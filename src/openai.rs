//! A brain backed by any OpenAI-compatible `/chat/completions` endpoint
//! (OpenAI, OpenRouter, vLLM, llama.cpp server, ...). Keeps the call's
//! history in memory: one brain per call.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use futures::stream;
use serde_json::{json, Value};

use crate::brain::{Brain, Reply, ReplyEvent, Utterance};
use crate::sse::SseData;

#[derive(Clone)]
pub struct OpenAiBrain {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    history: Arc<Mutex<Vec<Value>>>,
    superseded: Arc<Mutex<Arc<AtomicBool>>>,
}

impl OpenAiBrain {
    pub fn new(
        http: reqwest::Client,
        base_url: &str,
        api_key: &str,
        model: &str,
        system: Option<&str>,
    ) -> Self {
        let history = system
            .map(|s| vec![json!({ "role": "system", "content": s })])
            .unwrap_or_default();
        Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            history: Arc::new(Mutex::new(history)),
            superseded: Arc::default(),
        }
    }
}

/// Appends the reply to the history when dropped, so a reply cut off by
/// barge-in still records what was generated before the cut.
struct Pending {
    history: Arc<Mutex<Vec<Value>>>,
    text: String,
    superseded: Arc<AtomicBool>,
}

impl Drop for Pending {
    fn drop(&mut self) {
        if !self.text.is_empty() && !self.superseded.load(Ordering::Acquire) {
            if let Ok(mut history) = self.history.lock() {
                history.push(
                    json!({ "role": "assistant", "content": std::mem::take(&mut self.text) }),
                );
            }
        }
    }
}

impl Brain for OpenAiBrain {
    async fn reply(&self, utterance: Utterance) -> Result<Reply> {
        let superseded = Arc::new(AtomicBool::new(false));
        let messages = {
            let mut history = self.history.lock().expect("history lock");
            let previous = std::mem::replace(
                &mut *self.superseded.lock().expect("superseded lock"),
                superseded.clone(),
            );
            if utterance.continues {
                previous.store(true, Ordering::Release);
                if history.last().is_some_and(|m| m["role"] == "assistant") {
                    history.pop();
                }
            }
            match history.last_mut() {
                Some(last) if utterance.continues && last["role"] == "user" => {
                    let joined = format!(
                        "{} {}",
                        last["content"].as_str().unwrap_or(""),
                        utterance.text
                    );
                    last["content"] = json!(joined);
                }
                _ => history.push(json!({ "role": "user", "content": utterance.text })),
            }
            history.clone()
        };
        let response = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&json!({ "model": self.model, "messages": messages, "stream": true }))
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            bail!(
                "chat completions {status}: {}",
                response.text().await.unwrap_or_default()
            );
        }
        let state = Some((
            SseData::new(response),
            Pending {
                history: self.history.clone(),
                text: String::new(),
                superseded,
            },
        ));
        // An error ends the reply: it is yielded once, then the stream stops.
        Ok(Box::pin(stream::unfold(state, |state| async move {
            let (mut data, mut pending) = state?;
            loop {
                match data.next().await {
                    Ok(Some(payload)) => {
                        let delta = serde_json::from_str::<Value>(&payload).ok().and_then(|v| {
                            v["choices"][0]["delta"]["content"]
                                .as_str()
                                .map(str::to_string)
                        });
                        if let Some(delta) = delta.filter(|d| !d.is_empty()) {
                            pending.text.push_str(&delta);
                            return Some((Ok(ReplyEvent::Text(delta)), Some((data, pending))));
                        }
                    }
                    Ok(None) => return None,
                    Err(error) => return Some((Err(error), None)),
                }
            }
        })))
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::mock::{self, Script};

    fn said(text: &str, continues: bool) -> Utterance {
        Utterance {
            text: text.into(),
            index: 1,
            continues,
        }
    }

    fn roles(brain: &OpenAiBrain) -> Vec<(String, String)> {
        brain
            .history
            .lock()
            .unwrap()
            .iter()
            .map(|m| {
                let text = |v: &Value| v.as_str().unwrap_or_default().to_string();
                (text(&m["role"]), text(&m["content"]))
            })
            .collect()
    }

    /// A pause mid-question got an answer the caller talked over at once:
    /// the brain sees one question, and none of the answer it cut off.
    #[tokio::test]
    async fn a_continued_utterance_extends_the_question_and_drops_the_cut_reply() {
        let mocks = mock::start(Script::default()).await.unwrap();
        let brain = OpenAiBrain::new(
            reqwest::Client::new(),
            &format!("http://{mocks}/v1"),
            "mock",
            "mock",
            None,
        );
        let mut cut = brain
            .reply(said("Here is a question.", false))
            .await
            .unwrap();
        assert!(cut.next().await.is_some(), "the reply started");
        let continued = brain.reply(said("Is it true?", true)).await.unwrap();
        drop(cut); // dropped after the continuation started: must not land
        let finished: Vec<_> = continued.collect().await;
        assert!(!finished.is_empty());
        let history = roles(&brain);
        assert_eq!(
            history[0],
            ("user".into(), "Here is a question. Is it true?".into())
        );
        assert_eq!(history.len(), 2, "{history:?}");
        assert_eq!(history[1].0, "assistant");

        // Not a continuation: a new user message after the reply.
        let _ = brain
            .reply(said("Thanks.", false))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        let history = roles(&brain);
        assert_eq!(history[2], ("user".into(), "Thanks.".into()));
    }
}
