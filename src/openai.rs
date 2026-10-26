//! A brain backed by any OpenAI-compatible `/chat/completions` endpoint
//! (OpenAI, OpenRouter, vLLM, llama.cpp server, ...). Keeps the call's
//! history in memory: one brain per call.

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
        }
    }
}

/// Appends the reply to the history when dropped, so a reply cut off by
/// barge-in still records what was generated before the cut.
struct Pending {
    history: Arc<Mutex<Vec<Value>>>,
    text: String,
}

impl Drop for Pending {
    fn drop(&mut self) {
        if !self.text.is_empty() {
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
        let messages = {
            let mut history = self.history.lock().expect("history lock");
            history.push(json!({ "role": "user", "content": utterance.text }));
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
