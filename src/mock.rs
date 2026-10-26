//! Local stand-ins for the speech services and the brain, with scripted
//! timing, for benchmarks and tests. They speak the real wire protocols
//! (ElevenLabs realtime STT and stream-input TTS, OpenAI chat completions), so
//! chirp runs its production code paths against them; every delay is a known
//! constant, so whatever time remains is chirp's own.

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Result;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub struct Script {
    /// What the mock transcriber hears, whatever the audio.
    pub transcript: String,
    /// Commit → committed transcript.
    pub stt_commit: Duration,
    /// Request → first token.
    pub brain_first_token: Duration,
    pub brain_token_gap: Duration,
    pub reply_words: usize,
    /// First text → first audio.
    pub tts_first_audio: Duration,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            transcript: "What is the weather like today?".into(),
            stt_commit: Duration::from_millis(100),
            brain_first_token: Duration::from_millis(300),
            brain_token_gap: Duration::from_millis(10),
            reply_words: 20,
            tts_first_audio: Duration::from_millis(150),
        }
    }
}

impl Script {
    /// The scripted part of the time from end of turn to first audio.
    pub fn scripted_reply_delay(&self) -> Duration {
        self.stt_commit + self.brain_first_token + self.tts_first_audio
    }
}

/// Serves the mocks on an ephemeral 127.0.0.1 port and returns the address.
pub async fn start(script: Script) -> Result<SocketAddr> {
    start_on(("127.0.0.1", 0), script).await
}

pub async fn start_on(
    address: impl tokio::net::ToSocketAddrs,
    script: Script,
) -> Result<SocketAddr> {
    let router = Router::new()
        .route("/v1/speech-to-text/realtime", get(stt))
        .route("/v1/text-to-speech/{voice}/stream-input", get(tts))
        .route(
            "/v1/text-to-speech/{voice}/multi-stream-input",
            get(tts_contexts),
        )
        .route("/v1/chat/completions", post(completions))
        .with_state(script);
    let listener = tokio::net::TcpListener::bind(address).await?;
    let address = listener.local_addr()?;
    tokio::spawn(async move { axum::serve(listener, router).await });
    Ok(address)
}

/// Splits a socket so replies can be sent on a delay without stalling reads.
fn outbox(
    socket: WebSocket,
) -> (
    mpsc::UnboundedSender<Value>,
    futures::stream::SplitStream<WebSocket>,
) {
    let (mut sink, stream) = socket.split();
    let (sender, mut queue) = mpsc::unbounded_channel::<Value>();
    tokio::spawn(async move {
        while let Some(value) = queue.recv().await {
            if sink
                .send(Message::Text(value.to_string().into()))
                .await
                .is_err()
            {
                break;
            }
        }
    });
    (sender, stream)
}

async fn stt(State(script): State<Script>, upgrade: WebSocketUpgrade) -> impl IntoResponse {
    upgrade.on_upgrade(move |socket| async move {
        let (send, mut stream) = outbox(socket);
        let _ = send.send(json!({ "message_type": "session_started" }));
        while let Some(Ok(Message::Text(text))) = stream.next().await {
            let Ok(message) = serde_json::from_str::<Value>(&text) else { continue };
            if message["commit"] == true {
                let (send, script) = (send.clone(), script.clone());
                tokio::spawn(async move {
                    tokio::time::sleep(script.stt_commit).await;
                    let _ = send.send(json!({ "message_type": "committed_transcript", "text": script.transcript }));
                });
            }
        }
    })
}

async fn tts(State(script): State<Script>, upgrade: WebSocketUpgrade) -> impl IntoResponse {
    upgrade.on_upgrade(move |socket| async move {
        let (send, mut stream) = outbox(socket);
        // 100 ms of 16 kHz PCM16 per text message: enough to be audio, cheap to make.
        let chunk = BASE64.encode(vec![0u8; 3_200]);
        let mut first = true;
        while let Some(Ok(Message::Text(text))) = stream.next().await {
            let Ok(message) = serde_json::from_str::<Value>(&text) else {
                continue;
            };
            let words = message["text"].as_str().unwrap_or_default().to_string();
            if words.is_empty() {
                let _ = send.send(json!({ "isFinal": true }));
                break;
            }
            if words.trim().is_empty() {
                continue; // the stream's opening " "
            }
            if first {
                first = false;
                tokio::time::sleep(script.tts_first_audio).await;
            }
            let _ = send.send(json!({ "audio": chunk }));
        }
    })
}

/// The multi-context variant (one socket, many utterances keyed by
/// `context_id`), which other engines use. Same timing per context.
async fn tts_contexts(
    State(script): State<Script>,
    upgrade: WebSocketUpgrade,
) -> impl IntoResponse {
    upgrade.on_upgrade(move |socket| async move {
        let (send, mut stream) = outbox(socket);
        let chunk = BASE64.encode(vec![0u8; 3_200]);
        let mut started = std::collections::HashSet::<String>::new();
        while let Some(Ok(Message::Text(text))) = stream.next().await {
            let Ok(message) = serde_json::from_str::<Value>(&text) else {
                continue;
            };
            let context = message["context_id"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            if message["close_context"] == true {
                let _ = send.send(json!({ "isFinal": true, "contextId": context }));
                continue;
            }
            let words = message["text"].as_str().unwrap_or_default();
            if words.trim().is_empty() {
                continue; // opening " ", keepalive "", or a flush
            }
            if started.insert(context.clone()) {
                tokio::time::sleep(script.tts_first_audio).await;
            }
            let _ = send.send(json!({ "audio": chunk, "contextId": context }));
        }
    })
}

async fn completions(State(script): State<Script>) -> impl IntoResponse {
    let body = futures::stream::unfold(0usize, move |word| {
        let script = script.clone();
        async move {
            if word > script.reply_words {
                return None;
            }
            let line = if word == script.reply_words {
                tokio::time::sleep(script.brain_token_gap).await;
                "data: [DONE]\n\n".to_string()
            } else {
                let wait = if word == 0 {
                    script.brain_first_token
                } else {
                    script.brain_token_gap
                };
                tokio::time::sleep(wait).await;
                // Sentences, as real model output has: engines that buffer to a
                // sentence boundary before speaking are not penalized by the mock.
                let end = if word % 6 == 5 { "." } else { "" };
                let chunk =
                    json!({ "choices": [{ "delta": { "content": format!("word{word}{end} ") } }] });
                format!("data: {chunk}\n\n")
            };
            Some((Ok::<_, std::convert::Infallible>(line), word + 1))
        }
    });
    (
        [("content-type", "text/event-stream")],
        axum::body::Body::from_stream(body),
    )
}
