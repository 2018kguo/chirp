//! ElevenLabs realtime speech-to-text and streaming text-to-speech over
//! WebSockets. Wire formats follow Pipecat's ElevenLabs services.

use anyhow::{anyhow, Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use futures::stream::{SplitSink, StreamExt};
use futures::SinkExt;
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Clone)]
pub struct ElevenLabs {
    pub api_key: String,
    pub voice_id: String,
    pub tts_model: String,
    pub stt_model: String,
    pub host: String,
}

impl ElevenLabs {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            api_key: std::env::var("ELEVENLABS_API_KEY").context("ELEVENLABS_API_KEY")?,
            voice_id: std::env::var("ELEVENLABS_VOICE_ID")
                .unwrap_or_else(|_| "21m00Tcm4TlvDq8ikWAM".into()),
            tts_model: std::env::var("ELEVENLABS_TTS_MODEL")
                .unwrap_or_else(|_| "eleven_flash_v2_5".into()),
            stt_model: "scribe_v2_realtime".into(),
            host: "api.elevenlabs.io".into(),
        })
    }

    async fn open(&self, path_and_query: &str) -> Result<Socket> {
        let mut request = format!("wss://{}{path_and_query}", self.host).into_client_request()?;
        request
            .headers_mut()
            .insert("xi-api-key", self.api_key.parse()?);
        let (socket, _) = connect_async(request).await.context("ElevenLabs connect")?;
        Ok(socket)
    }

    /// One realtime transcription session for a call. Commits are manual: the
    /// call commits when Smart Turn says the user's turn is over.
    pub async fn transcriber(&self) -> Result<(Transcriber, mpsc::Receiver<Transcript>)> {
        let socket = self
            .open(&format!(
                "/v1/speech-to-text/realtime?model_id={}&audio_format=pcm_16000&commit_strategy=manual",
                self.stt_model
            ))
            .await?;
        let (sink, mut stream) = socket.split();
        let (events, received) = mpsc::channel(64);
        tokio::spawn(async move {
            while let Some(Ok(message)) = stream.next().await {
                let Message::Text(text) = message else {
                    continue;
                };
                let Ok(value) = serde_json::from_str::<Value>(&text) else {
                    continue;
                };
                let words = value
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let event = match value.get("message_type").and_then(Value::as_str) {
                    Some("partial_transcript") => Transcript::Partial(words),
                    Some("committed_transcript") => Transcript::Committed(words),
                    Some("session_started" | "committed_transcript_with_timestamps") => continue,
                    other => Transcript::Error(format!("{}: {text}", other.unwrap_or("unknown"))),
                };
                if events.send(event).await.is_err() {
                    break;
                }
            }
        });
        Ok((Transcriber { sink }, received))
    }

    /// One spoken response. Audio arrives as 16 kHz PCM16 chunks; dropping the
    /// `Speaker` and the receiver stops it (barge-in).
    pub async fn speaker(&self) -> Result<(Speaker, mpsc::Receiver<Vec<u8>>)> {
        let socket = self
            .open(&format!(
                "/v1/text-to-speech/{}/stream-input?model_id={}&output_format=pcm_16000&auto_mode=true",
                self.voice_id, self.tts_model
            ))
            .await?;
        let (mut sink, mut stream) = socket.split();
        // The first message opens the stream; a single space is the documented start.
        sink.send(Message::Text(json!({ "text": " " }).to_string().into()))
            .await?;
        let (audio, received) = mpsc::channel(256);
        tokio::spawn(async move {
            while let Some(Ok(message)) = stream.next().await {
                let Message::Text(text) = message else {
                    continue;
                };
                let Ok(value) = serde_json::from_str::<Value>(&text) else {
                    continue;
                };
                if let Some(chunk) = value.get("audio").and_then(Value::as_str) {
                    if let Ok(pcm) = BASE64.decode(chunk) {
                        if audio.send(pcm).await.is_err() {
                            break;
                        }
                    }
                }
                if value.get("isFinal").and_then(Value::as_bool) == Some(true) {
                    break;
                }
            }
        });
        Ok((Speaker { sink }, received))
    }
}

#[derive(Debug)]
pub enum Transcript {
    Partial(String),
    Committed(String),
    Error(String),
}

pub struct Transcriber {
    sink: SplitSink<Socket, Message>,
}

impl Transcriber {
    /// 16 kHz mono PCM16 little-endian.
    pub async fn audio(&mut self, pcm16: &[u8]) -> Result<()> {
        self.send(BASE64.encode(pcm16), false).await
    }

    /// End the current segment; a `Committed` transcript follows.
    pub async fn commit(&mut self) -> Result<()> {
        self.send(String::new(), true).await
    }

    async fn send(&mut self, audio_base_64: String, commit: bool) -> Result<()> {
        let message = json!({
            "message_type": "input_audio_chunk",
            "audio_base_64": audio_base_64,
            "commit": commit,
            "sample_rate": 16_000,
        });
        self.sink
            .send(Message::Text(message.to_string().into()))
            .await
            .map_err(|e| anyhow!("ElevenLabs STT send: {e}"))
    }
}

pub struct Speaker {
    sink: SplitSink<Socket, Message>,
}

impl Speaker {
    pub async fn say(&mut self, text: &str) -> Result<()> {
        self.sink
            .send(Message::Text(json!({ "text": text }).to_string().into()))
            .await
            .map_err(|e| anyhow!("ElevenLabs TTS send: {e}"))
    }

    /// No more text: ElevenLabs speaks what it has and ends the stream.
    pub async fn finish(&mut self) -> Result<()> {
        self.say("").await
    }
}
