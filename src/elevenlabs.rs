//! ElevenLabs realtime speech-to-text and streaming text-to-speech over
//! WebSockets. Wire formats follow Pipecat's ElevenLabs services.

use anyhow::{anyhow, Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use bytes::BytesMut;
use futures::stream::{SplitSink, StreamExt};
use futures::SinkExt;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::{Message, Utf8Bytes};

use tokio_tungstenite::{
    connect_async_tls_with_config, Connector, MaybeTlsStream, WebSocketStream,
};

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Clone)]
pub struct ElevenLabs {
    pub api_key: String,
    pub voice_id: String,
    pub tts_model: String,
    pub stt_model: String,
    /// `wss://api.elevenlabs.io` in production; a local mock in benchmarks.
    pub base_url: String,
    /// Built once per process and shared by every connection: without it,
    /// each connection loads and parses the system's root certificates
    /// (~74 ms of CPU on macOS) and keeps its own TLS config, and no
    /// connection can resume a previous session.
    tls: Arc<rustls::ClientConfig>,
}

impl ElevenLabs {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Result<Self> {
        let mut roots = rustls::RootCertStore::empty();
        roots.add_parsable_certificates(rustls_native_certs::load_native_certs().certs);
        let tls = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        Ok(Self {
            api_key: api_key.into(),
            voice_id: "21m00Tcm4TlvDq8ikWAM".into(),
            tts_model: "eleven_flash_v2_5".into(),
            stt_model: "scribe_v2_realtime".into(),
            base_url: base_url.into(),
            tls: Arc::new(tls),
        })
    }

    pub fn from_env() -> Result<Self> {
        let mut elevenlabs = Self::new(
            std::env::var("ELEVENLABS_API_KEY").context("ELEVENLABS_API_KEY")?,
            std::env::var("ELEVENLABS_BASE_URL")
                .unwrap_or_else(|_| "wss://api.elevenlabs.io".into()),
        )?;
        if let Ok(voice) = std::env::var("ELEVENLABS_VOICE_ID") {
            elevenlabs.voice_id = voice;
        }
        if let Ok(model) = std::env::var("ELEVENLABS_TTS_MODEL") {
            elevenlabs.tts_model = model;
        }
        Ok(elevenlabs)
    }

    async fn open(&self, path_and_query: &str) -> Result<Socket> {
        let mut request = format!("{}{path_and_query}", self.base_url).into_client_request()?;
        request
            .headers_mut()
            .insert("xi-api-key", self.api_key.parse()?);
        // Read buffers default to 128 KiB per socket, and every call holds two
        // of these sockets; transcripts and audio chunks need far less.
        let config = WebSocketConfig::default()
            .read_buffer_size(8 * 1024)
            .write_buffer_size(0);
        // No Nagle: audio and transcripts are small writes that must leave now.
        let connector = Connector::Rustls(self.tls.clone());
        let (socket, _) =
            connect_async_tls_with_config(request, Some(config), true, Some(connector))
                .await
                .context("ElevenLabs connect")?;
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
        Ok((
            Transcriber {
                sink,
                buffer: BytesMut::new(),
            },
            received,
        ))
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
    /// Each message is written here and split off; once the socket has sent
    /// it, the next `reserve` takes the same memory back. Steady state: no
    /// allocation per frame.
    buffer: BytesMut,
}

impl Transcriber {
    /// 16 kHz mono PCM16 little-endian.
    pub async fn audio(&mut self, pcm16: &[u8]) -> Result<()> {
        self.send(pcm16, false).await
    }

    /// End the current segment; a `Committed` transcript follows.
    pub async fn commit(&mut self) -> Result<()> {
        self.send(&[], true).await
    }

    async fn send(&mut self, pcm16: &[u8], commit: bool) -> Result<()> {
        let message = input_audio_chunk(&mut self.buffer, pcm16, commit);
        self.sink
            .send(Message::Text(message))
            .await
            .map_err(|e| anyhow!("ElevenLabs STT send: {e}"))
    }
}

/// `{"message_type":"input_audio_chunk","audio_base_64":…,"commit":…,"sample_rate":16000}`,
/// written into `buffer` (base64 has no characters JSON must escape).
fn input_audio_chunk(buffer: &mut BytesMut, pcm16: &[u8], commit: bool) -> Utf8Bytes {
    const START: &[u8] = br#"{"message_type":"input_audio_chunk","audio_base_64":""#;
    let end: &[u8] = if commit {
        br#"","commit":true,"sample_rate":16000}"#
    } else {
        br#"","commit":false,"sample_rate":16000}"#
    };
    let encoded = base64::encoded_len(pcm16.len(), true).unwrap_or(0);
    buffer.reserve(START.len() + encoded + end.len());
    buffer.extend_from_slice(START);
    let at = buffer.len();
    buffer.resize(at + encoded, 0);
    let written = BASE64.encode_slice(pcm16, &mut buffer[at..]).unwrap_or(0);
    buffer.truncate(at + written);
    buffer.extend_from_slice(end);
    // SAFETY: ASCII only: the two constants and base64 output.
    unsafe { Utf8Bytes::from_bytes_unchecked(buffer.split().freeze()) }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The hand-written message is the JSON `serde_json` would produce.
    #[test]
    fn input_audio_chunk_is_the_expected_json() {
        let mut buffer = BytesMut::new();
        for (pcm, commit) in [
            (&[1u8, 2, 3, 250, 0, 7][..], false),
            (&[][..], true),
            (&[9u8; 640][..], false),
        ] {
            let text = input_audio_chunk(&mut buffer, pcm, commit);
            let parsed: Value = serde_json::from_str(text.as_str()).unwrap();
            assert_eq!(
                parsed,
                json!({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": BASE64.encode(pcm),
                    "commit": commit,
                    "sample_rate": 16_000,
                })
            );
        }
    }
}
