//! A scripted caller, for tests and benchmarks: plays 16 kHz mono PCM16 into
//! a call at real-time pace, then silence, and timestamps what comes back.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message;

pub struct Outcome {
    /// When the scripted speech ended, from the start of the call.
    pub speech_end: Duration,
    /// JSON events with their arrival times.
    pub events: Vec<(Duration, Value)>,
    pub first_audio: Option<Duration>,
    /// Arrival time of every audio message.
    pub audio_at: Vec<Duration>,
    pub audio: Vec<i16>,
    /// When the caller hung up, and when the server closed the connection.
    pub hung_up: Duration,
    pub closed: Duration,
}

impl Outcome {
    /// Arrival time of the first event of `kind` (and matching `filter`).
    pub fn event_at(&self, kind: &str, filter: impl Fn(&Value) -> bool) -> Option<Duration> {
        self.events
            .iter()
            .find(|(_, event)| event["type"] == kind && filter(event))
            .map(|(at, _)| *at)
    }
}

/// Plays `speech`, then `tail` of silence, then hangs up.
pub async fn call(url: &str, speech: &[i16], tail: Duration) -> Result<Outcome> {
    let mut audio = speech.to_vec();
    audio.extend(std::iter::repeat_n(0i16, (tail.as_millis() * 16) as usize));
    let mut outcome = play(url, audio).await?;
    outcome.speech_end = speech_end(speech);
    Ok(outcome)
}

/// Plays `audio` at real-time pace, hangs up, and waits for the server to
/// close the connection.
pub async fn play(url: &str, audio: Vec<i16>) -> Result<Outcome> {
    play_with_backlog(url, audio, Duration::ZERO..Duration::ZERO).await
}

/// As `play`, but the audio in `backlog` (call time) is sent all at once, the
/// way it reaches a server that fell behind and then caught up.
pub async fn play_with_backlog(
    url: &str,
    audio: Vec<i16>,
    backlog: std::ops::Range<Duration>,
) -> Result<Outcome> {
    let (socket, _) = tokio_tungstenite::connect_async(url)
        .await
        .context("connect")?;
    let (mut sink, mut stream) = socket.split();
    let speech_end = speech_end(&audio);
    let frames: Vec<Vec<u8>> = audio
        .chunks(320)
        .map(|frame| frame.iter().flat_map(|s| s.to_le_bytes()).collect())
        .collect();
    let started = Instant::now();
    let sender = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(20));
        for (index, frame) in frames.into_iter().enumerate() {
            if !backlog.contains(&(Duration::from_millis(20) * index as u32)) {
                tick.tick().await;
            }
            if sink.send(Message::Binary(frame.into())).await.is_err() {
                break;
            }
        }
        let hung_up = started.elapsed();
        let _ = sink.close().await;
        hung_up
    });
    let mut outcome = Outcome {
        speech_end,
        events: Vec::new(),
        first_audio: None,
        audio_at: Vec::new(),
        audio: Vec::new(),
        hung_up: Duration::ZERO,
        closed: Duration::ZERO,
    };
    while let Some(Ok(message)) = stream.next().await {
        let at = started.elapsed();
        match message {
            Message::Binary(pcm) => {
                outcome.first_audio.get_or_insert(at);
                outcome.audio_at.push(at);
                outcome.audio.extend(
                    pcm.chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]])),
                );
            }
            Message::Text(text) => {
                if let Ok(event) = serde_json::from_str(&text) {
                    outcome.events.push((at, event));
                }
            }
            _ => {}
        }
    }
    outcome.closed = started.elapsed();
    outcome.hung_up = sender.await.unwrap_or_default();
    Ok(outcome)
}

pub fn read_wav(path: &std::path::Path) -> Result<Vec<i16>> {
    let mut reader = hound::WavReader::open(path).with_context(|| path.display().to_string())?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_rate == 16_000 && spec.channels == 1,
        "need 16 kHz mono"
    );
    Ok(reader.samples::<i16>().collect::<Result<_, _>>()?)
}

/// Where the speech actually ends: the last sample louder than about -36 dBFS,
/// not the end of the clip (clips carry some trailing quiet).
pub fn speech_end(speech: &[i16]) -> Duration {
    let last = speech
        .iter()
        .rposition(|s| s.unsigned_abs() > 500)
        .unwrap_or(0);
    Duration::from_micros(last as u64 * 1_000_000 / 16_000)
}
