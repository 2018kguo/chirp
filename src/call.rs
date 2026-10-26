//! One call: a client WebSocket carrying 16 kHz mono PCM16 both ways, plus
//! JSON events to the client. The call is a small state machine driven by one
//! `select!` loop; each spoken reply runs as its own task so barge-in can drop
//! it (closing its text-to-speech stream) without touching the brain, which
//! decides for itself whether to keep working.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{sleep_until, Duration, Instant as TokioInstant};

use crate::brain::{Brain, Reply, Speakable, Utterance};
use crate::elevenlabs::{ElevenLabs, Transcriber, Transcript};
use crate::endpoint::TurnModel;
use crate::vad::{CallVad, Speech, VadModel, SAMPLE_RATE};

/// Audio kept from just before speech is detected, so the turn model hears
/// the start of the utterance.
const PRE_ROLL: usize = SAMPLE_RATE / 2;
const TURN_AUDIO_MAX: usize = 8 * SAMPLE_RATE;
/// Smart Turn's threshold.
const COMPLETE: f32 = 0.5;
/// After an "incomplete" decision, silence this long ends the turn anyway.
const INCOMPLETE_GRACE: Duration = Duration::from_millis(2500);
/// How often a quiet call asks its brain for an unprompted reply.
const UNPROMPTED_POLL: Duration = Duration::from_secs(2);

/// Process-wide pieces every call shares.
#[derive(Clone)]
pub struct Shared {
    pub vad: VadModel,
    pub turn: TurnModel,
    pub elevenlabs: ElevenLabs,
    pub speech: SpeechPolicy,
}

/// How much of a reply is spoken.
#[derive(Clone)]
pub struct SpeechPolicy {
    /// Past this many characters, speech stops at the next sentence end.
    pub budget_chars: usize,
    /// Said after the cut, e.g. where the rest of the answer can be found.
    pub budget_tail: String,
}

impl Default for SpeechPolicy {
    fn default() -> Self {
        Self {
            budget_chars: 350,
            budget_tail: " The rest is in the transcript.".into(),
        }
    }
}

enum ToClient {
    Audio(Vec<u8>),
    Event(Value),
}

pub async fn run<B: Brain>(socket: WebSocket, shared: Shared, brain: B) {
    if let Err(error) = run_inner(socket, shared, Arc::new(brain)).await {
        tracing::warn!(%error, "call ended with an error");
    }
}

async fn run_inner<B: Brain>(socket: WebSocket, shared: Shared, brain: Arc<B>) -> Result<()> {
    let (mut client_tx, mut client_rx) = socket.split();
    let (to_client, mut outbox) = mpsc::channel::<ToClient>(512);
    let writer = tokio::spawn(async move {
        while let Some(message) = outbox.recv().await {
            let message = match message {
                ToClient::Audio(pcm) => Message::Binary(pcm.into()),
                ToClient::Event(event) => Message::Text(event.to_string().into()),
            };
            if client_tx.send(message).await.is_err() {
                break;
            }
        }
    });
    let event = |value: Value| ToClient::Event(value);
    let _ = to_client.send(event(json!({ "type": "ready" }))).await;

    let (mut transcriber, mut transcripts) = shared.elevenlabs.transcriber().await?;
    let mut vad = CallVad::new(shared.vad.clone());
    let mut recent: Vec<f32> = Vec::with_capacity(PRE_ROLL * 2);
    let mut turn_audio: Vec<f32> = Vec::new();
    let mut user_turn = 0u32;
    let mut awaiting_transcript = false;
    let mut turn_ended_at: Option<Instant> = None;
    let mut grace: Option<TokioInstant> = None;
    let mut speaking: Option<JoinHandle<()>> = None;
    let mut unprompted = tokio::time::interval(UNPROMPTED_POLL);

    loop {
        let grace_deadline = grace;
        tokio::select! {
            message = client_rx.next() => {
                let Some(Ok(message)) = message else { break };
                let Message::Binary(pcm) = message else { continue };
                let samples: Vec<f32> = pcm
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
                transcriber.audio(&pcm).await?;
                recent.extend_from_slice(&samples);
                if recent.len() > PRE_ROLL {
                    recent.drain(..recent.len() - PRE_ROLL);
                }
                if vad.speaking() || grace.is_some() {
                    turn_audio.extend_from_slice(&samples);
                    if turn_audio.len() > TURN_AUDIO_MAX {
                        turn_audio.drain(..turn_audio.len() - TURN_AUDIO_MAX);
                    }
                }
                match vad.push(&samples).await? {
                    Some(Speech::Started) => {
                        grace = None;
                        if turn_audio.is_empty() {
                            turn_audio.extend_from_slice(&recent);
                        }
                        if let Some(task) = speaking.take() {
                            // Barge-in: stop speaking; the brain decides what
                            // happens to the work behind that reply.
                            task.abort();
                            let _ = to_client.send(event(json!({ "type": "interrupt" }))).await;
                        }
                    }
                    Some(Speech::Stopped) => {
                        let probability = shared.turn.complete(turn_audio.clone()).await?;
                        let _ = to_client.send(event(json!({ "type": "turn", "complete": probability }))).await;
                        if probability > COMPLETE {
                            end_turn(&mut transcriber, &mut turn_audio, &mut awaiting_transcript, &mut turn_ended_at).await?;
                        } else {
                            grace = Some(TokioInstant::now() + INCOMPLETE_GRACE);
                        }
                    }
                    None => {}
                }
            }
            _ = async { sleep_until(grace_deadline.unwrap()).await }, if grace_deadline.is_some() => {
                grace = None;
                end_turn(&mut transcriber, &mut turn_audio, &mut awaiting_transcript, &mut turn_ended_at).await?;
            }
            transcript = transcripts.recv() => {
                match transcript {
                    Some(Transcript::Partial(text)) => {
                        let _ = to_client.send(event(json!({ "type": "transcript", "text": text, "final": false }))).await;
                    }
                    Some(Transcript::Committed(text)) if awaiting_transcript => {
                        awaiting_transcript = false;
                        let _ = to_client.send(event(json!({ "type": "transcript", "text": text, "final": true }))).await;
                        if !text.trim().is_empty() {
                            user_turn += 1;
                            if let Some(task) = speaking.take() {
                                task.abort();
                            }
                            let (brain, shared, to_client) = (brain.clone(), shared.clone(), to_client.clone());
                            let started = turn_ended_at.take();
                            let utterance = Utterance { text, index: user_turn };
                            speaking = Some(tokio::spawn(async move {
                                let reply = async {
                                    let (speaker, reply) = tokio::join!(shared.elevenlabs.speaker(), brain.reply(utterance));
                                    speak(reply?, speaker?, &shared.speech, &to_client, started).await
                                };
                                report(reply.await, &to_client).await;
                            }));
                        }
                    }
                    Some(Transcript::Committed(_)) => {}
                    Some(Transcript::Error(error)) => {
                        let _ = to_client.send(event(json!({ "type": "error", "message": error }))).await;
                    }
                    None => anyhow::bail!("speech-to-text stream closed"),
                }
            }
            _ = unprompted.tick() => {
                let busy = speaking.as_ref().is_some_and(|task| !task.is_finished());
                if busy || vad.speaking() || awaiting_transcript || grace.is_some() {
                    continue;
                }
                let Some(reply) = brain.unprompted().await? else { continue };
                let (shared, to_client) = (shared.clone(), to_client.clone());
                speaking = Some(tokio::spawn(async move {
                    let result = async { speak(reply, shared.elevenlabs.speaker().await?, &shared.speech, &to_client, None).await };
                    report(result.await, &to_client).await;
                }));
            }
        }
    }
    if let Some(task) = speaking {
        task.abort();
    }
    drop(to_client);
    let _ = writer.await;
    Ok(())
}

async fn end_turn(
    transcriber: &mut Transcriber,
    turn_audio: &mut Vec<f32>,
    awaiting_transcript: &mut bool,
    turn_ended_at: &mut Option<Instant>,
) -> Result<()> {
    transcriber.commit().await?;
    turn_audio.clear();
    *awaiting_transcript = true;
    *turn_ended_at = Some(Instant::now());
    Ok(())
}

async fn report(result: Result<()>, to_client: &mpsc::Sender<ToClient>) {
    if let Err(error) = result {
        tracing::warn!(%error, "reply failed");
        let _ = to_client
            .send(ToClient::Event(
                json!({ "type": "error", "message": error.to_string() }),
            ))
            .await;
    }
}

/// Speak one reply. `turn_ended_at` is when the user's turn was judged over;
/// the first audio's delay from it is reported to the client.
async fn speak(
    mut reply: Reply,
    (mut speaker, mut audio): (crate::elevenlabs::Speaker, mpsc::Receiver<Vec<u8>>),
    policy: &SpeechPolicy,
    to_client: &mpsc::Sender<ToClient>,
    turn_ended_at: Option<Instant>,
) -> Result<()> {
    let forward_to = to_client.clone();
    // Aborted with this reply: on barge-in, audio already queued must not
    // keep reaching the client after the interrupt event.
    let mut forwarder = AbortOnDrop(tokio::spawn(async move {
        let mut first = true;
        while let Some(pcm) = audio.recv().await {
            if first {
                first = false;
                if let Some(start) = turn_ended_at {
                    let ms = start.elapsed().as_millis() as u64;
                    let _ = forward_to
                        .send(ToClient::Event(
                            json!({ "type": "metrics", "firstAudioMs": ms }),
                        ))
                        .await;
                }
            }
            if forward_to.send(ToClient::Audio(pcm)).await.is_err() {
                break;
            }
        }
    }));
    let mut speakable = Speakable::new(policy.budget_chars, policy.budget_tail.clone());
    while let Some(event) = reply.next().await {
        if let Some(text) = speakable.event(event?) {
            speaker.say(&text).await?;
        }
    }
    if let Some(rest) = speakable.finish() {
        speaker.say(&rest).await?;
    }
    speaker.finish().await?;
    let _ = (&mut forwarder.0).await;
    Ok(())
}

struct AbortOnDrop(JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}
