//! One call: a client WebSocket carrying 16 kHz mono PCM16 both ways, plus
//! JSON events to the client. The call is a small state machine driven by one
//! `select!` loop; each spoken reply runs as its own task so barge-in can drop
//! it (closing its text-to-speech stream) without touching the brain, which
//! decides for itself whether to keep working.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use axum::extract::ws::{Message, WebSocket};
use bytes::Bytes;
use futures::{FutureExt, SinkExt, Stream, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::Duration;

use crate::brain::{Brain, Reply, Speakable, Utterance};
use crate::call_state::{Action, CallState, Input};
use crate::elevenlabs::{ElevenLabs, Transcript};
use crate::endpoint::TurnModel;
use crate::metrics;
use crate::turn_taking::{Policy, TurnEvent, TurnTaker};
use crate::vad::{CallVad, VadModel};

/// How often a quiet call asks its brain for an unprompted reply.
const UNPROMPTED_POLL: Duration = Duration::from_secs(2);
/// At most 1 s of 16 kHz PCM16 per speech-to-text message when catching up.
const MAX_COALESCED_BYTES: usize = 32_000;
/// A commit the provider never answers resolves as empty, so one lost
/// reply can't leave the turn unanswered and the call deaf.
const COMMIT_DEADLINE: Duration = Duration::from_secs(3);

/// Process-wide pieces every call shares.
#[derive(Clone)]
pub struct Shared {
    pub vad: VadModel,
    pub turn: TurnModel,
    pub elevenlabs: ElevenLabs,
    pub speech: SpeechPolicy,
    pub turn_policy: Policy,
    /// Calls over WebRTC (`POST /v1/webrtc`), when enabled.
    pub webrtc: Option<crate::webrtc::WebRtc>,
    /// The limit on calls at once, across transports.
    pub admission: crate::admission::Admission,
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

/// What a call sends its caller: 16 kHz mono PCM16 audio, as fast as it is
/// made (the transport paces playback), and JSON events. `interrupt` means
/// drop any audio not yet played.
pub enum ToCaller {
    Audio(Vec<u8>),
    Event(Value),
}

/// Serves one call on any transport: `audio` is the caller's 16 kHz mono
/// PCM16 in any frame size, ending when they hang up; everything for them
/// goes to `to_caller`, which is dropped when the call ends.
pub async fn run<B, S>(audio: S, to_caller: mpsc::Sender<ToCaller>, shared: Shared, brain: B)
where
    B: Brain,
    S: Stream<Item = Bytes> + Unpin + Send,
{
    metrics::CALLS.add();
    if let Err(error) = run_inner(audio, to_caller, shared, Arc::new(brain)).await {
        metrics::CALL_ERRORS.add();
        tracing::warn!(%error, "call ended with an error");
    }
}

/// A call over a WebSocket: binary messages are PCM16 both ways, events are
/// text.
pub async fn run_websocket<B: Brain>(socket: WebSocket, shared: Shared, brain: B) {
    let (mut client_tx, client_rx) = socket.split();
    let (to_caller, mut outbox) = mpsc::channel::<ToCaller>(512);
    let mut writer = AbortOnDrop(tokio::spawn(async move {
        while let Some(message) = outbox.recv().await {
            let message = match message {
                ToCaller::Audio(pcm) => Message::Binary(pcm.into()),
                ToCaller::Event(event) => Message::Text(event.to_string().into()),
            };
            if client_tx.send(message).await.is_err() {
                break;
            }
        }
    }));
    let audio = client_rx
        .take_while(|message| std::future::ready(message.is_ok()))
        .filter_map(|message| {
            std::future::ready(match message {
                Ok(Message::Binary(pcm)) => Some(pcm),
                _ => None,
            })
        });
    run(Box::pin(audio), to_caller, shared, brain).await;
    // The caller hung up: let what is already queued reach them.
    let _ = (&mut writer.0).await;
}

async fn run_inner<B: Brain, S: Stream<Item = Bytes> + Unpin>(
    mut client_rx: S,
    to_client: mpsc::Sender<ToCaller>,
    shared: Shared,
    brain: Arc<B>,
) -> Result<()> {
    // Every task this call starts is aborted when the call ends, on any path:
    // a reply left running would hold the caller's connection open.
    let event = |value: Value| ToCaller::Event(value);
    let _ = to_client.send(event(json!({ "type": "ready" }))).await;

    let (mut transcriber, mut transcripts) = shared.elevenlabs.transcriber().await?;
    let mut turns = TurnTaker::new(
        CallVad::new(shared.vad.clone()),
        shared.turn.clone(),
        shared.turn_policy.clone(),
    );
    // Every decision is the reducer's (checked over all interleavings in
    // tests/call_state_model.rs); this loop only does the I/O it asks for.
    let mut state = CallState::default();
    let mut ended_at: Option<Instant> = None;
    let mut speaking: Option<AbortOnDrop> = None;
    // Replies say when they finish, so the state knows the call is quiet.
    // Each reply carries its generation, so a finish queued just before a
    // newer reply started can't mark the newer one finished.
    let (reply_done_tx, mut reply_done) = mpsc::channel::<u64>(4);
    let mut generation = 0u64;
    // Reused for every frame: the frame path allocates nothing of its own.
    let mut samples: Vec<f32> = Vec::with_capacity(320);
    // Only used when frames queued up; empty (and unallocated) otherwise.
    let mut backlog: Vec<u8> = Vec::new();
    // The caller hung up while a backlog was being read: finish it, then stop.
    let mut hung_up = false;
    let mut unprompted = tokio::time::interval(UNPROMPTED_POLL);
    // Last commit sent or answered: the deadline runs from here.
    // One timer, moved when a commit is sent or answered (not rebuilt per frame).
    let commit_deadline = tokio::time::sleep(COMMIT_DEADLINE);
    tokio::pin!(commit_deadline);

    loop {
        let input = tokio::select! {
            pcm = client_rx.next() => {
                let Some(pcm) = pcm else { break };
                // Frames that queued while this loop was held up (a slow
                // host) go to speech-to-text as one message: ElevenLabs ends
                // a session flooded with hundreds of 20 ms messages at once.
                backlog.clear();
                while backlog.len() + pcm.len() < MAX_COALESCED_BYTES {
                    match client_rx.next().now_or_never() {
                        Some(Some(more)) => {
                            if backlog.is_empty() {
                                backlog.extend_from_slice(&pcm);
                            }
                            backlog.extend_from_slice(&more);
                        }
                        Some(None) => {
                            hung_up = true;
                            break;
                        }
                        None => break,
                    }
                }
                let pcm: &[u8] = if backlog.is_empty() { &pcm } else { &backlog };
                samples.clear();
                samples.extend(pcm.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0));
                transcriber.audio(pcm).await?;
                let mut actions = Vec::new();
                for turn_event in turns.push(&samples).await? {
                    if let TurnEvent::Assessed { probability } = turn_event {
                        let _ = to_client.send(event(json!({ "type": "turn", "complete": probability }))).await;
                    }
                    actions.extend(state.on(Input::Turn(turn_event)));
                }
                actions
            }
            transcript = transcripts.recv() => match transcript {
                Some(Transcript::Partial(text)) => {
                    let _ = to_client.send(event(json!({ "type": "transcript", "text": text, "final": false }))).await;
                    continue;
                }
                Some(Transcript::Committed(text)) => {
                    commit_deadline.as_mut().reset(tokio::time::Instant::now() + COMMIT_DEADLINE);
                    state.on(Input::Committed(text))
                }
                Some(Transcript::Error(error)) => {
                    let _ = to_client.send(event(json!({ "type": "error", "message": error }))).await;
                    continue;
                }
                None => anyhow::bail!("speech-to-text stream closed"),
            },
            _ = &mut commit_deadline, if state.pending().1 > 0 => {
                tracing::warn!(commits = state.pending().1, "speech-to-text never answered; treating as empty");
                commit_deadline.as_mut().reset(tokio::time::Instant::now() + COMMIT_DEADLINE);
                (0..state.pending().1).flat_map(|_| state.on(Input::Committed(String::new()))).collect()
            }
            Some(finished) = reply_done.recv() => {
                if finished != generation {
                    continue;
                }
                state.on(Input::ReplyFinished)
            }
            _ = unprompted.tick() => {
                if !state.quiet() {
                    continue;
                }
                let reply = match brain.unprompted().await {
                    Ok(Some(reply)) => reply,
                    Ok(None) => continue,
                    // A poll that failed is retried on the next tick; the call goes on.
                    Err(error) => {
                        tracing::warn!(%error, "unprompted poll failed");
                        continue;
                    }
                };
                let actions = state.on(Input::UnpromptedReady);
                if actions.contains(&Action::SpeakUnprompted) {
                    generation += 1;
                    let id = generation;
                    let (shared, to_client, done) = (shared.clone(), to_client.clone(), reply_done_tx.clone());
                    speaking = Some(AbortOnDrop(tokio::spawn(async move {
                        let result = async { speak(reply, shared.elevenlabs.speaker().await?, &shared.speech, &to_client, None).await };
                        report(result.await, &to_client).await;
                        let _ = done.send(id).await;
                    })));
                }
                continue;
            }
        };
        for action in input {
            match action {
                Action::Commit => {
                    commit_deadline
                        .as_mut()
                        .reset(tokio::time::Instant::now() + COMMIT_DEADLINE);
                    transcriber.commit().await?
                }
                Action::Interrupt => {
                    // Barge-in: stop speaking; the brain decides what
                    // happens to the work behind that reply.
                    speaking = None;
                    let _ = to_client.send(event(json!({ "type": "interrupt" }))).await;
                }
                Action::TurnEnded(by) => {
                    ended_at = Some(Instant::now());
                    metrics::turn_ended(by);
                    let _ = to_client
                        .send(event(
                            json!({ "type": "turn_end", "by": format!("{by:?}") }),
                        ))
                        .await;
                }
                Action::FinalTranscript(text) => {
                    let _ = to_client
                        .send(event(
                            json!({ "type": "transcript", "text": text, "final": true }),
                        ))
                        .await;
                }
                Action::Reply { text, index } => {
                    generation += 1;
                    let id = generation;
                    let (brain, shared, to_client, done) = (
                        brain.clone(),
                        shared.clone(),
                        to_client.clone(),
                        reply_done_tx.clone(),
                    );
                    let started = ended_at.take();
                    let utterance = Utterance { text, index };
                    // Replacing the previous reply aborts it.
                    speaking = Some(AbortOnDrop(tokio::spawn(async move {
                        let reply = async {
                            let (speaker, reply) =
                                tokio::join!(shared.elevenlabs.speaker(), brain.reply(utterance));
                            speak(reply?, speaker?, &shared.speech, &to_client, started).await
                        };
                        report(reply.await, &to_client).await;
                        let _ = done.send(id).await;
                    })));
                }
                Action::SpeakUnprompted => {}
            }
        }
        if hung_up {
            break;
        }
    }
    drop(speaking);
    Ok(())
}

async fn report(result: Result<()>, to_client: &mpsc::Sender<ToCaller>) {
    if let Err(error) = result {
        metrics::REPLY_ERRORS.add();
        tracing::warn!(%error, "reply failed");
        let _ = to_client
            .send(ToCaller::Event(
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
    to_client: &mpsc::Sender<ToCaller>,
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
                    metrics::FIRST_AUDIO.record(start.elapsed());
                    let ms = start.elapsed().as_millis() as u64;
                    let _ = forward_to
                        .send(ToCaller::Event(
                            json!({ "type": "metrics", "firstAudioMs": ms }),
                        ))
                        .await;
                }
            }
            if forward_to.send(ToCaller::Audio(pcm)).await.is_err() {
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
