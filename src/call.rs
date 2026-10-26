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
use tokio::time::Duration;

use crate::brain::{Brain, Reply, Speakable, Utterance};
use crate::call_state::{Action, CallState, Input};
use crate::elevenlabs::{ElevenLabs, Transcript};
use crate::endpoint::TurnModel;
use crate::turn_taking::{Policy, TurnEvent, TurnTaker};
use crate::vad::{CallVad, VadModel};

/// How often a quiet call asks its brain for an unprompted reply.
const UNPROMPTED_POLL: Duration = Duration::from_secs(2);
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
    // Every task this call starts is aborted when the call ends, on any path:
    // a reply or writer left running would hold the client's socket open.
    let mut writer = AbortOnDrop(tokio::spawn(async move {
        while let Some(message) = outbox.recv().await {
            let message = match message {
                ToClient::Audio(pcm) => Message::Binary(pcm.into()),
                ToClient::Event(event) => Message::Text(event.to_string().into()),
            };
            if client_tx.send(message).await.is_err() {
                break;
            }
        }
    }));
    let event = |value: Value| ToClient::Event(value);
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
    let mut unprompted = tokio::time::interval(UNPROMPTED_POLL);
    // Last commit sent or answered: the deadline runs from here.
    let mut commit_progress = tokio::time::Instant::now();

    loop {
        let input = tokio::select! {
            message = client_rx.next() => {
                let Some(Ok(message)) = message else { break };
                let Message::Binary(pcm) = message else { continue };
                samples.clear();
                samples.extend(pcm.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0));
                transcriber.audio(&pcm).await?;
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
                    commit_progress = tokio::time::Instant::now();
                    state.on(Input::Committed(text))
                }
                Some(Transcript::Error(error)) => {
                    let _ = to_client.send(event(json!({ "type": "error", "message": error }))).await;
                    continue;
                }
                None => anyhow::bail!("speech-to-text stream closed"),
            },
            _ = tokio::time::sleep_until(commit_progress + COMMIT_DEADLINE), if state.pending().1 > 0 => {
                tracing::warn!(commits = state.pending().1, "speech-to-text never answered; treating as empty");
                commit_progress = tokio::time::Instant::now();
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
                let Some(reply) = brain.unprompted().await? else { continue };
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
                    commit_progress = tokio::time::Instant::now();
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
    }
    drop(speaking);
    drop(to_client);
    // The caller hung up: let what is already queued reach them.
    let _ = (&mut writer.0).await;
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
