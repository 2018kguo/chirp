//! Silero VAD for every call in the process, from one ONNX session.
//!
//! The model is stateless apart from what each call carries (64 samples of
//! context and a `[2, 128]` recurrent state), so calls share one session and
//! a few inference threads. A thread takes whatever windows are queued when it
//! wakes and scores them in one batched run: an idle server scores a lone
//! window at once, a busy one amortises the call overhead across hundreds.
//! One thread keeps up with about 650 calls (`vad_load`); add threads for more.
//!
//! Framing, context, state and the reset policy follow Pipecat's `silero.py`
//! (the reference this crate is checked against).

use std::path::Path;
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::TensorRef;
use tokio::sync::Notify;

pub const SAMPLE_RATE: usize = 16_000;
/// Samples the model scores at a time (32 ms at 16 kHz).
pub const WINDOW: usize = 512;
/// Samples of the previous window prepended to each one.
const CONTEXT: usize = 64;
const STATE: usize = 2 * 128;
/// Cap on one batched run; more queued windows wait for the next.
const MAX_BATCH: usize = 256;

/// `CHIRP_VAD_MAX_BATCH` overrides [`MAX_BATCH`] (experiments).
fn max_batch() -> usize {
    std::env::var("CHIRP_VAD_MAX_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_BATCH)
}

/// Pipecat `VADParams` defaults.
pub const CONFIDENCE: f32 = 0.7;
const START_WINDOWS: u32 = 6; // round(0.2 s / 32 ms)
const STOP_WINDOWS: u32 = 6;
/// Pipecat's model reset: after 2 s of audio, on a clearly quiet stretch.
const RESET_INTERVAL_WINDOWS: u32 = (2 * SAMPLE_RATE / WINDOW) as u32;
const RESET_MAX_CONFIDENCE: f32 = 0.1;
const RESET_LOW_WINDOWS: u32 = 5;

/// A window's confidence and the call's next recurrent state.
type Scored = (f32, [f32; STATE]);

/// Where a call's scored windows land: one per call, made once, reused for
/// every window (a channel would preallocate blocks of 32 ~1 KB slots).
#[derive(Default)]
struct Reply {
    /// The window's confidence and the call's next state; `None` if its batch failed.
    scored: Mutex<Option<Option<Scored>>>,
    ready: Notify,
}

impl Reply {
    fn put(&self, scored: Option<Scored>) {
        if let Ok(mut slot) = self.scored.lock() {
            *slot = Some(scored);
        }
        self.ready.notify_one();
    }
}

struct Request {
    input: [f32; CONTEXT + WINDOW],
    state: [f32; STATE],
    reply: Arc<Reply>,
}

/// Cheap to clone; every call holds one.
#[derive(Clone)]
pub struct VadModel {
    requests: std_mpsc::Sender<Request>,
}

impl VadModel {
    /// Load the model and start one inference thread.
    pub fn spawn(model_path: &Path) -> Result<Self> {
        Self::spawn_threads(model_path, 1)
    }

    /// Load the model once per thread; the threads share one queue.
    pub fn spawn_threads(model_path: &Path, threads: usize) -> Result<Self> {
        let (requests, queue) = std_mpsc::channel::<Request>();
        let queue = Arc::new(Mutex::new(queue));
        for index in 0..threads.max(1) {
            let session = crate::onnx::session_dynamic_batch(model_path)?;
            let queue = queue.clone();
            thread::Builder::new()
                .name(format!("vad-{index}"))
                .spawn(move || run_batches(session, &queue))?;
        }
        Ok(Self { requests })
    }
}

/// Batch buffers, kept by each VAD thread and reused for every batch.
#[derive(Default)]
struct Scratch {
    input: Vec<f32>,
    state: Vec<f32>,
}

fn run_batches(mut session: Session, queue: &Mutex<std_mpsc::Receiver<Request>>) {
    let max_batch = max_batch();
    let mut batch: Vec<Request> = Vec::with_capacity(max_batch);
    let mut scratch = Scratch::default();
    loop {
        {
            // Hold the queue only while taking a batch, not while scoring it.
            let Ok(queue) = queue.lock() else { return };
            let Ok(first) = queue.recv() else { return };
            batch.push(first);
            while batch.len() < max_batch {
                match queue.try_recv() {
                    Ok(next) => batch.push(next),
                    Err(_) => break,
                }
            }
        }
        if let Err(error) = score_batch(&mut session, &mut batch, &mut scratch) {
            tracing::error!(%error, size = batch.len(), "VAD batch failed");
            for request in batch.drain(..) {
                request.reply.put(None);
            }
        }
        batch.clear();
    }
}

fn score_batch(
    session: &mut Session,
    batch: &mut Vec<Request>,
    scratch: &mut Scratch,
) -> Result<()> {
    let n = batch.len();
    let Scratch { input, state } = scratch;
    input.clear();
    // The model's state is [2, batch, 128]: layer-major, so gather per layer.
    state.clear();
    state.resize(n * STATE, 0.0);
    for (i, request) in batch.iter().enumerate() {
        input.extend_from_slice(&request.input);
        for layer in 0..2 {
            let to = (layer * n + i) * 128;
            state[to..to + 128].copy_from_slice(&request.state[layer * 128..(layer + 1) * 128]);
        }
    }
    let outputs = session.run(ort::inputs![
        "input" => TensorRef::from_array_view(([n, CONTEXT + WINDOW], &input[..]))?,
        "state" => TensorRef::from_array_view(([2usize, n, 128], &state[..]))?,
        "sr" => TensorRef::from_array_view(([0usize; 0], &[SAMPLE_RATE as i64][..]))?,
    ])?;
    let (_, probabilities) = outputs["output"].try_extract_tensor::<f32>()?;
    let (_, next_state) = outputs["stateN"].try_extract_tensor::<f32>()?;
    for (i, request) in batch.drain(..).enumerate() {
        let mut state = [0f32; STATE];
        for layer in 0..2 {
            let from = (layer * n + i) * 128;
            state[layer * 128..(layer + 1) * 128].copy_from_slice(&next_state[from..from + 128]);
        }
        request.reply.put(Some((probabilities[i], state)));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speech {
    Started,
    Stopped,
}

/// One call's view of the shared model: its buffered audio, context, state,
/// and whether the user is speaking.
pub struct CallVad {
    model: VadModel,
    reply: Arc<Reply>,
    pending: Vec<f32>,
    context: [f32; CONTEXT],
    state: [f32; STATE],
    speaking: bool,
    run: u32,
    windows_since_reset: u32,
    low_windows: u32,
    resets: bool,
    last_confidence: f32,
}

impl CallVad {
    pub fn new(model: VadModel) -> Self {
        Self {
            model,
            reply: Arc::default(),
            pending: Vec::with_capacity(2 * WINDOW),
            context: [0.0; CONTEXT],
            state: [0.0; STATE],
            speaking: false,
            run: 0,
            windows_since_reset: 0,
            low_windows: 0,
            resets: true,
            last_confidence: 0.0,
        }
    }

    /// The raw model, as Pipecat's `SileroOnnxModel` scores it (parity tests).
    pub fn without_model_resets(mut self) -> Self {
        self.resets = false;
        self
    }

    /// The model's speech confidence for the last window scored.
    pub fn last_confidence(&self) -> f32 {
        self.last_confidence
    }

    pub fn speaking(&self) -> bool {
        self.speaking
    }

    /// Feed 16 kHz mono samples in [-1, 1]; returns speech transitions, if any.
    pub async fn push(&mut self, samples: &[f32]) -> Result<Option<Speech>> {
        self.pending.extend_from_slice(samples);
        let mut transition = None;
        while self.pending.len() >= WINDOW {
            let mut input = [0f32; CONTEXT + WINDOW];
            input[..CONTEXT].copy_from_slice(&self.context);
            input[CONTEXT..].copy_from_slice(&self.pending[..WINDOW]);
            self.pending.drain(..WINDOW);
            self.model
                .requests
                .send(Request {
                    input,
                    state: self.state,
                    reply: self.reply.clone(),
                })
                .map_err(|_| anyhow::anyhow!("VAD thread stopped"))?;
            // One window in flight per call: wait for its answer.
            let scored = loop {
                if let Some(scored) = self
                    .reply
                    .scored
                    .lock()
                    .ok()
                    .and_then(|mut slot| slot.take())
                {
                    break scored;
                }
                self.reply.ready.notified().await;
            };
            let (confidence, state) = scored.context("VAD batch failed")?;
            self.context.copy_from_slice(&input[WINDOW..]);
            self.state = state;
            self.last_confidence = confidence;
            if let Some(change) = self.observe(confidence) {
                transition = Some(change);
            }
        }
        Ok(transition)
    }

    /// Pipecat's start/stop debouncing, then its model-reset policy.
    fn observe(&mut self, confidence: f32) -> Option<Speech> {
        let voiced = confidence >= CONFIDENCE;
        let mut change = None;
        if voiced != self.speaking {
            self.run += 1;
            let needed = if voiced { START_WINDOWS } else { STOP_WINDOWS };
            if self.run >= needed {
                self.speaking = voiced;
                self.run = 0;
                change = Some(if voiced {
                    Speech::Started
                } else {
                    Speech::Stopped
                });
            }
        } else {
            self.run = 0;
        }
        self.windows_since_reset += 1;
        self.low_windows = if confidence < RESET_MAX_CONFIDENCE {
            self.low_windows + 1
        } else {
            0
        };
        if self.resets
            && self.windows_since_reset >= RESET_INTERVAL_WINDOWS
            && !self.speaking
            && self.low_windows >= RESET_LOW_WINDOWS
        {
            self.context = [0.0; CONTEXT];
            self.state = [0.0; STATE];
            self.windows_since_reset = 0;
        }
        change
    }
}
