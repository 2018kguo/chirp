//! Smart Turn end-of-turn decisions for every call, on a small thread pool.
//! A decision is ~32 ms of CPU and happens only when a caller pauses, so a
//! few threads serve many calls; each thread owns its own session.
//!
//! Under overload a decision must not stall its call: each request carries a
//! deadline, expired requests are skipped rather than computed, and a missed
//! deadline reads as "no verdict" so the caller falls back to the VAD alone.
//! The pool's queue wait is tracked so the server can refuse new calls while
//! it is behind.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use tokio::sync::oneshot;

use crate::turn::SmartTurn;

/// How long a caller waits for a verdict before deciding without one.
pub const DECISION_DEADLINE: Duration = Duration::from_millis(250);
/// Queue wait (moving average) above which the pool counts as overloaded.
const OVERLOADED_WAIT: Duration = Duration::from_millis(100);

struct Request {
    audio: Vec<f32>,
    enqueued: Instant,
    deadline: Instant,
    reply: oneshot::Sender<Result<f32>>,
}

#[derive(Clone)]
pub struct TurnModel {
    requests: std_mpsc::Sender<Request>,
    /// Moving average of queue wait, in microseconds.
    wait_micros: Arc<AtomicU64>,
}

impl TurnModel {
    pub fn spawn_threads(model_path: &Path, threads: usize) -> Result<Self> {
        let (requests, queue) = std_mpsc::channel::<Request>();
        let queue = Arc::new(Mutex::new(queue));
        let wait_micros = Arc::new(AtomicU64::new(0));
        for index in 0..threads.max(1) {
            let mut model = SmartTurn::new(model_path)?;
            // One run at startup: the session sizes its buffers now, not on
            // some caller's first pause, and resident memory is all there
            // before the first call.
            model.predict(&[0.0; crate::vad::SAMPLE_RATE])?;
            let (queue, wait_micros) = (queue.clone(), wait_micros.clone());
            thread::Builder::new()
                .name(format!("turn-{index}"))
                .spawn(move || loop {
                    let request = match queue.lock() {
                        Ok(queue) => queue.recv(),
                        Err(_) => return,
                    };
                    let Ok(request) = request else { return };
                    let now = Instant::now();
                    let waited = now.duration_since(request.enqueued);
                    crate::metrics::TURN_QUEUE.record(waited);
                    let waited = waited.as_micros() as u64;
                    // Exponential moving average, weight 1/8; a lost race between
                    // threads only blurs the average.
                    let average = wait_micros.load(Ordering::Relaxed);
                    wait_micros.store(average - average / 8 + waited / 8, Ordering::Relaxed);
                    if now >= request.deadline || request.reply.is_closed() {
                        continue; // nobody is waiting for this verdict any more
                    }
                    let _ = request.reply.send(model.predict(&request.audio));
                })?;
        }
        Ok(Self {
            requests,
            wait_micros,
        })
    }

    /// Probability that the caller has finished their turn, from up to the
    /// last 8 s of their speech (16 kHz mono); `None` if no verdict came back
    /// within [`DECISION_DEADLINE`].
    pub async fn complete(&self, audio: Vec<f32>) -> Result<Option<f32>> {
        let (reply, answer) = oneshot::channel();
        let enqueued = Instant::now();
        let deadline = enqueued + DECISION_DEADLINE;
        self.requests
            .send(Request {
                audio,
                enqueued,
                deadline,
                reply,
            })
            .map_err(|_| anyhow!("turn threads stopped"))?;
        match tokio::time::timeout_at(deadline.into(), answer).await {
            Ok(Ok(verdict)) => verdict.map(Some),
            // Skipped as expired, or the deadline passed while it was queued.
            Ok(Err(_)) | Err(_) => Ok(None),
        }
    }

    /// Whether decisions are queueing long enough that new calls should wait.
    pub fn overloaded(&self) -> bool {
        Duration::from_micros(self.wait_micros.load(Ordering::Relaxed)) > OVERLOADED_WAIT
    }
}
