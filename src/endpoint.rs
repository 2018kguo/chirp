//! Smart Turn end-of-turn decisions for every call, on a small thread pool.
//! A decision is ~40 ms of CPU and happens only when the user pauses, so a
//! couple of threads serve many calls; each thread owns its own session.

use std::path::Path;
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{anyhow, Result};
use tokio::sync::oneshot;

use crate::turn::SmartTurn;

type Request = (Vec<f32>, oneshot::Sender<Result<f32>>);

#[derive(Clone)]
pub struct TurnModel {
    requests: std_mpsc::Sender<Request>,
}

impl TurnModel {
    pub fn spawn_threads(model_path: &Path, threads: usize) -> Result<Self> {
        let (requests, queue) = std_mpsc::channel::<Request>();
        let queue = Arc::new(Mutex::new(queue));
        for index in 0..threads.max(1) {
            let mut model = SmartTurn::new(model_path)?;
            let queue = queue.clone();
            thread::Builder::new()
                .name(format!("turn-{index}"))
                .spawn(move || loop {
                    let request = match queue.lock() {
                        Ok(queue) => queue.recv(),
                        Err(_) => return,
                    };
                    let Ok((audio, reply)) = request else { return };
                    let _ = reply.send(model.predict(&audio));
                })?;
        }
        Ok(Self { requests })
    }

    /// Probability that the user has finished their turn, from up to the last
    /// 8 s of their speech (16 kHz mono).
    pub async fn complete(&self, audio: Vec<f32>) -> Result<f32> {
        let (reply, answer) = oneshot::channel();
        self.requests
            .send((audio, reply))
            .map_err(|_| anyhow!("turn threads stopped"))?;
        answer
            .await
            .map_err(|_| anyhow!("turn thread dropped a request"))?
    }
}
