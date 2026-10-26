//! One ONNX Runtime environment for the process, and the one way this crate
//! opens a model: single-threaded, because concurrency comes from running
//! many sessions' work on dedicated threads, not from intra-op threads.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use ort::environment::Environment;
use ort::session::builder::AutoDevicePolicy;
use ort::session::Session;

pub fn session(model_path: &Path) -> Result<Session> {
    // A Mutex, not a OnceLock: two threads building the environment at once
    // make the second `build` fail ("only one environment").
    static ENVIRONMENT: Mutex<Option<Environment>> = Mutex::new(None);
    let mut guard = ENVIRONMENT
        .lock()
        .map_err(|_| anyhow!("ONNX environment lock poisoned"))?;
    if guard.is_none() {
        *guard = Some(
            ort::init()
                .with_name("chirp")
                .build()
                .map_err(|e| anyhow!("{e}"))?,
        );
    }
    let environment = guard.as_ref().expect("set above");
    Session::builder(environment)
        .map_err(|e| anyhow!("{e}"))?
        // ort's default policy registers CoreML on macOS, whose fp16 outputs
        // differ from the CPU kernels production (Linux) runs.
        .with_auto_device(AutoDevicePolicy::PreferCPU)
        .map_err(|e| anyhow!("{e}"))?
        .with_intra_threads(1)
        .map_err(|e| anyhow!("{e}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("loading {}: {e}", model_path.display()))
}
