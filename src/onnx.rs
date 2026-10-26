//! One ONNX Runtime environment for the process, and the one way this crate
//! opens a model: single-threaded, because concurrency comes from running
//! many sessions' work on dedicated threads, not from intra-op threads.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use ort::environment::Environment;
use ort::session::builder::AutoDevicePolicy;
use ort::session::Session;

/// Opens a model whose input shape never changes (Smart Turn: batch 1).
pub fn session(model_path: &Path) -> Result<Session> {
    open(model_path, false)
}

/// Opens a model run at varying batch sizes (the VAD). ONNX Runtime's arena
/// and memory pattern size themselves to the largest batch seen and keep
/// it, which grew resident memory with the call count; without them, a run's
/// activations are freed after it (PLAN.md findings).
pub fn session_dynamic_batch(model_path: &Path) -> Result<Session> {
    open(model_path, true)
}

fn open(model_path: &Path, dynamic_batch: bool) -> Result<Session> {
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
    let arena = !dynamic_batch;
    Session::builder(environment)
        .map_err(|e| anyhow!("{e}"))?
        // CPU only. ort's default device policy registers CoreML on macOS
        // (fp16 outputs that differ from the CPU kernels production runs), and
        // still does alongside an explicitly registered provider.
        .with_auto_device(AutoDevicePolicy::PreferCPU)
        .map_err(|e| anyhow!("{e}"))?
        .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(arena).build()])
        .map_err(|e| anyhow!("{e}"))?
        .with_memory_pattern(arena)
        .map_err(|e| anyhow!("{e}"))?
        .with_intra_threads(1)
        .map_err(|e| anyhow!("{e}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("loading {}: {e}", model_path.display()))
}
