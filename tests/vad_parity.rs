//! Calls that share the batched VAD score exactly as Pipecat's Python model
//! scores each signal alone: batching must never mix one call's state into
//! another's. Reference: scripts/vad_reference.py.

use std::path::{Path, PathBuf};

use chirp::vad::{CallVad, VadModel, WINDOW};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/vad")
        .join(name)
}

fn read_pcm(name: &str) -> Vec<f32> {
    std::fs::read(fixture(&format!("{name}.pcm")))
        .unwrap()
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect()
}

fn read_probs(name: &str) -> Vec<f32> {
    std::fs::read(fixture(&format!("{name}.probs.f32")))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_calls_match_the_reference_window_for_window() {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/silero_vad.onnx");
    let model = VadModel::spawn(&model_path).unwrap();
    let names = ["speech", "reversed", "later", "silence"];
    // Several calls per signal, all at once, so windows of different calls
    // land in the same batch.
    let mut calls = Vec::new();
    for copy in 0..8 {
        for name in names {
            let model = model.clone();
            calls.push(tokio::spawn(async move {
                let audio = read_pcm(name);
                let mut vad = CallVad::new(model).without_model_resets();
                let mut scored = Vec::new();
                // 20 ms frames, as a client sends them, not window-aligned.
                for frame in audio.chunks(320) {
                    vad.push(frame).await.unwrap();
                    scored.push(vad.last_confidence());
                }
                (name, copy, scored)
            }));
        }
    }
    for call in calls {
        let (name, copy, per_frame) = call.await.unwrap();
        let reference = read_probs(name);
        // A frame's last_confidence is the latest window completed by then;
        // pick the frames where a new window completed.
        let windows: Vec<f32> = (0..reference.len())
            .map(|w| {
                let frame = ((w + 1) * WINDOW).div_ceil(320) - 1;
                per_frame[frame]
            })
            .collect();
        let worst = windows
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // 1e-4, not exact: CI runs x86 kernels; a state mix-up differs by ~1.
        assert!(worst <= 1e-4, "{name} copy {copy}: max |diff| {worst}");
    }
}
