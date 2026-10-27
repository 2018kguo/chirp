//! The two ONNX models chirp runs, pinned by URL and SHA-256: Silero VAD as
//! Pipecat 1.13 vendors it (the parity reference) and Smart Turn v3.2 (int8).

use std::path::Path;

use anyhow::{bail, Context, Result};

pub const VAD: &str = "silero_vad.onnx";
pub const TURN: &str = "smart-turn-v3.2-cpu.onnx";

const PINNED: [(&str, &str, &str); 2] = [
    (
        VAD,
        "https://raw.githubusercontent.com/pipecat-ai/pipecat/v1.13.0/src/pipecat/audio/vad/data/silero_vad.onnx",
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    ),
    (
        TURN,
        "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/f766f81d3cfdf7737ac64aad813d91bbfd56bf93/smart-turn-v3.2-cpu.onnx",
        "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
    ),
];

/// Downloads each model `dir` lacks, or holds with the wrong checksum.
pub async fn ensure(dir: &Path) -> Result<()> {
    tokio::fs::create_dir_all(dir).await?;
    for (file, url, sha256) in PINNED {
        let path = dir.join(file);
        if tokio::fs::read(&path)
            .await
            .is_ok_and(|bytes| hex_sha256(&bytes) == sha256)
        {
            continue;
        }
        tracing::info!(%url, "downloading model");
        let bytes = reqwest::get(url)
            .await
            .and_then(reqwest::Response::error_for_status)
            .with_context(|| url.to_string())?
            .bytes()
            .await?;
        if hex_sha256(&bytes) != sha256 {
            bail!("checksum mismatch: {url}");
        }
        // Written aside and renamed, so a crash never leaves half a model.
        let part = path.with_extension("part");
        tokio::fs::write(&part, &bytes).await?;
        tokio::fs::rename(&part, &path).await?;
    }
    Ok(())
}

fn hex_sha256(bytes: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, bytes)
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
