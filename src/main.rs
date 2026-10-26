//! chirp server: `GET /v1/calls` upgrades to a call WebSocket (16 kHz mono
//! PCM16 both ways, JSON events as text). This binary answers with an
//! OpenAI-compatible chat model; to answer with something else, implement
//! `chirp::brain::Brain` and serve calls with `chirp::call::run`.

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result};
use chirp::call::{Shared, SpeechPolicy};
use chirp::elevenlabs::ElevenLabs;
use chirp::endpoint::TurnModel;
use chirp::openai::OpenAiBrain;
use chirp::vad::VadModel;

#[derive(Clone)]
struct App {
    shared: Shared,
    http: reqwest::Client,
    brain_url: String,
    brain_key: String,
    brain_model: String,
    system_prompt: Option<String>,
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let _ = rustls::crypto::ring::default_provider().install_default();
    let models = PathBuf::from(
        std::env::var("CHIRP_MODELS_DIR")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/models").into()),
    );
    let app = App {
        shared: Shared {
            vad: VadModel::spawn_threads(
                &models.join("silero_vad.onnx"),
                env_or("CHIRP_VAD_THREADS", 2),
            )?,
            turn: TurnModel::spawn_threads(
                &models.join("smart-turn-v3.2-cpu.onnx"),
                env_or(
                    "CHIRP_TURN_THREADS",
                    // Smart Turn is the throughput limit (~32 ms per decision);
                    // VAD needs about one thread per 600 calls.
                    std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).max(1)),
                ),
            )?,
            elevenlabs: ElevenLabs::from_env()?,
            speech: SpeechPolicy::default(),
            turn_policy: chirp::turn_taking::Policy::default(),
        },
        http: reqwest::Client::new(),
        brain_url: std::env::var("CHIRP_BRAIN_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".into()),
        brain_key: std::env::var("CHIRP_BRAIN_KEY").context("CHIRP_BRAIN_KEY")?,
        brain_model: std::env::var("CHIRP_BRAIN_MODEL").context("CHIRP_BRAIN_MODEL")?,
        system_prompt: std::env::var("CHIRP_SYSTEM_PROMPT").ok(),
    };
    let router = chirp::server::router(app.shared.clone(), move || {
        OpenAiBrain::new(
            app.http.clone(),
            &app.brain_url,
            &app.brain_key,
            &app.brain_model,
            app.system_prompt.as_deref(),
        )
    });
    let address: SocketAddr = ([0, 0, 0, 0], env_or("PORT", 7861u16)).into();
    tracing::info!(%address, "chirp listening");
    axum::serve(tokio::net::TcpListener::bind(address).await?, router).await?;
    Ok(())
}
