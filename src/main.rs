//! chirp server: `GET /v1/calls` upgrades to a call WebSocket (16 kHz mono
//! PCM16 both ways, JSON events as text). This binary answers with an
//! OpenAI-compatible chat model; to answer with something else, implement
//! `chirp::brain::Brain` and run `chirp::server::serve`.

use anyhow::{Context, Result};
use chirp::openai::OpenAiBrain;

// The allocator: jemalloc on Linux (measured lowest RSS per call), the
// system's elsewhere, mimalloc by feature; dhat or a live-byte counter
// (wrapping the chosen one) for measurements.
#[cfg(feature = "mimalloc")]
type Allocator = mimalloc::MiMalloc;
#[cfg(feature = "mimalloc")]
const ALLOCATOR: Allocator = mimalloc::MiMalloc;
#[cfg(all(target_os = "linux", not(feature = "mimalloc")))]
type Allocator = tikv_jemallocator::Jemalloc;
#[cfg(all(target_os = "linux", not(feature = "mimalloc")))]
const ALLOCATOR: Allocator = tikv_jemallocator::Jemalloc;
#[cfg(not(any(target_os = "linux", feature = "mimalloc")))]
type Allocator = std::alloc::System;
#[cfg(not(any(target_os = "linux", feature = "mimalloc")))]
const ALLOCATOR: Allocator = std::alloc::System;

#[cfg(feature = "heap-stats")]
#[global_allocator]
static GLOBAL: chirp::heap::Counting<Allocator> = chirp::heap::Counting(ALLOCATOR);

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static GLOBAL: dhat::Alloc = dhat::Alloc;

#[cfg(not(any(feature = "heap-stats", feature = "dhat-heap")))]
#[global_allocator]
static GLOBAL: Allocator = ALLOCATOR;

#[tokio::main]
async fn main() -> Result<()> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let http = reqwest::Client::new();
    let url =
        std::env::var("CHIRP_BRAIN_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into());
    let key = std::env::var("CHIRP_BRAIN_KEY").context("CHIRP_BRAIN_KEY")?;
    let model = std::env::var("CHIRP_BRAIN_MODEL").context("CHIRP_BRAIN_MODEL")?;
    let system_prompt = std::env::var("CHIRP_SYSTEM_PROMPT").ok();
    chirp::server::serve(move || {
        OpenAiBrain::new(http.clone(), &url, &key, &model, system_prompt.as_deref())
    })
    .await
}
