//! Past capacity, calls degrade instead of wedging: an end-of-turn decision
//! that cannot be made in time falls back to the VAD, every call still sees
//! its caller hang up at once, and new calls are refused while the pool is
//! behind. Reproduces the 1,500-call run where 515 calls never closed.

use std::path::Path;
use std::time::Duration;

use chirp::call::{Shared, SpeechPolicy};
use chirp::client;
use chirp::elevenlabs::ElevenLabs;
use chirp::endpoint::TurnModel;
use chirp::mock::{self, Script};
use chirp::openai::OpenAiBrain;
use chirp::turn_taking::Policy;
use chirp::vad::VadModel;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn overloaded_calls_degrade_and_close_promptly() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mocks = mock::start(Script::default()).await.unwrap();
    let shared = Shared {
        vad: VadModel::spawn_threads(&root.join("models/silero_vad.onnx"), 2).unwrap(),
        // One thread for 120 simultaneous pauses: ~3.8 s of queued decisions.
        turn: TurnModel::spawn_threads(&root.join("models/smart-turn-v3.2-cpu.onnx"), 1).unwrap(),
        elevenlabs: ElevenLabs::new("mock", format!("ws://{mocks}")).unwrap(),
        speech: SpeechPolicy::default(),
        turn_policy: Policy::default(),
        webrtc: None,
        admission: chirp::admission::Admission::unlimited(),
    };
    let turn = shared.turn.clone();
    let http = reqwest::Client::new();
    let brain_url = format!("http://{mocks}/v1");
    let router = chirp::server::router(shared, move || {
        OpenAiBrain::new(http.clone(), &brain_url, "mock", "mock", None)
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(chirp::server::nodelay(listener), router).await });
    let url = format!("ws://{address}/v1/calls");

    let mut audio =
        client::read_wav(&root.join("tests/fixtures/turn/complete_question.wav")).unwrap();
    audio.extend(vec![0i16; 16 * 600]); // hang up 0.6 s after speaking
    let calls: Vec<_> = (0..120)
        .map(|_| {
            let (url, audio) = (url.clone(), audio.clone());
            tokio::spawn(async move { client::play(&url, audio).await })
        })
        .collect();
    let mut fallbacks = 0;
    let mut slowest_close = Duration::ZERO;
    for call in calls {
        let outcome = call.await.unwrap().unwrap();
        slowest_close = slowest_close.max(outcome.closed.saturating_sub(outcome.hung_up));
        fallbacks += outcome
            .events
            .iter()
            .filter(|(_, e)| e["type"] == "turn_end" && e["by"] == "Fallback")
            .count();
    }
    assert!(
        slowest_close < Duration::from_millis(600),
        "a call stayed open {slowest_close:?} after its caller hung up"
    );
    assert!(fallbacks > 0, "no turn fell back to the VAD under overload");
    assert!(
        turn.overloaded(),
        "the pool should report it is behind right after the burst"
    );
    let refused = tokio_tungstenite::connect_async(url.as_str()).await;
    assert!(
        refused.is_err(),
        "a new call was accepted while the pool was behind"
    );
}
