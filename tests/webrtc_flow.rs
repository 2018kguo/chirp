//! A call over WebRTC, end to end: `client::play_webrtc` stands in for the
//! browser (Opus, DTX, an `events` data channel) over a real UDP socket.

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
use chirp::webrtc::WebRtc;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

async fn chirp_webrtc(script: Script) -> String {
    chirp_webrtc_limited(script, usize::MAX).await
}

async fn chirp_webrtc_limited(script: Script, max_calls: usize) -> String {
    let mocks = mock::start(script).await.unwrap();
    let shared = Shared {
        vad: VadModel::spawn(&root().join("models/silero_vad.onnx")).unwrap(),
        turn: TurnModel::spawn_threads(&root().join("models/smart-turn-v3.2-cpu.onnx"), 1).unwrap(),
        elevenlabs: ElevenLabs::new("mock", format!("ws://{mocks}")).unwrap(),
        speech: SpeechPolicy::default(),
        turn_policy: Policy::default(),
        admission: chirp::admission::Admission::new(max_calls),
        webrtc: Some(
            WebRtc::start("127.0.0.1:0".parse().unwrap(), None, 2)
                .await
                .unwrap(),
        ),
    };
    let http = reqwest::Client::new();
    let brain_url = format!("http://{mocks}/v1");
    let router = chirp::server::router(shared, move |_| {
        Some(OpenAiBrain::new(
            http.clone(),
            &brain_url,
            "mock",
            "mock",
            None,
        ))
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1/webrtc", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await });
    url
}

/// Plays `audio` like a browser, listening until `seconds` into the call.
async fn play(url: &str, mut audio: Vec<i16>, seconds: usize) -> client::Outcome {
    audio.resize(seconds * 16_000, 0);
    client::play_webrtc(url, audio).await.unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_call_over_webrtc_is_heard_and_answered() {
    let script = Script::default();
    let url = chirp_webrtc(script.clone()).await;
    let question =
        client::read_wav(&root().join("tests/fixtures/turn/complete_question.wav")).unwrap();
    let heard = play(&url, question, 8).await;
    let kinds: Vec<_> = heard
        .events
        .iter()
        .filter_map(|(_, e)| e["type"].as_str())
        .collect();
    assert!(
        kinds.contains(&"ready"),
        "no ready event over the data channel: {kinds:?}"
    );
    let transcript = heard
        .events
        .iter()
        .find(|(_, e)| e["type"] == "transcript" && e["final"] == true)
        .unwrap_or_else(|| panic!("no final transcript: {kinds:?}"));
    assert_eq!(transcript.1["text"], script.transcript);
    // The mock speaks 100 ms of audio per text message; well over a second
    // of the reply must have been played back.
    assert!(
        heard.audio.len() > 16_000,
        "heard only {} samples of the reply",
        heard.audio.len()
    );
}

/// Barge-in over WebRTC: chirp paces the reply itself, so once the caller
/// speaks over it, no more of it may be played, including what was queued.
#[tokio::test(flavor = "multi_thread")]
async fn barge_in_over_webrtc_stops_the_reply_at_once() {
    let url = chirp_webrtc(Script {
        reply_words: 300,
        brain_token_gap: Duration::from_millis(2),
        ..Script::default()
    })
    .await;
    let question =
        client::read_wav(&root().join("tests/fixtures/turn/complete_question.wav")).unwrap();
    let mut audio = question.clone();
    audio.extend(std::iter::repeat_n(0i16, 1500 * 16));
    let second_start = audio.len();
    audio.extend(&question);
    let heard = play(&url, audio, 9).await;
    let interrupt = heard
        .events
        .iter()
        .find(|(_, e)| e["type"] == "interrupt")
        .map(|(at, _)| *at)
        .expect("an interrupt");
    let second_turn_end =
        Duration::from_millis(second_start as u64 / 16) + client::speech_end(&question);
    let late: Vec<u128> = heard
        .audio_at
        .iter()
        .filter(|at| **at > interrupt + Duration::from_millis(100) && **at < second_turn_end)
        .map(|at| (*at - interrupt).as_millis())
        .collect();
    assert!(
        heard.audio_at.iter().any(|at| *at < interrupt),
        "the first reply was never heard"
    );
    assert!(
        late.is_empty(),
        "reply audio kept playing after the interrupt: {late:?} ms"
    );
    // Between the two replies chirp was silent; the RTP clock kept real
    // time, so the browser's jitter buffer sees a pause, not late audio.
    let (wall, rtp) = heard
        .rtp
        .windows(2)
        .map(|pair| ((pair[1].0 - pair[0].0).as_secs_f64(), pair[1].1 - pair[0].1))
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .expect("reply packets");
    assert!(
        wall > 1.0,
        "expected a pause between replies, longest gap {wall:.2} s"
    );
    assert!(
        (rtp - wall).abs() < 0.3,
        "a {wall:.2} s pause advanced the RTP clock by {rtp:.2} s"
    );
}

/// Past the call limit, an offer is refused before any session opens.
#[tokio::test(flavor = "multi_thread")]
async fn an_offer_over_the_call_limit_is_refused() {
    let url = chirp_webrtc_limited(Script::default(), 0).await;
    let response = reqwest::Client::new()
        .post(&url)
        .body("v=0")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 503);
}
