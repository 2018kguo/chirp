//! Whole calls against the scripted mocks: what a caller hears, and when.
//! Each test pins a behavior that was once wrong.

use std::path::Path;
use std::time::Duration;

use chirp::call::{Shared, SpeechPolicy};
use chirp::client::{self, Outcome};
use chirp::elevenlabs::ElevenLabs;
use chirp::endpoint::TurnModel;
use chirp::mock::{self, Script};
use chirp::openai::OpenAiBrain;
use chirp::turn_taking::Policy;
use chirp::vad::VadModel;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// A chirp server on the given mocks; returns its call URL.
async fn chirp_on(script: Script) -> String {
    let mocks = mock::start(script).await.unwrap();
    let shared = Shared {
        vad: VadModel::spawn(&root().join("models/silero_vad.onnx")).unwrap(),
        turn: TurnModel::spawn_threads(&root().join("models/smart-turn-v3.2-cpu.onnx"), 1).unwrap(),
        elevenlabs: ElevenLabs {
            api_key: "mock".into(),
            voice_id: "mock".into(),
            tts_model: "mock".into(),
            stt_model: "mock".into(),
            base_url: format!("ws://{mocks}"),
        },
        speech: SpeechPolicy::default(),
        turn_policy: Policy::default(),
    };
    let http = reqwest::Client::new();
    let brain_url = format!("http://{mocks}/v1");
    let router = chirp::server::router(shared, move || {
        OpenAiBrain::new(http.clone(), &brain_url, "mock", "mock", None)
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/v1/calls", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await });
    url
}

fn question() -> Vec<i16> {
    client::read_wav(&root().join("tests/fixtures/turn/complete_question.wav")).unwrap()
}

fn silence(duration: Duration) -> Vec<i16> {
    vec![0; (duration.as_millis() * 16) as usize]
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn turn_end(outcome: &Outcome) -> Duration {
    outcome
        .event_at("turn_end", |_| true)
        .expect("the turn ended")
}

/// The transcript is committed speculatively at the first pause, so it is
/// already in hand when the turn is called: no commit round trip after it.
#[tokio::test(flavor = "multi_thread")]
async fn the_transcript_is_ready_when_the_turn_ends() {
    let script = Script {
        stt_commit: Duration::from_millis(100),
        ..Script::default()
    };
    let url = chirp_on(script.clone()).await;
    let outcome = client::call(&url, &question(), Duration::from_secs(2))
        .await
        .unwrap();
    let transcript = outcome
        .event_at("transcript", |e| e["final"] == true)
        .expect("final transcript");
    assert_eq!(
        outcome
            .events
            .iter()
            .find(|(_, e)| e["final"] == true)
            .unwrap()
            .1["text"],
        script.transcript
    );
    let wait = ms(transcript) - ms(turn_end(&outcome));
    assert!(
        wait < 50.0,
        "waited {wait:.0} ms for the transcript after the turn ended"
    );
}

/// Barge-in: once the caller speaks over a reply, no more of it is played,
/// including audio that was already queued for the caller.
#[tokio::test(flavor = "multi_thread")]
async fn barge_in_stops_the_reply_at_once() {
    // A long reply: text arrives far faster than it would be spoken, so
    // plenty of audio is queued when the caller interrupts.
    let url = chirp_on(Script {
        reply_words: 300,
        brain_token_gap: Duration::from_millis(2),
        ..Script::default()
    })
    .await;
    let mut audio = question();
    audio.extend(silence(Duration::from_millis(1500)));
    let second_start = audio.len();
    audio.extend(question());
    audio.extend(silence(Duration::from_secs(1)));
    let outcome = client::play(&url, audio).await.unwrap();
    let interrupt = outcome
        .event_at("interrupt", |_| true)
        .expect("an interrupt");
    let second_turn_end =
        Duration::from_millis(second_start as u64 / 16) + client::speech_end(&question());
    let late: Vec<f64> = outcome
        .audio_at
        .iter()
        .filter(|at| **at > interrupt + Duration::from_millis(100) && **at < second_turn_end)
        .map(|at| ms(*at) - ms(interrupt))
        .collect();
    assert!(
        late.is_empty(),
        "reply audio kept arriving after the interrupt: {late:?} ms"
    );
}

/// A provider failing mid-call ends the call and closes the caller's socket
/// at once, even with a reply still waiting on the brain.
#[tokio::test(flavor = "multi_thread")]
async fn a_failing_provider_closes_the_call_promptly() {
    let script = Script {
        brain_first_token: Duration::from_secs(4),
        stt_drop_after_commit: Some(Duration::from_millis(600)),
        ..Script::default()
    };
    let url = chirp_on(script).await;
    let mut audio = question();
    audio.extend(silence(Duration::from_secs(8)));
    let outcome = client::play(&url, audio).await.unwrap();
    let after_turn = ms(outcome.closed) - ms(turn_end(&outcome));
    assert!(
        outcome.closed < outcome.hung_up && after_turn < 1500.0,
        "the call stayed open {after_turn:.0} ms after its turn ended (the brain needed 4,000)"
    );
}
