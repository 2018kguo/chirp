//! Whole calls against the scripted mocks: what a caller hears, and when.
//! Each test pins a behavior that was once wrong.

use std::path::Path;
use std::time::Duration;

use chirp::brain::{Brain, Reply, Utterance};
use chirp::call::{Shared, SpeechPolicy};
use chirp::client::{self, Outcome};
use chirp::elevenlabs::ElevenLabs;
use chirp::endpoint::TurnModel;
use chirp::mock::{self, Script};
use chirp::openai::OpenAiBrain;
use chirp::request::CallRequest;
use chirp::turn_taking::Policy;
use chirp::vad::VadModel;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// A chirp server on the given mocks; returns its call URL.
async fn chirp_on(script: Script) -> String {
    chirp_with(script, |_, brain| Some(brain)).await
}

/// As `chirp_on`, with the mock-backed brain wrapped (or the call refused)
/// by `wrap`.
async fn chirp_with<B: Brain>(
    script: Script,
    wrap: impl Fn(&CallRequest, OpenAiBrain) -> Option<B> + Clone + Send + Sync + 'static,
) -> String {
    let mocks = mock::start(script).await.unwrap();
    let shared = Shared {
        vad: VadModel::spawn(&root().join("models/silero_vad.onnx")).unwrap(),
        turn: TurnModel::spawn_threads(&root().join("models/smart-turn-v3.2-cpu.onnx"), 1).unwrap(),
        elevenlabs: ElevenLabs::new("mock", format!("ws://{mocks}")).unwrap(),
        speech: SpeechPolicy::default(),
        turn_policy: Policy::default(),
        webrtc: None,
        admission: chirp::admission::Admission::unlimited(),
    };
    let http = reqwest::Client::new();
    let brain_url = format!("http://{mocks}/v1");
    let router = chirp::server::router(shared, move |request| {
        wrap(
            request,
            OpenAiBrain::new(http.clone(), &brain_url, "mock", "mock", None),
        )
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/v1/calls", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(chirp::server::nodelay(listener), router).await });
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

/// A commit the transcriber never answers can't leave the turn unanswered:
/// after the deadline it counts as empty and the turn is still closed out.
#[tokio::test(flavor = "multi_thread")]
async fn an_unanswered_commit_does_not_leave_the_turn_hanging() {
    let script = Script {
        stt_unanswered_commits: usize::MAX,
        ..Script::default()
    };
    let url = chirp_on(script).await;
    let outcome = client::call(&url, &question(), Duration::from_secs(5))
        .await
        .unwrap();
    let transcript = outcome
        .event_at("transcript", |e| e["final"] == true)
        .expect("the turn was never closed out");
    let wait = ms(transcript) - ms(turn_end(&outcome));
    assert!(
        wait < 3_500.0,
        "closed out {wait:.0} ms after the turn ended"
    );
}

/// A brain whose unprompted poll always fails.
struct FailingPoll(OpenAiBrain);

impl Brain for FailingPoll {
    async fn reply(&self, utterance: Utterance) -> anyhow::Result<Reply> {
        self.0.reply(utterance).await
    }

    async fn unprompted(&self) -> anyhow::Result<Option<Reply>> {
        anyhow::bail!("network down")
    }
}

/// A failed unprompted poll is retried later; it doesn't end the call.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_unprompted_poll_does_not_end_the_call() {
    let url = chirp_with(Script::default(), |_, brain| Some(FailingPoll(brain))).await;
    // Quiet long enough for the poll to fail at least twice, then a question.
    let mut audio = silence(Duration::from_secs(5));
    audio.extend(question());
    audio.extend(silence(Duration::from_secs(3)));
    let outcome = client::play(&url, audio).await.unwrap();
    assert!(
        outcome.first_audio.is_some(),
        "the question after the failed polls went unanswered"
    );
}

/// Audio that queued while chirp was held up reaches speech-to-text in a few
/// messages, not one per frame: a flood of small chunks ends the session.
#[tokio::test(flavor = "multi_thread")]
async fn a_backlog_of_audio_is_sent_in_a_few_messages() {
    let script = Script {
        stt_max_messages_per_100ms: Some(20),
        ..Script::default()
    };
    let url = chirp_on(script.clone()).await;
    let mut audio = question();
    audio.extend(silence(Duration::from_secs(8)));
    // One second live, then the next six arrive at once.
    let outcome =
        client::play_with_backlog(&url, audio, Duration::from_secs(1)..Duration::from_secs(7))
            .await
            .unwrap();
    let errors: Vec<_> = outcome
        .events
        .iter()
        .filter(|(_, e)| e["type"] == "error")
        .map(|(_, e)| e["message"].to_string())
        .collect();
    assert!(errors.is_empty(), "the session failed: {errors:?}");
    let transcript = outcome
        .events
        .iter()
        .find(|(_, e)| e["type"] == "transcript" && e["final"] == true)
        .expect("a final transcript");
    assert_eq!(transcript.1["text"], script.transcript);
    assert!(outcome.first_audio.is_some(), "the turn was never answered");
}

/// A brain factory that refuses a call refuses it before it starts: no token,
/// no call; the right token, in the query as a browser sends it, gets one.
#[tokio::test(flavor = "multi_thread")]
async fn a_call_the_brain_factory_refuses_never_starts() {
    let url = chirp_with(Script::default(), |request, brain| {
        request.has_token("s3cret").then_some(brain)
    })
    .await;
    match tokio_tungstenite::connect_async(url.as_str()).await {
        Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
            assert_eq!(response.status(), 403)
        }
        other => panic!("a call without a token was not refused: {other:?}"),
    }
    let wrong = format!("{url}?token=nope");
    assert!(tokio_tungstenite::connect_async(wrong.as_str())
        .await
        .is_err());
    let right = format!("{url}?token=s3cret");
    tokio_tungstenite::connect_async(right.as_str())
        .await
        .expect("a call with the token");
}

/// Remembers whether each utterance continued the one before.
struct Recording(OpenAiBrain, std::sync::Arc<std::sync::Mutex<Vec<bool>>>);

impl Brain for Recording {
    async fn reply(&self, utterance: Utterance) -> anyhow::Result<Reply> {
        self.1.lock().unwrap().push(utterance.continues);
        self.0.reply(utterance).await
    }
}

/// Whether the second question continued the first, when the caller speaks
/// again `gap` after the first question (over a long reply).
async fn second_continues(gap: Duration) -> Vec<bool> {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let record = seen.clone();
    let url = chirp_with(
        Script {
            reply_words: 300,
            ..Script::default()
        },
        move |_, brain| Some(Recording(brain, record.clone())),
    )
    .await;
    let mut audio = question();
    audio.extend(silence(gap));
    audio.extend(question());
    audio.extend(silence(Duration::from_secs(2)));
    client::play(&url, audio).await.unwrap();
    let seen = seen.lock().unwrap().clone();
    seen
}

/// Talking over a reply before hearing a second of it continues the question
/// (the turn end was a pause); talking over it later starts a new one.
#[tokio::test(flavor = "multi_thread")]
async fn talking_over_a_reply_at_once_continues_the_question() {
    let at_once = second_continues(Duration::from_millis(1200)).await;
    assert_eq!(at_once, [false, true], "spoke again right after the pause");
    let later = second_continues(Duration::from_secs(4)).await;
    assert_eq!(later, [false, false], "spoke again after hearing the reply");
}
