//! `tests/call_state_model.rs` checks the call's decisions over every order
//! of turn events a caller can produce. This checks the other half: that the
//! real `TurnTaker`, on real audio, only produces orders from that grammar.

use std::path::Path;
use std::time::Duration;

use chirp::client::read_wav;
use chirp::endpoint::TurnModel;
use chirp::turn_taking::{EndedBy, Policy, TurnEvent, TurnTaker};
use chirp::vad::{CallVad, VadModel};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Caller {
    Idle,
    Speaking,
    Paused,
    /// Smart Turn said "unfinished": waiting out the grace period.
    Grace,
    /// Smart Turn said "finished": `Ended { by: Model }` comes next.
    Finished,
}

/// The model's grammar (see `moves` in call_state_model.rs), as a transition
/// function; `None` is an order the model never explored.
fn next(caller: Caller, event: &TurnEvent, complete: f32) -> Option<Caller> {
    use Caller::*;
    Some(match (caller, event) {
        (Idle | Paused | Grace, TurnEvent::SpeechStarted) => Speaking,
        (Speaking, TurnEvent::Paused) => Paused,
        (Speaking | Paused, TurnEvent::Assessed { probability }) if *probability > complete => {
            Finished
        }
        (Speaking | Paused, TurnEvent::Assessed { .. }) => Grace,
        (
            Speaking | Paused,
            TurnEvent::Ended {
                by: EndedBy::Fallback,
            },
        ) => Idle,
        (Finished, TurnEvent::Ended { by: EndedBy::Model }) => Idle,
        (Grace, TurnEvent::Ended { by: EndedBy::Grace }) => Idle,
        _ => return None,
    })
}

fn silence(duration: Duration) -> Vec<f32> {
    vec![0.0; (duration.as_secs_f64() * 16_000.0) as usize]
}

#[tokio::test(flavor = "multi_thread")]
async fn turn_events_stay_in_the_checked_grammar() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let vad = VadModel::spawn_threads(&root.join("models/silero_vad.onnx"), 1).unwrap();
    let turn = TurnModel::spawn_threads(&root.join("models/smart-turn-v3.2-cpu.onnx"), 2).unwrap();
    let clip = |name: &str| -> Vec<f32> {
        read_wav(&root.join("tests/fixtures/turn").join(name))
            .unwrap()
            .into_iter()
            .map(|s| s as f32 / 32768.0)
            .collect()
    };
    let (question, unfinished) = (
        clip("complete_question.wav"),
        clip("incomplete_mid_sentence.wav"),
    );
    // Gaps around the pause report (64 ms), the VAD's stop (192 ms) and the
    // grace period, so the caller resumes in each phase.
    let gaps = [100, 150, 250, 400, 900];
    let mut calls = Vec::new();
    for first in [&question, &unfinished] {
        for gap in gaps {
            let mut audio = first.clone();
            audio.extend(silence(Duration::from_millis(gap)));
            audio.extend(&question);
            audio.extend(silence(Duration::from_secs(3)));
            calls.push((format!("gap {gap} ms"), audio));
        }
    }
    let mut checked = 0;
    for after in [0, 1, 2, 5, 8] {
        let policy = Policy {
            pause_after_quiet_windows: after,
            grace: Duration::from_millis(600),
            ..Policy::default()
        };
        for (name, audio) in &calls {
            let mut taker = TurnTaker::new(CallVad::new(vad.clone()), turn.clone(), policy.clone());
            let mut caller = Caller::Idle;
            let mut seen = Vec::new();
            for frame in audio.chunks(320) {
                for event in taker.push(frame).await.unwrap() {
                    seen.push(event.clone());
                    caller = next(caller, &event, policy.complete).unwrap_or_else(|| {
                        panic!("pause after {after}, {name}: {event:?} from {caller:?}; events: {seen:?}")
                    });
                }
            }
            assert_eq!(
                caller,
                Caller::Idle,
                "pause after {after}, {name}: the turn never ended: {seen:?}"
            );
            checked += seen.len();
        }
    }
    eprintln!("{checked} events checked");
}
