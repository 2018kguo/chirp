//! Every interleaving, up to a depth, of what a caller, the transcription
//! service, replies and the brain can do, run through the production
//! `CallState`, with its invariants checked at every step:
//!
//! 1. no reply starts while the caller holds the floor;
//! 2. no speech is lost or duplicated: replies + pending + in-flight text
//!    equal everything said, in order;
//! 3. barge-in: speaking over a reply never stops it by itself, but real
//!    words (or an overlap gone on too long) always do, while a backchannel
//!    ("uh-huh") over a reply is never answered;
//! 4. once the caller is done and all transcripts are in, nothing is left
//!    unanswered.

use std::collections::{HashSet, VecDeque};

use chirp::call_state::{Action, CallState, Input};
use chirp::turn_taking::{EndedBy, TurnEvent};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Caller {
    Idle,
    Speaking,
    /// A few quiet windows in (Paused was reported).
    Paused,
    /// Smart Turn said "unfinished": waiting out the grace period.
    Grace,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct World {
    call: CallState,
    caller: Caller,
    /// Speech since the last commit (what the next commit will transcribe).
    unsent: bool,
    /// That speech is only a backchannel.
    unsent_backchannel: bool,
    /// The current speech started over a reply.
    over_reply: bool,
    /// The reply's playback is paused (Hold), awaiting Resume or Interrupt.
    held: bool,
    next_word: u32,
    /// Segments committed but not yet delivered, in order.
    in_flight: VecDeque<String>,
    /// Everything the caller said, as segments, in order.
    said: Vec<String>,
    /// Everything replied to, in order.
    replied: Vec<String>,
}

/// What can happen next, as (input to the call, caller's next phase).
/// Whether a move starts speech, and what kind.
#[derive(Clone, Copy, PartialEq)]
enum Speech {
    None,
    Words,
    Backchannel,
}

/// What the transcriber makes of the unsent speech.
fn unsent_text(world: &World) -> String {
    match (world.unsent_backchannel, world.over_reply) {
        (true, true) => "uh-huh".into(),
        (true, false) => "yeah".into(),
        _ => format!("w{}", world.next_word + 1),
    }
}

fn moves(world: &World) -> Vec<(Input, Caller, Speech)> {
    use Caller::*;
    let mut next = Vec::new();
    let turn = |event| Input::Turn(event);
    match world.caller {
        Idle => {
            next.push((turn(TurnEvent::SpeechStarted), Speaking, Speech::Words));
            next.push((
                turn(TurnEvent::SpeechStarted),
                Speaking,
                Speech::Backchannel,
            ));
        }
        // The VAD can also call the pause before a `Paused` (reporting off,
        // or set later than the VAD's stop).
        Speaking => {
            next.push((turn(TurnEvent::Paused), Paused, Speech::None));
            next.push((
                turn(TurnEvent::Ended {
                    by: EndedBy::Fallback,
                }),
                Idle,
                Speech::None,
            ));
            next.push((
                turn(TurnEvent::Assessed { probability: 0.2 }),
                Grace,
                Speech::None,
            ));
        }
        Paused => {
            // Speaking again before the VAD calls the pause.
            next.push((turn(TurnEvent::SpeechStarted), Speaking, Speech::Words));
            next.push((
                turn(TurnEvent::Ended { by: EndedBy::Model }),
                Idle,
                Speech::None,
            ));
            next.push((
                turn(TurnEvent::Assessed { probability: 0.2 }),
                Grace,
                Speech::None,
            ));
        }
        Grace => {
            next.push((turn(TurnEvent::SpeechStarted), Speaking, Speech::Words));
            next.push((
                turn(TurnEvent::Ended { by: EndedBy::Grace }),
                Idle,
                Speech::None,
            ));
        }
    }
    if let Some(text) = world.in_flight.front() {
        next.push((Input::Committed(text.clone()), world.caller, Speech::None));
    }
    if world.call.speaking() {
        next.push((Input::ReplyFinished, world.caller, Speech::None));
    }
    next.push((Input::UnpromptedReady, world.caller, Speech::None));
    // The transcriber's partial of what's unsent, and the overlap timer.
    if world.unsent {
        next.push((Input::Heard(unsent_text(world)), world.caller, Speech::None));
    }
    if world.call.overlapping() {
        next.push((Input::OverlapTimedOut, world.caller, Speech::None));
    }
    next
}

fn step(world: &World, input: Input, caller: Caller, speech: Speech) -> Result<World, String> {
    let mut next = world.clone();
    let was_speaking = next.call.speaking();
    let was_overlapping = next.call.overlapping();
    let holds_floor_after = caller != Caller::Idle;
    let actions = next.call.on(input.clone());
    next.caller = caller;
    if speech != Speech::None {
        // A segment is one kind: words spoken after a backchannel make it words.
        next.unsent_backchannel =
            (!next.unsent || next.unsent_backchannel) && speech == Speech::Backchannel;
        next.unsent = true;
        // A reply on hold hasn't been heard out: speech then is over it.
        next.over_reply = was_speaking || world.held;
    }
    if let Input::Committed(_) = input {
        next.in_flight.pop_front();
    }
    let interrupted = actions.contains(&Action::Interrupt);
    let held = actions.contains(&Action::Hold);
    let resumed = actions.contains(&Action::Resume);
    if held && !(was_speaking && speech != Speech::None) {
        return Err(format!(
            "held playback with no speech over a reply: {input:?}"
        ));
    }
    if resumed && !next.held {
        return Err(format!("resumed playback that wasn't held: {input:?}"));
    }
    if resumed && interrupted {
        return Err(format!("both resumed and interrupted on {input:?}"));
    }
    if held {
        next.held = true;
    }
    if interrupted || resumed {
        next.held = false;
    }
    if interrupted && !was_speaking && !world.held {
        return Err(format!("interrupted nothing on {input:?}"));
    }
    if interrupted && speech != Speech::None {
        return Err("speaking over a reply stopped it before any words".into());
    }
    let takes_turn = match &input {
        Input::Heard(text) | Input::Committed(text) => !chirp::call_state::is_backchannel(text),
        Input::OverlapTimedOut => world.caller == Caller::Speaking,
        _ => false,
    };
    if was_speaking && was_overlapping && takes_turn && !interrupted {
        return Err(format!(
            "the caller took the turn but the reply went on: {input:?}"
        ));
    }
    for action in actions {
        match action {
            Action::Commit => {
                let text = if next.unsent {
                    let text = unsent_text(&next);
                    if !next.unsent_backchannel {
                        next.next_word += 1;
                    }
                    next.said.push(text.clone());
                    text
                } else {
                    String::new()
                };
                next.unsent = false;
                next.unsent_backchannel = false;
                next.in_flight.push_back(text);
            }
            Action::Reply { text, .. } => {
                if holds_floor_after {
                    return Err(format!("replied {text:?} while the caller holds the floor"));
                }
                if text.split(' ').all(|w| w == "uh-huh") {
                    return Err(format!("answered a backchannel: {text:?}"));
                }
                next.replied.push(text);
            }
            _ => {}
        }
    }
    // No speech lost or duplicated (a backchannel over a reply may be let go).
    let kept = |w: &String| w != "uh-huh";
    let mut accounted: Vec<String> = next
        .replied
        .iter()
        .flat_map(|r| r.split(' ').map(str::to_string))
        .collect();
    let (pending, _) = next.call.pending();
    accounted.extend(
        pending
            .split(' ')
            .filter(|w| !w.is_empty())
            .map(str::to_string),
    );
    accounted.extend(next.in_flight.iter().filter(|w| !w.is_empty()).cloned());
    accounted.retain(kept);
    let said: Vec<String> = next.said.iter().filter(|w| kept(w)).cloned().collect();
    if accounted != said {
        return Err(format!("said {:?} but accounted {accounted:?}", next.said));
    }
    // Quiescent: caller done, nothing in flight → nothing left unanswered,
    // and no reply left paused.
    if next.caller == Caller::Idle && next.in_flight.is_empty() {
        if !next.call.pending().0.is_empty() {
            return Err(format!("left unanswered: {:?}", next.call.pending()));
        }
        if next.held {
            return Err("a reply was left on hold".into());
        }
    }
    Ok(next)
}

#[test]
fn every_interleaving_keeps_the_invariants() {
    // Speech kinds, partials and the overlap timer widened every step:
    // 22 is ~3M states (24 is 9M and passes too, in ~50 s); every bug the
    // held barge-in had was found within 8 steps.
    const DEPTH: usize = 22;
    let start = World {
        call: CallState::default(),
        caller: Caller::Idle,
        unsent: false,
        unsent_backchannel: false,
        over_reply: false,
        held: false,
        next_word: 0,
        in_flight: VecDeque::new(),
        said: Vec::new(),
        replied: Vec::new(),
    };
    let mut seen = HashSet::new();
    let mut frontier = vec![(start, Vec::<String>::new())];
    let mut explored = 0usize;
    for _ in 0..DEPTH {
        let mut next_frontier = Vec::new();
        for (world, trace) in frontier {
            let mut successors: Vec<(World, String)> = Vec::new();
            for (input, caller, speech) in moves(&world) {
                let label = format!("{input:?}");
                match step(&world, input, caller, speech) {
                    Ok(next) => successors.push((next, label)),
                    Err(error) => panic!("{error}\n  after: {trace:?}\n  then: {label}"),
                }
            }
            for (next, label) in successors {
                if seen.insert(next.clone()) {
                    explored += 1;
                    let mut trace = trace.clone();
                    trace.push(label);
                    next_frontier.push((next, trace));
                }
            }
        }
        frontier = next_frontier;
    }
    println!("explored {explored} distinct states to depth {DEPTH}");
    assert!(
        explored > 5_000,
        "the search should cover a real state space ({explored})"
    );
}
