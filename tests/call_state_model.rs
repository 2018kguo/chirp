//! Every interleaving, up to a depth, of what a caller, the transcription
//! service, replies and the brain can do, run through the production
//! `CallState`, with its invariants checked at every step:
//!
//! 1. no reply starts while the caller holds the floor;
//! 2. no speech is lost or duplicated: replies + pending + in-flight text
//!    equal everything said, in order;
//! 3. barge-in: an interrupt exactly when the caller speaks over a reply;
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
    next_word: u32,
    /// Segments committed but not yet delivered, in order.
    in_flight: VecDeque<String>,
    /// Everything the caller said, as segments, in order.
    said: Vec<String>,
    /// Everything replied to, in order.
    replied: Vec<String>,
}

/// What can happen next, as (input to the call, caller's next phase).
fn moves(world: &World) -> Vec<(Input, Caller, bool)> {
    use Caller::*;
    let mut next = Vec::new();
    let turn = |event| Input::Turn(event);
    match world.caller {
        Idle => next.push((turn(TurnEvent::SpeechStarted), Speaking, true)),
        // The VAD can also call the pause before a `Paused` (reporting off,
        // or set later than the VAD's stop).
        Speaking => {
            next.push((turn(TurnEvent::Paused), Paused, false));
            next.push((
                turn(TurnEvent::Ended {
                    by: EndedBy::Fallback,
                }),
                Idle,
                false,
            ));
            next.push((turn(TurnEvent::Assessed { probability: 0.2 }), Grace, false));
        }
        Paused => {
            // Speaking again before the VAD calls the pause.
            next.push((turn(TurnEvent::SpeechStarted), Speaking, true));
            next.push((turn(TurnEvent::Ended { by: EndedBy::Model }), Idle, false));
            next.push((turn(TurnEvent::Assessed { probability: 0.2 }), Grace, false));
        }
        Grace => {
            next.push((turn(TurnEvent::SpeechStarted), Speaking, true));
            next.push((turn(TurnEvent::Ended { by: EndedBy::Grace }), Idle, false));
        }
    }
    if let Some(text) = world.in_flight.front() {
        next.push((Input::Committed(text.clone()), world.caller, false));
    }
    if world.call.speaking() {
        next.push((Input::ReplyFinished, world.caller, false));
    }
    next.push((Input::UnpromptedReady, world.caller, false));
    next
}

fn step(world: &World, input: Input, caller: Caller, speech_starts: bool) -> Result<World, String> {
    let mut next = world.clone();
    let was_speaking = next.call.speaking();
    let holds_floor_after = caller != Caller::Idle;
    let actions = next.call.on(input.clone());
    next.caller = caller;
    if speech_starts {
        next.unsent = true;
    }
    if let Input::Committed(_) = input {
        next.in_flight.pop_front();
    }
    let interrupted = actions.contains(&Action::Interrupt);
    let barged_in = speech_starts && was_speaking;
    if interrupted != barged_in {
        return Err(format!(
            "interrupt {interrupted} but barge-in {barged_in} on {input:?}"
        ));
    }
    for action in actions {
        match action {
            Action::Commit => {
                let text = if next.unsent {
                    next.next_word += 1;
                    let text = format!("w{}", next.next_word);
                    next.said.push(text.clone());
                    text
                } else {
                    String::new()
                };
                next.unsent = false;
                next.in_flight.push_back(text);
            }
            Action::Reply { text, .. } => {
                if holds_floor_after {
                    return Err(format!("replied {text:?} while the caller holds the floor"));
                }
                next.replied.push(text);
            }
            _ => {}
        }
    }
    // No speech lost or duplicated.
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
    if accounted != next.said {
        return Err(format!("said {:?} but accounted {accounted:?}", next.said));
    }
    // Quiescent: caller done, nothing in flight → nothing left unanswered.
    if next.caller == Caller::Idle && next.in_flight.is_empty() && !next.call.pending().0.is_empty()
    {
        return Err(format!("left unanswered: {:?}", next.call.pending()));
    }
    Ok(next)
}

#[test]
fn every_interleaving_keeps_the_invariants() {
    const DEPTH: usize = 32;
    let start = World {
        call: CallState::default(),
        caller: Caller::Idle,
        unsent: false,
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
            for (input, caller, speech_starts) in moves(&world) {
                let label = format!("{input:?}");
                match step(&world, input, caller, speech_starts) {
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
