//! What a call decides, separate from how it does I/O: a synchronous reducer
//! from inputs (turn-taking events, transcripts, reply lifecycle) to actions
//! (commit the transcript, interrupt, start or stop a reply). `call::run`
//! executes the actions; `tests/call_state_model.rs` runs this same code over
//! every interleaving of a bounded set of inputs and checks its invariants.

use crate::turn_taking::{EndedBy, TurnEvent};

#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    Turn(TurnEvent),
    /// A committed transcript segment arrived (in commit order).
    Committed(String),
    /// The reply being spoken finished on its own.
    ReplyFinished,
    /// The brain has an unprompted reply ready (polled while the call is quiet).
    UnpromptedReady,
    /// The transcriber's running text of the speech so far (a partial).
    Heard(String),
    /// Speech over a reply has gone on long enough to be a turn, words or not.
    OverlapTimedOut,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Commit the transcript segment so far.
    Commit,
    /// The caller spoke over a reply and took the turn: stop it and tell the
    /// client to flush.
    Interrupt,
    /// The caller started speaking over a reply: pause its playback (keep
    /// what's queued) until their words decide.
    Hold,
    /// That speech was only a backchannel or noise: play on from the pause.
    Resume,
    /// Start replying to this utterance (replacing any reply in progress).
    Reply { text: String, index: u32 },
    /// Speak the brain's unprompted reply.
    SpeakUnprompted,
    /// Tell the client the turn ended (and how).
    TurnEnded(EndedBy),
    /// Tell the client the final transcript of the turn being answered.
    FinalTranscript(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CallState {
    /// The turn's text so far, from segments committed at each pause.
    turn_text: String,
    commits_in_flight: u32,
    /// Speech since the last commit: the turn's end must commit again.
    uncommitted_speech: bool,
    /// The caller holds the floor: speaking, or paused mid-turn.
    in_turn: bool,
    /// The turn ended and waits for its outstanding commits.
    awaiting_answer: bool,
    speaking: bool,
    /// The caller started speaking over a reply that still plays: undecided
    /// until their words show a real turn (interrupt) or only a backchannel
    /// or noise (ignored, and not answered).
    overlapping: bool,
    /// Playback is paused for that speech: a `Resume` or `Interrupt` is owed.
    held: bool,
    /// The current turn began over a reply: if it is only a backchannel, it
    /// isn't answered (even if it went on long enough to stop the reply).
    began_over_reply: bool,
    /// The caller is speaking now (not paused mid-turn).
    voiced: bool,
    user_turn: u32,
}

impl CallState {
    pub fn speaking(&self) -> bool {
        self.speaking
    }

    /// Text heard but not yet answered, and commits still in flight
    /// (introspection for tests and debugging).
    pub fn pending(&self) -> (&str, u32) {
        (&self.turn_text, self.commits_in_flight)
    }

    /// Speech over a reply is still undecided (see `Input::OverlapTimedOut`).
    pub fn overlapping(&self) -> bool {
        self.overlapping
    }

    /// The caller is taking the turn: stop the reply, and drop what playback
    /// held (even if the reply itself has finished meanwhile).
    fn interrupt(&mut self, actions: &mut Vec<Action>) {
        self.overlapping = false;
        let held = std::mem::take(&mut self.held);
        if self.speaking || held {
            self.speaking = false;
            actions.push(Action::Interrupt);
        }
    }

    /// The speech over the reply was nothing to stop for: play on.
    fn resume(&mut self, actions: &mut Vec<Action>) {
        if std::mem::take(&mut self.held) {
            actions.push(Action::Resume);
        }
    }

    /// Whether the call is quiet enough for an unprompted reply.
    pub fn quiet(&self) -> bool {
        !self.speaking && !self.in_turn && !self.awaiting_answer && self.commits_in_flight == 0
    }

    pub fn on(&mut self, input: Input) -> Vec<Action> {
        let mut actions = Vec::new();
        match input {
            Input::Turn(TurnEvent::SpeechStarted) => {
                self.in_turn = true;
                self.voiced = true;
                self.uncommitted_speech = true;
                // Not yet a barge-in: "mm-hmm" mustn't stop the reply. Each
                // stretch of speech is judged by whether a reply was playing
                // when it began.
                // A reply that finished while held hasn't been heard out:
                // speech then is still over it.
                let over_reply = self.speaking || self.held;
                if over_reply && !self.held {
                    self.held = true;
                    actions.push(Action::Hold);
                }
                self.overlapping = over_reply;
                self.began_over_reply = over_reply;
            }
            // The VAD called the pause.
            Input::Turn(TurnEvent::Assessed { .. }) => self.voiced = false,
            Input::Turn(TurnEvent::Paused) => {
                self.voiced = false;
                actions.push(Action::Commit);
                self.commits_in_flight += 1;
                self.uncommitted_speech = false;
            }
            Input::Turn(TurnEvent::Ended { by }) => {
                if self.uncommitted_speech {
                    actions.push(Action::Commit);
                    self.commits_in_flight += 1;
                    self.uncommitted_speech = false;
                }
                self.in_turn = false;
                self.voiced = false;
                self.awaiting_answer = true;
                actions.push(Action::TurnEnded(by));
            }
            Input::Committed(text) => {
                if self.commits_in_flight == 0 {
                    return actions; // not ours to wait for
                }
                self.commits_in_flight -= 1;
                let text = text.trim();
                if self.overlapping {
                    if is_backchannel(text) {
                        // Heard, and let be: the reply plays on.
                        self.overlapping = false;
                        self.resume(&mut actions);
                        return self.answer(actions);
                    }
                    self.interrupt(&mut actions);
                }
                if !text.is_empty() {
                    if !self.turn_text.is_empty() {
                        self.turn_text.push(' ');
                    }
                    self.turn_text.push_str(text);
                }
            }
            // Speech begun over the reply still answers the reply, even if
            // the reply ends first: a backchannel stays one.
            Input::ReplyFinished => self.speaking = false,
            Input::Heard(text) => {
                if self.overlapping && !is_backchannel(&text) {
                    self.interrupt(&mut actions);
                }
            }
            // Only while they are speaking: in a pause, or once they stop,
            // their words decide.
            Input::OverlapTimedOut => {
                if self.overlapping && self.voiced {
                    self.interrupt(&mut actions);
                }
            }
            Input::UnpromptedReady => {
                if self.quiet() {
                    self.speaking = true;
                    actions.push(Action::SpeakUnprompted);
                }
            }
        }
        self.answer(actions)
    }

    /// The turn is over and all its segments are in: answer it.
    fn answer(&mut self, mut actions: Vec<Action>) -> Vec<Action> {
        if self.awaiting_answer && self.commits_in_flight == 0 && !self.in_turn {
            self.awaiting_answer = false;
            // Speech over the reply that never became a turn: it goes on.
            self.overlapping = false;
            self.resume(&mut actions);
            let backchannel =
                std::mem::take(&mut self.began_over_reply) && is_backchannel(&self.turn_text);
            let text = std::mem::take(&mut self.turn_text);
            actions.push(Action::FinalTranscript(text.clone()));
            if !text.is_empty() && !backchannel {
                self.user_turn += 1;
                self.speaking = true;
                actions.push(Action::Reply {
                    text,
                    index: self.user_turn,
                });
            }
        }
        actions
    }
}

/// Words that acknowledge without taking the turn ("mm-hmm", "yeah, right"),
/// or none at all (a cough, noise): speech over a reply made only of these
/// doesn't interrupt it.
pub fn is_backchannel(text: &str) -> bool {
    const WORDS: &[&str] = &[
        "uh",
        "huh",
        "um",
        "mm",
        "mmm",
        "hmm",
        "hm",
        "mhm",
        "mmhmm",
        "yeah",
        "yes",
        "yep",
        "yup",
        "ok",
        "okay",
        "right",
        "sure",
        "alright",
        "cool",
        "nice",
        "great",
        "wow",
        "oh",
        "ah",
        "i",
        "see",
        "got",
        "it",
        "gotcha",
        "true",
        "totally",
        "exactly",
        "interesting",
        "uh-huh",
    ];
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .all(|word| WORDS.contains(&word.to_lowercase().as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backchannels_and_noise_are_told_from_turns() {
        for text in [
            "Mm-hmm, yeah.",
            "uh-huh",
            "Okay. Right.",
            "I see.",
            "",
            "  ",
            "Got it!",
        ] {
            assert!(is_backchannel(text), "{text:?}");
        }
        for text in [
            "Wait, stop.",
            "Yeah, but what about Tuesday?",
            "No",
            "I got a question",
        ] {
            assert!(!is_backchannel(text), "{text:?}");
        }
    }
}
