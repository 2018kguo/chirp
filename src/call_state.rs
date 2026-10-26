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
}

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Commit the transcript segment so far.
    Commit,
    /// The caller spoke over a reply: stop it and tell the client to flush.
    Interrupt,
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

    /// Whether the call is quiet enough for an unprompted reply.
    pub fn quiet(&self) -> bool {
        !self.speaking && !self.in_turn && !self.awaiting_answer && self.commits_in_flight == 0
    }

    pub fn on(&mut self, input: Input) -> Vec<Action> {
        let mut actions = Vec::new();
        match input {
            Input::Turn(TurnEvent::SpeechStarted) => {
                self.in_turn = true;
                self.uncommitted_speech = true;
                if self.speaking {
                    self.speaking = false;
                    actions.push(Action::Interrupt);
                }
            }
            Input::Turn(TurnEvent::Assessed { .. }) => {}
            Input::Turn(TurnEvent::Paused) => {
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
                self.awaiting_answer = true;
                actions.push(Action::TurnEnded(by));
            }
            Input::Committed(text) => {
                if self.commits_in_flight == 0 {
                    return actions; // not ours to wait for
                }
                self.commits_in_flight -= 1;
                let text = text.trim();
                if !text.is_empty() {
                    if !self.turn_text.is_empty() {
                        self.turn_text.push(' ');
                    }
                    self.turn_text.push_str(text);
                }
            }
            Input::ReplyFinished => self.speaking = false,
            Input::UnpromptedReady => {
                if self.quiet() {
                    self.speaking = true;
                    actions.push(Action::SpeakUnprompted);
                }
            }
        }
        // The turn is over and all its segments are in: answer it.
        if self.awaiting_answer && self.commits_in_flight == 0 && !self.in_turn {
            self.awaiting_answer = false;
            let text = std::mem::take(&mut self.turn_text);
            actions.push(Action::FinalTranscript(text.clone()));
            if !text.is_empty() {
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
