//! When has the caller finished their turn? A state machine over the call's
//! audio: Silero VAD per 32 ms window, Smart Turn on pauses. Time is the audio
//! clock (samples received), so it behaves the same live and offline, which
//! is how `bench_turns` scores policies against labeled turns.

use std::time::Duration;

use anyhow::Result;

use crate::endpoint::TurnModel;
use crate::ring::Ring;
use crate::vad::{CallVad, Speech, SAMPLE_RATE, WINDOW};

/// Audio kept from just before speech is detected, so the turn model hears
/// the start of the utterance.
const PRE_ROLL: usize = SAMPLE_RATE / 2;
const TURN_AUDIO_MAX: usize = 8 * SAMPLE_RATE;

#[derive(Clone, Debug)]
pub struct Policy {
    /// Smart Turn's threshold for "finished".
    pub complete: f32,
    /// After an "unfinished" decision, silence this long ends the turn anyway.
    pub grace: Duration,
    /// Report a pause (`TurnEvent::Paused`) after this many quiet windows,
    /// well before the VAD calls it, so work can start speculatively; 0 never.
    /// Whether the turn ends is still decided only once the VAD calls the
    /// pause: ending early instead traded cut-offs for speed (PLAN.md).
    pub pause_after_quiet_windows: u32,
}

impl Default for Policy {
    /// What Pipecat does by default: decide once the VAD has heard 0.2 s of
    /// quiet, end the turn above 0.5, otherwise wait up to the grace period.
    fn default() -> Self {
        Self {
            complete: 0.5,
            grace: Duration::from_millis(2500),
            pause_after_quiet_windows: 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TurnEvent {
    /// The caller started speaking (a barge-in, if the call is talking).
    SpeechStarted,
    /// The caller went quiet mid-turn; the turn may be ending. Fires once per
    /// pause, a few windows in.
    Paused,
    /// Smart Turn's verdict on a pause.
    Assessed { probability: f32 },
    /// The caller's turn is over.
    Ended { by: EndedBy },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndedBy {
    /// Smart Turn, once the VAD called the pause.
    Model,
    /// Silence for the whole grace period after an "unfinished" verdict.
    Grace,
    /// The VAD's pause alone: Smart Turn was too busy to answer in time.
    Fallback,
}

pub struct TurnTaker {
    policy: Policy,
    vad: CallVad,
    turn: TurnModel,
    window: Vec<f32>,
    recent: Ring,
    turn_audio: Ring,
    /// Consecutive quiet windows while the caller holds the floor.
    quiet: u32,
    pause_reported: bool,
    /// Audio-clock sample at which the grace period runs out.
    grace_until: Option<u64>,
    clock: u64,
}

impl TurnTaker {
    pub fn new(vad: CallVad, turn: TurnModel, policy: Policy) -> Self {
        Self {
            policy,
            vad,
            turn,
            window: Vec::with_capacity(WINDOW),
            recent: Ring::new(PRE_ROLL),
            turn_audio: Ring::new(TURN_AUDIO_MAX),
            quiet: 0,
            pause_reported: false,
            grace_until: None,
            clock: 0,
        }
    }

    /// Samples received so far (the audio clock).
    pub fn clock(&self) -> u64 {
        self.clock
    }

    /// Whether the caller holds the floor: speaking, or paused mid-turn.
    pub fn in_turn(&self) -> bool {
        self.vad.speaking() || self.grace_until.is_some() || !self.turn_audio.is_empty()
    }

    /// Feed 16 kHz mono samples; returns what happened, in order.
    pub async fn push(&mut self, samples: &[f32]) -> Result<Vec<TurnEvent>> {
        let mut events = Vec::new();
        for &sample in samples {
            self.window.push(sample);
            if self.window.len() == WINDOW {
                let window = std::mem::replace(&mut self.window, Vec::with_capacity(WINDOW));
                self.clock += WINDOW as u64;
                self.window_done(&window, &mut events).await?;
            }
        }
        Ok(events)
    }

    async fn window_done(&mut self, window: &[f32], events: &mut Vec<TurnEvent>) -> Result<()> {
        self.recent.push(window);
        let holding = self.vad.speaking() || self.grace_until.is_some();
        if holding {
            self.turn_audio.push(window);
        }
        let transition = self.vad.push(window).await?;
        let quiet = self.vad.last_confidence() < crate::vad::CONFIDENCE;
        match transition {
            Some(Speech::Started) => {
                self.grace_until = None;
                self.quiet = 0;
                self.pause_reported = false;
                if self.turn_audio.is_empty() {
                    self.turn_audio.push(&self.recent.to_vec());
                }
                events.push(TurnEvent::SpeechStarted);
                return Ok(());
            }
            Some(Speech::Stopped) => {
                match self.assess(events).await? {
                    Some(probability) if probability > self.policy.complete => {
                        self.end(EndedBy::Model, events)
                    }
                    // No verdict in time: the VAD's call stands, as in engines
                    // without an end-of-turn model.
                    None => self.end(EndedBy::Fallback, events),
                    Some(_) => {
                        self.grace_until = Some(
                            self.clock
                                + (self.policy.grace.as_secs_f64() * SAMPLE_RATE as f64) as u64,
                        );
                    }
                }
                return Ok(());
            }
            None => {}
        }
        if self.vad.speaking() {
            self.quiet = if quiet { self.quiet + 1 } else { 0 };
            if !quiet {
                self.pause_reported = false;
            }
            let after = self.policy.pause_after_quiet_windows;
            if after > 0 && self.quiet >= after && !self.pause_reported {
                self.pause_reported = true;
                events.push(TurnEvent::Paused);
            }
        }
        if self.grace_until.is_some_and(|until| self.clock >= until) {
            self.end(EndedBy::Grace, events);
        }
        Ok(())
    }

    /// Smart Turn's verdict on the current turn, or `None` if the pool could
    /// not answer in time (overload).
    async fn assess(&mut self, events: &mut Vec<TurnEvent>) -> Result<Option<f32>> {
        let probability = self.turn.complete(self.turn_audio.to_vec()).await?;
        if let Some(probability) = probability {
            events.push(TurnEvent::Assessed { probability });
        }
        Ok(probability)
    }

    fn end(&mut self, by: EndedBy, events: &mut Vec<TurnEvent>) {
        self.turn_audio.clear();
        self.grace_until = None;
        self.quiet = 0;
        self.pause_reported = false;
        events.push(TurnEvent::Ended { by });
    }
}
