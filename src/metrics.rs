//! Process-wide counters, served as Prometheus text at `/metrics`.
//! ponytail: atomics and a format string; a metrics crate if histograms are needed.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

use crate::turn_taking::EndedBy;

pub struct Counter(AtomicU64);

impl Counter {
    const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    pub fn add(&self) {
        self.0.fetch_add(1, Relaxed);
    }

    fn get(&self) -> u64 {
        self.0.load(Relaxed)
    }
}

/// A `_sum` (in microseconds) and `_count` pair: a mean over any window.
pub struct Timing {
    micros: AtomicU64,
    count: AtomicU64,
}

impl Timing {
    const fn new() -> Self {
        Self {
            micros: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    pub fn record(&self, duration: Duration) {
        self.micros.fetch_add(duration.as_micros() as u64, Relaxed);
        self.count.fetch_add(1, Relaxed);
    }
}

pub static REFUSED_FORBIDDEN: Counter = Counter::new();
pub static REFUSED_OVERLOADED: Counter = Counter::new();
pub static REFUSED_LIMIT: Counter = Counter::new();
pub static CALLS: Counter = Counter::new();
pub static CALL_ERRORS: Counter = Counter::new();
pub static REPLY_ERRORS: Counter = Counter::new();
static TURNS: [Counter; 3] = [Counter::new(), Counter::new(), Counter::new()];
/// From the end of a caller's turn to the first audio of the reply.
pub static FIRST_AUDIO: Timing = Timing::new();
/// How long a Smart Turn decision waited for a thread.
pub static TURN_QUEUE: Timing = Timing::new();

pub fn turn_ended(by: EndedBy) {
    TURNS[by as usize].add();
}

/// Everything, in the Prometheus text format; `active` is calls in progress.
pub fn render(active: usize) -> String {
    let mut out = String::with_capacity(2048);
    let mut counter = |name: &str, help: &str, values: &[(&str, u64)]| {
        let _ = writeln!(
            out,
            "# HELP chirp_{name} {help}\n# TYPE chirp_{name} counter"
        );
        for (labels, value) in values {
            let _ = writeln!(out, "chirp_{name}{labels} {value}");
        }
    };
    counter("calls_total", "Calls started.", &[("", CALLS.get())]);
    counter(
        "calls_refused_total",
        "Calls refused before starting.",
        &[
            ("{reason=\"forbidden\"}", REFUSED_FORBIDDEN.get()),
            ("{reason=\"overloaded\"}", REFUSED_OVERLOADED.get()),
            ("{reason=\"limit\"}", REFUSED_LIMIT.get()),
        ],
    );
    counter(
        "call_errors_total",
        "Calls ended by an error (a provider failed).",
        &[("", CALL_ERRORS.get())],
    );
    counter(
        "reply_errors_total",
        "Replies that failed.",
        &[("", REPLY_ERRORS.get())],
    );
    counter(
        "turns_total",
        "Caller turns ended, by what ended them.",
        &[
            ("{by=\"model\"}", TURNS[EndedBy::Model as usize].get()),
            ("{by=\"grace\"}", TURNS[EndedBy::Grace as usize].get()),
            ("{by=\"fallback\"}", TURNS[EndedBy::Fallback as usize].get()),
        ],
    );
    let _ = writeln!(out, "# HELP chirp_active_calls Calls in progress.\n# TYPE chirp_active_calls gauge\nchirp_active_calls {active}");
    for (name, help, timing) in [
        (
            "first_audio_seconds",
            "End of the caller's turn to the reply's first audio.",
            &FIRST_AUDIO,
        ),
        (
            "turn_queue_seconds",
            "Smart Turn decisions' wait for a thread.",
            &TURN_QUEUE,
        ),
    ] {
        let seconds = timing.micros.load(Relaxed) as f64 / 1e6;
        let count = timing.count.load(Relaxed);
        let _ = writeln!(
            out,
            "# HELP chirp_{name} {help}\n# TYPE chirp_{name} summary\nchirp_{name}_sum {seconds}\nchirp_{name}_count {count}"
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_every_series() {
        turn_ended(EndedBy::Fallback);
        FIRST_AUDIO.record(Duration::from_millis(250));
        let text = render(3);
        assert!(text.contains("chirp_active_calls 3"));
        assert!(text.contains("chirp_turns_total{by=\"fallback\"} 1"));
        assert!(text.contains("chirp_first_audio_seconds_sum 0.25"));
        assert!(text.contains("chirp_first_audio_seconds_count 1"));
    }
}
