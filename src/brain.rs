//! The brain: whatever answers the caller. chirp hands it each finished user
//! turn and speaks what it streams back. A brain is made per call, so it can
//! keep that call's history or point at a conversation stored elsewhere.

use std::future::Future;

use anyhow::Result;
use futures::stream::BoxStream;

/// One finished user turn, as transcribed.
#[derive(Clone, Debug)]
pub struct Utterance {
    pub text: String,
    /// 1 for the call's first turn. Stable across a retried reply, so a brain
    /// can derive idempotency keys from it.
    pub index: u32,
    /// The caller went on talking over the previous reply before hearing a
    /// second of it: that turn end was premature (a pause mid-thought), so
    /// this text continues the previous utterance.
    pub continues: bool,
}

/// What a reply asks chirp to say.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplyEvent {
    /// Model text, as it streams.
    Text(String),
    /// A fixed line while the brain works (a tool call, a search). Said at most
    /// once per distinct line per reply, never cut by the spoken budget.
    Progress(String),
}

pub type Reply = BoxStream<'static, Result<ReplyEvent>>;

pub trait Brain: Send + Sync + 'static {
    /// Start the reply to one user turn. Dropping the stream (barge-in) must
    /// be safe; whether the brain keeps working is its own business.
    fn reply(&self, utterance: Utterance) -> impl Future<Output = Result<Reply>> + Send;

    /// A reply the brain started on its own (a follow-up to earlier work), if
    /// one is ready. chirp asks between turns, when nobody is speaking.
    fn unprompted(&self) -> impl Future<Output = Result<Option<Reply>>> + Send {
        async { Ok(None) }
    }
}

/// Turns reply events into what to say: `[n]` citation markers stripped (also
/// when split across deltas), each progress line once, and speech cut at the
/// first sentence end past a character budget.
pub struct Speakable {
    budget: usize,
    budget_tail: String,
    pending: String,
    spoken_chars: usize,
    done: bool,
    progress_said: Vec<String>,
}

impl Speakable {
    pub fn new(budget: usize, budget_tail: impl Into<String>) -> Self {
        Self {
            budget,
            budget_tail: budget_tail.into(),
            pending: String::new(),
            spoken_chars: 0,
            done: false,
            progress_said: Vec::new(),
        }
    }

    /// The text to say for one event, if any.
    pub fn event(&mut self, event: ReplyEvent) -> Option<String> {
        match event {
            ReplyEvent::Text(delta) => self.text(&delta),
            ReplyEvent::Progress(line) => {
                if self.done || self.progress_said.contains(&line) {
                    return None;
                }
                let spoken = format!(" {line} ");
                self.progress_said.push(line);
                Some(spoken)
            }
        }
    }

    fn text(&mut self, delta: &str) -> Option<String> {
        if self.done {
            return None;
        }
        let joined = std::mem::take(&mut self.pending) + delta;
        // A trailing "[" or "[12" may be the first half of a marker.
        let cut = open_marker_tail(&joined);
        self.pending = joined[cut..].to_string();
        let mut text = strip_citations(&joined[..cut]);
        if text.is_empty() {
            return None;
        }
        if self.spoken_chars + text.len() > self.budget {
            if let Some(end) = sentence_end(&text) {
                text.truncate(end);
                text.push_str(&self.budget_tail);
                self.done = true;
            }
        }
        self.spoken_chars += text.len();
        Some(text)
    }

    /// Whatever was held back when the reply ended.
    pub fn finish(&mut self) -> Option<String> {
        let rest = std::mem::take(&mut self.pending);
        (!self.done && !rest.is_empty()).then_some(rest)
    }
}

/// Index where a possibly-incomplete `[123` marker starts at the end, or len.
fn open_marker_tail(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut i = bytes.len();
    while i > 0 && bytes[i - 1].is_ascii_digit() {
        i -= 1;
    }
    if i > 0 && bytes[i - 1] == b'[' {
        i - 1
    } else {
        text.len()
    }
}

/// Removes complete `[123]` markers.
fn strip_citations(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find('[') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && after.as_bytes().get(digits) == Some(&b']') {
            rest = &after[digits + 1..];
        } else {
            out.push('[');
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Byte index just past the first sentence end (". ", "!", "?" or end of text).
fn sentence_end(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    (0..bytes.len()).find_map(|i| {
        let terminal = matches!(bytes[i], b'.' | b'!' | b'?');
        let followed = bytes.get(i + 1).is_none_or(|b| b.is_ascii_whitespace());
        (terminal && followed).then_some(i + 1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAIL: &str = " The rest is in the transcript.";

    fn say(events: Vec<ReplyEvent>) -> String {
        let mut speakable = Speakable::new(350, TAIL);
        let mut said: String = events
            .into_iter()
            .filter_map(|e| speakable.event(e))
            .collect();
        said.push_str(&speakable.finish().unwrap_or_default());
        said
    }

    fn text(parts: &[&str]) -> Vec<ReplyEvent> {
        parts
            .iter()
            .map(|p| ReplyEvent::Text((*p).into()))
            .collect()
    }

    #[test]
    fn citation_markers_are_stripped_even_when_split() {
        assert_eq!(
            say(text(&[
                "The label carries a boxed warning [1",
                "2] for tumors [3]."
            ])),
            "The label carries a boxed warning  for tumors ."
        );
        assert_eq!(say(text(&["see [note] and [1]"])), "see [note] and ");
        assert_eq!(say(text(&["ends with ["])), "ends with [");
    }

    #[test]
    fn speech_stops_at_the_first_sentence_end_past_the_budget() {
        let long = "word ".repeat(80); // 400 chars, no sentence end
        let said = say(text(&[
            &long,
            "Last sentence here. More text that is not spoken.",
        ]));
        assert!(
            said.ends_with(&format!("Last sentence here.{TAIL}")),
            "{said}"
        );
    }

    #[test]
    fn each_progress_line_is_said_once() {
        let line = |s: &str| ReplyEvent::Progress(s.into());
        assert_eq!(
            say(vec![
                line("Searching."),
                line("Searching."),
                line("Reading.")
            ]),
            " Searching.  Reading. "
        );
    }
}
