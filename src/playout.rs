//! Server-paced playback for real-time transports (WebRTC, SIP). Reply audio
//! arrives faster than real time; each 20 ms is encoded as it arrives (Opus,
//! G.711: a fraction of the PCM's size) and queued, the transport takes one
//! encoded frame per tick, and an `interrupt` (barge-in) drops whatever has
//! not been played yet.

use std::collections::VecDeque;

use bytes::Bytes;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::call::ToCaller;

/// 20 ms at 16 kHz.
pub const FRAME: usize = 320;

/// A transport's encoder for one 20 ms frame (`None` drops the frame).
pub type Encoder = Box<dyn FnMut(&[i16; FRAME]) -> Option<Bytes> + Send>;

pub struct Playout<E> {
    from_call: mpsc::Receiver<ToCaller>,
    /// Encodes one 20 ms frame of 16 kHz audio for the wire.
    encode: E,
    /// Audio short of a whole frame, waiting for the rest.
    partial: Vec<i16>,
    frames: VecDeque<Bytes>,
    /// The caller is speaking over the reply: nothing plays until `resume`
    /// (or an `interrupt` drops it).
    held: bool,
    call_ended: bool,
}

impl<E: FnMut(&[i16; FRAME]) -> Option<Bytes>> Playout<E> {
    pub fn new(from_call: mpsc::Receiver<ToCaller>, encode: E) -> Self {
        Self {
            from_call,
            encode,
            partial: Vec::with_capacity(FRAME),
            frames: VecDeque::new(),
            held: false,
            call_ended: false,
        }
    }

    /// Takes everything the call has sent; events go to `on_event`.
    pub fn take(&mut self, mut on_event: impl FnMut(Value)) {
        loop {
            match self.from_call.try_recv() {
                Ok(ToCaller::Audio(pcm)) => {
                    for sample in pcm
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]))
                    {
                        self.partial.push(sample);
                        if self.partial.len() == FRAME {
                            self.encode_partial();
                        }
                    }
                }
                Ok(ToCaller::Event(event)) => {
                    if event["type"] == "hold" {
                        self.held = true;
                    } else if event["type"] == "resume" {
                        self.held = false;
                    }
                    if event["type"] == "interrupt" {
                        self.held = false;
                        self.partial.clear();
                        self.frames.clear();
                        // A long reply's queue is not kept for the rest of the call.
                        self.frames.shrink_to(64);
                    }
                    on_event(event);
                }
                Err(mpsc::error::TryRecvError::Empty) => return,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.call_ended = true;
                    return;
                }
            }
        }
    }

    fn encode_partial(&mut self) {
        let mut frame = [0i16; FRAME];
        frame[..self.partial.len()].copy_from_slice(&self.partial);
        self.partial.clear();
        if let Some(encoded) = (self.encode)(&frame) {
            self.frames.push_back(encoded);
        }
    }

    /// The next encoded 20 ms to play, if any. When the queue runs dry, a
    /// short tail is padded with silence and played rather than held.
    pub fn frame(&mut self) -> Option<Bytes> {
        if self.held {
            return None;
        }
        if self.frames.is_empty() && !self.partial.is_empty() {
            self.encode_partial();
        }
        self.frames.pop_front()
    }

    /// The call has ended and everything it said has been played.
    pub fn finished(&self) -> bool {
        self.call_ended && self.frames.is_empty() && self.partial.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pcm(samples: usize) -> ToCaller {
        ToCaller::Audio(vec![1; samples * 2])
    }

    #[tokio::test]
    async fn frames_come_out_whole_and_an_interrupt_drops_the_rest() {
        let (to, from) = mpsc::channel(16);
        let mut playout = Playout::new(from, |frame: &[i16; FRAME]| {
            Some(Bytes::copy_from_slice(
                &frame[..2]
                    .iter()
                    .flat_map(|s| s.to_le_bytes())
                    .collect::<Vec<_>>(),
            ))
        });
        to.send(pcm(FRAME * 2 + 100)).await.unwrap();
        playout.take(|_| {});
        assert!(playout.frame().is_some());
        assert!(playout.frame().is_some());
        // The 100-sample tail plays, padded, once the queue runs dry.
        assert!(playout.frame().is_some());
        assert!(playout.frame().is_none());
        to.send(pcm(FRAME * 5)).await.unwrap();
        to.send(ToCaller::Event(json!({ "type": "interrupt" })))
            .await
            .unwrap();
        let mut events = Vec::new();
        playout.take(|event| events.push(event));
        assert_eq!(events.len(), 1);
        assert!(
            playout.frame().is_none(),
            "audio before the interrupt was played"
        );
        drop(to);
        playout.take(|_| {});
        assert!(playout.finished());
    }

    #[tokio::test]
    async fn a_hold_pauses_playback_and_a_resume_plays_on_from_there() {
        let (to, from) = mpsc::channel(16);
        let mut playout = Playout::new(from, |frame: &[i16; FRAME]| {
            Some(Bytes::copy_from_slice(&frame[0].to_le_bytes()))
        });
        to.send(pcm(FRAME * 3)).await.unwrap();
        to.send(ToCaller::Event(json!({ "type": "hold" })))
            .await
            .unwrap();
        playout.take(|_| {});
        assert!(playout.frame().is_none(), "played while held");
        to.send(ToCaller::Event(json!({ "type": "resume" })))
            .await
            .unwrap();
        playout.take(|_| {});
        let played = std::iter::from_fn(|| playout.frame()).count();
        assert_eq!(played, 3, "the held frames all play after the resume");
    }
}
