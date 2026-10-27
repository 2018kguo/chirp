//! Server-paced playback for real-time transports (WebRTC, SIP): what a call
//! sends is queued here and taken out one 20 ms frame at a time, and an
//! `interrupt` (barge-in) drops whatever has not been played yet.

use std::collections::VecDeque;

use serde_json::Value;
use tokio::sync::mpsc;

use crate::call::ToCaller;

/// 20 ms at 16 kHz.
pub const FRAME: usize = 320;

pub struct Playout {
    from_call: mpsc::Receiver<ToCaller>,
    queue: VecDeque<i16>,
    call_ended: bool,
}

impl Playout {
    pub fn new(from_call: mpsc::Receiver<ToCaller>) -> Self {
        Self {
            from_call,
            queue: VecDeque::new(),
            call_ended: false,
        }
    }

    /// Takes everything the call has sent; events go to `on_event`.
    pub fn take(&mut self, mut on_event: impl FnMut(Value)) {
        loop {
            match self.from_call.try_recv() {
                Ok(ToCaller::Audio(pcm)) => self.queue.extend(
                    pcm.chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]])),
                ),
                Ok(ToCaller::Event(event)) => {
                    if event["type"] == "interrupt" {
                        self.queue.clear();
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

    /// The next 20 ms to play (the last one padded with silence), if any.
    pub fn frame(&mut self) -> Option<[i16; FRAME]> {
        if self.queue.is_empty() {
            return None;
        }
        let mut frame = [0i16; FRAME];
        let take = FRAME.min(self.queue.len());
        for (slot, sample) in frame.iter_mut().zip(self.queue.drain(..take)) {
            *slot = sample;
        }
        Some(frame)
    }

    /// The call has ended and everything it said has been played.
    pub fn finished(&self) -> bool {
        self.call_ended && self.queue.is_empty()
    }
}
