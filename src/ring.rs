//! Audio kept for the end-of-turn model: the newest samples up to a cap.
//!
//! Stored as i16, the format callers send, and converted to f32 only when
//! Smart Turn reads it: x / 32768 round-trips exactly, at half the memory.
//! It grows with the turn and gives its memory back when cleared, so an idle
//! call holds nothing and a typical 2–4 s turn never reaches the 8 s cap.

use std::collections::VecDeque;

pub struct Ring {
    samples: VecDeque<i16>,
    capacity: usize,
}

/// Memory a cleared ring keeps (about 64 ms of audio); anything more is freed.
const KEEP_AFTER_CLEAR: usize = 1024;

impl Ring {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::new(),
            capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn clear(&mut self) {
        self.samples.clear();
        self.samples.shrink_to(KEEP_AFTER_CLEAR);
    }

    /// Appends samples in [-1, 1), dropping the oldest past the cap.
    pub fn push(&mut self, samples: &[f32]) {
        let samples = &samples[samples.len().saturating_sub(self.capacity)..];
        let overflow = (self.samples.len() + samples.len()).saturating_sub(self.capacity);
        self.samples.drain(..overflow);
        self.samples.extend(
            samples
                .iter()
                .map(|s| (s * 32768.0).round().clamp(-32768.0, 32767.0) as i16),
        );
    }

    /// The buffered samples, oldest first.
    pub fn to_vec(&self) -> Vec<f32> {
        self.samples.iter().map(|s| *s as f32 / 32768.0).collect()
    }

    /// Bytes this ring holds (for memory tests).
    pub fn allocated_bytes(&self) -> usize {
        self.samples.capacity() * std::mem::size_of::<i16>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_newest_samples_in_order_and_exactly() {
        let mut ring = Ring::new(5);
        let mut kept: Vec<f32> = Vec::new();
        // Pushes of every size up to and past capacity, against a Vec model,
        // with values that are exact in 16-bit PCM.
        for (i, size) in [1, 3, 2, 7, 4, 5, 1, 0, 6].into_iter().enumerate() {
            let frame: Vec<f32> = (0..size)
                .map(|j| ((i * 10 + j) as f32 - 40.0) / 32768.0)
                .collect();
            ring.push(&frame);
            kept.extend_from_slice(&frame);
            let keep_from = kept.len().saturating_sub(5);
            kept.drain(..keep_from);
            assert_eq!(ring.to_vec(), kept, "after push {i}");
        }
        ring.clear();
        assert!(ring.to_vec().is_empty());
    }

    #[test]
    fn memory_follows_the_turn_not_the_cap() {
        let mut ring = Ring::new(8 * 16_000);
        assert_eq!(ring.allocated_bytes(), 0, "an idle call holds no audio");
        ring.push(&vec![0.1; 3 * 16_000]);
        assert!(
            ring.allocated_bytes() < 2 * 3 * 16_000 * 2,
            "a 3 s turn holds about 3 s"
        );
        ring.clear();
        assert!(
            ring.allocated_bytes() <= 2 * KEEP_AFTER_CLEAR * 2,
            "a finished turn gives its memory back"
        );
    }
}
