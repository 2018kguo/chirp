//! A fixed-capacity audio ring: pushing a 20 ms frame costs the frame, not
//! the buffer, however long the call has run.

pub struct Ring {
    data: Vec<f32>,
    /// Next write position.
    head: usize,
    len: usize,
}

impl Ring {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            head: 0,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Appends `samples`, dropping the oldest once full.
    pub fn push(&mut self, samples: &[f32]) {
        let capacity = self.data.len();
        let samples = &samples[samples.len().saturating_sub(capacity)..];
        let first = samples.len().min(capacity - self.head);
        self.data[self.head..self.head + first].copy_from_slice(&samples[..first]);
        self.data[..samples.len() - first].copy_from_slice(&samples[first..]);
        self.head = (self.head + samples.len()) % capacity;
        self.len = (self.len + samples.len()).min(capacity);
    }

    /// The buffered samples, oldest first.
    pub fn to_vec(&self) -> Vec<f32> {
        let capacity = self.data.len();
        let start = (self.head + capacity - self.len) % capacity;
        let mut out = Vec::with_capacity(self.len);
        if start + self.len <= capacity {
            out.extend_from_slice(&self.data[start..start + self.len]);
        } else {
            out.extend_from_slice(&self.data[start..]);
            out.extend_from_slice(&self.data[..self.len - (capacity - start)]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_newest_samples_in_order() {
        let mut ring = Ring::new(5);
        let mut kept: Vec<f32> = Vec::new();
        // Pushes of every size up to and past capacity, against a Vec model.
        for (i, size) in [1, 3, 2, 7, 4, 5, 1, 0, 6].into_iter().enumerate() {
            let frame: Vec<f32> = (0..size).map(|j| (i * 10 + j) as f32).collect();
            ring.push(&frame);
            kept.extend_from_slice(&frame);
            let keep_from = kept.len().saturating_sub(5);
            kept.drain(..keep_from);
            assert_eq!(ring.to_vec(), kept, "after push {i}");
        }
        ring.clear();
        assert!(ring.to_vec().is_empty());
    }
}
