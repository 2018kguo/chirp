//! Server-sent events: the `data:` payloads of a streaming HTTP response.

use anyhow::Result;
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::StreamExt;

pub struct SseData {
    bytes: BoxStream<'static, Result<Bytes>>,
    buffer: Vec<u8>,
    /// Start of the unread part of `buffer`; consumed lines are dropped in
    /// bulk, not one memmove per line.
    read: usize,
    done: bool,
}

impl SseData {
    pub fn new(response: reqwest::Response) -> Self {
        Self::from_stream(response.bytes_stream().map(|piece| Ok(piece?)).boxed())
    }

    pub fn from_stream(bytes: BoxStream<'static, Result<Bytes>>) -> Self {
        Self {
            bytes,
            buffer: Vec::new(),
            read: 0,
            done: false,
        }
    }

    /// The next `data:` payload; `None` at the end of the body or at `[DONE]`.
    pub async fn next(&mut self) -> Result<Option<String>> {
        loop {
            while let Some(offset) = self.buffer[self.read..].iter().position(|b| *b == b'\n') {
                let line = &self.buffer[self.read..self.read + offset];
                self.read += offset + 1;
                let line = String::from_utf8_lossy(line);
                let Some(data) = line.trim_end().strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim_start();
                if data == "[DONE]" {
                    self.done = true;
                    return Ok(None);
                }
                return Ok(Some(data.to_string()));
            }
            if self.done {
                return Ok(None);
            }
            match self.bytes.next().await {
                Some(piece) => {
                    self.buffer.drain(..self.read);
                    self.read = 0;
                    self.buffer.extend_from_slice(&piece?);
                }
                None => return Ok(None),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn parse(events: usize) -> (usize, std::time::Duration) {
        let body: String = (0..events)
            .map(|i| format!("data: {{\"n\":{i}}}\n\n"))
            .collect();
        let stream = futures::stream::iter([Ok(Bytes::from(body))]).boxed();
        let mut data = SseData::from_stream(stream);
        let started = std::time::Instant::now();
        let mut count = 0;
        while data.next().await.unwrap().is_some() {
            count += 1;
        }
        (count, started.elapsed())
    }

    #[tokio::test]
    async fn parses_every_event_and_split_reads() {
        assert_eq!(parse(1000).await.0, 1000);
        let pieces = [
            "data: {\"a\"",
            ":1}\n\ndata",
            ": [DONE]\n\n",
            "data: ignored\n\n",
        ];
        let stream = futures::stream::iter(pieces.map(|p| Ok(Bytes::from(p)))).boxed();
        let mut data = SseData::from_stream(stream);
        assert_eq!(data.next().await.unwrap().as_deref(), Some("{\"a\":1}"));
        assert_eq!(data.next().await.unwrap(), None);
    }

    /// Doubling check: one big read of 2N events costs about twice N, not four
    /// times (a per-line memmove of the remaining buffer made it quadratic).
    #[tokio::test]
    async fn cost_is_linear_in_events_per_read() {
        let mut ratios = Vec::new();
        for _ in 0..5 {
            let (_, small) = parse(20_000).await;
            let (_, large) = parse(40_000).await;
            ratios.push(large.as_secs_f64() / small.as_secs_f64());
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            ratios[2] < 3.0,
            "doubling the events multiplied the time by {:.2}",
            ratios[2]
        );
    }
}
