//! Server-sent events: the `data:` payloads of a streaming HTTP response.

use anyhow::{Context, Result};
use futures::stream::BoxStream;
use futures::StreamExt;

pub struct SseData {
    bytes: BoxStream<'static, reqwest::Result<bytes::Bytes>>,
    buffer: Vec<u8>,
    done: bool,
}

impl SseData {
    pub fn new(response: reqwest::Response) -> Self {
        Self {
            bytes: response.bytes_stream().boxed(),
            buffer: Vec::new(),
            done: false,
        }
    }

    /// The next `data:` payload; `None` at the end of the body or at `[DONE]`.
    pub async fn next(&mut self) -> Result<Option<String>> {
        loop {
            while let Some(newline) = self.buffer.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = self.buffer.drain(..=newline).collect();
                let line = String::from_utf8_lossy(&line);
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
                Some(piece) => self
                    .buffer
                    .extend_from_slice(&piece.context("reading event stream")?),
                None => return Ok(None),
            }
        }
    }
}
