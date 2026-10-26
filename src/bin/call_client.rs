//! Test client: plays a 16 kHz mono WAV into a call at real-time pace, then
//! silence, and records what the gateway says back.
//!
//!     cargo run --release --bin call_client -- <ws url> <question.wav> <out.wav> [seconds of silence after]

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

#[tokio::main]
async fn main() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let args: Vec<String> = std::env::args().collect();
    let (url, input, output) = (&args[1], &args[2], &args[3]);
    let tail_seconds: u64 = args.get(4).map_or(25, |a| a.parse().unwrap());
    let mut reader = hound::WavReader::open(input).context("input wav")?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_rate == 16_000 && spec.channels == 1,
        "need 16 kHz mono"
    );
    let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<_, _>>()?;
    let (socket, _) = tokio_tungstenite::connect_async(url.as_str())
        .await
        .context("connect")?;
    let (mut sink, mut stream) = socket.split();
    let started = Instant::now();
    let speech_ends = Duration::from_millis(samples.len() as u64 / 16);
    let frames: Vec<Vec<u8>> = samples
        .chunks(320)
        .chain(std::iter::repeat_n(
            &[0i16; 320][..],
            (tail_seconds * 50) as usize,
        ))
        .map(|frame| frame.iter().flat_map(|s| s.to_le_bytes()).collect())
        .collect();
    let sender = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(20));
        for frame in frames {
            tick.tick().await;
            if sink.send(Message::Binary(frame.into())).await.is_err() {
                break;
            }
        }
        let _ = sink.close().await;
    });
    let mut received: Vec<i16> = Vec::new();
    let mut first_audio: Option<Duration> = None;
    while let Some(Ok(message)) = stream.next().await {
        let at = started.elapsed();
        match message {
            Message::Binary(pcm) => {
                first_audio.get_or_insert(at);
                received.extend(
                    pcm.chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]])),
                );
            }
            Message::Text(event) => println!("{:>6} ms  {event}", at.as_millis()),
            _ => {}
        }
    }
    sender.abort();
    let mut writer = hound::WavWriter::create(output, spec)?;
    for sample in &received {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    println!(
        "speech ended at {} ms; first reply audio at {:?} ms; {:.1} s of reply audio -> {output}",
        speech_ends.as_millis(),
        first_audio.map(|d| d.as_millis()),
        received.len() as f64 / 16_000.0
    );
    Ok(())
}
