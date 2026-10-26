//! Plays a 16 kHz mono WAV into a call, then silence, and records the reply.
//!
//!     cargo run --release --bin call_client -- <ws url> <question.wav> <out.wav> [seconds of silence after]

use std::time::Duration;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let args: Vec<String> = std::env::args().collect();
    let speech = chirp::client::read_wav(args[2].as_ref())?;
    let tail = Duration::from_secs(args.get(4).map_or(25, |a| a.parse().unwrap()));
    let outcome = chirp::client::call(&args[1], &speech, tail).await?;
    for (at, event) in &outcome.events {
        println!("{:>6} ms  {event}", at.as_millis());
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&args[3], spec)?;
    for sample in &outcome.audio {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    println!(
        "speech ended at {} ms; first reply audio at {:?} ms; {:.1} s of reply audio -> {}",
        outcome.speech_end.as_millis(),
        outcome.first_audio.map(|d| d.as_millis()),
        outcome.audio.len() as f64 / 16_000.0,
        args[3]
    );
    Ok(())
}
