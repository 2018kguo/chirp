//! Smart Turn inference cost per decision, by batch size (one thread).
//!
//!     cargo run --release --bin turn_bench

use std::path::Path;
use std::time::Instant;

use ort::value::Tensor;

fn main() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut session = chirp::onnx::session(&root.join("models/smart-turn-v3.2-cpu.onnx"))?;
    let audio: Vec<f32> =
        chirp::client::read_wav(&root.join("tests/fixtures/turn/complete_question.wav"))?
            .iter()
            .map(|s| *s as f32 / 32768.0)
            .collect();
    let mut padded = vec![0f32; chirp::turn::N_SAMPLES];
    chirp::turn::last_8s_left_padded(&audio, &mut padded);
    let features = chirp::turn::whisper_log_mel(&padded);
    for batch in [1usize, 2, 4, 8, 16] {
        let input: Vec<f32> = features
            .iter()
            .copied()
            .cycle()
            .take(batch * features.len())
            .collect();
        let mut run = || -> anyhow::Result<f64> {
            let tensor = Tensor::from_array(([batch, 80usize, 800usize], input.clone()))?;
            let started = Instant::now();
            let outputs = session.run(ort::inputs!["input_features" => tensor])?;
            let (_, p) = outputs["logits"].try_extract_tensor::<f32>()?;
            assert!(p.len() == batch);
            Ok(started.elapsed().as_secs_f64() * 1000.0)
        };
        for _ in 0..3 {
            run()?;
        }
        let mut times: Vec<f64> = (0..15).map(|_| run()).collect::<anyhow::Result<_>>()?;
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        println!(
            "batch {batch:>2}: {median:>7.1} ms per run, {:>5.1} ms per decision",
            median / batch as f64
        );
    }
    Ok(())
}
