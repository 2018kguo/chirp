//! Engine latency against scripted mocks: how long chirp itself takes between
//! the caller's last word and the first reply audio, with the speech services
//! and the brain replaced by local mocks whose delays are known constants.
//!
//!     cargo run --release --bin bench_latency -- [runs] [concurrent calls]
//!
//! Runs chirp and the reference (Pipecat, reference/pipecat_bot.py) on the
//! same mocks and the same audio, interleaved. Reports medians with bootstrap
//! 95% CIs of each engine's own time: speech end → first reply audio, minus
//! the mocks' scripted delays; for chirp also speech end → end-of-turn
//! decision ("detect") and decision → final transcript ("commit").

use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use chirp::call::{Shared, SpeechPolicy};
use chirp::client;
use chirp::elevenlabs::ElevenLabs;
use chirp::endpoint::TurnModel;
use chirp::mock::{self, Script};
use chirp::openai::OpenAiBrain;
use chirp::vad::VadModel;

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

/// Bootstrap 95% CI of the median (fixed seed, so reruns agree on the method).
fn median_ci(values: &[f64]) -> (f64, f64) {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut medians: Vec<f64> = (0..2000)
        .map(|_| {
            let mut sample: Vec<f64> = (0..values.len())
                .map(|_| values[(next() % values.len() as u64) as usize])
                .collect();
            median(&mut sample)
        })
        .collect();
    medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (medians[50], medians[1949])
}

fn report(name: &str, mut values: Vec<f64>) {
    let (low, high) = median_ci(&values);
    println!(
        "{name:<12} median {:>7.1} ms  95% CI [{low:.1}, {high:.1}]  n={}",
        median(&mut values),
        values.len()
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let args: Vec<String> = std::env::args().collect();
    let runs: usize = args.get(1).map_or(20, |a| a.parse().unwrap());
    let concurrent: usize = args.get(2).map_or(1, |a| a.parse().unwrap());
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script = Script::default();
    let mocks = mock::start(script.clone()).await?;
    let shared = Shared {
        vad: VadModel::spawn_threads(&root.join("models/silero_vad.onnx"), 2)?,
        turn: TurnModel::spawn_threads(&root.join("models/smart-turn-v3.2-cpu.onnx"), 2)?,
        elevenlabs: ElevenLabs {
            api_key: "mock".into(),
            voice_id: "mock".into(),
            tts_model: "mock".into(),
            stt_model: "mock".into(),
            base_url: format!("ws://{mocks}"),
        },
        speech: SpeechPolicy::default(),
        turn_policy: chirp::turn_taking::Policy::default(),
    };
    let http = reqwest::Client::new();
    let brain_url = format!("http://{mocks}/v1");
    let router = chirp::server::router(shared, move || {
        OpenAiBrain::new(http.clone(), &brain_url, "mock", "mock", None)
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("ws://{}/v1/calls", listener.local_addr()?);
    tokio::spawn(async move { axum::serve(listener, router).await });

    // The reference: Pipecat on the same mocks, as a subprocess.
    let pipecat_port = std::net::TcpListener::bind("127.0.0.1:0")?
        .local_addr()?
        .port();
    let mut pipecat = std::process::Command::new("uv")
        .args([
            "run",
            "--project",
            "reference",
            "python",
            "reference/pipecat_bot.py",
        ])
        .args([pipecat_port.to_string(), mocks.to_string()])
        .current_dir(root)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    let pipecat_url = format!("ws://127.0.0.1:{pipecat_port}/v1/calls");
    for _ in 0..120 {
        if tokio::net::TcpStream::connect(("127.0.0.1", pipecat_port))
            .await
            .is_ok()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    let speech = client::read_wav(&root.join("tests/fixtures/turn/complete_question.wav"))?;
    let (mut detect, mut commit) = (vec![], vec![]);
    let mut first_audio = [vec![], vec![]];
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    // Interleaved, so drift (thermals, background load) hits both engines alike.
    for batch in 0..runs.div_ceil(concurrent) {
        for (engine, url) in [&url, &pipecat_url].into_iter().enumerate() {
            let calls: Vec<_> = (0..concurrent.min(runs - batch * concurrent))
                .map(|_| {
                    let (url, speech) = (url.clone(), speech.clone());
                    tokio::spawn(async move {
                        client::call(&url, &speech, Duration::from_secs(4)).await
                    })
                })
                .collect();
            for call in calls {
                let outcome = call.await??;
                let Some(audio) = outcome.first_audio else {
                    eprintln!("{} run without reply audio", ["chirp", "pipecat"][engine]);
                    continue;
                };
                first_audio[engine].push(ms(audio) - ms(outcome.speech_end));
                if engine == 0 {
                    let decided =
                        outcome.event_at("turn", |e| e["complete"].as_f64().unwrap_or(0.0) > 0.5);
                    let transcribed = outcome.event_at("transcript", |e| e["final"] == true);
                    if let (Some(decided), Some(transcribed)) = (decided, transcribed) {
                        detect.push(ms(decided) - ms(outcome.speech_end));
                        commit.push(ms(transcribed) - ms(decided));
                    }
                }
            }
        }
    }
    let _ = pipecat.kill();
    println!(
        "scripted delays: stt commit {:?}, brain first token {:?}, tts first audio {:?}",
        script.stt_commit, script.brain_first_token, script.tts_first_audio
    );
    let scripted = ms(script.scripted_reply_delay());
    report("chirp detect", detect);
    report("chirp commit", commit);
    for (name, values) in ["chirp", "pipecat"].into_iter().zip(first_audio) {
        report(
            &format!("{name} engine"),
            values.iter().map(|v| v - scripted).collect(),
        );
    }
    Ok(())
}
