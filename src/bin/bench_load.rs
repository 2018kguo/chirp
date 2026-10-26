//! Throughput: N concurrent callers holding conversations with one engine
//! process, against the scripted mocks in another process.
//!
//!     cargo run --release --bin bench_load -- chirp|pipecat <calls> [seconds]
//!
//! Every caller loops: ask the question (real speech), stay silent while the
//! reply plays, ask again. Reports the engine process's CPU (cores busy and
//! ms per call-second) and memory (from `ps`), and each turn's latency from
//! the end of speech to the first reply audio (p50/p99), which should stay
//! flat as calls are added if the engine keeps up.

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

/// Silence after each question: long enough for the reply to finish.
const GAP: Duration = Duration::from_millis(3000);

/// A child in its own process group, killed with everything it started
/// (`uv run` leaves its Python child running if only `uv` is killed).
struct Process(Child);

impl Process {
    fn spawn(command: &mut Command) -> Result<Self> {
        use std::os::unix::process::CommandExt;
        Ok(Self(command.process_group(0).spawn()?))
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = Command::new("kill")
            .args(["-TERM", &format!("-{}", self.0.id())])
            .status();
        let _ = self.0.wait();
    }
}

fn free_port() -> Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?
        .local_addr()?
        .port())
}

async fn wait_for(port: u16) -> Result<()> {
    for _ in 0..240 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    bail!("nothing listening on {port}")
}

/// (cumulative CPU seconds, resident MB) of a process, from `ps`.
fn usage(pid: u32) -> Result<(f64, f64)> {
    let out = Command::new("ps")
        .args(["-o", "time=,rss=", "-p", &pid.to_string()])
        .output()?;
    let text = String::from_utf8(out.stdout)?;
    let mut fields = text.split_whitespace();
    let time = fields.next().context("ps time")?;
    let rss_kb: f64 = fields.next().context("ps rss")?.parse()?;
    // [[dd-]hh:]mm:ss.cc
    let seconds = time.split([':', '-']).fold(0.0, |total, part| {
        total * 60.0 + part.parse::<f64>().unwrap_or(0.0)
    });
    Ok((seconds, rss_kb / 1024.0))
}

/// One caller: returns each turn's latency (speech end → first reply audio).
async fn converse(
    url: String,
    speech: std::sync::Arc<Vec<i16>>,
    speech_end: Duration,
    until: Instant,
) -> Result<Vec<f64>> {
    let (socket, _) = tokio_tungstenite::connect_async(url.as_str()).await?;
    let (mut sink, mut stream) = socket.split();
    let silence = vec![0i16; (GAP.as_millis() * 16) as usize];
    let turn: Vec<Vec<u8>> = speech
        .iter()
        .chain(silence.iter())
        .copied()
        .collect::<Vec<i16>>()
        .chunks(320)
        .map(|frame| frame.iter().flat_map(|s| s.to_le_bytes()).collect())
        .collect();
    let turn_length = Duration::from_millis(turn.len() as u64 * 20);
    let started = Instant::now();
    let sender = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(20));
        while Instant::now() < until {
            for frame in &turn {
                tick.tick().await;
                if sink
                    .send(Message::Binary(frame.clone().into()))
                    .await
                    .is_err()
                {
                    return;
                }
            }
        }
        let _ = sink.close().await;
    });
    // The first audio after each turn's speech end is that turn's reply.
    let mut latencies = Vec::new();
    let mut answered = u64::MAX;
    while let Some(Ok(message)) = stream.next().await {
        if !matches!(message, Message::Binary(_)) {
            continue;
        }
        let at = started.elapsed();
        let turn = (at.as_millis() / turn_length.as_millis()) as u64;
        let end = turn_length * turn as u32 + speech_end;
        if at > end && turn != answered {
            answered = turn;
            latencies.push((at - end).as_secs_f64() * 1000.0);
        }
    }
    sender.abort();
    Ok(latencies)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() as f64 - 1.0) * p) as usize]
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let engine = args.get(1).map_or("chirp", String::as_str).to_string();
    let calls: usize = args.get(2).map_or(10, |a| a.parse().unwrap());
    let seconds: u64 = args.get(3).map_or(30, |a| a.parse().unwrap());
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let release = root.join("target/release");

    let mocks_port = free_port()?;
    let _mocks = Process::spawn(
        Command::new(release.join("mocks"))
            .arg(mocks_port.to_string())
            .stdout(Stdio::null()),
    )?;
    wait_for(mocks_port).await?;
    let mocks = format!("127.0.0.1:{mocks_port}");
    let port = free_port()?;
    let log = std::fs::File::create(
        std::env::temp_dir().join(format!("bench_load-{engine}-{calls}.log")),
    )?;
    let mut command = match engine.as_str() {
        "chirp" => {
            let mut command = Command::new(release.join("chirp"));
            command
                .env("PORT", port.to_string())
                .env("ELEVENLABS_BASE_URL", format!("ws://{mocks}"))
                .env("ELEVENLABS_API_KEY", "mock")
                .env("CHIRP_BRAIN_URL", format!("http://{mocks}/v1"))
                .env("CHIRP_BRAIN_KEY", "mock")
                .env("CHIRP_BRAIN_MODEL", "mock")
                .env("RUST_LOG", "warn");
            for name in ["CHIRP_VAD_THREADS", "CHIRP_TURN_THREADS"] {
                if let Ok(value) = std::env::var(name) {
                    command.env(name, value);
                }
            }
            command
        }
        "pipecat" => {
            let mut command = Command::new("uv");
            command
                .args([
                    "run",
                    "--project",
                    "reference",
                    "python",
                    "reference/pipecat_bot.py",
                ])
                .args([port.to_string(), mocks.clone()])
                .current_dir(root);
            command
        }
        other => bail!("unknown engine {other}"),
    };
    let server = Process::spawn(command.stdout(Stdio::null()).stderr(log))?;
    wait_for(port).await?;
    // `uv run` is a parent of the Python process; measure the listener itself.
    let pid = {
        let out = Command::new("lsof")
            .args(["-nP", &format!("-iTCP:{port}"), "-sTCP:LISTEN", "-t"])
            .output()?;
        String::from_utf8(out.stdout)?
            .trim()
            .lines()
            .next()
            .context("listener pid")?
            .parse::<u32>()?
    };

    let speech = std::sync::Arc::new(chirp::client::read_wav(
        &root.join("tests/fixtures/turn/complete_question.wav"),
    )?);
    let speech_end = chirp::client::speech_end(&speech);
    let url = format!("ws://127.0.0.1:{port}/v1/calls");
    let (cpu_before, _) = usage(pid)?;
    let started = Instant::now();
    let until = started + Duration::from_secs(seconds);
    let callers: Vec<_> = (0..calls)
        .map(|i| {
            let (url, speech) = (url.clone(), speech.clone());
            tokio::spawn(async move {
                // Stagger starts across one turn so callers don't speak in lockstep.
                tokio::time::sleep(Duration::from_millis(
                    (i as u64 * 5_300) / calls.max(1) as u64,
                ))
                .await;
                // A call that never ends is a failure, not a hung benchmark.
                let deadline = until + Duration::from_secs(20);
                match tokio::time::timeout_at(
                    deadline.into(),
                    converse(url, speech, speech_end, until),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_) => bail!("call still open 20 s after the caller hung up"),
                }
            })
        })
        .collect();
    let mut peak_rss = 0f64;
    while Instant::now() < until {
        tokio::time::sleep(Duration::from_secs(1)).await;
        peak_rss = peak_rss.max(usage(pid)?.1);
    }
    let (cpu_after, _) = usage(pid)?;
    let wall = started.elapsed().as_secs_f64();
    let (mut latencies, mut failed) = (Vec::new(), 0);
    let mut reasons = std::collections::BTreeMap::<String, usize>::new();
    for caller in callers {
        match caller.await? {
            Ok(mut turns) => latencies.append(&mut turns),
            Err(error) => {
                failed += 1;
                *reasons.entry(error.to_string()).or_default() += 1;
            }
        }
    }
    for (reason, count) in &reasons {
        eprintln!("{count} calls failed: {reason}");
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cpu = cpu_after - cpu_before;
    let expected_turns = calls as f64 * (seconds as f64 / 5.3).floor();
    println!(
        "{engine} calls={calls} cores={:.2} cpu_ms_per_call_s={:.2} rss_mb={peak_rss:.0} rss_mb_per_call={:.2} turns={} (~{expected_turns:.0} expected) failed_calls={failed} latency_ms p50={:.0} p99={:.0}",
        cpu / wall,
        cpu * 1000.0 / (calls as f64 * wall),
        peak_rss / calls as f64,
        latencies.len(),
        if latencies.is_empty() { f64::NAN } else { percentile(&latencies, 0.5) },
        if latencies.is_empty() { f64::NAN } else { percentile(&latencies, 0.99) },
    );
    drop(server);
    Ok(())
}
