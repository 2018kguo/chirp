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
    /// SIGTERM first (so a profiling build can write its report), then
    /// SIGKILL to the whole group after a few seconds.
    fn drop(&mut self) {
        let _ = Command::new("kill")
            .args(["-TERM", &self.0.id().to_string()])
            .status();
        for _ in 0..100 {
            if matches!(self.0.try_wait(), Ok(Some(_))) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = Command::new("kill")
            .args(["-KILL", &format!("-{}", self.0.id())])
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
    // Sibling binaries live next to this one (also in a container whose
    // target directory is not the crate's).
    let release = std::env::current_exe()?
        .parent()
        .context("bench_load's directory")?
        .to_path_buf();

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
            // CHIRP_BIN: another build of the server (allocator experiments).
            let binary = std::env::var_os("CHIRP_BIN")
                .map_or_else(|| release.join("chirp"), std::path::PathBuf::from);
            let mut command = Command::new(binary);
            command
                .env("PORT", port.to_string())
                .env("ELEVENLABS_BASE_URL", format!("ws://{mocks}"))
                .env("ELEVENLABS_API_KEY", "mock")
                .env("CHIRP_BRAIN_URL", format!("http://{mocks}/v1"))
                .env("CHIRP_BRAIN_KEY", "mock")
                .env("CHIRP_BRAIN_MODEL", "mock")
                .env("RUST_LOG", "warn");
            for name in [
                "CHIRP_VAD_THREADS",
                "CHIRP_TURN_THREADS",
                "CHIRP_VAD_MAX_BATCH",
            ] {
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
    // chirp is the process we spawned; for Pipecat, `uv run` is a parent of
    // the Python process, so measure whatever listens on the port.
    let pid = if engine == "chirp" {
        server.0.id()
    } else {
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
    let (cpu_before, rss_before) = usage(pid)?;
    // Live heap, when the engine serves it (chirp built with `heap-stats`).
    let heap = |port: u16| async move {
        let body = reqwest::get(format!("http://127.0.0.1:{port}/debug/heap"))
            .await
            .ok()?
            .text()
            .await
            .ok()?;
        let mut fields = body.split_whitespace().map(|f| f.parse::<f64>().ok());
        Some((fields.next()??, fields.next()??, Instant::now()))
    };
    let heap_before = heap(port).await;
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
    let mut heap_mid = None;
    let mut heap_late = None;
    while Instant::now() < until {
        tokio::time::sleep(Duration::from_secs(1)).await;
        peak_rss = peak_rss.max(usage(pid)?.1);
        // Mid-run, every caller connected and conversing.
        if heap_mid.is_none() && started.elapsed() > Duration::from_secs(seconds / 2) {
            heap_mid = heap(port).await;
        } else if heap_mid.is_some()
            && heap_late.is_none()
            && started.elapsed() > Duration::from_secs(seconds.saturating_sub(2))
        {
            heap_late = heap(port).await;
        }
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
    if let (Some((before, _, _)), Some((mid, mid_calls, mid_at))) = (heap_before, heap_mid) {
        // Allocation rate late in the run, all callers live.
        let per_call_second = heap_late.map(|(_, end_calls, end_at)| {
            (end_calls - mid_calls) / end_at.duration_since(mid_at).as_secs_f64() / calls as f64
        });
        println!(
            "heap: {:.1} MB idle, {:.1} MB mid-run, {:.1} KB per call; RSS {:.1} KB per call over idle; {:.0} allocations per call-second",
            before / 1e6,
            mid / 1e6,
            (mid - before) / calls as f64 / 1e3,
            (peak_rss - rss_before) * 1024.0 / calls as f64,
            per_call_second.unwrap_or(f64::NAN)
        );
    }
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
