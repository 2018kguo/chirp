//! Load test: N concurrent calls stream real speech at real-time pace through
//! the shared VAD; reports process CPU per call and per-frame VAD latency.
//!
//!     cargo run --release --bin vad_load -- <calls> <seconds> [vad threads]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chirp::vad::{CallVad, VadModel};
use tokio::sync::Mutex;

fn cpu_seconds() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc_rusage::Rusage>::zeroed();
    unsafe { libc_rusage::getrusage(0, usage.as_mut_ptr()) };
    let usage = unsafe { usage.assume_init() };
    let secs = |tv: libc_rusage::Timeval| tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6;
    secs(usage.ru_utime) + secs(usage.ru_stime)
}

mod libc_rusage {
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Timeval {
        pub tv_sec: i64,
        pub tv_usec: i32,
    }
    #[repr(C)]
    pub struct Rusage {
        pub ru_utime: Timeval,
        pub ru_stime: Timeval,
        _rest: [i64; 14],
    }
    extern "C" {
        pub fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let calls: usize = args.get(1).map_or(100, |a| a.parse().unwrap());
    let seconds: u64 = args.get(2).map_or(20, |a| a.parse().unwrap());
    let threads: usize = args.get(3).map_or(1, |a| a.parse().unwrap());
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = VadModel::spawn_threads(&root.join("models/silero_vad.onnx"), threads)?;
    let audio: Arc<Vec<f32>> = Arc::new(
        std::fs::read(root.join("tests/fixtures/vad/speech.pcm"))?
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect(),
    );
    let latencies = Arc::new(Mutex::new(Vec::<f64>::new()));
    let cpu_start = cpu_seconds();
    let wall = Instant::now();
    let mut tasks = Vec::new();
    for call in 0..calls {
        let (model, audio, latencies) = (model.clone(), audio.clone(), latencies.clone());
        tasks.push(tokio::spawn(async move {
            // Stagger starts so calls don't tick in lockstep.
            tokio::time::sleep(Duration::from_micros(
                (call as u64 * 20_000) / calls.max(1) as u64,
            ))
            .await;
            let mut vad = CallVad::new(model);
            let mut tick = tokio::time::interval(Duration::from_millis(20));
            let frames = audio.len() / 320;
            let mut mine = Vec::new();
            for i in 0..(seconds as usize * 50) {
                tick.tick().await;
                let frame = &audio[(i % frames) * 320..(i % frames + 1) * 320];
                let t = Instant::now();
                vad.push(frame).await.unwrap();
                mine.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            latencies.lock().await.extend(mine);
        }));
    }
    for task in tasks {
        task.await?;
    }
    let cpu = cpu_seconds() - cpu_start;
    let elapsed = wall.elapsed().as_secs_f64();
    let mut lat = latencies.lock().await.clone();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| lat[((lat.len() as f64 - 1.0) * p) as usize];
    println!(
        "calls={calls} threads={threads} wall={elapsed:.1}s cpu={cpu:.2}s cores_busy={:.2} cpu_ms_per_call_second={:.2} frame_latency_ms p50={:.3} p99={:.3} max={:.3}",
        cpu / elapsed,
        cpu * 1000.0 / (calls as f64 * seconds as f64),
        pct(0.5),
        pct(0.99),
        lat.last().unwrap()
    );
    Ok(())
}
