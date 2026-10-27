//! Turn-taking quality on labeled audio: Pipecat's Smart Turn v3.2 test set
//! (CC-BY-4.0; extract with scripts/turn_dataset.py). Each clip stops either
//! where the speaker finished ("complete") or mid-thought; the bench plays
//! it, then silence, through `TurnTaker` and scores each policy on:
//!
//! - complete clips: share ended by the model (not the silence fallback), and
//!   latency from the clip's end to the end-of-turn decision;
//! - incomplete clips: false cut-offs (ended by the model instead of waiting);
//! - any clip: premature ends (a turn ended, then the speaker kept talking).
//!
//! `cargo run --release --bin bench_turns -- [language dir, default eng] [limit]`

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use chirp::endpoint::TurnModel;
use chirp::turn_taking::{EndedBy, Policy, TurnEvent, TurnTaker};
use chirp::vad::{CallVad, VadModel};

struct Clip {
    audio: Vec<f32>,
    complete: bool,
    synthetic: bool,
}

#[derive(Default)]
struct Score {
    complete: usize,
    ended_by_model: usize,
    latencies_ms: Vec<f64>,
    incomplete: usize,
    false_cuts: usize,
    premature: usize,
    clips: usize,
}

const TAIL: Duration = Duration::from_secs(3);

/// Where the clip's speech ends: the last 20 ms frame within 30 dB of the
/// clip's loudest frame (recordings carry trailing room noise or silence).
fn speech_end(audio: &[f32]) -> u64 {
    let rms: Vec<f32> = audio
        .chunks(320)
        .map(|f| (f.iter().map(|s| s * s).sum::<f32>() / f.len() as f32).sqrt())
        .collect();
    let loudest = rms.iter().copied().fold(0f32, f32::max);
    let last = rms.iter().rposition(|r| *r > loudest * 0.0316).unwrap_or(0);
    ((last + 1) * 320) as u64
}

/// The clip's final verdict (the last end after the last speech start) with
/// its latency from the end of speech, and whether any earlier end was
/// followed by more speech (a cut-off).
async fn score(
    clip: &Clip,
    vad: VadModel,
    turn: TurnModel,
    policy: Policy,
) -> Result<(Option<(EndedBy, f64)>, bool)> {
    let mut taker = TurnTaker::new(CallVad::new(vad), turn, policy);
    let silence = vec![0f32; (TAIL.as_secs_f64() * 16_000.0) as usize];
    let mut timeline = Vec::new();
    for frame in clip.audio.chunks(320).chain(silence.chunks(320)) {
        for event in taker.push(frame).await? {
            timeline.push((taker.clock(), event));
        }
    }
    let last_start = timeline
        .iter()
        .rposition(|(_, e)| *e == TurnEvent::SpeechStarted);
    let premature = timeline.iter().enumerate().any(|(i, (_, e))| {
        matches!(e, TurnEvent::Ended { .. }) && last_start.is_some_and(|start| start > i)
    });
    let end = speech_end(&clip.audio);
    let verdict = timeline[last_start.map_or(0, |i| i + 1)..]
        .iter()
        .find_map(|(at, e)| match e {
            TurnEvent::Ended { by } => Some((*by, (*at as f64 - end as f64) / 16.0)),
            _ => None,
        });
    Ok((verdict, premature))
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[((values.len() - 1) as f64 * p) as usize]
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let language = args.get(1).map_or("eng", String::as_str);
    let limit: usize = args.get(2).map_or(usize::MAX, |a| a.parse().unwrap());
    let data: PathBuf = std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME")?
        .join(".cache/chirp/turn-data")
        .join(language);
    let labels = std::fs::read_to_string(data.join("labels.tsv"))
        .context("run scripts/turn_dataset.py first")?;
    let clips: Arc<Vec<Clip>> = Arc::new(
        labels
            .lines()
            .skip(1)
            .take(limit)
            .map(|line| {
                let fields: Vec<&str> = line.split('\t').collect();
                let audio = std::fs::read(data.join(format!("{}.pcm", fields[0])))?
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
                Ok(Clip {
                    audio,
                    complete: fields[1] == "1",
                    synthetic: fields[2] == "1",
                })
            })
            .collect::<Result<_>>()?,
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let vad = VadModel::spawn_threads(&root.join("models/silero_vad.onnx"), 2)?;
    let turn = TurnModel::spawn_threads(&root.join("models/smart-turn-v3.2-cpu.onnx"), threads)?;
    // Pause reports must not change a single verdict: they only start work.
    let policies: Vec<(String, Policy)> = vec![
        ("default".into(), Policy::default()),
        (
            "no pause reports".into(),
            Policy {
                pause_after_quiet_windows: 0,
                ..Policy::default()
            },
        ),
    ];
    let mut verdicts: Vec<Vec<Option<(EndedBy, i64)>>> = Vec::new();
    println!("{} clips ({language})", clips.len());
    if let Ok(index) = std::env::var("CHIRP_TRACE_CLIP") {
        // Event timeline for one clip under the default policy.
        let clip = &clips[index.parse::<usize>()?];
        let mut taker = TurnTaker::new(CallVad::new(vad.clone()), turn.clone(), Policy::default());
        let peak = clip.audio.iter().fold(0f32, |m, s| m.max(s.abs()));
        println!(
            "clip {index}: {:.2} s, complete={}, synthetic={}, peak {peak:.3}",
            clip.audio.len() as f64 / 16_000.0,
            clip.complete,
            clip.synthetic
        );
        let silence = vec![0f32; 48_000];
        for frame in clip.audio.chunks(320).chain(silence.chunks(320)) {
            for event in taker.push(frame).await? {
                println!("  {:>6.2} s  {event:?}", taker.clock() as f64 / 16_000.0);
            }
        }
        return Ok(());
    }
    for (name, policy) in policies {
        let mut by_kind = [Score::default(), Score::default()]; // human, synthetic
        let mut these = vec![None; clips.len()];
        let tasks: Vec<_> = (0..clips.len())
            .map(|i| {
                let (clips, vad, turn, policy) =
                    (clips.clone(), vad.clone(), turn.clone(), policy.clone());
                tokio::spawn(
                    async move { score(&clips[i], vad, turn, policy).await.map(|r| (i, r)) },
                )
            })
            .collect();
        for task in tasks {
            let (i, (ended, premature)) = task.await??;
            these[i] = ended.map(|(by, ms)| (by, ms.round() as i64));
            let clip = &clips[i];
            let s = &mut by_kind[clip.synthetic as usize];
            s.clips += 1;
            s.premature += premature as usize;
            if clip.complete {
                s.complete += 1;
                if let Some((by, latency)) = ended {
                    if by != EndedBy::Grace {
                        s.ended_by_model += 1;
                        s.latencies_ms.push(latency);
                    }
                }
            } else {
                s.incomplete += 1;
                if matches!(ended, Some((EndedBy::Model, _))) {
                    s.false_cuts += 1;
                }
            }
        }
        for (kind, mut s) in ["human", "synthetic"].into_iter().zip(by_kind) {
            println!(
                "{name:<28} {kind:<9} complete: ended by model {:>5.1}%  latency p50 {:>4.0} ms p90 {:>4.0} ms | incomplete: false cut {:>5.1}% | premature {:>5.1}% (n={})",
                100.0 * s.ended_by_model as f64 / s.complete.max(1) as f64,
                percentile(&mut s.latencies_ms, 0.5),
                percentile(&mut s.latencies_ms, 0.9),
                100.0 * s.false_cuts as f64 / s.incomplete.max(1) as f64,
                100.0 * s.premature as f64 / s.clips.max(1) as f64,
                s.clips
            );
        }
        verdicts.push(these);
    }
    let differing = verdicts[0]
        .iter()
        .zip(&verdicts[1])
        .filter(|(a, b)| a != b)
        .count();
    println!("verdicts that differ between the policies: {differing}");
    Ok(())
}
