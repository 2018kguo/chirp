//! Parity oracle: Rust Smart Turn vs Pipecat 1.13's Python (scripts/turn_reference.py).
//!
//! Fixtures in tests/fixtures/turn/: `<clip>.wav` (16 kHz mono PCM16),
//! `reference.tsv` (clip -> p_pipecat, p_rust_ort) and `<clip>.features.f32`
//! (Python log-mel features, LE f32 [80][800]) for a subset of clips.
//! Regenerate with the script; never hand-edit.

use std::path::{Path, PathBuf};
use std::time::Instant;

use chirp::turn::{last_8s_left_padded, whisper_log_mel, SmartTurn, N_FRAMES, N_MELS, N_SAMPLES};

const FEATURE_TOL: f32 = 1e-4;
/// Against the same ONNX Runtime version the `ort` crate links (1.31.0), CPU EP.
const PROB_TOL: f32 = 1e-3;
/// Against Pipecat's runtime (ORT 1.24.4). The int8 model's output moves by up to
/// ~0.027 between ORT 1.24 and 1.31 on bit-identical features (different quantized
/// kernels; Python 1.31.0 reproduces Rust to ~1e-7), so only a loose bound plus an
/// exact decision match is meaningful here.
const CROSS_VERSION_PROB_TOL: f32 = 0.05;

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixtures() -> PathBuf {
    crate_dir().join("tests/fixtures/turn")
}

fn model() -> SmartTurn {
    SmartTurn::new(&crate_dir().join("models/smart-turn-v3.2-cpu.onnx")).expect("load model")
}

fn read_wav(path: &Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let spec = r.spec();
    assert_eq!(
        (spec.channels, spec.sample_rate, spec.bits_per_sample),
        (1, 16_000, 16)
    );
    r.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

/// (clip, p_pipecat, p_rust_ort)
fn references() -> Vec<(String, f32, f32)> {
    let tsv = std::fs::read_to_string(fixtures().join("reference.tsv")).unwrap();
    let refs: Vec<(String, f32, f32)> = tsv
        .lines()
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            (
                c[0].to_string(),
                c[1].parse().unwrap(),
                c[2].parse().unwrap(),
            )
        })
        .collect();
    assert!(refs.len() >= 5, "expected >= 5 reference clips");
    refs
}

fn padded(audio: &[f32]) -> Vec<f32> {
    let mut w = vec![0.0; N_SAMPLES];
    last_8s_left_padded(audio, &mut w);
    w
}

#[test]
fn features_match_python() {
    let mut checked = 0;
    for (name, _, _) in references() {
        let path = fixtures().join(format!("{name}.features.f32"));
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let expected: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(expected.len(), N_MELS * N_FRAMES);
        let got = whisper_log_mel(&padded(&read_wav(&fixtures().join(format!("{name}.wav")))));
        let max_diff = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("features {name:26} max|diff| = {max_diff:.3e}");
        assert!(
            max_diff <= FEATURE_TOL,
            "{name}: feature max diff {max_diff} > {FEATURE_TOL}"
        );
        checked += 1;
    }
    assert!(checked >= 2, "expected feature fixtures for >= 2 clips");
}

#[test]
fn probabilities_match_python() {
    let mut st = model();
    for (name, p_pipecat, p_same_ort) in references() {
        let audio = read_wav(&fixtures().join(format!("{name}.wav")));
        let p = st.predict(&audio).unwrap();
        let d_same = (p - p_same_ort).abs();
        let d_pipecat = (p - p_pipecat).abs();
        println!(
            "prob     {name:26} rust={p:.6} py_ort1.31={p_same_ort:.6} (|d|={d_same:.1e}) \
             pipecat_ort1.24={p_pipecat:.6} (|d|={d_pipecat:.1e})"
        );
        assert!(
            d_same <= PROB_TOL,
            "{name}: |{p} - {p_same_ort}| > {PROB_TOL} vs Python on the same ORT version \
             (a CoreML EP registered by ort's default auto-device policy on macOS does this)"
        );
        assert!(
            d_pipecat <= CROSS_VERSION_PROB_TOL,
            "{name}: |{p} - {p_pipecat}| vs Pipecat"
        );
        assert_eq!(
            p > 0.5,
            p_pipecat > 0.5,
            "{name}: decision differs from Pipecat"
        );
    }
}

#[test]
fn complete_and_incomplete_land_on_expected_sides() {
    let refs = references();
    let get = |n: &str| refs.iter().find(|(k, _, _)| k == n).unwrap().1;
    // Guards the fixtures themselves: the corpus must contain both decisions.
    assert!(get("complete_question") > 0.5);
    assert!(get("incomplete_mid_sentence") < 0.5);
    let mut st = model();
    let p = |st: &mut SmartTurn, n: &str| {
        st.predict(&read_wav(&fixtures().join(format!("{n}.wav"))))
            .unwrap()
    };
    assert!(p(&mut st, "complete_question") > 0.5);
    assert!(p(&mut st, "incomplete_mid_sentence") < 0.5);
}

#[test]
fn input_lengths_cover_truncate_and_pad_paths() {
    let len = |n: &str| read_wav(&fixtures().join(format!("{n}.wav"))).len();
    assert!(len("long_over_8s") > N_SAMPLES);
    assert!(len("short_under_1s") < 16_000);
    assert!(read_wav(&fixtures().join("silence.wav"))
        .iter()
        .all(|&s| s == 0.0));
}

/// Release-mode timing, single thread. `cargo test --release --test turn_parity -- --nocapture`.
#[test]
fn bench_predict() {
    let mut st = model();
    let audio = read_wav(&fixtures().join("long_over_8s.wav"));
    let feats = st.features(&audio).to_vec();
    for _ in 0..10 {
        st.infer(&feats).unwrap();
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (mut fe, mut inf, mut total) = (vec![], vec![], vec![]);
    for _ in 0..100 {
        let t0 = Instant::now();
        let f = st.features(&audio).to_vec();
        let t1 = Instant::now();
        st.infer(&f).unwrap();
        let t2 = Instant::now();
        st.predict(&audio).unwrap();
        let t3 = Instant::now();
        fe.push((t1 - t0).as_secs_f64() * 1e3);
        inf.push((t2 - t1).as_secs_f64() * 1e3);
        total.push((t3 - t2).as_secs_f64() * 1e3);
    }
    println!(
        "rust median ms: features {:.3}  inference {:.3}  predict {:.3}",
        median(fe),
        median(inf),
        median(total)
    );
}
