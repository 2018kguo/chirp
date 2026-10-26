"""Reference oracle for src/turn.rs (Smart Turn v3, Pipecat 1.13).

Run with the pinned reference environment:
  uv run --project reference python scripts/turn_reference.py [--generate] [--bench]

--generate  (re)creates the speech clips with macOS `say` (16 kHz mono PCM16).
Writes into tests/fixtures/turn/:
  <clip>.wav            input audio
  <clip>.features.f32   reference features, little-endian f32 [80][800] (FEATURE_CLIPS only)
  reference.tsv         clip \t p_pipecat \t p_rust_ort
                        p_pipecat:  Pipecat's own onnxruntime (1.24.4 in this venv)
                        p_rust_ort: onnxruntime==RUST_ORT_VERSION (the runtime the `ort`
                                    crate links), CPU EP, run in a throwaway uv env.
    This int8 model's output moves by up to ~0.03 between ORT 1.24 and 1.31 on
    bit-identical inputs (different quantized kernels), so 1e-3 probability parity
    is only meaningful against the same runtime version; decisions are checked
    against both.
--bench prints median ms for feature extraction and inference (single thread).

The math is Pipecat's own: compute_whisper_log_mel_features + the truncate/pad
in LocalSmartTurnAnalyzerV3._predict_endpoint, with the same session options.
"""

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    from pipecat.audio.turn.smart_turn._whisper_features import compute_whisper_log_mel_features
except ImportError:  # --infer-only runs in a bare onnxruntime env
    compute_whisper_log_mel_features = None

CRATE = Path(__file__).resolve().parent.parent
FIXTURES = CRATE / "tests" / "fixtures" / "turn"
MODEL = CRATE / "models" / "smart-turn-v3.2-cpu.onnx"
SR = 16000
# ort 2.0.0-rc.15 downloads ONNX Runtime 1.31.0 (ort-sys build/download/dist.tsv).
RUST_ORT_VERSION = "1.31.0"

# name -> text spoken by `say` (None: generated programmatically)
CLIPS = {
    "complete_question": "What time does the pharmacy close tomorrow?",
    "incomplete_mid_sentence": "So I was wondering if you could help me find the, um",
    "long_over_8s": (
        "Last week I called about my prescription refill, and they told me it would be ready "
        "on Tuesday, but when I went in on Wednesday they said it had never been ordered. "
        "Can you check what happened?"
    ),
    "short_under_1s": "Yes.",
    "silence": None,
}
FEATURE_CLIPS = ["complete_question", "long_over_8s", "short_under_1s"]


def generate() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, text in CLIPS.items():
        out = FIXTURES / f"{name}.wav"
        if text is None:
            write_wav(out, np.zeros(2 * SR, dtype=np.int16))
            continue
        subprocess.run(
            ["say", "-v", "Samantha", "-o", str(out), "--data-format=LEI16@16000", text],
            check=True,
        )


def write_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == SR, path
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    # Same int16 -> f32 scaling the Rust test uses.
    return pcm.astype(np.float32) / 32768.0


def last_8s(audio: np.ndarray) -> np.ndarray:
    # Verbatim from LocalSmartTurnAnalyzerV3._predict_endpoint.
    max_samples = 8 * SR
    if len(audio) > max_samples:
        return audio[-max_samples:]
    if len(audio) < max_samples:
        return np.pad(audio, (max_samples - len(audio), 0), mode="constant", constant_values=0)
    return audio


def make_session() -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(MODEL), sess_options=so, providers=["CPUExecutionProvider"])


def infer_dir(feature_dir: Path) -> None:
    """--infer-only: print `name\tprob` for every <name>.f32 in feature_dir."""
    session = make_session()
    for path in sorted(feature_dir.glob("*.f32")):
        feats = np.fromfile(path, "<f4").reshape(1, 80, 800)
        print(f"{path.stem}\t{session.run(None, {'input_features': feats})[0][0].item()!r}")


def rust_ort_probs(all_feats: dict) -> dict:
    with tempfile.TemporaryDirectory() as d:
        for name, feats in all_feats.items():
            feats.astype("<f4").tofile(Path(d) / f"{name}.f32")
        out = subprocess.run(
            ["uv", "run", "-q", "--no-project", "--python", "3.12",
             "--with", f"onnxruntime=={RUST_ORT_VERSION}", "--with", "numpy",
             "python", "-I", str(Path(__file__).resolve()), "--infer-only", d],
            check=True, capture_output=True, text=True,
        ).stdout
    return {k: float(v) for k, v in (line.split("\t") for line in out.strip().splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--infer-only", type=Path)
    args = ap.parse_args()
    if args.infer_only:
        return infer_dir(args.infer_only)
    if args.generate:
        generate()

    session = make_session()
    rows, all_feats, probs = [], {}, {}
    for name in CLIPS:
        audio = read_wav(FIXTURES / f"{name}.wav")
        feats = compute_whisper_log_mel_features(last_8s(audio), do_normalize=True)
        assert feats.shape == (80, 800) and feats.dtype == np.float32
        prob = session.run(None, {"input_features": feats[None]})[0][0].item()
        all_feats[name], probs[name] = feats, prob
        if name in FEATURE_CLIPS:
            feats.astype("<f4").tofile(FIXTURES / f"{name}.features.f32")
    rust_probs = rust_ort_probs(all_feats)
    for name in CLIPS:
        rows.append(f"{name}\t{probs[name]!r}\t{rust_probs[name]!r}")
        print(f"{name:26s} p_pipecat(ort {ort.__version__})={probs[name]:.6f}  "
              f"p(ort {RUST_ORT_VERSION})={rust_probs[name]:.6f}")
    (FIXTURES / "reference.tsv").write_text("\n".join(rows) + "\n")

    if args.bench:
        audio = last_8s(read_wav(FIXTURES / "long_over_8s.wav"))
        feats = compute_whisper_log_mel_features(audio, do_normalize=True)[None]
        fe, inf = [], []
        for _ in range(10):
            session.run(None, {"input_features": feats})
        for _ in range(100):
            t0 = time.perf_counter()
            f = compute_whisper_log_mel_features(audio, do_normalize=True)[None]
            t1 = time.perf_counter()
            session.run(None, {"input_features": f})
            t2 = time.perf_counter()
            fe.append((t1 - t0) * 1e3)
            inf.append((t2 - t1) * 1e3)
        print(
            f"python median ms: features {statistics.median(fe):.3f}  "
            f"inference {statistics.median(inf):.3f}  "
            f"total {statistics.median(a + b for a, b in zip(fe, inf)):.3f}"
        )


if __name__ == "__main__":
    sys.exit(main())
