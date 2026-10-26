"""Reference Silero probabilities for tests/vad_parity.rs, from Pipecat's model wrapper.

    uv run --project reference python scripts/vad_reference.py [source.wav]

Scores each tests/fixtures/vad/<signal>.pcm (16 kHz mono PCM16) window by
window with Pipecat's SileroOnnxModel and writes <signal>.probs.f32. With a
source WAV (16 kHz mono PCM16, >= 30 s of speech; the committed fixtures came
from macOS `say`), first cuts the signals from it.
"""
import sys
import wave

import numpy as np
from pipecat.audio.vad.silero import SileroOnnxModel

out = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "tests/fixtures/vad")
wav_path = sys.argv[1] if len(sys.argv) > 1 else None
names = ["speech", "reversed", "later", "silence"]
if wav_path:
    with wave.open(wav_path) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    speech = pcm[: 16000 * 12]
    signals = {
        "speech": speech,
        "reversed": speech[::-1].copy(),
        "later": pcm[16000 * 18 : 16000 * 30],
        "silence": np.zeros(16000 * 4, np.int16),
    }
    for name, signal in signals.items():
        signal.tofile(f"{out}/{name}.pcm")
signals = {name: np.fromfile(f"{out}/{name}.pcm", np.int16) for name in names}
import importlib.resources as res

model_path = str(res.files("pipecat.audio.vad.data").joinpath("silero_vad.onnx"))
for name, signal in signals.items():
    model = SileroOnnxModel(model_path, force_onnx_cpu=True)
    audio = signal.astype(np.float32) / 32768.0
    probs = [
        float(model(audio[i : i + 512][None, :], 16000)[0][0])
        for i in range(0, len(audio) - 511, 512)
    ]
    np.asarray(probs, np.float32).tofile(f"{out}/{name}.probs.f32")
    print(name, len(probs), "windows, max", round(max(probs), 3))
