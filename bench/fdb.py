# /// script
# requires-python = ">=3.11"
# dependencies = ["websockets>=13", "httpx>=0.27", "numpy", "soundfile"]
# ///
"""Full-Duplex-Bench v1/v1.5 inference against a chirp server: each sample's
input.wav is streamed in real time, and chirp's reply is laid on the same
timeline as a listener would hear it (played in real time, cut at chirp's
interrupts), truncated to the input's length, as the reference inference
scripts do. Word timestamps for output.json come from ElevenLabs Scribe
(the benchmark uses NVIDIA Parakeet, which needs CUDA).

    uv run fdb.py --samples DIR --out DIR [--limit N] [--clean]

--clean also runs clean_input.wav -> clean_output.wav (v1.5 tasks).
"""

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets

FRAME = 640  # 20 ms of 16 kHz PCM16


def read_16k(path: Path) -> np.ndarray:
    """Any WAV as 16 kHz mono int16 (the samples are float WAVs)."""
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.int16)


async def call(url: str, pcm: np.ndarray) -> np.ndarray:
    """Streams `pcm` in real time; returns what a listener heard, same length."""
    duration = len(pcm) / 16000
    segments: list[list] = []  # [start_s, int16 array], in play order
    playing_until = 0.0
    async with websockets.connect(url, max_size=None) as ws:
        start = time.monotonic()
        now = lambda: time.monotonic() - start  # noqa: E731

        async def receive():
            nonlocal playing_until
            async for message in ws:
                t = now()
                if isinstance(message, bytes):
                    audio = np.frombuffer(message, dtype=np.int16)
                    begin = max(t, playing_until)
                    segments.append([begin, audio])
                    playing_until = begin + len(audio) / 16000
                elif json.loads(message).get("type") == "interrupt":
                    # Whatever hadn't been played yet is dropped.
                    for segment in segments:
                        played = int(max(0.0, t - segment[0]) * 16000)
                        segment[1] = segment[1][:played]
                    playing_until = min(playing_until, t)

        receiver = asyncio.create_task(receive())
        data = pcm.tobytes()
        frames = (len(data) + FRAME - 1) // FRAME
        for frame in range(frames):
            delay = frame * 0.02 - now()
            if delay > 0:
                await asyncio.sleep(delay)
            await ws.send(data[frame * FRAME : (frame + 1) * FRAME].ljust(FRAME, b"\0"))
        receiver.cancel()
    out = np.zeros(int(duration * 16000), dtype=np.int16)
    for begin, audio in segments:
        at = int(begin * 16000)
        if at >= len(out) or not len(audio):
            continue
        piece = audio[: len(out) - at]
        out[at : at + len(piece)] = piece
    return out


async def words(http: httpx.AsyncClient, path: Path) -> dict:
    """output.json in the benchmark's shape: {"text", "chunks": [{"text", "timestamp"}]}."""
    audio = sf.read(path)[0]
    if not np.any(np.abs(audio) > 1e-4):
        return {"text": "", "chunks": []}
    for attempt in range(5):
        response = await http.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
            data={"model_id": "scribe_v1", "timestamps_granularity": "word", "tag_audio_events": "false"},
            files={"file": (path.name, path.read_bytes(), "audio/wav")},
            timeout=120,
        )
        if response.status_code != 429:
            break
        await asyncio.sleep(2 + attempt * 3)
    response.raise_for_status()
    body = response.json()
    chunks = [
        {"text": w["text"], "timestamp": [w["start"], w["end"]]}
        for w in body.get("words", [])
        if w.get("type") == "word"
    ]
    return {"text": body.get("text", ""), "chunks": chunks}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True, help="dir of sample dirs (each has input.wav)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="ws://127.0.0.1:7882/v1/calls")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    samples = sorted(p.parent for p in args.samples.glob("*/input.wav"))
    random.Random(args.seed).shuffle(samples)
    samples = samples[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as http:

        async def one(sample: Path) -> None:
            target = args.out / sample.name
            target.mkdir(exist_ok=True)
            # The sample's own files (annotations, input transcripts) beside ours.
            for f in sample.iterdir():
                if f.is_file() and not (target / f.name).exists():
                    shutil.copy(f, target / f.name)
            for prefix in (["", "clean_"] if args.clean else [""]):
                wav = target / f"{prefix}output.wav"
                if not wav.exists():
                    pcm = read_16k(sample / f"{prefix}input.wav")
                    async with gate:
                        try:
                            heard = await call(args.url, pcm)
                        except Exception as error:  # a dropped call heard nothing more
                            print(f"call failed {sample.name}: {error!r}", flush=True)
                            heard = np.zeros(len(pcm), dtype=np.int16)
                    sf.write(wav, heard, 16000, subtype="PCM_16")
                transcript = target / f"{prefix}output.json"
                if not transcript.exists():
                    transcript.write_text(json.dumps(await words(http, wav), indent=1))
            print(f"done {sample.name}", flush=True)

        await asyncio.gather(*(one(s) for s in samples))


if __name__ == "__main__":
    asyncio.run(main())
