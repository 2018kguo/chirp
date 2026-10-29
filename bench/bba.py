# /// script
# requires-python = ">=3.11"
# dependencies = ["websockets>=13", "httpx>=0.27"]
# ///
"""Big Bench Audio against a chirp server, the way Artificial Analysis runs
speech-to-speech models: the spoken question goes in in real time, the spoken
answer is transcribed and judged against the official answer.

    uv run bba.py --data DIR --out DIR [--url ws://127.0.0.1:7861/v1/calls]

DIR holds sample.json (rows of metadata.jsonl) and wav/question_N.wav (16 kHz
mono PCM16). Needs ELEVENLABS_API_KEY (reply transcription) and
JUDGE_API_KEY for an OpenAI-compatible endpoint serving the judge (JUDGE_BASE_URL,
default the Vercel AI Gateway; JUDGE_MODEL, default Claude Sonnet 4.6, the
judge of Artificial Analysis's v1.1).
"""

import argparse
import asyncio
import io
import json
import os
import statistics
import time
import wave
from pathlib import Path

import httpx
import websockets

FRAME = 640  # 20 ms of 16 kHz PCM16
QUIET_AFTER_REPLY = 8.0  # reply over after this long with no audio
GIVE_UP_AFTER_QUESTION = 90.0
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "anthropic/claude-sonnet-4.6")


async def call(url: str, pcm: bytes) -> dict:
    """One call: the question in real time, then silence until the reply ends."""
    events, audio = [], []  # (t, event) and (t, bytes), t from call start
    async with websockets.connect(url, max_size=None) as ws:
        start = time.monotonic()
        now = lambda: time.monotonic() - start  # noqa: E731
        question_end = len(pcm) / 32000

        async def receive():
            async for message in ws:
                if isinstance(message, bytes):
                    audio.append((now(), message))
                else:
                    events.append((now(), json.loads(message)))

        receiver = asyncio.create_task(receive())
        silence = b"\0" * FRAME
        frame = 0
        while True:
            # Real time: frame n leaves at n * 20 ms.
            delay = frame * 0.02 - now()
            if delay > 0:
                await asyncio.sleep(delay)
            chunk = pcm[frame * FRAME : (frame + 1) * FRAME] or silence
            await ws.send(chunk.ljust(FRAME, b"\0"))
            frame += 1
            t = now()
            after = [at for at, _ in audio if at > question_end]
            if after and t - audio[-1][0] > QUIET_AFTER_REPLY:
                break
            if t > question_end + GIVE_UP_AFTER_QUESTION:
                break
        receiver.cancel()
    turn_ends = [at for at, e in events if e.get("type") == "turn_end"]
    speech_end = speech_end_s(pcm)
    # The answer: audio after the last turn end, if that turn end came once
    # the question had been spoken (allowing for the transcriber's lag).
    final_turn = turn_ends[-1] if turn_ends and turn_ends[-1] >= speech_end - 0.3 else None
    reply = [(at, b) for at, b in audio if final_turn is not None and at >= final_turn]
    return {
        "question_s": question_end,
        "speech_end_s": speech_end,
        "turn_ends": turn_ends,
        # Turns ended (and possibly answered) before the question was over.
        "early_turn_ends": sum(1 for at in turn_ends if at < speech_end - 0.3),
        # From the end of the spoken question (not of the file's trailing
        # silence) to the first audio of the answer.
        "first_audio_after_question_s": (reply[0][0] - speech_end) if reply else None,
        "reply_pcm": b"".join(b for _, b in reply),
        "events": [e for _, e in events if e.get("type") != "turn"],
    }


def speech_end_s(pcm: bytes) -> float:
    """Where the question's speech ends: the last 20 ms frame above -45 dBFS."""
    import array

    samples = array.array("h", pcm)
    last = 0
    for frame in range(len(samples) // 320):
        window = samples[frame * 320 : (frame + 1) * 320]
        if max(abs(x) for x in window) > 180:  # ~ -45 dBFS
            last = frame + 1
    return last * 0.02


def wav_bytes(pcm: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return out.getvalue()


async def transcribe(http: httpx.AsyncClient, pcm: bytes) -> str:
    if not pcm:
        return ""
    response = await http.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        data={"model_id": "scribe_v1"},
        files={"file": ("reply.wav", wav_bytes(pcm), "audio/wav")},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("text", "")


async def judge(http: httpx.AsyncClient, category: str, answer: str, official: str) -> bool:
    prompt = (
        "You are grading an answer to a Big Bench Hard question "
        f"({category}). The official answer is: {official!r}.\n\n"
        f"The candidate's spoken answer, transcribed:\n<answer>\n{answer}\n</answer>\n\n"
        "Is the candidate's final answer the same as the official answer? "
        "A missing, hedged or contradictory final answer is incorrect. "
        "Reply with exactly CORRECT or INCORRECT."
    )
    response = await http.post(
        f"{os.environ.get('JUDGE_BASE_URL', 'https://ai-gateway.vercel.sh/v1')}/chat/completions",
        headers={"authorization": f"Bearer {os.environ['JUDGE_API_KEY']}"},
        json={
            "model": JUDGE_MODEL,
            "max_tokens": 8,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip().upper()
    return text.startswith("CORRECT")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="ws://127.0.0.1:7861/v1/calls")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.data / "sample.json").read_text())
    if args.limit:
        rows = rows[: args.limit]
    gate = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as http:

        async def one(row: dict) -> dict:
            path = args.out / f"{row['id']}.json"
            if path.exists():
                return json.loads(path.read_text())
            name = Path(row["file_name"]).stem
            with wave.open(str(args.data / "wav" / f"{name}.wav")) as w:
                pcm = w.readframes(w.getnframes())
            async with gate:
                try:
                    result = await call(args.url, pcm)
                except Exception as error:  # a dropped call is a failed question
                    result = {
                        "question_s": len(pcm) / 32000,
                        "turn_ends": [],
                        "early_turn_ends": 0,
                        "first_audio_after_question_s": None,
                        "reply_pcm": b"",
                        "events": [],
                        "error": repr(error),
                    }
            reply_pcm = result.pop("reply_pcm")
            (args.out / f"{row['id']}.wav").write_bytes(wav_bytes(reply_pcm))
            text = await transcribe(http, reply_pcm)
            correct = await judge(http, row["category"], text, row["official_answer"])
            record = {**row, **result, "reply_text": text, "correct": correct}
            path.write_text(json.dumps(record, indent=1))
            print(
                f"{row['id']:4} {row['category']:17} {'ok ' if correct else 'BAD'} "
                f"ttfa={record['first_audio_after_question_s']} early={record['early_turn_ends']}",
                flush=True,
            )
            return record

        records = await asyncio.gather(*(one(row) for row in rows))
    summarize(records)


def summarize(records: list[dict]) -> None:
    n = len(records)
    correct = sum(r["correct"] for r in records)
    print(f"\nBig Bench Audio, {n} questions: {correct}/{n} correct ({100 * correct / n:.1f}%)")
    for category in sorted({r["category"] for r in records}):
        rs = [r for r in records if r["category"] == category]
        print(f"  {category:17} {sum(r['correct'] for r in rs)}/{len(rs)}")
    ttfa = sorted(r["first_audio_after_question_s"] for r in records if r["first_audio_after_question_s"] is not None)
    if ttfa:
        print(
            f"time to first audio after the question: mean {statistics.mean(ttfa):.2f} s, "
            f"p50 {ttfa[len(ttfa) // 2]:.2f}, p90 {ttfa[int(len(ttfa) * 0.9)]:.2f} (n={len(ttfa)})"
        )
    early = sum(1 for r in records if r["early_turn_ends"])
    print(f"questions with a turn ended before the question was over: {early}/{n}")
    print(f"no answer: {sum(1 for r in records if r['first_audio_after_question_s'] is None)}/{n}")
    print(f"calls that failed: {sum(1 for r in records if r.get('error'))}/{n}")


if __name__ == "__main__":
    asyncio.run(main())
