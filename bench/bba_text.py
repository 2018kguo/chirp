# /// script
# requires-python = ">=3.11"
# dependencies = ["websockets>=13", "httpx>=0.27"]
# ///
"""The brain alone on Big Bench Audio: each question's audio transcribed whole
(ElevenLabs Scribe), sent as one message to the same model and system prompt
chirp's brain uses, judged the same way. The gap to bba.py is what the
voice pipeline costs (turns cut at pauses, streaming transcription).

    uv run bba_text.py --data DIR --out DIR [--model gpt-4.1]
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

from bba import judge

SYSTEM = (
    "You are a voice assistant. Your answers are spoken aloud. "
    "Think the problem through briefly, then clearly state your final answer."
)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4.1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.data / "sample.json").read_text())
    gate = asyncio.Semaphore(8)
    async with httpx.AsyncClient(timeout=180) as http:

        async def one(row: dict) -> dict:
            path = args.out / f"{row['id']}.json"
            if path.exists():
                return json.loads(path.read_text())
            async with gate:
                name = Path(row["file_name"]).stem
                wav = (args.data / "wav" / f"{name}.wav").read_bytes()
                stt = await http.post(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                    data={"model_id": "scribe_v1", "tag_audio_events": "false"},
                    files={"file": ("q.wav", wav, "audio/wav")},
                )
                stt.raise_for_status()
                question = stt.json()["text"]
                answer = await http.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                    json={
                        "model": args.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": question},
                        ],
                    },
                )
                answer.raise_for_status()
                text = answer.json()["choices"][0]["message"]["content"]
                correct = await judge(http, row["category"], text, row["official_answer"])
            record = {**row, "question_text": question, "reply_text": text, "correct": correct}
            path.write_text(json.dumps(record, indent=1))
            return record

        records = await asyncio.gather(*(one(r) for r in rows))
    n = len(records)
    print(f"text-only {args.model}: {sum(r['correct'] for r in records)}/{n}")
    for category in sorted({r["category"] for r in records}):
        rs = [r for r in records if r["category"] == category]
        print(f"  {category:17} {sum(r['correct'] for r in rs)}/{len(rs)}")


if __name__ == "__main__":
    asyncio.run(main())
