# Public benchmarks

Small-scale runs of two benchmarks speech-to-speech leaderboards use, against
a chirp server with real providers. Each harness drives chirp's WebSocket in
real time, as a caller would.

| Harness | Benchmark | What it measures |
|---|---|---|
| `bba.py` | [Big Bench Audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio) | reasoning on spoken questions; time to first audio |
| `bba_text.py` | the same questions as text | the brain alone: the ceiling for `bba.py` |
| `fdb.py` | [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) v1/v1.5 | pause handling, turn-taking, interruptions, backchannels |

```sh
ELEVENLABS_API_KEY=... CHIRP_BRAIN_KEY=... CHIRP_BRAIN_MODEL=gpt-4.1 PORT=7881 cargo run --release
# Big Bench Audio: DIR has sample.json (rows of metadata.jsonl) and wav/question_N.wav (16 kHz mono)
ELEVENLABS_API_KEY=... JUDGE_API_KEY=... uv run bench/bba.py --data DIR --out OUT --url ws://127.0.0.1:7881/v1/calls
# Full-Duplex-Bench: then score OUT with the benchmark's own evaluation/evaluate.py
ELEVENLABS_API_KEY=... uv run bench/fdb.py --samples DATA/v1.0/candor_turn_taking --out OUT --limit 40
```

Deviations from the published methods: `fdb.py` transcribes chirp's output
with ElevenLabs Scribe (the benchmark uses NVIDIA Parakeet, which needs
CUDA); Big Bench Audio's judge sees the official answer and the transcribed
reply, not the question text.

## Results (Oct 29 2026, M-series Mac, `gpt-4.1`, ElevenLabs)

Big Bench Audio, 100 questions (25 per category), Claude Sonnet 4.6 judge:

| | Correct |
|---|---|
| `gpt-4.1` given the question as text | 79 |
| chirp 0.3 (350-character spoken cap) | 53 |
| without the cap | 67.6 (mean of 5 runs) |
| and continued utterances merged | 75.6 (mean of 5 runs) |
| 0.4.0 (also 1 s grace, commit pacing) | 75, 76; 0 failed calls; first audio p50 1.0-1.1 s, p90 2.0 s |

Runs of the same build differ by up to ~6 points.

Full-Duplex-Bench v1, full sets, 1 s grace (0.4) vs 2.5 s (Pipecat's):

| | 1 s | 2.5 s |
|---|---|---|
| Pause handling, take-overs (synthetic 137 + Candor 216) | 19 | 20 |
| Smooth turn-taking (Candor 119): turns taken | 1.00 | 0.97 |
| Smooth turn-taking: latency | 1.59 s | 2.54 s |

Full-Duplex-Bench v1.5 (40 each, GPT-4o behavior judge): user interruption
C_RESPOND 0.38, C_RESUME 0.33, C_UNKNOWN 0.23; user backchannel C_RESUME
0.30, C_UNKNOWN 0.68 (chirp stops for an "uh-huh" and says nothing more,
the next thing to fix).
