# chirp

A voice engine that serves many calls per process.

Each call is a WebSocket carrying 16 kHz mono PCM16 in and out, plus JSON
events. chirp listens (Silero VAD), decides when the caller has finished
(Smart Turn v3), transcribes and speaks through streaming services
(ElevenLabs), handles barge-in, and gets its answers from a pluggable
**brain**: any streaming chat backend, behind a two-method trait.

```text
caller audio ─► VAD (batched across calls) ─► end-of-turn model ─► STT commit
                                                                       │
caller ◄── audio ◄── TTS ◄── speakable text ◄── brain (your backend) ◄─┘
```

## How it compares

Measured against Pipecat 1.13 running the same models on the same machine,
with scripted mock STT/TTS/LLM services so only the engine differs. Numbers
are from an AWS c7a.4xlarge (16 vCPU); [PLAN.md](PLAN.md) has every
measurement, its method, and what was tried and rejected.

| | chirp | Pipecat |
|---|---|---|
| Engine time, end of speech → first reply audio (median, mock services) | 182 ms | 315 ms |
| CPU per call-second | ~13 ms | ~305 ms |
| Calls per 16-vCPU box | ~1,000, 0 failed (p99 890 ms) | one process falls behind from ~10 |
| Memory per call | ~133 KB (Linux, jemalloc) | 60–70 MB |

Turn decisions match Pipecat's: speculative transcript commits changed 0 of
789 verdicts on Pipecat's Smart Turn v3.2 test set.

## Design

- **One process, many calls.** Tokio does only I/O. Model inference runs on
  dedicated threads: each VAD thread owns one Silero session and scores every
  call's 32 ms windows in batches. Smart Turn runs on its own pool, only when
  a caller pauses, with a deadline. When that pool falls behind, new calls get
  a 503 instead of everyone getting slow.
- **Speculative commits.** The transcript is committed when the caller first
  goes quiet, so it is ready the moment the turn is judged over.
- **Barge-in** drops the spoken reply, including audio already queued, and
  tells the client to flush playback. The brain decides what happens to the
  work behind it.
- **Checked against the reference.** The models run through ONNX Runtime on
  the files Pipecat ships, pinned by URL and checksum (`src/models.rs`).
  `tests/vad_parity.rs` and `tests/turn_parity.rs` compare framing, batching,
  features and probabilities with Pipecat's Python outputs.
- **Checked over every interleaving.** Each call's decisions (commit,
  interrupt, reply) are a pure reducer, `call_state.rs`, and the call loop
  only executes them. `tests/call_state_model.rs` explores every ordering of
  turn events, transcripts and reply completions to depth 32 and checks that
  chirp never answers while the caller holds the floor, never loses or
  duplicates speech, and leaves nothing unanswered.
  `tests/turn_grammar.rs` checks that real audio only produces orderings the
  model explored.

## Run

```sh
ELEVENLABS_API_KEY=... CHIRP_BRAIN_KEY=... CHIRP_BRAIN_MODEL=gpt-4.1-mini cargo run --release
cargo run --release --bin call_client -- ws://127.0.0.1:7861/v1/calls question.wav reply.wav
```

| Variable | Default | |
|---|---|---|
| `ELEVENLABS_API_KEY` | required | speech-to-text and text-to-speech |
| `ELEVENLABS_VOICE_ID`, `ELEVENLABS_TTS_MODEL` | a stock voice, Flash | |
| `CHIRP_BRAIN_URL` | OpenAI | any OpenAI-compatible `/chat/completions` API |
| `CHIRP_BRAIN_KEY`, `CHIRP_BRAIN_MODEL` | required | |
| `CHIRP_SYSTEM_PROMPT` | none | |
| `CHIRP_VAD_THREADS` | cores / 4, at least 2 | |
| `CHIRP_TURN_THREADS` | cores − 2 | |
| `CHIRP_MODELS_DIR` | `models` | missing models are downloaded here at startup |
| `PORT` | 7861 | |

## Your own brain

To answer with something other than a chat-completions API, implement
`Brain` and hand `chirp::server::serve` a function that makes one per call.
It reads the same environment variables:

```rust
use chirp::brain::{Brain, Reply, ReplyEvent, Utterance};
use futures::stream;

struct Echo;

impl Brain for Echo {
    async fn reply(&self, utterance: Utterance) -> anyhow::Result<Reply> {
        let said = format!("You said: {}", utterance.text);
        Ok(Box::pin(stream::iter([Ok(ReplyEvent::Text(said))])))
    }
}

# async fn run() -> anyhow::Result<()> {
chirp::server::serve(|| Echo).await
# }
```

To mount the call routes beside your own, build them with
`chirp::server::router(chirp::server::shared_from_env().await?, make_brain)`
and serve the combined axum `Router` with `chirp::server::listen`.

`ReplyEvent::Progress` speaks a short status line (for example while a tool
runs), at most once each. `Brain::unprompted` lets the brain speak something
the caller didn't ask for, such as a result that finished later; chirp polls
it while the call is quiet.

## Benchmarks

`bench_latency` (engine time, interleaved with Pipecat), `bench_load`
(throughput, CPU and memory per call at N calls), and `bench_turns` (turn
verdicts on Pipecat's labeled test set) run against `mocks`, scripted local
STT, TTS and LLM services. Run load tests at scale on a dedicated machine.
They saturate it on purpose.

## Models

chirp downloads [Silero VAD](https://github.com/snakers4/silero-vad)
(MIT) and [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3)
(BSD-2-Clause) on first start; `cargo run --bin fetch_models` fetches them
for the tests. The turn-taking test set used by `bench_turns` is
[Pipecat's](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test)
(CC-BY-4.0).

## License

MIT
