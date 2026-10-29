<h1 align="center">chirp</h1>
<p align="center">A voice engine that serves many calls per process.</p>
<p align="center">
  <a href="#run">Get started</a> ·
  <a href="#design">Design</a> ·
  <a href="#how-it-compares">Benchmarks</a> ·
  <a href="#models">Models</a>
</p>

A call is a WebSocket carrying 16 kHz mono PCM16 in and out plus JSON
events, a WebRTC session (Opus audio, events on a data channel), or a phone
call over SIP (G.711). chirp listens (Silero VAD), decides when the caller has finished
(Smart Turn v3), transcribes and speaks through streaming services
(ElevenLabs), handles barge-in, and gets its answers from a pluggable
**brain**: any streaming chat backend, behind a two-method trait.

```text
caller audio ─► VAD (batched across calls) ─► end-of-turn model ─► STT commit
                                                                       │
caller ◄── audio ◄── TTS ◄── speakable text ◄── brain (your backend) ◄─┘
```

## How it compares

Measured against Pipecat running the same models on the same machine,
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
789 verdicts on Pipecat's Smart Turn v3.2 test set. After an "unfinished"
verdict chirp waits 1 s of silence, not Pipecat's 2.5 s: on
Full-Duplex-Bench that cut the delay before answering a natural turn from
2.54 s to 1.59 s, with no more cut-ins during pauses.

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
| `CHIRP_TOKEN` | none (open) | WebSocket and WebRTC callers must send it (`Authorization: Bearer`, or `?token=` from a browser); others get 403 |
| `CHIRP_VAD_THREADS` | cores / 4, at least 2 | |
| `CHIRP_TURN_THREADS` | cores − 2 | |
| `CHIRP_MAX_CALLS` | 60 per core | calls at once, across transports; past it new calls get 503 |
| `CHIRP_RTC_PORT` | 7862 | first UDP port for WebRTC (one per hub); `off` disables WebRTC |
| `CHIRP_RTC_SHARDS` | cores / 4 | WebRTC hubs, on consecutive ports from `CHIRP_RTC_PORT` |
| `CHIRP_PUBLIC_ADDR` | this host's outbound address, that port | where callers send WebRTC media |
| `CHIRP_SIP_PORT` | off | UDP port to answer SIP calls on (e.g. 5060) |
| `CHIRP_SIP_RTP_PORT` | 7864 | UDP port every phone call's audio shares |
| `CHIRP_SIP_ALLOW` | any (warned) | addresses or CIDR blocks allowed to place calls: your trunk's |
| `CHIRP_SIP_RTP_TIMEOUT_SECS` | 10 | a caller silent on RTP this long has gone; chirp hangs up |
| `CHIRP_TURN_GRACE_MS` | 1000 | after Smart Turn says "unfinished", this much silence ends the turn anyway (Pipecat: 2500) |
| `CHIRP_SPOKEN_CHARS` | no cap | past this many characters a reply stops at the next sentence end; use only where the rest is shown elsewhere |
| `CHIRP_SPOKEN_TAIL` | none | said after such a cut, e.g. " The rest is in the transcript." |
| `CHIRP_DRAIN_SECS` | 30 | after SIGTERM, new calls are refused and calls in progress get this long to end |
| `CHIRP_MODELS_DIR` | `models` | missing models are downloaded here at startup |
| `PORT` | 7861 | |

Building needs cmake (libopus is compiled in).

`GET /metrics` serves Prometheus metrics: calls active, started, refused
(by reason) and failed; turns by what ended them; and histograms (5 ms to
10 s buckets) of the time from the end of a turn to the reply's first
audio and of Smart Turn's queue wait, for percentiles with
`histogram_quantile`. `GET /health` answers `ok`. Neither needs
`CHIRP_TOKEN`. A token sent as `?token=` can end up in proxy access logs;
prefer the header where the client can set one.

## Transports

- **WebSocket**, `GET /v1/calls`: binary messages are 16 kHz mono PCM16 both
  ways (20 ms frames suit), text messages are JSON events. On `interrupt`,
  drop any reply audio not yet played.
- **WebRTC**, `POST /v1/webrtc` with the browser's SDP offer as the body
  (`201` with the answer, as WHIP). Offer one audio track and a data channel;
  events arrive on it. chirp is ICE-lite with one host candidate, and paces
  and flushes reply audio itself. A few hubs (one UDP socket each) carry
  every WebRTC call; each call's Opus work runs in its own task.
- **SIP** (phone calls), when `CHIRP_SIP_PORT` is set: a SIP trunk or
  softphone sends an INVITE with G.711 (PCMU or PCMA) offered; chirp answers
  and every call's RTP shares one socket. Re-INVITEs (session refreshes) are
  answered; a call whose audio stops is hung up. It is a minimal user agent
  server: UDP only, no digest authentication (set `CHIRP_SIP_ALLOW` to the
  trunk's addresses), no 200 OK retransmission, no hold or DTMF, and media
  must come from the address in the caller's SDP (no NAT latching).

## Your own brain

To answer with something other than a chat-completions API, implement
`Brain` and hand `chirp::server::serve` a function that makes one per call.
It sees the call's `CallRequest` (transport, caller address, headers, query;
for SIP, the INVITE's `from`, `to` and `call_id`) and returns `None` to
refuse it (403), so one server can give each caller their own brain. It
reads the same environment variables:

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
chirp::server::serve(|_| Some(Echo)).await
# }
```

To mount the call routes beside your own, take `shared_from_env().await?`,
build the routes with `chirp::server::router(shared.clone(), make_brain)`,
and serve the combined axum `Router` with
`chirp::server::listen(router, shared.admission.clone())` (`serve` also
wraps `make_brain` in `require_token_from_env`);
`chirp::server::sip_from_env` adds phone calls the same way `serve` does.

`Utterance::continues` is set when the caller talked over the previous
reply before hearing a second of it: the turn end was a pause mid-thought,
so the text continues the previous utterance (`OpenAiBrain` merges the two
and drops the reply that was cut off).

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
