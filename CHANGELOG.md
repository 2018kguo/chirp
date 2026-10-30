# Changelog

## 0.5.0 (2026-10-30)

- **Backchannels don't stop the reply.** Speech over a reply pauses it at
  once (`hold`), and the transcriber's words decide: a backchannel or noise
  ("mm-hmm", "yeah", a cough) plays on from the pause (`resume`) and isn't
  answered as a turn; real words, or more than 1.5 s of speech, interrupt.
  WebRTC and SIP pause and resume server-side; WebSocket clients get
  `hold` and `resume` events. Full-Duplex-Bench v1.5, 40 each, vs 0.4.0:
  backchannels the reply carried on through 0.28 -> 0.85 (silent after
  0.65 -> 0.15; speaking again 0.23 s after vs 1.06 s), interruptions
  answered 0.45 -> 0.47, the reply stopping 0.38 s vs 0.35 s after the
  caller cut in.
- The call reducer's model check covers backchannels, partial transcripts
  and the overlap timer (it found five bugs in this change before it ran).


## 0.4.0 (2026-10-29)

Fixes found by running two public speech-to-speech benchmarks against chirp
with real providers (ElevenLabs, `gpt-4.1` as the brain): Big Bench Audio
(100 questions) and Full-Duplex-Bench v1/v1.5 (`bench/` has the harnesses).

### Breaking

- **`Utterance::continues`.** When the caller talks over a reply before
  hearing a second of it, the turn end was a pause mid-thought: the next
  utterance says it continues the previous one, and `OpenAiBrain` merges the
  two and drops the reply that was cut off. Big Bench Audio: 75.6 vs 67.6 of
  100 (mean of 5 runs each; it won all 3 interleaved pairs).
- **No spoken cap by default.** Replies used to stop after 350 characters
  with " The rest is in the transcript.", which on Big Bench Audio cut 24 of
  25 web-of-lies answers before the verdict (53 -> 75-81 of 100). Opt back
  in with `CHIRP_SPOKEN_CHARS` and `CHIRP_SPOKEN_TAIL` where the rest is
  shown elsewhere.

### Fixed

- **Calls no longer die on natural speech.** ElevenLabs ends a realtime
  session (`1000 commit_throttled`) when two commits come within ~200 ms;
  on natural conversation that ended 10 of 40 calls. Commits are now at
  least 500 ms apart (0 of 40).
- **Why speech-to-text failed is kept.** Its close code and reason, or the
  connection error, become the call's error.

### Changed

- **Faster turn-taking.** After Smart Turn says "unfinished", 1 s of silence
  ends the turn, not 2.5 s (`CHIRP_TURN_GRACE_MS`). Full-Duplex-Bench, full
  sets: the same take-overs during pauses (19 vs 20 of 353), turn-taking
  latency on natural conversation 2.54 s -> 1.59 s.


## 0.3.1 (2026-10-27)

- **SIP:** ACK and BYE must carry chirp's To tag (sent only in the 200
  OK), so someone who knows only a Call-ID can't stop an answer's
  retransmission or end a call; a BYE without it gets 481. A CANCEL is
  always too late (every INVITE is answered at once) and no longer ends
  the call (RFC 3261 9.2).
- **Metrics:** a histogram's `+Inf` and `_count` are never below a finite
  bucket, even while timings are being recorded.


## 0.3.0 (2026-10-27)

### SIP

- **Answers survive packet loss.** Over UDP a 200 OK can be lost; chirp
  now resends it (RFC 3261: 500 ms, doubling to 4 s) until the caller's
  ACK arrives, and hangs up if none comes within 32 s. Re-INVITE answers
  too. Before, one lost answer dropped the call.
- A call ended by chirp is reported by Call-ID, so a late notice can no
  longer hang up a newer call that reused the same media address.

### Metrics

- First-audio and Smart Turn queue-wait timings are Prometheus histograms
  (5 ms to 10 s buckets): percentiles with `histogram_quantile`, not only
  means. `_sum` and `_count` are unchanged.

### Measured, not shipped

- Batching Smart Turn decisions across calls: 36.9 ms per decision alone,
  35.7 at best batched (M-series, one thread). Its matrices are already
  large, so batching would only add queueing delay (PLAN.md).


## 0.2.0 (2026-10-27)

### Breaking

- **Brain factories see the call.** `make_brain` is now
  `Fn(&CallRequest) -> Option<B>`: it gets the transport, the caller's
  address, the HTTP headers and query (for SIP, the INVITE's `from`, `to`
  and `call_id`), so one server can give each caller their own brain.
  Returning `None` refuses the call (403) before any paid session opens.
  Migrate `serve(|| brain)` to `serve(|_| Some(brain))`.
- `server::listen` takes the `Admission` too (`shared.admission.clone()`),
  so it can drain.

### Added

- **Tokens:** set `CHIRP_TOKEN` and `serve` refuses WebSocket and WebRTC
  callers without it (`Authorization: Bearer`, or `?token=` from a
  browser). `server::require_token_from_env` wraps your own factory the
  same way. SIP stays behind `CHIRP_SIP_ALLOW`.
- **Graceful drain:** on SIGTERM, new calls are refused on every transport
  and calls in progress get `CHIRP_DRAIN_SECS` (default 30) to end.
- **`GET /metrics`** (Prometheus): calls active, started, refused (by
  reason) and failed; reply errors; turns by what ended them; mean time
  from end of turn to first reply audio; Smart Turn queue wait.

### Faster and smaller

Interleaved A/B at 50 calls (live heap per call):

- **Memory:** WebSocket 130 → 112 KB, SIP 132 → 114 KB, WebRTC 268 →
  218 KB. The speech-to-text TLS handshake (~10 KB) no longer sits in every
  call's future for its whole life, WebRTC negotiates no congestion
  feedback it never used, and str0m's 10 KB session state is boxed.
- **Benchmark:** WebRTC callers now send every 20 ms frame, silence
  included, as browsers do (chirp's answer doesn't offer `usedtx=1`); they
  used to drop silent frames, which made WebRTC look slower than it is.

### Fixed

- A turn still ends when a WebRTC caller sends nothing in silence (now
  tested directly).

## 0.1.0 (2026-10-27)

The first release: a voice engine that serves many calls per process.

### Calls

- **Transports:** WebSocket (16 kHz PCM16 and JSON events), WebRTC (Opus;
  events on a data channel; `POST /v1/webrtc`, as WHIP) and SIP phone calls
  (G.711 over RTP; opt in with `CHIRP_SIP_PORT`). All share one call loop
  (`call::run`) behind a transport seam.
- **Turn-taking:** Silero VAD batched across calls, then Smart Turn v3.2 on
  pauses, with a deadline and a VAD fallback under load. Transcripts are
  committed speculatively at the first pause, so they are ready when the
  turn ends.
- **Speech:** ElevenLabs realtime speech-to-text and streaming text-to-speech.
  Barge-in stops the reply at once, including audio already queued.
- **Your backend:** a `Brain` trait (`reply`, and `unprompted` for things the
  caller didn't ask for). `server::serve` runs a server around it;
  `server::router` / `shared_from_env` / `listen` / `sip_from_env` mount it
  beside your own routes. An OpenAI-compatible brain is included.
- **Models** download at first start, pinned by URL and SHA-256.

### Measured

On an AWS c7a.4xlarge (16 vCPU), against Pipecat 1.13 with the same models
and scripted mock services:

- **Engine time:** 182 ms from end of speech to first reply audio (Pipecat
  315 ms).
- **Calls per box:** about 1,000 calls over WebSocket with 0 failures (p99
  890 ms); WebRTC at 800 calls with 0 failures (p99 903 ms). Pipecat's one
  process falls behind from about 10 calls.
- **CPU per call-second:** about 12 ms over WebSocket, 13 ms over WebRTC,
  10 ms over SIP (Pipecat about 305 ms). Flat from 100 to 1,000 calls: each
  call costs O(1).
- **Memory:** about 133 KB per call (Linux, jemalloc).

[PLAN.md](PLAN.md) has every measurement, its method, and what was tried and
rejected.

### Correctness

- **Call decisions** are a pure reducer checked over every interleaving of
  turn events, transcripts and replies (`tests/call_state_model.rs`). Real
  audio is checked to produce only orderings the model explored
  (`tests/turn_grammar.rs`).
- **Parity:** Silero and Smart Turn outputs match Pipecat's Python reference
  on the same model files.
- **Failure handling:** calls end promptly when a provider fails. A lost
  transcript commit can't leave a call deaf, a failed poll doesn't end a
  call, and a stalled host catches up without flooding the transcriber.
- **Limits:** a call limit (`CHIRP_MAX_CALLS`) refuses calls before any paid
  session opens. Overload refuses new calls (503) instead of slowing every
  call. SIP accepts only allow-listed addresses (`CHIRP_SIP_ALLOW`) and hangs
  up on callers whose audio stops.

### Known limits

- **SIP:** UDP only, no digest authentication, no NAT latching, DTMF or hold.
- **WebRTC:** one host candidate per hub and no TURN, so `CHIRP_PUBLIC_ADDR`
  must be reachable by callers.
