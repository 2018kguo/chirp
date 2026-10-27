# Changelog

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
- **WebRTC turns end 10 ms sooner** (p50 765 → 755 ms): a header-only Opus
  packet (the browser entering DTX) means silence now, without first
  waiting out network jitter.


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
