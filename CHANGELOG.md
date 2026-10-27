# Changelog

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
