# chirp plan

chirp is a voice engine: one Rust process serves many concurrent voice calls,
each a WebSocket carrying audio in and out, with turn-taking (VAD + end-of-turn
model), barge-in, streaming speech-to-text and text-to-speech, and a pluggable
"brain" (any streaming chat backend). Goal: the best open voice engine on the
numbers that matter, not a port.

## Rules

1. **Oracles before claims.** Every model path is checked against the reference
   implementation's outputs on the same model files (Pipecat, Python). Every
   behavior change lands with a test that fails without it (mutation-checked).
2. **Profile first.** Before optimizing anything, profile exactly what work runs
   at that moment, and profile the reference doing the same job side by side.
   Theory only picks what to profile next.
3. **Honest numbers.** Benchmarks are interleaved, repeated, reported with
   bootstrap confidence intervals, on named hardware, against the reference
   running the same scenario. No number in docs that a script in this repo
   can't reproduce.
4. **Aim for #1, question ceilings.** "X can't go below Y" is a hypothesis to
   attack with an experiment, not a stopping point.
5. **Prior art first when stuck:** read the fastest real implementations
   (file:line) before inventing.
6. **Big-O and soak gates in CI:** per-frame cost must not grow with call age,
   transcript length or call count (doubling tests), and a long soak must not
   leak.

## Scoreboard (what "best" means)

| Metric | How measured | Reference |
|---|---|---|
| Engine latency: user's last word → first reply audio byte out, minus brain and TTS time | `bench/latency` with local mock STT/TTS/brain (deterministic delays), real audio | Pipecat on the same mocks |
| Turn-taking quality: false end-of-turn rate on mid-sentence pauses; end latency on complete turns | `bench/turns` labeled utterance set | Pipecat Smart Turn pipeline |
| CPU per call-second, at 1 / 100 / 1,000 / 5,000 calls | `bench/load` real audio, real-time pace | Pipecat process-per-call, same audio |
| Memory per call; process RSS at N calls | same | same |
| p99 frame latency under load (VAD, turn) | same | same |
| Robustness | fuzzed client frames, provider disconnects, slow clients, soak 6 h | — |

## Phases

- **0. Split.** Generic engine here; product adapters (a brain backed by a
  specific chat product) live in their own repos. Brain trait + an
  OpenAI-compatible example brain. Parity tests + CI green.
- **1. Truth harness.** Mock STT/TTS/brain servers with scripted timing;
  latency, turn-quality and load benches with CIs; the same scenarios driven
  through Pipecat as the reference. Profiling recipes (samply/Instruments,
  tokio-console, per-stage timelines in every call) as first-class tools.
- **2. Optimize from profiles.** Candidates, to be confirmed or killed by
  profiles: speculative brain start at VAD stop (cancel if the user resumes);
  pre-warmed TTS connections (one TLS handshake per call, not per reply);
  not waiting on the STT commit round trip when partials already cover the
  turn; allocation-free frame path; no network round trip inside the audio
  loop (the unprompted-reply poll and the STT send currently run inline); Smart Turn inference cost (threading,
  execution providers, earlier/overlapped decisions); Opus in and out.
- **3. Hardening.** Doubling + soak + fuzz gates in CI; graceful drain on
  deploy; backpressure for slow clients.
- **4. Release and adoption.** Public since Oct 27 2026. The first product
  brain lives in its product's repo and depends on this crate by git revision.

## Findings log

- **Batching Smart Turn doesn't help** (`turn_bench`): 31.6 ms per decision at
  batch 1 and 32.6 at batch 16, one thread, M5 Pro. Profile (ORT 1.31):
  QLinearMatMul 48%, QLinearSoftmax 15.5%, Erf (unfused GELU) 9.5%,
  LayerNorm 7.6%. The matmuls already run ~107 GMAC/s; the fusible ~35% is
  the only fat. Candidate: a fused pure-Rust encoder behind the parity test.
- **Smart Turn is the throughput limit** (`bench_load`): 500 calls on 2 turn
  threads answered 350 of ~1,500 turns; on 8 threads, all of them at flat
  latency (p50 833 ms, p99 837 ms with 550 ms scripted), 10 ms CPU per
  call-second. Thread count must follow load.
- **Deciding the end of turn early trades quality for speed**
  (`bench_turns`, Smart Turn v3.2 test set, English, 789 clips): ending at 2
  quiet windows when p >= 0.9 decides ~125 ms sooner than the baseline but
  raises cut-offs (a turn ended, then the speaker kept talking) from 52% to
  60% on human speech. So: keep the decision, speculate the work.
- **Memory per call** (`bench_load` with `--features heap-stats`, slope
  between call counts so fixed costs cancel): live heap 890 → 120 KB per call
  (i16 rings that grow with the turn: −492 KB; WebSocket read buffers 128 KiB
  → 4–8 KiB: −282 KB); RSS slope ~175 KB per call. The earlier "1.4 MB per
  call" divided fixed costs by the call count. Allocations 1,112 → ~230 per
  call-second (hand-written STT messages into a reclaimed BytesMut, a reply
  slot per call instead of a oneshot per window, reused buffers).
- **Fixed cost ~450 MB under load**, mostly the Smart Turn sessions' ONNX
  Runtime arenas. Turning the arena off saves ~116 MB but costs +20% per
  decision; capping the VAD batch changed nothing. Smart Turn threads are now
  warmed at startup: arenas exist before the first call, and p99 at 1,000
  calls fell from 975–1,649 ms to 740 ms (cold sessions had paid their first
  run on some caller's pause).
- **TLS config per connection was the biggest production cost the mocks
  hid**: tokio-tungstenite builds a fresh rustls config (root certificates
  loaded and parsed, ~74 ms CPU on macOS) for every connection, and chirp
  connects per call (STT) and per reply (TTS). One shared config: live
  ElevenLabs connects 182–322 ms → 86–172 ms (`tests/live_elevenlabs.rs`,
  `--ignored`). Sockets also skip Nagle now (small audio writes).
- **Allocator on Linux** (`scripts/linux_memory.sh`, Docker, 4 CPUs; RSS
  slope 50→200 calls, live heap ~118 KB/call in all): glibc 207 KB/call
  (185 MB at 200 calls); glibc MALLOC_ARENA_MAX=2 187 KB (176 MB); mimalloc
  153 KB but +70 MB baseline (247 MB); jemalloc 133 KB (179 MB). jemalloc is
  now the Linux default. CPU and latency equal across all four.
- **Rejected: VAD state in a per-call slot, 1 KB requests** (instead of
  3.3 KB through the queue). Interleaved A/B at 400 calls: no difference
  (CPU 8.95–9.38 vs 9.04–9.91 ms per call-second; same latency). At 1,000
  calls on a machine already loaded by other work it lost both rounds
  (p99 ~3.0 s vs 1.0–1.9 s); plausible cause: the VAD thread holds a call's
  std mutex during gather/scatter while that call's task locks it from async
  code, so a descheduled VAD thread can block a whole tokio worker. Reverted.
- **Noise floor:** with the host at load ~11–18 from other work, 1,000-call
  runs saturate and their latency varies by 2–3x between runs; compare
  builds interleaved, and below saturation for small effects.
- **Smart Turn kernel profile** (samply, one thread, M5 Pro, ORT 1.31):
  int8 GEMM 40% using MLAS's `UDOT` kernel (not the i8mm `UMMLA` one this
  CPU supports; MLAS detects i8mm on Linux, so Graviton 3+ likely runs
  faster than this laptop and x86 uses VNNI kernels: laptop numbers are not
  production numbers); QLinearSoftmax 15.8%; LayerNorm 9.1%; packing,
  quantize/dequantize and elementwise the rest. The softmax and layer norm
  quarter is the fusible part.
- **Benchmarking policy:** no load tests at scale on the owner's laptop.
  Local: ≤50 calls, short, niced; single-thread microbenchmarks and
  profiles. Scale curves go to remote hardware.
- **Production CPU, AWS c7a.4xlarge (16 vCPU, EPYC 9R14), Oct 26 2026.**
  Engine latency on the mocks: chirp 182 ms vs Pipecat 315 ms (medians,
  n=30 each, CIs ±1 ms). Smart Turn: 38 ms per decision on one thread
  (Pipecat publishes 9 ms int8 on c7a; thread count unverified).
  Load, 30 s, real audio, mocks on the same box:

  | Engine | Calls | Cores | CPU ms per call-s | Failed | p50 / p99 ms |
  |---|---|---|---|---|---|
  | chirp | 500 | 5.9 | 11.8 | 0 | 734 / 741 |
  | chirp | 1,000 | 12.6 | 12.6 | 0 | 777 / 890 |
  | chirp | 1,500 | 10.2 | 6.8 | 605 refused (503) | 986 / 1,235 |
  | chirp | 10 | 0.13 | 13.2 | 0 | 731 / 737 |
  | chirp | 25 | 0.33 | 13.2 | 0 | 731 / 753 |
  | Pipecat | 10 | 3.1 | 312 | 0 (40 of ~50 turns) | 871 / 885 |
  | Pipecat | 25 | 7.6 | 302 | 0 (83 of ~125 turns) | 1,299 / 2,635 |
  | Pipecat | 50 / 100 | — | — | 28 / 0 turns completed | — |

  Pipecat runs every call in one uvicorn process with a Silero session per
  call; its event loop falls behind from 10 calls on this box, so its CPU
  per call-second is a floor. Rerun after the fd-limit fix: unchanged.
  Python onnxruntime 1.24.4 times Smart Turn at 37.5 ms on one thread
  (chirp: 38.4), 21.5 ms on 4, 14.6 ms with its default threads: Pipecat's
  9 ms is a many-thread number; one thread per decision is the throughput
  optimum, and chirp's kernels are at onnxruntime parity.
  Capacity on this box is ~1,000 calls, bound by Smart Turn CPU.
- **Three silent caps found on Linux, each invisible on macOS:**
  1. The open-file soft limit (1,024): past ~300 calls, reply connections
     failed and calls went unanswered (550 turns of ~2,500 at 500 calls).
     chirp now raises its soft limit to the hard limit at startup, and
     bench_load fails a call on any `error` event instead of ignoring it.
  2. Thread defaults: half the cores for Smart Turn refused 388 of 1,000
     calls. Now all but two cores for Smart Turn and a quarter (min 2) for
     the VAD. Two VAD threads ran at their limit at 1,000 calls (p99 frame
     wait 23 ms vs 5.5 ms with 4, same CPU).
  3. bench_load's `kill -KILL -<pgid>`: procps-ng 4.0.4 reads it as every
     process the user owns while the group is alive, killing the benchmark
     suite and its SSH session. Now `kill -s KILL -- -<pgid>`.
- **CPU profile at 500 calls (c7a, perf, 15 s):** chirp's own code ~3%
  (mel 1.3%, call loop 0.8%, ring 0.4%); kernel networking 12.5%; the rest
  onnxruntime: int8 GEMM `MlasGemmU8X8KernelAvx512Core` 27.4%,
  QLinearSoftmax 12.0%, fp32 GEMM 10.5%, requantize 3.1%. The GEMMs run
  (correction below: the weight GEMMs already run VNNI).
- **Rejected: dynamic U8S8 requantization** of the fp32 model
  (`smart-turn-v3.2-gpu.onnx`, value_info stripped): 48.8 ms vs 37.5 ms
  for the published static U8U8, one thread (fp32: 74.0 ms); per-channel
  the same. Closer to fp32's output, but a DynamicQuantizeLinear per matmul
  and fp32 softmax cost more than VNNI saves.
- **Correction: the published Smart Turn int8 model is already static QDQ
  U8S8, and already runs VNNI.** Its 31 weight tensors are INT8; Pipecat's
  train.py quantizes with QDQ, QUInt8 activations, QInt8 per-channel
  weights, Entropy calibration on 1,024 stratified clips, Conv/MatMul/Gemm
  only. perf annotate of `MlasGemmU8X8KernelAvx512Core` on the c7a: 42% of
  its samples on `vpdpbusds` (VNNI, the weight matmuls), 16% on `vpmaddwd`
  (U8U8: attention's activation x activation matmuls). Our own static
  requantization (MinMax, 300 train-split clips) scored 86.3% vs the
  published 94.9% (fp32 95.3%) on the 789 English test clips, 120 verdicts
  flipped: rejected. Smart Turn is at onnxruntime's floor here; what is
  left is QLinearSoftmax (~20% of a decision, onnxruntime's per-row
  std::function) and the attention U8U8 matmuls.
- **Rejected: HTTP/2 to the brain.** Against api.openai.com (reqwest 0.12,
  ALPN): 20 concurrent requests on a warm client open 1 connection over
  HTTP/2 vs 20 over HTTP/1.1; cold, both open 20 (none knows it may share
  until ALPN completes). But OpenAI advertises max_concurrent_streams 100,
  and hyper queues past it: 300 concurrent requests still used 1
  connection. At 1,000 calls ~200-500 replies stream at once, so most would
  wait for a slot, and one dropped connection would fail them all.
  HTTP/1.1 keep-alive already reuses idle connections (no handshake per
  reply); it only costs a socket per concurrent reply.
- **Skipped: a pool of pre-opened TTS connections.** `speaker()` already
  connects in parallel with the brain's first token (`tokio::join!`), so
  the 86-172 ms connect is off the critical path whenever the brain's first
  token takes longer; and a stream-input connection ends with its reply, so
  a spare pool would still handshake once per reply. Multi-context TTS
  would save the handshake (~1 ms CPU each, ~1.5% of CPU at 1,000 calls)
  at the price of shared fate across calls: not worth it.
- **A stall used to end the call; now the backlog is coalesced.** Under heavy
  host load the call loop fell ~7 s behind, then forwarded the queued 20 ms
  frames to ElevenLabs one message each (~350 in 60 ms, with a pause commit
  mid-burst), and ElevenLabs ended the session: `queue_overflow`, "audio data
  is being sent too frequently". Reproduced by freezing chirp (SIGSTOP) for
  7 s mid-call; a logging proxy recorded the exact send pattern, and replaying
  it against ElevenLabs failed 3 of 3. The same audio and timing with each
  backlog merged into one message (capped at 1 s) passed 3 of 3. Bursts alone,
  long bursts and idle-then-burst all passed on a fresh session, so the
  trigger is the flood after a session has been streaming. Fix: the call loop
  reads frames already waiting and sends them as one message, up to 1 s.
  Frozen live calls: 4 of 4 answered (were 2 of 2 killed). The mock STT can
  enforce a message rate (`stt_max_messages_per_100ms`) for the regression
  test `call_flow::a_backlog_of_audio_is_sent_in_a_few_messages`.
- **WebRTC transport (Oct 27 2026).** str0m (sans-IO, pure-Rust DTLS) with
  one UDP socket and one hub task for every call; Opus at 16 kHz both ways
  (no resampling); events on a data channel; signaling is one WHIP-style
  POST. Server-side playout queue, flushed on interrupt. Opus cost per
  call-second: encode 5.3 ms at complexity 10, 3.5 ms at 5 (chosen), 1.6 ms
  at 1; decode 0.3 ms. `call::run` now takes any transport (caller audio in,
  `ToCaller` out). Tests: a str0m peer over real UDP is heard and answered,
  and barge-in stops playout within 100 ms (both fail on the matching
  mutation).
- **SIP transport (Oct 27 2026).** A minimal UAS (INVITE/ACK/BYE/OPTIONS,
  retransmitted INVITEs answered from the dialog, BYE sent when chirp ends a
  call) and G.711 RTP on one shared socket keyed by the caller's media
  address. 8 -> 16 kHz by interpolation in; out, an 11-tap half-band
  low-pass then decimation. Shares the server-paced `Playout` with WebRTC.
  Tested end to end with an in-test softphone (independent μ-law encoder);
  fails when inbound audio is silenced. Not yet: digest auth, TCP/TLS,
  200 OK retransmission timers, NAT latching, DTMF, hold. No carrier account
  yet, so no real phone call has been placed.
- **WebRTC: browsers send no audio while the caller is silent (Opus DTX).**
  Found by driving a browser page in headless Chrome: the call received
  6.2 s of audio over 12 s, the VAD never saw the quiet windows that end a
  turn, and no turn ever ended. The str0m test peer had sent silent frames,
  so it missed this; it now skips silent frames like a browser, and both
  WebRTC tests failed until the hub filled caller gaps over 60 ms with
  silence in real time.
- **Transport hardening after an adversarial review (Oct 27 2026).** A SIP
  caller that vanished without BYE held its call (and paid speech sessions)
  forever: calls now end after `CHIRP_SIP_RTP_TIMEOUT_SECS` of no RTP, with a
  BYE. Any client could open unbounded paid sessions: a global call limit
  (`CHIRP_MAX_CALLS`, 60 per core) gates every transport, and
  `CHIRP_SIP_ALLOW` restricts who may place calls. Re-INVITEs got the first
  transaction's stale 200 OK; a second call could take over another's media
  address (now 486). The WebRTC RTP clock stopped while chirp was silent
  (now real time). The WebRTC hub polled every peer on every packet
  (O(N^2) per second at N calls): packets now route by source address and
  only due peers get timeouts. Each has a test that fails without its fix.
- **Transport scaling audit and restructure (Oct 27 2026).** An independent
  big-O review flagged the WebRTC hub as a single-task ceiling: Opus for
  every call, and per-wakeup scans over every peer. Now each call's media
  task does its codec work and pacing; hubs (`CHIRP_RTC_SHARDS`, one UDP
  socket each) do socket I/O and str0m only, route packets by source address
  (a new address only via a STUN check naming a session's ICE ufrag; other
  stray packets cost one hash lookup), and drive only touched or due
  sessions (a deadline heap). `Playout` encodes reply audio as it arrives
  and queues encoded frames (WebRTC ~16x less memory than PCM). SIP media:
  allocation-free inbound path, a polyphase half-band downsampler, O(1)
  dialog lookups, hang-ups that can't be dropped. The call loop reuses one
  commit-deadline timer. c7a.4xlarge, 30 s, CPU ms per call-second and turn
  p99 (bench_load chirp-webrtc / chirp-sip, callers on the same box):

  | | before | after |
  |---|---|---|
  | WebRTC 300 | 13.6 ms, p99 808 | 13.8 ms, p99 810 |
  | WebRTC 600 | 102 refused, p99 2,859 | 0 refused, p99 830 |
  | SIP 1,000 | 10.1 ms, p99 1,839 | 10.4 ms, p99 1,006 |
  | WebSocket 1,000 | 12.2 ms, p99 803 | 12.2 ms, p99 805 |

  WebRTC at 1,000 calls refused 410 with the box at load 16.6 of 16 vCPUs:
  half of it was the benchmark's own callers encoding Opus, so that run
  measures the harness, not chirp.
- **WebRTC scales linearly (Oct 27 2026).** With the benchmark's callers
  sharing one encoding of the question (they used half the box), WebRTC on
  the c7a.4xlarge: 800 calls, 0 refused, p99 903 ms, 12.9 ms CPU per
  call-second; 1,000 calls refused 122 with chirp at 11.4 cores and the box
  at load 16.5 (CPU-bound, as WebSocket is at ~1,000). CPU per call-second
  by calls, WebRTC 100/300/600/800: 14.2 / 13.8 / 13.0 / 12.9; SIP
  100/300/600/1,000: 10.6 / 10.6 / 10.4 / 10.4. Flat: each call costs O(1).
