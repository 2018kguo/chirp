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
- **4. Adoption, then release** (only when the owner says so).

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
