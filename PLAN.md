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
