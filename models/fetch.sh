#!/usr/bin/env bash
# Download the two ONNX models this crate runs, pinned by URL and checksum.
set -euo pipefail
cd "$(dirname "$0")"
fetch() {
  local file=$1 url=$2 sha=$3
  if [ -f "$file" ] && echo "$sha  $file" | shasum -a 256 -c --status; then return; fi
  curl -fsSL "$url" -o "$file.part"
  echo "$sha  $file.part" | shasum -a 256 -c --status || { echo "checksum mismatch: $file" >&2; rm -f "$file.part"; exit 1; }
  mv "$file.part" "$file"
}
# Silero VAD as Pipecat 1.13 vendors it (the parity reference).
fetch silero_vad.onnx \
  https://raw.githubusercontent.com/pipecat-ai/pipecat/v1.13.0/src/pipecat/audio/vad/data/silero_vad.onnx \
  1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
# Smart Turn v3.2 (CPU, int8).
fetch smart-turn-v3.2-cpu.onnx \
  https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/f766f81d3cfdf7737ac64aad813d91bbfd56bf93/smart-turn-v3.2-cpu.onnx \
  2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f
