#!/usr/bin/env bash
# Per-call memory on Linux by allocator (jemalloc is the default build;
# glibc measured 207 KB/call, 187 with MALLOC_ARENA_MAX=2; see PLAN.md), in a rust:1.96.0 container:
#   docker run --rm -v "$PWD":/src -v chirp-target:/target -w /src rust:1.96.0 scripts/linux_memory.sh
set -euo pipefail
export CARGO_TARGET_DIR=/target
for variant in jemalloc mimalloc; do
  features=heap-stats
  [ "$variant" != jemalloc ] && features="heap-stats,$variant"
  cargo build -q --release --features "$features" --bin chirp
  cp /target/release/chirp "/target/chirp-$variant"
done
cargo build -q --release --bin mocks --bin bench_load
for run in "jemalloc" "mimalloc"; do
  set -- $run
  variant=$1; shift
  for calls in 50 200; do
    line=$(env "$@" CHIRP_BIN=/target/chirp-$variant CHIRP_TURN_THREADS=2 /target/release/bench_load chirp $calls 20 2>&1 | grep -E '^(heap|chirp)' | tr '\n' ' ' || true)
    echo "$run calls=$calls :: $line"
  done
done
