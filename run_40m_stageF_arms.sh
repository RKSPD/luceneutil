#!/usr/bin/env bash
#
# Stage F rerun driver. Builds the 40M p=2 index ONCE (unballooned-GT fix in run_40m_p2_sweep.sh), then
# runs the nprobe sweep for each async-I/O arm as a SEPARATE PHASE=search invocation. Each search phase
# re-inflates its balloon and fadvise-evicts the index, so every arm starts genuinely cold and the arms
# never share a warm cache -- the asymmetry that made only row 1 real in benchmarks.md §13.
#
# Arms (each is a full nprobe sweep 40,55,70,90,120 x 1000 queries, one cold index):
#   BASELINE : no async I/O (per-doc mmap). The floor.
#   CD       : Stage C (batched sketch) + Stage D (blocking coalesced rerank gather). Current default.
#   SEF      : Stage E (streaming coarse) + Stage F (streaming rerank). The overlap under test.
#   DIRECT   : Stage C/D + O_DIRECT (pure-device ceiling; submit/reap defer is unavailable under O_DIRECT
#              so E/F fall back to the blocking direct readBatch -- this arm measures no-cache, not overlap).
#
# usage:  ./run_40m_stageF_arms.sh              # build + all arms
#         SKIP_BUILD=1 ./run_40m_stageF_arms.sh # arms only (index must already be cached)
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb
BUILD_LOG="$OUT/run_40m_stageF_build_${STAMP}.log"

# ---- BUILD ONCE (also computes+caches the 1000-query GT under the roomy build balloon) ----
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "=== [build] $(date) -> $BUILD_LOG ==="
  PHASE=build RESULTS="$OUT/results_40m_stageF_build_${STAMP}.txt" \
    ./run_40m_p2_sweep.sh > "$BUILD_LOG" 2>&1
  BRC=$?
  echo "=== [build] exit=$BRC ==="
  if [ "$BRC" -ne 0 ]; then
    echo "BUILD FAILED (exit $BRC) -- not running arms. Tail:"; tail -20 "$BUILD_LOG"; exit "$BRC"
  fi
fi

# ---- SIZE THE SEARCH BALLOON SMALL ENOUGH TO STAY COLD --------------------------------------------
# WARNING -- vec_RAM IS NOT A MEASUREMENT. KnnGraphTester.java:1301 computes it as a FORMULA:
#   vectorRAMSizeBytes = totalVectorCount * (realEncodingByteSize * dim + overhead)
# It depends ONLY on doc count / dim / encoding byte size -- it knows nothing about nlist, spill
# duplication, or the sketch table, so it prints the SAME 39442 MB for wildly different configs (verified:
# identical for nlist=100k/spill10/qb4 and nlist=6306/spill5/qb8). The pre-existing "size against vec_RAM,
# not du" comment in run_40m_p2_sweep.sh is therefore built on a wrong premise -- do not trust it.
#
# What a query ACTUALLY touches on this index: the whole sketch table (~14 GiB at spill=5, scanned by the
# coarse tier) plus scattered reads into a ~116 GiB code table. Neither is 38.5 GiB.
#
# So we do not try to compute the touched set precisely -- we just pick a cache FAR BELOW it, which is the
# only property that matters for a cold measurement. vec_RAM/4 is used purely as a convenient small number
# (~9 GiB), NOT because it means anything. The du-based default (INDEX_GIB/4 = 29 GiB) is the thing to
# avoid: it can leave the sketch table fully resident and turn the "cold" row WARM (§13/§15 trap).
#
# VERIFY, DO NOT ASSUME: the result rows' avgCpuCount is the ground truth. ~1.0 => CPU-bound => WARM (the
# measurement is invalid as a cold number); well below 1.0 => genuinely I/O-stalled => cold.
VEC_RAM_MB=$(grep -oE "vec_RAM\(MB\) = [0-9.]+" "$BUILD_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
if [ -n "${VEC_RAM_MB:-}" ]; then
  # bash has no float math; take the integer MB, /1024 for GiB, /4 for the quarter-cache target.
  VEC_RAM_GIB=$(( ${VEC_RAM_MB%.*} / 1024 ))
  TARGET=$(( VEC_RAM_GIB / 4 ))
  [ "$TARGET" -lt 8 ] && TARGET=8
  export TARGET_CACHE_GIB="$TARGET"
  echo "=== balloon sizing: vec_RAM=${VEC_RAM_MB} MB (~${VEC_RAM_GIB} GiB touched) -> TARGET_CACHE_GIB=${TARGET_CACHE_GIB} ==="
else
  echo "=== WARNING: could not parse vec_RAM from build log; falling back to sweep's du-based INDEX_GIB/4 ==="
  echo "    (that OVERSHOOTS -- the cold arms may be partly warm. Check the first-row avgCpuCount ~1.0 = warm.)"
fi

# ---- ARMS ----
# Each entry: NAME|extra env assignments (space-separated KEY=VAL). Defaults in run_40m_p2_sweep.sh are
# SKETCH_SCAN=1, RERANK=1, PIPELINE=0, RERANK_PIPELINE unset(0), DIRECT unset(0) -- i.e. arm "CD".
run_arm() {
  local name="$1"; shift
  local res="$OUT/results_40m_stageF_${name}_${STAMP}.txt"
  local log="$OUT/run_40m_stageF_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) -> $log ==="
  env "$@" PHASE=search RESULTS="$res" ./run_40m_p2_sweep.sh > "$log" 2>&1
  local rc=$?
  echo "=== [arm $name] exit=$rc ==="
  # Surface the result rows and the engagement/guard verdict without dumping the whole log.
  grep -E "recall|latency\(ms\)|FATAL|engagement OK|search phase exit" "$res" 2>/dev/null | tail -20
  # STAGE F GUARD: the SEF arm is only meaningful if Stage F actually streamed the rerank. If it silently
  # fell back to Stage D (e.g. a coalesced range exceeded a lane), the numbers would be CD wearing SEF's
  # label -- the §13 bug-1 trap. The reader prints "Stage-F streaming rerank engaged" once when it does.
  if [ "$name" = "SEF" ]; then
    if grep -q "Stage-F streaming rerank engaged" "$log"; then
      echo "  [guard] SEF: Stage F engaged (OK)"
    else
      echo "  [guard] FATAL: SEF arm requested Stage F but the reader never printed it engaged --"
      echo "          these rows are the Stage D blocking path, NOT the streaming overlap. Do NOT record."
    fi
  fi
  return 0
}

run_arm BASELINE LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0
run_arm CD       LLOYD_URING_SKETCH_SCAN=1 LLOYD_URING_RERANK=1
run_arm SEF      LLOYD_URING_SKETCH_SCAN=1 LLOYD_URING_PIPELINE=1 LLOYD_URING_RERANK=1 LLOYD_URING_RERANK_PIPELINE=1
run_arm DIRECT   LLOYD_URING_SKETCH_SCAN=1 LLOYD_URING_RERANK=1 LLOYD_URING_DIRECT=1

echo "=== all arms done $(date). results: $OUT/results_40m_stageF_*_${STAMP}.txt ==="
