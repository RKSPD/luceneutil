#!/usr/bin/env bash
#
# >RAM test at 10M vectors, nlist=10000 (~1000 docs/cell), spill=5, margin=1.15, qb8.
#
# WHY 10M RATHER THAN 40M: the 40M cold arms were unusable in practice -- the BASELINE arm alone measured
# 1286.8 ms/query at nprobe=40 (99% I/O stall, 0 QPS) and would have taken ~3.4 h for its 5-point sweep.
# 10M shrinks the index to ~33 GiB, so a cold 1000-query pass is ~2-7 min per nprobe point instead of
# 21+. Same >RAM regime, ~10x the iteration speed.
#
# HOW THE >RAM REGIME IS FORCED: the balloon pins ~229 GiB of anon so only TARGET_CACHE_GIB is left for
# page cache, then posix_fadvise(DONTNEED) evicts the index, then KNN_DROP_CACHE_AFTER_WARMUP=1 re-evicts
# AFTER the harness's warmup pass and immediately before timing (without that last step the warmup
# re-faults the working set and the "cold" row is a WARM number -- benchmarks.md 13 bug 2 / 15).
#
# Cache is sized to ~18% of the index (6 GiB of ~33 GiB). Deliberately NOT sized off vec_RAM: that is a
# FORMULA (docCount * encodingByteSize * dim), blind to nlist/spill/sketch, and prints the same value for
# configs with 2x different touched sets -- see the correction in run_40m_p2_sweep.sh.
#
# VERIFY COLDNESS FROM THE ROWS, NOT THE CONFIG: avgCpuCount ~1.0 means CPU-bound => the arm is WARM and
# invalid as a cold measurement. Well below 1.0 means genuinely I/O-stalled.
#
# ARMS: BASELINE (no async) is the serialized-fault floor; CD is Stage C+D (batch+coalesce); SEF adds
# Stage E (streaming coarse) + Stage F (streaming rerank). Recall MUST be identical across all three --
# these only change how bytes are fetched. A recall difference is a bug, not a tuning artifact.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb

export KNN_NDOC=10000000
export KNN_NLIST="${KNN_NLIST:-10000}"          # ~1000 docs/cell
export KNN_SPILL_BITS=5
export IVF_SPILL_MARGIN=1.15
export IVF_QUANTIZER=blocksphere
export IVF_BLOCK_P=2
export IVF_QUANT_BITS=8
export IVF_BEAM_SPILL=1
export IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32
export IVF_CENTROID_HNSW_BEAM_WIDTH=64
export IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000
export IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE="${KNN_NPROBE:-20,40,80}"
export KNN_NQUERY="${KNN_NQUERY:-1000}"
export KNN_SKIP_SMELL=1
export LLOYD_URING_DEBUG=1
export LLOYD_RERANK_AUDIT=1

TARGET_CACHE="${TARGET_CACHE_GIB:-6}"           # ~18% of a ~33 GiB index

# ---------------- BUILD (unballooned: the GT + index build need RAM) ----------------
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "=== [build] $(date) ==="
  ./ram_balloon.sh --release >/dev/null 2>&1 || true
  KNN_CLEAR_CACHE=1 KNN_HEAP=32g KNN_NPROBE=40 KNN_DROP_CACHE_AFTER_WARMUP=0 \
    ./run_knn_bench.sh 1 > "$OUT/run_10m_build_${STAMP}.log" 2>&1
  echo "    exit=$?"
  grep -E "NOTE: (recall|index\(s\)|force_merge|index_size)" "$OUT/run_10m_build_${STAMP}.log" | sed 's/^/    /'
  du -sh knn-reuse/indices 2>/dev/null | sed 's/^/    index on disk: /'
fi

# ---------------- COLD ARMS ----------------
run_arm() {
  local name="$1"; shift
  local log="$OUT/run_10m_ram_${name}_${STAMP}.log"
  echo "=== [cold arm $name] $(date) ==="
  ./ram_balloon.sh --release >/dev/null 2>&1 || true
  BALLOON_HEAP_HEADROOM=12 ./ram_balloon.sh "$TARGET_CACHE" 2>&1 | tail -2 | sed 's/^/    /'
  # Evict the index so row 1 is cold too (a balloon alone starts warm if the build left files cached).
  python3 - <<'PY' 2>&1 | sed 's/^/    /'
import os, pathlib
t = 0
for p in pathlib.Path("knn-reuse/indices").rglob("*"):
    if p.is_file():
        try:
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            t += p.stat().st_size
        except OSError as e:
            print(f"warn {p}: {e}")
print(f"evicted ~{t/2**30:.2f} GiB")
PY
  KNN_CLEAR_CACHE=0 KNN_HEAP=8g KNN_DROP_CACHE_AFTER_WARMUP=1 env "$@" \
    ./run_knn_bench.sh 1 > "$log" 2>&1
  echo "    exit=$?"
  grep "^SUMMARY" "$log" 2>/dev/null | awk -F'\t' '{printf "    recall=%s  lat=%s ms\n",$1,$2}' | sed 's/SUMMARY: //'
  # avgCpuCount is the coldness proof; ~1.0 => warm => the row is not a >RAM measurement.
  grep -E "avgCpuCount" "$log" 2>/dev/null | tail -2 | sed 's/^/    /'
  # Engagement guard: an async arm whose ring never engaged is the pre-existing 13 bug-1 trap.
  if [ "$name" != "BASELINE" ]; then
    grep -c "\[lloyd uring\].*engaged" "$log" 2>/dev/null | sed 's/^/    uring engagement lines: /'
  fi
  ./ram_balloon.sh --release >/dev/null 2>&1 || true
}

run_arm BASELINE LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_PREFETCH_CELLS=0
run_arm CD       LLOYD_URING_SKETCH_SCAN=1 LLOYD_URING_RERANK=1 LLOYD_PREFETCH_CELLS=1
run_arm SEF      LLOYD_URING_SKETCH_SCAN=1 LLOYD_URING_RERANK=1 LLOYD_URING_PIPELINE=1 LLOYD_URING_RERANK_PIPELINE=1 LLOYD_PREFETCH_CELLS=1

./ram_balloon.sh --release >/dev/null 2>&1 || true
echo "=== done $(date). logs: $OUT/run_10m_ram_*_${STAMP}.log ==="
