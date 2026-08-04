#!/usr/bin/env bash
#
# A/B of the three WARM-CPU optimizations found by the JFR profile (nlist=6306 run, 56 ms/query,
# avgCpuCount=1.000 so 100% CPU / zero I/O stall). Profile breakdown that motivated each:
#
#   26.5%  xorBitCountInt          -> UNROLLED POPCOUNT (4 independent accumulators, int stride on ARM)
#   18.8%  FixedBitSet.getAndSet   -> SHORTLIST DEDUP (dedup 2000 shortlist entries, not ~250k slots)
#   13.7%  sketchScanCells loop
#   10.5%  MemorySegment bounds/session checks \
#   10.0%  MemorySegmentIndexInput readInt/readBytes > SEGMENT SLICE (hoist Panama checks out of the loop,
#                                                      + skip the per-slot docId read when nothing needs it)
#    2.2%  blockSphereScore (int8 rerank)
#
# All THREE are reader-side: no reindex, no format change. The segment-slice + docId-skip are unconditional
# (they are strictly less work for identical bytes); the other two are flagged so each can be priced alone.
#
# Every arm must report IDENTICAL RECALL to the control. These are pure CPU optimizations -- a recall
# difference means a bug, not a tuning artifact, and that is the whole reason to A/B rather than assume.
# (Unit tests already assert bit-identity: TestXorBitCountSegment vs VectorUtil.xorBitCount across 18
# lengths/odd offsets/extremes, and TestUringRerankGather's shortlist-dedup equivalence cases.)
#
# WARM, async OFF: warm is the ~10 GB/s production proxy, and the ring stages are pure overhead warm
# (measured +7-19%), which would mask the CPU deltas we are trying to price.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb

# Point at whichever index is built. Override NLIST/NPROBE to match it.
export KNN_NDOC=39767748
export KNN_NLIST="${KNN_NLIST:-40000}"
export KNN_SPILL_BITS="${KNN_SPILL_BITS:-5}"
export IVF_SPILL_MARGIN="${IVF_SPILL_MARGIN:-1.15}"
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
export KNN_NPROBE="${KNN_NPROBE:-40,80,150}"
export KNN_NQUERY="${KNN_NQUERY:-1000}"

export KNN_CLEAR_CACHE=0
export KNN_DROP_CACHE_AFTER_WARMUP=0   # WARM
export KNN_HEAP="${KNN_HEAP:-8g}"
export KNN_SKIP_SMELL=1
export LLOYD_URING_SKETCH_SCAN=0
export LLOYD_URING_RERANK=0
export LLOYD_URING_PIPELINE=0
export LLOYD_URING_RERANK_PIPELINE=0
export LLOYD_PREFETCH_CELLS=0

./ram_balloon.sh --release >/dev/null 2>&1 || true

run_arm() {
  local name="$1"; shift
  local log="$OUT/run_40m_cpuopt_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  echo "    exit=$?"
  grep "^SUMMARY" "$log" 2>/dev/null | awk -F'\t' '{printf "    recall=%s  lat=%s ms\n",$1,$2}' | sed 's/SUMMARY: //'
}

# CONTROL: old single-accumulator popcount + per-slot bitset dedup (i.e. pre-optimization behaviour).
run_arm CONTROL      LLOYD_SCALAR_POPCOUNT=1
# Each lever alone.
run_arm POPCOUNT     LLOYD_SCALAR_POPCOUNT=0
run_arm DEDUP        LLOYD_SCALAR_POPCOUNT=1 LLOYD_SHORTLIST_DEDUP=1
# Both.
run_arm BOTH         LLOYD_SCALAR_POPCOUNT=0 LLOYD_SHORTLIST_DEDUP=1

echo "=== done $(date). logs: $OUT/run_40m_cpuopt_*_${STAMP}.log ==="
