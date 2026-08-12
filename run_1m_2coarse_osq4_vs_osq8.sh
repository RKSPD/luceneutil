#!/usr/bin/env bash
# Two questions in one sweep, with the FIXED shared lo-threshold (doc==query at COARSE_LO_C*measuredStd):
#   (1) validate the threshold fix end-to-end (2-bit coarse should now rank better -> lower bruteN at 0.95).
#   (2) osq4 vs osq8 rerank on the SAME 2-bit coarse shortlist: is 16-level (osq4, 512 B/doc, dense int4, NO
#       reconstruction tax) enough for 0.95, or does the near-twin shortlist need osq8's 256 levels?
# 2-bit Gray Hamming coarse (sketchLoBits), sp2/m1.20, np40. Each quantBits arm is a distinct format (own
# cache key) -> builds once, then bruteN sweep reuses. runs=3.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=osq IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
# COARSE_LO_C default 1.5 is baked; leave lloyd.sketchLoC unset to use it.

BNS=(750 500 350 250)

run_arm () {
  local qb="$1"; local first=1
  export IVF_QUANT_BITS=$qb
  for BN in "${BNS[@]}"; do
    export LLOYD_BRUTE_N=$BN
    local LOG="$OUT/run_1m_2c_osq${qb}_bn${BN}_${STAMP}.log"
    if [[ $first == 1 ]]; then local CLEAR=1; first=0; else local CLEAR=0; fi
    echo "=== 2-coarse osq$qb bruteN=$BN (clear=$CLEAR) $(date) ===" | tee "$LOG"
    KNN_CLEAR_CACHE=$CLEAR ./run_knn_bench.sh 3 >> "$LOG" 2>&1
    echo "osq$qb bn$BN exit=$?"
  done
}

run_arm 8
run_arm 4

echo; echo "=== 2-bit COARSE + osq{8,4} rerank (np40, fixed threshold) ==="
for qb in 8 4; do
  echo "-- osq$qb --"
  for BN in "${BNS[@]}"; do
    L=$(ls -t "$OUT"/run_1m_2c_osq${qb}_bn${BN}_${STAMP}.log 2>/dev/null | head -1)
    S=$([ -n "$L" ] && grep "^SUMMARY" "$L" | sed 's/SUMMARY: //' | awk -F'\t' '{print $1"/"$2}' | sort -u | head -1)
    echo "   bn$BN: $S"
  done
done
echo "=== stamp: $STAMP ==="
