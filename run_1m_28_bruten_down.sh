#!/usr/bin/env bash
# 2/8: osq8 dense int8 rerank + 2-bit Gray Hamming coarse (sharp shortlist). The 2-bit coarse is DECOUPLED
# from the rerank (osq8 has no relation to the sketch), so bruteN should drop FAR below osq8's 1750. Sweep
# bruteN DOWN {1000,750,500,350,250} to find the 0.95 crossing, np40, sp2. Two coarse-threshold arms:
#   arm EW  = equal-WIDTH lo plane (clip*std/2), clip swept via sketchLoClip=2.5 (coarse-only optimum)
#   arm EM  = equal-MASS/density lo plane (QUARTILE*std) via RESIDUAL4_DECOUPLED=1 (osq path: only moves the
#             lo-plane threshold; residual4 code is never reached under quantizer=osq -- verified).
# First build per arm clears cache (osq8+sketch is a distinct format from the bare-osq8 h2h index). runs=3.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

BNS=(1000 750 500 350 250)

run_arm () {
  local tag="$1"; shift
  local first=1
  for BN in "${BNS[@]}"; do
    export LLOYD_BRUTE_N=$BN
    local LOG="$OUT/run_1m_28_${tag}_bn${BN}_${STAMP}.log"
    if [[ $first == 1 ]]; then local CLEAR=1; first=0; else local CLEAR=0; fi
    echo "=== 2/8 $tag bruteN=$BN (clear=$CLEAR) $(date) ===" | tee "$LOG"
    KNN_CLEAR_CACHE=$CLEAR ./run_knn_bench.sh 3 >> "$LOG" 2>&1
    echo "$tag bn$BN exit=$?"
  done
}

# arm EW: equal-width, clip 2.5
export LLOYD_SKETCH_LO_CLIP=2.5
unset IVF_R4_DECOUPLED
run_arm EW

# arm EM: equal-mass quartile threshold (decoupled lo plane); clip irrelevant to the threshold here
export IVF_R4_DECOUPLED=1
run_arm EM

echo; echo "=== 2/8 bruteN-DOWN RESULTS (np40) ==="
for L in "$OUT"/run_1m_28_*_${STAMP}.log; do
  echo "--- $(basename "$L") ---"
  grep "^SUMMARY" "$L" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms sel=%s\n",$1,$2,$4}' | sort -u | head -2
done
echo "=== stamp: $STAMP ==="
