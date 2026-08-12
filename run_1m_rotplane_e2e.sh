#!/usr/bin/env bash
# E2E gate for the ROTATION second sketch plane vs the incumbent Gray sign+lo plane, 1M golden config.
#
# Offline (byte-matched, 256 B/doc, shortlist recall of exact top-100) said this is a strictly better
# coarse filter, with the gain concentrated at SMALL N:
#     N=            250      500      750     1000
#     Gray sign+lo  0.8886   0.9696   0.9879   0.9943
#     sign+rotation 0.9291   0.9870   0.9962   0.9988
#
# So the e2e lever is NOT "higher recall at the same BRUTE_N" -- it is the SAME recall at a SMALLER
# BRUTE_N, because rerank cost is linear in the shortlist and rerank is the dominant term. Hence each
# arm is swept over BRUTE_N: the question is which curve reaches ~0.95 recall at the lowest latency.
#
# Both arms are pure symmetric XOR+popcount at identical bytes/doc and identical kernel shape, so any
# latency difference at equal BRUTE_N is noise, and any recall difference is the code itself.
#
# KNN_CLEAR_CACHE=1 on the FIRST run of each arm: the plane content is write-time and is NOT in the index
# cache key, so reusing an index across arms would silently compare a code against itself. Subsequent
# BRUTE_N points reuse that arm's index (BRUTE_N is search-time only).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
S="$OUT/run_1m_rotplane_e2e_${STAMP}_SUMMARY.txt"

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8          # osq8 rerank: the tier that RANKS, unchanged
export IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BEAM_FACTOR=2
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}" KNN_SKIP_MODEL_CHECK=1
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1   # two planes; content set per arm below
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

echo "=== rotation plane vs Gray lo plane, 1M golden (np40, nq=$KNN_NQUERY) $(date) ===" | tee "$S"
echo "both arms: 256 B/doc coarse, symmetric XOR+popcount, osq8 rerank. Sweeping BRUTE_N." | tee -a "$S"
echo | tee -a "$S"

arm() {
  local name="$1" rot="$2"; shift 2
  local first=1
  echo "--- arm=$name (sketchRotPlane=$rot) ---" | tee -a "$S"
  for BN in 500 350 250 175; do
    local log="$OUT/run_1m_rotplane_e2e_${STAMP}_${name}_bn${BN}.log"
    local cc=0; [ $first -eq 1 ] && cc=1; first=0
    export LLOYD_SKETCH_ROT_PLANE="$rot"
    LLOYD_BRUTE_N=$BN KNN_CLEAR_CACHE=$cc ./run_knn_bench.sh 1 > "$log" 2>&1
    local flags; flags="$(grep -oE 'Dlloyd\.sketchRotPlane=[a-z]+' "$log" | sort -u | tr '\n' ' ')"
    if [ "$rot" = "1" ] && [ -z "$flags" ]; then
      echo "  !! ABORT: arm=$name bn=$BN did NOT pass -Dlloyd.sketchRotPlane -- row is VOID" | tee -a "$S"
      return 1
    fi
    local x; x="$(grep '^SUMMARY' "$log" | tail -1 | sed 's/SUMMARY: //')"
    if [ -z "$x" ]; then
      printf '  bn=%-5s FAILED (see log)\n' "$BN" | tee -a "$S"
      grep -iE "exception|error|Caused by" "$log" | head -3 | sed 's/^/     /' | tee -a "$S"
    else
      printf '  bn=%-5s recall=%s ceil=%s lat=%s ms  flags=[%s]\n' "$BN" \
        "$(printf '%s' "$x" | awk -F'\t' '{print $1}')" \
        "$(printf '%s' "$x" | awk -F'\t' '{print $4}')" \
        "$(printf '%s' "$x" | awk -F'\t' '{print $2}')" "$flags" | tee -a "$S"
    fi
  done
  echo | tee -a "$S"
}

arm gray 0
arm rotation 1

echo "GATE: compare LATENCY AT ~0.95 RECALL between the two curves. The rotation plane should reach" | tee -a "$S"
echo "0.95 at a SMALLER bn, hence lower latency; equal-bn rows should differ in recall but not latency." | tee -a "$S"
echo "=== summary: $S ===" | tee -a "$S"
