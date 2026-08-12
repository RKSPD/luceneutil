#!/usr/bin/env bash
# THE GOLDEN QUANTITY: new coarse tier (rotation sign plane) + osq8 rerank, at the codec's actual objective --
# LATENCY AT 0.95 RECALL.
#
# The prior sweep established recall at fixed bn (rotation +0.017..+0.030 over Gray at identical 256 B/doc),
# but the objective is the other axis: Gray needs bn=500 to reach 0.945-0.949 (~1.15 ms), while rotation is
# already 0.940 at bn=350 and 0.962 at bn=500 -- so it should clear 0.95 somewhere near bn=400. This run
# brackets that crossing on BOTH arms so the comparison is "ms at equal recall", not "recall at equal bn".
#
# One index per arm (the plane content is write-time), then bn swept on the cached index -- so this is 2 builds
# and 8 search points. Engagement is asserted from the child JVM's own command line; a missing flag ABORTS.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
S="$OUT/run_1m_rotplane_at95_${STAMP}_SUMMARY.txt"

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8          # osq8 rerank on BOTH arms: only the coarse plane differs
export IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BEAM_FACTOR=2
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}" KNN_SKIP_MODEL_CHECK=1
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

echo "=== GOLDEN: rotation coarse + osq8, LATENCY AT 0.95 (1M, np40, nq=$KNN_NQUERY) $(date) ===" | tee "$S"
echo "both arms identical except the second coarse plane's CONTENT (same 256 B/doc, same XOR kernel)" | tee -a "$S"
echo | tee -a "$S"

arm() {
  local name="$1" rot="$2"; shift 2
  local first=1
  echo "--- arm=$name (sketchRotPlane=$rot) ---" | tee -a "$S"
  # Bracket 0.95 from below and above on both arms.
  for BN in 300 400 450 500 600; do
    local log="$OUT/run_1m_rotplane_at95_${STAMP}_${name}_bn${BN}.log"
    local cc=0; [ $first -eq 1 ] && cc=1; first=0
    export LLOYD_SKETCH_ROT_PLANE="$rot"
    LLOYD_BRUTE_N=$BN KNN_CLEAR_CACHE=$cc ./run_knn_bench.sh 1 > "$log" 2>&1
    local flags; flags="$(grep -oE 'Dlloyd\.sketchRotPlane=[a-z]+' "$log" | sort -u | tr '\n' ' ')"
    if [ "$rot" = "1" ] && [ -z "$flags" ]; then
      echo "  !! ABORT: bn=$BN did NOT pass -Dlloyd.sketchRotPlane -- row VOID" | tee -a "$S"
      return 1
    fi
    local x; x="$(grep '^SUMMARY' "$log" | tail -1 | sed 's/SUMMARY: //')"
    if [ -z "$x" ]; then
      printf '  bn=%-4s FAILED\n' "$BN" | tee -a "$S"
      grep -iE "exception|error|Caused by" "$log" | head -3 | sed 's/^/     /' | tee -a "$S"
    else
      printf '  bn=%-4s recall=%s ceil=%s lat=%s ms\n' "$BN" \
        "$(printf '%s' "$x" | awk -F'\t' '{print $1}')" \
        "$(printf '%s' "$x" | awk -F'\t' '{print $4}')" \
        "$(printf '%s' "$x" | awk -F'\t' '{print $2}')" | tee -a "$S"
    fi
  done
  echo | tee -a "$S"
}

arm gray 0
arm rotation 1

echo "READ: find the LOWEST bn in each arm with recall >= 0.950 and compare those two latencies." | tee -a "$S"
echo "That difference is the deployable speedup at the codec's stated objective." | tee -a "$S"
echo "=== summary: $S ===" | tee -a "$S"
