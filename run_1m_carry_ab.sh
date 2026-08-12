#!/usr/bin/env bash
# 3-arm A/B for the PLANE CARRY on the residual4 golden 1M index. All three are SEARCH-time flags on ONE
# cached index (no rebuild between arms), so this isolates the two effects cleanly:
#
#   A HAMMING      : symmetric 2-bit Gray Hamming coarse ranker + full rerank (re-reads both planes). Today's
#                    default coarse path.
#   B BUCKETDOT    : asymmetric bucket-dot coarse ranker + full rerank (no carry). Isolates the RANKER cost
#                    and recall vs Hamming at the same BRUTE_N.
#   C BUCKETDOT+CARRY: bucket-dot coarse + carry -- rerank reads ONLY the 520 B nibble record, taking
#                    Sum(q*bucket) from the coarse heap. Isolates the CARRY win over B (fewer bytes on the
#                    DRAM-bound rerank pole).
#
# The real question is whether B/C's better ranking (0.965 vs 0.911 @N=1000) + C's byte cut beats A's cheap
# coarse scan END TO END. Recall for B and C should match to ~3 decimals (same ranker; carry only changes
# WHICH bytes rerank reads, not the score). A may differ (different ranker).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

# --- residual4 golden config (matches the 0.956 @ np40 operating point) ---
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=residual4 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_TOPK="${KNN_TOPK:-100}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_SKETCH_LO_CLIP=3.5 LLOYD_BRUTE_N="${LLOYD_BRUTE_N:-1000}"
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
unset KNN_JFR  # low-latency measurement: no profiler overhead

REPS="${REPS:-3}"
run_arm() {
  local name="$1"; shift
  local log="$OUT/run_1m_carry_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  # First arm builds the index (KNN_CLEAR_CACHE unset => reuse if present); subsequent arms reuse it.
  KNN_CLEAR_CACHE=0 env "$@" ./run_knn_bench.sh "$REPS" > "$log" 2>&1
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' \
    | awk -v nm="$name" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "    %-18s recall=%s  MIN lat=%s ms\n",nm,r,min}'
}

# A: Hamming (bucket-dot OFF). B: bucket-dot ON, carry OFF. C: bucket-dot ON, carry ON (default).
run_arm HAMMING           LLOYD_COARSE_BUCKET_DOT=0
run_arm BUCKETDOT         LLOYD_COARSE_BUCKET_DOT=1 LLOYD_NO_CARRY_BUCKET_DOT=1
run_arm BUCKETDOT_CARRY   LLOYD_COARSE_BUCKET_DOT=1 LLOYD_NO_CARRY_BUCKET_DOT=0
echo "=== done $(date) ==="
