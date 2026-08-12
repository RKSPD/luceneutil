#!/usr/bin/env bash
# How much recall does the 4-bit fine tier actually buy? Two SEARCH-time arms on ONE cached residual4 index
# (Hamming coarse), golden low-latency config (topK=100, nquery=1000, no JFR):
#   FULL6BIT   : normal 2-bit bucket + 4-bit nibble rerank (6 effective bits).
#   BUCKET2BIT : rerank with the 2-bit bucket term ONLY (nibble dropped, record skipped).
# If BUCKET2BIT still clears ~0.95, recall is coarse-bound and the nibble (22% of query CPU) is
# over-provisioned -- a cheaper fine tier reclaims it. If it falls to ~0.93, the nibble earns its keep.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=residual4 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BRUTE_N=1000 KNN_TOPK=100 KNN_NQUERY=1000
export KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_SKETCH_LO_CLIP=3.5
export LLOYD_COARSE_BUCKET_DOT=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
unset KNN_JFR

REPS="${REPS:-3}"
run_arm() {
  local name="$1"; shift
  local log="$OUT/run_1m_finetier_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  KNN_CLEAR_CACHE=0 env "$@" ./run_knn_bench.sh "$REPS" > "$log" 2>&1
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' \
    | awk -v nm="$name" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "    %-12s recall=%s  MIN lat=%s ms\n",nm,r,min}'
}
run_arm FULL6BIT   LLOYD_RERANK_BUCKET_ONLY=0
run_arm BUCKET2BIT LLOYD_RERANK_BUCKET_ONLY=1
echo "=== done $(date) ==="
