#!/bin/bash
# Lloyd convergence curve on the golden config: run many iterations and trace per-iter centroid
# displacement, so we can see whether the default 3 iters converges or stops short.
set -uo pipefail
cd "$(dirname "$0")"
if ps -eo args 2>/dev/null | grep -qE '^[^ ]*java .*knn\.KnnGraphTester'; then
  echo "ERROR: a KnnGraphTester run is already in flight." >&2; exit 1
fi
export LUCENE_DIR=${LUCENE_DIR:-/local/home/rikhil/vectordb/lucene}
export KNN_INDEX_TYPE=ivfaster
export KNN_NDOC=${KNN_NDOC:-1000000} KNN_TOPK=100 KNN_NLIST=${KNN_NLIST:-2000}
export KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.2 KNN_HEAP=48g KNN_CLEAR_CACHE=1
export IVFASTER_FINE_TIER=int8 IVFASTER_LLOYD_ITERS=${IVFASTER_LLOYD_ITERS:-15}
export IVFASTER_CONVERGENCE_TRACE=1
export KNN_NPROBE=32 IVFASTER_BRUTE_N=400
STAMP="$(date +%Y%m%d_%H%M%S)"; LOG=/local/home/rikhil/vectordb/ivfaster_converge_${STAMP}.log
echo "=== convergence trace nlist=$KNN_NLIST iters=$IVFASTER_LLOYD_ITERS $(date) ===" | tee "$LOG"
./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "exit=$?"
echo "--- final-merge convergence curve (count=1000000) ---"
grep "\[ivfaster-converge\]" "$LOG" | tail -20
echo "--- recall/latency ---"
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms\n",$1,$2}'
echo "=== log: $LOG ==="
