#!/bin/bash
# A/B iters 3 vs 10, SPILL=0, to isolate the convergence gain without spill masking boundary docs.
set -uo pipefail
cd "$(dirname "$0")"
if ps -eo args 2>/dev/null | grep -qE '^[^ ]*java .*knn\.KnnGraphTester'; then
  echo "ERROR: a KnnGraphTester run is already in flight." >&2; exit 1
fi
export LUCENE_DIR=${LUCENE_DIR:-/local/home/rikhil/vectordb/lucene}
export KNN_INDEX_TYPE=ivfaster KNN_NDOC=1000000 KNN_TOPK=100 KNN_NLIST=2000
export KNN_SPILL_BITS=0 IVF_SPILL_MARGIN=1.0 KNN_HEAP=48g KNN_CLEAR_CACHE=1
export IVFASTER_FINE_TIER=int8
export KNN_NPROBE="16,32,48" IVFASTER_BRUTE_N=600
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
for ITERS in 3 10; do
  LOG=$OUT/ivfaster_nospill_iters${ITERS}_${STAMP}.log
  export IVFASTER_LLOYD_ITERS=$ITERS
  echo "=== spill=0 iters=$ITERS $(date) ===" | tee "$LOG"
  ./run_knn_bench.sh 1 >> "$LOG" 2>&1
  echo "  exit=$?"
  grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' -v it=$ITERS '{printf "  iters=%s recall=%s lat=%s ms\n",it,$1,$2}'
done
echo "=== done $(date) ==="
