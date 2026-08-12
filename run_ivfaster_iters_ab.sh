#!/bin/bash
# A/B: does closing the Lloyd convergence gap (iters 3 -> 12) buy recall? Same search params both arms,
# full rebuild each (iters is write-time). Reports recall/latency per arm.
set -uo pipefail
cd "$(dirname "$0")"
if ps -eo args 2>/dev/null | grep -qE '^[^ ]*java .*knn\.KnnGraphTester'; then
  echo "ERROR: a KnnGraphTester run is already in flight." >&2; exit 1
fi
export LUCENE_DIR=${LUCENE_DIR:-/local/home/rikhil/vectordb/lucene}
export KNN_INDEX_TYPE=ivfaster KNN_NDOC=1000000 KNN_TOPK=100 KNN_NLIST=2000
export KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.2 KNN_HEAP=48g KNN_CLEAR_CACHE=1
export IVFASTER_FINE_TIER=int8
export KNN_NPROBE="${KNN_NPROBE:-16,32,48}" IVFASTER_BRUTE_N="${IVFASTER_BRUTE_N:-600}"
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
for ITERS in 3 12; do
  LOG=$OUT/ivfaster_iters${ITERS}_${STAMP}.log
  export IVFASTER_LLOYD_ITERS=$ITERS
  echo "=== iters=$ITERS $(date) ===" | tee "$LOG"
  ./run_knn_bench.sh 1 >> "$LOG" 2>&1
  echo "  exit=$?  reindex=$(grep -oE 'reindex[^ ]* [0-9.]+ s|indexed .* in [0-9.]+' "$LOG" | tail -1)"
  grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' -v it=$ITERS '{printf "  iters=%s recall=%s lat=%s ms\n",it,$1,$2}'
done
echo "=== done $(date) ==="
