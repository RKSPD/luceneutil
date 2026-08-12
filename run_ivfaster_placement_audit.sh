#!/bin/bash
# Assignment-time coarse-retention audit: how many primaries the coarse shortlist misplaces versus an
# exact all-nlist scan, on the golden config. Build-time diagnostic; nprobe is irrelevant so it is fixed.
set -uo pipefail
cd "$(dirname "$0")"
if ps -eo args 2>/dev/null | grep -qE '^[^ ]*java .*knn\.KnnGraphTester'; then
  echo "ERROR: a KnnGraphTester run is already in flight." >&2; exit 1
fi
export LUCENE_DIR=${LUCENE_DIR:-/local/home/rikhil/vectordb/lucene}
export KNN_INDEX_TYPE=ivfaster
export KNN_NDOC=${KNN_NDOC:-1000000} KNN_TOPK=100 KNN_NLIST=${KNN_NLIST:-2000}
export KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.2 KNN_HEAP=48g KNN_CLEAR_CACHE=1
export IVFASTER_FINE_TIER=int8 IVFASTER_LLOYD_ITERS=${IVFASTER_LLOYD_ITERS:-3}
export IVFASTER_EXACT_PLACEMENT_AUDIT=1
export KNN_NPROBE=32 IVFASTER_BRUTE_N=400
STAMP="$(date +%Y%m%d_%H%M%S)"; LOG=/local/home/rikhil/vectordb/ivfaster_placement_audit_${STAMP}.log
echo "=== placement audit nlist=$KNN_NLIST iters=$IVFASTER_LLOYD_ITERS $(date) ===" | tee "$LOG"
./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "exit=$?"
grep "\[ivfaster\]" "$LOG" | tail -5
echo "=== log: $LOG ==="
