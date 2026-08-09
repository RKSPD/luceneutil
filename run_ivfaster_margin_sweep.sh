#!/usr/bin/env bash
#
# ADAPTIVE-NPROBE SWEEP: probe wide, prune on quality.
#
# THE HYPOTHESIS. After the docVec fix, the search profile is dominated by memory access -- mmap loads
# 20.6% plus ~11% of bounds/liveness machinery around them -- against 11.4% of actual vector arithmetic.
# All of that scales with DOCUMENTS SCANNED, which at nprobe=40 is ~62k per query. A fixed nprobe spends
# that budget identically on every query, but queries are not identical: one near a centroid has its
# neighbours concentrated in a few cells, and its tail cells contribute bytes and no recall.
#
# The margin drops a selected cell whose exact distance is worse than d1*margin. It is free to evaluate --
# rerankCells already computes those distances to pick the top `probe` -- so every dropped cell is ~1550
# documents that are never loaded or scored, at the cost of one compare.
#
# WHY WIDE-AND-PRUNE RATHER THAN A SMALLER NPROBE. The predecessor measured exactly this: its best
# >=0.95 point was nprobe=70/margin=0.75 at 0.951 @ 0.897ms, against 0.955 @ 1.129ms at a fixed nprobe.
# Raising nprobe keeps the cells that matter available; the margin removes the ones that do not. Choosing
# a small nprobe outright cannot do the second part.
#
# The grid therefore pairs a WIDER nprobe with a margin, and includes margin=1.0 (off) at each nprobe as
# the control -- so the margin's effect is separable from nprobe's, rather than confounded with it.
#
# Both axes are SEARCH-TIME, so this whole grid runs against ONE cached index. Nothing here reindexes.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

if ps -eo args 2>/dev/null | grep -qE '^[^ ]*java .*knn\.KnnGraphTester'; then
  echo "ERROR: a KnnGraphTester run is already in flight -- rebuilding jars under it corrupts it." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_ivfaster_margin_${STAMP}.log"

export LUCENE_DIR=${LUCENE_DIR:-/local/home/rikhil/vectordb/lucene}
export KNN_INDEX_TYPE=ivfaster
export KNN_NDOC=1000000
export KNN_TOPK=100
export KNN_NLIST=2000
export KNN_SPILL_BITS=2
export IVFASTER_FINE_TIER=int8
export IVFASTER_LLOYD_ITERS=3
export IVFASTER_BRUTE_N=400
export KNN_HEAP=48g
export KNN_NQUERY=1000
# Reuse the cached index: every axis swept here is search-time.
export KNN_CLEAR_CACHE=0
# No profiler: this is a latency measurement, and JFR overhead would blur small differences.
unset KNN_JFR || true

echo "=== ivfaster adaptive-nprobe sweep $(date) ===" | tee "$LOG"
printf '%-8s %-8s %-9s %s\n' nprobe margin recall "latency(ms)" | tee -a "$LOG"

for np in 40 70 100; do
  for m in 1.0 0.85 0.75 0.70; do
    export KNN_NPROBE=$np
    export IVFASTER_NPROBE_MARGIN=$m
    R="$OUT/.margin_${STAMP}_${np}_${m}.log"
    ./run_knn_bench.sh 1 > "$R" 2>&1
    rec=$(grep -oE "recall = [0-9.]+" "$R" | tail -1 | awk '{print $3}')
    lat=$(grep -A3 "latency(ms)" "$R" | tail -1 | awk '{print $1}')
    printf '%-8s %-8s %-9s %s\n' "$np" "$m" "${rec:-?}" "${lat:-?}" | tee -a "$LOG"
  done
done

echo "=== done $(date). log: $LOG ===" | tee -a "$LOG"
