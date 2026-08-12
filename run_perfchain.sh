#!/usr/bin/env bash
# One arm of the ivfaster search-perf chain at the golden operating point.
#
# WHY A SEPARATE SCRIPT. Each optimization in this chain needs the SAME measurement repeated: warm
# cache, np=40, bn=400, 1M golden config, several runs so a 3-5% effect is separable from noise. The
# JFR script re-profiles (which costs wall clock and perturbs timing) and the baseline script sweeps
# nprobe. Neither shape is what an A/B wants.
#
# ARM is a label only; the code under test is whatever is in the working tree.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

if ps -eo args 2>/dev/null | grep -qE '^[^ ]*java .*knn\.KnnGraphTester'; then
  echo "ERROR: a KnnGraphTester run is already in flight -- rebuilding jars under it corrupts it." >&2
  exit 1
fi

ARM=${ARM:-arm}
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_perfchain_${ARM}_${STAMP}.log"

export LUCENE_DIR=${LUCENE_DIR:-/local/home/rikhil/vectordb/lucene}
export KNN_INDEX_TYPE=ivfaster
export KNN_NDOC=${KNN_NDOC:-1000000}
export KNN_TOPK=${KNN_TOPK:-100}
export KNN_NLIST=${KNN_NLIST:-2000}
export KNN_SPILL_BITS=${KNN_SPILL_BITS:-2}
export IVFASTER_FINE_TIER=${IVFASTER_FINE_TIER:-int8}
export IVFASTER_LLOYD_ITERS=${IVFASTER_LLOYD_ITERS:-3}
export KNN_HEAP=${KNN_HEAP:-48g}
export KNN_NPROBE=${KNN_NPROBE:-40}
export IVFASTER_BRUTE_N=${IVFASTER_BRUTE_N:-400}
export KNN_NQUERY=${KNN_NQUERY:-1000}
# Search-time params only, so the cached index is valid across every arm of this chain. Set
# KNN_CLEAR_CACHE=1 explicitly for an arm that changes the on-disk FORMAT.
export KNN_CLEAR_CACHE=${KNN_CLEAR_CACHE:-0}
export KNN_JFR=${KNN_JFR:-0}

RUNS=${RUNS:-5}
echo "=== perfchain arm=$ARM runs=$RUNS np=$KNN_NPROBE bn=$IVFASTER_BRUTE_N $(date) ===" | tee "$LOG"
git -C "$LUCENE_DIR" log --oneline -1 | tee -a "$LOG"
./run_knn_bench.sh "$RUNS" >> "$LOG" 2>&1
echo "    exit=$?" | tee -a "$LOG"
grep -E "recall|^ *[0-9]+\.[0-9]+ +[0-9]" "$LOG" | tail -12 | tee -a "$LOG"
echo "=== done $(date). log: $LOG ===" | tee -a "$LOG"
