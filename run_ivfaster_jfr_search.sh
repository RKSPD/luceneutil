#!/usr/bin/env bash
#
# JFR profile of the ivfaster SEARCH path at the golden operating point.
#
# WHY SEARCH IS THE RIGHT TARGET NOW. The coarse Hamming kernel is SHARED between search and assignment
# -- routing, reap, spill selection and the graph descent all score through the same
# PanamaHammingKernel.bulkDistances2 the cell scan uses. So a kernel-level win pays twice: once per query
# and once per document per Lloyd pass. That is the opposite of the index-side work just landed, which was
# all waste-removal specific to the build.
#
# The last search profile put the shape at: 39% coarse kernel + vector plumbing, 21.8% mmap loads,
# 3.6% fine tier -- i.e. the fine tier is nearly free and the coarse scan is the whole game. This run
# re-measures that AFTER the plane-packing change, on a warm cache, with enough queries to be real.
#
# WHAT IT SHOULD ANSWER:
#   1. Is the coarse scan still ~39%, and how does it split between the ARITHMETIC (reduceLanes, XOR,
#      popcount) and the LOADS (loadFromMemorySegment)? Arithmetic-bound and load-bound want opposite
#      fixes, and the previous profile had both large.
#   2. How much is bounds/liveness checking (checkValidStateRaw, sessionImpl, getIntUnaligned)? The
#      whole-plane segment slice is one of the two untested leads on the 3.08ms-vs-1.21ms gap, and this
#      is where it would show.
#   3. Does the counting-sort select or the dedup show up at all? Both were poles in the predecessor and
#      should be small here.
#
# NOT cache-cleared: this is the WARM path, which is the operating point the objective is stated at
# (latency at 0.95 recall, warm). A cold run measures the disk, not the kernel.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

if ps -eo args 2>/dev/null | grep -qE '^[^ ]*java .*knn\.KnnGraphTester'; then
  echo "ERROR: a KnnGraphTester run is already in flight." >&2
  echo "  Rebuilding jars under a live JVM corrupts it -- wait for it, or kill it first." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_ivfaster_jfr_search_${STAMP}.log"

export LUCENE_DIR=${LUCENE_DIR:-/local/home/rikhil/vectordb/lucene}

export KNN_INDEX_TYPE=ivfaster
export KNN_NDOC=${KNN_NDOC:-1000000}
export KNN_TOPK=${KNN_TOPK:-100}
export KNN_NLIST=${KNN_NLIST:-2000}
export KNN_SPILL_BITS=${KNN_SPILL_BITS:-2}
export IVFASTER_FINE_TIER=${IVFASTER_FINE_TIER:-int8}
export IVFASTER_LLOYD_ITERS=${IVFASTER_LLOYD_ITERS:-3}
export KNN_HEAP=${KNN_HEAP:-48g}

# The golden operating point, one nprobe so the profile is not a blend of regimes.
export KNN_NPROBE=${KNN_NPROBE:-40}
export IVFASTER_BRUTE_N=${IVFASTER_BRUTE_N:-400}
# Enough queries that the search phase dominates the recording and the recall figure is real. The
# index-JFR runs used nquery=50 to SHRINK search -- quoting a recall off those was a mistake worth not
# repeating.
export KNN_NQUERY=${KNN_NQUERY:-1000}

# REUSE the cached index: nprobe and bruteN are search-time, so this measures the kernel, not a rebuild.
export KNN_CLEAR_CACHE=0
export KNN_JFR=1

echo "=== ivfaster SEARCH JFR (warm, np=$KNN_NPROBE bn=$IVFASTER_BRUTE_N) $(date) ===" | tee "$LOG"
./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "    exit=$?" | tee -a "$LOG"

# Preserve it: every profiled JVM writes knn-perf-test-<seq>.jfr and the sequence restarts per JVM, so
# the next run overwrites this. That is how the first index profile was lost.
for f in "$OUT"/logs/knn-perf-test-*.jfr; do
  [ -e "$f" ] || continue
  cp -p "$f" "$OUT/logs/ivfaster-search-${STAMP}-$(basename "$f")"
done

grep -E "recall = |^ *[0-9]+\.[0-9]+ +[0-9]" "$LOG" | tail -4
echo "=== done $(date). log: $LOG ; jfr: $OUT/logs/ivfaster-search-${STAMP}-*.jfr ==="
