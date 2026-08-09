#!/usr/bin/env bash
#
# JFR profile of the ivfaster INDEX path at 1M.
#
# WHY THIS EXISTS. Every ivfaster index-side claim so far is read off the source, not measured -- the one
# index-time recording this codec ever had was deleted before it was read. KNN_JFR=1 records the whole
# JVM with dumponexit, so with the cache cleared the JVM does a fresh build that dwarfs the search phase
# and the CPU-time profile is overwhelmingly the writer path. nquery is cut low to shrink the search
# contribution further.
#
# WHAT IT SHOULD ANSWER, in priority order:
#   1. Where does build time actually go? Predicted: routeAll + reap dominated by exactDistance (a 4 KB
#      float dot on ~8-27 candidates per document per pass), with recomputeCentroids second.
#   2. Does CentroidCodes.encodeAll matter at nlist=2000? It is the one build step with NO parallelism
#      (single-threaded mean over nlist x dim, then a serial encode loop). Predicted negligible at 2000
#      and a pole at 40000 -- if it is already visible here, that ordering is wrong.
#   3. Is selectCells really a second full route of the corpus? It calls route(keep=9) => shortlist 27,
#      against routing's shortlist 8, so it should appear as its OWN large share rather than as noise.
#   4. Did parallelizing the rotation take? HadamardRotation.rotate should now appear under
#      ivfaster-build-* threads, not only under "main". A profile still showing it single-threaded means
#      the Parallel.overRange did not engage (count/MIN_PER_THREAD gate).
#
# ENGAGEMENT, not just latency: a run whose kernel counters show scalar fallback is VOID, not slow. This
# codec has twice published a number from a path that never ran.
#
# NOTE: this CLEARS the cached ivfaster index -- a fresh build is the entire point.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

# A LIVE RUN AND A REBUILD CANNOT COEXIST: the JVM mmaps the Lucene jars and resolves classes lazily, so
# rebuilding them under a live run corrupts it (NoClassDefFoundError on a class that was present all
# along). Matches only a JAVA process running the tester -- `pgrep -f KnnGraphTester` also matched this
# script and every diagnostic command mentioning it, which blocked legitimate runs and made absent runs
# look live.
if ps -eo args 2>/dev/null | grep -qE '^[^ ]*java .*knn\.KnnGraphTester'; then
  echo "ERROR: a KnnGraphTester run is already in flight." >&2
  echo "  Rebuilding jars under a live JVM corrupts it -- wait for it, or kill it first." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_ivfaster_jfr_index_${STAMP}.log"

export LUCENE_DIR=${LUCENE_DIR:-/local/home/rikhil/vectordb/lucene}

# Golden config, write-time half: 1M Cohere-v3 wikipedia-en, 1024-d, DOT_PRODUCT.
export KNN_INDEX_TYPE=ivfaster
export KNN_NDOC=${KNN_NDOC:-1000000}
export KNN_TOPK=${KNN_TOPK:-100}
export KNN_NLIST=${KNN_NLIST:-2000}
export KNN_SPILL_BITS=${KNN_SPILL_BITS:-2}
export IVF_SPILL_MARGIN=${IVF_SPILL_MARGIN:-1.2}
export IVFASTER_FINE_TIER=${IVFASTER_FINE_TIER:-int8}
export IVFASTER_LLOYD_ITERS=${IVFASTER_LLOYD_ITERS:-3}

# Merge reconstructs float vectors; the default 2g OOMs in mergeOneField and reads as "it failed".
export KNN_HEAP=${KNN_HEAP:-48g}

# Single search point, few queries: the search phase is not what is being profiled here.
export KNN_NPROBE=${KNN_NPROBE:-40}
export IVFASTER_BRUTE_N=${IVFASTER_BRUTE_N:-400}
export KNN_NQUERY=${KNN_NQUERY:-50}

# A fresh build is the thing being profiled, and the cache key does not capture jar changes.
export KNN_CLEAR_CACHE=1
export KNN_JFR=1

echo "=== ivfaster INDEX JFR (fresh 1M build) $(date) ===" | tee "$LOG"
echo "  nlist=$KNN_NLIST spill=$KNN_SPILL_BITS iters=$IVFASTER_LLOYD_ITERS fine=$IVFASTER_FINE_TIER" | tee -a "$LOG"
./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "    exit=$?" | tee -a "$LOG"
grep -E "reindex takes|will now reindex|indexing took|force merge|simdEngaged|scalar" "$LOG" 2>/dev/null | head -20
echo "=== done $(date). log: $LOG ; jfr in $OUT/logs/ ==="
