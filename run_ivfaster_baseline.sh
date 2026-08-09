#!/bin/bash
# ivfaster's first end-to-end measurement on the golden config.
#
# WHAT THIS ANSWERS. Every performance claim about ivfaster so far is a microbenchmark or a
# simulation. Three of them need converting into measurements, and they are separable:
#
#   1. BUILD TIME. Brute-scan routing is predicted CHEAPER than the beam it replaces, because it
#      deletes the repeated float->int8 re-quantization of the whole centroid matrix that profiling
#      put at 42.9% of build CPU. Predicted ~0.7 s/iteration at nlist=2000. Read the reindex time.
#   2. SEARCH AT PARITY. The scoring path is the same as the predecessor's best arm -- same coarse
#      kernel, same int8 rerank, same counting-select and dedup -- so the expectation at HIGH nprobe
#      is parity, near 0.951 @ ~1.4 ms. A large miss means something regressed in the port.
#   3. THE LOW-NPROBE ADVANTAGE, which is the real thesis. Every doc is in its TRUE nearest cell
#      (asserted by test, against brute force), where a beam at ef=16 misplaces ~41% of them. A
#      simulation put the cell-recall ceiling at 0.976 with nprobe=1 versus needing nprobe=4 for the
#      beam. If that transfers, ivfaster reaches a target recall at far lower nprobe -- which is
#      fewer docs through the coarse scan, and the coarse scan is ~half of query CPU.
#
# So the sweep starts at nprobe=1, not at the predecessor's 40-70. Sweeping only the high end would
# measure the regime where both arms sit at ceiling and the advantage is arithmetically nil.
#
# nprobe and bruteN are SEARCH-time: the reader reads both per query, so this whole grid runs against
# ONE cached index. Everything write-time (nlist, spill, soar, iters, fine tier) is in the index key.

set -euo pipefail
cd "$(dirname "$0")"

# A LIVE RUN AND A REBUILD CANNOT COEXIST. The JVM memory-maps the Lucene jars and resolves classes
# LAZILY, so rebuilding them mid-run replaces files it is still holding: the next class it had not yet
# touched fails with NoClassDefFoundError. That happened here as
# "NoClassDefFoundError: org/apache/lucene/index/ReaderUtil" at the recall step -- a Lucene core class
# that was present in the jar the whole time, and loadable on the same classpath afterwards.
#
# The misleading part is that it looks catastrophic and codec-shaped when it is neither: search had
# already finished. But the numbers were void regardless, because the codec binary changed three times
# mid-flight.
# Matches only a JAVA process running the tester, not any shell command that happens to contain the
# string. `pgrep -f KnnGraphTester` matched this script's own launcher and every diagnostic command
# mentioning it, so the guard blocked legitimate runs and made absent runs look live.
if ps -eo args 2>/dev/null | grep -qE '^[^ ]*java .*knn\.KnnGraphTester'; then
  echo "ERROR: a KnnGraphTester run is already in flight." >&2
  echo "  Rebuilding jars under a live JVM corrupts it -- wait for it, or kill it first." >&2
  exit 1
fi

export LUCENE_DIR=${LUCENE_DIR:-/local/home/rikhil/vectordb/lucene}

# Golden config: 1M Cohere-v3 wikipedia-en, 1024-d, DOT_PRODUCT, topK=100, one search thread,
# force-merged to a single segment.
export KNN_INDEX_TYPE=ivfaster
export KNN_NDOC=${KNN_NDOC:-1000000}
export KNN_TOPK=${KNN_TOPK:-100}
export KNN_NLIST=${KNN_NLIST:-2000}
export KNN_SPILL_BITS=${KNN_SPILL_BITS:-2}
export IVF_SPILL_MARGIN=${IVF_SPILL_MARGIN:-1.2}

# Merge reconstructs float vectors, so this needs real heap. The default 2g OOMs in mergeOneField and
# reads as "it failed" rather than as an out-of-memory.
export KNN_HEAP=${KNN_HEAP:-48g}

# The index cache key does not capture jar changes, so a code change with a stale cache silently
# measures the previous build.
export KNN_CLEAR_CACHE=${KNN_CLEAR_CACHE:-1}

export IVFASTER_FINE_TIER=${IVFASTER_FINE_TIER:-int8}
export IVFASTER_LLOYD_ITERS=${IVFASTER_LLOYD_ITERS:-3}

# The nprobe grid, deliberately weighted to the LOW end -- see (3) above. KNN_NPROBE takes a
# comma-separated list and becomes a PARAMS axis, so the whole curve is ONE invocation against one
# cached index rather than a shell loop that reindexes if anything in the key drifts.
export KNN_NPROBE=${KNN_NPROBE:-1,2,4,8,16,32,64}
export IVFASTER_BRUTE_N=${IVFASTER_BRUTE_N:-400}

echo "=== ivfaster baseline ==="
echo "  nlist=$KNN_NLIST spill=$KNN_SPILL_BITS fine=$IVFASTER_FINE_TIER bruteN=$IVFASTER_BRUTE_N"
echo "  nprobe grid: $KNN_NPROBE"
echo

./run_knn_bench.sh "$@"
