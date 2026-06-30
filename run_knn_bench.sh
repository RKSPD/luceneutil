#!/usr/bin/env bash
#
# run the luceneutil knn benchmark end-to-end:
#   1. build the lucene checkout jars (the codec under test)  -- gradlew
#   2. compile the knn harness (KnnGraphTester) against those jars -- gradlew
#   3. clear the knn-reuse cache (stale exact-nn / index caches silently give 0.0 recall)
#   4. run knnPerfTest.py with python 3.11
#
# usage: ./run_knn_bench.sh [runs]   (runs defaults to 1)

set -euo pipefail

# --- config -------------------------------------------------------------------
LUCENE_DIR=/Users/rikhil/Desktop/lucene
LUCENEUTIL_DIR=/Users/rikhil/Desktop/luceneutil
JAVA_HOME=/Library/Java/JavaVirtualMachines/amazon-corretto-26.jdk/Contents/Home
PYTHON=python3
RUNS="${1:-1}"
# JVM heap cap for the search/index JVM (-Xms/-Xmx), read by constants.py. Override per-run:
#   KNN_HEAP=1g ./run_knn_bench.sh   (cap below index size to force the disk/off-heap path -- findings §20)
KNN_HEAP="${KNN_HEAP:-24g}"

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"
export KNN_HEAP

# --- 1. build the lucene jars (sandbox codec lands in the SNAPSHOT jars) -------
echo "=== [1/4] building lucene jars in $LUCENE_DIR ==="
cd "$LUCENE_DIR"
./gradlew jar > /tmp/lucene-build.log 2>&1 \
  && echo "lucene jar build OK" \
  || { echo "lucene jar build FAILED -- last 30 lines:"; tail -n 30 /tmp/lucene-build.log; exit 1; }

# --- 2. compile the knn harness against the fresh jars ------------------------
echo "=== [2/4] compiling knn harness in $LUCENEUTIL_DIR ==="
cd "$LUCENEUTIL_DIR"
./gradlew compileKnn > /tmp/knn-compile.log 2>&1 \
  && echo "compileKnn OK" \
  || { echo "compileKnn FAILED -- last 30 lines:"; tail -n 30 /tmp/knn-compile.log; exit 1; }

# empty dir satisfies a knnPerfTest.py preflight classpath check (real classes live in ./build)
mkdir -p "$LUCENEUTIL_DIR/src/main/build/classes/java/main"

# --- 3. knn-reuse cache -------------------------------------------------------
# Indexes are cached under knn-reuse/indices keyed by ALL index-affecting params (hashBits, nprobe-is-
# NOT-in-key since it is search-time, spillBits, pca, itq, quantize, ...). Search-only params (nprobe,
# searchThreads, overquery) are not in the key, so a matching index is correctly reused across those --
# letting you iterate on search params WITHOUT rebuilding the (20M) index each run.
#
# DEFAULT: keep the cache (fast search-param iteration).
# CLEAR IT (KNN_CLEAR_CACHE=1) whenever you changed the CODEC / on-disk format -- the cache key does not
# capture jar changes, so a stale index would silently give wrong/0.0 recall. Rule of thumb: cleared
# any LSHVectors*.java or rebuilt the format => set KNN_CLEAR_CACHE=1 for that run.
if [[ "${KNN_CLEAR_CACHE:-1}" == "1" ]]; then
  echo "=== [3/4] clearing knn-reuse cache (KNN_CLEAR_CACHE=1) ==="
  rm -rf "$LUCENEUTIL_DIR/knn-reuse"
else
  echo "=== [3/4] keeping knn-reuse cache (set KNN_CLEAR_CACHE=1 to clear after codec/format changes) ==="
fi

# --- 4. run the benchmark with python 3.11 ------------------------------------
# NOTE: do NOT use './gradlew runKnnPerfTest' -- it hardcodes the system python3 (3.7, too old).
echo "=== [4/4] running knnPerfTest.py (runs=$RUNS) ==="
"$PYTHON" --version
"$PYTHON" -u src/python/knnPerfTest.py --runs "$RUNS"
