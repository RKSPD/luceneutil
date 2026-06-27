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
LUCENE_DIR=/local/home/rikhil/lucene
LUCENEUTIL_DIR=/home/rikhil/luceneutil
JAVA_HOME=/local/home/rikhil/brazil-pkg-cache/packages/JDK25/JDK25-1.0.6380.0/AL2_aarch64/DEV.STD.PTHREAD/build/jdk-25
PY311_LIB=/usr/patching-agent/python3.11/lib
VENV_PY="$LUCENEUTIL_DIR/.venv/bin/python"
RUNS="${1:-1}"

# jdk 25 is required to build lucene main (versions.toml minJava=25)
export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

# the python3.11 venv links libpython3.11.so.1.0 from this dir; export it so EVERY invocation of
# "$VENV_PY" (not just the benchmark run) can load it -- otherwise the bare `--version` check below
# dies with "error while loading shared libraries: libpython3.11.so.1.0".
export LD_LIBRARY_PATH="$PY311_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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

# --- 3. clear the knn-reuse cache --------------------------------------------
echo "=== [3/4] clearing knn-reuse cache ==="
rm -rf "$LUCENEUTIL_DIR/knn-reuse"

# --- 4. run the benchmark with python 3.11 ------------------------------------
# NOTE: do NOT use './gradlew runKnnPerfTest' -- it hardcodes the system python3 (3.7, too old).
echo "=== [4/4] running knnPerfTest.py (runs=$RUNS) with python 3.11 ==="
"$VENV_PY" --version
"$VENV_PY" -u src/python/knnPerfTest.py --runs "$RUNS"
