#!/usr/bin/env bash
#
# COLD-CACHE variant of run_knn_bench.sh -- models the production case we care about:
# a KNN index far larger than the RAM available to serve it, so timed queries must
# fault posting lists in from storage instead of hitting a fully-cached mmap.
#
# Why a memory cap (not just `echo 3 > drop_caches`):
#   KnnGraphTester reindexes IN-PROCESS and then runs a mandatory warmup pass over every
#   query before the timer starts (KnnGraphTester.java:1407-1424). So any cache we drop
#   before launch is immediately repopulated by the index write + warmup, and the timed
#   loop runs fully warm. The ONLY faithful way to keep the timed search cold is to cap the
#   memory available to the JVM below the index size, so the kernel must evict index pages
#   under pressure -- exactly what happens on a prod node whose index dwarfs RAM.
#
# How: the whole python driver runs inside a transient systemd cgroup scope with
# MemoryLimit=$MEM_CAP. That cap counts JVM heap RSS *plus* file-backed page cache together,
# so set it ABOVE heap+overhead but BELOW heap+index. With the default -Xmx2g heap, a 4G cap
# leaves only ~2G for caching a multi-hundred-MB-to-multi-GB index -> page cache thrashes and
# posting reads hit the block device on the timed path.
#
# usage: ./run_knn_bench_coldcache.sh [runs] [mem_cap]
#   runs     defaults to 1
#   mem_cap  systemd memory limit for the benchmark JVM, e.g. 4G, 3500M (default 4G)
#
# To make the index genuinely large (recommended for a realistic prod measurement), raise
# "ndoc" in src/python/knnPerfTest.py PARAMS (the source vec file holds 1,000,000 docs) and/or
# lower mem_cap. Aim for index_size >> (mem_cap - heap).

set -euo pipefail

# --- config -------------------------------------------------------------------
LUCENE_DIR=/local/home/rikhil/lucene
LUCENEUTIL_DIR=/home/rikhil/luceneutil
JAVA_HOME=/local/home/rikhil/brazil-pkg-cache/packages/JDK25/JDK25-1.0.6380.0/AL2_aarch64/DEV.STD.PTHREAD/build/jdk-25
PY311_LIB=/usr/patching-agent/python3.11/lib
VENV_PY="$LUCENEUTIL_DIR/.venv/bin/python"
RUNS="${1:-1}"
MEM_CAP="${2:-4G}"

# jdk 25 is required to build lucene main (versions.toml minJava=25)
export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

# --- 1. build the lucene jars (sandbox codec lands in the SNAPSHOT jars) -------
echo "=== [1/5] building lucene jars in $LUCENE_DIR ==="
cd "$LUCENE_DIR"
./gradlew jar > /tmp/lucene-build.log 2>&1 \
  && echo "lucene jar build OK" \
  || { echo "lucene jar build FAILED -- last 30 lines:"; tail -n 30 /tmp/lucene-build.log; exit 1; }

# --- 2. compile the knn harness against the fresh jars ------------------------
echo "=== [2/5] compiling knn harness in $LUCENEUTIL_DIR ==="
cd "$LUCENEUTIL_DIR"
./gradlew compileKnn > /tmp/knn-compile.log 2>&1 \
  && echo "compileKnn OK" \
  || { echo "compileKnn FAILED -- last 30 lines:"; tail -n 30 /tmp/knn-compile.log; exit 1; }

# empty dir satisfies a knnPerfTest.py preflight classpath check (real classes live in ./build)
mkdir -p "$LUCENEUTIL_DIR/src/main/build/classes/java/main"

# --- 3. clear the knn-reuse cache --------------------------------------------
echo "=== [3/5] clearing knn-reuse cache ==="
rm -rf "$LUCENEUTIL_DIR/knn-reuse"

# --- 4. drop the OS page cache for a clean cold baseline ----------------------
# (sudo can't run a shell here -- sudo -l forbids /bin/sh etc -- so tee, not redirect.)
echo "=== [4/5] dropping OS page cache (sync + drop_caches=3) ==="
sync
if echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null 2>&1; then
  echo "page cache dropped"
else
  echo "WARNING: could not drop page cache (need passwordless sudo for tee /proc/sys/vm/drop_caches);"
  echo "         the memory cap below is what actually forces cold reads, so continuing anyway."
fi

# --- 5. run the benchmark inside a memory-capped cgroup -----------------------
# The cap (heap + page cache) forces index eviction during the timed search -> cold storage reads.
echo "=== [5/5] running knnPerfTest.py (runs=$RUNS) under MemoryLimit=$MEM_CAP ==="
"$VENV_PY" --version

# sudo strips the environment (secure_path is set), so inject what python/JVM need explicitly
# via an `env` wrapper INSIDE the scope. --uid=$USER keeps the JVM running as us, not root.
sudo -n systemd-run --scope --uid="$USER" -p MemoryLimit="$MEM_CAP" \
  /usr/bin/env \
    LD_LIBRARY_PATH="$PY311_LIB" \
    JAVA_HOME="$JAVA_HOME" \
    PATH="$JAVA_HOME/bin:$PATH" \
    HOME="$HOME" \
  "$VENV_PY" -u src/python/knnPerfTest.py --runs "$RUNS"

echo
echo "=== interpreting the result row ==="
echo "Compare latency(ms) vs netCPU: if latency >> netCPU now (it was ~equal in the warm run),"
echo "the gap is I/O wait -- that delta is what a faster storage tier could remove."
echo "index_size(MB) in the row vs the cap headroom (cap - 2G heap) tells you how much of the"
echo "index could NOT be cached; the more it spills, the more storage speed matters."
