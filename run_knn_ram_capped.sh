#!/usr/bin/env bash
#
# >RAM regime test for lloyd_ivf (LINUX ONLY).
#
# The warm 1M benchmark keeps the whole ~1.9 GB code table in the OS page cache, so it measures pure
# compute and CANNOT show any I/O effect (see open.md §1 "Measurement caveat"). This script forces the
# >RAM regime two ways at once, both rootless:
#
#   1. cgroup v2 memory cap (systemd-run --user --scope -p MemoryMax=): bounds the page cache the search
#      JVM may retain, so probed cells must be re-faulted from SSD instead of being served from RAM.
#      Verified behavior: a 3 GB file under a 400 MB cap re-faults on EVERY pass (12.5 s/pass) while
#      uncapped later passes are ~19 ms -- i.e. the cap really does evict file-backed pages, not just heap.
#   2. posix_fadvise(DONTNEED) on the index files before the run, so the FIRST query is cold too
#      (a cap alone still starts warm if a previous run left the file cached).
#
# The cap must cover heap + page cache + JVM overhead. Keep -Xmx well under KNN_RAM_CAP or the JVM OOMs
# instead of paging (that is the failure open.md warns about: heap alone does not constrain page cache).
#
# usage:
#   KNN_RAM_CAP=1200M KNN_HEAP=600m ./run_knn_ram_capped.sh
#   LLOYD_SCORE_IN_PLACE=1 KNN_RAM_CAP=1200M ./run_knn_ram_capped.sh     # A/B the scan kernel
#
set -euo pipefail

LUCENEUTIL_DIR=/local/home/rikhil/vectordb/luceneutil
RAM_CAP="${KNN_RAM_CAP:-1200M}"
export KNN_HEAP="${KNN_HEAP:-600m}"
export KNN_CLEAR_CACHE="${KNN_CLEAR_CACHE:-0}"   # default: reuse the cached index (we only vary search)

cd "$LUCENEUTIL_DIR"

# --- 1. drop the page cache for every cached index file (rootless, per-file) ---
echo "=== dropping page cache for cached index files (posix_fadvise DONTNEED) ==="
python3 - <<'PY'
import os, pathlib
root = pathlib.Path("knn-reuse/indices")
total = 0
for p in root.rglob("*"):
    if p.is_file():
        try:
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            total += p.stat().st_size
        except OSError as e:
            print(f"  warn: {p}: {e}")
print(f"  evicted ~{total/2**30:.2f} GiB of index files from page cache")
PY

# --- 2. run the benchmark inside a memory-capped cgroup scope ---
# NOTE: the cap applies to the WHOLE scope (python driver + forked JVMs). Index BUILDING under a tight
# cap is slow/fragile, so build the index first with the normal warm runner, then run this with
# KNN_CLEAR_CACHE=0 so this pass is search-only.
# The cap is applied by constants.py, which wraps the JVM COMMAND ITSELF in a systemd-run scope.
# Do NOT wrap this script instead: knnPerfTest.py forks the JVM as a child and those forks escape a
# scope placed around the driver (they land in the login session's uncapped scope), so the run would
# silently be WARM. Verified: with the driver wrapped, the JVM's cgroup showed memory.max=max.
echo "=== running with per-JVM cgroup memory cap: MemoryMax=$RAM_CAP (heap -Xmx$KNN_HEAP) ==="
echo "    index size on disk: $(du -sh knn-reuse/indices 2>/dev/null | cut -f1) -- cap should be BELOW this"
export KNN_RAM_CAP="$RAM_CAP"
export KNN_SKIP_SMELL="${KNN_SKIP_SMELL:-1}"
exec ./run_knn_bench.sh "${1:-1}"
