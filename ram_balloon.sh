#!/usr/bin/env bash
#
# Force a genuine >RAM regime by RESERVING anonymous memory, so the kernel must evict page cache.
#
# Why this works where everything else failed:
#   - cgroup MemoryMax  : page cache is charged to whichever cgroup FIRST FAULTS a page in, so a file
#                         already warm from a previous run is never re-charged and the cap never binds.
#                         Measured: 46 MiB charged warm vs 1200 MiB cold, same file, same cap.
#   - fadvise(DONTNEED) : pages simply re-fault and re-cache during the timed pass.
#   - O_DIRECT          : works, but bypasses cache ENTIRELY -- no realistic hit rate.
#   - this balloon      : swap is 0 on this box, so anon pages CANNOT be paged out; the kernel's only
#                         reclaim target is page cache. Verified: cache 173 GiB -> 33 GiB under 200 GiB.
#                         Leaves a PARTIAL cache, which is the realistic >RAM state.
#
# DO NOT hold a balloon during INDEXING. Indexing needs ~9 GiB of unreclaimable heap (dominated by the
# float seed sample) and rewrites a ~20 GiB temp code file once per refine iteration; squeezing cache
# there makes the build thrash unpredictably, and with zero swap a too-small balloon gets the JVM
# OOM-killed rather than slowed. Build first, then balloon for the SEARCH pass only.
#
# usage:  ./ram_balloon.sh <target_cache_gib>       # size the balloon to leave ~this much for cache
#         ./ram_balloon.sh --release
set -uo pipefail

if [ "${1:-}" = "--release" ]; then
  pkill -TERM -f "[b]alloon_worker.py" 2>/dev/null && echo "balloon released" || echo "no balloon running"
  sleep 3; grep -E "^(MemFree|MemAvailable|Cached):" /proc/meminfo
  exit 0
fi

TARGET_CACHE_GIB="${1:?usage: $0 <target_cache_gib> | --release}"
TOTAL_GIB=$(awk '/^MemTotal:/{printf "%.0f", $2/1048576}' /proc/meminfo)
# Leave headroom for the search JVM's heap + kernel; balloon = total - heap_headroom - target_cache.
HEAP_HEADROOM_GIB="${BALLOON_HEAP_HEADROOM:-12}"
BALLOON_GIB=$(( TOTAL_GIB - HEAP_HEADROOM_GIB - TARGET_CACHE_GIB ))
if [ "$BALLOON_GIB" -le 0 ]; then
  echo "nothing to reserve (total=${TOTAL_GIB} headroom=${HEAP_HEADROOM_GIB} target=${TARGET_CACHE_GIB})"; exit 1
fi

cat > /tmp/balloon_worker.py <<'PY'
import ctypes, os, signal, sys, time
n = int(float(sys.argv[1]) * (1 << 30))
buf = ctypes.create_string_buffer(n)            # anon mapping
step, addr = 4096, ctypes.addressof(buf)
for off in range(0, n, step):                   # touch every page => real physical backing
    ctypes.c_char.from_address(addr + off).value = b'\x01'
print(f"resident pid={os.getpid()}", flush=True)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
try:
    while True: time.sleep(3600)
except (KeyboardInterrupt, SystemExit):
    pass
PY

echo "total=${TOTAL_GIB} GiB | heap headroom=${HEAP_HEADROOM_GIB} GiB | target cache=${TARGET_CACHE_GIB} GiB"
echo "reserving ${BALLOON_GIB} GiB anon (takes ~$((BALLOON_GIB/3))s to touch)..."
setsid nohup python3 /tmp/balloon_worker.py "$BALLOON_GIB" > /tmp/balloon_worker.log 2>&1 < /dev/null &
disown
for _ in $(seq 1 120); do grep -q resident /tmp/balloon_worker.log 2>/dev/null && break; sleep 2; done
cat /tmp/balloon_worker.log
grep -E "^(MemFree|MemAvailable|Cached):" /proc/meminfo
echo "release with: $0 --release"
