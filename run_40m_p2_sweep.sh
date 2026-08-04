#!/usr/bin/env bash
#
# >RAM nprobe sweep at 40M docs, BlockSphere p=2, spillBits=10 (adaptive beam spill).
#
# BALLOONED IN BOTH PHASES, but sized DIFFERENTLY -- the build's balloon is the delicate one:
#
#   Phase 1 (BUILD):  ballooned, so the write path also runs against a bounded page cache (temp code
#                     file writes + the merge's per-cell regroup reads must hit the device, not a
#                     165 GiB cache). This is what killed both earlier 40M attempts, so the sizing is
#                     the whole game: dmesg shows global_oom with the BALLOON itself as the largest
#                     anon consumer (anon-rss 244567040 kB) -- with swap=0 the kernel cannot evict anon,
#                     so a balloon that leaves too little headroom outbids the JVM and the JVM dies.
#                     The fix is headroom that covers PEAK JVM anon, not the 12 GiB default that broke
#                     it: heap (48g) + the cap-sized docCentroids array + centroids + native/GC
#                     overhead. BUILD_CACHE_GIB is therefore generous and the headroom explicit.
#   Phase 2 (SEARCH): balloon re-sized much tighter (search heap is only 8g and the reader mmaps rather
#                     than slurping), index already on disk, KNN_CLEAR_CACHE=0 so the cached index is
#                     reused. nprobe is search-time (reader honors -Dlloyd.nprobe) so the WHOLE sweep
#                     runs against ONE index -- no rebuild per nprobe.
#
# Why a balloon and not a cgroup cap / fadvise / O_DIRECT: benchmarks.md §13 measured all three. Page
# cache is charged to whichever cgroup FIRST faults a page, so warm rows are never re-charged and a cap
# never binds (46 MiB charged warm vs 1200 MiB cold, same file, same cap). fadvise pages just re-fault
# during the timed pass. O_DIRECT bypasses cache ENTIRELY, so there is no realistic hit rate. Swap is 0
# here, so anon pages cannot be evicted and page cache is the kernel's ONLY reclaim target -- the
# balloon leaves a PARTIAL cache, which is the realistic >RAM state.
#
# usage:
#   ./run_40m_p2_sweep.sh              # both phases
#   PHASE=build  ./run_40m_p2_sweep.sh # build only
#   PHASE=search ./run_40m_p2_sweep.sh # search only (index must already be cached)
set -uo pipefail

LUCENEUTIL_DIR=/local/home/rikhil/vectordb/luceneutil
RESULTS="${RESULTS:-/local/home/rikhil/vectordb/results_40m_p2_spill10.txt}"
PHASE="${PHASE:-both}"

# --- the config under test ----------------------------------------------------
# Whole corpus: the .vec file holds EXACTLY 39,767,748 vectors (162,888,695,808 B / 4096 B), so this is
# every doc, not a 40M prefix. Queries come from a separate file, so there is no train/test overlap.
export KNN_NDOC=39767748
# nlist = 40,000 => ~994 docs/cell at 40M. Between the two configs measured so far.
#
# WHY NOT sqrt(N)=6306 (the classic FAISS heuristic): MEASURED AND REJECTED 2026-08-04. Warm at nlist=6306
# (~6306 docs/cell), recall/latency was 0.717@10.8ms, 0.855@20.6ms, 0.931@56.2ms -- i.e. recall 0.931 cost
# ~56 ms where nlist=100k does similar recall at ~13.5 ms (§15). Recall also degraded too STEEPLY with
# nprobe to buy the docs/cell cost back by probing less (the §11a spill-frontier win did not transfer at
# margin=1.15). Root cause: sqrt(N) balances a LINEAR coarse-scan against the fine scan, but this codec
# routes through an HNSW centroid graph -- selection is O(log nlist), so the term sqrt(N) exists to balance
# nearly vanishes and the optimum moves to MANY SMALLER cells. §6 measured the same gradient at 1M
# (20k->65k nlist cut docs-visited ~23% and latency ~17% at fixed recall).
# nlist is capped at KMeans MAX_NUM_CENTROIDS = 1<<20, so 40k is well inside it.
export KNN_NLIST=40000
# spillBits=5: a CAP, not the fan-out -- with beamSpill the writer keeps only the leading cells within
# ivf.spillMargin x the nearest cell's distance (marginKeep), so interior docs stay single-cell and only
# boundary docs approach 6 copies. spillMargin set explicitly to 1.15 below.
export KNN_SPILL_BITS=5
export IVF_SPILL_MARGIN=1.15
# nprobe sweep: search-time => one index serves all five points.
# Grid sized for ~994 docs/cell. Matching the DOCS-SCANNED volume of the nlist=6306 run's recall points
# (0.717/0.855/0.931) needs nprobe ~32/76/254 here -- and recall should land BETTER than those, because
# smaller cells drag in fewer irrelevant neighbours per probe (the §6 effect: 20k->65k cut docs-visited
# 23% at FIXED recall). So this spans the predicted 0.93-0.95+ region with a low anchor for the curve.
export KNN_NPROBE="${KNN_NPROBE:-20,40,80,150,250}"

# p=2 Block-Sphere: -Divf.quantizer=blocksphere selects the p-dim-block layout, ivf.blockP=2 sets p.
# (The codec now DEFAULTS to blocksphere4, i.e. p=4, so this MUST be set explicitly to get p=2.)
# Both are write-time; the quantizer is in the index key as qzblocksphere, so it cannot silently reuse
# a p=4 index. blockP is NOT in the key -- see the guard below.
export IVF_QUANTIZER=blocksphere
export IVF_BLOCK_P=2
# 8-bit quantization (was 4-bit). quantBits IS in the index key (qb8 vs qb4), so this builds a fresh index
# rather than silently reusing the 4-bit one.
export IVF_QUANT_BITS=8

# Adaptive beam spill (the mechanism that makes spillBits a cap rather than a multiplier).
export IVF_BEAM_SPILL=1
export IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32
export IVF_CENTROID_HNSW_BEAM_WIDTH=64
export IVF_STREAM_REFINE_ITERS=3
# Streaming write path: quantize-on-arrival at flush + a bounded training sample, so resident write RAM
# is O(1)/doc instead of O(4 KB/doc). Non-negotiable at 40M (§12.6: the buffered path OOMs at 700k).
export IVF_STREAM_FLUSH_MIN_DOCS=100000
export IVF_TRAIN_SAMPLE_CAP=100000
export LLOYD_URING_DEBUG=1
# --- async I/O for the cold coarse scan (READER-SIDE ONLY -- no rebuild) -------
# Both flags only change HOW the same bytes are fetched, never which bytes or the results (reader
# asserts bit-identical; §13 confirmed it, recall 0.946/0.946/0.949/0.949 row-for-row between arms).
# Their index-time preconditions are already satisfied by the existing index: cellOrder=1 in meta
# (forced anyway by spillBits>0) and a persisted 1024-dim sketch table.
#
# Why they matter HERE and not in the warm 1M runs: the previous 40M sweep spent ~90% of wall blocked
# on page faults (201-503 ms latency against only 19-47 ms CPU, avgCpuCount 0.093-0.100) with the
# device at queue depth ~1.9 and 220 MB/s -- pure latency x depth, not a bandwidth ceiling. That is
# the serialized-fault regime where §13 measured 14.8x (55.6 -> 3.1 ms, depth 1 -> 64).
#
# NOTE §13 also measured Stage C costing 26-41% WARM, so these default ON only because this sweep is
# genuinely cold (balloon + fadvise DONTNEED on a 148 GiB index). Set LLOYD_URING_SKETCH_SCAN=0 to get
# the baseline arm back.
export LLOYD_URING_SKETCH_SCAN="${LLOYD_URING_SKETCH_SCAN:-1}"
# Stage D: the rerank gather. Stage C batches the 40 CONTIGUOUS sketch runs, which were never the
# problem -- measured at nprobe=40, sketch+postings is 12.6 MB of the 43.4 MB read per query, and the
# other 30.8 MB is readahead around 2000 scattered 536 B code-record reads (~15.1 KiB per record, ~29x
# on that component). Stage D sorts+coalesces those into a few hundred merged ranges in one submission.
export LLOYD_URING_RERANK="${LLOYD_URING_RERANK:-1}"
export LLOYD_RERANK_AUDIT="${LLOYD_RERANK_AUDIT:-1}"
# Stage E: overlap the cell reads with the Hamming scan (scan-on-completion) instead of
# read-everything-then-scan-everything. OFF by default -- MEASURED SLOWER than Stage C alone at 40M
# nprobe=40 (187.4/187.5 ms vs 175.1/171.8 ms, two paired rows each), and the CPU column says why:
# 19.4 -> 23.0 ms (+18%). Overlapping I/O with compute should not ADD compute. Two causes, both fixable:
#   1. Queue got SHALLOWER, not deeper: 32 lanes vs nprobe=40 means only ~8 cells are ever refilled, so
#      measured depth fell to 6.01 vs Stage C's 7.56.
#   2. One io_uring_enter PER CELL (40 syscalls) where Stage C submits all 40 ranges in ONE enter.
# The overhead is real but the payoff is missing because dispatch still happens AFTER selectCentroids()
# has returned the whole probe set -- so it pays per-cell submission cost without gaining the
# routing-window overlap that would justify it. Making this win needs selectCentroids() to emit cells
# incrementally (submit during traversal) plus grouped submission. Until then Stage C alone is best.
export LLOYD_URING_PIPELINE="${LLOYD_URING_PIPELINE:-0}"
# Stage A: madvise(WILLNEED) the whole probe set's sketch AND code runs up front. Advisory, ~free, and
# it covers the code table -- which Stage C does NOT batch (see the rerank note below).
export LLOYD_PREFETCH_CELLS="${LLOYD_PREFETCH_CELLS:-1}"
# Report whether the hints actually reach the storage layer, so "no effect" can be told apart from
# "never issued" -- the exact ambiguity that made §12.5 draw the wrong conclusion.
export LLOYD_PREFETCH_AUDIT="${LLOYD_PREFETCH_AUDIT:-1}"
# Skips the O(ndoc) pure-python duplicate scan, which holds an ndoc-entry dict and OOMs well below 40M.
export KNN_SKIP_SMELL=1

mkdir -p "$(dirname "$RESULTS")"
cd "$LUCENEUTIL_DIR"

{
  echo "############################################################"
  echo "# 40M docs (39,767,748 = whole corpus), nlist=$KNN_NLIST"
  echo "# BlockSphere p=2 (qzblocksphere, blockP=2), quantBits=$IVF_QUANT_BITS"
  echo "# spillBits=$KNN_SPILL_BITS (CAP; adaptive beam spill, spillMargin=$IVF_SPILL_MARGIN)"
  echo "# nprobe sweep: $KNN_NPROBE  (search-time -> ONE index serves all)"
  echo "# phase=$PHASE  started $(date)"
  echo "############################################################"
} | tee -a "$RESULTS"

# --- guard: blockP is NOT in the index key ------------------------------------
# The key records qz<quantizer> but not blockP, so a p=4 index built earlier under the name
# "blocksphere" would be silently reused for a p=2 run and the reader would misparse every record.
# Fail loudly instead of reporting a garbage recall.
STAMP="$LUCENEUTIL_DIR/knn-reuse/.blockP"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$IVF_BLOCK_P" ]; then
  echo "FATAL: cached indices were built with blockP=$(cat "$STAMP") but this run wants blockP=$IVF_BLOCK_P." | tee -a "$RESULTS"
  echo "       blockP is not in the index key, so reuse would silently misparse records." | tee -a "$RESULTS"
  echo "       Clear it first:  rm -rf $LUCENEUTIL_DIR/knn-reuse/indices" | tee -a "$RESULTS"
  exit 1
fi

# =============================== PHASE 1: BUILD ===============================
if [ "$PHASE" = "both" ] || [ "$PHASE" = "build" ]; then
  echo "=== releasing any stale balloon before re-sizing for the build ===" | tee -a "$RESULTS"
  ./ram_balloon.sh --release 2>&1 | tee -a "$RESULTS"

  # Heap: the spill pass allocates docCentroids = int[count * (1+spillBits)] up front, sized at the CAP
  # regardless of how much actually spills: 39,767,748 * 11 * 4 B = 1.63 GiB in ONE array (fits int, no
  # overflow). Plus centroids (100k * 1024 floats = 391 MiB), the int8 centroid codes, the training
  # sample, and the graph. 48g leaves generous room; the box has 247 GB and nothing else needs it here.
  export KNN_HEAP="${KNN_HEAP:-48g}"
  export KNN_CLEAR_CACHE=1

  # Balloon the BUILD too. Headroom must cover PEAK JVM anon, not just -Xmx: heap 48g + the 1.63 GiB
  # cap-sized docCentroids + 391 MiB centroids + native/GC/thread overhead across 8 index threads.
  # 80 GiB of headroom against a 48g heap is deliberately ~1.6x -- with swap=0 an under-estimate here
  # does not degrade gracefully, it gets the JVM OOM-killed (which is exactly how the two prior 40M
  # runs died). Everything left over after headroom + balloon is page cache the build must live within.
  export BALLOON_HEAP_HEADROOM="${BUILD_HEAP_HEADROOM:-80}"
  BUILD_CACHE_GIB="${BUILD_CACHE_GIB:-40}"
  echo "=== inflating build balloon: ~${BUILD_CACHE_GIB} GiB page cache, ${BALLOON_HEAP_HEADROOM} GiB JVM headroom ===" | tee -a "$RESULTS"
  ./ram_balloon.sh "$BUILD_CACHE_GIB" 2>&1 | tee -a "$RESULTS"
  # Release on any exit so a killed build cannot leave 130 GiB pinned for the next run.
  trap './ram_balloon.sh --release >/dev/null 2>&1' EXIT INT TERM
  grep -E "^(MemFree|MemAvailable|Cached):" /proc/meminfo | tee -a "$RESULTS"
  # Build only: one nprobe value. nprobe is search-time and out of the index key, so the index this
  # produces serves every nprobe in phase 2.
  #
  # NQUERY MUST MATCH THE SEARCH PHASE (1000, not 100): the exact-NN ground truth is cached under a key
  # that INCLUDES nquery (knn-reuse/exact-nn/...-<nquery>-...-knn-100.bin). If the build computes only the
  # 100-query GT, the search phase recomputes the 1000-query GT fresh -- and it does so AFTER inflating the
  # tight ~10 GiB search balloon, so the numpy GT pass (1 GiB doc block + ~1 GiB BLAS score scratch +
  # copies) OOMs. That is exactly how the 2026-08-03 23:41 run died at 95.6% of the GT pass (exit 137, no
  # matching oom-kill line because it was the tight-envelope death, not the balloon-vs-JVM one). Computing
  # the 1000-query GT HERE, under the roomy build balloon (~120 GiB free), caches it for phase 2 to reuse.
  KNN_NPROBE=40 KNN_NQUERY="${KNN_NQUERY:-1000}" \
    ./run_knn_bench.sh 1 2>&1 | tee -a "$RESULTS"
  BUILD_RC=${PIPESTATUS[0]}
  echo "$IVF_BLOCK_P" > "$STAMP"

  echo "=== build phase exit=$BUILD_RC  $(date) ===" | tee -a "$RESULTS"

  # Distinguish "build failed" from "balloon starved the JVM" -- exit 137 (SIGKILL) plus an oom-kill line
  # naming our uid is the signature of the failure mode that killed both prior 40M runs. Say so loudly,
  # because the actionable fix is different: raise BUILD_CACHE_GIB / BUILD_HEAP_HEADROOM, not the codec.
  if [ "$BUILD_RC" -ne 0 ]; then
    if dmesg -T 2>/dev/null | tail -n 200 | grep -qiE "oom-kill|Out of memory"; then
      echo "BUILD OOM-KILLED: the build balloon left too little headroom for the JVM." | tee -a "$RESULTS"
      echo "  retry with e.g. BUILD_CACHE_GIB=80 BUILD_HEAP_HEADROOM=100 PHASE=build $0" | tee -a "$RESULTS"
      dmesg -T 2>/dev/null | grep -iE "oom-kill|Out of memory" | tail -3 | tee -a "$RESULTS"
    fi
    echo "BUILD FAILED -- not starting the search phase (a balloon on a broken index measures nothing)." | tee -a "$RESULTS"
    exit "$BUILD_RC"
  fi
  du -sh knn-reuse/indices 2>&1 | tee -a "$RESULTS"

  # Drop the build balloon so phase 2 can size its own (a 130 GiB build balloon would leave the search
  # phase nothing to re-reserve, and ram_balloon.sh does not stack).
  echo "=== releasing build balloon before the search phase ===" | tee -a "$RESULTS"
  ./ram_balloon.sh --release 2>&1 | tee -a "$RESULTS"
fi

# ============================== PHASE 2: SEARCH ==============================
if [ "$PHASE" = "both" ] || [ "$PHASE" = "search" ]; then
  INDEX_GIB=$(du -sB1 knn-reuse/indices 2>/dev/null | cut -f1)
  INDEX_GIB=$(( ${INDEX_GIB:-0} / 1073741824 ))
  echo "=== index on disk: ${INDEX_GIB} GiB ===" | tee -a "$RESULTS"

  # Target cache = ~1/4 of the index, so the working set genuinely exceeds RAM but a realistic partial
  # cache remains. Floor of 8 GiB so the JVM + kernel are not starved into thrashing.
  #
  # WARNING: sizing off the ON-DISK total OVERSHOOTS badly -- much of the index is spill record
  # DUPLICATION that no single query reads, so INDEX_GIB/4 can leave the whole touched set cached, the
  # pass re-warms itself, and the "cold" row is a WARM number (the §13/§15 trap in yet another costume).
  #
  # CORRECTION (2026-08-04): an earlier version of this comment said to "size against vec_RAM, measured
  # 39,442 MB". That was WRONG -- vec_RAM is not a measurement. KnnGraphTester.java:1301 computes it as
  # totalVectorCount * (realEncodingByteSize * dim + overhead), i.e. from doc count / dim / encoding ONLY.
  # It is blind to nlist, spill duplication and the sketch table, and prints the SAME value for configs
  # with 2x different touched sets (verified identical for nlist=100k/sp10/qb4 vs nlist=6306/sp5/qb8).
  # Do not use it as a touched-set proxy. Prefer an explicit small TARGET_CACHE_GIB, and VERIFY coldness
  # from the result rows' avgCpuCount (~1.0 => warm/CPU-bound => invalid as a cold number).
  TARGET_CACHE="${TARGET_CACHE_GIB:-$(( INDEX_GIB / 4 ))}"
  [ "$TARGET_CACHE" -lt 8 ] && TARGET_CACHE=8
  # Search heap is small on purpose: the code/sketch tables are read through mmap (§12.6, the reader no
  # longer slurps), so heap does NOT need to hold them -- and every GiB of heap is a GiB the page cache
  # cannot use. Must stay under the balloon's headroom.
  export KNN_HEAP="${KNN_SEARCH_HEAP:-8g}"
  export BALLOON_HEAP_HEADROOM=12
  export KNN_CLEAR_CACHE=0          # REUSE the built index; do NOT rebuild per nprobe
  # REQUIRED for a genuinely cold row, and the reason is subtle enough that it was missed twice:
  # the balloon + fadvise below evict the index, but the HARNESS then runs a full warmup over every query
  # BEFORE the timed pass, which re-faults the ~13 MB/query working set into whatever cache the balloon
  # left. Measured without this: 13.5 ms at avgCpuCount 0.997 -- pure CPU, zero I/O stall, i.e. a WARM
  # number wearing a cold run's clothes (benchmarks.md §15, and §13 bug 2 in a new costume). This flag
  # evicts AFTER warmup and immediately before timing, which is the only ordering that yields a cold pass.
  export KNN_DROP_CACHE_AFTER_WARMUP="${KNN_DROP_CACHE_AFTER_WARMUP:-1}"
  export KNN_NQUERY="${KNN_NQUERY:-1000}"

  echo "=== inflating balloon to leave ~${TARGET_CACHE} GiB for page cache ===" | tee -a "$RESULTS"
  ./ram_balloon.sh "$TARGET_CACHE" 2>&1 | tee -a "$RESULTS"
  # Release the balloon on ANY exit, or it survives the script and silently skews the next run.
  trap './ram_balloon.sh --release >/dev/null 2>&1' EXIT INT TERM

  # Evict the index so the FIRST nprobe row is cold too (a balloon alone starts warm if the build just
  # left the file cached -- that asymmetry is what made only row 1 a real >RAM measurement in §13).
  echo "=== evicting cached index files (posix_fadvise DONTNEED) ===" | tee -a "$RESULTS"
  python3 - <<'PY' 2>&1 | tee -a "$RESULTS"
import os, pathlib
total = 0
for p in pathlib.Path("knn-reuse/indices").rglob("*"):
    if p.is_file():
        try:
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            total += p.stat().st_size
        except OSError as e:
            print(f"  warn: {p}: {e}")
print(f"  evicted ~{total/2**30:.2f} GiB from page cache")
PY

  grep -E "^(MemFree|MemAvailable|Cached):" /proc/meminfo | tee -a "$RESULTS"
  echo "=== search sweep nprobe=$KNN_NPROBE  $(date) ===" | tee -a "$RESULTS"
  SEARCH_START_LINE=$(wc -l < "$RESULTS")
  ./run_knn_bench.sh 1 2>&1 | tee -a "$RESULTS"
  SEARCH_RC=${PIPESTATUS[0]}
  echo "=== search phase exit=$SEARCH_RC  $(date) ===" | tee -a "$RESULTS"

  # --- ENGAGEMENT GUARD ---------------------------------------------------------
  # The single most expensive bug in this file's history is a run whose numbers were read as "async I/O
  # does not help" when the async path was never entered at all (§13 bug 1: the >RAM A/B passed
  # -Dlloyd.sketchScan but not -Dlloyd.uringSketchScan, so uringSketchScanCells() ran zero times and the
  # 44.5 ms / 0.166-cores row was the UNACCELERATED path). It cost a wrong conclusion in benchmarks.md
  # that stood until §13. It recurred in the Aug 1 sweep: LLOYD_URING_DEBUG=1 was set here but
  # LLOYD_URING_SKETCH_SCAN was not, giving 5 nprobe rows at 0.09 cores and zero engagement lines.
  #
  # So do not trust the flag -- verify the reader SAID it engaged, and fail loudly otherwise. A silently
  # unaccelerated row is worse than no row, because it gets written down as a measurement.
  # Stage E prints "Stage-E pipeline engaged" and takes priority over Stage C, so accept either -- keying
  # only on Stage C would false-alarm the moment the pipeline is on (and a guard that cries wolf gets
  # ignored, which defeats its whole purpose).
  if [ "${LLOYD_URING_SKETCH_SCAN}" = "1" ] || [ "${LLOYD_URING_PIPELINE}" = "1" ]; then
    # Match any "[lloyd uring] ... engaged" line, not the old "Stage-C engaged" wording: the reader now
    # prints "[lloyd uring] engaged on thread <t> usingUring= O_DIRECT= depth= file=" and the Stage-C/E
    # spelling is gone. Keying on the old text made this guard fire on a run where the ring HAD engaged
    # (usingUring=true depth=128) and discard a valid result -- a false alarm is as damaging as a miss,
    # because a guard that cries wolf gets ignored, which is what this guard's own rationale warns about.
    ENGAGED=$(tail -n +"$SEARCH_START_LINE" "$RESULTS" | grep -cE "\[lloyd uring\].*(engaged|Stage-(C|E))")
    if [ "$ENGAGED" -eq 0 ]; then
      echo "FATAL: async I/O was requested but the reader never printed a Stage-C/Stage-E engagement line." | tee -a "$RESULTS"
      echo "       Every latency row above is the UNACCELERATED per-doc path -- do NOT record them as" | tee -a "$RESULTS"
      echo "       an async-I/O measurement (this is exactly benchmarks.md §13 bug 1)." | tee -a "$RESULTS"
      echo "       Check, in order:" | tee -a "$RESULTS"
      echo "         - the JVM cmd line above actually contains -Dlloyd.uringSketchScan=true" | tee -a "$RESULTS"
      echo "         - meta says cellOrder=1 (Stage C requires contiguous per-cell sketch runs)" | tee -a "$RESULTS"
      echo "         - CellBatchReader.isAvailable() -- io_uring_setup may be blocked on this kernel" | tee -a "$RESULTS"
      echo "         - a single cell's sketch run did not exceed lloyd.uringBufBytes (64 MiB default)" | tee -a "$RESULTS"
      SEARCH_RC=1
    else
       echo "=== engagement OK: $ENGAGED async-I/O engagement line(s) ===" | tee -a "$RESULTS"
      tail -n +"$SEARCH_START_LINE" "$RESULTS" | grep -m1 "\[lloyd uring\] Stage-C engaged" | tee -a "$RESULTS"
    fi
  fi
fi

echo "=== done $(date) -- results in $RESULTS ===" | tee -a "$RESULTS"
