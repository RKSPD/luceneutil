#!/usr/bin/env bash
#
# JFR CPU PROFILE of the warm 40M search at nlist=6306 (=sqrt N), spill=5, margin=1.15, qb8.
#
# WHY: the build phase measured 87.4 / 55.1 ms per query WARM with avgCpuCount=1.000 and netCPU==latency
# (87.395 of 87.437 ms) -- i.e. ZERO I/O stall, 100% CPU. That cost is IRREDUCIBLE by any async I/O work
# (Stage C/D/E/F all attack I/O stall, and there is none here), so the only way to move it is to find and
# cut the dominant CPU term. This run attributes that 87 ms.
#
# The standing hypothesis, from benchmarks.md 10M-persist's clean profile:
#   coarse Hamming (xorBitCount)  ~46%   <- scales with docs_scanned = nprobe * docs/cell * spill_infl
#   rerank (planeDot)             ~27%   <- scales with BRUTE_N (2000), NOT with nlist
#   rest (routing, heap, quant)   ~27%
# At nlist=6306 there are ~6306 docs/cell vs ~398 at nlist=100k -- ~16x more docs per probe -- so coarse
# Hamming should dominate FAR more than 46% here. If the profile confirms that, the fix is nprobe/nlist
# (fewer docs scanned), not kernels. If it does NOT -- if something unexpected dominates -- that is a bug
# or a missed optimization, which is exactly what makes this worth profiling rather than assuming.
#
# ONE nprobe point (40) and nquery=1000: a single param combination keeps it to ONE .jfr with everything
# attributed to one operating point. Sweeping would interleave several profiles and muddy attribution.
#
# WARM + async OFF: warm is the ~10 GB/s production proxy, and with no I/O stall the ring stages are pure
# overhead (13 measured 26-41%) that would show up as syscall noise on top of the CPU we are trying to
# attribute. Isolate the scan CPU.
#
# Reuses the built index (nl6306-sp5-qb8) and the cached exact-NN -- search only, no rebuild.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb
RES="$OUT/results_40m_jfr_warm_${STAMP}.txt"
LOG="$OUT/run_40m_jfr_warm_${STAMP}.log"

# MUST match the built index key or the harness rebuilds (~90 min).
export KNN_NDOC=39767748
export KNN_NLIST=6306
export KNN_SPILL_BITS=5
export IVF_SPILL_MARGIN=1.15
export IVF_QUANTIZER=blocksphere
export IVF_BLOCK_P=2
export IVF_QUANT_BITS=8
export IVF_BEAM_SPILL=1
export IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32
export IVF_CENTROID_HNSW_BEAM_WIDTH=64
export IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000
export IVF_TRAIN_SAMPLE_CAP=100000

# Single operating point -> single clean profile.
export KNN_NPROBE="${KNN_NPROBE:-40}"
export KNN_NQUERY="${KNN_NQUERY:-1000}"

export KNN_CLEAR_CACHE=0             # reuse the index; never rebuild here
export KNN_DROP_CACHE_AFTER_WARMUP=0 # WARM (production proxy)
export KNN_HEAP="${KNN_HEAP:-8g}"
export KNN_SKIP_SMELL=1

# THE POINT OF THIS RUN.
export KNN_JFR=1

# Async off: no I/O stall to hide warm, so the ring is only overhead + syscall noise in the profile.
export LLOYD_URING_SKETCH_SCAN=0
export LLOYD_URING_RERANK=0
export LLOYD_URING_PIPELINE=0
export LLOYD_URING_RERANK_PIPELINE=0
export LLOYD_PREFETCH_CELLS=0

./ram_balloon.sh --release >/dev/null 2>&1 || true

{
  echo "############################################################"
  echo "# JFR CPU PROFILE -- warm 40M search"
  echo "# nlist=$KNN_NLIST (=sqrt N), spill=$KNN_SPILL_BITS, margin=$IVF_SPILL_MARGIN, qb$IVF_QUANT_BITS"
  echo "# nprobe=$KNN_NPROBE, nquery=$KNN_NQUERY, WARM, async OFF"
  echo "# attributing the 87.4/55.1 ms warm rows (avgCpuCount=1.000 => 100% CPU)"
  echo "# started $(date)"
  echo "############################################################"
} | tee -a "$RES"

./run_knn_bench.sh 1 2>&1 | tee "$LOG" | tail -40 | tee -a "$RES"
RC=${PIPESTATUS[0]}
echo "=== exit=$RC  $(date) ===" | tee -a "$RES"

# The harness runs ProfileResults automatically (benchUtil.profilerOutput) and prints PROFILE SUMMARY.
echo "" | tee -a "$RES"
echo "=== PROFILE SUMMARY (top CPU frames) ===" | tee -a "$RES"
sed -n '/PROFILE SUMMARY/,/^$/p' "$LOG" 2>/dev/null | head -45 | tee -a "$RES"
echo "" | tee -a "$RES"
echo "=== .jfr files written ===" | tee -a "$RES"
ls -lt /local/home/rikhil/vectordb/logs/*.jfr 2>/dev/null | head -3 | tee -a "$RES"
echo "=== done. results: $RES  full log: $LOG ==="
