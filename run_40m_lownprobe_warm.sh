#!/usr/bin/env bash
#
# LOW-NPROBE WARM SWEEP at nlist=6306 (=sqrt N), spill=5, margin=1.15, qb8.
#
# THE QUESTION: does adaptive spill let us cut nprobe far enough to pay back the 16x docs/cell that
# nlist=sqrt(N) costs? benchmarks.md §11a measured that spill collapses the p95 NN-cell rank 146 -> 15-33
# ("the actual lever is nprobe reduction, which §11a proves is available but §11b has NOT yet spent"), and
# the build phase measured 87.4/55.1 ms at nprobe=40 WARM with avgCpuCount=1.000 (pure CPU, zero I/O
# stall). Linear-in-nprobe extrapolation puts nlist=6306 at parity with the old nlist=100k config
# (~13.5 ms, §15) around nprobe~10, and possibly AHEAD at nprobe~5. This sweep measures that directly --
# specifically whether RECALL holds as nprobe drops, which is the whole bet.
#
# WHY WARM, NOT COLD: production runs ~10 GB/s NVMe. Page cache (~10-50 GB/s) is the same order; this
# box's EBS is ~300 MB/s, ~30x slower. So the WARM number is the production proxy and the cold >RAM sweep
# is a worst case production will not see. It is also the regime where this cost is IRREDUCIBLE: at
# avgCpuCount=1.000 there is no I/O stall, so no amount of async I/O work (Stage C/D/E/F) can hide it --
# only scanning fewer docs can. That makes nprobe the load-bearing lever, not an optimization.
#
# Search-time only: nprobe is not in the index key, so this reuses the SAME index the arms ran on. No
# rebuild, no ground-truth recompute (exact-nn is cached and config-independent).
#
# NOTE the async stages are OFF here. Warm, they are pure overhead (§13 measured 26-41%), and this sweep
# is about docs-scanned CPU, not I/O overlap. Keeping them off isolates the operating-point question.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb
RES="$OUT/results_40m_lownprobe_warm_${STAMP}.txt"
LOG="$OUT/run_40m_lownprobe_warm_${STAMP}.log"

# Must MATCH the built index exactly or the harness rebuilds: nl6306-sp5-qb8-qzblocksphere.
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

# The sweep itself: spans the predicted parity point (~10) and the §11a spill region (15-33), with 40 as
# the bridge back to the build-phase measurement.
export KNN_NPROBE="${KNN_NPROBE:-3,5,8,12,20,30,40}"
export KNN_NQUERY="${KNN_NQUERY:-1000}"

export KNN_CLEAR_CACHE=0             # REUSE the built index -- never rebuild here
export KNN_DROP_CACHE_AFTER_WARMUP=0 # WARM: keep the warmup's pages resident (production proxy)
export KNN_HEAP="${KNN_HEAP:-8g}"
export KNN_SKIP_SMELL=1
# Async I/O off: warm it is pure overhead and this sweep isolates docs-scanned CPU.
export LLOYD_URING_SKETCH_SCAN=0
export LLOYD_URING_RERANK=0
export LLOYD_URING_PIPELINE=0
export LLOYD_URING_RERANK_PIPELINE=0
export LLOYD_PREFETCH_CELLS=0

# No balloon: this is the WARM arm. Release any stale one so it cannot skew the cache.
./ram_balloon.sh --release >/dev/null 2>&1 || true

{
  echo "############################################################"
  echo "# LOW-NPROBE WARM sweep -- operating point for nlist=sqrt(N)"
  echo "# 40M docs, nlist=$KNN_NLIST (=sqrt N), spill=$KNN_SPILL_BITS, margin=$IVF_SPILL_MARGIN, qb$IVF_QUANT_BITS"
  echo "# nprobe: $KNN_NPROBE   (search-time -> ONE index serves all)"
  echo "# WARM (no balloon, no fadvise) = the ~10 GB/s production proxy"
  echo "# reference: nprobe=40 build rows were 87.4/55.1 ms at avgCpuCount=1.000"
  echo "#            old nlist=100k warm was ~13.5 ms @ nprobe=40 (benchmarks.md 15)"
  echo "# started $(date)"
  echo "############################################################"
} | tee -a "$RES"

./run_knn_bench.sh 1 2>&1 | tee -a "$RES" > "$LOG"
RC=${PIPESTATUS[0]}
echo "=== exit=$RC  $(date) ===" | tee -a "$RES"

# Pull the result table. recall vs latency across nprobe IS the deliverable here.
echo "=== RESULT TABLE ===" | tee -a "$RES"
grep -E "recall|latency\(ms\)|avgCpuCount|^ *0\.[0-9]+ +[0-9]+\.[0-9]+" "$LOG" 2>/dev/null | tail -25 | tee -a "$RES"
echo "=== done. results: $RES ==="
