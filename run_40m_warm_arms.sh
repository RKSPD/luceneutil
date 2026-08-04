#!/usr/bin/env bash
#
# WARM (in-page-cache) arms at 40M. Run AFTER run_40m_stageF_arms.sh has built the index.
#
# WHY THIS IS THE PRODUCTION-RELEVANT ARM, not the cold one: production runs ~10 GB/s NVMe. Page cache is
# ~10-50 GB/s -- the SAME ORDER as that device -- while this box's EBS volume is ~300 MB/s, i.e. ~30x
# SLOWER than production storage. So the WARM number is the better proxy for a 10 GB/s deployment, and the
# cold >RAM number is a worst-case that production will not see.
#
# What that reframes: benchmarks.md §13 measured the uring stages costing 26-41% WARM (ring syscalls +
# buffer copies are pure overhead when there is no I/O stall to hide). On EBS that cost is irrelevant --
# it is dwarfed by the fault latency it removes. At 10 GB/s it may DOMINATE. These arms measure that
# directly, and they are what should decide whether any async stage is safe to default ON.
#
# NOTE what stays true at 10 GB/s: bandwidth is not what makes scattered reads slow, PER-OP LATENCY is
# (~50-80us/op on NVMe). 2000 serialized scattered record reads is ~160 ms even on a fast device, so the
# COALESCING + QUEUE DEPTH that Stage C/D provide still matter -- arguably more, since a fast device is
# the one you waste most by serializing. It is specifically the OVERLAP (Stage E/F) whose value shrinks,
# because overlap can only hide min(I/O, CPU) and at 10 GB/s the I/O side is ~1.6 ms of a ~22 ms query.
#
# No balloon, no fadvise, no KNN_DROP_CACHE_AFTER_WARMUP: the harness's own warmup pass leaves the ~13
# MB/query working set resident, which is exactly the state we want to measure here.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb

# Same knobs as the sweep, minus everything that manufactures a cold cache.
# CRITICAL: these MUST match the BUILT index key (nl6306-sp5-qb8-qzblocksphere) or the harness treats it
# as a cache miss and REBUILDS -- ~90 min of indexing + force-merge for what should be a search-only pass.
export KNN_NDOC=39767748
# Overridable so a caller can point the warm arms at a different built index (e.g. nlist=40000) without
# editing this file. A hard assignment here would SILENTLY override the caller's KNN_NLIST and search the
# wrong index -- or miss the cache key and trigger a ~90 min rebuild.
export KNN_NLIST="${KNN_NLIST:-6306}"
export KNN_SPILL_BITS="${KNN_SPILL_BITS:-5}"
export IVF_SPILL_MARGIN=1.15
# nprobe spans the low region where spill may pay back nlist=sqrt(N)'s docs/cell cost, plus 40 as the
# bridge to the build-phase warm rows (87.4/55.1 ms at avgCpuCount=1.000).
export KNN_NPROBE="${KNN_NPROBE:-5,12,40}"
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
export KNN_SKIP_SMELL=1
export KNN_CLEAR_CACHE=0            # reuse the built index
export KNN_DROP_CACHE_AFTER_WARMUP=0 # WARM: keep the warmup's pages resident
export KNN_HEAP="${KNN_HEAP:-8g}"
export KNN_NQUERY="${KNN_NQUERY:-1000}"
export LLOYD_URING_DEBUG=1
export LLOYD_PREFETCH_CELLS=0        # advisory hints are noise warm; isolate the ring stages
export LLOYD_RERANK_AUDIT=1

./ram_balloon.sh --release >/dev/null 2>&1 || true

run_arm() {
  local name="$1"; shift
  local res="$OUT/results_40m_warm_${name}_${STAMP}.txt"
  local log="$OUT/run_40m_warm_${name}_${STAMP}.log"
  echo "=== [warm arm $name] $(date) -> $log ==="
  env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  echo "=== [warm arm $name] exit=$? ==="
  grep -E "recall|latency\(ms\)" "$log" 2>/dev/null | tail -8
  if [ "$name" = "SEF" ] && ! grep -q "Stage-F streaming rerank engaged" "$log"; then
    echo "  [guard] FATAL: SEF warm arm never engaged Stage F -- these rows are Stage D. Do NOT record."
  fi
}

# BASELINE is the one that matters most here: if it WINS warm, no async stage should default on.
run_arm BASELINE LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0
run_arm CD       LLOYD_URING_SKETCH_SCAN=1 LLOYD_URING_RERANK=1
run_arm SEF      LLOYD_URING_SKETCH_SCAN=1 LLOYD_URING_PIPELINE=1 LLOYD_URING_RERANK=1 LLOYD_URING_RERANK_PIPELINE=1

echo "=== warm arms done $(date). results: $OUT/results_40m_warm_*_${STAMP}.txt ==="
