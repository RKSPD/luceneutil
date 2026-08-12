#!/usr/bin/env bash
# Tests the "routing quality caps nprobe" thesis on the cached 1M localized-refine index (search-time,
# no rebuild). Three probes:
#   1. routeAudit: mean overlap of flat-graph top-nprobe vs EXACT nearest-nprobe (the routing gap, direct).
#   2. noGraph nprobe sweep: nprobe-to-0.95 with PERFECT (exact) routing -> the headroom ceiling.
#   3. beamFactor sweep at fixed nprobe=45: does widening the beam (free, <1% CPU) recover recall so the
#      SAME nprobe clears 0.95 at lower latency?
# If noGraph hits 0.95 at much lower nprobe than the graph, routing is the limiter and beamFactor is the win.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
run() {
  local name="$1"; shift; local log="$OUT/run_1m_routing_${name}_${STAMP}.log"
  echo "=== [$name] $(date) ==="
  KNN_CLEAR_CACHE=0 env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  grep -iE "routeAudit" "$log" | tail -1 | sed 's/^/    /'
  grep "ivfNprobe = " "$log" | head -1 | sed 's/^/    /'
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "    recall=%s lat=%s ms\n",$1,$2}'
}
# 1. Routing gap at the golden operating point.
run AUDIT      KNN_NPROBE=45 LLOYD_ROUTE_AUDIT=1
# 2. Perfect-routing headroom: how low can nprobe go if routing is exact?
run NOGRAPH    KNN_NPROBE=25,30,35,40,45 LLOYD_NO_GRAPH=1
# 3. Free beam widening at the graph's nprobe=45.
run BEAM2      KNN_NPROBE=45 LLOYD_BEAM_FACTOR=2.0
run BEAM4      KNN_NPROBE=45 LLOYD_BEAM_FACTOR=4.0
echo "=== done $(date) ==="
