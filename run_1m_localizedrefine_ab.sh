#!/usr/bin/env bash
# A/B the localized-refine build optimization at 1M (streaming path exercised at streamFlushMinDocs<1M).
# CONTROL = full-beam Assign #2 (-Divf.fullBeamRefine=true, old behavior); TREAT = localized (default).
# Measures BUILD TIME (reindex takes) and RECALL (must hold -- localized reassignment changes clustering).
# Golden config: nlist=2000, spill=2, margin=1.10, osq int8, 64 index threads, streamRefineIters=3.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g KNN_INDEX_THREADS=64
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
run_arm() {
  local name="$1"; shift
  local log="$OUT/run_1m_locref_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  KNN_CLEAR_CACHE=1 env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  grep -E "reindex takes|force merge" "$log" | sed 's/^/    /'
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v nm="$name" -F'\t' '{printf "    %s: recall=%s lat=%s ms\n",nm,$1,$2}'
}
# CONTROL: full-beam Assign #2 (must clear cache -- write-time change).
run_arm FULLBEAM  JAVA_TOOL_OPTIONS=-Divf.fullBeamRefine=true
# TREAT: localized reassignment (codec default now).
run_arm LOCALIZED JAVA_TOOL_OPTIONS=
echo "=== done $(date) ==="
