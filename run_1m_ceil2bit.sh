#!/usr/bin/env bash
# 2-bit THERMOMETER sketch A/B inside the ceilBrute audit: does a 2-bit coarse tier lift the recall@N curve
# vs the 1-bit sign sketch (so we could HALVE bruteN)? No format change / no reindex -- reconstructs docs
# from int8 codes and encodes a candidate 2-bit sketch on the fly. Sweeps alpha (threshold = alpha*mean|v|),
# the one free knob. Reuses the cached 1M index (already at sketchDims=1024 from the ceilBrute run).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY="${KNN_NQUERY:-100}" KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
run_alpha() {
  local a="$1"; local log="$OUT/run_1m_ceil2bit_a${a}_${STAMP}.log"
  echo "=== [alpha=$a] $(date) ==="
  KNN_CLEAR_CACHE=0 JAVA_TOOL_OPTIONS="-Dlloyd.ceilAudit=true -Dlloyd.ceilBrute=true -Dlloyd.ceil2bit=true -Dlloyd.ceil2bitAlpha=$a" ./run_knn_bench.sh 1 > "$log" 2>&1
  echo "  --- 1-bit sign baseline ---"; grep -E "brute sign-.*N=" "$log" | sed 's/^ *//'
  echo "  --- 2-bit thermometer alpha=$a ---"; grep -E "THERM-2bit.*N=" "$log" | sed 's/^ *//'
}
run_alpha "${1:-1.0}"
echo "=== done $(date) ==="
