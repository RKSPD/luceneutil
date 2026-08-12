#!/usr/bin/env bash
# Sketch-WIDTH sweep: does a shorter 1-bit sketch cut the coarse scan (50.8% of kernel is sketch-byte loads)
# without losing too much recall? WRITE-TIME (sketch is in the index), so each width REBUILDS the 1M index.
# The index dir name does NOT encode sketchDims, so we must delete the 1M index subdir per width or it would
# silently reuse the prior sketch (flag-not-passed trap). exact-nn is sketch-independent and is PRESERVED.
# Golden config, nprobe=60, counting-select on. 3 passes, MIN latency.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
IDXDIR="${LUCENEUTIL_DIR:-/local/home/rikhil/vectordb/luceneutil}/knn-reuse/indices"
run_width() {
  local dims="$1"; local log="$OUT/run_1m_skw_${dims}_${STAMP}.log"
  echo "=== [sketchDims=$dims] $(date) ==="
  # Delete ONLY the 1M lloydivf index subdir (keep the 40M one and exact-nn). Forces a rebuild at this width.
  find "$IDXDIR" -maxdepth 1 -type d -name "*-1000000-lloydivf-*" -exec rm -rf {} + 2>/dev/null
  KNN_CLEAR_CACHE=0 LLOYD_SKETCH_DIMS="$dims" ./run_knn_bench.sh 3 > "$log" 2>&1
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v d="$dims" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "    sketchDims=%s recall=%s MIN lat=%s ms\n",d,r,min}'
}
run_width 1024
run_width 512
run_width 768
echo "=== done $(date) ==="
