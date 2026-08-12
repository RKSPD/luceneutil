#!/usr/bin/env bash
# A/B spillMargin 1.10 vs 1.15 at the new best operating point (nprobe=50, bruteN=2000, counting-select on).
# spillMargin is WRITE-TIME (controls adaptive beam-spill: higher margin -> more boundary docs spill to
# multiple cells -> more distinct-doc coverage per nprobe, bigger index). So each value REBUILDS the index.
# Question: does 1.15 lift recall enough to let us drop nprobe/bruteN further (net latency win)?
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=50 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g LLOYD_BRUTE_N=2000
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
run_m() {
  local m="$1"; local log="$OUT/run_1m_sm_${m}_${STAMP}.log"
  echo "=== [spillMargin=$m] $(date) ==="
  KNN_CLEAR_CACHE=1 IVF_SPILL_MARGIN="$m" ./run_knn_bench.sh 3 > "$log" 2>&1
  local idx=$(grep -iE "reindex takes" "$log" | tail -1 | grep -oE "[0-9]+\.[0-9]+ sec" | head -1)
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v m="$m" -v ix="$idx" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "    spillMargin=%s recall=%s MIN lat=%s ms  (reindex %s)\n",m,r,min,ix}'
}
run_m 1.10
run_m 1.15
echo "=== done $(date) ==="
