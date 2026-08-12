#!/usr/bin/env bash
# Map the int8-query rerank curve: does a LOWER nprobe now clear 0.95 under 1.5ms? int8 banked 0.951@1.73ms
# at bn750/np40; sweep np {25,35,40} x bruteN {600,750} to find the cheapest 0.95 point. 2-bit Hamming coarse,
# sp2/m1.20, int8 query ON, reusing the baseline index (read-path flags only). runs=3 medians.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=residual4 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=25,35,40
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_SKETCH_LO_CLIP=3.5
export LLOYD_INT8_QUERY=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

declare -a LOGS
for BN in 600 750; do
  export LLOYD_BRUTE_N=$BN
  LOG="$OUT/run_1m_r4_int8np_bn${BN}_${STAMP}.log"
  LOGS+=("$LOG")
  echo "=== r4 INT8 npsweep bruteN=$BN (reuse baseline, clear=0) $(date) ===" | tee "$LOG"
  KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 3 >> "$LOG" 2>&1
  echo "bn$BN exit=$?"
done

echo; echo "=== INT8 npsweep RESULTS: recall / latency(ms) at np {25,35,40} ==="
for L in "${LOGS[@]}"; do
  echo "--- $(basename "$L") ---"
  grep "^SUMMARY" "$L" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms sel=%s\n",$1,$2,$4}' | sort -u
done
echo "=== logs: ${LOGS[*]} ==="
