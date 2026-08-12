#!/usr/bin/env bash
# Sweep bruteN DOWN at nprobe=60 on the cached golden m1.10 index (bulk dedup, SIMD kernels). Profile said
# rerank (~26%) + heap bookkeeping (~18%) both scale with bruteN, so a smaller shortlist should cut latency
# with little recall loss (nprobe=60 already feeds a strong shortlist). bruteN is read-side (-Dlloyd.bruteN,
# not in the index key) -> every value reuses the cached index, NO rebuild. One JVM per value (static read).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_bruten_down_${STAMP}.log"
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
BRUTENS="${BRUTENS:-2000 1500 1000 750 500}"
echo "=== bruteN-down sweep, nprobe=60, bulk dedup $(date) ===" | tee "$LOG"
for bn in $BRUTENS; do
  echo "--- bruteN=$bn ---" | tee -a "$LOG"
  KNN_CLEAR_CACHE=0 LLOYD_BRUTE_N="$bn" ./run_knn_bench.sh 1 > "$OUT/run_1m_brutendown_${bn}_${STAMP}.log" 2>&1
  grep "hammingKernel" "$OUT/run_1m_brutendown_${bn}_${STAMP}.log" | grep -v "cellsScanned=0" | tail -1 | sed 's/^/    /'
  grep "^SUMMARY" "$OUT/run_1m_brutendown_${bn}_${STAMP}.log" | sed 's/SUMMARY: //' | awk -v bn="$bn" -F'\t' '{printf "    bruteN=%s  recall=%s  lat=%s ms\n",bn,$1,$2}' | tee -a "$LOG"
done
echo "=== done $(date). log: $LOG ==="
