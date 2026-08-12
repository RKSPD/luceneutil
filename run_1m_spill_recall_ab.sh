#!/usr/bin/env bash
# Golden 1M, margin=1.10, osq int8, bulk dedup, nprobe=40 -- spill=2 vs spill=5, to see if higher spill
# recovers recall. spill is WRITE-time (in the index key), so each arm builds its own index. Bulk dedup on
# (bulk Hamming kernel). Also sweeps a couple nprobe points per arm to see the recall/latency curve.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE="${KNN_NPROBE:-40,60,80}" KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
run_arm() {
  local sp="$1"; local log="$OUT/run_1m_spillrecall_sp${sp}_${STAMP}.log"
  echo "=== [spill=$sp] $(date) ==="
  KNN_SPILL_BITS="$sp" KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 1 > "$log" 2>&1
  grep -E "reindex takes" "$log" | sed 's/^/    /'
  grep "hammingKernel" "$log" | grep -v "cellsScanned=0" | tail -1 | sed 's/^/    /'
  echo "    nprobe order: $KNN_NPROBE  index_size(MB) below"
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v sp="$sp" -F'\t' '{printf "    spill=%s recall=%s lat=%s ms idxMB=%s\n",sp,$1,$2,$22}'
}
run_arm 2
run_arm 5
echo "=== done $(date) ==="
