#!/usr/bin/env bash
#
# bruteN sweep on the GOLDEN 1M index (nlist=2000, spillBits=2, beam spill, osq int8), SIMD Hamming +
# shortlist dedup on. bruteN is the coarse-shortlist heap size and is READ-SIDE ONLY (-Dlloyd.bruteN,
# reader static, NOT in the index key), so every point reuses the cached index -- NO rebuild.
#
# WHY: shortlist dedup engages the fast BULK kernel but lets spill duplicates occupy heap slots, so the
# effective DISTINCT-doc rerank pool is < bruteN and recall plateaus ~0.93. Growing bruteN restores the
# distinct pool while keeping the bulk kernel. Find the smallest bruteN that clears 0.95 and see the
# latency -- the goal is sub-2ms @ 0.95 with the bulk path still live.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_bruten_sweep_${STAMP}.log"

export KNN_NDOC=1000000
export KNN_NLIST=2000
export KNN_SPILL_BITS=2
export IVF_QUANTIZER=osq
export IVF_QUANT_BITS=8
export IVF_BEAM_SPILL=1
export IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32
export IVF_CENTROID_HNSW_BEAM_WIDTH=64
export IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000
export IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40
export KNN_NQUERY="${KNN_NQUERY:-1000}"
export KNN_SKIP_SMELL=1
export KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1
# Shortlist dedup => bulk kernel engaged; bruteN is the lever that gives it enough distinct survivors.
export LLOYD_SHORTLIST_DEDUP=1
export KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0
export LLOYD_URING_RERANK=0
export LLOYD_URING_PIPELINE=0
export LLOYD_URING_RERANK_PIPELINE=0
export LLOYD_PREFETCH_CELLS=0

# One JVM per bruteN (it is a static read once per classload, so it cannot be swept within a run).
BRUTENS="${BRUTENS:-2000 3000 4000 6000 8000}"
echo "=== bruteN sweep (SIMD + shortlist dedup, golden spill=2) $(date) ===" | tee "$LOG"
for bn in $BRUTENS; do
  echo "--- bruteN=$bn ---" | tee -a "$LOG"
  # KNN_CLEAR_CACHE=0: reuse the golden osq index. bruteN is not in the key.
  KNN_CLEAR_CACHE=0 LLOYD_BRUTE_N="$bn" ./run_knn_bench.sh 1 > "$OUT/run_1m_bruten_${bn}_${STAMP}.log" 2>&1
  echo "    exit=$?"
  grep "hammingKernel" "$OUT/run_1m_bruten_${bn}_${STAMP}.log" 2>/dev/null | grep -v "cellsScanned=0" | tail -1 | sed 's/^/    /'
  grep "^SUMMARY" "$OUT/run_1m_bruten_${bn}_${STAMP}.log" 2>/dev/null | sed 's/SUMMARY: //' | awk -v bn="$bn" -F'\t' '{printf "    bruteN=%s  recall=%s  lat=%s ms\n",bn,$1,$2}' | tee -a "$LOG"
done
echo "=== done $(date). logs: $OUT/run_1m_bruten_*_${STAMP}.log ===" | tee -a "$LOG"
