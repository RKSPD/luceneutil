#!/usr/bin/env bash
# Recover low-nprobe recall via higher SPILL MARGIN: more spill => each doc reachable in more probed cells =>
# recall back at low nprobe (where bf2 hit 0.77ms @ 0.925). margin is WRITE-time (rebuild per arm). Arms:
# 1.20 (current baseline) and 1.25. 2/8 golden (fused coarse + osq8 + bf2), sp2, bn500. Sweep np {20,25,30,40}
# per margin so we see the whole curve shift. First arm per margin clears cache (margin in the build); the
# np sweep within a margin reuses. Baseline (m1.20): np20=0.925/0.77, np30=0.941/0.97, np40=0.948/1.16.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=20,25,30,40 LLOYD_BRUTE_N=500 LLOYD_BEAM_FACTOR=2
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

for M in 1.20 1.25; do
  export IVF_SPILL_MARGIN=$M
  LOG="$OUT/run_1m_2c_m${M}_${STAMP}.log"
  echo "=== spillMargin=$M (rebuild) $(date) ===" | tee "$LOG"
  KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 3 >> "$LOG" 2>&1
  echo "m$M exit=$?"
done

echo; echo "=== recall/lat vs nprobe, by spill margin (bn500, bf2) ==="
for M in 1.20 1.25; do
  echo "-- margin $M --"
  L=$(ls -t "$OUT"/run_1m_2c_m${M}_${STAMP}.log 2>/dev/null | head -1)
  grep "^SUMMARY" "$L" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms visited=%s sel=%s\n",$1,$2,$16,$4}' | sort -u
done
echo "=== stamp: $STAMP ==="
