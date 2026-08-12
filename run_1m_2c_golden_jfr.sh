#!/usr/bin/env bash
# JFR the CURRENT-BEST 2/8 golden: 2-bit Gray Hamming coarse (fixed shared threshold) + osq8 dense int8 rerank
# + beamFactor=2 routing, sp2/m1.20, np40, bn500 (~0.948). nquery=1000 so the exact-NN scaffolding stays
# small (nquery=10000 made DocsFileNNTask.dotProduct 82% of a prior JFR). Reuses the cached sp2 index (no
# build) so the profile is pure search. GOAL: 1/8 hit 0.985ms at bn1750; 2/8 uses ~500 rerank docs (1/3) yet
# sits ~1.3ms -- find what became the pole now that rerank shrank (coarse scan? routing? heap? record reads?).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_2c_golden_jfr_${STAMP}.log"

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BRUTE_N=500 LLOYD_BEAM_FACTOR=2
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
export KNN_JFR=1

echo "=== 2/8 golden JFR (2-bit coarse + osq8 + bf2, np40 bn500, nq=$KNN_NQUERY) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "exit=$?"
grep -E "will now reindex|reused|^SUMMARY" "$LOG" | head -3
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms\n",$1,$2}'
echo "=== jfr ==="; ls -lat "$OUT"/logs/*.jfr 2>/dev/null | head -2
echo "=== log: $LOG ==="
