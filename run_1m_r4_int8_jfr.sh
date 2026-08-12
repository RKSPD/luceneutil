#!/usr/bin/env bash
# JFR the CURRENT BEST residual4 search: int8 query + 2-bit Hamming coarse (bucket-dot OFF), R4_TILE=1,
# sp2/m1.20, np40, bruteN=750 -- the 0.951 @ 1.77ms operating point. High nquery so the SEARCH phase (not
# reader-open) dominates the whole-JVM CPUTime profile. Reuses the cached baseline index (no build). Goal:
# see where the 1.77ms now goes -- nibble unpack vs DRAM record read vs coarse Hamming -- to decide whether
# dim-major storage (Option B) is worth the writer change.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_r4_int8_jfr_${STAMP}.log"

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=residual4 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BRUTE_N=750
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_SKETCH_LO_CLIP=3.5
export LLOYD_INT8_QUERY=1 LLOYD_R4_TILE=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
export KNN_JFR=1

echo "=== int8 golden JFR (bucket-dot OFF, T1, np40, bn750, nquery=$KNN_NQUERY) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "exit=$?"
grep -E "will now reindex|reused|^SUMMARY" "$LOG" | head -3
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms\n",$1,$2}'
echo "=== jfr files ==="; ls -lat "$OUT"/tool-logs/*.jfr 2>/dev/null | head -3
echo "=== log: $LOG ==="
