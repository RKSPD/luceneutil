#!/usr/bin/env bash
# JFR the residual4 SEARCH path at the carry operating point: bucket-dot coarse ranker + plane carry, np40,
# BRUTE_N=1000. Reuses the cached index (KNN_CLEAR_CACHE controlled below) so the CPU profile is search, not
# build. High nquery so search dominates the whole-JVM JFR. Goal: see whether, WITH the carry live, rerank
# is still the pole and where (kernel arithmetic vs the remaining 520 B/doc record read vs coarse scan).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_r4_carry_jfr_${STAMP}.log"

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=residual4 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BRUTE_N=1000
export KNN_NQUERY="${KNN_NQUERY:-10000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_SKETCH_LO_CLIP=3.5
export LLOYD_COARSE_BUCKET_DOT=1 LLOYD_NO_CARRY_BUCKET_DOT=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
export KNN_JFR=1

# First run of this config must build the index against the freshly rebuilt reader jar (KNN_CLEAR_CACHE=1);
# override to 0 on repeat runs to reuse. Default 1 here to be safe after a codec change.
CLEAR="${KNN_CLEAR_CACHE:-1}"
echo "=== residual4 carry JFR (bucketdot+carry, np40, bruteN=1000, clear=$CLEAR) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE="$CLEAR" ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "exit=$?"
grep -E "will now reindex|reused|^SUMMARY" "$LOG" | head -3
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms\n",$1,$2}'
echo "=== done $(date). log: $LOG ; jfr in $OUT/logs/*.jfr ==="
ls -lat "$OUT"/logs/*.jfr 2>/dev/null | head -2
