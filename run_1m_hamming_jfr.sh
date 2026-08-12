#!/usr/bin/env bash
# JFR the HAMMING-coarse residual4 search path (bucket-dot OFF) at the low-latency operating point:
# topK=100, nquery=1000, cached index reused. This is arm A (the fast config: 0.957 @ 3.0 ms). Goal: the
# REAL pole of the Hamming path -- coarse Hamming scan vs residual4 rerank vs heap/dedup -- now that the
# bucket-dot experiment is retired. nquery kept at 1000 (not 10000) so the exact-NN ground-truth pass does
# not swamp the profile with DocsFileNNTask samples the way the 10k run did.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_hamming_jfr_${STAMP}.log"

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=residual4 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BRUTE_N=1000 KNN_TOPK=100 KNN_NQUERY=1000
export KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_SKETCH_LO_CLIP=3.5
export LLOYD_COARSE_BUCKET_DOT=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
export KNN_JFR=1

echo "=== HAMMING-coarse residual4 JFR (np40, bruteN=1000, topK=100, nq=1000, cached) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "exit=$?"
grep -E "will now reindex|reused|^SUMMARY" "$LOG" | head -3
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms\n",$1,$2}'
echo "=== done $(date). log: $LOG ==="
