#!/usr/bin/env bash
# PAIRED residual4 (decoupled rejected). Find the TIGHTEST shared clip that still holds 0.95 recall while
# SHARPENING the coarse scan. clip trades coarse bucket quality (tighter = closer to equal-mass = better
# spill/ranking) against fine-tier tail fidelity (tighter = more clamp = lower recall). Known: clip=3.5 ->
# 0.950 @ N=750; clip=2.0 -> 0.912 (ceiling). Balance point is in 2.5-3.2, unmeasured.
#
# Each clip, TWO search arms on ONE build (both search-time): FULL (does end-to-end hold 0.95?) and
# BUCKETONLY (-Dlloyd.rerankBucketOnly, recall from the 2-bit COARSE alone = coarse-quality indicator).
# Rising bucket-only recall as clip drops = coarse is genuinely sharpening. clip is write-time -> reindex.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=residual4 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 KNN_TOPK=100 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_COARSE_BUCKET_DOT=0 LLOYD_BRUTE_N=750
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
unset KNN_JFR

CLIPS="${CLIPS:-3.5 3.2 3.0 2.75 2.5}"
for clip in $CLIPS; do
  # FULL arm rebuilds (clip changed); BUCKETONLY arm reuses that build (search-time flag only).
  fl="$OUT/run_1m_balclip${clip}_full_${STAMP}.log"
  bl="$OUT/run_1m_balclip${clip}_bucket_${STAMP}.log"
  KNN_CLEAR_CACHE=1 LLOYD_SKETCH_LO_CLIP=$clip LLOYD_RERANK_BUCKET_ONLY=0 ./run_knn_bench.sh 3 > "$fl" 2>&1
  KNN_CLEAR_CACHE=0 LLOYD_SKETCH_LO_CLIP=$clip LLOYD_RERANK_BUCKET_ONLY=1 ./run_knn_bench.sh 1 > "$bl" 2>&1
  fr=$(grep "^SUMMARY" "$fl"|sed 's/SUMMARY: //'|awk -F'\t' '{if(min==""||$2<min){min=$2;rc=$1}}END{print rc" "min}')
  br=$(grep "^SUMMARY" "$bl"|sed 's/SUMMARY: //'|awk -F'\t' '{print $1}'|head -1)
  printf "  clip=%-4s FULL recall=%s lat=%sms | COARSE-only recall=%s\n" \
    "$clip" "$(echo $fr|cut -d' ' -f1)" "$(echo $fr|cut -d' ' -f2)" "$br"
done
echo "=== done $(date) ==="
