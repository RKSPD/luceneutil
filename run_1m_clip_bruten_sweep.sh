#!/usr/bin/env bash
# GOAL: minimize latency @ recall=0.95. Lever: coarse 2-bit bucket MASS, tuned by the lo-plane clip (still
# equal-WIDTH/affine -- lowering clip moves the |v|<clip*std/2 threshold toward the quartile, so the 4
# buckets approach equal MASS = 2.0 bits instead of 1.55, WITHOUT the non-affine equal-mass reconstruction).
# A sharper coarse ranker should let BRUTE_N (the rerank candidate count = the ~50% rerank pole) SHRINK at
# equal recall.
#
#   clip=3.5 : today's default, threshold 1.75 std, masses ~[.04,.46,.46,.04] = 1.55 bits
#   clip=2.0 : threshold 1.00 std, masses ~[.16,.34,.34,.16]
#   clip=1.35: threshold 0.67 std ~= QUARTILE, masses ~[.25,.25,.25,.25] = 2.0 bits (equal-mass, affine)
#
# clip is WRITE-time (reindex per value, NOT in cache key -> KNN_CLEAR_CACHE=1). BRUTE_N is SEARCH-time
# (one build sweeps all N). At each clip we sweep N and read off the smallest N clearing 0.95 and its latency.
# Golden low-latency config: Hamming coarse, topK=100, nquery=1000, no JFR.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=residual4 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 KNN_TOPK=100 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_COARSE_BUCKET_DOT=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
unset KNN_JFR

CLIPS="${CLIPS:-3.5 2.0 1.35}"
NS="${NS:-2000 1500 1000 750 500 350}"

for clip in $CLIPS; do
  echo "############ clip=$clip (reindex) $(date) ############"
  first=1
  for n in $NS; do
    log="$OUT/run_1m_clip${clip}_n${n}_${STAMP}.log"
    # First N at this clip rebuilds (clip changed -> clear cache); rest reuse that build.
    clr=0; [ "$first" = "1" ] && clr=1
    KNN_CLEAR_CACHE=$clr LLOYD_SKETCH_LO_CLIP=$clip LLOYD_BRUTE_N=$n ./run_knn_bench.sh 3 > "$log" 2>&1
    first=0
    grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' \
      | awk -v c="$clip" -v n="$n" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "  clip=%s N=%-5s recall=%s  MIN lat=%s ms\n",c,n,r,min}'
  done
done
echo "=== done $(date) ==="
