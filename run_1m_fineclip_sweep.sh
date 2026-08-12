#!/usr/bin/env bash
# DECOUPLED residual4, fine-tier CLAMP sweep. The decoupled fine tier is 5-bit signed-magnitude with 16 |v|
# levels over [0, fineClip*std] -- a CRUDE (uniform) density match. |v| is half-normal, so the fidelity
# optimum is a tighter clamp than the coarse 3.5 (which wasted ~30% of range on the tail: that run got 0.939
# @ N=1000). Sweep fineClip at FIXED N=750 (the paired 0.95 point) to find the uniform-5-bit ceiling; if it
# clears ~0.95, we then sweep N. If it stalls ~0.94, the half-normal density-POLY (no LUT) is the recovery.
# fineClip is WRITE-time -> reindex per value.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=residual4 IVF_R4_DECOUPLED=1 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 KNN_TOPK=100 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_SKETCH_LO_CLIP=3.5 LLOYD_COARSE_BUCKET_DOT=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
export LLOYD_BRUTE_N="${LLOYD_BRUTE_N:-750}"
unset KNN_JFR

CLIPS="${CLIPS:-3.0 2.5 2.0 1.6}"
for fc in $CLIPS; do
  log="$OUT/run_1m_fineclip${fc}_${STAMP}.log"
  # fineClip is write-time -> every value reindexes.
  KNN_CLEAR_CACHE=1 IVF_R4_FINE_CLIP=$fc ./run_knn_bench.sh 3 > "$log" 2>&1
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' \
    | awk -v fc="$fc" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "  fineClip=%-4s N=%s recall=%s  MIN lat=%s ms\n",fc,ENVIRON["LLOYD_BRUTE_N"],r,min}'
done
echo "=== done $(date). Paired baseline @N=750: 0.950 @ 2.40ms ==="
