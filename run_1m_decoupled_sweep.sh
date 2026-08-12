#!/usr/bin/env bash
# DECOUPLED residual4: equal-mass 2.0-bit coarse (lo plane freed to quartile threshold) + 5-bit signed-mag
# fine tier (sign+nibble, no lo read). The bet: sharper coarse -> smaller BRUTE_N at 0.95, net latency win,
# despite the fine tier dropping 6->5 bit. RISK GATE (from the clip sweep: fine-precision loss ceilinged
# clip=2.0 at 0.919) -- if 5-bit fine can't clear 0.95 at any N, the design is dead.
#
# ONE decoupled build (reindex), then a BRUTE_N sweep (search-time, reuses the build). Compare latency@0.95
# against the PAIRED baseline (clip=3.5: 0.950 @ 2.40 ms at N=750).
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
unset KNN_JFR

NS="${NS:-1000 750 500 350 250}"
first=1
for n in $NS; do
  log="$OUT/run_1m_decoupled_n${n}_${STAMP}.log"
  clr=0; [ "$first" = "1" ] && clr=1   # first N reindexes (decoupled build); rest reuse
  KNN_CLEAR_CACHE=$clr LLOYD_BRUTE_N=$n ./run_knn_bench.sh 3 > "$log" 2>&1
  first=0
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' \
    | awk -v n="$n" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "  decoupled N=%-5s recall=%s  MIN lat=%s ms\n",n,r,min}'
done
echo "=== done $(date) ==="
