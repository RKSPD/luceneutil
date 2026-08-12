#!/usr/bin/env bash
# margin=1.2 (sb2): does MORE adaptive spill lift the recall ceiling past m1.15 (0.949-0.957 @ np50), or tip
# into over-spill regression (like sb3+m1.15 did)? Build once, then sweep nprobe on the same index.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
FIRST=1
run_pt() {
  local np="$1" bn="$2"; local log="$OUT/run_1m_m120_np${np}_bn${bn}_${STAMP}.log"
  local clear=0; [ "$FIRST" = "1" ] && clear=1 && FIRST=0
  KNN_NPROBE="$np" KNN_CLEAR_CACHE="$clear" LLOYD_BRUTE_N="$bn" ./run_knn_bench.sh 3 > "$log" 2>&1
  local ix=""; [ "$clear" = "1" ] && ix=$(grep -iE "reindex takes" "$log"|tail -1|grep -oE "[0-9]+\.[0-9]+ sec"|head -1)
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v np="$np" -v b="$bn" -v ix="$ix" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "  nprobe=%-3s bruteN=%-4s recall=%s MIN lat=%s ms  %s\n",np,b,r,min,ix}'
}
run_pt 50 2000
run_pt 40 2000
run_pt 40 1750
echo "=== done $(date) ==="
