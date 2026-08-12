#!/usr/bin/env bash
# residual4 golden at the hamming1+osq8 geometry (spillBits=2, spillMargin=1.20) so the comparison is
# bytes-per-doc, not spill-count. Sweep nprobe {15,25,35,40} at bruteN {350,500,600,750}. spillBits/margin
# are INDEX-affecting (sp2 in the cache key), so the FIRST bruteN pass builds once (KNN_CLEAR_CACHE=1) and
# every later pass reuses that index (clear=0), since nprobe and bruteN are SEARCH-time. runs=3 medians.
# GOAL: find the smallest bruteN x nprobe that holds recall>=0.95 -- 2/4's sharper 2-bit coarse gate should
# let bruteN drop below where 1-bit hamming could, and at 512 B/doc rerank (half osq8's 1024) that is the
# sub-1ms lever. NO JFR -- profile the winner separately.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=residual4 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=15,25,35,40
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_SKETCH_LO_CLIP=3.5
export LLOYD_COARSE_BUCKET_DOT=1 LLOYD_NO_CARRY_BUCKET_DOT=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

BNS=(350 500 600 750)
FIRST=1
declare -a LOGS
for BN in "${BNS[@]}"; do
  export LLOYD_BRUTE_N=$BN
  LOG="$OUT/run_1m_r4_sp2_bn${BN}_${STAMP}.log"
  LOGS+=("$LOG")
  if [[ $FIRST == 1 ]]; then CLEAR=1; TAG="BUILD"; FIRST=0; else CLEAR=0; TAG="REUSE"; fi
  echo "=== r4 sp2 m1.20 npsweep bruteN=$BN ($TAG, clear=$CLEAR) $(date) ===" | tee "$LOG"
  KNN_CLEAR_CACHE=$CLEAR ./run_knn_bench.sh 3 >> "$LOG" 2>&1
  echo "bn$BN exit=$?"
done

echo; echo "=== RESULTS: recall / latency(ms) at nprobe {15,25,35,40} ==="
for L in "${LOGS[@]}"; do
  echo "--- $(basename "$L") ---"
  grep -E "will now reindex|reused" "$L" | head -1
  grep "^SUMMARY" "$L" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms visited=%s sel=%s\n",$1,$2,$16,$4}'
done
echo "=== logs: ${LOGS[*]} ==="
