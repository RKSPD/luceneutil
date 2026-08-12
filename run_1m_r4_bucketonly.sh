#!/usr/bin/env bash
# DIAGNOSTIC: residual4 rerank scored by the 2-bit BUCKET term ALONE (LLOYD_RERANK_BUCKET_ONLY=1), dropping
# the 4-bit nibble. Measures (a) the recall the 2-bit tier holds without the fine tier, and (b) the latency
# FLOOR when rerank skips the 512 B nibble read (reads planes only, 256 B/doc). Same 2-bit Hamming coarse
# (bucket-dot OFF), sp2/m1.20 geometry, reusing the baseline index. Sweep np {25,40} x bruteN {500,750,1000}.
# NB: bucket-only is a WRITE-time-neutral read-path flag; the index is identical, so KNN_CLEAR_CACHE=0 reuses
# the baseline build.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=residual4 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=25,40
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_SKETCH_LO_CLIP=3.5
export LLOYD_RERANK_BUCKET_ONLY=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

BNS=(500 750 1000)
declare -a LOGS
for BN in "${BNS[@]}"; do
  export LLOYD_BRUTE_N=$BN
  LOG="$OUT/run_1m_r4_bucketonly_bn${BN}_${STAMP}.log"
  LOGS+=("$LOG")
  echo "=== r4 BUCKET-ONLY sp2 m1.20 bruteN=$BN (reuse baseline, clear=0) $(date) ===" | tee "$LOG"
  KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 3 >> "$LOG" 2>&1
  echo "bn$BN exit=$?"
done

echo; echo "=== BUCKET-ONLY RESULTS: recall / latency(ms) at nprobe {25,40} ==="
for L in "${LOGS[@]}"; do
  echo "--- $(basename "$L") ---"
  grep -E "will now reindex|reused" "$L" | head -1
  grep "^SUMMARY" "$L" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms sel=%s\n",$1,$2,$4}' | sort -u
done
echo "=== logs: ${LOGS[*]} ==="
