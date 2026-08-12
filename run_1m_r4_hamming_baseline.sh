#!/usr/bin/env bash
# CORRECTED residual4 baseline: 2-bit HAMMING coarse (bucket-dot OFF -- the fast ~3.5 ns/doc path), sp2,
# margin 1.20 (matches the hamming1+osq8 geometry). This is the first CORRECT end-to-end number tonight --
# every earlier sweep carried LLOYD_COARSE_BUCKET_DOT=1, forcing the rejected 382 ns/doc float coarse path.
# Sweep nprobe {15,25,35,40} at bruteN {350,500,750}. First bruteN pass builds (sp2 in cache key); the rest
# reuse it. runs=3 medians. Float rerank (dotFused) -- the int8 kernel A/B comes after this establishes the
# split. Deliberately DOES NOT set LLOYD_COARSE_BUCKET_DOT / LLOYD_NO_CARRY_BUCKET_DOT.
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
# NB: NO LLOYD_COARSE_BUCKET_DOT (=> 2-bit Hamming coarse), NO LLOYD_NO_CARRY_BUCKET_DOT.
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

BNS=(350 500 750)
FIRST=1
declare -a LOGS
for BN in "${BNS[@]}"; do
  export LLOYD_BRUTE_N=$BN
  LOG="$OUT/run_1m_r4_ham_bn${BN}_${STAMP}.log"
  LOGS+=("$LOG")
  if [[ $FIRST == 1 ]]; then CLEAR=1; TAG="BUILD"; FIRST=0; else CLEAR=0; TAG="REUSE"; fi
  echo "=== r4 HAMMING baseline sp2 m1.20 bruteN=$BN ($TAG, clear=$CLEAR) $(date) ===" | tee "$LOG"
  KNN_CLEAR_CACHE=$CLEAR ./run_knn_bench.sh 3 >> "$LOG" 2>&1
  echo "bn$BN exit=$?"
done

echo; echo "=== RESULTS: recall / latency(ms) at nprobe {15,25,35,40} ==="
for L in "${LOGS[@]}"; do
  echo "--- $(basename "$L") ---"
  grep -E "will now reindex|reused" "$L" | head -1
  # prove the coarse path: sketchLoQueries / hammingKernel engaged, bucketDot NOT
  grep -E "hammingKernel|sketchLo\]|bucketDot" "$L" | head -2
  grep "^SUMMARY" "$L" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms visited=%s sel=%s\n",$1,$2,$16,$4}'
done
echo "=== logs: ${LOGS[*]} ==="
