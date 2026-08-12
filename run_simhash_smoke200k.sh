#!/usr/bin/env bash
# Cheap WIRING smoke for rotation-SimHash before committing to three 1M builds: 200k docs, R=8.
# Proves the encode -> persist -> coarse-shortlist -> Hamming-rerank path runs and returns sane recall.
# NOT a perf number (200k, different nlist). Compares against an osq8 arm at the SAME scale so a
# recall collapse is visible rather than inferred.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
SUMMARY="$OUT/run_simhash_smoke200k_${STAMP}_SUMMARY.txt"

export KNN_NDOC="${KNN_NDOC:-200000}" KNN_NLIST=1000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BRUTE_N=750 LLOYD_BEAM_FACTOR=2
export KNN_NQUERY="${KNN_NQUERY:-500}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 KNN_SKIP_MODEL_CHECK=1
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

echo "=== simhash 200k SMOKE (R=8 vs osq8, np40 bn750) $(date) ===" | tee "$SUMMARY"

arm() {
  local name="$1"; shift
  local log="$OUT/run_simhash_smoke200k_${STAMP}_${name}.log"
  echo "--- $name $(date +%H:%M:%S) ---" | tee -a "$SUMMARY"
  ( "$@" KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 1 ) > "$log" 2>&1
  local rc=$?
  local flags; flags="$(grep -oE '\-Divf\.quantizer=[a-z0-9]+|\-Divf\.simhashRounds=[0-9]+' "$log" | sort -u | tr '\n' ' ')"
  local s; s="$(grep '^SUMMARY' "$log" | tail -1 | sed 's/SUMMARY: //')"
  printf '  %s rc=%s flags=[%s]\n' "$name" "$rc" "$flags" | tee -a "$SUMMARY"
  if [ -z "$s" ]; then
    echo "  !! NO SUMMARY -- failed. First errors:" | tee -a "$SUMMARY"
    grep -iE "exception|error|Caused by|assert" "$log" | head -8 | sed 's/^/     /' | tee -a "$SUMMARY"
  else
    printf '  %s recall=%s lat=%s ms\n' "$name" \
      "$(printf '%s' "$s" | awk -F'\t' '{print $1}')" \
      "$(printf '%s' "$s" | awk -F'\t' '{print $2}')" | tee -a "$SUMMARY"
  fi
  echo "  log: $log" | tee -a "$SUMMARY"
}

arm osq8 env IVF_QUANTIZER=osq IVF_QUANT_BITS=8
arm R8 env IVF_QUANTIZER=simhash IVF_SIMHASH_ROUNDS=8

echo "=== summary: $SUMMARY ===" | tee -a "$SUMMARY"
