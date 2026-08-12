#!/usr/bin/env bash
# EXCELSIXOR vs osq8, 1M golden config, with JFR on the excelsixor arm.
#
# THE CLAIM UNDER TEST. excelsixor stores b=4 bits/dim as bit planes (513 B/doc incl. the norm scalar) and
# scores with XOR/AND + popcount ONLY -- no multiply-accumulate against doc bytes. osq8 stores 1024 B/doc and
# scores with a VNNI int8 dot. Offline (rerank inside an exact top-1000 shortlist) excelsixor measured 0.9557
# vs osq8's 0.9857, and it BEAT the byte-matched MAC baseline (osq4 at 0.9489).
#
# So the honest expectations are:
#   - FOOTPRINT: 513 vs 1152 B/doc (osq8 code + coarse filter) = 2.2x smaller. This is the real prize.
#   - LATENCY: the rerank scan streams 2x fewer bytes, but does MORE instructions (8 doc planes x 4 query
#     planes = 32 popcount terms per 64-bit word vs ~1 VNNI instruction per 64 B). Whether that nets out
#     faster depends on how memory-bound the scan actually is -- which is what the JFR is for.
#   - RECALL: expect slightly BELOW osq8. If it lands far below the offline 0.9557, suspect the pipeline
#     (cell density differs from the probe's, and e2e adds coarse-tier loss), not the estimator.
#
# KNN_CLEAR_CACHE=1 is MANDATORY on each arm's first run: the quantizer is NOT part of the index cache key,
# so without it the second arm silently reuses the first arm's index. Engagement is asserted from the child
# JVM's own command line, and a missing flag ABORTS rather than reporting a void row.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
S="$OUT/run_1m_excelsixor_${STAMP}_SUMMARY.txt"

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BEAM_FACTOR=2
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}" KNN_SKIP_MODEL_CHECK=1
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

echo "=== excelsixor(b=4, 513 B) vs osq8(1024 B), 1M golden np40 nq=$KNN_NQUERY $(date) ===" | tee "$S"
echo | tee -a "$S"

arm() {
  local name="$1" quant="$2" jfr="$3"; shift 3
  local first=1
  echo "--- arm=$name ($quant) ---" | tee -a "$S"
  for BN in 500 350 250; do
    local log="$OUT/run_1m_excelsixor_${STAMP}_${name}_bn${BN}.log"
    local cc=0; [ $first -eq 1 ] && cc=1
    # JFR only on the first (largest bn) point, so the profile is one clean run.
    local usejfr=0; [ $first -eq 1 ] && usejfr=$jfr
    first=0
    env IVF_QUANTIZER="$quant" \
        ${quant:+IVF_EXCELSIXOR_BITS=4} \
        $([ "$quant" = "osq" ] && echo "IVF_QUANT_BITS=8") \
        KNN_JFR=$usejfr LLOYD_BRUTE_N=$BN KNN_CLEAR_CACHE=$cc \
        ./run_knn_bench.sh 1 > "$log" 2>&1
    local flags; flags="$(grep -oE '\-Divf\.quantizer=[a-z0-9]+|\-Divf\.excelsixorBits=[0-9]+' "$log" | sort -u | tr '\n' ' ')"
    if [ -z "$flags" ]; then
      echo "  !! ABORT: bn=$BN passed no -Divf.quantizer -- row VOID" | tee -a "$S"
      return 1
    fi
    local x; x="$(grep '^SUMMARY' "$log" | tail -1 | sed 's/SUMMARY: //')"
    if [ -z "$x" ]; then
      printf '  bn=%-5s FAILED\n' "$BN" | tee -a "$S"
      grep -iE "exception|error|Caused by" "$log" | head -4 | sed 's/^/     /' | tee -a "$S"
    else
      printf '  bn=%-5s recall=%s ceil=%s lat=%s ms  index=%s MB  flags=[%s]\n' "$BN" \
        "$(printf '%s' "$x" | awk -F'\t' '{print $1}')" \
        "$(printf '%s' "$x" | awk -F'\t' '{print $4}')" \
        "$(printf '%s' "$x" | awk -F'\t' '{print $2}')" \
        "$(printf '%s' "$x" | awk -F'\t' '{print $18}')" "$flags" | tee -a "$S"
    fi
  done
  echo | tee -a "$S"
}

arm osq8 osq 0
arm excelsixor excelsixor 1

echo "=== JFR (excelsixor arm) ===" | tee -a "$S"
ls -lat "$OUT"/logs/*.jfr 2>/dev/null | head -2 | tee -a "$S"
echo "=== summary: $S ===" | tee -a "$S"
