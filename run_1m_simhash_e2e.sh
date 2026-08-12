#!/usr/bin/env bash
# END-TO-END gate for ROTATION-SimHash rerank vs the 2/8 osq8 golden, 1M docs, golden config.
#
# Pipeline under test: 2-bit Gray Hamming coarse shortlist (unchanged) -> SimHash Hamming rerank over
# R*dim sign-bits, scored with the SIMD popcount kernel. Compared against the SAME config with osq8
# dense int8 rerank, which is the incumbent (~0.951 @ ~1.97 ms in the int8 golden JFR).
#
# THREE ARMS, all at np40/bn500 so only the rerank tier differs:
#   osq8  -- incumbent control, 1024 B/doc rerank record.
#   R=8   -- SimHash at 1024 B/doc: EQUAL BYTES to osq8, the honest head-to-head.
#   R=4   -- SimHash at 512 B/doc: half the bytes, tests the bytes/recall trade.
#
# KNN_CLEAR_CACHE=1 IS MANDATORY AND NOT OPTIONAL HERE. IVF_QUANTIZER is forwarded only as a JVM flag and
# is NOT part of the index cache key, so without a clear the simhash arms would silently reuse the osq8
# index -- the exact "flag not passed / wrong conclusion" trap that has produced bogus benchmarks.md
# entries twice. Each arm therefore pays a full ~1M build.
#
# Every arm asserts ENGAGEMENT from the child JVM's own command line before its number is reported: a
# missing -Divf.quantizer=simhash means the arm measured osq8 and the row is void.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
SUMMARY="$OUT/run_1m_simhash_e2e_${STAMP}_SUMMARY.txt"

# Golden config, shared by every arm (from run_1m_2c_golden_jfr.sh -- the current-best 2/8 baseline).
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BRUTE_N=500 LLOYD_BEAM_FACTOR=2
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 KNN_SKIP_MODEL_CHECK=1
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

echo "=== ROTATION-SimHash vs osq8 e2e, 1M golden (np40 bn500, nq=$KNN_NQUERY) $(date) ===" | tee "$SUMMARY"
echo "each arm rebuilds (KNN_CLEAR_CACHE=1): quantizer is NOT in the index key" | tee -a "$SUMMARY"
echo | tee -a "$SUMMARY"

run_arm() {
  local name="$1"; shift
  local log="$OUT/run_1m_simhash_e2e_${STAMP}_${name}.log"
  echo "--- arm=$name starting $(date +%H:%M:%S) ---" | tee -a "$SUMMARY"
  ( "$@" KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 1 ) > "$log" 2>&1
  local rc=$?

  # ENGAGEMENT: read the flags off the child JVM command line the harness echoed into the log.
  local engaged
  engaged="$(grep -oE '\-Divf\.quantizer=[a-z0-9]+|\-Divf\.simhashRounds=[0-9]+|\-Divf\.quantBits=[0-9]+' "$log" | sort -u | tr '\n' ' ')"
  local summary_line
  summary_line="$(grep '^SUMMARY' "$log" | tail -1 | sed 's/SUMMARY: //')"
  local recall lat
  recall="$(printf '%s' "$summary_line" | awk -F'\t' '{print $1}')"
  lat="$(printf '%s' "$summary_line" | awk -F'\t' '{print $2}')"

  printf '  arm=%-6s rc=%s flags=[%s]\n' "$name" "$rc" "$engaged" | tee -a "$SUMMARY"
  if [ -z "$summary_line" ]; then
    echo "  !! NO SUMMARY LINE -- arm FAILED, see $log" | tee -a "$SUMMARY"
    grep -iE "exception|error|OutOfMemory|Caused by" "$log" | head -5 | sed 's/^/     /' | tee -a "$SUMMARY"
  else
    printf '  arm=%-6s recall=%s latency=%s ms\n' "$name" "$recall" "$lat" | tee -a "$SUMMARY"
  fi
  echo "  log: $log" | tee -a "$SUMMARY"
  echo | tee -a "$SUMMARY"
}

# Arm 1: incumbent osq8 control.
run_arm osq8 env IVF_QUANTIZER=osq IVF_QUANT_BITS=8

# Arm 2: SimHash R=8 -> 8 bits/dim = 1024 B/doc, EQUAL to osq8.
run_arm R8 env IVF_QUANTIZER=simhash IVF_SIMHASH_ROUNDS=8

# Arm 3: SimHash R=4 -> 4 bits/dim = 512 B/doc, half of osq8.
run_arm R4 env IVF_QUANTIZER=simhash IVF_SIMHASH_ROUNDS=4

echo "=== RESULT TABLE ===" | tee -a "$SUMMARY"
printf '%-8s %-12s %-10s %s\n' "arm" "bytes/doc" "recall" "latency(ms)" | tee -a "$SUMMARY"
grep -E "^  arm=.* recall=" "$SUMMARY" | while read -r line; do
  a="$(printf '%s' "$line" | sed -n 's/.*arm=\([^ ]*\).*/\1/p')"
  r="$(printf '%s' "$line" | sed -n 's/.*recall=\([^ ]*\).*/\1/p')"
  l="$(printf '%s' "$line" | sed -n 's/.*latency=\([^ ]*\).*/\1/p')"
  case "$a" in osq8) b=1024 ;; R8) b=1024 ;; R4) b=512 ;; *) b="?" ;; esac
  printf '%-8s %-12s %-10s %s\n' "$a" "$b" "$r" "$l" | tee -a "$SUMMARY"
done
echo | tee -a "$SUMMARY"
echo "GATE: R8 must reach ~0.95 recall at <= osq8 latency to be worth pursuing. A large recall" | tee -a "$SUMMARY"
echo "shortfall at EQUAL bytes means sign-only coding is inherently weaker than int8 here." | tee -a "$SUMMARY"
echo "=== summary: $SUMMARY ===" | tee -a "$SUMMARY"
