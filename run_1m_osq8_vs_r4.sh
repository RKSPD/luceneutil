#!/usr/bin/env bash
# HEAD-TO-HEAD at matched recall (~0.95), each at its OWN golden operating point (different on-disk formats,
# different cache keys -> each builds its own index):
#   arm A = osq8 (the old 1/8 setup: 1-bit sign Hamming coarse + dense int8 rerank), margin1.25, bn1750, np40.
#   arm B = residual4 (2-bit Gray Hamming coarse + 4-bit nibble rerank, INT8 QUERY), margin1.20, bn750, np40.
# Decides whether 2/4 (compact storage, reconstruction ALU) beats 1/8 (dense bytes, trivial kernel) on this
# Graviton3 warm. runs=3 medians. Each arm CLEARs its own build once (safe: distinct format keys).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

# ---- shared ----
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2
export IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

# ---- arm A: osq8 (old 1/8) ----
unset IVF_QUANTIZER LLOYD_SKETCH_LO_BITS LLOYD_SKETCH_LO_CLIP LLOYD_INT8_QUERY LLOYD_SKETCH_DIMS
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_SPILL_MARGIN=1.25 LLOYD_BRUTE_N=1750
LOGA="$OUT/run_1m_h2h_osq8_${STAMP}.log"
echo "=== arm A: osq8 1/8 (margin1.25 bn1750 np40) $(date) ===" | tee "$LOGA"
KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 3 >> "$LOGA" 2>&1
echo "osq8 exit=$?"

# ---- arm B: residual4 int8 (2/4) ----
export IVF_QUANTIZER=residual4 IVF_SPILL_MARGIN=1.20 LLOYD_BRUTE_N=750
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_SKETCH_LO_CLIP=3.5 LLOYD_INT8_QUERY=1
unset IVF_QUANT_BITS
LOGB="$OUT/run_1m_h2h_r4int8_${STAMP}.log"
echo "=== arm B: residual4 int8 2/4 (margin1.20 bn750 np40) $(date) ===" | tee "$LOGB"
KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 3 >> "$LOGB" 2>&1
echo "r4 exit=$?"

echo; echo "=== HEAD-TO-HEAD (np40, matched ~0.95) ==="
for L in "$LOGA" "$LOGB"; do
  echo "--- $(basename "$L") ---"
  grep -oE "quantizer=[a-z0-9]+|quantBits=[0-9]+|int8Query=true|bruteN=[0-9]+" "$L" | sort -u | tr '\n' ' '; echo
  grep "^SUMMARY" "$L" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms sel=%s\n",$1,$2,$4}' | sort -u
done
echo "=== logs: $LOGA  $LOGB ==="
