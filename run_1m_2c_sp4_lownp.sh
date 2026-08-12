#!/usr/bin/env bash
# spill=4 at LOW nprobe -- the correct test of the ceilAudit "p95 NN in the 12th probed cell at sp4" claim.
# sp4 replicates each doc into more cells, so the true neighbors reach FEWER probed cells -> nprobe can drop
# FAR below sp2's 40. The early-session sp4 disaster held nprobe=40 (fat cells + same cell count = pure cost,
# 8-20ms); this sweeps nprobe {12,15,20,25} to exploit the routing win instead. osq8 2-bit coarse, bn500,
# margin 1.20. sp4 is a REBUILD (spillBits in cache key) -> clear once. Compare against the sp2 baseline:
# sp2/np55/bn500 = 0.951 @ 1.67ms; sp2/np40/bn500 = 0.945 @ 1.27ms. WIN = sp4 hits >=0.95 at LOWER latency.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_2c_sp4_lownp_${STAMP}.log"

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=4 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=12,15,20,25,40
export LLOYD_BRUTE_N=500
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

echo "=== sp4 low-nprobe @ bn500 (osq8 2-coarse) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 3 >> "$LOG" 2>&1
echo "exit=$?"

echo; echo "=== sp4 recall/lat vs nprobe (bn500) -- vs sp2 baseline np40=0.945/1.27 np55=0.951/1.67 ==="
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms visited=%s sel=%s\n",$1,$2,$16,$4}' | sort -u
echo "=== log: $LOG ==="
