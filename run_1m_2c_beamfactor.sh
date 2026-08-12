#!/usr/bin/env bash
# Lever B: BETTER ROUTING GRAPH without rebuild or cell inflation. beamFactor widens the HNSW graph beam
# (efSearch = ceil(nprobe*beamFactor)) so the top-nprobe cells actually scanned are a MORE ACCURATE nearest
# set -- pure coarse-select cost, no posting-scan inflation (unlike sp4 which fattens cells). routeAudit
# logs the graph's overlap vs EXACT nearest-nprobe (1.0 = optimal routing; <1.0 = recall left on the table).
# sp2 osq8 2-coarse, bn500, np40 fixed. Reuses cached sp2 index (search-time flags). Compare to sp2/np40/
# bn500 = 0.945/1.27ms and sp4/np40 = 0.951/1.79ms. WIN = beamFactor lifts recall toward 0.95 near 1.27ms.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BRUTE_N=500
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

# One routeAudit run first (beamFactor=1, baseline graph) to quantify the routing gap, then sweep beamFactor.
for BF in 1 2 4 8; do
  LOG="$OUT/run_1m_2c_bf${BF}_${STAMP}.log"
  export LLOYD_BEAM_FACTOR=$BF
  if [[ $BF == 1 ]]; then export LLOYD_ROUTE_AUDIT=1; else unset LLOYD_ROUTE_AUDIT; fi
  echo "=== beamFactor=$BF $( [[ $BF == 1 ]] && echo '+routeAudit') $(date) ===" | tee "$LOG"
  KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 3 >> "$LOG" 2>&1
  echo "bf$BF exit=$?"
  grep -iE "routeAudit|routing overlap|mean overlap|route" "$LOG" | grep -iv "cmd:" | head -3
  grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  bf'"$BF"': recall=%s lat=%s ms\n",$1,$2}' | sort -u | head -1
done
echo "=== stamp: $STAMP ==="
