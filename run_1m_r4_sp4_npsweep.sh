#!/usr/bin/env bash
# residual4 golden re-tune: spillBits=4, spillMargin=1.15. Sweep nprobe {15,25,35,40} at bruteN 600 and
# 750. spillBits/margin are INDEX-affecting (sp4 is in the cache key), so the first pass builds once with
# KNN_CLEAR_CACHE=1; nprobe and bruteN are SEARCH-time (not in the key), so the np sweep runs in one JVM
# over one index, and the bruteN=750 pass reuses that same index (clear=0). runs=3 for stable medians.
# NO JFR here -- profile the winning operating point separately once recall/latency is known.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=4 IVF_SPILL_MARGIN=1.15
export IVF_QUANTIZER=residual4 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=15,25,35,40
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_SKETCH_LO_CLIP=3.5
export LLOYD_COARSE_BUCKET_DOT=1 LLOYD_NO_CARRY_BUCKET_DOT=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

# --- pass 1: bruteN=600, build the sp4/m1.15 index fresh (clip 3.2 rerun left the shared key stale) ---
export LLOYD_BRUTE_N=600
LOG600="$OUT/run_1m_r4_sp4_bn600_${STAMP}.log"
echo "=== r4 sp4 m1.15 npsweep bruteN=600 (BUILD, clear=1) $(date) ===" | tee "$LOG600"
KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 3 >> "$LOG600" 2>&1
echo "bn600 exit=$?"

# --- pass 2: bruteN=750, REUSE the index just built (clear=0) ---
export LLOYD_BRUTE_N=750
LOG750="$OUT/run_1m_r4_sp4_bn750_${STAMP}.log"
echo "=== r4 sp4 m1.15 npsweep bruteN=750 (REUSE, clear=0) $(date) ===" | tee "$LOG750"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 3 >> "$LOG750" 2>&1
echo "bn750 exit=$?"

echo; echo "=== RESULTS (recall / latency ms per nprobe) ==="
for L in "$LOG600" "$LOG750"; do
  echo "--- $(basename "$L") ---"
  grep -E "will now reindex|reused" "$L" | head -1
  grep "^SUMMARY" "$L" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms visited=%s sel=%s\n",$1,$2,$16,$4}'
done
echo "=== logs: $LOG600  $LOG750 ==="
