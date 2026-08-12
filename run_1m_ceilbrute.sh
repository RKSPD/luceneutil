#!/usr/bin/env bash
# ceilBrute audit: how much recall does the 1-bit sketch SHORTLISTING lose, and what N recovers it?
# Drops routing entirely -- full-corpus 1-bit Hamming scan -> shortlist top-N -> int8 rerank -> recall@100
# vs the codec's OWN full-scan int8 top-100 (isolates the sketch loss, not the quantizer loss). Sweeps
# N = {200,500,1000,2000,5000,10000}, bracketing the current bruteN=2000. Answers directly: can we rerank
# FEWER than 2000 and keep recall? Diagnostic path (-Dlloyd.ceilAudit + -Dlloyd.ceilBrute), NO reindex of
# its own, but the last width-sweep left the 1M cache at sketchDims=768 -- so REBUILD at 1024 first.
# Heavy: exact-scans all 1M records PER QUERY, so nquery is small.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY="${KNN_NQUERY:-100}" KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
IDXDIR="${LUCENEUTIL_DIR:-/local/home/rikhil/vectordb/luceneutil}/knn-reuse/indices"
LOG="$OUT/run_1m_ceilbrute_${STAMP}.log"
echo "=== ceilBrute (rebuild 1M @ sketchDims=1024, nquery=$KNN_NQUERY) $(date) ===" | tee "$LOG"
# Force a rebuild at 1024 (cache does not encode sketchDims; last sweep left 768).
find "$IDXDIR" -maxdepth 1 -type d -name "*-1000000-lloydivf-*" -exec rm -rf {} + 2>/dev/null
KNN_CLEAR_CACHE=0 JAVA_TOOL_OPTIONS="-Dlloyd.ceilAudit=true -Dlloyd.ceilBrute=true" ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "=== done $(date). log: $LOG ===" | tee -a "$LOG"
grep -E "brute sign-.*rerank N=" "$LOG"
