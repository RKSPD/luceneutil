#!/usr/bin/env bash
#
# 1M A/B of the ONE-GRAPH-PER-SEGMENT change (GraphCarry in LloydIVFVectorsWriter).
#
# WHAT CHANGED: previously every routing stage of a write called CentroidHnsw.build() whenever donorGraph
# was null -- which is ALWAYS true on a first-time build -- so one segment paid ~6-7 full
# O(nlist*log nlist*dim) HNSW builds (sample-train router, assign #2 per refine, spill router, persisted
# graph, and once inside graphRoutedAssign). Now the FIRST build is carried and every later stage calls
# withRefreshedCodes on that same adjacency: O(nlist*dim) re-quantize instead. The adjacency is
# position-free node-id lists, so it stays a valid navigable graph as Lloyd moves the centroids; only the
# int8 scoring codes go stale.
#
# WHY 1M FIRST: at 40M this costs ~4.7 h per arm; at 1M it is minutes, and the question is a YES/NO
# (did recall move?) that does not need 40M to answer. The 40M rerun is only worth paying once this holds.
#
# THE RISK BEING TESTED: this changes CLUSTERING, not just speed. A refreshed-codes graph routes slightly
# differently than a freshly-built one, so doc assignment -- and therefore recall -- can move. That is
# exactly why -Divf.reuseGraphTopology=false exists as the control arm.
#
# READ THE RESULT AS:
#   index(s)  CARRY << CONTROL  => the build-elimination worked
#   recall    CARRY ~= CONTROL  => topology reuse is safe (the codec's stated design assumption holds)
#   recall    CARRY <  CONTROL  => the assumption does NOT hold at this nlist; do not ship it
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb

# 1M reference config (matches the prior 1M spill5 runs: nlist=2000, ~500 docs/cell).
export KNN_NDOC=1000000
export KNN_NLIST="${KNN_NLIST:-2000}"
export KNN_SPILL_BITS=5
export IVF_SPILL_MARGIN=1.15
export IVF_QUANTIZER=blocksphere
export IVF_BLOCK_P=2
export IVF_QUANT_BITS=8
export IVF_BEAM_SPILL=1
export IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32
export IVF_CENTROID_HNSW_BEAM_WIDTH=64
export IVF_STREAM_REFINE_ITERS=3
# Force the STREAMING write path at 1M so this exercises the same code the 40M build uses (otherwise a
# 1M index is small enough to take the buffered path and the A/B would test different code).
export IVF_STREAM_FLUSH_MIN_DOCS=100000
export IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE="${KNN_NPROBE:-40}"
export KNN_NQUERY="${KNN_NQUERY:-1000}"
export KNN_SKIP_SMELL=1
export KNN_HEAP="${KNN_HEAP:-16g}"
# WARM search: this A/B is about INDEXING cost + recall, not I/O. Async stages off for the same reason.
export KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0
export LLOYD_URING_RERANK=0
export LLOYD_URING_PIPELINE=0
export LLOYD_URING_RERANK_PIPELINE=0
export LLOYD_PREFETCH_CELLS=0

./ram_balloon.sh --release >/dev/null 2>&1 || true

run_arm() {
  local name="$1"; shift
  local log="$OUT/run_1m_graphcarry_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  # MUST clear: reuseGraphTopology is WRITE-TIME and NOT in the index key, so without clearing, arm 2
  # would silently reuse arm 1's index and measure nothing (the trap benchmarks.md 13 documents).
  KNN_CLEAR_CACHE=1 env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  echo "    exit=$?"
  grep -E "NOTE: (index\(s\)|index_docs|force_merge|merge\(s\))" "$log" 2>/dev/null | sed 's/^/    /'
  grep "^SUMMARY" "$log" 2>/dev/null | awk -F'\t' '{printf "    recall=%s  lat=%s ms\n",$1,$2}' | sed 's/SUMMARY: //'
}

# CONTROL: rebuild the HNSW at every routing stage (pre-change behaviour).
run_arm CONTROL IVF_REUSE_GRAPH_TOPOLOGY=0
# CARRY: one graph per segment, refreshed codes thereafter (the change under test; codec default).
run_arm CARRY   IVF_REUSE_GRAPH_TOPOLOGY=1

echo "=== done $(date). logs: $OUT/run_1m_graphcarry_*_${STAMP}.log ==="
