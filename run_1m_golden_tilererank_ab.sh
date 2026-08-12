#!/usr/bin/env bash
#
# A/B of the TILED int8 rerank against the per-candidate rerank, on the §16.4 GOLDEN 1M config.
#
# WHY: JFR (knn-perf-test-0.jfr, 2026-08-06, warm) put rerankCands at 42.9% of sketchScanCells CPU vs
# bulkHamming's 37.4% -- the rerank has OVERTAKEN the coarse Hamming as the pole. The coarse kernel is
# within 1-5% of its pure load+XOR bandwidth floor (measured), so there is nothing left there; the rerank
# was still dotting ONE candidate at a time, re-widening the query for each of the ~bruteN candidates.
# BulkDotKernel.bulkDotAt shares the query load+ZERO_EXTEND across 4 records. Isolated: 126.6 -> 107.6
# ns/candidate (1.18x) at dim=1024/n=1750.
#
# Config is a faithful copy of run_1m_golden_simd_ab.sh (the golden setup): nlist=2000, spillBits=2, BEAM
# SPILL on, spillMargin left at the codec default 1.10, osq int8, soarLambda=1.0, flushIters=5, nprobe=40,
# force-merged. Both arms share ONE cached index -- the tile is a read-time scoring choice and is not in the
# index key -- so the only difference between the arms is the flag.
#
# Correctness is NOT established here: TestTiledRerankEquivalence pins identical top-k (docIds AND scores)
# for both paths on one index in one JVM. This script measures latency only. Recall is printed as a
# guard -- the arms MUST agree to ~0.001 (they are bit-identical), and a real gap means a wiring bug.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000
export KNN_NLIST=2000
export KNN_SPILL_BITS=2
# spillMargin DELIBERATELY UNSET -- §16.4 pinned no margin, so the codec default (1.10) applies. Setting it
# would change clustering and this would no longer be the golden index.
export IVF_QUANTIZER=osq
export IVF_QUANT_BITS=8
export IVF_BEAM_SPILL=1
export IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32
export IVF_CENTROID_HNSW_BEAM_WIDTH=64
export IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000
export IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40
export KNN_NQUERY="${KNN_NQUERY:-1000}"
export KNN_SKIP_SMELL=1
export KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1
export KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0
export LLOYD_URING_RERANK=0
export LLOYD_URING_PIPELINE=0
export LLOYD_URING_RERANK_PIPELINE=0
export LLOYD_PREFETCH_CELLS=0

run_arm() {
  local name="$1"; shift
  local clear="$1"; shift
  local log="$OUT/run_1m_tilererank_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  KNN_CLEAR_CACHE="$clear" env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  echo "    exit=$?"
  grep -E "NOTE: (index\(s\)|force_merge)" "$log" 2>/dev/null | sed 's/^/    /'
  grep "hammingKernel" "$log" 2>/dev/null | grep -v "cellsScanned=0" | tail -1 | sed 's/^/    /'
  # ENGAGEMENT PROOF. A flag that silently did not take effect has produced wrong conclusions here twice,
  # so print the tile counter and shout if the TILED arm scored zero tiles.
  local tl
  tl="$(grep "tileRerank" "$log" 2>/dev/null | tail -1)"
  echo "    ${tl:-[lloyd tileRerank] NO LINE -- path never ran}"
  if [ "$name" = "TILED" ] && ! echo "$tl" | grep -q 'tilesScored=[1-9]'; then
    echo "    *** WARNING: tiled rerank never engaged -- this arm does NOT measure the optimization ***"
  fi
  if [ "$name" = "PERCAND" ] && echo "$tl" | grep -q 'tilesScored=[1-9]'; then
    echo "    *** WARNING: control arm ran tiles -- the A/B is not comparing two different paths ***"
  fi
  grep "^SUMMARY" "$log" 2>/dev/null | sed 's/SUMMARY: //' | awk -F'\t' '{printf "    recall=%s  lat=%s ms\n",$1,$2}'
}

# TILED arm (the new default) builds/reuses the golden index.
run_arm TILED   "${KNN_CLEAR_CACHE:-0}" JAVA_TOOL_OPTIONS=
# PERCAND control reuses the SAME index; only the scoring path differs.
run_arm PERCAND 0 JAVA_TOOL_OPTIONS=-Dlloyd.noTileRerank=true

echo "=== done $(date). logs: $OUT/run_1m_tilererank_*_${STAMP}.log ==="
