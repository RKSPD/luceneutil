#!/usr/bin/env/python

# TODO
#   - hmm what is "normalized" boolean at KNN indexing time? -- COSINE similarity sets this to true
#   - try turning diversity off -- faster forceMerge?  better recall?
#   - why force merge 12X slower
#   - why only one thread
#   - report net concurrency utilized in the table
#   - report total cpu for all indexing threads too
#   - hmm how come so much faster to compute exact NN at queryStartIndex=0 than 10000, 20000?  60 sec vs ~470 sec!?
#     - not always the first run!  sometimes 2nd run is super-fast

import argparse
import itertools
import mmap
import multiprocessing
import os
import random
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

try:
  import numpy as np
except ImportError:
  print("\nERROR: numpy is required but not installed.\n")
  print("To fix, run from the luceneutil root directory:\n")
  print("  make env")
  print("  source .venv/bin/activate")
  print("  python -u src/python/knnPerfTest.py\n")
  raise SystemExit(1) from None

import autologger
import benchUtil
import constants
import knnExactNN
import ps_head
import ram_monitor
from benchUtil import GNUPLOT_PATH, PERF_EXE
from common import getLuceneDirFromGradleProperties

# toggle between 'pread' and 'mmap' for concurrent random vector reads when smelling vectors -- pread is
# maybe a bit faster?
IO_METHOD = "pread"
# Debug: pass -Dlsh.scanStats=true to the search JVM so LSHVectorsReader dumps columnar scan I/O stats
# (filter-region bytes read vs payload bytes faulted, and survivor fraction) at exit. Off for real runs.
LSH_SCAN_STATS = True
# When True, pass -Dlsh.offHeapCentroids=true so LSHVectorsReader does NOT pin per-bucket centroids on
# the heap (reference centroids reconstructed from the bucket code; TRUE centroids read from the mmap'd
# index per probed bucket). Drops per-bucket heap from ~2*dim floats to ~8 bytes -- needed at the >RAM /
# high-bucket-count scale -- at the cost of recompute + small reads per probed bucket. Leave False when
# the whole index fits in RAM (the eager resident arrays are then pure speed upside, ~2x lower latency).
LSH_OFF_HEAP_CENTROIDS = False
# When True, pass -Dlsh.positionalScan=true so LSHVectorsReader's per-bucket scan reads filter pairs +
# survivor payload rows POSITIONALLY off a single whole-file random-access view (created once per query)
# instead of allocating a fresh filter slice + payload view and issuing a prefetch (madvise WILLNEED) per
# probed bucket -- i.e. it drops O(nprobe) view allocations + syscalls/query. Result-neutral (bit-identical
# bytes/order), reader-only, so it needs no reindex: A/B it by toggling this on the same index. Most likely
# to help at high nprobe and warm/in-RAM (where the dropped prefetch did nothing but cost). NOTE: opposed to
# LSH_OFF_HEAP_CENTROIDS in the >RAM/cold regime, where the explicit prefetch front-runs the page fault --
# measure the two together there, not just warm.
LSH_POSITIONAL_SCAN = False
# When True, pass -Dlsh.hnswRouting=true so LSHVectorsReader selects buckets by navigating an in-RAM HNSW
# graph over the bucket centroids (built at reader open) instead of query-directed Hamming multi-probe.
# Multi-probe can only reach buckets Hamming-near the query's code; the graph reaches centroid-near but
# Hamming-far buckets too, so the searcher visits FEWER, better buckets at matched recall -- letting
# lshHashBits be pushed high (small buckets, precision retained on quantize) without the router losing the
# right ones. Reader-only and search-time: the graph is a RAM-only DERIVED structure, never persisted, so
# the on-disk format and concat merge are untouched and NO reindex is needed -- A/B it by toggling on the
# same index (like LSH_POSITIONAL_SCAN). Requires the eager centroid path, so it is INCOMPATIBLE with
# LSH_OFF_HEAP_CENTROIDS (which keeps no resident centroid arrays to build a graph over).
LSH_HNSW_ROUTING = True
# HNSW routing graph params (only used when LSH_HNSW_ROUTING). hnswM = max connections per node (graph
# fanout), hnswBeamWidth = construction-time beam, hnswOverquery = search-time pool multiplier (the router
# asks the graph for hnswOverquery * bucketPoolFactor * nprobe centroids per table, mirroring multi-probe's
# over-fetch). None => use the codec defaults (16 / 100 / 1).
LSH_HNSW_M = 20
LSH_HNSW_BEAM_WIDTH = 300
LSH_HNSW_OVERQUERY = 1
# When True, pass -Dlsh.routeOnReferenceCentroids=true so the routing graph is built over REFERENCE
# centroids (pure function of the bucket code) instead of the default TRUE centroids (empirical member
# means). Reference centroids are merge-invariant (identical in every segment / after any merge), the
# prerequisite for a persisted/shared routing graph; true centroids route better but change with bucket
# membership so the graph must be rebuilt per merged segment. This A/Bs the routing-quality gap: as
# hashBits rises the SimHash wedge narrows and reference -> true, so the gap should shrink at high bits.
# Search-time, reader-only (no reindex) — toggle on the same index. Routing-graph SOURCE only; the
# downstream pool re-rank still uses true centroids, so this isolates graph quality, not final scoring.
LSH_ROUTE_ON_REFERENCE_CENTROIDS = False
# When True, pass -Dlsh.referenceCentroidsOnly=true so the WRITER skips per-bucket TRUE centroids
# entirely (no compute, no dim-float on-disk block, and — the big win — no count-weighted recombine +
# radius shift in the concat merge, since reference centroids are identical across segments). The
# reader falls back to the code-derived REFERENCE centroid for routing, pool re-rank, AND the scan
# bound (radius is reference-anchored at write so the early-termination bound stays admissible). This
# is the merge-speed / merge-invariant-graph direction: costs some routing/bound precision (wedge axis
# vs member mean) — small at high hashBits (narrow wedges), larger at low bits. WRITE-TIME: changes the
# on-disk format (centroidsLength==0), so it IS effectively in the index and a sweep reindexes per value.
LSH_REFERENCE_CENTROIDS_ONLY = True
# IVF coarse-routing A/B knobs (search-time, reader-only — reused on the SAME cached index, no reindex).
#   IVF_CENTROID_HNSW: False => -Divf.centroidHnsw=false, forcing the EXACT linear coarse scan (rank all
#     nlist centroids on the full dim). This is the routing-quality ceiling: if recall at fixed nprobe
#     jumps vs the graph, the graph router is the bottleneck; if it doesn't, coarse routing is already
#     tight and recall loss is elsewhere (frozen centroids / spill=0).
#   IVF_REFINE_FACTOR: None => use the persisted value; an int => -Divf.refineFactor=N, so the graph
#     over-collects N*nprobe centroids and re-ranks them on the FULL dim before probing. Larger N drives
#     the graph's probe set toward the exact-scan probe set at a fraction of exact-scan cost.
IVF_CENTROID_HNSW = None      # None => leave default (graph on); False => exact linear coarse scan
IVF_REFINE_FACTOR = None      # None => persisted; e.g. 4 => -Divf.refineFactor=4
# IVF_RERANK_FACTOR: None => persisted; an int => -Divf.rerankFactor=N. Cranked high (e.g. 400) every
# visited doc gets exact float32 rerank, so recall becomes a PURE COVERAGE measurement (partition
# quality) with quantization misranking removed. Search-time, reader-only — same cached index.
IVF_RERANK_FACTOR = 1
# IVF_ADAPTIVE_NPROBE_MARGIN: None/>=1.0 => disabled (always probe full nprobe). A float in (0,1) =>
# -Divf.adaptiveNprobeMargin=M: probe UP TO nprobe cells but stop early, keeping cell j only while its
# query→centroid squared distance dj <= d0/M (d0 = nearest cell). M→1 keeps ~all cells; smaller prunes
# harder (lower avg latency, risks recall). A floor of 8 cells guards recall. SEARCH-TIME, reader-only —
# same cached index, so A/B it without reindexing. Sweep e.g. 0.5/0.7/0.85. See findings §19.
IVF_ADAPTIVE_NPROBE_MARGIN = None
# LLOYD_BEAM_FACTOR: lloyd_ivf only. Widens the centroid-graph beam to efSearch=ceil(nprobe*factor)
# while still scanning only the top nprobe cells -> better cell selection at ~free coarse-select cost
# (select is cheap vs the posting scan). None/1.0 => beam == nprobe (old behavior). Search-time only
# (-Dlloyd.beamFactor), so a factor sweep reuses ONE cached index. Try 1.0/1.5/2.0/3.0.
LLOYD_BEAM_FACTOR = float(os.environ["LLOYD_BEAM_FACTOR"]) if os.environ.get("LLOYD_BEAM_FACTOR") else None
# LLOYD_SCORE_IN_PLACE=1 => -Dlloyd.scoreInPlace=true: in the posting scan, score the doc code where
# it already sits in the record buffer instead of System.arraycopy-ing it into a scratch array first
# (VectorUtil.int4DotProductSinglePackedAt / uint8DotProductAt). Same kernel, same integer dot, so
# scores are BIT-IDENTICAL: same recall, same index, latency only. SEARCH-TIME only, so an A/B reuses
# ONE cached index. Measured on Graviton3 (SVE 256-bit, dim=1024): 4-bit 55.7->48.0 ns/doc scan.
LLOYD_SCORE_IN_PLACE = os.environ.get("LLOYD_SCORE_IN_PLACE") == "1"
# LLOYD_PREFETCH_CELLS=1 => -Dlloyd.prefetchCells=true: Stage A of the >RAM I/O plan (open.md §1).
# Batch-prefetches each probed cell's sketch + code run before the scan, instead of faulting one doc's
# sketch at a time (~100k serialized faults/query at nlist=2000/spill=2/nprobe=100). Advisory madvise,
# so recall is unchanged; only helps when reads MISS the page cache (cold-cache / >RAM), no-op when warm.
# SEARCH-TIME only => an A/B reuses ONE cached index.
LLOYD_PREFETCH_CELLS = os.environ.get("LLOYD_PREFETCH_CELLS") == "1"
# IVF_STREAM_REFINE_ITERS: full-corpus Lloyd refinement passes in the STREAMING merge (0 = sample-only,
# the old behavior). The streaming merge no longer has to hold every vector in RAM, so centroids need not
# be limited to the ≤trainSampleCap subsample: each pass streams all vectors and accumulates per-cell sums
# (resident cost nlist*dim floats, independent of N). WRITE-time => in the index key, so a sweep reindexes.
IVF_STREAM_REFINE_ITERS = os.environ.get("IVF_STREAM_REFINE_ITERS")
# IVF_STREAM_FLUSH_MIN_DOCS: above this many docs in a segment's field buffer, FLUSH stops holding
# full-dim floats and instead quantizes each vector on arrival into a temp record file (data-blind zero-mu
# codes make this safe), clustering from that file. Resident RAM becomes O(1)/doc instead of ~4 KB/doc.
# 0/unset = previous heap-buffered behavior. WRITE-time (changes centroids) => in the index key.
IVF_STREAM_FLUSH_MIN_DOCS = os.environ.get("IVF_STREAM_FLUSH_MIN_DOCS")
# IVF_TRAIN_SAMPLE_CAP: cap on the training subsample used to seed streaming-merge centroids. This is
# only the SEED -- streamRefineIters then runs full-corpus Lloyd over every doc -- but the seed still has
# to be non-degenerate: at nlist=100k the default 200k cap leaves 2 sample vectors per centroid, so raise
# it with nlist to keep a sane vectors/centroid ratio. Held as float[sample][dim] (~4 KB each), so a 2M
# sample is ~8 GiB resident: raise KNN_HEAP alongside it.
IVF_TRAIN_SAMPLE_CAP = os.environ.get("IVF_TRAIN_SAMPLE_CAP")
# Centroid-graph (routing among the nlist centroids) build params. The defaults M=16/beamWidth=16 were
# tuned at nlist=2000; a 100k-centroid graph needs more connectivity or routing recall drops, which shows
# up as missed cells (direct recall loss), not as an error. WRITE-time => in the index key.
IVF_CENTROID_HNSW_M = os.environ.get("IVF_CENTROID_HNSW_M")
# Derived (fixed random topology) centroid graph: no HNSW build, adjacency from a seed. WRITE-time
# (changes routing/spill) => in the index key.
IVF_DERIVED_GRAPH = os.environ.get("IVF_DERIVED_GRAPH") == "1"
IVF_DERIVED_GRAPH_M = os.environ.get("IVF_DERIVED_GRAPH_M")
IVF_CENTROID_HNSW_BEAM_WIDTH = os.environ.get("IVF_CENTROID_HNSW_BEAM_WIDTH")
# Beam ef for the per-doc spill fan-out. With IVF_BEAM_SPILL=1 the SOAR pool is max(4*spillPerDoc, 64)
# instead of ALL nlist, which is what makes spill affordable at large nlist (the full scan is
# O(count*nlist*dim) -- projected ~33 h at 40M x nlist=100k).
IVF_SPILL_EF_SEARCH = os.environ.get("IVF_SPILL_EF_SEARCH")
# Per-cell cap (bytes) on the CODE run hinted by Stage A; 0 disables code prefetch (sketch only).
LLOYD_PREFETCH_CODE_MAX_BYTES = os.environ.get("LLOYD_PREFETCH_CODE_MAX_BYTES")
# LLOYD_CEIL_AUDIT=1 => -Dlloyd.ceilAudit=true: per-query, the reader exact-scans its own code table
# for the achievable top-k, then reports how the NN docs' primary cells scatter across the centroid-
# distance frontier (meanDistinctNNCells, meanMaxNNCellRank). Diagnostic only; slow (full scan/query).
LLOYD_CEIL_AUDIT = os.environ.get("LLOYD_CEIL_AUDIT") == "1"
# LLOYD_CEIL_DIR=1 => -Dlloyd.ceilDir=true: adds the EXACT directional-oracle ranking to the ceiling
# audit (needs LLOYD_CEIL_AUDIT=1). RAM-heavy (retains count*dim float residuals). Diagnostic only.
LLOYD_CEIL_DIR = os.environ.get("LLOYD_CEIL_DIR") == "1"
# LLOYD_CEIL_ANISO=1 => -Dlloyd.ceilAniso=true: anisotropic-reassignment audit (§10c). Needs
# LLOYD_CEIL_AUDIT=1. Cheap-ish (full centroids, no residual RAM). Sweeps eta internally.
LLOYD_CEIL_ANISO = os.environ.get("LLOYD_CEIL_ANISO") == "1"
# LLOYD_CEIL_SUB=1 => -Dlloyd.ceilSub=true: 2-level sub-centroid ceiling probe (§10e). Needs
# LLOYD_CEIL_AUDIT=1. Builds K sub-centroids/cell, ranks by min query->subcentroid dist. RAM-heavy.
LLOYD_CEIL_SUB = os.environ.get("LLOYD_CEIL_SUB") == "1"
# LLOYD_CEIL_SEP=1 => -Dlloyd.ceilSep=true: separability probe (§10f). Needs LLOYD_CEIL_AUDIT=1.
LLOYD_CEIL_SEP = os.environ.get("LLOYD_CEIL_SEP") == "1"
# LLOYD_CEIL_ONLINE=1 => -Dlloyd.ceilOnline=true: online-steering probe (§10h). Needs LLOYD_CEIL_AUDIT=1.
LLOYD_CEIL_ONLINE = os.environ.get("LLOYD_CEIL_ONLINE") == "1"
# LLOYD_CEIL_ADJ=1 => -Dlloyd.ceilAdj=true: graph-adjacency probe (§10i). Needs LLOYD_CEIL_AUDIT=1.
LLOYD_CEIL_ADJ = os.environ.get("LLOYD_CEIL_ADJ") == "1"
# LLOYD_CEIL_SELEXP=1 => -Dlloyd.ceilSelExp=true: selective-expansion audit (§10j). Needs LLOYD_CEIL_AUDIT=1.
LLOYD_CEIL_SELEXP = os.environ.get("LLOYD_CEIL_SELEXP") == "1"
# LLOYD_CEIL_BRUTE=1 => -Dlloyd.ceilBrute=true: full-corpus 1-bit sign scan → int8 rerank recall (§10L). Needs LLOYD_CEIL_AUDIT=1.
LLOYD_CEIL_BRUTE = os.environ.get("LLOYD_CEIL_BRUTE") == "1"
# LLOYD_CEIL_SPILL=1 => -Dlloyd.ceilSpill=true: §11 spill audit — credit each NN with the NEAREST-ranked
# cell it is SPILLED into (min rank over its posting lists) vs the primary-cell-only baseline. The gap
# shows how far spill pulls the NN-cell frontier forward. Needs LLOYD_CEIL_AUDIT=1, ivfSpillBits>0, and
# ORD-order (IVF_CELL_ORDER=0 — the audit needs ord==doc). No-op under cell-order.
LLOYD_CEIL_SPILL = os.environ.get("LLOYD_CEIL_SPILL") == "1"
# GCUT_TREE_ROUTE_DIMS=<n> => -Dgcut.treeRouteDims=<n>: WRITE-time. Persist the gcut cut tree with
# axis/center rows truncated to the leading n (post-rotation) dims, enabling the partition-consistent
# margin router. 0/unset disables (routes by nearest-centroid HNSW as before). In the index → clear
# cache when changed. Use the full dim for an exact tree.
GCUT_TREE_ROUTE_DIMS = int(os.environ["GCUT_TREE_ROUTE_DIMS"]) if os.environ.get("GCUT_TREE_ROUTE_DIMS") else None
# GCUT_TREE_ROUTE=1 => -Dgcut.treeRoute=true: SEARCH-time. Route by descending the persisted cut tree
# (best-first by boundary margin) instead of nearest-centroid. No-op unless the segment carries a tree
# (GCUT_TREE_ROUTE_DIMS was set at index time). Reuses one cached index for the A-B vs nearest-centroid.
GCUT_TREE_ROUTE = os.environ.get("GCUT_TREE_ROUTE") == "1"
# LLOYD_BRUTE_SEARCH=1 => -Dlloyd.bruteSearch=true + -Dlloyd.ceilBruteDims=1024: the real two-phase
# search path (sign-1024 Hamming shortlist → int8 rerank). Drops IVF routing as a recall mechanism.
LLOYD_BRUTE_SEARCH = os.environ.get("LLOYD_BRUTE_SEARCH") == "1"
# LLOYD_SKETCH_SCAN=1 => -Dlloyd.sketchScan=true + sign-1024: routed sketch scan (§10M). Keeps HNSW
# routing, scans selected cells with 1-bit Hamming → int8 rerank. RAM-friendly (sublinear). Pair w/ high nprobe.
# DEFAULT ON: the §10N shippable operating point (cell-order 5-bit sketch-scan, 0.944 / 12.3 ms). Set
# LLOYD_SKETCH_SCAN=0 to force it off.
LLOYD_SKETCH_SCAN = os.environ.get("LLOYD_SKETCH_SCAN", "1") == "1"
# IVF_QUANT_BITS: rerank code bit-depth. Write+read side. DEFAULT 8 (int8, one byte/dim): it measured
# FASTER at BETTER recall than the 4-bit codebook (8.4x scan speedup), at ~2x the bytes/doc. 4 => 4-bit
# codes (512 B/doc at dim=1024) but 4-bit cannot clear 0.95 recall; 5 => 5-bit bit-plane codes, which have
# no SIMD path (planeDot is a bit-scan) and are slow.
# NOTE: this only takes effect when IVF_QUANTIZER is the coordinate-wise 'osq' layout (now the codec
# default). Under -Divf.quantizer=blocksphere4 the codebook layout wins and quantBits is IGNORED.
IVF_QUANT_BITS = os.environ.get("IVF_QUANT_BITS", "8")
# IVF_BEAM_SPILL=1 => -Divf.beamSpill=true: ADAPTIVE HNSW-beam spilling (§11). When ivfSpillBits>0 the
# writer routes each doc through the centroid HNSW beam for its 1+spillBits nearest cells, then keeps only
# the leading cells within IVF_SPILL_MARGIN× the nearest cell's distance — boundary docs spill, interior
# docs stay single-cell. Coexists with cell-order (spilled records duplicated per cell block). WRITE-TIME
# (in the index) → reindex per value. Requires ivfSpillBits>0 in PARAMS to do anything.
IVF_BEAM_SPILL = os.environ.get("IVF_BEAM_SPILL") == "1"
# IVF_SPILL_MARGIN: adaptive-spill distance ratio (default 1.30, matches the codec default). Keep cell j
# while dist_j <= margin*dist_0. Larger => more spill (higher recall, bigger index). WRITE-TIME → reindex.
IVF_SPILL_MARGIN = os.environ.get("IVF_SPILL_MARGIN")
# LLOYD_RERANK_BITS: simulate B-bit rerank in sketch-scan (§10M gate). Default 8 (real int8). 4 => 4-bit.
LLOYD_RERANK_BITS = os.environ.get("LLOYD_RERANK_BITS")
# IVF_ENABLE_COPY_MERGE: the codec default is now the warm-start re-cluster merge path (seed centroids
# from the largest donor segment + GRAPH_ROUTE_ITERS graph-routed Lloyd passes over ALL merged docs +
# requantize) — centroids adapt to the merged distribution, best recall. Set True => -Divf.enableCopyMerge
# =true to opt INTO the frozen-centroid copy-merge fast path (faster merge, worse recall) for a merge-speed
# A/B. WRITE-TIME: reindexes. Leave False for the recall hot path.
IVF_ENABLE_COPY_MERGE = False
# IVF_EXACT_ASSIGN: True => -Divf.exactAssign=true, writer places each doc via an exact full-dim
# nearest-centroid scan (matches reader cell selection) instead of the lossy efSearch=8 graph beam.
# WRITE-TIME: reindexes. Tests whether coarse-partition recall loss is assignment fidelity.
IVF_EXACT_ASSIGN = False
# IVF_WORK_DIMS: None => no truncation (full dim). An int => -Divf.workDims=N: after the full-dim
# Hadamard rotation, truncate to N leading dims as the WORKING representation for centroids/codes/
# routing/ADC (≈dim/N cheaper scan/route/quantize + smaller postings); full dim kept only for rotation
# + exact rerank. JL-safe. WRITE-TIME: reindexes. Sweep 64/128/256/512 for the recall/latency frontier.
IVF_WORK_DIMS = None
# IVF_GRAPH_ROUTE_ITERS: None => leave default (2); an int => -Divf.graphRouteIters=N. This is the
# "Lloyd iters on merge/flush" lever: each pass builds an HNSW over the current centroids, graph-routes
# every doc (~log(nlist) comparisons), and recomputes centroids. N=1 = single online-k-means step (seed
# + one refine); N=2 (default) converges incremental centroids; higher N = more centroid refinement per
# flush/merge at O(N·count·log(nlist)·dim). WRITE-TIME: reindexes. A/B whether extra merge-time Lloyd
# refinement actually buys recall (findings: more Lloyd iters made high-nlist recall WORSE — the gap is
# writer assignment vs reader selection fidelity, not centroid position). Set to 1 to test cheaper merge.
IVF_GRAPH_ROUTE_ITERS = None
# IVF_ANISO_ETA: ScaNN anisotropic k-means ratio eta=h_par/h_orth (write-time, IN the index -> reindex).
# None/1.0 => today's spherical Lloyd. >1 => anisotropic assign + WLS centroid update, no renorm.
IVF_ANISO_ETA = float(os.environ["IVF_ANISO_ETA"]) if os.environ.get("IVF_ANISO_ETA") else None
# IVF_SHARED_CODES: False => legacy per-record postings (self-contained centroid-relative OSQ records).
# True => -Divf.sharedCodes=true: each doc's OSQ code is stored ONCE in a per-ord code table (quantized
# against a single GLOBAL reference = the rotated field mean, NOT the per-cell centroid), and postings
# become 4-byte ord references. Decouples index size from spillBits (a spilled doc replicates 4 bytes,
# not a ~1KB code). Cost: coarser per-cell quantization (§16b says recall-neutral WITH rerank; without
# rerank the coarse scan is worse). WRITE-TIME: reindexes. OSQ only (ignored for PQ/Block).
IVF_SHARED_CODES = True
# IVF_DROP_RAW_VECTORS: False => codec default (store the original full-precision vectors in .vec/.vemf,
# used for exact rerank + merge re-clustering on true originals). True => -Divf.dropRawVectors=true: do
# NOT persist the full-precision vectors. Shrinks the index by the raw float32 footprint (dim*4 bytes/doc)
# but disables exact rerank (rerankFactor is forced to 1) and forces merges to re-cluster on reconstructed
# (quantized) vectors instead of true originals. SAFE ONLY when rerank is off (IVF_RERANK_FACTOR=1), since
# no reranking is done on the IVF codec. WRITE-TIME: reindexes.
IVF_DROP_RAW_VECTORS = False
# GCUT partitioner knobs are swept via the PARAMS dict (gcutMode/gcutSplitBy/gcutBalancePenalty/
# gcutAxes), passed as CLI args to KnnGraphTester, which sets the corresponding -Dgcut.* sysprops.
# Number of concurrent indexing threads passed to KnnGraphTester (-numIndexThreads). Affects build
# wall-clock only; with -forceMerge the final single-segment index is concurrency-independent. This box
# has 12 cores. Used at both the search-and-stats and the search-only command builders below.
NUM_INDEX_THREADS = 8
# IO_METHOD = "mmap"


def advise_will_need(file_name, offset_bytes=0, length_bytes=0):
  """Proactively hint to the OS to load a range of file into RAM, using the configured IO_METHOD."""
  if not os.path.exists(file_name):
    return

  file_size = os.path.getsize(file_name)
  if length_bytes <= 0 or offset_bytes + length_bytes > file_size:
    length_bytes = file_size - offset_bytes

  if length_bytes <= 0:
    return

  with open(file_name, "rb") as f:
    if IO_METHOD == "pread":
      # os.posix_fadvise is Linux-only; skip the hint where it is unavailable.
      if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(f.fileno(), offset_bytes, length_bytes, os.POSIX_FADV_WILLNEED)
    elif IO_METHOD == "mmap":
      # map the part of the file we need
      mm = mmap.mmap(f.fileno(), length_bytes, offset=offset_bytes, access=mmap.ACCESS_READ)
      try:
        mm.madvise(mmap.MADV_WILLNEED)
      finally:
        mm.close()


# see also https://share.google/aimode/IDYCxtTyGhFUwC1pX for clean-ish
# ways to use io_uring-like async io from Python

# Measure vector search recall and latency while exploring hyperparameters

# SETUP:
### Download and extract data files: Wikipedia line docs + GloVe
# python src/python/initial_setup.py -download    OR    curl -O  https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip -k
# cd ../data
# unzip glove.6B.zip
# unlzma enwiki-20120502-lines-1k.txt.lzma    OR    xz enwiki-20120502-lines-1k.txt.lzma
### Create document and task vectors
# ./gradlew vectors-100
#
# change the parameters below and then run (you can still manually run this file, but using gradle command
# below will auto recompile if you made any changes to java files in luceneutils)
# ./gradlew runKnnPerfTest
#
# for the median result of n runs with the same parameters:
# ./gradlew runKnnPerfTest -Pruns=n
#
# you may want to modify the following settings:

# uses CPUTime sampling (newly available/experimental in Java 25, seems to work on the tasks benchmark)
# KNN_JFR=1 enables it without editing this file (useful for one-off profiling runs, e.g. attributing
# >RAM latency between CPU and I/O wait).
DO_PROFILING = os.environ.get("KNN_JFR") == "1"
# Monotonic counter so each profiled JVM invocation gets its own .jfr (see jfr_output).
_JFR_SEQ = [0]
DO_PS = True
# vmstat is Linux-only; disable when the executable is unavailable (e.g. macOS)
DO_VMSTAT = benchUtil.VMSTAT_PATH is not None
# Live per-run RAM (RSS) monitor for the search/index JVM: prints a \r-updating gauge to the terminal,
# logs (elapsed,rss_mb) to a CSV, and writes a per-run RSS-over-time HTML chart. Tracks the codec's actual
# resident footprint (heap + faulted mmap pages) -- the number the off-heap-centroids work targets (§20).
DO_RAM_MONITOR = True

# precompute exact NN using numpy (multi-threaded BLAS matmul).  when False,
# KnnGraphTester.java computes exact NN itself (slower, single-threaded Java).
USE_NUMPY_EXACT_NN = True

# Run check_vector_overlap (doc-vs-query duplicate / test-on-train scan). This hashes every doc vector
# in pure Python (O(ndoc)) and holds an ndoc-entry dict in RAM, which is prohibitively slow and memory
# hungry at large ndoc (e.g. 20M -> hangs + OOMs). Off by default; the docs and queries here come from
# distinct files/ranges, so the scan is unnecessary. Enable only for small ndoc when validating a new
# dataset for accidental train/test overlap.
CHECK_VECTOR_OVERLAP = False
# Enable to also catch duplicates within the doc set or within the query set (requires CHECK_VECTOR_OVERLAP)
CHECK_DOC_DOC_DUPLICATES = False
CHECK_QUERY_QUERY_DUPLICATES = False

# Sample doc and query vectors and statistically compare their distributions to
# detect dual-encoder model mismatch (e.g. someone re-embedded one side with a
# different model/checkpoint).  Cheap (~seconds): computes per-dim z-scores plus
# cross-score spread vs within-side spread.  Raises if mismatch looks severe.
CHECK_QUERY_DOC_MODEL_CONSISTENCY = True
QUERY_DOC_MODEL_CONSISTENCY_SAMPLE = 5000

# Set this to True to use perf tool to record instructions executed and confirm SIMD
# instructions were executed
# TODO: how much overhead / perf impact from this?  can we always run?
CONFIRM_SIMD_ASM_MODE = False

# perf stat SIMD validation: uses hardware counters to confirm SIMD (SSE/AVX2/AVX-512)
# is actually used during vector operations.  negligible overhead -- always on when perf
# is available.
DO_PERF_STAT_SIMD = PERF_EXE is not None

# set this to True to collect all HNSW traversal scores and generate a histogram
DO_HNSW_SCORE_HISTOGRAM = False

# sample 1 in every N HNSW traversal scores when building the histogram (to keep HTML size reasonable)
HNSW_SAMPLE_EVERY_N = 100

# set this to True to compute sampled all query x doc distances and generate a histogram
DO_ALL_DISTANCES_HISTOGRAM = False

# sample 1 in every N all-distances scores (sampling done in Java).
# with 400K docs and 10K queries (4B distances), 1000 -> ~4M samples -> ~40 MB HTML
ALL_DISTANCES_SAMPLE_EVERY_N = 1000

# enable to compute intrinsic-dimensionality estimates of the doc and query
# vectors as part of smell-vector analysis.
#
# why care about ID?  vector-search difficulty (HNSW recall vs beamWidth/fanout,
# quantization tolerance, NN-distance contrast) is governed by the *intrinsic*
# dimensionality of the data manifold, not by the *ambient* dimension of the
# embedding.  two corpora with ambient dim=1024 but ID=8 vs ID=80 will look
# wildly different at search time even though they "look the same" by shape.
#
# how it relates to the existing isotropy / effective-rank measure:
#   - isotropy and effective_rank are LINEAR measures: they ask "how many
#     coordinate axes does the data spread across?", which is a property of
#     the covariance spectrum.
#   - intrinsic dim is a MANIFOLD measure: it asks "how many independent
#     latent variables does the data actually vary along?", which can be
#     much smaller than the linear answer when the data lives on a curved
#     low-D surface inside the ambient space (very common for embeddings).
#   - the gap between PCA-95% and TwoNN reveals manifold curvature.
DO_INTRINSIC_DIM = True

# sample size for ID estimation.  TwoNN and MLE-Levina-Bickel both rely on
# nearest-neighbor distance ratios that are noisy at small N.  10k samples
# gives stable d-hat for ID up to ~50 while keeping the all-pairs distance
# matrix at ~400MB float32 (10k x 10k) and BLAS matmul under ~5s on a
# typical multicore box.  at higher ID the estimator needs more points;
# bump this if reported R^2 is consistently low.
_NUM_INTRINSIC_DIM_SAMPLE_VECS = 10_000

# k values for MLE Levina-Bickel.  small k -> more local (closer to true
# manifold dimension under curvature) but noisier.  large k -> smoother
# but more biased high under non-uniform sampling density.  reporting both
# gives a sense of stability across scales.
_MLE_K_VALUES = (10, 20)

# fraction of TwoNN ratios to discard from the upper tail.  very large mu
# values come from points near density boundaries and from outliers; they
# bend the regression away from the true slope.  Facco et al. (2017)
# recommend ~10%.
_TWONN_DISCARD_FRAC = 0.10

# WARNING thresholds for ID analysis -- triggers loud output if data looks
# pathological for KNN benchmarking:
#
# TwoNN regression R^2 below this -> the Pareto-tail model doesn't fit,
# which means the data isn't a clean single manifold (mixture of clusters,
# heavy duplicate population, isolated outliers, etc).  ID estimate is
# unreliable; the dataset itself may be unsuitable as a benchmark.
_THRESH_TWONN_BAD_FIT_R2 = 0.95

# TwoNN ID below this -> vectors are essentially collinear; HNSW and
# quantization will be trivially easy and benchmark numbers won't transfer
# to real workloads.
_THRESH_TWONN_DEGENERATE_ID = 2.0

# TwoNN ID above this fraction of ambient dim -> vectors look like noise
# (poorly trained / random embeddings); benchmarks will be hard but not
# meaningfully so.
_THRESH_TWONN_NEAR_AMBIENT_FRAC = 0.5

# PCA-95% / TwoNN ratio above this -> highly curved manifold within a
# moderate-rank linear subspace.  random rotations (RaBitQ) less effective
# here than learned rotations (PCA, OPQ) for quantization.
_THRESH_HIGH_CURVATURE_RATIO = 20.0

# max(MLE) / TwoNN (or vice versa) above this -> ID estimators disagree
# strongly; treat reported ID as ~uncertain by 2x.
_THRESH_ID_DISAGREEMENT_RATIO = 2.0

# fraction of duplicate (r1==0) sample points above which we consider the
# corpus problematic for KNN benchmarking.
_THRESH_DUPLICATES_FRAC = 0.001

if CONFIRM_SIMD_ASM_MODE and PERF_EXE is None:
  raise RuntimeError("CONFIRM_SIMD_ASM_MODE is True but PERF_EXE is not found; install 'perf' tool and rerun?")

# e.g. to compile KnnIndexer:
#
#   javac -d build -cp /l/trunk/lucene/core/build/libs/lucene-core-10.0.0-SNAPSHOT.jar:/l/trunk/lucene/join/build/libs/lucene-join-10.0.0-SNAPSHOT.jar src/main/knn/*.java src/main/WikiVectors.java src/main/perf/VectorDictionary.java
#

NOISY = True

# TODO
#  - can we expose greediness (global vs local queue exploration in KNN search) here?

# test parameters. This script will run KnnGraphTester on every combination of these parameters

def _env_tuple(name, default, cast=int):
  """Comma-separated env override for a PARAMS sweep axis; unset => the literal default.

  Added so HNSW-vs-lloydivf comparisons can be scripted. Without this, maxConn/fanout/indexType were
  reachable only by editing PARAMS by hand, so a shell script that "swept" them silently ran the defaults
  N times -- the same class of vacuous A/B that benchmarks.md 13 documents for the uring flags.
  """
  v = os.environ.get(name)
  if not v:
    return default
  return tuple(cast(x.strip()) for x in v.split(",") if x.strip())


PARAMS = {
  "ndoc": (1_000_000,),
  "indexType": _env_tuple("KNN_INDEX_TYPE", ("lloyd_ivf",), str),
  # IVF params (ignored for hnsw runs)
  # Target ~50 docs/Voronoi cell: nlist = ndoc / 50 = 1_000_000 / 50 = 20_000.
  # nlist sweep: smaller cells (higher nlist) should reach recall=0.95 while visiting FEWER total docs
  # (~docs/cell * nprobe), even though they need more probes. nlist is in the index key → one reindex
  # per value; nprobe is a search-time override so each index sweeps nprobe cheaply.
  # Sweep nlist SMALL with nprobe=1: find the nlist where a query's true top-100 is (almost) all
  # within its single nearest cluster. Recall here == "fraction of top-100 in the 1 nearest cluster".
  # nlist is write-time (in the index key) so each value reindexes. Smaller nlist = bigger cells =
  # more coverage per probe (nlist=1 is trivially 1.0: one cell = whole corpus).
  # Find the LARGEST nlist (smallest cells = cheapest scan) that still gives 0.95+ recall at
  # nprobe=20 with full-scan cells (subNprobe=subNlist). Recall here = coarse coverage of the 20
  # nearest cells. nlist is write-time -> each value reindexes.
  # Target operating point: coverage law says 0.95@nprobe=20 needs nlist~40 (big ~25k-doc cells).
  # Per-cell navigable graph makes searching those big cells cheap. nlist is write-time -> reindex.
  "ivfNlist": (2_000,),
  # nprobe scans the same corpus FRACTION as a well-tuned high-nlist run (~1.3% of cells), which at
  # 50 docs/cell means ~50*nprobe docs visited. Light (ScaNN-style) spilling instead of the heavy
  # spillBits=30 exact-SOAR tax; the larger nprobe recovers the coverage.
  # nprobe is a pure search-time scan budget for lloyd_ivf (reader honors -Dlloyd.nprobe), so ONE
  # cached index serves this whole sweep. nprobe=256 gave ~0.86 recall; sweeping up to reach ~0.95.
  # hier_ivf: nprobe is a pure search-time scan budget (reader honors -Dhier.nprobe), so this whole
  # sweep reuses ONE cached index. At nlist=200 (~5000 docs/coarse cell) probe ~10-20 coarse cells.
  "ivfNprobe": (55, 70),
  # hier_ivf only: subNlist sub-centroids per coarse cell (WRITE-time → in the index key, a sweep
  # reindexes). subNprobe sub-cells scanned per probed cell (SEARCH-time → reader honors
  # -Dhier.subNprobe, reuses the cached index). At subNlist=25 each sub-cell holds ~200 docs;
  # subNprobe=5 scans ~1000 docs/probed cell (~5000*5/25). Ignored for non-hier index types.
  "ivfSubNlist": (25,),
  # subNprobe = subNlist => sub-level is a NO-OP (scan every doc in each probed cell). This isolates
  # the COARSE coverage ceiling: recall at nprobe=N == fraction of true top-100 within the N nearest
  # clusters. Tests "are all 100 NN encompassed within the nearest (few) cluster(s)?".
  # reuses ivfSubNprobe knob -> hier.efSearch (intra-cell graph beam). Sweep: within-cell recall knob.
  "ivfSubNprobe": (256,),  # -> hier.efSearch (intra-cell beam)
  "ivfClusterTrainDims": (1024,),
  # Spilling DISABLED (spillBits=0): each doc lives in exactly one cell (its k-means assignment), so
  # the merge skips the per-doc O(nlist*dim) spill-select scan entirely — that scan dominated merge
  # time. Recall coverage that spilling would have bought is instead recovered by scanning more cells
  # per query via the (adaptive) nprobe above.
  # Lloyd iterations at FLUSH (random seeds, no donor to warm-start from). Iteration 1 only moves the
  # centroids off their random start, so a single pass leaves the partition far from converged; the extra
  # passes are localized reassignment (O(count*M), not O(count*nlist)) plus a PARALLEL centroid recompute,
  # so 5 costs only ~+5% index time. WRITE-time (it determines the persisted centroids) => in the index
  # key, so each value reindexes. Goes hand-in-hand with ivfSpillBits: better-converged centroids make
  # each spill copy land in a more useful cell.
  "ivfFlushIters": (5,),
  "ivfSpillBits": (2,),
  # SOAR is a spill-selection method; no-op when spillBits=0.
  "ivfSoarLambda": (1.0,),
  # hier_ivf: rerankFactor is SEARCH-time (reader honors -Dhier.rerankFactor); pool = factor*topK
  # candidates from the int8 scan, re-scored EXACT against persisted rotated raw vectors. 1 == off.
  # This sweep reuses ONE cached index (rerankFactor out of the key).
  "ivfRerankFactor": (1,),  # 1-bit BlockQuant scan needs exact rerank backstop
  # hier_ivf: workDims truncates the SUB-CELL routing scan to the top-W (Hadamard-rotated) dims.
  # SEARCH-time (reader honors -Dhier.workDims), so this sweep reuses ONE cached index. 0 == full dim
  # (1024). This is a RANDOM projection (Hadamard spreads energy uniformly) — measures how fast recall
  # decays under JL truncation, i.e. how much headroom a data-aware PCA basis would have.
  # LOCAL variance-sorted truncation now: each cell ranks sub-cells on its top-W highest-variance
  # dims (search-time, reuses one index). 0 == full dim (exact). Tests whether local dim-selection
  # recovers the sub-routing recall that GLOBAL/random truncation (earlier sweep) could not.
  "ivfWorkDims": (0,),
  "ivfCentroidRefineFactor": (1,),
  "ivfPqSubspaces": (0,),
  "ivfBlockSize": (0,),
  # LSH params (ignored unless indexType="lsh"). hashBits => up to 2^hashBits buckets;
  # nprobe buckets probed per query; rerankFactor>1 enables exact full-precision rerank.
  # bucketPoolFactor: multi-probe over-fetches bucketPoolFactor*nprobe buckets, re-ranks them by
  # query-to-bucket-reference similarity, and scans the top nprobe best-first (with early termination).
  "lshHashBits": (17,),  # NOTE: hard cap is MAX_HASH_BITS=30 (bucket code must fit a positive int)
  "lshNumTables": (1,),
  "lshNprobe": (600,),
  "lshBucketPoolFactor": (2,),
  "lshRerankFactor": (1,),
  # Frozen PCA hash basis subspace dim (0 = data-independent hash, no PCA). >0 trains (mu,V) once from
  # a doc sample and injects it, correcting rotational anisotropy while staying merge-stable. On the
  # Cohere data m=256 lifts recall ~0.685 -> ~0.799 at matched params vs the data-independent hash.
  "lshPcaDim": (20,),
  # When True (and lshPcaDim>0), also train frozen per-table ITQ rotations that minimize sign-
  # quantization loss on the used hash bits (tighter buckets, higher recall per candidate scanned).
  "lshItq": (True,),
  # OSQ posting precision for the LSH postings: 8 (default), 4, or 1.
  #   8 = symmetric unsigned byte (1 byte/dim).
  #   4 = UNPACKED 4-bit (still 1 byte/dim, same posting size & scoring path as 8-bit) — isolates the
  #       recall impact of lower precision. MEASURED: recall DOWN vs 8-bit, NO size/latency benefit. Skip.
  #   1 = RaBitQ BINARY: 1-bit packed doc (dim/8 bytes) + 4-bit transposed query, asymmetric int4 dot.
  #       ~6.9x smaller postings (the real packed layout win 4-bit lacked) => ~6.8x smaller no-raw index
  #       (crosses under box RAM at 20M). Merge-stable (quantizes vs the code-derived centroid). Phase 1
  #       is quantized-only (NO rerank) — recall recovery untuned; expect to raise nprobe/spill or
  #       re-enable rerank. MEASURED-NEGATIVE on this (unit-norm Cohere) data: killed through three doors
  #       — quantized-only 25% scan = 0.6; coverage-inert; rerank=2 @ 100% visited = only 0.92 (1-bit
  #       misranks ~8% of true neighbors out of the rerank pool). Root cause: 1 bit/dim destroys the
  #       subspace magnitude this data's signal lives in. Metric-invariant (unit-norm). Parked; keep 8-bit.
  #       See findings §24b. (Re-A/B only on a NON-normalized / lower-recall-target corpus.)
  "lshQuantizeBits": (8,),
  # Index-time SPILLING: 0 (default) = off. >0 assigns each doc to its home bucket PLUS the buckets
  # reached by flipping its N lowest-confidence signature bits, so a query finds it without probing more
  # buckets — more recall per probe ⇒ lower nprobe ⇒ fewer docs visited (the only structural lever on
  # `visited`, §18). Write-only (reader unchanged); index grows ≈(1+spillBits)×. Sweep e.g. (0,1,2,3).
  "lshSpillBits": (1,),
  # SOAR spill selection (ScaNN, Sun et al. 2023): when True AND lshSpillBits>0, choose spill buckets by
  # the anisotropic residual loss (steer spills toward query directions the home bucket mis-scores)
  # instead of the lowest-|projection| geometry. Same K and on-disk layout as plain spilling, so it is
  # merge-stable and matched on INDEX SIZE — a clean A/B for "does SOAR reach matched recall at lower
  # nprobe?". No-op unless lshSpillBits>0. Sweep e.g. (False, True).
  "lshSoar": (False,),
  # SOAR orthogonality weight lambda (used only when lshSoar). ScaNN reports robustness ~1.0-1.5;
  # 0 ~= plain nearest-centroid spill selection. Sweep e.g. (0.5, 1.0, 1.5).
  "lshSoarLambda": (1,),
  # When True, DROP the stored full-precision originals (.vec/.vemf written empty) via
  # -Dlsh.storeRawVectors=false. The raw float32 copy is the DOMINANT merge cost at >RAM scale (~80% of
  # merge I/O: ~410GB at 100M×1024d, vs ~105GB×(1+spill)×tables of quantized postings the concat path
  # actually needs), so dropping it is the structural force-merge lever. SAFE ONLY when: rerank is off
  # (lshRerankFactor=1 — §10 measured rerank inert on this data), queries are UNFILTERED (the exact-search
  # fallback needs originals), and you don't run CheckIndex on the result. Concat merge + quantized search
  # are unaffected. Write-only, persisted per field (VERSION_NO_RAW). Sweep e.g. (False, True). See §22.
  "lshNoRaw": (True,),
  # Per-segment LSH search parallelism: number of threads the reader uses to scan ONE segment's selected
  # buckets per query (1 = sequential, the original path). The bucket scan is embarrassingly parallel and
  # parallelizes the profiled-dominant per-bucket query-quantize (§11) — the one warm-latency lever HNSW's
  # sequential graph walk can't pull within a segment (§13/§17). SEARCH-time only: the index is unchanged,
  # so it is NOT in the index cache key and the same index is reused across thread counts. Results are
  # bit-identical to sequential (deterministic min-position merge). Sweep e.g. (1,2,4,8) for a latency A/B.
  "lshSearchThreads": (1,),
  # GCUT (graph-cut IVF) partitioner knobs (ignored unless indexType="gcut"; write-time -> each value
  # reindexes, and all are in the index key). Swept like any other PARAM.
  #   gcutMode: GCUT_FREEZE (keep the cut, recompute centroids only) | GCUT_SEED (seed cut + Lloyd refine)
  #   gcutSplitBy: "variance" (bisect largest Sum||x-mu||^2 -> shrinks cell radius / NN-cell-rank tail)
  #                | "count" (bisect largest population -> balanced posting lists)
  #   gcutBalancePenalty: lambda in density*(1+lambda*imbalance^2); higher => more compact children
  #   gcutAxes: K principal axes tried per split (deepest valley wins); K>1 costs Kx per split
  "gcutMode": ("GCUT_FREEZE",),
  "gcutSplitBy": ("variance",),
  "gcutBalancePenalty": (4.0,),
  "gcutAxes": (1,2,),
  # HNSW params (ignored for ivf runs); defaults maxConn=16, beamWidth=100 for good recall.
  "maxConn": _env_tuple("KNN_MAXCONN", (16,)),
  "beamWidthIndex": _env_tuple("KNN_BEAM_WIDTH", (100,)),
  # fanout is SEARCH-time (efSearch = topK + fanout), so sweeping it reuses ONE graph -- the cheap recall
  # lever benchmarks.md 7 flagged as "not run".
  "fanout": _env_tuple("KNN_FANOUT", (100,)),
  "numSearchThread": (1,),
  "encoding": ("float32",),
  "metric": ("dot_product",),
  # 8-bit scalar-quantized HNSW, for a fair int8-vs-int8 comparison against the int8 lloyd_ivf codec.
  "quantizeBits": (8,),
  "topK": (100,),
  "forceMerge": (True,),
  "nquery": (1000,),
}


# Env overrides for the knobs a >RAM sweep varies between its BUILD and SEARCH phases, so one PARAMS
# block serves both without hand-editing between them (an edit mid-sweep is how a "40M spill=10" run
# silently becomes a 1M spill=2 one). Each parses a comma-separated list into the tuple PARAMS wants.
#   KNN_NDOC=39767748  KNN_NLIST=100000  KNN_SPILL_BITS=10  KNN_NPROBE=40,55,70,90,120
# ndoc/nlist/spillBits are WRITE-time (in the index key => reindex per value); nprobe is search-time,
# so ONE cached index serves the whole nprobe sweep.
def _env_int_tuple(name):
  raw = os.environ.get(name)
  if not raw:
    return None
  return tuple(int(x) for x in raw.replace(" ", "").split(",") if x)


for _env_name, _param in (
  ("KNN_NDOC", "ndoc"),
  ("KNN_NLIST", "ivfNlist"),
  ("KNN_SPILL_BITS", "ivfSpillBits"),
  ("KNN_NPROBE", "ivfNprobe"),
  ("KNN_NQUERY", "nquery"),
  ("KNN_FLUSH_ITERS", "ivfFlushIters"),
):
  _vals = _env_int_tuple(_env_name)
  if _vals:
    PARAMS[_param] = _vals
    print(f"{_env_name}: {_param} = {_vals}")


OUTPUT_HEADERS = [
  "recall",
  "latency(ms)",
  "netCPU",
  "avgCpuCount",
  "nDoc",
  "searchType",
  "topK",
  "fanout",
  "resultSimilarity",
  "decay",
  "resultCount",
  "maxConn",
  "beamWidth",
  "quantized",
  "visited",
  "index(s)",
  "index_docs/s",
  "merge(s)",
  "force_merge(s)",
  "num_segments",
  "index_size(MB)",
  "filterStrategy",
  "filterSelectivity",
  "overSample",
  "vec_disk(MB)",
  "vec_RAM(MB)",
  "bp-reorder",
  "indexType",
  "rerank",
]
# TODO:  "bp",

# output metrics (right-hand side of table); everything else is a hyperparameter
_METRIC_HEADERS = {"recall", "latency(ms)", "netCPU", "avgCpuCount"}

_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"


def _try_float(s):
  try:
    return float(s)
  except (ValueError, TypeError):
    return None


def advance(ix, values):
  for i in reversed(range(len(ix))):
    # scary to rely on dict key enumeration order?  but i guess if dict never changes while we do this, it's stable?
    param = list(values.keys())[i]
    # print("advance " + param)
    if type(values[param]) in (list, tuple) and ix[i] == len(values[param]) - 1:
      ix[i] = 0
    else:
      ix[i] += 1
      return True
  return False


_SPARK_CHARS = " ▁▂▃▄▅▆▇█"
_NUM_SPARK_BINS = 20
_NUM_DIM_SAMPLE_VECS = 2000
_THRESH_CONSTANT_STD = 1e-6
# fraction of samples equal to a single dominant value above which a dim is "near-constant"
# (degenerate-but-not-quite-CONSTANT: a few stray values keep std above _THRESH_CONSTANT_STD,
# but the dim still carries almost no information)
_THRESH_NEAR_CONSTANT_DOMINANT_FRAC = 0.95
_THRESH_SPARSE_PCT_ZEROS = 0.50
_THRESH_SKEWED_ABS = 1.0
_THRESH_HEAVY_TAILS_KURTOSIS = 3.0
_THRESH_FLAT_KURTOSIS = -1.0
_THRESH_OUTLIER_SPREAD_SIGMA = 3.0
# isotropy participation ratio below this triggers a WARNING; values closer to 1.0 are isotropic
_THRESH_ANISOTROPIC = 0.5


def _sparklines_2row(counts):
  """Returns (top_row_str, bot_row_str) for a two-row histogram using unicode block chars.

  Block chars fill from the bottom of the cell, so with 8 levels per row the two rows
  connect seamlessly: a partial char in the top row sits at the bottom of its cell,
  flush against the full block below it.  Total resolution: 16 height levels.
  """
  max_count = max(counts) if max(counts) > 0 else 1
  top_chars = []
  bot_chars = []
  for c in counts:
    level = round(c / max_count * 16)
    if level <= 8:
      bot_chars.append(_SPARK_CHARS[level])
      top_chars.append(" ")
    else:
      bot_chars.append("█")
      top_chars.append(_SPARK_CHARS[level - 8])
  return "".join(top_chars), "".join(bot_chars)


def _print_dim_line(d, mean, std, pct_zeros, counts, labels, dim_idx_width):
  """Prints sparkline + stats, with any smell labels (with details) on separate lines below."""
  top_row, bot_row = _sparklines_2row(counts)
  prefix = f"  dim {d:0{dim_idx_width}d} μ={mean:+8.3f} σ={std:8.3f} zeros={pct_zeros * 100:3.0f}% "  # noqa: RUF001 sigma (std deviation) is intentional
  print(f"{' ' * len(prefix)}[{top_row}]")
  print(f"{prefix}[{bot_row}]")
  indent = f"  {' ' * dim_idx_width}  "
  for name, detail in labels:
    print(f"{indent}-> {name}: {detail}")
  print()


def _read_vectors_pread(file_name, sample_indices, vec_size_bytes):
  """Generator that issues concurrent readahead hints and yields vectors via pread."""
  # os.posix_fadvise is Linux-only; on macOS/Windows it is absent, so the
  # readahead hints are simply skipped (pread still works correctly).
  have_fadvise = hasattr(os, "posix_fadvise")
  with open(file_name, "rb") as f:
    fd = f.fileno()

    # hint random access for the whole file, to suppress wasteful readahead
    if have_fadvise:
      os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)

      # concurrently send all requests to the OS as hints
      for vec_idx in sample_indices:
        os.posix_fadvise(fd, vec_idx * vec_size_bytes, vec_size_bytes, os.POSIX_FADV_WILLNEED)

    # yield vectors; they should be pre-fetched by the kernel
    for vec_idx in sample_indices:
      yield vec_idx, np.frombuffer(os.pread(fd, vec_size_bytes, vec_idx * vec_size_bytes), dtype="<f4")

    # hint sequential access for the subsequent (sequential) indexing test
    if have_fadvise:
      os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)


def _read_vectors_mmap(file_name, sample_indices, vec_size_bytes, dim):
  """Generator that issues concurrent readahead hints and yields vectors via mmap."""
  with open(file_name, "rb") as f:
    # mmap.PAGESIZE is typically 4096 on Linux
    pagesize = mmap.PAGESIZE

    # map the entire file
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    # hint random access to the whole mmap, to suppress wasteful readahead
    mm.madvise(mmap.MADV_RANDOM)

    try:
      # concurrently send all requests to the OS as hints
      for vec_idx in sample_indices:
        offset = vec_idx * vec_size_bytes
        page_offset = (offset // pagesize) * pagesize
        length = offset + vec_size_bytes - page_offset
        mm.madvise(mmap.MADV_WILLNEED, page_offset, length)

      # yield vectors; they should be pre-fetched by the kernel
      for vec_idx in sample_indices:
        offset = vec_idx * vec_size_bytes
        yield vec_idx, np.frombuffer(mm, dtype="<f4", count=dim, offset=offset)

      # hint sequential access for the subsequent (sequential) indexing test
      mm.madvise(mmap.MADV_SEQUENTIAL)
    finally:
      mm.close()

  print()


def _check_dim_distributions(dim, file_name, num_vectors, vec_size_bytes):
  """Samples vectors and computes per-dim statistics to detect degenerate dimensions."""
  if num_vectors == 0:
    return

  num_sample = min(_NUM_DIM_SAMPLE_VECS, num_vectors)
  sample_indices = random.sample(range(num_vectors), num_sample)

  if NOISY:
    print(f"smell: sampling {num_sample} of {num_vectors} vectors for per-dim distribution...")

  # load all sampled vectors concurrently into a (num_sample, dim) float32 array
  samples = np.empty((num_sample, dim), dtype=np.float32)
  t0_sec = time.monotonic()

  if IO_METHOD == "pread":
    reader = _read_vectors_pread(file_name, sample_indices, vec_size_bytes)
  elif IO_METHOD == "mmap":
    reader = _read_vectors_mmap(file_name, sample_indices, vec_size_bytes, dim)
  else:
    raise ValueError(f'unknown IO_METHOD "{IO_METHOD}"')

  not_norm_count = 0
  for i, (vec_idx, vec) in enumerate(reader):
    samples[i] = vec

    # CPU work: check norm immediately as vector arrives
    norm = np.linalg.norm(vec)
    if not np.isclose(norm, 1.0, rtol=0.0001, atol=0.0001):
      # print warning on new line so it doesn't get overwritten by next progress \r
      print(f'\nWARNING: vec {vec_idx} in "{file_name}" has norm={norm} (not normalized)')
      not_norm_count += 1

    completed = i + 1
    if completed % max(1, num_sample // 20) == 0 or completed == num_sample:
      elapsed_sec = time.monotonic() - t0_sec
      if NOISY:
        print(f"\rsmell:   {completed}/{num_sample} ({100 * completed / num_sample:3.0f}%) {elapsed_sec:.1f}s", end="", flush=True)

  if NOISY:
    print()

  if not_norm_count > 0:
    print(f'WARNING: dimension or vector file name might be wrong?  {not_norm_count} of {num_sample} randomly checked vectors are not normalized in "{file_name}"')

  # per-dim stats, all vectorized over axis=0, result shape: (dim,)
  mean = samples.mean(axis=0)
  std = samples.std(axis=0)
  pct_zeros = (samples == 0.0).mean(axis=0)

  # per-dim "most-frequent exact value" + its fraction. catches dims that are essentially
  # a single value (e.g. 0.0) but not flagged as CONSTANT because a few stray samples keep
  # std just above _THRESH_CONSTANT_STD. uses exact float equality (no quantization).
  dominant_val = np.zeros(dim, dtype=np.float32)
  dominant_frac = np.zeros(dim, dtype=np.float64)
  for d in range(dim):
    vals, counts = np.unique(samples[:, d], return_counts=True)
    top = counts.argmax()
    dominant_val[d] = vals[top]
    dominant_frac[d] = counts[top] / num_sample

  centered = samples - mean
  m3 = (centered**3).mean(axis=0)
  m4 = (centered**4).mean(axis=0)
  with np.errstate(invalid="ignore", divide="ignore"):
    skewness = m3 / std**3
    excess_kurtosis = m4 / std**4 - 3.0
  # zero out stats that are undefined for constant dims
  non_const = std > _THRESH_CONSTANT_STD
  skewness = np.where(non_const, skewness, 0.0)
  excess_kurtosis = np.where(non_const, excess_kurtosis, 0.0)

  # Isotropy = participation ratio of the covariance spectrum, in [0, 1]:
  #   PR = (tr C)^2 / (dim * ||C||_F^2) = (sum eigvals)^2 / (dim * sum eigvals^2)
  # Roughly: "what fraction of dims does the data actually use?" 1.0 = energy spread
  # evenly across all dims (isotropic, no preconditioning needed); 1/dim = data lives
  # on a single direction (highly anisotropic).
  #
  # We compute it two ways:
  #   - isotropy_full uses the real covariance C = X^T X / N (captures cross-dim
  #     correlations -- the "true" answer)
  #   - isotropy_diag pretends C is diagonal, using only per-dim variances (cheap;
  #     blind to correlations)
  # Always isotropy_full <= isotropy_diag. A large gap means anisotropy is rotational
  # (data on a low-dim subspace at an angle to coord axes) rather than axis-aligned.
  variances = std**2
  sum_var = variances.sum()
  sum_var_sq = (variances**2).sum()
  if sum_var_sq > 0:
    isotropy_diag = (sum_var**2) / (dim * sum_var_sq)
  else:
    isotropy_diag = 0.0

  C = (centered.T @ centered) / num_sample
  tr_C = np.trace(C)
  frob_sq = np.sum(C**2)
  if frob_sq > 0:
    isotropy_full = (tr_C**2) / (dim * frob_sq)
  else:
    isotropy_full = 0.0

  # OUTLIER_SPREAD: flag dims whose std is an outlier across all dims' stds
  mean_of_stds = std.mean()
  std_of_stds = std.std()

  # use global range so histogram widths are directly comparable across dims:
  # a dim with larger sigma spans more bins visually, matching the printed sigma
  global_min = float(samples.min())
  global_max = float(samples.max())

  # per-dim histogram (loop is over dims not over samples, so it's cheap)
  dim_counts = []
  for d in range(dim):
    col = samples[:, d]
    if col.min() == col.max():
      counts = [0] * _NUM_SPARK_BINS
      counts[_NUM_SPARK_BINS // 2] = num_sample
    else:
      counts = np.histogram(col, bins=_NUM_SPARK_BINS, range=(global_min, global_max))[0].tolist()
    dim_counts.append(counts)

  # assign labels per dim -- each label is (name, detail_string)
  dim_labels = []
  for d in range(dim):
    labels = []

    if std[d] <= _THRESH_CONSTANT_STD:
      labels.append(("CONSTANT", f"std={std[d]:.6f}, threshold={_THRESH_CONSTANT_STD}"))
    elif dominant_frac[d] >= _THRESH_NEAR_CONSTANT_DOMINANT_FRAC:
      labels.append(
        (
          "NEAR_CONSTANT",
          f"value={float(dominant_val[d]):+g} occurs in {dominant_frac[d] * 100:.1f}% of samples, threshold={_THRESH_NEAR_CONSTANT_DOMINANT_FRAC * 100:.0f}%",
        )
      )

    if pct_zeros[d] > _THRESH_SPARSE_PCT_ZEROS:
      labels.append(("SPARSE", f"{pct_zeros[d] * 100:.1f}% zeros, threshold={_THRESH_SPARSE_PCT_ZEROS * 100:.0f}%"))

    if non_const[d]:
      if abs(skewness[d]) > _THRESH_SKEWED_ABS:
        labels.append(("SKEWED", f"skew={skewness[d]:+.2f}, threshold=|{_THRESH_SKEWED_ABS}|"))
      if excess_kurtosis[d] > _THRESH_HEAVY_TAILS_KURTOSIS:
        labels.append(("HEAVY_TAILS", f"kurtosis={excess_kurtosis[d]:+.2f}, threshold>{_THRESH_HEAVY_TAILS_KURTOSIS}"))
      if excess_kurtosis[d] < _THRESH_FLAT_KURTOSIS:
        labels.append(("FLAT", f"kurtosis={excess_kurtosis[d]:+.2f}, threshold<{_THRESH_FLAT_KURTOSIS}"))

    if std_of_stds > 0 and abs(std[d] - mean_of_stds) > _THRESH_OUTLIER_SPREAD_SIGMA * std_of_stds:
      z = (std[d] - mean_of_stds) / std_of_stds
      labels.append(("OUTLIER_SPREAD", f"this_std={std[d]:.4f}, mean_std={mean_of_stds:.4f}, {z:+.1f}sigma vs threshold={_THRESH_OUTLIER_SPREAD_SIGMA}sigma"))

    dim_labels.append(labels)

  # format and print output
  dim_idx_width = len(str(dim - 1))
  bad_dims = [d for d in range(dim) if len(dim_labels[d]) > 0]

  elapsed_sec = time.monotonic() - t0_sec

  # Estimate effective rank for human-readable context: full*dim ~= number of dims actually used
  eff_rank_full = isotropy_full * dim
  eff_rank_diag = isotropy_diag * dim
  print(f"smell: isotropy={isotropy_full:.3f} full-cov, {isotropy_diag:.3f} diag-only (effective rank ~{eff_rank_full:.1f}/{dim} full, ~{eff_rank_diag:.1f}/{dim} diag)")
  print("  isotropy = (tr C)^2 / (D * ||C||_F^2): 1.0 = energy spread evenly across all dims; 1/D = data on one direction")
  print("  full-cov uses real covariance (sees cross-dim correlations); diag-only uses per-dim variances (blind to correlations)")
  if isotropy_full < _THRESH_ANISOTROPIC:
    if isotropy_diag - isotropy_full > 0.2:
      cause = "ROTATED anisotropy: per-dim variances look balanced but data lives in a lower-dim subspace at an angle to coord axes"
      fix = "consider PCA or a random rotation before quantization/HNSW"
    else:
      cause = "AXIS-ALIGNED anisotropy: a few dims carry most of the variance"
      fix = "consider per-dim standardization (subtract mean, divide by std)"
    print(f"  WARNING: anisotropic vectors (isotropy_full={isotropy_full:.3f} < {_THRESH_ANISOTROPIC})")
    print(f"           {cause}")
    print(f"           {fix}")

  # tally degenerate-or-near-degenerate dims (CONSTANT + NEAR_CONSTANT) -- these are
  # dims that effectively carry no information and are wasted ambient axes
  num_constant = int(sum(1 for lbs in dim_labels for name, _ in lbs if name == "CONSTANT"))
  num_near_constant = int(sum(1 for lbs in dim_labels for name, _ in lbs if name == "NEAR_CONSTANT"))
  if num_constant > 0 or num_near_constant > 0:
    print(f"smell: {num_constant + num_near_constant}/{dim} dims are degenerate or near-degenerate (carry ~no information): {num_constant} CONSTANT, {num_near_constant} NEAR_CONSTANT")

  if bad_dims:
    print(f"smell: {len(bad_dims)} degenerate dim(s) found in {elapsed_sec:.1f}s:")
    if any(name == "OUTLIER_SPREAD" for lbs in dim_labels for name, _ in lbs):
      print(f"  (OUTLIER_SPREAD: std of all dims: μ={mean_of_stds:.3f}, σ={std_of_stds:.3f}, threshold={_THRESH_OUTLIER_SPREAD_SIGMA}σ)")  # noqa: RUF001 sigma (std deviation) is intentional

    for d in bad_dims:
      _print_dim_line(d, float(mean[d]), float(std[d]), float(pct_zeros[d]), dim_counts[d], dim_labels[d], dim_idx_width)

    if NOISY:
      print(f"smell: all {dim} dims:")
      for d in range(dim):
        _print_dim_line(d, float(mean[d]), float(std[d]), float(pct_zeros[d]), dim_counts[d], dim_labels[d], dim_idx_width)
  elif NOISY:
    print(f"smell: no degenerate dims found in {elapsed_sec:.1f}s")


def _load_id_samples(dim, file_name, num_vectors, vec_size_bytes):
  """Load a fresh, larger sample of vectors for intrinsic-dim estimation.

  Kept separate from the histogram sample (which only needs ~2k vectors) so
  the histogram cost doesn't grow with the bigger ID sample size.  Re-uses
  the same posix_fadvise/pread or mmap path as the histogram loader.
  """
  num_sample = min(_NUM_INTRINSIC_DIM_SAMPLE_VECS, num_vectors)
  sample_indices = random.sample(range(num_vectors), num_sample)

  if NOISY:
    print(f"smell ID: sampling {num_sample} of {num_vectors} vectors for intrinsic-dim estimation...")

  samples = np.empty((num_sample, dim), dtype=np.float32)
  t0_sec = time.monotonic()

  if IO_METHOD == "pread":
    reader = _read_vectors_pread(file_name, sample_indices, vec_size_bytes)
  elif IO_METHOD == "mmap":
    reader = _read_vectors_mmap(file_name, sample_indices, vec_size_bytes, dim)
  else:
    raise ValueError(f'unknown IO_METHOD "{IO_METHOD}"')

  for i, (_vec_idx, vec) in enumerate(reader):
    samples[i] = vec
    completed = i + 1
    if completed % max(1, num_sample // 20) == 0 or completed == num_sample:
      elapsed_sec = time.monotonic() - t0_sec
      if NOISY:
        print(f"\rsmell ID:   {completed}/{num_sample} ({100 * completed / num_sample:3.0f}%) {elapsed_sec:.1f}s", end="", flush=True)

  if NOISY:
    print()

  return samples


def _twonn_estimate(D2):
  """TwoNN intrinsic-dim estimator (Facco, d'Errico, Rodriguez, Laio 2017).

  Theory: if data is sampled uniformly from a smooth manifold of intrinsic
  dimension d, then for each point the ratio mu = r2 / r1 of its 2nd to 1st
  nearest-neighbor distances follows a Pareto distribution:

      P(mu) = d * mu^(-d - 1)         for mu >= 1
      F(mu) = 1 - mu^(-d)             (CDF)
      => -log(1 - F(mu)) = d * log(mu)

  So the slope of -log(1 - F_hat) regressed on log(mu) IS the intrinsic
  dimension.  Beautiful trick: the per-point density cancels out (because
  r1 and r2 are at the same point), so this works even when sampling
  density varies wildly across the manifold.

  We discard the upper tail (Facco et al. recommend ~10%) because very
  large mu values come from points near density boundaries and outliers,
  which bend the regression.

  Returns dict with keys: d, r2, n_kept, n_dup, n_used.
    d      = estimated intrinsic dimension
    r2     = R^2 of the linear fit (1.0 = perfect Pareto, lower = data is
             not really a single manifold)
    n_kept = points kept after tail discard
    n_dup  = points dropped because r1 == 0 (exact duplicates)
    n_used = points with valid r1, r2 before tail discard
  """
  N = D2.shape[0]

  # 2nd and 3rd smallest of each row of D2 (the 1st-smallest is the diagonal,
  # which we set to inf upstream).  partition is O(N) per row so this is
  # O(N^2) total, dwarfed by the matmul that produced D2.
  partitioned = np.partition(D2, 1, axis=1)
  r1_sq = partitioned[:, 0]
  r2_sq = partitioned[:, 1]

  # numerical guard: tiny negatives from float roundoff in the matmul
  r1_sq = np.maximum(r1_sq, 0.0)
  r2_sq = np.maximum(r2_sq, 0.0)
  r1 = np.sqrt(r1_sq)
  r2 = np.sqrt(r2_sq)

  # drop points whose nearest neighbor is a duplicate (r1 == 0): mu is
  # undefined and probably indicates corpus-level duplication.  count them
  # so the caller can warn.
  is_dup = r1 == 0.0
  n_dup = int(is_dup.sum())
  valid = ~is_dup
  if valid.sum() < 100:
    raise RuntimeError(f"TwoNN: too few non-duplicate points ({int(valid.sum())} of {N}); the corpus may be heavily duplicated or the sample too small")

  mu = r2[valid] / r1[valid]
  # mu must be >= 1 by construction (r2 >= r1); float roundoff can produce
  # mu just under 1 -- clamp.
  mu = np.maximum(mu, 1.0)
  n_used = int(mu.size)

  # discard upper tail
  mu_sorted = np.sort(mu)
  n_keep = int(round(n_used * (1.0 - _TWONN_DISCARD_FRAC)))
  if n_keep < 50:
    raise RuntimeError(f"TwoNN: too few kept points ({n_keep}); sample size too small")
  mu_kept = mu_sorted[:n_keep]

  # empirical CDF: F_hat_i = i / (N+1) for i = 1..N (avoids log(0) at the top)
  i_arr = np.arange(1, n_keep + 1, dtype=np.float64)
  f_hat = i_arr / (n_keep + 1)

  x = np.log(mu_kept)
  y = -np.log(1.0 - f_hat)

  # linear fit forced through origin (the model is y = d * x exactly when
  # the Pareto holds).  drop x==0 (mu==1 points) to avoid degeneracies.
  nz = x > 0
  x = x[nz]
  y = y[nz]
  if x.size < 50:
    raise RuntimeError(f"TwoNN: too few non-degenerate points ({x.size}) after dropping mu==1")

  d_hat = float(np.sum(x * y) / np.sum(x * x))

  # R^2 vs the through-origin fit
  y_pred = d_hat * x
  ss_res = float(np.sum((y - y_pred) ** 2))
  ss_tot = float(np.sum((y - np.mean(y)) ** 2))
  r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

  return {"d": d_hat, "r2": r_squared, "n_kept": n_keep, "n_dup": n_dup, "n_used": n_used}


def _mle_estimate(D2, k):
  """MLE Levina-Bickel intrinsic-dim estimator (2004) at neighborhood size k.

  Theory: model the local point process around each query point as a Poisson
  process with intensity rho * r^(d-1).  The MLE for d given the k nearest
  neighbor distances r_1 < r_2 < ... < r_k is:

      d_hat_k(x) = (k - 1) / sum_{j=1}^{k-1} log(r_k / r_j)

  This is a per-point estimate; we report mean and median over all sample
  points.  The harmonic-mean form makes it scale-invariant (good) but
  biased high under non-uniform sampling density (bad).  Comparing across
  k=10 vs k=20 reveals stability.
  """
  N = D2.shape[0]
  if k + 1 > N:
    raise RuntimeError(f"MLE: k={k} requires N>{k}; have N={N}")

  # per-row top-k smallest distances (excluding self, which is inf)
  partitioned = np.partition(D2, k - 1, axis=1)[:, :k]
  partitioned = np.sort(partitioned, axis=1)
  partitioned = np.maximum(partitioned, 0.0)
  r = np.sqrt(partitioned)  # shape (N, k); r[:, 0]..r[:, k-1]

  r_k = r[:, k - 1]  # shape (N,)
  r_inner = r[:, : k - 1]  # shape (N, k-1)

  # need r_k > 0 and all r_inner > 0; drop rows that fail
  r_inner_min = r_inner.min(axis=1)
  valid = (r_k > 0) & (r_inner_min > 0)
  if valid.sum() < 100:
    raise RuntimeError(f"MLE k={k}: too few non-degenerate points ({int(valid.sum())} of {N})")

  log_ratios = np.log(r_k[valid, None] / r_inner[valid])  # (M, k-1)
  inv_d = log_ratios.sum(axis=1) / (k - 1)
  # inv_d must be > 0 (since r_k >= r_j); guard against numerical zero
  good = inv_d > 0
  if good.sum() < 100:
    raise RuntimeError(f"MLE k={k}: too few rows with positive log-ratio sum ({int(good.sum())})")
  d_per_point = 1.0 / inv_d[good]

  return {"mean": float(d_per_point.mean()), "median": float(np.median(d_per_point)), "n": int(d_per_point.size)}


def _pca_id(samples, thresholds=(0.90, 0.95, 0.99)):
  """Linear ID via cumulative variance of the centered covariance.

  Returns a dict {threshold -> num_components_needed}.  This is the LINEAR
  intrinsic dim -- minimum number of orthogonal axes that span X% of the
  total variance.  Comparing PCA-95% to TwoNN reveals manifold curvature:
  if PCA-95% >> TwoNN, the data lives on a curved low-D manifold inside
  a higher-D linear subspace.
  """
  centered = samples - samples.mean(axis=0)
  # use SVD on centered data for numerical stability vs eigh of X^T X
  # (squared-condition-number is bad when some dims are tiny)
  s = np.linalg.svd(centered, compute_uv=False)
  variances = (s**2) / max(samples.shape[0] - 1, 1)
  total = variances.sum()
  if total <= 0:
    raise RuntimeError("PCA ID: total variance is zero; samples are constant")
  cumvar = np.cumsum(variances) / total

  result = {}
  for thresh in thresholds:
    # smallest k such that cumvar[k-1] >= thresh
    k = int(np.searchsorted(cumvar, thresh) + 1)
    k = min(k, len(cumvar))
    result[thresh] = k
  return result


def _estimate_intrinsic_dim(samples, label, dim):
  """Compute and report intrinsic-dimensionality estimates for `samples`.

  Runs three estimators (TwoNN, MLE-Levina-Bickel at multiple k, PCA-cumvar)
  and prints a compact report plus interpretation guidance.  Issues loud
  WARNINGs when the data looks pathological for KNN benchmarking.

  See module-level docstring on DO_INTRINSIC_DIM for the why; the function
  bodies _twonn_estimate / _mle_estimate / _pca_id document the math for
  each estimator.

  Args:
    samples: (N, D) float32 array of sample vectors
    label:   "docs" or "queries", used in printed report
    dim:     ambient dimension (== samples.shape[1])

  """
  N = samples.shape[0]
  assert samples.shape[1] == dim, f"shape mismatch: samples.shape[1]={samples.shape[1]} vs dim={dim}"

  print(f"\nsmell ID: estimating intrinsic dim of {label} vectors (N={N}, ambient dim={dim})...")
  t0_sec = time.monotonic()

  # all-pairs squared euclidean distance via BLAS.  for unit-normalized
  # vectors euclidean ranking is monotone-equivalent to angular ranking and
  # to dot-product ranking, so a single distance covers all metrics in
  # use here (smell_vectors already enforced near-unit norm upstream).
  #
  # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 <a, b>
  sq_norms = (samples * samples).sum(axis=1)
  D2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (samples @ samples.T)
  # numerical guard for tiny negatives from float roundoff
  np.maximum(D2, 0.0, out=D2)
  # exclude self-distance from NN search by setting diagonal to +inf
  np.fill_diagonal(D2, np.inf)

  matmul_sec = time.monotonic() - t0_sec
  if NOISY:
    print(f"smell ID:   distance matrix ({N}x{N}) in {matmul_sec:.1f}s")

  # --- TwoNN ---
  twonn = _twonn_estimate(D2)

  # --- MLE Levina-Bickel ---
  mle = {}
  for k in _MLE_K_VALUES:
    mle[k] = _mle_estimate(D2, k)

  # free distance matrix before PCA (PCA only needs samples)
  del D2

  # --- PCA cumulative variance ---
  pca = _pca_id(samples)

  total_sec = time.monotonic() - t0_sec

  # --- report ---
  print(f"smell ID: results for {label} (computed in {total_sec:.1f}s):")
  print(f"  TwoNN:    d={twonn['d']:6.2f}  (R^2={twonn['r2']:.3f} over {twonn['n_kept']} points; {twonn['n_dup']} duplicates dropped)")
  for k in _MLE_K_VALUES:
    print(f"  MLE k={k:<3d} d_mean={mle[k]['mean']:6.2f}  d_median={mle[k]['median']:6.2f}  (over {mle[k]['n']} points)")
  pca_strs = []
  for thresh, k_needed in pca.items():
    pca_strs.append(f"d_{int(thresh * 100)}={k_needed}")
  print(f"  PCA:      {'  '.join(pca_strs)}   (linear ID; ambient={dim})")

  # --- interpretation block ---
  pca95 = pca.get(0.95)
  twonn_d = twonn["d"]
  curvature_ratio = pca95 / twonn_d if (pca95 is not None and twonn_d > 0) else float("inf")
  mle_max = max(m["mean"] for m in mle.values())
  mle_min = min(m["mean"] for m in mle.values())
  disagreement = max(mle_max / twonn_d, twonn_d / max(mle_min, 1e-9)) if twonn_d > 0 else float("inf")

  print("  -- interpretation --")
  print("  TwoNN d is the manifold (nonlinear) intrinsic dim; PCA-95% is the linear-subspace dim.")
  print(f"  PCA-95% / TwoNN = {curvature_ratio:.1f}x measures how curved the manifold is inside")
  print("  that linear subspace.  high ratio -> learned rotations (PCA, OPQ) help quantization;")
  print("  random rotations (RaBitQ) less so.  low ratio (~1) -> data is near-flat; random rotation suffices.")
  print(f"  TwoNN d~={twonn_d:.1f} predicts HNSW recall scaling roughly like dim-{int(round(twonn_d))} data:")
  print("  lower ID -> can afford smaller beamWidth, more aggressive quantization, fewer fanout;")
  print("  higher ID -> need richer beam search and conservative quantization.")

  # --- WARNINGs ---
  warnings = []

  if twonn["n_dup"] > _THRESH_DUPLICATES_FRAC * N:
    warnings.append(
      f"found {twonn['n_dup']} exact-duplicate sample points (>{_THRESH_DUPLICATES_FRAC * 100:.1f}% of {N}); "
      "duplicates inflate recall numbers and may indicate dataset corruption or a leaky train/test split"
    )

  if twonn["r2"] < _THRESH_TWONN_BAD_FIT_R2:
    warnings.append(
      f"TwoNN R^2={twonn['r2']:.3f} < {_THRESH_TWONN_BAD_FIT_R2}; "
      "the Pareto-tail model does not fit the data, which means the sample is not a single clean "
      "manifold (likely a mixture of clusters, heavy duplicates, or extreme outliers).  ID estimate "
      "is unreliable; the dataset may be a poor KNN benchmark."
    )

  if twonn_d < _THRESH_TWONN_DEGENERATE_ID:
    warnings.append(
      f"TwoNN d={twonn_d:.2f} < {_THRESH_TWONN_DEGENERATE_ID}; "
      "vectors are nearly collinear / degenerate.  HNSW and quantization will be trivially easy; "
      "benchmark numbers will not transfer to real-world embedding workloads."
    )

  if twonn_d > _THRESH_TWONN_NEAR_AMBIENT_FRAC * dim:
    warnings.append(
      f"TwoNN d={twonn_d:.1f} > {_THRESH_TWONN_NEAR_AMBIENT_FRAC} * ambient dim {dim}; "
      "vectors look like high-D noise (poorly trained or random embeddings).  benchmarks will be hard "
      "but in a way that does not reflect real embeddings, which typically have ID well below ambient."
    )

  if curvature_ratio > _THRESH_HIGH_CURVATURE_RATIO:
    warnings.append(
      f"PCA-95%/TwoNN ratio {curvature_ratio:.1f} > {_THRESH_HIGH_CURVATURE_RATIO}: highly curved manifold; "
      "expect quantization with random rotation (RaBitQ) to underperform learned rotation (PCA/OPQ)."
    )

  if disagreement > _THRESH_ID_DISAGREEMENT_RATIO:
    warnings.append(
      f"ID estimators disagree by {disagreement:.1f}x (TwoNN={twonn_d:.1f}, MLE range "
      f"[{mle_min:.1f}, {mle_max:.1f}]); treat reported ID as uncertain by ~2x.  "
      "this can happen when sampling density varies a lot across the manifold."
    )

  if warnings:
    print(f"\nWARNING: {len(warnings)} issue(s) flagged for {label} vectors:")
    for w in warnings:
      print(f"  WARNING: {w}")
  elif NOISY:
    print(f"smell ID: no issues flagged for {label} vectors")


def smell_vectors(dim, file_name, label):
  """Runs some simple sanity checks on the vector source file, because we don't store any
  self-describing metadata in the .vec source file.

  label: "docs" or "queries", used to label ID-estimation output.
  """
  size_bytes = os.path.getsize(file_name)

  vec_size_bytes = dim * 4

  # cool, i didn't know about divmod!
  num_vectors, leftover = divmod(size_bytes, vec_size_bytes)

  if leftover != 0:
    raise RuntimeError(
      f'vector file "{file_name}" cannot be dimension {dim}: its size is not a multiple of each vector\'s size in bytes ({vec_size_bytes}); wrong vector source file or dimensionality?'
    )

  if NOISY:
    print(f"smell vectors from {file_name}")

  _check_dim_distributions(dim, file_name, num_vectors, vec_size_bytes)

  if DO_INTRINSIC_DIM:
    samples = _load_id_samples(dim, file_name, num_vectors, vec_size_bytes)
    _estimate_intrinsic_dim(samples, label, dim)


# all candidate perf stat counters for SIMD validation -- probed at import time
# to discover which ones this CPU actually supports.
#
# two families:
#   fp_arith_inst_retired.*  -- counts float SIMD instructions (float32 vectors)
#   int_vec_retired.*        -- counts integer SIMD instructions (quantized vectors)
#                               available on Ice Lake+ (different naming on older CPUs)
_FP_SIMD_CANDIDATE_COUNTERS = (
  "fp_arith_inst_retired.scalar_single",
  "fp_arith_inst_retired.128b_packed_single",
  "fp_arith_inst_retired.256b_packed_single",
  "fp_arith_inst_retired.512b_packed_single",
)

_INT_SIMD_CANDIDATE_COUNTERS = (
  "int_vec_retired.128bit",
  "int_vec_retired.256bit",
  "int_vec_retired.512bit",
)

# label, counter name, weight (proportional to SIMD width so FP and INT are comparable)
_FP_SIMD_LEVEL_DEFS = (
  ("FP-scalar", "fp_arith_inst_retired.scalar_single", 1),
  ("FP-SSE", "fp_arith_inst_retired.128b_packed_single", 4),
  ("FP-AVX2", "fp_arith_inst_retired.256b_packed_single", 8),
  ("FP-AVX512", "fp_arith_inst_retired.512b_packed_single", 16),
)

_INT_SIMD_LEVEL_DEFS = (
  ("INT-SSE", "int_vec_retired.128bit", 4),
  ("INT-AVX2", "int_vec_retired.256bit", 8),
  ("INT-AVX512", "int_vec_retired.512bit", 16),
)


def _probe_perf_counter(counter):
  """Return True if perf recognizes this counter on the current CPU."""
  try:
    result = subprocess.run(
      [PERF_EXE, "stat", "-e", counter, "--", "true"],
      capture_output=True,
      text=True,
      timeout=5,
      check=False,
    )
    # perf exits 0 and prints the counter (possibly "<not counted>" for short
    # commands) when the counter is valid.  it exits non-zero or prints
    # "<not supported>" when the counter doesn't exist on this CPU.
    return result.returncode == 0 and "<not supported>" not in result.stderr
  except (subprocess.TimeoutExpired, OSError):
    return False


def _probe_simd_perf_counters():
  """Probe which SIMD perf counters this CPU supports.

  Probes each counter individually because perf refuses to run at all if any
  counter name is unrecognized (e.g. 512b on a non-AVX-512 CPU).

  Returns (available_counters, fp_levels, int_levels).
  """
  if PERF_EXE is None:
    return (), (), ()

  available = []
  for counter in _FP_SIMD_CANDIDATE_COUNTERS + _INT_SIMD_CANDIDATE_COUNTERS:
    if _probe_perf_counter(counter):
      available.append(counter)

  available_set = set(available)
  counters = tuple(available)
  fp_levels = tuple(t for t in _FP_SIMD_LEVEL_DEFS if t[1] in available_set)
  int_levels = tuple(t for t in _INT_SIMD_LEVEL_DEFS if t[1] in available_set)
  return counters, fp_levels, int_levels


# probe once at import time
_AVAILABLE_SIMD_COUNTERS, FP_SIMD_LEVELS, INT_SIMD_LEVELS = _probe_simd_perf_counters()
if _AVAILABLE_SIMD_COUNTERS:
  fp_names = [c for c in _AVAILABLE_SIMD_COUNTERS if c.startswith("fp_")]
  int_names = [c for c in _AVAILABLE_SIMD_COUNTERS if c.startswith("int_")]
  parts = []
  if fp_names:
    parts.append(f"FP: {', '.join(fp_names)}")
  if int_names:
    parts.append(f"INT: {', '.join(int_names)}")
  print(f"NOTE: perf SIMD counters available: {'; '.join(parts)}")
  if not int_names:
    print("NOTE: integer SIMD counters (int_vec_retired.*) not available on this CPU; quantized runs will only show FP counters")
elif DO_PERF_STAT_SIMD:
  print("WARNING: perf SIMD counters not available on this CPU; disabling SIMD validation")
  DO_PERF_STAT_SIMD = False


def wrap_cmd_with_perf_stat_simd(cmd, output_file):
  """Prepend perf stat with SIMD counters to a command, writing stats to output_file."""
  return [
    PERF_EXE,
    "stat",
    "-o",
    output_file,
    "-e",
    ",".join(_AVAILABLE_SIMD_COUNTERS),
    "--",
  ] + cmd


def parse_perf_stat_file(path):
  """Parse a perf stat output file and return dict of counter_name -> count (int).

  Returns empty dict if file is missing or unparseable.
  """
  result = {}
  try:
    text = Path(path).read_text()
  except OSError:
    return result
  for line in text.splitlines():
    # format: "     12,345,678      cpu_core/fp_arith_inst_retired.256b_packed_single/    (88.34%)"
    # or:     "     12345678      fp_arith_inst_retired.256b_packed_single"
    # or:     "     12345678      int_vec_retired.256bit"
    m = re.match(r"^\s+([\d,]+)\s+(?:\S+/)?((?:fp_arith_inst_retired|int_vec_retired)\.\S+?)(?:/|\s)", line)
    if m:
      count = int(m.group(1).replace(",", ""))
      counter = m.group(2)
      result[counter] = count
  return result


def _summarize_levels(counters, level_defs):
  """Compute per-level breakdown for a set of SIMD level defs.

  Returns (parts_list, total_ops, dominant_label) where parts_list has
  (label, count, ops) tuples for levels with count > 0.
  """
  parts = []
  total_ops = 0
  dominant_label = None
  dominant_ops = 0
  for label, counter, ops_per_insn in level_defs:
    count = counters.get(counter, 0)
    ops = count * ops_per_insn
    total_ops += ops
    if count > 0:
      parts.append((label, count, ops))
    if ops > dominant_ops:
      dominant_ops = ops
      dominant_label = label
  return parts, total_ops, dominant_label


def format_simd_report(counters, is_quantized=False):
  """Format a one-line SIMD usage summary from parsed perf stat counters.

  Combines FP (fp_arith_inst_retired) and integer (int_vec_retired) counters
  into a single flat percentage breakdown weighted by SIMD width, so all
  percentages sum to 100%.

  Returns (report_string, dominant_level) where dominant_level is
  "FP-scalar", "FP-SSE", "FP-AVX2", "FP-AVX512",
  "INT-SSE", "INT-AVX2", "INT-AVX512", or "none".
  """
  if not counters:
    return "SIMD: no perf data", "none"

  all_levels = list(FP_SIMD_LEVELS) + list(INT_SIMD_LEVELS)
  all_parts, grand_total, dominant = _summarize_levels(counters, all_levels)

  if grand_total == 0:
    if is_quantized and not INT_SIMD_LEVELS:
      return "SIMD: no FP instructions (expected for quantized); integer SIMD counters not available on this CPU", "none"
    return "SIMD: no instructions detected", "none"

  pct_parts = []
  for label, _count, ops in all_parts:
    pct_parts.append(f"{label} {100.0 * ops / grand_total:.1f}%")

  report = f"SIMD: {' | '.join(pct_parts)} (dominant: {dominant})"

  # warn only when the relevant family shows no SIMD
  if dominant == "FP-scalar" and not is_quantized:
    report += " WARNING: no SIMD vectorization detected!"
  elif is_quantized and not any(label.startswith("INT-") for label, _, _ in all_parts):
    if not INT_SIMD_LEVELS:
      report += " (int_vec_retired counters not available on this CPU)"
    else:
      report += " WARNING: no integer SIMD detected!"

  return report, dominant


def get_unique_log_name(log_path, sub_tool):
  log_dir_name, log_base_name = log_path
  upto = 0
  while True:
    log_file_name = f"{log_dir_name}/{log_base_name}-{sub_tool}"
    if upto > 0:
      log_file_name += f"-{upto}"
    log_file_name += ".log"
    if not os.path.exists(log_file_name):
      return log_file_name
    upto += 1


def print_run_summary(values):
  options = []
  fixed = []
  combos = 1

  # print these important params first, in this order:
  print_order = ["forceMerge", "ndoc", "nquery", "topK", "quantizeBits"]
  key_to_ord = {}
  other_keys = []

  max_key_len = 0

  for key, value in values.items():
    max_key_len = max(max_key_len, len(key))
    try:
      key_ord = print_order.index(key)
    except ValueError:
      other_keys.append(key)

  for key_ord, key in enumerate(print_order):
    key_to_ord[key] = key_ord
  other_keys.sort()
  for key_ord, key in enumerate(other_keys):
    key_to_ord[key] = len(print_order) + key_ord
  ord_to_key = [-1] * len(key_to_ord)
  for key, key_ord in key_to_ord.items():
    ord_to_key[key_ord] = key

  for key in ord_to_key:
    value = values[key]
    if len(value) > 1:
      # yay, it turns out you can nest {...} in f-strings!
      options.append(f"  {key:<{max_key_len}s}: {','.join(str(v) for v in value)}")
      combos *= len(value)
    else:
      # yay, it turns out you can nest {...} in f-strings!
      fixed.append(f"  {key:<{max_key_len}s}: {value[0]}")

  if len(options) == 0:
    assert combos == 1
    print("NOTE: will run single warmup+test with these params:")
  else:
    print(f"NOTE: will run {combos} total warmups+tests with all combinations of:")
    for s in options:
      print(s)
  for s in fixed:
    print(s)


METRIC_LABELS = {
  "dot_product": ("dot_product similarity", "higher --->"),
  "angular": ("dot_product similarity", "higher --->"),
  "cosine": ("cosine similarity", "higher --->"),
  "euclidean": ("euclidean distance", "<--- lower"),
  "mip": ("max inner product", "higher --->"),
}


def generate_exact_nn_histogram(scores_path, output_dir, log_base_name, metric=None):
  """Read the binary exact NN scores file and generate an HTML histogram using Google Charts."""
  if not os.path.exists(scores_path):
    print(f"WARNING: exact NN scores file not found: {scores_path}")
    return

  file_size = os.path.getsize(scores_path)
  num_floats = file_size // 4
  if num_floats == 0:
    print("WARNING: exact NN scores file is empty")
    return

  all_scores = struct.unpack(f"<{num_floats}f", Path(scores_path).read_bytes())

  metric_name, direction = METRIC_LABELS.get(metric or "", ("similarity score", ""))

  min_score = min(all_scores)
  max_score = max(all_scores)

  if min_score == max_score:
    print(f"WARNING: all exact NN scores are identical ({min_score}); skipping histogram")
    return

  # Sort scores so JS can use binary search for fast range queries
  sorted_scores = sorted(all_scores)

  # Emit as a compact JSON array (6 decimal places keeps file reasonable)
  scores_js_lines = []
  chunk_size = 500
  for i in range(0, len(sorted_scores), chunk_size):
    chunk = sorted_scores[i : i + chunk_size]
    scores_js_lines.append(",".join(f"{s:.6f}" for s in chunk))
  scores_js = ",\n".join(scores_js_lines)

  html = f"""<!DOCTYPE html>
<html>
  <head>
    <script type="text/javascript" src="https://www.google.com/jsapi"></script>
    <script type="text/javascript">
      google.load("visualization", "1", {{packages:["corechart"]}});
      google.setOnLoadCallback(init);

      // sorted scores for fast range queries via binary search
      var scores = [
{scores_js}
      ];
      var globalMin = {min_score:.6f};
      var globalMax = {max_score:.6f};
      var curMin = globalMin;
      var curMax = globalMax;
      var chart, chartDiv;
      var zoomStack = [];

      function lowerBound(arr, val) {{
        var lo = 0, hi = arr.length;
        while (lo < hi) {{
          var mid = (lo + hi) >> 1;
          if (arr[mid] < val) lo = mid + 1; else hi = mid;
        }}
        return lo;
      }}

      function buildHistogram(lo, hi, numBins) {{
        var range = hi - lo;
        var binWidth = range / numBins;
        var bins = new Array(numBins).fill(0);
        var startIdx = lowerBound(scores, lo);
        var endIdx = lowerBound(scores, hi);
        var count = 0;
        for (var i = startIdx; i < scores.length && scores[i] <= hi; i++) {{
          var idx = Math.floor((scores[i] - lo) / binWidth);
          if (idx >= numBins) idx = numBins - 1;
          if (idx < 0) idx = 0;
          bins[idx]++;
          count++;
        }}
        return {{bins: bins, binWidth: binWidth, count: count}};
      }}

      function drawChart(lo, hi) {{
        var numBins = 50;
        var h = buildHistogram(lo, hi, numBins);
        var rows = [['{metric_name} ({direction})', 'Count']];
        for (var i = 0; i < numBins; i++) {{
          var center = lo + (i + 0.5) * h.binWidth;
          rows.push([center, h.bins[i]]);
        }}
        var data = google.visualization.arrayToDataTable(rows);

        var zoomLabel = (lo === globalMin && hi === globalMax) ? '' : ' [zoomed]';
        var options = {{
          title: 'Exact NN {metric_name} distribution (' + h.count + ' of {num_floats} scores)' + zoomLabel,
          legend: {{position: 'none'}},
          hAxis: {{
            title: '{metric_name} ({direction})',
            viewWindow: {{min: lo, max: hi}}
          }},
          vAxis: {{title: 'Count'}},
          bar: {{groupWidth: '95%'}},
          chartArea: {{left: 80, right: 20, top: 40, bottom: 60}}
        }};

        chart.draw(data, options);
        document.getElementById('info').innerHTML =
          'Range: ' + lo.toFixed(6) + ' .. ' + hi.toFixed(6) +
          ' &nbsp; Bin width: ' + h.binWidth.toFixed(6) +
          ' &nbsp; Scores in view: ' + h.count;
      }}

      function init() {{
        chartDiv = document.getElementById('chart_div');
        chart = new google.visualization.ColumnChart(chartDiv);
        drawChart(globalMin, globalMax);

        // drag-to-zoom
        var dragStart = null;
        var overlay = document.getElementById('overlay');

        chartDiv.addEventListener('mousedown', function(e) {{
          var rect = chartDiv.getBoundingClientRect();
          dragStart = {{x: e.clientX - rect.left, clientX: e.clientX}};
          overlay.style.left = dragStart.x + 'px';
          overlay.style.width = '0px';
          overlay.style.display = 'block';
        }});

        chartDiv.addEventListener('mousemove', function(e) {{
          if (!dragStart) return;
          var rect = chartDiv.getBoundingClientRect();
          var curX = e.clientX - rect.left;
          var left = Math.min(dragStart.x, curX);
          var width = Math.abs(curX - dragStart.x);
          overlay.style.left = left + 'px';
          overlay.style.width = width + 'px';
        }});

        chartDiv.addEventListener('mouseup', function(e) {{
          if (!dragStart) return;
          overlay.style.display = 'none';
          var rect = chartDiv.getBoundingClientRect();
          var endX = e.clientX - rect.left;
          var x0 = Math.min(dragStart.x, endX);
          var x1 = Math.max(dragStart.x, endX);
          dragStart = null;

          // need at least 5px drag to count as zoom
          if (x1 - x0 < 5) return;

          var cli = chart.getChartLayoutInterface();
          var val0 = cli.getHAxisValue(x0);
          var val1 = cli.getHAxisValue(x1);
          if (val0 === null || val1 === null) return;
          var newMin = Math.max(Math.min(val0, val1), globalMin);
          var newMax = Math.min(Math.max(val0, val1), globalMax);
          if (newMax - newMin < (globalMax - globalMin) * 0.001) return;

          zoomStack.push({{min: curMin, max: curMax}});
          curMin = newMin;
          curMax = newMax;
          drawChart(curMin, curMax);
          document.getElementById('resetBtn').style.display = 'inline';
          document.getElementById('backBtn').style.display = 'inline';
        }});

        document.getElementById('resetBtn').addEventListener('click', function() {{
          zoomStack = [];
          curMin = globalMin;
          curMax = globalMax;
          drawChart(curMin, curMax);
          this.style.display = 'none';
          document.getElementById('backBtn').style.display = 'none';
        }});

        document.getElementById('backBtn').addEventListener('click', function() {{
          if (zoomStack.length === 0) return;
          var prev = zoomStack.pop();
          curMin = prev.min;
          curMax = prev.max;
          drawChart(curMin, curMax);
          if (zoomStack.length === 0) {{
            document.getElementById('resetBtn').style.display = 'none';
            this.style.display = 'none';
          }}
        }});
      }}
    </script>
    <style>
      #chart_container {{ position: relative; width: 1200px; height: 600px; }}
      #chart_div {{ width: 100%; height: 100%; }}
      #overlay {{ position: absolute; top: 0; height: 100%; background: rgba(66,133,244,0.15);
                  border-left: 1px solid rgba(66,133,244,0.5); border-right: 1px solid rgba(66,133,244,0.5);
                  display: none; pointer-events: none; z-index: 10; }}
      button {{ margin: 4px 4px 4px 0; padding: 4px 12px; }}
    </style>
  </head>
  <body>
    <div id="chart_container">
      <div id="chart_div"></div>
      <div id="overlay"></div>
    </div>
    <button id="backBtn" style="display:none">Back</button>
    <button id="resetBtn" style="display:none">Reset zoom</button>
    <p id="info"></p>
    <p style="color:#888">Click and drag on the chart to zoom in. Source: {scores_path}</p>
  </body>
</html>
"""
  output_file = f"{output_dir}/{log_base_name}-knnDistanceHistogram.html"
  Path(output_file).write_text(html)
  print(f"Wrote exact NN distance histogram to {output_file}")


def generate_all_distances_histogram(scores_path, output_dir, log_base_name, metric=None, sample_every_n=None):
  """Read the sampled all-distances binary file and generate an HTML histogram.

  The binary format is raw little-endian float32 scores (no per-query structure).
  """
  if not os.path.exists(scores_path):
    print(f"WARNING: all-distances scores file not found: {scores_path}")
    return

  file_size = os.path.getsize(scores_path)
  num_floats = file_size // 4
  if num_floats == 0:
    print("WARNING: all-distances scores file is empty")
    return

  all_scores = struct.unpack(f"<{num_floats}f", Path(scores_path).read_bytes())

  sample_label = ""
  if sample_every_n is not None:
    sample_label = f" (sampled 1-in-{sample_every_n})"

  metric_name, direction = METRIC_LABELS.get(metric or "", ("similarity score", ""))

  min_score = min(all_scores)
  max_score = max(all_scores)

  if min_score == max_score:
    print(f"WARNING: all distances scores are identical ({min_score}); skipping histogram")
    return

  sorted_scores = sorted(all_scores)

  scores_js_lines = []
  chunk_size = 500
  for i in range(0, len(sorted_scores), chunk_size):
    chunk = sorted_scores[i : i + chunk_size]
    scores_js_lines.append(",".join(f"{s:.6f}" for s in chunk))
  scores_js = ",\n".join(scores_js_lines)

  html = f"""<!DOCTYPE html>
<html>
  <head>
    <script type="text/javascript" src="https://www.google.com/jsapi"></script>
    <script type="text/javascript">
      google.load("visualization", "1", {{packages:["corechart"]}});
      google.setOnLoadCallback(init);

      var scores = [
{scores_js}
      ];
      var globalMin = {min_score:.6f};
      var globalMax = {max_score:.6f};
      var curMin = globalMin;
      var curMax = globalMax;
      var chart, chartDiv;
      var zoomStack = [];

      function lowerBound(arr, val) {{
        var lo = 0, hi = arr.length;
        while (lo < hi) {{
          var mid = (lo + hi) >> 1;
          if (arr[mid] < val) lo = mid + 1; else hi = mid;
        }}
        return lo;
      }}

      function buildHistogram(lo, hi, numBins) {{
        var range = hi - lo;
        var binWidth = range / numBins;
        var bins = new Array(numBins).fill(0);
        var startIdx = lowerBound(scores, lo);
        var count = 0;
        for (var i = startIdx; i < scores.length && scores[i] <= hi; i++) {{
          var idx = Math.floor((scores[i] - lo) / binWidth);
          if (idx >= numBins) idx = numBins - 1;
          if (idx < 0) idx = 0;
          bins[idx]++;
          count++;
        }}
        return {{bins: bins, binWidth: binWidth, count: count}};
      }}

      function drawChart(lo, hi) {{
        var numBins = 50;
        var h = buildHistogram(lo, hi, numBins);
        var rows = [['{metric_name} ({direction})', 'Count']];
        for (var i = 0; i < numBins; i++) {{
          var center = lo + (i + 0.5) * h.binWidth;
          rows.push([center, h.bins[i]]);
        }}
        var data = google.visualization.arrayToDataTable(rows);

        var zoomLabel = (lo === globalMin && hi === globalMax) ? '' : ' [zoomed]';
        var options = {{
          title: 'All query x doc distances{sample_label} (' + h.count + ' of {num_floats} scores)' + zoomLabel,
          legend: {{position: 'none'}},
          hAxis: {{
            title: '{metric_name} ({direction})',
            viewWindow: {{min: lo, max: hi}}
          }},
          vAxis: {{title: 'Count'}},
          bar: {{groupWidth: '95%'}},
          chartArea: {{left: 80, right: 20, top: 40, bottom: 60}}
        }};

        chart.draw(data, options);
        document.getElementById('info').innerHTML =
          'Range: ' + lo.toFixed(6) + ' .. ' + hi.toFixed(6) +
          ' &nbsp; Bin width: ' + h.binWidth.toFixed(6) +
          ' &nbsp; Scores in view: ' + h.count;
      }}

      function init() {{
        chartDiv = document.getElementById('chart_div');
        chart = new google.visualization.ColumnChart(chartDiv);
        drawChart(globalMin, globalMax);

        var dragStart = null;
        var overlay = document.getElementById('overlay');

        chartDiv.addEventListener('mousedown', function(e) {{
          var rect = chartDiv.getBoundingClientRect();
          dragStart = {{x: e.clientX - rect.left, clientX: e.clientX}};
          overlay.style.left = dragStart.x + 'px';
          overlay.style.width = '0px';
          overlay.style.display = 'block';
        }});

        chartDiv.addEventListener('mousemove', function(e) {{
          if (!dragStart) return;
          var rect = chartDiv.getBoundingClientRect();
          var curX = e.clientX - rect.left;
          var left = Math.min(dragStart.x, curX);
          var width = Math.abs(curX - dragStart.x);
          overlay.style.left = left + 'px';
          overlay.style.width = width + 'px';
        }});

        chartDiv.addEventListener('mouseup', function(e) {{
          if (!dragStart) return;
          overlay.style.display = 'none';
          var rect = chartDiv.getBoundingClientRect();
          var endX = e.clientX - rect.left;
          var x0 = Math.min(dragStart.x, endX);
          var x1 = Math.max(dragStart.x, endX);
          dragStart = null;

          if (x1 - x0 < 5) return;

          var cli = chart.getChartLayoutInterface();
          var val0 = cli.getHAxisValue(x0);
          var val1 = cli.getHAxisValue(x1);
          if (val0 === null || val1 === null) return;
          var newMin = Math.max(Math.min(val0, val1), globalMin);
          var newMax = Math.min(Math.max(val0, val1), globalMax);
          if (newMax - newMin < (globalMax - globalMin) * 0.001) return;

          zoomStack.push({{min: curMin, max: curMax}});
          curMin = newMin;
          curMax = newMax;
          drawChart(curMin, curMax);
          document.getElementById('resetBtn').style.display = 'inline';
          document.getElementById('backBtn').style.display = 'inline';
        }});

        document.getElementById('resetBtn').addEventListener('click', function() {{
          zoomStack = [];
          curMin = globalMin;
          curMax = globalMax;
          drawChart(curMin, curMax);
          this.style.display = 'none';
          document.getElementById('backBtn').style.display = 'none';
        }});

        document.getElementById('backBtn').addEventListener('click', function() {{
          if (zoomStack.length === 0) return;
          var prev = zoomStack.pop();
          curMin = prev.min;
          curMax = prev.max;
          drawChart(curMin, curMax);
          if (zoomStack.length === 0) {{
            document.getElementById('resetBtn').style.display = 'none';
            this.style.display = 'none';
          }}
        }});
      }}
    </script>
    <style>
      #chart_container {{ position: relative; width: 1200px; height: 600px; }}
      #chart_div {{ width: 100%; height: 100%; }}
      #overlay {{ position: absolute; top: 0; height: 100%; background: rgba(66,133,244,0.15);
                  border-left: 1px solid rgba(66,133,244,0.5); border-right: 1px solid rgba(66,133,244,0.5);
                  display: none; pointer-events: none; z-index: 10; }}
      button {{ margin: 4px 4px 4px 0; padding: 4px 12px; }}
    </style>
  </head>
  <body>
    <div id="chart_container">
      <div id="chart_div"></div>
      <div id="overlay"></div>
    </div>
    <button id="backBtn" style="display:none">Back</button>
    <button id="resetBtn" style="display:none">Reset zoom</button>
    <p id="info"></p>
    <p style="color:#888">Click and drag on the chart to zoom in. Source: {scores_path}</p>
  </body>
</html>
"""
  output_file = f"{output_dir}/{log_base_name}-allDistancesHistogram.html"
  Path(output_file).write_text(html)
  print(f"Wrote all-distances histogram to {output_file}")


def generate_hnsw_traversal_histogram(scores_path, output_dir, log_base_name, metric=None):
  """Read the HNSW traversal scores binary file and generate an HTML histogram.

  The binary format is: for each query, a little-endian int32 count followed
  by that many little-endian float32 scores.
  """
  if not os.path.exists(scores_path):
    print(f"WARNING: HNSW traversal scores file not found: {scores_path}")
    return

  all_scores = []
  total_scores = 0
  data = Path(scores_path).read_bytes()

  offset = 0
  while offset < len(data):
    count = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    scores = struct.unpack_from(f"<{count}f", data, offset)
    offset += count * 4
    total_scores += count
    all_scores.extend(scores[::HNSW_SAMPLE_EVERY_N])

  num_sampled = len(all_scores)
  print(f"HNSW traversal: {total_scores} total scores, sampled 1-in-{HNSW_SAMPLE_EVERY_N} -> {num_sampled} scores for histogram")
  if num_sampled == 0:
    print("WARNING: HNSW traversal scores file is empty")
    return

  metric_name, direction = METRIC_LABELS.get(metric or "", ("similarity score", ""))

  min_score = min(all_scores)
  max_score = max(all_scores)

  if min_score == max_score:
    print(f"WARNING: all HNSW traversal scores are identical ({min_score}); skipping histogram")
    return

  sorted_scores = sorted(all_scores)

  scores_js_lines = []
  chunk_size = 500
  for i in range(0, len(sorted_scores), chunk_size):
    chunk = sorted_scores[i : i + chunk_size]
    scores_js_lines.append(",".join(f"{s:.6f}" for s in chunk))
  scores_js = ",\n".join(scores_js_lines)

  html = f"""<!DOCTYPE html>
<html>
  <head>
    <script type="text/javascript" src="https://www.google.com/jsapi"></script>
    <script type="text/javascript">
      google.load("visualization", "1", {{packages:["corechart"]}});
      google.setOnLoadCallback(init);

      var scores = [
{scores_js}
      ];
      var globalMin = {min_score:.6f};
      var globalMax = {max_score:.6f};
      var curMin = globalMin;
      var curMax = globalMax;
      var chart, chartDiv;
      var zoomStack = [];

      function lowerBound(arr, val) {{
        var lo = 0, hi = arr.length;
        while (lo < hi) {{
          var mid = (lo + hi) >> 1;
          if (arr[mid] < val) lo = mid + 1; else hi = mid;
        }}
        return lo;
      }}

      function buildHistogram(lo, hi, numBins) {{
        var range = hi - lo;
        var binWidth = range / numBins;
        var bins = new Array(numBins).fill(0);
        var startIdx = lowerBound(scores, lo);
        var count = 0;
        for (var i = startIdx; i < scores.length && scores[i] <= hi; i++) {{
          var idx = Math.floor((scores[i] - lo) / binWidth);
          if (idx >= numBins) idx = numBins - 1;
          if (idx < 0) idx = 0;
          bins[idx]++;
          count++;
        }}
        return {{bins: bins, binWidth: binWidth, count: count}};
      }}

      function drawChart(lo, hi) {{
        var numBins = 50;
        var h = buildHistogram(lo, hi, numBins);
        var rows = [['{metric_name} ({direction})', 'Count']];
        for (var i = 0; i < numBins; i++) {{
          var center = lo + (i + 0.5) * h.binWidth;
          rows.push([center, h.bins[i]]);
        }}
        var data = google.visualization.arrayToDataTable(rows);

        var zoomLabel = (lo === globalMin && hi === globalMax) ? '' : ' [zoomed]';
        var options = {{
          title: 'HNSW traversal {metric_name} distribution (' + h.count + ' of {num_sampled} sampled from {total_scores} total, 1-in-{HNSW_SAMPLE_EVERY_N})' + zoomLabel,
          legend: {{position: 'none'}},
          hAxis: {{
            title: '{metric_name} ({direction})',
            viewWindow: {{min: lo, max: hi}}
          }},
          vAxis: {{title: 'Count'}},
          bar: {{groupWidth: '95%'}},
          chartArea: {{left: 80, right: 20, top: 40, bottom: 60}}
        }};

        chart.draw(data, options);
        document.getElementById('info').innerHTML =
          'Range: ' + lo.toFixed(6) + ' .. ' + hi.toFixed(6) +
          ' &nbsp; Bin width: ' + h.binWidth.toFixed(6) +
          ' &nbsp; Scores in view: ' + h.count;
      }}

      function init() {{
        chartDiv = document.getElementById('chart_div');
        chart = new google.visualization.ColumnChart(chartDiv);
        drawChart(globalMin, globalMax);

        var dragStart = null;
        var overlay = document.getElementById('overlay');

        chartDiv.addEventListener('mousedown', function(e) {{
          var rect = chartDiv.getBoundingClientRect();
          dragStart = {{x: e.clientX - rect.left, clientX: e.clientX}};
          overlay.style.left = dragStart.x + 'px';
          overlay.style.width = '0px';
          overlay.style.display = 'block';
        }});

        chartDiv.addEventListener('mousemove', function(e) {{
          if (!dragStart) return;
          var rect = chartDiv.getBoundingClientRect();
          var curX = e.clientX - rect.left;
          var left = Math.min(dragStart.x, curX);
          var width = Math.abs(curX - dragStart.x);
          overlay.style.left = left + 'px';
          overlay.style.width = width + 'px';
        }});

        chartDiv.addEventListener('mouseup', function(e) {{
          if (!dragStart) return;
          overlay.style.display = 'none';
          var rect = chartDiv.getBoundingClientRect();
          var endX = e.clientX - rect.left;
          var x0 = Math.min(dragStart.x, endX);
          var x1 = Math.max(dragStart.x, endX);
          dragStart = null;

          if (x1 - x0 < 5) return;

          var cli = chart.getChartLayoutInterface();
          var val0 = cli.getHAxisValue(x0);
          var val1 = cli.getHAxisValue(x1);
          if (val0 === null || val1 === null) return;
          var newMin = Math.max(Math.min(val0, val1), globalMin);
          var newMax = Math.min(Math.max(val0, val1), globalMax);
          if (newMax - newMin < (globalMax - globalMin) * 0.001) return;

          zoomStack.push({{min: curMin, max: curMax}});
          curMin = newMin;
          curMax = newMax;
          drawChart(curMin, curMax);
          document.getElementById('resetBtn').style.display = 'inline';
          document.getElementById('backBtn').style.display = 'inline';
        }});

        document.getElementById('resetBtn').addEventListener('click', function() {{
          zoomStack = [];
          curMin = globalMin;
          curMax = globalMax;
          drawChart(curMin, curMax);
          this.style.display = 'none';
          document.getElementById('backBtn').style.display = 'none';
        }});

        document.getElementById('backBtn').addEventListener('click', function() {{
          if (zoomStack.length === 0) return;
          var prev = zoomStack.pop();
          curMin = prev.min;
          curMax = prev.max;
          drawChart(curMin, curMax);
          if (zoomStack.length === 0) {{
            document.getElementById('resetBtn').style.display = 'none';
            this.style.display = 'none';
          }}
        }});
      }}
    </script>
    <style>
      #chart_container {{ position: relative; width: 1200px; height: 600px; }}
      #chart_div {{ width: 100%; height: 100%; }}
      #overlay {{ position: absolute; top: 0; height: 100%; background: rgba(66,133,244,0.15);
                  border-left: 1px solid rgba(66,133,244,0.5); border-right: 1px solid rgba(66,133,244,0.5);
                  display: none; pointer-events: none; z-index: 10; }}
      button {{ margin: 4px 4px 4px 0; padding: 4px 12px; }}
    </style>
  </head>
  <body>
    <div id="chart_container">
      <div id="chart_div"></div>
      <div id="overlay"></div>
    </div>
    <button id="backBtn" style="display:none">Back</button>
    <button id="resetBtn" style="display:none">Reset zoom</button>
    <p id="info"></p>
    <p style="color:#888">Click and drag on the chart to zoom in. Source: {scores_path}</p>
    <p style="color:#888">These are all scores computed during HNSW graph traversal, not just the final top-K results.</p>
  </body>
</html>
"""
  output_file = f"{output_dir}/{log_base_name}-hnswTraversalHistogram.html"
  Path(output_file).write_text(html)
  print(f"Wrote HNSW traversal score histogram to {output_file}")


def precompute_exact_nn(values, dim, doc_vectors, query_vectors):
  """Precompute exact nearest neighbors using numpy for all parameter combinations."""
  ndoc = values.get("ndoc", (1000,))
  nquery = values.get("nquery", (1000,))
  metrics = values.get("metric", ("dot_product",))
  top_ks = values.get("topK", (100,))
  query_start_indices = values.get("queryStartIndex", (0,))
  encodings = values.get("encoding", ("float32",))

  # wrap scalar values
  if not isinstance(ndoc, (tuple, list)):
    ndoc = (ndoc,)
  if not isinstance(nquery, (tuple, list)):
    nquery = (nquery,)
  if not isinstance(metrics, (tuple, list)):
    metrics = (metrics,)
  if not isinstance(top_ks, (tuple, list)):
    top_ks = (top_ks,)
  if not isinstance(query_start_indices, (tuple, list)):
    query_start_indices = (query_start_indices,)
  if not isinstance(encodings, (tuple, list)):
    encodings = (encodings,)

  combos = list(itertools.product(ndoc, nquery, metrics, top_ks, query_start_indices, encodings))
  print(f"\nprecomputing exact NN for {len(combos)} parameter combination(s) using numpy...")
  knnExactNN.check_blas_config()
  for ndoc, nquery, metric, top_k, query_start_index, encoding in combos:
    knnExactNN.run_one(doc_vectors, query_vectors, dim, ndoc, nquery, metric, top_k, query_start_index, encoding)
  print()


def run_knn_benchmark(checkout, values, log_path):
  indexes = [0] * len(values.keys())
  indexes[-1] = -1
  args = []
  # dim = 100
  # doc_vectors = "%s/lucene_util/tasks/enwiki-20120502-lines-1k-100d.vec" % constants.BASE_DIR
  # query_vectors = "%s/lucene_util/tasks/vector-task-100d.vec" % constants.BASE_DIR

  # Cohere Wikipedia en vectors - see cohere-v3-README.txt. Set False for the older 768d v2 corpus.
  v3 = True

  # The FULL English Cohere-v3 Wikipedia corpus: 41,488,110 passages x 1024d unit-norm float32
  # (~170GB), downloaded straight from HuggingFace by
  #   python src/python/download_cohere_v3_en.py --data-dir $BASE_DIR/data
  # then shuffled to the 'scattered' distribution (adjacent Wikipedia passages must not stay adjacent,
  # or they land in the same IVF/LSH bucket and inflate recall):
  #   python src/python/shuffle_vecs.py <docs.en-full.vec> <docs.en-full-scattered.vec> --dim 1024
  # Same 1024d unit-norm distribution as the bundled 1M sample, so a frozen PCA/ITQ basis stays
  # comparable. This is the >RAM regime where this codec is meant to win and where disk I/O finally
  # matters (findings.md §6/§13); the bundled 1M sample was HNSW's best case (fit in RAM).
  #
  # Queries are held out by the downloader at the ARTICLE level (1-in-24 wiki_ids by hash, ~1.7M
  # vectors), matching the bundled corpus's design: docs and queries share no vector AND no article,
  # so a query's own sibling paragraphs aren't sitting in the index as trivial top-1 hits.
  if v3:
    dim = 1024
    data_dir = f"{constants.BASE_DIR}/data"
    # Prefer the shuffled ('scattered') corpus; fall back to corpus order if it hasn't been built yet.
    # ndoc in PARAMS must be <= the doc count (~39.7M: 41,488,110 total minus the ~1.7M query holdout);
    # the ndoc-vs-file-size check below enforces this exactly.
    doc_vectors = f"{data_dir}/cohere-v3-wikipedia-en-scattered-1024d.docs.en-full.vec"
    if not os.path.exists(doc_vectors):
      unshuffled = f"{data_dir}/cohere-v3-wikipedia-en-1024d.docs.en-full.vec"
      if not os.path.exists(unshuffled):
        raise RuntimeError(
          f"full-English doc vectors not found: {doc_vectors}\n"
          f"  download them first:  python src/python/download_cohere_v3_en.py --data-dir {data_dir}\n"
          f"  then shuffle:         python src/python/shuffle_vecs.py {unshuffled} {doc_vectors} --dim {dim}"
        )
      print(f"WARNING: using CORPUS-ORDER doc vectors ({unshuffled}).\n  Adjacent Wikipedia passages share a bucket -> recall is OPTIMISTIC. Shuffle for honest numbers:\n    python src/python/shuffle_vecs.py {unshuffled} {doc_vectors} --dim {dim}")
      doc_vectors = unshuffled
    query_vectors = f"{data_dir}/cohere-v3-wikipedia-en-1024d.queries.1in24-articles.vec"
    if not os.path.exists(query_vectors):
      raise RuntimeError(
        f"query vectors not found: {query_vectors}\n"
        f"  download them first:  python src/python/download_cohere_v3_en.py --data-dir {data_dir}"
      )
  else:
    dim = 768
    doc_vectors = f"/lucenedata/enwiki/cohere-wikipedia-docs-{dim}d.vec"
    query_vectors = f"/lucenedata/enwiki/cohere-wikipedia-queries-{dim}d.vec"

  # dim = 768
  # doc_vectors = '/lucenedata/enwiki/enwiki-20120502-lines-1k-mpnet.vec'
  # query_vectors = '/lucenedata/enwiki/enwiki-20120502.mpnet.vec'
  # dim = 384
  # doc_vectors = '%s/data/enwiki-20120502-lines-1k-minilm.vec' % constants.BASE_DIR
  # query_vectors = '%s/luceneutil/tasks/vector-task-minilm.vec' % constants.BASE_DIR
  # dim = 300
  # doc_vectors = '%s/data/enwiki-20120502-lines-1k-300d.vec' % constants.BASE_DIR
  # query_vectors = '%s/luceneutil/tasks/vector-task-300d.vec' % constants.BASE_DIR

  # Cohere dataset
  # dim = 768
  # doc_vectors = f"{constants.BASE_DIR}/data/cohere-wikipedia-docs-{dim}d.vec"
  # query_vectors = f"{constants.BASE_DIR}/data/cohere-wikipedia-queries-{dim}d.vec"
  # doc_vectors = f"/lucenedata/enwiki/{'cohere-wikipedia'}-docs-{dim}d.vec"
  # query_vectors = f"/lucenedata/enwiki/{'cohere-wikipedia'}-queries-{dim}d.vec"
  # parentJoin_meta_file = f"{constants.BASE_DIR}/data/{'cohere-wikipedia'}-metadata.csv"

  # One .jfr PER INVOCATION. This function is called once per param combination (e.g. gcutAxes 1 then 2),
  # and only the FIRST invocation reindexes -- later ones reuse the cached index and just search. With a
  # single shared filename the last (search-only) JVM overwrote the recording that contained indexing and
  # force-merge, so an indexing profile silently came back ~14 s of pure search.
  jfr_output = f"{constants.LOGS_DIR}/knn-perf-test-{_JFR_SEQ[0]}.jfr"
  _JFR_SEQ[0] += 1

  cp = benchUtil.classPathToString(benchUtil.getClassPath(checkout) + (f"{constants.BENCH_BASE_DIR}/build",))
  cmd = constants.JAVA_EXE.split(" ") + [
    # Heap cap: JAVA_EXE is the bare java path (no -Xmx), so without this the index/search JVM ran at
    # the JVM DEFAULT max heap (~1/4 RAM), silently ignoring KNN_HEAP. At high hashBits the per-segment
    # centroid arrays (2^hashBits * dim floats) blow past that default -> OOM. KNN_HEAP now applies.
    # Only -Xmx (no -Xms): the JVM starts with a small heap and grows lazily to the cap, so it COMMITS
    # only what the live set needs instead of reserving the full cap up front. On a 32GB box -Xms=-Xmx=24g
    # reserved 24GB resident -> ~8GB swap even though peak USED was ~15-20GB. Lazy commit avoids that.
    f"-Xmx{constants.KNN_HEAP}",
    "-cp",
    cp,
    "--add-modules",
    "jdk.incubator.vector",  # no need to add these flags -- they are on by default now?
    "--enable-native-access=ALL-UNNAMED",
    f"-Djava.util.concurrent.ForkJoinPool.common.parallelism={multiprocessing.cpu_count()}",  # so that brute force computeNN uses all cores
    "-XX:+UnlockDiagnosticVMOptions",
    "-XX:+DebugNonSafepoints",
  ]

  # Debug: dump LSH columnar scan I/O stats (filter vs payload bytes, survivor fraction) at JVM exit.
  if LSH_SCAN_STATS:
    cmd += ["-Dlsh.scanStats=true"]

  if LSH_OFF_HEAP_CENTROIDS:
    cmd += ["-Dlsh.offHeapCentroids=true"]

  if LSH_POSITIONAL_SCAN:
    cmd += ["-Dlsh.positionalScan=true"]

  if LSH_HNSW_ROUTING:
    cmd += ["-Dlsh.hnswRouting=true"]
    if LSH_HNSW_M is not None:
      cmd += [f"-Dlsh.hnswM={LSH_HNSW_M}"]
    if LSH_HNSW_BEAM_WIDTH is not None:
      cmd += [f"-Dlsh.hnswBeamWidth={LSH_HNSW_BEAM_WIDTH}"]
    if LSH_HNSW_OVERQUERY is not None:
      cmd += [f"-Dlsh.hnswOverquery={LSH_HNSW_OVERQUERY}"]
    if LSH_ROUTE_ON_REFERENCE_CENTROIDS:
      cmd += ["-Dlsh.routeOnReferenceCentroids=true"]

  if LSH_REFERENCE_CENTROIDS_ONLY:
    cmd += ["-Dlsh.referenceCentroidsOnly=true"]

  if IVF_CENTROID_HNSW is not None:
    cmd += [f"-Divf.centroidHnsw={'true' if IVF_CENTROID_HNSW else 'false'}"]
  if IVF_REFINE_FACTOR is not None:
    cmd += [f"-Divf.refineFactor={IVF_REFINE_FACTOR}"]
  if IVF_ADAPTIVE_NPROBE_MARGIN is not None:
    cmd += [f"-Divf.adaptiveNprobeMargin={IVF_ADAPTIVE_NPROBE_MARGIN}"]
  if LLOYD_BEAM_FACTOR is not None:
    cmd += [f"-Dlloyd.beamFactor={LLOYD_BEAM_FACTOR}"]
  if LLOYD_SCORE_IN_PLACE:
    cmd += ["-Dlloyd.scoreInPlace=true"]
  # LLOYD_SHORTLIST_DEDUP=1 => -Dlloyd.shortlistDedup=true: move the spill dedup off the per-slot scan
  # (JFR: FixedBitSet.getAndSet = 18.8% of warm CPU, ~250k random writes into a 4.75 MB bitset per query,
  # plus a fresh maxDoc bitset allocated per query) and onto the BRUTE_N shortlist instead. Reader-side,
  # no reindex; asserted bit-identical by TestUringRerankGather's shortlist-dedup cases.
  if os.environ.get("LLOYD_SHORTLIST_DEDUP") == "1":
    cmd += ["-Dlloyd.shortlistDedup=true"]
  # LLOYD_CO_RESIDENT_CODES=1 => -Dlloyd.coResidentCodes=true: Stage G. Read each probed cell's CODE run
  # contiguously alongside its sketch run, so the shortlist's int8 records are already in RAM and Stage D's
  # ~2000 scattered reads are skipped. Reads ~8.2x more coarse bytes (only ~1% get reranked) in exchange
  # for zero random reads -- wins where I/O time is hidden (CPU-bound) or storage is fast.
  if os.environ.get("LLOYD_CO_RESIDENT_CODES") == "1":
    cmd += ["-Dlloyd.coResidentCodes=true"]
  # LLOYD_SCALAR_POPCOUNT=1 => -Dlloyd.scalarPopcount=true: CONTROL arm that restores the old
  # single-accumulator Hamming loop. The default (unset) uses four independent accumulators so the
  # XOR+CNT work pipelines and C2 auto-vectorizes -- xorBitCountInt was 26.5% of warm CPU, the top term.
  if os.environ.get("LLOYD_SCALAR_POPCOUNT") == "1":
    cmd += ["-Dlloyd.scalarPopcount=true"]
  if LLOYD_PREFETCH_CELLS:
    cmd += ["-Dlloyd.prefetchCells=true"]
  if os.environ.get("LLOYD_URING_SKETCH_SCAN") == "1":
    cmd += ["-Dlloyd.uringSketchScan=true"]
  # Stage D: batch+coalesce the rerank code-record reads (the component that dominates cold reads --
  # Stage C only batches the sketch runs). Reader-side; bit-identical results.
  if os.environ.get("LLOYD_URING_RERANK") == "1":
    cmd += ["-Dlloyd.uringRerank=true"]
  # Stage E: dispatch each cell's read as it is selected and scan cells as their bytes land, so reads
  # overlap the Hamming scan instead of running as a separate blocking phase.
  if os.environ.get("LLOYD_URING_PIPELINE") == "1":
    cmd += ["-Dlloyd.uringPipeline=true"]
  # Stage F: STREAMING rerank -- submit the coalesced code-record ranges and score each candidate as its
  # bytes land, so the int8 rerank overlaps the cold reads instead of blocking for the whole gather first.
  if os.environ.get("LLOYD_URING_RERANK_PIPELINE") == "1":
    cmd += ["-Dlloyd.uringRerankPipeline=true"]
  # O_DIRECT: open the reader's data file cache-bypassing, forcing the >RAM regime for the ring alone.
  # A/B arm only -- measures the pure-device ceiling; not the realistic partial-cache number.
  if os.environ.get("LLOYD_URING_DIRECT") == "1":
    cmd += ["-Dlloyd.uringDirect=true"]
  if os.environ.get("LLOYD_PIPELINE_DEPTH") is not None:
    cmd += [f"-Dlloyd.pipelineDepth={os.environ['LLOYD_PIPELINE_DEPTH']}"]
  if os.environ.get("LLOYD_RERANK_AUDIT") == "1":
    cmd += ["-Dlloyd.rerankAudit=true"]
  if os.environ.get("LLOYD_RERANK_COALESCE_GAP") is not None:
    cmd += [f"-Dlloyd.rerankCoalesceGap={os.environ['LLOYD_RERANK_COALESCE_GAP']}"]
  if os.environ.get("LLOYD_URING_DEBUG") == "1":
    cmd += ["-Dlloyd.uringDebug=true"]
  if os.environ.get("LLOYD_NO_HEAP_SKIP") == "1":
    cmd += ["-Dlloyd.noHeapSkip=true"]
  if os.environ.get("KNN_DROP_CACHE_AFTER_WARMUP") == "1":
    cmd += ["-Dknn.dropCacheAfterWarmup=true"]
  if os.environ.get("LLOYD_CEIL_K"):
    cmd += [f'-Dlloyd.ceilK={os.environ["LLOYD_CEIL_K"]}']
  if IVF_STREAM_REFINE_ITERS is not None:
    cmd += [f"-Divf.streamRefineIters={IVF_STREAM_REFINE_ITERS}"]
  if IVF_CENTROID_HNSW_M is not None:
    cmd += [f"-Divf.centroidHnswM={IVF_CENTROID_HNSW_M}"]
  if IVF_DERIVED_GRAPH:
    cmd += ["-Divf.derivedGraph=true"]
  if IVF_DERIVED_GRAPH_M is not None:
    cmd += [f"-Divf.derivedGraphM={IVF_DERIVED_GRAPH_M}"]
  if IVF_CENTROID_HNSW_BEAM_WIDTH is not None:
    cmd += [f"-Divf.centroidHnswBeamWidth={IVF_CENTROID_HNSW_BEAM_WIDTH}"]
  if IVF_SPILL_EF_SEARCH is not None:
    cmd += [f"-Divf.spillEfSearch={IVF_SPILL_EF_SEARCH}"]
  # IVF_REUSE_GRAPH_TOPOLOGY=0 => -Divf.reuseGraphTopology=false, restoring a full HNSW build at every
  # merge routing stage instead of refreshing the donor graph's int8 codes over the moved centroids. The
  # codec DEFAULTS this on, so 0 is the control arm for pricing the indexing win.
  #
  # WRITE-TIME and NOT in the index key: it changes the persisted clustering (a reused topology routes docs
  # slightly differently than a freshly built one), so the two arms MUST NOT share a cached index. Run the
  # control with KNN_CLEAR_CACHE=1 -- otherwise the "off" arm silently reuses the "on" arm's index and
  # measures nothing. Same class of trap as blockP below.
  if os.environ.get("IVF_REUSE_GRAPH_TOPOLOGY") == "0":
    cmd += ["-Divf.reuseGraphTopology=false"]
  if os.environ.get("IVF_QUANTIZER"):
    cmd += [f'-Divf.quantizer={os.environ["IVF_QUANTIZER"]}']
  # Block size p for -Divf.quantizer=blocksphere. Write-time (it sets the on-disk bytes/block), and the
  # codec default is already 2, so an unset value happens to give p=2 today -- pass it explicitly anyway
  # so the run does not silently depend on that default. NOTE: blockP is NOT in the index key (only
  # qz<quantizer> is), so switching p against a cached index would misparse every record; clear
  # knn-reuse/indices when changing it. run_40m_p2_sweep.sh enforces this with a .blockP stamp.
  if os.environ.get("IVF_BLOCK_P"):
    cmd += [f'-Divf.blockP={os.environ["IVF_BLOCK_P"]}']
  if IVF_STREAM_FLUSH_MIN_DOCS is not None:
    cmd += [f"-Divf.streamFlushMinDocs={IVF_STREAM_FLUSH_MIN_DOCS}"]
  if IVF_TRAIN_SAMPLE_CAP is not None:
    cmd += [f"-Divf.trainSampleCap={IVF_TRAIN_SAMPLE_CAP}"]
  if os.environ.get("LLOYD_PREFETCH_AUDIT") == "1":
    cmd += ["-Dlloyd.prefetchAudit=true"]
  if LLOYD_PREFETCH_CODE_MAX_BYTES is not None:
    cmd += [f"-Dlloyd.prefetchCodeMaxBytes={LLOYD_PREFETCH_CODE_MAX_BYTES}"]
  if LLOYD_CEIL_AUDIT:
    cmd += ["-Dlloyd.ceilAudit=true"]
  if LLOYD_CEIL_DIR:
    cmd += ["-Dlloyd.ceilDir=true"]
  if LLOYD_CEIL_ANISO:
    cmd += ["-Dlloyd.ceilAniso=true"]
  if LLOYD_CEIL_SUB:
    cmd += ["-Dlloyd.ceilSub=true"]
  if LLOYD_CEIL_SEP:
    cmd += ["-Dlloyd.ceilSep=true"]
  if LLOYD_CEIL_ONLINE:
    cmd += ["-Dlloyd.ceilOnline=true"]
  if LLOYD_CEIL_ADJ:
    cmd += ["-Dlloyd.ceilAdj=true"]
  if LLOYD_CEIL_SELEXP:
    cmd += ["-Dlloyd.ceilSelExp=true"]
  if LLOYD_CEIL_BRUTE:
    cmd += ["-Dlloyd.ceilBrute=true"]
  if LLOYD_CEIL_SPILL:
    cmd += ["-Dlloyd.ceilSpill=true"]
  if GCUT_TREE_ROUTE_DIMS is not None:
    cmd += [f"-Dgcut.treeRouteDims={GCUT_TREE_ROUTE_DIMS}"]
  if GCUT_TREE_ROUTE:
    cmd += ["-Dgcut.treeRoute=true"]
  if LLOYD_BRUTE_SEARCH:
    cmd += ["-Dlloyd.bruteSearch=true"]
    cmd += ["-Dlloyd.ceilBruteDims=1024"]
  if IVF_QUANT_BITS:
    cmd += [f"-Divf.quantBits={IVF_QUANT_BITS}"]
  if os.environ.get("IVF_CELL_ORDER", "1") == "1":
    cmd += ["-Divf.cellOrder=true"]
  if IVF_BEAM_SPILL:
    cmd += ["-Divf.beamSpill=true"]
    if IVF_SPILL_MARGIN:
      cmd += [f"-Divf.spillMargin={IVF_SPILL_MARGIN}"]
  if os.environ.get("LLOYD_DBG_CELL") == "1":
    cmd += ["-Dlloyd.dbgCell=true"]
  if LLOYD_SKETCH_SCAN:
    cmd += ["-Dlloyd.sketchScan=true"]
    if os.environ.get("LLOYD_BRUTE_N"):
      cmd += [f'-Dlloyd.bruteN={os.environ["LLOYD_BRUTE_N"]}']
    _sd = os.environ.get("LLOYD_SKETCH_DIMS", "1024")
    cmd += [f"-Dlloyd.ceilBruteDims={_sd}"]
    cmd += [f"-Dlloyd.sketchDims={_sd}"]
    if LLOYD_RERANK_BITS:
      cmd += [f"-Dlloyd.rerankBits={LLOYD_RERANK_BITS}"]
  if IVF_RERANK_FACTOR is not None:
    cmd += [f"-Divf.rerankFactor={IVF_RERANK_FACTOR}"]
  if IVF_ENABLE_COPY_MERGE:
    cmd += ["-Divf.enableCopyMerge=true"]
  if IVF_EXACT_ASSIGN:
    cmd += ["-Divf.exactAssign=true"]
  if IVF_WORK_DIMS is not None:
    cmd += [f"-Divf.workDims={IVF_WORK_DIMS}"]
  if IVF_GRAPH_ROUTE_ITERS is not None:
    cmd += [f"-Divf.graphRouteIters={IVF_GRAPH_ROUTE_ITERS}"]
  if IVF_ANISO_ETA is not None:
    cmd += [f"-Divf.anisoEta={IVF_ANISO_ETA}"]
  if IVF_SHARED_CODES:
    cmd += ["-Divf.sharedCodes=true"]
  if IVF_DROP_RAW_VECTORS:
    cmd += ["-Divf.dropRawVectors=true"]

  if DO_PROFILING:
    cmd += [
      f"-XX:StartFlightRecording=jdk.CPUTimeSample#enabled=true,dumponexit=true,maxsize={constants.JFR_MAX_SIZE_MB}M,settings={constants.BENCH_BASE_DIR}/src/python/profiling.jfc,filename={jfr_output}"
    ]

  cmd += ["knn.KnnGraphTester"]

  if NOISY:
    print_run_summary(values)

  # KNN_SKIP_SMELL=1: skip the vector-distribution / intrinsic-dim analysis. It random-reads a large
  # sample straight out of the (162 GB) docs file, which under a >RAM memory cap thrashes for many
  # minutes before the benchmark even starts -- and it tells us nothing about search latency.
  if os.environ.get("KNN_SKIP_SMELL") == "1":
    print("KNN_SKIP_SMELL=1: skipping smell_vectors (vector distribution / intrinsic-dim analysis)")
  else:
    smell_vectors(dim, doc_vectors, "docs")
    smell_vectors(dim, query_vectors, "queries")

  n_doc_check = max(values.get("ndoc", (1000,)))
  n_query_check = max(values.get("nquery", (1000,)))
  query_start_check = min(values.get("queryStartIndex", (0,)))

  # Fail fast if PARAMS asks for more vectors than the source files actually hold: KnnGraphTester would
  # otherwise read past EOF (or silently index fewer docs than the run is labelled with).
  for label, path, needed in (("ndoc", doc_vectors, n_doc_check), ("nquery+queryStartIndex", query_vectors, query_start_check + n_query_check)):
    available = os.path.getsize(path) // (dim * 4)
    if needed > available:
      raise RuntimeError(f"{label}={needed:,} exceeds the {available:,} vectors in {path}")

  encoding_check = values.get("encoding", ("float32",))[0]
  if CHECK_VECTOR_OVERLAP:
    knnExactNN.check_vector_overlap(
      doc_vectors, 0, n_doc_check, query_vectors, query_start_check, n_query_check, dim, encoding_check, check_doc_doc=CHECK_DOC_DOC_DUPLICATES, check_query_query=CHECK_QUERY_QUERY_DUPLICATES
    )
  else:
    print("check_vector_overlap: SKIPPED (CHECK_VECTOR_OVERLAP=False); not scanning for doc/query duplicates")

  if CHECK_QUERY_DOC_MODEL_CONSISTENCY:
    metric_check = values.get("metric", ("dot_product",))[0]
    knnExactNN.check_query_doc_distribution_match(
      doc_vectors, 0, n_doc_check, query_vectors, query_start_check, n_query_check, dim, encoding_check, metric=metric_check, n_sample=QUERY_DOC_MODEL_CONSISTENCY_SAMPLE
    )

  # precompute exact nearest neighbors using numpy (much faster than Java brute force)
  if USE_NUMPY_EXACT_NN:
    precompute_exact_nn(values, dim, doc_vectors, query_vectors)

  index_run = 1
  all_results = []
  all_simd_reports = []
  log_dir_name, log_file_name = log_path
  if DO_VMSTAT and GNUPLOT_PATH is not None:
    vmstat_index_html_path = f"{log_dir_name}/{log_file_name}-vmstats.html"
    print(f"\nNOTE: open {vmstat_index_html_path} in browser to see CPU/IO telemetry of each run")
    vmstat_index_out = open(vmstat_index_html_path, "w")
    vmstat_index_out.write("<h2>vmstat results for each run</h2>\n\n")
  while advance(indexes, values):
    if NOISY:
      print("\nNEXT:")
    pv = {}
    args = []
    quantize_bits = None
    do_quantize_compress = False
    do_rerank = False
    rerank_quantize_bits = 32
    for i, p in enumerate(values.keys()):
      if values[p]:
        value = values[p][indexes[i]]
        if p == "quantizeBits":
          if value != 32:
            pv[p] = value
            print(f"  -{p}={value}")
            print("  -quantize")
            args += ["-quantize"]
            quantize_bits = value
        elif p == "rerank":
          if value:
            do_rerank = True
        elif p == "rerankQuantizeBits":
          rerank_quantize_bits = value
        elif type(value) is bool:
          if p == "quantizeCompress":
            # carefully only add this flag (below) if we are quantizing to 4 bits:
            do_quantize_compress = True
          elif value:
            args += ["-" + p]
            print(f"  -{p}")
        else:
          print(f"  -{p}={value}")
          pv[p] = value
      else:
        args += ["-" + p]
        print(f"  -{p}")

    if not do_rerank and "rerankQuantizeBits" in values:
      rerank_qb_vals = values["rerankQuantizeBits"]
      if type(rerank_qb_vals) in (list, tuple) and rerank_quantize_bits != rerank_qb_vals[0]:
        continue

    if quantize_bits == 4 and do_quantize_compress:
      args += ["-quantizeCompress"]
      print("  -quantizeCompress")

    if do_rerank:
      args += ["-rerank"]
      print("  -rerank")
      if rerank_quantize_bits != 32:
        args += ["-rerankQuantizeBits", str(rerank_quantize_bits)]
        print(f"  -rerankQuantizeBits={rerank_quantize_bits}")

    args += [a for (k, v) in pv.items() for a in ("-" + k, str(v)) if a]

    this_cmd = (
      cmd
      + args
      + [
        "-dim",
        str(dim),
        "-docs",
        doc_vectors,
        # "-reindex",
        "-search-and-stats",
        query_vectors,
        "-numIndexThreads",
        str(NUM_INDEX_THREADS),
        # "-metric",
        # "mip",
        # "-parentJoin",
        # parentJoin_meta_file,
        # '-numMergeThread', '8', '-numMergeWorker', '8',
        #'-forceMerge',
        #'-stats',
        #'-quiet'
      ]
    )

    if DO_HNSW_SCORE_HISTOGRAM:
      this_cmd += ["-hnswScoreHistogram"]

    if DO_ALL_DISTANCES_HISTOGRAM:
      this_cmd += ["-allDistancesHistogram", "-allDistancesSampleEveryN", str(ALL_DISTANCES_SAMPLE_EVERY_N)]

    if CONFIRM_SIMD_ASM_MODE:
      perf_data_file = f"perf{index_run}.data"
      print(f"NOTE: adding 'perf record' command, to {perf_data_file}, to sample instructions being executed to later confirm SIMD usage")
      this_cmd = [PERF_EXE, "record", "-m", "2M", "-v", "--call-graph", "lbr", "-e", "instructions:u", "-o", perf_data_file, "-g"] + this_cmd

    perf_stat_simd_file = None
    if DO_PERF_STAT_SIMD:
      perf_stat_simd_file = get_unique_log_name(log_path, "perf-simd").replace(".log", ".txt")
      this_cmd = wrap_cmd_with_perf_stat_simd(this_cmd, perf_stat_simd_file)

    if NOISY:
      print(f"  cmd: {this_cmd}")
    else:
      cmd += ["-quiet"]

    # hint that we will read the vectors files, to get the OS starting on the I/O now:
    vec_size_bytes = dim * 4
    query_start_byte = pv.get("queryStartIndex", 0) * vec_size_bytes
    advise_will_need(query_vectors, query_start_byte, pv.get("nquery", 0) * vec_size_bytes)
    if "-reindex" in this_cmd or DO_ALL_DISTANCES_HISTOGRAM:
      advise_will_need(doc_vectors, 0, pv.get("ndoc", 0) * vec_size_bytes)

    if DO_PS:
      # TODO: get k=v into log file name instead of confusing/error-prone 0, 1, 2, ...
      ps_log_file_name = get_unique_log_name(log_path, "ps")
      ps_process = ps_head.PSTopN(1, ps_log_file_name)
      print(f"\nsaving top (ps) processes: {ps_process.cmd}")
    else:
      print("WARNING: top (ps) processes is disabled!")
      ps_process = None

    if DO_VMSTAT:
      vmstat_log_file_name = get_unique_log_name(log_path, "vmstat")
      vmstat_cmd = f"{benchUtil.VMSTAT_PATH} --active --wide --timestamp --unit M 1 > {vmstat_log_file_name} 2>/dev/null &"
      print(f'saving vmstat: "{vmstat_cmd}"\n')
      vmstat_process = subprocess.Popen(vmstat_cmd, shell=True, preexec_fn=os.setsid)
    else:
      print("WARNING: vmstat is disabled!")

    ram_mon = None  # bound before the try so the finally can reference it even if Popen throws
    try:
      job = subprocess.Popen(this_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")

      # Live RAM monitor on the just-launched search/index JVM (samples job.pid's RSS on a bg thread).
      if DO_RAM_MONITOR:
        ram_csv_file_name = get_unique_log_name(log_path, "ram").replace(".log", ".csv")
        ram_label = f"{pv.get('indexType', '?')} ndoc={pv.get('ndoc', '?')}"
        ram_mon = ram_monitor.RAMMonitor(job.pid, ram_csv_file_name, label=ram_label)
        print(f"saving live RAM (RSS) monitor: {ram_csv_file_name}")

      re_summary = re.compile(r"^SUMMARY: (.*?)$", re.MULTILINE)
      re_scores_path = re.compile(r"^EXACT_NN_SCORES_PATH: (.+)$")
      re_nn_metric = re.compile(r"^EXACT_NN_METRIC: (.+)$")
      re_hnsw_scores_path = re.compile(r"^HNSW_TRAVERSAL_SCORES_PATH: (.+)$")
      re_hnsw_metric = re.compile(r"^HNSW_TRAVERSAL_METRIC: (.+)$")
      re_all_distances_path = re.compile(r"^ALL_DISTANCES_SCORES_PATH: (.+)$")
      re_all_distances_metric = re.compile(r"^ALL_DISTANCES_METRIC: (.+)$")
      re_all_distances_sample = re.compile(r"^ALL_DISTANCES_SAMPLE_EVERY_N: (.+)$")
      summary = None
      exact_nn_scores_path = None
      exact_nn_metric = None
      hnsw_traversal_scores_path = None
      hnsw_traversal_metric = None
      all_distances_scores_path = None
      all_distances_metric = None
      all_distances_sample_every_n = None
      hit_exception = False
      while job.poll() is None:
        line = job.stdout.readline()
        if not line:
          continue
        if NOISY:
          sys.stdout.write(line)
          sys.stdout.flush()
        m = re_summary.match(line)
        if m is not None:
          summary = m.group(1)
        m = re_scores_path.match(line)
        if m is not None:
          exact_nn_scores_path = m.group(1).strip()
        m = re_nn_metric.match(line)
        if m is not None:
          exact_nn_metric = m.group(1).strip()
        m = re_hnsw_scores_path.match(line)
        if m is not None:
          hnsw_traversal_scores_path = m.group(1).strip()
        m = re_hnsw_metric.match(line)
        if m is not None:
          hnsw_traversal_metric = m.group(1).strip()
        m = re_all_distances_path.match(line)
        if m is not None:
          all_distances_scores_path = m.group(1).strip()
        m = re_all_distances_metric.match(line)
        if m is not None:
          all_distances_metric = m.group(1).strip()
        m = re_all_distances_sample.match(line)
        if m is not None:
          all_distances_sample_every_n = int(m.group(1).strip())
        if "Exception in" in line:
          hit_exception = True
    finally:
      if DO_RAM_MONITOR and ram_mon is not None:
        peak_mb = ram_mon.stop()
        heap_str = f"{ram_mon.peak_heap_mb:.1f} MB" if ram_mon.have_heap else "n/a (no jstat?)"
        print(
          f"peak RSS for this run: {peak_mb:.1f} MB  |  peak JVM heap used: {heap_str}  "
          f"(chart: {ram_mon.html_file_name})"
        )

      if DO_PS:
        print("now stop ps process...")
        ps_process.stop()

      if DO_VMSTAT:
        print(f"now stop vmstat (pid={vmstat_process.pid})...")
        # TODO: messy!  can we get process group working so we can kill bash and its child reliably?
        # pkill returns 1 when no process matched (already exited); that is not an error here
        subprocess.call(["pkill", "-u", benchUtil.get_username(), "vmstat"])
        if vmstat_process.poll() is None:
          raise RuntimeError("failed to kill vmstat child process?  pid={vmstat_process.pid}")

    if DO_VMSTAT and GNUPLOT_PATH is not None:
      vmstat_subdir_name = write_vmstat_pretties(vmstat_log_file_name, this_cmd)
      str_this_cmd = shlex.join(this_cmd)
      # each run creates a new subdir with the N (cpu, io, memory, ...) charts
      vmstat_index_out.write(f'\n<a href="{vmstat_subdir_name}/index.html"><tt>run {index_run}</tt>: <tt>{str_this_cmd}</tt></a><br><br>\n')

    if hit_exception:
      raise RuntimeError("unhandled java exception while running")
    job.wait()
    if job.returncode != 0:
      raise RuntimeError(f"command failed with exit {job.returncode}")
    if summary is None:
      raise RuntimeError("could not find summary line in output! ")

    if exact_nn_scores_path is not None:
      generate_exact_nn_histogram(exact_nn_scores_path, log_dir_name, log_file_name, exact_nn_metric)

    if hnsw_traversal_scores_path is not None:
      generate_hnsw_traversal_histogram(hnsw_traversal_scores_path, log_dir_name, log_file_name, hnsw_traversal_metric)

    if all_distances_scores_path is not None:
      generate_all_distances_histogram(all_distances_scores_path, log_dir_name, log_file_name, all_distances_metric, all_distances_sample_every_n)

    summary_fields = summary.split("\t")
    if len(summary_fields) != len(OUTPUT_HEADERS):
      raise RuntimeError(f"SUMMARY has {len(summary_fields)} fields but expected {len(OUTPUT_HEADERS)}; summary line was: {summary!r}")
    all_results.append((summary, args))
    if DO_PROFILING:
      benchUtil.profilerOutput(constants.JAVA_EXE, jfr_output, benchUtil.checkoutToPath(checkout), 30, (1, 4, 12))

    if perf_stat_simd_file is not None:
      counters = parse_perf_stat_file(perf_stat_simd_file)
      is_quant = quantize_bits is not None and quantize_bits != 32
      report, dominant = format_simd_report(counters, is_quantized=is_quant)
      all_simd_reports.append((index_run, report, dominant))
      print(f"  run {index_run} {report}")
      if dominant in ("FP-scalar", "none"):
        print(f"  raw perf stat output ({perf_stat_simd_file}):")
        try:
          for line in Path(perf_stat_simd_file).read_text().splitlines():
            print(f"    {line}")
        except OSError as e:
          print(f"    (could not read: {e})")

    index_run += 1

  if NOISY:
    print("\nResults:")

  skip_headers = set()

  # skip columns that have the same value for every row
  if len(all_results) > 1:
    for col in range(len(OUTPUT_HEADERS)):
      unique_values = set([result[0].split("\t")[col] for result in all_results])
      if len(unique_values) == 1:
        skip_headers.add(OUTPUT_HEADERS[col])
        print(f"NOTE: {OUTPUT_HEADERS[col]} = {unique_values.pop()} for all runs; skipping column")

  print_fixed_width(all_results, skip_headers)
  print_chart(all_results)

  # SIMD summary across all runs
  if all_simd_reports:
    print("\nSIMD validation summary:")
    for run_num, report, dominant in all_simd_reports:
      print(f"  run {run_num}: {report}")
    # warn if any run had scalar-dominant or no SIMD
    bad_runs = [(r, d) for r, _, d in all_simd_reports if d in ("FP-scalar", "none")]
    if bad_runs:
      print(f"\n  WARNING: {len(bad_runs)} run(s) without SIMD vectorization: runs {', '.join(str(r) for r, _ in bad_runs)}")
      print("  Check that JVM is using --add-modules jdk.incubator.vector and that the Lucene")
      print("  codec supports vectorized similarity functions for your metric/encoding.")
    else:
      levels = {d for _, _, d in all_simd_reports}
      print(f"\n  All {len(all_simd_reports)} run(s) used SIMD: {', '.join(sorted(levels))}")

  if DO_VMSTAT and GNUPLOT_PATH is not None:
    print(f"\nNOTE: open {vmstat_index_html_path} in browser to see CPU/IO telemetry of each run")
  return all_results, skip_headers


def write_vmstat_pretties(vmstat_log_file_name, full_cmd):
  # print(f"write vmstat pretties from log={vmstat_log_file_name}")

  vmstat_log_path = Path(vmstat_log_file_name)
  dir_name = vmstat_log_path.parent
  base_name = vmstat_log_path.stem
  ext = vmstat_log_path.suffix

  vmstat_dir_name = f"{dir_name}/{base_name}-vmstat-charts"
  job_index_html_file = f"{vmstat_dir_name}/index.html"

  # print(f"see {vmstat_dir_name}/index.html for vmstat visualization")
  os.mkdir(vmstat_dir_name)

  # our own little pushd/popd!
  cwd = os.getcwd()
  try:
    # TODO: optimize to single shared copy!
    # TODO: don't hardwire version / full path to this js file!
    shutil.copy("/usr/share/gnuplot/6.0/js/gnuplot_svg.js", vmstat_dir_name)
    shutil.copy(f"{constants.BENCH_BASE_DIR}/src/vmstat/index.html.template", f"{vmstat_dir_name}/index.html")

    # because gnuplot needs to be in this directory (?)
    # the gnuplot script (src/vmstat/vmstat.gpi) writes output to ".":
    # print(f"cd {vmstat_dir_name=}")
    os.chdir(dir_name)
    try:
      subprocess.check_call(f"{GNUPLOT_PATH} -c {constants.BENCH_BASE_DIR}/src/vmstat/vmstat.gpi {vmstat_log_file_name} {base_name}-vmstat-charts", shell=True)
    except subprocess.CalledProcessError as e:
      print(f"WARNING: gnuplot failed to generate vmstat charts (run may have been too short?): {e}")
  finally:
    os.chdir(cwd)

  return os.path.split(vmstat_dir_name)[1]


def print_fixed_width(all_results, columns_to_skip):
  headers = OUTPUT_HEADERS
  num_columns = len(headers)
  skip_column_index = {headers.index(h) for h in columns_to_skip}

  data_rows = [result[0].split("\t") for result in all_results]

  for row in data_rows:
    if len(row) != num_columns:
      raise RuntimeError(f'wrong number of columns: expected {num_columns} but got {len(row)} in row "{chr(9).join(row)}"')

  # active columns (not skipped), split into hyperparams (inputs) and metrics (outputs)
  active_cols = [i for i in range(num_columns) if i not in skip_column_index]
  metric_col_indices = {i for i, h in enumerate(headers) if h in _METRIC_HEADERS}
  hyperparam_cols = [c for c in active_cols if c not in metric_col_indices]
  metric_cols = [c for c in active_cols if c in metric_col_indices]

  # sort hyperparam columns by transition count: fewest transitions = slowest-changing = leftmost (odometer MSB)
  def count_transitions(col):
    if len(data_rows) <= 1:
      return 0
    return sum(1 for r in range(1, len(data_rows)) if data_rows[r][col] != data_rows[r - 1][col])

  hyperparam_cols.sort(key=count_transitions)

  # sort rows by hyperparam column values in that odometer order; numeric comparison where possible
  def row_sort_key(row):
    key = []
    for col in hyperparam_cols:
      f = _try_float(row[col])
      key.append((0, f) if f is not None else (1, row[col]))
    return key

  data_rows_sorted = sorted(data_rows, key=row_sort_key)

  # final column order: hyperparams (odometer) then metrics
  ordered_cols = hyperparam_cols + metric_cols

  # compute max column widths (plain text, no ANSI codes)
  all_rows = [headers] + data_rows_sorted
  max_by_col = [0] * num_columns
  for row in all_rows:
    for i, s in enumerate(row):
      max_by_col[i] = max(max_by_col[i], len(s))

  # ANSI color setup (only when writing to a terminal)
  use_color = sys.stdout.isatty()

  recall_col = headers.index("recall") if "recall" in headers else None
  latency_col = headers.index("latency(ms)") if "latency(ms)" in headers else None

  lat_floats = []
  if latency_col is not None and latency_col not in skip_column_index:
    for row in data_rows_sorted:
      f = _try_float(row[latency_col])
      if f is not None:
        lat_floats.append(f)
  min_lat_ms = min(lat_floats) if lat_floats else None
  max_lat_ms = max(lat_floats) if lat_floats else None

  def colorize(col_idx, val_str):
    if not use_color:
      return val_str
    code = None
    if col_idx == recall_col:
      f = _try_float(val_str)
      if f is not None:
        if f >= 0.99:
          code = _ANSI_GREEN
        elif f >= 0.9:
          code = _ANSI_YELLOW
        else:
          code = _ANSI_RED
    elif col_idx == latency_col and min_lat_ms is not None and max_lat_ms != min_lat_ms:
      f = _try_float(val_str)
      if f is not None:
        t = (f - min_lat_ms) / (max_lat_ms - min_lat_ms)
        if t <= 0.33:
          code = _ANSI_GREEN
        elif t <= 0.67:
          code = _ANSI_YELLOW
        else:
          code = _ANSI_RED
    if code is not None:
      return f"{code}{val_str}{_ANSI_RESET}"
    return val_str

  for row_idx, row in enumerate(all_rows):
    is_header = row_idx == 0
    parts = []
    for col in ordered_cols:
      w = max_by_col[col]
      val = row[col]
      # pad first (plain), then wrap value in color so ANSI codes don't affect alignment
      padding = w - len(val)
      cell = " " * padding + (val if is_header else colorize(col, val))
      parts.append(cell)
    print("  ".join(parts))


def arglist_to_argmap(arglist):
  """Map args starting with - to the next value in the list unless it starts with -
  in which case map to the empty string
  """
  argmap = dict()
  for i in range(len(arglist)):
    if arglist[i][0] == "-":
      if i < len(arglist) - 1 and arglist[i + 1][0] != "-":
        argmap[arglist[i]] = arglist[i + 1]
      else:
        argmap[arglist[i]] = ""
  return argmap


def remove_common_args(argmaps):
  common_args = argmaps[0].copy()
  for args in argmaps[1:]:
    for k in list(common_args.keys()):
      if k not in args or args[k] != common_args[k]:
        del common_args[k]
  # don't use fanout as a dimension in a data series label
  # TODO: also remove other "minor" dimensions such as beam_width and maxconn?
  # or place under user control somehow
  common_args["-fanout"] = 1
  # everything remaining is in common to all rows, now remove them
  unique_args = []
  for args in argmaps:
    ua = dict()
    for k, v in args.items():
      if k not in common_args:
        ua[k] = v
    unique_args.append(ua)
  return unique_args


CHART_HEADER = """
<!DOCTYPE html>
<html>
  <head>
    <script type="text/javascript" src="https://www.google.com/jsapi"></script>
    <script type="text/javascript">
      google.load("visualization", "1", {packages:["corechart"]});
      google.setOnLoadCallback(drawChart);
      function drawChart() {
        var data = google.visualization.arrayToDataTable([
"""

CHART_FOOTER = """        ]);

        var options = {
          //title: '$TITLE$',
          pointSize: 5,
          //legend: {position: 'none'},
          hAxis: {title: 'Recall'},
          vAxis: {title: 'CPU (msec)', direction: -1},
          interpolateNulls: true
        };

        var chart = new google.visualization.LineChart(document.getElementById('chart_div'));
        chart.draw(data, options);
      }
    </script>
  </head>
  <body>
    <div id="chart_div" style="width: 1200px; height: 600px;"></div>
  </body>
</html>
"""


def print_chart(results):
  # (recall, nCpu) for each result
  argmaps = [arglist_to_argmap(r[1]) for r in results]
  argmaps = remove_common_args(argmaps)
  # TODO: also remove "minor" args that may vary the performance but are not shown on the axes
  # and then show the corresponding value(s) in the tooltips
  output = CHART_HEADER
  labels = dict()
  for argmap in argmaps:
    label = chart_args_label(argmap)
    if label in labels:
      index = labels[label]
    else:
      index = len(labels)
      labels[label] = index

  output += str([""] + list(labels.keys()))
  output += ",\n"

  for i, row in enumerate(results):
    values = row[0].split("\t")
    label = chart_args_label(argmaps[i])
    index = labels[label]
    # recall on x axis, cpu on y axis. nulls for the other label indices
    data_row = [float(values[0])] + ["null"] * index + [float(values[2])] + ["null"] * (len(labels) - index - 1)
    output += str(data_row).replace("'null'", "null")
    output += ",\n"

  output += CHART_FOOTER
  with open("knnPerfChart.html", "w") as fout:
    print(output, file=fout)


def chart_args_label(args):
  if len(args) == 0:
    return "baseline"
  return str(args)


def print_cpu_info():
  """Read and print CPU information from /proc/cpuinfo if present (linux only)"""
  cpuinfo_path = "/proc/cpuinfo"

  if not os.path.exists(cpuinfo_path):
    print("CPU info: /proc/cpuinfo not found (not running on Linux)")
    return

  # parse /proc/cpuinfo - it repeats info for each logical core
  cpu_info = {}
  processor_count = 0

  with open(cpuinfo_path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue

      if ":" in line:
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        # count processors
        if key == "processor":
          processor_count += 1

        # capture these fields (they repeat for each core, so we only need one)
        elif key == "model name" and "model_name" not in cpu_info:
          cpu_info["model_name"] = value
        elif key == "cpu cores" and "cpu_cores" not in cpu_info:
          cpu_info["cpu_cores"] = value
        elif key == "microcode" and "microcode" not in cpu_info:
          cpu_info["microcode"] = value
        elif key == "flags" and "flags" not in cpu_info:
          cpu_info["flags"] = value
        elif key == "vmx flags" and "vmx_flags" not in cpu_info:
          cpu_info["vmx_flags"] = value
        elif key == "bugs" and "bugs" not in cpu_info:
          cpu_info["bugs"] = value

  # print cpu information
  print("\nCPU Information:")
  print(f"  model: {cpu_info.get('model_name', 'unknown')}")
  print(f"  logical cores: {processor_count}")
  print(f"  physical cores per socket: {cpu_info.get('cpu_cores', 'unknown')}")
  print(f"  microcode: {cpu_info.get('microcode', 'unknown')}")

  flags = cpu_info.get("flags", "")
  if flags:
    print(f"  flags: {flags}")
  else:
    print("  flags: unknown")

  vmx_flags = cpu_info.get("vmx_flags", "")
  if vmx_flags:
    print(f"  vmx flags: {vmx_flags}")
  else:
    print("  vmx flags: none")

  bugs = cpu_info.get("bugs", "")
  if bugs:
    print(f"  bugs: {bugs}")
  else:
    print("  bugs: none")

  print()


def format_memory_kb(value_str):
  """Convert memory value from kB to appropriate units (kB, MB, GB, TB)"""
  try:
    # parse value (format is like "65916396 kB")
    parts = value_str.split()
    if len(parts) < 1:
      return value_str

    value_kb = int(parts[0])

    # choose appropriate unit
    if value_kb >= 1024 * 1024 * 1024:  # >= 1 TB
      return f"{value_kb / (1024 * 1024 * 1024):.2f} TB"
    if value_kb >= 1024 * 1024:  # >= 1 GB
      return f"{value_kb / (1024 * 1024):.2f} GB"
    if value_kb >= 1024:  # >= 1 MB
      return f"{value_kb / 1024:.2f} MB"
    return f"{value_kb} kB"
  except (ValueError, IndexError):
    return value_str


def print_mem_info():
  """Read and print memory information from /proc/meminfo if present (linux only)"""
  meminfo_path = "/proc/meminfo"

  if not os.path.exists(meminfo_path):
    print("Memory info: /proc/meminfo not found (not running on Linux)")
    return

  mem_info = {}

  with open(meminfo_path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue

      if ":" in line:
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        # capture memory fields
        if key == "MemTotal":
          mem_info["mem_total"] = value
        elif key == "MemFree":
          mem_info["mem_free"] = value
        elif key == "MemAvailable":
          mem_info["mem_available"] = value
        elif key == "Dirty":
          mem_info["mem_dirty"] = value

  # print memory information
  print("Memory Information:")

  mem_total = mem_info.get("mem_total", "unknown")
  mem_free = mem_info.get("mem_free", "unknown")
  mem_available = mem_info.get("mem_available", "unknown")
  mem_dirty = mem_info.get("mem_dirty", "unknown")

  print(f"  total RAM: {format_memory_kb(mem_total) if mem_total != 'unknown' else 'unknown'}")
  print(f"  free RAM: {format_memory_kb(mem_free) if mem_free != 'unknown' else 'unknown'}")
  print(f"  available RAM: {format_memory_kb(mem_available) if mem_available != 'unknown' else 'unknown'}")

  # calculate used ram if we have total and available
  if "mem_total" in mem_info and "mem_available" in mem_info:
    try:
      # parse values (they come in format like "65916396 kB")
      total_kb = int(mem_info["mem_total"].split()[0])
      available_kb = int(mem_info["mem_available"].split()[0])
      used_kb = total_kb - available_kb
      print(f"  used RAM: {format_memory_kb(f'{used_kb} kB')}")
    except (ValueError, IndexError):
      print("  used RAM: unknown")
  else:
    print("  used RAM: unknown")

  print(f"  dirty RAM: {format_memory_kb(mem_dirty) if mem_dirty != 'unknown' else 'unknown'}")
  print()


def check_knn_compiled():
  """Hard exit if KNN Java classes are missing or out of date vs source files."""
  build_dir = Path(constants.BENCH_BASE_DIR) / "build"
  src_dir = Path(constants.BENCH_BASE_DIR) / "src" / "main"

  marker = build_dir / "knn" / "KnnGraphTester.class"

  source_files = list((src_dir / "knn").glob("*.java"))
  source_files.extend([src_dir / "WikiVectors.java", src_dir / "perf" / "VectorDictionary.java"])

  gradle_cmd = "  JAVA_HOME=/usr/lib/jvm/java-26-openjdk ./gradlew compileKnn"

  if not marker.exists():
    print(f"\nERROR: {marker} does not exist. Run:\n\n{gradle_cmd}\n")
    raise SystemExit(1)

  marker_mtime = marker.stat().st_mtime
  stale = [src for src in source_files if src.exists() and src.stat().st_mtime > marker_mtime]

  if len(stale) > 0:
    print("\nERROR: KNN Java classes are out of date. Stale source files:")
    for src in stale:
      print(f"  {src}")
    print(f"\nRun:\n\n{gradle_cmd}\n")
    raise SystemExit(1)

  lucene_checkout = getLuceneDirFromGradleProperties()
  lucene_path = benchUtil.checkoutToPath(lucene_checkout)
  lucene_gradle_cmd = f"  JAVA_HOME=/usr/lib/jvm/java-26-openjdk ./gradlew compileJava\n  (in {lucene_path})"

  try:
    cp_entries = benchUtil.getClassPath(lucene_checkout)
  except RuntimeError as e:
    print(f"\nERROR: Lucene checkout at {lucene_path} is missing built artifacts: {e}\n\nRun:\n\n{lucene_gradle_cmd}\n")
    raise SystemExit(1) from e

  for entry in cp_entries:
    p = Path(entry)
    if not str(p).startswith(lucene_path):
      continue
    if not p.exists():
      print(f"\nERROR: Lucene classpath entry missing: {p}\n\nRun:\n\n{lucene_gradle_cmd}\n")
      raise SystemExit(1)


def build_java_base_cmd(checkout):
  """Build the base Java command (JVM flags + classpath) for KnnGraphTester."""
  cp = benchUtil.classPathToString(benchUtil.getClassPath(checkout) + (f"{constants.BENCH_BASE_DIR}/build",))
  cmd = constants.JAVA_EXE.split(" ") + [
    # Heap cap: JAVA_EXE is the bare java path (no -Xmx), so without this the index/search JVM ran at
    # the JVM DEFAULT max heap (~1/4 RAM), silently ignoring KNN_HEAP. At high hashBits the per-segment
    # centroid arrays (2^hashBits * dim floats) blow past that default -> OOM. KNN_HEAP now applies.
    # Only -Xmx (no -Xms): the JVM starts with a small heap and grows lazily to the cap, so it COMMITS
    # only what the live set needs instead of reserving the full cap up front. On a 32GB box -Xms=-Xmx=24g
    # reserved 24GB resident -> ~8GB swap even though peak USED was ~15-20GB. Lazy commit avoids that.
    f"-Xmx{constants.KNN_HEAP}",
    "-cp",
    cp,
    "--add-modules",
    "jdk.incubator.vector",
    "--enable-native-access=ALL-UNNAMED",
    f"-Djava.util.concurrent.ForkJoinPool.common.parallelism={multiprocessing.cpu_count()}",
    "-XX:+UnlockDiagnosticVMOptions",
    "-XX:+DebugNonSafepoints",
  ]
  # Debug: dump LSH columnar scan I/O stats (filter vs payload bytes, survivor fraction) at JVM exit.
  if LSH_SCAN_STATS:
    cmd += ["-Dlsh.scanStats=true"]
  if LSH_OFF_HEAP_CENTROIDS:
    cmd += ["-Dlsh.offHeapCentroids=true"]
  if LSH_POSITIONAL_SCAN:
    cmd += ["-Dlsh.positionalScan=true"]
  if LSH_HNSW_ROUTING:
    cmd += ["-Dlsh.hnswRouting=true"]
    if LSH_HNSW_M is not None:
      cmd += [f"-Dlsh.hnswM={LSH_HNSW_M}"]
    if LSH_HNSW_BEAM_WIDTH is not None:
      cmd += [f"-Dlsh.hnswBeamWidth={LSH_HNSW_BEAM_WIDTH}"]
    if LSH_HNSW_OVERQUERY is not None:
      cmd += [f"-Dlsh.hnswOverquery={LSH_HNSW_OVERQUERY}"]
    if LSH_ROUTE_ON_REFERENCE_CENTROIDS:
      cmd += ["-Dlsh.routeOnReferenceCentroids=true"]
  if LSH_REFERENCE_CENTROIDS_ONLY:
    cmd += ["-Dlsh.referenceCentroidsOnly=true"]

  if IVF_CENTROID_HNSW is not None:
    cmd += [f"-Divf.centroidHnsw={'true' if IVF_CENTROID_HNSW else 'false'}"]
  if IVF_REFINE_FACTOR is not None:
    cmd += [f"-Divf.refineFactor={IVF_REFINE_FACTOR}"]
  if IVF_ADAPTIVE_NPROBE_MARGIN is not None:
    cmd += [f"-Divf.adaptiveNprobeMargin={IVF_ADAPTIVE_NPROBE_MARGIN}"]
  if LLOYD_BEAM_FACTOR is not None:
    cmd += [f"-Dlloyd.beamFactor={LLOYD_BEAM_FACTOR}"]
  if LLOYD_SCORE_IN_PLACE:
    cmd += ["-Dlloyd.scoreInPlace=true"]
  # LLOYD_SHORTLIST_DEDUP=1 => -Dlloyd.shortlistDedup=true: move the spill dedup off the per-slot scan
  # (JFR: FixedBitSet.getAndSet = 18.8% of warm CPU, ~250k random writes into a 4.75 MB bitset per query,
  # plus a fresh maxDoc bitset allocated per query) and onto the BRUTE_N shortlist instead. Reader-side,
  # no reindex; asserted bit-identical by TestUringRerankGather's shortlist-dedup cases.
  if os.environ.get("LLOYD_SHORTLIST_DEDUP") == "1":
    cmd += ["-Dlloyd.shortlistDedup=true"]
  # LLOYD_CO_RESIDENT_CODES=1 => -Dlloyd.coResidentCodes=true: Stage G. Read each probed cell's CODE run
  # contiguously alongside its sketch run, so the shortlist's int8 records are already in RAM and Stage D's
  # ~2000 scattered reads are skipped. Reads ~8.2x more coarse bytes (only ~1% get reranked) in exchange
  # for zero random reads -- wins where I/O time is hidden (CPU-bound) or storage is fast.
  if os.environ.get("LLOYD_CO_RESIDENT_CODES") == "1":
    cmd += ["-Dlloyd.coResidentCodes=true"]
  # LLOYD_SCALAR_POPCOUNT=1 => -Dlloyd.scalarPopcount=true: CONTROL arm that restores the old
  # single-accumulator Hamming loop. The default (unset) uses four independent accumulators so the
  # XOR+CNT work pipelines and C2 auto-vectorizes -- xorBitCountInt was 26.5% of warm CPU, the top term.
  if os.environ.get("LLOYD_SCALAR_POPCOUNT") == "1":
    cmd += ["-Dlloyd.scalarPopcount=true"]
  if LLOYD_PREFETCH_CELLS:
    cmd += ["-Dlloyd.prefetchCells=true"]
  if os.environ.get("LLOYD_URING_SKETCH_SCAN") == "1":
    cmd += ["-Dlloyd.uringSketchScan=true"]
  # Stage D: batch+coalesce the rerank code-record reads (the component that dominates cold reads --
  # Stage C only batches the sketch runs). Reader-side; bit-identical results.
  if os.environ.get("LLOYD_URING_RERANK") == "1":
    cmd += ["-Dlloyd.uringRerank=true"]
  # Stage E: dispatch each cell's read as it is selected and scan cells as their bytes land, so reads
  # overlap the Hamming scan instead of running as a separate blocking phase.
  if os.environ.get("LLOYD_URING_PIPELINE") == "1":
    cmd += ["-Dlloyd.uringPipeline=true"]
  # Stage F: STREAMING rerank -- submit the coalesced code-record ranges and score each candidate as its
  # bytes land, so the int8 rerank overlaps the cold reads instead of blocking for the whole gather first.
  if os.environ.get("LLOYD_URING_RERANK_PIPELINE") == "1":
    cmd += ["-Dlloyd.uringRerankPipeline=true"]
  # O_DIRECT: open the reader's data file cache-bypassing, forcing the >RAM regime for the ring alone.
  # A/B arm only -- measures the pure-device ceiling; not the realistic partial-cache number.
  if os.environ.get("LLOYD_URING_DIRECT") == "1":
    cmd += ["-Dlloyd.uringDirect=true"]
  if os.environ.get("LLOYD_PIPELINE_DEPTH") is not None:
    cmd += [f"-Dlloyd.pipelineDepth={os.environ['LLOYD_PIPELINE_DEPTH']}"]
  if os.environ.get("LLOYD_RERANK_AUDIT") == "1":
    cmd += ["-Dlloyd.rerankAudit=true"]
  if os.environ.get("LLOYD_RERANK_COALESCE_GAP") is not None:
    cmd += [f"-Dlloyd.rerankCoalesceGap={os.environ['LLOYD_RERANK_COALESCE_GAP']}"]
  if os.environ.get("LLOYD_URING_DEBUG") == "1":
    cmd += ["-Dlloyd.uringDebug=true"]
  if os.environ.get("LLOYD_NO_HEAP_SKIP") == "1":
    cmd += ["-Dlloyd.noHeapSkip=true"]
  if os.environ.get("KNN_DROP_CACHE_AFTER_WARMUP") == "1":
    cmd += ["-Dknn.dropCacheAfterWarmup=true"]
  if os.environ.get("LLOYD_CEIL_K"):
    cmd += [f'-Dlloyd.ceilK={os.environ["LLOYD_CEIL_K"]}']
  if IVF_STREAM_REFINE_ITERS is not None:
    cmd += [f"-Divf.streamRefineIters={IVF_STREAM_REFINE_ITERS}"]
  if IVF_CENTROID_HNSW_M is not None:
    cmd += [f"-Divf.centroidHnswM={IVF_CENTROID_HNSW_M}"]
  if IVF_DERIVED_GRAPH:
    cmd += ["-Divf.derivedGraph=true"]
  if IVF_DERIVED_GRAPH_M is not None:
    cmd += [f"-Divf.derivedGraphM={IVF_DERIVED_GRAPH_M}"]
  if IVF_CENTROID_HNSW_BEAM_WIDTH is not None:
    cmd += [f"-Divf.centroidHnswBeamWidth={IVF_CENTROID_HNSW_BEAM_WIDTH}"]
  if IVF_SPILL_EF_SEARCH is not None:
    cmd += [f"-Divf.spillEfSearch={IVF_SPILL_EF_SEARCH}"]
  # IVF_REUSE_GRAPH_TOPOLOGY=0 => -Divf.reuseGraphTopology=false, restoring a full HNSW build at every
  # merge routing stage instead of refreshing the donor graph's int8 codes over the moved centroids. The
  # codec DEFAULTS this on, so 0 is the control arm for pricing the indexing win.
  #
  # WRITE-TIME and NOT in the index key: it changes the persisted clustering (a reused topology routes docs
  # slightly differently than a freshly built one), so the two arms MUST NOT share a cached index. Run the
  # control with KNN_CLEAR_CACHE=1 -- otherwise the "off" arm silently reuses the "on" arm's index and
  # measures nothing. Same class of trap as blockP below.
  if os.environ.get("IVF_REUSE_GRAPH_TOPOLOGY") == "0":
    cmd += ["-Divf.reuseGraphTopology=false"]
  if os.environ.get("IVF_QUANTIZER"):
    cmd += [f'-Divf.quantizer={os.environ["IVF_QUANTIZER"]}']
  # Block size p for -Divf.quantizer=blocksphere. Write-time (it sets the on-disk bytes/block), and the
  # codec default is already 2, so an unset value happens to give p=2 today -- pass it explicitly anyway
  # so the run does not silently depend on that default. NOTE: blockP is NOT in the index key (only
  # qz<quantizer> is), so switching p against a cached index would misparse every record; clear
  # knn-reuse/indices when changing it. run_40m_p2_sweep.sh enforces this with a .blockP stamp.
  if os.environ.get("IVF_BLOCK_P"):
    cmd += [f'-Divf.blockP={os.environ["IVF_BLOCK_P"]}']
  if IVF_STREAM_FLUSH_MIN_DOCS is not None:
    cmd += [f"-Divf.streamFlushMinDocs={IVF_STREAM_FLUSH_MIN_DOCS}"]
  if IVF_TRAIN_SAMPLE_CAP is not None:
    cmd += [f"-Divf.trainSampleCap={IVF_TRAIN_SAMPLE_CAP}"]
  if os.environ.get("LLOYD_PREFETCH_AUDIT") == "1":
    cmd += ["-Dlloyd.prefetchAudit=true"]
  if LLOYD_PREFETCH_CODE_MAX_BYTES is not None:
    cmd += [f"-Dlloyd.prefetchCodeMaxBytes={LLOYD_PREFETCH_CODE_MAX_BYTES}"]
  if LLOYD_CEIL_AUDIT:
    cmd += ["-Dlloyd.ceilAudit=true"]
  if LLOYD_CEIL_DIR:
    cmd += ["-Dlloyd.ceilDir=true"]
  if LLOYD_CEIL_ANISO:
    cmd += ["-Dlloyd.ceilAniso=true"]
  if LLOYD_CEIL_SUB:
    cmd += ["-Dlloyd.ceilSub=true"]
  if LLOYD_CEIL_SEP:
    cmd += ["-Dlloyd.ceilSep=true"]
  if LLOYD_CEIL_ONLINE:
    cmd += ["-Dlloyd.ceilOnline=true"]
  if LLOYD_CEIL_ADJ:
    cmd += ["-Dlloyd.ceilAdj=true"]
  if LLOYD_CEIL_SELEXP:
    cmd += ["-Dlloyd.ceilSelExp=true"]
  if LLOYD_CEIL_BRUTE:
    cmd += ["-Dlloyd.ceilBrute=true"]
  if LLOYD_CEIL_SPILL:
    cmd += ["-Dlloyd.ceilSpill=true"]
  if GCUT_TREE_ROUTE_DIMS is not None:
    cmd += [f"-Dgcut.treeRouteDims={GCUT_TREE_ROUTE_DIMS}"]
  if GCUT_TREE_ROUTE:
    cmd += ["-Dgcut.treeRoute=true"]
  if LLOYD_BRUTE_SEARCH:
    cmd += ["-Dlloyd.bruteSearch=true"]
    cmd += ["-Dlloyd.ceilBruteDims=1024"]
  if IVF_QUANT_BITS:
    cmd += [f"-Divf.quantBits={IVF_QUANT_BITS}"]
  if os.environ.get("IVF_CELL_ORDER", "1") == "1":
    cmd += ["-Divf.cellOrder=true"]
  if IVF_BEAM_SPILL:
    cmd += ["-Divf.beamSpill=true"]
    if IVF_SPILL_MARGIN:
      cmd += [f"-Divf.spillMargin={IVF_SPILL_MARGIN}"]
  if os.environ.get("LLOYD_DBG_CELL") == "1":
    cmd += ["-Dlloyd.dbgCell=true"]
  if LLOYD_SKETCH_SCAN:
    cmd += ["-Dlloyd.sketchScan=true"]
    if os.environ.get("LLOYD_BRUTE_N"):
      cmd += [f'-Dlloyd.bruteN={os.environ["LLOYD_BRUTE_N"]}']
    _sd = os.environ.get("LLOYD_SKETCH_DIMS", "1024")
    cmd += [f"-Dlloyd.ceilBruteDims={_sd}"]
    cmd += [f"-Dlloyd.sketchDims={_sd}"]
    if LLOYD_RERANK_BITS:
      cmd += [f"-Dlloyd.rerankBits={LLOYD_RERANK_BITS}"]
  if IVF_RERANK_FACTOR is not None:
    cmd += [f"-Divf.rerankFactor={IVF_RERANK_FACTOR}"]
  if IVF_ENABLE_COPY_MERGE:
    cmd += ["-Divf.enableCopyMerge=true"]
  if IVF_EXACT_ASSIGN:
    cmd += ["-Divf.exactAssign=true"]
  if IVF_WORK_DIMS is not None:
    cmd += [f"-Divf.workDims={IVF_WORK_DIMS}"]
  if IVF_GRAPH_ROUTE_ITERS is not None:
    cmd += [f"-Divf.graphRouteIters={IVF_GRAPH_ROUTE_ITERS}"]
  if IVF_ANISO_ETA is not None:
    cmd += [f"-Divf.anisoEta={IVF_ANISO_ETA}"]
  if IVF_SHARED_CODES:
    cmd += ["-Divf.sharedCodes=true"]
  if IVF_DROP_RAW_VECTORS:
    cmd += ["-Divf.dropRawVectors=true"]
  cmd += ["knn.KnnGraphTester"]
  return cmd


def build_knn_args_from_params(params):
  """Convert a flat dict of param_name->value into the arg list for KnnGraphTester."""
  args = []
  quantize_bits = None
  do_quantize_compress = False
  do_rerank = False
  rerank_quantize_bits = 32
  for p, value in params.items():
    if p == "quantizeBits":
      if value != 32:
        args += ["-quantize", "-quantizeBits", str(value)]
        quantize_bits = value
    elif p == "quantizeCompress":
      do_quantize_compress = value
    elif p == "rerank":
      if value:
        do_rerank = True
    elif p == "rerankQuantizeBits":
      rerank_quantize_bits = value
    elif isinstance(value, bool):
      if value:
        args += ["-" + p]
    else:
      args += ["-" + p, str(value)]

  if quantize_bits == 4 and do_quantize_compress:
    args += ["-quantizeCompress"]

  if do_rerank:
    args += ["-rerank"]
    if rerank_quantize_bits != 32:
      args += ["-rerankQuantizeBits", str(rerank_quantize_bits)]

  return args


def run_single_knn_iteration(checkout, params, dim, doc_vectors, query_vectors, work_dir, extra_java_args=None):
  """Run a single KNN benchmark iteration in work_dir.  Always reindexes.

  params: flat dict of param_name -> single value (not tuple)
  Returns (summary_string, full_output_string) or raises on failure.
  """
  base_cmd = build_java_base_cmd(checkout)
  knn_args = build_knn_args_from_params(params)

  full_cmd = (
    base_cmd
    + knn_args
    + [
      "-dim",
      str(dim),
      "-docs",
      str(doc_vectors),
      "-reindex",
      "-search-and-stats",
      str(query_vectors),
      "-numIndexThreads",
      str(NUM_INDEX_THREADS),
    ]
  )

  if extra_java_args is not None:
    full_cmd += extra_java_args

  perf_stat_simd_file = None
  if DO_PERF_STAT_SIMD:
    perf_stat_simd_file = str(Path(work_dir) / "perf-simd.txt")
    full_cmd = wrap_cmd_with_perf_stat_simd(full_cmd, perf_stat_simd_file)

  print(f"[variance] running in {work_dir}")
  print(f"[variance] cmd: {full_cmd}")

  os.makedirs(work_dir, exist_ok=True)

  job = subprocess.Popen(
    full_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    encoding="utf-8",
    cwd=str(work_dir),
  )

  output_lines = []
  re_summary = re.compile(r"^SUMMARY: (.*?)$", re.MULTILINE)
  summary = None
  hit_exception = False

  while job.poll() is None:
    line = job.stdout.readline()
    if not line:
      continue
    output_lines.append(line)
    sys.stdout.write(line)
    sys.stdout.flush()
    m = re_summary.match(line)
    if m is not None:
      summary = m.group(1)
    if "Exception in" in line:
      hit_exception = True

  # drain remaining output
  for line in job.stdout:
    output_lines.append(line)
    sys.stdout.write(line)
    sys.stdout.flush()
    m = re_summary.match(line)
    if m is not None:
      summary = m.group(1)
    if "Exception in" in line:
      hit_exception = True

  full_output = "".join(output_lines)

  if hit_exception:
    raise RuntimeError(f"java exception in {work_dir}:\n{full_output}")
  job.wait()
  if job.returncode != 0:
    raise RuntimeError(f"command failed with exit {job.returncode} in {work_dir}:\n{full_output}")
  if summary is None:
    raise RuntimeError(f"could not find SUMMARY line in output from {work_dir}:\n{full_output}")

  if perf_stat_simd_file is not None:
    counters = parse_perf_stat_file(perf_stat_simd_file)
    is_quant = params.get("quantizeBits", 32) != 32
    report, dominant = format_simd_report(counters, is_quantized=is_quant)
    print(f"  {report}")
    if dominant in ("scalar", "none"):
      print(f"  raw perf stat output ({perf_stat_simd_file}):")
      try:
        for line in Path(perf_stat_simd_file).read_text().splitlines():
          print(f"    {line}")
      except OSError as e:
        print(f"    (could not read: {e})")

  return summary, full_output


def run_n_knn_benchmarks(LUCENE_CHECKOUT, PARAMS, n, log_path):
  rec, lat, net, avg = [], [], [], []
  tests = []
  for i in range(n):
    results, skip_headers = run_knn_benchmark(LUCENE_CHECKOUT, PARAMS, log_path)
    tests.append(results)
    first_4_numbers = results[0][0].split("\t")[:4]
    first_4_numbers = [float(num) for num in first_4_numbers]

    # store relevant data points
    rec.append(first_4_numbers[0])
    lat.append(first_4_numbers[1])
    net.append(first_4_numbers[2])
    avg.append(first_4_numbers[3])

  # reconstruct string with median results
  med_results = []
  med_string = ""
  med_string += f"{round(statistics.median(rec), 3)}\t"
  med_string += f"{round(statistics.median(lat), 3)}\t"
  med_string += f"{round(statistics.median(net), 3)}\t"
  med_string += f"{round(statistics.median(avg), 3)}\t"

  split_results = results[0][0].split("\t")
  split_string = "\t".join(split_results[4:])
  med_string += split_string
  med_tuple = (med_string, results[0][1])
  med_results.append(med_tuple)

  # re-print all tables in a row
  print("\nFinal Results:")
  for i in range(n):
    print(f"\nTest {i + 1}:")
    print_fixed_width(tests[i], skip_headers)

  # print median results in table
  print("\nMedian Results:")
  print_chart(med_results)
  print_fixed_width(med_results, skip_headers)


if __name__ == "__main__":
  with autologger.capture_output() as log_path:
    log_path = Path(log_path)
    log_dir_name = log_path.parent
    log_base_name = log_path.stem
    log_ext = log_path.suffix

    # print cpu and memory information at the start
    print_cpu_info()
    print_mem_info()

    parser = argparse.ArgumentParser(description="Run KNN benchmarks")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to run the benchmark (default: 1)")
    n = parser.parse_args()

    constants.check_java_home()
    check_knn_compiled()

    # Where the version of Lucene is that will be tested. Now this will be sourced from gradle.properties
    LUCENE_CHECKOUT = getLuceneDirFromGradleProperties()
    if n.runs == 1:
      run_knn_benchmark(LUCENE_CHECKOUT, PARAMS, (log_dir_name, log_base_name))
    else:
      run_n_knn_benchmarks(LUCENE_CHECKOUT, PARAMS, n.runs, (log_dir_name, log_base_name))
