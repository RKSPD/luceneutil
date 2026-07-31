#!/usr/bin/env python3

# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download the FULL English config of Cohere's v3 multilingual Wikipedia embeddings into raw .vec.

Source: CohereLabs/wikipedia-2023-11-embed-multilingual-v3, config 'en' -- 41,488,110 paragraphs x
1024d float32, shipped as 415 parquet shards (~90GB compressed) via HF's auto-parquet endpoint.
Output is luceneutil's .vec format: one vector after another, little-endian float32, NO header
(41.5M x 4KB ~= 170GB).

Why this exists alongside initial_setup.py's build_large_vecs(): that one is serial
(download -> extract -> delete, one shard at a time) and multilingual. A single shard downloads at
~50MB/s but 4 concurrent shards reach ~115MB/s, so here downloads run in a small thread pool while
extraction consumes shards IN ORDER on the main thread. Ordered extraction keeps resume trivial
(truncate to the recorded vector count, restart at the next shard).

Queries: the bundled 200K query .vec is a separate luceneutil download, so instead we hold out whole
ARTICLES, exactly as cohere-v3-README.txt describes for the bundled corpus ("pick 250,000 random
wiki_ids as queries, the rest are docs ... no vectors are in common between docs and queries").

Article-level holdout is load-bearing, not tidiness. Rows arrive grouped by article (`_id` is
`20231101.en_<wiki_id>_<paragraph>`, ~7.1 consecutive paragraphs per article), and paragraphs of one
article are near-duplicates in embedding space. Holding out ROWS instead of ARTICLES breaks the
benchmark two ways: the held-out paragraphs' siblings stay in the doc set as trivial top-1 hits
(recall is then measured against giveaways), and a contiguous row range covers only a handful of
articles, so the query set is a clump of near-duplicates rather than a sample of the corpus. Measured
on a row-holdout build of this same corpus, knnExactNN's consistency check flagged both: query-query
similarity spread 0.2203 vs doc-doc 0.1720 (near-duplicate clumping) and 29.6% of dims with |z|>5
mean drift (unrepresentative sample).

An article is selected by a deterministic hash of its wiki_id, so the choice is uniform over the
whole corpus and identical no matter which shard a paragraph shows up in (articles can straddle shard
boundaries). ALL paragraphs of a selected article go to queries; docs and queries therefore share no
vector AND no article.

Both files land in --data-dir. The docs .vec comes out in corpus order; run shuffle_vecs.py on it to
get the 'scattered' distribution the IVF/LSH benchmarks want (adjacent Wikipedia paragraphs would
otherwise land in the same bucket and inflate recall).

Reachability note (inherited from initial_setup.py, still true): HF is reachable via `curl` using the
system CA store, but Python's urllib fails TLS in this venv (no CA certs) -- so downloads shell out
to curl.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq

HF_REPO = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"
LANG = "en"
DIM = 1024
EMB_COLUMN = "emb"
# `_id` looks like "20231101.en_13194570_4" = <dump>_<wiki_id>_<paragraph_index>; the wiki_id is what
# groups paragraphs into an article, so it's what the query holdout must be keyed on.
ID_COLUMN = "_id"
BYTES_PER_VEC = DIM * 4

# Concurrency. WINDOW must be >= DOWNLOAD_WORKERS: downloads are submitted in shard order into a FIFO
# pool, so the shard the extractor is waiting on always starts before any later shard, and with at
# least one window slot per worker no later shard can hold the slot the extractor needs.
DOWNLOAD_WORKERS = 6
WINDOW = 12


def _curl_to_file(url, target_path, retries=4):
  """Download url -> target_path via curl. Returns on success, raises after `retries` attempts."""
  for attempt in range(1, retries + 1):
    # -f: fail on HTTP >=400 rather than writing an error body; -L: follow redirects; --retry: curl's
    # own transient-error retries. NOTE: no -C - (resume) here; each attempt starts the file over, so
    # a truncated attempt can't be appended onto and corrupt the parquet.
    rc = subprocess.call(["curl", "-fsSL", "--retry", "5", "--retry-delay", "3", "-o", target_path, url])
    if rc == 0:
      return target_path
    print(f"  curl failed (rc={rc}) for {url} (attempt {attempt}/{retries})", flush=True)
    if os.path.exists(target_path):
      os.remove(target_path)
    time.sleep(2 * attempt)
  raise RuntimeError(f"curl failed {retries}x for {url}")


def list_shard_urls():
  """Return the ordered list of HF auto-parquet shard URLs for the 'en' config."""
  api = f"https://huggingface.co/api/datasets/{HF_REPO}/parquet/{LANG}/train"
  out = subprocess.run(["curl", "-fsSL", api], capture_output=True, text=True, check=True)
  return json.loads(out.stdout)


def check_parquet_decode(shard_path):
  """Fail loudly if this pyarrow mis-decodes the shards' RLE_DICTIONARY-encoded 'emb' column.

  pyarrow 25.0.0 silently substitutes earlier dictionary values when decoding these shards: vectors
  come out with ~0.5 norms instead of unit norm, and the SAME row decodes differently at different
  batch sizes. Left unchecked that yields a corrupt 170GB corpus that looks fine until recall numbers
  come out subtly wrong, so verify before writing anything.

  Two independent checks on one real shard:
    1. every vector is unit-norm (these embeddings are L2-normalized -- see cohere-v3-README.txt, and
       the bundled 1M sample measures at exactly 1.0),
    2. the batched read path this script uses agrees byte-for-byte with a whole-column read.
  """
  import pyarrow as pa

  gt = np.array(pq.read_table(shard_path, columns=[EMB_COLUMN]).column(0).combine_chunks().to_pylist()[:600], dtype=np.float32)
  norms = np.linalg.norm(gt, axis=1)
  if not np.all(np.abs(norms - 1.0) < 1e-3):
    raise RuntimeError(
      f"pyarrow {pa.__version__} decodes non-unit-norm vectors from {shard_path} "
      f"(norm mean {norms.mean():.4f}, expected 1.0) -- known RLE_DICTIONARY decode bug in 25.0.0. "
      f"Install a known-good version:  pip install 'pyarrow==21.0.0'"
    )

  pf = pq.ParquetFile(shard_path)
  batched = []
  got = 0
  for batch in pf.iter_batches(batch_size=10_000, columns=[EMB_COLUMN]):
    col = batch.column(0)
    batched.append(col.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(col), DIM))
    got += len(col)
    if got >= 600:
      break
  if not np.array_equal(np.vstack(batched)[:600], gt):
    raise RuntimeError(
      f"pyarrow {pa.__version__}: batched read disagrees with whole-column read on {shard_path} "
      f"-- corrupt parquet decode. Install a known-good version:  pip install 'pyarrow==21.0.0'"
    )
  print(f"parquet decode check passed (pyarrow {pa.__version__}): unit-norm vectors, batched read matches")


def _wiki_id(row_id):
  """'20231101.en_13194570_4' -> '13194570' (the article key). Empty string if unparseable."""
  parts = row_id.rsplit("_", 2)
  return parts[1] if len(parts) == 3 else ""


def _is_query_article(wiki_id, query_frac_num, query_frac_den, seed):
  """Deterministically select whole articles for the query set.

  Hashing the wiki_id (rather than tracking a chosen-id set, or numbering articles as we go) means the
  decision needs no cross-shard state: an article straddling a shard boundary is classified the same
  way in both shards, and a resumed run reproduces the split exactly. blake2b keyed by `seed` gives a
  uniform spread over wiki_ids; sha/blake are stable across Python versions and platforms, unlike
  hash().
  """
  h = hashlib.blake2b(f"{seed}:{wiki_id}".encode(), digest_size=8).digest()
  return int.from_bytes(h, "big") % query_frac_den < query_frac_num


def _extract_shard(path, docs_f, queries_f, query_frac_num, query_frac_den, seed):
  """Append one shard's vectors, routing WHOLE ARTICLES to either queries or docs.

  Reads in row-group batches so memory stays bounded regardless of shard size. Returns
  (docs_written, queries_written) for this shard.
  """
  pf = pq.ParquetFile(path)
  n_docs = n_q = 0
  for batch in pf.iter_batches(batch_size=10_000, columns=[EMB_COLUMN, ID_COLUMN]):
    col = batch.column(0)
    # list<float> -> contiguous (n, DIM) float32. Flattening the child values and reshaping avoids a
    # Python-level per-row loop.
    flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    n = len(col)
    if n * DIM != flat.size:
      raise RuntimeError(f"unexpected emb width in {path}: {flat.size} for {n} rows")
    rows = flat.reshape(n, DIM)

    # Classify per row via its article. Paragraphs of one article are consecutive, so cache the last
    # decision instead of re-hashing ~7x per article.
    mask = np.zeros(n, dtype=bool)
    last_wid = None
    last_is_q = False
    for i, row_id in enumerate(batch.column(1).to_pylist()):
      wid = _wiki_id(row_id)
      if wid != last_wid:
        last_wid = wid
        last_is_q = _is_query_article(wid, query_frac_num, query_frac_den, seed)
      mask[i] = last_is_q

    q_rows = rows[mask]
    d_rows = rows[~mask]
    if q_rows.size:
      queries_f.write(np.ascontiguousarray(q_rows).tobytes())
      n_q += len(q_rows)
    if d_rows.size:
      docs_f.write(np.ascontiguousarray(d_rows).tobytes())
      n_docs += len(d_rows)
  return n_docs, n_q


def download_en(data_dir, query_frac_num, query_frac_den, seed, max_shards=None):
  os.makedirs(data_dir, exist_ok=True)
  docs_path = os.path.join(data_dir, f"cohere-v3-wikipedia-en-1024d.docs.{LANG}-full.vec")
  # Query count isn't known up front (it depends on how many paragraphs the selected articles have),
  # so name the file by the holdout rate rather than a count.
  queries_path = os.path.join(data_dir, f"cohere-v3-wikipedia-en-1024d.queries.{query_frac_num}in{query_frac_den}-articles.vec")
  progress_path = docs_path + ".progress"

  shard_urls = list_shard_urls()
  if max_shards is not None:
    shard_urls = shard_urls[:max_shards]
  print(f"{len(shard_urls)} shards for config '{LANG}'")

  # Resume: restart at the first unfinished shard and truncate BOTH outputs back to the recorded
  # counts, dropping anything a shard interrupted mid-write had appended.
  next_shard = 0
  docs_written = queries_written = 0
  if os.path.exists(progress_path):
    with open(progress_path) as f:
      state = json.load(f)
    # Refuse to resume under different query-holdout settings: the docs/queries split would change
    # mid-file, so the two outputs would no longer be the disjoint split they claim to be.
    prior = (state.get("query_frac_num"), state.get("query_frac_den"), state.get("seed"))
    if prior != (query_frac_num, query_frac_den, seed):
      raise RuntimeError(
        f"{progress_path} was written with query holdout {prior[0]}/{prior[1]} seed={prior[2]}, but this "
        f"run asks for {query_frac_num}/{query_frac_den} seed={seed}. Use the original values, or delete "
        f"the .vec files and .progress to start over."
      )
    next_shard = state["next_shard"]
    docs_written = state["docs_written"]
    queries_written = state["queries_written"]
    for path, count in ((docs_path, docs_written), (queries_path, queries_written)):
      clean_len = count * BYTES_PER_VEC
      if os.path.exists(path) and os.path.getsize(path) != clean_len:
        with open(path, "r+b") as f:
          f.truncate(clean_len)
    print(f"resuming at shard {next_shard}: {docs_written:,} docs, {queries_written:,} queries already written")

  if next_shard >= len(shard_urls):
    print(f"already complete: {docs_written:,} docs, {queries_written:,} queries")
    return docs_path, queries_path

  staging = os.path.join(data_dir, ".en_shard_staging")
  os.makedirs(staging, exist_ok=True)
  # Bound how many downloaded-but-not-yet-extracted shards sit on disk (~200MB each).
  window = threading.Semaphore(WINDOW)

  def fetch(shard_idx):
    window.acquire()
    try:
      target = os.path.join(staging, f"{shard_idx}.parquet")
      # Re-download rather than trusting a leftover from a killed run: a partial parquet would fail
      # to read anyway, and re-fetching one shard is ~4s.
      if os.path.exists(target):
        os.remove(target)
      return _curl_to_file(shard_urls[shard_idx], target)
    except BaseException:
      window.release()  # never leak a slot on failure -- the extractor would deadlock
      raise

  t0 = time.time()
  start_docs = docs_written
  docs_f = open(docs_path, "ab")
  queries_f = open(queries_path, "ab")
  pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS, thread_name_prefix="dl")
  try:
    # Submit in shard order; the FIFO queue keeps worker pickup in order too, so the shard the
    # extractor blocks on is always already in flight.
    futures = {i: pool.submit(fetch, i) for i in range(next_shard, len(shard_urls))}
    checked_decode = False
    for shard_idx in range(next_shard, len(shard_urls)):
      path = futures.pop(shard_idx).result()
      try:
        # Verify the parquet decoder on the first real shard, BEFORE writing any vectors.
        if not checked_decode:
          check_parquet_decode(path)
          checked_decode = True
        d, q = _extract_shard(path, docs_f, queries_f, query_frac_num, query_frac_den, seed)
      finally:
        os.remove(path)
        window.release()
      docs_written += d
      queries_written += q
      docs_f.flush()
      queries_f.flush()
      with open(progress_path, "w") as f:
        json.dump(
          {
            "next_shard": shard_idx + 1,
            "docs_written": docs_written,
            "queries_written": queries_written,
            "query_frac_num": query_frac_num,
            "query_frac_den": query_frac_den,
            "seed": seed,
          },
          f,
        )
      gb = docs_written * BYTES_PER_VEC / 1e9
      elapsed = max(1e-9, time.time() - t0)
      rate = (docs_written - start_docs) / elapsed
      eta = (len(shard_urls) - shard_idx - 1) / max(1e-9, (shard_idx + 1 - next_shard) / elapsed)
      print(
        f"  shard {shard_idx + 1}/{len(shard_urls)}: {docs_written:,} docs ({gb:.1f} GB), "
        f"{queries_written:,} queries, {rate:,.0f} docs/s, ETA {eta / 60:.0f} min",
        flush=True,
      )
  finally:
    pool.shutdown(wait=False, cancel_futures=True)
    docs_f.close()
    queries_f.close()

  print(f"\nDONE: {docs_written:,} docs ({docs_written * BYTES_PER_VEC / 1e9:.1f} GB) -> {docs_path}")
  print(f"      {queries_written:,} queries ({queries_written * BYTES_PER_VEC / 1e9:.1f} GB) -> {queries_path}")
  print("\nNOTE: docs are in corpus order. For the 'scattered' distribution the IVF/LSH benchmarks want:")
  print(f"  python src/python/shuffle_vecs.py {docs_path} <out.vec> --dim {DIM}")
  return docs_path, queries_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Download full Cohere-v3 English Wikipedia embeddings as raw float32 .vec")
  parser.add_argument("--data-dir", required=True, help="Output directory for the .vec files")
  # 250K of the corpus's 5,854,887 articles is the bundled corpus's holdout (~4.3%), which yields
  # ~1.78M query vectors. 1-in-24 articles reproduces that rate closely.
  parser.add_argument("--query-frac-num", type=int, default=1, help="Numerator of the article holdout fraction (default: 1)")
  parser.add_argument("--query-frac-den", type=int, default=24, help="Denominator of the article holdout fraction (default: 24 -> ~4.2%% of articles)")
  parser.add_argument("--seed", type=int, default=42, help="Hash seed for article selection (default: 42)")
  parser.add_argument("--max-shards", type=int, default=None, help="Only process this many shards (for testing)")
  args = parser.parse_args()
  if not 0 < args.query_frac_num < args.query_frac_den:
    sys.exit(f"need 0 < --query-frac-num ({args.query_frac_num}) < --query-frac-den ({args.query_frac_den})")
  download_en(args.data_dir, args.query_frac_num, args.query_frac_den, args.seed, args.max_shards)
