#!/usr/bin/env python

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

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from urllib import request

PYTHON_MAJOR_VER = sys.version_info.major

# --- Large-scale SOTA vector source: Cohere v3 multilingual Wikipedia embeddings -------------------
# CohereLabs/wikipedia-2023-11-embed-multilingual-v3 (the SOTA superset the bundled
# cohere-v3-wikipedia-en-scattered-1M sample is drawn from): ~250M passages x 1024d, unit-norm, the
# same embedding model/distribution as the existing data so a frozen PCA/ITQ basis stays comparable.
# It ships as hundreds of per-language parquet shards (HF's auto-converted parquet), NOT a .vec, so
# build_large_vecs() streams them shard-by-shard into a raw little-endian float32 .vec (the format
# knnPerfTest/KnnGraphTester read). Reachability notes for this (restricted) network, all verified:
#   - HF + the parquet API + the r2.dev buckets are reachable via `curl` (system CA store), but Python's
#     urllib fails TLS (no CA certs in the venv) -- so downloads shell out to curl, not urllib.
#   - parquet is read with pyarrow (prebuilt wheel: `pip install --only-binary=:all: pyarrow`).
# The HF auto-parquet shard list per language config:
#   https://huggingface.co/api/datasets/<repo>/parquet/<lang>/train  -> JSON array of shard URLs
LARGE_VEC_HF_REPO = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"
LARGE_VEC_DIM = 1024
LARGE_VEC_EMB_COLUMN = "emb"
# Language configs to consume IN ORDER until the target doc count is reached. en (~41M) leads (matches
# the bundled English sample); the rest are large Wikipedias appended only if a bigger target needs them.
LARGE_VEC_LANG_ORDER = ("en", "de", "fr", "ru", "es", "it", "ja", "zh", "pt", "nl")
# Bytes per vector on disk (raw float32, no header) -- used for size/target math.
_LARGE_VEC_BYTES_PER_VEC = LARGE_VEC_DIM * 4

BASE_URL = "https://home.apache.org/~mikemccand"
BASE_URL2 = "https://home.apache.org/~sokolov"

DATA_FILES = [
  # remote url, local name
  "https://pub-6de3254d7180436684278e0ec33ada22.r2.dev/enwiki-20120502-lines-1k-fixed-utf8-with-random-label.txt.lzma",
  # https://pub-6de3254d7180436684278e0ec33ada22.r2.dev/enwiki-20120502-lines-with-random-label.txt.lzma, <-- wiki big
  "https://pub-a0911d22bca84510bc906f76af183a65.r2.dev/cohere-v3-wikipedia-en-scattered-1024d.docs.first1M.vec",
  "https://pub-a0911d22bca84510bc906f76af183a65.r2.dev/cohere-v3-wikipedia-en-scattered-1024d.docs.first1M.csv",
  "https://pub-a0911d22bca84510bc906f76af183a65.r2.dev/cohere-v3-wikipedia-en-scattered-1024d.queries.first200K.vec",
  "https://pub-a0911d22bca84510bc906f76af183a65.r2.dev/cohere-v3-wikipedia-en-scattered-1024d.queries.first200K.csv",
  "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
]

USAGE = """
Usage: python initial_setup.py [-download]

Options:
  -download downloads a 5GB linedoc file and untold GB of vector files

"""
DEFAULT_LOCAL_CONST = """
BASE_DIR = '%(base_dir)s'
BENCH_BASE_DIR = '%(base_dir)s/%(cwd)s'
"""


def _curl_to_file(url, target_path, retries=3):
  """Download url -> target_path via curl (system CA store; venv urllib lacks CA certs on this network).

  Uses curl's own resume (-C -) and fail-on-error (-f). Returns on success, raises after `retries`.
  """
  for attempt in range(1, retries + 1):
    # -f: fail (non-zero) on HTTP >=400 instead of writing an error body; -L: follow redirects;
    # -C -: resume a partial file; --retry: curl-level transient-error retries.
    rc = subprocess.call(
      ["curl", "-fL", "-C", "-", "--retry", "5", "--retry-delay", "3", "-o", target_path, url]
    )
    if rc == 0:
      return
    print(f"  curl failed (rc={rc}) for {url} (attempt {attempt}/{retries})")
    time.sleep(2 * attempt)
  raise RuntimeError(f"curl failed {retries}x for {url}")


def _curl_to_string(url):
  """GET url and return the response body as text (follows redirects), via curl."""
  out = subprocess.run(["curl", "-fsSL", url], capture_output=True, text=True, check=True)
  return out.stdout


def _list_shard_urls(repo, lang):
  """Return the ordered list of HF auto-parquet shard URLs for one language config of `repo`."""
  api = f"https://huggingface.co/api/datasets/{repo}/parquet/{lang}/train"
  return json.loads(_curl_to_string(api))


def build_large_vecs(data_dir, target_docs, langs=LARGE_VEC_LANG_ORDER, repo=LARGE_VEC_HF_REPO):
  """Stream Cohere-v3 multilingual parquet shards into a single raw float32 .vec of up to target_docs.

  Memory- and disk-frugal and RESUMABLE:
    - one shard on disk at a time (download -> extract -> delete the parquet),
    - parquet read in row-group batches (constant RAM regardless of shard size),
    - appends raw little-endian float32 (dim*4 bytes/vec, no header) -- the .vec format the harness reads,
    - a sidecar .progress json records (#docs written, last completed (lang, shard-index)) so a re-run
      continues instead of restarting; the .vec is truncated back to the recorded length first so a
      shard interrupted mid-write can't leave a partial vector.

  Returns the output .vec path. Companion query file is the bundled 200K queries (same distribution).
  """
  import numpy as np
  import pyarrow.parquet as pq

  out_path = os.path.join(data_dir, f"cohere-v3-multilingual-1024d.docs.{target_docs}.vec")
  progress_path = out_path + ".progress"

  # Resume: read how many docs we already wrote and which shards are done, then truncate the .vec to
  # exactly that many docs (drops any partial trailing vector from an interrupted run).
  done_shards = set()
  docs_written = 0
  if os.path.exists(progress_path):
    with open(progress_path) as f:
      state = json.load(f)
    docs_written = state.get("docs_written", 0)
    done_shards = {tuple(s) for s in state.get("done_shards", [])}
    clean_len = docs_written * _LARGE_VEC_BYTES_PER_VEC
    if os.path.exists(out_path) and os.path.getsize(out_path) != clean_len:
      with open(out_path, "r+b") as f:
        f.truncate(clean_len)
    print(f"resuming: {docs_written:,} docs already written, {len(done_shards)} shards done")

  if docs_written >= target_docs:
    print(f"already have {docs_written:,} >= target {target_docs:,} docs at {out_path}")
    return out_path

  tmp_parquet = os.path.join(data_dir, ".large_vec_shard.parquet.tmp")
  t0 = time.time()
  out_f = open(out_path, "ab")
  try:
    for lang in langs:
      if docs_written >= target_docs:
        break
      print(f"=== language config '{lang}' ===")
      try:
        shard_urls = _list_shard_urls(repo, lang)
      except Exception as e:
        print(f"  could not list shards for '{lang}': {e} -- skipping")
        continue
      print(f"  {len(shard_urls)} shards")
      for shard_idx, url in enumerate(shard_urls):
        if docs_written >= target_docs:
          break
        if (lang, shard_idx) in done_shards:
          continue
        # Remove any leftover temp from the previous shard FIRST: _curl_to_file resumes with `-C -`,
        # which would otherwise append this shard's bytes onto the prior shard's file and corrupt it.
        if os.path.exists(tmp_parquet):
          os.remove(tmp_parquet)
        _curl_to_file(url, tmp_parquet)
        pf = pq.ParquetFile(tmp_parquet)
        # Read just the embedding column, in row-group batches, to bound memory.
        for batch in pf.iter_batches(batch_size=10_000, columns=[LARGE_VEC_EMB_COLUMN]):
          # list<float> column -> contiguous (n, dim) float32. to_numpy(zero_copy_only=False) on the
          # flattened child values then reshape avoids a Python-level per-row loop.
          col = batch.column(0)
          flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
          n = len(col)
          if n * LARGE_VEC_DIM != flat.size:
            raise RuntimeError(f"unexpected emb width in {lang}/{shard_idx}: {flat.size} for {n} rows")
          take = min(n, target_docs - docs_written)
          out_f.write(np.ascontiguousarray(flat[: take * LARGE_VEC_DIM]).tobytes())
          docs_written += take
          if docs_written >= target_docs:
            break
        out_f.flush()
        done_shards.add((lang, shard_idx))
        with open(progress_path, "w") as pf_out:
          json.dump({"docs_written": docs_written, "done_shards": sorted(done_shards)}, pf_out)
        gb = docs_written * _LARGE_VEC_BYTES_PER_VEC / 1e9
        rate = docs_written / max(1e-9, time.time() - t0)
        print(f"  {lang}/{shard_idx}: {docs_written:,} docs ({gb:.1f} GB) {rate:,.0f} docs/s")
  finally:
    out_f.close()
    if os.path.exists(tmp_parquet):
      os.remove(tmp_parquet)

  print(f"DONE: wrote {docs_written:,} docs ({docs_written * _LARGE_VEC_BYTES_PER_VEC / 1e9:.1f} GB) to {out_path}")
  return out_path


def runSetup(download):
  cwd = os.getcwd()
  parent, base = os.path.split(cwd)
  data_dir = os.path.join(parent, "data")
  idx_dir = os.path.join(parent, "indices")

  if not os.path.exists(data_dir):
    print("create data directory at %s" % (data_dir))
    os.mkdir(data_dir)
  else:
    print("data directory already exists %s" % (data_dir))

  if not os.path.exists(idx_dir):
    os.mkdir(idx_dir)
    print("create indices directory at %s" % (idx_dir))
  else:
    print("indices directory already exists %s" % (idx_dir))

  pySrcDir = os.path.join(cwd, "src", "python")
  local_const = os.path.join(pySrcDir, "localconstants.py")
  if not os.path.exists(local_const):
    f = open(local_const, "w")
    try:
      f.write(DEFAULT_LOCAL_CONST % ({"base_dir": parent, "cwd": base}))
    finally:
      f.close()
  else:
    print("localconstants.py already exists - skipping")

  local_run = os.path.join(pySrcDir, "localrun.py")
  example = os.path.join(pySrcDir, "example.py")
  if not os.path.exists(local_run):
    shutil.copyfile(example, local_run)
  else:
    print("localrun.py already exists - skipping")

  if download:
    for tup in DATA_FILES:
      if type(tup) is str:
        url_source = tup
        local_filename = os.path.basename(url_source)
      elif type(tup) is tuple and len(tup) == 2:
        url_source, local_filename = tup
      else:
        raise RuntimeError(f"DATA_FILES elements should be single string or length 2 tuple; got: {tup}")
      target_file = os.path.join(data_dir, local_filename)
      if os.path.exists(target_file):
        print("file %s already exists - skipping" % target_file)
      else:
        print("download %s to %s - might take a long time!" % (url_source, target_file))
        Downloader(url_source, target_file).download()
        print()
        print("downloading %s to %s done " % (url_source, target_file))

      for suffix in (".bz2", ".lzma", ".zip", ".xz"):
        if target_file.endswith(suffix):
          print("NOTE: make sure you decompress %s" % target_file)
          break

  print("setup successful")


class Downloader:
  HISTORY_SIZE = 100

  def __init__(self, url, target_path):
    self.__url = url
    self.__target_path = target_path
    Downloader.times = [time.time()] * Downloader.HISTORY_SIZE
    Downloader.sizes = [0] * Downloader.HISTORY_SIZE
    Downloader.index = 0

  def download(self):
    opener = request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")]
    request.install_opener(opener)
    request.urlretrieve(self.__url, self.__target_path, Downloader.reporthook)

  @staticmethod
  def reporthook(count, block_size, total_size):
    current_time = time.time()
    current_size = int(count * block_size)
    last_time = Downloader.times[Downloader.index]
    last_size = Downloader.sizes[Downloader.index]
    delta_size = current_size - last_size
    delta_time = current_time - last_time
    Downloader.times[Downloader.index] = current_time
    Downloader.sizes[Downloader.index] = current_size
    Downloader.index = (Downloader.index + 1) % Downloader.HISTORY_SIZE

    speed = float(delta_size) / (1024 * delta_time)
    percent = int(current_size * 100 / total_size)
    sys.stdout.write("\r ")
    #    sys.stdout.write('(%d, %d), (%d, %d), (%d, %d) ' % (current_size, current_time, last_size, last_time, delta_size, delta_time))
    sys.stdout.write("downloading ... %d%%, %.2f MB/%.2fMB, speed %.2f KB/s" % (percent, float(current_size) / (1024 * 1024), float(total_size) / (1024 * 1024), speed))
    sys.stdout.flush()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog="luceneutil setup", description="Benchmarking setup for lucene")
  parser.add_argument(
    "-d",
    "-download",
    "--download",
    action="store_true",
    help="Download datasets to run benchmarks. A 6 GB compressed Wikipedia line doc file, and a 13 GB vectors file is downloaded from Apache mirrors",
  )
  parser.add_argument(
    "--build-large-vecs",
    type=int,
    metavar="TARGET_DOCS",
    default=0,
    help=(
      "Stream Cohere-v3 multilingual Wikipedia parquet shards into a single raw float32 .vec of up to "
      "TARGET_DOCS vectors (1024d, ~4KB/vec: e.g. 100_000_000 ~= 400GB). Resumable. Writes to ../data/. "
      "NOTE: leave disk room for the LSH index (~1.6x the .vec) + spilling."
    ),
  )
  parser.add_argument(
    "--data-dir",
    default=None,
    help="Override the output data directory for --build-large-vecs (default: ../data relative to cwd).",
  )
  args = parser.parse_args()
  if args.build_large_vecs > 0:
    cwd = os.getcwd()
    parent = os.path.split(cwd)[0]
    data_dir = args.data_dir or os.path.join(parent, "data")
    os.makedirs(data_dir, exist_ok=True)
    build_large_vecs(data_dir, args.build_large_vecs)
  else:
    runSetup(args.download)
