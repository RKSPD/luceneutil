#!/usr/bin/env python3

"""Converts the dense vectors of Qdrant's Supernova benchmark datasets into luceneutil .vec files.

The datasets are only published as parquet on Hugging Face.  This script reads just the dense
embedding column of each shard using HTTP range requests, so the text, sparse and multi-vector
columns (most of the bytes) are never downloaded.

See qdrant-supernova-README.txt.
"""

import argparse
import http.client
import io
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

HF_HOST = "huggingface.co"


@dataclass(frozen=True)
class Dataset:
  repo: str
  column: str
  dim: int
  # parquet file holding query embeddings in the same column, or None to hold out docs as queries
  query_file: str | None


DATASETS = {
  # 10.07B docs, 768d float16 gte-multilingual-base; query vectors are not published (MS MARCO license)
  "fineweb": Dataset("Qdrant/FineWeb-10B", "dense_embedding", 768, None),
  # 23.9M docs, 1024d float32 BGE-M3; 10K bioRxiv title queries
  "pubmed": Dataset("Qdrant/PubMed-MV", "bge_m3_dense_embedding", 1024, "queries/bf_queries_dense_cosine_k1000.parquet"),
  # 15.4M docs, 2048d float32 Qwen3-VL-Embedding-2B; 25K image+text queries
  "coyo": Dataset("Qdrant/Coyo-VE", "qwen_multimodal_embedding", 2048, "queries/queries_gt_k1000.parquet"),
}

# each request is tried this many times; a stalled read fails after the timeout
DOWNLOAD_ATTEMPTS = 6
DOWNLOAD_TIMEOUT_SEC = 60
# first retry waits this long, doubling each time, unless the server sends Retry-After
RETRY_BASE_SEC = 5
MAX_REDIRECTS = 5

# range responses are copied in pieces of this size
COPY_BUFFER_BYTES = 4 * 1024 * 1024

# vectors whose length differs from 1.0 by more than this are counted as not unit-length (float16 sources lose some precision)
NORM_TOLERANCE = 1e-2

# http status codes worth retrying
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
# a signed cdn url that has expired returns one of these; resolve it again
EXPIRED_STATUS = {401, 403, 410}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
  def redirect_request(self, req, fp, code, msg, headers, newurl):
    return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)


def hf_headers(url):
  """Returns the auth header for huggingface.co urls when HF_TOKEN is set; never sends the token elsewhere."""
  token = os.environ.get("HF_TOKEN")
  if token and urllib.parse.urlsplit(url).hostname == HF_HOST:
    return {"Authorization": f"Bearer {token}"}
  return {}


def retry_delay_sec(attempt, error):
  """Seconds to wait before the next attempt, honoring a numeric Retry-After header."""
  if isinstance(error, urllib.error.HTTPError):
    retry_after = error.headers.get("Retry-After")
    if retry_after is not None and retry_after.isdigit():
      return int(retry_after)
  return RETRY_BASE_SEC * 2 ** (attempt - 1)


def with_retries(description, fn):
  """Calls fn() until it succeeds, retrying network errors and retryable http statuses."""
  for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
    try:
      return fn()
    except urllib.error.HTTPError as e:
      if e.code not in RETRY_STATUS or attempt == DOWNLOAD_ATTEMPTS:
        raise
      error = e
    except (OSError, http.client.HTTPException) as e:
      if attempt == DOWNLOAD_ATTEMPTS:
        raise
      error = e
    delay_sec = retry_delay_sec(attempt, error)
    print(f"  {description} failed ({error}), retrying in {delay_sec} sec")
    time.sleep(delay_sec)
  raise AssertionError("unreachable")


def open_url(url, headers, method="GET"):
  """Opens url without following redirects, so the token is only ever sent to huggingface.co."""
  request = urllib.request.Request(url, headers={**headers, **hf_headers(url)}, method=method)
  return _NO_REDIRECT_OPENER.open(request, timeout=DOWNLOAD_TIMEOUT_SEC)


def resolve(url):
  """Follows redirects from a huggingface.co resolve url to the (signed) download url and returns (url, size)."""
  for _ in range(MAX_REDIRECTS + 1):
    try:
      response = with_retries(f"resolving {url}", lambda u=url: open_url(u, {}, method="HEAD"))
    except urllib.error.HTTPError as e:
      if e.code not in (301, 302, 303, 307, 308):
        raise
      url = urllib.parse.urljoin(url, e.headers["Location"])
      continue
    with response:
      size = response.headers.get("Content-Length")
    if size is None:
      raise RuntimeError(f"{url}: no Content-Length")
    return url, int(size)
  raise RuntimeError(f"too many redirects resolving {url}")


def list_files(repo, path):
  """Returns the sorted (path, type) entries directly under path in the repo, following pagination."""
  url = f"https://{HF_HOST}/api/datasets/{repo}/tree/main/{urllib.parse.quote(path)}"
  entries = []
  while url is not None:

    def fetch(u=url):
      with open_url(u, {}) as response:
        return json.load(response), response.headers.get("Link")

    page, link = with_retries(f"listing {path}", fetch)
    entries.extend((e["path"], e["type"]) for e in page)
    match = re.search(r'<([^>]+)>;\s*rel="next"', link or "")
    url = match.group(1) if match else None
  return sorted(entries)


def iter_data_files(repo):
  """Yields the parquet files under data/ in sorted order, listing one sub directory at a time."""
  for path, kind in list_files(repo, "data"):
    if kind == "directory":
      for sub_path, sub_kind in list_files(repo, path):
        if sub_kind == "file" and sub_path.endswith(".parquet"):
          yield sub_path
    elif path.endswith(".parquet"):
      yield path


class HttpRangeFile(io.RawIOBase):
  """Read-only seekable file over http range requests, so pyarrow only fetches the bytes it needs."""

  def __init__(self, resolve_url):
    super().__init__()
    self._resolve_url = resolve_url
    self._url, self._size = resolve(resolve_url)
    self._pos = 0
    self.request_count = 0
    self.bytes_read = 0

  def readable(self):
    return True

  def seekable(self):
    return True

  def tell(self):
    return self._pos

  def size(self):
    return self._size

  def seek(self, offset, whence=io.SEEK_SET):
    if whence == io.SEEK_SET:
      self._pos = offset
    elif whence == io.SEEK_CUR:
      self._pos += offset
    elif whence == io.SEEK_END:
      self._pos = self._size + offset
    else:
      raise ValueError(f"bad whence {whence}")
    return self._pos

  def _fetch(self, start, length):
    out = bytearray()

    def read_range():
      out.clear()
      with open_url(self._url, {"Range": f"bytes={start}-{start + length - 1}"}) as response:
        if response.status != 206:
          raise OSError(f"expected 206 partial content, got {response.status}")
        while len(out) < length:
          chunk = response.read(min(COPY_BUFFER_BYTES, length - len(out)))
          if not chunk:
            break
          out.extend(chunk)
      if len(out) != length:
        raise OSError(f"got {len(out)} bytes, expected {length}")

    try:
      with_retries(f"reading {self._resolve_url} bytes {start}-{start + length - 1}", read_range)
    except urllib.error.HTTPError as e:
      if e.code not in EXPIRED_STATUS:
        raise
      # the signed url expired: resolve again and retry once
      self._url, self._size = resolve(self._resolve_url)
      with_retries(f"reading {self._resolve_url} bytes {start}-{start + length - 1}", read_range)
    self.request_count += 1
    self.bytes_read += length
    return bytes(out)

  def read(self, size=-1):
    if size is None or size < 0:
      size = self._size - self._pos
    size = max(0, min(size, self._size - self._pos))
    if size == 0:
      return b""
    data = self._fetch(self._pos, size)
    self._pos += size
    return data

  def readinto(self, buffer):
    data = self.read(len(buffer))
    buffer[: len(data)] = data
    return len(data)


def hf_url(repo, path):
  return f"https://{HF_HOST}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"


def column_to_matrix(table, column, dim):
  """Returns a parquet list column as a (rows, dim) float32 matrix."""
  lists = table.column(column).combine_chunks()
  # check every row, not just the total, so one short row next to one long row can't shift the vectors after it
  lengths = pc.min_max(pc.list_value_length(lists)).as_py()
  if lists.null_count > 0 or lengths["min"] != dim or lengths["max"] != dim:
    raise RuntimeError(f"{column}: expected every row to have {dim} values, got {lists.null_count} null rows and lengths {lengths['min']}..{lengths['max']}")
  values = lists.flatten().to_numpy()
  if values.size != table.num_rows * dim:
    raise RuntimeError(f"{column}: expected {table.num_rows * dim} values, got {values.size}")
  # float16 -> float32 is exact; anything else would need a precision check
  if values.dtype not in (np.float16, np.float32):
    raise RuntimeError(f"{column}: unsupported element type {values.dtype}")
  return values.reshape(table.num_rows, dim).astype(np.float32, copy=False)


def check_vectors(matrix, description):
  """Fails on NaN/infinite vectors and returns how many are not unit-length."""
  norms = np.linalg.norm(matrix, axis=1)
  if not np.isfinite(norms).all():
    raise RuntimeError(f"{description}: {int((~np.isfinite(norms)).sum())} vectors are NaN or infinite")
  return int((np.abs(norms - 1.0) > NORM_TOLERANCE).sum())


def open_parquet(dataset, path):
  """Opens a remote parquet file, reading only its footer, and checks it has the embedding column."""
  source = HttpRangeFile(hf_url(dataset.repo, path))
  parquet = pq.ParquetFile(source)
  if dataset.column not in parquet.schema_arrow.names:
    raise RuntimeError(f"{path}: no column {dataset.column}; columns are {parquet.schema_arrow.names}")
  return source, parquet


def write_row_group(dataset, parquet, group, out_path, description, max_rows=None):
  """Writes the embedding column of one row group (optionally only its first max_rows rows) to a .vec file. Returns the row count."""
  table = parquet.read_row_group(group, columns=[dataset.column])
  if max_rows is not None:
    table = table.slice(0, max_rows)
  matrix = column_to_matrix(table, dataset.column, dataset.dim)
  not_unit_count = check_vectors(matrix, description)
  if not_unit_count > 0:
    print(f"  WARNING: {not_unit_count} vectors in {description} are not unit-length")
  tmp_path = out_path + ".part"
  matrix.tofile(tmp_path)
  expected_bytes = table.num_rows * dataset.dim * 4
  if os.path.getsize(tmp_path) != expected_bytes:
    raise RuntimeError(f"{tmp_path}: expected {expected_bytes} bytes, got {os.path.getsize(tmp_path)}")
  Path(tmp_path).replace(out_path)
  return table.num_rows


def report(path, source, row_count, t0):
  elapsed_time_sec = time.time() - t0
  print(f"  {path}: {row_count} vectors, fetched {source.bytes_read / 1024 / 1024:.1f} of {source.size() / 1024 / 1024:.1f} MB in {source.request_count} requests, {elapsed_time_sec:.1f} sec")


def convert_docs(dataset, parts_dir, needed_rows):
  """Converts row groups of the corpus shards, in order, until needed_rows vectors are on disk. Returns the part paths.

  each row group becomes its own part file, so an interrupted run resumes where it stopped, and a later run
  asking for more docs reuses the parts already converted.
  """
  part_paths = []
  have_rows = 0
  for path in iter_data_files(dataset.repo):
    if have_rows >= needed_rows:
      break
    t0 = time.time()
    source = None
    parquet = None
    shard_rows = 0
    group = 0
    while have_rows < needed_rows:
      part_path = str(parts_dir / f"{path.replace('/', '_').removesuffix('.parquet')}.rg{group:05d}.vec")
      if not os.path.exists(part_path):
        # only fetch the footer when a row group is missing, so fully converted shards cost no requests
        if parquet is None:
          source, parquet = open_parquet(dataset, path)
        if group >= parquet.metadata.num_row_groups:
          break
        write_row_group(dataset, parquet, group, part_path, f"{path} row group {group}")
      rows = vec_rows(part_path, dataset.dim)
      part_paths.append(part_path)
      have_rows += rows
      shard_rows += rows
      group += 1
    if source is not None:
      report(path, source, shard_rows, t0)
    else:
      print(f"  {path}: {shard_rows} vectors already converted")
  if have_rows < needed_rows:
    raise RuntimeError(f"{dataset.repo} only has {have_rows} vectors, need {needed_rows}")
  return part_paths


def convert_queries(dataset, out_path, max_rows):
  """Writes the published query embeddings (optionally only the first max_rows) to a .vec file. Returns the row count."""
  t0 = time.time()
  source, parquet = open_parquet(dataset, dataset.query_file)
  group_paths = []
  row_count = 0
  for group in range(parquet.metadata.num_row_groups):
    if max_rows is not None and row_count >= max_rows:
      break
    group_path = f"{out_path}.rg{group:05d}"
    remaining_rows = None if max_rows is None else max_rows - row_count
    row_count += write_row_group(dataset, parquet, group, group_path, f"{dataset.query_file} row group {group}", remaining_rows)
    group_paths.append(group_path)
  concat_vectors(group_paths, dataset.dim, 0, row_count, out_path)
  for group_path in group_paths:
    os.remove(group_path)
  report(dataset.query_file, source, row_count, t0)
  return row_count


def vec_rows(path, dim):
  return os.path.getsize(path) // (dim * 4)


def concat_vectors(part_paths, dim, skip_rows, row_count, out_path):
  """Writes row_count vectors from the concatenated parts, after skipping the first skip_rows, to out_path."""
  vector_bytes = dim * 4
  skip_bytes = skip_rows * vector_bytes
  remaining_bytes = row_count * vector_bytes
  tmp_path = out_path + ".part"
  with open(tmp_path, "wb") as out:
    for path in part_paths:
      if remaining_bytes == 0:
        break
      part_bytes = os.path.getsize(path)
      if skip_bytes >= part_bytes:
        skip_bytes -= part_bytes
        continue
      with open(path, "rb") as f:
        f.seek(skip_bytes)
        copy_bytes = min(part_bytes - skip_bytes, remaining_bytes)
        skip_bytes = 0
        while copy_bytes > 0:
          chunk = f.read(min(COPY_BUFFER_BYTES, copy_bytes))
          out.write(chunk)
          copy_bytes -= len(chunk)
          remaining_bytes -= len(chunk)
  if remaining_bytes != 0:
    raise RuntimeError(f"{out_path}: ran out of vectors, {remaining_bytes // vector_bytes} short")
  Path(tmp_path).replace(out_path)
  print(f"  wrote {out_path}: {row_count} x {dim}d")


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--dataset", required=True, choices=sorted(DATASETS), help="which Supernova dataset to convert")
  parser.add_argument("--out-dir", required=True, help="where to put the output files (and the per row group parts while converting)")
  parser.add_argument("--num-docs", type=int, default=1_000_000, help="number of doc vectors to write (default 1000000)")
  parser.add_argument(
    "--num-queries",
    type=int,
    default=None,
    help="number of query vectors to write (default: all published queries, or 10000 held out from the corpus for fineweb)",
  )
  parser.add_argument("--keep-parts", action="store_true", help="keep the per row group .vec parts, so a later run can reuse them")
  args = parser.parse_args()

  dataset = DATASETS[args.dataset]
  if args.num_docs < 1:
    parser.error("--num-docs must be at least 1")
  if args.num_queries is not None and args.num_queries < 1:
    parser.error("--num-queries must be at least 1")
  # fineweb has no published query vectors, so the first rows of the corpus are held out as queries
  holdout_rows = 0
  if dataset.query_file is None:
    holdout_rows = args.num_queries if args.num_queries is not None else 10_000

  t0 = time.time()
  out_dir = Path(args.out_dir)
  parts_dir = out_dir / "parts" / args.dataset
  parts_dir.mkdir(parents=True, exist_ok=True)
  prefix = f"qdrant-{args.dataset}-{dataset.dim}d"
  docs_path = str(out_dir / f"{prefix}.docs.vec")
  queries_path = str(out_dir / f"{prefix}.queries.vec")

  print(f"converting {dataset.repo} column {dataset.column} ({dataset.dim}d)")
  if "HF_TOKEN" not in os.environ:
    print("  HF_TOKEN is not set: downloading anonymously, which Hugging Face rate limits more aggressively")

  part_paths = convert_docs(dataset, parts_dir, holdout_rows + args.num_docs)

  print("writing outputs")
  concat_vectors(part_paths, dataset.dim, holdout_rows, args.num_docs, docs_path)
  if dataset.query_file is None:
    concat_vectors(part_paths, dataset.dim, 0, holdout_rows, queries_path)
    query_count = holdout_rows
  else:
    query_count = convert_queries(dataset, queries_path, args.num_queries)
    if args.num_queries is not None and query_count < args.num_queries:
      raise RuntimeError(f"{dataset.query_file} only has {query_count} queries, asked for {args.num_queries}")

  if not args.keep_parts:
    shutil.rmtree(parts_dir)
    if not any(parts_dir.parent.iterdir()):
      parts_dir.parent.rmdir()
  print(f"done: {args.num_docs} docs, {query_count} queries, {time.time() - t0:.1f} sec; use -metric dot_product")


if __name__ == "__main__":
  main()
