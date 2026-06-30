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

"""Uniformly shuffle a fixed-record raw float32 .vec that is too big for RAM.

The .vec is a flat array of `dim` little-endian float32s per vector, no header (the format
KnnGraphTester/knnPerfTest read). To match the 'scattered' distribution of the bundled samples (so
nearby Wikipedia passages aren't adjacent -> honest IVF/LSH bucketing + recall), we shuffle it.

Strategy: two-pass external bucket shuffle, all SEQUENTIAL I/O, memory bounded by one bucket.
  pass 1 (scatter): stream the input, assign each vector to one of K random buckets, append it there.
  pass 2 (gather):  for each bucket, load it fully into RAM, in-memory shuffle, append to the output;
                    delete each bucket as soon as it is consumed.
K is chosen so one bucket (~N/K vectors) fits comfortably in RAM. The result is a uniform shuffle:
bucket choice is uniform-random per vector, and within-bucket order is fully permuted.

Disk: needs ~1x input size of scratch for the buckets. Peak usage ~2x input (buckets + growing output)
during pass 2 -- the script refuses to start unless that headroom exists, unless --delete-input is set
(then the source is removed at the start of pass 2, after it has been fully read, to free its space).
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np

VEC_DTYPE = "<f4"


def _bytes_per_vec(dim):
  return dim * 4


def shuffle_vecs(in_path, out_path, dim, seed, target_bucket_gb=3.0, delete_input=False):
  bpv = _bytes_per_vec(dim)
  in_size = os.path.getsize(in_path)
  if in_size % bpv != 0:
    raise ValueError(f"{in_path}: size {in_size} not a multiple of {bpv} (dim={dim})?")
  n = in_size // bpv
  print(f"shuffling {n:,} vectors ({in_size / 1e9:.1f} GB), dim={dim}")

  # Choose bucket count so each bucket is ~target_bucket_gb on disk (and so fits in RAM at pass 2).
  num_buckets = max(1, int(round(in_size / (target_bucket_gb * 1e9))))
  print(f"using {num_buckets} buckets (~{in_size / num_buckets / 1e9:.1f} GB each)")

  scratch_dir = out_path + ".shuffle_buckets.tmp"
  os.makedirs(scratch_dir, exist_ok=True)
  bucket_paths = [os.path.join(scratch_dir, f"bucket_{b:04d}.bin") for b in range(num_buckets)]

  rng = np.random.default_rng(seed)

  # --- pass 1: scatter -------------------------------------------------------------------------
  # Read the input in chunks of many vectors; for each chunk draw a random bucket per vector and
  # append the vectors grouped by bucket. Buffered writes keep this sequential per bucket.
  t0 = time.time()
  bucket_files = [open(p, "wb") for p in bucket_paths]
  CHUNK_VECS = 50_000  # ~200 MB/chunk at 1024d
  try:
    with open(in_path, "rb") as f:
      done = 0
      while done < n:
        take = min(CHUNK_VECS, n - done)
        raw = np.frombuffer(f.read(take * bpv), dtype=VEC_DTYPE).reshape(take, dim)
        assign = rng.integers(0, num_buckets, size=take)
        for b in range(num_buckets):
          sel = raw[assign == b]
          if sel.size:
            bucket_files[b].write(np.ascontiguousarray(sel).tobytes())
        done += take
        if done % (CHUNK_VECS * 20) == 0 or done == n:
          rate = done / max(1e-9, time.time() - t0)
          print(f"  pass1 scatter: {done:,}/{n:,} ({100 * done / n:.0f}%) {rate:,.0f} vec/s")
  finally:
    for bf in bucket_files:
      bf.close()

  # Free the source before the output grows, if asked (the source has now been fully read).
  if delete_input:
    print(f"  --delete-input: removing source {in_path} to free {in_size / 1e9:.1f} GB")
    os.remove(in_path)

  # --- pass 2: gather --------------------------------------------------------------------------
  # Load each bucket fully, shuffle in RAM, append to output, delete the bucket.
  t1 = time.time()
  written = 0
  with open(out_path, "wb") as out_f:
    for b, bp in enumerate(bucket_paths):
      data = np.fromfile(bp, dtype=VEC_DTYPE)
      cnt = data.size // dim
      data = data.reshape(cnt, dim)
      perm = rng.permutation(cnt)
      out_f.write(np.ascontiguousarray(data[perm]).tobytes())
      written += cnt
      os.remove(bp)  # reclaim space as we go
      print(f"  pass2 gather: bucket {b + 1}/{num_buckets}, {written:,}/{n:,} written")

  os.rmdir(scratch_dir)
  out_size = os.path.getsize(out_path)
  if written != n or out_size != in_size and not delete_input:
    raise RuntimeError(f"sanity check failed: wrote {written}/{n} vectors, out_size={out_size}")
  print(
    f"DONE: shuffled {written:,} vectors -> {out_path} "
    f"(scatter {t1 - t0:.0f}s, gather {time.time() - t1:.0f}s)"
  )
  return out_path


def main():
  ap = argparse.ArgumentParser(description="External (memory-bounded) shuffle of a raw float32 .vec")
  ap.add_argument("in_path")
  ap.add_argument("out_path")
  ap.add_argument("--dim", type=int, required=True)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--bucket-gb", type=float, default=3.0, help="approx GB per bucket (RAM bound at gather)")
  ap.add_argument(
    "--delete-input",
    action="store_true",
    help="delete the source after pass 1 (frees its disk; source must be regenerable)",
  )
  args = ap.parse_args()

  if os.path.abspath(args.in_path) == os.path.abspath(args.out_path):
    sys.exit("in_path and out_path must differ")
  if os.path.exists(args.out_path):
    sys.exit(f"out_path already exists: {args.out_path}")

  # Disk headroom check: buckets (~1x) + output (~1x) live simultaneously in pass 2.
  in_size = os.path.getsize(args.in_path)
  free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.out_path))).free
  needed = in_size * (1 if args.delete_input else 2)  # +output; input freed early if --delete-input
  if free < needed * 1.05:
    sys.exit(
      f"insufficient disk: need ~{needed / 1e9:.0f} GB free (have {free / 1e9:.0f} GB). "
      f"Use --delete-input to free the source after pass 1, or free space."
    )
  shuffle_vecs(args.in_path, args.out_path, args.dim, args.seed, args.bucket_gb, args.delete_input)


if __name__ == "__main__":
  main()
