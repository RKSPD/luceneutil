Qdrant Supernova benchmark datasets
===================================

load_qdrant_supernova.py converts the dense vectors of the datasets Qdrant
released with its Supernova benchmark into luceneutil .vec files:

  https://qdrant.tech/blog/qdrant-fineweb-10b-release/

The datasets are only published as parquet on Hugging Face; there is no
other mirror.  Each parquet row holds text and metadata, plus sparse and
(for PubMed) multi-vector embeddings that luceneutil does not use.  The
script reads only the dense embedding column, using HTTP range requests, so
most of each file is never downloaded:

  --dataset  Hugging Face repo     docs    dense vectors        queries   license
  fineweb    Qdrant/FineWeb-10B    10.07B  768d  float16 mGTE   none (*)  ODC-BY-1.0
  pubmed     Qdrant/PubMed-MV      23.9M   1024d float32 BGE-M3 10,000    Apache-2.0
  coyo       Qdrant/Coyo-VE        15.4M   2048d float32 Qwen3  25,000    CC-BY-4.0

  fineweb: ~1.4 GB fetched per 1M docs (of ~4.4 GB per 1M rows of parquet)
  pubmed:  ~4 MB fetched per 1K docs (the shards are ~99% multi-vector data)
  coyo:    ~8 KB per doc; the vectors are nearly all of the file

(*) FineWeb's queries are MS MARCO queries, whose text and embeddings Qdrant
does not redistribute.  So for fineweb the first --num-queries corpus vectors
(default 10000) are held out as queries, and the docs start after them; the
two never overlap.  For pubmed and coyo the published query embeddings are
used.

float16 FineWeb vectors are widened to float32 exactly.  All vectors are
unit-length, so use -metric dot_product.

The published ground truth is exact top-1000 over the whole corpus, so it
does not apply to a subset; let knnPerfTest.py compute exact NN as usual.

Be sure to follow the licenses above, and the source data's own terms
(FineWeb is also subject to the Common Crawl terms of use).


Running
-------

  pip install numpy pyarrow
  python src/python/load_qdrant_supernova.py --dataset fineweb --out-dir /path/to/data --num-docs 1000000

This writes OUT_DIR/qdrant-DATASET-DIMd.docs.vec and
OUT_DIR/qdrant-DATASET-DIMd.queries.vec.  Docs are taken in order from the
sorted shard list, so the same --num-docs always gives the same vectors, and
a smaller --num-docs is a prefix of a larger one.  To shuffle them, use
shuffle_vec_file.py.

Each parquet row group is first written to its own part file under
OUT_DIR/parts/DATASET, and the parts are concatenated at the end.  If a run
is interrupted, rerun the same command and it resumes from the parts already
on disk.  The parts are removed when the run completes, unless you pass
--keep-parts, in which case a later run asking for more docs only downloads
the new row groups.  Peak disk use is about twice the size of the outputs.


Rate limits
-----------

Hugging Face rate limits anonymous downloads.  The script retries throttled
and failed requests with backoff (honoring Retry-After), but for large runs
set a (free) Hugging Face access token, which gets higher limits:

  export HF_TOKEN=hf_...

The token is only sent to huggingface.co, to resolve each file; the range
reads go to the signed CDN url Hugging Face redirects to, without the token.
