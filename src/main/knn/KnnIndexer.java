/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package knn;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.ConcurrentMergeScheduler;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.MergePolicy.OneMerge;
import org.apache.lucene.index.MergeRateLimiter;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.misc.index.BPReorderingMergePolicy;
import org.apache.lucene.misc.index.BpVectorReorderer;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.PrintStreamInfoStream;

import static knn.KnnGraphTester.DOCTYPE_CHILD;
import static knn.KnnGraphTester.DOCTYPE_PARENT;

public class KnnIndexer implements FormatterLogger {

  // Larger buffer => fewer flush segments. This matters a lot for IVF codecs (ivfaster), which re-run
  // Lloyd clustering on every flush AND every background merge: 128MB gave ~31 segments for 1M x 1024d
  // (~4x the clustering work of the 512MB the lsh_ivf branch uses). HNSW/flat also benefit (fewer, larger
  // segment graphs). Raised 128 -> 512 to match lsh_ivf's ivfaster indexing speed.
  // Overridable via -Dknn.writerBufferMB. Bigger buffer => larger initial flush segments => fewer
  // merge tiers => less IVF re-clustering amplification on the way up to large merged segments.
  private static final double WRITER_BUFFER_MB = Double.parseDouble(System.getProperty("knn.writerBufferMB", "1024"));

  private final Path docsPath;
  private final Path indexPath;
  private final VectorEncoding vectorEncoding;
  private final int dim;
  private final VectorSimilarityFunction similarityFunction;
  private final Codec codec;
  private final int numDocs;
  private final int docsStartIndex;
  private final int numIndexThreads;
  private final boolean quiet;
  private final boolean parentJoin;
  private final Path parentJoinMetaPath;
  private final boolean useBp;
  private final FilterScheme filterScheme;
  private final TrackingConcurrentMergeScheduler tcms;
  private final TrackingTieredMergePolicy ttmp;
  private final boolean rerank;
  // When true the docs file is read as flat fp16 (dim*2 bytes/vec) and widened to float; the field
  // is still FLOAT32. Only affects the docs read, not queries. Halves the on-disk docs file.
  private final boolean halfFloatDocs;

  public KnnIndexer(Path docsPath, Path indexPath, Codec codec, int numIndexThreads,
                    VectorEncoding vectorEncoding, int dim,
                    VectorSimilarityFunction similarityFunction, int numDocs, int docsStartIndex, boolean quiet,
                    boolean parentJoin, Path parentJoinMetaPath, boolean useBp, FilterScheme filterScheme,
                    boolean rerank, boolean halfFloatDocs) {
    this.docsPath = docsPath;
    this.indexPath = indexPath;
    this.codec = codec;
    this.numIndexThreads = numIndexThreads;
    this.vectorEncoding = vectorEncoding;
    this.dim = dim;
    this.similarityFunction = similarityFunction;
    this.numDocs = numDocs;
    this.docsStartIndex = docsStartIndex;
    this.quiet = quiet;
    this.parentJoin = parentJoin;
    this.parentJoinMetaPath = parentJoinMetaPath;
    this.useBp = useBp;
    this.filterScheme = filterScheme;
    this.rerank = rerank;
    this.halfFloatDocs = halfFloatDocs;
    if (halfFloatDocs && vectorEncoding != VectorEncoding.FLOAT32) {
      throw new IllegalArgumentException("halfFloatDocs requires FLOAT32 vector encoding (fp16 widens to float)");
    }
    this.tcms = new TrackingConcurrentMergeScheduler();
    this.ttmp = new TrackingTieredMergePolicy();
  }

  // result of indexing: time spent indexing (excluding merges) and time spent waiting for merges,
  // both in milliseconds
  public record IndexResult(int indexTimeMsec, int mergeTimeMsec) {}

  public IndexResult createIndex() throws IOException, InterruptedException {
    IndexWriterConfig iwc = new IndexWriterConfig().setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    iwc.setCodec(codec);
    iwc.setMergeScheduler(tcms);
    // iwc.setMergePolicy(NoMergePolicy.INSTANCE);
    iwc.setRAMBufferSizeMB(WRITER_BUFFER_MB);
    
    iwc.setUseCompoundFile(false);
    // iwc.setMaxBufferedDocs(10000);

    // sidestep an apparent IW bug that causes merges kicked off during commit to be aborted on later commit/close instead of waited on, hrmph
    iwc.setMaxFullFlushMergeWaitMillis(0);

    // aim for more compact/realistic index:
    
    iwc.setMergePolicy(ttmp);
    ttmp.setFloorSegmentMB(256);
    // TieredMergePolicy caps merged segments at maxMergedSegmentMB (Lucene default 5120MB ~= 1.6M
    // docs at 768d fp32). To build large (~50M-doc) IVF segments we must raise this; unset leaves the
    // Lucene default. Set generously -- TMP treats segments above ~maxMergedSegmentMB/2 as merge-
    // ineligible, so the value is roughly 2x the target segment size on disk.
    final double maxMergedSegmentMB = Double.parseDouble(System.getProperty("knn.maxMergedSegmentMB", "-1"));
    if (maxMergedSegmentMB > 0) {
      ttmp.setMaxMergedSegmentMB(maxMergedSegmentMB);
      System.out.println("TieredMergePolicy maxMergedSegmentMB=" + maxMergedSegmentMB);
    }
    iwc.getCodec().compoundFormat().setShouldUseCompoundFile(false);
    // tmp.setSegmentsPerTier(5);
    if (useBp) {
      iwc.setMergePolicy(new BPReorderingMergePolicy(iwc.getMergePolicy(), new BpVectorReorderer(KnnGraphTester.KNN_FIELD)));
    }

    ConcurrentMergeScheduler cms = (ConcurrentMergeScheduler) iwc.getMergeScheduler();
    // cms.setMaxMergesAndThreads(24, 12);

    // CMS concurrency, overridable, because the default interacts badly with a codec that is itself
    // internally parallel. CMS defaults maxThreadCount to min(4, cores/2) on an SSD, so up to 4 merges
    // run at once -- and an IVF codec like ivfaster fans each merge out over every core internally, so
    // total build concurrency is the PRODUCT of the two. maxThreadCount=1 gives the other arrangement:
    // one merge at a time, with all cores routing documents inside it.
    //   -Dknn.cmsMaxThreadCount=N   (maxMergeCount defaults to N+5, as CMS itself does)
    final int cmsThreads = Integer.getInteger("knn.cmsMaxThreadCount", -1);
    if (cmsThreads > 0) {
      final int cmsMerges = Integer.getInteger("knn.cmsMaxMergeCount", cmsThreads + 5);
      cms.setMaxMergesAndThreads(cmsMerges, cmsThreads);
      System.out.println("CMS maxMergeCount=" + cmsMerges + " maxThreadCount=" + cmsThreads);
    }

    FieldType fieldType =
        switch (vectorEncoding) {
          case BYTE -> KnnByteVectorField.createFieldType(dim, similarityFunction);
          case FLOAT32 -> KnnFloatVectorField.createFieldType(dim, similarityFunction);
          default -> throw new IllegalArgumentException("unsupported vector encoding: " + vectorEncoding);
        };
    if (rerank && vectorEncoding != VectorEncoding.FLOAT32) {
      throw new IllegalArgumentException("rerank requires FLOAT32 vector encoding");
    }
    FieldType rerankFieldType = rerank ? KnnFloatVectorField.createFieldType(dim, similarityFunction) : null;
    if (quiet == false) {
      // iwc.setInfoStream(new PrintStreamInfoStream(System.out));
      System.out.println("creating index in " + indexPath);
    }

    // Opt-in InfoStream that surfaces ONLY the ivfaster-evo codec's "IVFE" build messages (cold/hot
    // start origin, carried assignments, seed cells, Lloyd iterations) without the full IndexWriter
    // firehose. Enable with -Dknn.ivfInfoStream=true. Used to verify hot-start engagement.
    if (Boolean.getBoolean("knn.ivfInfoStream")) {
      iwc.setInfoStream(new InfoStream() {
        @Override
        public void message(String component, String message) {
          System.out.println("[" + component + "] " + message);
        }

        @Override
        public boolean isEnabled(String component) {
          return "SIVF".equals(component);
        }

        @Override
        public void close() {}
      });
    }

    if (indexPath.toFile().exists() == false) {
      indexPath.toFile().mkdirs();
    }

    long startNS = System.nanoTime(), elapsedNS, mergeNS;
    try (FSDirectory dir = FSDirectory.open(indexPath);
         IndexWriter iw = new IndexWriter(dir, iwc);
         FileChannel in = FileChannel.open(docsPath)) {
      long docsPathSizeInBytes = in.size();
      int recordBytes = halfFloatDocs ? dim * Short.BYTES : dim * vectorEncoding.byteSize;
      if (docsPathSizeInBytes % recordBytes != 0) {
        throw new IllegalArgumentException("docsPath \"" + docsPath + "\" does not contain a whole number of vectors?  size=" + docsPathSizeInBytes);
      }
      System.out.println((int) (docsPathSizeInBytes / recordBytes) + " doc vectors in docsPath \"" + docsPath + "\""
          + (halfFloatDocs ? " (fp16)" : ""));

      VectorReader vectorReader = halfFloatDocs
          ? VectorReader.createHalfFloat(in, dim, docsStartIndex)
          : VectorReader.create(in, dim, vectorEncoding, docsStartIndex);
      log("parentJoin=%s\n", parentJoin);
      if (parentJoin == false) {
        ExecutorService exec = Executors.newFixedThreadPool(numIndexThreads);
        AtomicInteger numDocsIndexed = new AtomicInteger();
        List<Thread> threads = new ArrayList<>();
        for (int i=0;i<numIndexThreads;i++) {
          Thread t = new IndexerThread(iw, dim, vectorReader, vectorEncoding, fieldType, numDocsIndexed, numDocs, filterScheme, rerankFieldType);
          t.setDaemon(true);
          t.start();
          threads.add(t);
        }
        for (Thread t : threads) {
          t.join();
        }
      } else {
        // TODO: multi-threaded!
        // create parent-block join documents
        try (BufferedReader br = Files.newBufferedReader(parentJoinMetaPath)) {
          String[] headers = br.readLine().trim().split(",");
          if (headers.length != 2) {
            throw new IllegalStateException("Expected two columns in parentJoinMetadata csv. Found: " + headers.length);
          }
          log("Parent join metaFile columns: %s | %s\n", headers[0], headers[1]);
          int childDocs = 0;
          int parentDocs = 0;
          int docId = 0;
          String prevWikiId = "null";
          String currWikiId;
          List<Document> block = new ArrayList<>();
          do {
            String[] line = br.readLine().trim().split(",");
            currWikiId = line[0];
            String currParaId = line[1];
            Document doc = new Document();
            switch (vectorEncoding) {
              case BYTE -> {
                byte[] vector = ((VectorReaderByte) vectorReader).nextBytes();
                if (filterScheme == null || filterScheme.keepUnfiltered) {
                  doc.add(new KnnByteVectorField(KnnGraphTester.KNN_FIELD, vector, fieldType));
                }
                if (filterScheme != null && filterScheme.filter.get(docId)) {
                  doc.add(new KnnByteVectorField(KnnGraphTester.KNN_FIELD_FILTERED, vector, fieldType));
                }
              }
              case FLOAT32 -> {
                float[] vector = vectorReader.next();
                if (filterScheme == null || filterScheme.keepUnfiltered) {
                  doc.add(new KnnFloatVectorField(KnnGraphTester.KNN_FIELD, vector, fieldType));
                }
                if (filterScheme != null && filterScheme.filter.get(docId)) {
                  doc.add(new KnnFloatVectorField(KnnGraphTester.KNN_FIELD_FILTERED, vector, fieldType));
                }
                if (rerankFieldType != null) {
                  doc.add(new KnnFloatVectorField(KnnGraphTester.KNN_FIELD_RERANK, vector, rerankFieldType));
                }
              }
            }
            doc.add(new StoredField(KnnGraphTester.ID_FIELD, docId++));
            doc.add(new StringField(KnnGraphTester.WIKI_ID_FIELD, currWikiId, Field.Store.YES));
            doc.add(new StringField(KnnGraphTester.WIKI_PARA_ID_FIELD, currParaId, Field.Store.YES));
            doc.add(new StringField(KnnGraphTester.DOCTYPE_FIELD, DOCTYPE_CHILD, Field.Store.NO));
            childDocs++;

            // Close block and create a new one when wiki article changes.
            if (!currWikiId.equals(prevWikiId) && !"null".equals(prevWikiId)) {
              Document parent = new Document();
              parent.add(new StoredField(KnnGraphTester.ID_FIELD, docId++));
              parent.add(new StringField(KnnGraphTester.DOCTYPE_FIELD, DOCTYPE_PARENT, Field.Store.NO));
              parent.add(new StringField(KnnGraphTester.WIKI_ID_FIELD, prevWikiId, Field.Store.YES));
              parent.add(new StringField(KnnGraphTester.WIKI_PARA_ID_FIELD, "_", Field.Store.YES));
              block.add(parent);
              iw.addDocuments(block);
              parentDocs++;
              // create new block for the next article
              block = new ArrayList<>();
              block.add(doc);
            } else {
              block.add(doc);
            }
            prevWikiId = currWikiId;
            if (childDocs % 25000 == 0) {
              log("indexed %d child documents, with %d parents\n", childDocs, parentDocs);
            }
          } while (childDocs < numDocs);
          if (!block.isEmpty()) {
            Document parent = new Document();
            parent.add(new StoredField(KnnGraphTester.ID_FIELD, docId++));
            parent.add(new StringField(KnnGraphTester.DOCTYPE_FIELD, DOCTYPE_PARENT, Field.Store.NO));
            parent.add(new StringField(KnnGraphTester.WIKI_ID_FIELD, prevWikiId, Field.Store.YES));
            parent.add(new StringField(KnnGraphTester.WIKI_PARA_ID_FIELD, "_", Field.Store.YES));
            block.add(parent);
            iw.addDocuments(block);
          }
          log("Indexed %d documents with %d parent docs. now flush\n", childDocs, parentDocs);
        }
      }

      // give merges a chance to kick off and finish:
      log("now IndexWriter.commit()\n");
      iw.commit();

      elapsedNS = System.nanoTime() - startNS;

      // merge time is intentionally excluded from the indexing time reported above, but we still
      // track it explicitly here so it is not silently dropped (otherwise both index(s) and the
      // later force_merge(s) become noisy depending on how much merge work happened in this phase)
      long mergeStartNS = System.nanoTime();
      waitForMergesWithStatus(ttmp, tcms, this);
      mergeNS = System.nanoTime() - mergeStartNS;
    }
    log("Indexed %d docs in %d seconds (plus %d seconds waiting for merges)\n",
        numDocs, TimeUnit.NANOSECONDS.toSeconds(elapsedNS), TimeUnit.NANOSECONDS.toSeconds(mergeNS));
    return new IndexResult((int) TimeUnit.NANOSECONDS.toMillis(elapsedNS),
                           (int) TimeUnit.NANOSECONDS.toMillis(mergeNS));
  }

  public static void waitForMergesWithStatus(TrackingTieredMergePolicy ttmp, TrackingConcurrentMergeScheduler tcms, FormatterLogger log) throws InterruptedException {
    long startNS = System.nanoTime();
    
    // wait for running merges to complete, and print coarse status updates
    log.log("now wait for already running merges to finish\n");

    // silliness to just be able to print progress in waiting (so long...) for merges:
    long nextPrintNS = System.nanoTime();
    long lastNonZeroNS = -1;
    
    while (true) {
      long nowNS = System.nanoTime();
      int mergeCount = tcms.mergeThreadCount();
      if (mergeCount > 0) {
        lastNonZeroNS = nowNS;
      } else if (nowNS - lastNonZeroNS > 1_000_000_000) {
        // hackity -- in case merges finish and new merges kick off but that is not atomic, we wait
        // for 1 second period of no merges.  we are still calling CMS.sync below, so worst case that will
        // accurately wait for all merges in that call.
        break;
      }
      if (nowNS > nextPrintNS) {
        log.log("%.1f sec: %d segments, %d merges\n%s\n",
                (nowNS - startNS)/1_000_000_000.,
                ttmp.getLastSegmentCount(),
                mergeCount, tcms.whatsUp());
        // print status every 5 sec
        nextPrintNS += 5_000_000_000L;
      }
      Thread.sleep(100);
    }
    log.log("done first pass waiting for merges\n");
    tcms.sync();
    log.log("done second pass waiting for merges (CMS.sync())\n");
  }

  public void log(String msg, Object... args) {
    if (quiet == false) {
      System.out.printf(msg, args);
      System.out.flush();
    }
  }

  private static record OneMergeAndStartTime(long startTimeNS, OneMerge oneMerge) {};

  // Simple little parasite class to track index stats (total segment count)
  static class TrackingTieredMergePolicy extends TieredMergePolicy {

    private volatile int lastSegmentCount = -1;
    
    @Override
    public MergeSpecification findMerges(
                                         MergeTrigger mergeTrigger, SegmentInfos infos, MergeContext mergeContext) throws IOException {
      lastSegmentCount = infos.size();
      return super.findMerges(mergeTrigger, infos, mergeContext);
    }

    @Override
    public MergeSpecification findForcedMerges(
                                               SegmentInfos infos,
                                               int maxSegmentCount,
                                               Map<SegmentCommitInfo, Boolean> segmentsToMerge,
                                               MergeContext mergeContext)
      throws IOException {
      lastSegmentCount = infos.size();
      return super.findForcedMerges(infos, maxSegmentCount, segmentsToMerge, mergeContext);
    }
    
    public int getLastSegmentCount() {
      return lastSegmentCount;
    }
  }

  // Simple little parasite class to track progress in merges...
  static class TrackingConcurrentMergeScheduler extends ConcurrentMergeScheduler {

    private final List<OneMergeAndStartTime> running = new ArrayList<>();
    
    @Override
    protected void doMerge(MergeSource mergeSource, OneMerge merge) throws IOException {
      OneMergeAndStartTime mos = new OneMergeAndStartTime(System.nanoTime(), merge);
      synchronized(running) {
        running.add(mos);
      }
      try {
        super.doMerge(mergeSource, merge);
      } finally {
        synchronized(running) {
          running.remove(mos);
        }
      }
    }

    // TODO: maybe wrap Directory to watch IO?
    public String whatsUp() {
      List<String> addEm = new ArrayList<>();
      long nowNS = System.nanoTime();
      synchronized(running) {
        for (OneMergeAndStartTime mos : running) {
          // there is a brief window when a merge is kicked off (and added to our
          // "running" List) but it has not yet had a SegmentCommitInfo set for
          // the merged segment, so we sidestep NPE:
          SegmentCommitInfo info = mos.oneMerge.getMergeInfo();
          String segName;
          if (info == null) {
            segName = "?";
          } else {
            segName = info.info.name;
          }
          addEm.add(String.format(Locale.ROOT, "%6.1fs %3s: %2d segs, %.1f K docs, %6.1f MB",
                                  (nowNS - mos.startTimeNS)/1000000000.,
                                  segName,
                                  mos.oneMerge.segments.size(),
                                  mos.oneMerge.totalNumDocs()/1000.,
                                  mos.oneMerge.totalBytesSize()/1024./1024.));
        }
      }
      return "  " + String.join("\n  ", addEm);
    }
  }

  record FilterScheme(Bits filter, boolean keepUnfiltered) { }
}
