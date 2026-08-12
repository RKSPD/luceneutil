#!/bin/zsh
# ivfaster publish sweep: 1-plane NitroxMF coarse + native udot8 rerank, bruteN=600, 1M, warm.
# Search-time grid (nprobe x nprobeMargin) reuses ONE cached index per spillMargin.
# Usage: ivf_sweep.sh <spillMargin> <out_tsv>
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate

SM=$1
OUT=$2
NPROBES=(${(s: :)3})
MARGINS=(${(s: :)4})

# spillMargin=1.10 is the codec default; passing it is harmless but only non-default lands in the index key.
export KNN_INDEX_TYPE=ivfaster
export IVFASTER_FINE_TIER=int8
export IVFASTER_COARSE_BITS=1
export IVFASTER_COARSE_MF=1
export IVFASTER_BRUTE_N=600
export IVFASTER_SPILL_MARGIN=$SM
export KNN_NLIST=2000
export KNN_GCUT_AXES=1
export IVFASTER_REPORT=1

for np in $NPROBES; do
  for m in $MARGINS; do
    export KNN_NPROBE=$np
    export IVFASTER_NPROBE_MARGIN=$m
    LOG=/tmp/ivfsweep_sm${SM}_np${np}_m${m}.log
    python -u src/python/knnPerfTest.py > $LOG 2>&1 || { echo "FAIL sm=$SM np=$np m=$m" >> $OUT; continue; }
    # SUMMARY: recall<TAB>latency<TAB>netCPU<TAB>avgCpuCount ...
    S=$(grep -m1 '^SUMMARY:' $LOG | sed 's/^SUMMARY:[[:space:]]*//')
    REC=$(echo "$S" | awk -F'\t' '{print $1}')
    LAT=$(echo "$S" | awk -F'\t' '{print $2}')
    # engagement: cellsProbed/query, docsScanned/query, udot rows/query
    ENG=$(grep -m1 'graphDescents=2000' $LOG | grep -oE 'docsScanned/query=[0-9]+|cellsProbed/query=[0-9]+|rows/query=[0-9]+' | paste -sd' ')
    echo -e "sm=$SM\tnp=$np\tmargin=$m\trecall=$REC\tlatency_ms=$LAT\t$ENG" | tee -a $OUT
  done
done
echo "DONE sm=$SM" >> $OUT
