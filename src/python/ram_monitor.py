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

"""Live RAM (resident set size) monitor for the KNN search/index JVM.

Polls `ps -o rss= -p <pid>` (resident KB, works on macOS and Linux) on a
background thread, prints a live updating one-line gauge to the terminal,
records (elapsed_sec, rss_mb) samples to a CSV, and on stop writes a small
self-contained HTML line chart of RSS over time (styled after knnPerfChart.html).

This measures the ACTUAL resident footprint of the codec under test -- the
number that matters for the off-heap-centroids work (findings.md §20): heap +
mmap pages currently faulted in. It is the live counterpart to the per-run
ps/vmstat logs, focused on a single process rather than top-N.
"""

import os
import shutil
import subprocess
import sys
import threading
import time

PS_EXE_PATH = shutil.which("ps")
# jstat ships with the JDK. Prefer the one next to JAVA_HOME (matches the JVM under test), else PATH.
_java_home = os.environ.get("JAVA_HOME")
JSTAT_EXE_PATH = (
  (_java_home + "/bin/jstat")
  if _java_home and os.path.exists(_java_home + "/bin/jstat")
  else shutil.which("jstat")
)

# Live-heap-used columns in `jstat -gc` (KB): survivor-0/1 used, eden used, old used. Summed = used heap.
# (Metaspace MU is excluded -- it is off-heap and not what -Xmx bounds.) Parsed by NAME so column
# reordering across JVM versions can't silently corrupt the reading.
_HEAP_USED_COLS = ("S0U", "S1U", "EU", "OU")


def _read_rss_mb(pid):
  """Returns the resident set size of `pid` in MB, or None if the process is gone."""
  if PS_EXE_PATH is None:
    return None
  try:
    # -o rss= : resident KB, no header. Same flag on macOS and Linux.
    out = subprocess.check_output(
      [PS_EXE_PATH, "-o", "rss=", "-p", str(pid)], stderr=subprocess.DEVNULL
    )
  except subprocess.CalledProcessError:
    # ps exits non-zero once the pid no longer exists.
    return None
  text = out.decode().strip()
  if not text:
    return None
  try:
    return int(text.split()[0]) / 1024.0
  except (ValueError, IndexError):
    return None


def _read_heap_used_mb(pid):
  """Returns the JVM's used heap (eden+survivors+old) in MB via `jstat -gc`, or None if unavailable.

  None means "couldn't read it" (no jstat, pid gone, JVM not yet up, or parse failure) -- the caller
  treats heap as optional so the RSS monitor still works on a non-JVM process or a stripped JRE.
  """
  if JSTAT_EXE_PATH is None:
    return None
  try:
    out = subprocess.check_output(
      [JSTAT_EXE_PATH, "-gc", str(pid)], stderr=subprocess.DEVNULL, encoding="utf-8"
    )
  except (subprocess.CalledProcessError, OSError):
    return None
  lines = out.strip().splitlines()
  if len(lines) < 2:
    return None
  header = lines[0].split()
  values = lines[1].split()
  if len(header) != len(values):
    return None
  col = dict(zip(header, values))
  try:
    kb = sum(float(col[c]) for c in _HEAP_USED_COLS)
  except (KeyError, ValueError):
    return None
  return kb / 1024.0


class RAMMonitor:
  """Background thread that samples a process's RSS until stopped.

  Usage:
      mon = RAMMonitor(job.pid, "run-ram.csv", label="LSH ndoc=1M")
      ...                       # run the workload
      peak_mb = mon.stop()      # also writes run-ram.html next to the CSV
  """

  def __init__(self, pid, csv_file_name, poll_interval_sec=1.0, label="", live=True):
    self.pid = pid
    self.csv_file_name = csv_file_name
    self.html_file_name = csv_file_name.rsplit(".", 1)[0] + ".html"
    self.poll_interval_sec = poll_interval_sec
    self.label = label
    self.live = live and sys.stdout.isatty()
    self.samples = []  # list of (elapsed_sec, rss_mb, heap_mb_or_None)
    self.peak_mb = 0.0  # peak RSS (resident, incl. evictable mmap page cache)
    self.peak_heap_mb = 0.0  # peak JVM used heap (what -Xmx bounds; the off-heap-centroids signal, §20)
    self.have_heap = False
    self.stop_now = False
    self.wakey = threading.Condition()
    self.thread = threading.Thread(target=self._run, daemon=True)
    self.thread.start()

  def _run(self):
    start = time.monotonic()
    target = start
    with open(self.csv_file_name, "w") as f:
      f.write("elapsed_sec,rss_mb,heap_used_mb\n")
    while not self.stop_now:
      rss_mb = _read_rss_mb(self.pid)
      heap_mb = _read_heap_used_mb(self.pid)
      elapsed = time.monotonic() - start
      if rss_mb is not None:
        self.samples.append((elapsed, rss_mb, heap_mb))
        if rss_mb > self.peak_mb:
          self.peak_mb = rss_mb
        if heap_mb is not None:
          self.have_heap = True
          if heap_mb > self.peak_heap_mb:
            self.peak_heap_mb = heap_mb
        with open(self.csv_file_name, "a") as f:
          # heap column blank when unreadable (no jstat / JVM not up yet)
          f.write(f"{elapsed:.1f},{rss_mb:.1f},{'' if heap_mb is None else f'{heap_mb:.1f}'}\n")
        if self.live:
          # \r-updated single line: RSS (total resident) and heap (capped) side by side.
          bar_units = int(rss_mb / 256)  # one block per 256 MB
          bar = "█" * min(bar_units, 40)
          heap_str = "  n/a" if heap_mb is None else f"{heap_mb:8.1f} MB"
          sys.stdout.write(
            f"\r[RAM] {self.label} t={elapsed:6.1f}s  rss={rss_mb:8.1f} MB (peak {self.peak_mb:7.1f})  "
            f"heap={heap_str} (peak {self.peak_heap_mb:7.1f})  {bar}"
          )
          sys.stdout.flush()
      target += self.poll_interval_sec
      with self.wakey:
        wait = target - time.monotonic()
        if wait > 0:
          self.wakey.wait(wait)

  def stop(self):
    """Stops sampling, finishes the live line, writes the HTML chart, returns peak RSS MB."""
    self.stop_now = True
    with self.wakey:
      self.wakey.notify()
    self.thread.join()
    if self.live:
      sys.stdout.write("\n")
      sys.stdout.flush()
    self._write_html()
    return self.peak_mb

  def _write_html(self):
    # Two series: RSS (total resident, incl. page cache) and JVM used heap (what -Xmx bounds). Heap is
    # the signal for the off-heap-centroids work (§20): eager creeps toward the cap as buckets grow,
    # off-heap stays flat. Blank heap cells -> null so interpolateNulls bridges the JVM-startup gap.
    rows = ",\n".join(
      f"[{t:.1f}, {rss:.1f}, {'null' if heap is None else f'{heap:.1f}'}]"
      for t, rss, heap in self.samples
    )
    heap_note = "" if self.have_heap else " (heap unavailable -- no jstat?)"
    title = (
      f"RAM over time{(': ' + self.label) if self.label else ''} "
      f"(peak RSS {self.peak_mb:.0f} MB, peak heap {self.peak_heap_mb:.0f} MB){heap_note}"
    )
    html = f"""
<!DOCTYPE html>
<html>
  <head>
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">
      google.charts.load("current", {{packages: ["corechart"]}});
      google.charts.setOnLoadCallback(drawChart);
      function drawChart() {{
        var data = google.visualization.arrayToDataTable([
['elapsed (s)', 'RSS (MB)', 'JVM heap used (MB)'],
{rows}
        ]);
        var options = {{
          title: {title!r},
          pointSize: 2,
          hAxis: {{title: 'elapsed (s)'}},
          vAxis: {{title: 'MB', minValue: 0}},
          series: {{
            0: {{color: '#3366cc'}},  // RSS
            1: {{color: '#dc3912'}}   // heap used
          }},
          interpolateNulls: true
        }};
        var chart = new google.visualization.LineChart(document.getElementById('chart_div'));
        chart.draw(data, options);
      }}
    </script>
  </head>
  <body>
    <h3>{title}</h3>
    <div id="chart_div" style="width: 1200px; height: 600px;"></div>
  </body>
</html>
"""
    with open(self.html_file_name, "w") as f:
      f.write(html)


if __name__ == "__main__":
  # smoke test: monitor this python process for a few seconds
  mon = RAMMonitor(os.getpid(), "ram_test.csv", poll_interval_sec=0.5, label="self-test")
  blobs = []
  for _ in range(6):
    blobs.append(bytearray(50 * 1024 * 1024))  # grow ~50MB/step
    time.sleep(0.5)
  print(f"\npeak = {mon.stop():.1f} MB -> {mon.html_file_name}")
