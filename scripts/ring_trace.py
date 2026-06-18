#!/usr/bin/env python3
# Ring-buffer reader for a gvsoc --trace stream.
# Reads the trace on stdin, keeps only the last RING_LINES lines in memory,
# and on stall (no new data for STALL_SECS), c.unimp, or EOF, dumps that tail
# to RING_OUT and exits. Keeps disk bounded (tail only) and finds the hang PC.
import sys, os, select, time, collections

N = int(os.environ.get("RING_LINES", "300000"))
STALL = float(os.environ.get("STALL_SECS", "45"))
OUT = os.environ.get("RING_OUT", "/scratch/runw/traces/hang.tail")
fd = sys.stdin.fileno()
buf = collections.deque(maxlen=N)
carry = b""
total = 0
last = time.time()
reason = "?"
last_cyc = b"?"
while True:
    r, _, _ = select.select([fd], [], [], 5.0)
    if r:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            reason = "EOF (gvsoc exited — ran clean / no hang)"
            break
        data = carry + chunk
        lines = data.split(b"\n")
        carry = lines.pop()  # last partial line carried to next read
        buf.extend(lines)
        total += len(lines)
        last = time.time()
        if b"c.unimp" in chunk:
            reason = "PC-CORRUPTION (c.unimp trap loop)"
            break
        if b"Invalid access" in chunk or b"Invalid fetch" in chunk:
            reason = "Invalid access/fetch — core aborted"
            break
    else:
        if time.time() - last > STALL:
            reason = "STALLED (no new insn for %ds — idle/hung)" % int(STALL)
            break
# find last cycle from the tail
for ln in reversed(buf):
    p = ln.split(b":")
    if len(p) > 1 and p[1].strip().isdigit():
        last_cyc = p[1].strip()
        break
with open(OUT, "wb") as f:
    f.write(b"\n".join(buf))
print("RING REASON=%s  ~total_lines=%d  last_cyc=%s  out=%s" %
      (reason, total, last_cyc.decode(errors="replace"), OUT), flush=True)
