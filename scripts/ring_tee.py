#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""
Ring-buffer tee.  Reads stdin, writes to a fixed set of rotating files,
discarding the oldest data once total disk usage hits `--keep * --size`.

Used by cmake/simulation.cmake to keep gvsoc's `--trace` output from
filling the disk: instead of one growing trace.log, you get
  <prefix>.0  <prefix>.1  <prefix>.2  ...
each capped at --size, overwritten round-robin.  <prefix>.current is a
symlink to the file being written right now.

While running, a heartbeat is printed to stderr every --heartbeat
seconds so you can tell the sim is actually progressing.

SIGUSR1 dumps a concatenated <prefix>.snapshot (oldest -> newest) of
everything still on disk, for grabbing the recent window when gvsoc
hangs:  `kill -USR1 <ring_tee_pid>`.
"""
import argparse
import os
import re
import signal
import sys
import time


def parse_size(s):
    m = re.fullmatch(r'\s*(\d+)\s*([KMGkmg]?)B?\s*', s)
    if not m:
        raise ValueError("bad size %r (use e.g. 500M, 1G)" % s)
    return int(m.group(1)) * {'': 1, 'k': 1 << 10, 'm': 1 << 20, 'g': 1 << 30}[m.group(2).lower()]


class RingTee:

    def __init__(self, prefix, size, keep, passthrough, heartbeat):
        self.prefix = prefix
        self.size = size
        self.keep = keep
        self.passthrough = passthrough
        self.heartbeat = heartbeat
        self.idx = 0
        self.bytes_in_file = 0
        self.total_bytes = 0
        self.start = time.monotonic()
        self.last_hb = self.start
        self.fh = None
        # Wipe any stale ring files from a previous run so old data
        # can't masquerade as part of the current trace.
        for i in range(keep):
            try:
                os.unlink("%s.%d" % (prefix, i))
            except OSError:
                pass
        self._open_current()
        signal.signal(signal.SIGUSR1, self._snapshot)

    def _path(self, i):
        return "%s.%d" % (self.prefix, i % self.keep)

    def _open_current(self):
        if self.fh is not None:
            self.fh.close()
        path = self._path(self.idx)
        self.fh = open(path, 'wb')
        self.bytes_in_file = 0
        cur = "%s.current" % self.prefix
        try:
            if os.path.lexists(cur):
                os.unlink(cur)
            os.symlink(os.path.basename(path), cur)
        except OSError:
            pass

    def _snapshot(self, *_):
        try:
            self.fh.flush()
        except Exception:
            pass
        snap = "%s.snapshot" % self.prefix
        try:
            with open(snap, 'wb') as out:
                lo = max(0, self.idx - self.keep + 1)
                for i in range(lo, self.idx + 1):
                    try:
                        with open(self._path(i), 'rb') as f:
                            while True:
                                buf = f.read(1 << 20)
                                if not buf:
                                    break
                                out.write(buf)
                    except OSError:
                        pass
            sys.stderr.write("[ring_tee] snapshot -> %s\n" % snap)
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write("[ring_tee] snapshot failed: %s\n" % e)

    def _hb(self, force = False):
        now = time.monotonic()
        if not force and now - self.last_hb < self.heartbeat:
            return
        self.last_hb = now
        sys.stderr.write("[ring_tee] +%4ds total=%dMB file=%d fill=%dMB/%dMB\n" % (
            int(now - self.start),
            self.total_bytes >> 20,
            self.idx % self.keep,
            self.bytes_in_file >> 20,
            self.size >> 20,
        ))
        sys.stderr.flush()

    def run(self):
        rd = sys.stdin.buffer
        out = sys.stdout.buffer if self.passthrough else None
        BUF = 1 << 16
        try:
            while True:
                # read1: return as soon as any bytes are available, so heartbeats
                # and SIGUSR1 handlers fire promptly even on a slow trace stream.
                chunk = rd.read1(BUF)
                if not chunk:
                    break
                # Rotate before writing if this chunk would push us over.
                if self.bytes_in_file + len(chunk) > self.size:
                    self.idx += 1
                    self._open_current()
                self.fh.write(chunk)
                self.bytes_in_file += len(chunk)
                self.total_bytes += len(chunk)
                if out is not None:
                    try:
                        out.write(chunk)
                        out.flush()
                    except BrokenPipeError:
                        out = None
                self._hb()
        finally:
            try:
                self.fh.flush()
                self.fh.close()
            except Exception:
                pass
            self._hb(force = True)
            sys.stderr.write("[ring_tee] done, total=%dMB across %d file(s)\n" %
                             (self.total_bytes >> 20, min(self.idx + 1, self.keep)))
            sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prefix', required = True, help = 'Output prefix; writes <prefix>.0 .. <prefix>.{keep-1}')
    ap.add_argument('--size', default = '500M', help = 'Per-file size (default 500M). Suffixes K/M/G.')
    ap.add_argument('--keep', type = int, default = 3, help = 'Number of rotating files to keep (default 3)')
    ap.add_argument('--passthrough',
                    action = 'store_true',
                    help = 'Also echo stdin to stdout (default: silent on stdout)')
    ap.add_argument('--heartbeat',
                    type = float,
                    default = 5.0,
                    help = 'Heartbeat to stderr every N seconds (default 5)')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.prefix)) or '.', exist_ok = True)
    RingTee(args.prefix, parse_size(args.size), args.keep, args.passthrough, args.heartbeat).run()


if __name__ == '__main__':
    main()
