#!/usr/bin/env python3
"""Print the device-vs-reference loss lines + error count of one or more BP device logs (exp21e/fix)."""
import re, sys
for p in sys.argv[1:]:
    t = open(p, errors='replace').read()
    L = re.findall(r'\[loss (\d+)\] computed=([-\d.]+)\s+ref=([-\d.]+)\s+diff=([-\d.]+)', t)
    e = re.findall(r'Errors: (\d+) out of (\d+)', t)
    c = re.search(r'commit (\w{8})', t)
    print(p.split('/')[-1], c.group(1) if c else '', 'errors=' + ('/'.join(e[-1]) if e else 'None'),
          ' '.join(f'{i}:{a}/{b}' for i, a, b, d in L[:8]))
