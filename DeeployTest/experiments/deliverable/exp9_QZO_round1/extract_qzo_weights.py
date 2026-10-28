# SPDX-License-Identifier: MIT
"""QZO-aware WDUMP extractor (route (b) — code injection, no dequantization).

Parses the `[WDUMP s=.. wi=.. n=..] <hex words>` lines from a QZO GVSoC log and decodes each of
the 22 tensors with its TRUE dtype, taken from the zo_train graph's input list (wi = input
index - TRAINING_NUM_DATA_INPUTS): fp32 BN gamma/beta + fc (1 value/word), int8 conv weights
(4 values/word, little-endian), int32 conv bias/rqs-add (1 value/word). Saves an npz of
{input_name: array} carrying the EXACT device representation — ready to be injected into the
quantized inference graph by build_qzo_infer_fixture.py (no float round trip anywhere).

Optionally bit-compares the dump against the export reference's updated_* tensors.

Usage:
  python3 extract_qzo_weights.py --gvsoc-log logs/round1_gvsoc.log \
      --train-onnx fixture/network_zo_train.onnx \
      --out results/dumped_weights.npz [--ref-outputs fixture/outputs.npz]
"""
import argparse
import re
import struct

import numpy as np
import onnx

ELEM = {1: ("f4", 1), 3: ("i1", 4), 6: ("i4", 1)}   # onnx elem_type -> (np dtype, values/word)


def wi_map(train_onnx, num_data_inputs=2):
    m = onnx.load(train_onnx)
    out = []
    for i in m.graph.input[num_data_inputs:]:
        et = i.type.tensor_type.elem_type
        shape = [d.dim_value for d in i.type.tensor_type.shape.dim]
        assert et in ELEM, f"{i.name}: unsupported elem_type {et}"
        out.append((i.name, et, shape))
    return out


def parse_wdump_last_step(logpath):
    pat = re.compile(r"\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*([0-9a-fA-F ]+)")
    dumps = {}
    for line in open(logpath, errors="replace"):
        mm = pat.search(line)
        if mm:
            dumps[(int(mm.group(1)), int(mm.group(2)))] = \
                [int(w, 16) for w in mm.group(4).split()]
    if not dumps:
        raise RuntimeError(f"no [WDUMP] lines in {logpath}")
    last = max(s for s, _ in dumps)
    return last, {wi: words for (s, wi), words in dumps.items() if s == last}


def decode(words, elem_type, shape):
    dt, per_word = ELEM[elem_type]
    raw = b"".join(struct.pack("<I", w) for w in words)
    arr = np.frombuffer(raw, dtype=np.dtype(dt))
    n = int(np.prod(shape)) if shape else arr.size
    assert arr.size >= n, f"decoded {arr.size} < expected {n}"
    return arr[:n].reshape(shape) if shape else arr[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvsoc-log", required=True)
    ap.add_argument("--train-onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-outputs", help="outputs.npz with updated_* for a bit-compare")
    a = ap.parse_args()

    wmap = wi_map(a.train_onnx)
    last, dev = parse_wdump_last_step(a.gvsoc_log)
    print(f"[extract] last dump step s={last}: {len(dev)} tensors (expect {len(wmap)})")
    assert len(dev) == len(wmap), "dump/graph tensor count mismatch"

    out = {}
    for wi, (name, et, shape) in enumerate(wmap):
        arr = decode(dev[wi], et, shape)
        out[name] = arr
        print(f"  wi{wi:2d} {name:28s} {str(arr.dtype):7s} {arr.shape}")
    np.savez(a.out, **out)
    print(f"[extract] wrote {a.out}")

    if a.ref_outputs:
        ref = np.load(a.ref_outputs)
        n_exact = n_diff = n_missing = 0
        for name, arr in out.items():
            for key in (f"updated_{name}", f"updated_{name.replace('.', '_')}"):
                if key in ref.files:
                    r = np.asarray(ref[key]).reshape(arr.shape)
                    if arr.dtype.kind == "f":
                        same = np.array_equal(arr.astype(np.float32), r.astype(np.float32))
                    else:
                        same = np.array_equal(arr.astype(np.int64),
                                              np.rint(np.asarray(r, np.float64)).astype(np.int64))
                    n_exact += int(same); n_diff += int(not same)
                    if not same:
                        d = np.abs(arr.astype(np.float64) - r.astype(np.float64))
                        print(f"  DIFF {name}: n={int((d>0).sum())} max={d.max():.3e}")
                    break
            else:
                n_missing += 1
        print(f"[extract] vs reference updated_*: {n_exact} bit-exact, {n_diff} differ, "
              f"{n_missing} not found in reference")


if __name__ == "__main__":
    main()
