"""Write the tiny deterministic F32 safetensors fixture used by tests/regression/*.

Tensors: L0.e0.w, L0.e1.w (routed-expert stand-ins) and dense.w, each [OUT, IN] = [16, 128].
Data: seeded Gaussian (sigma 0.05) with ~1% 8x outliers so per-group maxabs scaling is exercised.

D-reg-1: synthetic seeded fixture instead of a real checkpoint.
  WHY : the regressions are packer/decoder/dispatch agreement bugs, which are data independent; a
        fixture this small keeps the whole suite runnable in seconds on any host.
  COST: says nothing about quality on real weights.
  EXIT: swap tensor() for a slice of a real checkpoint if a quality regression test is wanted.
"""
import json
import random
import struct
import sys

OUT, IN = 16, 128
random.seed(20261019)


def tensor():
    vals = []
    for _ in range(OUT * IN):
        v = random.gauss(0.0, 0.05)
        if random.random() < 0.01:
            v *= 8
        vals.append(v)
    return struct.pack('<%df' % len(vals), *vals)


names = ['L0.e0.w', 'L0.e1.w', 'dense.w']
blobs = {n: tensor() for n in names}
hdr = {}
off = 0
for n in names:
    hdr[n] = {'dtype': 'F32', 'shape': [OUT, IN], 'data_offsets': [off, off + len(blobs[n])]}
    off += len(blobs[n])
h = json.dumps(hdr, separators=(',', ':')).encode()
h += b' ' * ((8 - len(h) % 8) % 8)
with open(sys.argv[1], 'wb') as f:
    f.write(struct.pack('<Q', len(h)))
    f.write(h)
    for n in names:
        f.write(blobs[n])
