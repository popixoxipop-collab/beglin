import sys
sys.path.insert(0, "/Users/bob/llamacpp_kleidi_build/gguf-py")
import gguf, numpy as np

path = sys.argv[1]
names = sys.argv[2:]
r = gguf.GGUFReader(path)
for name in names:
    t = [x for x in r.tensors if x.name == name][0]
    deq = gguf.quants.dequantize(t.data, t.tensor_type).reshape(-1).astype(np.float32)
    n = deq.shape[0]
    print(f"TENSOR {name} n={n}")
    idxs = list(range(1,97+1))
    weights = np.array([(i % 97) + 1 for i in range(n)], dtype=np.float64)
    checksum = float(np.sum(deq.astype(np.float64) * weights))
    for i in range(min(20, n)):
        print(f"  [{i}]={deq[i]:.9g}")
    print(f"  ... checksum={checksum:.9g}")
    for i in range(max(0, n-20), n):
        print(f"  [{i}]={deq[i]:.9g}")
