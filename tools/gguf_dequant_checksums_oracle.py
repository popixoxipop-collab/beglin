import sys
sys.path.insert(0, "/Users/bob/llamacpp_kleidi_build/gguf-py")
import gguf, numpy as np

path = sys.argv[1]
r = gguf.GGUFReader(path)
for t in r.tensors:
    try:
        deq = gguf.quants.dequantize(t.data, t.tensor_type).reshape(-1).astype(np.float64)
    except Exception:
        print(f"{t.name} SKIP_UNSUPPORTED_TYPE")
        continue
    n = deq.shape[0]
    weights = np.array([(i % 97) + 1 for i in range(n)], dtype=np.float64)
    checksum = float(np.sum(deq * weights))
    print(f"{t.name} checksum={checksum:.9g} n={n}")
