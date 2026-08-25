import sys
import numpy as np
sys.path.insert(0, "/Users/bob/llamacpp_kleidi_build/gguf-py")
import gguf

# Independent re-implementation of gguf_transcode.c's algorithm, ported straight from
# eval/quantize_int4.py's quant_group_ef()/quant_group_int8() (the SAME functions
# gguf_transcode.c itself is a C port of) -- this checks the C port against the Python
# original, not against itself. The error_feedback residual term below is carried forward
# exactly as quant_group_ef() does -- not omitted.

GROUP = 64

def quant_group_error_feedback(w2d):
    out, inn = w2d.shape
    ng = inn // GROUP
    g = w2d.reshape(out, ng, GROUP).astype(np.float32)
    scale = np.max(np.abs(g), axis=2, keepdims=True) / 7.0
    scale = np.where(scale < 1e-12, 1.0, scale).astype(np.float32)
    inv = (1.0 / scale)[..., 0]
    s = scale[..., 0]
    codes = np.empty(g.shape, np.int8)
    error_feedback = np.zeros((out, ng), np.float32)
    for p in range(GROUP):
        x = g[:, :, p] + error_feedback
        q = np.clip(np.round(x * inv), -8, 7).astype(np.float32)
        deq = q * s
        error_feedback = x - deq
        codes[:, :, p] = q.astype(np.int8)
    return codes.reshape(out, inn), s  # s: [out,ng]

def quant_group_int8(w2d):
    out, inn = w2d.shape
    ng = inn // GROUP
    g = w2d.reshape(out, ng, GROUP).astype(np.float32)
    scale = np.max(np.abs(g), axis=2, keepdims=True) / 127.0
    scale = np.where(scale < 1e-12, 1.0, scale).astype(np.float32)
    q = np.clip(np.round(g / scale), -127, 127).astype(np.int8)
    return q.reshape(out, inn), scale[..., 0]

def pack_nibbles(codes):
    even = (codes[:, 0::2].astype(np.int32) + 8) & 0x0F
    odd  = (codes[:, 1::2].astype(np.int32) + 8) & 0x0F
    return (even | (odd << 4)).astype(np.uint8)

path, name, mode, prefix = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
r = gguf.GGUFReader(path)
t = [x for x in r.tensors if x.name == name][0]
deq = gguf.quants.dequantize(t.data, t.tensor_type).reshape(-1).astype(np.float32)

# ne[0] = fastest-varying = "in" (row length); ne[1] = "out" (row count) -- same convention
# gguf_transcode_dump.c uses, and already proven to match gguf_dequant_row's flat element
# order in Phase 1 sub-step 2's checksum verification.
in_dim, out_dim = int(t.shape[0]), int(t.shape[1])
w2d = deq.reshape(out_dim, in_dim)

if mode == "q4":
    codes, scales = quant_group_error_feedback(w2d)
    packed = pack_nibbles(codes)
    with open(f"{prefix}.packed.bin", "wb") as f: f.write(packed.tobytes())
    with open(f"{prefix}.scales.bin", "wb") as f: f.write(scales.astype("<f4").tobytes())
    print(f"q4 out={out_dim} in={in_dim} ng={in_dim//GROUP} packed_bytes={packed.nbytes} scales_floats={scales.size}")
elif mode == "q8":
    codes, scales = quant_group_int8(w2d)
    with open(f"{prefix}.codes.bin", "wb") as f: f.write(codes.tobytes())
    with open(f"{prefix}.scales.bin", "wb") as f: f.write(scales.astype("<f4").tobytes())
    print(f"q8 out={out_dim} in={in_dim} ng={in_dim//GROUP} codes_bytes={codes.nbytes} scales_floats={scales.size}")
else:
    raise SystemExit("mode must be q4 or q8")
