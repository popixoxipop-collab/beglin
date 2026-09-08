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

# D-qNg64-1: n-parameterized generalization of quant_group_error_feedback() above -- same
# error-feedback loop shape, reciprocal (`inv`, NOT direct division -- q4's convention, not
# q8's), np.round() (== rintf(), matches gguf_quantize_qNg64()'s own header comment on why).
def quant_group_qN(n, w2d):
    out, inn = w2d.shape
    ng = inn // GROUP
    g = w2d.reshape(out, ng, GROUP).astype(np.float32)
    qmax = float((1 << (n - 1)) - 1)
    qmin = float(-(1 << (n - 1)))
    scale = np.max(np.abs(g), axis=2, keepdims=True) / qmax
    scale = np.where(scale < 1e-12, 1.0, scale).astype(np.float32)
    inv = (1.0 / scale)[..., 0]
    s = scale[..., 0]
    codes = np.empty(g.shape, np.int16)  # int16: n can reach 7 (range [-64,63]), int8 too narrow
    error_feedback = np.zeros((out, ng), np.float32)
    for p in range(GROUP):
        x = g[:, :, p] + error_feedback
        q = np.clip(np.round(x * inv), qmin, qmax).astype(np.float32)
        deq = q * s
        error_feedback = x - deq
        codes[:, :, p] = q.astype(np.int16)
    return codes.reshape(out, inn), s  # s: [out,ng]

# D-qNg64-1: bit-plane packer, must match gguf_quantize_qNg64()'s layout EXACTLY (bias
# FIRST via `+ 2^(n-1)`, THEN bit-extract -- not the other order, see that function's own
# comment for why this matters) -- bitorder='little' is REQUIRED and explicit (np.packbits'
# default is 'big', which would silently produce a different-but-still-self-consistent
# packing that a round-trip test alone could never catch).
def pack_planes(codes, n):
    out, inn = codes.shape
    ng = inn // GROUP
    bias = 1 << (n - 1)
    u = (codes.astype(np.int32) + bias).reshape(out, ng, GROUP)  # [out,ng,64], range [0, 2^n-1]
    planes = np.empty((out, ng, n, 8), np.uint8)
    for j in range(n):
        bits_j = ((u >> j) & 1).astype(np.uint8)
        planes[:, :, j, :] = np.packbits(bits_j, axis=-1, bitorder="little")
    return planes.reshape(out, ng * n * 8)

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
elif mode == "qN":
    n = int(sys.argv[5])
    codes, scales = quant_group_qN(n, w2d)
    planes = pack_planes(codes, n)
    with open(f"{prefix}.planes.bin", "wb") as f: f.write(planes.tobytes())
    with open(f"{prefix}.scales.bin", "wb") as f: f.write(scales.astype("<f4").tobytes())
    print(f"qN n={n} out={out_dim} in={in_dim} ng={in_dim//GROUP} planes_bytes={planes.nbytes} scales_floats={scales.size}")
else:
    raise SystemExit("mode must be q4, q8, or qN")
