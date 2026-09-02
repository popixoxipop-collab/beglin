#!/usr/bin/env python3
# ROI-G Phase 1: offline software simulation of arbitrary-n group-64 symmetric
# RTN quantization, stdlib-only (no numpy/safetensors package -- neither is
# installed locally, and this project's own established preference is zero
# new dependencies, e.g. d4_supabase_push.py's urllib-only design).
#
# Reads one real tensor's BF16 bytes straight out of a real safetensors
# checkpoint (single file or sharded via *.index.json, same layout this
# project's C loader (safetensors_load.c) already parses), quantizes+
# dequantizes it in software at an arbitrary bit-width n using the SAME
# group-64/scale=maxabs/qmax convention as gguf_transcode.c's own
# gguf_quantize_q4g64_error_feedback() (qmax=2^(n-1)-1, clamp
# [-2^(n-1), 2^(n-1)-1]) -- generalized to arbitrary n, PLAIN round-to-
# nearest-even (no error-feedback residual; Phase 1 only needs "good
# enough" quantization to locate the accuracy knee, not a bit-exact match
# to a future real packed format -- see the Opus plan's own note on this).
#
# Writes each simulated result back out as a minimal single-tensor F32
# safetensors file (tensor name "sim") -- safetensors_load.c's own parser
# (confirmed by direct read: 8-byte header length + JSON header, no
# required __metadata__, no alignment padding) accepts this with zero
# changes, and qwen_infer.c's st_register_moe_f32_as_af() already loads
# any F32 tensor with zero additional precision loss beyond what's baked
# into these bytes.

import argparse
import array
import json
import math
import os
import struct
import sys

GROUP = 64


def bf16_bytes_to_f32_array(raw: bytes) -> array.array:
    # BF16 is exactly the top 16 bits of an F32 (same sign/exponent, truncated
    # mantissa) -- widening is a plain left-shift by 16, zero library needed.
    n = len(raw) // 2
    u16 = array.array("H")
    u16.frombytes(raw)
    if sys.byteorder != "little":
        u16.byteswap()
    u32 = array.array("I", (v << 16 for v in u16))
    out = array.array("f")
    out.frombytes(u32.tobytes())
    return out


def f32_array_to_bytes(a: array.array) -> bytes:
    b = array.array("f", a)
    if sys.byteorder != "little":
        b.byteswap()
    return b.tobytes()


def read_safetensors_header(path: str):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    return header, 8 + hlen


def resolve_shard(checkpoint_path: str, tensor_name: str):
    """checkpoint_path is either a *.index.json (sharded) or a single
    *.safetensors file. Returns (shard_file_path, dtype, shape, byte_offsets)."""
    if checkpoint_path.endswith(".index.json"):
        with open(checkpoint_path, "r") as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
        if tensor_name not in weight_map:
            raise SystemExit(f"tensor '{tensor_name}' not in weight_map of {checkpoint_path}")
        shard_basename = weight_map[tensor_name]
        shard_path = os.path.join(os.path.dirname(checkpoint_path), shard_basename)
    else:
        shard_path = checkpoint_path
    header, data_start = read_safetensors_header(shard_path)
    if tensor_name not in header:
        raise SystemExit(f"tensor '{tensor_name}' not found in shard {shard_path}")
    info = header[tensor_name]
    return shard_path, data_start, info["dtype"], info["shape"], info["data_offsets"]


def read_tensor_f32(checkpoint_path: str, tensor_name: str):
    shard_path, data_start, dtype, shape, offsets = resolve_shard(checkpoint_path, tensor_name)
    with open(shard_path, "rb") as f:
        f.seek(data_start + offsets[0])
        raw = f.read(offsets[1] - offsets[0])
    if dtype == "BF16":
        vals = bf16_bytes_to_f32_array(raw)
    elif dtype == "F32":
        vals = array.array("f")
        vals.frombytes(raw)
        if sys.byteorder != "little":
            vals.byteswap()
    else:
        raise SystemExit(f"unsupported source dtype {dtype} for '{tensor_name}' (only BF16/F32 wired up)")
    if len(shape) != 2:
        raise SystemExit(f"expected a 2D weight tensor, got shape {shape} for '{tensor_name}'")
    return vals, shape[0], shape[1]


def quantize_dequantize_n(vals: array.array, out: int, in_: int, n: int) -> array.array:
    """Group-64 symmetric RTN, generalized from gguf_quantize_q4g64_error_feedback()'s
    own scale/clamp convention (qmax=2^(n-1)-1), plain round-to-nearest-even
    (Python's round() on floats is already round-half-to-even, matching this
    project's own rintf()-not-roundf() finding for the same reason)."""
    if in_ % GROUP != 0:
        raise SystemExit(f"in={in_} not a multiple of GROUP={GROUP}")
    qmax = float((1 << (n - 1)) - 1)
    qmin = float(-(1 << (n - 1)))
    ng = in_ // GROUP
    result = array.array("f", [0.0]) * (out * in_)
    for r in range(out):
        row_off = r * in_
        for g in range(ng):
            base = row_off + g * GROUP
            grp = vals[base:base + GROUP]
            maxabs = max(abs(x) for x in grp)
            scale = maxabs / qmax if maxabs > 1e-12 else 1.0
            inv = 1.0 / scale
            for p in range(GROUP):
                x = grp[p]
                q = round(x * inv)  # Python round() is round-half-to-even
                if q > qmax:
                    q = qmax
                elif q < qmin:
                    q = qmin
                result[base + p] = q * scale
    return result


def rel_l2(a: array.array, b: array.array) -> float:
    num = 0.0
    den = 0.0
    for x, y in zip(a, b):
        d = x - y
        num += d * d
        den += x * x
    return math.sqrt(num / den) if den > 0 else 0.0


def write_single_tensor_safetensors(path: str, tensor_name: str, shape, vals: array.array):
    data = f32_array_to_bytes(vals)
    header = {
        tensor_name: {"dtype": "F32", "shape": list(shape), "data_offsets": [0, len(data)]},
        "__metadata__": {"source": "quant_sim_n.py"},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    # safetensors convention: header padded so data starts 8-byte aligned --
    # not required by this project's own parser (no alignment check found in
    # safetensors_load.c), but harmless and keeps the file spec-conformant
    # for any other tool that might read it.
    pad = (-(len(header_bytes)) - 8) % 8
    header_bytes += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="path to *.index.json or a single *.safetensors file")
    ap.add_argument("--tensor", required=True, help="real tensor name inside the checkpoint")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, nargs="+", required=True, help="bit-widths to simulate, e.g. 2 3 4 ... 16")
    ap.add_argument("--passthrough", action="store_true",
                     help="also write a zero-quantization pure-F32 passthrough file (pipeline sanity check)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[quant_sim_n] reading '{args.tensor}' from {args.checkpoint}", file=sys.stderr)
    vals, out, in_ = read_tensor_f32(args.checkpoint, args.tensor)
    print(f"[quant_sim_n] shape=({out},{in_}) n_elements={out*in_}", file=sys.stderr)

    if args.passthrough:
        p = os.path.join(args.out_dir, "sim_passthrough.safetensors")
        write_single_tensor_safetensors(p, "sim", (out, in_), vals)
        print(f"[quant_sim_n] passthrough -> {p} (rel_l2=0.0 by construction)", file=sys.stderr)

    results = []
    for n in args.n:
        qd = quantize_dequantize_n(vals, out, in_, n)
        err = rel_l2(vals, qd)
        eff_bpw = n + 32.0 / (GROUP * n)  # same convention as q4g64's own 4+32/64=4.5
        out_path = os.path.join(args.out_dir, f"sim_n{n}.safetensors")
        write_single_tensor_safetensors(out_path, "sim", (out, in_), qd)
        results.append({"n": n, "effective_bpw": eff_bpw, "rel_l2": err, "path": out_path})
        print(f"[quant_sim_n] n={n:2d} eff_bpw={eff_bpw:.3f} rel_l2={err:.6e} -> {out_path}", file=sys.stderr)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"tensor": args.tensor, "shape": [out, in_], "results": results}, f, indent=2)
    print(f"[quant_sim_n] summary -> {summary_path}")


if __name__ == "__main__":
    main()
