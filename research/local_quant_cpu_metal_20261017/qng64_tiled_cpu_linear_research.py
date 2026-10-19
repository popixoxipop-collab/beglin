"""BEGLIN CPU qNg64 GEMM: reusable source-BF16 PyTorch linear research adapter.

Weights are genuine qNg64 n-bit signed bitplanes (64 weights per F32 scale),
encoded by native qng64_dense.c and executed by NEON/Apple Accelerate tiled
C code. Only packed uint8 planes + float32 group scales remain as resident
model weights; one bounded transient F32 tile is used by C GEMM. It does NOT
dequantize/cache a full dense matrix in Python, is NOT a live engine backend,
and does NOT claim GPU/whole-model performance improvements.

Build an isolated dylib from the accompanying research C plus original native:
  clang -O3 -std=c11 -DACCELERATE_NEW_LAPACK -fPIC -dynamiclib \
    -Wall -Wextra -Werror -I. \
    qng64_dense.c research/local_quant_cpu_metal_20261017/\
qng64_tiled_accelerate_portable_poc.c -framework Accelerate -lm \
    -o /tmp/beglin_qng64_tiled_research.dylib
"""
import ctypes
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

__all__=["PackedQNg64TiledLinear"]

class PackedQNg64TiledLinear(nn.Module):
    """Read-only BF16 CPU linear op with packed n-bit qNg64 code+scale payload."""

    def __init__(self, source_weight, n, library_path, *,
                 expected_library_sha256=None, tile_rows=64):
        super().__init__()
        if type(n) is not int or n not in tuple(x for x in range(2,16) if x!=4):
            raise ValueError("CPU BEGLIN qNg64 n is 2..15 excluding q4g64 n4")
        if type(tile_rows) is not int or not 1<=tile_rows<=64:
            raise ValueError("native tiled C supports tile_rows in 1..64")
        if not isinstance(source_weight,torch.Tensor) or (
            source_weight.ndim!=2 or source_weight.device.type!="cpu" or
            source_weight.dtype not in (torch.bfloat16,torch.float32)):
            raise TypeError("source_weight must be rank-two BF16/F32 CPU tensor")
        out,in_features=source_weight.shape
        if (out<1 or in_features<64 or in_features%64 or
            out>2**31-1 or in_features>2**31-1):
            raise ValueError("rank-two qNg64 dimensions/group64 invalid")
        if not bool(torch.isfinite(source_weight).all()):
            raise ValueError("nonfinite source weights")
        path=Path(library_path)
        if not path.is_file():
            raise FileNotFoundError("qNg64 research C dylib unavailable")
        blob_digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_library_sha256 is not None and (
            not isinstance(expected_library_sha256,str) or
            blob_digest.lower()!=expected_library_sha256.lower()):
            raise ValueError("qNg64 library digest mismatch")
        lib=ctypes.CDLL(str(path.resolve()))
        pF=ctypes.POINTER(ctypes.c_float)
        pU=ctypes.POINTER(ctypes.c_uint8)
        for sym in ("qng64_quantize_f32","qng64_packed_bytes",
                    "beglin_qng64_tiled_cblas_matmul"):
            if not hasattr(lib,sym):
                raise RuntimeError("C library lacks required BEGLIN native entrypoint")
        lib.qng64_packed_bytes.argtypes=(ctypes.c_int,ctypes.c_int,ctypes.c_int)
        lib.qng64_packed_bytes.restype=ctypes.c_size_t
        lib.qng64_quantize_f32.argtypes=(pF,ctypes.c_int,ctypes.c_int,ctypes.c_int,pU,pF)
        lib.qng64_quantize_f32.restype=ctypes.c_int
        lib.beglin_qng64_tiled_cblas_matmul.argtypes=(
            pU,pF,ctypes.c_int,pF,pF,pF,
            ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)
        lib.beglin_qng64_tiled_cblas_matmul.restype=ctypes.c_int

        out=int(out)
        in_features=int(in_features)
        expected_code=(out*in_features//64)*(n*8)
        code_bytes=int(lib.qng64_packed_bytes(out,in_features,n))
        if code_bytes!=expected_code:
            raise RuntimeError("C serialized qNg64 packed bytes/shape mismatch")
        group_count=out*(in_features//64)
        raw=np.ascontiguousarray(source_weight.detach().float().numpy(),dtype=np.float32)
        codes=np.empty(code_bytes,dtype=np.uint8)
        scales=np.empty(group_count,dtype=np.float32)
        ok=lib.qng64_quantize_f32(
            raw.ctypes.data_as(pF),out,in_features,n,
            codes.ctypes.data_as(pU),scales.ctypes.data_as(pF))
        if ok!=0 or not np.isfinite(scales).all() or np.any(scales<=0):
            raise RuntimeError("C native qNg64 encoding failed")
        self.n=n
        self.out_features=out
        self.in_features=in_features
        self.tile_rows=tile_rows
        self.serialized_weight_bytes=int(codes.nbytes+scales.nbytes)
        self.code_bytes=int(codes.nbytes)
        self.scale_bytes=int(scales.nbytes)
        self.library_sha256=blob_digest
        self._codes=codes
        self._scales=scales
        self._lib=lib
        self._pF=pF
        self._pU=pU
        # raw is intentionally released at constructor exit: no FP32 dense
        # source weights remain in this packed module.
        del raw

    @torch.inference_mode()
    def forward(self, hidden_states):
        if (not isinstance(hidden_states,torch.Tensor) or
            hidden_states.device.type!="cpu" or
            hidden_states.dtype!=torch.bfloat16 or
            hidden_states.ndim not in (2,3) or
            hidden_states.shape[-1]!=self.in_features):
            raise TypeError("qNg64 packed linear input must be CPU BF16 rank2/3 with correct width")
        x=np.ascontiguousarray(
            hidden_states.detach().float().numpy().reshape(-1,self.in_features),
            dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite BF16 activation")
        batch=x.shape[0]
        if batch==0:raise ValueError("empty qNg64 input batch")
        out=np.empty((batch,self.out_features),dtype=np.float32)
        status=self._lib.beglin_qng64_tiled_cblas_matmul(
            self._codes.ctypes.data_as(self._pU),
            self._scales.ctypes.data_as(self._pF),
            self.n,
            x.ctypes.data_as(self._pF),None,
            out.ctypes.data_as(self._pF),
            self.out_features,self.in_features,batch,self.tile_rows)
        if status!=0 or not np.isfinite(out).all():
            raise RuntimeError("BEGLIN packed native C tiled GEMM returned failure")
        return torch.from_numpy(out.reshape(*hidden_states.shape[:-1],
                                            self.out_features)).to(torch.bfloat16)

    def packed_bytes(self):
        """Exact serialized qNg64 code+one F32 scale per 64 weights, no KV/cache."""
        return self.serialized_weight_bytes

    def evidence(self):
        return {"format":"qng64_g64_ef_v1",
                "bitwidth":self.n,
                "group_size":64,
                "native_library_sha256":self.library_sha256,
                "packed_codes_sha256":hashlib.sha256(self._codes.tobytes()).hexdigest(),
                "scales_sha256":hashlib.sha256(self._scales.tobytes()).hexdigest(),
                "serialized_weight_bytes":self.serialized_weight_bytes,
                "no_dense_resident_model_weight":True,
                "automatic_live_promotion":False}
