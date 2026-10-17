"""BEGLIN qNg64 multi-expert gather GEMV directly on Apple MLX Metal.

Matches the original engine mlx_moe.cpp qng64_gather_gemv_kernel:
  planes   [E,out,(in/64)*n*8] raw uint8 bitplanes
  scales   [E,out,in/64] float32
  X        [N,in] float32
  expert_idx [N] int32
  output   [N,out] float32

One Metal dispatch evaluates N arbitrary token-expert pairs; weights stay
compressed on GPU. n=7/9..15 (engine's qNg64 custom Metal path). n=6 can be
used only in explicitly experimental mode; current engine C++ native dispatch
normally repacks n6 into MLX-native quantized_matmul.

No model files, no network, no production runtime connection.
"""
import numpy as np

SOURCE_ENGINE_MLX_MOE_GIT_BLOB="b6cbc1c98cfedcd0f3c99f437d1f6c5978f2ce5a"

GATHER_SOURCE = r"""
    uint p = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint z = thread_position_in_grid.z;
    if (p >= 64) return;
    int e = expert_idx[z];
    int bias_code = 1 << (n - 1);
    float partial = 0.0f;
    for (uint g = 0; g < ng; g++) {
        uint plane_base = (uint)e * ((uint)out_dim * ng * (uint)n * 8)
                         + row * (ng * (uint)n * 8) + g * (uint)n * 8;
        uint byte_in_plane = p >> 3;
        uint bit_in_byte = p & 7;
        int u = 0;
        for (int j = 0; j < n; j++) {
            uint8_t byte = planes[plane_base + (uint)j * 8 + byte_in_plane];
            int bit = (byte >> bit_in_byte) & 1;
            u |= (bit << j);
        }
        int code = u - bias_code;
        uint scale_base = (uint)e * ((uint)out_dim * ng) + row * ng + g;
        float scale = scales[scale_base];
        float decoded = (float)code * scale;
        partial += decoded * x[z * (ng * 64u) + g * 64 + p];
    }
    threadgroup float shared_sums[2];
    uint simd_lane = p % 32;
    uint simd_group = p / 32;
    float simd_partial = simd_sum(partial);
    if (simd_lane == 0) shared_sums[simd_group] = simd_partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p == 0) {
        out[z * (uint)out_dim + row] = shared_sums[0] + shared_sums[1];
    }
"""
_KERNEL=None

def _kernel():
    import mlx.core as mx
    global _KERNEL
    if "gpu" not in str(mx.default_device()):
        raise RuntimeError("real Apple MLX GPU is required")
    if _KERNEL is None:
        _KERNEL=mx.fast.metal_kernel(
            name="beglin_p18_qng64_expert_gather_research_v1",
            input_names=["planes","scales","x","expert_idx"],
            output_names=["out"],
            source=GATHER_SOURCE)
    return _KERNEL

class QNg64MetalExpertBank:
    """Validated, immutable shape/type GPU-resident bitplane expert bank."""

    def __init__(self, planes, scales, n, experts, out, in_features, *,
                 experimental_n6=False):
        import mlx.core as mx
        if type(n) is not int or (
            n not in (7,9,10,11,12,13,14,15) and
            not (n==6 and experimental_n6)):
            raise ValueError("unsupported BEGLIN custom Metal qNg64 bitwidth")
        if (any(type(x) is not int for x in (experts,out,in_features)) or
            experts<=0 or out<=0 or in_features<=0 or in_features%64):
            raise ValueError("invalid expert bank shape or group alignment")
        if (not isinstance(planes,np.ndarray) or planes.dtype!=np.uint8 or
            planes.ndim!=1 or not planes.flags.c_contiguous):
            raise TypeError("planes must be C-contiguous uint8 vector")
        if (not isinstance(scales,np.ndarray) or scales.dtype!=np.float32 or
            scales.ndim!=1 or not scales.flags.c_contiguous):
            raise TypeError("scales must be C-contiguous float32 vector")
        ng=in_features//64
        if planes.size!=experts*out*ng*n*8 or scales.size!=experts*out*ng:
            raise ValueError("packed bank byteplanes/group scale size mismatch")
        if not np.isfinite(scales).all() or np.any(scales<=0):
            raise ValueError("invalid finite positive scale required")
        self.n=n
        self.experts=experts
        self.out=out
        self.in_features=in_features
        self.groups=ng
        self.serialized_bytes=planes.nbytes+scales.nbytes
        self._kernel=_kernel()
        self._planes_gpu=mx.array(planes)
        self._scales_gpu=mx.array(scales)
        mx.eval(self._planes_gpu,self._scales_gpu)

    def gather(self, X, expert_idx):
        """One actual Metal kernel dispatch for all specified token/expert pairs.

        Requires host np int32 IDs for fail-closed index range checking before
        the device kernel dereferences raw weights. X may be host np float32
        or an already resident MLX float32 [N,in] matrix.
        """
        import mlx.core as mx
        if (not isinstance(expert_idx,np.ndarray) or
            expert_idx.dtype!=np.int32 or expert_idx.ndim!=1 or
            not expert_idx.flags.c_contiguous):
            raise TypeError("expert_idx must be C-contiguous 1D numpy int32")
        N=expert_idx.size
        if N<1 or np.any(expert_idx<0) or np.any(expert_idx>=self.experts):
            raise ValueError("expert_idx out of range")
        if isinstance(X,np.ndarray):
            if (X.ndim!=2 or X.dtype!=np.float32 or
                not X.flags.c_contiguous or X.shape!=(N,self.in_features)):
                raise TypeError("input must be C-contiguous float32 [N,in]")
            X_gpu=mx.array(X)
        else:
            if (not isinstance(X,mx.array) or X.ndim!=2 or
                X.dtype!=mx.float32 or X.shape!=(N,self.in_features)):
                raise TypeError("GPU input must be MLX float32 [N,in]")
            X_gpu=X
        index_gpu=mx.array(expert_idx)
        return self._kernel(
            inputs=[self._planes_gpu,self._scales_gpu,X_gpu,index_gpu],
            template=[("n",self.n),("ng",self.groups),("out_dim",self.out)],
            grid=(64,self.out,N),
            threadgroup=(64,1,1),
            output_shapes=[(N,self.out)],
            output_dtypes=[mx.float32],
        )[0]
