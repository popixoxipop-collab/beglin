"""BEGLIN qNg64 packed bitplane MLX Metal research adapter.

This mirrors the two-level 64-thread SIMD-group reduction of the existing
mlx_moe.cpp::qng64_gemv_kernel (research Git blob
b6cbc1c98cfedcd0f3c99f437d1f6c5978f2ce5a).
- Packed uint8 planes + group-wise F32 scales remain RESIDENT on GPU.
- Does NOT upload dense dequantized weights or use a dense fallback.
- n=7 and n=9..15 mirror C++ custom Metal support.
- n=6 may be enabled explicitly as EXPERIMENTAL. Existing BEGLIN C++ uses
  a different MLX-native quantized_matmul/repack path for n=6.
- This utility is not wired to live serving, not full model GPU decode.
"""
import math

__all__ = ["QNg64MetalPackedMatrix", "MODEL_KERNEL_SOURCE_BLOB_SHA256"]
MODEL_KERNEL_SOURCE_BLOB_SHA256 = "b6cbc1c98cfedcd0f3c99f437d1f6c5978f2ce5a"
METAL_SOURCE = r"""
    uint p = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint z = thread_position_in_grid.z;
    if (p >= 64) return;
    int bias_code = 1 << (n - 1);
    float partial = 0.0f;
    for (uint g = 0; g < ng; g++) {
        uint plane_base = row * (ng * (uint)n * 8) + g * (uint)n * 8;
        uint byte_in_plane = p >> 3;
        uint bit_in_byte = p & 7;
        int u = 0;
        for (int j = 0; j < n; j++) {
            uint8_t byte = planes[plane_base + (uint)j * 8 + byte_in_plane];
            int bit = (byte >> bit_in_byte) & 1;
            u |= (bit << j);
        }
        int code = u - bias_code;
        float scale = scales[row * ng + g];
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
_kernel = None

def _get_gpu_kernel():
    import mlx.core as mx
    if "gpu" not in str(mx.default_device()):
        raise RuntimeError("qNg64 Metal packed execution requires MLX Apple GPU")
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="beglin_qng64_metal_two_simdgroup_research",
            input_names=["planes", "scales", "x"],
            output_names=["out"],
            source=METAL_SOURCE)
    return _kernel

class QNg64MetalPackedMatrix:
    """An immutable GPU-resident qNg64 [out,in] packed matrix.

    This wrapper only implements GEMM from already-quantized group-64 code
    planes. Callers MUST use a matching qNg64 format, including signed code
    offset 2^(n-1) and per-64-weight F32 scale. All checks are fail-closed.
    """
    def __init__(self, packed, scales, n, out, in_features, *,
                 experimental_n6=False):
        import numpy as np
        import mlx.core as mx
        if type(n) is not int or (
            n not in (7,9,10,11,12,13,14,15) and
            not (n==6 and experimental_n6)):
            raise ValueError("unsupported custom Metal qNg64 bits")
        if type(out) is not int or type(in_features) is not int or (
            out<=0 or in_features<=0 or in_features%64):
            raise ValueError("invalid packed matrix shape/group alignment")
        if not isinstance(packed,np.ndarray) or packed.dtype!=np.uint8 or (
            packed.ndim!=1 or not packed.flags.c_contiguous):
            raise TypeError("planes must be contiguous 1D numpy uint8")
        if not isinstance(scales,np.ndarray) or scales.dtype!=np.float32 or (
            scales.ndim!=1 or not scales.flags.c_contiguous):
            raise TypeError("scales must be contiguous 1D numpy float32")
        self.n=n
        self.out=out
        self.in_features=in_features
        self.ng=in_features//64
        self.serialized_bytes=(out*self.ng)*(n*8+4)
        if packed.size!=out*self.ng*n*8 or scales.size!=out*self.ng:
            raise ValueError("encoded plane/scale size mismatch")
        if not np.isfinite(scales).all() or not (scales>0).all():
            raise ValueError("nonfinite/nonpositive group scales")
        self._kernel=_get_gpu_kernel()
        self.planes_gpu=mx.array(packed)
        self.scales_gpu=mx.array(scales)
        mx.eval(self.planes_gpu,self.scales_gpu)
        # NO dense matrix of any dtype exists in GPU representation.

    def matmul(self, x):
        import numpy as np
        import mlx.core as mx
        if isinstance(x,np.ndarray):
            if x.dtype!=np.float32 or x.ndim!=2 or (
                not x.flags.c_contiguous or x.shape[1]!=self.in_features):
                raise TypeError("CPU activation must be C-contiguous [M,in] F32")
            xin=mx.array(x)
        else:
            if not isinstance(x,mx.array) or x.dtype!=mx.float32 or (
                x.ndim!=2 or x.shape[1]!=self.in_features):
                raise TypeError("MLX activation must be [M,in] float32")
            xin=x
        m=xin.shape[0]
        if m<1:
            raise ValueError("empty activation batch")
        output=self._kernel(
            inputs=[self.planes_gpu,self.scales_gpu,xin],
            template=[("n",self.n),("ng",self.ng),("out_dim",self.out)],
            grid=(64,self.out,m),
            threadgroup=(64,1,1),
            output_shapes=[(m,self.out)],
            output_dtypes=[mx.float32])[0]
        return output
