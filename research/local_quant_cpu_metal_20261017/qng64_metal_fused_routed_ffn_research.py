"""BEGLIN fused BF16 arithmetic around packed qNg64 MoE Metal gather.

Experimental, GPU device-resident research module (not C++ live serving).
- same routed Gate/Up/Down packed bitplane banks and shared packed matrices
- custom Metal kernel 1: BF16-round(Gate), BF16-round(Up), SiLU + BF16
  boundaries, Gate*Up -> BF16 represented in F32 for down gather
- custom Metal kernel 2: TopK weighted router reduction + BF16-rounded
  shared MoE output + BF16 residual, producing BF16-valued F32 output
- no host round-trip between the three routed expert gather stages
- no dense GPU weight arrays.

The reference DeepSeek BF16 source computes source MLP SiLU on bfloat16,
gate/up multiply on bfloat16, down output cast to bfloat16, route weighted
sum in float32, result cast to bfloat16, shared addition cast to bfloat16,
and residual addition cast to bfloat16. Metal explicitly rounds F32 to BF16
with RNE to match those boundaries. Floating-point math implementations may
still differ: validate against the CPU reference and previous packed MLX
path with explicit tolerances; never claim bitwise equivalence unless shown.
"""
import numpy as np
from qng64_metal_routed_ffn_research import QNg64MetalRoutedFFN

ROUND_BF16_METAL = r"""
    inline float beglin_bf16(float x) {
        uint w = as_type<uint>(x);
        uint lsb = (w >> 16) & 1u;
        uint rounded = (w + 0x7fffu + lsb) & 0xffff0000u;
        return as_type<float>(rounded);
    }
"""
FUSED_SILU_MULT_SOURCE = r"""
    uint idx=thread_position_in_grid.x;
    if (idx >= (uint)elements) return;
    float gv=beglin_bf16(gate[idx]);
    float uv=beglin_bf16(up[idx]);
    float silu=gv / (1.0f + metal::exp(-gv));
    float activated=beglin_bf16(silu);
    out[idx]=beglin_bf16(activated*uv);
"""
FUSED_ROUTED_COMBINE_SOURCE = r"""
    uint idx=thread_position_in_grid.x;
    uint total=(uint)tokens*(uint)hidden;
    if (idx>=total) return;
    uint t=idx/(uint)hidden;
    uint h=idx%(uint)hidden;
    float acc=0.0f;
    for (uint k=0;k<(uint)topk;k++) {
        uint offset=(t*(uint)topk+k)*(uint)hidden+h;
        float value=beglin_bf16(down[offset]);
        acc += value*route_w[t*(uint)topk+k];
    }
    float weighted=beglin_bf16(acc);
    float shared_part=beglin_bf16(shared_out[idx]);
    float result=beglin_bf16(weighted+shared_part);
    float resid=beglin_bf16(residual[idx]);
    out[idx]=beglin_bf16(resid+result);
"""
_FAST_SILU=None
_FAST_REDUCE=None

def _kernels():
    import mlx.core as mx
    global _FAST_SILU,_FAST_REDUCE
    if "gpu" not in str(mx.default_device()):
        raise RuntimeError("real Apple MLX GPU required")
    if _FAST_SILU is None:
        _FAST_SILU=mx.fast.metal_kernel(
            name="beglin_p18_moe_fused_silu_bf16_v1",
            input_names=["gate","up"],output_names=["out"],
            source=FUSED_SILU_MULT_SOURCE,header=ROUND_BF16_METAL)
    if _FAST_REDUCE is None:
        _FAST_REDUCE=mx.fast.metal_kernel(
            name="beglin_p18_moe_fused_route_shared_residual_bf16_v1",
            input_names=["down","route_w","shared_out","residual"],
            output_names=["out"],
            source=FUSED_ROUTED_COMBINE_SOURCE,header=ROUND_BF16_METAL)
    return _FAST_SILU,_FAST_REDUCE

class QNg64MetalFusedRoutedFFN(QNg64MetalRoutedFFN):
    """Exactly 3 routed expert gather calls, 3 packed shared matmul calls,
    2 BF16 fused SiLU calls, 1 fused TopK/Shared/Residual reduction.
    """

    def __init__(self, gate, up, down, *, shared):
        super().__init__(gate,up,down,shared=shared)
        if self.shared is None:
            raise ValueError("fused pipeline requires packed shared FFN")
        self._silu_mul,self._reduce=_kernels()

    @staticmethod
    def _validated_pair_routes(T,indices,weights,experts):
        if not isinstance(indices,np.ndarray) or (
            indices.dtype!=np.int32 or indices.ndim!=2 or
            not indices.flags.c_contiguous or indices.shape[0]!=T):
            raise TypeError("compact expert IDs must be contiguous int32 [T,TOPK]")
        K=indices.shape[1]
        if K<=0 or np.any(indices<0) or np.any(indices>=experts):
            raise ValueError("invalid or out of range compact expert IDs")
        if not isinstance(weights,np.ndarray) or (
            weights.dtype!=np.float32 or weights.shape!=(T,K) or
            not weights.flags.c_contiguous):
            raise TypeError("router weights must be contiguous float32 [T,TOPK]")
        if np.any(weights<0) or not np.isfinite(weights).all():
            raise ValueError("route weights must be finite and nonnegative")
        return K

    def _silu(self,gate,up):
        import mlx.core as mx
        if gate.shape!=up.shape or gate.dtype!=mx.float32 or up.dtype!=mx.float32:
            raise TypeError("Gate and Up must be same-shape float32 packed GEMM outputs")
        return self._silu_mul(
            inputs=[gate,up],template=[("elements",int(gate.size))],
            grid=(int(gate.size),1,1),
            threadgroup=(256,1,1),
            output_shapes=[gate.shape],output_dtypes=[mx.float32])[0]

    def forward(self,x,expert_indices,route_weights,residual):
        import mlx.core as mx
        xx=self._input(x)
        resid=self._input(residual)
        if xx.shape!=resid.shape or xx.shape[0]<1:
            raise ValueError("residual and hidden [T,in] shapes must match/nonempty")
        T=int(xx.shape[0])
        K=self._validated_pair_routes(T,expert_indices,route_weights,self.nexperts)
        pairs=mx.repeat(xx,K,axis=0)
        ids=np.ascontiguousarray(expert_indices.reshape(-1))
        gate=self.gate.gather(pairs,ids)
        up=self.up.gather(pairs,ids)
        down_input=self._silu(gate,up)
        routed_down=self.down.gather(down_input,ids)
        sg,su,sd=self.shared
        sgate=sg.matmul(xx)
        sup=su.matmul(xx)
        shared_input=self._silu(sgate,sup)
        shared_down=sd.matmul(shared_input)
        weights=mx.array(route_weights)
        out=self._reduce(
            inputs=[routed_down,weights,shared_down,resid],
            template=[("tokens",T),("topk",K),("hidden",self.T_hidden)],
            grid=(T*self.T_hidden,1,1),
            threadgroup=(256,1,1),
            output_shapes=[(T,self.T_hidden)],
            output_dtypes=[mx.float32])[0]
        return out
