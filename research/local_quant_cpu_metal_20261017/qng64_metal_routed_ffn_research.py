"""Packed qNg64 routed-FFN on real MLX GPU, three multi-expert gather stages.

Research engine component; does NOT modify live C++ serving code. Inference:
- all active routed experts' Gate/Up/Down weights remain uint8 bitplanes +
  F32 scales in 3 QNg64MetalExpertBank instances;
- source MoEGate independently computes [T,TOPK] expert indices/weights;
- Gate+Up gather on device; SiLU and BF16 casts on device; Down gather;
- route-weighted sum and optional packed SHARED Gate/Up/Down, BF16 output.

BEGLIN mlx_moe.cpp source uses the same gather kernel with a canonicalized
(token,expert) pair convention. n=7/9-15 existing custom code; n=6 here
requires explicitly experimental bank construction elsewhere.
No BF16 hidden activations or dense dequantized GPU weights persisted.
"""
import numpy as np

from qng64_metal_expert_gather_research import QNg64MetalExpertBank
from qng64_metal_packed_matrix_research import QNg64MetalPackedMatrix

__all__=["QNg64MetalRoutedFFN"]

class QNg64MetalRoutedFFN:
    """Engine-research operator. Calls 3 gather Metal kernels, with optional 3
    shared-expert packed matmuls. All nonlinear/routing arithmetic remains MLX
    device-resident until caller chooses to materialize outputs.
    """

    def __init__(self, gate, up, down, *, shared=None):
        if not all(isinstance(z,QNg64MetalExpertBank) for z in (gate,up,down)):
            raise TypeError("all routed FFN weights must be packed expert banks")
        if gate.experts!=up.experts or gate.experts!=down.experts:
            raise ValueError("expert banks must cover identical compact IDs")
        if gate.in_features!=up.in_features or gate.out!=up.out:
            raise ValueError("Gate and Up projection shapes must match")
        if down.in_features!=gate.out or down.out!=gate.in_features:
            raise ValueError("Down projection incompatible with Gate/Up")
        self.gate=gate
        self.up=up
        self.down=down
        self.T_hidden=gate.in_features
        self.nexperts=gate.experts
        self.intermediate=gate.out
        self.shared=None
        if shared is not None:
            if not isinstance(shared,(tuple,list)) or len(shared)!=3:
                raise TypeError("shared must be packed (gate,up,down) triple")
            sg,su,sd=shared
            if not all(isinstance(z,QNg64MetalPackedMatrix) for z in shared):
                raise TypeError("shared FFN weights must be packed Metal matrices")
            if (sg.in_features!=self.T_hidden or su.in_features!=self.T_hidden or
                sg.out!=su.out or sd.in_features!=sg.out or
                sd.out!=self.T_hidden):
                raise ValueError("shared projection shape incompatible")
            self.shared=(sg,su,sd)

    @staticmethod
    def _silu_bf16(g):
        import mlx.core as mx
        # Torch F.silu(BF16) calculates the nonlinear function in higher
        # precision before rounding back to BF16. Hold explicit boundaries.
        g=mx.astype(g,mx.bfloat16).astype(mx.float32)
        return mx.astype(g*mx.sigmoid(g),mx.bfloat16)

    @staticmethod
    def _mul_silu_gate_up(g,u):
        import mlx.core as mx
        a=QNg64MetalRoutedFFN._silu_bf16(g)
        b=mx.astype(u,mx.bfloat16)
        return mx.astype(a*b,mx.bfloat16)

    def _input(self,x):
        import mlx.core as mx
        if isinstance(x,np.ndarray):
            if (x.ndim!=2 or x.dtype!=np.float32 or not x.flags.c_contiguous or
                x.shape[1]!=self.T_hidden):
                raise TypeError("input numpy must be [T,in] C-contiguous float32")
            return mx.array(x)
        if not isinstance(x,mx.array) or x.ndim!=2 or x.dtype!=mx.float32 or x.shape[1]!=self.T_hidden:
            raise TypeError("input mlx must be [T,in] float32 on MLX device")
        return x

    def forward(self, x, expert_indices, route_weights, residual=None):
        """Return BF16 [T,hidden] MoE output or residual+MoE output.

        Expert indices here are COMPACT indices in [0, E). Caller is responsible
        for source MoEGate original-ID -> compact-ID mapping. The explicit
        float32 route weights must come from the actual model gate, not from
        an invented normalization.
        """
        import mlx.core as mx
        xx=self._input(x)
        T=int(xx.shape[0])
        if T<=0:raise ValueError("empty token batch")
        if (not isinstance(expert_indices,np.ndarray) or
            expert_indices.dtype!=np.int32 or expert_indices.ndim!=2 or
            not expert_indices.flags.c_contiguous or
            expert_indices.shape[0]!=T):
            raise TypeError("expert ids must be contiguous numpy int32 [T,TOPK]")
        K=int(expert_indices.shape[1])
        if K<=0:raise ValueError("empty topK")
        if (np.any(expert_indices<0) or
            np.any(expert_indices>=self.nexperts)):
            raise ValueError("out-of-range expert indices")
        if (not isinstance(route_weights,np.ndarray) or
            route_weights.dtype!=np.float32 or route_weights.shape!=(T,K) or
            not route_weights.flags.c_contiguous):
            raise TypeError("route weights must be contiguous numpy float32 [T,TOPK]")
        if not np.isfinite(route_weights).all() or np.any(route_weights<0):
            raise ValueError("route weights must be finite nonnegative")
        pair_idx=np.ascontiguousarray(expert_indices.reshape(-1))
        pair_input=mx.repeat(xx,K,axis=0)
        g=self.gate.gather(pair_input,pair_idx)
        u=self.up.gather(pair_input,pair_idx)
        intermediate=self._mul_silu_gate_up(g,u)
        d=self.down.gather(intermediate.astype(mx.float32),pair_idx)
        d=mx.astype(d,mx.bfloat16)
        # Original DeepSeek source: BF16 per expert, then cast to router F32,
        # multiply topK weights and sum in F32, cast back to BF16.
        wr=mx.array(route_weights)
        routed=mx.sum(d.astype(mx.float32).reshape(T,K,self.T_hidden) *
                      wr.reshape(T,K,1),axis=1).astype(mx.bfloat16)
        if self.shared is not None:
            sg,su,sd=self.shared
            gs=sg.matmul(xx)
            us=su.matmul(xx)
            zs=self._mul_silu_gate_up(gs,us)
            shared_out=sd.matmul(zs.astype(mx.float32)).astype(mx.bfloat16)
            routed=mx.astype(routed+shared_out,mx.bfloat16)
        if residual is not None:
            residual_x=self._input(residual)
            if residual_x.shape!=xx.shape:
                raise ValueError("residual shape must match hidden input")
            routed=mx.astype(residual_x.astype(mx.bfloat16)+routed,mx.bfloat16)
        return routed
