// Scratch-only MLX 0.32.1 forward-mode adapters for JE01 research.
// The primal operations remain native compressed gather_qmm and native scatter.
// No installed MLX files, parameter arrays, or backend source are modified.
#pragma once

#include "mlx/mlx.h"
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

namespace je01_native_jvp {
namespace mx = mlx::core;
using Arrays = std::vector<mx::array>;
using ArrayFn = std::function<Arrays(const Arrays&)>;
using JvpFn = std::function<Arrays(const Arrays&,const Arrays&,const std::vector<int>&)>;

inline void need(bool condition,const char *message) {
    if(!condition)throw std::runtime_error(message);
}

inline ArrayFn jvp_only_custom(const ArrayFn &primal,const JvpFn &tangent) {
    // The stopped inner primal is deliberately unusable as a default reverse
    // or batching rule. Fail explicitly instead of silently returning zero
    // derivatives when an untested transformation composition is attempted.
    std::function<Arrays(const Arrays&,const Arrays&,const Arrays&)> reject_vjp =
        [](const Arrays&,const Arrays&,const Arrays&) -> Arrays {
            throw std::runtime_error("JE01_JVP_ADAPTER_REVERSE_MODE_NOT_SUPPORTED");
        };
    std::function<std::pair<Arrays,std::vector<int>>(const Arrays&,const std::vector<int>&)> reject_vmap =
        [](const Arrays&,const std::vector<int>&) -> std::pair<Arrays,std::vector<int>> {
            throw std::runtime_error("JE01_JVP_ADAPTER_DIRECT_VMAP_NOT_SUPPORTED");
        };
    auto wrapped=mx::custom_function(primal,reject_vjp,tangent,reject_vmap);
    return [wrapped](const Arrays &inputs) {
        // MLX 0.32.1 direct vmap calls the function first with graphless tracer
        // placeholders. Its later graph walk can enter GatherQMM::vmap before
        // reaching CustomTransforms::vmap. Refuse that entry before constructing
        // the inner native primal. jvp/vjp use Copy tracers with a primitive.
        // Only public array queries are used; no tracing state is changed.
        // Arbitrary nested transformation compositions remain unsupported.
        for(const auto &input:inputs)
            if(input.is_tracer() && !input.has_primitive())
                throw std::runtime_error("JE01_JVP_ADAPTER_DIRECT_VMAP_NOT_SUPPORTED");
        return wrapped(inputs);
    };
}

// Quantized weights, scales, biases, and discrete routing indices are frozen.
// This adapter differentiates only the floating-point activation input.
// Route-boundary and finite-difference checks remain caller responsibilities.
inline ArrayFn frozen_gather_qmm(const mx::array &packed,const mx::array &scales,
    const std::optional<mx::array> &biases,const std::optional<mx::array> &lhs_indices,
    const std::optional<mx::array> &rhs_indices,int group_size,int bits,
    bool transpose=true,bool sorted_indices=false) {
    need(group_size==64 && (bits==4 || bits==5),"JE01_JVP_GATHER_NATIVE_Q4_Q5_GROUP64_ONLY");
    auto lhs=lhs_indices ? std::optional<mx::array>(mx::stop_gradient(*lhs_indices)) : std::nullopt;
    auto rhs=rhs_indices ? std::optional<mx::array>(mx::stop_gradient(*rhs_indices)) : std::nullopt;
    auto call=[=](const mx::array &x) {
        return mx::gather_qmm(x,packed,scales,biases,lhs,rhs,transpose,
                             group_size,bits,"affine",sorted_indices);
    };
    ArrayFn primal=[call](const Arrays &inputs) {
        need(inputs.size()==1,"JE01_JVP_GATHER_ONE_ACTIVATION_INPUT_REQUIRED");
        // In MLX 0.32.1 JVP traversal visits a stopped output's producer first.
        // Stop the inner primal INPUT as well so GatherQMM::jvp is not entered;
        // CustomTransforms still receives the original, differentiable args.
        return Arrays{call(mx::stop_gradient(inputs[0]))};
    };
    JvpFn tangent=[call](const Arrays &primals,const Arrays &tangents,
                         const std::vector<int> &argnums) {
        need(primals.size()==1 && tangents.size()==argnums.size(),
             "JE01_JVP_GATHER_TANGENT_SIGNATURE");
        auto dx=mx::zeros_like(primals[0]);
        for(size_t j=0;j<argnums.size();j++) {
            need(argnums[j]==0,"JE01_JVP_GATHER_FROZEN_WEIGHT_OR_INDEX_DERIVATIVE_FORBIDDEN");
            dx=tangents[j];
        }
        return Arrays{call(dx)};
    };
    return jvp_only_custom(primal,tangent);
}

// Native B1 MLA cache update: cache {1,H,L,D}, updates {1,1,H,1,D}.
// One (slot,position) pair means update coordinates cannot overlap.
// The caller must already have checked slot==0 and 0<=position<L, as the
// native cbatch entry point does. No host evaluation occurs in this adapter.
inline ArrayFn fixed_b1_kv_scatter(const mx::array &slot,const mx::array &position) {
    need(slot.ndim()==1 && position.ndim()==1 && slot.size()==1 && position.size()==1 &&
         slot.dtype()==mx::int32 && position.dtype()==mx::int32,
         "JE01_JVP_SCATTER_ONE_FIXED_INT32_COORDINATE_REQUIRED");
    auto s=mx::stop_gradient(slot),p=mx::stop_gradient(position);
    auto call=[s,p](const mx::array &cache,const mx::array &updates) {
        need(cache.ndim()==4 && cache.shape(0)==1 && cache.dtype()==mx::float32 &&
             updates.ndim()==5 && updates.shape(0)==1 && updates.shape(1)==1 &&
             updates.shape(2)==cache.shape(1) && updates.shape(3)==1 &&
             updates.shape(4)==cache.shape(3) && updates.dtype()==mx::float32,
             "JE01_JVP_SCATTER_NATIVE_B1_WINDOW_SHAPE_REQUIRED");
        return mx::scatter(cache,{s,p},updates,{0,2});
    };
    ArrayFn primal=[call](const Arrays &inputs) {
        need(inputs.size()==2,"JE01_JVP_SCATTER_CACHE_AND_UPDATE_REQUIRED");
        return Arrays{call(mx::stop_gradient(inputs[0]),mx::stop_gradient(inputs[1]))};
    };
    JvpFn tangent=[call](const Arrays &primals,const Arrays &tangents,
                         const std::vector<int> &argnums) {
        need(primals.size()==2 && tangents.size()==argnums.size(),
             "JE01_JVP_SCATTER_TANGENT_SIGNATURE");
        auto dcache=mx::zeros_like(primals[0]);
        auto dnew=mx::zeros_like(primals[1]);
        for(size_t j=0;j<argnums.size();j++) {
            if(argnums[j]==0)dcache=tangents[j];
            else if(argnums[j]==1)dnew=tangents[j];
            else throw std::runtime_error("JE01_JVP_SCATTER_INDEX_DERIVATIVE_FORBIDDEN");
        }
        // A replaced cache location receives dnew, not dcache+dnew. Missing
        // update tangents therefore zero derivatives at overwritten positions.
        return Arrays{call(dcache,dnew)};
    };
    return jvp_only_custom(primal,tangent);
}
} // namespace je01_native_jvp
