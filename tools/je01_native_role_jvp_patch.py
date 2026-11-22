#!/usr/bin/env python3
"""Isolated role-selectable full-prefix native JVP at the AF4 baseline.

The pinned fixed-route native source remains the OFF implementation. Each run
selects exactly one of the frozen bank's 24 gate/up/down roles and one Q4/Q5
alternative through the original loader. Hidden/K/V state and alpha propagate
through the complete prefix in forward mode. Only alpha=0 is admitted.

This extends the validated single-Gate mechanism, not its validation claim.
Every newly exercised role/tier requires exact OFF/primal parity and independent
fixed-route finite-difference convergence/agreement before it can be used.
The TRAIN16 counters and original guarded adapters remain unchanged. No full48
Fisher, policy allocation, independent holdout, or production change is implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


CPP_SHA256 = "390fb6b136bdd9f4978e6482b7a18b11951333e453752a15c3fd094a7d3b07f3"
C_SHA256 = "8d1a8fce7d578319ceb45d17948ebe0349758acb2ae0b786944c8119ea795c3d"
ADAPTER_SHA256 = "9ae9d98b921334bae321e583908e71bdc9c08178cbf3ca5d7a1d4d4e552a09ec"
ADAPTER_NAME = "je01_mlx_native_jvp_adapters.h"
MODE = "ROLE_DIRECTION_FIXED_Q4Q5"
GOLD_MAGIC = b"JE01_NATIVE_ONE_COLUMN_JVP_GOLD_V1\n"
PREFIX_TOKENS = 4334
PREFIX_CASE_RESETS = 32
PREFIX_LAYERS = 117018
GOLD_COUNT = 656


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def once(text, old, new, label):
    if text.count(old) != 1:
        raise ValueError("JE01_ONE_JVP_ANCHOR:" + label + ":" + str(text.count(old)))
    return text.replace(old, new, 1)


HELPERS = r'''
// JE01 one selected role/tier per run. Native weights/IDs stay fixed;
// router values do not. The bank loader owns the immutable role index map.
using JE01JvpArrays=je01_native_jvp::Arrays;
using JE01JvpFn=je01_native_jvp::ArrayFn;
static std::vector<mx::array> g_je01_one_dK,g_je01_one_dV;
static std::optional<mx::array> g_je01_one_dhidden,g_je01_one_dlogits;
static uint64_t g_je01_one_tokens=0,g_je01_one_cases=0,g_je01_one_layers=0;
static int g_je01_one_position=-1,g_je01_one_logits_position=-1;

static bool je01_one_jvp_requested(void) {
    const char *mode=std::getenv("JE01_NATIVE_JVP_MODE");
    if(!mode || !*mode)return false;
    if(std::string(mode)!="ROLE_DIRECTION_FIXED_Q4Q5")
        throw std::runtime_error("JE01_ONE_COLUMN_JVP_MODE_FORBIDDEN");
    return true;
}

static bool je01_role_jvp_direction_matches_bank(void) {
    const char *role_env=std::getenv("JE01_DIRECTION_ROLE");
    const char *tier_env=std::getenv("JE01_DIRECTION_TIER");
    if(!role_env || !*role_env || !tier_env)return false;
    // The pinned original loader has already checked the 8-slot bank and
    // selected exactly one alternative. Revalidate the public selector here,
    // including its association to the loaded native role and tensor name.
    char *end=nullptr;
    long selected=std::strtol(role_env,&end,10);
    if(!end || *end || selected<0 || selected>=24 ||
       std::string(role_env)!=std::to_string(selected) ||
       g_je01_direction_layer<1 || g_je01_direction_layer>=27 ||
       g_je01_direction_expert<0 || g_je01_direction_expert>=64 ||
       g_je01_direction_role<0 || g_je01_direction_role>=3 ||
       selected%3!=g_je01_direction_role ||
       (std::strcmp(tier_env,"Q4") && std::strcmp(tier_env,"Q5")))return false;
    static constexpr int slots[8][2]={{12,13},{13,22},{13,28},{14,63},
                                     {20,17},{21,61},{24,26},{24,60}};
    if(g_je01_direction_layer!=slots[selected/3][0] ||
       g_je01_direction_expert!=slots[selected/3][1])return false;
    const char *roles[]={"gate_proj","up_proj","down_proj"};
    char expected[128];
    int written=std::snprintf(expected,sizeof(expected),"je01.dir.L%d.E%d.%s.%s",
        g_je01_direction_layer,g_je01_direction_expert,roles[g_je01_direction_role],tier_env);
    return written>0 && written<(int)sizeof(expected) &&
           g_je01_direction_alternative==expected;
}

static void je01_one_jvp_require_contract(void) {
    if(g_fused_B!=1 || g_mla_n_heads!=16 || g_mla_q_head_dim!=192 ||
       g_mla_v_hd!=128 || MLA_L0_MAXPOS!=384 || g_layer_hidden!=2048 ||
       g_layer_im_dim!=1408 || g_layer_n_experts!=64 || g_layer_top_k!=6 ||
       g_layer_group!=64 || g_gpu_sort_threshold!=64 || g_risk_signal_telemetry_on ||
       !je01_role_jvp_direction_matches_bank() || g_je01_direction_alpha!=0.0f ||
       !je01_direction_optional_path("JE01_DIRECTION_BASELINE_ROUTES") ||
       !je01_direction_optional_path("JE01_DIRECTION_ROUTE_OUTPUT") ||
       !je01_direction_optional_path("JE01_NATIVE_JVP_GOLD_OUTPUT") ||
       je01_direction_optional_path("JE01_DIRECTION_MARGIN_OUTPUT"))
        throw std::runtime_error("JE01_ONE_COLUMN_JVP_CONTRACT_DRIFT");
}

static mx::array je01_one_jvp_gather(const FfnRoleRef &t,const mx::array &x,
    const std::optional<mx::array> &lhs,const mx::array &rhs,bool transpose,bool sorted) {
    if(t.ng64 || t.is_dense || !t.q || t.q->E!=64 || t.q->bits!=4 ||
       g_layer_group!=64 || !transpose || sorted)
        throw std::runtime_error("JE01_ONE_COLUMN_JVP_NATIVE_AF4_E64_ONLY");
    auto fn=je01_native_jvp::frozen_gather_qmm(t.q->w,t.q->scales,t.q->biases,
                                             lhs,rhs,64,4,true,false);
    return fn({x})[0];
}

static mx::array je01_role_jvp_output(int layer,int role,const mx::array &input,
    const mx::array &base_native,const mx::array &indices,const mx::array &alpha) {
    if(layer!=g_je01_direction_layer || role!=g_je01_direction_role)return base_native;
    const int width=role==2?2048:1408;
    const int alternate_rows=role==2?6:1;
    if(role<0 || role>=3 || base_native.size()!=6*width ||
       indices.shape()!=mx::Shape{1,6} || alpha.size()!=1 || alpha.dtype()!=mx::float32 ||
       (role<2 && input.shape()!=mx::Shape{1,2048}) ||
       (role==2 && input.shape()!=mx::Shape{6,1,1408}))
        throw std::runtime_error("JE01_ROLE_JVP_NATIVE_SHAPE");
    const auto shape=base_native.shape();
    auto mask=mx::stop_gradient(mx::reshape(mx::equal(indices,
        mx::array(g_je01_direction_expert,mx::int32)),{6,1}));
    // At alpha=0, d(alpha*(alt(input)-base(input))) is
    // d_alpha*(alt-base); the input derivative of alt is multiplied by zero.
    // Down uses the native six different SwiGLU inputs, not one broadcast row.
    auto alternate=mx::stop_gradient(lazy_matvec_e0(g_je01_direction_alternative.c_str(),input));
    JE01JvpFn primal=[=](const JE01JvpArrays &a) {
        if(a.size()!=3)throw std::runtime_error("JE01_ONE_COLUMN_JVP_GATE_ARGUMENTS");
        auto base=mx::reshape(mx::stop_gradient(a[0]),{6,width});
        auto alt=mx::reshape(mx::stop_gradient(a[1]),{alternate_rows,width});
        auto coordinate=mx::stop_gradient(a[2]);
        auto endpoint=mx::where(mask,alt,base);
        auto interpolation=mx::where(mask,base+coordinate*(alt-base),base);
        // This defines a nonconstant continuous extension, while selecting
        // exact endpoint bytes. No .item() or host eval occurs during tracing.
        auto value=mx::where(mx::equal(coordinate,mx::array(0.0f)),base,
                   mx::where(mx::equal(coordinate,mx::array(1.0f)),endpoint,interpolation));
        return JE01JvpArrays{mx::reshape(value,shape)};
    };
    je01_native_jvp::JvpFn tangent=[=](const JE01JvpArrays &p,const JE01JvpArrays &t,
                                      const std::vector<int> &argnums) {
        if(p.size()!=3 || t.size()!=argnums.size())
            throw std::runtime_error("JE01_ONE_COLUMN_JVP_GATE_TANGENT_ARGUMENTS");
        auto dbase=mx::zeros_like(p[0]),da=mx::zeros_like(p[2]);
        for(size_t j=0;j<argnums.size();++j) {
            if(argnums[j]==0)dbase=t[j];
            else if(argnums[j]==2)da=t[j];
            else throw std::runtime_error("JE01_ONE_COLUMN_JVP_ALT_MUST_BE_STOPPED");
        }
        auto base=mx::reshape(p[0],{6,width}),alt=mx::reshape(p[1],{alternate_rows,width});
        auto residual=mx::where(mask,alt-base,mx::zeros_like(base));
        return JE01JvpArrays{dbase+da*mx::reshape(residual,shape)};
    };
    // Only alpha0 is admitted by this lane. The custom rule is the derivative
    // at that point of the native role interpolation defined above.
    auto fn=je01_native_jvp::jvp_only_custom(primal,tangent);
    return fn({base_native,alternate,alpha})[0];
}
'''


PRE_ATTENTION_PREFIX = r'''
static JE01JvpArrays je01_one_jvp_layer_prefix(int l,int A,const int *slot,const int *spos,
    int is_dense,const float *w_inln,const float *w_postln,const float *w_kvaln,
    const float *w_gate,const JE01JvpArrays &state) {
    if(state.size()!=3)throw std::runtime_error("JE01_ONE_COLUMN_JVP_PREFIX_STATE_REQUIRED");
    const int H=g_mla_n_heads,QHD=g_mla_q_head_dim,NOPE=g_mla_qk_nope,
              ROPE=g_mla_qk_rope,VHD=g_mla_v_hd,KVLORA=g_mla_kv_lora;
    const int HIDDEN=g_layer_hidden,IM=g_layer_im_dim,DENSE_IM=g_layer_dense_im,
              NE=g_layer_n_experts,NS=g_layer_n_shared,TOPK=g_layer_top_k;
    mx::array slot_arr(slot,{A},mx::int32),pos_arr(spos,{A},mx::int32);
    auto cache_k=state[1],cache_v=state[2];
'''


POST_ATTENTION_PREFIX = r'''
static JE01JvpArrays je01_one_jvp_layer_postfix(int l,int A,const int *slot,const int *spos,
    int is_dense,const float *w_postln,const float *w_gate,const JE01JvpArrays &state) {
    if(state.size()!=3)throw std::runtime_error("JE01_ONE_COLUMN_JVP_POSTFIX_STATE_REQUIRED");
    const int H=g_mla_n_heads,VHD=g_mla_v_hd,HIDDEN=g_layer_hidden,IM=g_layer_im_dim,
              DENSE_IM=g_layer_dense_im,NE=g_layer_n_experts,NS=g_layer_n_shared,TOPK=g_layer_top_k;
    auto x=state[0],attn=state[1];
    auto returned_route=mx::zeros({1,6},mx::int32);
    char no[96];std::snprintf(no,sizeof(no),"model.layers.%d.self_attn.o_proj",l);
'''


LAYER_WRAPPER = r'''
int mlx_gpu_cbatch_layer_step_lazy(int l,int A,const int *slot,const int *spos,int is_dense,
    const float *x_in_host,const float *w_inln,const float *w_postln,
    const float *w_kvaln,const float *w_gate) {
    try {
        if(!je01_one_jvp_requested())
            return je01_one_jvp_original_layer(l,A,slot,spos,is_dense,x_in_host,
                                               w_inln,w_postln,w_kvaln,w_gate);
        je01_one_jvp_require_contract();
        if(A!=1 || !slot || !spos || slot[0]!=0 || spos[0]<0 || spos[0]>=384 ||
           l<0 || l>=27 || is_dense!=(l==0))
            throw std::runtime_error("JE01_ONE_COLUMN_JVP_LAYER_COORDINATES");
        ensure_fused_kv_init();
        if(l==0) {
            if(g_cbatch_x || g_cbatch_layers_done!=0 || g_cbatch_A!=0)
                throw std::runtime_error("JE01_ONE_COLUMN_JVP_PREVIOUS_TOKEN_NOT_FINALIZED");
            if(spos[0]==0) {
                g_je01_one_dK.clear();g_je01_one_dV.clear();
                for(const auto &a:g_fused_K)g_je01_one_dK.push_back(mx::zeros_like(a));
                for(const auto &a:g_fused_V)g_je01_one_dV.push_back(mx::zeros_like(a));
                ++g_je01_one_cases;
                g_je01_one_position=-1;
            }
            if(spos[0]!=g_je01_one_position+1 ||
               g_je01_one_dK.size()!=g_fused_K.size() || g_je01_one_dV.size()!=g_fused_V.size())
                throw std::runtime_error("JE01_ONE_COLUMN_JVP_FULL_PREFIX_CONTINUITY");
            g_je01_one_position=spos[0];
            g_cbatch_x=new mx::array(wrap_host_f32(x_in_host,{1,g_layer_hidden}));
            g_je01_one_dhidden=mx::zeros_like(*g_cbatch_x);
            g_je01_one_dlogits.reset();g_je01_one_logits_position=-1;
            g_cbatch_A=1;g_cbatch_layers_done=0;
        }
        if(g_cbatch_A!=1 || g_cbatch_layers_done!=l || !g_cbatch_x ||
           !g_je01_one_dhidden || g_je01_one_position!=spos[0])
            throw std::runtime_error("JE01_ONE_COLUMN_JVP_LAYER_STATE_ORDER");
        JE01JvpFn prefix=[=](const JE01JvpArrays &state) {
            return je01_one_jvp_layer_prefix(l,A,slot,spos,is_dense,
                                               w_inln,w_postln,w_kvaln,w_gate,state);
        };
        auto before=mx::jvp(prefix,{*g_cbatch_x,g_fused_K[l],g_fused_V[l]},
                           {*g_je01_one_dhidden,g_je01_one_dK[l],g_je01_one_dV[l]});
        if(before.first.size()!=6 || before.second.size()!=6)
            throw std::runtime_error("JE01_ONE_COLUMN_JVP_PREFIX_RESULT_ARITY");
        auto mask=mx::stop_gradient(before.first[5]);
        // MLX0.32.1 forces SDPA fallback inside grad tracing. Build the genuine
        // fused native primal HERE, after prefix JVP returned and before the
        // next transformation. Only the derivative uses the analytical fallback.
        auto native_attention=mx::fast::scaled_dot_product_attention(
            before.first[0],before.first[1],before.first[2],
            (float)g_mla_attn_scale,"array",mask);
        JE01JvpFn attention=[=](const JE01JvpArrays &qkv) {
            return JE01JvpArrays{mx::fast::scaled_dot_product_attention(qkv[0],qkv[1],qkv[2],
                                      (float)g_mla_attn_scale,"array",mask)};
        };
        auto attn_derivative=mx::jvp(attention,
            {before.first[0],before.first[1],before.first[2]},
            {before.second[0],before.second[1],before.second[2]});
        JE01JvpFn postfix=[=](const JE01JvpArrays &state) {
            return je01_one_jvp_layer_postfix(l,A,slot,spos,is_dense,w_postln,w_gate,state);
        };
        auto result=mx::jvp(postfix,{*g_cbatch_x,native_attention,mx::array(0.0f)},
                            {*g_je01_one_dhidden,attn_derivative.second[0],mx::array(1.0f)});
        if(result.first.size()!=2 || result.second.size()!=2)
            throw std::runtime_error("JE01_ONE_COLUMN_JVP_POSTFIX_RESULT_ARITY");
        g_fused_K[l]=before.first[3];g_fused_V[l]=before.first[4];
        g_je01_one_dK[l]=before.second[3];g_je01_one_dV[l]=before.second[4];
        delete g_cbatch_x;g_cbatch_x=new mx::array(result.first[0]);
        g_je01_one_dhidden=result.second[0];
        // Route trace receives returned primal arrays after the transformation,
        // never a captured tracer or a repeated side effect inside the function.
        if(l>0)je01_direction_route_trace(l,A,slot,spos,result.first[1]);
        g_cbatch_layers_done=l+1;++g_je01_one_layers;
        return 1;
    } catch(const std::exception &e) {
        std::fprintf(stderr,"JE01_ONE_COLUMN_JVP_LAYER_FAILED layer=%d position=%d reason=%s\n",
                     l,spos?spos[0]:-1,e.what());
        delete g_cbatch_x;g_cbatch_x=nullptr;g_cbatch_A=0;g_cbatch_layers_done=0;
        return 0;
    }
}
'''


FINALIZE_WRAPPER = r'''
int mlx_gpu_cbatch_forward_finalize(const float *w_finalnorm,float *logits_out) {
    try {
        if(!je01_one_jvp_requested())return je01_one_jvp_original_finalize(w_finalnorm,logits_out);
        if(!g_cbatch_x || !g_je01_one_dhidden || g_cbatch_A!=1 || g_cbatch_layers_done!=27)
            throw std::runtime_error("JE01_ONE_COLUMN_JVP_FINALIZE_LAYER_COVERAGE");
        const int HIDDEN=g_layer_hidden;
        JE01JvpFn head=[=](const JE01JvpArrays &a) {
            auto h=mx::fast::rms_norm(a[0],wrap_host_f32(w_finalnorm,{HIDDEN}),(float)g_mla_rms_eps);
            return JE01JvpArrays{lazy_matvec_e0("lm_head",h)};
        };
        auto result=mx::jvp(head,{*g_cbatch_x},{*g_je01_one_dhidden});
        auto logits=result.first[0];
        g_je01_one_dlogits=mx::contiguous(result.second[0]);
        if(logits.shape()!=mx::Shape{1,102400} || g_je01_one_dlogits->shape()!=logits.shape())
            throw std::runtime_error("JE01_ONE_COLUMN_JVP_VOCAB_SHAPE");
        std::vector<mx::array> all{logits,*g_je01_one_dlogits};
        for(auto &a:g_fused_K)all.push_back(a);
        for(auto &a:g_fused_V)all.push_back(a);
        for(auto &a:g_je01_one_dK)all.push_back(a);
        for(auto &a:g_je01_one_dV)all.push_back(a);
        je01_direction_route_append(all);
        mx::eval(all); // One boundary per token, before x_in_host can be reused.
        je01_direction_route_emit();
        std::memcpy(logits_out,logits.data<float>(),sizeof(float)*102400);
        g_je01_one_logits_position=g_je01_one_position;
        ++g_je01_one_tokens;
        delete g_cbatch_x;g_cbatch_x=nullptr;
        g_je01_one_dhidden.reset();g_cbatch_A=0;g_cbatch_layers_done=0;
        return 1;
    } catch(const std::exception &e) {
        std::fprintf(stderr,"JE01_ONE_COLUMN_JVP_FINALIZE_FAILED reason=%s\n",e.what());
        delete g_cbatch_x;g_cbatch_x=nullptr;g_cbatch_A=0;g_cbatch_layers_done=0;
        return 0;
    }
}

extern "C" int mlx_gpu_je01_one_column_jvp_copy_logits(float *out,int vocabulary,int position) {
    if(!out || vocabulary!=102400 || !g_je01_one_dlogits ||
       g_je01_one_logits_position!=position || g_cbatch_x || g_cbatch_layers_done!=0)return 0;
    std::memcpy(out,g_je01_one_dlogits->data<float>(),sizeof(float)*102400);
    return 1;
}

extern "C" int mlx_gpu_je01_one_column_jvp_finish(uint64_t *tokens,uint64_t *cases,uint64_t *layers) {
    if(!tokens || !cases || !layers || g_cbatch_x || g_cbatch_layers_done!=0 ||
       !g_je01_direction_pending_routes.empty() ||
       g_je01_one_tokens!=4334 || g_je01_one_cases!=32 || g_je01_one_layers!=117018)return 0;
    *tokens=g_je01_one_tokens;*cases=g_je01_one_cases;*layers=g_je01_one_layers;
    return 1;
}
'''


C_OPEN = r'''
    // Separate token derivative export. The ordinary gold bytes stay identical.
    const char *je01_jvp_mode=getenv("JE01_NATIVE_JVP_MODE");
    const char *je01_jvp_path=getenv("JE01_NATIVE_JVP_GOLD_OUTPUT");
    FILE *je01_jvp_file=NULL;
    float *je01_jvp_dlogits=NULL;
    unsigned je01_jvp_gold_count=0;
    if(je01_jvp_mode && *je01_jvp_mode) {
        const char prefix[]="/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_";
        if(strcmp(je01_jvp_mode,"ROLE_DIRECTION_FIXED_Q4Q5") || !je01_gold_file ||
           !je01_jvp_path || strncmp(je01_jvp_path,prefix,sizeof(prefix)-1)) {
            fprintf(stderr,"FATAL: JE01 one-column JVP output contract\n");exit(1);
        }
        int fd=open(je01_jvp_path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
        if(fd<0 || !(je01_jvp_file=fdopen(fd,"wb"))) {
            fprintf(stderr,"FATAL: JE01 one-column JVP output must be new\n");exit(1);
        }
        je01_jvp_dlogits=malloc(sizeof(float)*MOE_VOCAB);
        if(!je01_jvp_dlogits)exit(1);
        const char magic[]="JE01_NATIVE_ONE_COLUMN_JVP_GOLD_V1\n";
        if(fwrite(magic,1,sizeof(magic)-1,je01_jvp_file)!=sizeof(magic)-1)exit(1);
    } else if(je01_jvp_path && *je01_jvp_path) {
        fprintf(stderr,"FATAL: JE01 JVP output requested without JVP mode\n");exit(1);
    }
'''


C_GOLD = r'''
                    if(je01_jvp_file) {
                        extern int mlx_gpu_je01_one_column_jvp_copy_logits(float *,int,int);
                        if(!mlx_gpu_je01_one_column_jvp_copy_logits(je01_jvp_dlogits,MOE_VOCAB,pos)) {
                            fprintf(stderr,"FATAL: JE01 one-column JVP logit tangent missing\n");exit(1);
                        }
                        double tangent_weighted=0.0,tangent_squared=0.0;
                        for(int c=0;c<MOE_VOCAB;c++) {
                            double d=(double)je01_jvp_dlogits[c];
                            if(!isfinite(d)) {fprintf(stderr,"FATAL: JE01 nonfinite logit tangent\n");exit(1);}
                            tangent_weighted+=exp((double)lg[c]-maximum)*d;
                            tangent_squared+=d*d;
                        }
                        double values[4]={nll,tangent_weighted/total-(double)je01_jvp_dlogits[target],
                                           (double)je01_jvp_dlogits[target],sqrt(tangent_squared/MOE_VOCAB)};
                        if(!isfinite(values[1]) || !isfinite(values[3]) ||
                           fwrite(record,sizeof(uint32_t),4,je01_jvp_file)!=4 ||
                           fwrite(values,sizeof(double),4,je01_jvp_file)!=4) {
                            fprintf(stderr,"FATAL: JE01 one-column JVP score write failed\n");exit(1);
                        }
                        ++je01_jvp_gold_count;
                    }
'''


C_CLOSE = r'''
    if(je01_jvp_file) {
        extern int mlx_gpu_je01_one_column_jvp_finish(uint64_t *,uint64_t *,uint64_t *);
        uint64_t tokens=0,cases=0,layers=0;
        if(je01_jvp_gold_count!=656 || !mlx_gpu_je01_one_column_jvp_finish(&tokens,&cases,&layers) ||
           fflush(je01_jvp_file)!=0 || fsync(fileno(je01_jvp_file))!=0 || fclose(je01_jvp_file)!=0) {
            fprintf(stderr,"FATAL: JE01 one-column JVP full-prefix coverage or close failed\n");exit(1);
        }
        fprintf(stderr,"JE01_ONE_COLUMN_JVP_PREFIX_COMPLETE tokens=%llu cases=%llu layers=%llu gold=%u\n",
                (unsigned long long)tokens,(unsigned long long)cases,(unsigned long long)layers,je01_jvp_gold_count);
        free(je01_jvp_dlogits);
    }
'''


def pure_layer(original):
    start = original.index("        char nq[96], nkva[96], nkvb[96], no[96];")
    end = original.index("        delete g_cbatch_x;", start)
    body = original[start:end]
    body = once(body, "mx::array x = *g_cbatch_x;", "mx::array x = state[0];", "EXPLICIT_HIDDEN")
    body = once(body,
        "g_fused_K[l] = mx::scatter(g_fused_K[l], {slot_arr, pos_arr}, k_new_win, {0, 2});",
        "cache_k = je01_native_jvp::fixed_b1_kv_scatter(slot_arr,pos_arr)({cache_k,k_new_win})[0];", "K_SCATTER")
    body = once(body,
        "g_fused_V[l] = mx::scatter(g_fused_V[l], {slot_arr, pos_arr}, v_new_win, {0, 2});",
        "cache_v = je01_native_jvp::fixed_b1_kv_scatter(slot_arr,pos_arr)({cache_v,v_new_win})[0];", "V_SCATTER")
    body = body.replace("mx::take(g_fused_K[l], slot_arr, 0)", "mx::take(cache_k, slot_arr, 0)")
    body = body.replace("mx::take(g_fused_V[l], slot_arr, 0)", "mx::take(cache_v, slot_arr, 0)")
    # This telemetry mode is forbidden by the JVP contract. Remove its dormant
    # host evaluation/global writes from the pure function, keeping native OFF.
    telemetry = body.index("            if (g_risk_signal_telemetry_on) {")
    selection = body.index("            mx::array order = mx::argsort(scores, -1);", telemetry)
    body = body[:telemetry] + body[selection:]
    body = once(body, "            je01_direction_baseline_margin_capture(l,A,slot,spos,scores,top_idx);\n", "", "NO_MARGIN_SIDE_EFFECT")
    body = once(body, "            top_idx=je01_direction_fixed_routes(l,A,slot,spos,top_idx);",
        "            top_idx=mx::stop_gradient(je01_direction_fixed_routes(l,A,slot,spos,top_idx));\n"
        "            returned_route=top_idx;", "FIXED_IDS_RETURNED")
    body = once(body, "            je01_direction_route_trace(l,A,slot,spos,top_idx);\n", "", "TRACE_OUTSIDE_TRANSFORM")
    # Inject alpha into exactly the chosen linear role; all other roles retain
    # their native output and input tangent. Do not replace an FFN residual.
    body = once(body, "gate_all=je01_direction_role_output(l,0,h2,gate_all,top_idx);",
                "gate_all=je01_role_jvp_output(l,0,h2,gate_all,top_idx,state[2]);", "GATE_ALPHA_INJECTION")
    body = once(body, "up_all=je01_direction_role_output(l,1,h2,up_all,top_idx);",
                "up_all=je01_role_jvp_output(l,1,h2,up_all,top_idx,state[2]);", "UP_ALPHA_INJECTION")
    body = once(body, "down_all=je01_direction_role_output(l,2,swiglu_3d,down_all,top_idx);",
                "down_all=je01_role_jvp_output(l,2,swiglu_3d,down_all,top_idx,state[2]);", "DOWN_ALPHA_INJECTION")
    if body.count("ffn_gather(") != 6:
        raise ValueError("JE01_ONE_JVP_EXPECTED_SIX_NATIVE_GATHER_SITES")
    body = body.replace("ffn_gather(", "je01_one_jvp_gather(")
    if "g_fused_K[l]" in body or "g_fused_V[l]" in body or "*g_cbatch_x" in body:
        raise ValueError("JE01_ONE_JVP_IMPLICIT_STATE_REMAINS")
    attention_start = body.index("        mx::array attn = mx::fast::scaled_dot_product_attention(")
    attention_end = body.index("        mx::array attn_flat = mx::reshape(attn,", attention_start)
    prefix = PRE_ATTENTION_PREFIX + body[:attention_start]
    prefix += "    return {q_full_r,k_win,v_win,cache_k,cache_v,mask_arr};\n}\n"
    postfix = POST_ATTENTION_PREFIX + body[attention_end:]
    postfix += "    return {x_out,returned_route};\n}\n"
    return prefix + postfix


def apply(cpp, c):
    if digest(cpp) != CPP_SHA256 or digest(c) != C_SHA256:
        raise ValueError("JE01_ONE_JVP_FIXED_BASE_SOURCE_SHA_DRIFT")
    cpp = once(cpp, '#include "mlx/mlx.h"\n',
               '#include "mlx/mlx.h"\n#include "'+ADAPTER_NAME+'"\n', "ADAPTER_INCLUDE")
    a = cpp.index("int mlx_gpu_cbatch_layer_step_lazy(")
    b = cpp.index("int mlx_gpu_cbatch_forward_finalize(", a)
    original = cpp[a:b]
    pure = pure_layer(original)
    renamed = once(original, "int mlx_gpu_cbatch_layer_step_lazy(",
                   "static int je01_one_jvp_original_layer(", "ORIGINAL_LAYER_PRESERVED")
    cpp = cpp[:a] + HELPERS + pure + renamed + LAYER_WRAPPER + cpp[b:]
    a = cpp.index("int mlx_gpu_cbatch_forward_finalize(")
    b = cpp.index("\n}\n", a) + 3
    original = cpp[a:b]
    renamed = once(original, "int mlx_gpu_cbatch_forward_finalize(",
                   "static int je01_one_jvp_original_finalize(", "ORIGINAL_FINALIZER_PRESERVED")
    cpp = cpp[:a] + renamed + FINALIZE_WRAPPER + cpp[b:]
    a = c.index("static int run_moe_gpu_cbatch_online_gate(")
    b = c.index("// V5i Phase B:", a)
    cpu = c[a:b]
    # Teacher-forcing's existing setup block ends immediately before its
    # persistent outer loop. Derivative output setup belongs at this scope.
    anchor = '    const char *je01_gold_out=getenv("JE01_ORIGINAL_GOLD_OUTPUT");'
    if cpu.count(anchor) != 1:
        raise ValueError("JE01_ONE_JVP_GOLD_SETUP_MISSING")
    marker = '        if (fwrite(magic,1,sizeof(magic)-1,je01_gold_file)!=sizeof(magic)-1)exit(1);\n    }\n'
    cpu = once(cpu, marker, marker+C_OPEN, "DERIVATIVE_OUTPUT_OPEN")
    marker = '                        {fprintf(stderr,"FATAL: JE01 teacher forced score write failed\\n");exit(1);}\n'
    cpu = once(cpu, marker, marker+C_GOLD, "PER_TOKEN_DERIVATIVE_AFTER_GOLD")
    marker = "    free(x_embed); free(gpu_logits);\n    return 1;\n"
    cpu = once(cpu, marker, C_CLOSE+marker, "DERIVATIVE_CHECKED_CLOSE")
    return cpp, c[:a]+cpu+c[b:]


def stage(source):
    source = Path(source)
    work = Path("/Volumes/D50/TailnetCommander/work")
    if source.name != "source" or source.parent.parent != work or \
       not source.parent.name.startswith("je01_native_role_jvp_") or \
       any(p.is_symlink() for p in (source, *source.parents)):
        raise ValueError("JE01_ONE_JVP_SEPARATE_D50_SOURCE_REQUIRED")
    cpp, c = source / "mlx_moe.cpp", source / "qwen_infer.c"
    if not all(p.is_file() and not p.is_symlink() for p in (cpp, c)):
        raise ValueError("JE01_ONE_JVP_SOURCE_FILES_REQUIRED")
    adapter = Path(__file__).resolve().with_name(ADAPTER_NAME)
    if not adapter.is_file() or adapter.is_symlink() or \
       hashlib.sha256(adapter.read_bytes()).hexdigest() != ADAPTER_SHA256:
        raise ValueError("JE01_ONE_JVP_VALIDATED_ADAPTER_SOURCE_REQUIRED")
    target = source / ADAPTER_NAME
    if target.exists() or target.is_symlink():
        raise ValueError("JE01_ONE_JVP_ADAPTER_DESTINATION_MUST_BE_NEW")
    patched_cpp, patched_c = apply(cpp.read_text(), c.read_text())
    shutil.copyfile(adapter, target)
    cpp.write_text(patched_cpp)
    c.write_text(patched_c)
    return {"schema": "beglin-je01-native-role-jvp-source-patch/1",
            "status": "SEPARATE_SCRATCH_SOURCE_PATCHED_NOT_EXECUTED",
            "base_cpp_sha256": CPP_SHA256, "base_c_sha256": C_SHA256,
            "patched_cpp_sha256": digest(patched_cpp), "patched_c_sha256": digest(patched_c),
            "adapter_sha256": ADAPTER_SHA256, "runtime_mode": MODE,
            "supported_role_indices": list(range(24)), "supported_tiers": ["Q4","Q5"],
            "one_direction_per_run": True, "alpha": 0,
            "input_scope": "PREVIOUSLY_SEEN_TRAIN16_656_GOLD_TOKENS",
            "full48_empirical_fisher": None, "allocation_selected": False,
            "native_new_role_validation": "NOT_RUN", "production_modified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    print(json.dumps(stage(parser.parse_args().source), sort_keys=True))
