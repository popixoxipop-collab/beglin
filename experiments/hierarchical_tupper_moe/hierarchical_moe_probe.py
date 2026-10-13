#!/usr/bin/env python3
"""Offline Beglin hierarchical micro-expert assembly: real-shaped synthetic probe.

Features:
 - Exact SwiGLU assembly into arbitrary neuron-channel widths, nested by 16 lanes.
 - Generalized Tupper N x M binary router-mask packing and reversible pixel decoding.
 - Full 2048x1408x64 macro-expert dimensions, generated tile-by-tile without
   ever storing the complete 64-expert model in RAM.
 - Leakage-conscious heldout-expert/coordinate shared-dictionary factorization
   across all experts and four channel widths on IDENTICAL generated weights.
 - Q4 vs shared-basis + Q2 residual proxy, including dictionary/coeff/index bytes.
 - Separate structured vs independently sampled synthetic controls.
 - Calibration-only static-channel routing and heldout-input oracle upper bound.

Does NOT load pretrained weights, implement Beglin's native qNg64, run GPU kernels,
or benchmark inference throughput. Never touches a production model or service.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable
import numpy as np

SCHEMA = 'beglin-hierarchical-microexpert-synthetic-v1'


def tupper_encode(mask: np.ndarray) -> dict[str, int]:
    """N rows x M cols, least-significant bit is row zero, column zero."""
    a = np.asarray(mask)
    if a.ndim != 2 or not a.shape[0] or not a.shape[1] or not np.all((a == 0) | (a == 1)):
        raise ValueError('mask must be a nonempty binary N by M matrix')
    n, m = map(int, a.shape)
    bitmap = 0
    for x in range(m):
        col = a[:, x]
        for y in np.flatnonzero(col):
            bitmap |= 1 << (n*x + int(y))
    return {'N': n, 'M': m, 'k': bitmap*n, 'payload_bits': n*m}


def tupper_decode(k: int, n: int, m: int) -> np.ndarray:
    if n <= 0 or m <= 0 or k < 0 or k % n != 0:
        raise ValueError('N,M must be positive and k a nonnegative multiple of N')
    bitmap = k // n
    out = np.empty((n, m), dtype=np.uint8)
    for x in range(m):
        bits = bitmap >> (n*x)
        for y in range(n):
            out[y, x] = (bits >> y) & 1
    return out


def assemble_swiglu(rows: np.ndarray, h: np.ndarray, group: int) -> np.ndarray:
    """rows [intermediate,3*hidden] = [G, U, D.T] for ONE macro expert."""
    intermediate, triple = rows.shape
    if triple % 3 or intermediate % group:
        raise ValueError('shape/group mismatch')
    d = triple // 3
    if h.shape != (d,):
        raise ValueError('input hidden mismatch')
    out = np.zeros(d, np.float64)
    for start in range(0, intermediate, group):
        a = rows[start:start+group]
        g = a[:, :d].astype(np.float64) @ h
        u = a[:, d:2*d].astype(np.float64) @ h
        z = (g/(1.0 + np.exp(-g))) * u
        out += a[:, 2*d:].astype(np.float64).T @ z
    return out


def permute_channels(rows: np.ndarray, order: np.ndarray) -> np.ndarray:
    if order.shape != (rows.shape[0],) or not np.array_equal(np.sort(order), np.arange(len(order))):
        raise ValueError('invalid neuron permutation')
    # G/U row and D columns all move together, preserving SwiGLU.
    return rows[order].copy()


def quant_sq_error(values: np.ndarray, n: int, max_temp_elements: int = 1_600_000) -> float:
    """Quantize in consecutive group64s without a packed runtime codec.

    Proxy only: symmetric ties-to-even rounding, NO Beglin error feedback.
    Returns squared reconstruction error in double precision.
    """
    if values.size % 64:
        raise ValueError('quant values need group64 divisibility')
    flat = np.asarray(values, dtype=np.float32).reshape(-1, 64)
    qmax, qmin = (1 << (n-1))-1, -(1 << (n-1))
    sq = 0.0
    chunk = max(1, max_temp_elements//64)
    for lo in range(0, len(flat), chunk):
        src = flat[lo:lo+chunk]
        scales = np.maximum(np.max(np.abs(src), axis=1, keepdims=True)/qmax, 1e-12)
        quant = np.clip(np.rint(src/scales), qmin, qmax) * scales
        diff = (src - quant).astype(np.float64)
        sq += float(np.sum(diff*diff))
    return sq


def role_quant_sq_error(rows: np.ndarray, n: int, hidden: int) -> float:
    """Rows [E, tile, 3H], tile multiple 64; D.T grouped along tile axis."""
    if rows.ndim != 3 or rows.shape[-1] != 3*hidden or rows.shape[1] % 64:
        raise ValueError('expected expert by rows by 3*H, rows multiple 64')
    sq = 0.0
    sq += quant_sq_error(rows[:, :, :hidden], n)
    sq += quant_sq_error(rows[:, :, hidden:2*hidden], n)
    sq += quant_sq_error(rows[:, :, 2*hidden:].transpose(0,2,1).copy(), n)
    return sq


def byte_budget(experts: int, hidden: int, intermediate: int, rank: int,
                width: int, *, with_permutation: bool = True) -> dict[str, int | float]:
    if intermediate%width or hidden%64 or intermediate%64:
        raise ValueError('dimensions must match group widths')
    params = 3*experts*hidden*intermediate
    groups_per_role = params // 64
    q4 = params*4//8 + groups_per_role*4
    q2 = params*2//8 + groups_per_role*4
    dic = (rank+1)*(3*hidden*intermediate)*2  # fp16 shared mean+basis
    coeff = experts * (intermediate//width) * rank * 2
    perm = experts*intermediate*2 if with_permutation else 0
    # 16 bytes per channel group as rough non-native descriptor cost
    desc = experts*(intermediate//width)*16
    total = q2+dic+coeff+perm+desc
    return {'parameters':params,'q4_proxy_bytes':q4, 'q2_residual_proxy_bytes':q2,
            'dictionary_fp16_bytes':dic,'coeff_fp16_bytes':coeff,
            'permutation_table_bytes':perm,'channel_descriptor_bytes':desc,
            'total_hybrid_proxy_bytes':total,'hybrid_over_q4':total/q4}


def fit_and_predict(X: np.ndarray, ntrain: int, rank: int, calib_idx: np.ndarray,
                    basis_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized train-only shared basis; partial observed coords for expert coeffs.

    X [E,D], test experts excluded from basis construction; only their
    calib coordinates determine coefficients. Evaluation elsewhere is untouched.
    FP16 roundtrip of mean/basis/coeff charged in budget.
    """
    tr = X[:ntrain]
    mu = np.mean(tr, axis=0).astype(np.float32)
    center_s = tr[:,basis_idx] - mu[basis_idx]
    gram = center_s @ center_s.T
    vals, vectors = np.linalg.eigh(gram)
    select = np.argsort(vals)[::-1][:rank]
    directions = vectors[:,select].T.astype(np.float32)
    # project complete training coordinates to the fitted principal subspace
    basis = directions @ (tr-mu[None,:])
    basis /= np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)
    mu = mu.astype(np.float16).astype(np.float32)
    basis = basis.astype(np.float16).astype(np.float32)
    z = basis[:,calib_idx]
    lhs = z @ z.T + np.eye(rank, dtype=np.float32)*1e-5
    rhs = z @ (X[:,calib_idx]-mu[calib_idx]).T
    coeff = np.linalg.solve(lhs, rhs).T
    coeff = coeff.astype(np.float16).astype(np.float32)
    pred = mu[None,:] + coeff @ basis
    return pred, mu, basis


def generate_tile(rng: np.random.Generator, mode: str, experts: int,
                  rows: int, hidden: int, latent_lane: int, amplitude: float,
                  noise: float) -> np.ndarray:
    """Generate all experts in a tile. True latent coefficients change every lane."""
    cols = hidden*3
    if mode == 'independent':
        return rng.normal(0, amplitude, (experts,rows,cols)).astype(np.float32)
    if mode != 'structured':
        raise ValueError('mode must be structured or independent')
    data = np.empty((experts,rows,cols), np.float32)
    for off in range(0,rows,latent_lane):
        shape = (latent_lane,cols)
        anchor = rng.normal(0,amplitude,shape).astype(np.float32)
        pattern = rng.normal(0,amplitude*.36,(2,)+shape).astype(np.float32)
        coeffs = rng.normal(0,.65,(experts,2)).astype(np.float32)
        for i in range(experts):
            # Use array-shaped noise only per expert to keep peak RSS low.
            data[i,off:off+latent_lane] = (
                anchor + np.einsum('r,rgd->gd',coeffs[i],pattern)
                + rng.normal(0,noise,shape).astype(np.float32))
    return data


def lane_contributions(tile: np.ndarray, h: np.ndarray, *, lane: int = 16) -> np.ndarray:
    """Per expert and per lane SwiGLU contribution [E, tile/lane, H]."""
    E, rows, triple = tile.shape
    d = triple//3
    if h.shape != (d,) or rows % lane:
        raise ValueError('input shape incompatible with tile/lane')
    outs = np.empty((E,rows//lane,d),np.float32)
    for g in range(rows//lane):
        a = tile[:,g*lane:(g+1)*lane,:]
        gate = np.einsum('egh,h->eg',a[:,:,:d],h)
        up = np.einsum('egh,h->eg',a[:,:,d:2*d],h)
        gate = np.clip(gate,-45,45)
        z = (gate/(1.0+np.exp(-gate)))*up
        outs[:,g,:] = np.einsum('egh,eg->eh',a[:,:,2*d:],z)
    return outs


def tupper_routing_and_sparsity(expert_lanes_cal: np.ndarray,
                                 expert_lanes_test: np.ndarray,
                                 *, total_experts: int, top_ids: list[int],
                                 widths: list[int], lane: int,
                                 fractions: list[float]) -> dict[str,Any]:
    """Calibration-only static scores vs TEST oracle; oracle requires full FFN.

    Full-channel baseline faithfully sums all micro channel contributions.
    Theoretical FFN fraction excludes shared/attention/router overhead.
    """
    E, channels, hidden = expert_lanes_test.shape
    if E!=len(top_ids) or expert_lanes_cal.shape!=expert_lanes_test.shape:
        raise ValueError('macro-router selection mismatch')
    full_expert=expert_lanes_test.astype(np.float64).sum(axis=1)
    result={}
    for width in widths:
        factor=width//lane
        if width % lane or channels % factor:
            raise ValueError('incompatible width/lane')
        ng=channels//factor
        a_cal=expert_lanes_cal.reshape(E,ng,factor,hidden).sum(axis=2,dtype=np.float64)
        a_test=expert_lanes_test.reshape(E,ng,factor,hidden).sum(axis=2,dtype=np.float64)
        static_score=np.linalg.norm(a_cal,axis=2)
        oracle_score=np.linalg.norm(a_test,axis=2)
        by_frac={}
        for f in fractions:
            retained=max(1,math.ceil(ng*f))
            mask=np.zeros((ng,total_experts),np.uint8)
            static_recon=np.zeros((E,hidden),np.float64)
            oracle_recon=np.zeros((E,hidden),np.float64)
            for i,eid in enumerate(top_ids):
                stat=np.argsort(static_score[i],kind='stable')[-retained:]
                best=np.argsort(oracle_score[i],kind='stable')[-retained:]
                mask[stat,eid]=1
                static_recon[i]=a_test[i,stat].sum(axis=0)
                oracle_recon[i]=a_test[i,best].sum(axis=0)
            packed=tupper_encode(mask)
            if not np.array_equal(mask,tupper_decode(packed['k'],ng,total_experts)):
                raise AssertionError('generalized Tupper routing mask failed to round-trip')
            denom=max(float(np.linalg.norm(full_expert)),1e-15)
            by_frac[str(f)]={
                'retained_per_macro_expert':retained,
                'activated_micro_experts':retained*E,
                'full_micro_experts_for_selected_macros':ng*E,
                'static_calibration_relative_l2':float(np.linalg.norm(full_expert-static_recon)/denom),
                'oracle_test_relative_l2_upper_bound':float(np.linalg.norm(full_expert-oracle_recon)/denom),
                'theoretical_macro_ffn_compute_fraction':retained/ng,
                'tupper_mask_shape':[ng,total_experts],
                'tupper_mask_raw_bytes':math.ceil(ng*total_experts/8),
                'tupper_roundtrip':True,
                'note':'Oracle computes all channel outputs first; cannot yield runtime speedup.',
            }
        result[str(width)]=by_frac
    return result


def probe(*,mode: str, seed: int, experts: int, hidden: int, intermediate: int,
          groups: list[int], lane: int, train_experts: int, rank: int,
          calibration_fraction: float, tiles: int | None,
          top_k: int, amplitude: float=.035, noise: float=.001,
          tile_source: Callable[[int], np.ndarray] | None = None,
          source_provenance: dict[str,Any] | None = None) -> dict[str,Any]:
    if hidden<=0 or hidden%64 or intermediate%128 or intermediate%lane:
        raise ValueError('hidden group64, intermediate divisible 128 and lane required')
    if any(w<lane or w%lane or intermediate%w or 128%w for w in groups):
        raise ValueError('widths must be lane multiples and divide tile128 & intermediate')
    if experts<train_experts+2 or rank>=train_experts-1 or train_experts<4:
        raise ValueError('invalid train/expert/rank counts')
    if top_k<1 or top_k>experts or not(0<calibration_fraction<.5):
        raise ValueError('invalid topK/calibration')
    total_tiles=intermediate//128
    used=total_tiles if tiles is None else min(total_tiles,tiles)
    if used<1:
        raise ValueError('at least one tile required')
    # Every config sees exactly the same underlying synthetic weights.
    rng=np.random.default_rng(seed)
    h_rng=np.random.default_rng(seed+101)
    h_cal=h_rng.normal(0,1,hidden).astype(np.float32)
    h_test=h_rng.normal(0,1,hidden).astype(np.float32)
    top_ids=list(range(top_k))
    contributions_cal=[]
    contributions_test=[]
    stats={w:{'err_sq':0.0,'mean_sq':0.0,'total_sq':0.0,'q2_sq':0.0,
              'q4_sq':0.0, 'original_full_sq':0.0, 'eval_numel':0} for w in groups}
    per_tile_digest=hashlib.sha256()
    start=time.perf_counter()
    for ti in range(used):
        full=(generate_tile(rng,mode,experts,128,hidden,lane,amplitude,noise)
              if tile_source is None else tile_source(ti))
        if full.shape!=(experts,128,3*hidden) or full.dtype!=np.float32 or not np.isfinite(full).all():
            raise ValueError('tile source returned unsupported shape/dtype or nonfinite weights')
        per_tile_digest.update(np.ascontiguousarray(full[0,0,:32]).tobytes())
        cal_lanes=lane_contributions(full[top_ids],h_cal,lane=lane)
        test_lanes=lane_contributions(full[top_ids],h_test,lane=lane)
        contributions_cal.append(cal_lanes)
        contributions_test.append(test_lanes)
        ref_q4_sq=role_quant_sq_error(full,4,hidden)
        original_sq=float(np.sum(full.astype(np.float64)**2))
        for w in groups:
            reconstructed=np.empty_like(full)
            for start_row in range(0,128,w):
                segment=full[:,start_row:start_row+w,:].reshape(experts,-1)
                D=segment.shape[1]
                # deterministic coordinate holdout; identical masked coordinate pattern
                # within a given group for ALL experts, with a separate train-only basis.
                obs_rng=np.random.default_rng(seed+100003*ti+211*w+start_row)
                choice=obs_rng.permutation(D)
                ncal=max(rank+8,int(D*calibration_fraction))
                cal_idx=choice[:ncal]
                basis_idx=choice[ncal:ncal+min(4096,D-ncal)]
                pred,mu,basis=fit_and_predict(segment,train_experts,rank,cal_idx,basis_idx)
                held=segment[train_experts:]
                pheld=pred[train_experts:]
                # We evaluate on coordinates not used to estimate expert coefficients.
                # Basis is trained exclusively on first train_experts.
                test_idx=choice[ncal:]
                diff=held[:,test_idx]-pheld[:,test_idx]
                mdiff=held[:,test_idx]-mu[test_idx]
                stats[w]['err_sq']+=float(np.sum(diff.astype(np.float64)**2))
                stats[w]['mean_sq']+=float(np.sum(mdiff.astype(np.float64)**2))
                stats[w]['total_sq']+=float(np.sum(held[:,test_idx].astype(np.float64)**2))
                stats[w]['eval_numel']+=int(diff.size)
                reconstructed[:,start_row:start_row+w,:]=pred.reshape(experts,w,3*hidden)
            residual=full-reconstructed
            stats[w]['q2_sq']+=role_quant_sq_error(residual,2,hidden)
            stats[w]['q4_sq']+=ref_q4_sq
            stats[w]['original_full_sq']+=original_sq
            del reconstructed,residual
        del full
    cal=np.concatenate(contributions_cal,axis=1)
    test=np.concatenate(contributions_test,axis=1)
    if cal.shape[1] != used*128//lane:
        raise AssertionError('intermediate channel accounting mismatch')
    routing=tupper_routing_and_sparsity(cal,test,total_experts=experts,
                                      top_ids=top_ids,widths=groups,lane=lane,
                                      fractions=[.25,.5,.75,1.0])
    del cal,test,contributions_cal,contributions_test
    result={}
    frac=used/total_tiles
    for w,s in stats.items():
        budget=byte_budget(experts,hidden,intermediate,rank,w)
        result[str(w)]={
            'grid_full':[intermediate//w,experts],
            'grid_tested':[used*128//w,experts],
            'channel_width':w,
            'tested_parameter_fraction':frac,
            'storage_full_layer_projection':budget,
            'metric_holdout_relative_l2':math.sqrt(s['err_sq']/max(s['total_sq'],1e-30)),
            'metric_explained_vs_shared_mean_mse':1-s['err_sq']/max(s['mean_sq'],1e-30),
            'metric_proxy_q4_full_weight_relative_l2':math.sqrt(s['q4_sq']/max(s['original_full_sq'],1e-30)),
            'metric_proxy_law_plus_q2_full_weight_relative_l2':math.sqrt(s['q2_sq']/max(s['original_full_sq'],1e-30)),
            'full_tested_weight_q4_squared_error':s['q4_sq'],
            'full_tested_weight_hybrid_q2_squared_error':s['q2_sq'],
            'heldout_evaluated_parameters':s['eval_numel'],
            'routing':routing[str(w)],
        }
    return {
        'schema':SCHEMA,
        'synthetic_mode':mode,
        'model_equivalent_shape':{'hidden':hidden,'intermediate':intermediate,'experts_per_layer':experts,
                                  'base_lane_width':lane,'total_physical_swiglu_parameters_per_layer':3*experts*hidden*intermediate},
        'shared_rank':rank,
        'train_expert_count':train_experts,'heldout_expert_count':experts-train_experts,
        'partial_expert_calibration_fraction':calibration_fraction,
        'processed_intermediate_rows':used*128,'complete_intermediate_rows':intermediate,
        'all_rows_processed':used==total_tiles,
        'top_k_macro_experts_for_assembly':top_k,
         'source': (dict(source_provenance or {},
                         scanned_tile_probe_sha256=per_tile_digest.hexdigest(),
                         implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                    if tile_source is not None else
                    {'type':'SYNTHETIC_ONLY','seed':seed,'first_row_slice_sha256':per_tile_digest.hexdigest(),
                     'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}),
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,
                   'seconds':round(time.perf_counter()-start,3),'production_touched':False},
        'widths':result,
        'limitations':[
            ('PARTIAL CHECKPOINT IDENTITY: no full shard-hash receipt' if tile_source is not None
             else 'NO REAL CHECKPOINT: cannot establish pretrained LLM weight law'),
            ('Synthetic activation inputs are not pretrained transformer activations' if tile_source is not None
             else 'Synthetic inputs and weights are not trained transformer distributions'),
            'Q2/Q4 numeric proxy is not native Beglin qNg64 with error feedback',
            'Full-dimension input does not imply full pre-trained model execution',
            'Storage estimates omit cache alignment, kernel code, and runtime buffers',
            'Sparse-channel quality is tested on synthetic holdout activation only; no LLM task evaluation',
            'Oracle routing evaluates all channel outputs first, and offers NO runtime speedup',
            'Smaller channels require new grouped GPU/CPU kernels for runtime gain',
            'Real mode does NOT align expert hidden neuron permutations across experts',
            'Heldout experts expose calibration coordinates for coefficient fitting; not zero-shot unseen expert prediction',
        ],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    origin=p.add_mutually_exclusive_group()
    origin.add_argument('--mode',choices=['structured','independent'],default=None)
    origin.add_argument('--checkpoint-index',type=Path,default=None)
    p.add_argument('--layer',type=int,default=1)
    p.add_argument('--seed',type=int,default=20261013)
    p.add_argument('--experts',type=int,default=64)
    p.add_argument('--hidden',type=int,default=2048)
    p.add_argument('--intermediate',type=int,default=1408)
    p.add_argument('--groups',type=int,nargs='+',default=[16,32,64,128])
    p.add_argument('--lane',type=int,default=16)
    p.add_argument('--train-experts',type=int,default=16)
    p.add_argument('--rank',type=int,default=2)
    p.add_argument('--calibration-fraction',type=float,default=.2)
    p.add_argument('--tiles',type=int,default=None,help='override full 11 tiles for quick smoke; metrics then partial')
    p.add_argument('--top-k',type=int,default=6)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    tile_source=None
    source_provenance=None
    mode=a.mode or 'structured'
    if a.checkpoint_index is not None:
        from real_checkpoint_reader import RealExpertTileReader
        reader=RealExpertTileReader(a.checkpoint_index,layer=a.layer,
                                    expert_ids=list(range(a.experts)),
                                    hidden=a.hidden,intermediate=a.intermediate)
        tile_source=reader.tile
        source_provenance=reader.provenance()
        mode='real_checkpoint'
    res=probe(mode=mode,seed=a.seed,experts=a.experts,hidden=a.hidden,
              intermediate=a.intermediate,groups=a.groups,lane=a.lane,
              train_experts=a.train_experts,rank=a.rank,
              calibration_fraction=a.calibration_fraction,tiles=a.tiles,top_k=a.top_k,
              tile_source=tile_source,source_provenance=source_provenance)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'mode':mode,'all_rows_processed':res['all_rows_processed'],
                      'seconds':res['runtime']['seconds'],
                      'widths':{w:{'hybrid/q4':v['storage_full_layer_projection']['hybrid_over_q4'],
                                   'explained':v['metric_explained_vs_shared_mean_mse'],
                                   'q2_sq/q4_sq':v['full_tested_weight_hybrid_q2_squared_error']/max(v['full_tested_weight_q4_squared_error'],1e-30),
                                   'static_channel_half_l2':v['routing']['0.5']['static_calibration_relative_l2']}
                                for w,v in res['widths'].items()}},indent=2))


if __name__=='__main__':
    main()
