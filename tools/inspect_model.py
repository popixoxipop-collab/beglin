#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_capability as mc
import model_evidence_registry as mer
import loader_runtime_v1 as lrv
import tokenizer_runtime_v1 as trv

def _load_evidence(path):
    if not path:
        return None
    p=Path(path)
    try:
        value=json.loads(p.read_text())
    except (OSError,json.JSONDecodeError) as exc:
        raise mc.ModelCapabilityError(f"invalid evidence JSON: {p}") from exc
    if not isinstance(value,dict):
        raise mc.ModelCapabilityError(f"evidence must be a JSON object: {p}")
    return value

def _load_evidence_list(paths):
    return [_load_evidence(path) for path in (paths or [])]

def _runtime_plan_report(bundle,args):
    report={}
    try:
        loader=lrv.plan_from_bundle(bundle)
        entry={'status':'AVAILABLE','plan':loader}
        if args.verify_loader_sources:
            entry['source_verification']=lrv.verify_source_files(loader)
        report['loader']=entry
    except lrv.LoaderRuntimeError as exc:
        report['loader']={'status':'UNAVAILABLE','error':str(exc)}
    try:
        tokenizer=trv.plan_from_bundle(
            bundle,
            external_executable=args.tokenizer_executable,
            external_executable_sha256=args.tokenizer_executable_sha256,
        )
        report['tokenizer']={'status':'AVAILABLE','plan':tokenizer}
    except trv.TokenizerRuntimeError as exc:
        report['tokenizer']={'status':'UNAVAILABLE','error':str(exc)}
    return report


def main()->int:
    ap=argparse.ArgumentParser(description='Inspect a model and compile the P12 Beglin capability bundle.')
    ap.add_argument('path')
    ap.add_argument('--backend',choices=['cpu','mlx_metal'])
    ap.add_argument('--cpu-runtime-verified',action='store_true',help='deprecated: requires --cpu-runtime-evidence')
    ap.add_argument('--mlx-runtime-verified',action='store_true',help='deprecated: requires --mlx-runtime-evidence')
    ap.add_argument('--cpu-runtime-evidence')
    ap.add_argument('--mlx-runtime-evidence')
    ap.add_argument('--tokenizer-evidence')
    ap.add_argument('--loader-evidence')
    ap.add_argument('--cpu-qng64-evidence',action='append',default=[])
    ap.add_argument('--mlx-qng64-evidence',action='append',default=[])
    ap.add_argument('--cpu-mutation-evidence',action='append',default=[])
    ap.add_argument('--mlx-mutation-evidence',action='append',default=[])
    ap.add_argument('--evidence',action='append',default=[],help='verification evidence JSON file or directory; repeatable')
    ap.add_argument('--json',action='store_true')
    ap.add_argument('--target')
    ap.add_argument('--runtime-plans',action='store_true',help='include loader/tokenizer runtime readiness outside the immutable capability bundle')
    ap.add_argument('--verify-loader-sources',action='store_true',help='with --runtime-plans, re-hash all loader source files')
    ap.add_argument('--tokenizer-executable',help='absolute executable for EXTERNAL_VERIFIED tokenizer runtime planning')
    ap.add_argument('--tokenizer-executable-sha256',help='expected SHA-256 of --tokenizer-executable')
    args=ap.parse_args()
    try:
        scalar_explicit={
            'cpu_runtime_evidence':_load_evidence(args.cpu_runtime_evidence),
            'mlx_runtime_evidence':_load_evidence(args.mlx_runtime_evidence),
            'tokenizer_evidence':_load_evidence(args.tokenizer_evidence),
            'loader_evidence':_load_evidence(args.loader_evidence),
        }
        list_explicit={
            'cpu_qng64_evidence':_load_evidence_list(args.cpu_qng64_evidence),
            'mlx_qng64_evidence':_load_evidence_list(args.mlx_qng64_evidence),
            'cpu_mutation_evidence':_load_evidence_list(args.cpu_mutation_evidence),
            'mlx_mutation_evidence':_load_evidence_list(args.mlx_mutation_evidence),
        }
        resolved={**{key:None for key in scalar_explicit},**{key:[] for key in list_explicit}}
        if args.evidence:
            source=mc.inspect_model_source(args.path)
            descriptor=mc.build_architecture_descriptor(source)
            registry=mer.VerificationEvidenceRegistry.from_paths(args.evidence)
            resolved=registry.resolve_model_set(
                architecture_id=descriptor['architecture_id'],
                checkpoint_identity_sha256=source['checkpoint_identity_sha256'],
            )
        evidence={
            key:mer.merge_explicit_and_registry(
                scalar_explicit[key],resolved.get(key),label=key)
            for key in scalar_explicit
        }
        evidence.update({
            key:mer.merge_explicit_list_and_registry(
                list_explicit[key],resolved.get(key,[]),label=key)
            for key in list_explicit
        })
        bundle=mc.compile_model_capabilities(
            args.path,
            backend=args.backend,
            cpu_runtime_verified=args.cpu_runtime_verified,
            mlx_runtime_verified=args.mlx_runtime_verified,
            **evidence,
        )
    except (mc.ModelCapabilityError,mer.EvidenceRegistryError) as exc:
        print(json.dumps({'status':'ERROR','error':str(exc)},sort_keys=True),file=sys.stderr)
        return 2
    runtime_plans=_runtime_plan_report(bundle,args) if args.runtime_plans else None
    if args.target:
        rows=[]
        for key in ('backend_capability_matrix','quant_capability_matrix','runtime_mutation_matrix'):
            for row in bundle[key].get('rows',[]):
                if args.target.lower() in str(row.get('target_key','')).lower():
                    rows.append({'matrix':key,**row})
        out={'target':args.target,'rows':rows,'bundle_sha256':bundle['bundle_sha256']}
        if runtime_plans is not None:
            out['runtime_plans']=runtime_plans
        print(json.dumps(out,indent=2,sort_keys=True))
        return 0
    if args.json:
        out=bundle if runtime_plans is None else {'bundle':bundle,'runtime_plans':runtime_plans}
        print(json.dumps(out,indent=2,sort_keys=True))
    else:
        out=mc.capability_summary(bundle)
        if runtime_plans is not None:
            out['runtime_plans']=runtime_plans
        print(json.dumps(out,indent=2,sort_keys=True))
    return 0

if __name__=='__main__': raise SystemExit(main())
