#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_capability as mc

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

def main()->int:
    ap=argparse.ArgumentParser(description='Inspect a model and compile the P12 Beglin capability bundle.')
    ap.add_argument('path')
    ap.add_argument('--backend',choices=['cpu','mlx_metal'])
    ap.add_argument('--cpu-runtime-verified',action='store_true',help='deprecated: requires --cpu-runtime-evidence')
    ap.add_argument('--mlx-runtime-verified',action='store_true',help='deprecated: requires --mlx-runtime-evidence')
    ap.add_argument('--cpu-runtime-evidence')
    ap.add_argument('--mlx-runtime-evidence')
    ap.add_argument('--tokenizer-evidence')
    ap.add_argument('--json',action='store_true')
    ap.add_argument('--target')
    args=ap.parse_args()
    try:
        bundle=mc.compile_model_capabilities(
            args.path,
            backend=args.backend,
            cpu_runtime_verified=args.cpu_runtime_verified,
            mlx_runtime_verified=args.mlx_runtime_verified,
            cpu_runtime_evidence=_load_evidence(args.cpu_runtime_evidence),
            mlx_runtime_evidence=_load_evidence(args.mlx_runtime_evidence),
            tokenizer_evidence=_load_evidence(args.tokenizer_evidence),
        )
    except mc.ModelCapabilityError as exc:
        print(json.dumps({'status':'ERROR','error':str(exc)},sort_keys=True),file=sys.stderr)
        return 2
    if args.target:
        rows=[]
        for key in ('backend_capability_matrix','quant_capability_matrix','runtime_mutation_matrix'):
            for row in bundle[key].get('rows',[]):
                if args.target.lower() in str(row.get('target_key','')).lower(): rows.append({'matrix':key,**row})
        print(json.dumps({'target':args.target,'rows':rows,'bundle_sha256':bundle['bundle_sha256']},indent=2,sort_keys=True))
        return 0
    if args.json:
        print(json.dumps(bundle,indent=2,sort_keys=True))
    else:
        print(json.dumps(mc.capability_summary(bundle),indent=2,sort_keys=True))
    return 0

if __name__=='__main__': raise SystemExit(main())
