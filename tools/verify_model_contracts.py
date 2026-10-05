#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json

EXPECTED={
 'model-source-v1','architecture-descriptor-v1','model-skeleton-v1','tensor-role-graph-v1',
 'operator-graph-v1','tokenizer-contract-v1','loader-contract-v1','loader-runtime-plan-v1','backend-capability-v1',
 'backend-transition-result-v1',
 'quant-capability-v1','runtime-mutation-v1','model-capability-bundle-v1','pipeline-eligibility-v1',
 'validation-plan-v1','verification-evidence-v1','tokenizer-runtime-plan-v1',
 'qt-observation-identity-v1','qheatmap-v2','theatmap-v2','local-precision-policy-v1',
 'qt-numeric-observation-v1','qt-quant-perturbation-v1','qt-training-sensitivity-v1',
 'q-uncertainty-v1','qpolicy-candidate-v1',
}

def verify(root: str|Path|None=None)->dict:
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]/'schemas'/'model'
    found=set()
    for path in sorted(root.glob('*.schema.json')):
        obj=json.loads(path.read_text())
        name=path.name.removesuffix('.schema.json')
        found.add(name)
        if obj.get('$schema')!='https://json-schema.org/draft/2020-12/schema':
            raise AssertionError(f'bad $schema: {path}')
        if obj.get('type')!='object':
            raise AssertionError(f'non-object contract: {path}')
        if 'schema' not in obj.get('required',[]):
            raise AssertionError(f'missing required schema discriminator: {path}')
        expected_schema=f'beglin-{name}'
        actual_schema=obj.get('properties',{}).get('schema',{}).get('const')
        if actual_schema!=expected_schema:
            raise AssertionError(
                f'bad schema discriminator: {path} '
                f'expected={expected_schema!r} actual={actual_schema!r}'
            )
    missing=EXPECTED-found; extra=found-EXPECTED
    if missing or extra:
        raise AssertionError(f'contract set mismatch missing={sorted(missing)} extra={sorted(extra)}')
    index=root/'CONTRACT_INDEX.md'
    if not index.is_file():
        raise AssertionError('CONTRACT_INDEX.md missing')
    return {'status':'PASS','schema':'beglin-p12-bskel-verification-v1','schema_count':len(found),'index':str(index)}

def main()->int:
    result=verify()
    print(json.dumps(result,sort_keys=True))
    return 0

if __name__=='__main__': raise SystemExit(main())
