#!/usr/bin/env python3
"""Read-only, mmap-backed Safetensors Expert [G,U,D.T] tile reader.

No checkpoint modification, no network, no eager loading of all weights.
This is a source adapter for hierarchical_moe_probe.py; only standard unfused
HF expert keys and BF16/F16/F32 are supported, fail closed for other layouts.
Index hash is PARTIAL provenance; full shard content identity must be verified
separately using Beglin's checkpoint_identity.py before accepting evidence.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

DTYPES={'BF16':('<u2',2),'F16':('<f2',2),'F32':('<f4',4)}


class ReaderError(ValueError):
    pass


class RealExpertTileReader:
    def __init__(self, index_file: str|Path, *, layer:int, expert_ids:list[int],
                 hidden:int, intermediate:int):
        self.index=Path(index_file).resolve(strict=True)
        self.root=self.index.parent
        self.layer=int(layer)
        self.ids=list(expert_ids)
        self.hidden=int(hidden)
        self.intermediate=int(intermediate)
        if not self.ids or len(set(self.ids))!=len(self.ids) or min(self.ids)<0:
            raise ReaderError('expert IDs must be unique nonnegative integers')
        if self.hidden<1 or self.intermediate<1:
            raise ReaderError('invalid model dimensions')
        raw=self.index.read_bytes()
        self.index_sha256=hashlib.sha256(raw).hexdigest()
        idx=json.loads(raw)
        mapping=idx.get('weight_map')
        if not isinstance(mapping,dict):
            raise ReaderError('checkpoint index must contain weight_map')
        self.tensors={}
        self.stats={}
        self.arrays={}
        headers={}
        for eid in self.ids:
            for role in ['gate_proj','up_proj','down_proj']:
                key=f'model.layers.{self.layer}.mlp.experts.{eid}.{role}.weight'
                name=mapping.get(key)
                if not isinstance(name,str) or not name:
                    raise ReaderError(f'missing standard expert key {key}')
                candidate=self.root / name
                path=candidate.resolve(strict=True)
                if path.is_symlink() or candidate.is_symlink() or not path.is_relative_to(self.root) or not path.is_file():
                    raise ReaderError('unsafe shard location')
                if path not in headers:
                    with path.open('rb') as f:
                        length_bytes=f.read(8)
                        if len(length_bytes)!=8:raise ReaderError('invalid safetensors header')
                        n=int.from_bytes(length_bytes,'little')
                        if n<2 or n>100_000_000 or n>path.stat().st_size-8:
                            raise ReaderError('invalid header byte length')
                        header=json.loads(f.read(n))
                    headers[path]=(8+n,header)
                    st=path.stat()
                    self.stats[path.name]={'size':st.st_size,'mtime_ns':st.st_mtime_ns}
                offset,header=headers[path]
                entry=header.get(key)
                if not isinstance(entry,dict):raise ReaderError(f'key not in shard header: {key}')
                shape=tuple(map(int,entry.get('shape',[])))
                required=(self.intermediate,self.hidden) if role!='down_proj' else (self.hidden,self.intermediate)
                if shape!=required:raise ReaderError(f'wrong tensor shape for {key}: {shape} != {required}')
                dtype=str(entry.get('dtype'))
                if dtype not in DTYPES:raise ReaderError(f'unsupported dtype for {key}: {dtype}')
                kind,size=DTYPES[dtype]
                start,end=map(int,entry.get('data_offsets',[]))
                if start<0 or end-start!=size*np.prod(shape) or offset+end>path.stat().st_size:
                    raise ReaderError(f'out-of-bounds tensor {key}')
                self.tensors[(eid,role)]={'path':path,'offset':offset+start,'dtype':dtype,'shape':shape,'key':key}

    def _array(self, eid:int,role:str)->np.ndarray:
        k=(eid,role)
        if k not in self.arrays:
            info=self.tensors[k]
            self.arrays[k]=np.memmap(info['path'],dtype=DTYPES[info['dtype']][0],mode='r',
                                     offset=info['offset'],shape=info['shape'])
        return self.arrays[k]

    def _float(self,a:np.ndarray,dtype:str)->np.ndarray:
        if dtype=='BF16':
            return (np.asarray(a,dtype='<u2').astype('<u4')<<16).view('<f4')
        return np.asarray(a,dtype=np.float32)

    def tile(self,tile_index:int,tile_rows:int=128)->np.ndarray:
        offset=int(tile_index)*tile_rows
        if tile_rows<=0 or offset<0 or offset+tile_rows>self.intermediate:
            raise ReaderError('out-of-bounds tile request')
        E=len(self.ids)
        out=np.empty((E,tile_rows,3*self.hidden),dtype=np.float32)
        for i,eid in enumerate(self.ids):
            for role,sl in [('gate_proj',slice(0,self.hidden)),
                            ('up_proj',slice(self.hidden,2*self.hidden)),
                            ('down_proj',slice(2*self.hidden,3*self.hidden))]:
                meta=self.tensors[(eid,role)]
                source=self._array(eid,role)
                selected=source[:,offset:offset+tile_rows].T if role=='down_proj' else source[offset:offset+tile_rows,:]
                out[i,:,sl]=self._float(selected,meta['dtype'])
        if not np.isfinite(out).all():raise ReaderError('nonfinite checkpoint tensor encountered')
        return out

    def provenance(self)->dict[str,Any]:
        return {'type':'REAL_CHECKPOINT_PARTIAL_IDENTITY',
                'layer':self.layer,'expert_ids':self.ids,
                'index_sha256':self.index_sha256,
                'shards_metadata_only':self.stats,
                'warning':'Not full checkpoint identity; run Beglin checkpoint_identity.py for all shard hashes'}
