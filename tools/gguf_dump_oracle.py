import sys
sys.path.insert(0, "/Users/bob/llamacpp_kleidi_build/gguf-py")
import gguf

path = sys.argv[1]
r = gguf.GGUFReader(path)

version = int(r.fields["GGUF.version"].contents())
alignment = 32
if "general.alignment" in r.fields:
    alignment = int(r.fields["general.alignment"].contents())

print(f"VERSION {version}")
print(f"ALIGNMENT {alignment}")

kv_keys = [k for k in r.fields.keys() if k not in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count")]
print(f"N_KV {len(kv_keys)}")
print(f"N_TENSORS {len(r.tensors)}")

for k in kv_keys:
    f = r.fields[k]
    is_array = len(f.types) > 1 or (len(f.types) == 1 and f.types[0] == gguf.GGUFValueType.ARRAY)
    # gguf-py's ReaderField.types is [ARRAY, elem_type] for arrays, [scalar_type] otherwise.
    if f.types and f.types[0] == gguf.GGUFValueType.ARRAY:
        elem_type = f.types[1]
        arr_len = len(f.data)
        print(f"KV {k} ARRAY_OF_{elem_type.name} len={arr_len}")
        continue
    t = f.types[0]
    val = f.contents()
    if t == gguf.GGUFValueType.STRING:
        print(f"KV {k} STRING len={len(val.encode())} val={val}")
    elif t in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
        print(f"KV {k} {t.name} val={val:.6g}")
    elif t == gguf.GGUFValueType.BOOL:
        print(f"KV {k} BOOL val={int(val)}")
    else:
        print(f"KV {k} {t.name} val={int(val)}")

for tt in r.tensors:
    ne = list(tt.shape) + [1]*(4-len(tt.shape))
    print(f"TENSOR {tt.name} type={tt.tensor_type.name} n_dims={len(tt.shape)} "
          f"ne=[{ne[0]},{ne[1]},{ne[2]},{ne[3]}] n_elements={tt.n_elements} "
          f"n_bytes={tt.n_bytes} data_offset={tt.data_offset}")
