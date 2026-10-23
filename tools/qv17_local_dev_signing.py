#!/usr/bin/env python3
"""BEGLIN Qv17 research-only Ed25519 offline signing for persistent quantized
alternates. The private dev key stays owner-only on EOE D50 and MUST NOT be
committed. The public trust-root is committed. An isolated caller must verify
the pinned public root and detached signature BEFORE invoking any native load.
"""
import argparse,base64,hashlib,json,os,pathlib,stat,struct,sys,time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey,Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
ROOT=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv13-github-commit-20261023")
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023")
SEC=LAB/"dev_signing_secret"
PRIVATE=SEC/"QV17_DEV_ED25519_PRIVATE_PKCS8.pem"
PUB_SRC=ROOT/"docs/research/BEGLIN_QV17_ED25519_DEV_TRUST_ROOT_PUBLIC_20261023.txt"
ARTIFACT=LAB/"owner_only_artifacts/alternate40_Qv8_random73_qng64_v1.pack"
MANIFEST=LAB/"owner_only_artifacts/alternate40_Qv8_random73_manifest.json"
SIGNATURE=LAB/"owner_only_artifacts/alternate40_Qv8_random73_manifest.sig"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
MODEL=SNAP/"model.safetensors"
CFG=SNAP/"config.json"
Q16=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
Q17=LAB/"native_persisted_C_v2"
EXE=Q17/"beglin_qv17_signed_packed_cache"
CSRC=Q17/"qv17_original_Qwen28_persisted_cache.c"
FROZEN=Q16/"independent_arc_validation48"
BIN_SHA="16085378788b64b229f321b49c11540e286edffbff832db4e41fbf8d2a87643b"
CSRC_SHA="0c9d797c0066fdb1f8ca5972bdf7611ae75431050d0d55c0f0fdef0653f6bdaa"
MASKS={"source_qv8":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
       "random73":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde"}
FROZEN_SHA="a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296"
CONFIG_SHA="86afa7d8ef2e622fb9a917c75008198e791eac225937c687c0910fc6e46f377e"
MODEL_REVISION="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ROLE_ORDER=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
SHAPES={"q_proj":(1536,1536),"k_proj":(256,1536),"v_proj":(256,1536),
        "o_proj":(1536,1536),"gate_proj":(8960,1536),"up_proj":(8960,1536),"down_proj":(1536,8960)}
# One-time generated key root SHA pinned in research runner. Do not replace automatically.
DEV_PUB_ROOT_SHA_PLACEHOLDER="5f535c54985ac18257b50c4bd7969ec4ce57aecb766def485c68c8b0a4ad4847"
def digest_bytes(blob):return hashlib.sha256(blob).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
 return h.hexdigest()
def source_file(p,h,meaning):
 if not p.is_file() or sha_file(p)!=h:raise RuntimeError("SIGNED_QV17_SOURCE_DRIFT "+meaning)
def put_new(p,b):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def canonical(o):return (json.dumps(o,ensure_ascii=False,sort_keys=True,indent=2,separators=(",",": "))+"\n").encode()
def require_private_dir():
 SEC.mkdir(parents=True,exist_ok=True,mode=0o700);SEC.chmod(0o700)
 if SEC.is_symlink() or SEC.stat().st_mode&0o077:raise RuntimeError("private key dir accessible to others")
def root_public():
 raw=PUB_SRC.read_text().strip()
 if not raw.startswith("ed25519-raw-hex:"):raise RuntimeError("public pinned dev root schema error")
 key=bytes.fromhex(raw.split(":",1)[1])
 if len(key)!=32:raise RuntimeError("pinned Ed25519 dev root key length wrong")
 return key
def create_root():
 if PRIVATE.exists() or PUB_SRC.exists():raise FileExistsError("Qv17 dev signer was already initialized; do not reissue")
 require_private_dir()
 key=Ed25519PrivateKey.generate()
 raw=key.private_bytes(encoding=serialization.Encoding.PEM,format=serialization.PrivateFormat.PKCS8,
                       encryption_algorithm=serialization.NoEncryption())
 put_new(PRIVATE,raw)
 pub=key.public_key().public_bytes(encoding=serialization.Encoding.Raw,format=serialization.PublicFormat.Raw)
 put_new(PUB_SRC,b"ed25519-raw-hex:"+pub.hex().encode()+b"\n")
 print("QV17_DEV_ED25519_KEY_CREATED",json.dumps({"public_trust_root_sha256":sha_file(PUB_SRC),
   "public_key_sha256":digest_bytes(pub),"private_key_owner_only":True,
   "WARNING":"locally generated test signing key is NOT a production attestation certificate"},sort_keys=True))
def verify_roots():
 if DEV_PUB_ROOT_SHA_PLACEHOLDER.startswith("TO_BE_"):raise RuntimeError("Qv17 trust root not pinned yet")
 source_file(PUB_SRC,DEV_PUB_ROOT_SHA_PLACEHOLDER,"pinned developer public key")
 return root_public()
def check_sources():
 source_file(EXE,BIN_SHA,"native executable")
 source_file(CSRC,CSRC_SHA,"derived C compiler source")
 source_file(CFG,CONFIG_SHA,"original model config")
 source_file(FROZEN/"frozen_inputs_private.json",FROZEN_SHA,"frozen source-only input contract")
 for name,h in MASKS.items():source_file(FROZEN/"masks"/(name+".txt"),h,"original mask "+name)
def verify_packed_layout():
 if ARTIFACT.is_symlink() or not ARTIFACT.is_file():raise RuntimeError("pack must be a normal file")
 if ARTIFACT.stat().st_mode&0o077:raise RuntimeError("packed file accessible to group/others")
 head=struct.Struct("<8sIIQ");entry=struct.Struct("<HHHHIIQQ")
 with ARTIFACT.open("rb") as f:
  magic,version,count,payload=head.unpack(f.read(head.size))
  if magic!=b"BEGQ17A1" or version!=1 or count!=40 or payload>96*1048576:
   raise RuntimeError("pack header invalid")
  ma={name:{tuple(line.split()) for line in (FROZEN/"masks"/(name+".txt")).read_text().splitlines()} for name in MASKS}
  src,rnd=ma["source_qv8"],ma["random73"]
  differing=[(role,layer) for role in ROLE_ORDER for layer in range(28) if ((role,str(layer)) in src)!=((role,str(layer)) in rnd)]
  if len(differing)!=40:raise RuntimeError("signed source/random must differ in 40")
  bytes_read=0
  for role,layer in differing:
   rid,l,bits,res,out,inp,npacked,nscales=entry.unpack(f.read(entry.size))
   target=8 if (role,str(layer)) in rnd else 4
   eout,ein=SHAPES[role]
   if rid!=ROLE_ORDER.index(role) or l!=layer or bits!=target or res or (out,inp)!=(eout,ein):
    raise RuntimeError("packed tensor role/layer/bit/shape wrong")
   expected_pack=eout*ein if bits==8 else eout*ein//2
   expected_scales=eout*ein//64*4
   if (npacked,nscales)!=(expected_pack,expected_scales):raise RuntimeError("packed tensor byte layout wrong")
   for n in (npacked,nscales):
    f.seek(n,1);bytes_read+=n
  if bytes_read!=payload or f.read(1):raise RuntimeError("packed trailing bytes or wrong total")
  expect_len=head.size+40*entry.size+payload
  if ARTIFACT.stat().st_size!=expect_len:raise RuntimeError("packed file length error")
 return {"tensor_count":40,"payload_bytes":payload,"file_bytes":expect_len,
  "inactive_pack_mib":payload/1048576}
def sign():
 verify_roots();check_sources()
 if MANIFEST.exists() or SIGNATURE.exists():raise FileExistsError("signed artifact immutable after issuing")
 if PRIVATE.is_symlink() or not PRIVATE.is_file() or PRIVATE.stat().st_mode&0o077:
  raise RuntimeError("owner-only local private key not authorized")
 key=serialization.load_pem_private_key(PRIVATE.read_bytes(),password=None)
 if not isinstance(key,Ed25519PrivateKey):raise RuntimeError("not Ed25519")
 pub=key.public_key().public_bytes(encoding=serialization.Encoding.Raw,format=serialization.PublicFormat.Raw)
 if pub!=root_public():raise RuntimeError("signing secret does not match pinned developer root")
 layout=verify_packed_layout()
 t=time.perf_counter();original_bf16_sha=sha_file(MODEL);model_ms=(time.perf_counter()-t)*1000
 packsha=sha_file(ARTIFACT)
 manifest={"schema":"BEGLIN-Qv17-LOCAL-DEV-Ed25519-signed-alternate-precision-packed-cache/1",
  "signature_algorithm":"Ed25519-local-developer-only",
  "signer_public_key_sha256":digest_bytes(pub),
  "model":"Qwen/Qwen2.5-1.5B-Instruct",
  "model_source_revision_label":MODEL_REVISION,
  "BF16_original_model_file_sha256":original_bf16_sha,
  "original_Qwen_hf_config_sha256":CONFIG_SHA,
  "frozen_Qv16_source_only_profile_manifest_sha256":FROZEN_SHA,
  "original_Qv16_two_role_layer_masks_SHA256":MASKS,
  "native_Qv17_executable_sha256":BIN_SHA,
  "native_Qv17_C_source_sha256":CSRC_SHA,
  "packed_file_sha256":packsha,"packed_layout":layout,
  "cache_cap_bytes":96*1048576,
  "source_CPU_engine_and_score_equations_unchanged":True,
  "this_is_a_research_local_dev_signature_not_a_production_certificate":True}
 content=canonical(manifest);signature=key.sign(content)
 put_new(MANIFEST,content);put_new(SIGNATURE,signature)
 print("QV17_ED25519_SIGNED",json.dumps({"manifest_sha256":digest_bytes(content),
 "detached_signature_sha256":digest_bytes(signature),"pack_sha256":packsha,
 "model_full_hash_ms":round(model_ms,3),"full_model_sha256":original_bf16_sha,
 "layout":layout,"developer_signature_only":True},sort_keys=True))
def verify(strict_model=True,alternate_pack=None):
 verify_roots();check_sources()
 pack=ARTIFACT if alternate_pack is None else pathlib.Path(alternate_pack)
 if pack!=ARTIFACT:
  # Alternate path is exclusively for tests of tamper rejection; never treated as loadable.
  raise RuntimeError("Qv17 pack must be the exact pinned owner-only artifact path")
 if not MANIFEST.is_file() or not SIGNATURE.is_file():raise RuntimeError("missing detached manifest signature")
 if any(p.is_symlink() for p in (MANIFEST,SIGNATURE,ARTIFACT)):raise RuntimeError("signed asset symlink forbidden")
 t=time.perf_counter()
 pub=Ed25519PublicKey.from_public_bytes(root_public())
 content=MANIFEST.read_bytes();sig=SIGNATURE.read_bytes()
 pub.verify(sig,content)
 doc=json.loads(content)
 if doc.get("schema")!="BEGLIN-Qv17-LOCAL-DEV-Ed25519-signed-alternate-precision-packed-cache/1" or doc["native_Qv17_executable_sha256"]!=BIN_SHA or doc["native_Qv17_C_source_sha256"]!=CSRC_SHA:
  raise RuntimeError("signed artifact code/model schema binding failed")
 if doc["original_Qwen_hf_config_sha256"]!=CONFIG_SHA or doc["original_Qv16_two_role_layer_masks_SHA256"]!=MASKS or doc["frozen_Qv16_source_only_profile_manifest_sha256"]!=FROZEN_SHA:
  raise RuntimeError("signed artifact masks/config/input identity binding failed")
 layout=verify_packed_layout()
 if layout!=doc["packed_layout"] or doc["cache_cap_bytes"]!=96*1048576:raise RuntimeError("signed packed 40 entry layout wrong")
 packsha=sha_file(ARTIFACT)
 if packsha!=doc["packed_file_sha256"]:raise RuntimeError("packed artifact content SHA does not match signed manifest")
 if strict_model:
  if sha_file(MODEL)!=doc["BF16_original_model_file_sha256"]:raise RuntimeError("source model file not identical to signed model")
 elapsed=(time.perf_counter()-t)*1000
 return {"status":"ED25519_LOCAL_DEVELOPER_SIGNATURE_AND_PACK_SHA256_VERIFIED",
 "pack_sha256":packsha,"signed_manifest_SHA256":sha_file(MANIFEST),
 "model_verified_full_content_sha256":bool(strict_model),
 "verifier_total_ms":round(elapsed,3),"payload_mib":layout["inactive_pack_mib"],
 "certificate_scope":"local research dev key only"}
if __name__=="__main__":
 cli=argparse.ArgumentParser()
 cli.add_argument("mode",choices=["setup","sign","verify","verify-no-model"])
 x=cli.parse_args()
 if x.mode=="setup":create_root()
 elif x.mode=="sign":sign()
 else:print("QV17_ED25519_VERIFIED",json.dumps(verify(strict_model=x.mode=="verify"),sort_keys=True))
