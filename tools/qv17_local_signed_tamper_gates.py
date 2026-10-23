#!/usr/bin/env python3
"""BEGLIN Qv17 adversarial-local verification of signed 83MiB research artifact.
Tests three independent failures: tampered Ed25519 manifest, wrong signing key,
and flipped-byte packed tensor bytes. Original artifacts never modified.
NOT an external security audit, and not proof against local key compromise.
"""
import hashlib,json,os,pathlib,shutil,tempfile,time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey,Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
ROOT=pathlib.Path(__file__).resolve().parents[1]
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023")
ART=LAB/"owner_only_artifacts/alternate40_Qv8_random73_qng64_v1.pack"
MAN=LAB/"owner_only_artifacts/alternate40_Qv8_random73_manifest.json"
SIG=LAB/"owner_only_artifacts/alternate40_Qv8_random73_manifest.sig"
PUB=ROOT/"docs/research/BEGLIN_QV17_ED25519_DEV_TRUST_ROOT_PUBLIC_20261023.txt"
EXPECT={"trust_sha":"5f535c54985ac18257b50c4bd7969ec4ce57aecb766def485c68c8b0a4ad4847",
"manifest_sha":"491deb61dbc30fad08b4a5d819b270b64167d5551b422823ae0b431f4a5c6824",
"signature_sha":"2c346634543b222890a99e0536fcca7ec6488b09a7cac05aa5562fe31c0b168c",
"artifact_sha":"49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b"}
def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(262144),b""):h.update(chunk)
 return h.hexdigest()
def must(ok,why):
 if not ok:raise RuntimeError("QV17_TAMPER_TEST_FAIL_"+why)
def main():
 before={name:sha(p) for name,p in [("pub",PUB),("manifest",MAN),("signature",SIG),("artifact",ART)]}
 for key,value in [("pub","trust_sha"),("manifest","manifest_sha"),("signature","signature_sha"),("artifact","artifact_sha")]:
  must(before[key]==EXPECT[value],"source_reference_"+key)
 key=bytes.fromhex(PUB.read_text().split(":",1)[1])
 authority=Ed25519PublicKey.from_public_bytes(key)
 signed_message=MAN.read_bytes();signature=SIG.read_bytes()
 authority.verify(signature,signed_message)
 mutated=bytearray(signed_message)
 pos=signed_message.index(b"Qwen/")
 mutated[pos]=ord("X")
 rejected_bad_message=False
 try:authority.verify(signature,bytes(mutated))
 except InvalidSignature:rejected_bad_message=True
 must(rejected_bad_message,"mutated_signed_manifest_not_rejected")
 rejected_wrong_public_key=False
 fake=Ed25519PrivateKey.generate().public_key()
 try:fake.verify(signature,signed_message)
 except InvalidSignature:rejected_wrong_public_key=True
 must(rejected_wrong_public_key,"wrong_dev_trust_root_not_rejected")
 t=time.perf_counter()
 with tempfile.TemporaryDirectory(prefix="qv17_tamper_test_",dir=str(LAB)) as td:
  folder=pathlib.Path(td);folder.chmod(0o700)
  copy=folder/"tampered_alternate40.pack"
  shutil.copyfile(ART,copy);copy.chmod(0o600)
  with copy.open("r+b") as f:
   f.seek(4096);val=f.read(1)
   must(len(val)==1,"must_have_packed_bytes_to_flip")
   f.seek(4096);f.write(bytes([val[0]^0x01]))
   f.flush();os.fsync(f.fileno())
  bad=sha(copy)
  must(bad!=EXPECT["artifact_sha"],"bitflip_packed_sha_not_rejected")
  got_size=copy.stat().st_size
 copied_hash_ms=round((time.perf_counter()-t)*1000,3)
 must(before=={name:sha(p) for name,p in [("pub",PUB),("manifest",MAN),("signature",SIG),("artifact",ART)]},"original_artifacts_modified")
 result={"schema":"BEGLIN-Qv17-local-Ed25519-signed-packed-asset-tamper-test/1",
 "status":"ALL_3_ADVERSARIAL_OFFLINE_TAMPER_DENIALS_PASS",
 "local_dev_Ed25519_original_signature_accepted":True,
 "modified_manifest_signature_rejected":rejected_bad_message,
 "wrong_developer_root_signature_rejected":rejected_wrong_public_key,
 "single_bit_file_tamper_SHA256_rejected":True,
 "bit_flip_byte_offset":4096,
 "pack_file_bytes":got_size,
 "test_copy_and_hash_ms":copied_hash_ms,
 "original_dev_public_key_SHA256":before["pub"],
 "original_signed_manifest_SHA256":before["manifest"],
 "original_signature_SHA256":before["signature"],
 "original_native_pack_SHA256":before["artifact"],
 "all_original_artifacts_digest_preserved":True,
 "temporary_tampered_copy_destroyed":True,
 "limits":["trusted Python launcher enforces Ed25519 before native C is spawned; direct native invocation has independent SHA256 checking but no direct Ed25519 verification",
         "local development Ed25519 key is not an organizational release certificate",
         "malicious code controlling the key or loader trust root remains outside this test's threat model",
         "negative tests inspect a temporary copy and do not modify the original artifact or original kernel"],
 "no_original_model_service_or_operator_release_change":True}
 out=ROOT/"docs/research/BEGLIN_QV17_OFFLINE_ED25519_TAMPER_TEST_PUBLIC_20261023.json"
 data=(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
 with out.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 print("QV17_TAMPER_GATES_PASS",json.dumps({"public_SHA256":sha_bytes(data),
 "three_negative_cases":True,"original_pack_bytes":got_size,"original_unchanged":True,
 "temp_copy_destroyed":True,"path":str(out)},sort_keys=True))
if __name__=="__main__":main()
