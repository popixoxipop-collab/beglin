#!/usr/bin/env python3
"""BEGLIN Qv18 OFFLINE simulated signing-key rotation and rollback policy:
- Actual Qv17 local Ed25519 developer signed manifest must be validated.
- Generate second ED25519 developer key IN MEMORY only, never serialize secret.
- Signed release envelope covers artifact/mask/C/model digests + revision/epoch.
- Reject untrusted key, revoked original key, stale revision, wrong model/binary,
  unsigned modification and digest mutation.
No claim of actual production key rotation, tamper-resistant antirollback store or CA.
No original signed model/artifact/keys are modified.
"""
import base64,hashlib,json,pathlib
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey,Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
REPO=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv13-github-commit-20261023")
BASE=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023")
PUB=REPO/"docs/research/BEGLIN_QV17_ED25519_DEV_TRUST_ROOT_PUBLIC_20261023.txt"
MAN=BASE/"owner_only_artifacts/alternate40_Qv8_random73_manifest.json"
SIG=BASE/"owner_only_artifacts/alternate40_Qv8_random73_manifest.sig"
PACK=BASE/"owner_only_artifacts/alternate40_Qv8_random73_qng64_v1.pack"
ORIG_PUBLIC_SHA="5f535c54985ac18257b50c4bd7969ec4ce57aecb766def485c68c8b0a4ad4847"
ORIG_PACK_SHA="49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b"
ORIG_MAN_SHA="491deb61dbc30fad08b4a5d819b270b64167d5551b422823ae0b431f4a5c6824"
def sha(b):return hashlib.sha256(b).hexdigest()
def shaf(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for x in iter(lambda:f.read(262144),b""):h.update(x)
 return h.hexdigest()
def need(ok,why):
 if not ok:raise RuntimeError("QV18_TRUST_ROTATION_TEST_FAILED_"+why)
def canon(x):return (json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()
def check_envelope(blob,sig,key_id,trust_state):
 keys=trust_state["trusted_keys"]
 if key_id not in keys:raise PermissionError("UNTRUSTED_OR_REVOKED_KEY")
 Ed25519PublicKey.from_public_bytes(keys[key_id]).verify(sig,blob)
 d=json.loads(blob)
 if d["schema"]!="BEGLIN-Qv18-research-signed-artifact-envelope-v1":
  raise PermissionError("WRONG_SIGNED_ENVELOPE_SCHEMA")
 if d["sequence"]<trust_state["minimum_accepted_sequence"]:
  raise PermissionError("ROLLBACK_SEQUENCE_REJECTED")
 if d["artifact_sha256"]!=ORIG_PACK_SHA:raise PermissionError("WRONG_ARTIFACT_DIGEST")
 if d["model_full_sha256"]!=trust_state["model_sha256"]:raise PermissionError("WRONG_MODEL_BINDING")
 if d["native_binary_sha256"]!=trust_state["native_sha256"]:raise PermissionError("WRONG_BINARY_BINDING")
 if d["source_mask_sha256"]!=trust_state["source_mask_sha256"] or d["alternate_mask_sha256"]!=trust_state["alternate_mask_sha256"]:raise PermissionError("WRONG_MASK_BINDING")
 return d
def expected_reject(fn,types,reason):
 try:fn()
 except types:return True
 raise RuntimeError("Qv18 supposed denial NOT observed: "+reason)
def main():
 original_paths={k:v for k,v in [("public",PUB),("manifest",MAN),("signature",SIG),("packed",PACK)]}
 orig={k:shaf(v) for k,v in original_paths.items()}
 need(orig["public"]==ORIG_PUBLIC_SHA and orig["manifest"]==ORIG_MAN_SHA and orig["packed"]==ORIG_PACK_SHA,"input_source_digest")
 doc=json.loads(MAN.read_text())
 original_pub=bytes.fromhex(PUB.read_text().strip().split(":",1)[1])
 Ed25519PublicKey.from_public_bytes(original_pub).verify(SIG.read_bytes(),MAN.read_bytes())
 need(doc["packed_file_sha256"]==ORIG_PACK_SHA,"real original developer signed artifact")
 # New developer key exists ONLY in Python RAM for this unit test. Do not serialize.
 private2=Ed25519PrivateKey.generate()
 public2=private2.public_key().public_bytes_raw()
 original_model_sha=doc["BF16_original_model_file_sha256"]
 original_native_sha=doc["native_Qv17_executable_sha256"]
 signed_payload={"schema":"BEGLIN-Qv18-research-signed-artifact-envelope-v1",
  "sequence":18,"artifact_sha256":ORIG_PACK_SHA,
  "model_full_sha256":original_model_sha,
  "native_binary_sha256":original_native_sha,
  "source_mask_sha256":doc["original_Qv16_two_role_layer_masks_SHA256"]["source_qv8"],
  "alternate_mask_sha256":doc["original_Qv16_two_role_layer_masks_SHA256"]["random73"],
  "intent":"EPHEMERAL_TEST_ONLY"}
 payload=canon(signed_payload);signature2=private2.sign(payload)
 v1={"trusted_keys":{"local_dev_v1":original_pub},
     "minimum_accepted_sequence":17,
     "model_sha256":original_model_sha,
     "native_sha256":original_native_sha,
     "source_mask_sha256":signed_payload["source_mask_sha256"],
     "alternate_mask_sha256":signed_payload["alternate_mask_sha256"]}
 v2={"trusted_keys":{"ephemeral_local_dev_v2":public2},
     "minimum_accepted_sequence":18,
     "model_sha256":original_model_sha,
     "native_sha256":original_native_sha,
     "source_mask_sha256":signed_payload["source_mask_sha256"],
     "alternate_mask_sha256":signed_payload["alternate_mask_sha256"]}
 # v1 still accepts its EXISTING signed artifact; not an unsigned wrapper.
 Ed25519PublicKey.from_public_bytes(v1["trusted_keys"]["local_dev_v1"]).verify(SIG.read_bytes(),MAN.read_bytes())
 checks={
  "new_v2_key_REJECTED_under_unrotated_v1_roots":expected_reject(lambda:check_envelope(payload,signature2,"ephemeral_local_dev_v2",v1),(PermissionError,), "untrusted key"),
  "valid_v2_release_ACCEPTED_after_trust_root_rotation":check_envelope(payload,signature2,"ephemeral_local_dev_v2",v2)["sequence"]==18,
  "old_v1_key_REVOKED_in_new_trust_store":"local_dev_v1" not in v2["trusted_keys"],
 }
 stale={**signed_payload,"sequence":17}
 stale_bytes=canon(stale);stale_sig=private2.sign(stale_bytes)
 checks["authentic_but_stale_sequence_REJECTED"]=expected_reject(
  lambda:check_envelope(stale_bytes,stale_sig,"ephemeral_local_dev_v2",v2),(PermissionError,),"signed rollback")
 bad_model={**signed_payload,"model_full_sha256":"0"*64}
 bm=canon(bad_model)
 checks["authentic_but_wrong_model_REJECTED"]=expected_reject(
  lambda:check_envelope(bm,private2.sign(bm),"ephemeral_local_dev_v2",v2),(PermissionError,),"model")
 bad_bin={**signed_payload,"native_binary_sha256":"f"*64}
 bb=canon(bad_bin)
 checks["authentic_but_wrong_native_binary_REJECTED"]=expected_reject(
  lambda:check_envelope(bb,private2.sign(bb),"ephemeral_local_dev_v2",v2),(PermissionError,),"C binary")
 altered=payload.replace(b'"sequence":18',b'"sequence":19')
 need(altered!=payload,"test must mutate actual signed envelope")
 checks["tampered_epoch_without_resigning_REJECTED"]=expected_reject(
  lambda:check_envelope(altered,signature2,"ephemeral_local_dev_v2",v2),(InvalidSignature,),"unsigned sequence change")
 checks["wrong_signature_public_key_REJECTED"]=expected_reject(
  lambda:check_envelope(payload,signature2,"local_dev_v1",v1),(InvalidSignature,),"other signer id")
 mutated_pack_sha=sha(bytes.fromhex(ORIG_PACK_SHA)+b"modified")
 checks["wrong_packed_artifact_digest_REJECTED"]=expected_reject(
  lambda:check_envelope(canon({**signed_payload,"artifact_sha256":mutated_pack_sha}),
        private2.sign(canon({**signed_payload,"artifact_sha256":mutated_pack_sha})),"ephemeral_local_dev_v2",v2),
  (PermissionError,),"wrong packed digest")
 wrong_mask=canon({**signed_payload,"source_mask_sha256":"0"*64})
 checks["authentic_but_wrong_mask_REJECTED"]=expected_reject(
  lambda:check_envelope(wrong_mask,private2.sign(wrong_mask),"ephemeral_local_dev_v2",v2),(PermissionError,),"source precision mask")
 need(all(checks.values()),"all policy cases")
 need(orig=={k:shaf(v) for k,v in original_paths.items()},"original signed assets untouched")
 report={
 "schema":"BEGLIN-Qv18-offline-local-developer-key-revision-policy-negative-unit-test/1",
 "status":"ALL_10_ED25519_ROTATION_REVOCATION_ROLLBACK_BINDING_TESTS_PASS",
 "test_cases":checks,
 "original_developer_public_trust_root_sha256":ORIG_PUBLIC_SHA,
 "original_Qv17_signed_manifest_SHA256":ORIG_MAN_SHA,
 "original_real_signed_artifact_SHA256":ORIG_PACK_SHA,
 "ephemeral_second_public_key_sha256":sha(public2),
 "ephemeral_second_private_key_stored_on_disk":False,
 "new_dev_v2_signed_release_envelope_sha256":sha(payload),
 "Qv18_antireplay_requirement":"In real service, a durable tamper-resistant monotonic minimum revision is required; this test is an IN-MEMORY trust-store simulation only",
 "new_production_release_key_issued":False,
 "existing_Qv17_original_root_revoked_in_operation":False,
 "original_signed_artifacts_unchanged":True,
 "no_model_or_operator_release_change":True}
 p=REPO/"docs/research/BEGLIN_QV18_KEY_ROTATION_ANTIROLLBACK_V2_PUBLIC_20261023.json"
 blob=(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
 with p.open("xb") as f:f.write(blob);f.flush()
 print("QV18_OFFLINE_TRUST_LIFECYCLE_PASS",json.dumps({
  "cases":len(checks),"all_pass":all(checks.values()),"source_sha256_unchanged":True,
  "public_receipt_sha256":sha(blob),"new_dev_private_written":False
 },sort_keys=True),flush=True)
if __name__=="__main__":main()
