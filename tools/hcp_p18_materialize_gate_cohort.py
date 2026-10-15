#!/usr/bin/env python3
"""Create a P18 no-raw-text 480-question *gate* cohort from independently audited token bank."""
import hashlib,json,pathlib,sys
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
sys.path.insert(0,str(ROOT))
from hcp_p18_quality_gate import frozen_cohort
CANDIDATE="ea7cb77649fcdc285393234e019d56db0d7178fb7787cf23268a7c52c072bb5a"
FROZEN="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
AUDIT="4e43064669116c9e4c670f969e230c0ddb29a1535f9b191c89f68f9be0a91ea1"
DEVSHA="2d9f6e63fe49cfe646493f0e3e79870ff2eca34adac5081b3bb8fb305b28d836"
def sha(path):
 p=pathlib.Path(path)
 if p.is_symlink() or not p.is_file():raise ValueError("source must be regular file")
 return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if sha(ROOT/"candidate_bank_PRIVATE.json")!=CANDIDATE or sha(ROOT/"p18_multishot_frozen_PRIVATE.json")!=FROZEN or sha(ROOT/"p18_multishot_independent_audit.json")!=AUDIT or sha(ROOT/"dev_source_manifest.json")!=DEVSHA:
  raise ValueError("independent P18 cohort inputs are not exact")
 freeze=json.loads((ROOT/"p18_multishot_frozen_PRIVATE.json").read_text())
 audit=json.loads((ROOT/"p18_multishot_independent_audit.json").read_text())
 protocol=json.loads((ROOT/"p18_protocol_DRAFT.json").read_text())
 if freeze["state"]!="P18_COHORT_TOKEN_FROZEN_NATIVE_CAPABILITY_PENDING" or audit["status"]!="P18_480_TOKEN_AND_SOURCE_INDEPENDENT_AUDIT_PASS":
  raise ValueError("freeze or independent audit is not complete")
 keep=("item_id","dataset","dataset_revision","split","record_sha256","prompt_token_sha256",
       "gold_token_sha256","gold_token_count","prompt_token_count","max_new_tokens","decode_steps","shot_count")
 cohort={"schema":"beglin-hcp-p18-frozen-external-cohort-v1",
  "status":"PRE_FROZEN_SHA_BOUND_EXTERNAL_COHORT",
  "original_token_exact_bank_sha256":FROZEN,
  "independent_input_audit_sha256":AUDIT,
  "source_test_selection_sha256":CANDIDATE,
  "source_dev_manifest_sha256":DEVSHA,
  "original_tokenizer_sha256":freeze["original_tokenizer_sha256"],
  "external_public_benchmark_training_contamination_unknown":True,
  "native_context_4096_verified":False,
  "automatic_live_promotion":False,"production_touched":False,
  "items":[{k:r[k] for k in keep} for r in freeze["items"]]}
 checked=frozen_cohort(cohort,protocol)
 if len(checked)!=480:raise ValueError("P18 statistics contract cannot accept cohort")
 target=ROOT/"p18_gate_cohort_PRIVATE.json"
 if target.exists():raise FileExistsError("refusing re-freeze")
 with target.open("x") as f:json.dump(cohort,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":"P18_480_COHORT_GATE_SCHEMA_VALIDATED","count":len(checked),
                   "gate_cohort_sha256":sha(target),"exact_input_sha256":FROZEN,
                   "independent_audit_sha256":AUDIT,"native_4096_verified":False},sort_keys=True))
if __name__=="__main__":main()
