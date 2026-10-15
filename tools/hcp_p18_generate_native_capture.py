#!/usr/bin/env python3
"""Generate a separate SHA-pinned P18 FULL-logit observer; never patch P17.

Only the 673 preregistered P18 teacher-forced gold prefix token SHAs are allowed.
Research binary requires its existing BEGLIN_P18_SCRATCH_ONLY=1 fence.
Strict P18 path, exclusive 409600-byte source logit file, no generic capture.
"""
from __future__ import annotations
import hashlib,json,pathlib,re,shutil
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
BASE=ROOT/"native_p1024_buildroot"
DEST=ROOT/"native_p18_capture_buildroot"
SOURCE_SHA="4887485be4c5ea13be7e14a33bafd4eb153a6a9439929cced5586608ede5c3b9"
MODEL_SHA="a0122c8b2010573fb3b20b1c58ab4a3abb26b73602eddf92dd008dda7892f82b"
BANK_SHA="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
CAPTURE="/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015/capture"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def thash(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def replace_unique(s,a,b,detail):
 if s.count(a)!=1:raise ValueError("Unexpected source count for "+detail+": "+str(s.count(a)))
 return s.replace(a,b,1)
def main():
 if DEST.exists() or not BASE.is_dir() or not ROOT.is_dir():
  raise ValueError("No existing/overwritten P18 capture source and authentic source required")
 if sha(BASE/"qwen_infer.c")!=SOURCE_SHA or sha(BASE/"mlx_moe.cpp")!=MODEL_SHA or sha(ROOT/"p18_multishot_frozen_PRIVATE.json")!=BANK_SHA:
  raise ValueError("P18 native baseline and frozen pre-registration changed")
 src=json.loads((ROOT/"p18_multishot_frozen_PRIVATE.json").read_text())
 items=src.get("items",[])
 if len(items)!=480 or src.get("question_count")!=480:raise ValueError("P18 source has no 480 question units")
 hashes=[]
 for r in items:
  ids=r["prompt_ids"];gold=r["gold_ids"]
  if not 1<=len(gold)<=8 or len(ids)+len(gold)>1024 or r["prompt_token_sha256"]!=thash(ids) or r["gold_token_sha256"]!=thash(gold):
   raise ValueError("P18 source item token/hash/length changed")
  for i in range(len(gold)):
   h=thash(ids+gold[:i])
   hashes.append(h)
 if len(hashes)!=673 or len(set(hashes))!=673:raise ValueError("P18 frozen 673 unique gold-prefix hash count violated")
 code=(BASE/"qwen_infer.c").read_text()
 insert='''\n/* BEGLIN_P18_SOURCE_HASH_ALLOWLIST_BEGIN: generated from immutable 480 question items. */\n'''
 insert+='static int p18_prefix_allowed(const char *hex) {\n'
 insert+='    static const char *const hashes[] = {\n'
 insert+="".join('        "'+h+'",\n' for h in sorted(hashes))
 insert+='    };\n'
 insert+='    if(!hex || strlen(hex)!=64) return 0;\n'
 insert+='    for(size_t i=0;i<sizeof(hashes)/sizeof(hashes[0]);i++)\n'
 insert+='        if(strcmp(hex,hashes[i])==0) return 1;\n'
 insert+='    return 0;\n}\n'
 insert+='/* BEGLIN_P18_SOURCE_HASH_ALLOWLIST_END */\n'
 code=replace_unique(code,"static void p16_write_native_logits(",insert+"static void p16_write_native_logits(","P18 capture allowlist function before call")
 code=replace_unique(code,'static const char allowed_dir[]="/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p17-unseen-multitoken-20261015/capture";',
                     'static const char allowed_dir[]="'+CAPTURE+'";',"capture path")
 code=replace_unique(code,'(plen<20 || plen>112)','(plen<20 || plen>=1024)',"P18 maximum prefix bound")
 code=replace_unique(code,'char text[1024];','char text[8192];',"1024 maximum token preimage buffer")
 code=replace_unique(code,"if (!p17_prefix_allowed(tok_hash)) {",
                     "if (!p18_prefix_allowed(tok_hash)) {","P18 SHA allowlist use")
 code=replace_unique(code,"P17_NATIVE_GOLD_PREFIX_LOGITS rid=","P18_NATIVE_GOLD_PREFIX_LOGITS rid=","P18 actual native log witness")
 if code.count('BEGLIN_P17_LOGITS_DIR')!=2:raise ValueError("Unexpected opt-in capture env references")
 code=code.replace('BEGLIN_P17_LOGITS_DIR','BEGLIN_P18_LOGITS_DIR')
 if not all(v in code for v in ('BEGLIN_P18_SCRATCH_ONLY','P18_NATIVE_GOLD_PREFIX_LOGITS','p18_prefix_allowed','char text[8192]')):
  raise ValueError("P18 capture security fence or source signature missing")
 DEST.mkdir()
 for item in BASE.iterdir():
  if item.name in ("build", ".git") :continue
  if item.is_dir():shutil.copytree(item,DEST/item.name,symlinks=True)
  else:shutil.copy2(item,DEST/item.name)
 (DEST/"qwen_infer.c").write_text(code)
 manifest={"schema":"beglin-hcp-p18-native-one-owner-raw-logits-source-v1",
  "status":"P18_FROZEN_TOKEN_EXACT_NATIVE_LOGITS_OBSERVER_SOURCE_PREPARED_NOT_EXECUTED",
  "frozen_P18_bank_sha256":BANK_SHA,"parent_native_C_sha256":SOURCE_SHA,
  "parent_native_MLX_sha256":MODEL_SHA,"allowed_gold_prefix_count":len(hashes),
  "allowed_capture_path":CAPTURE,"out_of_scope_P17_prompt_capture_allowed":False,
  "native_1024_candidate_C_SHA256":sha(DEST/"qwen_infer.c"),
  "native_1024_candidate_MLX_SHA256":sha(DEST/"mlx_moe.cpp"),
  "original_27layer_BF16_source_not_changed":True,
  "automatic_live_promotion":False,"production_touched":False}
 with (DEST/"P18_CAPTURE_SOURCE_MANIFEST.json").open("x") as f:
  json.dump(manifest,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":manifest["status"],"allowlisted_gold_prefixes":len(hashes),
   "candidate_C_SHA256":manifest["native_1024_candidate_C_SHA256"],
   "candidate_MLX_SHA256":manifest["native_1024_candidate_MLX_SHA256"]},sort_keys=True))
if __name__=="__main__":main()
