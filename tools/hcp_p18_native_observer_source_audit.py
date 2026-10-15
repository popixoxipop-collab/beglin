#!/usr/bin/env python3
"""Independently prove exact 673 frozen P18 token hashes in scratch C capture policy."""
import hashlib,json,pathlib,re
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
BANK_SHA="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
C_SHA="82475145a4351b934cfccda6763112cdf25751e2ca470ac8417a16e6dc8fdb1c"
MLX_SHA="a0122c8b2010573fb3b20b1c58ab4a3abb26b73602eddf92dd008dda7892f82b"
P17_SHA="a8e6e42b11a0f881317ab8afddebf7b266a965f4100bfddd7342183e73728521"
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def tok(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def run():
 bankfile=ROOT/"p18_multishot_frozen_PRIVATE.json"
 p=ROOT/"native_p18_capture_buildroot/qwen_infer.c"
 m=ROOT/"native_p18_capture_buildroot/mlx_moe.cpp"
 if (sha(bankfile)!=BANK_SHA or sha(p)!=C_SHA or sha(m)!=MLX_SHA):
  raise ValueError("P18 original frozen input/C/MLX SHA mismatch")
 j=json.loads(bankfile.read_text())
 expected=set()
 for row in j["items"]:
  for i in range(len(row["gold_ids"])):
   expected.add(tok(row["prompt_ids"]+row["gold_ids"][:i]))
 if len(expected)!=673:raise ValueError("P18 unexpected gold prefix count")
 c=p.read_text()
 start="/* BEGLIN_P18_SOURCE_HASH_ALLOWLIST_BEGIN:"
 end="/* BEGLIN_P18_SOURCE_HASH_ALLOWLIST_END */"
 if c.count(start)!=1 or c.count(end)!=1:raise ValueError("P18 allowlist markers absent")
 chunk=c.split(start,1)[1].split(end,1)[0]
 found=re.findall(r'"([a-f0-9]{64})"',chunk)
 if len(found)!=673 or len(set(found))!=673 or set(found)!=expected:
  raise ValueError("C generated 673 token hash list differs from frozen 480")
 fp=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/hcp-p17-unseen-multitoken-20261015/repository/experiments/hcp_p17/P17_FROZEN_TOKENS.json")
 if sha(fp)!=P17_SHA:raise ValueError("P17 reference SHA differs")
 old=json.loads(fp.read_text())
 old_prefixes={item["forced_prefixes"][i]["prefix_token_sha256"]
   for item in old["items"] for i in range(len(item["forced_prefixes"]))}
 if expected&old_prefixes:raise ValueError("P18 capture includes P17 old token hashes")
 scope="/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015/capture"
 checks={
  "strict_scope":c.count('static const char allowed_dir[]="'+scope+'";')==1,
  "separate_env":c.count('getenv("BEGLIN_P18_LOGITS_DIR")')==2,
  "size_guard":c.count("(plen<20 || plen>=1024)")>=1,
  "full_vocab_bytes":'MOE_VOCAB!=102400 || sizeof(float)!=4' in c,
  "hex_and_exclusive":'O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW' in c,
  "full_fd_sync":'fsync(fd)' in c,
  "research_scratch_optin":'BEGLIN_P18_SCRATCH_ONLY' in c,
  "native_log_witness":'P18_NATIVE_GOLD_PREFIX_LOGITS rid=' in c,
  "p18_only_allowlist_check":'if (!p18_prefix_allowed(tok_hash))' in c,
  "max_token_text_capacity":'char text[8192];' in c
 }
 if not all(checks.values()):raise ValueError("Mandatory P18 P16 source/exclusive file security rule absent")
 out={"schema":"beglin-hcp-p18-frozen-native-capture-source-audit-v1",
      "status":"P18_673_FROZEN_FULL_LOGIT_CAPTURE_SOURCE_AUDIT_PASS",
      "source_bank_sha256":BANK_SHA,"C_source_sha256":C_SHA,
      "MLX_source_sha256":MLX_SHA,
      "allowlisted_prefixes":len(found),"all_hashes_equal_frozen_bank":True,
      "P17_prefixes_excluded":len(old_prefixes),
      "capture_security_invariants":checks,
      "native_GPU_executed":False,"source_modified_in_existing_P17":False,
      "production_touched":False,"automatic_live_promotion":False}
 output=ROOT/"P18_NATIVE_OBSERVER_SOURCE_INDEPENDENT_AUDIT.json"
 if output.exists():raise FileExistsError("Source audit already exists")
 with output.open("x") as f:json.dump(out,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":out["status"],"verified_hashes":673,
     "P17_prefixes_excluded":len(old_prefixes),"security_check_pass":len(checks),
     "receipt_sha256":sha(output)},sort_keys=True))
if __name__=="__main__":run()
