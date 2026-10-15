#!/usr/bin/env python3
"""Read-only parity check of EOE exact-byte XOX tokenizer against frozen P17 inputs."""
import hashlib,json,pathlib,sys
from tokenizers import Tokenizer
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
FROZEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/hcp-p17-unseen-multitoken-20261015/repository/experiments/hcp_p17/P17_FROZEN_TOKENS.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashids(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def main():
 files={"tokenizer.json":"41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669",
        "tokenizer_config.json":"31181eaf79394ea26728d95ecb54fe7c8413e6f56085dbabc8b0818134380ec8",
        "config.json":"f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d"}
 for fn,expected in files.items():
  p=ROOT/"tokenizer_exact_XOX"/fn
  if not p.is_file() or p.is_symlink() or sha(p)!=expected:raise RuntimeError("Original XOX token source bytes changed "+fn)
 if sha(FROZEN)!="a8e6e42b11a0f881317ab8afddebf7b266a965f4100bfddd7342183e73728521":
  raise RuntimeError("P17 reference inputs are not original pinned corpus")
 data=json.loads(FROZEN.read_text())
 tok=Tokenizer.from_file(str(ROOT/"tokenizer_exact_XOX/tokenizer.json"))
 rows=[]
 for q in data["items"]:
  prompt=tok.encode(q["prompt"],add_special_tokens=True).ids
  both=tok.encode(q["prompt"]+q["reference_answer"],add_special_tokens=True).ids
  rows.append({"id":q["id"],
    "prompt_id_exact":prompt==q["prompt_ids"],
    "full_context_exact":both==q["teacher_full_context_ids"],
    "gold_id_exact":both[len(prompt):]==q["gold_answer_ids"],
    "prefix_stable":both[:len(prompt)]==prompt,
    "prompt_hash_exact":hashids(prompt)==q["prompt_token_sha256"],
    "first_expected":q["prompt_ids"][0],"first_observed":prompt[0] if prompt else None,
    "prompt_length":len(prompt),"expected_prompt_length":len(q["prompt_ids"])})
 summary={"schema":"beglin-hcp-p18-original-tokenizer-p17-parity-v1",
   "tokenizer_sha256":files["tokenizer.json"],"P17_frozen_sha256":sha(FROZEN),
   "P17_reference_items":len(rows),
   "exact_prompt_id":sum(x["prompt_id_exact"] for x in rows),
   "exact_full_context_id":sum(x["full_context_exact"] for x in rows),
   "exact_gold_id":sum(x["gold_id_exact"] for x in rows),
   "prefix_stable":sum(x["prefix_stable"] for x in rows),
   "per_item":rows,
   "original_model_forward_count":0,"production_touched":False}
 dst=ROOT/"p17_tokenizer_parity.json"
 with dst.open("x") as f:json.dump(summary,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps({"prompt_exact":summary["exact_prompt_id"],"context_exact":summary["exact_full_context_id"],
 "gold_exact":summary["exact_gold_id"],"total":len(rows),"receipt_sha256":sha(dst)},sort_keys=True))
 if any(not (r["prompt_id_exact"] and r["full_context_exact"] and r["gold_id_exact"]) for r in rows):
  return 2
 return 0
if __name__=="__main__":sys.exit(main())
