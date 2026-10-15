#!/usr/bin/env python3
"""Independently audit original P17 27-layer BF16 teacher vs P18 long-window copy.
Exactly two source changes are allowed:
  class GoldWindowTeacher(StreamedTeacher) -> subclass of original class
  input position bound >112 -> >1024
The method's remaining code must be byte-identical to SHA-pinned P17 source.
"""
import hashlib,json,pathlib

ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
ORIGINAL=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/hcp-p17-unseen-multitoken-20261015/repository/tools/hcp_p17_teacher_gold.py")
DERIVED=ROOT/"hcp_p18_extended_original_bf16_teacher_core.py"
O_SHA="40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3"
D_SHA="c2a631ee7d82feef83efce59087b0a144f7561b9da26db920e1201b45b689820"
def sha(p):
 if not p.is_file() or p.is_symlink():raise RuntimeError("Source absent/symlink")
 return hashlib.sha256(p.read_bytes()).hexdigest()
def audit():
 if sha(ORIGINAL)!=O_SHA or sha(DERIVED)!=D_SHA:
  raise RuntimeError("Original P17 / derived P18 class source SHA drift")
 original=ORIGINAL.read_text()
 begin=original.index("class GoldWindowTeacher(StreamedTeacher):")
 end=original.index("\ndef run(index):",begin)
 class_original=original[begin:end]
 derived=DERIVED.read_text().split("class P18LongGoldWindowTeacher(og.GoldWindowTeacher):",1)[1]
 compare="class GoldWindowTeacher(StreamedTeacher):"+derived
 if compare.count("ids.shape[1]>1024")!=1 or compare.replace("ids.shape[1]>1024","ids.shape[1]>112") .strip()!=class_original.strip():
  raise RuntimeError("P18 extended teacher changed semantics beyond source limit")
 result={"schema":"beglin-hcp-p18-original-bf16-112-to-1024-source-exact-diff-v1",
  "status":"P18_ORIGINAL_BF16_27L_SOURCE_EXACT_PARITY_EXCEPT_POSITION_LIMIT_PASS",
  "original_P17_teacher_sha256":O_SHA,"new_P18_teacher_sha256":D_SHA,
  "only_class_parent_and_position_constant_changed":True,
  "original_teacher_context_limit":112,"extended_teacher_context_limit":1024,
  "original_layer_forward_formula_identical":True,
  "original_P17_paths_written":False,"production_touched":False,
  "automatic_live_promotion":False}
 dest=ROOT/"P18_EXTENDED_BF16_EXACT_SOURCE_DIFF_AUDIT.json"
 if dest.exists():raise FileExistsError("Immutable class source receipt already exists")
 with dest.open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":result["status"],"receipt_sha256":sha(dest)},sort_keys=True))
if __name__=="__main__":audit()
