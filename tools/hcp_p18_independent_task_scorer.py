#!/usr/bin/env python3
"""P18 independent benchmark-output scoring rules and *completion* contract.

Ground truth always comes from the SHA-pinned external public test selection,
and all model tokens are decoded with the hash-pinned ORIGINAL P17 tokenizer.
This module cannot claim that a model produced supplied tokens: a separate,
SHA-verified native/teacher generation trace is required for authoritative
task accuracy. Native 1-token probes with cap 8 are ALWAYS incomplete.
Never authorizes production.
"""
from __future__ import annotations
import decimal,hashlib,json,pathlib,re
from tokenizers import Tokenizer

BASE=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
SOURCES={
 "candidate_bank_PRIVATE.json":"ea7cb77649fcdc285393234e019d56db0d7178fb7787cf23268a7c52c072bb5a",
 "p18_multishot_frozen_PRIVATE.json":"c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df",
 "p18_gate_cohort_PRIVATE.json":"19e7080b7208b8c7dd6fd06a656bb400dd2a9f72ca5f5674a80f5cc9a654cf65",
 "p18_protocol_FROZEN.json":"77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c",
 "p18_multishot_independent_audit.json":"4e43064669116c9e4c670f969e230c0ddb29a1535f9b191c89f68f9be0a91ea1",
 "p17_tokenizer_parity.json":"3b7f99193e26c7bc6bca6c9cf8252256b880ed97f66e15d94836fa06ee0ad5f9",
 "tokenizer_exact_XOX/tokenizer.json":"41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"}
SHA_RE=re.compile(r"[0-9a-f]{64}")
DECIMAL_RE=re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?",re.ASCII)
LETTER_RE=re.compile(r"[A-E]",re.ASCII)
EOS_TOKEN_ID=100001
BOS_TOKEN_ID=100000
MAX_VOCAB=102400

class SourceOrOutputError(ValueError):pass
def digest(path):
 p=pathlib.Path(path)
 if not p.is_file() or p.is_symlink():raise SourceOrOutputError("missing/symlink source")
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def shatext(value):return hashlib.sha256(value.encode("utf-8")).hexdigest()
def shaids(ids):
 return hashlib.sha256(",".join(map(str,ids)).encode("ascii")).hexdigest()
def fail(why):raise SourceOrOutputError(why)
def pin_authority(base=BASE):
 base=pathlib.Path(base)
 if not base.is_dir() or base.is_symlink():fail("P18 private authority directory missing")
 for name,expected in SOURCES.items():
  if digest(base/name)!=expected:fail("immutable P18 authority SHA mismatch: "+name)
 audit=json.loads((base/"p18_multishot_independent_audit.json").read_text())
 parity=json.loads((base/"p17_tokenizer_parity.json").read_text())
 if audit.get("audited_question_count")!=480 or parity.get("exact_prompt_id")!=12 or parity.get("exact_gold_id")!=12:
  fail("P18 source or P17 tokenizer independent audit not passed")
 t=Tokenizer.from_file(str(base/"tokenizer_exact_XOX/tokenizer.json"))
 raw=json.loads((base/"candidate_bank_PRIVATE.json").read_text())
 frozen=json.loads((base/"p18_multishot_frozen_PRIVATE.json").read_text())
 cohort=json.loads((base/"p18_gate_cohort_PRIVATE.json").read_text())
 if len(raw["items"])!=480 or len(frozen["items"])!=480 or len(cohort["items"])!=480:
  fail("not the preregistered 480-question source")
 byid={}
 original={v["item_id"]:v for v in raw["items"]}
 score={v["item_id"]:v for v in cohort["items"]}
 for f in frozen["items"]:
  ident=f["item_id"]
  if ident in byid or ident not in original or ident not in score:fail("duplicate/missing original source")
  ref=original[ident];q=score[ident]
  if (ref["record_sha256"]!=f["record_sha256"] or
      ref["dataset"]!=f["dataset"] or
      ref["dataset_revision"]!=f["dataset_revision"] or
      shatext(ref["reference_answer"])!=f["gold_text_sha256"] or
      ref["task_scoring"]!=f["task_scoring"]):
   fail("original test source and frozen answer identity mismatch "+ident)
  for key in ("prompt_token_sha256","gold_token_sha256","record_sha256","dataset","gold_token_count","max_new_tokens"):
   if f[key]!=q[key]:fail("frozen score cohort item identity differs "+ident+" "+key)
  gold=f["gold_ids"]
  if (shaids(gold)!=f["gold_token_sha256"] or
      shaids(f["prompt_ids"])!=f["prompt_token_sha256"] or
      t.decode(gold,skip_special_tokens=True)!=ref["reference_answer"] or
      not all(type(v)==int and 0<=v<MAX_VOCAB for v in gold)):
   fail("gold source tokenizer BPE/reference text drift "+ident)
  byid[ident]=(ref,f)
 return t,byid

def numeric_value(s):
 if not DECIMAL_RE.fullmatch(s):return None
 # only the valid standard comma grouping accepted; no units/chain-of-thought/####.
 try:
  out=decimal.Decimal(s.replace(",",""))
  if not out.is_finite():return None
  return out
 except decimal.InvalidOperation:return None

def score_complete(t,ref,f,generated,finish_reason):
 """Pure testable exact-format correctness function; does NOT attest authenticity."""
 if not isinstance(generated,list) or not generated:fail("missing actual generated token sequence")
 if (len(generated)>f["max_new_tokens"] or
     not all(type(v)==int and 0<=v<MAX_VOCAB for v in generated)):
  fail("out-of-range or over-cap generated token sequence")
 if BOS_TOKEN_ID in generated:fail("unexpected model BOS inside generated answer")
 if EOS_TOKEN_ID in generated[:-1]:fail("EOS inside generated answer (trailing tokens invalid)")
 if finish_reason=="eos":
  if generated[-1]!=EOS_TOKEN_ID:fail("claimed EOS termination without real terminal EOS token")
 elif finish_reason=="max_new_tokens":
  if len(generated)!=f["max_new_tokens"] or EOS_TOKEN_ID in generated:
   fail("claimed full-cap generation has wrong token count or contains terminal EOS")
 else:fail("no independently established complete-generation finish reason")
 decoded=t.decode(generated,skip_special_tokens=True)
 answer=decoded.strip()
 expected=ref["reference_answer"]
 if f["task_scoring"]=="letter_exact":
  correct=bool(LETTER_RE.fullmatch(answer)) and answer==expected
 elif f["task_scoring"]=="numeric_exact":
  exp=numeric_value(expected);actual=numeric_value(answer)
  if exp is None:fail("source benchmark numeric gold was not standard finite Decimal")
  correct=actual is not None and actual==exp
 else:fail("unsupported frozen task correctness mode")
 return {"item_id":f["item_id"],"dataset":f["dataset"],
  "source_record_sha256":f["record_sha256"],
  "prompt_token_sha256":f["prompt_token_sha256"],
  "gold_token_sha256":f["gold_token_sha256"],
  "generated_token_sha256":shaids(generated),
  "generated_token_count":len(generated),
  "finished_with":finish_reason,
  "decoded_output_sha256":shatext(decoded),
  "correct":bool(correct),
  "full_generation_finish_contract_verified":True,
  "model_provenance_verified_by_this_function":False}

def audit_raw_generation_record(t,byid,record):
 """Read signed/externally attested generation elsewhere; checks frozen content only.

 An arbitrary manifest is NOT proof of authentic native or BF16 execution. The
 caller must independently cross-check raw token output files against C/MLX
 logs/ACK, model SHA, PID, clock, complete generation and custody receipts.
 """
 if not isinstance(record,dict):fail("malformed actual generation row")
 item_id=record.get("item_id")
 if item_id not in byid:fail("generation item not in the fixed 480")
 ref,f=byid[item_id]
 if(record.get("dataset")!=f["dataset"] or
    record.get("record_sha256")!=f["record_sha256"] or
    record.get("prompt_token_sha256")!=f["prompt_token_sha256"] or
    record.get("policy") not in ("BF16","A","B")):
  fail("generation dataset/policy/source item not frozen")
 ids=record.get("generated_token_ids")
 if not isinstance(ids,list) or shaids(ids)!=record.get("generated_token_sha256"):
  fail("generated token stream does not match source SHA")
 result=score_complete(t,ref,f,ids,record.get("finish_reason"))
 return {**result,"policy":record["policy"],"source_generation_claim_only":True,
         "authoritative_task_accuracy_allowed":False}

if __name__=="__main__":
 import argparse
 a=argparse.ArgumentParser()
 a.add_argument("--validate-sources",action="store_true")
 a.add_argument("--record-json")
 args=a.parse_args()
 token,source=pin_authority()
 if args.record_json:
  p=pathlib.Path(args.record_json)
  raw=json.loads(p.read_text())
  result=audit_raw_generation_record(token,source,raw)
  print(json.dumps({"status":"P18_GENERATED_TASK_GRAMMAR_CHECKED_NOT_NATIVE_ATTESTED",
                    "item_id":result["item_id"],"policy":result["policy"],
                    "correct_claim":result["correct"],
                    "authoritative_task_accuracy_allowed":False,
                    "production_touched":False},sort_keys=True))
 else:print(json.dumps({"status":"P18_480_TASK_SCORING_AUTHORITY_AND_ORIGINAL_TOKENIZER_PINS_PASS",
             "question_count":len(source),"production_touched":False,
             "model_inference_performed":False},sort_keys=True))
