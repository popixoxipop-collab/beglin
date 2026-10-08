#!/usr/bin/env python3
"""Immutable held-out natural-language Qwen2.5 L0 V-focus mixed Q4/Q5 diagnostic.

No training, model updates, production writes, or automatic promotion.
Frozen: prior V-focus Q/K/V/O map 16/16/64/32 and map SHA, model,
engine, tokenizer revision, natural language cases, context positions.
Evaluate canonical Q4 vs original-FP32 for L0 Q/K/V/O vs V-focus.
"""
from __future__ import annotations
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid

from tokenizers import Tokenizer
import tokenizers

SCHEMA="beglin-qt-vfocus-natural-heldout/1"
MODEL=Path("/tmp/qt-l0-fp32-cache/model.safetensors")
CONFIG=Path("/tmp/qt-l0-fp32-cache/config.json")
ENGINE=Path("/tmp/beglin-qt-l0-fp32-control/build-qt-l0-fp32/qwen_infer_gpu")
REPO=Path("/Users/xox/beglin")
RESULT_ROOT=Path("/Users/xox/mcp-sandbox/tailnet-commander/qt-natural-vfocus-results")
REV="7ae557604adf67be50417f59c2c2f167def9a775"
TOKENIZER_ROOT=Path("/tmp/qt-natural-vfocus-tokenizer-"+REV)
TOK_SHA="c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
TOKCONF_SHA="5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
MODEL_SHA="fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
CONFIG_SHA="18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
ENGINE_SHA="a023fe2e15da39eddbf7b2a483d65a25e3897eb3c3d9825db211b4ec2adce96e"
OLD_MAP=Path("/tmp/qt-role-grid-ld4quo2c/bits_16_16_64_32")
ROLES=("q_proj","k_proj","v_proj","o_proj")
COUNTS={"q_proj":16,"k_proj":16,"v_proj":64,"o_proj":32}
ROLE_SHA={
 "q_proj":"a5c8ad63bd93746850d9035824f06128a1a536f9b8268633ed8b953c136f0d4d",
 "k_proj":"98eb5bc1cfbdff603c2bfd433909b15572730d196828f137dc58b35ab7e1ac4b",
 "v_proj":"8f272ed4639bc2b2427b9b40eeb9e1575a30980f3124cd345a08a8058042482f",
 "o_proj":"1c5dba1e57a4d8271932339d8e2736f9276873ad047d2022e93a19f25a6f0290",
}
HIDDEN=896
LAYERS=24
ANSWER_TOKENS=12

# These were written *after* the previous synthetic-token experiment and are
# fixed before running any natural-language results. No prompt optimization.
SCENARIOS=(
 {"id":"english-128","language":"en","tokens":128,
  "role":"laboratory maintenance report",
  "lead":"The engineer inspected a cold-storage test rig. The signed bearing inspection identifier is RAVEN-42. The inspector retained the test certificate and refused to alter the approved batch number.",
  "notes":(
   "The technician checked the torque reading after the housing reached thermal equilibrium.",
   "A second engineer inspected the shaft alignment and recorded the variance in the service notebook.",
   "The vibration monitor remained stable during the morning trial.",
   "The team kept the original measurements separate from the field adjustments.",
   "The maintenance checklist required two signatures before release.",
   "The workshop documented the tooling calibration and the replacement sequence.",
   "The report included a timeline of observations, warnings and corrective actions.",
   "The supervisor requested a short answer drawn only from the signed inspection record."),
  "question":"What is the signed bearing inspection identifier?","expected":"RAVEN-42","positions":("mid","end")},
 {"id":"korean-128","language":"ko","tokens":128,
  "role":"로봇 센서 교정 점검 기록",
  "lead":"이동 로봇의 최종 교정 승인 코드는 SEIRA-27입니다.",
  "notes":(
   "작업자는 실험실 온도와 센서 초기 상태를 먼저 확인했습니다.",
   "연속 측정 결과는 점검 문서에 시간순으로 기록했습니다.",
   "장비 진동과 측정 잡음의 영향을 별도로 구분했습니다.",
   "교정 이후에는 기준 좌표와 실측 좌표를 다시 비교했습니다.",
   "관리자는 검사표의 누락된 항목이 없는지 살폈습니다.",
   "모든 결과는 승인된 교정 절차에 따라 보관했습니다."),
  "question":"최종 교정 승인 코드는 무엇입니까? 코드만 답하십시오.","expected":"SEIRA-27","positions":("end",)},
 {"id":"english-512","language":"en","tokens":512,
  "role":"software incident review",
  "lead":"The service team investigated a late-night restart. The confirmed incident ticket is MAPLE-61. Only the confirmed ticket should be used in the incident summary, not the earlier internal drafts.",
  "notes":(
   "The support engineer noticed a rise in response latency shortly after a scheduled cache refresh.",
   "The on-call analyst compared the event timeline with the recorded service metrics.",
   "A dependency update had been installed earlier, but no connection to the incident was proven.",
   "Operators captured evidence from the actual running process rather than from a symbolic link.",
   "The incident commander asked that restoration and root-cause analysis remain separate tasks.",
   "The recovery procedure preserved the previous certified binary and its process identity.",
   "After service readiness returned, the team reran the necessary smoke tests.",
   "The review distinguished observed facts, assumptions and unresolved hypotheses.",
   "A follow-up proposal recommended alerts for health and readiness transitions.",
   "The final report prohibited automatic deployment based solely on a successful unit test.",
   "An independent reviewer checked the incident evidence before filing the report.",
   "The next maintenance window was left unchanged pending additional verification."),
  "question":"What is the confirmed incident ticket? Respond with the ticket only.","expected":"MAPLE-61","positions":("mid","end")},
 {"id":"korean-512","language":"ko","tokens":512,
  "role":"에너지 관리 회의록",
  "lead":"실험팀은 태양광 설비의 수집 데이터를 검토했습니다. 승인된 성능 시험 번호는 HARBOR-52입니다. 이 번호는 여러 초안 중 최종 서명본에 기재된 식별자입니다.",
  "notes":(
   "오전 회의에서는 설비의 전압과 온도 데이터를 먼저 정리했습니다.",
   "품질 담당자는 원본 센서 기록을 수정 없이 보존하도록 요청했습니다.",
   "시험 과정에서 기록한 이상치는 원인 분석 결과와 분리했습니다.",
   "전력 변환 효율은 동일한 외부 조건에서 비교해야 한다고 합의했습니다.",
   "운영팀은 시험 시작 전 장비 상태와 교정 이력을 확인했습니다.",
   "측정값 누락이 발견되면 추정값으로 채우지 않고 결측으로 남겼습니다.",
   "검토자는 성능 수치와 장비 상태의 관련성을 반복 확인했습니다.",
   "결과 보고서에는 근거 자료의 해시와 작성 시각을 포함했습니다.",
   "추가 검증에 사용할 기록은 최초 평가 자료와 분리해 보관했습니다.",
   "회의에서는 실제 관측과 실험자의 해석을 다른 항목으로 기록했습니다."),
  "question":"최종 승인된 성능 시험 번호는 무엇입니까? 번호만 답하십시오.","expected":"HARBOR-52","positions":("end",)},
 {"id":"english-1024","language":"en","tokens":1024,
  "role":"long technical audit and release memo",
  "lead":"The research team prepared a formal reliability audit for a robotic inspection system. The approved rollback checkpoint identifier is LYRA-84. The checkpoint was approved after independent verification, and no other checkpoint is authorized in this memo.",
  "notes":(
   "The inspection system combines camera observations with inertial measurements under a shared timestamp contract.",
   "The audit distinguishes sensor calibration from online filtering so their errors cannot be silently combined.",
   "A repeatable experiment begins with a frozen model artifact and a documented execution environment.",
   "Every trial records the input identifier, software revision, configuration checksum and output checksum.",
   "The team tracks median error, worst-case error, memory consumption and runtime latency separately.",
   "A successful build verifies only that the program was compiled, not that inference quality improved.",
   "A candidate that performs well on calibration tasks may fail when evaluated on a new workload.",
   "The validation committee therefore reserves independent tasks until all candidate selection is finished.",
   "The logging system records service health and readiness as observations from the actual process.",
   "An immutable release is selected by an operation-scoped manifest rather than by a mutable link.",
   "If readiness fails during an attempted release, the system must restore the previously verified process.",
   "A failed rollback must retain a fence until an authorized recovery process confirms process identity.",
   "The workspace inventory includes both local storage and network-mounted volumes.",
   "A network volume may disappear after a restart and should not be replaced by an empty directory.",
   "The engineering group evaluates floating-point and quantized reference paths under identical prompts.",
   "Local reconstruction loss does not always correlate with the maximum error at a downstream layer.",
   "Role-specific precision budgets can redistribute resources across query, key, value and output projections.",
   "Resource planning considers free memory and free GPU capacity rather than only physical hardware totals.",
   "Performance claims require synchronized measurements of quality, latency and actual process memory.",
   "An accepted candidate remains experimental until it passes completely new validation inputs.",
   "The team avoids automatic promotion when an experiment reveals even a small strict-error regression.",
   "The final audit includes sufficient evidence to reproduce the prompts and verify the model checksum."),
  "question":"What is the approved rollback checkpoint identifier? Answer with only the identifier.","expected":"LYRA-84","positions":("mid","end")},
)

def sha_bytes(b:bytes)->str:
 return hashlib.sha256(b).hexdigest()
def verify_file(p:Path,expected:str)->str:
 if not p.is_file() or p.stat().st_size==0:
  raise RuntimeError("MISSING_PINNED_FILE:"+str(p))
 hasher=hashlib.sha256()
 with p.open("rb") as reader:
  for part in iter(lambda:reader.read(1024*1024),b""):
   hasher.update(part)
 actual=hasher.hexdigest()
 if actual!=expected:raise RuntimeError("SHA_MISMATCH:"+str(p)+":"+actual)
 return actual
def relative_l2(a,b)->float:
 num=sum((x-y)**2 for x,y in zip(a,b))
 den=sum(x*x for x in a)
 return math.sqrt(num)/(math.sqrt(den)+1e-12)
def compare_dumps(ref,cand):
 if len(ref)!=LAYERS*HIDDEN or len(cand)!=LAYERS*HIDDEN:
  raise RuntimeError("HIDDEN_DIMENSION_MISMATCH")
 return [relative_l2(ref[i*HIDDEN:(i+1)*HIDDEN],cand[i*HIDDEN:(i+1)*HIDDEN]) for i in range(LAYERS)]
def load_dump(p):
 b=p.read_bytes()
 if len(b)!=LAYERS*HIDDEN*4:raise RuntimeError("INVALID_DUMP_SIZE:"+str(p)+":"+str(len(b)))
 a=array.array("f");a.frombytes(b)
 if sys.byteorder!="little":a.byteswap()
 if not all(math.isfinite(x) for x in a):raise RuntimeError("NONFINITE_DUMP:"+str(p))
 return a,sha_bytes(b)
def pin_maps(work):
 frozen=work/"vfocus";frozen.mkdir()
 q4=work/"allq4";q4.mkdir()
 result={}
 for role in ROLES:
  name="model_layers_0_self_attn_"+role+"_weight.bits"
  old=OLD_MAP/name
  raw=old.read_bytes()
  sha=sha_bytes(raw)
  if sha!=ROLE_SHA[role] or raw.count(5)!=COUNTS[role] or set(raw)!={4,5}:
   raise RuntimeError("VFOCUS_MAP_CHANGED:"+role+":"+sha)
  expected_size=12544 if role in ("q_proj","o_proj") else 1792
  if len(raw)!=expected_size:raise RuntimeError("MAP_SIZE_CHANGED:"+role)
  (frozen/name).write_bytes(raw)
  (q4/name).write_bytes(bytes([4])*len(raw))
  result[role]={"sha256":sha,"n5":raw.count(5),"cells":len(raw)}
 return frozen,q4,result
def tokenizer():
 verify_file(TOKENIZER_ROOT/"tokenizer.json",TOK_SHA)
 verify_file(TOKENIZER_ROOT/"tokenizer_config.json",TOKCONF_SHA)
 tok=Tokenizer.from_file(str(TOKENIZER_ROOT/"tokenizer.json"))
 if tok.token_to_id("<|im_start|>")!=151644 or tok.token_to_id("<|im_end|>")!=151645:
  raise RuntimeError("TOKENIZER_SPECIAL_TOKENS_MISMATCH")
 return tok
def encode_case(tok,c):
 lang=c["language"];target=c["tokens"]
 system="Return only the document identifier requested in the question. Do not guess or elaborate."
 if lang=="ko":system="승인 코드만 답하십시오."
 head=(f"<|im_start|>system\n{system}<|im_end|>\n"
       f"<|im_start|>user\nDocument type: {c['role']}\n"
       f"Approved fact: {c['lead']}\nEvidence notes:\n")
 tail=f"\nQuestion: {c['question']}\n<|im_end|>\n<|im_start|>assistant\n"
 h=tok.encode(head,add_special_tokens=False).ids
 t=tok.encode(tail,add_special_tokens=False).ids
 available=target-len(h)-len(t)
 if available<12:raise RuntimeError("PROMPT_TOO_SMALL_FOR_HEADERS:"+c["id"])
 lines=[]
 for i in range(700):
  marker=f"Record {i+1}: " if lang=="en" else f"기록 {i+1}: "
  lines.append(marker+c["notes"][i%len(c["notes"])]+"\n")
  if len(tok.encode("".join(lines),add_special_tokens=False).ids)>=available+32:
   break
 body=tok.encode("".join(lines),add_special_tokens=False).ids
 if len(body)<available:raise RuntimeError("INSUFFICIENT_NATURAL_CONTEXT")
 tokens=h+body[:available]+t
 if len(tokens)!=target or any(x<0 or x>=151936 for x in tokens):
  raise RuntimeError("PROMPT_IDS_INVALID")
 decoded=tok.decode(tokens,skip_special_tokens=False)
 if c["expected"] not in decoded or c["question"] not in decoded:
  raise RuntimeError("QA_FACT_OR_QUESTION_MISSING")
 encoded=struct.pack("<%di"%len(tokens),*tokens)
 sha=sha_bytes(encoded)
 positions=[target//2-1 if x=="mid" else target-1 for x in c["positions"]]
 if any(x<0 or x>=target for x in positions):raise RuntimeError("INVALID_POSITION")
 return encoded,{"case":c["id"],"target_tokens":target,"language":lang,
                  "prompt_sha256":sha,"decoded_sha256":sha_bytes(decoded.encode()),
                  "decoded_preview_start":decoded[:240],
                  "decoded_preview_end":decoded[-240:],
                  "expected_answer":c["expected"],"question":c["question"],
                  "positions":positions}
def ps_rss_kb(pid:int)->int|None:
 p=subprocess.run(["/bin/ps","-p",str(pid),"-o","rss="],
                  stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,timeout=3)
 try:return int(p.stdout.strip())
 except (ValueError,TypeError):return None
def launch(work,case_id,tokens_file,pos:int,arm:str,manifest,fp32=False,generate=1):
 tag=f"{case_id}_pos{pos}_{arm}"
 dump=work/"dumps"/(tag+".bin")
 stdout_path=work/"logs"/(tag+".stdout")
 stderr_path=work/"logs"/(tag+".stderr")
 dump.parent.mkdir(parents=True,exist_ok=True)
 stdout_path.parent.mkdir(parents=True,exist_ok=True)
 env={k:v for k,v in os.environ.items() if not k.startswith("QWEN_")}
 env.update(QWEN_SAFETENSORS=str(MODEL),QWEN_HF_CONFIG=str(CONFIG),
            QWEN_PROMPT=str(tokens_file),QWEN_DEBUG_LAYERDUMP=str(dump),
            QWEN_DEBUG_LAYERDUMP_POS=str(pos))
 if manifest:env["QWEN_QT_MIXED_MANIFEST"]=str(manifest)
 if fp32:
  if not manifest:raise RuntimeError("MISSING_F32_MANIFEST")
  env["QWEN_QT_FP32_PASSTHROUGH"]="1"
 started=time.perf_counter()
 peak=0;sample_count=0
 with stdout_path.open("wb") as out,stderr_path.open("wb") as err:
  proc=subprocess.Popen([str(ENGINE),"greedy",str(generate)],cwd=str(REPO),
                        env=env,stdout=out,stderr=err)
  try:
   while proc.poll() is None:
    if time.perf_counter()-started>900:
     proc.terminate()
     try:proc.wait(timeout=8)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
     raise RuntimeError("ENGINE_RUN_TIMEOUT:"+tag)
    rss=ps_rss_kb(proc.pid)
    if rss is not None:
     peak=max(peak,rss);sample_count+=1
    time.sleep(0.18)
   status=proc.wait()
  finally:
   if proc.poll() is None:proc.kill();proc.wait()
 wall=time.perf_counter()-started
 stdout=stdout_path.read_text(errors="replace")
 stderr=stderr_path.read_text(errors="replace")
 if status!=0:
  raise RuntimeError(f"ENGINE_FAILED:{tag}:rc={status}:stderr={stderr[-1700:]}")
 markers=[line[:260] for line in stderr.splitlines() if "[qt mixed dense]" in line]
 if manifest:
  n_targets=sum("target=model.layers.0.self_attn." in x for x in markers)
  if n_targets<4:
   raise RuntimeError("NOT_ALL_4_MIXED_ROLES_REGISTERED:"+tag)
 if fp32 and sum("fp32-passthrough" in x for x in markers)<4:
  raise RuntimeError("NOT_ALL_4_F32_ROLES_USED:"+tag)
 lines=[x for x in stdout.splitlines() if x.startswith("greedy:")]
 if len(lines)!=1:raise RuntimeError("MISSING_GREEDY_RESULT:"+tag+":"+stdout[:200])
 generation=lines[0][len("greedy:"):].strip().split()
 if not generation or any(not re.fullmatch(r"\d+",x) for x in generation):
  raise RuntimeError("MALFORMED_GENERATED_IDS:"+tag)
 tokens=[int(x) for x in generation]
 if any(x>=151936 for x in tokens):raise RuntimeError("GENERATED_ID_RANGE")
 arr,dump_sha=load_dump(dump)
 return arr,{"tag":tag,"position":pos,"arm":arm,"wall_seconds":wall,
              "peak_process_rss_kib_sampled":peak if peak else None,
              "rss_samples":sample_count,"process_exit":status,
              "generated_tokens":tokens,"dump_sha256":dump_sha,
              "stderr_sha256":sha_bytes(stderr.encode()),
              "qt_markers":markers[:10],
              "generation_requested":generate}
def score(tokens,expected:str,tok):
 try:
  decoded=tok.decode(tokens,skip_special_tokens=True)
 except Exception as e:
  decoded="<decode-error:"+str(e)+">"
 normalize=lambda x:"".join(c for c in x.upper() if c.isalnum())
 return {"text":decoded[:360],"expected":expected,"correct_identifier":normalize(expected) in normalize(decoded)}
def metrics_for_cases(measured):
 result={}
 for case in measured:
  q4=case["q4_error_per_layer"];v=case["vfocus_error_per_layer"]
  if len(q4)!=24 or len(v)!=24:raise RuntimeError("MISSING_LAYER_METRIC")
  qmax=max(q4);vmax=max(v)
  result[case["case"]+"@"+str(case["position"])]={
   "case":case["case"],"position":case["position"],"length":case["tokens"],
   "q4_max":qmax,"vfocus_max":vmax,"q4_mean":sum(q4)/24,"vfocus_mean":sum(v)/24,
   "max_improvement":qmax-vmax,"case_max_non_worse":vmax<=qmax,
   "max_relative_improvement":(qmax-vmax)/qmax if qmax else 0}
 return result
def summarize(measured):
 per=metrics_for_cases(measured)
 q4=[item["q4_max"] for item in per.values()]
 v=[item["vfocus_max"] for item in per.values()]
 q4layers=[x for case in measured for x in case["q4_error_per_layer"]]
 vlayers=[x for case in measured for x in case["vfocus_error_per_layer"]]
 grouped={}
 for n in (128,512,1024):
  vals=[x for x in per.values() if x["length"]==n]
  grouped[str(n)]={"positions":len(vals),"q4_worst":max(x["q4_max"] for x in vals),
   "vfocus_worst":max(x["vfocus_max"] for x in vals),
   "non_worse_positions":sum(x["case_max_non_worse"] for x in vals)}
 return {"cases":len(per),"q4_global_max":max(q4),"vfocus_global_max":max(v),
         "strict_max_relative_improvement":(max(q4)-max(v))/max(q4),
         "q4_layer_mean":sum(q4layers)/len(q4layers),
         "vfocus_layer_mean":sum(vlayers)/len(vlayers),
         "mean_relative_improvement":(sum(q4layers)-sum(vlayers))/sum(q4layers),
         "non_worse_case_max_count":sum(x["case_max_non_worse"] for x in per.values()),
         "per_context_length":grouped,"per_case":per}
def main():
 for p,sha in ((MODEL,MODEL_SHA),(CONFIG,CONFIG_SHA),(ENGINE,ENGINE_SHA)):
  verify_file(p,sha)
 if not REPO.is_dir():raise RuntimeError("MISSING_REPO")
 tok=tokenizer()
 work=Path(tempfile.mkdtemp(prefix="qt-natural-vfocus-",dir="/tmp"))
 focus,q4map,map_meta=pin_maps(work)
 records=[]
 cases=[]
 for scenario in SCENARIOS:
  data,metadata=encode_case(tok,scenario)
  p=work/"prompts"/(scenario["id"]+".i32")
  p.parent.mkdir(parents=True,exist_ok=True)
  p.write_bytes(data)
  cases.append((scenario,metadata,p))
 read_start=time.perf_counter()
 for scen,meta,prompt in cases:
  length=scen["tokens"]
  for pos in meta["positions"]:
   last=(pos==length-1)
   gen=ANSWER_TOKENS if last else 1
   # Warmup is intrinsically same for all arms via identical file and process.
   ref,rref=launch(work,scen["id"],prompt,pos,"original_l0_qkvo_fp32",q4map,fp32=True,generate=gen)
   baseline,rq4=launch(work,scen["id"],prompt,pos,"canonical_q4",None,generate=gen)
   focus_hidden,rv=launch(work,scen["id"],prompt,pos,"vfocus_q5",focus,generate=gen)
   errors_q4=compare_dumps(ref,baseline)
   errors_v=compare_dumps(ref,focus_hidden)
   control=None
   if last and scen["id"] in ("english-128","english-512","english-1024"):
    canonical_q4_also,rctrl=launch(work,scen["id"],prompt,pos,"mixed_allq4",q4map,generate=gen)
    contract=relative_l2(baseline,canonical_q4_also)
    if contract>0.001:raise RuntimeError("MIXED_CANONICAL_Q4_CONTRACT_FAIL:"+scen["id"]+":"+str(contract))
    control={"mixed_allq4_direct_relative_l2":contract,"run":rctrl}
   quality=None
   if last:
    quality={"q4":score(rq4["generated_tokens"],scen["expected"],tok),
             "vfocus":score(rv["generated_tokens"],scen["expected"],tok),
             "l0_qkvo_fp32":score(rref["generated_tokens"],scen["expected"],tok),
             "vfocus_same_tokens_as_q4":rv["generated_tokens"]==rq4["generated_tokens"]}
   record={"case":scen["id"],"tokens":length,"position":pos,
           "q4_error_per_layer":errors_q4,"vfocus_error_per_layer":errors_v,
           "q4_worst":max(errors_q4),"vfocus_worst":max(errors_v),
           "q4_vs_vfocus_max_relative_change":(max(errors_v)-max(errors_q4))/max(errors_q4),
           "runs":{"fp32":rref,"q4":rq4,"vfocus":rv},
           "mixed_q4_control":control,"quality":quality}
   records.append(record)
   # Each checkpoint is durable in user-owned sandbox. Never overwrite prior run.
   print(json.dumps({"event":"CASE_DONE","case":scen["id"],"pos":pos,
      "q4_max":max(errors_q4),"vfocus_max":max(errors_v),
      "q4_wall_s":rq4["wall_seconds"],"vfocus_wall_s":rv["wall_seconds"],
      "quality_q4":quality["q4"]["correct_identifier"] if quality else None,
      "quality_vfocus":quality["vfocus"]["correct_identifier"] if quality else None},sort_keys=True),flush=True)
 aggregate=summarize(records)
 qa=[{"case":r["case"],"q4":r["quality"]["q4"]["correct_identifier"],
      "vfocus":r["quality"]["vfocus"]["correct_identifier"],
      "vfocus_same_tokens_as_q4":r["quality"]["vfocus_same_tokens_as_q4"]}
     for r in records if r["quality"]]
 regressed=[x["case"] for x in qa if x["q4"] and not x["vfocus"]]
 strict_pass=(aggregate["non_worse_case_max_count"]==aggregate["cases"]
  and aggregate["strict_max_relative_improvement"]>0 and not regressed)
 elapsed=time.perf_counter()-read_start
 payload={"schema":SCHEMA,"state":"SUCCEEDED",
  "production_write_allowed":False,"automatic_live_promotion":False,
  "source_selection":"V-focus 16/16/64/32 chosen AFTER prior synthetic-token heldout (pre-frozen for this new test)",
  "frozen_model_revision":REV,
  "model_sha256":MODEL_SHA,"config_sha256":CONFIG_SHA,"engine_sha256":ENGINE_SHA,
  "tokenizer_sha256":TOK_SHA,"tokenizer_config_sha256":TOKCONF_SHA,
  "tokenizers_library_version":tokenizers.__version__,
  "fixture_sha256":sha_bytes(Path(__file__).read_bytes()),
  "strategy":{"q_proj":16,"k_proj":16,"v_proj":64,"o_proj":32},
  "bitmaps":map_meta,
  "model_scope":"L0 Q/K/V/O original FP32 vs canonical Q4 vs frozen V-focused Q4/Q5",
  "actual_prompt_type":"official HF tokenizers encode of handwritten natural-language documents with length-trimmed body and preserved question tail; prompt IDs persisted",
  "question_answer_protocol":"12 greedy decode token IDs, decoded by same pinned official tokenizer; normalized expected identifier presence only, not broad LM accuracy",
  "latency_memory_scope":"single-process wall seconds for prefill+greedy decode; process RSS sampled (not GPU VRAM, no warm cache control); exploratory only",
  "prompt_cases":[metadata for _,metadata,_ in cases],
  "runs":records,
  "summary":aggregate,
  "qa":qa,"qa_regressions":regressed,
  "strict_hidden_non_worse_plus_quality_gate_pass":strict_pass,
  "total_elapsed_seconds":elapsed}
 data=(json.dumps(payload,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode("utf-8")
 RESULT_ROOT.mkdir(parents=True,exist_ok=True)
 uid=uuid.uuid4().hex
 dst=RESULT_ROOT/("QT_NATURAL_VFOCUS_"+uid+".json")
 with dst.open("xb") as f:
  f.write(data);f.flush();os.fsync(f.fileno())
 info={"schema":SCHEMA,"status":"SUCCEEDED","fixture_sha256":payload["fixture_sha256"],
       "result_path":str(dst),"result_sha256":sha_bytes(data),"result_bytes":len(data),
       "model_sha256":MODEL_SHA,"engine_sha256":ENGINE_SHA,
       "tokenizer_sha256":TOK_SHA,"maps":map_meta,
       "strict_hidden_non_worse_plus_quality_gate_pass":strict_pass,
       "qa_regressions":regressed,"qa":qa,"hidden":aggregate,
       "elapsed_s":elapsed,"automatic_live_promotion":False}
 print("QT_NATURAL_VFOCUS_RESULT="+json.dumps(info,sort_keys=True,ensure_ascii=False),flush=True)

if __name__=="__main__":
 main()
