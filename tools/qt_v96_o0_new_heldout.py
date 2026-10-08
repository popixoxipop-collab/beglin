#!/usr/bin/env python3
"""Frozen *new* natural-language heldout for a fixed 128-cell O0 Q16/K16/V96/O0 candidate.

No candidate tuning, no auto-promotion, no training. Disjoint from synthetic and
previous natural QA prompts. Compare identical 128 Q5-cell budgets against
Q4 and prior Q16/K16/V64/O32 in one engine, model and tokenizer.
"""
from __future__ import annotations

import array
import hashlib
import importlib.util
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

SCHEMA="beglin-qt-v96-o0-frozen-natural-heldout/1"
MODEL=Path("/tmp/qt-l0-fp32-cache/model.safetensors")
CONFIG=Path("/tmp/qt-l0-fp32-cache/config.json")
ENGINE=Path("/tmp/beglin-qt-l0-fp32-control/build-qt-l0-fp32/qwen_infer_gpu")
REPO=Path("/Users/xox/beglin")
TOKENIZER=Path("/tmp/qt-natural-vfocus-tokenizer-7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json")
TOKCFG=Path("/tmp/qt-natural-vfocus-tokenizer-7ae557604adf67be50417f59c2c2f167def9a775/tokenizer_config.json")
PRIOR_MAPS=Path("/tmp/qt-role-grid-ld4quo2c/bits_16_16_64_32")
OUTPUT=Path("/Users/xox/mcp-sandbox/tailnet-commander/qt-v96-o0-new-heldout-results")
RANK_SOURCE=Path(__file__).with_name("qt_role_downstream_grid.py")
MODEL_SHA="fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
CONFIG_SHA="18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
ENGINE_SHA="a023fe2e15da39eddbf7b2a483d65a25e3897eb3c3d9825db211b4ec2adce96e"
TOKENIZER_SHA="c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
TOKCFG_SHA="5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
RANK_SHA="ff33df5e1af3d412f2312c6e5adf2507ff02c219fc3956b543384c1ffb62be6f"
ROLE_ORDER=("q_proj","k_proj","v_proj","o_proj")
PRIOR_COUNTS={"q_proj":16,"k_proj":16,"v_proj":64,"o_proj":32}
TARGET_COUNTS={"q_proj":16,"k_proj":16,"v_proj":96,"o_proj":0}
PRIOR_SHAS={
    "q_proj":"a5c8ad63bd93746850d9035824f06128a1a536f9b8268633ed8b953c136f0d4d",
    "k_proj":"98eb5bc1cfbdff603c2bfd433909b15572730d196828f137dc58b35ab7e1ac4b",
    "v_proj":"8f272ed4639bc2b2427b9b40eeb9e1575a30980f3124cd345a08a8058042482f",
    "o_proj":"1c5dba1e57a4d8271932339d8e2736f9276873ad047d2022e93a19f25a6f0290",
}
V96_SHA="46bbbe0a7f12b0c04c7ab4e7e229ab1de2996fba402b620000e83bfefd829819"
VOCAB=151936
LAYERS=24
HIDDEN=896
NEW_TOKENS=16

# Pre-declared after old RAVEN/MAPLE/LYRA failure and before observing any
# candidate result. No text/IDs/approved code reused from previous prompts.
# All are artificially constructed fictional document QA probes.
CORPUS=(
 {"id":"en-128-air","lang":"en","length":128,
  "topic":"aircraft materials inspection",
  "fact":"The certified alloy-batch inspection identifier is BIRCH-58. The signed certificate applies to the sampled wing bracket.",
  "notes":(
   "Quality engineers checked the metal surface under uniform lighting and logged the dimensions.",
   "The supervisor inspected fixture alignment before accepting the tensile test measurements.",
   "The record separates preliminary technician notes from the signed certificate.",
   "Machine calibration used the approved measuring blocks and the current inspection procedure.",
   "No substitutions were made for the original readings."),
  "question":"What is the certified alloy-batch inspection identifier? Give the identifier only.","answer":"BIRCH-58","positions":("mid","end")},
 {"id":"en-128-ocean","lang":"en","length":128,
  "topic":"ocean navigation sensor calibration",
  "fact":"The verified sonar maintenance log reference is NOVA-31. It refers to the completed harbor test.",
  "notes":(
   "The ship's sensor package was held at a constant reference speed during calibration.",
   "Maintenance staff checked the power supply and temperature before capturing new data.",
   "The printed maintenance summary documents every exception and the follow-up checks.",
   "The final status was reviewed by a second operator."),
  "question":"Which verified sonar maintenance log reference appears in the report? Answer with only the reference.","answer":"NOVA-31","positions":("mid","end")},
 {"id":"ko-128-energy","lang":"ko","length":128,
  "topic":"도시 에너지 계측 점검",
  "fact":"태양광 패널 계측 기록의 최종 승인 번호는 DAON-41입니다. 검수 담당자가 확인한 번호입니다.",
  "notes":(
   "관리자는 발전량과 기상 조건을 구분하여 자료를 기록했습니다.",
   "현장에서는 계측기가 기준 규격을 만족하는지 다시 검사했습니다.",
   "결과 보고서는 원본 데이터를 변경하지 않고 보관하도록 작성했습니다.",
   "시험 종료 후 점검자는 기록의 누락 여부를 확인했습니다."),
  "question":"최종 승인 번호가 무엇입니까? 번호만 쓰십시오.","answer":"DAON-41","positions":("end",)},
 {"id":"en-512-medical","lang":"en","length":512,
  "topic":"laboratory sample custody review",
  "fact":"The official sample chain-of-custody ticket is MESA-74. The ticket was signed before the specimen was moved.",
  "notes":(
   "The laboratory confirmed container integrity before the receiving clerk logged the transfer.",
   "The temperature monitor captured each reading during transport and storage.",
   "The independent reviewer checked the seal but did not edit historical sensor measurements.",
   "The report explicitly distinguishes verified facts from staff comments.",
   "All calibration certificates were indexed with their original timestamps.",
   "The receiving team preserved the physical documentation under the approved handling policy.",
   "The log includes both routine observations and exceptional conditions for later review.",
   "A supervisor signed the final paperwork only after checking all mandatory fields.",
   "The process records the instrument identity without disclosing any personal information.",
   "A follow-up inspection focused on the storage environment rather than the test specimen."),
  "question":"What is the official sample chain-of-custody ticket? Reply with only the ticket.","answer":"MESA-74","positions":("mid","end")},
 {"id":"en-512-satellite","lang":"en","length":512,
  "topic":"satellite telemetry recovery procedure",
  "fact":"The approved telemetry recovery request code is FERN-26. It is the only request authorized in the current review.",
  "notes":(
   "Engineers examined the downlink timing against the spacecraft's authenticated event log.",
   "The maintenance record requires the previous validated software version to remain accessible.",
   "A thermal model was used as background context, not as replacement for measured telemetry.",
   "The recovery checklist distinguishes process readiness from an interface health indicator.",
   "Each command was linked to an audit entry with a timestamp and an independent checksum.",
   "The team did not alter the flight-control settings during analysis of the incident.",
   "Sensor values were converted to engineering units using the documented calibration table.",
   "Reviewers checked the evidence and recorded uncertainties in a separate section.",
   "A proposed patch was held back until its regression tests and operator review were complete.",
   "The service team preferred reversible corrective actions over unverified state changes."),
  "question":"Identify the approved telemetry recovery request code. Output only the code.","answer":"FERN-26","positions":("mid","end")},
 {"id":"ko-512-robot","lang":"ko","length":512,
  "topic":"자율 이동 로봇의 안전 점검 보고서",
  "fact":"로봇 안전 검사의 최종 인증 식별자는 YUNA-73입니다. 담당자는 검증을 마친 뒤 해당 식별자를 승인했습니다.",
  "notes":(
   "점검팀은 라이다 거리 측정과 카메라 관측의 시간 동기화를 확인했습니다.",
   "안전 시험은 정해진 속도와 조명 조건에서 반복하여 수행했습니다.",
   "기준 좌표계의 차이는 별도의 오차 항목으로 기록했습니다.",
   "운영자는 시험 중 발생한 특이 사항을 날짜별로 정리했습니다.",
   "검증팀은 기록된 결과를 별도의 승인 문서와 대조했습니다.",
   "센서 캘리브레이션은 안전 담당자의 참관 아래 진행됐습니다.",
   "사후 분석은 실측 기록과 추정 모델의 출력을 구분해 수행했습니다.",
   "평가 결과는 실험 단계별로 보존하고 요약문과 원본을 연결했습니다."),
  "question":"로봇 안전 검사의 최종 인증 식별자가 무엇입니까? 식별자만 답하십시오.","answer":"YUNA-73","positions":("end",)},
 {"id":"en-1024-disaster","lang":"en","length":1024,
  "topic":"flood resilience engineering and bridge survey",
  "fact":"The emergency bridge inspection authorization key is ORBIT-65. It belongs to the approved assessment after the flood.",
  "notes":(
   "The civil engineering team collected photographic evidence without modifying the original files.",
   "Inspectors compared the measured structural deflection with the pre-event baseline.",
   "An archived weather report provided rainfall context but not direct evidence of beam damage.",
   "The certified engineer recorded load-bearing observations under each surveyed span.",
   "Site staff restricted traffic while a separate crew verified the safety equipment.",
   "The risk register distinguished previously known defects from newly observed conditions.",
   "All measurements used documented reference markers and their associated calibration records.",
   "The team logged each observation and linked it to the geospatial inspection map.",
   "Access to the lower support assembly was delayed until water levels returned to safe limits.",
   "An independent reviewer assessed the intervention plan before authorizing field work.",
   "The maintenance notes described observed corrosion without overstating its likely cause.",
   "Engineering calculations were reproduced independently before they were attached to the report.",
   "No automated equipment deployment was permitted until the site clearance was signed.",
   "The team separated speculative failure scenarios from measured structural response.",
   "Corrective actions were recorded alongside the responsible team and an expected completion date.",
   "Final safety observations were compared against the official accepted criteria.",
   "A second field visit confirmed that the reference markers remained in their surveyed locations."),
  "question":"Give the emergency bridge inspection authorization key from the approved assessment.","answer":"ORBIT-65","positions":("mid","end")},
 {"id":"en-1024-data","lang":"en","length":1024,
  "topic":"secure data center migration approval memo",
  "fact":"The signed data migration checkpoint code is CIRRUS-49. It uniquely identifies the authorized snapshot.",
  "notes":(
   "The operations review starts by listing the active services and their dependencies.",
   "The cutover protocol preserves the previous immutable application image for recovery.",
   "Readiness checks must inspect the process currently serving requests rather than a cached response.",
   "A reversible deployment requires an independently verified rollback target.",
   "The data team reconciles record counts and checksums between source and destination.",
   "Secrets are never copied into incident reports or attached to diagnostic artifacts.",
   "An audit file records the release configuration and the operator decision timestamp.",
   "Network monitoring distinguishes expected retry traffic from a genuine communication failure.",
   "Performance baselines are captured before any changes in production.",
   "The proposed rollout proceeds only if all acceptance checks are satisfied.",
   "A synthetic workload may supplement tests but cannot replace real service verification.",
   "Test results include both median latency and tail latency under identical conditions.",
   "If a background process fails, the supervisor examines its logs before attempting recovery.",
   "Immutable evidence must identify the exact input, software version and output state.",
   "The team verifies storage capacity using currently available space instead of installed size.",
   "A configuration change remains pending until the independent reviewer approves it.",
   "The maintenance schedule avoids overlapping concurrent writes to the release authority.",
   "After completion, observers retain the original monitoring snapshots for later analysis."),
  "question":"What is the signed data migration checkpoint code? Respond with only that code.","answer":"CIRRUS-49","positions":("mid","end")},
 {"id":"ko-1024-water","lang":"ko","length":1024,
  "topic":"상수도 정수 시설의 운영 평가",
  "fact":"정수 시설의 승인된 품질 시험 식별자는 MIRAE-62입니다. 최종 검사표에 이 식별자가 기재돼 있습니다.",
  "notes":(
   "시설 운영자는 시료 채취 시간과 장비 온도를 확인했습니다.",
   "시험 담당자는 유량과 압력 기록을 각각 독립된 표로 저장했습니다.",
   "검사 자료는 원본과 분석 보고서를 분리해 관리했습니다.",
   "오차 분석에서는 측정기 교정 이력과 환경 변화를 함께 고려했습니다.",
   "운영팀은 시험 전에 비상 정지 장치의 작동 여부를 검사했습니다.",
   "제어 시스템의 상태는 계측값과 별개의 확인 항목으로 기록됐습니다.",
   "기록에는 실제 관측한 이상 현상과 가능한 원인 가설을 구분해 작성했습니다.",
   "현장 직원은 기록된 각 수치를 별도의 검증표와 비교했습니다.",
   "관리자는 최종 품질 인증 전 검사 절차의 누락 여부를 확인했습니다.",
   "이후 자료는 정해진 보존 기간에 맞춰 접근 권한과 함께 관리했습니다.",
   "점검 후에는 담당자의 서명과 장비 상태 기록을 추가했습니다.",
   "진단 결과는 후속 업무의 입력 정보로 사용되지만 독립 재시험을 대신하지 않습니다."),
  "question":"승인된 품질 시험 식별자가 무엇입니까? 식별자만 기재하십시오.","answer":"MIRAE-62","positions":("end",)},
)

# Forbidden old IDs and exact document contents from the previous cohort.
OLD_IDENTIFIERS=("RAVEN-42","SEIRA-27","MAPLE-61","HARBOR-52","LYRA-84")
COUNTS_TOTAL=128

def sha_bytes(b):
 return hashlib.sha256(b).hexdigest()

def sha_file(p,expected):
 if not p.is_file() or not p.stat().st_size:
  raise RuntimeError("MISSING_PINNED_FILE:"+str(p))
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):
   h.update(b)
 actual=h.hexdigest()
 if actual!=expected:raise RuntimeError("SHA_PIN_MISMATCH:"+str(p)+":"+actual)
 return actual

def prep_maps(work):
 rank_source=sha_file(RANK_SOURCE,RANK_SHA)
 spec=importlib.util.spec_from_file_location("qt_verified_cell_ranking",str(RANK_SOURCE))
 mod=importlib.util.module_from_spec(spec)
 spec.loader.exec_module(mod)
 rank,shapes=mod.cell_ranking()
 candidates={}
 base=(work/"maps")
 base.mkdir(parents=True)
 src={}
 for role in ROLE_ORDER:
  filename="model_layers_0_self_attn_"+role+"_weight.bits"
  p=PRIOR_MAPS/filename
  sha_file(p,PRIOR_SHAS[role])
  raw=p.read_bytes()
  if raw.count(5)!=PRIOR_COUNTS[role] or set(raw)!={4,5}:
   raise RuntimeError("PRIOR_BITMAP_MISMATCH:"+role)
  src[role]=raw
  r,g=shapes[role]
  if len(raw)!=r*g:raise RuntimeError("RANK_DIM_MISMATCH")
  expected=bytearray([4])*len(raw)
  for _,row,group in rank[role][:PRIOR_COUNTS[role]]:
   expected[row*g+group]=5
  if bytes(expected)!=raw:
   raise RuntimeError("PRIOR_MAP_NOT_REPRODUCIBLE:"+role)
 v=bytearray([4])*len(src["v_proj"])
 _,vg=shapes["v_proj"]
 for _,row,group in rank["v_proj"][:96]:
  v[row*vg+group]=5
 if sha_bytes(v)!=V96_SHA or v.count(5)!=96:
  raise RuntimeError("V96_MAP_SHA_OR_COUNT_WRONG")
 if any(a==5 and b!=5 for a,b in zip(src["v_proj"],v)):
  raise RuntimeError("OLD_V64_NOT_SUBSET_OF_NEW_V96")
 v96={
  "q_proj":src["q_proj"],
  "k_proj":src["k_proj"],
  "v_proj":bytes(v),
  "o_proj":bytes([4])*len(src["o_proj"])
 }
 def write_group(name,bits):
  folder=base/name
  folder.mkdir()
  metadata={}
  for role in ROLE_ORDER:
   b=bits[role]
   out=folder/("model_layers_0_self_attn_"+role+"_weight.bits")
   out.write_bytes(b)
   metadata[role]={"sha256":sha_bytes(b),"n4":b.count(4),"n5":b.count(5),"cells":len(b)}
  return folder,metadata
 q4,zero_meta=write_group("all_q4",{role:bytes([4])*len(src[role]) for role in ROLE_ORDER})
 old,old_meta=write_group("previous_v64_o32",src)
 new,new_meta=write_group("candidate_v96_o0",v96)
 if sum(x["n5"] for x in old_meta.values())!=COUNTS_TOTAL or sum(x["n5"] for x in new_meta.values())!=COUNTS_TOTAL:
  raise RuntimeError("NOT_EQUAL_128_Q5_BUDGET")
 return {"canonical_q4":None,"mixed_all_q4":q4,"prior_vfocus_128":old,"o0_v96_128":new,"l0_qkvo_fp32":q4},\
        {"prior":old_meta,"candidate":new_meta,"control":zero_meta,
         "ranking_source_sha256":rank_source,"new_v96_sha256":V96_SHA}

def get_tokenizer():
 sha_file(TOKENIZER,TOKENIZER_SHA)
 sha_file(TOKCFG,TOKCFG_SHA)
 tok=Tokenizer.from_file(str(TOKENIZER))
 if tok.token_to_id("<|im_start|>")!=151644 or tok.token_to_id("<|im_end|>")!=151645:
  raise RuntimeError("WRONG_QWEN_SPECIAL_TOKENS")
 return tok

def encode_doc(tok,doc):
 length=doc["length"]
 system="Answer the final question using only the verified identifier found in the document. Do not quote unrelated notes."
 if doc["lang"]=="ko":
  system="식별자만 답하십시오."
 head=("<|im_start|>system\n"+system+"<|im_end|>\n"+
       "<|im_start|>user\n"+doc["topic"]+"\n"+
       "Verified document finding: "+doc["fact"]+"\nBackground notes:\n")
 tail="\nFinal question: "+doc["question"]+"\n<|im_end|>\n<|im_start|>assistant\n"
 a=tok.encode(head,add_special_tokens=False).ids
 b=tok.encode(tail,add_special_tokens=False).ids
 capacity=length-len(a)-len(b)
 if capacity<8:raise RuntimeError("INSUFFICIENT_CONTENT_CAPACITY:"+doc["id"]+":"+str(capacity))
 notes=[]
 for i in range(1500):
  label=(f"Context entry {i+1}: " if doc["lang"]=="en" else f"참고 사항 {i+1}: ")
  notes.append(label+doc["notes"][i%len(doc["notes"])]+"\n")
  if len(tok.encode("".join(notes),add_special_tokens=False).ids)>=capacity+60:break
 filler=tok.encode("".join(notes),add_special_tokens=False).ids
 if len(filler)<capacity:raise RuntimeError("BODY_FILLER_INSUFFICIENT")
 tokens=a+filler[:capacity]+b
 if len(tokens)!=length or not all(0<=x<VOCAB for x in tokens):
  raise RuntimeError("INCORRECT_TOKEN_LENGTH_OR_RANGE")
 decoded=tok.decode(tokens,skip_special_tokens=False)
 if doc["answer"] not in decoded or doc["question"] not in decoded:
  raise RuntimeError("FACT_OR_QUESTION_LOST")
 for code in OLD_IDENTIFIERS:
  if code in decoded:raise RuntimeError("PREVIOUS_QA_IDENTIFIER_LEAKED")
 return struct.pack("<"+str(len(tokens))+"i",*tokens),\
        {"case":doc["id"],"language":doc["lang"],"length":length,"answer":doc["answer"],
         "question":doc["question"],"positions":[length//2-1,length-1] if "mid" in doc["positions"] else [length-1],
         "preview_head":decoded[:230],"preview_tail":decoded[-240:]}

def parse_dump(path):
 raw=path.read_bytes()
 if len(raw)!=LAYERS*HIDDEN*4:raise RuntimeError("BAD_HIDDEN_DUMP_DIMENSIONS")
 v=array.array("f");v.frombytes(raw)
 if sys.byteorder!="little":v.byteswap()
 if not all(math.isfinite(x) for x in v):
  raise RuntimeError("NONFINITE_HIDDEN")
 return v,sha_bytes(raw)

def relative_l2(a,b):
 den=sum(float(x)*float(x) for x in a)
 num=sum((float(x)-float(y))**2 for x,y in zip(a,b))
 return math.sqrt(num)/(math.sqrt(den)+1e-12)

def per_layer(reference,trial):
 if len(reference)!=LAYERS*HIDDEN or len(trial)!=len(reference):
  raise RuntimeError("HIDDEN_SHAPE_MISMATCH")
 return [relative_l2(reference[i*HIDDEN:(i+1)*HIDDEN],trial[i*HIDDEN:(i+1)*HIDDEN]) for i in range(LAYERS)]

def ps_rss(pid):
 p=subprocess.run(["/bin/ps","-p",str(pid),"-o","rss="],
                  capture_output=True,text=True,timeout=3)
 try:return int(p.stdout.strip())
 except (ValueError,TypeError):return None

def run(work,name,prompt,pos,engine,manifest,gen=1,fp32=False):
 target=(work/"runs"/name)
 target.mkdir(parents=True,exist_ok=False)
 dump=target/"layers.bin"
 env={k:v for k,v in os.environ.items() if not k.startswith("QWEN_")}
 env.update(QWEN_SAFETENSORS=str(MODEL),QWEN_HF_CONFIG=str(CONFIG),
            QWEN_PROMPT=str(prompt),QWEN_DEBUG_LAYERDUMP=str(dump),
            QWEN_DEBUG_LAYERDUMP_POS=str(pos))
 if manifest is not None:
  env["QWEN_QT_MIXED_MANIFEST"]=str(manifest)
 if fp32:
  if manifest is None:raise RuntimeError("FP32_REQUIRES_MAP")
  env["QWEN_QT_FP32_PASSTHROUGH"]="1"
 started=time.monotonic()
 peak=0;samples=0
 with (target/"stdout").open("wb") as stdout,(target/"stderr").open("wb") as stderr:
  proc=subprocess.Popen([str(ENGINE),"greedy",str(gen)],
     cwd=str(REPO),env=env,stdout=stdout,stderr=stderr)
  try:
   while proc.poll() is None:
    if time.monotonic()-started>900:
     proc.terminate()
     try:proc.wait(timeout=8)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
     raise RuntimeError("ENGINE_TIMEOUT:"+name)
    rss=ps_rss(proc.pid)
    if rss is not None:peak=max(peak,rss);samples+=1
    time.sleep(.18)
   code=proc.wait()
  finally:
   if proc.poll() is None:proc.kill();proc.wait()
 elapsed=time.monotonic()-started
 stdout=(target/"stdout").read_text(errors="replace")
 stderr=(target/"stderr").read_text(errors="replace")
 if code!=0:raise RuntimeError("ENGINE_FAILED:"+name+":exit="+str(code)+":"+stderr[-1000:])
 match=re.search(r"(?m)^greedy:((?:[ \t]+\d+)+)\s*$",stdout)
 if not match:raise RuntimeError("GREEDY_TOKEN_OUTPUT_MISSING:"+name+":"+stdout[:160])
 tokens=[int(a) for a in match.group(1).strip().split()]
 if len(tokens)!=gen or any(v<0 or v>=VOCAB for v in tokens):raise RuntimeError("INVALID_GENERATED_TOKENS")
 markers=[s for s in stderr.splitlines() if "[qt mixed dense]" in s]
 if manifest is not None:
  if sum("target=model.layers.0.self_attn." in s for s in markers)<4:
   raise RuntimeError("NOT_ALL_L0_ROLES_REGISTERED:"+name)
 if fp32:
  if sum("fp32-passthrough" in s for s in markers)<4:
   raise RuntimeError("NOT_ALL_L0_FP32_PASSTHROUGH:"+name)
 hidden,hashval=parse_dump(dump)
 return hidden,{"id":name,"wall_seconds":elapsed,"peak_sampled_rss_kib":peak or None,
         "rss_samples":samples,"generated_ids":tokens,
         "hidden_sha256":hashval,"stderr_sha256":sha_bytes(stderr.encode()),
         "registered_role_markers":markers[:8],"exit_code":code}

def quality(tokens,answer,tok):
 s=tok.decode(tokens,skip_special_tokens=True)
 normalize=lambda x:"".join(c for c in x.upper() if c.isalnum())
 expected=normalize(answer)
 actual=normalize(s)
 contains=expected in actual
 starts=actual.startswith(expected)
 return {"decoded":s[:400],"expected":answer,
         "contains_identifier":contains,"begins_with_identifier":starts}

def aggregate(rows):
 allcases={}
 for row in rows:
  b=row["variants"]["canonical_q4"]["error_per_layer"]
  candidate=row["variants"]["o0_v96_128"]["error_per_layer"]
  prev=row["variants"]["prior_vfocus_128"]["error_per_layer"]
  allcases[row["case"]+"@"+str(row["position"])]={
   "length":row["length"],"q4_max":max(b),"o0_max":max(candidate),
   "prior_max":max(prev),"q4_mean":sum(b)/LAYERS,
   "o0_mean":sum(candidate)/LAYERS,"prior_mean":sum(prev)/LAYERS,
   "o0_vs_q4_max_nonworse":max(candidate)<=max(b),
   "o0_vs_prior_max_nonworse":max(candidate)<=max(prev)}
 def value(var,stat):
  return [v[stat] for v in allcases.values()]
 qmax=max(value("baseline","q4_max"))
 omax=max(value("target","o0_max"))
 pmax=max(value("prior","prior_max"))
 qmean=sum(value("baseline","q4_mean"))/len(allcases)
 omean=sum(value("target","o0_mean"))/len(allcases)
 pmean=sum(value("prior","prior_mean"))/len(allcases)
 return {"cases":len(allcases),"q4_global_max":qmax,"o0_global_max":omax,
   "prior_global_max":pmax,
   "o0_strict_max_relative_improvement_vs_q4":(qmax-omax)/qmax,
   "o0_strict_max_relative_improvement_vs_prior":(pmax-omax)/pmax,
   "q4_mean":qmean,"o0_mean":omean,"prior_mean":pmean,
   "o0_mean_relative_improvement_vs_q4":(qmean-omean)/qmean,
   "o0_max_nonworse_vs_q4_positions":sum(v["o0_vs_q4_max_nonworse"] for v in allcases.values()),
   "o0_max_nonworse_vs_prior_positions":sum(v["o0_vs_prior_max_nonworse"] for v in allcases.values()),
   "per_case":allcases,
   "by_length":{str(n):{
      "count":sum(v["length"]==n for v in allcases.values()),
      "q4_max":max(v["q4_max"] for v in allcases.values() if v["length"]==n),
      "o0_max":max(v["o0_max"] for v in allcases.values() if v["length"]==n),
      "prior_max":max(v["prior_max"] for v in allcases.values() if v["length"]==n)
    } for n in (128,512,1024)}}

def main():
 for p,sha in ((MODEL,MODEL_SHA),(CONFIG,CONFIG_SHA),(ENGINE,ENGINE_SHA)):
  sha_file(p,sha)
 if not REPO.is_dir():raise RuntimeError("BEGLIN_MISSING")
 tok=get_tokenizer()
 work=Path(tempfile.mkdtemp(prefix="qt-v96-o0-heldout-",dir="/tmp"))
 paths,maps=prep_maps(work)
 cases=[]
 for doc in CORPUS:
  raw,meta=encode_doc(tok,doc)
  p=work/"prompts"/(doc["id"]+".i32")
  p.parent.mkdir(parents=True,exist_ok=True)
  p.write_bytes(raw)
  meta["prompt_sha256"]=sha_bytes(raw)
  cases.append((doc,meta,p))
 if len(cases)!=9 or len({x[1]["prompt_sha256"] for x in cases})!=9:
  raise RuntimeError("HOLDOUT_SET_NOT_UNIQUE")
 if sum(len(x[1]["positions"]) for x in cases)!=15:
  raise RuntimeError("EXPECTED_15_POSITIONS")
 records=[]
 for doc,meta,p in cases:
  for pos in meta["positions"]:
   last=(pos==meta["length"]-1)
   gen=NEW_TOKENS if last else 1
   tested={}
   order=("l0_qkvo_fp32","canonical_q4","prior_vfocus_128","o0_v96_128")
   # All arms execute on identical prompt, same length, same launch mode.
   for arm in order:
    hidden,receipt=run(work,doc["id"]+"_"+str(pos)+"_"+arm,p,pos,ENGINE,
       paths[arm],gen=gen,fp32=(arm=="l0_qkvo_fp32"))
    tested[arm]={"hidden":hidden,"receipt":receipt}
   baseline=tested["l0_qkvo_fp32"]["hidden"]
   variants={}
   quality_results={}
   for arm in ("canonical_q4","prior_vfocus_128","o0_v96_128"):
    vec=tested[arm]["hidden"]
    m=per_layer(baseline,vec)
    variants[arm]={"error_per_layer":m,"max_relative_l2":max(m),
      "mean_relative_l2":sum(m)/LAYERS}
    if last:quality_results[arm]=quality(tested[arm]["receipt"]["generated_ids"],doc["answer"],tok)
   if last:quality_results["l0_qkvo_fp32"]=quality(tested["l0_qkvo_fp32"]["receipt"]["generated_ids"],doc["answer"],tok)
   # At the final English position, check the mixed-all-Q4 adapter versus
   # canonical Q4; no attempt to equate its tiny difference to exact bits.
   control=None
   if last and doc["lang"]=="en":
    check,info=run(work,doc["id"]+"_"+str(pos)+"_allq4_control",
      p,pos,ENGINE,paths["mixed_all_q4"],gen=1)
    rel=relative_l2(tested["canonical_q4"]["hidden"],check)
    if rel>0.001:raise RuntimeError("Q4_KERNAL_CONTRACT_CHANGED:"+doc["id"])
    control={"direct_relative_l2":rel,"run":info}
   row={"case":doc["id"],"length":meta["length"],"position":pos,
        "prompt_sha256":meta["prompt_sha256"],"first_token_arms":{
          arm:tested[arm]["receipt"]["generated_ids"][0] for arm in order},
        "variants":variants,"quality":quality_results if last else None,
        "control":control,
        "receipts":{arm:tested[arm]["receipt"] for arm in order}}
   records.append(row)
   print("QT_V96_O0_CASE="+json.dumps({"case":doc["id"],"pos":pos,
     "q4_max":variants["canonical_q4"]["max_relative_l2"],
     "o0_max":variants["o0_v96_128"]["max_relative_l2"],
     "prior_max":variants["prior_vfocus_128"]["max_relative_l2"],
     "qa_q4":quality_results.get("canonical_q4",{}).get("contains_identifier"),
     "qa_prior":quality_results.get("prior_vfocus_128",{}).get("contains_identifier"),
     "qa_o0":quality_results.get("o0_v96_128",{}).get("contains_identifier")},sort_keys=True),flush=True)
 all_results=aggregate(records)
 qa=[{"case":r["case"],"length":r["length"],
      **{arm:r["quality"][arm] for arm in ("canonical_q4","prior_vfocus_128","o0_v96_128")}}
     for r in records if r["quality"]]
 regressed_from_q4=[x["case"] for x in qa if x["canonical_q4"]["contains_identifier"] and not x["o0_v96_128"]["contains_identifier"]]
 regressed_from_prior=[x["case"] for x in qa if x["prior_vfocus_128"]["contains_identifier"] and not x["o0_v96_128"]["contains_identifier"]]
 gate=(not regressed_from_q4 and not regressed_from_prior
       and all_results["o0_strict_max_relative_improvement_vs_q4"]>0
       and all_results["o0_max_nonworse_vs_q4_positions"]==all_results["cases"])
 payload={
  "schema":SCHEMA,"state":"SUCCEEDED","production_write_allowed":False,
  "automatic_live_promotion":False,
  "experimental_status":"FROZEN_CANDIDATE_EVALUATION_ONLY",
  "candidate_definition":"O0, keep old Q16/K16, replace O32 with next V32 from SHA-pinned ranking; total 128 Q5",
  "validation_type":"new pre-registered artificial natural-language document QA cohort, not reused prior RAVEN/SEIRA/MAPLE/HARBOR/LYRA",
  "model_sha256":MODEL_SHA,"config_sha256":CONFIG_SHA,
  "engine_sha256":ENGINE_SHA,"tokenizer_sha256":TOKENIZER_SHA,
  "tokenizer_config_sha256":TOKCFG_SHA,"rank_sha256":RANK_SHA,
  "fixture_sha256":sha_bytes(Path(__file__).read_bytes()),
  "budgets":{"prior":PRIOR_COUNTS,"candidate":TARGET_COUNTS},
  "maps":maps,"prompts":[meta for _,meta,_ in cases],
  "prompt_text_source":[{"case":x["id"],"fact":x["fact"],"question":x["question"],"answer":x["answer"]} for x,_,_ in cases],
  "runs":records,"summary":all_results,"qa":qa,
  "qa_regressions_vs_q4":regressed_from_q4,
  "qa_regressions_vs_prior":regressed_from_prior,
  "quality_strict_gate_pass":gate,
  "performance_scope":"one run per condition, includes model load+prefill+greedy; sampled process RSS only, not VRAM or packed-size benchmarking"}
 raw=(json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode("utf-8")
 OUTPUT.mkdir(parents=True,exist_ok=True)
 dest=OUTPUT/("QT_V96_O0_HELDOUT_"+uuid.uuid4().hex+".json")
 with dest.open("xb") as f:
  f.write(raw);f.flush();os.fsync(f.fileno())
 print("QT_V96_O0_FINAL="+json.dumps({"schema":SCHEMA,"state":"SUCCEEDED",
  "result_path":str(dest),"result_sha256":sha_bytes(raw),"result_bytes":len(raw),
  "fixture_sha256":payload["fixture_sha256"],"candidate_v96_sha256":V96_SHA,
  "budgets":payload["budgets"],"summary":all_results,
  "qa":[{"case":x["case"],"q4":x["canonical_q4"]["contains_identifier"],
    "prior":x["prior_vfocus_128"]["contains_identifier"],
    "candidate":x["o0_v96_128"]["contains_identifier"]} for x in qa],
  "qa_regressions_vs_q4":regressed_from_q4,
  "qa_regressions_vs_prior":regressed_from_prior,
  "quality_strict_gate_pass":gate,"automatic_live_promotion":False},sort_keys=True,ensure_ascii=False),flush=True)

if __name__=="__main__":
 main()
