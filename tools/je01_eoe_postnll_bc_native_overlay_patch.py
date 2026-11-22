#!/usr/bin/env python3
"""Scratch-only original27 BC pre-registered 7-expert native loader patch.

Apply after the existing source-verified CAL240 A loader patch. Only the
two registered exact-byte B/C policies are accepted; original E64 AF4
unchanged roles and 6 fixed Q5 tensors stay untouched. Runtime arm is
selected by JE01_CAL240_POSTNLL_ARM; it is NOT accepted from TSV content.
No model/holdout execution on import or when used by native compiler.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PREREG=ROOT/"docs/research/JE01_REAL27_NLL_POST_FAILURE_EARLY_EXPERT_SWAP_PREREG_20261122.json"
ORIGINAL=ROOT/"tools/je01_eoe_original384_cal240_sse_role_specific_overlay_stage.py"
PRE_BLOB="08c493215078b83481c871203d1e08232f2646e3"
ORIGINAL_BLOB="e25125d56bccef14fbb6984b6ca89cfb5d3b5839"
NEW_MARK="JE01_CAL240_POSTNLL_BC_SEVEN_EXPERTS_REGISTERED"
SCHEMA="beglin-je01-original27-scratch-only-postnll-bc-pre_registered-7expert-overlay/1"


def blob(p):
    b=p.read_bytes()
    return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()


def arm_roles():
    if blob(PREREG)!=PRE_BLOB or blob(ORIGINAL)!=ORIGINAL_BLOB:
        raise ValueError("JE01_BC_SOURCE_OR_POLICY_SHA_DRIFT")
    data=json.loads(PREREG.read_text())
    if data.get("stage")!="PRE_REGISTERED_CONDITIONAL_UNRUN":
        raise ValueError("JE01_BC_POLICY_NOT_FROZEN")
    arms={}
    universe={tuple(s.split("/")) for s in data["budget"]["original_role_universe"]}
    q5={tuple(s.split("/")) for s in data["budget"]["Q5_FROZEN"]}
    for key in ("B_PROTECT_EARLY_GATE","C_PROTECT_EARLY_UP"):
        a=next((a for a in data["arms"] if a["id"]==key),None)
        if a is None:raise ValueError("JE01_BC_ARM_UNREGISTERED")
        q4={tuple(s.split("/")) for s in a["q4"]}
        if len(q4)!=6 or q4&q5 or q4|q5>universe or a["actual_43mb_bytes"]!=43253760:
            raise ValueError("JE01_BC_INCORRECT_Q4_GROUP_POLICY")
        arms[key]=q4
    if arms["B_PROTECT_EARLY_GATE"]==arms["C_PROTECT_EARLY_UP"]:
        raise ValueError("JE01_BC_ARMS_NOT_INDEPENDENT_CONTRAST")
    return arms


def q4_expr(roles):
    expr=[]
    for layer,expert,role in sorted(roles):
        idx={"gate_proj":0,"up_proj":1,"down_proj":2}[role]
        expr.append("(r.layer=="+layer+" && r.expert=="+expert+
                    " && idx=="+str(idx)+")")
    return "("+" || ".join(expr)+")"


def patch(cpp,c):
    arms=arm_roles()
    spec=importlib.util.spec_from_file_location("je01_original_cal240_frozen_stage",ORIGINAL)
    original=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    a,b=original.patch(cpp,c)
    old="""const bool q4=((r.layer==12 && r.expert==13) ||
                            (r.layer==13 && r.expert==22) ||
                            (r.layer==13 && r.expert==28)) && idx>=0 && idx<2;"""
    if a.count(old)!=1 or b.count(original.NEW_MARK)!=1:
        raise ValueError("JE01_BC_EXACT_ORIGINAL_A_LOADER_ANCHOR_LOST")
    dynamic="""const bool q4=(
                (arm=="B_PROTECT_EARLY_GATE" && """+q4_expr(arms["B_PROTECT_EARLY_GATE"])+""") ||
                (arm=="C_PROTECT_EARLY_UP" && """+q4_expr(arms["C_PROTECT_EARLY_UP"])+"""));"""
    a=a.replace(old,dynamic,1)
    manifest="std::ifstream manifest(tsv);"
    if a.count(manifest)!=1:
        raise ValueError("JE01_BC_MANIFEST_ANCHOR_AMBIGUOUS")
    a=a.replace(manifest,"""const char* arm_env=std::getenv("JE01_CAL240_POSTNLL_ARM");
         if(!arm_env) return 0;
         const std::string arm(arm_env);
         if(arm!="B_PROTECT_EARLY_GATE" && arm!="C_PROTECT_EARLY_UP")return 0;
         std::ifstream manifest(tsv);""",1)
    a="#include <cstdlib>\n"+a
    b=b.replace(original.NEW_MARK,NEW_MARK,1)
    if (a.count("JE01_CAL240_POSTNLL_ARM")!=1 or
        a.count("const bool q4=")!=1):
        raise ValueError("JE01_BC_Q4_MASK_UNIQUENESS_FAILED")
    if b.count(NEW_MARK)!=1 or b.count("mlx_gpu_je01_overlay_count()!=7")!=1:
        raise ValueError("JE01_BC_NATIVE_ROLEWISE_REGISTER_COUNT_OR_MARKER")
    return a,b
