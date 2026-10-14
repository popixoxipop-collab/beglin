#!/usr/bin/env python3
"""Independent, no-inference P8 evidence seal.

Repairs only the *post-run receipt* left absent by a Python path expression bug
after native 12-domain capture had written parity.json and native-phases.json.
Never replays sealed native captures. Makes no model speedup/quality claim.
"""
import hashlib,json,math,pathlib,statistics,subprocess,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from hcp_p8_profile_patch import validate_phases,strip_observer

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p8-native-20261014-1245")
STAGE=pathlib.Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
ROUTE=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
P6ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/hcp-p6-af-logits-20261014-1104")
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
PROFILE_SOURCE_SHA="84ff47596d6912741b8dcd14c8bdb2028f09031c4c48b78272314a4374d88edf"
PLAIN_SOURCE_SHA="bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"

def sh(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def live_pid_set():
    stdout=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True)
    return sorted(int(l.split()[0]) for l in stdout.splitlines()
                  if str(STAGE) in l)

def seal():
    assert ROOT.is_dir() and ROOT.resolve()==ROOT and not ROOT.is_symlink()
    pre=json.loads((ROOT/"preflight.json").read_text())
    a=ROOT/"results/parity.json";p=ROOT/"results/native-phases.json"
    if any(not f.is_file() or f.is_symlink() for f in (a,p)):raise ValueError("untrusted missing completed trial")
    parity=json.loads(a.read_text());profile=json.loads(p.read_text())
    if (parity.get("schema")!="beglin-hcp-p8-native-observer-parity-v1"
        or parity.get("pairs")!=8 or parity.get("all_native_sequences_exact") is not True
        or parity.get("ACKs_verified") is not True):
        raise ValueError("incomplete real on/off parity")
    x=parity["control"];y=parity["observed"]
    if len(x)!=8 or len(y)!=8: raise ValueError("wrong replay size")
    for u,v in zip(x,y):
        if any(u[k]!=v[k] for k in ("prompt_id","position","policy","seq_sha256","generation_length")):
            raise ValueError("nonidentical observer output")
    r=profile["records"]
    if (profile.get("schema")!="beglin-hcp-p8-xox-real-9phase-v1"
        or profile.get("transition_count")!=len(r) or len(r)!=35
        or profile.get("seq_ack_parity_pass") is not True):
        raise ValueError("P8 phase corpus incomplete")
    if len({rec["domain"] for rec in r})!=12:raise ValueError("not all 12 domains")
    for row in r:validate_phases(row)
    if sh(ROOT/"src/qwen_infer.c")!=PROFILE_SOURCE_SHA or sh(ROOT/"plain_cb128_qwen_infer.c")!=PLAIN_SOURCE_SHA:
        raise ValueError("P8 observer/plain source bytes mismatch")
    if strip_observer((ROOT/"src/qwen_infer.c").read_text())!=(ROOT/"plain_cb128_qwen_infer.c").read_text():
        raise ValueError("P8 observer removed does not restore exact original math")
    if sh(STAGE)!=STAGE_SHA or sh(ROUTE)!=ROUTE_SHA:
        raise ValueError("production original changed")
    if pre["route_sha_before"]!=ROUTE_SHA or pre["original_stage_sha"]!=STAGE_SHA:
        raise ValueError("preflight live identity mismatch")
    current=live_pid_set()
    if current!=[35481,35493]:raise ValueError("unexpected production PID drift")
    d={"schema":"beglin-hcp-p8-xox-measurement-recovered-receipt-v1",
       "status":"NATIVE_EVIDENCE_COMPLETE_POSTRUN_RECEIPT_EXCEPTION_DISCLOSED",
       "original_native_phase_process_exit_code":1,
       "postrun_exception":"TypeError: PosixPath + str in final receipt filename only",
       "native_phase_replayed_to_repair":False,
       "phase_count":35,"source_domains":12,"control_observer_output_pairs":8,
       "logit_token_parity_preserved":True,
       "phase_sum_35_35_pass":True,
       "means_ms":{key:statistics.mean(z[key] for z in r) for key in
          ("entry_to_ack_ms","mlx_bind_ms","qng64_cache_lookup_or_build_ms","durable_ack_ms")},
       "source_commit":pre["source_head"],
       "source_sha256":{"plain_C":PLAIN_SOURCE_SHA,"observer_C":PROFILE_SOURCE_SHA},
       "files_sha256":{"parity.json":sh(a),"native-phases.json":sh(p)},
       "preflight_sha256":sh(ROOT/"preflight.json"),
       "live_stage_sha256":STAGE_SHA,"live_route_sha256":ROUTE_SHA,
       "live_stage_pids":current,"production_touched":False,"automatic_live_promotion":False,
       "new_speedup_proven":False}
    dest=ROOT/"results/research-receipt-profile.json"
    with dest.open("x") as out:json.dump(d,out,indent=2,sort_keys=True,allow_nan=False);out.write("\n")
    print("P8_POSTRUN_RECEIPT_RECOVERED",json.dumps({"sha256":sh(dest),"phase_count":35,
              "no_replay":True,"live_pids":current,"original_process_exit":1}),flush=True)

if __name__=="__main__":seal()
