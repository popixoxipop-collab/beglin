#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;MANIFEST=ROOT/"fixtures/qt_fixture_manifest_v2.json"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--action",required=True,choices=["uncertainty","active_observe","calibrate_beval"]);a=p.parse_args();m=json.loads(MANIFEST.read_text())
 if m["schema"]!="beglin-qt-fixture/2" or m.get("automatic_live_promotion") is not False:raise SystemExit("BAD_MANIFEST")
 for rel,want in m["files"].items():
  if sha(ROOT/rel)!=want:raise SystemExit("HASH_MISMATCH:"+rel)
 x=m["actions"][a.action];r=subprocess.run([sys.executable,str(ROOT/x["script"]),*x["args"]],cwd=ROOT,text=True,capture_output=True,timeout=x.get("timeout_seconds",600));sys.stdout.write(r.stdout);sys.stderr.write(r.stderr);raise SystemExit(r.returncode)
if __name__=="__main__":main()
