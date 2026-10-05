#!/usr/bin/env python3
"""Closed QT experiment dispatcher. No production promotion."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
MANIFEST=ROOT/"fixtures/qt_fixture_manifest_v1.json"

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def verify(m):
 if m["schema"]!="beglin-qt-fixture/1": raise SystemExit("BAD_SCHEMA")
 if m.get("automatic_live_promotion") is not False: raise SystemExit("PROMOTION_MUST_BE_FALSE")
 for rel,want in m["files"].items():
  p=ROOT/rel
  if not p.is_file() or sha(p)!=want: raise SystemExit("FIXTURE_HASH_MISMATCH:"+rel)

def run(action,m):
 a=m["actions"][action]
 cmd=[sys.executable,str(ROOT/a["script"])]
 for x in a.get("args",[]): cmd.append(x)
 r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=a.get("timeout_seconds",600))
 sys.stdout.write(r.stdout);sys.stderr.write(r.stderr)
 if r.returncode: raise SystemExit(r.returncode)

def main():
 p=argparse.ArgumentParser();p.add_argument("--action",required=True,choices=["uncertainty"])
 a=p.parse_args();m=json.loads(MANIFEST.read_text());verify(m);run(a.action,m)
if __name__=="__main__":main()
