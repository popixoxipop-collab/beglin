#!/usr/bin/env python3
import argparse,glob,shutil,subprocess
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--shards-dir",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--windows",type=int,default=4);a=p.parse_args()
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);n=0
 for i in range(8):
  d=out/f"shard-{i:02d}";d.mkdir(exist_ok=True)
  subprocess.run(["python3","tools/qt_attention_shard_heatmap.py","--checkpoint",a.checkpoint,"--shard",f"{a.shards_dir}/shard-{i:02d}.json","--output-dir",str(d),"--windows",str(a.windows)],check=True)
  for x in d.glob("*.bits"): shutil.copy2(x,out/x.name);n+=1
 if n!=96: raise SystemExit(f"expected 96 maps, got {n}")
 print(f"QT_FULL_MAPS_PASS count={n}")
if __name__=="__main__":main()
