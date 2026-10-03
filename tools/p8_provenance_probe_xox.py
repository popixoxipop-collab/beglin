#!/usr/bin/env python3
import json,shutil
from pathlib import Path
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
ps.PERSISTENT_BINARY=Path("/Users/xox/vdsp-precision-p7-soak-20261003/build-p7/qwen_infer_gpu")
root=Path("/Users/xox/vdsp_serving/p8-replay-provenance-probe")
shutil.rmtree(root,ignore_errors=True)
w=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root/"worker")
try:
    w.start()
    cfg=w.configure_replay_provenance_capture(root=root/"provenance")
    got=w.submit_base([(base._read_first_certified_prompt(),10)])
    print(json.dumps({"config":cfg,"finite":got["finite_logits"],"token8":got["responses"][0][8],"event_count":len(got["neartie_events"]),"provenance":got.get("replay_provenance")},indent=2))
finally:
    w.stop(force=True)
