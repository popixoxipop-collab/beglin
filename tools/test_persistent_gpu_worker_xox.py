#!/usr/bin/env python3
"""Real XOX regression for the opt-in persistent MLX worker protocol."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


REPO = Path("/Users/xox/mcp-sandbox/tailnet-commander/beglin-persistent-worker")
BINARY = REPO / "build-gpu-persistent/qwen_infer_gpu"
MOE_BASE = Path("/Users/xox/vdsp_local_data/moe_base_deepseek")
SAFETENSORS = Path(
    "/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/"
    "model.safetensors.index.json"
)
SOURCE_MANIFEST = Path(
    "/Users/xox/vdsp-engine-gpu-precision/g5_drill_manifest.txt"
)

MARKER_RE = re.compile(
    r"GPU_PERSIST_RESULT_V1 generation=(\d+) requests=(\d+) "
    r"finite_logits=(\d+) wall_ms=([0-9.]+) weight_epoch=(\d+)"
)
REQ_RE = re.compile(
    r"\[moe gpu cb online\] req (\d+) .*? tokens:\s*([^\n\r]*)"
)


def atomic_text(path: Path, text: str) -> None:
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp,path)


def source_prompt() -> tuple[Path,int]:
    rows=[
        line.strip() for line in SOURCE_MANIFEST.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    parts=rows[0].split()
    return Path(parts[0]), 10


def minimal_env() -> dict[str,str]:
    out={}
    for key in ("PATH","HOME","USER","LOGNAME","LANG","LC_ALL","TMPDIR"):
        if os.environ.get(key):
            out[key]=os.environ[key]
    return out


def read_cycle(proc: subprocess.Popen, generation: int, timeout: float=120.0) -> dict:
    deadline=time.monotonic()+timeout
    lines=[]
    while time.monotonic()<deadline:
        line=proc.stdout.readline()
        if line == "":
            if proc.poll() is not None:
                raise RuntimeError(
                    f"persistent worker exited rc={proc.returncode} before generation={generation}\n"
                    + "".join(lines[-80:])
                )
            time.sleep(0.01)
            continue
        lines.append(line)
        m=MARKER_RE.search(line)
        if m and int(m.group(1)) == generation:
            text="".join(lines)
            reqs=REQ_RE.findall(text)
            if not reqs:
                raise RuntimeError("request token line missing before persistent result marker")
            request_tokens={
                int(req_id): [int(x) for x in token_text.split()]
                for req_id, token_text in reqs[-int(m.group(2)):]
            }
            if len(request_tokens) != int(m.group(2)):
                raise RuntimeError(
                    f"expected {m.group(2)} request token lines, got {len(request_tokens)}"
                )
            return {
                "generation":generation,
                "requests":int(m.group(2)),
                "finite_logits":m.group(3)=="1",
                "engine_wall_ms":float(m.group(4)),
                "weight_epoch":int(m.group(5)),
                "request_tokens":request_tokens,
                "log_tail":"".join(lines[-40:]),
            }
    raise TimeoutError(f"timed out waiting for generation {generation}")


def run_variant(name: str, *, candidate: bool, expected_token: int) -> dict:
    raw,maxnew=source_prompt()
    with tempfile.TemporaryDirectory(prefix=f"beglin-persist-{name}-") as td:
        root=Path(td)
        manifest=root/"manifest.txt"
        gen=root/"generation.txt"
        ack=root/"ack.json"
        txn=root/"txn.cmd"
        promo=root/"promotion_nq.txt"
        manifest.write_text(f"{raw} {maxnew}\n")
        atomic_text(gen,"1\n")
        promo.write_text("shared_up_proj 3 6\n" if candidate else "")

        env=minimal_env()
        env.update({
            "QWEN_MOE_GPU_CBATCH_ONLINE":"1",
            "QWEN_MOE_BASE":str(MOE_BASE),
            "QWEN_MOE_NEARTIE_CORRECT":"0",
            "QWEN_MOE_CB_PROMPT_MANIFEST":str(manifest),
            "QWEN_MOE_CB_SLOTS":"1",
            "QWEN_MOE_CB_REQS":"1",
            "QWEN_MOE_GPU_VALIDATION_REPORT":"1",
            "QWEN_MOE_GPU_APPLIED_ACK":str(ack),
            "QWEN_MOE_GPU_TXN_FILE":str(txn),
            "QWEN_MOE_GPU_PERSIST_GENERATION_FILE":str(gen),
            "QWEN_MOE_PROMOTION_SAFETENSORS":str(SAFETENSORS),
        })
        if candidate:
            env["QWEN_MOE_PROMOTION_FILE_NQ"]=str(promo)

        started=time.monotonic()
        proc=subprocess.Popen(
            [str(BINARY)],
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        pid=proc.pid
        try:
            first=read_cycle(proc,1)
            first_end=time.monotonic()
            first_tokens=first["request_tokens"][0]
            if len(first_tokens) <= 8 or first_tokens[8] != expected_token:
                raise RuntimeError(
                    f"{name} generation1 token mismatch: {first_tokens}"
                )

            manifest.write_text("".join(f"{raw} {maxnew}\n" for _ in range(4)))
            atomic_text(gen,"2\n")
            second_start=time.monotonic()
            second=read_cycle(proc,2)
            second_end=time.monotonic()
            if proc.pid != pid:
                raise RuntimeError("worker PID changed across persistent generations")
            if second["requests"] != 4:
                raise RuntimeError(f"{name} generation2 request count mismatch: {second['requests']}")
            for req_id,tokens in second["request_tokens"].items():
                if len(tokens) <= 8 or tokens[8] != expected_token:
                    raise RuntimeError(
                        f"{name} generation2 req={req_id} token mismatch: {tokens}"
                    )

            manifest.write_text(f"{raw} {maxnew}\n")
            atomic_text(gen,"3\n")
            third_start=time.monotonic()
            third=read_cycle(proc,3)
            third_end=time.monotonic()
            third_tokens=third["request_tokens"][0]
            if len(third_tokens) <= 8 or third_tokens[8] != expected_token:
                raise RuntimeError(
                    f"{name} generation3 token mismatch: {third_tokens}"
                )
            if proc.pid != pid:
                raise RuntimeError("worker PID changed after batch shrink")

            return {
                "name":name,
                "pid":pid,
                "candidate":candidate,
                "expected_token":expected_token,
                "generation1":first,
                "generation2":second,
                "generation3":third,
                "startup_to_generation1_ms":round((first_end-started)*1000,3),
                "generation2_end_to_end_ms":round((second_end-second_start)*1000,3),
                "generation3_end_to_end_ms":round((third_end-third_start)*1000,3),
                "same_pid":True,
            }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def main() -> int:
    baseline=run_variant("baseline",candidate=False,expected_token=3268)
    candidate=run_variant("candidate",candidate=True,expected_token=1224)
    result={
        "schema":"beglin-persistent-gpu-worker-xox-v1",
        "status":"PASS",
        "baseline":baseline,
        "candidate":candidate,
        "persistent_pid_reuse":baseline["same_pid"] and candidate["same_pid"],
    }
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
