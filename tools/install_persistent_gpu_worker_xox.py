#!/usr/bin/env python3
"""Install the reviewed persistent Beglin GPU worker artifact on XOX.

Fixed source:
- repository popixoxipop-collab/beglin
- workflow run 37025197232
- artifact beglin-persistent-gpu-worker-arm64
- artifact ZIP digest sha256:4bad32a0...
- binary SHA256 a47e01dc...

The production qwen_infer_gpu binary is never replaced.  The persistent worker
is installed to /Users/xox/vdsp_serving/bin/qwen_infer_gpu_persistent.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import zipfile

REPO="popixoxipop-collab/beglin"
RUN_ID="37025197232"
ARTIFACT_NAME="beglin-persistent-gpu-worker-arm64"
EXPECTED_BINARY_SHA="a47e01dcf1c5f4fc8cb9c94eab82abaecf5f9f35d5f55d49af28284565b7931e"
DEST=Path("/Users/xox/vdsp_serving/bin/qwen_infer_gpu_persistent")


class InstallError(RuntimeError):
    pass


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def install()->dict:
    gh=shutil.which("gh")
    if not gh:
        raise InstallError("gh CLI unavailable")
    with tempfile.TemporaryDirectory(prefix="beglin-persistent-install-") as td:
        root=Path(td)
        proc=subprocess.run(
            [gh,"run","download",RUN_ID,"-R",REPO,"-n",ARTIFACT_NAME,"-D",str(root)],
            text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,
        )
        if proc.returncode != 0:
            raise InstallError("GitHub artifact download failed: "+proc.stderr.strip())

        tar_path=root/"beglin-persistent-gpu-worker-arm64.tar.gz"
        sha_path=root/"persistent_binary.sha256"
        if not tar_path.is_file() or not sha_path.is_file():
            raise InstallError("artifact contents missing")
        declared=sha_path.read_text().strip().split()[0]
        if declared != EXPECTED_BINARY_SHA:
            raise InstallError(
                f"artifact declared binary SHA mismatch expected={EXPECTED_BINARY_SHA} actual={declared}"
            )

        unpack=root/"unpack"
        unpack.mkdir()
        with tarfile.open(tar_path,"r:gz") as tf:
            names=tf.getnames()
            if names != ["qwen_infer_gpu"]:
                raise InstallError(f"unexpected tar members: {names!r}")
            tf.extractall(unpack)
        src=unpack/"qwen_infer_gpu"
        actual=sha256_file(src)
        if actual != EXPECTED_BINARY_SHA:
            raise InstallError(
                f"binary SHA mismatch expected={EXPECTED_BINARY_SHA} actual={actual}"
            )

        DEST.parent.mkdir(parents=True,exist_ok=True)
        tmp=DEST.with_name(DEST.name+f".tmp.{os.getpid()}")
        shutil.copyfile(src,tmp)
        os.chmod(tmp,0o755)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp,DEST)
        dfd=os.open(str(DEST.parent),os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    installed=sha256_file(DEST)
    if installed != EXPECTED_BINARY_SHA:
        raise InstallError("post-install SHA verification failed")
    return {
        "schema":"beglin-persistent-gpu-worker-install-v1",
        "status":"INSTALLED",
        "path":str(DEST),
        "sha256":installed,
        "mode":oct(DEST.stat().st_mode & 0o777),
        "production_binary_replaced":False,
        "workflow_run_id":RUN_ID,
        "artifact_name":ARTIFACT_NAME,
    }


if __name__=="__main__":
    print(json.dumps(install(),indent=2,sort_keys=True))
