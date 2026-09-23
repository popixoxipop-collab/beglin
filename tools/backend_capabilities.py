#!/usr/bin/env python3
"""Collect immutable backend capability/build identity without running inference."""
import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone


GPU_NATIVE_QNG64 = (2, 3, 5, 6)
GPU_CUSTOM_QNG64 = (7, 9, 10, 11, 12, 13, 14, 15)


def _run(host, command, timeout=30):
    argv = ["ssh", host, command] if host else ["sh", "-lc", command]
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return p.stdout


def _identity(host, binary):
    q = shlex.quote(binary)
    out = _run(
        host,
        (
            f"set -e; shasum -a 256 {q}; stat -f '%z' {q}; "
            "hostname; uname -m; "
            f"(otool -L {q} || true); "
            f"(nm -gU {q} 2>/dev/null | grep 'mlx_gpu_available' || true)"
        ),
    )
    lines = out.splitlines()
    if len(lines) < 4:
        raise RuntimeError(f"incomplete identity output: {lines!r}")
    sha = lines[0].split()[0].lower()
    if len(sha) != 64:
        raise RuntimeError(f"invalid sha256 line: {lines[0]!r}")
    tail = "\n".join(lines[4:])
    return {
        "host": lines[2].strip(),
        "arch": lines[3].strip(),
        "binary_path": binary,
        "binary_sha256": sha,
        "binary_size": int(lines[1].strip()),
        "mlx_symbol_present": "mlx_gpu_available" in tail,
        "link_report": tail,
    }


def collect(host, binary):
    ident = _identity(host, binary)
    gpu_compiled = bool(ident["mlx_symbol_present"])
    gpu_widths = {}
    for n in GPU_NATIVE_QNG64:
        gpu_widths[str(n)] = (
            "IMPLEMENTED_UNVERIFIED" if gpu_compiled else "UNSUPPORTED_BINARY"
        )
    for n in GPU_CUSTOM_QNG64:
        gpu_widths[str(n)] = (
            "IMPLEMENTED_UNVERIFIED" if gpu_compiled else "UNSUPPORTED_BINARY"
        )
    result = {
        "schema": "precision-capability-v1",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "worker": ident,
        "backends": {
            "cpu": {"compiled": True, "status": "PRESENT"},
            "mlx_metal": {
                "compiled": gpu_compiled,
                "status": "IMPLEMENTED_UNVERIFIED" if gpu_compiled else "UNSUPPORTED_BINARY",
                "qng64_widths": gpu_widths,
                "native_widths": list(GPU_NATIVE_QNG64),
                "custom_metal_widths": list(GPU_CUSTOM_QNG64),
            },
        },
    }
    canonical = json.dumps(result["worker"], sort_keys=True, separators=(",", ":"))
    result["worker_identity_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", help="SSH host; omit for local machine")
    ap.add_argument("--binary", required=True)
    ap.add_argument("--output")
    args = ap.parse_args()
    result = collect(args.host, args.binary)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
