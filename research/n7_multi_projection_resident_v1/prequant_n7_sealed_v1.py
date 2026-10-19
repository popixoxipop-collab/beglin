#!/usr/bin/env python3
"""Research-only, sealed *prequantized* n7/group64 role reader for BEGLIN v11.

Exports already-quantized canonical bitplanes+F32 scales, then reloads with
read-only mmap and externally pinned manifest / checkpoint-index SHA-256.
Never reads BF16 source tensors while loading or performs quantization.
The caller must independently attest the offline transform/source provenance.
No serving/runtime switch or all-model-resident claim is implied.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import stat
import sys

import numpy as np

SCHEMA = "beglin.qng64.n7.prequant-sealed/1"
MAX_MANIFEST = 4 * 1024 * 1024
MAX_ROLES = 5300
MAX_TENSOR_BYTES = 96 * 1024 * 1024
ROLE = re.compile(r"^model[.]layers[.](?:0|[1-9][0-9]?)[.][a-zA-Z0-9_.]{1,145}[.]weight$")
SHA = re.compile(r"^[0-9a-f]{64}$")


def require(ok, message):
    if not ok:
        raise ValueError("BEGLIN_PREQUANT_N7_FAIL_CLOSED: " + message)


def file_sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_role(role):
    require(type(role) is str and len(role) <= 180 and ROLE.fullmatch(role) is not None,
            "invalid or unsafe source role")
    layer = int(role.split(".")[2])
    require(layer < 27, "not a DeepSeek-V2-Lite decoder source layer")
    return role


def geometry(out, in_features):
    require(type(out) is int and type(in_features) is int and
            1 <= out <= 16384 and 64 <= in_features <= 16384 and
            in_features % 64 == 0, "unsupported original n7/group64 geometry")
    groups = out * (in_features // 64)
    p, s = groups * 56, groups * 4
    require(p + s <= MAX_TENSOR_BYTES, "single packed role exceeds native limit")
    return p, s


def role_files(role):
    ident = file_sha(canonical_role(role).encode("ascii"))
    return ident + ".planes", ident + ".scales"


class PackedN7:
    """Read-only mmap-backed source module shape expected by ResidentN7Cache."""
    def __init__(self, role, out, in_features, planes, scales, mmaps):
        self.role = role
        self.n = 7
        self.out = out
        self.in_features = in_features
        self.packed = planes
        self.scales = scales
        self.serialized_bytes = planes.nbytes + scales.nbytes
        self._maps = mmaps


class PackWriter:
    """Offline staging writer; *inputs must already be canonically quantized*.

    Creates a NEW output directory with exclusive file creation. Never
    overwrites an existing artifact, checkpoint, manifest, or role.
    Unsealed/incomplete directories are intentionally unreadable.
    """
    def __init__(self, root, source_checkpoint_index_sha256):
        require(sys.byteorder == "little", "little-endian F32 required")
        require(type(source_checkpoint_index_sha256) is str and
                SHA.fullmatch(source_checkpoint_index_sha256) is not None,
                "externally attested checkpoint index SHA-256 required")
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.root = self.root.resolve(strict=True)
        self.source_sha = source_checkpoint_index_sha256
        self.roles = {}
        self.sealed = False

    def add(self, role, out, in_features, planes, scales):
        require(not self.sealed and len(self.roles) < MAX_ROLES, "writer sealed/full")
        canonical_role(role)
        require(role not in self.roles, "duplicate source role")
        pbytes, sbytes = geometry(out, in_features)
        require(isinstance(planes, np.ndarray) and planes.dtype == np.uint8 and
                planes.flags.c_contiguous and planes.nbytes == pbytes and
                isinstance(scales, np.ndarray) and scales.dtype == np.float32 and
                scales.flags.c_contiguous and scales.nbytes == sbytes and
                np.isfinite(scales).all() and np.all(scales > 0),
                "canonical native packed bitplanes/F32 positive scales required")
        pn, sn = role_files(role)
        pb, sb = planes.tobytes(order="C"), scales.tobytes(order="C")
        for filename, raw in ((pn, pb), (sn, sb)):
            dest = self.root / filename
            with dest.open("xb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            dest.chmod(0o400)
        self.roles[role] = dict(
            role=role, out=out, in_features=in_features,
            planes_sha256=file_sha(pb), scales_sha256=file_sha(sb))
        return pbytes + sbytes

    def seal(self):
        require(not self.sealed and self.roles, "empty/already sealed writer")
        manifest = dict(schema=SCHEMA,
                        source_checkpoint_index_sha256=self.source_sha,
                        roles=[self.roles[k] for k in sorted(self.roles)])
        raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        require(len(raw) <= MAX_MANIFEST, "manifest exceeds bounded size")
        path = self.root / "manifest.json"
        with path.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        path.chmod(0o400)
        self.sealed = True
        return file_sha(raw)


class PackReader:
    def __init__(self, root, expected_manifest_sha256, expected_checkpoint_index_sha256):
        require(sys.byteorder == "little", "little-endian F32 required")
        require(type(expected_manifest_sha256) is str and
                SHA.fullmatch(expected_manifest_sha256) is not None and
                type(expected_checkpoint_index_sha256) is str and
                SHA.fullmatch(expected_checkpoint_index_sha256) is not None,
                "manifest and source identity must be externally SHA pinned")
        self.root = Path(root)
        require(not self.root.is_symlink() and self.root.is_dir(),
                "sealed artifact directory absent or symlink")
        self.root = self.root.resolve(strict=True)
        raw = self._read_manifest()
        require(file_sha(raw) == expected_manifest_sha256,
                "sealed manifest SHA-256 mismatch")
        obj = json.loads(raw)
        require(type(obj) is dict and set(obj) ==
                {"schema", "source_checkpoint_index_sha256", "roles"} and
                obj["schema"] == SCHEMA and
                obj["source_checkpoint_index_sha256"] == expected_checkpoint_index_sha256,
                "schema/source checkpoint identity mismatch")
        rows = obj["roles"]
        require(type(rows) is list and 0 < len(rows) <= MAX_ROLES,
                "bounded nonempty role manifest required")
        self.roles = {}
        total = 0
        prev = ""
        for row in rows:
            require(type(row) is dict and set(row) ==
                    {"role", "out", "in_features", "planes_sha256", "scales_sha256"},
                    "unknown/missing role fields")
            role = canonical_role(row["role"])
            require(role > prev and role not in self.roles, "roles must be unique and sorted")
            prev = role
            p, s = geometry(row["out"], row["in_features"])
            require(type(row["planes_sha256"]) is str and SHA.fullmatch(row["planes_sha256"]) and
                    type(row["scales_sha256"]) is str and SHA.fullmatch(row["scales_sha256"]),
                    "invalid packed file hash")
            total += p + s
            require(total <= 64 * 1024**3, "manifest total exceeds 64 GiB")
            self.roles[role] = row
        self.total_packed_bytes = total
        self.manifest_sha256 = expected_manifest_sha256

    def _read_manifest(self):
        path = self.root / "manifest.json"
        require(not path.is_symlink(), "manifest symlink rejected")
        info = path.stat()
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= MAX_MANIFEST,
                "bounded regular sealed manifest required")
        return path.read_bytes()

    def _map(self, name, expected_bytes, digest, dtype):
        path = self.root / name  # name is computed from SHA(role), never manifest-controlled
        require(not path.is_symlink(), "packed role symlink rejected")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and before.st_size == expected_bytes,
                    "packed source payload size/type mismatch")
            h = hashlib.sha256()
            with os.fdopen(os.dup(fd), "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(block)
            require(h.hexdigest() == digest, "packed source payload SHA-256 mismatch")
            after = os.fstat(fd)
            require((before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns,
                     before.st_size) ==
                    (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns,
                     after.st_size), "payload changed during SHA verification")
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            data = np.frombuffer(mm, dtype=dtype)
            require(data.nbytes == expected_bytes and not data.flags.writeable,
                    "read-only mmap byte accounting changed")
            return data, mm
        finally:
            os.close(fd)

    def load(self, role):
        canonical_role(role)
        require(role in self.roles, "unapproved/unlisted source role")
        row = self.roles[role]
        pbytes, sbytes = geometry(row["out"], row["in_features"])
        pn, sn = role_files(role)
        planes, pm = self._map(pn, pbytes, row["planes_sha256"], np.uint8)
        scales, sm = self._map(sn, sbytes, row["scales_sha256"], np.dtype("<f4"))
        require(scales.dtype == np.float32 and np.isfinite(scales).all() and
                np.all(scales > 0), "source scales invalid after mmap")
        return PackedN7(role, row["out"], row["in_features"],
                        planes, scales, (pm, sm))
