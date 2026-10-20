"""Dataset provenance and preregistration leakage guards (stdlib only).
Requires externally collected real dataset row IDs; never generates fake trials.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path

VALID_SPLITS = {
    "pilot", "train", "dev", "calibration", "final_iid", "final_ood",
    "anchors", "behavior_train", "behavior_dev", "behavior_final",
}
FINAL_SPLITS = {"final_iid", "final_ood", "behavior_final"}

def sha_text(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        raise ValueError("missing text")
    # Only whitespace normalization. NOT a semantic-dedup guarantee.
    t = " ".join(s.split())
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

def stable_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()

def audit_rows(rows: list[dict], *, consumed_v6_text_sha256: set[str] | None = None) -> dict:
    """Check exact-text and declared semantic-cluster disjointness across splits.
    When V6 identities are unknown, freeze is BLOCKED, not assumed safe.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("a nonempty set of source rows is required")
    consumed = set(consumed_v6_text_sha256 or [])
    # Cardinality or a list of plausible hashes is NOT independent evidence of
    # V6 holdout identities. Do not certify an attacker-supplied set of 24 strings.
    v6_format_checked = (len(consumed) >= 24 and all(isinstance(x, str) and len(x) == 64
                   and all(c in "0123456789abcdef" for c in x) for x in consumed))
    v6_verified = False  # Needs authenticated source-to-row provenance, not just count.
    seen_id, seen_text, seen_group = {}, {}, {}
    counts = Counter()
    issues = []
    hashes = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {i} must be a dictionary")
        req = ("dataset", "dataset_revision", "source_id", "cluster_id", "split", "text")
        if any(k not in row or not isinstance(row[k], str) or not row[k].strip() for k in req):
            raise ValueError(f"row {i} is missing required provenance fields")
        split = row["split"]
        if split not in VALID_SPLITS:
            raise ValueError(f"unrecognized split: {split}")
        txt_sha = sha_text(row["text"])
        ident = (row["dataset"], row["dataset_revision"], row["source_id"])
        grp = (row["dataset"], row["cluster_id"])
        if ident in seen_id:
            issues.append(f"duplicate source record at rows {seen_id[ident]}/{i}")
        if txt_sha in seen_text:
            issues.append(f"duplicate normalized text at rows {seen_text[txt_sha]}/{i}")
        if grp in seen_group and seen_group[grp][0] != split:
            issues.append(f"semantic group split leakage: row {i} conflicts with {seen_group[grp][1]}")
        if txt_sha in consumed:
            issues.append(f"reused V6 evaluated content at row {i}")
        seen_id[ident], seen_text[txt_sha], seen_group[grp] = i, i, (split, i)
        counts[split] += 1
        hashes.append(stable_sha({k: row[k] for k in req if k != "text"} | {"text_sha256": txt_sha}))
    status = "INVALID" if issues else "V6_EXCLUSION_UNVERIFIED"
    return {"schema": "oml-acs-dataset-audit/1", "status": status,
            "row_count": len(rows), "counts": dict(sorted(counts.items())),
            "issues": issues, "row_manifest_sha256": stable_sha(sorted(hashes)),
            "v6_identity_verified": v6_verified,
            "v6_supplied_hashes_format_valid": v6_format_checked,
            "v6_exclusion_rule": "requires independently attested origin-and-identity binding of original 24 documents",
            "note": "Declared cluster IDs are not a substitute for independent near-duplicate audit."}

def load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("source_jsonl", help="actual pinned-source rows, NOT synthetic final data")
    ap.add_argument("--v6-hashes-json", help="JSON array of SHA256 hashes of consumed V6 documents")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    v6 = set(json.loads(Path(args.v6_hashes_json).read_text())) if args.v6_hashes_json else None
    report = audit_rows(load_jsonl(args.source_jsonl), consumed_v6_text_sha256=v6)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(report["status"], report["row_count"], report["row_manifest_sha256"])
    if report["status"] == "INVALID":
        raise SystemExit(2)
