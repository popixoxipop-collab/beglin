import os
import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="beglin-dashboard-test-")) / "journal.db"
os.environ["BEGLIN_DASHBOARD_DB"] = str(tmp)

from app.journal import ensure_a0_seeded, list_events
from app.projectors import coverage, event_log, heatmap, metrics
from app.stream import encode_sse


def main() -> None:
    ensure_a0_seeded()
    ensure_a0_seeded()
    rows = list_events()
    assert len(rows) == 2
    assert [r["event_type"] for r in rows] == [
        "canary.passed",
        "production.policy_verified",
    ]

    log = event_log()
    assert log.cursor == 2
    assert len(log.entries) == 2
    assert log.entries[0].evidence_id == "signed-canary"
    assert log.entries[1].evidence_id == "persistent-production"

    hm = heatmap()
    assert hm.source_kind == "evidence_aggregate"
    assert len(hm.cells) == 1
    cell = hm.cells[0]
    assert cell.layer == 3
    assert cell.role_group == "Shared Up"
    assert cell.sample_count == 36
    assert cell.correction_hits == 36
    assert cell.current_precision_n == 6
    assert cell.live_state == "PROMOTED"

    cov = coverage()
    assert cov.total_cells == 3
    assert cov.filled_cells == 3
    assert cov.percentage == 100.0
    assert [c.status for c in cov.cells] == ["PASS", "PROMOTED", "PROMOTED"]

    met = metrics()
    assert met.request_count == 36
    assert met.corrected_hits == 36
    assert met.attribution_before == 1.0
    assert met.attribution_after == 0.0
    assert met.rollback_ready is True
    assert met.resident_worker is False

    frame = encode_sse(rows[0])
    assert frame.startswith("id: 1\nevent: canary.passed\n")
    assert "signed-canary" in frame

    print("journal/projector/SSE PASS")


if __name__ == "__main__":
    main()
