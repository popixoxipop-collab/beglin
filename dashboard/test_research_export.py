from app.research import comparison_view, export_view


def main() -> None:
    comparison = comparison_view()
    assert comparison.entries, "comparison entries must not be empty"

    controls = comparison.control_consistency
    assert controls.same_source_commit is False
    assert len(controls.distinct_source_commits) >= 2
    assert controls.same_binary_sha256 is True
    assert controls.same_checkpoint_sha256 is True

    kv = [
        entry
        for entry in comparison.entries
        if entry.target is not None
        and entry.target.role == "kv_a_proj_with_mqa"
        and entry.target.layer == 11
        and entry.real_gpu
        and not entry.decision_only
    ]
    assert {entry.target.n for entry in kv} >= {5, 6, 7}

    regressions = [
        entry
        for entry in comparison.entries
        if entry.target is not None
        and entry.target.role == "shared_up_proj"
        and entry.target.layer == 3
        and entry.target.n == 7
        and entry.status == "REGRESSION_DETECTED"
    ]
    assert regressions
    assert all(entry.rollback_required for entry in regressions)

    no_bad = [
        entry
        for entry in comparison.entries
        if entry.status == "NO_BAD_CANDIDATE_WITHIN_BUDGET"
    ]
    assert no_bad
    assert all(entry.status != "PASS" for entry in no_bad)

    a0 = [
        entry
        for entry in comparison.entries
        if entry.source_kind in {"A0_CANARY", "A0_PERSISTENT"}
    ]
    assert len(a0) == 2
    assert all(
        entry.control.source_commit
        == "330954b27f146b8a17db2cb353c3e620968bad5e"
        for entry in a0
    )

    export = export_view()
    assert export.csv_columns
    assert export.csv_rows
    assert export.svg_ready_heatmap
    assert any(
        row["status"] == "NO_BAD_CANDIDATE_WITHIN_BUDGET"
        for row in export.csv_rows
    )
    assert any(
        row["status"] == "REGRESSION_DETECTED"
        and row["rollback_required"] is True
        for row in export.csv_rows
    )

    print(
        "research comparison/export PASS "
        f"entries={len(comparison.entries)} "
        f"csv_rows={len(export.csv_rows)} "
        f"heatmap_cells={len(export.svg_ready_heatmap)}"
    )


def test_csv_http() -> None:
    import csv
    import io
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/v1/research/export.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"] == 'attachment; filename="beglin-research-export.csv"'
    assert response.headers["x-beglin-export-schema"] == "beglin-research-export/1"

    rows = list(csv.DictReader(io.StringIO(response.text)))
    view = export_view()
    assert len(rows) == len(view.csv_rows)
    assert rows[0]["event_id"] == str(view.csv_rows[0]["event_id"])
    assert any(row["status"] == "NO_BAD_CANDIDATE_WITHIN_BUDGET" for row in rows)
    assert any(
        row["status"] == "REGRESSION_DETECTED"
        and row["rollback_required"] == "True"
        for row in rows
    )
    print("research CSV HTTP PASS", len(rows))


if __name__ == "__main__":
    main()
    test_csv_http()
