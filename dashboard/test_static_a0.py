from app.evidence import (
    build_canary,
    build_experiments,
    build_policy,
    build_production,
    build_summary,
    build_variables,
)


def main() -> None:
    summary = build_summary()
    assert summary.source_commit == "330954b27f146b8a17db2cb353c3e620968bad5e"
    assert summary.resident_worker is False
    assert summary.active_policy[0].role == "shared_up_proj"
    assert summary.active_policy[0].layer == 3
    assert summary.active_policy[0].n == 6

    variables = build_variables()
    assert variables.controlled.checkpoint_sha256.startswith("1ea6be7e")
    assert variables.experimental.before_n == 4
    assert variables.experimental.after_n == 6
    assert "model_id" in variables.unsupported_fields

    experiments = build_experiments()
    assert len(experiments) == 2
    assert experiments[0].status == "CANARY_PASS"
    assert experiments[1].status == "PERSISTENT_APPLY_VERIFIED"

    canary = build_canary()
    assert canary.signed is True
    assert canary.attribution_before == 1.0
    assert canary.attribution_after == 0.0
    assert canary.rollback_required is False

    production = build_production()
    assert production.active_policy[0].n == 6
    assert len(production.restart_verification) == 2
    assert all(row.corrected_hits == 12 for row in production.restart_verification)
    assert production.rollback.ready is True
    assert production.rollback.exercised is False

    policy = build_policy()
    assert policy.persistent is True
    assert policy.resident_worker is False

    print("static A0 adapter PASS")


if __name__ == "__main__":
    main()
