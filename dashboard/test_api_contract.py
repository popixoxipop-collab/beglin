from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

paths = [
    "/health",
    "/api/v1/dashboard/summary",
    "/api/v1/dashboard/variables",
    "/api/v1/experiments",
    "/api/v1/policy",
    "/api/v1/canary",
    "/api/v1/production",
    "/api/v1/evidence/signed-canary",
]

responses = {}
for path in paths:
    r = client.get(path)
    assert r.status_code == 200, (path, r.status_code, r.text)
    responses[path] = r.json()

summary = responses["/api/v1/dashboard/summary"]
assert summary["source_commit"] == "330954b27f146b8a17db2cb353c3e620968bad5e"
assert summary["active_policy"] == [{"role": "shared_up_proj", "layer": 3, "n": 6}]
assert summary["resident_worker"] is False

variables = responses["/api/v1/dashboard/variables"]
assert variables["experimental"]["before_n"] == 4
assert variables["experimental"]["after_n"] == 6
assert "model_id" in variables["unsupported_fields"]

canary = responses["/api/v1/canary"]
assert canary["signed"] is True
assert canary["signature_status"] == "VERIFIED"
assert canary["attribution_before"] == 1.0
assert canary["attribution_after"] == 0.0
assert canary["rollback_required"] is False

production = responses["/api/v1/production"]
assert production["state"] == "PERSISTENT_APPLY_VERIFIED"
assert production["active_policy"] == [{"role": "shared_up_proj", "layer": 3, "n": 6}]
assert len(production["restart_verification"]) == 2
assert all(x["corrected_hits"] == 12 for x in production["restart_verification"])
assert production["rollback"]["ready"] is True
assert production["rollback"]["exercised"] is False

evidence = responses["/api/v1/evidence/signed-canary"]
assert evidence["evidence"]["sha256"] == "e383c614a18dbc2539d9137c6d78323355711f0c919f0f7d4ef30044301985dc"

print("testclient dashboard API PASS")
