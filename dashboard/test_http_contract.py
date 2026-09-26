from fastapi.testclient import TestClient
from app.main import app

client=TestClient(app)

paths=[
    "/health",
    "/api/v1/dashboard/summary",
    "/api/v1/dashboard/variables",
    "/api/v1/experiments",
    "/api/v1/policy",
    "/api/v1/canary",
    "/api/v1/production",
    "/api/v1/evidence/signed-canary",
]
out={}
for path in paths:
    r=client.get(path)
    assert r.status_code==200, (path,r.status_code,r.text)
    out[path]=r.json()

assert out["/api/v1/dashboard/summary"]["active_policy"][0]=={"role":"shared_up_proj","layer":3,"n":6}
assert out["/api/v1/canary"]["signed"] is True
assert out["/api/v1/production"]["state"]=="PERSISTENT_APPLY_VERIFIED"
assert out["/api/v1/production"]["rollback"]["ready"] is True
assert out["/api/v1/dashboard/variables"]["experimental"]["before_n"]==4
assert out["/api/v1/dashboard/variables"]["experimental"]["after_n"]==6
print("testclient HTTP contract PASS", len(paths), "endpoints")
