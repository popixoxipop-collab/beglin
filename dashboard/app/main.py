from fastapi import FastAPI

from .dashboard import router as dashboard_router
from .evidence import evidence_root_readable
from .models import HealthView
from .stream import router as stream_router

app = FastAPI(
    title="beglin-experiment-dashboard",
    description="Read-only contract-first dashboard API for Beglin quantization experiments.",
    version="0.2.0",
)

app.include_router(dashboard_router)
app.include_router(stream_router)


@app.get("/health", response_model=HealthView, operation_id="health")
def health() -> HealthView:
    return HealthView(evidence_root_readable=evidence_root_readable())
