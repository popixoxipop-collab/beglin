from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .dashboard import router as dashboard_router
from .evidence import evidence_root_readable
from .models import HealthView
from .research import router as research_router
from .stream import router as stream_router
from .telemetry import router as telemetry_router

ROOT = Path(__file__).resolve().parents[1]
UI_DIST = ROOT / "ui" / "dist"

app = FastAPI(
    title="beglin-experiment-dashboard",
    description="Read-only contract-first dashboard API for Beglin quantization experiments.",
    version="0.3.0",
)

app.include_router(dashboard_router)
app.include_router(stream_router)
app.include_router(telemetry_router)
app.include_router(research_router)
app.mount("/assets", StaticFiles(directory=UI_DIST / "assets", check_dir=False), name="ui-assets")


@app.get("/", include_in_schema=False)
def dashboard_ui():
    index = UI_DIST / "index.html"
    if not index.is_file():
        return JSONResponse(
            status_code=503,
            content={
                "status": "ui_not_built",
                "detail": "Run cd ui && npm run build before serving the integrated dashboard.",
            },
        )
    return FileResponse(index)


@app.get("/health", response_model=HealthView, operation_id="health")
def health() -> HealthView:
    return HealthView(evidence_root_readable=evidence_root_readable())
