from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from .journal import ensure_a0_seeded, list_events

router = APIRouter(prefix="/api/v1/stream", tags=["stream"])


def encode_sse(row: dict) -> str:
    payload = {
        "seq": int(row["seq"]),
        "event_id": row["event_id"],
        "event_type": row["event_type"],
        "occurred_at": row["occurred_at"],
        "experiment_id": row["experiment_id"],
        "source_commit": row["source_commit"],
        "binary_sha256": row["binary_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "payload": row["payload"],
    }
    return (
        f"id: {row['seq']}\n"
        f"event: {row['event_type']}\n"
        f"data: {json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n\n"
    )


async def event_stream(after_seq: int):
    ensure_a0_seeded()
    cursor = int(after_seq)
    idle = 0
    while True:
        rows = list_events(after_seq=cursor, limit=100)
        if rows:
            idle = 0
            for row in rows:
                cursor = int(row["seq"])
                yield encode_sse(row)
        else:
            idle += 1
            if idle >= 15:
                yield ": heartbeat\n\n"
                idle = 0
            await asyncio.sleep(1)


@router.get("/events", operation_id="streamEvents")
async def stream_events(
    after_seq: int = Query(default=0, ge=0),
) -> StreamingResponse:
    return StreamingResponse(
        event_stream(after_seq),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
