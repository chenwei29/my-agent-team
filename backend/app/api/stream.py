"""`GET /api/stream` —— 全局 SSE 端点。

协议细节（前端 `stream-provider.tsx` 只认这一种形状，改了就静默收不到）：
- 每条消息只有**匿名 `data:` 行 + 空行**：`data: {json}\\n\\n`
  —— **不能**写 `event: xxx`，否则 `EventSource.onmessage` 收不到（命名事件走 addEventListener）。
- 建连后立刻推一帧 `{"type":"connected","timestamp":...}`。
- 每 15s 推一帧 `{"type":"heartbeat","timestamp":...}` 防中间代理空闲断连。
- 服务端**不按会话过滤**：所有会话共一条流，前端按 `conversationId` 自己分桶。

字段为 null 时这里照常输出（如 `"parentRunId": null`）—— 前端对 undefined/null 的处理
完全一致（都是 `?? null`），所以显式 null 更可预测。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.schemas.events import ConnectedEvent, HeartbeatEvent, StreamEvent
from app.services.event_bus import event_bus
from app.utils.time import now_ms

router = APIRouter(prefix="/api", tags=["stream"])

HEARTBEAT_SECONDS = 15


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _event_stream(request: Request) -> AsyncIterator[str]:
    yield _sse(ConnectedEvent(timestamp=now_ms()).model_dump())

    subscription = event_bus.subscribe()
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event: StreamEvent = await asyncio.wait_for(anext(subscription), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                yield _sse(HeartbeatEvent(timestamp=now_ms()).model_dump())
                continue
            except StopAsyncIteration:
                break
            yield _sse(event.model_dump())
    finally:
        await subscription.aclose()


@router.get("/stream")
async def stream(request: Request) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(request),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
