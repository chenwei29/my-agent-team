"""SSE 端点测试：帧格式与事件投递行为。

这里守的是「前端能不能收到」这件事：只要多写一个 `event:` 字段名，
`EventSource.onmessage` 就收不到任何事件，而前端不会报错（静默失效）。
"""

from __future__ import annotations

import asyncio
import json

from httpx import AsyncClient

from app.api import stream as stream_module
from app.api.stream import _event_stream, _sse
from app.schemas.events import RunEndEvent
from app.services.event_bus import event_bus


class _FakeRequest:
    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


def test_sse_frame_is_anonymous_data_line_with_blank_line():
    frame = _sse({"type": "heartbeat", "timestamp": 1})
    assert frame == 'data: {"type": "heartbeat", "timestamp": 1}\n\n'
    assert "event:" not in frame


def test_sse_keeps_chinese_readable_and_escapes_newlines():
    frame = _sse({"type": "part.delta", "text": "第一行\n第二行"})
    assert "第一行" in frame  # ensure_ascii=False
    assert "\\n" in frame  # 裸换行必须被转义，否则 SSE 会把它当成帧结束
    assert frame.count("\n\n") == 1
    json.loads(frame[len("data: ") : -2])  # 去掉前缀与空行后是可解析 JSON


async def test_first_frame_is_connected_then_published_events(monkeypatch):
    monkeypatch.setattr(stream_module, "HEARTBEAT_SECONDS", 5)
    request = _FakeRequest()
    generator = _event_stream(request)  # type: ignore[arg-type]

    first = await anext(generator)
    payload = json.loads(first[len("data: ") :].strip())
    assert payload["type"] == "connected"
    assert isinstance(payload["timestamp"], int) and payload["timestamp"] > 10**12

    # 生成器先发 connected 再订阅，所以要先把生成器推进到订阅状态，事件才不会丢
    next_frame = asyncio.create_task(anext(generator))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    event_bus.publish(
        RunEndEvent(conversationId="conv_1", timestamp=123, runId="run_1", status="complete")
    )
    second = json.loads((await next_frame)[len("data: ") :].strip())
    assert second["type"] == "run.end"
    assert second["conversationId"] == "conv_1"
    assert second["status"] == "complete"

    await generator.aclose()


async def test_heartbeat_is_sent_when_idle(monkeypatch):
    monkeypatch.setattr(stream_module, "HEARTBEAT_SECONDS", 0.05)
    generator = _event_stream(_FakeRequest())  # type: ignore[arg-type]
    await anext(generator)  # connected

    frames = json.loads((await asyncio.wait_for(anext(generator), timeout=1))[len("data: ") :].strip())
    assert frames["type"] == "heartbeat"
    assert isinstance(frames["timestamp"], int)

    await generator.aclose()


async def test_stream_headers_and_status(client: AsyncClient):
    async with client.stream("GET", "/api/stream") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache, no-transform"
        assert response.headers["x-accel-buffering"] == "no"
