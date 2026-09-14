"""进程内事件总线：单进程扇出，**没有任何过滤 / 缓冲 / 重放** —— 每个订阅者收到每条事件，
由前端按 `conversationId` 自己分发。实现是「每个订阅者一个 asyncio.Queue」。

背压：`publish` 是同步函数 + `put_nowait`，队列满了就**丢事件并记日志**（P2 明确要求的背压保护），
绝不让一个卡住的浏览器把整个后端拖死。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from app.schemas.events import StreamEvent

logger = logging.getLogger(__name__)

# 单个订阅者最多积压多少条事件：正常打字机速率远低于此，超过说明这个消费者已经废了
MAX_QUEUE_SIZE = 512


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[StreamEvent]] = set()
        self._dropped = 0

    def publish(self, event: StreamEvent) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self._dropped += 1
                logger.warning(
                    "event bus: subscriber queue full, dropped %s (total dropped=%d)",
                    event.type,
                    self._dropped,
                )

    async def subscribe(self) -> AsyncIterator[StreamEvent]:
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# 单进程单例：模块级变量就够，进程内所有订阅者共享同一个总线
event_bus = EventBus()
