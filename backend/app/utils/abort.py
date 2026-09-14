"""最小 AbortSignal 替身。

`adapter.stream(input, signal)` 把 signal 作为第二个位置参数传入，
adapter 在循环头 `if signal.aborted: break` 自行轮询。
语义是不抛异常、不 cancel task —— 已经进入的 `asyncio.sleep()` 会睡完，
中止在下一次循环头才被观察到（这是对外可见行为，别"优化"成即时中断）。
"""

from __future__ import annotations

import asyncio


class AbortSignal:
    def __init__(self) -> None:
        self._aborted = False
        self._event = asyncio.Event()

    @property
    def aborted(self) -> bool:
        return self._aborted

    def abort(self) -> None:
        self._aborted = True
        self._event.set()

    async def wait(self) -> None:
        """给 P3 起的 SDK adapter 用（它们可以 await 一个 abort 事件）。"""
        await self._event.wait()
