"""最小 AbortSignal 替身。

`adapter.stream(input, signal)` 把 signal 作为第二个位置参数传入，
adapter 在循环头 `if signal.aborted: break` 自行轮询。
语义是不抛异常、不 cancel task —— 已经进入的 `asyncio.sleep()` 会睡完，
中止在下一次循环头才被观察到（这是对外可见行为，别"优化"成即时中断）。

监听器（P4 引入）：挂起的审批 Future / bash 子进程需要在 abort 瞬间被回调
（杀进程、resolve Future），而不是等下一次轮询。listener 语义与轮询并存：
`abort()` 同步触发全部监听器；往已 abort 的 signal 上注册会立即回调。
"""

from __future__ import annotations

import asyncio


class AbortSignal:
    def __init__(self) -> None:
        self._aborted = False
        self._event = asyncio.Event()
        self._listeners: list = []

    @property
    def aborted(self) -> bool:
        return self._aborted

    def abort(self) -> None:
        self._aborted = True
        self._event.set()
        for listener in list(self._listeners):
            listener()

    def add_listener(self, listener) -> None:
        """注册 abort 回调；signal 已 abort 则立即同步回调一次。"""
        if self._aborted:
            listener()
            return
        self._listeners.append(listener)

    def remove_listener(self, listener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    async def wait(self) -> None:
        """给 SDK adapter 用（它们可以 await 一个 abort 事件）。"""
        await self._event.wait()
