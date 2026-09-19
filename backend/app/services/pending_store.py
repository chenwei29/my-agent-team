"""审批中转的公共骨架：in-memory 注册表 + asyncio.Future 挂起。

三种 pending（fs_write / ask_user / bash 命令）共用同一套机制：
工具 handler 注册一项 → 发布 *.pending 事件 → 前端弹面板 → 用户操作打
HTTP 端点 → store 把结果 resolve 进 Future → handler 醒来继续 run。

约定：
- 挂起状态只在内存里（重启即丢，前端刷新后 GET 会拿到空列表，这是接受的语义）；
- ``_finalize`` 是 Future 的唯一写入方：先摘表项再 set_result（done 守卫），
  谁先到谁生效 —— abort 与 approve 的竞态靠这个收口；
- cancel（abort 路径）是否发布 *.resolved 事件，各 store 自己定（fs_write /
  ask_user 静默、bash 发布），语义见各自模块。
"""

from __future__ import annotations

from typing import Any, Callable

from app.schemas.events import StreamEvent
from app.services.event_bus import event_bus
from app.utils.time import now_ms


class _Entry:
    __slots__ = ("payload", "future")

    def __init__(self, payload: dict, future) -> None:
        self.payload = payload
        self.future = future


class PendingStore:
    """按 pending id 索引的挂起项注册表。子类补充事件构造与业务动作。"""

    def __init__(self, id_factory: Callable[[], str]) -> None:
        self._id_factory = id_factory
        self._entries: dict[str, _Entry] = {}

    # ---- 注册 / 查询 ----

    def register(self, payload_fields: dict) -> dict:
        """落表 + 发布 pending 事件，返回完整 payload（含 id / createdAt）。

        此时 resolver 还没挂上（handler 拿到返回值后才建 Future 再 attach）；
        若用户在 attach 之前就操作（极端竞态），attach 返回 False，调用方按已拒绝处理。
        """
        pending_id = self._id_factory()
        payload = {"id": pending_id, "createdAt": now_ms(), **payload_fields}
        self._entries[pending_id] = _Entry(payload, None)
        return payload

    def attach_resolver(self, pending_id: str, future) -> bool:
        entry = self._entries.get(pending_id)
        if entry is None:
            return False
        entry.future = future
        return True

    def get(self, pending_id: str) -> dict | None:
        entry = self._entries.get(pending_id)
        return dict(entry.payload) if entry else None

    def list_by_conversation(self, conversation_id: str) -> list[dict]:
        items = [
            dict(e.payload)
            for e in self._entries.values()
            if e.payload.get("conversationId") == conversation_id
        ]
        items.sort(key=lambda p: p["createdAt"])
        return items

    # ---- 收口 ----

    def _finalize(self, pending_id: str, result: Any, resolved_event: StreamEvent | None = None) -> bool:
        """摘表项 → （可选）发布 resolved 事件 → resolve Future。表项不存在返回 False。"""
        entry = self._entries.pop(pending_id, None)
        if entry is None:
            return False
        if resolved_event is not None:
            event_bus.publish(resolved_event)
        if entry.future is not None and not entry.future.done():
            entry.future.set_result(result)
        return True

    # ---- abort 路径 ----

    def _cancel_result(self) -> Any:
        raise NotImplementedError

    def _cancel_resolved_event(self, pending_id: str) -> StreamEvent | None:
        return None

    def cancel(self, pending_id: str) -> bool:
        return self._finalize(pending_id, self._cancel_result(), self._cancel_resolved_event(pending_id))

    def cancel_for_run(self, run_id: str) -> int:
        """run 被 abort / 失败时清掉它名下所有挂起项。返回清理数量。"""
        ids = [pid for pid, e in self._entries.items() if e.payload.get("runId") == run_id]
        for pending_id in ids:
            self.cancel(pending_id)
        return len(ids)
