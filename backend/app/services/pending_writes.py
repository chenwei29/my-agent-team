"""fs_write 的审批中转（review 模式）。

approve 由 store 落盘（除非 skip_write —— 那是后续 CLI adapter 自己写盘的路径，
AgentHub 自己的 fs_write 工具恒为 False）；写失败按拒绝收口：端点回 500，
工具侧看到的错误文案与用户拒绝一致（前端面板照常关闭）。
abort 清理走静默路径（不发 SSE）：前端随 run 收尾自行清面板。
"""

from __future__ import annotations

from app.schemas.events import FsWritePendingEvent, FsWriteResolvedEvent
from app.services.event_bus import event_bus
from app.services.fs_service import write_file_in_workspace
from app.services.pending_store import PendingStore
from app.utils.ids import new_pending_write_id
from app.utils.time import now_ms


class PendingWriteStore(PendingStore):
    def __init__(self) -> None:
        super().__init__(new_pending_write_id)
        # pending_id -> {workspace, skip_write}；与表项同生命周期
        self._extra: dict[str, dict] = {}

    def register(
        self,
        *,
        conversation_id: str,
        agent_id: str,
        run_id: str,
        path: str,
        absolute_path: str,
        old_content: str | None,
        new_content: str,
        workspace,
        skip_write: bool = False,
    ) -> dict:
        payload = super().register(
            {
                "conversationId": conversation_id,
                "agentId": agent_id,
                "runId": run_id,
                "path": path,
                "absolutePath": absolute_path,
                "oldContent": old_content,
                "newContent": new_content,
            }
        )
        self._extra[payload["id"]] = {"workspace": workspace, "skip_write": skip_write}
        event_bus.publish(
            FsWritePendingEvent(
                conversationId=conversation_id, timestamp=now_ms(), pendingWrite=payload  # type: ignore[arg-type]
            )
        )
        return payload

    def approve(self, pending_id: str) -> bool:
        """批准：由 store 落盘（skip_write 除外），再收口。落盘失败按拒绝收口并返回 False。"""
        entry = self._entries.get(pending_id)
        extra = self._extra.get(pending_id)
        if entry is None or extra is None:
            return False

        if not extra["skip_write"]:
            try:
                write_file_in_workspace(extra["workspace"], entry.payload["path"], entry.payload["newContent"])
            except Exception:
                self._finalize(pending_id, {"applied": False})
                self._extra.pop(pending_id, None)
                return False

        self._finalize(
            pending_id,
            {"applied": True},
            FsWriteResolvedEvent(
                conversationId=entry.payload["conversationId"],
                timestamp=now_ms(),
                pendingId=pending_id,
                applied=True,
            ),
        )
        self._extra.pop(pending_id, None)
        return True

    def reject(self, pending_id: str) -> bool:
        entry = self._entries.get(pending_id)
        if entry is None:
            return False
        self._finalize(
            pending_id,
            {"applied": False},
            FsWriteResolvedEvent(
                conversationId=entry.payload["conversationId"],
                timestamp=now_ms(),
                pendingId=pending_id,
                applied=False,
            ),
        )
        self._extra.pop(pending_id, None)
        return True

    # ---- abort 路径：静默（不发 SSE） ----

    def _cancel_result(self):
        return {"applied": False}

    def cancel(self, pending_id: str) -> bool:
        ok = super().cancel(pending_id)
        self._extra.pop(pending_id, None)
        return ok


pending_writes = PendingWriteStore()
