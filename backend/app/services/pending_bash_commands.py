"""bash 高危命令的审批中转。

与另外两个 store 的差异：abort 清理**会**发布 bash_command.resolved{approved:false}
（前端面板需要这条事件才能关掉对应卡片）。
"""

from __future__ import annotations

from app.schemas.events import BashCommandPendingEvent, BashCommandResolvedEvent
from app.services.event_bus import event_bus
from app.services.pending_store import PendingStore
from app.utils.time import now_ms
from app.utils.ids import new_pending_bash_command_id


class PendingBashCommandStore(PendingStore):
    def __init__(self) -> None:
        super().__init__(new_pending_bash_command_id)

    def register(
        self, *, conversation_id: str, agent_id: str, run_id: str, command: str, cwd: str, reason: str
    ) -> dict:
        payload = super().register(
            {
                "conversationId": conversation_id,
                "agentId": agent_id,
                "runId": run_id,
                "command": command,
                "cwd": cwd,
                "reason": reason,
            }
        )
        event_bus.publish(
            BashCommandPendingEvent(
                conversationId=conversation_id, timestamp=now_ms(), pendingCommand=payload  # type: ignore[arg-type]
            )
        )
        return payload

    def approve(self, pending_id: str) -> bool:
        return self._resolve(pending_id, True)

    def reject(self, pending_id: str) -> bool:
        return self._resolve(pending_id, False)

    def _resolve(self, pending_id: str, approved: bool) -> bool:
        entry = self._entries.get(pending_id)
        if entry is None:
            return False
        return self._finalize(
            pending_id,
            {"approved": approved},
            BashCommandResolvedEvent(
                conversationId=entry.payload["conversationId"], timestamp=now_ms(), pendingId=pending_id, approved=approved
            ),
        )

    # ---- abort 路径：发布 resolved（与其他 store 不同，见模块注释） ----

    def _cancel_result(self):
        return {"approved": False}

    def _cancel_resolved_event(self, pending_id: str):
        entry = self._entries.get(pending_id)
        if entry is None:
            return None
        return BashCommandResolvedEvent(
            conversationId=entry.payload["conversationId"], timestamp=now_ms(), pendingId=pending_id, approved=False
        )


pending_bash_commands = PendingBashCommandStore()
