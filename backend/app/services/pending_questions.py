"""ask_user 的中转：问题挂起 → 用户作答 → 答案 resolve 回工具。

cancel（abort 路径）静默、以 None 唤醒 —— 工具侧把 None 翻译成
"User did not answer the question (aborted)"。
"""

from __future__ import annotations

from app.schemas.events import AskUserPendingEvent, AskUserResolvedEvent
from app.services.event_bus import event_bus
from app.services.pending_store import PendingStore
from app.utils.time import now_ms
from app.utils.ids import new_pending_question_id


class PendingQuestionStore(PendingStore):
    def __init__(self) -> None:
        super().__init__(new_pending_question_id)

    def register(self, *, conversation_id: str, agent_id: str, run_id: str, questions: list[dict]) -> dict:
        payload = super().register(
            {
                "conversationId": conversation_id,
                "agentId": agent_id,
                "runId": run_id,
                "questions": questions,
            }
        )
        event_bus.publish(
            AskUserPendingEvent(
                conversationId=conversation_id, timestamp=now_ms(), pendingQuestion=payload  # type: ignore[arg-type]
            )
        )
        return payload

    def answer(self, pending_id: str, answers: dict) -> bool:
        """用户提交答案。answers: Record<questionText, {selectedLabels, freeformNote}>。"""
        entry = self._entries.get(pending_id)
        if entry is None:
            return False
        return self._finalize(
            pending_id,
            answers,
            AskUserResolvedEvent(
                conversationId=entry.payload["conversationId"], timestamp=now_ms(), pendingId=pending_id, answered=True
            ),
        )

    # ---- abort 路径：静默、None 唤醒 ----

    def _cancel_result(self):
        return None


pending_questions = PendingQuestionStore()
