"""ask_user 问答端点。

POST body: {answers: Record<questionText, {selectedLabels, freeformNote?}>}。
与 fs_write 审批一样，不做 conversationId 从属校验（未知 id 一律 404）。
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import ValidationError

from app.errors import HttpError, InvalidBody
from app.schemas.entities import AnswerQuestionsBody
from app.services.pending_questions import pending_questions

router = APIRouter(prefix="/api/conversations/{conversation_id}/pending-questions", tags=["pending-questions"])


@router.get("")
async def list_pending_questions(conversation_id: str) -> dict:
    return {"pendingQuestions": pending_questions.list_by_conversation(conversation_id)}


@router.post("/{pending_id}")
async def submit_answers(conversation_id: str, pending_id: str, body: dict) -> dict:
    try:
        parsed = AnswerQuestionsBody.model_validate(body)
    except ValidationError as err:
        issues = [{"path": list(e.get("loc", [])), "message": e.get("msg", "")} for e in err.errors()]
        raise InvalidBody(issues) from err

    if pending_questions.get(pending_id) is None:
        raise HttpError(404, "Pending question not found")

    answers = {q: a.model_dump(by_alias=True) for q, a in parsed.answers.items()}
    if not pending_questions.answer(pending_id, answers):
        raise HttpError(500, "Failed to record answer")
    return {"ok": True}
