"""ask_user 工具：向用户提结构化问题（1–4 问 × 2–4 选项），挂起等回答。

答案回灌给 LLM 的格式（每问一行字符串）：
- 选中项用 ", " 连接；有备注时追加 " ; note: <备注>"
- 用户没答该问 → "(no answer)"；一个都没选且无备注 → "(empty)"
abort → None 唤醒 → 错误 "User did not answer the question (aborted)"。
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services.pending_questions import pending_questions
from app.tools.types import ToolContext, ToolDef, ToolResult


class _OptionArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str = Field(min_length=1)
    description: str = ""
    preview: str | None = None


class _QuestionArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1)
    header: str = Field(min_length=1, max_length=40)
    multiSelect: bool = False
    options: list[_OptionArgs] = Field(min_length=2, max_length=4)


class AskUserArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    questions: list[_QuestionArgs] = Field(min_length=1, max_length=4)


def _format_answer(answer) -> str:
    if answer is None:
        return "(no answer)"
    parts: list[str] = []
    selected = ", ".join(answer.get("selectedLabels") or [])
    if selected:
        parts.append(selected)
    note = (answer.get("freeformNote") or "").strip()
    if note:
        parts.append(f"note: {note}")
    return " ; ".join(parts) if parts else "(empty)"


async def _handle(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        parsed = AskUserArgs.model_validate(args or {})
    except ValidationError as err:
        return ToolResult(ok=False, error=f"Invalid args: {err}")

    questions = [
        {
            "question": q.question,
            "header": q.header,
            "multiSelect": q.multiSelect,
            "options": [
                {"label": o.label, "description": o.description, "preview": o.preview}
                if o.preview is not None
                else {"label": o.label, "description": o.description}
                for o in q.options
            ],
        }
        for q in parsed.questions
    ]

    pending = pending_questions.register(
        conversation_id=ctx.conversation_id,
        agent_id=ctx.agent_id,
        run_id=ctx.run_id,
        questions=questions,
    )

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    if not pending_questions.attach_resolver(pending["id"], future):
        return ToolResult(ok=False, error="User did not answer the question (aborted)")

    def _on_abort() -> None:
        pending_questions.cancel(pending["id"])

    if ctx.abort_signal is not None and ctx.abort_signal.aborted:
        _on_abort()
    elif ctx.abort_signal is not None:
        ctx.abort_signal.add_listener(_on_abort)
    try:
        answers = await future
    finally:
        if ctx.abort_signal is not None:
            ctx.abort_signal.remove_listener(_on_abort)

    if answers is None:
        return ToolResult(ok=False, error="User did not answer the question (aborted)")

    formatted = {q["question"]: _format_answer(answers.get(q["question"])) for q in questions}
    return ToolResult(ok=True, value={"answers": formatted})


ASK_USER_TOOL = ToolDef(
    name="ask_user",
    description=(
        "Ask the user 1-4 structured questions with 2-4 options each. "
        "Use this when you need clarification or a decision before proceeding. "
        "The call blocks until the user answers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "header": {"type": "string", "maxLength": 40},
                        "multiSelect": {"type": "boolean"},
                        "options": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "description": {"type": "string"},
                                    "preview": {"type": "string"},
                                },
                                "required": ["label"],
                            },
                        },
                    },
                    "required": ["question", "header", "options"],
                },
            },
        },
        "required": ["questions"],
    },
    handler=_handle,
)
