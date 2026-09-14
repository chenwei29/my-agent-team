"""StreamEvent 契约测试：事件类型与字段必须与前端 web/src/shared/types.ts 的联合逐字对齐。

这些测试是「事件契约」的护栏：字段名写错、type 拼错、漏定义变体，
前端不会报错、只会静默不生效，所以只能靠这里拦住。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.events import (
    ConnectedEvent,
    HeartbeatEvent,
    MessageEndEvent,
    MessageRecord,
    PartDeltaEvent,
    PartStartEvent,
    RunEndEvent,
    RunStartEvent,
    StreamEvent,
    TextAppendDelta,
    TextPart,
    parse_event,
)

# 前端联合里的 28 个事件类型，逐字对齐，一个都不能少
FRONTEND_EVENT_TYPES = {
    "run.start",
    "run.end",
    "run.usage",
    "message.start",
    "message.end",
    "message.usage",
    "message.added",
    "message.removed",
    "part.start",
    "part.delta",
    "part.end",
    "tool.call",
    "tool.result",
    "artifact.create",
    "artifact.update",
    "deploy.status",
    "dispatch.plan.pending",
    "dispatch.plan.resolved",
    "dispatch.plan",
    "dispatch.start",
    "dispatch.end",
    "fs_write.pending",
    "fs_write.resolved",
    "bash_command.pending",
    "bash_command.resolved",
    "ask_user.pending",
    "ask_user.resolved",
    "heartbeat",
}


def _union_types() -> set[str]:
    """从判别联合里取出所有变体的 type 字面量。"""
    from typing import get_args

    from app.schemas import events as events_module

    union = get_args(get_args(StreamEvent)[0])
    types = set()
    for variant in union:
        type_field = variant.model_fields["type"]
        types.add(type_field.default)
    return types


def test_union_covers_every_frontend_event_type():
    assert _union_types() == FRONTEND_EVENT_TYPES
    assert len(FRONTEND_EVENT_TYPES) == 28


def test_run_start_round_trips_in_camel_case():
    event = RunStartEvent(
        conversationId="conv_1",
        timestamp=1_700_000_000_000,
        runId="run_1",
        agentId="ag_1",
        triggerMessageId="msg_1",
    )
    dumped = event.model_dump()
    assert dumped["type"] == "run.start"
    assert dumped["conversationId"] == "conv_1"
    assert dumped["triggerMessageId"] == "msg_1"
    assert dumped["parentRunId"] is None


def test_run_end_status_is_restricted_to_frontend_union():
    for status in ("complete", "failed", "aborted"):
        RunEndEvent(conversationId="c", timestamp=1, runId="r", status=status)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RunEndEvent(conversationId="c", timestamp=1, runId="r", status="error")  # type: ignore[arg-type]


def test_part_events_discriminate_by_type():
    started = parse_event(
        {
            "type": "part.start",
            "conversationId": "conv_1",
            "timestamp": 1,
            "messageId": "msg_1",
            "partIndex": 0,
            "part": {"type": "text", "content": ""},
        }
    )
    assert isinstance(started, PartStartEvent)
    assert isinstance(started.part, TextPart)

    delta = parse_event(
        {
            "type": "part.delta",
            "conversationId": "conv_1",
            "timestamp": 1,
            "messageId": "msg_1",
            "partIndex": 0,
            "delta": {"type": "text.append", "text": "你"},
        }
    )
    assert isinstance(delta, PartDeltaEvent)
    assert isinstance(delta.delta, TextAppendDelta)
    assert delta.delta.text == "你"


def test_message_record_fields_match_frontend_message_row():
    record = MessageRecord(
        id="msg_1",
        conversationId="conv_1",
        role="agent",
        agentId=None,
        parts=[TextPart(content="hi")],
        status="streaming",
        parentMessageId=None,
        mentionedAgentIds=[],
        runId="run_1",
        usage=None,
        createdAt=123,
    )
    assert set(record.model_dump()) == {
        "id",
        "conversationId",
        "role",
        "agentId",
        "parts",
        "status",
        "parentMessageId",
        "mentionedAgentIds",
        "runId",
        "usage",
        "createdAt",
    }


def test_unknown_event_type_is_rejected():
    with pytest.raises(ValidationError):
        parse_event({"type": "run.started", "conversationId": "c", "timestamp": 1})


def test_extra_fields_are_rejected_so_typos_surface_early():
    with pytest.raises(ValidationError):
        MessageEndEvent(conversationId="c", timestamp=1, messageId="m", messagId="typo")  # type: ignore[call-arg]


def test_connected_and_heartbeat_are_not_in_the_union_but_carry_timestamp():
    assert "connected" not in FRONTEND_EVENT_TYPES  # 握手帧不是 StreamEvent（就是一个裸对象）
    connected = ConnectedEvent(timestamp=1)
    assert connected.model_dump() == {"type": "connected", "timestamp": 1}

    heartbeat = HeartbeatEvent(timestamp=2)
    assert heartbeat.model_dump() == {"type": "heartbeat", "timestamp": 2, "conversationId": None}
