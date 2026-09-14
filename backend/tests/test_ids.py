"""ID 契约：前缀 + 12 位 base62。"""

from __future__ import annotations

import pytest

from app.utils import ids

# (函数, 期望前缀) —— 前缀是前端判断类型的依据，必须保持
CASES = [
    (ids.new_agent_id, "ag_"),
    (ids.new_conversation_id, "conv_"),
    (ids.new_message_id, "msg_"),
    (ids.new_artifact_id, "art_"),
    (ids.new_workspace_id, "ws_"),
    (ids.new_run_id, "run_"),
    (ids.new_tool_call_id, "call_"),
    (ids.new_attachment_id, "att_"),
    (ids.new_pending_write_id, "pwr_"),
    (ids.new_pending_bash_command_id, "pbc_"),
    (ids.new_pending_question_id, "pq_"),
    (ids.new_pending_dispatch_plan_id, "pdp_"),
    (ids.new_context_summary_id, "ctx_"),
    (ids.new_deployment_id, "dep_"),
]

_ALPHABET = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


@pytest.mark.parametrize(("factory", "prefix"), CASES)
def test_prefix_and_length(factory, prefix):
    value = factory()
    assert value.startswith(prefix)
    suffix = value[len(prefix) :]
    assert len(suffix) == 12
    assert set(suffix) <= _ALPHABET


def test_ids_are_unique():
    generated = {ids.new_message_id() for _ in range(500)}
    assert len(generated) == 500
