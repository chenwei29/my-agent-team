"""ID 生成：前缀 + 12 位 base62（如 `ag_` / `conv_` / `msg_` / `run_`）。

前缀是前端判断类型的依据，长度固定 12，不要换成 uuid4。
"""

from __future__ import annotations

import secrets

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ID_LENGTH = 12


def _nano(n: int = _ID_LENGTH) -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(n))


def new_agent_id() -> str:
    return f"ag_{_nano()}"


def new_conversation_id() -> str:
    return f"conv_{_nano()}"


def new_message_id() -> str:
    return f"msg_{_nano()}"


def new_artifact_id() -> str:
    return f"art_{_nano()}"


def new_workspace_id() -> str:
    return f"ws_{_nano()}"


def new_run_id() -> str:
    return f"run_{_nano()}"


def new_tool_call_id() -> str:
    return f"call_{_nano()}"


def new_attachment_id() -> str:
    return f"att_{_nano()}"


def new_pending_write_id() -> str:
    return f"pwr_{_nano()}"


def new_pending_bash_command_id() -> str:
    return f"pbc_{_nano()}"


def new_pending_question_id() -> str:
    return f"pq_{_nano()}"


def new_pending_dispatch_plan_id() -> str:
    return f"pdp_{_nano()}"


def new_context_summary_id() -> str:
    return f"ctx_{_nano()}"


def new_deployment_id() -> str:
    return f"dep_{_nano()}"
