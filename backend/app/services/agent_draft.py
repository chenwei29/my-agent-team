"""Agent 草稿生成：从一句用户描述启发式拼出可保存的 Agent 配置草稿。

纯本地规则、不调 LLM —— 生成的草稿一律是**普通自建 Agent**（custom adapter、
无 Orchestrator 专用工具），保存仍走 POST /api/agents，草稿只负责把「起名 /
能力 / 工具预设 / system prompt / 假设与理由」一次填好，用户可在详细配置里改。
"""

from __future__ import annotations

import re
from typing import Any

from app.services.agent_builder_config import (
    AGENT_BUILDER_PROVIDER_DEFAULTS,
    build_tool_permission_summaries,
    get_agent_tool_preset,
    infer_agent_tool_preset,
)

DEFAULT_PROVIDER = "deepseek"


def create_agent_config_draft(intent: str, follow_up: str | None = None) -> dict[str, Any]:
    """生成草稿并过一遍输出契约校验（形状不对直接抛错，不放行到前端）。"""
    from app.schemas.entities import AgentConfigDraftOut

    draft = build_heuristic_agent_config_draft(intent, follow_up)
    return AgentConfigDraftOut.model_validate(draft).model_dump(by_alias=True)


def build_heuristic_agent_config_draft(intent: str, follow_up: str | None = None) -> dict[str, Any]:
    intent_text = _normalize_text(intent)
    follow_up_text = _normalize_text(follow_up or "")
    combined = "\n".join(t for t in (intent_text, follow_up_text) if t)
    preset_id = infer_agent_tool_preset(intent_text, follow_up_text)
    preset = get_agent_tool_preset(preset_id)
    name = _infer_agent_name(combined, preset_id)
    capabilities = _infer_capabilities(combined, preset_id)
    permission_summaries = build_tool_permission_summaries(preset["tools"])
    defaults = AGENT_BUILDER_PROVIDER_DEFAULTS[DEFAULT_PROVIDER]

    return {
        "name": name,
        "avatar": "🤖",
        "description": _infer_description(combined, preset_id),
        "capabilities": capabilities,
        "systemPrompt": _build_system_prompt(
            name=name,
            intent=intent_text,
            follow_up=follow_up_text,
            preset_label=preset["label"],
            permission_summaries=permission_summaries,
        ),
        "adapterName": "custom",
        "modelProvider": DEFAULT_PROVIDER,
        "modelId": defaults["defaultModel"],
        "toolNames": [s["toolName"] for s in permission_summaries],
        "supportsVision": True,
        "rationale": [
            f"根据描述匹配到「{preset['label']}」工具预设。",
            "按普通自建 Agent 生成，不包含 Orchestrator 专用工具。",
            "最终保存仍会走现有 Agent 创建接口，保存前可切到详细配置继续调整。",
        ],
        "assumptions": [
            {
                "label": "模型",
                "detail": (
                    f"默认使用 {defaults['label']} / {defaults['defaultModel']}，"
                    "可在详细配置中改成其他 provider。"
                ),
            },
            {
                "label": "视觉",
                "detail": (
                    "默认开启视觉能力，方便处理截图、设计稿、图示和图片附件；"
                    "如果模型不支持可在详细配置中关闭。"
                ),
            },
            {
                "label": "权限",
                "detail": (
                    f"工具权限来自「{preset['label']}」预设，保存前会逐项展示，可切到详细配置增减。"
                ),
            },
        ],
        "toolPermissionSummaries": permission_summaries,
    }


def _build_system_prompt(
    *,
    name: str,
    intent: str,
    follow_up: str,
    preset_label: str,
    permission_summaries: list[dict[str, str]],
) -> str:
    permission_line = "、".join(f"{s['label']}({s['toolName']})" for s in permission_summaries)
    lines = [
        f"你是 {name}。",
        "",
        f"用户创建你的目标：{intent}",
        f"补充偏好：{follow_up}" if follow_up else "",
        "",
        "工作方式：",
        "- 先判断用户真正想完成的交付物、约束和验收标准。",
        "- 信息不足时，优先使用结构化提问澄清关键选择；不要假装已经知道用户偏好。",
        "- 执行前简要说明计划，执行中保持结果可检查，交付前做自检。",
        "- 涉及文件写入、命令执行或部署时，明确说明影响范围和结果。",
        "",
        f"默认工具策略：{preset_label}。可用权限包括：{permission_line or 'SDK 内置工具集'}。",
        "不要尝试使用未授权工具；普通自建 Agent 不承担 Orchestrator 的任务拆分职责。",
    ]
    return "\n".join(line for line in lines if line)


_NAME_PATTERN = re.compile(
    r"(?:叫|命名为|名字叫|名称(?:是|为)?|name(?:d)?\s*)(?:「|“|\"|')?([^，,。.\n\"”』']{2,24})"
)


def _infer_agent_name(text: str, preset_id: str) -> str:
    explicit = _NAME_PATTERN.search(text)
    if explicit:
        return _truncate(_clean_name(explicit.group(1)), 64)

    lower = text.lower()
    if re.search(r"ppt|幻灯片|演示|presentation|slides", lower):
        return "PPT 设计师"
    if re.search(r"图示|图表|流程图|mermaid|diagram", lower):
        return "图示架构师"
    if re.search(r"文档|报告|document|report", lower):
        return "文档写作助手"
    if re.search(r"网页|页面|原型|website|prototype|landing", lower):
        return "网页原型助手"

    return {
        "local-code": "代码工程师",
        "artifact": "产物设计师",
        "review": "审查验证助手",
    }.get(preset_id, "专属助手")


def _infer_description(text: str, preset_id: str) -> str:
    target = _truncate(text, 72)
    prefix = {
        "local-code": "围绕本地代码与命令行任务提供实现、修改和验证支持",
        "artifact": "围绕网页、文档、PPT 等产物提供规划、生成和迭代支持",
        "review": "围绕已有产物或代码提供审查、验证和风险发现",
    }.get(preset_id, "围绕用户目标提供规划、执行和交付支持")
    return _truncate(f"{prefix}：{target}", 280)


def _infer_capabilities(text: str, preset_id: str) -> list[str]:
    lower = text.lower()
    capabilities = {
        "local-code": ["代码实现", "本地验证", "命令行"],
        "artifact": ["产物交付", "内容生成", "原型设计"],
        "review": ["审查验证", "风险发现", "改进建议"],
    }.get(preset_id, ["需求澄清", "任务执行", "交付自检"])

    if re.search(r"ppt|幻灯片|演示|presentation|slides", lower):
        capabilities.append("PPT")
    if re.search(r"图示|图表|流程图|mermaid|diagram", lower):
        capabilities.append("图示")
    if re.search(r"网页|页面|website|prototype|landing", lower):
        capabilities.append("网页")
    if re.search(r"图片|截图|视觉|image|screenshot|visual", lower):
        capabilities.append("视觉理解")

    # 去重保序 + 最多 8 项
    return list(dict.fromkeys(capabilities))[:8]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_name(text: str) -> str:
    return re.sub(r"[「」“”\"']", "", text).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"
