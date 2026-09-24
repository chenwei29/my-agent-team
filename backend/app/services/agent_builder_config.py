"""Agent Builder 的静态配置：可选工具清单、工具中文元数据、工具预设与推断规则。

草稿生成（app/services/agent_draft.py）与前端共用同一套口径：
- AVAILABLE_AGENT_TOOLS 是普通自建 Agent 能授权的**全部**工具（不含 Orchestrator 专用工具）；
- 预设把工具打包成四种典型工作方式，按用户描述里的关键词推断；
- 工具权限摘要（label/desc）直接取 AGENT_TOOL_META，保存前逐项展示给用户确认。
"""

from __future__ import annotations

import re
from typing import Any

AVAILABLE_AGENT_TOOLS: tuple[str, ...] = (
    "write_artifact",
    "deploy_artifact",
    "deploy_workspace",
    "read_artifact",
    "read_attachment",
    "ask_user",
    "fs_list",
    "fs_read",
    "fs_write",
    "bash",
)

AGENT_TOOL_META: dict[str, dict[str, str]] = {
    "write_artifact": {
        "label": "创建产物",
        "desc": "生成可预览的代码 / 网页 / 文档 / PPT，支持多版本迭代",
    },
    "deploy_artifact": {
        "label": "部署网页",
        "desc": "把网页产物发布为本地静态站点，生成预览链接与下载包",
    },
    "deploy_workspace": {
        "label": "部署目录",
        "desc": "把工作区内 dist/build/out 等静态目录生成预览链接与下载包",
    },
    "read_artifact": {
        "label": "读取产物",
        "desc": "查看会话中已有产物的完整内容，便于在其基础上继续改",
    },
    "read_attachment": {
        "label": "读取附件",
        "desc": "读取用户上传的文本 / 文件附件内容",
    },
    "ask_user": {
        "label": "结构化提问",
        "desc": "让用户在明确选项中选择，用于范围、风格、平台等关键澄清",
    },
    "fs_list": {
        "label": "列出文件",
        "desc": "列出工作区内的目录和文件，用于安全探索项目结构",
    },
    "fs_read": {
        "label": "读取文件",
        "desc": "读取工作区内的文件（源码 / 配置等），仅限沙箱目录",
    },
    "fs_write": {
        "label": "写入文件",
        "desc": "在工作区内新建 / 修改文件；review 模式下需用户批准",
    },
    "bash": {
        "label": "执行命令",
        "desc": "在工作区内运行命令行；受命令黑名单与沙箱目录约束",
    },
}

AGENT_BUILDER_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "deepseek": {"label": "DeepSeek", "defaultModel": "deepseek-v4-flash"},
    "anthropic": {"label": "Anthropic", "defaultModel": "claude-opus-4-7"},
    "openai": {"label": "OpenAI", "defaultModel": "gpt-4o"},
    "volcano-ark": {"label": "火山方舟 (豆包)", "defaultModel": "doubao-seed-2-0-lite-260428"},
    "openai-compatible": {"label": "OpenAI-compatible", "defaultModel": ""},
}

AGENT_TOOL_PRESETS: list[dict[str, Any]] = [
    {
        "id": "all-purpose",
        "label": "全栈通用",
        "desc": "本地代码 + artifact 交付",
        "tools": list(AVAILABLE_AGENT_TOOLS),
    },
    {
        "id": "local-code",
        "label": "本地代码",
        "desc": "读写 workspace 并运行命令",
        "tools": [
            "deploy_workspace",
            "read_artifact",
            "read_attachment",
            "ask_user",
            "fs_list",
            "fs_read",
            "fs_write",
            "bash",
        ],
    },
    {
        "id": "artifact",
        "label": "产物交付",
        "desc": "网页、文档、原型卡片",
        "tools": [
            "write_artifact",
            "deploy_artifact",
            "deploy_workspace",
            "read_artifact",
            "read_attachment",
            "ask_user",
        ],
    },
    {
        "id": "review",
        "label": "审查验证",
        "desc": "读取产物/文件并跑检查",
        "tools": ["read_artifact", "read_attachment", "ask_user", "fs_list", "fs_read", "bash"],
    },
]


def normalize_agent_tool_names(tool_names: list[str] | tuple[str, ...]) -> list[str]:
    """过滤掉不在可选清单里的（如 Orchestrator 专用工具），并去重保序。"""
    allowed = set(AVAILABLE_AGENT_TOOLS)
    seen: set[str] = set()
    normalized: list[str] = []
    for tool_name in tool_names:
        if tool_name not in allowed or tool_name in seen:
            continue
        seen.add(tool_name)
        normalized.append(tool_name)
    return normalized


def get_agent_tool_preset(preset_id: str) -> dict[str, Any]:
    for preset in AGENT_TOOL_PRESETS:
        if preset["id"] == preset_id:
            return preset
    return AGENT_TOOL_PRESETS[0]


def build_tool_permission_summaries(tool_names: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {"toolName": tool_name, **AGENT_TOOL_META[tool_name]}
        for tool_name in normalize_agent_tool_names(tool_names)
    ]


def infer_agent_tool_preset(intent: str, follow_up: str | None = None) -> str:
    """按描述关键词推断工具预设 id：review > local-code > artifact > all-purpose。"""
    text = f"{intent}\n{follow_up or ''}".lower()
    wants_to_write = bool(
        re.search(r"写|实现|开发|生成|创建|搭建|部署|build|implement|create|write|ship", text)
        or re.search(r"修改(?!建议)", text)
    )
    wants_review = bool(
        re.search(r"审查|评审|检查|验证|验收|风险|review|audit|inspect|validate|verify", text)
    )
    if wants_review and not wants_to_write:
        return "review"

    if re.search(
        r"代码|源码|仓库|本地|文件|命令|终端|测试|修复|重构|调试|workspace|repo|repository|code|cli|bash|test|lint|debug|refactor",
        text,
    ):
        return "local-code"

    if re.search(
        r"产物|网页|页面|原型|文档|报告|幻灯片|演示|图示|图表|设计稿|ppt|slides|presentation|website|document|diagram|mermaid|prototype",
        text,
    ):
        return "artifact"

    return "all-purpose"
