"""Mermaid 源码的静态规整与校验。

产物 diagram 类型的 content 在入库前必须过这里：
- 剥掉 ```mermaid 围栏、统一换行
- 校验首行是受支持的图声明（flowchart / sequenceDiagram / classDiagram ...）
- flowchart 标签自动加引号（中文 / 数学符号等会让 Mermaid 解析器炸掉）
- 逐行静态校验：围栏残留、style 语法、括号配平

只做「不需要真跑 Mermaid 就能发现」的检查；深度语法交给前端渲染器。
"""

from __future__ import annotations

import re
from typing import NamedTuple

_MERMAID_DECLARATION_RE = re.compile(
    r"^(?:flowchart|graph|sequenceDiagram|classDiagram|stateDiagram(?:-v2)?|erDiagram|gantt|pie"
    r"|journey|gitGraph|mindmap|timeline|quadrantChart|requirementDiagram|C4Context|C4Container"
    r"|C4Component|C4Dynamic|C4Deployment|architecture-beta|block-beta|packet-beta|sankey-beta"
    r"|xychart-beta)\b",
    re.IGNORECASE,
)

_FENCE_RE = re.compile(r"^```(?:mermaid|mmd)?[ \t]*\n(.*?)\n```$", re.DOTALL | re.IGNORECASE)

_IS_FLOWCHART_RE = re.compile(r"^(?:flowchart|graph)\b", re.IGNORECASE)

_SUBGRAPH_RE = re.compile(
    r"^(\s*subgraph\s+)([A-Za-z][A-Za-z0-9_-]*)(\[)([^\]\"']+)(\]\s*)$"
)

_NODE_LABEL_RE = re.compile(
    r"(^|[\s;&])([A-Za-z][A-Za-z0-9_-]*)(\[)([^\]\"'\n]+)(\])"
)

_STYLE_LINE_RE = re.compile(
    r"^style\s+[A-Za-z][A-Za-z0-9_-]*\s+[A-Za-z-]+:[^\s,]+(?:,[A-Za-z-]+:[^\s,]+)*$",
    re.IGNORECASE,
)

_IS_STYLE_LINE_RE = re.compile(r"^style\s+", re.IGNORECASE)


class MermaidOutcome(NamedTuple):
    """normalise_mermaid_source 的结果：ok=False 时 error 有值。"""

    ok: bool
    source: str | None = None
    error: str | None = None


def normalise_mermaid_source(raw_source: str) -> MermaidOutcome:
    source = _strip_fence(raw_source).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not source:
        return MermaidOutcome(False, error="Mermaid source is empty.")

    first_line = _first_significant_line(source)
    if first_line is None or not _MERMAID_DECLARATION_RE.match(first_line):
        return MermaidOutcome(
            False,
            error=(
                "Mermaid source must start with a supported diagram declaration such as "
                '"flowchart TD", "sequenceDiagram", or "classDiagram".'
            ),
        )

    normalised = _normalise_flowchart_labels(source) if _IS_FLOWCHART_RE.match(first_line) else source
    validation_error = _validate_static(normalised)
    if validation_error:
        return MermaidOutcome(False, error=validation_error)

    return MermaidOutcome(True, source=normalised)


def _strip_fence(source: str) -> str:
    match = _FENCE_RE.match(source.strip())
    return match.group(1) if match else source


def _first_significant_line(source: str) -> str | None:
    for line in source.split("\n"):
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("%%"):
            continue
        return trimmed
    return None


def _normalise_flowchart_labels(source: str) -> str:
    out: list[str] = []
    for line in source.split("\n"):
        subgraph = _SUBGRAPH_RE.match(line)
        if subgraph:
            prefix, node_id, _, label, trailing = subgraph.groups()
            out.append(f'{prefix}{node_id}["{_escape_label(label)}"]{trailing[1:]}')
            continue
        out.append(
            _NODE_LABEL_RE.sub(
                lambda m: f'{m.group(1)}{m.group(2)}{m.group(3)}"{_escape_label(m.group(4))}"{m.group(5)}',
                line,
            )
        )
    return "\n".join(out)


def _escape_label(label: str) -> str:
    return label.strip().replace("\\", "\\\\").replace('"', '\\"')


def _validate_static(source: str) -> str | None:
    for index, line in enumerate(source.split("\n")):
        line_number = index + 1
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("%%"):
            continue

        if trimmed.startswith("```"):
            return (
                f"Line {line_number}: remove Markdown code fences before saving a "
                "Mermaid diagram."
            )

        if _IS_STYLE_LINE_RE.match(trimmed) and not _STYLE_LINE_RE.match(trimmed):
            return (
                f'Line {line_number}: invalid style syntax. Use "style ID fill:#hex,color:#hex" '
                f"without trailing prose.\n{trimmed}"
            )

        balance_error = _validate_bracket_balance(trimmed)
        if balance_error:
            return f"Line {line_number}: {balance_error}\n{trimmed}"

    return None


def _validate_bracket_balance(line: str) -> str | None:
    stack: list[str] = []
    quote: str | None = None
    escaped = False

    for ch in line:
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue

        if ch in ('"', "'"):
            quote = ch
            continue
        if ch in "[({":
            stack.append(ch)
            continue
        if ch in "])}":
            if not stack or not _matches_bracket(stack.pop(), ch):
                return f'unmatched "{ch}"'

    if quote:
        return f"unclosed {quote} quote"
    if stack:
        return f'unclosed "{stack[-1]}"'
    return None


def _matches_bracket(opening: str, closing: str) -> bool:
    return (opening, closing) in {("[", "]"), ("(", ")"), ("{", "}")}
