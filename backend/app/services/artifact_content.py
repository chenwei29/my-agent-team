"""产物内容规整：把 agent / 用户面板给的松散 content 收成强类型 JSON。

build_artifact_content(type, raw) 是产物内容校验的单一来源：
- write_artifact 工具（agent 路径）
- 产物版本端点（用户面板路径）
共用本函数。非法输入返回 None；具体错误文案只有 diagram 提供
（describe_artifact_content_error），其余类型靠前端各自兜底。

模型常见毛病：把整个 content 对象 JSON.stringify 成字符串再传——
unwrap_stringified_content 会保守解包一层（仅当字符串带包装签名时）。
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from app.utils.mermaid_normalize import normalise_mermaid_source
from app.utils.ppt_normalize import normalize_blocks, strict_ppt_layout

MAX_DIAGRAM_SOURCE_CHARS = 50_000

# 已知 content 包装对象的键：决定一个解析结果是否「像被字符串化的 content」
_CONTENT_WRAPPER_KEYS = (
    "format",
    "content",
    "markdown",
    "text",
    "files",
    "entry",
    "html",
    "css",
    "js",
    "code",
    "url",
    "source",
    "mermaid",
    "targetArtifactId",
    "targetId",
    "hunks",
    "diff",
    "patch",
    "workspacePath",
    "path",
    "language",
    "sizeBytes",
    "checksum",
    "slides",
    "blocks",
    "subtitle",
)

_WRAPPER_SIGNATURE_RE = re.compile(
    r'"(?:format|content|markdown|text|files|entry|html|source|mermaid|targetArtifactId'
    r'|targetId|hunks|diff|patch|workspacePath|path|slides|blocks|subtitle)"\s*:'
)

_DATA_URI_RE = re.compile(r"^data:[^,]+;base64,", re.IGNORECASE)

_UNIFIED_DIFF_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
_METADATA_LINE_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@")

# 合法 JSON 转义字符（fix_invalid_json_escapes 之外的都补成 \\X）
_VALID_JSON_ESCAPE_CHARS = '"\\/bfnrtu'

_LANGUAGE_ALIASES = {
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "md": "markdown",
    "markdown": "markdown",
    "html": "html",
    "css": "css",
    "json": "json",
    "yml": "yaml",
    "yaml": "yaml",
}

_MERMAID_THEMES = ("default", "base", "dark", "forest", "neutral")


def build_artifact_content(artifact_type: str, raw_input: Any) -> dict[str, Any] | None:
    raw = _unwrap_stringified_content(raw_input)

    if artifact_type == "web_app":
        return _build_web_app_content(raw)

    if artifact_type == "document":
        return _build_document_content(raw)

    if artifact_type == "diagram":
        return _build_diagram_content(raw)

    if artifact_type == "image":
        return _build_image_content(raw)

    if artifact_type == "diff":
        return _build_diff_content(raw)

    if artifact_type == "code_file":
        return _build_code_file_content(raw)

    if artifact_type == "ppt":
        return _build_ppt_content(raw)

    return None


def describe_artifact_content_error(artifact_type: str, raw_input: Any) -> str | None:
    """非法 content 的用户可读错误文案；目前只有 diagram 提供细节。"""
    if artifact_type != "diagram":
        return None
    raw = _unwrap_stringified_content(raw_input)
    if isinstance(raw, dict):
        source = _coalesce(
            _read_string(raw.get("source")),
            _read_string(raw.get("mermaid")),
            _read_string(raw.get("code")),
            _read_string(raw.get("content")),
        )
    elif isinstance(raw, str):
        source = raw
    else:
        source = None
    if not source:
        return "Invalid diagram content: missing Mermaid source."
    if len(source.strip()) > MAX_DIAGRAM_SOURCE_CHARS:
        return (
            f"Invalid diagram content: Mermaid source exceeds "
            f"{MAX_DIAGRAM_SOURCE_CHARS} characters."
        )
    result = normalise_mermaid_source(source)
    if not result.ok:
        return f"Invalid Mermaid diagram: {result.error}"
    return None


# ─── web_app ────────────────────────────────────────────────


def _build_web_app_content(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        files = raw.get("files")
        # 形态 1：标准 { files, entry }
        if isinstance(files, dict):
            normalised = {k: v for k, v in files.items() if isinstance(v, str)}
            if not normalised:
                return None
            return {
                "type": "web_app",
                "files": normalised,
                "entry": raw["entry"] if isinstance(raw.get("entry"), str) else "index.html",
            }

        # 形态 2：扁平 { html, css, js }
        if (
            isinstance(raw.get("html"), str)
            or isinstance(raw.get("css"), str)
            or isinstance(raw.get("js"), str)
        ):
            web_files: dict[str, str] = {}
            if isinstance(raw.get("html"), str):
                web_files["index.html"] = raw["html"]
            if isinstance(raw.get("css"), str):
                web_files["style.css"] = raw["css"]
            if isinstance(raw.get("js"), str):
                web_files["script.js"] = raw["js"]
            return {"type": "web_app", "files": web_files, "entry": "index.html"}

        # 形态 3：{ content: '<html>...' } / { code: '...' }
        if isinstance(raw.get("content"), str):
            return {
                "type": "web_app",
                "files": {"index.html": raw["content"]},
                "entry": "index.html",
            }
        if isinstance(raw.get("code"), str):
            return {"type": "web_app", "files": {"index.html": raw["code"]}, "entry": "index.html"}

    # 形态 4：直接传 HTML 字符串
    if isinstance(raw, str):
        return {"type": "web_app", "files": {"index.html": raw}, "entry": "index.html"}

    return None


def _build_document_content(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        for key in ("content", "markdown", "text"):
            if isinstance(raw.get(key), str):
                return {"type": "document", "format": "markdown", "content": raw[key]}
    if isinstance(raw, str):
        return {"type": "document", "format": "markdown", "content": raw}
    return None


def _build_diagram_content(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        syntax = _coalesce(
            _read_string(raw.get("syntax")), _read_string(raw.get("format")), "mermaid"
        )
        if syntax.lower() != "mermaid":
            return None
        source = _coalesce(
            _read_string(raw.get("source")),
            _read_string(raw.get("mermaid")),
            _read_string(raw.get("code")),
            _read_string(raw.get("content")),
        )
        if not source:
            return None
        normalised = _normalise_diagram_source(source)
        if normalised is None:
            return None
        content: dict[str, Any] = {"type": "diagram", "syntax": "mermaid", "source": normalised}
        theme = _normalise_mermaid_theme(raw.get("theme"))
        if theme:
            content["theme"] = theme
        return content

    if isinstance(raw, str):
        normalised = _normalise_diagram_source(raw)
        if normalised is None:
            return None
        return {"type": "diagram", "syntax": "mermaid", "source": normalised}

    return None


def _build_image_content(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict) and isinstance(raw.get("url"), str):
        return {
            "type": "image",
            "url": raw["url"],
            "alt": raw["alt"] if isinstance(raw.get("alt"), str) else "",
        }
    if isinstance(raw, str):
        return {"type": "image", "url": raw, "alt": ""}
    return None


def _build_diff_content(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    target_artifact_id = _coalesce(
        _read_string(raw.get("targetArtifactId")), _read_string(raw.get("targetId"))
    )
    if not target_artifact_id:
        return None

    hunks = (
        _normalise_hunks(raw["hunks"])
        if isinstance(raw.get("hunks"), list)
        else _parse_unified_diff(_coalesce(_read_string(raw.get("diff")), _read_string(raw.get("patch"))))
    )
    if not hunks:
        return None
    return {
        "type": "diff",
        "targetArtifactId": target_artifact_id,
        "hunks": hunks,
        "applied": raw["applied"] if isinstance(raw.get("applied"), bool) else False,
    }


def _build_code_file_content(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    workspace_path = _coalesce(
        _read_string(raw.get("workspacePath")), _read_string(raw.get("path"))
    )
    if not workspace_path:
        return None
    return {
        "type": "code_file",
        "workspacePath": workspace_path,
        "language": _coalesce(_read_string(raw.get("language")), _guess_language(workspace_path)),
        "sizeBytes": _coalesce(_read_non_negative_number(raw.get("sizeBytes")), 0),
        "checksum": _coalesce(_read_string(raw.get("checksum")), ""),
    }


def _build_ppt_content(raw: Any) -> dict[str, Any] | None:
    obj = raw if isinstance(raw, dict) else None
    if _contains_unbounded_binary_payload(raw):
        return None
    if isinstance(raw, list):
        raw_slides: Any = raw
    elif obj is not None and isinstance(obj.get("slides"), list):
        raw_slides = obj["slides"]
    else:
        raw_slides = None
    if raw_slides is None:
        return None

    slides: list[dict[str, Any]] = []
    for item in raw_slides:
        if not isinstance(item, dict):
            continue
        title = _read_string(item.get("title"))
        subtitle = _read_string(item.get("subtitle"))
        bullets: list[str] | None = None
        if isinstance(item.get("bullets"), list):
            bullets = [b for b in item["bullets"] if isinstance(b, str)]
        elif isinstance(item.get("bullets"), str):
            bullets = [x.strip() for x in item["bullets"].split("\n") if x.strip()]
        elif isinstance(item.get("points"), list):
            bullets = [b for b in item["points"] if isinstance(b, str)]
        blocks = normalize_blocks(item.get("blocks"))
        notes = _read_string(item.get("notes"))
        layout = strict_ppt_layout(item.get("layout"))
        if (
            not title
            and not subtitle
            and not bullets
            and not blocks
            and layout != "blank"
        ):
            continue
        slide: dict[str, Any] = {}
        if title:
            slide["title"] = title
        if subtitle:
            slide["subtitle"] = subtitle
        if bullets:
            slide["bullets"] = bullets
        if blocks:
            slide["blocks"] = blocks
        if notes:
            slide["notes"] = notes
        if layout:
            slide["layout"] = layout
        slides.append(slide)

    if not slides:
        return None

    content: dict[str, Any] = {"type": "ppt"}
    deck_title = _read_string(obj.get("title")) if obj is not None else None
    if deck_title:
        content["title"] = deck_title
    theme = _normalise_ppt_theme(obj.get("theme")) if obj is not None else None
    if theme:
        content["theme"] = theme
    content["slides"] = slides
    return content


# ─── 字符串化 content 的解包 ─────────────────────────────────


def _unwrap_stringified_content(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    trimmed = raw.strip()
    if not trimmed.startswith("{"):
        return raw

    # 1) 合法 JSON 包装 → 直接解开
    try:
        parsed = json.loads(trimmed)
        if _is_wrapper_object(parsed):
            return parsed
    except ValueError:
        pass

    # 2) 非法 JSON 但带包装签名 → 容错救回（修转义 + 配平截取）
    if _WRAPPER_SIGNATURE_RE.search(trimmed):
        fixed = _fix_invalid_json_escapes(trimmed)
        candidate = _first_balanced_object(fixed) or _first_balanced_object(trimmed)
        if candidate:
            try:
                parsed = json.loads(candidate)
                if _is_wrapper_object(parsed):
                    return parsed
            except ValueError:
                pass
    return raw


def _is_wrapper_object(value: Any) -> bool:
    return isinstance(value, dict) and any(k in value for k in _CONTENT_WRAPPER_KEYS)


def _fix_invalid_json_escapes(s: str) -> str:
    # 非法转义 \X（X ∉ " \ / b f n r t u）补成 \\X，修掉模型常见的 \| 等
    return re.sub(
        r"\\(.)",
        lambda m: m.group(0) if m.group(1) in _VALID_JSON_ESCAPE_CHARS else "\\\\" + m.group(1),
        s,
    )


def _first_balanced_object(s: str) -> str | None:
    """取首个花括号配平的 {...}（处理字符串字面与转义），丢弃尾部杂字符。"""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


# ─── diff 解析 ───────────────────────────────────────────────


def _normalise_hunks(raw_hunks: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in raw_hunks:
        if not isinstance(raw, dict):
            continue
        raw_lines = raw.get("lines")
        lines = [
            line
            for line in (raw_lines if isinstance(raw_lines, list) else [])
            if isinstance(line, str) and not _METADATA_LINE_RE.match(line)
        ]
        if not lines:
            continue
        out.append(
            {
                "oldStart": _coalesce(_read_positive_number(raw.get("oldStart")), 1),
                "oldLines": _coalesce(
                    _read_non_negative_number(raw.get("oldLines")), _count_hunk_lines(lines, "+")
                ),
                "newStart": _coalesce(_read_positive_number(raw.get("newStart")), 1),
                "newLines": _coalesce(
                    _read_non_negative_number(raw.get("newLines")), _count_hunk_lines(lines, "-")
                ),
                "lines": lines,
            }
        )
    return out


def _parse_unified_diff(raw_diff: str | None) -> list[dict[str, Any]]:
    if not raw_diff:
        return []
    hunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in raw_diff.replace("\r\n", "\n").split("\n"):
        header = _UNIFIED_DIFF_HEADER_RE.match(line)
        if header:
            current = {
                "oldStart": int(header.group(1)),
                "oldLines": int(header.group(2)) if header.group(2) else 1,
                "newStart": int(header.group(3)),
                "newLines": int(header.group(4)) if header.group(4) else 1,
                "lines": [],
            }
            hunks.append(current)
            continue
        if current is None:
            continue
        if line.startswith("\\ No newline") or _METADATA_LINE_RE.match(line):
            continue
        if line.startswith("+") or line.startswith("-") or line.startswith(" "):
            current["lines"].append(line)

    return [h for h in hunks if h["lines"]]


def _count_hunk_lines(lines: list[str], excluded_prefix: str) -> int:
    return sum(1 for line in lines if not line.startswith(excluded_prefix))


# ─── ppt theme / mermaid theme / 语言猜测 ────────────────────


def _normalise_ppt_theme(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    def hex_color(v: Any) -> str | None:
        s = _read_string(v)
        return s[1:] if s is not None and s.startswith("#") else s

    theme: dict[str, Any] = {}
    pairs = (
        (("primary", "primaryColor", "color"), "primary"),
        (("background", "bg"), "background"),
        (("surface", "card"), "surface"),
        (("textBody", "text", "bodyColor"), "textBody"),
        (("textMuted", "muted"), "textMuted"),
        (("accentPositive", "positive", "success"), "accentPositive"),
        (("accentNegative", "negative", "danger", "warning"), "accentNegative"),
        (("divider", "border"), "divider"),
    )
    for aliases, key in pairs:
        for alias in aliases:
            color = hex_color(value.get(alias))
            if color:
                theme[key] = color
                break
    font_heading = _coalesce(
        _read_string(value.get("fontHeading")),
        _read_string(value.get("headingFont")),
        _read_string(value.get("fontFace")),
        _read_string(value.get("font")),
    )
    if font_heading:
        theme["fontHeading"] = font_heading
    font_body = _coalesce(
        _read_string(value.get("fontBody")),
        _read_string(value.get("bodyFont")),
        _read_string(value.get("font")),
    )
    if font_body:
        theme["fontBody"] = font_body
    return theme or None


def _normalise_mermaid_theme(value: Any) -> str | None:
    theme = _read_string(value)
    return theme if theme in _MERMAID_THEMES else None


def _guess_language(workspace_path: str) -> str:
    ext = workspace_path.split(".")[-1].lower() if "." in workspace_path else ""
    return _LANGUAGE_ALIASES.get(ext, ext or "text")


def _contains_unbounded_binary_payload(value: Any) -> bool:
    """拒收 data:base64 —— PPT 里塞原始二进制会撑爆 DB。"""
    if isinstance(value, str):
        return bool(_DATA_URI_RE.match(value.strip()))
    if isinstance(value, list):
        return any(_contains_unbounded_binary_payload(v) for v in value)
    if isinstance(value, dict):
        return any(_contains_unbounded_binary_payload(v) for v in value.values())
    return False


# ─── 标量读取助手 ────────────────────────────────────────────


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _read_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _to_js_number(value: Any) -> float | None:
    """模仿 JS Number()：bool → NaN；数字串（含空白包裹）→ 数值；其余 → NaN。"""
    if isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value == "":
            return 0.0
        try:
            return float(value.strip())
        except ValueError:
            return math.nan
    return math.nan


def _read_non_negative_number(value: Any) -> int | None:
    n = _to_js_number(value)
    if n is None or math.isnan(n) or math.isinf(n) or n < 0:
        return None
    return math.floor(n)


def _read_positive_number(value: Any) -> int | None:
    n = _read_non_negative_number(value)
    return n if n and n > 0 else None


def _normalise_diagram_source(source: str) -> str | None:
    if len(source.strip()) > MAX_DIAGRAM_SOURCE_CHARS:
        return None
    result = normalise_mermaid_source(source)
    return result.source if result.ok else None
