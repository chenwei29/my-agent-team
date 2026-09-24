"""PPT blocks 的规整：把 LLM 给的松散 blocks/slides 结构收成强类型 JSON。

artifact 内容规整（services/artifact_content.py）与前端渲染共用同一套块语义：
heading / paragraph / bullets / metric / quote / timeline / columns / callout /
divider / spacer；不认识的块直接丢弃（宁可少一块，不给渲染器喂未知结构）。
"""

from __future__ import annotations

from typing import Any

# 「取第一个非 None 的值」——等价 JS 的 `a ?? b ?? c`：
# 空串/空数组也是「已给出的值」，不能像 `or` 那样跳过。
def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None

PPT_LAYOUTS = (
    "title",
    "title-bullets",
    "section",
    "blank",
    "content",
    "two-column",
    "metrics",
    "timeline",
    "quote",
)

PPT_TONES = ("neutral", "positive", "negative", "info", "warning")

# columns 块内部允许的块类型（两栏排版容不下大块结构）
_COLUMN_BLOCK_TYPES = ("paragraph", "bullets", "metric", "callout")

# columns 最多 3 栏，超出直接截断（布局撑不下）
_MAX_COLUMNS = 3


def normalize_layout(layout: Any) -> str:
    return layout if layout in PPT_LAYOUTS else "title-bullets"


def strict_ppt_layout(value: Any) -> str | None:
    """内容规整用的严格版：合法布局原样返回（不 trim），非法返回 None。"""
    if isinstance(value, str) and value.strip() and value in PPT_LAYOUTS:
        return value
    return None


def normalize_blocks(raw_blocks: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_blocks, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_blocks:
        block = _normalize_block(raw)
        if block is not None:
            out.append(block)
    return out


def _normalize_block(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    block_type = clean_text(raw.get("type"))

    if block_type == "heading":
        text = _block_text(raw)
        if not text:
            return None
        block: dict[str, Any] = {"type": "heading", "text": text}
        level = raw.get("level")
        if isinstance(level, int) and not isinstance(level, bool) and level in (1, 2):
            block["level"] = level
        return block

    if block_type == "paragraph":
        text = _block_text(raw)
        return {"type": "paragraph", "text": text} if text else None

    if block_type == "bullets":
        items = normalize_string_list(
            _coalesce(raw.get("items"), raw.get("bullets"), raw.get("points"))
        )
        if not items:
            return None
        return {"type": "bullets", "items": items, "ordered": raw.get("ordered") is True}

    if block_type == "metric":
        label = clean_text(raw.get("label"))
        value = clean_text(raw.get("value"))
        if not label or not value:
            return None
        block = {"type": "metric", "label": label, "value": value}
        change = clean_text(raw.get("change"))
        if change:
            block["change"] = change
        tone = _normalize_tone(raw.get("tone"))
        if tone:
            block["tone"] = tone
        return block

    if block_type == "quote":
        text = _block_text(raw)
        if not text:
            return None
        block = {"type": "quote", "text": text}
        attribution = clean_text(
            _coalesce(raw.get("attribution"), raw.get("author"), raw.get("source"))
        )
        if attribution:
            block["attribution"] = attribution
        return block

    if block_type == "timeline":
        items = _normalize_timeline_items(raw.get("items"))
        return {"type": "timeline", "items": items} if items else None

    if block_type == "columns":
        columns = _normalize_columns(raw.get("columns"))
        return {"type": "columns", "columns": columns} if columns else None

    if block_type == "callout":
        text = _block_text(raw)
        if not text:
            return None
        block = {"type": "callout", "text": text}
        title = clean_text(raw.get("title"))
        if title:
            block["title"] = title
        tone = _normalize_tone(raw.get("tone"))
        if tone:
            block["tone"] = tone
        return block

    if block_type == "divider":
        return {"type": "divider"}

    if block_type == "spacer":
        block = {"type": "spacer"}
        if raw.get("size") in ("sm", "md", "lg"):
            block["size"] = raw["size"]
        return block

    return None


def _normalize_columns(raw_columns: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_columns, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_columns[:_MAX_COLUMNS]:
        if not isinstance(raw, dict):
            continue
        title = clean_text(raw.get("title"))
        blocks = _normalize_column_blocks(raw.get("blocks"))
        bullets = normalize_string_list(
            _coalesce(raw.get("bullets"), raw.get("items"), raw.get("points"))
        )
        if bullets:
            blocks.append({"type": "bullets", "items": bullets})
        if not title and not blocks:
            continue
        column: dict[str, Any] = {}
        if title:
            column["title"] = title
        column["blocks"] = blocks
        out.append(column)
    return out


def _normalize_column_blocks(raw_blocks: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_blocks, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_blocks:
        block = _normalize_block(raw)
        if block is not None and block["type"] in _COLUMN_BLOCK_TYPES:
            out.append(block)
    return out


def _normalize_timeline_items(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        label = clean_text(_coalesce(raw.get("label"), raw.get("date"), raw.get("phase")))
        if not label:
            continue
        item: dict[str, Any] = {"label": label}
        title = clean_text(raw.get("title"))
        if title:
            item["title"] = title
        text = clean_text(raw.get("text") or raw.get("description"))
        if text:
            item["text"] = text
        out.append(item)
    return out


def normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [line for item in value if isinstance(item, str) for line in _split_lines(item)]
    if isinstance(value, str):
        return _split_lines(value)
    return []


def _split_lines(value: str) -> list[str]:
    return [line for line in (chunk.strip() for chunk in value.split("\n")) if line]


def _block_text(obj: dict[str, Any]) -> str | None:
    return clean_text(_coalesce(obj.get("text"), obj.get("content"), obj.get("body")))


def clean_text(value: Any) -> str | None:
    """非空字符串 → trim 后返回；其余（含纯空白）→ None。"""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_tone(value: Any) -> str | None:
    return value if value in PPT_TONES else None
