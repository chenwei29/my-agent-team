"""产物（Artifact）的出入参模型与内容判别联合。

content 的 8 种形态逐字对齐 web/src/shared/types.ts 的 ArtifactContent：
web_app / code_file / diff / document / image / diagram / ppt / project。
规整后的 content（services/artifact_content.build_artifact_content 的产物）
保证符合这里的形状；API 响应里 content 仍按 dict 原样透传（历史行不做二次校验），
各消费端（preview / export / deploy）按需用具体子模型解析。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field

from app.schemas.base import CamelModel

# 内容子模型共同配置：字段名走 camelCase alias，多余键拒绝（形状是硬契约）
_STRICT = ConfigDict(extra="forbid")


# ─── 枚举 ───────────────────────────────────────────────────
ArtifactType = Literal[
    "web_app", "code_file", "diff", "document", "image", "diagram", "ppt", "project"
]
WritableArtifactType = Literal["web_app", "document", "image", "ppt", "diagram"]

MermaidTheme = Literal["default", "base", "dark", "forest", "neutral"]
PptLayout = Literal[
    "title",
    "title-bullets",
    "section",
    "blank",
    "content",
    "two-column",
    "metrics",
    "timeline",
    "quote",
]
PptTone = Literal["neutral", "positive", "negative", "info", "warning"]


# ─── diff ───────────────────────────────────────────────────
class DiffHunk(CamelModel):
    model_config = _STRICT

    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    lines: list[str]


# ─── ppt ────────────────────────────────────────────────────
class PptTimelineItem(CamelModel):
    model_config = _STRICT

    label: str
    title: str | None = None
    text: str | None = None


class PptHeadingBlock(CamelModel):
    model_config = _STRICT

    type: Literal["heading"]
    text: str
    level: Literal[1, 2] | None = None


class PptParagraphBlock(CamelModel):
    model_config = _STRICT

    type: Literal["paragraph"]
    text: str


class PptBulletsBlock(CamelModel):
    model_config = _STRICT

    type: Literal["bullets"]
    items: list[str]
    ordered: bool | None = None


class PptMetricBlock(CamelModel):
    model_config = _STRICT

    type: Literal["metric"]
    label: str
    value: str
    change: str | None = None
    tone: PptTone | None = None


class PptQuoteBlock(CamelModel):
    model_config = _STRICT

    type: Literal["quote"]
    text: str
    attribution: str | None = None


class PptTimelineBlock(CamelModel):
    model_config = _STRICT

    type: Literal["timeline"]
    items: list[PptTimelineItem]


class PptColumnParagraphBlock(CamelModel):
    model_config = _STRICT

    type: Literal["paragraph"]
    text: str


class PptColumnBulletsBlock(CamelModel):
    model_config = _STRICT

    type: Literal["bullets"]
    items: list[str]
    ordered: bool | None = None


class PptColumnMetricBlock(CamelModel):
    model_config = _STRICT

    type: Literal["metric"]
    label: str
    value: str
    change: str | None = None
    tone: PptTone | None = None


class PptColumnCalloutBlock(CamelModel):
    model_config = _STRICT

    type: Literal["callout"]
    text: str
    title: str | None = None
    tone: PptTone | None = None


PptColumnBlock = (
    PptColumnParagraphBlock | PptColumnBulletsBlock | PptColumnMetricBlock | PptColumnCalloutBlock
)


class PptColumn(CamelModel):
    model_config = _STRICT

    title: str | None = None
    blocks: list[PptColumnBlock] | None = None


class PptColumnsBlock(CamelModel):
    model_config = _STRICT

    type: Literal["columns"]
    columns: list[PptColumn]


class PptCalloutBlock(CamelModel):
    model_config = _STRICT

    type: Literal["callout"]
    text: str
    title: str | None = None
    tone: PptTone | None = None


class PptDividerBlock(CamelModel):
    model_config = _STRICT

    type: Literal["divider"]


class PptSpacerBlock(CamelModel):
    model_config = _STRICT

    type: Literal["spacer"]
    size: Literal["sm", "md", "lg"] | None = None


PptBlock = Annotated[
    PptHeadingBlock
    | PptParagraphBlock
    | PptBulletsBlock
    | PptMetricBlock
    | PptQuoteBlock
    | PptTimelineBlock
    | PptColumnsBlock
    | PptCalloutBlock
    | PptDividerBlock
    | PptSpacerBlock,
    Field(discriminator="type"),
]


class PptSlide(CamelModel):
    model_config = _STRICT

    title: str | None = None
    subtitle: str | None = None
    bullets: list[str] | None = None
    blocks: list[PptBlock] | None = None
    notes: str | None = None
    layout: PptLayout | None = None


class PptTheme(CamelModel):
    """幻灯片视觉 token；颜色为不带 # 的 hex（如 '1A3C6E'），全部可选。"""

    model_config = _STRICT

    primary: str | None = None
    background: str | None = None
    surface: str | None = None
    text_body: str | None = None
    text_muted: str | None = None
    accent_positive: str | None = None
    accent_negative: str | None = None
    divider: str | None = None
    font_heading: str | None = None
    font_body: str | None = None
    # 旧字段：渲染端兼容映射到 primary / fontHeading+fontBody，这里照契约保留
    primary_color: str | None = None
    font_face: str | None = None


# ─── ArtifactContent 判别联合 ───────────────────────────────
class WebAppContent(CamelModel):
    model_config = _STRICT

    type: Literal["web_app"]
    files: dict[str, str]
    entry: str


class CodeFileContent(CamelModel):
    model_config = _STRICT

    type: Literal["code_file"]
    workspace_path: str
    language: str
    size_bytes: int
    checksum: str


class DiffContent(CamelModel):
    model_config = _STRICT

    type: Literal["diff"]
    target_artifact_id: str
    hunks: list[DiffHunk]
    applied: bool


class DocumentContent(CamelModel):
    model_config = _STRICT

    type: Literal["document"]
    format: Literal["markdown"]
    content: str


class ImageContent(CamelModel):
    model_config = _STRICT

    type: Literal["image"]
    url: str
    alt: str
    width: int | None = None
    height: int | None = None


class DiagramContent(CamelModel):
    model_config = _STRICT

    type: Literal["diagram"]
    syntax: Literal["mermaid"]
    source: str
    theme: MermaidTheme | None = None


class PptContent(CamelModel):
    model_config = _STRICT

    type: Literal["ppt"]
    title: str | None = None
    theme: PptTheme | None = None
    slides: list[PptSlide]


class ProjectFile(CamelModel):
    model_config = _STRICT

    path: str
    size_bytes: int


class ProjectContent(CamelModel):
    """project 的正文留在 workspace，DB 只存文件清单。"""

    model_config = _STRICT

    type: Literal["project"]
    files: list[ProjectFile]
    task_id: str | None = None
    agent_id: str | None = None


ArtifactContent = Annotated[
    WebAppContent
    | CodeFileContent
    | DiffContent
    | DocumentContent
    | ImageContent
    | DiagramContent
    | PptContent
    | ProjectContent,
    Field(discriminator="type"),
]


# ─── 响应模型 ────────────────────────────────────────────────
class ArtifactOut(CamelModel):
    """产物行（与 DB 行同形）。content 原样透传 —— 历史行不重新校验。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    type: ArtifactType
    title: str
    content: dict[str, Any]
    version: int
    parent_artifact_id: str | None = None
    created_by_agent_id: str
    created_at: int


class ArtifactListItemOut(CamelModel):
    """全局产物库列表项：不带 content（前端列表只展示元信息），带会话标题显示归属。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    conversation_title: str | None = None
    type: ArtifactType
    title: str
    version: int
    parent_artifact_id: str | None = None
    created_by_agent_id: str
    created_at: int


class ArtifactsResponse(CamelModel):
    artifacts: list[ArtifactListItemOut]


class ArtifactResponse(CamelModel):
    artifact: ArtifactOut


class ArtifactVersionsResponse(CamelModel):
    versions: list[ArtifactOut]


# ─── 请求体 ─────────────────────────────────────────────────
class CreateArtifactVersionBody(CamelModel):
    """用户在产物面板编辑后提交新版本：content 形状随 parent 的 type 走。"""

    content: Any
    title: str | None = None
