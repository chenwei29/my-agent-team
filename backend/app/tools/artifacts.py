"""产物工具：write_artifact（创建/新版本）与 read_artifact（同会话读取）。

write_artifact 只写 DB 并返回 artifactId，不发 artifact.create 事件——
事件接线是后续阶段 adapter 的职责（保证事件流单一来源）。

P4 阶段 content 只做「必须是 JSON 对象」的最小校验（DB 列就是 JSON dict）；
按 type 的内容规整留给产物渲染阶段。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select

from app.db.models import Artifact
from app.db.session import SessionLocal
from app.tools.types import ToolContext, ToolDef, ToolResult
from app.utils.ids import new_artifact_id
from app.utils.time import now_ms

ArtifactType = Literal["web_app", "document", "image", "ppt", "diagram"]


class WriteArtifactArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: ArtifactType
    title: str
    content: object = None
    output_key: str | None = Field(default=None, alias="outputKey")
    parent_artifact_id: str | None = Field(default=None, alias="parentArtifactId")


class ReadArtifactArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_id: str = Field(alias="artifactId", min_length=1)


async def _write_artifact(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        parsed = WriteArtifactArgs.model_validate(args or {})
    except ValidationError as err:
        return ToolResult(ok=False, error=f"Invalid args: {err}")

    if not isinstance(parsed.content, dict):
        return ToolResult(
            ok=False,
            error=(
                f"Invalid content for type {parsed.type}: "
                "content must be a JSON object, not a string"
            ),
        )

    version = 1
    resolved_parent: str | None = None
    if parsed.parent_artifact_id:
        async with SessionLocal() as session:
            parent = await session.get(Artifact, parsed.parent_artifact_id)
        if parent is None:
            return ToolResult(
                ok=False, error=f"parentArtifactId not found: {parsed.parent_artifact_id}"
            )
        if parent.conversation_id != ctx.conversation_id:
            return ToolResult(
                ok=False, error="parentArtifactId belongs to a different conversation"
            )
        version = parent.version + 1
        resolved_parent = parent.id

    artifact_id = new_artifact_id()
    row = Artifact(
        id=artifact_id,
        conversation_id=ctx.conversation_id,
        type=parsed.type,
        title=parsed.title,
        content=parsed.content,
        version=version,
        parent_artifact_id=resolved_parent,
        created_by_agent_id=ctx.agent_id,
        created_at=now_ms(),
    )
    async with SessionLocal() as session:
        session.add(row)
        await session.commit()

    value = {
        "artifactId": artifact_id,
        "title": parsed.title,
        "type": parsed.type,
        "version": version,
        "parentArtifactId": resolved_parent,
    }
    if parsed.output_key:
        value["outputKey"] = parsed.output_key
    return ToolResult(ok=True, value=value)


async def _read_artifact(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        parsed = ReadArtifactArgs.model_validate(args or {})
    except ValidationError as err:
        return ToolResult(ok=False, error=f"Invalid args: {err}")

    async with SessionLocal() as session:
        row = await session.scalar(
            select(Artifact).where(
                Artifact.id == parsed.artifact_id,
                Artifact.conversation_id == ctx.conversation_id,
            )
        )
    if row is None:
        return ToolResult(ok=False, error=f"Artifact not found: {parsed.artifact_id}")

    return ToolResult(
        ok=True,
        value={
            "id": row.id,
            "type": row.type,
            "title": row.title,
            "content": row.content,
            "version": row.version,
        },
    )


WRITE_ARTIFACT_TOOL = ToolDef(
    name="write_artifact",
    description=(
        "Create a new artifact, or a new version of an existing one. Never call with empty "
        "args: type, title, and content are required in the same tool call. Pass "
        "parentArtifactId to create a version that links to the prior; version "
        "auto-increments. Use this to produce code/web/docs/images/PPT decks/diagrams "
        "that the user can preview."
    ),
    parameters={
        "type": "object",
        "required": ["type", "title", "content"],
        "properties": {
            "type": {
                "type": "string",
                "enum": ["web_app", "document", "image", "ppt", "diagram"],
                "description": (
                    "web_app for HTML/CSS/JS bundles, document for markdown text, "
                    "image for URL or data URI, ppt for slide decks (structured JSON), "
                    "diagram for Mermaid diagrams"
                ),
            },
            "title": {"type": "string", "description": "Short human-readable title"},
            "content": {
                "type": "object",
                "description": (
                    'Artifact body — pass as a JSON OBJECT, do NOT JSON-stringify it into a '
                    'quoted string. For web_app: { files: { "index.html": "..." }, entry: '
                    '"index.html" }. For document: { format: "markdown", content: "..." }. '
                    'For image: { url: "...", alt: "..." }. For diagram: { syntax: '
                    '"mermaid", source: "flowchart TD\\nA --> B" }.'
                ),
            },
            "parentArtifactId": {
                "type": "string",
                "description": (
                    "Optional: id of an existing artifact to base a new version on. "
                    "When provided, the new row links to it and version increments from "
                    "the parent."
                ),
            },
            "outputKey": {
                "type": "string",
                "description": "Optional handoff key for downstream task consumption.",
            },
        },
    },
    handler=_write_artifact,
)


READ_ARTIFACT_TOOL = ToolDef(
    name="read_artifact",
    description=(
        "Read full content of an existing artifact in the current conversation. "
        "Use when you need the actual body of an artifact referenced by id."
    ),
    parameters={
        "type": "object",
        "required": ["artifactId"],
        "properties": {
            "artifactId": {"type": "string", "description": "Id of the artifact, format art_xxx"},
        },
    },
    handler=_read_artifact,
)
