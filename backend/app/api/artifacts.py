"""`/api/artifacts` —— 产物全局列表 / 详情 / 删除 / 版本链 / 预览 / 导出。

状态码口径：
- GET /{id} 找不到 → 404 `{error: "Not found"}`（字面量，不是 "Artifact not found"）
- DELETE /{id} 找不到 → 404，error 带完整 message
- POST /{id}/versions：parent 不存在 → 404；内容非法 → 400
- preview 只服务 web_app；其他类型 → 400
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.helpers import read_json
from app.db.session import get_session
from app.errors import HttpError, NotFoundError, ServiceError
from app.schemas.artifacts import (
    ArtifactListItemOut,
    ArtifactOut,
    ArtifactResponse,
    ArtifactsResponse,
    ArtifactVersionsResponse,
    CreateArtifactVersionBody,
    WebAppContent,
)
from app.schemas.base import validate_body
from app.services import artifact_service
from app.utils.webapp_html import build_iframe_html

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])

_PREVIEW_CSP = "; ".join(
    [
        "sandbox allow-scripts",
        "default-src 'none'",
        "script-src 'unsafe-inline'",
        "style-src 'unsafe-inline'",
        "img-src data: blob: http: https:",
        "font-src data:",
        "connect-src 'none'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'self'",
    ]
)


@router.get("", response_model=ArtifactsResponse)
async def list_artifacts(session: AsyncSession = Depends(get_session)) -> dict:
    return {
        "artifacts": [
            ArtifactListItemOut.model_validate(item).model_dump(by_alias=True)
            for item in await artifact_service.list_artifacts(session)
        ]
    }


@router.get("/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(artifact_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    row = await artifact_service.get_artifact(session, artifact_id)
    if row is None:
        raise HttpError(404, "Not found")
    return {"artifact": ArtifactOut.model_validate(row).model_dump(by_alias=True)}


@router.delete("/{artifact_id}")
async def delete_artifact(artifact_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    try:
        await artifact_service.delete_artifact(session, artifact_id)
    except NotFoundError as err:
        raise HttpError(404, err.message) from err
    return {"ok": True}


@router.get("/{artifact_id}/versions", response_model=ArtifactVersionsResponse)
async def list_versions(
    artifact_id: str, session: AsyncSession = Depends(get_session)
) -> dict:
    versions = await artifact_service.list_artifact_versions(session, artifact_id)
    if versions is None:
        raise HttpError(404, f"Artifact not found: {artifact_id}")
    return {
        "versions": [ArtifactOut.model_validate(row).model_dump(by_alias=True) for row in versions]
    }


@router.post("/{artifact_id}/versions", response_model=ArtifactResponse)
async def create_version(
    artifact_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    body = validate_body(CreateArtifactVersionBody, await read_json(request))
    try:
        artifact = await artifact_service.create_artifact_version(
            session, artifact_id, body.content, body.title
        )
    except NotFoundError as err:
        raise HttpError(404, err.message) from err
    except ServiceError as err:
        raise HttpError(400, err.message) from err
    return {"artifact": ArtifactOut.model_validate(artifact).model_dump(by_alias=True)}


@router.get("/{artifact_id}/preview")
async def preview_artifact(
    artifact_id: str, session: AsyncSession = Depends(get_session)
) -> Response:
    row = await artifact_service.get_artifact(session, artifact_id)
    if row is None:
        raise HttpError(404, "Artifact not found")
    if row.content.get("type") != "web_app":
        raise HttpError(400, "Artifact is not a web_app")

    content = WebAppContent.model_validate(row.content)
    return Response(
        content=build_iframe_html(content.files, content.entry),
        media_type="text/html; charset=utf-8",
        headers={
            "Content-Security-Policy": _PREVIEW_CSP,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-store",
        },
    )


@router.get("/{artifact_id}/export")
async def export_artifact(
    artifact_id: str, mode: str = "editable", session: AsyncSession = Depends(get_session)
) -> Response:
    if mode not in ("editable", "visual"):
        raise HttpError(400, f"Unsupported export mode: {mode}")

    row = await artifact_service.get_artifact(session, artifact_id)
    if row is None:
        raise HttpError(404, "Artifact not found")

    base_name = f"{_sanitize_file_name(row.title) or 'artifact'}-v{row.version}"
    content: dict[str, Any] = row.content
    content_type = content.get("type")

    if content_type == "web_app":
        parsed = WebAppContent.model_validate(content)
        return _zip_response(base_name, _web_app_zip(parsed, row))

    if content_type == "document":
        return _download_response(
            content["content"], "text/markdown; charset=utf-8", f"{quote(base_name)}.md"
        )

    if content_type == "image":
        # 外部图片：302 让浏览器走原 URL
        return RedirectResponse(content["url"], status_code=302)

    if content_type == "diagram":
        return _download_response(
            content["source"], "text/plain; charset=utf-8", f"{quote(base_name)}.mmd"
        )

    if content_type == "ppt":
        if mode == "visual":
            raise HttpError(
                501,
                "Visual-priority PPTX export is not enabled yet. "
                "Use the default editable PPTX export instead.",
            )
        # editable 模式暂未做真 .pptx（可选能力），先按 JSON 兜底导出
        return _download_response(
            _pretty_json(content), "application/json", f"{quote(base_name)}.json"
        )

    # code_file / diff / project 及其它：原始 JSON
    return _download_response(
        _pretty_json(content), "application/json", f"{quote(base_name)}.json"
    )


# ─── 导出工具 ────────────────────────────────────────────────


def _download_response(body: str, media_type: str, filename: str) -> Response:
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _zip_response(base_name: str, data: bytes) -> Response:
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{quote(base_name)}.zip"'},
    )


def _web_app_zip(content: WebAppContent, row: Any) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in content.files.items():
            archive.writestr(name, body)
        archive.writestr(
            "README.txt",
            (
                f"Artifact: {row.title}\nVersion: v{row.version}\nEntry: {content.entry}\n\n"
                f"打开 {content.entry} 即可在浏览器中查看。\n"
                f"导出时间: {datetime.now(timezone.utc).isoformat()}\n"
            ),
        )
    return buffer.getvalue()


def _pretty_json(content: dict[str, Any]) -> str:
    return json.dumps(content, ensure_ascii=False, indent=2)


_UNSAFE_FILENAME_CHARS_RE = re.compile(r'[\\/:*?"<>|]')
_FILENAME_WHITESPACE_RE = re.compile(r"\s+")


def _sanitize_file_name(name: str) -> str:
    return _FILENAME_WHITESPACE_RE.sub(
        "_", _UNSAFE_FILENAME_CHARS_RE.sub("_", name)
    )[:60].strip()
