"""`/api/deployments` —— 部署产物的静态文件服务与下载包。

挂 `/api` 前缀的原因：前端 rewrites 只覆盖 `/api/:path*`，部署卡片的 iframe
直接拿 previewPath 当 src 用，路径必须落在转发范围内。

私有目录 `.agenthub`（manifest + 源码副本）一律当作不存在，连错误原因都不区分。
"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Response

from app.errors import HttpError
from app.services import deployment_service

router = APIRouter(prefix="/api/deployments", tags=["deployments"])


@router.get("/{deployment_id}/download/{kind}")
async def download_deployment(deployment_id: str, kind: str) -> Response:
    if kind not in ("source", "container"):
        raise HttpError(400, "Invalid download kind")

    download = (
        deployment_service.build_deployment_source_zip(deployment_id)
        if kind == "source"
        else deployment_service.build_deployment_container_zip(deployment_id)
    )
    if download is None:
        raise HttpError(404, "Deployment not found")

    return Response(
        content=download["body"],
        media_type=download["content_type"],
        headers={
            "Content-Disposition": (
                f'attachment; filename="{quote(download["file_name"])}"'
            ),
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )


@router.get("/{deployment_id}")
async def read_deployment_root(deployment_id: str) -> Response:
    return _asset_response(deployment_id, None)


@router.get("/{deployment_id}/{path:path}")
async def read_deployment_asset(deployment_id: str, path: str) -> Response:
    return _asset_response(deployment_id, path.split("/"))


def _asset_response(deployment_id: str, path_parts: list[str] | None) -> Response:
    asset = deployment_service.read_deployment_asset(deployment_id, path_parts)
    if not asset["ok"]:
        raise HttpError(asset["status"], asset["error"])
    return Response(
        content=asset["body"],
        media_type=asset["content_type"],
        headers=asset["headers"],
    )
