"""部署工具：deploy_artifact（发布 web_app 产物）与 deploy_workspace（发布工作区静态目录）。

两个工具都**不报错**：部署失败也是结果，返回 status='failed' 的记录让前端显示失败卡片，
LLM 也能读到 error 说明后自己决定怎么补救。所以这里把 art_ / 目录不存在、类型不对、
路径逃逸等全部收敛成 failed 记录。
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Artifact, Workspace
from app.db.session import SessionLocal
from app.security.workspace_utils import (
    PathOutsideWorkspaceError,
    assert_path_within_workspace,
    get_effective_cwd,
)
from app.services.deployment_service import (
    DeploymentError,
    create_local_static_deployment,
    create_workspace_static_deployment,
    publish_deployment_to_static_directory,
)
from app.services.settings_service import get_app_settings
from app.tools.types import ToolContext, ToolDef, ToolResult
from app.utils.ids import new_deployment_id
from app.utils.time import now_ms

# 外部发布后的说明：这时候 previewPath 已经是真实公开地址，可以原样引用
EXTERNAL_DEPLOYMENT_SUMMARY_INSTRUCTION = (
    "User-facing summaries may quote the returned previewPath/publicUrl exactly. "
    "Do not invent or rewrite hostnames. If localPreviewPath is present, mention it only "
    "as a local fallback inside AgentHub."
)


class DeployArtifactArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_id: str = Field(alias="artifactId", min_length=1)


class DeployWorkspaceArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = Field(min_length=1)
    title: str | None = None
    entry: str | None = None


async def deploy_artifact_for_conversation(
    session: AsyncSession, conversation_id: str, artifact_id: str
) -> dict[str, Any]:
    """按 id 建本地部署；产物必须属于该会话且是 web_app。"""
    artifact = await session.scalar(
        select(Artifact).where(
            Artifact.id == artifact_id, Artifact.conversation_id == conversation_id
        )
    )
    if artifact is None:
        return _failed_deployment(artifact_id, "Unknown artifact", "Artifact not found")

    content = artifact.content or {}
    if content.get("type") != "web_app":
        return _failed_deployment(
            artifact.id,
            artifact.title,
            f'Artifact type "{content.get("type")}" cannot be deployed as a web app',
            artifact.version,
        )

    try:
        local = create_local_static_deployment(
            deployment_id=new_deployment_id(),
            artifact_id=artifact.id,
            title=artifact.title,
            version=artifact.version,
            content=content,
            created_at=now_ms(),
        )
    except (DeploymentError, OSError) as err:
        return _failed_deployment(artifact.id, artifact.title, str(err), artifact.version)

    return await maybe_publish_externally(session, local)


async def deploy_workspace_for_conversation(
    session: AsyncSession, conversation_id: str, args: DeployWorkspaceArgs
) -> dict[str, Any]:
    """把工作区里的静态输出目录（dist / build / out…）发布成站点。"""
    workspace = await session.scalar(
        select(Workspace).where(Workspace.conversation_id == conversation_id)
    )
    if workspace is None:
        return _failed_workspace_deployment(args.path, "Workspace not found")

    try:
        source_dir = assert_path_within_workspace(workspace, args.path)
    except PathOutsideWorkspaceError as err:
        return _failed_workspace_deployment(args.path, str(err))

    cwd = get_effective_cwd(workspace)
    workspace_path = _relative_to(source_dir, cwd) or "."
    title = (args.title or "").strip() or f"Workspace {workspace_path}"

    try:
        local = create_workspace_static_deployment(
            deployment_id=new_deployment_id(),
            title=title,
            source_dir=source_dir,
            workspace_path=workspace_path,
            entry=args.entry,
            created_at=now_ms(),
        )
    except (DeploymentError, OSError) as err:
        return _failed_workspace_deployment(workspace_path, str(err), title)

    return await maybe_publish_externally(session, local)


def _relative_to(path: str, parent: str) -> str:
    try:
        return os.path.relpath(path, parent).replace("\\", "/")
    except ValueError:  # Windows 跨盘符无法求相对路径
        return path.replace("\\", "/")


async def maybe_publish_externally(
    session: AsyncSession, local: dict[str, Any]
) -> dict[str, Any]:
    """用户在设置里开了外部静态发布时，把本地部署再拷一份到发布目录。

    本地部署已经建好，所以这里失败也只是把记录标成 failed 并附上本地兜底路径，
    不回滚 —— 本地预览仍然可用。
    """
    settings = await get_app_settings(session)
    if not settings.get("deployment_publish_enabled"):
        return local

    publish_dir = settings.get("deployment_publish_dir")
    public_base_url = settings.get("deployment_public_base_url")
    if not publish_dir or not public_base_url:
        return {
            **local,
            "status": "failed",
            "deploymentType": "external_static",
            "localPreviewPath": local["previewPath"],
            "error": (
                "External static publishing is enabled, but deployment publish directory "
                "or public base URL is not configured"
            ),
        }

    try:
        published = publish_deployment_to_static_directory(
            local["id"], publish_dir, public_base_url
        )
    except (DeploymentError, OSError) as err:
        return {
            **local,
            "status": "failed",
            "deploymentType": "external_static",
            "localPreviewPath": local["previewPath"],
            "error": f"External static publish failed: {err}",
        }

    return {
        **local,
        "previewPath": published["publicUrl"],
        "deploymentPath": published["publicUrl"],
        "deploymentType": "external_static",
        "localPreviewPath": published["localPreviewPath"],
        "publicUrl": published["publicUrl"],
        "publishPath": published["publishPath"],
        "publishTargetType": published["publishTargetType"],
        "summaryInstruction": EXTERNAL_DEPLOYMENT_SUMMARY_INSTRUCTION,
    }


def _failed_deployment(
    artifact_id: str, title: str, error: str, version: int = 0
) -> dict[str, Any]:
    return {
        "id": new_deployment_id(),
        "artifactId": artifact_id,
        "title": title,
        "version": version,
        "previewPath": f"/api/artifacts/{artifact_id}/preview",
        "status": "failed",
        "sourceType": "artifact",
        "error": error,
        "createdAt": now_ms(),
    }


def _failed_workspace_deployment(
    workspace_path: str, error: str, title: str | None = None
) -> dict[str, Any]:
    return {
        "id": new_deployment_id(),
        "artifactId": f"workspace:{workspace_path}",
        "title": title or f"Workspace {workspace_path}",
        "version": 0,
        "previewPath": "",
        "status": "failed",
        "sourceType": "workspace",
        "workspacePath": workspace_path,
        "error": error,
        "createdAt": now_ms(),
    }


# ─── 工具入口 ────────────────────────────────────────────────


async def _deploy_artifact(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        parsed = DeployArtifactArgs.model_validate(args or {})
    except ValidationError as err:
        return ToolResult(ok=False, error=f"Invalid args: {err}")

    async with SessionLocal() as session:
        deployment = await deploy_artifact_for_conversation(
            session, ctx.conversation_id, parsed.artifact_id
        )
    return ToolResult(ok=True, value=deployment)


async def _deploy_workspace(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        parsed = DeployWorkspaceArgs.model_validate(args or {})
    except ValidationError as err:
        return ToolResult(ok=False, error=f"Invalid args: {err}")

    async with SessionLocal() as session:
        deployment = await deploy_workspace_for_conversation(
            session, ctx.conversation_id, parsed
        )
    return ToolResult(ok=True, value=deployment)


DEPLOY_ARTIFACT_TOOL = ToolDef(
    name="deploy_artifact",
    description=(
        "Create a local static deployment for a web_app artifact and return its stable "
        "previewPath plus downloadable packages. The previewPath is a relative path for the "
        "current AgentHub instance; do not invent or print a public hostname. In user-facing "
        "summaries, tell the user to use the deployment card buttons or quote previewPath exactly."
    ),
    parameters={
        "type": "object",
        "required": ["artifactId"],
        "properties": {
            "artifactId": {
                "type": "string",
                "description": "Id of the web_app artifact to deploy, format art_xxx",
            },
        },
    },
    handler=_deploy_artifact,
)


DEPLOY_WORKSPACE_TOOL = ToolDef(
    name="deploy_workspace",
    description=(
        "Create a deployment card from a static directory inside the current workspace, such "
        "as dist, build, out, or client/dist. Use this after building a local project. It copies "
        "existing static files only; it does not run npm/pnpm/build commands. The directory must "
        "contain index.html unless entry is provided."
    ),
    parameters={
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    'Static output directory inside the workspace, for example "dist", "build", '
                    '"out", "client/dist", or "apps/web/dist".'
                ),
            },
            "title": {
                "type": "string",
                "description": (
                    "Optional human-readable deployment title. Defaults to the directory name."
                ),
            },
            "entry": {
                "type": "string",
                "description": "Optional HTML entry file relative to path. Defaults to index.html.",
            },
        },
    },
    handler=_deploy_workspace,
)
