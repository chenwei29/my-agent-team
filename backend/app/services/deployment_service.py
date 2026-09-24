"""本地静态部署引擎：把 web_app 产物或工作区静态目录复制成一份可直接访问的站点。

部署不落 DB —— 全部状态都在文件系统上，一个部署就是一个目录：

    <dataDir>/deployments/<depId>/
        index.html            运行入口（web_app 由 iframe 骨架拼出来）
        <其它公开文件>         产物原始文件 / 工作区拷贝，浏览器直接取
        .agenthub/manifest.json   私有元数据（不进静态服务、不进源码包）
        .agenthub/source/         产物原始文件，供「源码包」下载

目录布局定死的原因：静态服务、下载包、外部发布三处都靠「公开文件 = 除 .agenthub 之外的一切」
这条规则，改动布局要同步这三处。

部署失败不是异常，而是数据：调用方拿 status='failed' 的记录继续走（前端要显示失败卡片）。
"""

from __future__ import annotations

import io
import json
import os
import posixpath
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from app.config import get_settings
from app.security.workspace_utils import is_path_within
from app.utils.webapp_html import build_iframe_html

DEPLOYMENT_ID_RE = re.compile(r"^dep_[0-9A-Za-z]+$")
PRIVATE_DIR = ".agenthub"
MANIFEST_PATH = f"{PRIVATE_DIR}/manifest.json"
SOURCE_ROOT = f"{PRIVATE_DIR}/source"
RUNTIME_ENTRY = "index.html"

# 告诉 LLM 怎么向用户描述部署结果：previewPath 是本机相对路径，不能自己编域名
DEPLOYMENT_SUMMARY_INSTRUCTION = (
    "User-facing summaries must not invent a hostname or public URL for this deployment. "
    "The previewPath is a local relative path for the current AgentHub instance; tell the user "
    "to use the deployment card buttons, or quote previewPath exactly."
)

WORKSPACE_DEPLOY_MAX_FILES = 2000
WORKSPACE_DEPLOY_MAX_BYTES = 100 * 1024 * 1024
WORKSPACE_DEPLOY_IGNORED_DIRS = {".agenthub", ".git", "node_modules"}


class DeploymentError(Exception):
    """部署过程中的可预期失败（内容不合法、路径逃逸、目标已存在等）。"""


# ─── 路径 ────────────────────────────────────────────────────


def deployments_root() -> Path:
    return get_settings().data_dir / "deployments"


def deployment_dir(deployment_id: str) -> Path:
    assert_deployment_id(deployment_id)
    return deployments_root() / deployment_id


def deployment_preview_path(deployment_id: str) -> str:
    """iframe 真实预览地址。

    挂在 /api 前缀下，前端 rewrites 已经覆盖 /api/:path*，不用改前端配置。
    """
    return f"/api/deployments/{deployment_id}"


def deployment_download_path(deployment_id: str, kind: str) -> str:
    return f"/api/deployments/{deployment_id}/download/{kind}"


def is_deployment_id(value: str) -> bool:
    return bool(DEPLOYMENT_ID_RE.match(value))


def assert_deployment_id(value: str) -> None:
    if not is_deployment_id(value):
        raise DeploymentError(f"Invalid deployment id: {value}")


def normalize_deployment_file_path(file_path: str) -> str | None:
    """把请求里的相对路径归一化成安全的 posix 相对路径；不合法返回 None。

    拒的是：空串、绝对路径、盘符、`..`、空段、`.agenthub` 私有目录。
    """
    if "\0" in file_path:
        return None
    raw = file_path.strip().replace("\\", "/")
    if not raw:
        return None
    if raw.startswith("/") or raw.startswith("//") or re.match(r"^[A-Za-z]:", raw):
        return None
    if any(not segment or segment == ".." for segment in raw.split("/")):
        return None

    normalized = str(PurePosixPath(raw))
    if normalized in (".", "..") or normalized.startswith("../") or normalized.startswith("/"):
        return None

    segments = normalized.split("/")
    if any(not segment or segment in (".", "..") for segment in segments):
        return None
    if segments[0].lower() == PRIVATE_DIR:
        return None
    return normalized


def _safe_join(root: Path, relative_path: str) -> Path:
    """root 下拼接相对路径；私有目录（manifest / source）走白名单例外。"""
    normalized = normalize_deployment_file_path(relative_path)
    allowed_private = relative_path == MANIFEST_PATH or relative_path.startswith(
        f"{SOURCE_ROOT}/"
    )
    if normalized is None and not allowed_private:
        raise DeploymentError(f"Invalid deployment file path: {relative_path}")

    parts = relative_path.replace("\\", "/").split("/")
    abs_path = root.joinpath(*parts).resolve()
    if not is_path_within(str(abs_path), str(root.resolve())):
        raise DeploymentError(f"Deployment path escapes root: {relative_path}")
    return abs_path


def _write_text_within(root: Path, relative_path: str, body: str) -> None:
    target = _safe_join(root, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def _write_binary_within(root: Path, relative_path: str, body: bytes) -> None:
    target = _safe_join(root, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)


# ─── 创建部署 ────────────────────────────────────────────────


def create_local_static_deployment(
    *,
    deployment_id: str,
    artifact_id: str,
    title: str,
    version: int,
    content: dict[str, Any],
    created_at: int,
) -> dict[str, Any]:
    """把 web_app 产物铺开成站点：原始文件进 source/，入口另外拼成 index.html。"""
    assert_deployment_id(deployment_id)
    root = deployments_root()
    target = deployment_dir(deployment_id)
    if target.exists():
        raise DeploymentError(f"Deployment already exists: {deployment_id}")

    files = _normalize_web_app_files(content.get("files") or {})
    source_entry = _resolve_source_entry(str(content.get("entry") or ""), files)
    manifest = {
        "id": deployment_id,
        "artifactId": artifact_id,
        "title": title,
        "version": version,
        "deploymentType": "local_static",
        "createdAt": created_at,
        "sourceEntry": source_entry,
        "runtimeEntry": RUNTIME_ENTRY,
        "sourceFiles": sorted(files.keys()),
        "sourceType": "artifact",
    }

    try:
        target.mkdir(parents=True, exist_ok=True)
        for name, body in files.items():
            _write_text_within(target, f"{SOURCE_ROOT}/{name}", body)
            _write_text_within(target, name, body)

        runtime_html = build_iframe_html(files, source_entry)
        _write_text_within(target, RUNTIME_ENTRY, runtime_html)
        _write_text_within(target, MANIFEST_PATH, json.dumps(manifest, indent=2))
    except Exception:
        _cleanup_partial(target, root)
        raise

    return {
        "id": deployment_id,
        "artifactId": artifact_id,
        "title": title,
        "version": version,
        "previewPath": deployment_preview_path(deployment_id),
        "deploymentType": "local_static",
        "deploymentPath": deployment_preview_path(deployment_id),
        "sourceDownloadPath": deployment_download_path(deployment_id, "source"),
        "containerDownloadPath": deployment_download_path(deployment_id, "container"),
        "summaryInstruction": DEPLOYMENT_SUMMARY_INSTRUCTION,
        "status": "ready",
        "createdAt": created_at,
        "sourceType": "artifact",
    }


def create_workspace_static_deployment(
    *,
    deployment_id: str,
    title: str,
    source_dir: str,
    workspace_path: str,
    entry: str | None,
    created_at: int,
) -> dict[str, Any]:
    """把工作区里的静态输出目录（dist / build / out…）原样拷成站点。"""
    assert_deployment_id(deployment_id)
    if not os.path.isdir(source_dir):
        raise DeploymentError(f"Workspace deployment source is not a directory: {workspace_path}")

    root = deployments_root()
    target = deployment_dir(deployment_id)
    if target.exists():
        raise DeploymentError(f"Deployment already exists: {deployment_id}")

    source_files = _list_workspace_static_files(source_dir)
    source_entry = _resolve_workspace_entry(entry, source_files)
    manifest = {
        "id": deployment_id,
        "artifactId": f"workspace:{workspace_path}",
        "title": title,
        "version": 0,
        "deploymentType": "local_static",
        "createdAt": created_at,
        "sourceEntry": source_entry,
        "runtimeEntry": RUNTIME_ENTRY,
        "sourceFiles": source_files,
        "sourceType": "workspace",
        "workspacePath": workspace_path,
    }

    try:
        target.mkdir(parents=True, exist_ok=True)
        for file in source_files:
            body = (Path(source_dir) / file).read_bytes()
            _write_binary_within(target, f"{SOURCE_ROOT}/{file}", body)
            _write_binary_within(target, file, body)

        # 入口不是 index.html 时补一份拷贝，静态服务的默认入口才找得到
        if source_entry != RUNTIME_ENTRY:
            _write_binary_within(
                target, RUNTIME_ENTRY, (Path(source_dir) / source_entry).read_bytes()
            )
        _write_text_within(target, MANIFEST_PATH, json.dumps(manifest, indent=2))
    except Exception:
        _cleanup_partial(target, root)
        raise

    return {
        "id": deployment_id,
        "artifactId": manifest["artifactId"],
        "title": title,
        "version": 0,
        "previewPath": deployment_preview_path(deployment_id),
        "deploymentType": "local_static",
        "deploymentPath": deployment_preview_path(deployment_id),
        "sourceDownloadPath": deployment_download_path(deployment_id, "source"),
        "containerDownloadPath": deployment_download_path(deployment_id, "container"),
        "summaryInstruction": DEPLOYMENT_SUMMARY_INSTRUCTION,
        "status": "ready",
        "createdAt": created_at,
        "sourceType": "workspace",
        "workspacePath": workspace_path,
    }


def _normalize_web_app_files(files: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, body in (files or {}).items():
        normalized = normalize_deployment_file_path(str(name))
        if normalized is None:
            raise DeploymentError(f"Unsafe web app file path: {name}")
        if normalized in out:
            raise DeploymentError(f"Duplicate web app file path after normalization: {name}")
        out[normalized] = body if isinstance(body, str) else str(body)
    if not out:
        raise DeploymentError("Web app artifact has no deployable files")
    return out


def _resolve_source_entry(entry: str, files: dict[str, str]) -> str:
    normalized_entry = normalize_deployment_file_path(entry)
    if normalized_entry is None:
        raise DeploymentError(f"Unsafe web app entry path: {entry}")
    if normalized_entry in files:
        return normalized_entry
    if "index.html" in files:
        return "index.html"
    first_html = next((name for name in files if name.lower().endswith(".html")), None)
    if first_html:
        return first_html
    raise DeploymentError(f"Web app entry file not found: {entry}")


def _list_workspace_static_files(
    source_dir: str, relative_dir: str = "", acc: dict[str, int] | None = None
) -> list[str]:
    """列出工作区静态文件；跳过点目录（`.well-known` 除外）与构建产物目录，并限量。"""
    acc = acc if acc is not None else {"files": 0, "bytes": 0}
    abs_dir = Path(source_dir) / relative_dir if relative_dir else Path(source_dir)
    out: list[str] = []

    for entry in sorted(abs_dir.iterdir(), key=lambda item: item.name):
        if entry.name.startswith(".") and entry.name != ".well-known":
            continue
        if entry.is_dir() and entry.name in WORKSPACE_DEPLOY_IGNORED_DIRS:
            continue

        rel = f"{relative_dir}/{entry.name}" if relative_dir else entry.name
        normalized = normalize_deployment_file_path(rel)
        if normalized is None:
            continue

        if entry.is_dir():
            out.extend(_list_workspace_static_files(source_dir, normalized, acc))
            continue
        if not entry.is_file():
            continue

        acc["files"] += 1
        acc["bytes"] += entry.stat().st_size
        if acc["files"] > WORKSPACE_DEPLOY_MAX_FILES:
            raise DeploymentError(
                f"Workspace deployment has too many files (>{WORKSPACE_DEPLOY_MAX_FILES})"
            )
        if acc["bytes"] > WORKSPACE_DEPLOY_MAX_BYTES:
            raise DeploymentError(
                "Workspace deployment is too large "
                f"(>{WORKSPACE_DEPLOY_MAX_BYTES // 1024 // 1024}MB)"
            )
        out.append(normalized)

    return sorted(out)


def _resolve_workspace_entry(entry: str | None, files: list[str]) -> str:
    raw = (entry or "").strip() or RUNTIME_ENTRY
    normalized = normalize_deployment_file_path(raw)
    if normalized is None:
        raise DeploymentError(f"Unsafe workspace deployment entry path: {raw}")
    if normalized not in files:
        raise DeploymentError(f"Workspace deployment entry not found: {normalized}")
    if not normalized.lower().endswith(".html"):
        raise DeploymentError(f"Workspace deployment entry must be an HTML file: {normalized}")
    return normalized


def _cleanup_partial(target: Path, root: Path) -> None:
    """写一半失败时把目录删干净 —— 否则残留目录会让同名重试直接报「已存在」。"""
    if is_path_within(str(target.resolve()), str(root.resolve())):
        shutil.rmtree(target, ignore_errors=True)


# ─── 外部静态发布 ────────────────────────────────────────────


def publish_deployment_to_static_directory(
    deployment_id: str, publish_dir: str, public_base_url: str
) -> dict[str, Any]:
    """把公开文件整目录拷到用户配置的发布目录，并算出公开 URL。"""
    manifest = read_deployment_manifest(deployment_id)
    if manifest is None:
        raise DeploymentError(f"Deployment not found: {deployment_id}")

    publish_root = _normalize_publish_root(publish_dir)
    target = publish_root / deployment_id
    if not is_path_within(str(target), str(publish_root)):
        raise DeploymentError(f"Publish path escapes configured directory: {target}")

    source_root = deployment_dir(deployment_id)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)

    for file in _list_public_deployment_files(source_root):
        source = _safe_join(source_root, file)
        dest = target.joinpath(*file.split("/"))
        if not is_path_within(str(dest.resolve()), str(target.resolve())):
            raise DeploymentError(f"Publish file path escapes deployment directory: {file}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)

    return {
        "publicUrl": _public_deployment_url(public_base_url, deployment_id),
        "publishPath": str(target),
        "localPreviewPath": deployment_preview_path(deployment_id),
        "publishTargetType": "static_directory",
    }


def _normalize_publish_root(publish_dir: str) -> Path:
    trimmed = (publish_dir or "").strip()
    if not trimmed:
        raise DeploymentError("Deployment publish directory is empty")
    if not os.path.isabs(trimmed):
        raise DeploymentError("Deployment publish directory must be an absolute path")
    resolved = Path(trimmed).resolve()
    if resolved == resolved.parent:
        raise DeploymentError("Deployment publish directory must not be the filesystem root")
    return resolved


def _public_deployment_url(base_url: str, deployment_id: str) -> str:
    """base URL 必须以 / 结尾拼出 `…/<depId>/`；query / fragment 直接丢弃。"""
    assert_deployment_id(deployment_id)
    trimmed = (base_url or "").strip()
    try:
        parts = urlsplit(trimmed)
    except ValueError as err:
        raise DeploymentError("Deployment public base URL must be a valid absolute URL") from err
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise DeploymentError("Deployment public base URL must use http or https")

    path = parts.path if parts.path.endswith("/") else f"{parts.path}/"
    return f"{parts.scheme}://{parts.netloc}{path}{deployment_id}/"


# ─── 读取 ────────────────────────────────────────────────────


def read_deployment_manifest(deployment_id: str) -> dict[str, Any] | None:
    if not is_deployment_id(deployment_id):
        return None
    manifest_path = _safe_join(deployment_dir(deployment_id), MANIFEST_PATH)
    if not manifest_path.is_file():
        return None
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return parsed if _is_deployment_manifest(parsed) else None


def _is_deployment_manifest(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("id"), str)
        and is_deployment_id(value["id"])
        and isinstance(value.get("artifactId"), str)
        and isinstance(value.get("title"), str)
        and isinstance(value.get("version"), int)
        and value.get("deploymentType") == "local_static"
        and isinstance(value.get("createdAt"), int)
        and isinstance(value.get("sourceEntry"), str)
        and isinstance(value.get("runtimeEntry"), str)
        and isinstance(value.get("sourceFiles"), list)
        and all(isinstance(item, str) for item in value["sourceFiles"])
    )


def read_deployment_asset(deployment_id: str, path_parts: list[str] | None) -> dict[str, Any]:
    """静态文件服务的一次读取。

    返回 `{ok: True, body, content_type, headers}` 或 `{ok: False, status, error}`。
    `path_parts` 为空时回落到 manifest 里的运行入口。
    """
    if not is_deployment_id(deployment_id):
        return {"ok": False, "status": 404, "error": "Deployment not found"}

    manifest = read_deployment_manifest(deployment_id)
    if manifest is None:
        return {"ok": False, "status": 404, "error": "Deployment not found"}

    requested = "/".join(path_parts) if path_parts else manifest["runtimeEntry"]
    # 私有目录一律当作「不存在」（连 400 都不给），免得暴露 manifest / 源码目录的存在
    normalized_requested = posixpath.normpath(requested.strip().replace("\\", "/"))
    if normalized_requested == PRIVATE_DIR or normalized_requested.startswith(f"{PRIVATE_DIR}/"):
        return {"ok": False, "status": 404, "error": "Deployment asset not found"}

    relative_path = normalize_deployment_file_path(requested)
    if relative_path is None:
        return {"ok": False, "status": 400, "error": "Invalid deployment path"}

    abs_path = _safe_join(deployment_dir(deployment_id), relative_path)
    if not abs_path.is_file():
        return {"ok": False, "status": 404, "error": "Deployment asset not found"}

    content_type = _content_type_for(relative_path)
    return {
        "ok": True,
        "body": abs_path.read_bytes(),
        "content_type": content_type,
        "headers": _response_headers_for(content_type),
    }


def _list_public_deployment_files(root: Path, relative_dir: str = "") -> list[str]:
    """公开文件 = 除 .agenthub 之外的树；下载包与外部发布共用这一条规则。"""
    abs_dir = _safe_join(root, relative_dir) if relative_dir else root
    out: list[str] = []
    for entry in abs_dir.iterdir():
        if entry.name == PRIVATE_DIR:
            continue
        rel = f"{relative_dir}/{entry.name}" if relative_dir else entry.name
        if entry.is_dir():
            out.extend(_list_public_deployment_files(root, rel))
        elif entry.is_file():
            out.append(rel)
    return sorted(out)


# ─── 下载包 ──────────────────────────────────────────────────


def build_deployment_source_zip(deployment_id: str) -> dict[str, Any] | None:
    manifest = read_deployment_manifest(deployment_id)
    if manifest is None:
        return None

    root = deployment_dir(deployment_id)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in manifest["sourceFiles"]:
            abs_path = _safe_join(root, f"{SOURCE_ROOT}/{file}")
            if abs_path.is_file():
                archive.writestr(file, abs_path.read_bytes())
        archive.writestr("README.txt", _source_readme(manifest))
    return {
        "body": buffer.getvalue(),
        "file_name": f"{_download_base_name(manifest)}-source.zip",
        "content_type": "application/zip",
    }


def build_deployment_container_zip(deployment_id: str) -> dict[str, Any] | None:
    """容器包 = 公开文件 + Dockerfile + nginx 配置，解压后 `docker build` 即可跑。"""
    manifest = read_deployment_manifest(deployment_id)
    if manifest is None:
        return None

    root = deployment_dir(deployment_id)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in _list_public_deployment_files(root):
            archive.writestr(f"app/{file}", _safe_join(root, file).read_bytes())
        archive.writestr("Dockerfile", _dockerfile())
        archive.writestr("nginx.conf", _nginx_conf())
        archive.writestr("README.txt", _container_readme(manifest))
    return {
        "body": buffer.getvalue(),
        "file_name": f"{_download_base_name(manifest)}-container.zip",
        "content_type": "application/zip",
    }


def _download_base_name(manifest: dict[str, Any]) -> str:
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", str(manifest["title"]))
    safe_title = re.sub(r"\s+", "_", safe_title)[:60].strip()
    return f"{safe_title or 'artifact'}-v{manifest['version']}-{manifest['id']}"


def _timestamp(created_at: int) -> str:
    return datetime.fromtimestamp(created_at / 1000, tz=timezone.utc).isoformat()


def _source_readme(manifest: dict[str, Any]) -> str:
    if manifest.get("sourceType") == "workspace":
        return "\n".join(
            [
                f"Workspace deployment: {manifest['title']}",
                f"Source path: {manifest.get('workspacePath') or '(unknown)'}",
                f"Deployment: {manifest['id']}",
                f"Entry: {manifest['sourceEntry']}",
                "",
                "This ZIP contains files copied from a workspace static output directory.",
                f"Generated at: {_timestamp(manifest['createdAt'])}",
                "",
            ]
        )
    return "\n".join(
        [
            f"Artifact: {manifest['title']}",
            f"Version: v{manifest['version']}",
            f"Deployment: {manifest['id']}",
            f"Entry: {manifest['sourceEntry']}",
            "",
            "This ZIP contains the original web_app artifact source files.",
            f"Generated at: {_timestamp(manifest['createdAt'])}",
            "",
        ]
    )


def _container_readme(manifest: dict[str, Any]) -> str:
    is_workspace = manifest.get("sourceType") == "workspace"
    return "\n".join(
        [
            f"{'Workspace deployment' if is_workspace else 'Artifact'}: {manifest['title']}",
            (
                f"Source path: {manifest.get('workspacePath') or '(unknown)'}"
                if is_workspace
                else f"Version: v{manifest['version']}"
            ),
            f"Deployment: {manifest['id']}",
            "",
            "Build and run:",
            f"  docker build -t agenthub-{manifest['id']} .",
            f"  docker run --rm -p 8080:80 agenthub-{manifest['id']}",
            "",
            "Then open http://127.0.0.1:8080/",
            f"Generated at: {_timestamp(manifest['createdAt'])}",
            "",
        ]
    )


def _dockerfile() -> str:
    return (
        "\n".join(
            [
                "FROM nginx:1.27-alpine",
                "COPY app/ /usr/share/nginx/html/",
                "COPY nginx.conf /etc/nginx/conf.d/default.conf",
                "EXPOSE 80",
            ]
        )
        + "\n"
    )


def _nginx_conf() -> str:
    return (
        "\n".join(
            [
                "server {",
                "  listen 80;",
                "  server_name _;",
                "  root /usr/share/nginx/html;",
                "  index index.html;",
                "",
                "  location / {",
                "    try_files $uri $uri/ /index.html;",
                "  }",
                "",
                '  add_header X-Content-Type-Options "nosniff" always;',
                "}",
            ]
        )
        + "\n"
    )


# ─── 响应头 / MIME ───────────────────────────────────────────

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


def _content_type_for(file_path: str) -> str:
    return _CONTENT_TYPES.get(os.path.splitext(file_path)[1].lower(), "application/octet-stream")


def _response_headers_for(content_type: str) -> dict[str, str]:
    """部署里的 HTML/JS 都来自 LLM，按不可信内容处理：禁 same-origin、禁外连。"""
    headers = {
        "Content-Type": content_type,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store",
    }
    if content_type.startswith("text/html"):
        headers["Content-Security-Policy"] = "; ".join(
            [
                "sandbox allow-scripts",
                "default-src 'self'",
                "script-src 'self' 'unsafe-inline'",
                "style-src 'self' 'unsafe-inline'",
                "img-src 'self' data: blob: http: https:",
                "font-src 'self' data:",
                "connect-src 'none'",
                "object-src 'none'",
                "base-uri 'none'",
                "form-action 'none'",
                "frame-ancestors 'self'",
            ]
        )
    elif content_type == "image/svg+xml":
        headers["Content-Security-Policy"] = "sandbox; default-src 'none'"
    return headers
