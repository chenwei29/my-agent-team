"""FastAPI 入口：CORS + 路由挂载 + 启动期建表/seed。

跑法（必须在 `backend/` 下 —— 模块内部用的是 `app.` 绝对导入，从仓库根起不来）：
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import (
    agents,
    artifacts,
    attachments,
    conversation_fs,
    conversations,
    deployments,
    fs,
    message_ops,
    messages,
    pending_bash_commands,
    pending_questions,
    pending_writes,
    platform,
    runs,
    settings,
    stream,
)
from app.config import get_settings
from app.db.bootstrap import bootstrap_database
from app.errors import HttpError, InvalidBody, ServiceError
from app.tools.builtin import register_builtin_tools

_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def _allowed_origins() -> list[str]:
    raw = os.environ.get("AGENTHUB_CORS_ORIGINS")
    if raw and raw.strip():
        return [o.strip() for o in raw.split(",") if o.strip()]
    return _DEFAULT_ORIGINS


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings_ = get_settings()
    settings_.ensure_dirs()
    await bootstrap_database()
    yield


app = FastAPI(title="AgentHub (Python)", lifespan=lifespan)

register_builtin_tools()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # SSE 相关响应头要能被前端读到（P2 用）
    expose_headers=["Content-Type", "Cache-Control", "X-Accel-Buffering"],
)

app.include_router(agents.router)
app.include_router(artifacts.router)
app.include_router(attachments.conv_router)
app.include_router(attachments.item_router)
app.include_router(conversations.router)
app.include_router(conversation_fs.router)
app.include_router(deployments.router)
app.include_router(message_ops.router)
app.include_router(messages.router)
app.include_router(pending_bash_commands.router)
app.include_router(pending_questions.router)
app.include_router(pending_writes.router)
app.include_router(platform.router)
app.include_router(fs.router)
app.include_router(settings.router)
app.include_router(stream.router)
app.include_router(runs.router)


# ─── 错误响应 ────────────────────────────────────────────────
# 前端把非 2xx 的 body 当纯文本抛出去，所以 body 形状自由，但状态码必须对。


@app.exception_handler(InvalidBody)
async def _invalid_body_handler(_request: Request, exc: InvalidBody) -> JSONResponse:
    return JSONResponse(
        status_code=400, content={"error": "Invalid body", "issues": exc.issues}
    )


@app.exception_handler(HttpError)
async def _http_error_handler(_request: Request, exc: HttpError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.exception_handler(ServiceError)
async def _service_error_handler(_request: Request, exc: ServiceError) -> JSONResponse:
    """兜底：路由忘了 catch 时按 400 处理（多数端点的默认口径）。"""
    return JSONResponse(status_code=400, content={"error": exc.message})


@app.exception_handler(RequestValidationError)
async def _request_validation_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """请求体校验失败统一返回 400，而不是 FastAPI 默认的 422。"""
    issues = [
        {"path": list(e.get("loc", [])), "message": e.get("msg", "")} for e in exc.errors()
    ]
    return JSONResponse(status_code=400, content={"error": "Invalid body", "issues": issues})
