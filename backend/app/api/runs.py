"""`POST /api/runs/{id}/abort` —— 中止一个正在进行的 run。

只查内存里的 run 注册表（不查 DB）：已结束的 run 找不到 → 404。
端点本身不写库、不推事件；可见效果全部来自那个 run 自己的收尾
（run 行落 aborted、仍 streaming 的消息落终态、补 [已中止] part、推 run.end{status:'aborted'}）。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.errors import HttpError
from app.services.agent_runner import abort_run

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("/{run_id}/abort")
async def abort(run_id: str) -> dict:
    if not abort_run(run_id):
        raise HttpError(404, "Run not found or already finished")
    return {"ok": True}
