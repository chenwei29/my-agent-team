"""测试夹具。

⚠️ AGENTHUB_DATA_DIR 必须在**导入 app 之前**设好 —— engine 是在 app.db.session 的模块级
创建的，路径一旦确定就固定了。conftest 先于测试模块加载，所以在这里设是安全的。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

os.environ["AGENTHUB_DATA_DIR"] = tempfile.mkdtemp(prefix="agenthub-test-")

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.db.bootstrap import bootstrap_database  # noqa: E402
from app.db.bootstrap_cli import E2E_MOCK_AGENT_ID, insert_mock_agent  # noqa: E402
from app.main import app  # noqa: E402

MOCK_AGENT_ID = E2E_MOCK_AGENT_ID


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """每个用例前跑一次幂等 bootstrap + 插入 E2E mock agent，再给一个 ASGI 客户端。

    不用 lifespan：httpx 的 ASGITransport 不会触发 lifespan，建表得自己来。
    """
    await bootstrap_database()
    await insert_mock_agent()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as async_client:
        yield async_client


@pytest.fixture
def safe_dir() -> Iterator[str]:
    """一个「路径安检」会放行的临时目录。

    不能用 pytest 的 tmp_path / tempfile：macOS 上它们在 /var/folders 或 /private/tmp 下，
    而 is_path_safe 把 /var 和 /private 都列进了系统根目录，一律拒绝，
    所以这里在 home 下开目录（home 子目录是明确允许的）。
    """
    target = tempfile.mkdtemp(prefix="agenthub-safe-", dir=str(Path.home()))
    try:
        yield target
    finally:
        shutil.rmtree(target, ignore_errors=True)


@pytest_asyncio.fixture
async def conversation(client: AsyncClient) -> dict:
    """一个单聊会话（与 mock agent），返回创建出来的 conversation 对象。"""
    res = await client.post(
        "/api/conversations", json={"mode": "single", "agentIds": [MOCK_AGENT_ID]}
    )
    assert res.status_code == 201, res.text
    return res.json()["conversation"]
