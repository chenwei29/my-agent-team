"""服务层异常 —— 路由据类型映射到状态码。

各路由对状态码的处理并不一致（同样的 "Conversation not found"，DELETE 返回 404
而 PATCH 返回 400），所以这里保留细分的异常类型，让每个路由能精确表达自己那一份行为。
"""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    """业务错误，路由自行决定状态码（多数是 400）。"""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(ServiceError):
    """资源不存在 —— 映射 404。"""


class ConflictError(ServiceError):
    """状态冲突（如清空历史时有 run 在跑）—— 映射 409。"""


class InvalidBody(Exception):
    """请求体校验失败 —— 映射 400，body 形状 { error, issues }。"""

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        super().__init__("Invalid body")
        self.issues = issues


class HttpError(Exception):
    """路由层显式指定的 HTTP 错误 —— body 形状 { error: message }。"""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
