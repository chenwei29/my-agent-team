"""CamelModel —— 所有 API 出入参的基类。

前端直接读 agent.systemPrompt / message.createdAt，所以 **API 出入参一律 camelCase**，
DB 列才是 snake_case。Python 侧字段写 snake_case，由 alias_generator 生成 camelCase alias。

漏继承 CamelModel 是本项目最容易犯的错 —— 症状是前端拿到 snake_case 字段后静默显示空白。
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic.alias_generators import to_camel

T = TypeVar("T", bound=BaseModel)


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,  # 允许 Python 侧用 snake_case 构造
        from_attributes=True,  # 允许从 SQLAlchemy 对象直接转换
    )


def validate_body(model: type[T], raw: Any) -> T:
    """校验请求体：失败抛 InvalidBody（→ 400），不是 FastAPI 默认的 422。"""
    from app.errors import InvalidBody

    try:
        return model.model_validate(raw)
    except ValidationError as err:
        issues = [
            {"path": list(e["loc"]), "message": e["msg"]} for e in err.errors(include_url=False)
        ]
        raise InvalidBody(issues) from err
