"""时间戳一律毫秒整数。

前端直接当数字用（new Date(ts) / 排序 / formatDistanceToNow）。
绝不要用 datetime —— 一旦输出 ISO 字符串前端会显示 Invalid Date。
"""

from __future__ import annotations

import time


def now_ms() -> int:
    return int(time.time() * 1000)
