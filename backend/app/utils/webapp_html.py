"""web_app 产物的运行时 HTML 组装（iframe 预览与本地部署共用）。

约定：入口 HTML 里若已有 </head> / </body>，把 style/script 注进去；
否则包一层最小 HTML 骨架。css/js 只认固定文件名（style.css/styles.css、
script.js/main.js/app.js）—— 多文件引用自身的产物不受影响。
"""

from __future__ import annotations

import re

_HEAD_CLOSE_RE = re.compile(r"</head>", re.IGNORECASE)
_BODY_CLOSE_RE = re.compile(r"</body>", re.IGNORECASE)


def build_iframe_html(files: dict[str, str], entry: str) -> str:
    html = files.get(entry) or files.get("index.html") or ""
    css = files.get("style.css") or files.get("styles.css") or ""
    js = files.get("script.js") or files.get("main.js") or files.get("app.js") or ""

    style_tag = f"<style>\n{css}\n</style>" if css else ""
    script_tag = f"<script>(function(){{\n{js}\n}})();</script>" if js else ""

    if _HEAD_CLOSE_RE.search(html):
        return _BODY_CLOSE_RE.sub(
            lambda _m: f"{script_tag}\n</body>",
            _HEAD_CLOSE_RE.sub(lambda _m: f"{style_tag}\n</head>", html, count=1),
            count=1,
        )

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            style_tag,
            "</head>",
            "<body>",
            html,
            script_tag,
            "</body>",
            "</html>",
        ]
    )
