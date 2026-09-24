"""产物内容规整的单元测试（与被测模块同语义对齐的 27 个用例）。"""

from __future__ import annotations

import json

import pydantic

from app.schemas.artifacts import ArtifactContent
from app.services.artifact_content import (
    build_artifact_content,
    describe_artifact_content_error,
)


def test_document_standard_object():
    assert build_artifact_content(
        "document", {"format": "markdown", "content": "# hi"}
    ) == {"type": "document", "format": "markdown", "content": "# hi"}


def test_document_plain_markdown_string():
    assert build_artifact_content("document", "# hi") == {
        "type": "document",
        "format": "markdown",
        "content": "# hi",
    }


def test_document_unwraps_stringified_wrapper():
    raw = json.dumps({"format": "markdown", "content": "# 番茄钟\n\n正文"})
    assert build_artifact_content("document", raw) == {
        "type": "document",
        "format": "markdown",
        "content": "# 番茄钟\n\n正文",
    }


def test_document_non_wrapper_json_stays_literal():
    assert build_artifact_content("document", '{"foo":1}') == {
        "type": "document",
        "format": "markdown",
        "content": '{"foo":1}',
    }


def test_web_app_raw_html_not_treated_as_json():
    assert build_artifact_content("web_app", "<!doctype html><h1>x</h1>") == {
        "type": "web_app",
        "files": {"index.html": "<!doctype html><h1>x</h1>"},
        "entry": "index.html",
    }


def test_web_app_unwraps_stringified_wrapper():
    raw = json.dumps({"files": {"index.html": "<h1>x</h1>"}, "entry": "index.html"})
    assert build_artifact_content("web_app", raw) == {
        "type": "web_app",
        "files": {"index.html": "<h1>x</h1>"},
        "entry": "index.html",
    }


def test_document_tolerates_invalid_escape_in_wrapper():
    raw = '{"format":"markdown","content":"表格 a \\| b 结束"}'
    assert build_artifact_content("document", raw) == {
        "type": "document",
        "format": "markdown",
        "content": "表格 a \\| b 结束",
    }


def test_document_tolerates_trailing_junk_after_wrapper():
    raw = '{"format":"markdown","content":"hi"} trailing-junk'
    assert build_artifact_content("document", raw) == {
        "type": "document",
        "format": "markdown",
        "content": "hi",
    }


def test_web_app_tolerates_invalid_escape_in_multi_file_wrapper():
    raw = '{"files":{"index.html":"<h1>x \\| y</h1>","style.css":"a{}"},"entry":"index.html"}'
    assert build_artifact_content("web_app", raw) == {
        "type": "web_app",
        "files": {"index.html": "<h1>x \\| y</h1>", "style.css": "a{}"},
        "entry": "index.html",
    }


def test_diff_standard_hunks_object():
    assert build_artifact_content(
        "diff",
        {
            "targetArtifactId": "art_target",
            "hunks": [
                {
                    "oldStart": 1,
                    "oldLines": 2,
                    "newStart": 1,
                    "newLines": 2,
                    "lines": [" import x", "-const a = 1", "+const a = 2"],
                }
            ],
        },
    ) == {
        "type": "diff",
        "targetArtifactId": "art_target",
        "applied": False,
        "hunks": [
            {
                "oldStart": 1,
                "oldLines": 2,
                "newStart": 1,
                "newLines": 2,
                "lines": [" import x", "-const a = 1", "+const a = 2"],
            }
        ],
    }


def test_diff_filters_hunk_header_lines_in_hunks():
    assert build_artifact_content(
        "diff",
        {
            "targetArtifactId": "art_target",
            "hunks": [
                {
                    "oldStart": 10,
                    "oldLines": 2,
                    "newStart": 11,
                    "newLines": 2,
                    "lines": ["@@ -10,2 +11,2 @@", " old", "-line a", "+line b"],
                }
            ],
        },
    ) == {
        "type": "diff",
        "targetArtifactId": "art_target",
        "applied": False,
        "hunks": [
            {
                "oldStart": 10,
                "oldLines": 2,
                "newStart": 11,
                "newLines": 2,
                "lines": [" old", "-line a", "+line b"],
            }
        ],
    }


def test_diff_parses_unified_diff_string():
    assert build_artifact_content(
        "diff",
        {
            "targetArtifactId": "art_target",
            "diff": "@@ -10,2 +10,2 @@\n old\n-line a\n+line b",
        },
    ) == {
        "type": "diff",
        "targetArtifactId": "art_target",
        "applied": False,
        "hunks": [
            {
                "oldStart": 10,
                "oldLines": 2,
                "newStart": 10,
                "newLines": 2,
                "lines": [" old", "-line a", "+line b"],
            }
        ],
    }


def test_code_file_standard_metadata_object():
    assert build_artifact_content(
        "code_file",
        {
            "workspacePath": "components/Button.tsx",
            "language": "typescript",
            "sizeBytes": 123,
            "checksum": "abc",
        },
    ) == {
        "type": "code_file",
        "workspacePath": "components/Button.tsx",
        "language": "typescript",
        "sizeBytes": 123,
        "checksum": "abc",
    }


def test_diagram_mermaid_standard_object():
    assert build_artifact_content(
        "diagram",
        {"syntax": "mermaid", "source": "flowchart TD\nA-->B", "theme": "neutral"},
    ) == {
        "type": "diagram",
        "syntax": "mermaid",
        "source": "flowchart TD\nA-->B",
        "theme": "neutral",
    }


def test_diagram_plain_mermaid_string():
    assert build_artifact_content("diagram", "sequenceDiagram\nA->>B: hello") == {
        "type": "diagram",
        "syntax": "mermaid",
        "source": "sequenceDiagram\nA->>B: hello",
    }


def test_diagram_normalises_flowchart_cjk_and_math_labels():
    source = "\n".join(
        [
            "flowchart LR",
            "    A[研究背景: 堆叠智能超表面SIM]",
            "    subgraph C[本文提出的通用多端口网络框架]",
            "        C21[全局矩阵求逆复杂度 从O(LN)^3 降至 O(L*N^2)]",
            "    end",
            "    A --> C",
            "    style A fill:#1A3C6E,color:#fff,font-weight:bold",
        ]
    )
    expected = "\n".join(
        [
            "flowchart LR",
            '    A["研究背景: 堆叠智能超表面SIM"]',
            '    subgraph C["本文提出的通用多端口网络框架"]',
            '        C21["全局矩阵求逆复杂度 从O(LN)^3 降至 O(L*N^2)"]',
            "    end",
            "    A --> C",
            "    style A fill:#1A3C6E,color:#fff,font-weight:bold",
        ]
    )
    assert build_artifact_content("diagram", source) == {
        "type": "diagram",
        "syntax": "mermaid",
        "source": expected,
    }


def test_diagram_returns_actionable_validation_error():
    source = 'flowchart LR\nA["开始"]\nstyle A fill:#fff 这里混入了正文'

    assert build_artifact_content("diagram", source) is None
    assert describe_artifact_content_error("diagram", source) == (
        'Invalid Mermaid diagram: Line 3: invalid style syntax. Use "style ID fill:#hex,color:#hex" '
        "without trailing prose.\nstyle A fill:#fff 这里混入了正文"
    )


def test_ppt_standard_slides_object():
    assert build_artifact_content(
        "ppt",
        {
            "title": "季度汇报",
            "slides": [
                {"title": "封面", "layout": "title"},
                {"title": "要点", "bullets": ["一", "二"]},
            ],
        },
    ) == {
        "type": "ppt",
        "title": "季度汇报",
        "slides": [
            {"title": "封面", "layout": "title"},
            {"title": "要点", "bullets": ["一", "二"]},
        ],
    }


def test_ppt_top_level_array_as_slides():
    assert build_artifact_content("ppt", [{"title": "A"}, {"title": "B", "bullets": ["x"]}]) == {
        "type": "ppt",
        "slides": [{"title": "A"}, {"title": "B", "bullets": ["x"]}],
    }


def test_ppt_bullets_string_split_by_lines_and_points_alias():
    assert build_artifact_content(
        "ppt",
        {
            "slides": [
                {"title": "A", "bullets": "一\n二\n"},
                {"title": "B", "points": ["x", "y"]},
            ]
        },
    ) == {
        "type": "ppt",
        "slides": [
            {"title": "A", "bullets": ["一", "二"]},
            {"title": "B", "bullets": ["x", "y"]},
        ],
    }


def test_ppt_filters_empty_slides():
    assert build_artifact_content(
        "ppt",
        {"slides": [{"title": "keep"}, {}, {"bullets": []}, {"notes": "only notes"}]},
    ) == {"type": "ppt", "slides": [{"title": "keep"}]}


def test_ppt_ignores_invalid_layout_maps_legacy_theme_fields_and_strips_hash():
    assert build_artifact_content(
        "ppt",
        {
            "theme": {"primaryColor": "#1E40AF", "fontFace": "Arial"},
            "slides": [{"title": "A", "layout": "fancy"}],
        },
    ) == {
        "type": "ppt",
        "theme": {"primary": "1E40AF", "fontHeading": "Arial"},
        "slides": [{"title": "A"}],
    }


def test_ppt_theme_full_tokens_with_aliases():
    assert build_artifact_content(
        "ppt",
        {
            "theme": {
                "primary": "#1A3C6E",
                "bg": "#F8F9FA",
                "textBody": "2C3E50",
                "positive": "2B7A4B",
                "danger": "#C0392B",
                "fontHeading": "Inter",
                "fontBody": "Inter",
            },
            "slides": [{"title": "A", "bullets": ["x"]}],
        },
    ) == {
        "type": "ppt",
        "theme": {
            "primary": "1A3C6E",
            "background": "F8F9FA",
            "textBody": "2C3E50",
            "accentPositive": "2B7A4B",
            "accentNegative": "C0392B",
            "fontHeading": "Inter",
            "fontBody": "Inter",
        },
        "slides": [{"title": "A", "bullets": ["x"]}],
    }


def test_ppt_unwraps_stringified_wrapper():
    raw = json.dumps({"slides": [{"title": "A", "bullets": ["x"]}]})
    assert build_artifact_content("ppt", raw) == {
        "type": "ppt",
        "slides": [{"title": "A", "bullets": ["x"]}],
    }


def test_ppt_accepts_enhanced_blocks_and_filters_unknown_block():
    assert build_artifact_content(
        "ppt",
        {
            "slides": [
                {
                    "title": "Metrics",
                    "subtitle": "Q2",
                    "layout": "metrics",
                    "blocks": [
                        {
                            "type": "metric",
                            "label": "收入",
                            "value": "1200万",
                            "change": "+18%",
                            "tone": "positive",
                        },
                        {"type": "unknown", "text": "ignored"},
                        {
                            "type": "timeline",
                            "items": [
                                {"label": "Q1", "title": "启动", "text": "完成验证"},
                                {"label": "", "title": "ignored"},
                            ],
                        },
                    ],
                }
            ]
        },
    ) == {
        "type": "ppt",
        "slides": [
            {
                "title": "Metrics",
                "subtitle": "Q2",
                "layout": "metrics",
                "blocks": [
                    {
                        "type": "metric",
                        "label": "收入",
                        "value": "1200万",
                        "change": "+18%",
                        "tone": "positive",
                    },
                    {"type": "timeline", "items": [{"label": "Q1", "title": "启动", "text": "完成验证"}]},
                ],
            }
        ],
    }


def test_ppt_rejects_inline_data_uri_binary_payload():
    assert (
        build_artifact_content(
            "ppt",
            {
                "slides": [
                    {
                        "title": "Image",
                        "blocks": [{"type": "paragraph", "text": "ok"}],
                        "image": "data:image/png;base64,AAAA",
                    }
                ]
            },
        )
        is None
    )


def test_invalid_inputs_return_none():
    assert build_artifact_content("document", 123) is None
    assert build_artifact_content("diff", {"targetArtifactId": "art_target", "hunks": []}) is None
    assert build_artifact_content("code_file", {"language": "typescript"}) is None
    assert (
        build_artifact_content("diagram", {"syntax": "plantuml", "source": "@startuml\n@enduml"})
        is None
    )
    assert build_artifact_content("diagram", {"source": ""}) is None
    assert build_artifact_content("ppt", {"slides": []}) is None
    assert build_artifact_content("ppt", {"slides": [{}]}) is None
    assert build_artifact_content("ppt", 123) is None


def test_all_built_contents_conform_to_schema_union():
    """规整产物必须全部落在 schemas 的 ArtifactContent 判别联合内（含各类 block）。"""
    adapter = pydantic.TypeAdapter(ArtifactContent)
    samples = [
        build_artifact_content("web_app", {"html": "<h1>x</h1>", "css": "a{}", "js": "b()"}),
        build_artifact_content("document", {"format": "markdown", "content": "# hi"}),
        build_artifact_content("image", {"url": "https://x/y.png", "alt": "y"}),
        build_artifact_content("diagram", {"syntax": "mermaid", "source": "flowchart TD\nA-->B"}),
        build_artifact_content(
            "diff",
            {"targetArtifactId": "art_t", "diff": "@@ -1,2 +1,2 @@\n old\n-a\n+b"},
        ),
        build_artifact_content("code_file", {"workspacePath": "src/a.ts"}),
        build_artifact_content(
            "ppt",
            {
                "title": "D",
                "theme": {"primary": "#111"},
                "slides": [
                    {
                        "title": "S",
                        "subtitle": "sub",
                        "bullets": ["x"],
                        "notes": "n",
                        "layout": "metrics",
                        "blocks": [
                            {"type": "heading", "text": "h", "level": 2},
                            {"type": "paragraph", "text": "p"},
                            {"type": "bullets", "items": ["a"], "ordered": True},
                            {
                                "type": "metric",
                                "label": "l",
                                "value": "v",
                                "change": "+1",
                                "tone": "positive",
                            },
                            {"type": "quote", "text": "q", "attribution": "me"},
                            {"type": "timeline", "items": [{"label": "L", "title": "t", "text": "x"}]},
                            {
                                "type": "columns",
                                "columns": [
                                    {"title": "c", "blocks": [{"type": "callout", "text": "z"}]}
                                ],
                            },
                            {"type": "divider"},
                            {"type": "spacer", "size": "md"},
                        ],
                    }
                ],
            },
        ),
    ]
    for content in samples:
        assert content is not None
        adapter.validate_python(content)
