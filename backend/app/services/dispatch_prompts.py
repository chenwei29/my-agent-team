"""编排各阶段的 prompt 与 XML 渲染（纯函数，不碰 DB）。

三类消费面：
- Orchestrator 的 plan / aggregate 两段 system prompt 扩展；
- 子 Agent 的隔离上下文 prompt（`<context>` + `<your_task>` + 执行规约）；
- 聚合阶段的 user prompt（`<user_request>` + `<task_results>` + `<file_conflicts>`）。

文本是行为契约的一部分（LLM 依它决定调什么工具、按什么格式交付），
改动前先想清楚下游门禁是否依赖其中的措辞。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.schemas.dispatch import DispatchExpectedOutput, DispatchPlanItem, TaskResultReport
from app.services.dispatch_file_writes import FileWriteConflict
from app.services.dispatch_scheduler import ResolvedTaskInput
from app.services.task_result_report import REPORT_TASK_RESULT_TOOL_NAME

ASK_USER_TOOL_NAME = "ask_user"


def escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def xml_attr(value: str) -> str:
    return f'"{escape_xml(value).replace(chr(34), "&quot;")}"'


def _quote(value: Any) -> str:
    """JSON.stringify 语义的属性值引用（不转义非 ASCII，与事件流里的引法一致）。"""
    return json.dumps(value, ensure_ascii=False)


# ─── 消息 parts → 纯文本 ────────────────────────────────────


def format_size(byte_count: int) -> str:
    if byte_count < 1024:
        return f"{byte_count}B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f}KB"
    return f"{byte_count / 1024 / 1024:.1f}MB"


def extract_text_from_parts(parts: list[Any]) -> str:
    """消息 parts 拼成纯文本：text/thinking 直出，code 进 fence，附件折成占位行。"""
    chunks: list[str] = []
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("text", "thinking"):
            chunks.append(part.get("content", ""))
        elif kind == "code":
            chunks.append("```" + str(part.get("language", "")) + "\n" + part.get("content", "") + "\n```")
        elif kind == "image_attachment":
            chunks.append(
                f"[图片附件: {part.get('fileName')} "
                f"({format_size(int(part.get('size') or 0))}, {part.get('mimeType')}) "
                f"· id={part.get('attachmentId')}]"
            )
        elif kind == "file_attachment":
            chunks.append(
                f"[文件附件: {part.get('fileName')} "
                f"({format_size(int(part.get('size') or 0))}, {part.get('mimeType')}) "
                f"· id={part.get('attachmentId')}]"
            )
    return "\n\n".join(chunk for chunk in chunks if chunk)


# ─── 工具调用规范（拼进 system prompt）──────────────────────


def build_agent_hub_tool_guidance(
    adapter_name: str, tool_names: list[str], workspace_mode: str
) -> str:
    """按 agent 实际工具集生成调用规范段落；没有工具时返回空串。"""
    tools = set(tool_names)
    is_sdk_agent = adapter_name in ("claude-code", "codex")
    if is_sdk_agent:
        sdk_agent_hub_tools = (
            ["read_artifact", "read_attachment", "fs_list", ASK_USER_TOOL_NAME]
            if "plan_tasks" in tools
            else [
                "write_artifact",
                "read_artifact",
                "deploy_artifact",
                "deploy_workspace",
                ASK_USER_TOOL_NAME,
                REPORT_TASK_RESULT_TOOL_NAME,
            ]
        )
        tools.update(sdk_agent_hub_tools)
    is_plan_stage = "plan_tasks" in tools

    sections: list[str] = []

    def add(lines: list[str]) -> None:
        sections.append("\n".join(lines))

    has_workspace_file_tools = not is_plan_stage and (
        "fs_read" in tools or "fs_write" in tools or "bash" in tools or is_sdk_agent
    )

    if tools:
        add(
            [
                "## AgentHub 工具调用规范",
                "- 需要调用工具时，必须用工具调用通道提交结构化参数，不要把 JSON 示例写进普通回复里假装调用。",
                "- 字段名必须严格使用工具 schema 里的 camelCase，例如 artifactId、attachmentId、parentArtifactId、outputKey、dependsOn、expectedOutputs、acceptanceCriteria、acceptanceResults。",
                "- 不要编造 artifactId、attachmentId、outputKey、文件路径；只能使用上下文里明确给出的 id / 路径。",
                "- 工具返回 ok:false 或 isError=true 时，先根据错误修正参数；不要继续基于失败结果推进。",
            ]
        )

    if workspace_mode == "local" and has_workspace_file_tools:
        add(
            [
                "## 本地项目模式",
                "当前 workspace 是用户绑定的真实本地文件夹。用户要求创建、修改、初始化、调试、构建前后端项目或源码文件时，必须优先直接操作 workspace 文件。",
                (
                    "- 使用 SDK 自带的 Read / Write / Edit / Bash / shell 工具读写文件、安装依赖、运行构建与测试。"
                    if is_sdk_agent
                    else "- 使用 fs_read / fs_write / bash 读写文件、安装依赖、运行构建与测试。"
                ),
                "- 不要用 write_artifact 保存应该落盘到本地项目的源码、package.json、tsconfig、server/client 文件或构建配置。",
                "- 如果本地项目已经构建出 dist / build / out / client/dist 等静态目录，可用 deploy_workspace 为该目录生成部署预览卡。",
                "- write_artifact 只用于用户明确要求 artifact / 可预览原型 / 独立 demo / 文档交接，或任务本身声明需要 artifact handoff。",
                "- 完成本地项目改动后，优先运行必要的验证命令（install / typecheck / build / test）；如果无法运行，说明具体原因。",
            ]
        )
    elif workspace_mode == "local" and "write_artifact" in tools:
        add(
            [
                "## 本地项目模式",
                "当前 workspace 是用户绑定的真实本地文件夹，但这个 agent 没有文件/命令工具，不能直接修改本地项目。",
                "- 如果用户要求写入本地项目源码，应说明当前 agent 缺少 fs_read / fs_write / bash 或 SDK 本地工具，而不是用 write_artifact 假装已经落盘。",
                "- 只有用户明确要求 artifact / 可预览原型 / 独立 demo / 文档交接时，才使用 write_artifact。",
            ]
        )

    if ASK_USER_TOOL_NAME in tools:
        add(
            [
                "### ask_user",
                "用途：当继续执行前需要用户在有限方案中选择时，发起结构化问答；不要只在普通文本里问。",
                '正确案例：产品范围不清，调用 ask_user({ questions: [{ header: "范围", question: "这次先做哪个范围?", options: [{ label: "核心流程", description: "先打通主路径，风险最低" }, { label: "完整后台", description: "覆盖更多页面，但耗时更长" }] }] })。',
                "参数规则：每次 1-4 个 questions，每题 2-4 个 options；header 是短标签，question 是完整问题，label 是按钮短文本，description 写清选择后果。",
                "错误案例：直接回复“你想做核心流程还是完整后台？”然后停止；这样 UI 不会出现结构化选择，也不会阻塞 run 等待答案。",
                "不要滥用：开放式讨论、非关键细节、或可以保守决策时，直接说明假设并继续。",
            ]
        )

    if "read_attachment" in tools:
        add(
            [
                "### read_attachment",
                "用途：用户上传了文本/文件附件且任务依赖附件内容时，先读取附件；不要只凭文件名猜测。",
                '正确案例：看到上下文有 attachmentId="att_123"，调用 read_attachment({ attachmentId: "att_123" }) 后再总结或实现。',
                "常见错误：传 { id: \"att_123\" } 或把 art_* 产物 id 传给 read_attachment；产物必须用 read_artifact。",
                "错误案例：把“需求.docx”文件名当作完整需求内容。",
            ]
        )

    if "read_artifact" in tools:
        add(
            [
                "### read_artifact",
                "用途：需要基于已有产物继续设计、实现、审查或修改时，先读取完整产物内容。",
                '正确案例：上游只给出 <artifact id="art_123" />，调用 read_artifact({ artifactId: "art_123" })。',
                '常见错误：传 { id: "art_123" }、{ artifact_id: "art_123" }，或把 att_* 附件 id 传给 read_artifact。',
                "错误案例：只根据 artifact 标题或摘要判断内容，直接改写或审查。",
            ]
        )

    if "write_artifact" in tools:
        add(
            [
                "### write_artifact",
                "用途：创建用户需要预览、下载、交接或长期保存的产物；不要用它记录普通聊天结论。",
                "硬性要求：调用前必须已经准备好完整参数；严禁 write_artifact({})，严禁先空调用工具再补参数。",
                "调用前自检：type 必须是工具 schema 允许的枚举值，title 必须是非空字符串，content 必须是对应类型的原始对象。",
                'web_app 正确参数：write_artifact({ type: "web_app", title: "登录页原型", content: { files: { "index.html": "<!doctype html>...", "style.css": "body { ... }", "script.js": "..." }, entry: "index.html" } })。',
                'document 完整模板：write_artifact({ type: "document", title: "PRD", content: { format: "markdown", content: "# PRD\\n\\n## 1. 背景\\n...\\n\\n## 2. 目标\\n...\\n\\n## 3. 方案\\n..." } })。',
                'diagram 正确参数：write_artifact({ type: "diagram", title: "系统调用流程", content: { syntax: "mermaid", source: "flowchart TD\\n  U[\\"用户\\"] --> A[\\"AgentHub\\"]\\n  A --> LLM[\\"LLM\\"]\\n  LLM --> T[\\"工具调用\\"]", theme: "default" } })。适合流程图、时序图、架构图、依赖关系图；不要把 Mermaid 放进 document 代码块里冒充图产物。',
                'diagram 规则：中文、数学公式、括号、冒号、斜杠等 label 一律写成 A["..."]；一行只写一条边；不要把 ```mermaid fence 传给 source。write_artifact 会校验 Mermaid，若返回 Invalid Mermaid diagram，必须根据错误修正 source 后重新调用工具。',
                'ppt 正确参数：write_artifact({ type: "ppt", title: "Q2 复盘", content: { title: "Q2 复盘", theme: { primary: "1A3C6E", background: "F8F9FA", surface: "FFFFFF", textBody: "2C3E50", textMuted: "6B7280", accentPositive: "2B7A4B", accentNegative: "C0392B", divider: "E0E4E8", fontHeading: "Inter", fontBody: "Inter" }, slides: [{ title: "Q2 复盘", subtitle: "关键指标", layout: "metrics", blocks: [{ type: "metric", label: "ARR", value: "$12M", change: "+18%", tone: "positive" }, { type: "callout", title: "下一步", text: "聚焦企业客户扩张", tone: "info" }] }] } })。',
                "ppt 支持 blocks：heading、paragraph、bullets、metric、quote、timeline、columns、callout、divider、spacer；columns 内只放 paragraph/bullets/metric/callout。不要在 ppt JSON 里嵌入 base64/data URI 大资产。",
                '常见错误：把 content 作为 JSON 字符串传入，例如 content: "{\\"format\\":\\"markdown\\"}"；必须传原始对象。',
                "字段名必须是 parentArtifactId、outputKey；不要写 parent_artifact_id、output_key。",
                "如果子任务声明非 project 的 expectedOutputs，创建对应产物时传 outputKey 等于 expectedOutputs.id。",
                "project 产物不能用 write_artifact 创建；代码任务通过 fs_write / bash 写入 workspace 文件后由 AgentHub 自动生成 project。",
            ]
        )

    if "deploy_artifact" in tools:
        add(
            [
                "### deploy_artifact",
                "用途：web_app 产物完成后生成可打开的预览部署卡。",
                '正确流程：先 write_artifact 得到 artifactId="art_123"，再 deploy_artifact({ artifactId: "art_123" })。',
                '常见错误：传 { id: "art_123" }、传还没创建的 id、或对旧版本 id 误部署。',
                "错误案例：自己编造 http://localhost:3000/... 或公网域名；只能引用工具返回的 previewPath，或让用户点击部署卡按钮。",
                "不要对 document/image/ppt 调用 deploy_artifact；它只接受 web_app。",
            ]
        )

    if "deploy_workspace" in tools:
        add(
            [
                "### deploy_workspace",
                "用途：把当前 workspace 内已有的静态输出目录部署成预览卡，例如 dist、build、out、client/dist。",
                '正确流程：先用 bash 运行项目构建命令，确认静态目录存在且包含 index.html，再 deploy_workspace({ path: "dist", title: "前端构建预览" })。',
                "常见错误：把源码根目录、node_modules、server 目录传给 deploy_workspace；它只复制静态文件，不会自动构建或启动服务。",
                "如果项目是 Vite/React/Next 静态导出，优先部署构建输出目录，而不是创建 web_app artifact。",
            ]
        )

    if not is_plan_stage and (
        "fs_list" in tools or "fs_read" in tools or "fs_write" in tools or "bash" in tools
    ):
        add(
            [
                "### workspace 文件与命令工具",
                "用途：只操作当前 workspace 内的真实文件；路径必须在 <workspace_info><cwd> 下。",
                'fs_list 正确案例：fs_list({ path: "" }) 查看根目录；fs_list({ path: "src/components" }) 查看子目录。探索项目结构优先用 fs_list，不要先用 bash 拼目录命令。',
                'fs_read 正确案例：fs_read({ path: "src/components/Chart.tsx" })，先看现有代码再改。',
                'fs_write 正确案例：fs_write({ path: "src/components/Chart.tsx", content: "完整的新文件内容" })；content 是完整文件内容，不是 diff patch。',
                'bash 正确案例：bash({ command: "pnpm typecheck" })；子目录命令用 bash({ command: "pnpm build", cwd: "frontend", timeoutMs: 300000 })，不要写 cd frontend && pnpm build。',
                '临时启动服务测试时，必须在同一个 bash 命令里清理后台进程，例如 `npm run dev > /tmp/agenthub-dev.log 2>&1 & pid=$!; trap "kill $pid" EXIT; sleep 3; curl http://127.0.0.1:3000`；不要裸 `server &` 留长驻后台进程。',
                "常见错误：fs_write 只写局部 diff、bash 里 cd 到 workspace 外、裸 `cmd &` 留后台服务、或在 Windows workspace 用 POSIX-only 参数。",
                "错误案例：读取 ~/.ssh、/etc、仓库外路径，或在没有看文件的情况下覆盖代码。",
            ]
        )

    if "plan_tasks" in tools:
        add(
            [
                "### plan_tasks",
                "用途：Orchestrator 用结构化计划拆分子任务；执行顺序只认 dependsOn 字段。",
                '正确案例：实现依赖设计时，t2.dependsOn=["t1"]，不要只在 task 文本里写“基于 t1”。',
                "字段名必须是 agentId、dependsOn、expectedOutputs、acceptanceCriteria、taskKind、targetPaths、expectedWorkspaceChanges、requiredCommands、requiredEvidence；不要写 snake_case。",
                "文字型审查/诊断任务不要声明 expectedOutputs；把完成条件写进 acceptanceCriteria。",
                '代码任务正确案例：{ taskKind: "code", expectedOutputs: [{ id: "project", type: "project", required: true }], targetPaths: ["frontend/"], acceptanceCriteria: ["项目构建/编译验证通过"], requiredCommands: [{ command: "pnpm build", cwd: "frontend", timeoutMs: 300000 }], requiredEvidence: ["至少一条构建/编译/测试/类型检查命令 exitCode=0"] }。',
            ]
        )

    if REPORT_TASK_RESULT_TOOL_NAME in tools:
        add(
            [
                "### report_task_result",
                "用途：被 Orchestrator 分派的子任务结束前必须调用一次，报告真实语义结果。",
                '正确案例：report_task_result({ status: "complete", summary: "已实现并通过类型检查", filesChanged: [{ path: "src/foo.ts", action: "modified" }], commandsRun: [{ command: "pnpm test src/foo.test.ts", exitCode: 0 }], acceptanceResults: [{ criterion: "通过 typecheck", passed: true, evidence: "pnpm typecheck exited 0" }] })。',
                "字段名必须是 acceptanceResults、filesChanged、commandsRun、tests；不要写 snake_case。",
                "错误案例：代码部分完成、测试失败、或缺少依赖时仍上报 complete；应使用 failed 或 blocked 并说明原因。",
            ]
        )

    return "\n\n".join(sections)


# ─── Orchestrator：plan / aggregate 阶段 system prompt ─────


def build_orchestrator_plan_prompt(
    base_system_prompt: str,
    other_agents: list[Any],
    workspace_mode: str,
) -> str:
    """plan 阶段 system prompt：工作流 + 可用 Agent 清单 + 拆解/依赖规则。

    other_agents 的元素需要有 id / name / capabilities / tool_names / description 属性。
    本地 workspace 规则会**出现两次**（拆解原则中段与末尾各一次）——这是刻意的强调。
    """
    agent_list = "\n".join(
        "- { "
        f"id: {_quote(a.id)}, name: {_quote(a.name)}, "
        f"capabilities: {_quote(a.capabilities)}, "
        f"tools: {_quote(a.tool_names)}, "
        f"description: {_quote(a.description)}"
        " }"
        for a in other_agents
    )
    local_workspace_rules = (
        [
            "",
            "## 本地 workspace 规划规则",
            "- 用户要求在当前文件夹创建 / 修改 / 初始化 / 调试前后端项目或源码文件时，优先派给具备 fs_read / fs_write / bash 或 SDK 本地工具的 agent。",
            "- 这类本地代码任务不要声明 expectedOutputs；用 acceptanceCriteria 描述应落盘的目录、文件、命令和验证结果。",
            "- 子任务文本必须明确写出“直接修改当前本地 workspace 文件，不要用 write_artifact 代替源码落盘”。",
            "- 只有需要聊天内交付的独立文档、设计稿、可预览原型或 artifact handoff，才声明 expectedOutputs。",
        ]
        if workspace_mode == "local"
        else []
    )

    lines = [
        base_system_prompt,
        "",
        "## 你的工作流",
        "1. 阅读用户最新请求与上下文。",
        "2. 如果存在会阻塞正确规划的关键歧义，且能归纳为 2-4 个清晰选项，先调用 ask_user 让用户选择；拿到答案后继续。",
        "3. 调用 plan_tasks 工具，输出结构化 plan。",
        "4. 系统会自动执行 plan 并把子任务结果回传给你，由你做最终总结。",
        "",
        "## 可用 Agent",
        agent_list or "（无）",
        "",
        "## 拆解原则",
        "- 充分利用每个 Agent 的 capabilities，不要把任务派给不合适的人。",
        "- 每个子任务必须独立可执行（被分派的 Agent 看不到完整群聊上下文，必要上下文要写进 task）。",
        "- 计划阶段只能调用 ask_user、plan_tasks 和只读侦察工具（fs_list/fs_read/read_artifact/read_attachment）；不要写文件或执行命令。",
        "- 若用户需求已足够明确，不要为了形式感提问，直接 plan_tasks。",
        "",
        "## 依赖关系（执行顺序的唯一来源，务必读完）",
        "- 系统【只】按每个任务的 dependsOn 决定顺序：dependsOn 为空的任务会【同时并发】启动。",
        "- 若任务 B 需要任务 A 的产物 / 结论 / 输出，你【必须】在 B 的 dependsOn 里写上 A 的 id。",
        "- 在 task 文本里写「先做 A」「基于上一步」之类【没有任何效果】——执行顺序只认 dependsOn 字段。",
        "- 只有彼此真正无关、可同时进行的任务才留空 dependsOn；拿不准时倾向加依赖（串行更安全）。",
        '- Code implementation tasks MUST set taskKind="code", declare expectedOutputs:[{ id:"project", type:"project", required:true }], and include an acceptanceCriteria item requiring build/compile/test/typecheck to pass.',
        "- project expectedOutputs are system-created from workspace file writes; do not ask the child agent to call write_artifact for project.",
        "- Only declare non-project expectedOutputs when the assigned agent must create a real artifact via write_artifact for downstream handoff or user inspection.",
        "- Do NOT declare expectedOutputs for text-only tasks such as review, validation, diagnosis, status check, explanation, or summary; put their completion checks in acceptanceCriteria.",
        "- If a task needs an upstream artifact, declare inputs with fromTaskId and outputId; the system will compile these into dependencies.",
        "- For tasks with quality requirements, add concise acceptanceCriteria that the assigned agent can verify.",
        *local_workspace_rules,
        "- For code or test tasks, set taskKind and declare targetPaths, expectedWorkspaceChanges, requiredCommands, and requiredEvidence whenever possible.",
        "- Frontend and backend implementation tasks usually both depend on PRD/API contracts, not on each other; plan them as parallel siblings unless one truly consumes the other output.",
        '- Prefer requiredCommands with cwd, for example { command: "pnpm build", cwd: "frontend", timeoutMs: 300000 }; avoid encoding directory changes as "cd frontend && ...".',
        "- A retry/remediation plan must preserve the original user goal. Do not replace implementation work with a narrower review-only task unless the user explicitly approved that scope change.",
        *local_workspace_rules,
        "",
        "示例（设计 → 前端 → 审查，逐级依赖；agentId 用上面可用列表里的真实 id）：",
        "tasks: [",
        '  { "id": "t1", "agentId": "<设计师 id>", "task": "产出 UI 设计稿" },',
        '  { "id": "t2", "agentId": "<前端 id>", "task": "按设计稿实现页面", "dependsOn": ["t1"] },',
        '  { "id": "t3", "agentId": "<Reviewer id>", "task": "审查 t2 的实现", "dependsOn": ["t2"] }',
        "]",
    ]
    return "\n".join(lines)


def build_orchestrator_aggregate_prompt(base_system_prompt: str) -> str:
    return "\n".join(
        [
            base_system_prompt,
            "",
            "## 当前阶段",
            "你处于「聚合阶段」。所有子任务已执行完成（含成功与失败），结果在 user 消息中以 XML 给出。",
            "请直接给用户输出一条总结消息：",
            "- 简明列出完成 / 失败的任务",
            "- 如果存在 failed / skipped / aborted 任务，必须明确说明整体未完成，不要把局部成功说成全部完成",
            '- 用 <artifact_ref id="art_xxx"/> 形式引用关键产物（如果有）',
            "- 给出明确的下一步建议",
            "不要再调用 plan_tasks，不要把任务再次分派。",
        ]
    )


# ─── 子 Agent prompt ───────────────────────────────────────


@dataclass
class SubAgentPromptData:
    """build_sub_agent_prompt 的全部入参（DB 查询由调用方做完后填进来）。"""

    task: DispatchPlanItem
    resolved_inputs: list[ResolvedTaskInput]
    workspace_mode: str
    upstream_artifacts: list[Any] = field(default_factory=list)
    existing_artifacts: list[Any] = field(default_factory=list)
    summary_block: str | None = None
    recent_messages: list[Any] = field(default_factory=list)
    pinned_messages: list[Any] = field(default_factory=list)
    agent_name_by_id: dict[str, str] = field(default_factory=dict)


def build_sub_agent_prompt(data: SubAgentPromptData) -> str:
    task = data.task
    upstream_artifacts_xml = "\n".join(
        render_artifact_summary_xml(artifact) for artifact in data.upstream_artifacts
    )
    existing_xml = "\n".join(
        render_artifact_summary_xml(artifact) for artifact in data.existing_artifacts
    )
    recent_xml = render_context_messages_xml(data.recent_messages, data.agent_name_by_id)
    pinned_xml = render_context_messages_xml(data.pinned_messages, data.agent_name_by_id)
    summary_xml = (
        "\n".join(f"  {line}" for line in data.summary_block.split("\n"))
        if data.summary_block
        else ""
    )
    task_inputs_xml = render_task_inputs_xml(data.resolved_inputs)
    expected_outputs_xml = render_expected_outputs_xml(task.expectedOutputs or [])
    acceptance_criteria_xml = render_acceptance_criteria_xml(task.acceptanceCriteria or [])
    evidence_contract_xml = render_task_evidence_contract_xml(task)

    sections = [
        "<context>",
        summary_xml,
        f"  <recent_conversation>\n{recent_xml}\n  </recent_conversation>" if recent_xml else None,
        f"  <pinned_messages>\n{pinned_xml}\n  </pinned_messages>" if pinned_xml else None,
        f"  <required_inputs>\n{task_inputs_xml}\n  </required_inputs>" if task_inputs_xml else None,
        f"  <expected_outputs>\n{expected_outputs_xml}\n  </expected_outputs>"
        if expected_outputs_xml
        else None,
        f"  <acceptance_criteria>\n{acceptance_criteria_xml}\n  </acceptance_criteria>"
        if acceptance_criteria_xml
        else None,
        f"  <evidence_contract>\n{evidence_contract_xml}\n  </evidence_contract>"
        if evidence_contract_xml
        else None,
        f"  <upstream_artifacts>\n{upstream_artifacts_xml}\n  </upstream_artifacts>"
        if upstream_artifacts_xml
        else None,
        f"  <existing_artifacts>\n{existing_xml or '    （无）'}\n  </existing_artifacts>",
        "</context>",
        "",
        "<your_task>",
        task.task,
        "</your_task>",
        "",
        "Before working, read every required input artifact with read_artifact(artifactId).",
        (
            "If this task is about local project source files, directly modify the current "
            "local workspace with file/command tools. Do not use write_artifact to store source "
            "files that should be written to disk."
            if data.workspace_mode == "local"
            else None
        ),
        'For expected_outputs with type="project", write the project files into the workspace with fs_write or bash; AgentHub will create and bind the project artifact automatically. Do not call write_artifact for project.',
        "For non-project expected_outputs, create the artifact with write_artifact and pass outputKey equal to that output id.",
        "If no expected_outputs are declared, complete the task with a normal message; do not create an artifact just to satisfy status tracking.",
        "Satisfy every acceptance_criteria item when present.",
        "If evidence_contract is present, include matching filesChanged, commandsRun, tests, and/or acceptanceResults evidence in report_task_result.",
        "For required_commands, you may run the command yourself, and AgentHub will also run it as a completion gate after your attempt. Use bash cwd instead of cd when running commands in subdirectories.",
        "If dependencies are missing, install them inside the workspace and continue; dependency installation is preparation, not completion evidence.",
        "If a required command fails, fix the issue and continue; do not report complete until the command can pass.",
        "For target_paths, list every changed or verified path in report_task_result.filesChanged.",
        "At the end, call report_task_result exactly once. A normal text response alone does not complete this dispatched task.",
        'Use report_task_result.status="complete" only when you have FULLY accomplished the assigned task.',
        "Never report complete if tests are failing, implementation is partial, unresolved errors remain, or you could not find necessary files/dependencies.",
        "If acceptance_criteria are present, include acceptanceResults and copy each criterion string exactly with passed/evidence.",
        'Use status="failed" when the task was attempted but did not satisfy the assignment; use status="blocked" when external input or unavailable prerequisites prevent progress.',
        "",
        "执行任务，必要时通过 read_artifact 获取上游产物详情。",
    ]
    return "\n".join(line for line in sections if line)


def render_context_messages_xml(
    messages: list[Any], agent_name_by_id: dict[str, str]
) -> str:
    """把消息行渲染成 `    <message from="...">...</message>`；空文本行剔除。"""
    lines: list[str] = []
    for message in messages:
        if message.role == "user":
            from_name = "user"
        else:
            from_name = agent_name_by_id.get(message.agent_id or "") or message.role
        text = extract_text_from_parts(message.parts).strip()
        if not text:
            continue
        lines.append(f"    <message from={_quote(from_name)}>{escape_xml(text)}</message>")
    return "\n".join(lines)


def render_task_inputs_xml(inputs: list[ResolvedTaskInput]) -> str:
    lines: list[str] = []
    for entry in inputs:
        input_ = entry.input
        attrs = [
            f"fromTaskId={xml_attr(input_.fromTaskId)}",
            f"outputId={xml_attr(input_.outputId)}",
            f'required={xml_attr("false" if input_.required is False else "true")}',
        ]
        if entry.type:
            attrs.append(f"type={xml_attr(entry.type)}")
        if entry.artifact_id:
            attrs.append(f"artifactId={xml_attr(entry.artifact_id)}")
        if entry.missing:
            attrs.append('missing="true"')
        attr_text = " ".join(attrs)
        description = escape_xml(input_.description) if input_.description else ""
        if description:
            lines.append(f"    <input {attr_text}>{description}</input>")
        else:
            lines.append(f"    <input {attr_text} />")
    return "\n".join(lines)


def render_expected_outputs_xml(outputs: list[DispatchExpectedOutput]) -> str:
    lines: list[str] = []
    for output in outputs:
        attrs = " ".join(
            [
                f"id={xml_attr(output.id)}",
                f"type={xml_attr(output.type)}",
                f'required={xml_attr("false" if output.required is False else "true")}',
            ]
        )
        description = escape_xml(output.description) if output.description else ""
        if description:
            lines.append(f"    <output {attrs}>{description}</output>")
        else:
            lines.append(f"    <output {attrs} />")
    return "\n".join(lines)


def render_acceptance_criteria_xml(criteria: list[str]) -> str:
    return "\n".join(f"    <item>{escape_xml(item)}</item>" for item in criteria)


def render_task_evidence_contract_xml(task: DispatchPlanItem) -> str:
    lines: list[str] = []
    if task.taskKind:
        lines.append(f"    <task_kind>{escape_xml(task.taskKind)}</task_kind>")
    for target_path in task.targetPaths or []:
        lines.append(f"    <target_path>{escape_xml(target_path)}</target_path>")
    for change in task.expectedWorkspaceChanges or []:
        lines.append(
            f"    <expected_workspace_change>{escape_xml(change)}</expected_workspace_change>"
        )
    for required_command in task.requiredCommands or []:
        attrs = [f"command={xml_attr(required_command.command)}"]
        if required_command.cwd:
            attrs.append(f"cwd={xml_attr(required_command.cwd)}")
        if required_command.timeoutMs:
            attrs.append(f"timeoutMs={xml_attr(str(required_command.timeoutMs))}")
        attr_text = " ".join(attrs)
        description = escape_xml(required_command.description) if required_command.description else ""
        if description:
            lines.append(f"    <required_command {attr_text}>{description}</required_command>")
        else:
            lines.append(f"    <required_command {attr_text} />")
    for evidence in task.requiredEvidence or []:
        lines.append(f"    <required_evidence>{escape_xml(evidence)}</required_evidence>")
    return "\n".join(lines)


def render_artifact_summary_xml(artifact: Any) -> str:
    return (
        f'  <artifact id="{artifact.id}" type="{artifact.type}" '
        f"title={_quote(artifact.title)} />"
    )


# ─── 聚合阶段 user prompt ──────────────────────────────────


def build_aggregate_prompt(
    original_user_prompt: str,
    plan: list[DispatchPlanItem],
    task_results: dict[str, Any],
    conflicts: list[FileWriteConflict],
    workspace_cwd: str,
    artifact_by_id: dict[str, Any],
) -> str:
    results_xml_lines: list[str] = []
    for item in plan:
        result = task_results.get(item.id)
        if result is None:
            continue
        output_key_by_artifact_id = {
            artifact_id: output_key
            for output_key, artifact_id in result.output_artifacts.items()
        }
        artifact_lines: list[str] = []
        for artifact_id in result.artifact_ids:
            artifact = artifact_by_id.get(artifact_id)
            if artifact is None:
                continue
            output_key = output_key_by_artifact_id.get(artifact_id)
            output_attr = f" outputKey={_quote(output_key)}" if output_key else ""
            artifact_lines.append(
                f'    <artifact id="{artifact.id}" type="{artifact.type}"'
                f"{output_attr} title={_quote(artifact.title)} />"
            )
        arts = "\n".join(artifact_lines)
        report = render_task_result_report_xml(result.task_report) if result.task_report else ""
        inner_content = "\n".join(part for part in (report, arts) if part)
        inner = f"\n{inner_content}\n  " if inner_content else ""
        err_attr = f" error={_quote(result.error)}" if result.error else ""
        results_xml_lines.append(
            f'  <result task="{item.id}" agent="{item.agentId}" '
            f'status="{result.status}"{err_attr}>{inner}</result>'
        )
    results_xml = "\n".join(results_xml_lines)

    lines = [
        f"<user_request>{original_user_prompt}</user_request>",
        "<task_results>",
        results_xml,
        "</task_results>",
    ]

    if conflicts:
        def to_rel(abs_path: str) -> str:
            if abs_path.startswith(workspace_cwd):
                return abs_path[len(workspace_cwd) :].lstrip("\\/")
            return abs_path

        lines.append("<file_conflicts>")
        lines.append(
            "  <!-- 多个并行子任务写了同一文件，后写已覆盖先写。请在总结里明确告知用户："
            "哪个文件、涉及哪些任务、当前保留的是最后写入的版本，并建议如何处理"
            "（例如改为串行重做或人工合并）。 -->"
        )
        for conflict in conflicts:
            tasks = ", ".join(
                f"{writer.taskId}({writer.agentId})" for writer in conflict.contributors
            )
            lines.append(
                f"  <conflict path={_quote(to_rel(conflict.path))} tasks={_quote(tasks)} />"
            )
        lines.append("</file_conflicts>")

    lines.append("")
    lines.append("请基于以上结果给用户输出最终总结消息。")
    return "\n".join(lines)


def render_task_result_report_xml(report: TaskResultReport) -> str:
    children = [f"      <summary>{escape_xml(report.summary)}</summary>"]
    for result in report.acceptanceResults or []:
        children.append(
            f"      <acceptance criterion={xml_attr(result.criterion)} "
            f'passed={xml_attr(str(result.passed).lower())}>{escape_xml(result.evidence)}</acceptance>'
        )
    for file_ in report.filesChanged or []:
        action_attr = f" action={xml_attr(file_.action)}" if file_.action else ""
        children.append(f"      <file path={xml_attr(file_.path)}{action_attr} />")
    for command in report.commandsRun or []:
        attrs = [f"command={xml_attr(command.command)}"]
        exit_text = "null" if command.exitCode is None else str(command.exitCode)
        attrs.append(f"exitCode={xml_attr(exit_text)}")
        if command.cwd:
            attrs.append(f"cwd={xml_attr(command.cwd)}")
        if command.timedOut is not None:
            attrs.append(f'timedOut={xml_attr(str(command.timedOut).lower())}')
        attr_text = " ".join(attrs)
        summary = escape_xml(command.summary) if command.summary else ""
        if summary:
            children.append(f"      <command {attr_text}>{summary}</command>")
        else:
            children.append(f"      <command {attr_text} />")
    for test in report.tests or []:
        summary = escape_xml(test.summary) if test.summary else ""
        attrs = (
            f"command={xml_attr(test.command)} "
            f'passed={xml_attr(str(test.passed).lower())}'
        )
        if summary:
            children.append(f"      <test {attrs}>{summary}</test>")
        else:
            children.append(f"      <test {attrs} />")
    for blocker in report.blockers or []:
        children.append(f"      <blocker>{escape_xml(blocker)}</blocker>")
    return "\n".join(
        [
            f'    <task_report status={xml_attr(report.status)}>',
            *children,
            "    </task_report>",
        ]
    )
