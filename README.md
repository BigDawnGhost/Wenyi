# trans-novel —— 多 Agent 协同长篇小说翻译（日/英 → 中）

一套以"媲美人类翻译、尽量减少漏译/出错、靠专有名词对照表保证一致性"为目标的流水线。
多个职责单一的 Agent 协同，模拟出版社流程：**全书理解预扫 → 分析 → 翻译 → 审校 → 标点规范化 → 术语 AI 审计统一 → 跨章一致性把关**，全程围绕一个持久化的术语库。**一条 `translate` 命令连续跑完并直接产出 EPUB。**

- **语言方向**：日/英 → 中，**AI 自动检测来源语言**（`config.yaml` 设 `language.source: auto`，也可写死 `ja`/`en`）。提示词按源语言切换（日语：敬称/假名读音/第一人称语气；英语：性别推断/从句重组/称谓处理）。
- **模型**：DeepSeek 双档，经 OpenAI SDK 调 `https://api.deepseek.com`，两档都开 thinking 模式、`reasoning_effort=high`（成本差异只靠模型区分）：
  - `strong` = `deepseek-v4-pro` → 翻译 / 全局分析 / 标题翻译 / 术语审计 /（可选）润色
  - `cheap`  = `deepseek-v4-flash` → 审校 / 一致性 QA / 回译比对（判断类，开 thinking）
  - `fast`   = `deepseek-v4-flash` → 梗概 / 全书概览 / 术语抽取 / 回译（机械任务，免思考大幅省钱提速）- **输入**：EPUB、FB2 与 纯文本（TXT/Markdown）。**输出默认 EPUB**（EPUB 输入按原模板回填保留排版，并改写书名/目录为译名；TXT 输入用 ebooklib 生成规范 EPUB3）；`--format txt` 可导出纯文本。
- **全书理解**：翻译前用廉价档**预扫整本源文**，产出"逐章梗概 + 全书概览"，作为恒定前缀注入每章翻译——让译者翻任意章前就把握主线/人物/伏笔，不盲译早章。
- **省成本（提示词缓存）**：system 全静态、术语表/全书概览等恒定块前置，命中 DeepSeek 自动前缀缓存（命中输入价≈0.1×）；章内批次**串行**逐批刷新上下文以保连贯。
- **标点**：译文统一为简体中文大陆通用全角标点（“”‘’、，。！？、……、——）。

## 安装

```bash
uv sync                          # 用 uv 安装依赖（pydantic / typer / rich / tenacity / ebooklib / lxml / openai …）
export DEEPSEEK_API_KEY=sk-...   # DeepSeek API key（运行真实翻译时需要）
```

> 仅离线跑切分/对齐/术语库/状态机等逻辑（不发网络请求）时，把 `config.yaml` 的
> `llm.provider` 设为 `fake` 即可。

## 使用

日常只需 `translate` 一个命令；细粒度/调试工具收敛到 `tools` 子命令。

```bash
# 连续全流程
uv run trans-novel translate book.epub    # 预扫→分析→翻译→审校→标点→术语审计→QA→出 EPUB（断点可续）
uv run trans-novel resume    book.epub    # 中断后续跑，跳过已完成章/批
uv run trans-novel status    book.epub    # 查看各章进度与术语库统计

# 高级/调试工具（收在 tools 下）
uv run trans-novel tools glossary book.epub list      # 查看术语表
uv run trans-novel tools glossary book.epub conflicts # 待裁决的译法冲突
uv run trans-novel tools glossary book.epub audit     # AI 审计统一译法并改写正文
uv run trans-novel tools qa        book.epub          # 全书跨章一致性扫描
uv run trans-novel tools report    book.epub          # 生成 QA 报告（漏译/冲突/低置信度汇总）
uv run trans-novel tools assemble  book.epub          # 回填生成译文 EPUB（--format txt 出纯文本）
```

`translate` 自带段级进度条（不止于章），长文也能看清进度。
开关：`--polish`（默认关，开启=用强档把全书再加工一遍，较烧 token）/ `--no-audit` / `--no-qa`；调试单章：`translate book.epub --chapter 0`。

> 自动改写边界：**术语审计**会自动改写正文（消除译法漂移）；**一致性 QA / 报告**只汇总不改正文——需要的改动通过 `tools glossary` 的编辑/裁决来落地。

### 连续流程（默认）

`translate` 一步到位：先**预扫整本源文**（fast 免思考并行）建立全书理解（逐章梗概+全书概览）+ 强档多点分析样章建立术语表 → 章内批次**串行**翻译（逐批刷新上下文、跨章串行保连贯）→ 标点规范化 → **章末整章审校 + 严重项自动重译**（autofix_severe，过长度校验才采纳）→ 术语 AI 审计统一（消除如 佳穂/佳穗 的译法漂移并改写正文）→ 跨章一致性 QA → 写报告 → 回填出 EPUB（书名/目录用译名）。

## 一致性 / 防漏译机制

- **句段对齐强制**：翻译按批输入 N 段、要求输出 N 段 JSON 数组；段数不符则重试，
  仍不符则逐段兜底翻译，从结构上杜绝整段漏译。
- **专有名词对照表（SQLite）**：人名/地名/术语/敬称统一译法，含读音、性别、别名、
  置信度、锁定位；每章增量抽取、冲突裁决，翻译时注入整章全量表（恒定块命中缓存）。
- **全书理解 + 滚动上下文**：预扫源文得"全书概览/本章梗概"（恒定前缀），加最近译文尾段（局部衔接），
  共同保证跨批次/跨章连贯、代词指代与对全书走向的把握。
- **校验 + 定向修复**：章末整章分块审校（漏译/误译/术语/人称，不阻塞翻译主路径）；
  严重项（missing/mistranslation）带审校意见定向重译一次，过长度校验才采纳（`autofix_severe` 开关，默认开）。
  无成本长度校验留在批内实时标记；回译抽检可选；全书跨章一致性扫描。润色可选（默认关）。
- **断点续跑（章/批级）**：每批译完即增量落盘，中断（含 Ctrl+C）后再次运行 `translate`/`resume`
  跳过已完成的批，只补未完成的。

## 配置（`config.yaml`）

模型 ID、三档 tier effort、流水线开关（审校 / 自动重译 / 润色 / 回译比例 / 一致性 / `book_understanding` 全书理解预扫 / 术语裁剪 / 预扫并行）、
敬称策略、切分粒度（`max_chars_per_batch` 批大小、`max_chars_per_segment` 超长段按句拆分）都在这里改。

> 若 `deepseek-v4-flash` / `deepseek-v4-pro` 模型 ID 与官方不符，直接在 `config.yaml` 的 `llm.tiers` 改即可。

## 目录

```
trans_novel/
  ingest/      摄取与切分（EPUB/FB2/TXT → Chapter/Segment）+ 语言检测 + 超长段拆分
  llm/         LLM 抽象接口 + DeepSeek provider + 离线 FakeClient
  glossary/    术语库(SQLite) + 抽取 + 冲突裁决
  agents/      base(Agent 基类) / analyzer / synopsis(全书理解) / translator / reviewer / polisher / consistency / glossary_auditor + 提示词
  pipeline/    orchestrator(状态机/续跑) / context(滚动上下文) / checks(对齐校验) / runstore
  postprocess/ 标点规范化
  assemble/    回填(EPUB/TXT，书名/目录译名) + QA 报告
prompts/       提示词覆盖（可选，见其 README）
tests/         离线测试（不发网络请求）
```

## 实现取舍

- LLM 层是**可插拔接口**：要换平台只需实现 `LLMClient`（见 `llm/base.py`），其余不动。
- 建模用 `pydantic`，LLM 重试用 `tenacity`，CLI/进度用 `typer`+`rich`，TXT→EPUB 用 `ebooklib`。
- EPUB 输入回填仍走 zip 原样拷贝 + 锚点替换，最大程度保留原排版/资源。
- 章内批次**串行**：逐批把刚译出的译文并入上下文供下一批参照，换取代词/术语/语气的跨批连贯；
  靠提示词缓存（恒定前缀）而非并发来控成本与时延。

## 测试

```bash
uv run python -m unittest discover -s tests
```
