"""编排器：驱动全流程，章级状态机 + 断点续跑。

单章流水线（章内批次**串行**，逐批刷新滚动上下文；跨章亦串行传递梗概）：
  每批：渲染上下文（含前一批刚译出的译文）→ 翻译（对齐保证）→ 长度校验（零成本）→
        润色（可选）→ 标点规范化 → 立即把本批译文并入滚动上下文（供下一批参照，保证连贯）。
  章末（串行）：整章分块审校（不阻塞翻译主路径）→ 严重项定向重译（autofix_severe，
        过长度校验才采纳）→ 回译抽检 → 术语抽取入库 → 写 TM → 落盘标记 done。
翻译前先预扫源文建立全书理解（逐章梗概+全书概览，fast 档并行），作恒定前缀注入每章翻译。

run_all：在翻译全书后接 术语 AI 审计统一 → 一致性 QA → 写报告 → 回填出 EPUB，一气呵成。
进度回调 progress(done_segments, total_segments, label) 与 UI 无关，每批完成即触发。
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..config import Config
from ..glossary.extractor import GlossaryExtractor
from ..glossary.store import GlossaryStore, TYPE_PERSON
from ..llm.base import LLMClient, build_client
from ..ingest.segmenter import load_document, batch_segments
from ..postprocess.punct import normalize_zh
from ..agents.analyzer import Analyzer
from ..agents.synopsis import Synopsizer
from ..agents.translator import Translator
from ..agents.reviewer import Reviewer, BackTranslator
from ..agents.polisher import Polisher
from . import checks
from .context import RollingContext
from .runstore import RunStore, slugify, STATUS_DONE

ProgressFn = Callable[[int, int, str], None]


# 语言名/代码 → ISO 639-1 两字母代码（AI 检测结果归一化）
_LANG_ALIASES = {
    "japanese": "ja", "日语": "ja", "日文": "ja", "jp": "ja", "jpn": "ja",
    "english": "en", "英语": "en", "英文": "en", "eng": "en",
    "russian": "ru", "俄语": "ru", "俄文": "ru", "rus": "ru",
    "chinese": "zh", "中文": "zh", "汉语": "zh", "zh-cn": "zh", "zho": "zh",
    "korean": "ko", "韩语": "ko", "韩文": "ko", "kor": "ko",
    "french": "fr", "法语": "fr", "法文": "fr",
    "german": "de", "德语": "de", "德文": "de",
    "spanish": "es", "西班牙语": "es", "西班牙文": "es",
}


def _normalize_lang(code: str) -> str:
    c = (code or "").strip().lower()
    if not c:
        return ""
    if c in _LANG_ALIASES:
        return _LANG_ALIASES[c]
    return c[:2] if c[:2].isalpha() else ""


@dataclass
class _BatchResult:
    targets: list[str]
    issues: list[dict] = field(default_factory=list)
    bt_samples: list[tuple[str, str]] = field(default_factory=list)


class Orchestrator:
    def __init__(self, config: Config, client: LLMClient | None = None):
        self.config = config
        self.client = client or build_client(config)
        self.analyzer = Analyzer(self.client, config)
        self.synopsizer = Synopsizer(self.client, config)
        self.translator = Translator(self.client, config)
        self.reviewer = Reviewer(self.client, config)
        self.backtrans = BackTranslator(self.client, config)
        self.polisher = Polisher(self.client, config)
        self.extractor = GlossaryExtractor(self.client, config)

    # ── 语言解析 ────────────────────────────────────────────────────────────
    def _apply_language(self, lang: str) -> None:
        """把解析出的源语言应用到 config 与各 agent（auto 检测后调用）。"""
        resolved = lang or self.config.source_lang
        self.config.source_lang = resolved
        for ag in (self.analyzer, self.synopsizer, self.translator, self.reviewer,
                   self.backtrans, self.polisher, self.extractor):
            ag.src = resolved

    # ── 准备 / 续跑入口 ──────────────────────────────────────────────────
    def prepare(self, input_path: str) -> RunStore:
        # 超长段按句拆分（max_chars_per_segment），续段标 cont 供回填并回
        doc = load_document(input_path, self.config.source_lang, self.config.target_lang,
                            split_segments=self.config.segment.max_chars_per_segment)
        run_dir = os.path.join(self.config.state_dir, slugify(doc.title))
        store = RunStore(run_dir)
        if store.exists():
            return store  # 已有进度 → 直接续跑，不重置（语言在 run() 里按 manifest 应用）

        # 新建：auto 时用 AI 检测主要语言（失败回退启发式结果）
        if self.config.source_lang in ("auto", "", None):
            doc.source_lang = self._detect_language_ai(doc) or doc.source_lang
        self._apply_language(doc.source_lang)

        store.init_from_document(doc)
        glossary = GlossaryStore(store.glossary_path)
        sample = self._sample_text(doc)
        analysis = self.analyzer.analyze(sample) if sample else {}
        if analysis:
            self.analyzer.seed_glossary(glossary, analysis)
        store.save_analysis(analysis)
        glossary.close()
        store.save_context(RollingContext().to_dict())
        return store

    def _detect_language_ai(self, doc) -> str:
        """用 LLM 检测正文主要语言，返回 ISO 代码（如 ja/en/ru）。失败返回空串。"""
        # labeled=False：纯源文样本，防多点采样的中文标签污染语言检测
        sample = self._sample_text(doc, labeled=False)[:1500]
        if not sample.strip():
            return ""
        system = (
            "你是语言识别器。判断给定文本的主要自然语言，"
            '仅输出 JSON：{"language":"<ISO 639-1 两字母代码，如 ja/en/ru/ko/fr/de/zh>"}。'
        )
        try:
            data = self.client.complete_json(
                [{"role": "system", "content": system},
                 {"role": "user", "content": sample}], tier="cheap")
            code = (data.get("language") if isinstance(data, dict) else "") or ""
            return _normalize_lang(str(code))
        except Exception:
            return ""

    @staticmethod
    def _sample_text(doc, *, labeled: bool = True) -> str:
        """取风格分析样章。labeled=True 时多点采样（开头/中部/结尾各一段，带中文标注），
        让分析覆盖全书风格全貌；labeled=False 返回单段纯源文（语言检测用，不能混入中文标签）。"""
        texts = ["\n".join(s.source for s in ch.text_segments) for ch in doc.chapters]
        texts = [t for t in texts if len(t) > 200]
        if not texts:  # 兜底：全书都是短章
            joined = "\n".join(
                s.source for ch in doc.chapters[:2] for s in ch.text_segments)
            return joined[:6000]
        if not labeled:
            return texts[0][:6000]
        picks = [(0, "开头样章"), (len(texts) // 2, "中部样章"), (len(texts) - 1, "结尾样章")]
        parts: list[str] = []
        seen: set[int] = set()
        for idx, tag in picks:
            if idx in seen:  # 短书（1-2 章）去重，不重复取同一章
                continue
            seen.add(idx)
            t = texts[idx]
            chunk = t[-2800:] if tag == "结尾样章" else t[:2800]
            parts.append(f"【{tag}】\n{chunk}")
        return "\n\n".join(parts)

    def run(self, input_path: str, *, only_chapter: int | None = None,
            progress: Optional[ProgressFn] = None) -> RunStore:
        store = self.prepare(input_path)
        manifest = store.load_manifest()
        self._apply_language(manifest.get("source_lang") or self.config.source_lang)
        glossary = GlossaryStore(store.glossary_path)
        context = RollingContext.from_dict(store.load_context() or {})
        style = self.analyzer.style_brief(store.load_analysis() or {})
        # 翻译前预扫源文，建立全书理解（幂等、可续跑）；全书概览注入每章翻译
        book_synopsis = self._build_understanding(store)

        if only_chapter is not None:
            targets = [only_chapter]
        else:
            targets = store.pending_chapters()

        total = self._count_segments(store, targets)
        done = 0
        try:
            for ci in targets:
                done = self._translate_chapter(
                    ci, store, glossary, context, style, book_synopsis,
                    progress=progress, done=done, total=total)
                store.save_context(context.to_dict())
            # 全书译完后翻译书名与各章标题（供目录/文件名使用，借术语表保持专名一致）
            if not store.pending_chapters():
                self._translate_titles(store, glossary)
        finally:
            glossary.close()
        if progress and total:
            progress(total, total, "翻译完成")
        return store

    @staticmethod
    def _count_segments(store: RunStore, chapter_indices: list[int]) -> int:
        total = 0
        for ci in chapter_indices:
            total += len(store.load_chapter(ci).text_segments)
        return total

    # ── 全书理解预扫（源文逐章梗概 + 全书概览）────────────────────────────────
    def _build_understanding(self, store: RunStore) -> str:
        """翻译前预扫源文：逐章梗概存入 chapter.meta，归并出全书概览存入 analysis。

        幂等、可续跑：已有梗概/概览则跳过。返回全书概览（注入各章翻译 prompt）。
        关闭 book_understanding 时直接返回空串。
        """
        if not self.config.pipeline.book_understanding:
            return ""
        manifest = store.load_manifest()
        chapters = manifest.get("chapters", [])

        # 各章梗概相互独立 → 并行调用（LLM 调用进线程池；落盘全部在主线程，
        # 保持原子写不竞争，且逐章增量落盘、续跑粒度不变）。已有梗概的章跳过（幂等）。
        loaded = {c.get("index", i): store.load_chapter(c.get("index", i))
                  for i, c in enumerate(chapters)}
        todo = [(ci, "\n".join(s.source for s in ch.text_segments))
                for ci, ch in loaded.items() if not ch.meta.get("source_digest")]
        if todo:
            workers = max(1, self.config.pipeline.prescan_concurrency)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(self.synopsizer.digest_chapter, src): ci
                        for ci, src in todo}
                for fut in as_completed(futs):
                    ci = futs[fut]
                    loaded[ci].meta["source_digest"] = fut.result()  # 失败时 _ask_text 已回退 ""
                    store.save_chapter(loaded[ci])

        # 按 manifest 章序组装（与并发完成顺序无关）
        digests = [loaded[c.get("index", i)].meta.get("source_digest", "") or ""
                   for i, c in enumerate(chapters)]

        analysis = store.load_analysis() or {}
        synopsis = analysis.get("book_synopsis", "")
        if not synopsis and any(d.strip() for d in digests):
            synopsis = self.synopsizer.book_synopsis(
                digests, self.analyzer.style_brief(analysis))
            analysis["book_synopsis"] = synopsis
            store.save_analysis(analysis)
        return synopsis

    # ── 书名 / 章节标题翻译（目录与输出文件名用）──────────────────────────────
    def _translate_titles(self, store: RunStore, glossary: GlossaryStore) -> None:
        """把书名 + 各章标题整体翻成中文，写回 manifest（幂等：已全部译过则跳过）。

        借术语表保证专名一致；一次调用翻译全部标题，互为上下文更连贯。
        """
        from ..agents import prompts

        m = store.load_manifest()
        chapters = m.get("chapters", [])
        if (m.get("title_translated")
                and all(c.get("title_translated") for c in chapters)):
            return  # 已译，断点续跑不重复调用

        # 标题压成单行，避免内嵌换行破坏 numbered 对齐
        def _flat(s: str) -> str:
            return " ".join((s or "").split())
        titles = [_flat(m.get("title", ""))] + [_flat(c.get("title", "")) for c in chapters]
        if not any(t.strip() for t in titles):
            return
        system = prompts.render("title_translator_system",
                                src=self.config.source_lang, tgt=self.config.target_lang,
                                n=len(titles))
        user = prompts.render("title_translator_user",
                              src=self.config.source_lang, tgt=self.config.target_lang,
                              glossary=prompts.render_glossary(glossary.all_terms()),
                              n=len(titles), numbered_titles=prompts.numbered(titles))
        try:
            data = self.client.complete_json(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}], tier="strong")
        except Exception:
            return
        out = data.get("titles") if isinstance(data, dict) else data
        if not isinstance(out, list) or len(out) != len(titles):
            return
        out = [str(t).strip() for t in out]
        m["title_translated"] = out[0] or m.get("title")
        for c, t in zip(chapters, out[1:]):
            c["title_translated"] = t or c.get("title")
        store.save_manifest(m)

    # ── 单章 ──────────────────────────────────────────────────────────────
    def _translate_chapter(self, ci: int, store: RunStore,
                           glossary: GlossaryStore, context: RollingContext,
                           style: str, book_synopsis: str = "", *,
                           progress: Optional[ProgressFn] = None,
                           done: int = 0, total: int = 0) -> int:
        chapter = store.load_chapter(ci)
        text_segs = chapter.text_segments
        if not text_segs:
            store.set_chapter_status(ci, STATUS_DONE)
            return done
        chapter_digest = chapter.meta.get("source_digest", "")

        batches = batch_segments(text_segs, self.config.segment.max_chars_per_batch)
        label = f"第{ci}章 {chapter.title}"
        # 章内术语快照：整章各批次共用同一份（章内恒定 → system+style+glossary 成为
        # 整章批次共享的稳定前缀，命中 DeepSeek 自动前缀缓存，命中部分输入价≈0.1×）。
        # glossary_scope=chapter 时裁剪为「本章源文实际出现的词条（含别名命中）+ 全部锁定人物」，
        # 长书后期全量表很大，裁剪能显著省去未命中缓存时的输入 token；full 保持全量。
        term_snapshot = glossary.all_terms()
        if self.config.pipeline.glossary_scope == "chapter":
            src_text = "\n".join(s.source for s in text_segs)
            hit = {t.source for t in GlossaryStore.terms_in(term_snapshot, src_text)}
            term_snapshot = [t for t in term_snapshot
                             if t.source in hit or (t.type == TYPE_PERSON and t.locked)]

        # 逐批串行：每批渲染最新上下文 → 处理 → 立即把译文并入上下文供下一批参照。
        # 不再并发，换取章内跨批的代词/术语/语气连贯。
        # 断点续跑（段/批级）：上次中断前已译完并落盘的批次，整批跳过、不重翻，只重建上下文。
        review_issues: list[dict] = list(chapter.meta.get("review_issues", []))
        bt_samples: list[tuple[str, str]] = []
        seg_base = 0   # 当前批首段的章内段号（issue 批内下标 → 章内段号）
        for b in batches:
            if all(s.target and s.target.strip() for s in b):
                # 该批上次已在原位、原上下文中译完 → 复用，重建滚动上下文后跳过
                context.add_targets([s.target for s in b])
                done += len(b)
                seg_base += len(b)
                if progress:
                    progress(done, total, label)
                continue

            ctx_text = context.render(self.config.pipeline.rolling_context_segments)
            # 传章内术语快照：批次间 glossary 块恒定，命中前缀缓存
            res = self._process_batch(b, term_snapshot, ctx_text, style,
                                      book_synopsis, chapter_digest)
            for s, t in zip(b, res.targets):
                s.target = t
            context.add_targets(res.targets)
            for it in res.issues:
                it["chapter"] = ci
                it["index"] += seg_base   # 批内下标 → 章内段号
            review_issues.extend(res.issues)
            bt_samples.extend(res.bt_samples)
            done += len(b)
            seg_base += len(b)
            if progress:
                progress(done, total, label)
            # 增量持久化：本批译文 + 累计问题落盘，下次中断从此批之后续跑
            chapter.meta["review_issues"] = review_issues
            store.save_chapter(chapter)

        # ── 章末整章审校（移出批内关键路径；块内 index 映射回章内段号）──
        # 幂等：续跑重入章末时清掉旧审校项（只留批内长度校验项），防重复累积
        if self.config.pipeline.review:
            review_issues = [i for i in review_issues if i.get("stage") == "length"]
            new_issues = self._review_chapter(text_segs, term_snapshot)
            if self.config.pipeline.autofix_severe:
                self._autofix_severe(text_segs, new_issues, term_snapshot, style,
                                     book_synopsis, chapter_digest)
            for it in new_issues:
                it["chapter"] = ci
                it.setdefault("fixed", False)
                it["stage"] = "review"
            review_issues.extend(new_issues)

        # 回译抽检
        bt_issues: list[dict] = []
        if bt_samples:
            srcs = [a for a, _ in bt_samples]
            tgts = [b for _, b in bt_samples]
            for it in self.backtrans.check(srcs, tgts):
                it["chapter"] = ci
                bt_issues.append(it)

        # 术语抽取入库
        src_text = "\n".join(s.source for s in text_segs)
        tgt_text = "\n".join(s.target or "" for s in text_segs)
        self.extractor.extract_and_store(glossary, src_text, tgt_text, ci)

        # 翻译记忆库（仅作记录/参考，不用于跨位置复用译文）
        for s in text_segs:
            if s.target:
                glossary.add_tm(s.source, s.target, ci)

        chapter.meta["review_issues"] = review_issues
        chapter.meta["backtranslation_issues"] = bt_issues
        store.save_chapter(chapter)
        store.set_chapter_status(ci, STATUS_DONE)
        return done

    _LEN_DETAIL = {
        "empty": "译文为空（疑似漏译）",
        "too_short": "译文明显偏短（疑似漏译）",
        "too_long": "译文明显偏长（疑似增译/失控）",
    }

    # ── 章末审校 + 严重项定向重译 ────────────────────────────────────────────
    _SEVERE_TYPES = ("missing", "mistranslation")

    def _review_chapter(self, text_segs, terms) -> list[dict]:
        """整章分块审校（章末统一做，不在批内阻塞翻译主路径）。

        块 = 连续段序列（约 3 倍翻译批大小，减少调用次数与重复注入的输入 token）；
        块内 reviewer 返回的 index 是块内下标，加块首段偏移映射回章内段号；
        越界 index 直接丢弃（模型幻觉防御）。
        """
        budget = self.config.segment.max_chars_per_batch * 3
        issues: list[dict] = []
        base = 0
        for chunk in self._pack_contiguous(text_segs, budget):
            srcs = [s.source for s in chunk]
            tgts = [s.target or "" for s in chunk]
            for it in self.reviewer.review(srcs, tgts, terms):
                idx = it.get("index")
                if isinstance(idx, int) and 0 <= idx < len(chunk):
                    it["index"] = base + idx
                    issues.append(it)
            base += len(chunk)
        return issues

    @staticmethod
    def _pack_contiguous(segs, budget: int) -> list[list]:
        """按源文字符预算把段保序打包成若干连续块。"""
        chunks: list[list] = []
        cur: list = []
        size = 0
        for s in segs:
            if cur and size + len(s.source) > budget:
                chunks.append(cur)
                cur, size = [], 0
            cur.append(s)
            size += len(s.source)
        if cur:
            chunks.append(cur)
        return chunks

    def _autofix_severe(self, text_segs, issues, terms, style,
                        book_synopsis: str = "", chapter_digest: str = "") -> None:
        """对审校严重项（漏译/误译）带审校意见定向重译，每段最多一次。

        采纳条件 = 重译非空且过长度校验：采纳则标点规范化后更新 seg.target 并标 fixed=True；
        不采纳保持 fixed=False 留人工。章末重译时原滚动上下文已失效，用该段前后各 2 段译文做局部上下文。
        """
        by_seg: dict[int, list[dict]] = {}
        for it in issues:
            if it.get("type") in self._SEVERE_TYPES:
                by_seg.setdefault(it["index"], []).append(it)
        for idx, seg_issues in sorted(by_seg.items()):
            seg = text_segs[idx]
            before = "\n".join(text_segs[j].target or ""
                               for j in range(max(0, idx - 2), idx))
            after = "\n".join(text_segs[j].target or ""
                              for j in range(idx + 1, min(len(text_segs), idx + 3)))
            feedback = "；".join(
                f"{it.get('detail', '')}（建议：{it.get('suggestion', '')}）"
                for it in seg_issues)
            new_t = self.translator.retranslate_with_feedback(
                seg.source, feedback=feedback, glossary_terms=terms, style=style,
                context_before=before, context_after=after,
                book_synopsis=book_synopsis, chapter_digest=chapter_digest)
            if new_t and not checks.length_flags([seg.source], [new_t]):
                if self.config.punctuation_normalize:
                    new_t = normalize_zh(new_t)
                seg.target = new_t
                for it in seg_issues:
                    it["fixed"] = True

    def _process_batch(self, batch, terms, ctx_text: str, style: str,
                       book_synopsis: str = "", chapter_digest: str = "") -> _BatchResult:
        """单个批次：整批翻译 → 长度校验（零成本，仅上报）→ 润色 → 标点规范化。

        每段都在自身上下文里翻译，不跨位置复用译文（避免丢失语境信息）。
        全书概览/本章梗概作为恒定前缀注入，让译者把握全局。
        LLM 审校不在批内做（移至章末统一做，见 _review_chapter，不阻塞翻译主路径）。
        """
        sources = [s.source for s in batch]
        targets = self.translator.translate_batch(
            sources, glossary_terms=terms, style=style, context=ctx_text,
            book_synopsis=book_synopsis, chapter_digest=chapter_digest)

        # 无成本长度校验：空译/过短/过长作为待人工项上报（index 为批内下标，调用方映射）
        issues: list[dict] = []
        for f in checks.length_flags(sources, targets):
            issues.append({
                "index": f.index, "type": f.reason,
                "detail": self._LEN_DETAIL.get(f.reason, f.reason),
                "suggestion": "", "stage": "length", "fixed": False,
            })

        if self.config.pipeline.polish:
            polished = self.polisher.polish(targets, glossary_terms=terms, style=style)
            if len(polished) == len(targets):
                targets = polished

        if self.config.punctuation_normalize:
            targets = [normalize_zh(t) if t else t for t in targets]

        bt_samples: list[tuple[str, str]] = []
        rate = self.config.pipeline.backtranslate_sample
        if rate > 0:
            for s, t in zip(sources, targets):
                if random.random() < rate:
                    bt_samples.append((s, t or ""))

        return _BatchResult(targets=targets, issues=issues, bt_samples=bt_samples)

    # ── 可选步骤 / 连续全流程 ────────────────────────────────────────────────
    ALL_STEPS = ("translate", "audit", "qa", "report", "assemble")

    def run_steps(self, input_path: str, steps, *,
                  progress: Optional[ProgressFn] = None,
                  out_format: str = "epub", out_path: str | None = None) -> dict[str, Any]:
        """按需执行步骤子集（可单选可全选）。steps ⊆ ALL_STEPS。"""
        from ..agents.glossary_auditor import GlossaryAuditor
        from ..agents.consistency import ConsistencyChecker
        from ..assemble.writer import assemble
        from ..assemble.report import build_report

        steps = set(steps)

        if "translate" in steps:
            store = self.run(input_path, progress=progress)
        else:
            store = self.prepare(input_path)
            m = store.load_manifest()
            self._apply_language(m.get("source_lang") or self.config.source_lang)

        glossary = GlossaryStore(store.glossary_path)
        audit_applied: list[dict] = []
        qa_issues: list[dict] = []
        report: dict[str, Any] | None = None
        try:
            if "audit" in steps:
                audit_applied = GlossaryAuditor(self.client, self.config).audit(store, glossary)

            if "qa" in steps:
                qa_issues = ConsistencyChecker(self.client, self.config).check(store, glossary)

            if "report" in steps:
                report = build_report(store, glossary)
                report["consistency_issues"] = qa_issues
                report["glossary_unifications"] = audit_applied
                store.save_report(report)
        finally:
            glossary.close()

        out = None
        if "assemble" in steps:
            out = assemble(store, input_path, out_path=out_path, out_format=out_format)

        return {"store": store, "output": out, "report": report,
                "qa_issues": qa_issues, "audit": audit_applied}

    def run_all(self, input_path: str, *, progress: Optional[ProgressFn] = None,
                out_format: str = "epub", out_path: str | None = None,
                do_audit: bool | None = None, do_qa: bool | None = None) -> dict[str, Any]:
        """翻译 → 术语审计统一 → 一致性 QA → 报告 → 回填 EPUB，返回结果汇总。"""
        steps = {"translate", "report", "assemble"}
        if do_audit if do_audit is not None else self.config.glossary_audit:
            steps.add("audit")
        if do_qa if do_qa is not None else self.config.pipeline.consistency_qa:
            steps.add("qa")
        return self.run_steps(input_path, steps, progress=progress,
                              out_format=out_format, out_path=out_path)
