"""全书理解预扫 Agent（廉价档）。

翻译开始前通读**源文**，产出：
- 逐章梗概（chapter digest）：每章一段中文梗概，存入 chapter.meta["source_digest"]；
- 全书概览（book synopsis）：把各章梗概 + 前期分析归并成一份全局概览。

二者作为**恒定前缀**注入翻译 prompt（见 prompts.py），让译者翻任意章节前都"对全书有理解"：
把握主线走向、人物弧光、伏笔与谜底，避免早期章节盲译。全局块全程不变，命中前缀缓存近免费复用。
归并对超长书做分组 map-reduce，避免单次 prompt 超长。
"""

from __future__ import annotations

from . import prompts
from .base import Agent

# 归并时单次喂入的各章梗概字符预算；超过则分组先归并再合并。
_REDUCE_BUDGET = 12000


class Synopsizer(Agent):
    def digest_chapter(self, source_text: str) -> str:
        """把单章源文压成一段中文梗概；空文本或失败返回空串。"""
        if not source_text.strip():
            return ""
        system = prompts.render("chapter_digest_system", src=self.src, tgt=self.tgt)
        user = prompts.render("chapter_digest_user", src=self.src, tgt=self.tgt,
                              source=source_text[:8000])
        # 机械任务走 fast 档（免思考）；梗概 ≤200 字，上限留足裕量防输出失控
        return self._ask_text(system, user, tier="fast", max_tokens=600)

    def book_synopsis(self, digests: list[str], analysis_brief: str) -> str:
        """把各章梗概 + 前期分析归并成全书概览。超长则分组 map-reduce。"""
        items = [d.strip() for d in digests if d and d.strip()]
        if not items:
            return ""
        while True:
            groups = self._group(items, _REDUCE_BUDGET)
            if len(groups) == 1:
                return self._synth(groups[0], analysis_brief)
            # 多组：每组先归并为一段较粗的概览，再进入下一轮归并
            items = [self._synth(g, analysis_brief) for g in groups]
            items = [s for s in items if s.strip()]
            if not items:
                return ""

    # ── 内部 ────────────────────────────────────────────────────────────────
    @staticmethod
    def _group(items: list[str], budget: int) -> list[list[str]]:
        """按字符预算贪心打包成若干组（每组 joined 长度尽量 ≤ budget）。"""
        groups: list[list[str]] = []
        cur: list[str] = []
        size = 0
        for it in items:
            if cur and size + len(it) > budget:
                groups.append(cur)
                cur, size = [], 0
            cur.append(it)
            size += len(it) + 1
        if cur:
            groups.append(cur)
        return groups

    def _synth(self, digests: list[str], analysis_brief: str) -> str:
        numbered = "\n".join(f"[{i}] {d}" for i, d in enumerate(digests))
        system = prompts.render("book_synopsis_system", src=self.src, tgt=self.tgt)
        user = prompts.render("book_synopsis_user", src=self.src, tgt=self.tgt,
                              analysis=analysis_brief or "（无）", digests=numbered)
        # 概览 ≤500 字，fast 档 + 上限
        return self._ask_text(system, user, tier="fast", max_tokens=1200)
