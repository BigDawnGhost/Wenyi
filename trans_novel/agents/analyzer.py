"""全局分析 Agent（强档）。

通读样章，产出风格指南、角色圣经（含性别/语气）、初始术语候选，
并把角色/术语种入术语库，作为全书翻译的统一基准。
"""

from __future__ import annotations

from typing import Any

from ..glossary.store import GlossaryStore, GlossaryTerm, TYPE_PERSON
from . import prompts
from .base import Agent


class Analyzer(Agent):
    def analyze(self, sample_text: str) -> dict[str, Any]:
        system = prompts.render("analyzer_system", src=self.src, tgt=self.tgt)
        user = prompts.render("analyzer_user", src=self.src, tgt=self.tgt,
                              sample=sample_text)
        # 不传 default：分析失败照常抛出，由调用方决定（prepare 阶段失败应显式暴露）
        data = self._ask_json(system, user, tier="strong")
        if not isinstance(data, dict):
            data = {}
        data.setdefault("genre", "")
        data.setdefault("tone", "")
        data.setdefault("style_guide", "")
        for key in ("narration", "pacing", "register", "dialogue_style", "rhetoric"):
            data.setdefault(key, "")
        data.setdefault("characters", [])
        data.setdefault("terms", [])
        return data

    def seed_glossary(self, store: GlossaryStore, analysis: dict[str, Any]) -> int:
        """把分析得到的角色/术语种入术语库，返回写入条目数。"""
        count = 0
        for ch in analysis.get("characters", []):
            if not ch.get("source") or not ch.get("target"):
                continue
            store.upsert_term(
                GlossaryTerm(
                    source=ch["source"], target=ch["target"],
                    reading=ch.get("reading", ""), type=TYPE_PERSON,
                    gender=ch.get("gender", ""), note=ch.get("note", ""),
                    confidence="medium", first_chapter=0,
                ),
                chapter=0,
            )
            count += 1
        for tm in analysis.get("terms", []):
            if not tm.get("source") or not tm.get("target"):
                continue
            store.upsert_term(
                GlossaryTerm(
                    source=tm["source"], target=tm["target"],
                    reading=tm.get("reading", ""), type=tm.get("type", "术语"),
                    note=tm.get("note", ""), confidence="medium", first_chapter=0,
                ),
                chapter=0,
            )
            count += 1
        return count

    def style_brief(self, analysis: dict[str, Any]) -> str:
        """把分析结果浓缩成给译者注入的风格/角色简报。"""
        lines = []
        if analysis.get("genre"):
            lines.append(f"体裁：{analysis['genre']}")
        if analysis.get("tone"):
            lines.append(f"语气文体：{analysis['tone']}")
        if analysis.get("style_guide"):
            lines.append(f"风格指南：{analysis['style_guide']}")
        # 细粒度风格维度（旧 analysis.json 缺字段时自动跳过，向后兼容）
        for key, tag in (("narration", "叙事"), ("pacing", "句式节奏"),
                         ("register", "语域"), ("dialogue_style", "对话风格"),
                         ("rhetoric", "修辞")):
            if analysis.get(key):
                lines.append(f"{tag}：{analysis[key]}")
        chars = analysis.get("characters", [])
        if chars:
            lines.append("角色：")
            for c in chars:
                g = f"，{c.get('gender')}" if c.get("gender") else ""
                note = f"，{c.get('note')}" if c.get("note") else ""
                lines.append(f"  - {c.get('target', c.get('source',''))}({c.get('source','')}{g}{note})")
        return "\n".join(lines)
