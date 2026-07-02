"""Agent 基类：统一 client/config/src/tgt 初始化，与带默认值的 LLM 调用帮助方法。

各 agent 的"渲染 system/user → complete_json → 失败回退默认值"模式收敛到这里；
默认值语义留在 agent 层（传输层 llm/base.py 不掺业务回退）。
orchestrator._apply_language 依赖每个 agent 都有 .src 属性——基类把该契约显式化。
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..llm.base import LLMClient

_RAISE = object()  # 哨兵：未提供 default 时异常照常抛出，由调用方自理


class Agent:
    def __init__(self, client: LLMClient, config: Config):
        self.client = client
        self.config = config
        self.src = config.source_lang
        self.tgt = config.target_lang

    def _ask_json(self, system: str, user: str, *, tier: str,
                  key: str | None = None, default: Any = _RAISE,
                  max_tokens: int | None = None) -> Any:
        """system/user → complete_json。

        异常时返回 default（未给 default 则照常抛出，如 Translator 交由重试逻辑处理）。
        key 给出时：结果为 dict 取 data[key]（缺失回退）；结果为非空 list 直接用；否则回退。
        """
        try:
            data = self.client.complete_json(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}], tier=tier,
                max_tokens=max_tokens)
        except Exception:
            if default is _RAISE:
                raise
            return default
        if key is None:
            return data
        fb = None if default is _RAISE else default
        if isinstance(data, dict):
            return data.get(key, fb)
        return data if data else fb

    def _ask_text(self, system: str, user: str, *, tier: str,
                  default: str = "", max_tokens: int | None = None) -> str:
        """complete 纯文本并 strip；异常返回 default。"""
        try:
            return (self.client.complete(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}], tier=tier,
                max_tokens=max_tokens) or "").strip()
        except Exception:
            return default

    @staticmethod
    def dict_items(items: Any) -> list[dict]:
        """过滤出 dict 元素（issues/terms 等模型返回列表的通用清洗）。"""
        return [i for i in items or [] if isinstance(i, dict)]
