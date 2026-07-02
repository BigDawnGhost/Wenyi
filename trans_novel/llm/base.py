"""LLM 抽象接口与具体实现。

设计要点：
- 三档 tier："strong"（deepseek-v4-pro + thinking，翻译/润色/分析/审计）、
  "cheap"（deepseek-v4-flash + thinking，审校/一致性等判断类）、
  "fast"（deepseek-v4-flash 免思考，梗概/术语抽取/回译等机械任务——
  thinking 推理 token 按输出计费，机械任务关掉可大幅省钱提速）。
  缺档时按回退链向"更便宜优先"回退（fast→cheap→strong），老双档配置行为不变。
- complete() 返回纯文本；complete_json() 强制 JSON 输出并 loose 解析。
- DeepSeekClient 经由 OpenAI SDK 调 https://api.deepseek.com，openai 惰性导入；
  未装 openai 时仍可用 FakeClient 跑通离线流程（切分/对齐/术语库/状态机）。
"""

from __future__ import annotations

import json
import re
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Config, LLMConfig, TierConfig

Messages = list[dict[str, str]]

# 缺档回退链：向"更便宜优先"回退，绝不因缺档反而升到更贵的档
_TIER_FALLBACK = {"fast": ("cheap", "strong"), "cheap": ("strong",), "strong": ()}


def resolve_tier(tiers: dict[str, TierConfig], tier: str) -> TierConfig:
    """按回退链解析 tier 配置。缺 strong 时 KeyError（与旧行为一致）。"""
    if tier in tiers:
        return tiers[tier]
    for fb in _TIER_FALLBACK.get(tier, ("strong",)):
        if fb in tiers:
            return tiers[fb]
    return tiers["strong"]


# ── JSON 宽松解析 ────────────────────────────────────────────────────────
def parse_json_loose(text: str) -> Any:
    """从模型输出里尽力解析 JSON。

    优先直接 json.loads；失败则剥离 ```json 围栏并截取首个 {…}/[…] 块再试。
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 去掉 markdown 代码围栏
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        inner = fenced.group(1).strip()
        try:
            return json.loads(inner)
        except Exception:
            text = inner
    # 截取首个 JSON 数组或对象
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    raise ValueError(f"无法解析为 JSON：{text[:200]!r}")


# ── 抽象接口 ──────────────────────────────────────────────────────────────
class LLMClient(ABC):
    """所有 provider 实现此接口。"""

    @abstractmethod
    def complete(
        self,
        messages: Messages,
        *,
        tier: str = "strong",
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        """返回模型回复的纯文本。"""
        raise NotImplementedError

    def complete_json(
        self,
        messages: Messages,
        *,
        tier: str = "strong",
        max_tokens: Optional[int] = None,
    ) -> Any:
        """要求 JSON 输出并解析。"""
        text = self.complete(messages, tier=tier, json_mode=True, max_tokens=max_tokens)
        return parse_json_loose(text)


# ── DeepSeek（OpenAI SDK 兼容）────────────────────────────────────────────
class DeepSeekClient(LLMClient):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        if not cfg.tiers:
            raise ValueError("配置缺少 llm.tiers")
        self._client = None  # 惰性创建
        self._client_lock = threading.Lock()  # 预扫并行时防惰性初始化竞态

    def _ensure_client(self):
        with self._client_lock:
            return self._ensure_client_locked()

    def _ensure_client_locked(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "需要 openai SDK：pip install openai（或把 llm.provider 设为 fake 做离线测试）"
                ) from e
            api_key = self.cfg.api_key
            if not api_key:
                raise RuntimeError(
                    f"未设置环境变量 {self.cfg.api_key_env}（DeepSeek API key）"
                )
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.cfg.base_url,
                timeout=self.cfg.timeout,
            )
        return self._client

    def complete(
        self,
        messages: Messages,
        *,
        tier: str = "strong",
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        tcfg = resolve_tier(self.cfg.tiers, tier)
        client = self._ensure_client()

        kwargs: dict[str, Any] = {
            "model": tcfg.model,
            "messages": messages,
            "stream": False,
        }
        if tcfg.thinking:
            kwargs["reasoning_effort"] = tcfg.reasoning_effort
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            # DeepSeek thinking 模式下 max_tokens 含推理 token（总输出上限）。
            # 带紧上限的调用若经回退链落到 thinking 档，抬到安全下限防推理被截断。
            kwargs["max_tokens"] = max(max_tokens, 4096) if tcfg.thinking else max_tokens

        # 网络/限流/超时 → tenacity 指数退避重试（最多 max_retries 次重试）
        @retry(
            stop=stop_after_attempt(self.cfg.max_retries + 1),
            wait=wait_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _call() -> str:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        return _call()


# ── 离线 Fake（测试 / 不发网络请求）───────────────────────────────────────
class FakeClient(LLMClient):
    """可编程的离线 client。

    handler(messages, tier, json_mode) -> str。默认对 json_mode 返回 "[]"，
    否则返回空串。测试通过注入 handler 模拟翻译/抽取等行为。
    """

    def __init__(self, handler: Optional[Callable[[Messages, str, bool], str]] = None):
        self.handler = handler
        self.calls: list[dict[str, Any]] = []  # 记录调用，便于断言

    def complete(
        self,
        messages: Messages,
        *,
        tier: str = "strong",
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        self.calls.append({"messages": messages, "tier": tier,
                           "json_mode": json_mode, "max_tokens": max_tokens})
        if self.handler is not None:
            return self.handler(messages, tier, json_mode)
        return "[]" if json_mode else ""


def build_client(config: Config) -> LLMClient:
    provider = config.llm.provider.lower()
    if provider == "deepseek":
        return DeepSeekClient(config.llm)
    if provider == "fake":
        return FakeClient()
    raise ValueError(f"未知 provider：{provider}（支持 deepseek / fake）")
