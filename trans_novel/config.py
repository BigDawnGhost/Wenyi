"""配置加载。读取 config.yaml，提供带默认值的类型化访问（pydantic v2）。"""

from __future__ import annotations

import os
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TierConfig(BaseModel):
    model: str
    reasoning_effort: str = "high"
    thinking: bool = True


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout: int = 600
    max_retries: int = 4
    tiers: dict[str, TierConfig] = Field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


class SegmentConfig(BaseModel):
    max_chars_per_batch: int = 1800
    max_chars_per_segment: int = 1200


class PipelineConfig(BaseModel):
    review: bool = True
    autofix_severe: bool = True      # 章末审校后自动重译严重项（漏译/误译）；关闭则仅上报留人工
    align_retry_limit: int = 2       # 批次翻译段数不符时的整批重试次数，超限后逐段兜底
    polish: bool = False             # 默认关：润色=用强档把全书再翻一遍，最烧钱；需要时显式开
    backtranslate_sample: float = 0.05
    consistency_qa: bool = True
    rolling_context_segments: int = 6
    # 翻译前预扫源文，生成全书概览+逐章梗概注入翻译 prompt（让译者对全书有理解）。
    # fast 档（免思考），且全局概览为恒定前缀可命中缓存复用；关掉可省去预扫成本。
    book_understanding: bool = True
    prescan_concurrency: int = 4     # 预扫逐章梗概的并发线程数（各章独立，1=串行）
    glossary_scope: str = "chapter"  # chapter=只注入本章出现的词条+锁定人物（省 token）；full=全量表


class Config(BaseModel):
    source_lang: str = "auto"        # auto | ja | en | …（auto 时按正文脚本自动检测）
    target_lang: str = "zh"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    honorific_strategy: str = "keep_style"
    punctuation_normalize: bool = True  # 译文标点规范化为简体中文通用
    glossary_audit: bool = True      # 收尾做术语 AI 审计统一（改写正文）
    state_dir: str = "state"

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        lang = raw.get("language", {})
        llm_raw = raw.get("llm", {})
        tiers = {
            name: TierConfig.model_validate(t)
            for name, t in (llm_raw.get("tiers", {}) or {}).items()
        }
        llm = LLMConfig(
            provider=llm_raw.get("provider", "deepseek"),
            base_url=llm_raw.get("base_url", "https://api.deepseek.com"),
            api_key_env=llm_raw.get("api_key_env", "DEEPSEEK_API_KEY"),
            timeout=llm_raw.get("timeout", 600),
            max_retries=llm_raw.get("max_retries", 4),
            tiers=tiers,
        )
        segment = SegmentConfig.model_validate(raw.get("segment", {}) or {})
        pipeline = PipelineConfig.model_validate(raw.get("pipeline", {}) or {})
        punct = raw.get("punctuation", {}) or {}
        return cls(
            source_lang=lang.get("source", "auto"),
            target_lang=lang.get("target", "zh"),
            llm=llm,
            segment=segment,
            pipeline=pipeline,
            honorific_strategy=raw.get("honorific", {}).get("strategy", "keep_style"),
            punctuation_normalize=bool(punct.get("normalize", True)),
            glossary_audit=bool(raw.get("glossary_audit", True)),
            state_dir=raw.get("paths", {}).get("state_dir", "state"),
        )
