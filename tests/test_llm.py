"""LLM 抽象层与 JSON 解析的测试（离线）。"""

from __future__ import annotations

import unittest

from trans_novel.llm.base import FakeClient, parse_json_loose


class TestParseJsonLoose(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_json_loose('{"a":1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(parse_json_loose("```json\n[1,2,3]\n```"), [1, 2, 3])

    def test_surrounded_by_prose(self):
        text = '思考结束。结果如下：["译文1","译文2"] 完毕。'
        self.assertEqual(parse_json_loose(text), ["译文1", "译文2"])

    def test_failure(self):
        with self.assertRaises(ValueError):
            parse_json_loose("没有任何 JSON 内容")


class TestResolveTier(unittest.TestCase):
    def test_fallback_chain(self):
        from trans_novel.config import TierConfig
        from trans_novel.llm.base import resolve_tier

        strong = TierConfig(model="pro")
        cheap = TierConfig(model="flash")
        fast = TierConfig(model="flash", thinking=False)

        # 三档全有 → 各归各
        tiers = {"strong": strong, "cheap": cheap, "fast": fast}
        self.assertIs(resolve_tier(tiers, "fast"), fast)
        self.assertIs(resolve_tier(tiers, "cheap"), cheap)
        self.assertIs(resolve_tier(tiers, "strong"), strong)
        # 无 fast → 落 cheap（不升到更贵的 strong）
        tiers2 = {"strong": strong, "cheap": cheap}
        self.assertIs(resolve_tier(tiers2, "fast"), cheap)
        # 只有 strong → 都落 strong
        tiers3 = {"strong": strong}
        self.assertIs(resolve_tier(tiers3, "fast"), strong)
        self.assertIs(resolve_tier(tiers3, "cheap"), strong)
        # 未知档 → 落 strong
        self.assertIs(resolve_tier(tiers, "unknown"), strong)


class TestFakeClient(unittest.TestCase):
    def test_default(self):
        c = FakeClient()
        self.assertEqual(c.complete([{"role": "user", "content": "x"}]), "")
        self.assertEqual(c.complete_json([{"role": "user", "content": "x"}]), [])

    def test_handler(self):
        def handler(messages, tier, json_mode):
            return '["A","B"]' if json_mode else "hello"

        c = FakeClient(handler=handler)
        self.assertEqual(c.complete([{"role": "user", "content": "x"}]), "hello")
        self.assertEqual(c.complete_json([{"role": "user", "content": "x"}]), ["A", "B"])
        self.assertEqual(len(c.calls), 2)


if __name__ == "__main__":
    unittest.main()
