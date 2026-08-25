"""正規化層のテスト。

**実測で見つけた 2 つのバグを回帰として固定している**（`test_fence_marker_*`）。
どちらも「モデルの癖ではなく検出器の癖を測る」形の壊れ方で、FLB-QB-001 §6 が
最大の危険として名指ししていたもの。
"""

from __future__ import annotations

import pytest

from quirkbench import parse as P

JSON_SPEC = {"format": "json", "extract": "fenced_or_first_object"}
NONE_SPEC = {"format": "none"}

FENCED_JSON = '```json\n{\n  "name": "田中一郎",\n  "age": 42\n}\n```'


# ------------------------------------------------------------ 回帰（実測由来）


def test_fence_marker_is_not_preamble() -> None:
    """フェンス行は抽出の足場であって前置きではない。

    これを前置きと数えると、フェンス付き JSON を返すモデル**全件**に preamble が付き、
    層3 が測るのはモデルの癖ではなくこの検出器の癖になる。
    """
    parsed = P.parse(FENCED_JSON, JSON_SPEC)
    assert parsed.preamble == ""
    assert parsed.json_value == {"name": "田中一郎", "age": 42}


def test_fence_marker_is_not_repetition() -> None:
    """``` の 3 連バッククォートを n-gram 反復として数えない。"""
    parsed = P.parse(FENCED_JSON, JSON_SPEC)
    assert parsed.repeat.unit != "`"
    assert parsed.repeat.span < 12


def test_real_preamble_still_detected() -> None:
    parsed = P.parse("はい、承知しました。\n" + FENCED_JSON, JSON_SPEC)
    assert parsed.preamble == "はい、承知しました。"


def test_preamble_without_fence() -> None:
    parsed = P.parse('以下です。\n{"a": 1}\n以上です。', JSON_SPEC)
    assert parsed.preamble == "以下です。"
    assert parsed.json_value == {"a": 1}


# ---------------------------------------------------------------- フェンス


def test_scan_fences_closed_and_unclosed() -> None:
    blocks = P.scan_fences("前\n```py\nx = 1\n```\n後\n```\ny = 2")
    assert len(blocks) == 2
    assert blocks[0].lang == "py" and blocks[0].body == "x = 1" and blocks[0].closed
    assert blocks[1].lang is None and blocks[1].body == "y = 2" and not blocks[1].closed


def test_unclosed_fence_recorded() -> None:
    parsed = P.parse('```json\n{"a": 1}', JSON_SPEC)
    assert "fence" in parsed.unclosed


def test_no_fence_means_no_blocks() -> None:
    assert P.scan_fences("ただの文章") == ()


# ------------------------------------------------------------ 均衡した構造


@pytest.mark.parametrize(
    ("text", "expected", "closed"),
    [
        ('{"a": 1}', '{"a": 1}', True),
        ('前置き {"a": {"b": [1, 2]}} 後置き', '{"a": {"b": [1, 2]}}', True),
        ('{"a": "}"}', '{"a": "}"}', True),  # 文字列内の括弧を数えない
        ('{"a": "\\""}', '{"a": "\\""}', True),  # エスケープされた引用符
        ("[1, 2, 3]", "[1, 2, 3]", True),
        ('{"a": 1', '{"a": 1', False),  # 閉じていない
    ],
)
def test_find_balanced(text: str, expected: str, closed: bool) -> None:
    found = P.find_balanced(text)
    assert found is not None
    start, end, is_closed = found
    assert text[start:end] == expected
    assert is_closed is closed


def test_find_balanced_none() -> None:
    assert P.find_balanced("括弧が無い") is None


def test_unclosed_json_recorded() -> None:
    parsed = P.parse('{"a": 1', JSON_SPEC)
    assert "json" in parsed.unclosed
    assert not parsed.format_ok


# -------------------------------------------------------------------- 件数


def test_items_counted_through_parse() -> None:
    parsed = P.parse(
        "1. あ\n2. い\n3. う\n4. え\n5. お",
        {"format": "none", "count": {"n": 5, "pattern": "numbered_list"}},
    )
    assert len(parsed.items) == 5


# ---------------------------------------------------------------- 空の判定


@pytest.mark.parametrize("text", ["", "   ", "\n\t \n", "​﻿", "```\n```", "```json\n```"])
def test_visible_chars_zero(text: str) -> None:
    assert P.visible_chars(text) == 0


def test_visible_chars_counts_content() -> None:
    assert P.visible_chars("```\nab\n```") == 2


# ------------------------------------------------------------------ python


def test_python_format_ok() -> None:
    parsed = P.parse("```python\ndef f():\n    return 1\n```", {"format": "python"})
    assert parsed.format_ok
    assert parsed.syntax_error is None


def test_python_syntax_error() -> None:
    parsed = P.parse("```python\ndef f(:\n```", {"format": "python"})
    assert not parsed.format_ok
    assert parsed.syntax_error


def test_python_unclosed_bracket() -> None:
    parsed = P.parse("```python\nx = [1, 2\n```", {"format": "python"})
    assert "python" in parsed.unclosed


# -------------------------------------------------------------------- 雑則


def test_unknown_format_rejected() -> None:
    with pytest.raises(ValueError, match=r"未知の failure\.format"):
        P.parse("x", {"format": "yaml"})


def test_unknown_extractor_rejected() -> None:
    with pytest.raises(ValueError, match=r"未知の failure\.extract"):
        P.parse("x", {"format": "json", "extract": "magic"})


def test_format_none_keeps_whole_body() -> None:
    parsed = P.parse("ただの日本語の応答です。", NONE_SPEC)
    assert parsed.preamble == ""
    assert parsed.body == "ただの日本語の応答です。"
    assert parsed.format_ok


def test_body_excludes_code_blocks_for_language() -> None:
    parsed = P.parse("説明です。\n```python\nprint('hello world')\n```", NONE_SPEC)
    assert "print" not in parsed.body
    assert "説明です。" in parsed.body


def test_normalize_folds_fullwidth() -> None:
    assert P.normalize("  ４２　") == "42"


# ------------------------------------------------- レビュー由来の回帰（実測で再現済み）


def test_explanatory_fence_does_not_break_extraction() -> None:
    """説明用フェンスが先にあっても、後ろの正しい JSON を採る。

    最初のフェンスを無条件に採ると、測るのは「最初のフェンスが JSON か」になる。
    余計に書いたことは preamble タグが記録する（Codex の指摘）。
    """
    parsed = P.parse('```text\npreliminary\n```\n{"a": 1}', JSON_SPEC)
    assert parsed.format_ok
    assert parsed.json_value == {"a": 1}
    assert parsed.preamble == "preliminary"


def test_json_fence_preferred_over_other_language() -> None:
    parsed = P.parse('```python\nx = 1\n```\n```json\n{"a": 2}\n```', JSON_SPEC)
    assert parsed.json_value == {"a": 2}


def test_all_candidates_failing_reports_the_first() -> None:
    """どれも通らなければ第一候補の誤りを報告する。format_broken は到達可能なまま。"""
    parsed = P.parse("```text\nこわれた\n```\nこれも JSON ではない", JSON_SPEC)
    assert not parsed.format_ok
    assert parsed.json_error


def test_unclosed_belongs_to_the_scored_block() -> None:
    """採点対象が完全なら、応答末尾の未閉じフェンスで incomplete にしない。

    応答全体の未閉じは trailing_unclosed_fence が別に持つ（Opus の指摘）。
    """
    parsed = P.parse('```json\n{"a": 1}\n```\n\nおまけ:\n```python\nx = (', JSON_SPEC)
    assert parsed.format_ok
    assert parsed.unclosed == ()


def test_json_null_is_valid_json() -> None:
    """null は正当な JSON。パース失敗と混同すると『JSON ですらなかった』と
    『JSON だが中身が違う』の区別が崩れる（Opus の指摘）。"""
    parsed = P.parse("null", JSON_SPEC)
    assert parsed.format_ok
    assert parsed.json_value is None
    assert parsed.json_error is None


def test_content_chars_excludes_fence_markers() -> None:
    """長さを len(raw) で測ると、フェンスを付けるモデルだけ数文字ぶん不利になる。"""
    parsed = P.parse("```\nabcde\n```", NONE_SPEC)
    assert parsed.content_chars == 5
    assert len(parsed.raw) > parsed.content_chars


# ------------------------------------------------- fenced_or_whole（§12.4）

CODE_SPEC = {
    "format": "python",
    "extract": "fenced_or_whole",
    "language": "none",
    "requires_def": "solve",
}


def test_explanatory_fence_before_the_answer_is_skipped() -> None:
    """説明用の断片を先に書くモデルを落とさない。落とすと測っているのが出力順序になる。"""
    raw = (
        "考え方です。\n```text\nこうします\n```\n"
        "実装:\n```python\ndef solve(x):\n    return x * 2\n```"
    )
    assert "x * 2" in P.parse(raw, CODE_SPEC).payload


def test_last_valid_candidate_wins() -> None:
    """誤答を示してから直す型では、**両方の候補が entry_point を定義する**。

    Python 自身が後の定義で前を上書きするので、最後を採るのが実行時の挙動と一致する。
    """
    raw = (
        "```python\ndef solve(x):\n    return x + 2\n```\n"
        "正しくは:\n```python\ndef solve(x):\n    return x * 2\n```"
    )
    assert "x * 2" in P.parse(raw, CODE_SPEC).payload
    assert "x + 2" not in P.parse(raw, CODE_SPEC).payload


def test_whole_text_is_the_last_resort_not_the_default() -> None:
    """全文を先に置くと「最後を採る」がいつも全文を選び、フェンスを見る意味が消える。"""
    raw = "```python\ndef solve(x):\n    return x * 2\n```"
    assert P.parse(raw, CODE_SPEC).payload.strip().startswith("def solve")


def test_falls_back_to_whole_text_without_fences() -> None:
    assert "x * 2" in P.parse("def solve(x):\n    return x * 2\n", CODE_SPEC).payload


def test_candidate_must_define_the_entry_point_at_top_level() -> None:
    """top-level に無いものは候補にしない。ネストした定義は import から見えない。"""

    assert P.has_toplevel_def("def solve(x):\n    return x\n", "solve")
    assert not P.has_toplevel_def("def outer():\n    def solve(x):\n        return x\n", "solve")
    assert not P.has_toplevel_def("def other(x):\n    return x\n", "solve")
    assert not P.has_toplevel_def("def solve(x)\n    broken\n", "solve")


def test_syntax_error_keeps_format_broken_reachable() -> None:
    parsed = P.parse("```python\ndef solve(x)\n    return x\n```", CODE_SPEC)
    assert parsed.syntax_error is not None
