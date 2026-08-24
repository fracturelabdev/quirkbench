"""失敗型の検出器のテスト。

要点は**タグが付くこと**より、``applicable``（分布の母数）が正しいこと。
宣言していない型を母数に入れると「適用して 0 件」と「そもそも見ていない」が
混ざり、層3 の分布が読めなくなる（FLB-QB-001 §6-B）。
"""

from __future__ import annotations

import pytest

from quirkbench import failures as F
from quirkbench.parse import parse

JSON_SPEC = {"format": "json", "extract": "fenced_or_first_object", "min_tokens": 4}
JA_SPEC = {"format": "none", "language": "ja", "min_tokens": 10}


def detect(
    text: str, spec: dict, *, done_reason: str = "stop", eval_count: int = 50
) -> F.FailureReport:
    return F.detect(parse(text, spec), spec, done_reason=done_reason, eval_count=eval_count)


# -------------------------------------------------------------------- 固い


def test_truncated_from_done_reason() -> None:
    report = detect('{"a": 1', JSON_SPEC, done_reason="length")
    assert F.TRUNCATED in report.tags


def test_truncated_and_format_broken_coexist() -> None:
    """長さ制限で切れた JSON は両方付く。単一ラベルにしない（§6-A）。"""
    report = detect('{"a": 1, "b": ', JSON_SPEC, done_reason="length")
    assert F.TRUNCATED in report.tags
    assert F.FORMAT_BROKEN in report.tags


@pytest.mark.parametrize("text", ["", "   \n ", "```\n```"])
def test_empty_by_visible_chars(text: str) -> None:
    report = detect(text, JSON_SPEC)
    assert report.tags == (F.EMPTY,)


def test_empty_by_eval_count() -> None:
    report = detect("なにか", JA_SPEC, eval_count=0)
    assert F.EMPTY in report.tags


def test_empty_short_circuits_derived_checks() -> None:
    """中身が無いものに派生判定を掛けても、測るのは検出器の既定値でしかない。"""
    report = detect("", JSON_SPEC)
    assert set(report.applicable) == {F.EMPTY, F.TRUNCATED}
    assert F.FORMAT_BROKEN not in report.applicable


# ------------------------------------------------------------ 宣言があれば固い


def test_incomplete_when_stopped_but_unclosed() -> None:
    report = detect('{"a": 1', JSON_SPEC, done_reason="stop")
    assert F.INCOMPLETE in report.tags


def test_incomplete_not_tagged_when_truncated() -> None:
    """打ち切りが原因なら incomplete ではない。done_reason で切り分ける。"""
    report = detect('{"a": 1', JSON_SPEC, done_reason="length")
    assert F.INCOMPLETE not in report.tags


def test_format_broken_only_when_declared() -> None:
    report = detect("{ これは JSON ではない", {"format": "none"})
    assert F.FORMAT_BROKEN not in report.applicable
    assert F.FORMAT_BROKEN not in report.tags


def test_preamble_requires_format_success() -> None:
    """抽出が失敗しているときは preamble を付けない（§6-C）。"""
    broken = detect("はい、承知しました。\n{壊れた", JSON_SPEC)
    assert F.FORMAT_BROKEN in broken.tags
    assert F.PREAMBLE not in broken.tags

    ok = detect('はい、承知しました。\n{"a": 1}', JSON_SPEC)
    assert F.PREAMBLE in ok.tags
    assert F.FORMAT_BROKEN not in ok.tags


def test_fenced_json_gets_no_preamble() -> None:
    """実測由来の回帰。フェンスを前置きと数えると全件に付く。"""
    report = detect('```json\n{"a": 1}\n```', JSON_SPEC)
    assert report.tags == ()


def test_count_mismatch_only_when_declared() -> None:
    spec = {"format": "none", "count": {"n": 5, "pattern": "numbered_list"}}
    short = detect("1. あ\n2. い\n3. う", spec)
    assert F.COUNT_MISMATCH in short.tags
    assert short.details["item_count"] == 3

    exact = detect("1. あ\n2. い\n3. う\n4. え\n5. お", spec)
    assert F.COUNT_MISMATCH not in exact.tags
    assert F.COUNT_MISMATCH in exact.applicable

    assert F.COUNT_MISMATCH not in detect("1. あ", {"format": "none"}).applicable


def test_overlong_by_chars() -> None:
    spec = {"format": "none", "max_chars": 10}
    assert F.OVERLONG in detect("あ" * 20, spec).tags
    assert F.OVERLONG not in detect("あ" * 5, spec).tags


def test_overlong_by_tokens() -> None:
    spec = {"format": "none", "max_tokens": 10}
    assert F.OVERLONG in detect("あ" * 30, spec, eval_count=40).tags
    assert F.OVERLONG not in detect("あ" * 30, spec, eval_count=5).tags


def test_overlong_not_applicable_without_declaration() -> None:
    assert F.OVERLONG not in detect("あ" * 100, {"format": "none"}).applicable


# ------------------------------------------------------------------ 候補階層


def test_wrong_language() -> None:
    report = detect("This is an English answer that is long enough to judge.", JA_SPEC)
    assert F.WRONG_LANGUAGE in report.tags
    assert report.details["detected_lang"] == "en"


def test_zh_leak_tags_both() -> None:
    """中国語漏れは wrong_language でもある。多タグを許す（§6-A）。"""
    report = detect("这是一段中文句子内容足够长可以用来判定语言种类了对吧", JA_SPEC)
    assert F.ZH_LEAK in report.tags
    assert F.WRONG_LANGUAGE in report.tags


def test_language_not_applied_to_short_text() -> None:
    """20 文字未満に言語ラベルを付けない。短い正答を誤判定するため。"""
    report = detect("はい", JA_SPEC)
    assert F.WRONG_LANGUAGE not in report.applicable


def test_language_not_applied_when_none() -> None:
    report = detect(
        "This is an English answer that is long enough.", {"format": "none", "language": "none"}
    )
    assert F.WRONG_LANGUAGE not in report.applicable


def test_repeated() -> None:
    report = detect("はい、承知しました。" * 5, JA_SPEC)
    assert F.REPEATED in report.tags
    assert report.details["repeat"]["runs"] >= 3


def test_repeated_not_fired_on_normal_list() -> None:
    report = detect(
        "1. 珈琲の木\n2. 街角の椅子\n3. 静かな窓辺\n4. 路地の灯\n5. 小さな隣人", JA_SPEC
    )
    assert F.REPEATED not in report.tags


# -------------------------------------------------------------- 順序と統合


def test_tags_follow_priority_order() -> None:
    report = detect(
        "A" * 200, {"format": "json", "language": "en", "max_chars": 10}, done_reason="length"
    )
    assert list(report.tags) == [tag for tag in F.PRIORITY if tag in report.tags]


def test_merge_is_immutable() -> None:
    original = F.FailureReport(tags=(F.TRUNCATED,), applicable=(F.TRUNCATED,))
    merged = original.merge(tags=(F.EMPTY,), applicable=(F.EMPTY,))
    assert original.tags == (F.TRUNCATED,)
    assert merged.tags == (F.EMPTY, F.TRUNCATED)  # PRIORITY 順に並ぶ


def test_every_tag_has_a_tier() -> None:
    assert set(F.PRIORITY) == set(F.TIER)


# --------------------------------------------------------------- non_attempt


def test_non_attempt_fires() -> None:
    tags, applicable = F.detect_non_attempt(score=0.0, eval_count=3, min_tokens=10, truncated=False)
    assert tags == (F.NON_ATTEMPT,) and applicable == (F.NON_ATTEMPT,)


@pytest.mark.parametrize(
    ("score", "eval_count", "truncated"),
    [
        (1.0, 3, False),  # 正解しているなら短くても非着手ではない
        (0.0, 50, False),  # 長く書いて間違えたのは非着手ではない
        (0.0, 3, True),  # 打ち切りが原因なら truncated が説明する
    ],
)
def test_non_attempt_does_not_fire(score: float, eval_count: int, truncated: bool) -> None:
    tags, applicable = F.detect_non_attempt(
        score=score, eval_count=eval_count, min_tokens=10, truncated=truncated
    )
    assert tags == ()
    assert applicable == (F.NON_ATTEMPT,)


def test_non_attempt_not_applicable_without_min_tokens() -> None:
    assert F.detect_non_attempt(score=0.0, eval_count=1, min_tokens=None, truncated=False) == (
        (),
        (),
    )


# ------------------------------------------------- レビュー由来の回帰（実測で再現済み）


def test_incomplete_not_applicable_on_truncated_rows() -> None:
    """打ち切られた行では incomplete は原理的に発火しない。母数に入れない。

    入れると、切れやすいモデルほど分母だけ膨らんで incomplete 率が低く見える
    （Grok の指摘）。overlong >= num_predict と同じ「死んだ検出器」の形。
    """
    report = detect('{"a": 1', JSON_SPEC, done_reason="length")
    assert F.INCOMPLETE not in report.applicable
    assert F.INCOMPLETE not in report.tags
    assert F.INCOMPLETE in detect('{"a": 1', JSON_SPEC, done_reason="stop").applicable


def test_unknown_language_is_not_wrong_language() -> None:
    """判定不能を別言語の証拠にしない。和風の店名は全漢字が普通（Codex の指摘）。"""
    report = detect("1. 珈琲小径\n2. 街角十二席\n3. 灯窓珈琲\n4. 静席\n5. 住宅街喫茶", JA_SPEC)
    assert F.WRONG_LANGUAGE not in report.tags
    assert F.WRONG_LANGUAGE not in report.applicable  # 判定していないので母数にも入れない


def test_repeated_not_applied_to_structured_payloads() -> None:
    """JSON / Python は構造上あたりまえに繰り返す。散文と同じ閾値を当てない。"""
    assert F.REPEATED not in detect('{"a": 1}', JSON_SPEC).applicable
    assert F.REPEATED in detect("ふつうの日本語の応答です。", JA_SPEC).applicable


def test_overlong_measured_without_fence_markers() -> None:
    """len(raw) で測るとフェンスを付けるモデルだけ数文字ぶん不利になる。"""
    spec = {"format": "none", "max_chars": 6}
    assert F.OVERLONG not in detect("```\nabcde\n```", spec).tags
    assert F.OVERLONG in detect("abcdefgh", spec).tags


def test_format_is_read_from_parsed_not_spec() -> None:
    """detect は Parsed.fmt を読む。spec から再導出すると二重解釈の余地が残る。"""
    parsed = parse('{"a": 1', JSON_SPEC)
    report = F.detect(parsed, {"language": "none"}, done_reason="stop", eval_count=50)
    assert F.FORMAT_BROKEN in report.tags
