"""ケースのゲート（FLB-QB-001 §12.2）。

**規約は書いた時点では効かない**ので、規約ごとに「落ちること」を確かめる。
正常系だけ通しても、検査が空振りしていることに気づけない。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quirkbench.cases import parse_case
from quirkbench.lint import MAX_ASSERTS, lint

BASE = {
    "id": "probe",
    "dim": "code-gen",
    "task": "t",
    "lang": "ja",
    "prompt": "p",
    "options": {"num_predict": 512},
    "failure": {
        "format": "python",
        "extract": "fenced_or_whole",
        "language": "none",
        "min_tokens": 20,
    },
    "score": {
        "kind": "pytest",
        "entry_point": "f",
        "timeout_seconds": 30,
        "test": "def check(c):\n    assert c(1) == 1\n",
        "reference": "def f(x):\n    return x\n",
    },
}


def case(case_id: str = "probe", **over):  # type: ignore[no-untyped-def]
    raw = {**BASE, "id": case_id, "score": {**BASE["score"], **over}}
    return parse_case(raw, Path("t.yaml"))


def test_valid_case_has_no_issue() -> None:
    assert lint([case()]) == []


@pytest.mark.parametrize(
    ("check", "over"),
    [
        (
            "assert_count",
            {
                "test": "def check(c):\n"
                + "".join(f"    assert c({i}) == {i}\n" for i in range(MAX_ASSERTS + 2))
            },
        ),
        ("assert_count", {"test": "def check(c):\n    pass\n"}),
        ("reference_entry_point", {"reference": "def g(x):\n    return x\n"}),
        (
            "reference_entry_point",
            {"reference": "def f(x):\n    return x\ndef f(x):\n    return x\n"},
        ),
        ("reference_parses", {"reference": "def f(x)\n    return x\n"}),
        ("timeout_range", {"timeout_seconds": 2}),
        ("timeout_range", {"timeout_seconds": 600}),
        ("test_parses", {"test": "def check(c)\n    assert c(1)\n"}),
    ],
)
def test_broken_case_is_rejected(check: str, over: dict) -> None:
    issues = lint([case(**over)])
    assert [i.check for i in issues] == [check], f"{check} を落とせていない"


def test_assert_spread_within_dimension() -> None:
    """個々が範囲内でも、次元内で揃っていなければ残差が粒度を測る。"""
    small = case("a", test="def check(c):\n    assert c(1) == 1\n")
    large = case(
        "b",
        test="def check(c):\n" + "".join(f"    assert c({i}) == {i}\n" for i in range(MAX_ASSERTS)),
    )
    assert [i.check for i in lint([small, large])] == ["assert_spread"]


def test_non_pytest_cases_are_ignored() -> None:
    """decisive scorer のケースまで粒度規約に掛けない。"""
    raw = {
        **BASE,
        "failure": {"format": "json", "language": "none"},
        "score": {"kind": "json_schema", "schema": {"type": "object"}},
    }
    assert lint([parse_case(raw, Path("t.yaml"))]) == []


def test_bundled_cases_pass_the_gate() -> None:
    """同梱ケースは常にゲートを通る。**これが通らないものを配ってはいけない。**"""
    from quirkbench.cases import load_cases

    assert lint(load_cases(Path("cases"))) == []


# ------------------------------------------------- ideate の規約（§13.7）

IDEATE_RAW = {
    "id": "idea",
    "dim": "ideate",
    "task": "t",
    "lang": "ja",
    "prompt": "5 つ考えてください",
    "failure": {"format": "none", "count": {"n": 5, "pattern": "numbered_list"}},
    "score": {
        "kind": "ideate",
        "count": {"n": 5, "pattern": "numbered_list"},
        "topic": "小さな喫茶店の名前",
        "coverage_terms": [["珈琲"], ["街"]],
    },
}


def _ideate(**overrides):
    """`score` / `failure` を差し替えた ideate ケースを 1 件作る。"""
    from pathlib import Path

    from quirkbench.cases import parse_case

    raw = {
        **IDEATE_RAW,
        "score": {**IDEATE_RAW["score"], **overrides.pop("score", {})},
        "failure": {**IDEATE_RAW["failure"], **overrides.pop("failure", {})},
    }
    for key in overrides.pop("drop_score", []):
        raw["score"].pop(key, None)
    for key in overrides.pop("drop_failure", []):
        raw["failure"].pop(key, None)
    return parse_case(raw, Path("idea.yaml"))


def _checks(case) -> set[str]:
    return {issue.check for issue in lint([case])}


def test_valid_ideate_case_passes() -> None:
    """**まず通ることを確かめる。** 常に落ちる検査はゲートではない。"""
    assert lint([_ideate()]) == []


def test_missing_failure_count_is_rejected() -> None:
    assert "ideate_count_required" in _checks(_ideate(drop_failure=["count"]))


def test_disagreeing_counts_are_rejected() -> None:
    """**採点器と失敗検出器が別の件数を見る**（§13.2）。

    どちらも例外を出さずに**それらしい数字を返す**ので、走らせても気づけない。
    """
    case = _ideate(score={"count": {"n": 3, "pattern": "numbered_list"}})
    assert "ideate_count_agrees" in _checks(case)


def test_disagreeing_pattern_is_rejected() -> None:
    """件数が同じでも数え方が違えば別の件数を見る。"""
    case = _ideate(score={"count": {"n": 5, "pattern": "bullet_list"}})
    assert "ideate_count_agrees" in _checks(case)


def test_empty_topic_is_rejected() -> None:
    assert "ideate_topic" in _checks(_ideate(score={"topic": "   "}))


def test_missing_coverage_terms_is_rejected() -> None:
    assert "ideate_terms" in _checks(_ideate(drop_score=["coverage_terms"]))


def test_empty_coverage_group_is_rejected() -> None:
    """空グループは常に未被覆になり、coverage の上限が 1 を下回る。"""
    assert "ideate_terms" in _checks(_ideate(score={"coverage_terms": [["珈琲"], []]}))


def test_single_item_request_is_rejected() -> None:
    """1 案では diversity が原理的に 0 になる（§13.1）。

    `cases.py` の ``max_tokens >= num_predict`` と同じ「原理的に発火しない検査」の型。
    """
    case = _ideate(
        score={"count": {"n": 1, "pattern": "numbered_list"}},
        failure={"count": {"n": 1, "pattern": "numbered_list"}},
    )
    assert "ideate_n_range" in _checks(case)


# ---------------------------------- exact / numeric の規約（§15.1）

ANSWER_RAW = {
    "id": "ans",
    "dim": "code-read",
    "task": "t",
    "lang": "ja",
    "prompt": "p",
    "failure": {"format": "none"},
    "score": {"kind": "exact", "match": "equals", "expect": "5"},
}


def _answer(**score):
    from pathlib import Path

    from quirkbench.cases import parse_case

    raw = {**ANSWER_RAW, "score": {**ANSWER_RAW["score"], **score}}
    for key in list(raw["score"]):
        if raw["score"][key] is _DROP:
            del raw["score"][key]
    return parse_case(raw, Path("ans.yaml"))


class _Drop:
    pass


_DROP = _Drop()


def test_valid_exact_case_passes() -> None:
    assert lint([_answer()]) == []


def test_exact_without_expect_is_rejected() -> None:
    assert "answer_expect" in _checks(_answer(expect=_DROP))


def test_unknown_match_mode_is_rejected() -> None:
    assert "exact_match_mode" in _checks(_answer(match="fuzzy"))


def test_contains_with_a_short_answer_is_rejected() -> None:
    """**禁止であって閾値ではない**（§15.1）。

    `5` のような短い答えに `contains` を使うと、無関係な出力に偶然含まれて
    正解になる。実測で 0.5b は ``1: 2`` を返しており、期待値が `2` なら
    偶然一致していた。
    """
    assert "contains_too_short" in _checks(_answer(match="contains", expect="5"))


def test_contains_with_a_long_answer_passes() -> None:
    assert lint([_answer(match="contains", expect="MULBERRY-7")]) == []


def test_contains_checks_every_element_of_a_list() -> None:
    """1 つでも短ければ落とす。リストの後ろに短い値を隠せないようにする。"""
    case = _answer(match="contains", expect=["MULBERRY-7", "M7"])
    assert "contains_too_short" in _checks(case)


def test_numeric_with_a_non_numeric_expect_is_rejected() -> None:
    case = _answer(kind="numeric", expect="たくさん", match=_DROP)
    assert "numeric_expect" in _checks(case)


def test_negative_tolerance_is_rejected() -> None:
    case = _answer(kind="numeric", expect=400, tolerance=-1, match=_DROP)
    assert "numeric_tolerance" in _checks(case)


# ------------------------------ プロンプトが num_ctx に収まるか（§15.2）


def test_a_prompt_that_overflows_num_ctx_is_rejected() -> None:
    """**収まらないとプロンプトは黙って切り詰められる**（§15.2）。

    実測では 7,821 文字を num_ctx 4096 に投げると 2,050 トークンしか入らず、
    モデルは「文中に記述はありません」と答えた。
    """
    from pathlib import Path

    from quirkbench.cases import parse_case

    case = parse_case(
        {
            "id": "big",
            "dim": "longctx",
            "task": "t",
            "lang": "ja",
            "prompt": "あ" * 4000,
            "options": {"num_ctx": 4096},
            "failure": {"format": "none"},
            "score": {"kind": "exact", "expect": "MULBERRY-7", "match": "contains"},
        },
        Path("big.yaml"),
    )
    assert "prompt_fits_context" in _checks(case)


def test_a_prompt_with_headroom_passes() -> None:
    from pathlib import Path

    from quirkbench.cases import parse_case

    case = parse_case(
        {
            "id": "ok",
            "dim": "longctx",
            "task": "t",
            "lang": "ja",
            "prompt": "あ" * 1000,
            "options": {"num_ctx": 4096},
            "failure": {"format": "none"},
            "score": {"kind": "exact", "expect": "MULBERRY-7", "match": "contains"},
        },
        Path("ok.yaml"),
    )
    assert lint([case]) == []


def test_the_context_check_applies_to_every_dimension() -> None:
    """**longctx だけに掛けない。** 他の次元で伸ばしたときも切り詰められる。"""
    from pathlib import Path

    from quirkbench.cases import parse_case

    case = parse_case(
        {
            "id": "wide",
            "dim": "instruct",
            "task": "t",
            "lang": "ja",
            "prompt": "あ" * 4000,
            "options": {"num_ctx": 4096},
            "failure": {"format": "json"},
            "score": {"kind": "json_schema", "schema": {"type": "object"}},
        },
        Path("wide.yaml"),
    )
    assert "prompt_fits_context" in _checks(case)


# ------------------------------------------------------------ task の数え方
#
# **この 3 ルールにはテストが 1 件も無かった**（独立レビューの指摘）。
# ロジックは動いていたが、`lint()` から `_lint_tasks` の呼び出しを消しても
# 1 件も落ちない状態だった。**呼び出しを検証しないテストはガードではない。**


def _task_case(case_id: str, dim: str, task: str, pair: str | None = None):
    raw = {
        "id": case_id,
        "dim": dim,
        "task": task,
        "lang": "ja",
        "prompt": "p",
        "failure": {"format": "none", "language": "none", "min_tokens": 1},
        "score": {"kind": "exact", "match": "equals", "expect": "1"},
    }
    if pair is not None:
        raw["pair"] = pair
    return parse_case(raw, Path(f"{case_id}.yaml"))


def _task_checks(cases) -> set[str]:
    """**既存の `_checks` は Case 1 件を取る。** 同名で定義すると後勝ちで上書きされ、
    このファイルの既存テスト 15 件が黙って壊れる（実際に踏んだ）。"""
    return {issue.check for issue in lint(cases)}


def test_task_confined_to_one_dimension_passes() -> None:
    """正常系。**落ちないことも確かめる** — 常に落ちる検査はゲートにならない。"""
    cases = [_task_case("a", "reason", "alpha"), _task_case("b", "instruct", "beta")]
    assert "task_spans_dims" not in _task_checks(cases)


def test_task_spanning_two_dimensions_is_rejected() -> None:
    """**次元をまたぐ task があると観測数が数えられない**（§17.1）。"""
    cases = [_task_case("a", "reason", "same"), _task_case("b", "instruct", "same")]
    assert "task_spans_dims" in _task_checks(cases)


def test_pair_with_a_different_task_is_rejected() -> None:
    """**対訳は定義上「同じ問題を別の言語で聞いたもの」**（§14.7）。"""
    cases = [
        _task_case("a", "reason", "alpha", pair="b"),
        _task_case("b", "reason", "beta", pair="a"),
    ]
    assert "pair_task_mismatch" in _task_checks(cases)


def test_pair_with_the_same_task_passes() -> None:
    cases = [
        _task_case("a", "reason", "same", pair="b"),
        _task_case("b", "reason", "same", pair="a"),
    ]
    assert "pair_task_mismatch" not in _task_checks(cases)


def test_pair_pointing_nowhere_is_rejected() -> None:
    """**宛先の無い pair は黙って無視される。** ja_penalty の母数から静かに消える。"""
    cases = [_task_case("a", "reason", "alpha", pair="does-not-exist")]
    assert "pair_missing" in _task_checks(cases)
