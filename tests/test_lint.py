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
