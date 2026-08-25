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
