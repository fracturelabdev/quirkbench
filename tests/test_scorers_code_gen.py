"""`code-gen` の採点（FLB-QB-001 §12.2・§12.10）。

**実行器は stub。コードを一切実行しない。** スコアは 0/1 なので採点ロジックの検査には
これで十分で、境界は test_sandbox.py が実機で測る。ubuntu の CI で回るのはこちら。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from quirkbench.cases import parse_case
from quirkbench.parse import parse
from quirkbench.scorers import ExecutorRequired, score
from quirkbench.scorers.code_gen import count_asserts, from_outcome

RAW = {
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
        "entry_point": "double",
        "timeout_seconds": 10,
        "test": "def check(c):\n    assert c(2) == 4\n",
        "reference": "def double(x):\n    return x * 2\n",
    },
}
CASE = parse_case(RAW, Path("t.yaml"))


@dataclass
class StubExecutor:
    """**あらかじめ用意した結果を返すだけ。** `LocalExecutor` を作らないための境界。"""

    verdict: str = "pass"
    calls: int = 0
    sandbox_applied: bool = True  # type: ignore[assignment]

    def run(self, **kwargs: object):  # type: ignore[no-untyped-def]
        self.calls += 1
        return {"verdict": self.verdict}


def _parsed(code: str):  # type: ignore[no-untyped-def]
    return parse(code, CASE.failure)


def test_executor_is_required() -> None:
    """**既定値で直接実行に落とさない。** ``executor=None → 直接実行`` は境界を消す最短経路。"""
    with pytest.raises(ExecutorRequired):
        score(_parsed("def double(x):\n    return x * 2\n"), CASE, executor=None)


def test_pass_is_one_and_fail_is_zero() -> None:
    """all-or-nothing。部分点にすると `pass@1` の定義から外れる。"""
    parsed = _parsed("def double(x):\n    return x * 2\n")
    assert score(parsed, CASE, executor=StubExecutor("pass")).score == 1.0
    assert score(parsed, CASE, executor=StubExecutor("fail")).score == 0.0


def test_syntax_error_does_not_reach_the_executor() -> None:
    """形式が壊れたコードは実行に届かない。

    節約ではなく、**exec_error と format_broken が同時に付いて原因が読めなくなる**のを避ける。
    """
    stub = StubExecutor()
    result = score(_parsed("def double(x)\n    return x\n"), CASE, executor=stub)
    assert stub.calls == 0
    assert result.score == 0.0
    assert result.sub_metrics["executed"] is False


@pytest.mark.parametrize(
    ("verdict", "tag"),
    [("exec_timeout", "exec_timeout"), ("exec_error", "exec_error")],
)
def test_execution_failures_are_tagged(verdict: str, tag: str) -> None:
    result = score(
        _parsed("def double(x):\n    return x * 2\n"), CASE, executor=StubExecutor(verdict)
    )
    assert result.tags == (tag,)
    assert result.score == 0.0


def test_applicable_is_declared_even_when_no_tag_fires() -> None:
    """「適用して 0 件」と「そもそも見ていない」を区別する（§6-B）。"""
    result = score(_parsed("def double(x):\n    return x * 2\n"), CASE, executor=StubExecutor())
    assert set(result.applicable) == {"exec_error", "exec_timeout"}


def test_assert_count_is_reported() -> None:
    assert count_asserts("def check(c):\n    assert 1\n    assert 2\n") == 2
    assert count_asserts("def check(c)\n    broken\n") == 0


def test_from_outcome_accepts_cache_rows() -> None:
    """**キャッシュ行からも同じ関数でスコアを作る**（§12.9 の規約 4）。"""
    row = {"verdict": "pass", "returncode": 0}
    assert from_outcome(row, check_source="def check(c):\n    assert 1\n").score == 1.0
