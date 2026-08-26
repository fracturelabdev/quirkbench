"""`exact` と `numeric`（FLB-QB-001 §15.1）。

**抽出規則は実測から決めた。** ここで固定するのはその規則で、
実測そのものは §15.1 の表にある。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quirkbench.cases import parse_case
from quirkbench.parse import parse
from quirkbench.scorers.answer import find_numbers, normalize_answer, score_exact, score_numeric


def case_of(score: dict, lang: str = "ja"):
    return parse_case(
        {
            "id": "c",
            "dim": "reason",
            "lang": lang,
            "prompt": "p",
            "failure": {"format": "none"},
            "score": score,
        },
        Path("c.yaml"),
    )


def run_exact(text: str, **spec):
    case = case_of({"kind": "exact", **spec})
    return score_exact(parse(text, case.failure), case, executor=None, embedder=None)


def run_numeric(text: str, **spec):
    case = case_of({"kind": "numeric", **spec})
    return score_numeric(parse(text, case.failure), case, executor=None, embedder=None)


# --------------------------------------------------------------- exact


def test_exact_matches_a_bare_answer() -> None:
    assert run_exact("5", expect="5").score == 1.0


def test_exact_normalises_fullwidth() -> None:
    """全角の揺れで不正解にしない。"""
    assert run_exact("５", expect="5").score == 1.0


def test_exact_strips_surrounding_whitespace() -> None:
    assert run_exact("  5\n", expect="5").score == 1.0


def test_exact_rejects_prose_around_the_answer() -> None:
    """``equals`` は素の一致だけを採る。"""
    assert run_exact("答えは 5 行目です", expect="5").score == 0.0


def test_exact_accepts_any_of_a_list() -> None:
    assert run_exact("welder", expect=["welder", "welder at a shipyard"]).score == 1.0


def test_exact_rejects_the_real_05b_output() -> None:
    """**実測の出力を回帰として固定する**（§15.1）。

    `qwen2.5:0.5b` は行番号を聞かれて ``1: 2`` / ``2: 1.0`` を返した。
    「最初の数値を採る」抽出器を書いていたら、これは「1 と答えた」ことになる。
    """
    assert run_exact("1: 2\n2: 1.0", expect="5").score == 0.0


def test_contains_mode_accepts_a_sentence() -> None:
    """長文脈のケースはモデルが文で返す（実測）。"""
    result = run_exact(
        "第 7 倉庫の照合番号は MULBERRY-7 です。", match="contains", expect="MULBERRY-7"
    )
    assert result.score == 1.0
    assert result.sub_metrics["bare_answer"] is False


def test_contains_mode_still_needs_the_answer() -> None:
    assert run_exact("文中に記述はありません。", match="contains", expect="MULBERRY-7").score == 0.0


def test_bare_answer_is_reported_but_not_scored() -> None:
    """**「答えは知っているが黙れない」を読めるようにする**（§15.1）。

    スコアには入れない。入れると `instruct` と同じものを測る。
    """
    result = run_exact("MULBERRY-7", match="contains", expect="MULBERRY-7")
    assert result.sub_metrics["bare_answer"] is True
    verbose = run_exact("答えは MULBERRY-7 です", match="contains", expect="MULBERRY-7")
    assert verbose.score == 1.0
    assert verbose.sub_metrics["bare_answer"] is False


def test_contains_answer_is_reported_even_in_equals_mode() -> None:
    result = run_exact("答えは 5 行目です", expect="5")
    assert result.score == 0.0
    assert result.sub_metrics["contains_answer"] is True


# ------------------------------------------------------------- numeric


def test_numeric_matches_a_bare_number() -> None:
    assert run_numeric("400", expect=400).score == 1.0


def test_numeric_takes_the_last_number_not_the_first() -> None:
    """**最初を採ると問題文の数値を書くモデルに点が入る**（§15.1）。"""
    result = run_numeric("1000 - 150 * 4 = 400", expect=400)
    assert result.score == 1.0
    assert result.sub_metrics["extracted"] == 400.0
    assert result.sub_metrics["numbers_found"] == 4


def test_numeric_ignores_a_trailing_unit() -> None:
    """実測で `qwen2.5:3b` は ``10円`` を返した。単位で落とさない。"""
    assert run_numeric("400円", expect=400).score == 1.0


def test_numeric_handles_thousands_separators() -> None:
    assert run_numeric("1,200", expect=1200).score == 1.0


def test_numeric_handles_fullwidth_digits() -> None:
    assert run_numeric("４００", expect=400).score == 1.0


def test_numeric_rejects_a_wrong_last_number() -> None:
    """実測の 0.5b: 説明の末尾が誤答だった。"""
    assert (
        run_numeric("1000 円 - 150 円 = 850 円。おつりは 850 円でした。", expect=400).score == 0.0
    )


def test_numeric_with_no_number_scores_zero() -> None:
    result = run_numeric("分かりません", expect=400)
    assert result.score == 0.0
    assert result.sub_metrics["extracted"] is None


def test_numeric_tolerance_is_absolute() -> None:
    """**相対誤差にしない** — 答えが 0 のとき定義できない（§15.1）。"""
    assert run_numeric("3.14", expect=3.14159, tolerance=0.01).score == 1.0
    assert run_numeric("3.1", expect=3.14159, tolerance=0.01).score == 0.0


def test_numeric_tolerance_defaults_to_exact() -> None:
    assert run_numeric("399", expect=400).score == 0.0


def test_numeric_bare_answer_flag() -> None:
    assert run_numeric("400", expect=400).sub_metrics["bare_answer"] is True
    assert run_numeric("答えは 400 です", expect=400).sub_metrics["bare_answer"] is False


def test_numeric_handles_negative_numbers() -> None:
    assert run_numeric("-5", expect=-5).score == 1.0


# ------------------------------------------------------------- 部品


def test_find_numbers_returns_every_run() -> None:
    assert find_numbers("a1 b2.5 c-3") == [1.0, 2.5, -3.0]


def test_normalize_answer_is_nfkc() -> None:
    assert normalize_answer(" ＭＵＬＢＥＲＲＹ ") == "MULBERRY"


def test_no_failure_vocabulary_is_owned_here() -> None:
    assert run_exact("x", expect="y").tags == ()
    assert run_numeric("x", expect=1).tags == ()


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_response_scores_zero(text: str) -> None:
    assert run_exact(text, expect="5").score == 0.0
    assert run_numeric(text, expect=5).score == 0.0
