"""`json_keys`（FLB-QB-001 §15.1）。

`json_schema` との違いは**構造の門が無い**こと。`extract` が測るのは
「拾えたか」であって「形式を守れたか」ではない。
"""

from __future__ import annotations

from pathlib import Path

from quirkbench.cases import parse_case
from quirkbench.parse import parse
from quirkbench.scorers.extract import score_json_keys

EXPECT = {"name": "山田花子", "age": 34, "city": "札幌市"}


def case_of(expect=None):
    return parse_case(
        {
            "id": "c",
            "dim": "extract",
            "task": "t",
            "lang": "ja",
            "prompt": "p",
            "failure": {"format": "json", "extract": "fenced_or_first_object"},
            "score": {"kind": "json_keys", "expect": EXPECT if expect is None else expect},
        },
        Path("c.yaml"),
    )


def run(text: str, expect=None):
    case = case_of(expect)
    return score_json_keys(parse(text, case.failure), case, executor=None, embedder=None)


def test_all_keys_matched() -> None:
    assert run('{"name": "山田花子", "age": 34, "city": "札幌市"}').score == 1.0


def test_partial_credit_is_the_point() -> None:
    """**キー単位の一致率**。`json_schema` の乗算とは違う。"""
    result = run('{"name": "山田花子", "age": 34, "city": "東京"}')
    assert result.score == 2 / 3
    assert result.sub_metrics["expect_missed"] == ["city"]


def test_a_missing_key_is_not_a_match() -> None:
    assert run('{"name": "山田花子"}').score == 1 / 3


def test_extra_keys_are_not_penalised() -> None:
    """**減点すると `instruct` と同じものを測り始める**（§15.1）。"""
    result = run('{"name": "山田花子", "age": 34, "city": "札幌市", "hobby": "登山"}')
    assert result.score == 1.0
    assert result.sub_metrics["extra_keys"] == ["hobby"]


def test_real_05b_output_loses_only_the_name() -> None:
    """実測: 0.5b は ``"山田花子さん"`` と敬称つきで返した（§15.1 の probe）。"""
    result = run('{"name": "山田花子さん", "age": 34, "city": "札幌市"}')
    assert result.score == 2 / 3
    assert result.sub_metrics["expect_missed"] == ["name"]


def test_fullwidth_values_still_match() -> None:
    """値の比較は `instruct` の ``_equal`` を共有する。"""
    assert run('{"name": "ﾔﾏﾀﾞ", "age": 34, "city": "札幌市"}', {"name": "ヤマダ"}).score == 1.0


def test_broken_json_scores_zero_without_tags() -> None:
    """**format_broken は failures.py が単独で所有する。**"""
    result = run("これは JSON ではありません")
    assert result.score == 0.0
    assert result.tags == ()
    assert result.sub_metrics["parse_ok"] is False


def test_a_json_array_is_not_a_key_map() -> None:
    """トップレベルが配列なら拾えない。0 点にして型を確認できるようにする。"""
    result = run("[1, 2, 3]")
    assert result.score == 0.0
    assert result.sub_metrics["expect_matched"] == 0


def test_string_number_is_not_an_integer() -> None:
    """型は厳密に見る（`instruct` と同じ規則）。"""
    assert run('{"name": "山田花子", "age": "34", "city": "札幌市"}').score == 2 / 3


def test_fenced_json_is_extracted() -> None:
    body = '```json\n{"name": "山田花子", "age": 34, "city": "札幌市"}\n```'
    assert run(body).score == 1.0
