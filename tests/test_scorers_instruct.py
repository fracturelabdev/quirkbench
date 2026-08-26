"""instruct 次元の採点。入力→期待スコアの表駆動で検証する。

**スコアは乗算**（schema_ok × expect 一致率）。魔法の重みを置かない。
「JSON ですらなかった」と「JSON だが中身が違う」の区別はスコアではなく
失敗型が持つので、どちらも 0.0 になるのが正しい。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quirkbench.cases import parse_case
from quirkbench.parse import parse
from quirkbench.scorers import score
from quirkbench.scorers.instruct import score_json_schema

SCHEMA = {
    "type": "object",
    "required": ["name", "age", "occupation"],
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "occupation": {"type": "string"},
    },
}

CASE = parse_case(
    {
        "id": "c",
        "dim": "instruct",
        "lang": "ja",
        "prompt": "p",
        "failure": {"format": "json", "extract": "fenced_or_first_object"},
        "score": {"kind": "json_schema", "schema": SCHEMA, "expect": {"age": 42}},
    },
    Path("c.yaml"),
)

GOOD = '```json\n{"name": "田中一郎", "age": 42, "occupation": "溶接工"}\n```'
# 実測: qwen2.5:0.5b は日本語で聞かれると JSON のキー名まで訳す
TRANSLATED_KEYS = '```json\n{"名前": "田中一郎", "年齢": 42, "職業": "溶接工"}\n```'
WRONG_VALUE = '{"name": "田中一郎", "age": 24, "occupation": "溶接工"}'
AGE_AS_STRING = '{"name": "田中一郎", "age": "42", "occupation": "溶接工"}'
NOT_JSON = "田中一郎さんは 42 歳の溶接工です。"


def run(text: str):
    return score_json_schema(parse(text, CASE.failure), CASE, executor=None, embedder=None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (GOOD, 1.0),
        (TRANSLATED_KEYS, 0.0),  # schema 不合格
        (WRONG_VALUE, 0.0),  # schema は通るが expect が外れる
        (AGE_AS_STRING, 0.0),  # 型が違う
        (NOT_JSON, 0.0),  # そもそも JSON でない
    ],
)
def test_scores(text: str, expected: float) -> None:
    assert run(text).score == expected


def test_scorer_does_not_own_failure_vocabulary() -> None:
    """**scorer は失敗型のタグを付けない。** format_broken は failures.py が所有する。

    ここで append すると、failures が empty で短絡した空応答にも format_broken が
    付き、「JSON を壊す癖」の母数に「何も出さない癖」が載る（Grok の指摘）。
    """
    result = run(NOT_JSON)
    assert result.tags == ()
    assert result.sub_metrics["parse_ok"] is False


def test_json_null_is_scored_not_rejected() -> None:
    """null は正当な JSON なので parse_ok は True。schema 違反として 0 点になる。"""
    result = run("null")
    assert result.sub_metrics["parse_ok"] is True
    assert result.sub_metrics["schema_ok"] is False
    assert result.score == 0.0


def test_schema_errors_are_recorded() -> None:
    """キー名を訳したことが sub_metrics に残る。ここが癖の実体になる。"""
    result = run(TRANSLATED_KEYS)
    assert result.sub_metrics["schema_ok"] is False
    assert any("name" in error for error in result.sub_metrics["schema_errors"])


def test_schema_failure_zeroes_a_matching_expect() -> None:
    """値は当てたが構造を守らなかった場合は 0。

    ``occupation`` が欠けているので schema は不合格。しかし expect の ``age`` は
    合っている。乗算にしていないと、**構造を無視したまま部分点が出てしまう**。
    instruct 次元は「形式制約が守れるか」を測る次元なので、これは 0 でなければ
    次元の定義そのものが崩れる。
    """
    parsed = parse('{"name": "田中一郎", "age": 42}', CASE.failure)
    result = score_json_schema(parsed, CASE, executor=None, embedder=None)
    assert result.sub_metrics["schema_ok"] is False
    assert result.sub_metrics["expect_matched"] == 1
    assert result.score == 0.0


def test_partial_credit_from_expect() -> None:
    case = parse_case(
        {
            "id": "d",
            "dim": "instruct",
            "lang": "ja",
            "prompt": "p",
            "failure": {"format": "json"},
            "score": {
                "kind": "json_schema",
                "schema": SCHEMA,
                "expect": {"age": 42, "name": "田中一郎"},
            },
        },
        Path("d.yaml"),
    )
    parsed = parse('{"name": "山田", "age": 42, "occupation": "溶接工"}', case.failure)
    result = score_json_schema(parsed, case, executor=None, embedder=None)
    assert result.score == 0.5
    assert result.sub_metrics["expect_missed"] == ["name"]


def test_no_expect_means_schema_only() -> None:
    case = parse_case(
        {
            "id": "e",
            "dim": "instruct",
            "lang": "ja",
            "prompt": "p",
            "failure": {"format": "json"},
            "score": {"kind": "json_schema", "schema": SCHEMA},
        },
        Path("e.yaml"),
    )
    assert (
        score_json_schema(parse(GOOD, case.failure), case, executor=None, embedder=None).score
        == 1.0
    )


def test_fullwidth_value_still_matches() -> None:
    """全角半角の揺れで不正解にしない。"""
    case = parse_case(
        {
            "id": "f",
            "dim": "instruct",
            "lang": "ja",
            "prompt": "p",
            "failure": {"format": "json"},
            "score": {
                "kind": "json_schema",
                "schema": {"type": "object"},
                "expect": {"name": "タナカ"},
            },
        },
        Path("f.yaml"),
    )
    parsed = parse('{"name": "ﾀﾅｶ"}', case.failure)
    assert score_json_schema(parsed, case, executor=None, embedder=None).score == 1.0


def test_bool_expect_is_strict() -> None:
    case = parse_case(
        {
            "id": "g",
            "dim": "instruct",
            "lang": "ja",
            "prompt": "p",
            "failure": {"format": "json"},
            "score": {"kind": "json_schema", "schema": {"type": "object"}, "expect": {"ok": True}},
        },
        Path("g.yaml"),
    )

    def run_one(text: str) -> float:
        return score_json_schema(
            parse(text, case.failure), case, executor=None, embedder=None
        ).score

    assert run_one('{"ok": true}') == 1.0
    assert run_one('{"ok": 1}') == 0.0


def test_registry_dispatch() -> None:
    assert score(parse(GOOD, CASE.failure), CASE).score == 1.0


def test_every_declared_kind_has_a_scorer() -> None:
    """**`SCORE_KINDS` と registry がずれていないこと。**

    S6 で 6 種すべてが実装され、`ScorerNotImplemented` は正規の経路からは
    到達できなくなった。**ガードを消すのではなく、ずれを検査に変える** —
    `SCORE_KINDS` に名前を足して採点器を書き忘れると、
    そのケースは「採点されないまま数えられない」で終わる。
    """
    from quirkbench.cases import SCORE_KINDS
    from quirkbench.scorers import _registry

    assert set(SCORE_KINDS) == set(_registry())


def test_unimplemented_kind_raises() -> None:
    """ガードそのものは残す。上の検査が落ちる前の最後の網になる。"""
    import dataclasses

    from quirkbench.scorers import ScorerNotImplemented

    case = parse_case(
        {"id": "h", "dim": "reason", "lang": "ja", "prompt": "p", "score": {"kind": "numeric"}},
        Path("h.yaml"),
    )
    # `parse_case` は未知の kind を弾くので、検証を通したあとで差し替える
    broken = dataclasses.replace(case, score={"kind": "not-implemented-yet"})
    with pytest.raises(ScorerNotImplemented, match="not-implemented-yet"):
        score(parse("x", broken.failure), broken)
