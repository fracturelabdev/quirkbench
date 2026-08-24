"""採点のオーケストレーション。

検証したいのは主に 3 点:
1. 採点は追記のみで、generations.jsonl を書き換えない
2. 同じ (gen_id, scorer_version, check_hash) を二重に採点しない
3. 採点しなかったものを**黙って落とさない**（未実装 kind・ケース欠落・生成失敗）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quirkbench import scorers
from quirkbench.cases import parse_case
from quirkbench.scoring import score_generation, score_run
from quirkbench.store import RunStore

SCHEMA = {"type": "object", "required": ["age"], "properties": {"age": {"type": "integer"}}}

CASE = parse_case(
    {
        "id": "c1",
        "dim": "instruct",
        "lang": "ja",
        "prompt": "p",
        "failure": {"format": "json", "extract": "fenced_or_first_object", "min_tokens": 4},
        "score": {"kind": "json_schema", "schema": SCHEMA, "expect": {"age": 42}},
    },
    Path("c1.yaml"),
)
IDEATE_CASE = parse_case(
    {"id": "i1", "dim": "ideate", "lang": "ja", "prompt": "p", "score": {"kind": "ideate"}},
    Path("i1.yaml"),
)


def gen(gen_id: str, response: str, *, case_id: str = "c1", **extra) -> dict:
    row = {
        "gen_id": gen_id,
        "case_id": case_id,
        "response": response,
        "done_reason": "stop",
        "eval_count": 30,
    }
    row.update(extra)
    return row


def test_score_generation_shape() -> None:
    row = score_generation(gen("g1", '{"age": 42}'), CASE)
    assert row["gen_id"] == "g1"
    assert row["score"] == 1.0
    assert row["scorer_version"] == scorers.SCORER_VERSION
    assert row["check_hash"] == CASE.check_hash
    assert row["failure_tags"] == []
    assert "format_broken" in row["failure_applicable"]


def test_score_generation_tags_truncated_and_format_broken() -> None:
    row = score_generation(gen("g2", '{"age": 4', done_reason="length"), CASE)
    assert row["score"] == 0.0
    assert set(row["failure_tags"]) >= {"truncated", "format_broken"}


def test_non_attempt_reaches_the_row() -> None:
    row = score_generation(gen("g3", "むり", eval_count=2), CASE)
    assert "non_attempt" in row["failure_tags"]


def test_parse_happens_once_and_agrees() -> None:
    """scorer と失敗検出器が同じ Parsed を見る。食い違いを構造で塞いでいる。"""
    row = score_generation(gen("g4", 'はい。\n```json\n{"age": 42}\n```'), CASE)
    assert row["score"] == 1.0
    assert "preamble" in row["failure_tags"]
    assert "format_broken" not in row["failure_tags"]


# -------------------------------------------------------------------- run


@pytest.fixture()
def store(tmp_path: Path):
    with RunStore(tmp_path, "r1") as opened:
        yield opened


def test_score_run_appends_and_is_idempotent(store: RunStore) -> None:
    store.append_generation(gen("g1", '{"age": 42}'))
    store.append_generation(gen("g2", '{"age": 7}'))

    first = score_run(store, [CASE])
    assert (first.total, first.scored, first.skipped) == (2, 2, 0)

    second = score_run(store, [CASE])
    assert (second.total, second.scored, second.skipped) == (2, 0, 2)

    rows, _ = store.scores()
    assert len(rows) == 2


def test_generations_are_not_rewritten(store: RunStore) -> None:
    store.append_generation(gen("g1", '{"age": 42}'))
    before = (store.dir / "generations.jsonl").read_bytes()
    score_run(store, [CASE])
    assert (store.dir / "generations.jsonl").read_bytes() == before


def test_check_hash_change_triggers_rescore(store: RunStore) -> None:
    """期待値を直したら再採点する。生成はやり直さない。"""
    store.append_generation(gen("g1", '{"age": 42}'))
    score_run(store, [CASE])

    revised = parse_case(
        {
            "id": "c1",
            "dim": "instruct",
            "lang": "ja",
            "prompt": "p",
            "failure": {"format": "json", "extract": "fenced_or_first_object", "min_tokens": 4},
            "score": {"kind": "json_schema", "schema": SCHEMA, "expect": {"age": 7}},
        },
        Path("c1.yaml"),
    )
    assert revised.check_hash != CASE.check_hash
    again = score_run(store, [revised])
    assert again.scored == 1

    rows, _ = store.scores()
    assert len(rows) == 2
    assert [r["score"] for r in rows] == [1.0, 0.0]


def test_unsupported_kind_is_reported_not_dropped(store: RunStore) -> None:
    store.append_generation(gen("g1", "1. あ", case_id="i1"))
    summary = score_run(store, [CASE, IDEATE_CASE])
    assert summary.scored == 0
    assert summary.unsupported == {"ideate": 1}


def test_missing_case_is_reported(store: RunStore) -> None:
    store.append_generation(gen("g1", "x", case_id="gone"))
    summary = score_run(store, [CASE])
    assert summary.missing_cases == {"gone": 1}
    assert summary.scored == 0


def test_generation_errors_are_skipped(store: RunStore) -> None:
    store.append_generation(gen("g1", "", error="timeout"))
    summary = score_run(store, [CASE])
    assert summary.generation_errors == 1
    assert summary.scored == 0
    assert store.scores()[0] == []


def test_scores_are_valid_jsonl(store: RunStore) -> None:
    store.append_generation(gen("g1", '{"age": 42}'))
    score_run(store, [CASE])
    text = (store.dir / "scores.jsonl").read_text(encoding="utf-8")
    assert all(json.loads(line) for line in text.splitlines())


# ------------------------------------------------- レビュー由来の回帰（実測で再現済み）


def test_empty_short_circuit_survives_the_merge() -> None:
    """空応答に format_broken を付けない。

    detect() 単体では短絡できていたが、scorer が独立に format_broken を append
    していたため結合後に破れていた。「JSON を壊す癖」の母数に「何も出さない癖」が
    載る経路だった（Grok の指摘・実測で再現）。
    """
    row = score_generation(gen("g1", "", eval_count=0), CASE)
    assert row["failure_tags"] == ["empty", "non_attempt"]
    assert "format_broken" not in row["failure_applicable"]


def test_score_rows_carry_a_timestamp() -> None:
    """scorer_version 同値・check_hash 違いの行を時刻で決着させるため。"""
    assert score_generation(gen("g1", '{"age": 42}'), CASE)["ts"]


def test_double_score_within_one_run_is_prevented(store: RunStore) -> None:
    """同じ gen_id の成功行が 2 行ある run でも、採点は 1 行だけ。"""
    store.append_generation(gen("g1", '{"age": 42}'))
    store.append_generation(gen("g1", '{"age": 42}'))
    summary = score_run(store, [CASE])
    assert summary.scored == 1
    assert len(store.scores()[0]) == 1
