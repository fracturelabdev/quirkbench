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


# ------------------------------------- 実行結果のキャッシュ（FLB-QB-001 §12.9）


class _CountingExecutor:
    """呼び出し回数を数える stub。**コードを一切実行しない。**"""

    sandbox_applied = True

    def __init__(self, verdict: str = "pass") -> None:
        self.verdict = verdict
        self.calls = 0

    def run(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        return {"verdict": self.verdict}


def _code_case(tmp_path):  # type: ignore[no-untyped-def]
    from pathlib import Path

    from quirkbench.cases import parse_case

    raw = {
        "id": "code-probe",
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
            "entry_point": "double",
            "timeout_seconds": 10,
            "test": "def check(c):\n    assert c(2) == 4\n",
            "reference": "def double(x):\n    return x * 2\n",
        },
    }
    return parse_case(raw, Path("t.yaml"))


def _gen_row(gen_id: str, response: str) -> dict[str, object]:
    return {
        "gen_id": gen_id,
        "case_id": "code-probe",
        "model": "m",
        "model_digest": "d",
        "prompt_hash": "p",
        "seed": 1,
        "options_hash": "o",
        "attempt": 0,
        "response": response,
        "done_reason": "stop",
        "eval_count": 40,
    }


def test_identical_code_across_gen_ids_executes_once(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**同一コードが別 gen_id に現れるのは常態**（実データで 8 組中 3 組）。

    第 1 段は gen_id 単位、第 2 段はコード内容単位なので、ここでキャッシュが効く。
    """
    from quirkbench.scoring import score_run
    from quirkbench.store import RunStore

    case = _code_case(tmp_path)
    code = "def double(x):\n    return x * 2\n"
    with RunStore(tmp_path, "r") as store:
        for i in range(3):
            store.append_generation(_gen_row(f"g{i}", code))
        executor = _CountingExecutor()
        summary = score_run(store, [case], executor=executor, fingerprint="fp")
    assert summary.scored == 3
    assert executor.calls == 1, "同じコードを 3 回実行している"
    assert summary.exec_hits == 2


def test_second_score_run_executes_nothing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**S3 の完了条件。** 2 回目でサンドボックスが 1 度も起動しない。"""
    from quirkbench.scoring import score_run
    from quirkbench.store import RunStore

    case = _code_case(tmp_path)
    with RunStore(tmp_path, "r") as store:
        store.append_generation(_gen_row("g0", "def double(x):\n    return x * 2\n"))
        first = _CountingExecutor()
        score_run(store, [case], executor=first, fingerprint="fp")
        second = _CountingExecutor()
        summary = score_run(store, [case], executor=second, fingerprint="fp")
    assert first.calls == 1
    assert second.calls == 0
    assert summary.scored == 0


def test_fingerprint_change_invalidates_the_cache(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """境界が変わったら作り直す。**隔離あり時代と隔離なし時代の判定を混ぜない。**"""
    from quirkbench.scoring import _CachedExecutor
    from quirkbench.store import RunStore

    with RunStore(tmp_path, "r") as store:
        inner = _CountingExecutor()
        a = _CachedExecutor(inner, store, "fp-a")
        a.run(payload="x", check_source="c", entry_point="e", timeout_seconds=1, check_hash="h")
        b = _CachedExecutor(inner, store, "fp-b")
        b.run(payload="x", check_source="c", entry_point="e", timeout_seconds=1, check_hash="h")
    assert inner.calls == 2


def test_check_hash_change_invalidates_the_cache(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """テストを直せば再実行が走る。``scorer_version`` を直しただけでは走らない。"""
    from quirkbench.scoring import _CachedExecutor
    from quirkbench.store import RunStore

    with RunStore(tmp_path, "r") as store:
        inner = _CountingExecutor()
        cached = _CachedExecutor(inner, store, "fp")
        cached.run(
            payload="x", check_source="c", entry_point="e", timeout_seconds=1, check_hash="h1"
        )
        cached.run(
            payload="x", check_source="c", entry_point="e", timeout_seconds=1, check_hash="h2"
        )
    assert inner.calls == 2


def test_error_score_rows_are_not_marked_done(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``error`` を ``done`` に入れると、一時的な基盤失敗が永久スキップになる（§12.9）。"""
    from quirkbench.scoring import score_run
    from quirkbench.store import RunStore

    case = _code_case(tmp_path)
    with RunStore(tmp_path, "r") as store:
        store.append_generation(_gen_row("g0", "def double(x):\n    return x * 2\n"))
        store.append_score(
            {
                "gen_id": "g0",
                "scorer_version": 2,
                "check_hash": case.check_hash,
                "error": "workdir が作れない",
            }
        )
        executor = _CountingExecutor()
        summary = score_run(store, [case], executor=executor, fingerprint="fp")
    assert summary.scored == 1, "error 行が完了として扱われている"
