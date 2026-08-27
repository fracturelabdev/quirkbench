"""レポートの集計（FLB-QB-001 §14）。

**残差と σ の計算はここだけで検証する。** レンダリング側は表の形を見る。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quirkbench.cases import parse_case
from quirkbench.report.aggregate import (
    SAME_SCORE_EPSILON,
    InconsistentRun,
    aggregate,
)
from quirkbench.store import RunStore

SCHEMA = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}


def make_case(case_id: str, dim: str = "instruct", lang: str = "ja", pair: str | None = None):
    raw = {
        "id": case_id,
        "dim": dim,
        "task": "t",
        "lang": lang,
        "prompt": "p",
        "failure": {"format": "json"},
        "score": {"kind": "json_schema", "schema": SCHEMA},
    }
    if pair:
        raw["pair"] = pair
    return parse_case(raw, Path(f"{case_id}.yaml"))


def build(tmp_path, rows, *, run="r", version="0.32.13", digests=None):
    """``rows`` は ``(model, case_id, seed, score)``。生成と採点を両方書く。"""
    store = RunStore(tmp_path, run)
    store.dir.mkdir(parents=True, exist_ok=True)
    for index, (model, case_id, seed, score) in enumerate(rows):
        gen_id = f"g{index}"
        store.append_generation(
            {
                "gen_id": gen_id,
                "model": model,
                "model_digest": (digests or {}).get((model, index), f"digest-{model}"),
                "ollama_version": version,
                "case_id": case_id,
                "seed": seed,
                "eval_count": 10,
                "eval_duration_ns": 100_000_000,
                "prompt_eval_count": 5,
                "prompt_eval_duration_ns": 10_000_000,
                "load_duration_ns": 50_000_000,
                "response": "x",
            }
        )
        store.append_score(
            {
                "gen_id": gen_id,
                "ts": f"2026-08-25T00:00:{index:02d}+00:00",
                "scorer_version": 2,
                "check_hash": "CH:" + case_id,
                "score": score,
                "sub_metrics": {},
                "failure_tags": [],
                "failure_applicable": [],
            }
        )
    return store


def cases_for(ids, **kw):
    out = []
    for case_id in ids:
        case = make_case(case_id, **kw.get(case_id, {}))
        out.append(
            type(case)(
                **{
                    **{f.name: getattr(case, f.name) for f in case.__dataclass_fields__.values()},
                    "check_hash": "CH:" + case_id,
                }
            )
        )
    return out


# ------------------------------------------------------------ 読み出し規則


def test_takes_the_highest_scorer_version(tmp_path) -> None:
    store = build(tmp_path, [("m", "c", 0, 0.0)])
    store.append_score(
        {
            "gen_id": "g0",
            "ts": "2026-08-25T00:00:00+00:00",
            "scorer_version": 3,
            "check_hash": "CH:c",
            "score": 1.0,
            "failure_tags": [],
            "failure_applicable": [],
        }
    )
    agg = aggregate(store, cases_for(["c"]))
    assert agg.cases[0].by_model["m"] == 1.0


def test_ties_on_version_are_broken_by_ts(tmp_path) -> None:
    """**期待値を直すと同一 version で `check_hash` 違いの行ができる**（§9）。"""
    store = build(tmp_path, [("m", "c", 0, 0.0)])
    store.append_score(
        {
            "gen_id": "g0",
            "ts": "2026-08-25T09:00:00+00:00",
            "scorer_version": 2,
            "check_hash": "CH:c",
            "score": 1.0,
            "failure_tags": [],
            "failure_applicable": [],
        }
    )
    agg = aggregate(store, cases_for(["c"]))
    assert agg.cases[0].by_model["m"] == 1.0


def test_rows_with_a_stale_check_hash_are_ignored(tmp_path) -> None:
    store = build(tmp_path, [("m", "c", 0, 1.0)])
    store.append_score(
        {
            "gen_id": "g0",
            "ts": "2026-08-25T09:00:00+00:00",
            "scorer_version": 9,
            "check_hash": "OLD",
            "score": 0.0,
            "failure_tags": [],
            "failure_applicable": [],
        }
    )
    agg = aggregate(store, cases_for(["c"]))
    assert agg.cases[0].by_model["m"] == 1.0


def test_unscored_generations_are_counted_not_dropped(tmp_path) -> None:
    """**黙って落とすと、採点し直し忘れに気づけない**（§14.2）。"""
    store = build(tmp_path, [("m", "c", 0, 1.0)])
    store.append_generation(
        {
            "gen_id": "orphan",
            "model": "m",
            "model_digest": "digest-m",
            "ollama_version": "0.32.13",
            "case_id": "c",
            "seed": 0,
        }
    )
    agg = aggregate(store, cases_for(["c"]))
    assert agg.unscored == 1


# ------------------------------------------------------ 測定条件のずれ


def test_digest_drift_within_a_run_is_rejected(tmp_path) -> None:
    """**差し引かれるのが難易度でなくなる**（§14.1）。"""
    store = build(tmp_path, [("m", "c", 0, 1.0)])
    store.append_generation(
        {
            "gen_id": "x",
            "model": "m",
            "model_digest": "DIFFERENT",
            "ollama_version": "0.32.13",
            "case_id": "c",
            "seed": 1,
        }
    )
    with pytest.raises(InconsistentRun, match="モデルの中身が変わっている"):
        aggregate(store, cases_for(["c"]))


def test_mixed_ollama_versions_are_rejected(tmp_path) -> None:
    store = build(tmp_path, [("m", "c", 0, 1.0)])
    store.append_generation(
        {
            "gen_id": "x",
            "model": "m",
            "model_digest": "digest-m",
            "ollama_version": "9.9.9",
            "case_id": "c",
            "seed": 1,
        }
    )
    with pytest.raises(InconsistentRun, match="ollama のバージョン"):
        aggregate(store, cases_for(["c"]))


# --------------------------------------------------------- 識別力の判定


def test_all_models_at_ceiling_is_excluded(tmp_path) -> None:
    store = build(tmp_path, [("a", "c", 0, 1.0), ("b", "c", 0, 1.0)])
    agg = aggregate(store, cases_for(["c"]))
    assert agg.cases[0].discriminating is False
    assert agg.cases[0].degenerate_kind == "ceiling"


def test_all_models_at_floor_is_reported_separately(tmp_path) -> None:
    """**同じ「除外」でも次の手が違う**（§14.3）。緩いのか、難しすぎるのか。"""
    store = build(tmp_path, [("a", "c", 0, 0.0), ("b", "c", 0, 0.0)])
    agg = aggregate(store, cases_for(["c"]))
    assert agg.cases[0].degenerate_kind == "floor"


def test_all_models_equal_but_mid_is_flat(tmp_path) -> None:
    store = build(tmp_path, [("a", "c", 0, 0.5), ("b", "c", 0, 0.5)])
    agg = aggregate(store, cases_for(["c"]))
    assert agg.cases[0].degenerate_kind == "flat"


def test_a_difference_below_epsilon_is_not_discriminating(tmp_path) -> None:
    """**厳密比較にすると浮動小数の最下位ビットで識別力ありと誤判定する**（§14.3）。"""
    store = build(tmp_path, [("a", "c", 0, 0.5), ("b", "c", 0, 0.5 + SAME_SCORE_EPSILON / 10)])
    agg = aggregate(store, cases_for(["c"]))
    assert agg.cases[0].discriminating is False


def test_a_real_difference_is_discriminating(tmp_path) -> None:
    store = build(tmp_path, [("a", "c", 0, 0.5), ("b", "c", 0, 0.6)])
    agg = aggregate(store, cases_for(["c"]))
    assert agg.cases[0].discriminating is True


# ---------------------------------------------------------- 残差と σ


def test_residual_cancels_case_difficulty(tmp_path) -> None:
    """全モデルが落とすケースは、全員の残差が 0 付近になる（§3）。"""
    store = build(
        tmp_path,
        [
            ("a", "hard", 0, 0.1),
            ("b", "hard", 0, 0.1 + 1e-3),
            ("a", "easy", 0, 0.9),
            ("b", "easy", 0, 0.9 + 1e-3),
        ],
    )
    agg = aggregate(store, cases_for(["hard", "easy"]))
    dim = agg.dims[0]
    assert dim.residual_by_model["a"] == pytest.approx(-5e-4)
    assert dim.residual_by_model["b"] == pytest.approx(5e-4)


def test_non_discriminating_cases_do_not_enter_the_residual(tmp_path) -> None:
    store = build(
        tmp_path,
        [
            ("a", "flat", 0, 1.0),
            ("b", "flat", 0, 1.0),
            ("a", "real", 0, 0.0),
            ("b", "real", 0, 1.0),
        ],
    )
    agg = aggregate(store, cases_for(["flat", "real"]))
    dim = agg.dims[0]
    assert dim.discriminating_cases == 1
    assert dim.excluded_cases == 1
    assert dim.residual_by_model["a"] == pytest.approx(-0.5)


def test_absolute_keeps_every_case(tmp_path) -> None:
    """**「全モデルが満点だった」もモデルの事実**なので絶対値からは外さない。"""
    store = build(
        tmp_path,
        [
            ("a", "flat", 0, 1.0),
            ("b", "flat", 0, 1.0),
            ("a", "real", 0, 0.0),
            ("b", "real", 0, 1.0),
        ],
    )
    agg = aggregate(store, cases_for(["flat", "real"]))
    assert agg.dims[0].absolute_by_model["a"] == pytest.approx(0.5)


def test_zero_sigma_yields_no_z_rather_than_zero(tmp_path) -> None:
    """**0 を返すと「差が無い」と読める**が、実際は「測れていない」（§14.4）。"""
    store = build(tmp_path, [("a", "c", 0, 1.0), ("b", "c", 0, 1.0)])
    agg = aggregate(store, cases_for(["c"]))
    dim = agg.dims[0]
    assert dim.sd == 0.0
    assert dim.z_by_model == {}
    assert dim.measurable is False


def test_z_is_marked_untrusted_with_one_case(tmp_path) -> None:
    store = build(tmp_path, [("a", "c", 0, 0.0), ("b", "c", 0, 1.0)])
    assert aggregate(store, cases_for(["c"])).dims[0].z_is_trusted is False


def test_z_is_trusted_with_two_cases(tmp_path) -> None:
    store = build(
        tmp_path,
        [("a", "c1", 0, 0.0), ("b", "c1", 0, 1.0), ("a", "c2", 0, 0.2), ("b", "c2", 0, 0.9)],
    )
    assert aggregate(store, cases_for(["c1", "c2"])).dims[0].z_is_trusted is True


def test_sigma_is_population_not_sample(tmp_path) -> None:
    """標本推定にすると n が小さいとき σ が膨らみ、**z が縮んで「差が無い」方向に誤る**。

    §14.4 の判断。母集団標準偏差を使うことを値で固定する。
    """
    store = build(tmp_path, [("a", "c", 0, 0.0), ("b", "c", 0, 1.0)])
    dim = aggregate(store, cases_for(["c"])).dims[0]
    # 残差は ±0.5。母集団 sd = 0.5、標本 sd = 0.7071
    assert dim.sd == pytest.approx(0.5)
    assert dim.z_by_model["b"] == pytest.approx(1.0)


# ------------------------------------------------------- 反復ばらつき


def test_spread_separates_stable_from_lucky(tmp_path) -> None:
    """**「安定して弱い」と「たまに当たる」を分ける**（§14.6）。"""
    store = build(
        tmp_path,
        [
            ("stable", "c", 0, 0.0),
            ("stable", "c", 1, 0.0),
            ("lucky", "c", 0, 0.0),
            ("lucky", "c", 1, 1.0),
        ],
    )
    dim = aggregate(store, cases_for(["c"])).dims[0]
    assert dim.spread_by_model["stable"] == 0.0
    assert dim.spread_by_model["lucky"] == pytest.approx(0.5)
    # 平均スコアは違うが、どちらも「弱い」側にいる
    assert dim.absolute_by_model["stable"] == 0.0


# ---------------------------------------------------------- ja_penalty


def test_ja_penalty_needs_both_sides(tmp_path) -> None:
    store = build(tmp_path, [("m", "c-ja", 0, 0.0), ("m", "c-en", 0, 1.0)])
    cases = [
        make_case("c-ja", lang="ja", pair="c-en"),
        make_case("c-en", lang="en", pair="c-ja"),
    ]
    cases = [
        type(c)(
            **{
                **{f.name: getattr(c, f.name) for f in c.__dataclass_fields__.values()},
                "check_hash": "CH:" + c.id,
            }
        )
        for c in cases
    ]
    dim = aggregate(store, cases).dims[0]
    assert dim.ja_pairs == 1
    # **符号は en − ja**（§4・§15.3）。ja=0.0 / en=1.0 なので penalty は +1.0。
    # 「penalty」は正の値が罰であるべきで、逆にすると読み手が毎回符号を反転させる
    assert dim.ja_penalty_by_model["m"] == pytest.approx(+1.0)


def test_a_dimension_without_pairs_has_no_penalty_not_zero(tmp_path) -> None:
    """**0 は「差が無い」であって「測っていない」ではない**（§14.7）。"""
    store = build(tmp_path, [("m", "c", 0, 0.5)])
    dim = aggregate(store, cases_for(["c"])).dims[0]
    assert dim.ja_pairs == 0
    assert dim.ja_penalty_by_model == {}


def test_a_pair_missing_from_the_run_is_skipped(tmp_path) -> None:
    store = build(tmp_path, [("m", "c-ja", 0, 0.0)])
    case = make_case("c-ja", lang="ja", pair="c-en")
    case = type(case)(
        **{
            **{f.name: getattr(case, f.name) for f in case.__dataclass_fields__.values()},
            "check_hash": "CH:c-ja",
        }
    )
    dim = aggregate(store, [case]).dims[0]
    assert dim.ja_pairs == 0
    assert dim.ja_penalty_by_model == {}


# --------------------------------------------------------------- 性能


def test_throughput_is_summed_not_averaged(tmp_path) -> None:
    """**行ごとに tok/s を出してから平均すると「短い応答の速さ」を測る**（§14.9）。"""
    store = RunStore(tmp_path, "r")
    store.dir.mkdir(parents=True, exist_ok=True)
    # **数値を非対称にする。** (1, 1s) と (99, 1s) だと総和も行ごと平均も 50 になり、
    # 変異検査で「行ごとに平均する」が生き残る（実際に生き残った）
    for index, (count, ns) in enumerate([(10, 1_000_000_000), (10, 9_000_000_000)]):
        store.append_generation(
            {
                "gen_id": f"g{index}",
                "model": "m",
                "model_digest": "d",
                "ollama_version": "0.32.13",
                "case_id": "c",
                "seed": index,
                "eval_count": count,
                "eval_duration_ns": ns,
                "prompt_eval_count": 1,
                "prompt_eval_duration_ns": 1_000_000_000,
                "load_duration_ns": 0,
            }
        )
        store.append_score(
            {
                "gen_id": f"g{index}",
                "ts": "t",
                "scorer_version": 2,
                "check_hash": "CH:c",
                "score": 1.0,
                "failure_tags": [],
                "failure_applicable": [],
            }
        )
    perf = aggregate(store, cases_for(["c"])).perf[0]
    # 総和: 20 トークン / 10 秒 = 2.0
    # 行ごとの平均なら (10.0 + 1.111) / 2 = 5.56 になる
    assert perf.tokens_per_second == pytest.approx(2.0)
    assert perf.tokens_per_second != pytest.approx(5.56, abs=0.01)


def test_load_uses_median_not_mean(tmp_path) -> None:
    """**run の最初の 1 件だけモデルのロードが入る**（§14.9）。"""
    store = RunStore(tmp_path, "r")
    store.dir.mkdir(parents=True, exist_ok=True)
    for index, load in enumerate([5_000_000_000, 1_000_000, 1_000_000]):
        store.append_generation(
            {
                "gen_id": f"g{index}",
                "model": "m",
                "model_digest": "d",
                "ollama_version": "0.32.13",
                "case_id": "c",
                "seed": index,
                "eval_count": 1,
                "eval_duration_ns": 1_000_000_000,
                "prompt_eval_count": 1,
                "prompt_eval_duration_ns": 1_000_000,
                "load_duration_ns": load,
            }
        )
        store.append_score(
            {
                "gen_id": f"g{index}",
                "ts": "t",
                "scorer_version": 2,
                "check_hash": "CH:c",
                "score": 1.0,
                "failure_tags": [],
                "failure_applicable": [],
            }
        )
    perf = aggregate(store, cases_for(["c"])).perf[0]
    assert perf.load_ms_median == pytest.approx(1.0)


# ---------------------------------------------------------- 失敗型


def test_failure_denominator_is_applicable_not_total(tmp_path) -> None:
    """**宣言の無い型は母数に入らない**（§6-B・§14.8）。"""
    store = RunStore(tmp_path, "r")
    store.dir.mkdir(parents=True, exist_ok=True)
    for index, (tags, applicable) in enumerate(
        [(["preamble"], ["preamble", "format_broken"]), ([], [])]
    ):
        store.append_generation(
            {
                "gen_id": f"g{index}",
                "model": "m",
                "model_digest": "d",
                "ollama_version": "0.32.13",
                "case_id": "c",
                "seed": index,
                "eval_count": 1,
                "eval_duration_ns": 1,
                "prompt_eval_count": 1,
                "prompt_eval_duration_ns": 1,
                "load_duration_ns": 0,
            }
        )
        store.append_score(
            {
                "gen_id": f"g{index}",
                "ts": "t",
                "scorer_version": 2,
                "check_hash": "CH:c",
                "score": 1.0,
                "failure_tags": tags,
                "failure_applicable": applicable,
            }
        )
    counts = aggregate(store, cases_for(["c"])).failure_counts
    assert counts["preamble"]["m"] == (1, 1)
    assert counts["format_broken"]["m"] == (0, 1)


def test_json_round_trip_of_a_run(tmp_path) -> None:
    """store に書いた行がそのまま読み直せる（集計の前提）。"""
    store = build(tmp_path, [("m", "c", 0, 0.25)])
    rows = [json.loads(line) for line in (store.dir / "scores.jsonl").read_text().splitlines()]
    assert rows[0]["score"] == 0.25


# ------------------------------------------------------------ 脱線ゲートの寄与
#
# **`offtopic_rate` は S4 から書かれていたのに、読み返すものが 1 つも無かった。**
# 「黙って落とした件数は必ず出す」を、ここだけ守れていなかった（§17.11）。


def _with_gate(tmp_path, rows):
    """``rows`` は ``(model, case_id, score, offtopic_rate | None)``。"""
    store = RunStore(tmp_path, "g")
    store.dir.mkdir(parents=True, exist_ok=True)
    for index, (model, case_id, score, rate) in enumerate(rows):
        store.append_generation(
            {
                "gen_id": f"g{index}",
                "model": model,
                "model_digest": f"d-{model}",
                "ollama_version": "0.32.13",
                "case_id": case_id,
                "seed": 1000,
                "eval_count": 10,
                "eval_duration_ns": 10**8,
                "prompt_eval_count": 5,
                "prompt_eval_duration_ns": 10**7,
                "load_duration_ns": 10**6,
                "response": "x",
            }
        )
        store.append_score(
            {
                "gen_id": f"g{index}",
                "ts": f"2026-08-27T00:00:{index:02d}+00:00",
                "scorer_version": 2,
                "check_hash": "CH:" + case_id,
                "score": score,
                "sub_metrics": {} if rate is None else {"offtopic_rate": rate},
                "failure_tags": [],
                "failure_applicable": [],
            }
        )
    return store


def test_gate_drop_is_read_from_sub_metrics(tmp_path) -> None:
    store = _with_gate(tmp_path, [("m", "c", 1.0, 0.4), ("n", "c", 0.0, 0.0)])
    agg = aggregate(store, cases_for(["c"]))
    stat = agg.cases[0]
    assert stat.gate_drop_by_model == {"m": 0.4, "n": 0.0}


def test_gate_drop_is_the_mean_over_repeats(tmp_path) -> None:
    store = _with_gate(tmp_path, [("m", "c", 1.0, 0.2), ("m", "c", 1.0, 0.6)])
    agg = aggregate(store, cases_for(["c"]))
    assert agg.cases[0].gate_drop_by_model["m"] == 0.4


def test_a_case_without_a_gate_holds_nothing(tmp_path) -> None:
    """**0 と書かない。** 0 は「落ちなかった」で、「ゲートが無い」とは違う。"""
    store = _with_gate(tmp_path, [("m", "c", 1.0, None), ("n", "c", 0.0, None)])
    agg = aggregate(store, cases_for(["c"]))
    assert agg.cases[0].gate_drop_by_model == {}
