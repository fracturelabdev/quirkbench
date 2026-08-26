"""レポートのレンダリング（FLB-QB-001 §14）。

**表の形だけを見る。** 数値の正しさは `test_report_aggregate.py` が持つ。
ここで数値を検証すると、計算がレンダリング側にあることを許してしまう。
"""

from __future__ import annotations

from pathlib import Path

from quirkbench.cases import parse_case
from quirkbench.report.aggregate import aggregate
from quirkbench.report.render_compare import MAX_CHARS, pick_seed
from quirkbench.report.render_compare import render as render_compare
from quirkbench.report.render_profile import NA, UNTRUSTED_MARK
from quirkbench.report.render_profile import render as render_profile
from quirkbench.store import RunStore


def case_of(case_id, dim="instruct", lang="ja", pair=None, prompt="p"):
    raw = {
        "id": case_id,
        "dim": dim,
        "lang": lang,
        "prompt": prompt,
        "failure": {"format": "json"},
        "score": {"kind": "json_schema", "schema": {"type": "object"}},
    }
    if pair:
        raw["pair"] = pair
    case = parse_case(raw, Path(f"{case_id}.yaml"))
    return type(case)(
        **{
            **{f.name: getattr(case, f.name) for f in case.__dataclass_fields__.values()},
            "check_hash": "CH:" + case_id,
        }
    )


def build(tmp_path, rows, *, responses=None):
    store = RunStore(tmp_path, "r")
    store.dir.mkdir(parents=True, exist_ok=True)
    for index, (model, case_id, seed, score) in enumerate(rows):
        store.append_generation(
            {
                "gen_id": f"g{index}",
                "model": model,
                "model_digest": "d",
                "ollama_version": "0.32.13",
                "case_id": case_id,
                "seed": seed,
                "eval_count": 10,
                "eval_duration_ns": 100_000_000,
                "prompt_eval_count": 5,
                "prompt_eval_duration_ns": 10_000_000,
                "load_duration_ns": 1_000_000,
                "wall_seconds": 0.1,
                "done_reason": "stop",
                "response": (responses or {}).get((model, case_id), "本文"),
            }
        )
        store.append_score(
            {
                "gen_id": f"g{index}",
                "ts": "t",
                "scorer_version": 2,
                "check_hash": "CH:" + case_id,
                "score": score,
                "failure_tags": ["preamble"] if score == 0.0 else [],
                "failure_applicable": ["preamble", "wrong_language"],
            }
        )
    return store


# ------------------------------------------------------------ プロファイル


def test_untrusted_z_is_marked_not_removed(tmp_path) -> None:
    """**消すと情報が減る。** 1 ケースで決まっていることが読めればよい（§14.5）。"""
    store = build(tmp_path, [("a", "c", 0, 0.0), ("b", "c", 0, 1.0)])
    text = render_profile(aggregate(store, [case_of("c")]))
    assert f"σ{UNTRUSTED_MARK}" in text


def test_trusted_z_has_no_mark(tmp_path) -> None:
    store = build(
        tmp_path,
        [("a", "c1", 0, 0.0), ("b", "c1", 0, 1.0), ("a", "c2", 0, 0.1), ("b", "c2", 0, 0.9)],
    )
    text = render_profile(aggregate(store, [case_of("c1"), case_of("c2")]))
    assert f"σ{UNTRUSTED_MARK}" not in text


def test_unmeasurable_dimension_says_so(tmp_path) -> None:
    """**「差が無い」ではなく「測れていない」**と書く（§14.4）。"""
    store = build(tmp_path, [("a", "c", 0, 1.0), ("b", "c", 0, 1.0)])
    text = render_profile(aggregate(store, [case_of("c")]))
    assert "測れていない" in text
    assert "z を出していない" in text


def test_excluded_cases_are_listed_with_a_reason(tmp_path) -> None:
    store = build(tmp_path, [("a", "c", 0, 1.0), ("b", "c", 0, 1.0)])
    text = render_profile(aggregate(store, [case_of("c")]))
    assert "集計から外したケース: 1 件" in text
    assert "ceiling" in text


def test_no_excluded_cases_says_so_explicitly(tmp_path) -> None:
    """**0 件と「見ていない」を区別する。**"""
    store = build(tmp_path, [("a", "c", 0, 0.0), ("b", "c", 0, 1.0)])
    text = render_profile(aggregate(store, [case_of("c")]))
    assert "集計から外したケースは無い" in text


def test_a_dimension_without_pairs_renders_na_not_zero(tmp_path) -> None:
    """**0 と書かない**（§14.7）。"""
    store = build(tmp_path, [("a", "c", 0, 0.0), ("b", "c", 0, 1.0)])
    text = render_profile(aggregate(store, [case_of("c")]))
    ja_section = text.split("## `ja_penalty`", 1)[1].split("## ", 1)[0]
    assert NA in ja_section
    assert "+0.000" not in ja_section


def test_candidate_tags_are_in_a_separate_table(tmp_path) -> None:
    """**主表に出すのは hard / declared だけ**（§6-C）。"""
    store = build(tmp_path, [("a", "c", 0, 0.0), ("b", "c", 0, 1.0)])
    text = render_profile(aggregate(store, [case_of("c")]))
    declared = text.split("### 宣言があれば固い", 1)[1].split("### ", 1)[0]
    candidate = text.split("### 候補（主表に出さない）", 1)[1]
    assert "`preamble`" in declared
    assert "`wrong_language`" not in declared
    assert "`wrong_language`" in candidate
    assert "人が行う" in candidate


def test_a_tier_with_no_tags_is_omitted(tmp_path) -> None:
    """**空の表を出さない。** 「0 件」と「その階層の型を一度も適用していない」は違う。"""
    store = build(tmp_path, [("a", "c", 0, 0.0), ("b", "c", 0, 1.0)])
    text = render_profile(aggregate(store, [case_of("c")]))
    assert "### 固い判定" not in text


def test_zero_denominator_renders_na(tmp_path) -> None:
    """**`0/0` を 0% と書かない**（§14.8）。"""
    store = RunStore(tmp_path, "r")
    store.dir.mkdir(parents=True, exist_ok=True)
    store.append_generation(
        {
            "gen_id": "g",
            "model": "a",
            "model_digest": "d",
            "ollama_version": "0.32.13",
            "case_id": "c",
            "seed": 0,
            "eval_count": 1,
            "eval_duration_ns": 1,
            "prompt_eval_count": 1,
            "prompt_eval_duration_ns": 1,
            "load_duration_ns": 0,
        }
    )
    store.append_score(
        {
            "gen_id": "g",
            "ts": "t",
            "scorer_version": 2,
            "check_hash": "CH:c",
            "score": 0.5,
            "failure_tags": ["truncated"],
            "failure_applicable": [],
        }
    )
    text = render_profile(aggregate(store, [case_of("c")]))
    assert f"| `truncated` | {NA} |" in text


def test_secondary_metric_is_labelled_untrustworthy(tmp_path) -> None:
    store = build(tmp_path, [("a", "c", 0, 0.0), ("b", "c", 0, 1.0)])
    text = render_profile(aggregate(store, [case_of("c")]))
    section = text.split("## 副指標", 1)[1]
    assert "単体では信用できない" in section
    assert "上を採る" in section


# ------------------------------------------------------------ 比較ビュー


def test_compare_uses_the_lowest_seed(tmp_path) -> None:
    """**選択が結論を作らないようにする**（§14.10）。"""
    store = build(
        tmp_path,
        [("a", "c", 7, 0.0), ("a", "c", 3, 1.0)],
        responses={("a", "c"): "共通本文"},
    )
    assert pick_seed(store) == 3
    text = render_compare(store, [case_of("c")], seed=3)
    assert "seed = 3" in text
    # **見出しだけを見ない。** 絞り込みが効いていないと 2 件とも出るが、
    # 見出しは同じなので気づけない（変異検査で実際に生き残った）
    assert text.count("### `a`") == 1


def test_compare_includes_every_model(tmp_path) -> None:
    store = build(tmp_path, [("a", "c", 0, 0.0), ("b", "c", 0, 1.0)])
    text = render_compare(store, [case_of("c")], seed=0)
    assert "`a`" in text
    assert "`b`" in text


def test_compare_shows_the_prompt(tmp_path) -> None:
    store = build(tmp_path, [("a", "c", 0, 0.0)])
    text = render_compare(store, [case_of("c", prompt="固有のお題")], seed=0)
    assert "固有のお題" in text


def test_fences_in_the_output_do_not_break_the_view(tmp_path) -> None:
    """**生出力にはコードフェンスが普通に含まれる。**

    ``` で囲むと途中で閉じ、以降の表示が壊れたまま
    「モデルの出力が壊れている」ように見える。
    """
    body = "```python\nx = 1\n```"
    store = build(tmp_path, [("a", "c", 0, 1.0)], responses={("a", "c"): body})
    text = render_compare(store, [case_of("c")], seed=0)
    assert "````" in text
    assert body in text


def test_long_output_is_clipped_and_says_so(tmp_path) -> None:
    body = "あ" * (MAX_CHARS + 100)
    store = build(tmp_path, [("a", "c", 0, 1.0)], responses={("a", "c"): body})
    text = render_compare(store, [case_of("c")], seed=0)
    assert "先頭のみ" in text
    assert body not in text


def test_all_seeds_includes_every_repeat(tmp_path) -> None:
    store = build(tmp_path, [("a", "c", 0, 0.0), ("a", "c", 1, 1.0)])
    text = render_compare(store, [case_of("c")], seed=0, all_seeds=True)
    assert text.count("### `a`") == 2


def test_empty_response_is_shown_as_empty_not_blank(tmp_path) -> None:
    """空を空欄で出すと、生成が無かったのか空だったのか読めない。"""
    store = build(tmp_path, [("a", "c", 0, 0.0)], responses={("a", "c"): "   "})
    text = render_compare(store, [case_of("c")], seed=0)
    assert "(空)" in text


# ------------------------------ モデル名の短縮（§15.5b）


def test_names_are_shortened_within_one_family() -> None:
    from quirkbench.report.render_profile import short_names

    assert short_names(["qwen2.5:0.5b", "qwen2.5:7b"]) == {
        "qwen2.5:0.5b": "0.5b",
        "qwen2.5:7b": "7b",
    }


def test_a_collision_disables_shortening_for_everyone() -> None:
    """**別ファミリを足した瞬間に別モデルが同じ名前で並ぶ**（S6 で実際に起きた）。

    一部だけ短縮すると、どれが短縮されているのかを読み手が判断できない。
    """
    from quirkbench.report.render_profile import short_names

    models = ["qwen2.5:3b", "llama3.2:3b", "qwen2.5:7b"]
    assert short_names(models) == {m: m for m in models}


def test_the_report_never_shows_two_models_under_one_name(tmp_path) -> None:
    store = build(
        tmp_path,
        [("qwen2.5:3b", "c", 0, 0.0), ("llama3.2:3b", "c", 0, 1.0)],
    )
    text = render_profile(aggregate(store, [case_of("c")]))
    assert "`qwen2.5:3b`" in text
    assert "`llama3.2:3b`" in text
