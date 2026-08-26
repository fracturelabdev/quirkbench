"""ケース 1 件抜きの再集計（FLB-QB-001 §17）。

**この検査が守るのは「結論がケース 1 件に乗っていたら気づける」ことだけ。**
z の計算そのものは `test_report_aggregate.py` が見る。
"""

from __future__ import annotations

from pathlib import Path

from quirkbench.cases import parse_case
from quirkbench.report.render_stability import CROSS_MARK, render
from quirkbench.report.stability import ModelStability, stability
from quirkbench.store import RunStore

SCHEMA = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}


def make_case(case_id: str, dim: str, task: str, lang: str = "ja"):
    case = parse_case(
        {
            "id": case_id,
            "dim": dim,
            "task": task,
            "lang": lang,
            "prompt": "p",
            "failure": {"format": "json"},
            "score": {"kind": "json_schema", "schema": SCHEMA},
        },
        Path(f"{case_id}.yaml"),
    )
    # `build` が書く採点行と `check_hash` を合わせる（§9 の読み出し規則）
    fields = {f.name: getattr(case, f.name) for f in case.__dataclass_fields__.values()}
    return type(case)(**{**fields, "check_hash": "CH:" + case_id})


def build(tmp_path, rows, *, run="r"):
    """``rows`` は ``(model, case_id, score)``。"""
    store = RunStore(tmp_path, run)
    store.dir.mkdir(parents=True, exist_ok=True)
    for index, (model, case_id, score) in enumerate(rows):
        gen_id = f"g{index}"
        store.append_generation(
            {
                "gen_id": gen_id,
                "model": model,
                "model_digest": f"digest-{model}",
                "ollama_version": "0.32.13",
                "case_id": case_id,
                "seed": 1000,
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
                "ts": f"2026-08-26T00:00:{index:02d}+00:00",
                "scorer_version": 2,
                "check_hash": "CH:" + case_id,
                "score": score,
                "sub_metrics": {},
                "failure_tags": [],
                "failure_applicable": [],
            }
        )
    return store


# ケースどうしが食い違う次元。1 件抜くと符号が入れ替わる
DISAGREE = [
    ("A", "d1", 1.0),
    ("B", "d1", 0.0),
    ("A", "d2", 0.0),
    ("B", "d2", 1.0),
]
# ケースどうしが一致する次元。1 件抜いても動かない
AGREE = [
    ("A", "s1", 1.0),
    ("B", "s1", 0.0),
    ("A", "s2", 1.0),
    ("B", "s2", 0.0),
]
DISAGREE_CASES = [make_case("d1", "reason", "alpha"), make_case("d2", "reason", "beta")]
AGREE_CASES = [make_case("s1", "instruct", "alpha"), make_case("s2", "instruct", "beta")]


def test_disagreeing_cases_flip_the_sign(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**これが検出できなければこの道具は無意味。**"""
    store = build(tmp_path, DISAGREE)
    result = stability(store, DISAGREE_CASES)
    dim = next(d for d in result.dims if d.dim == "reason")

    model_a = next(m for m in dim.models if m.model == "A")
    assert model_a.z_full == 0.0
    assert sorted(model_a.replicates) == [-1.0, 1.0]
    assert model_a.crosses_zero is True
    assert model_a.width == 2.0


def test_agreeing_cases_do_not_move(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = build(tmp_path, AGREE)
    result = stability(store, AGREE_CASES)
    dim = next(d for d in result.dims if d.dim == "instruct")

    model_a = next(m for m in dim.models if m.model == "A")
    assert model_a.z_full == 1.0
    assert model_a.replicates == (1.0, 1.0)
    assert model_a.width == 0.0
    assert model_a.crosses_zero is False


def test_replicate_count_matches_case_count(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**抜いた件数と複製の件数が合わないと、黙って一部が測られない。**"""
    store = build(tmp_path, DISAGREE + AGREE)
    result = stability(store, DISAGREE_CASES + AGREE_CASES)
    for dim in result.dims:
        for model in dim.models:
            assert len(model.replicates) == dim.cases


def test_a_replicate_that_yields_no_z_is_not_dropped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**測っていないものを、最も安定した値として出さない。**

    識別力のあるケースが 1 件しかない次元では、そのケースを抜くと σ=0 になって
    z が出ない。その複製を黙って捨てると、**残った 1 本だけで幅 0.00 =
    完全に安定**と表示される。実際には**その z はケース 1 件に依存している**
    という、最も不安定な状態である。
    """
    cases = [make_case("flat", "reason", "alpha"), make_case("real", "reason", "beta")]
    store = build(
        tmp_path,
        # `flat` は全モデル同点 → 識別力なし。`real` だけが z を作る
        [("A", "flat", 0.5), ("B", "flat", 0.5), ("A", "real", 0.0), ("B", "real", 1.0)],
    )
    result = stability(store, cases)
    dim = next(d for d in result.dims if d.dim == "reason")

    model_a = next(m for m in dim.models if m.model == "A")
    assert len(model_a.replicates) == 1
    assert model_a.missing == 1
    assert model_a.complete is False
    # **ここが本体。** 幅 0.00 を「安定している」として出してはいけない
    assert dim.measurable is False
    assert dim.why_not is not None and "複製で z が出ない" in dim.why_not
    assert "reason" in result.skipped


def test_tasks_counts_distinct_tasks_not_cases(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**同じ問題を ja/en で聞いても観測は 1 件**（§17.1）。"""
    cases = [
        make_case("p1", "reason", "same", lang="ja"),
        make_case("p2", "reason", "same", lang="en"),
    ]
    store = build(
        tmp_path, [("A", "p1", 1.0), ("B", "p1", 0.0), ("A", "p2", 0.0), ("B", "p2", 1.0)]
    )
    dim = next(d for d in stability(store, cases).dims if d.dim == "reason")
    assert dim.cases == 2
    assert dim.tasks == 1


def test_single_case_dimension_is_reported_not_dropped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**黙って飛ばすと、表に出ない次元が「安定していた」と読まれる。**"""
    cases = [make_case("only", "longctx", "solo")]
    store = build(tmp_path, [("A", "only", 1.0), ("B", "only", 0.0)])
    result = stability(store, cases)
    assert result.skipped == ["longctx"]
    dim = next(d for d in result.dims if d.dim == "longctx")
    assert dim.measurable is False
    assert dim.models == ()
    assert dim.why_not == "独立した観測が 1 件なので抜けるものが無い"


def test_dimension_without_generations_says_so(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**理由を 1 つの文言に潰すと嘘になる。**

    ケースは定義されていて、無いのは**その run の生成**という場合がある。
    実際 CI の煙テストで「ケースが 4 件なので抜けるものが無い」と出た。
    """
    cases = DISAGREE_CASES + AGREE_CASES
    store = build(tmp_path, DISAGREE)  # instruct 側の生成は書かない
    result = stability(store, cases)
    dim = next(d for d in result.dims if d.dim == "instruct")
    # **`cases` は「この run で採点された件数」**。定義済みの件数ではない
    assert dim.cases == 0
    assert dim.observed is False
    assert dim.measurable is False
    assert dim.why_not == "この run に採点済みの生成が無い"
    assert "instruct" in result.skipped


def test_case_spread_exposes_a_floor(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`discriminating` は通るが情報のほとんど無いケースを見えるようにする（§17.4）。"""
    cases = [make_case("f1", "reason", "alpha"), make_case("f2", "reason", "beta")]
    store = build(
        tmp_path,
        [("A", "f1", 0.25), ("B", "f1", 0.75), ("A", "f2", 1.0), ("B", "f2", 0.0)],
    )
    spreads = {c.case_id: c for c in stability(store, cases).case_spreads}
    assert spreads["f1"].spread == 0.5
    assert spreads["f1"].best == 0.75
    assert spreads["f1"].worst == 0.25
    assert spreads["f2"].spread == 1.0
    assert spreads["f1"].task == "alpha"


def test_case_spreads_are_in_dimension_order_not_by_size(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**幅の降順に並べたら、その表自体が順位表になる**（§16.1）。"""
    store = build(tmp_path, DISAGREE + AGREE)
    result = stability(store, DISAGREE_CASES + AGREE_CASES)
    order = [(c.dim, c.case_id) for c in result.case_spreads]
    assert order == sorted(order)


def test_the_dropped_unit_is_the_task_not_the_case(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**抜く単位は `task`**（§17.1）。

    ja/en の対訳をケース単位で抜くと**相手が残る**ので、その観測は消えない。
    消えないものを抜いても「この観測が無かったらどうなるか」には答えていない。
    """
    cases = [
        make_case("a_ja", "reason", "alpha", lang="ja"),
        make_case("a_en", "reason", "alpha", lang="en"),
        make_case("b_ja", "reason", "beta", lang="ja"),
        make_case("b_en", "reason", "beta", lang="en"),
    ]
    store = build(
        tmp_path,
        [
            ("A", "a_ja", 1.0),
            ("B", "a_ja", 0.0),
            ("A", "a_en", 1.0),
            ("B", "a_en", 0.0),
            ("A", "b_ja", 0.0),
            ("B", "b_ja", 1.0),
            ("A", "b_en", 0.0),
            ("B", "b_en", 1.0),
        ],
    )
    dim = next(d for d in stability(store, cases).dims if d.dim == "reason")
    assert dim.cases == 4
    assert dim.tasks == 2
    # **複製はタスクの数だけ。** ケース単位なら 4 本になる
    model_a = next(m for m in dim.models if m.model == "A")
    assert len(model_a.replicates) == 2
    # alpha を抜けば beta だけ（A は負）、beta を抜けば alpha だけ（A は正）
    assert sorted(model_a.replicates) == [-1.0, 1.0]


def test_an_unscored_case_does_not_pad_the_observation_count(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**定義済みのケース数で数えると、生成が無いケースが観測に混じる。**

    生成が無いケースを抜いても集計は動かないので、その複製は恒等になり
    幅が 0 に寄る。**観測 1 件の次元が「完全に安定」と出る。**
    """
    cases = [
        make_case("real", "reason", "alpha"),
        make_case("never_run", "reason", "beta"),
    ]
    store = build(tmp_path, [("A", "real", 1.0), ("B", "real", 0.0)])
    result = stability(store, cases)
    dim = next(d for d in result.dims if d.dim == "reason")
    assert dim.cases == 1
    assert dim.tasks == 1
    assert dim.measurable is False
    assert dim.why_not == "独立した観測が 1 件なので抜けるものが無い"


def test_a_replicate_that_is_numerically_zero_counts_as_crossing() -> None:
    """**最下位ビットの符号で判定を変えない。**

    実データで `2.05e-17` という複製が出た。数値としては 0 だが、
    厳密比較では正なので「0 をまたぐ」から漏れる。
    同じ次元の別モデルはたまたま負側に落ちて拾われていた。
    """
    positive_dust = ModelStability(model="A", z_full=0.6, replicates=(0.45, 2.05e-17))
    assert positive_dust.crosses_zero is True
    negative_dust = ModelStability(model="A", z_full=-0.6, replicates=(-0.45, -2.05e-17))
    assert negative_dust.crosses_zero is True
    clear = ModelStability(model="A", z_full=0.6, replicates=(0.45, 0.25))
    assert clear.crosses_zero is False


def test_crosses_zero_counts_touching_zero() -> None:
    """**見逃すより多めに拾う側に倒す。** 端が 0 ちょうどでも「またぐ」。"""
    touching = ModelStability(model="A", z_full=0.5, replicates=(0.0, 1.0))
    assert touching.crosses_zero is True
    clear = ModelStability(model="A", z_full=0.5, replicates=(0.25, 1.0))
    assert clear.crosses_zero is False


def test_max_width_is_the_worst_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = build(tmp_path, DISAGREE)
    dim = next(d for d in stability(store, DISAGREE_CASES).dims if d.dim == "reason")
    assert dim.max_width == max(m.width for m in dim.models)


def test_render_has_no_total_or_ranking_column(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**歯止めは散文ではなく、その欄が無いことに置く**（§16.1）。"""
    store = build(tmp_path, DISAGREE + AGREE)
    text = render(stability(store, DISAGREE_CASES + AGREE_CASES))
    header_cells = set()
    for line in text.splitlines():
        if line.startswith("|") and "---" not in line:
            header_cells.update(cell.strip() for cell in line.strip("|").split("|"))
    for banned in ("合計", "総合", "順位", "平均", "総計"):
        assert banned not in header_cells


def test_render_marks_the_crossing_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = build(tmp_path, DISAGREE)
    text = render(stability(store, DISAGREE_CASES))
    # **凡例にも印は出る。** `in text` で見ると、表から印を消しても通ってしまう
    marked = [
        line
        for line in text.splitlines()
        if line.startswith("|") and line.rstrip().rstrip("|").rstrip().endswith(CROSS_MARK)
    ]
    assert marked, "0 をまたいだモデルの行に印が付いていない"
    assert "独立な観測 2 件" in text


def test_render_says_a_single_case_dimension_was_not_measured(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = build(tmp_path, [("A", "only", 1.0), ("B", "only", 0.0)])
    text = render(stability(store, [make_case("only", "longctx", "solo")]))
    assert "抜けるものが無い" in text
