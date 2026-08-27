"""プロファイル表・失敗型・`ja_penalty`・性能を Markdown にする（FLB-QB-001 §14）。

**ここは計算しない。** `aggregate.py` が出した値を並べるだけ。
計算が混ざると、表形式を足すたびに残差の計算が増える（§10.1）。
"""

from __future__ import annotations

from .. import failures
from .aggregate import Aggregation, DimStat

#: z 値が 1 ケースで決まっていることを示す印（§14.5）。**消さずに印を付ける**
UNTRUSTED_MARK = "!"
#: 母数 0 / 対象外。**0 と書かない** — 0 は「差が無い」であって「測っていない」ではない
NA = "—"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _header(cells: list[str]) -> list[str]:
    return [_row(cells), _row(["---"] * len(cells))]


def short_names(models: list[str]) -> dict[str, str]:
    """表示用の短い名前。**衝突したら短縮しない**。

    `qwen2.5:3b` を `3b` と書くのは 1 ファミリのときだけ通用する。
    別ファミリを足した瞬間、`llama3.2:3b` も `3b` になり、
    **表の中で別のモデルが同じ名前で並ぶ**（S6 で実際に起きた）。

    短縮は読みやすさのためのもので、**同定を壊してよい理由にはならない。**
    1 つでも衝突したら、全部を元の名前で出す — 一部だけ短縮すると、
    どれが短縮されているのかを読み手が判断できない。
    """
    short = {m: (m.split(":", 1)[-1] if ":" in m else m) for m in models}
    if len(set(short.values())) != len(models):
        return {m: m for m in models}
    return short


def _z(dim: DimStat, model: str) -> str:
    if not dim.measurable:
        return NA
    value = dim.z_by_model.get(model)
    if value is None:
        return NA
    mark = "" if dim.z_is_trusted else UNTRUSTED_MARK
    return f"{value:+.2f}σ{mark}"


def render(agg: Aggregation) -> str:
    names = short_names(agg.models)
    out: list[str] = [
        f"# プロファイル — run `{agg.run_id}`",
        "",
        "**総合スコアで順位をつけるものではない。** 次元ごとの相対的な位置と、"
        "どう失敗するかの分布を見るための表。",
        "",
        f"- モデル **{len(agg.models)}** 本 / 生成 **{agg.total_generations}** 件"
        f"（失敗 {agg.generation_errors}）/ ollama `{agg.ollama_version}`",
        f"- 採点行が見つからなかった生成: **{agg.unscored}** 件",
        "",
    ]
    out += _profile_table(agg, names)
    out += _cases_table(agg, names)
    out += _gate_table(agg, names)
    out += _ja_penalty_table(agg, names)
    out += _failure_tables(agg, names)
    out += _perf_table(agg, names)
    out += _within_model_table(agg, names)
    return "\n".join(out) + "\n"


def _profile_table(agg: Aggregation, names: dict[str, str]) -> list[str]:
    out = [
        "## 次元プロファイル",
        "",
        "`z` は**ケース単位の残差を次元内で標準化した値**（§3）。"
        "ケースの難易度が相殺されるので次元をまたいで比較できる。"
        "`絶対` は反復の平均で、**次元をまたいで比較できない**（尺度が違う）。",
        "",
        "**母数が違う。** `z` は**識別力のあるケースだけ**から計算する"
        "（全モデル同値のケースは残差が全部 0 で情報を持たない）。"
        "`絶対` と `ばらつき` は**その次元の全ケース**の平均で、除外しない — "
        "「全モデルが満点だった」もモデルの事実だから。",
        "",
        "`ばらつき` は反復スコアの標準偏差。**小さい低スコアは「安定して弱い」**、"
        "大きければ「たまに当たる」。",
        "",
        f"**`{UNTRUSTED_MARK}` が付いた z は、識別力のあるケースが "
        f"1 件しかない次元のもの。** その 1 ケースで決まっている。",
        "",
    ]
    out += _header(["次元", "識別力あり/全", "モデル", "z", "絶対", "ばらつき"])
    for dim in agg.dims:
        for index, model in enumerate(agg.models):
            out.append(
                _row(
                    [
                        f"`{dim.dim}`" if index == 0 else "",
                        f"{dim.discriminating_cases} / "
                        f"{dim.discriminating_cases + dim.excluded_cases}"
                        if index == 0
                        else "",
                        f"`{names[model]}`",
                        _z(dim, model),
                        f"{dim.absolute_by_model.get(model, 0.0):.3f}",
                        f"{dim.spread_by_model.get(model, 0.0):.3f}",
                    ]
                )
            )
    out.append("")
    for dim in agg.dims:
        if not dim.measurable:
            out.append(
                f"**`{dim.dim}` は z を出していない。** 識別力のあるケースが 0 件で "
                f"σ が 0 になる。**「差が無い」のではなく「測れていない」。**"
            )
    out.append("")
    return out


def _cases_table(agg: Aggregation, names: dict[str, str]) -> list[str]:
    excluded = [c for c in agg.cases if not c.discriminating]
    out = ["## ケース単位のスコア", ""]
    out += _header(["ケース", "次元", *[f"`{names[m]}`" for m in agg.models], "識別力"])
    for stat in agg.cases:
        out.append(
            _row(
                [
                    f"`{stat.case_id}`",
                    stat.dim,
                    *[f"{stat.by_model.get(m, 0.0):.3f}" for m in agg.models],
                    "あり" if stat.discriminating else f"**なし**（{stat.degenerate_kind}）",
                ]
            )
        )
    out.append("")
    if excluded:
        out += [
            f"**集計から外したケース: {len(excluded)} 件。**",
            "",
            "- `ceiling` … 全モデルが満点。**ケースが緩い**",
            "- `floor` … 全モデルが 0 点。**難しすぎるか、採点器が壊れている**",
            "- `flat` … 満点でも 0 点でもないが全モデル同値",
            "",
        ]
        for stat in excluded:
            out.append(f"- `{stat.case_id}`（{stat.dim}）— {stat.degenerate_kind}")
        out.append("")
    else:
        out += ["集計から外したケースは無い。", ""]
    return out


def _gate_table(agg: Aggregation, names: dict[str, str]) -> list[str]:
    """脱線ゲートが落とした案の割合（§17.11）。

    **落とした件数を出さないと、削られたことに誰も気づけない。**
    実際、`offtopic_rate` は S4 から `scores.jsonl` に書かれていたのに
    読み返すものが 1 つも無く、**5 段階のあいだ誰も見ていなかった**。

    **ゲートを持たない次元は行に出さない。** 0 と書くと「落ちなかった」に読めるが、
    実際は「そもそもゲートが無い」で意味が違う。
    """
    gated = [c for c in agg.cases if c.gate_drop_by_model]
    if not gated:
        return []
    out = [
        "## 脱線ゲートが落とした割合",
        "",
        "案とお題の埋め込み類似度による足切り。**ゲートを持つケースだけ**が並ぶ。",
        "",
        "**この割合が課題ごとに大きく違うなら、ゲートは話題ではなく出力の形で発火している。**",
        "短い固有名詞（店名など）は類似度が一様に低く出るので、"
        "**閾値が分布の中腹を切る**（§17.11）。",
        "",
    ]
    out += _header(["ケース", *[f"`{names[m]}`" for m in agg.models]])
    for case in gated:
        cells = []
        for model in agg.models:
            value = case.gate_drop_by_model.get(model)
            cells.append(NA if value is None else f"{value:.3f}")
        out.append(_row([f"`{case.case_id}`", *cells]))
    out.append("")
    return out


def _ja_penalty_table(agg: Aggregation, names: dict[str, str]) -> list[str]:
    out = [
        "## `ja_penalty` — 同じ課題の日英差分",
        "",
        "**対訳ペアの両側が同じ run にあるケースだけ**が対象。"
        "`en − ja` なので、**正の値が「日本語で聞くと落ちる」**（§4）。",
        "",
    ]
    out += _header(["次元", "対訳ペア", *[f"`{names[m]}`" for m in agg.models]])
    for dim in agg.dims:
        cells = [f"`{dim.dim}`", str(dim.ja_pairs) if dim.ja_pairs else NA]
        for model in agg.models:
            value = dim.ja_penalty_by_model.get(model)
            cells.append(f"{value:+.3f}" if value is not None else NA)
        out.append(_row(cells))
    out += [
        "",
        f"`{NA}` は**対象外**（対訳ペアが無い）。**0 と書かない** — "
        "0 は「差が無い」であって「測っていない」ではない。",
        "",
    ]
    return out


def _failure_tables(agg: Aggregation, names: dict[str, str]) -> list[str]:
    out: list[str] = ["## 失敗型の分布", ""]
    for tier, title, note in [
        ("hard", "固い判定", "API の応答と実行の事実だけで決まる。"),
        ("declared", "宣言があれば固い", "ケース YAML が判定手続きを宣言したものだけが母数。"),
        (
            "candidate",
            "候補（主表に出さない）",
            "**機械タグは候補にすぎない。** 最終判断は比較ビューで人が行う。",
        ),
    ]:
        tags = [t for t in agg.failure_counts if failures.TIER.get(t) == tier]
        if not tags:
            continue
        out += [f"### {title}", "", note, ""]
        out += _header(["型", *[f"`{names[m]}`" for m in agg.models]])
        for tag in tags:
            cells = [f"`{tag}`"]
            for model in agg.models:
                numer, denom = agg.failure_counts[tag][model]
                cells.append(f"{numer} / {denom}" if denom else NA)
            out.append(_row(cells))
        out.append("")
    out += [
        f"分子は発火件数、分母は**適用されたケース数**。`{NA}` は母数 0 — "
        "**`0/0` を 0% と書かない**。",
        "",
    ]
    return out


def _perf_table(agg: Aggregation, names: dict[str, str]) -> list[str]:
    out = [
        "## 性能",
        "",
        "`tok/s` は**総和ベース**（Σトークン ÷ Σ時間）。行ごとの値を平均すると、"
        "短い生成ほど重みが大きくなり「短い応答の速さ」を測ることになる。"
        "`TTFT` と `load` は**中央値**（最初の 1 件だけモデルのロードが入る）。",
        "",
    ]
    out += _header(["モデル", "生成 tok/s", "プロンプト tok/s", "TTFT 中央値", "load 中央値"])
    for perf in agg.perf:
        out.append(
            _row(
                [
                    f"`{names[perf.model]}`",
                    f"{perf.tokens_per_second:.1f}",
                    f"{perf.prompt_tokens_per_second:.1f}",
                    f"{perf.ttft_ms_median:.1f} ms",
                    f"{perf.load_ms_median:.1f} ms",
                ]
            )
        )
    out.append("")
    return out


def _within_model_table(agg: Aggregation, names: dict[str, str]) -> list[str]:
    out = [
        "## 副指標 — モデル内の全次元平均からの偏差",
        "",
        "**単体では信用できない。** 尺度の違う次元を平均して差を取るので、"
        "難易度差が弱点に化け、天井効果が他次元を押し下げ、"
        "次元を 1 つ足すだけで全部の値が動く（§3）。"
        "**上のプロファイル表と食い違ったら、上を採る。**",
        "",
        "**読み方の例**: ある次元が全モデルで同じ符号に偏っていたら、"
        "それはその次元が弱いのではなく**その次元の尺度が他と違う**"
        "（合成指標のように上限が事実上 1 未満の次元は、全員が負に出る）。"
        "上の表はケース単位の残差を使うのでこの偏りが出ない。",
        "",
    ]
    dims = [d.dim for d in agg.dims]
    out += _header(["モデル", *[f"`{d}`" for d in dims]])
    for model in agg.models:
        values = agg.within_model_deviation.get(model, {})
        out.append(_row([f"`{names[model]}`", *[f"{values.get(d, 0.0):+.3f}" for d in dims]]))
    out.append("")
    return out
