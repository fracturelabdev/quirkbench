"""安定性の表を Markdown にする（FLB-QB-001 §17.6）。

**ここは計算しない**（§10.1）。`stability.py` が出した値を並べるだけ。
**合計欄・順位欄を置かない**（§16.1）— 歯止めは散文ではなく、その欄が無いことに置く。
"""

from __future__ import annotations

from .render_profile import NA, short_names
from .stability import Stability, median_width

#: 区間が 0 をまたいだ印。**符号をデータが決めていない**という意味で、
#: 「間違っている」という意味ではない（§17.7）
CROSS_MARK = "?"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _header(cells: list[str]) -> list[str]:
    return [_row(cells), _row(["---"] * len(cells))]


def render(result: Stability) -> str:
    names = short_names(result.models)
    out: list[str] = [
        f"# 安定性 — run `{result.run_id}`",
        "",
        "**ケースを 1 件抜いて集計し直したときに `z` がどこまで動くか。**",
        "新しい生成はしていない。既存の採点行を読み直しただけ（§17.2）。",
        "",
        "読み方は 4 つ（§17.7）。",
        "",
        "1. **タスク数の違う次元の幅を直接比べない。** "
        "タスクが 1 種類の次元では 1 件抜くと観測が半分消えるので、幅は構造的に大きく出る",
        f"2. **`{CROSS_MARK}` は「符号をデータが決めていない」印であって、誤りの印ではない。** "
        "`z(全件)` が 0 付近ならまたぐのは当然で、"
        f"**問題になるのは `z(全件)` が 0 から離れているのに `{CROSS_MARK}` が付くとき**",
        "3. **対訳だけの次元では、1 件抜きは「言語を 1 つ抜く」と同じ。** "
        "幅の一部は `ja_penalty` が既に報告している言語効果である",
        "4. **幅が小さいことは「正しい」を意味しない。** "
        "ケースが揃って同じ偏りを持てば、安定して間違える",
        "",
    ]
    out += _dims_table(result, names)
    out += _cases_table(result)
    return "\n".join(out) + "\n"


def _dims_table(result: Stability, names: dict[str, str]) -> list[str]:
    out = ["## 次元ごと", ""]
    for dim in result.dims:
        out += [
            f"### `{dim.dim}` — ケース {dim.cases} 件 / **独立な観測 {dim.tasks} 件**",
            "",
        ]
        if not dim.measurable:
            out += [f"**測っていない** — {dim.why_not}。", ""]
            continue
        out += _header(["モデル", "z（全件）", "1 件抜きの範囲", "幅", ""])
        for model in dim.models:
            out.append(
                _row(
                    [
                        f"`{names.get(model.model, model.model)}`",
                        f"{model.z_full:+.2f}σ",
                        f"{model.z_min:+.2f} … {model.z_max:+.2f}",
                        f"{model.width:.2f}",
                        CROSS_MARK if model.crosses_zero else "",
                    ]
                )
            )
        out += [
            "",
            f"最大幅 **{dim.max_width:.2f}** / 中央値 {median_width(dim):.2f}。",
            "",
        ]
    if result.skipped:
        out += [
            "**測れなかった次元**: " + ", ".join(f"`{d}`" for d in result.skipped),
            "",
        ]
    return out


def _cases_table(result: Stability) -> list[str]:
    """ケースごとの実効幅。**floor を読み手が見つけられるようにする**（§17.4）。"""
    out = [
        "## ケースごとの実効幅",
        "",
        "**`識別力あり` の判定は `最大 − 最小 >= 1e-9` でしかない**（§14.3）。",
        "6 本中 5 本が 0.00 で 1 本だけ 0.20 のケースも、この判定は通る。",
        "**幅の小さいケースは、集計に入っていても情報をほとんど足していない。**",
        "",
    ]
    if not result.case_spreads:
        out += [NA, ""]
        return out
    # **次元順**に並べる。幅の降順にすると、それ自体が順位表になる（§16.1）
    out += _header(["ケース", "次元", "タスク", "最小", "最大", "実効幅"])
    for case in result.case_spreads:
        out.append(
            _row(
                [
                    f"`{case.case_id}`",
                    case.dim,
                    f"`{case.task}`",
                    f"{case.worst:.2f}",
                    f"{case.best:.2f}",
                    f"{case.spread:.2f}",
                ]
            )
        )
    out.append("")
    return out
