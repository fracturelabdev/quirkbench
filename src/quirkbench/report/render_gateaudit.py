"""ゲート検査の表を Markdown にする（FLB-QB-001 §19.7）。

**ここは計算しない**（§10.1）。`gateaudit.py` が出した値を並べるだけ。
**合計欄・順位欄を置かない**（§16.1）。
"""

from __future__ import annotations

import math

from .gateaudit import FALSE_REJECT_TARGET, GateAudit

#: 測っていない欄。**0 と書かない**（§14.7）
NA = "—"


def _num(value: float, digits: int = 3) -> str:
    return NA if math.isnan(value) else f"{value:.{digits}f}"


def _pct(value: float) -> str:
    return NA if math.isnan(value) else f"{value:.1%}"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render(result: GateAudit) -> str:
    out: list[str] = [
        f"# 脱線ゲートの検査 — run `{result.run_id}`",
        "",
        "**ゲートが発火したかではなく、発火が正しかったかを測る。**",
        "新しい生成はしていない（§19.7）。",
        "",
        "**対照は人手で作っていない。** 同じ次元・同じ言語・別 `task` のケースへの案を、"
        "このケースのお題のゲートに掛けたもので、**それは定義上の脱線である**（§19.2）。"
        "正例のほうには本物の屑が混じるので、**誤除外は過大に出る側**に倒れている。",
        "",
        "読み方は 5 つ。",
        "",
        "1. **`AUC` は 0.5 が無情報**、1.0 が完全分離。"
        "**順序が逆転していれば、閾値をどこに置いても分けられない**（§17.11.2）",
        f"2. **見逃し率は「正例の誤除外を {FALSE_REJECT_TARGET:.0%} に抑えた位置」で読む。** "
        "閾値の置き方と切り離して分離能だけを見るための固定であって、**判定の線ではない**",
        "3. **合否の線は引いていない。** 何 % を無情報と呼ぶかは読み手が決める — "
        "ここで定数を決めると、**その定数を動かすだけで「効いている」と言えてしまう**",
        f"4. **`{NA}` は測っていない。** 対照が作れない（次元にタスクが 1 つ）か、"
        "この run に生成が無い。**0 と書かない**",
        "5. **最後の 2 列は別のことを言う。** `現行の誤除外` は"
        " **いま実際に落としている正例の割合**、`見逃し` は**分離能の上限**である",
        "",
    ]
    out += _table(result)
    out += _skipped(result)
    return "\n".join(out) + "\n"


def _table(result: GateAudit) -> list[str]:
    header = [
        "ケース",
        "言語",
        "正例",
        "対照",
        "`AUC`",
        "運用点",
        "見逃し",
        f"現行の誤除外（閾値 {result.threshold_in_use}）",
    ]
    out = [_row(header), _row(["---"] * len(header))]
    for case in result.cases:
        out.append(
            _row(
                [
                    f"`{case.case_id}`",
                    case.lang,
                    str(case.positives),
                    str(case.controls) if case.controls else NA,
                    _num(case.auc),
                    _num(case.threshold),
                    _pct(case.control_leak),
                    _pct(case.current_reject_rate),
                ]
            )
        )
    out.append("")
    return out


def _skipped(result: GateAudit) -> list[str]:
    """**測れなかったものを黙って落とさない。** 表に出ないものは「効いていた」と読まれる。"""
    unmeasured = [c for c in result.cases if not c.measurable]
    if not unmeasured:
        return []
    out = ["## 測っていないもの", ""]
    for case in unmeasured:
        out.append(f"- `{case.case_id}` — {case.why_not}")
    out.append("")
    return out
