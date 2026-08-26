#!/usr/bin/env python
"""README の主張を機械で検査する。

**2 つを見る。**

1. **合計欄が存在しないこと** — README がプロファイル表を載せる以上、
   見た目は順位表に近づく。散文で「順位表ではない」と書いても、
   **合計欄が 1 つあれば読み手はそこで並べ替える。**
   歯止めは文章ではなく、その欄が無いことに置く。**この検査は run データが無くても回る。**
2. **数字が実データと一致すること** — README の数字は手で書き写すので、
   **必ず 1 桁間違える。** `runs/<run>/` がある環境でだけ回る。

    uv run python tools/check-readme.py

**run データはリポジトリに入らない**（`runs/` は gitignore）ので、
CI で回るのは 1 だけ。2 は測定を更新したときに手元で回す。

**README 側のパースは表の形に依存する。** 表を作り直したらここも直す。
"""

from __future__ import annotations

import pathlib
import re
import sys

from quirkbench.cases import load_cases
from quirkbench.report.aggregate import aggregate
from quirkbench.store import RunStore

QB = pathlib.Path(__file__).resolve().parent.parent

#: README がどの run を引いているか。**README 側に書いてある値を読む**ので、
#: run を差し替えたときにここを直し忘れても照合が空振りしない
RUN_MARKER = re.compile(r"run `([\w.-]+)`")


def main() -> int:
    readme = (QB / "README.md").read_text(encoding="utf-8")
    problems: list[str] = []
    checked = 0

    # --- 0) **合計欄が存在しないこと**（§16.1）。run データが無くても回る
    #
    # **散文の語で検査しない。** 「総合スコアで順位をつけるのではなく」のような、
    # まさに順位表でないことを説明している文が引っかかる。**説明文を消す方向に
    # 圧力がかかる検査は、検査として間違っている。**
    banned_labels = ("合計", "総合", "平均", "total", "overall", "rank", "順位")
    for label, line in _table_labels(readme):
        low = label.casefold()
        if any(b in low for b in banned_labels):
            problems.append(f"表に合計を示唆する欄 {label!r}: {line.strip()[:60]}")
    checked += 1

    match = RUN_MARKER.search(readme)
    if match is None:
        problems.append("README に `run `...`` の記載が無い。どの run の数字か辿れない")
        return _report(problems, checked)
    run_id = match.group(1)

    store = RunStore(QB / "runs", run_id)
    if not store.dir.exists():
        # **run データはリポジトリに入らない。** 無いことは失敗ではないので、
        # 飛ばしたことを明示して先に進む。**黙って通さない。**
        print(f"  --  run {run_id!r} が無いので数値の照合は飛ばした（CI では常にこうなる）")
        return _report(problems, checked)
    agg = aggregate(store, load_cases(QB / "cases"))

    # --- 1) 次元 × モデルの z
    expected_z = {(d.dim, m): v for d in agg.dims for m, v in d.z_by_model.items()}
    for dim, model, value in _parse_matrix(readme, "### 次元プロファイル"):
        checked += 1
        want = expected_z.get((dim, model))
        if want is None:
            problems.append(f"README に実データに無い組 {dim} × {model}")
        elif abs(want - value) > 5e-3:
            problems.append(
                f"z がずれている {dim} × {model}: README {value:+.2f} / 実測 {want:+.2f}"
            )
    missing = set(expected_z) - {
        (d, m) for d, m, _ in _parse_matrix(readme, "### 次元プロファイル")
    }
    if missing:
        problems.append(f"README に載っていない組が {len(missing)} 件: {sorted(missing)[:3]}")

    # --- 2) ja_penalty
    expected_ja = {(d.dim, m): v for d in agg.dims for m, v in d.ja_penalty_by_model.items()}
    for dim, model, value in _parse_matrix(readme, "### `ja_penalty`"):
        checked += 1
        want = expected_ja.get((dim, model))
        if want is None:
            problems.append(f"README の ja_penalty に実データに無い組 {dim} × {model}")
        elif abs(want - value) > 5e-4:
            problems.append(
                f"ja_penalty がずれている {dim} × {model}: README {value:+.3f} / 実測 {want:+.3f}"
            )

    # --- 3) 母数。**「母数」行だけを見る。** 文書全体を検索すると、
    # 「ケース 3 件」のような別の文脈の数字を拾って毎回落ちる
    row = next((ln for ln in readme.splitlines() if ln.startswith("| 母数 ")), None)
    if row is None:
        problems.append("README に `| 母数 |` の行が無い。何件を測ったのか辿れない")
    else:
        for label, want in [
            ("モデル", len(agg.models)),
            ("ケース", len(agg.cases)),
            ("生成", agg.total_generations),
        ]:
            found = re.search(rf"{label}\s*\*\*(\d+)\*\*", row)
            if found is None:
                problems.append(f"母数の行に「{label}」の件数が無い")
                continue
            checked += 1
            if int(found.group(1)) != want:
                problems.append(f"{label}数がずれている: README {found.group(1)} / 実測 {want}")

    return _report(problems, checked)


def _report(problems: list[str], checked: int) -> int:
    for line in problems:
        print(f"  NG  {line}", file=sys.stderr)
    print(f"  照合した項目: {checked} 件 / 指摘 {len(problems)} 件")
    return 1 if problems else 0


def _parse_matrix(readme: str, heading: str) -> list[tuple[str, str, float]]:
    """`| `dim` | ... | +0.12σ | ...` の形の表を読む。

    **列の並びはヘッダ行から取る。** 位置を決め打ちすると、
    列を 1 つ足したときに全部の値が 1 つずれたまま「一致」する。
    """
    body = readme.split(heading, 1)
    if len(body) < 2:
        return []
    rows: list[tuple[str, str, float]] = []
    models: list[str] = []
    for line in body[1].splitlines():
        if not line.startswith("|"):
            if rows or models:
                break
            continue
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if not models:
            models = [c for c in cells if ":" in c]
            continue
        if set(cells[0]) <= {"-"}:
            continue
        dim = cells[0]
        values = [c for c in cells[1:] if _looks_numeric(c)]
        if len(values) != len(models):
            continue
        for model, raw in zip(models, values, strict=True):
            rows.append((dim, model, float(raw.replace("σ", "").replace("−", "-"))))
    return rows


def _table_labels(readme: str) -> list[tuple[str, str]]:
    r"""表の**欄名**（見出し行の全セル + データ行の 1 列目）を返す。

    **見出しは「区切り行の直前」で決まる。** 中身の見た目で推定すると、
    2 列の説明表（`| \`diversity\` | 案同士の… |`）が毎行「見出しに見える」ので、
    説明文のセルを欄名と誤認する。

    どこまでが欄名かは**位置で決まっていて、長さや中身では決まっていない。**
    """
    out: list[tuple[str, str]] = []
    lines = readme.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("`*") for c in line.strip("|").split("|")]
        if set("".join(cells)) <= {"-", ":"}:
            continue  # 区切り行そのもの
        nxt = lines[index + 1] if index + 1 < len(lines) else ""
        is_header = nxt.startswith("|") and set(nxt.replace("|", "").strip()) <= {"-", ":", " "}
        for cell in cells if is_header else cells[:1]:
            out.append((cell, line))
    return out


def _looks_numeric(cell: str) -> bool:
    return bool(re.fullmatch(r"[-+−]?\d+\.\d+σ?", cell))


if __name__ == "__main__":
    raise SystemExit(main())
