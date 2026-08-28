#!/usr/bin/env python
"""README の主張を機械で検査する。

**3 つを見る。**

1. **合計欄が存在しないこと** — README がプロファイル表を載せる以上、
   見た目は順位表に近づく。散文で「順位表ではない」と書いても、
   **合計欄が 1 つあれば読み手はそこで並べ替える。**
   歯止めは文章ではなく、その欄が無いことに置く。**この検査は run データが無くても回る。**
2. **README が挙げる出力が実装と一致すること** — 表を 1 つ足しても README は
   黙っている。実際 S8 の `stability.md`・S9 の `gate-audit.md` と
   **3 つの表が README に載らないまま 2 段階を通過した**。
   **数字の照合はこれを捕まえない** — 載っていない表の数字は照合対象にならないからで、
   **書いていないことは、間違って書くことより検出されにくい。**
   **run データが無くても回る**ので、CI で毎回効く。
3. **数字が実データと一致すること** — README の数字は手で書き写すので、
   **必ず 1 桁間違える。** `runs/<run>/` がある環境でだけ回る。

    uv run python tools/check-readme.py

**run データはリポジトリに入らない**（`runs/` は gitignore）ので、
CI で回るのは 1 と 2 だけ。3 は測定を更新したときに手元で回す。

**README 側のパースは表の形に依存する。** 表を作り直したらここも直す。
"""

from __future__ import annotations

import pathlib
import re
import sys

from quirkbench.cases import load_cases
from quirkbench.report.aggregate import aggregate
from quirkbench.report.stability import stability
from quirkbench.store import RunStore

QB = pathlib.Path(__file__).resolve().parent.parent

#: README がどの run を引いているか。**README 側に書いてある値を読む**ので、
#: run を差し替えたときにここを直し忘れても照合が空振りしない
#: **来歴の行に固定する。** 単に `run \`X\`` を探すと、本文中の別の run の話
#: （キャッシュの説明など）を拾う。実際 README の先頭から 2 番目までに
#: 過去の run への言及があり、**照合が古い run のデータに対して回っていた。**
RUN_MARKER = re.compile(r"^\| *測定日 *\|.*run `([\w.-]+)`", re.M)


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

    # --- 0b) **README が挙げる出力が、実装が出すものと一致すること**
    #
    # **run データを要さない。** 実装のソースに書いてある見出しとファイル名を読み、
    # README の表と**双方向で**突き合わせる。片方向にすると、
    # README に余分な行が残っても気づけない（§19 で踏んだのと同じ型）。
    for problem in _check_outputs(readme):
        problems.append(problem)
    checked += 1

    match = RUN_MARKER.search(readme)
    if match is None:
        problems.append(
            "README の来歴表に `| 測定日 | ... run `X` |` の行が無い。どの run の数字か辿れない"
        )
        return _report(problems, checked)
    checked += 1
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

    # --- 3) 結論の安定性（§17）。
    #
    # **数字だけ更新して安定性を置き去りにできないようにする。** ケースを 1 件でも
    # 足せば幅は変わるので、片方だけ新しいと、**古い安定性が新しい数字を裏書き**する。
    stab = stability(store, load_cases(QB / "cases"))
    expected_stab = {
        d.dim: (d.tasks, d.max_width, sum(1 for m in d.models if m.crosses_zero))
        for d in stab.dims
        if d.measurable
    }
    seen_dims = set()
    for dim, tasks, width, crossings in _parse_stability(readme):
        seen_dims.add(dim)
        want = expected_stab.get(dim)
        if want is None:
            problems.append(f"README の安定性に実データで測れていない次元 {dim}")
            continue
        checked += 1
        if tasks != want[0]:
            problems.append(f"観測数がずれている {dim}: README {tasks} / 実測 {want[0]}")
        if abs(width - want[1]) > 5e-3:
            problems.append(
                f"1 件抜きの幅がずれている {dim}: README {width:.2f} / 実測 {want[1]:.2f}"
            )
        if crossings != want[2]:
            problems.append(
                f"0 をまたぐ本数がずれている {dim}: README {crossings} / 実測 {want[2]}"
            )
    if expected_stab and not seen_dims:
        problems.append("README に安定性の表が無い。z だけが更新されうる")
    else:
        for dim in sorted(set(expected_stab) - seen_dims):
            problems.append(f"README の安定性に載っていない次元 {dim}")

    # --- 4) 母数。**「母数」行だけを見る。** 文書全体を検索すると、
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


#: `report.md` の中の表。**ソースの見出しリテラルを実体とする。**
#: 描画関数を呼んで見出しを集める手もあるが、`_gate_table` のように
#: **データが無ければ出ない表**があるので、合成データの作り方しだいで
#: 検査の網が変わってしまう。ソースなら run データに依らず一定になる。
HEADING = re.compile(r'"## (.+?)"')

#: `qb` が run ごとに書き出す Markdown。`cli.py` の書き込み先を実体とする。
REPORT_FILE = re.compile(r'out_dir / "([\w.-]+\.md)"')


def _normalize(text: str) -> str:
    """見出しと README の欄名を比べるための正規化。

    **記号だけを落とす。** 語まで落として緩めると、
    別の表と取り違えたまま一致してしまう。
    """
    return text.replace("`", "").replace("*", "").strip()


def _check_outputs(readme: str) -> list[str]:
    """README の出力の一覧が、実装が出すものと**双方向で**一致するか。"""
    problems: list[str] = []

    src = (QB / "src" / "quirkbench" / "report" / "render_profile.py").read_text(encoding="utf-8")
    headings = [_normalize(h) for h in HEADING.findall(src)]
    if not headings:
        return ["`render_profile.py` から表の見出しが 1 つも読めない。検査が空振りしている"]
    listed = [_normalize(c) for c in _first_column(readme, "`report.md` に出るもの:")]
    if not listed:
        return ["README に「`report.md` に出るもの」の表が無い。何が出るのか辿れない"]

    for heading in headings:
        if not any(heading.startswith(cell) for cell in listed):
            problems.append(f"`report.md` が出す表が README に無い: {heading}")
    for cell in listed:
        if not any(heading.startswith(cell) for heading in headings):
            problems.append(f"README が挙げる表を `report.md` は出さない: {cell}")

    cli = (QB / "src" / "quirkbench" / "cli.py").read_text(encoding="utf-8")
    written = sorted(set(REPORT_FILE.findall(cli)))
    if not written:
        return [*problems, "`cli.py` から出力ファイル名が読めない。検査が空振りしている"]
    for name in written:
        if f"reports/<run>/{name}" not in readme:
            problems.append(f"`qb` が書く出力が README の一覧に無い: reports/<run>/{name}")
    for name in re.findall(r"`reports/<run>/([\w.-]+\.md)`", readme):
        if name not in written:
            problems.append(f"README が挙げる出力を `qb` は書かない: reports/<run>/{name}")
    return problems


def _first_column(readme: str, after: str) -> list[str]:
    """`after` の直後にある表の 1 列目を返す（見出し行と区切り行は除く）。"""
    body = readme.split(after, 1)
    if len(body) < 2:
        return []
    cells: list[str] = []
    seen_separator = False
    for line in body[1].splitlines():
        if not line.startswith("|"):
            if cells:
                break
            continue
        row = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(row)) <= {"-", ":"}:
            seen_separator = True
            continue
        if seen_separator:
            cells.append(row[0])
    return cells


def _report(problems: list[str], checked: int) -> int:
    for line in problems:
        print(f"  NG  {line}", file=sys.stderr)
    print(f"  照合した項目: {checked} 件 / 指摘 {len(problems)} 件")
    return 1 if problems else 0


def _parse_stability(readme: str) -> list[tuple[str, int, float, int]]:
    """安定性の表を読む。**列の並びは 次元 / 観測数 / 最大幅 / またぐ本数。**"""
    out: list[tuple[str, int, float, int]] = []
    inside = False
    for line in readme.splitlines():
        if line.startswith("### 結論の安定性"):
            inside = True
            continue
        if inside and line.startswith("#"):
            break
        if not inside or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or set(cells[0]) <= {"-", ":"}:
            continue
        dim = cells[0].strip("`* ")
        try:
            out.append((dim, int(cells[1]), float(cells[2]), int(cells[3])))
        except ValueError:
            # 見出し行。**黙って読み飛ばすのはここだけ**
            continue
    return out


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
