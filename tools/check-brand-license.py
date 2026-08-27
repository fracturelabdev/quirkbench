#!/usr/bin/env python
"""`assets/` が MIT の対象外であることが、実際に書かれているかを見る。

**このリポジトリは MIT で、`assets/` だけが All rights reserved である。**
除外は 1 箇所に書けば効くものではない — 読み手が最初に見る場所が人によって違う。

    - `LICENSE`        : ライセンスを機械で判定する側が見る
    - `README.md`      : 人が最初に見る
    - `CONTRIBUTING.md`: PR を出す人が見る
    - `assets/LICENSE` : ディレクトリを丸ごと取った人が見る

**どれか 1 つでも欠けると、そこだけ見た人にとっては MIT のままになる。**
公開済みリポジトリなので、置いた版は取り消せない。

**コピーが原本とずれていないかも見る。** 原本は非公開の
`organization-design/brand/` にあり、`assets/` はそのコピーである。
ずれたまま公開すると、**どちらが正か分からないものが 2 つ**になる。
原本が無い環境（CI）では、その照合だけを飛ばす。

    uv run python tools/check-brand-license.py
"""

from __future__ import annotations

import pathlib
import sys

QB = pathlib.Path(__file__).resolve().parent.parent
ASSETS = QB / "assets"
#: 原本。**非公開リポジトリなので CI には存在しない**
MASTER = QB.parent / "organization-design" / "brand"

#: `assets/` のファイルと、原本での名前
COPIES = {
    "logo.svg": "quirkbench-mark.svg",
    "logo-mono.svg": "quirkbench-mark-mono.svg",
    "favicon.svg": "quirkbench-favicon.svg",
}

#: 除外が書かれていなければならない場所と、そこに要る語
REQUIRED = {
    "LICENSE": ["assets/", "NOT", "MIT"],
    "README.md": ["assets/", "All rights reserved"],
    "CONTRIBUTING.md": ["assets/", "All rights reserved"],
    "assets/LICENSE": ["All rights reserved", "MIT License の対象外"],
}


def main() -> int:
    problems: list[str] = []
    checked = 0

    for name, needles in REQUIRED.items():
        path = QB / name
        if not path.exists():
            problems.append(f"{name} が無い")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            checked += 1
            if needle not in text:
                problems.append(f"{name} に {needle!r} が無い。除外がそこだけ効いていない")

    # **assets/ にライセンスの無いファイルを増やさない。**
    # 除外はディレクトリ単位で書いてあるので、増えたファイルも自動で対象になる —
    # だからこそ、**何が入っているかを検査で固定する**
    if ASSETS.exists():
        found = {p.name for p in ASSETS.iterdir() if p.is_file()}
        expected = set(COPIES) | {"LICENSE"}
        checked += 1
        if found != expected:
            problems.append(
                f"assets/ の中身が想定と違う: 余分 {sorted(found - expected)} / "
                f"欠け {sorted(expected - found)}"
            )
    else:
        problems.append("assets/ が無い")

    if MASTER.exists():
        for copy_name, master_name in COPIES.items():
            checked += 1
            copy_path, master_path = ASSETS / copy_name, MASTER / master_name
            if not copy_path.exists() or not master_path.exists():
                problems.append(f"{copy_name} か {master_name} が無い")
            elif copy_path.read_bytes() != master_path.read_bytes():
                problems.append(f"{copy_name} が原本 {master_name} と一致しない")
    else:
        # **飛ばしたことは必ず出す。** 0 件と「検査していない」は違う
        print("  --  原本が無いので照合は飛ばした（CI では常にこうなる）")

    for problem in problems:
        print(f"  NG  {problem}", file=sys.stderr)
    print(f"  照合した項目: {checked} 件 / 指摘 {len(problems)} 件")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
