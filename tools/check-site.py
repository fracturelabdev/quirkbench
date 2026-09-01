#!/usr/bin/env python
"""公開ページ（`site/`）の主張を機械で検査する。

**3 つを見る。**

1. **表の欄名に合計を示唆する語が無いこと** — README と同じ検査で、語は
   `_shared.py` から引く。**ページは README より視覚的なので、この圧力は強い。**
   図は合計欄が無くても順位を作る（棒を並べれば長さが順位に、レーダーを重ねれば
   面積が総合スコアになる）ので、**文章では止められない。**
2. **表が実測値を持っていないこと** — 初回スコープは「数字を載せず README へリンク」
   （`FLB-QB-001` §20.3）。実測値は既に 2 箇所にあり両方に検査が付いているので、
   **3 箇所目を作ればそこだけ検査が無い**状態になる。
   **対象は HTML の表だけ。** 図解 SVG の中の数字は意図した記述である。
3. **`404.html` が存在すること** — Cloudflare Pages は
   トップレベルに `404.html` が無いとプロジェクトを**単一ページアプリケーションだと判断し**、
   未知のパスをすべて `/` に一致させて `index.html` を HTTP 200 で返す。
   `www` が実際に踏んだ罠で、**構造で防ぐ以外に手が無い**。

    uv run python tools/check-site.py

**`site/` が無い環境では飛ばさない。** 無いこと自体を指摘する —
ディレクトリごと消えたときに黙って緑になるのが一番まずい。
"""

from __future__ import annotations

import pathlib
import re
import sys
from html.parser import HTMLParser

from _shared import BANNED_LABELS

QB = pathlib.Path(__file__).resolve().parent.parent
SITE = QB / "site"

#: 実測値の書式。`+0.12σ` / `-1.00σ` / `0.344` を拾う。
#:
#: **σ の付かない裸の小数も見る。** `z` だけを禁じても、
#: `ja_penalty` や類似度をそのまま貼れば同じことになる。
MEASURED = re.compile(r"[-+−]?\d+\.\d{2,}σ?")


class _Tables(HTMLParser):
    """`<table>` の中のセルだけを集める。

    **`<svg>` の中は見ない。** パーサは要素の入れ子で判定するので、
    表の外にある図の数字は最初から入ってこない。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.cell: str | None = None
        self.buf: list[str] = []
        #: (欄名かどうか, テキスト)
        self.cells: list[tuple[bool, str]] = []
        self._row_started = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.depth += 1
        elif tag == "tr":
            self._row_started = True
        elif tag in ("th", "td") and self.depth:
            self.cell = tag
            self.buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self.depth = max(0, self.depth - 1)
        elif tag in ("th", "td") and self.cell == tag:
            text = "".join(self.buf).strip()
            # **欄名は `<th>` と、行の先頭の `<td>`。** 位置で決まっていて、
            # 中身では決まっていない（`check-readme.py` の `_table_labels` と同じ判断）
            is_label = tag == "th" or self._row_started
            self.cells.append((is_label, text))
            self._row_started = False
            self.cell = None

    def handle_data(self, data: str) -> None:
        if self.cell:
            self.buf.append(data)


def _check_page(path: pathlib.Path, problems: list[str]) -> int:
    checked = 0
    parser = _Tables()
    parser.feed(path.read_text(encoding="utf-8"))
    name = path.name

    for is_label, text in parser.cells:
        if is_label:
            checked += 1
            low = text.casefold()
            if any(b in low for b in BANNED_LABELS):
                problems.append(f"{name}: 表に合計を示唆する欄 {text[:30]!r}")
        found = MEASURED.search(text)
        if found is not None:
            problems.append(
                f"{name}: 表に実測値らしい数字 {found.group()!r}（{text[:30]!r}）。"
                "数字は README に置き、ページからは参照する"
            )
    return checked


def main() -> int:
    problems: list[str] = []
    checked = 0

    if not SITE.is_dir():
        print("  NG  site/ が無い", file=sys.stderr)
        return 1

    pages = sorted(SITE.glob("*.html"))
    if not pages:
        print("  NG  site/ に .html が 1 つも無い。検査が空振りしている", file=sys.stderr)
        return 1

    for page in pages:
        checked += _check_page(page, problems)

    # --- 3) **`404.html` が存在すること**
    checked += 1
    if not (SITE / "404.html").is_file():
        problems.append(
            "site/404.html が無い。Cloudflare Pages が単一ページアプリケーションだと判断し、"
            "未知のパスをすべて index.html で HTTP 200 で返す"
        )

    for line in problems:
        print(f"  NG  {line}", file=sys.stderr)
    print(f"  照合した項目: {checked} 件 / ページ {len(pages)} 枚 / 指摘 {len(problems)} 件")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
