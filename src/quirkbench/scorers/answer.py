"""短い答えの採点 — `exact` と `numeric`（FLB-QB-001 §15.1）。

**抽出規則を決める前に、モデルが実際に何を返すか測った。**

「バグのある行番号だけを出力してください」に対し `qwen2.5:0.5b` は
``1: 2`` / ``2: 1.0`` の 2 行を返した。**「最初の数値を採る」抽出器だと、
これは「1 と答えた」ことになる** — 測っているのはモデルではなく抽出器である。

「おつりはいくらですか」に対しては ``10円``（単位つき）と
``… おつりは 850 円でした。``（説明つき）の両方が出た。
**素の数値だけを正解にすると、測るのは算数ではなく指示追従になる。**
それは `instruct` が測る次元で、二重に数えることになる（§4 が `ja` を
次元にしなかったのと同じ理由）。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from ..cases import Case
from ..embed import Embedder
from ..parse import Parsed
from . import Executor, ScoreResult

#: `contains` を許す最小の文字数（§15.1）。これ未満だと、無関係な出力に
#: 偶然含まれて正解になる。実測で 0.5b は ``1: 2`` を返しており、
#: 期待値が ``2`` なら偶然一致していた。**`qb lint-cases` で強制する。**
MIN_CONTAINS_LENGTH = 4

#: 数値らしき並び。符号・桁区切りのカンマ・小数点を含む
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_answer(text: str) -> str:
    """NFKC 正規化 + 前後の空白除去。全角の揺れで不正解にしない。"""
    return unicodedata.normalize("NFKC", text).strip()


def find_numbers(text: str) -> list[float]:
    """本文中の数値をすべて拾う。桁区切りのカンマは取り除く。"""
    out: list[float] = []
    for match in _NUMBER_RE.finditer(normalize_answer(text)):
        try:
            out.append(float(match.group(0).replace(",", "")))
        except ValueError:  # pragma: no cover - 正規表現が保証する
            continue
    return out


def _expected(spec: dict[str, Any]) -> list[str]:
    """``expect`` は文字列か文字列のリスト。**いずれか 1 つに一致すれば正解**。"""
    value = spec.get("expect")
    values = value if isinstance(value, list) else [value]
    return [normalize_answer(str(v)) for v in values]


def score_exact(
    parsed: Parsed, case: Case, *, executor: Executor | None, embedder: Embedder | None
) -> ScoreResult:
    """正規化後の一致。``match`` は ``equals``（既定）か ``contains``。

    ``contains`` を置くのは `longctx` のため。長文に埋めた事実を問うと、
    モデルは「照合番号は MULBERRY-7 です」と文で返す（実測）。``equals`` だけだと、
    **測っているのは長文脈保持ではなく出力の素っ気なさ**になる。
    """
    del executor, embedder
    spec: dict[str, Any] = case.score
    mode = str(spec.get("match", "equals"))
    payload = normalize_answer(parsed.payload)
    wanted = _expected(spec)

    equals = payload in wanted
    contains = any(w and w in payload for w in wanted)
    hit = contains if mode == "contains" else equals

    return ScoreResult(
        score=1.0 if hit else 0.0,
        sub_metrics={
            "match": mode,
            # **素の答えだったかをスコアに入れない**（§15.1）。
            # 「答えは知っているが黙れない」を読めるようにするための診断値
            "bare_answer": equals,
            "contains_answer": contains,
            "answer_chars": len(payload),
        },
        tags=(),
        applicable=(),
    )


def score_numeric(
    parsed: Parsed, case: Case, *, executor: Executor | None, embedder: Embedder | None
) -> ScoreResult:
    """**最後の数値**を採り、絶対許容差と比べる（§15.1）。

    最初ではなく最後を採るのは、推論を書いてから答えを述べる形に合わせるため。
    実測で 0.5b は問題文の数値を先に並べてから誤答を書いた。

    ``tolerance`` は**絶対値**（既定 0）。相対誤差にしない — 答えが 0 のとき
    定義できない。
    """
    del executor, embedder
    spec: dict[str, Any] = case.score
    expect = float(spec["expect"])
    tolerance = float(spec.get("tolerance", 0.0))

    numbers = find_numbers(parsed.payload)
    extracted = numbers[-1] if numbers else None
    hit = extracted is not None and abs(extracted - expect) <= tolerance

    return ScoreResult(
        score=1.0 if hit else 0.0,
        sub_metrics={
            "extracted": extracted,
            "numbers_found": len(numbers),
            # 素の数値 1 つだけだったか。指示追従の診断で、スコアには入らない
            "bare_answer": len(numbers) == 1
            and normalize_answer(parsed.payload).replace(",", "") == _plain(extracted),
            "tolerance": tolerance,
        },
        tags=(),
        applicable=(),
    )


def _plain(value: float | None) -> str:
    """``380.0`` ではなく ``380`` と書く。整数の答えの見た目を揃えるため。"""
    if value is None:
        return ""
    return str(int(value)) if value == int(value) else str(value)
