"""正規化層 — 生テキストを見る唯一の場所。

`scorers/*` も `failures.py` も、この層が返す :class:`Parsed` しか見ない。
これを挟まないと、scorer が「JSON パース成功」と判断する一方で failures が
``format_broken`` を付けるという食い違いが必ず起きる（FLB-QB-001 §10.1）。
**二重パースを構造で塞ぐのがこのモジュールの唯一の存在理由。**

このモジュールは**構造の抽出**だけを持つ。文字種・言語・反復・件数の計量は
`textstats.py` にある。判定（何を失敗と呼ぶか）は `failures.py` が持つ。
"""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .textstats import Repeat, char_class_of, detect_language, extract_items, max_repeat

FORMATS = frozenset({"json", "python", "none"})
EXTRACTORS = frozenset({"fenced_or_first_object", "fenced_or_whole", "fenced", "whole"})

_DEFAULT_EXTRACTOR = {
    "json": "fenced_or_first_object",
    "python": "fenced",
    "none": "whole",
}

# 空白ではないが目に見えない文字。empty 判定で取り除く
_INVISIBLE = frozenset("​‌‍⁠﻿")

# 中身の無いフェンス行。これだけの応答は empty 扱いにする
_FENCE_LINE_RE = re.compile(r"^[ \t]*```[^\n]*$", re.M)


@dataclass(frozen=True)
class CodeBlock:
    """フェンスで囲まれたブロック 1 つ。``closed`` が False なら閉じフェンスが無い。"""

    lang: str | None
    body: str
    closed: bool
    start: int
    end: int


@dataclass(frozen=True)
class Parsed:
    """1 応答の解析結果。生テキストに触れてよいのはこの生成過程だけ。

    **フィールドを足すときの規則: 消費者を名指しできないものは足さない。**
    「あとで使うかもしれない」で増やすと、`Parsed` が何を約束しているのか
    読めなくなる。現時点で判定に使っていないのは次の 4 つで、いずれも
    消費者が決まっている:

    - ``raw`` … S5 の比較ビューが本文として出す。生成の監査証跡でもある
    - ``payload`` … S5 の比較ビューが「実際に採点した部分」として出す
    - ``body`` … S5 の比較ビューが ``wrong_language`` の根拠として出す
    - ``code_blocks`` … S3 の code-gen 採点器がコードを取り出す

    判定に使うのは ``preamble`` / ``json_value`` / ``format_ok`` / ``unclosed`` /
    ``detected_lang`` / ``repeat`` / ``items`` / ``visible_chars`` / ``content_chars``。
    長さを見るときは ``content_chars`` を使う（``len(raw)`` はフェンス記号を含む）。
    """

    raw: str
    fmt: str
    preamble: str
    payload: str
    body: str
    code_blocks: tuple[CodeBlock, ...]
    json_value: Any | None
    json_error: str | None
    syntax_error: str | None
    format_ok: bool
    unclosed: tuple[str, ...]
    detected_lang: str
    repeat: Repeat
    items: tuple[str, ...]
    visible_chars: int
    content_chars: int


# -------------------------------------------------------------- フェンス


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.split("\n")[:-1]:
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def scan_fences(text: str) -> tuple[CodeBlock, ...]:
    lines = text.split("\n")
    offsets = _line_offsets(text)
    blocks: list[CodeBlock] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.startswith("```"):
            index += 1
            continue
        lang = stripped[3:].strip() or None
        start = offsets[index]
        index += 1
        body_lines: list[str] = []
        closed = False
        while index < len(lines):
            if lines[index].strip().startswith("```"):
                closed = True
                break
            body_lines.append(lines[index])
            index += 1
        end = offsets[index] + len(lines[index]) if closed and index < len(lines) else len(text)
        if closed:
            index += 1
        blocks.append(
            CodeBlock(lang=lang, body="\n".join(body_lines), closed=closed, start=start, end=end)
        )
    return tuple(blocks)


def _body_offset(raw: str, block: CodeBlock) -> int:
    head = raw.find("\n", block.start)
    return head + 1 if head >= 0 else block.start


# ------------------------------------------------------ 均衡した構造の抽出


def find_balanced(text: str) -> tuple[int, int, bool] | None:
    """最初の ``{`` / ``[`` から均衡が取れる範囲を返す。``(start, end, closed)``。

    文字列リテラル内の括弧を数えないよう、引用符とエスケープを見ながら歩く。
    """
    start = -1
    for index, ch in enumerate(text):
        if ch in "{[":
            start = index
            break
    if start < 0:
        return None

    pairs = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return (start, index + 1, False)
            stack.pop()
            if not stack:
                return (start, index + 1, True)
    return (start, len(text), False)


# ------------------------------------------------------------ 抽出の候補


def _regions(
    raw: str, extractor: str, blocks: tuple[CodeBlock, ...]
) -> list[tuple[str, int, CodeBlock | None]]:
    """抽出対象の候補を優先順に返す。``(領域, raw 内オフセット, 由来ブロック)``。

    **最初のフェンスを無条件に採らない。** 説明用の ```text ブロックを先に書く
    モデルは、後ろに正しい JSON があっても format_broken になり、測っているのは
    「最初のフェンスが JSON か」になる（Codex の指摘・実測で再現）。
    余計に書いたことは `preamble` タグが記録する。
    """
    if extractor == "whole" or not blocks:
        return [(raw, 0, None)]
    if extractor == "fenced":
        block = blocks[0]
        return [(block.body, _body_offset(raw, block), block)]

    if extractor == "fenced_or_whole":
        # python 用。lang が python/py/空のフェンス → その他 → 全文。
        # **全文は必ず最後**。先に置くと「最後の有効候補を採る」規則が
        # 全文だけを選ぶようになり、フェンスを見る意味が消える（§12.4）。
        py_first = [b for b in blocks if (b.lang or "").lower() in {"", "py", "python"}]
        others = [b for b in blocks if b not in py_first]
        out: list[tuple[str, int, CodeBlock | None]] = [
            (b.body, _body_offset(raw, b), b) for b in py_first + others
        ]
        out.append((raw, 0, None))
        return out

    preferred = [b for b in blocks if (b.lang or "").lower() in {"", "json"}]
    rest = [b for b in blocks if b not in preferred]
    regions: list[tuple[str, int, CodeBlock | None]] = [
        (b.body, _body_offset(raw, b), b) for b in preferred + rest
    ]
    regions.append((raw, 0, None))
    return regions


def _apply_format(payload: str, fmt: str) -> tuple[Any | None, str | None, str | None, list[str]]:
    """``(json_value, json_error, syntax_error, unclosed)`` を返す。"""
    if fmt == "json":
        try:
            return json.loads(payload), None, None, []
        except (json.JSONDecodeError, ValueError) as exc:
            return None, str(exc), None, []
    if fmt == "python":
        try:
            ast.parse(payload)
            return None, None, None, []
        except SyntaxError as exc:
            message = f"{exc.msg} (line {exc.lineno})"
            eof = any(w in exc.msg for w in ("never closed", "EOF", "unterminated"))
            return None, None, message, ["python"] if eof else []
        except ValueError as exc:  # ヌルバイト等
            return None, None, str(exc), []
    return None, None, None, []


# -------------------------------------------------------------------- 本体


def strip_fence_lines(text: str) -> str:
    """コードフェンスの**マーカー行だけ**を落とす。ブロックの中身は残す。"""
    return _FENCE_LINE_RE.sub("", text)


def visible_chars(text: str) -> int:
    """空白・ゼロ幅・BOM・フェンス行を除いた文字数。empty 判定に使う。"""
    return sum(1 for ch in strip_fence_lines(text) if not ch.isspace() and ch not in _INVISIBLE)


def has_toplevel_def(payload: str, entry_point: str) -> bool:
    """``payload`` の AST の top-level に ``entry_point`` と同名の関数定義があるか。"""
    try:
        tree = ast.parse(payload)
    except (SyntaxError, ValueError):
        return False
    return any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == entry_point
        for node in tree.body
    )


def _extract(
    raw: str,
    fmt: str,
    extractor: str,
    blocks: tuple[CodeBlock, ...],
    entry_point: str | None = None,
) -> tuple[str, int, int, CodeBlock | None, Any | None, str | None, str | None, list[str]]:
    """候補を順に試し、形式検証を通ったものを採る。

    どれも通らなければ**第一候補**の結果を採る（誤りの説明として最も妥当なため）。
    「何かがパースできるまで試す」わけではないので、`format_broken` は到達可能なまま。

    ``fenced_or_whole`` では**最後の**有効候補を採る（§12.4）。誤りを示してから直す
    説明の型では、**両方の候補が `entry_point` を top-level に定義する**。
    Python 自身が後の定義で前を上書きするので、最後を採るのが実行時の挙動と一致する。
    さらに ``entry_point`` を渡された場合は、**top-level に同名の定義があること**も
    条件にする。これが無いと、説明用の断片を先に書くモデルを落とし、
    測っているのがコード能力ではなく出力順序になる。
    """
    take_last = extractor == "fenced_or_whole"
    first = None
    best = None
    for region, offset, block in _regions(raw, extractor, blocks):
        start, end = 0, len(region)
        structure_unclosed: list[str] = []
        if fmt == "json" and extractor != "fenced":
            found = find_balanced(region)
            if found is not None:
                start, end, closed = found
                if not closed:
                    structure_unclosed.append("json")
        payload = region[start:end]
        value, json_error, syntax_error, fmt_unclosed = _apply_format(payload, fmt)
        candidate = (
            payload,
            offset + start,
            offset + end,
            block,
            value,
            json_error,
            syntax_error,
            structure_unclosed + fmt_unclosed,
        )
        ok = json_error is None and syntax_error is None
        if ok and entry_point is not None and not has_toplevel_def(payload, entry_point):
            ok = False
        if ok:
            if not take_last:
                return candidate
            best = candidate
        if first is None:
            first = candidate
    if best is not None:
        return best
    assert first is not None
    return first


def parse(raw: str, spec: dict[str, Any] | None = None) -> Parsed:
    """生テキストを 1 回だけ解析する。``spec`` はケースの ``failure`` ブロック。"""
    spec = spec or {}
    fmt = str(spec.get("format", "none"))
    if fmt not in FORMATS:
        raise ValueError(f"未知の failure.format: {fmt!r}")
    extractor = str(spec.get("extract") or _DEFAULT_EXTRACTOR[fmt])
    if extractor not in EXTRACTORS:
        raise ValueError(f"未知の failure.extract: {extractor!r}")

    blocks = scan_fences(raw)
    entry_point = spec.get("requires_def")
    payload, span_start, span_end, source_block, json_value, json_error, syntax_error, unclosed = (
        _extract(raw, fmt, extractor, blocks, str(entry_point) if entry_point else None)
    )

    # 未閉じフェンスは**採点対象のブロック**から導く。応答の最後のブロックで
    # 判定すると、ペイロードが完全でも incomplete が付く（Opus の指摘・実測で再現）。
    if source_block is not None and not source_block.closed:
        unclosed = ["fence", *unclosed]

    # フェンス行は抽出の足場であって前置きではない。除かずに数えると、
    # フェンス付き JSON を返すモデル**全件**に preamble が付き、
    # 層3 が測るのはモデルの癖ではなくこの検出器の癖になる（実測で確認）。
    preamble = strip_fence_lines(raw[:span_start]).strip() if fmt != "none" else ""

    # 散文（言語判定用）。コードブロックと抽出ペイロードを抜いた残り
    removals = [(block.start, block.end) for block in blocks]
    if fmt != "none":
        removals.append((span_start, span_end))
    body = _remove_spans(raw, removals)

    # 反復と件数はフェンス記号を除いた全文に対して数える。body に対して数えると、
    # 応答全体がペイロードのとき対象が空になる。
    content = strip_fence_lines(raw)
    pattern = (spec.get("count") or {}).get("pattern")

    return Parsed(
        raw=raw,
        fmt=fmt,
        preamble=preamble,
        payload=payload,
        body=body,
        code_blocks=blocks,
        json_value=json_value,
        json_error=json_error,
        syntax_error=syntax_error,
        format_ok=json_error is None and syntax_error is None,
        unclosed=tuple(dict.fromkeys(unclosed)),
        detected_lang=detect_language(char_class_of(body)),
        repeat=max_repeat(content),
        items=extract_items(content, pattern) if pattern else (),
        visible_chars=visible_chars(raw),
        content_chars=len(content.strip()),
    )


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    kept: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            kept.append(text[cursor:start])
        cursor = max(cursor, end)
    kept.append(text[cursor:])
    return "".join(kept)


def normalize(text: str) -> str:
    """期待値との比較に使う正規化。全角半角の揺れで不正解にしない。"""
    return unicodedata.normalize("NFKC", text).strip()
