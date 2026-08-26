"""`extract` 次元の採点 — キー単位の一致率（FLB-QB-001 §15.1）。

`json_schema`（``schema_ok × expect 一致率``）との違いは**構造の門が無い**こと。
`extract` が測るのは「拾えたか」であって「形式を守れたか」ではない。
形式は `instruct` が測る次元で、両方に入れると二重に数えることになる。

**余分なキーは減点しない。** 減点すると `instruct` と同じものを測り始める。
余分は ``extra_keys`` として ``sub_metrics`` に出し、人が読む。
"""

from __future__ import annotations

from typing import Any

from ..cases import Case
from ..embed import Embedder
from ..parse import Parsed
from . import Executor, ScoreResult
from .instruct import _equal


def score_json_keys(
    parsed: Parsed, case: Case, *, executor: Executor | None, embedder: Embedder | None
) -> ScoreResult:
    """``expect`` の各キーについて一致を見て率を出す。

    値の比較は **`instruct` の ``_equal`` をそのまま使う**。2 箇所に持つと、
    全角半角の扱いが次元ごとに違うという形で静かにずれる。
    """
    del executor, embedder
    spec: dict[str, Any] = case.score
    expect: dict[str, Any] = spec.get("expect") or {}

    if not parsed.format_ok or not isinstance(parsed.json_value, dict):
        # **タグは付けない。** format_broken は failures.py が単独で所有する
        return ScoreResult(
            score=0.0,
            sub_metrics={
                "parse_ok": parsed.format_ok,
                "expect_total": len(expect),
                "expect_matched": 0,
            },
        )

    value: dict[str, Any] = parsed.json_value
    matched: list[str] = []
    missed: list[str] = []
    for key, wanted in expect.items():
        if key in value and _equal(value[key], wanted):
            matched.append(key)
        else:
            missed.append(key)

    return ScoreResult(
        score=len(matched) / len(expect) if expect else 1.0,
        sub_metrics={
            "parse_ok": True,
            "expect_total": len(expect),
            "expect_matched": len(matched),
            "expect_missed": missed,
            # 減点しないが、出しておく。「余計に拾う癖」は癖として読める
            "extra_keys": sorted(set(value) - set(expect)),
        },
        tags=(),
        applicable=(),
    )
