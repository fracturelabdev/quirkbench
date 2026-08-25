"""`instruct` 次元の採点 — 形式制約つきの依頼が守られたか。

**スコアは乗算で出す。**

    score = schema_ok × expect_一致率

魔法の重み（0.5 など）を置かない。schema を満たさないなら 0、満たしたうえで
値がどれだけ合っているかが部分点になる。「JSON ですらなかった」と
「JSON だが中身が違う」の区別は**スコアではなく失敗型が持つ**（FLB-QB-001 §6）。
スコアに両方の役割を持たせると、残差が何の残差か読めなくなる。
"""

from __future__ import annotations

from typing import Any

from .. import minischema
from ..cases import Case
from ..parse import Parsed, normalize
from . import Executor, ScoreResult


def _equal(actual: Any, expected: Any) -> bool:
    """期待値との一致。全角半角の揺れでは落とさないが、型は厳密に見る。"""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, str) and isinstance(actual, str):
        return normalize(actual) == normalize(expected)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return actual == expected
    return actual == expected


def score_json_schema(
    parsed: Parsed, case: Case, *, executor: Executor | None = None
) -> ScoreResult:
    """``executor`` は受け取るが使わない。決定的採点はコードを実行しない。

    **署名に含めるのは、registry の型（``Scorer``）を全採点器で 1 つに保つため**（§12.10）。
    ここだけ署名が違うと registry の値型を緩めることになり、
    「``executor`` を受け取らない関数は入らない」という保証が消える。
    """
    del executor
    spec = case.score
    schema = spec.get("schema")
    expect: dict[str, Any] = spec.get("expect") or {}

    if not parsed.format_ok:
        # **失敗型のタグは付けない。** format_broken は failures.py が単独で所有する。
        # ここで append すると、空応答（failures が empty で短絡した行）にも
        # format_broken が付き、「JSON を壊す癖」の母数に「何も出さない癖」が載る。
        return ScoreResult(
            score=0.0,
            sub_metrics={
                "parse_ok": False,
                "schema_ok": False,
                "expect_total": len(expect),
                "expect_matched": 0,
            },
        )

    errors = minischema.validate(parsed.json_value, schema) if schema else []
    schema_ok = not errors

    matched = 0
    missing: list[str] = []
    for key, wanted in expect.items():
        actual = parsed.json_value.get(key) if isinstance(parsed.json_value, dict) else None
        if _equal(actual, wanted):
            matched += 1
        else:
            missing.append(key)
    rate = matched / len(expect) if expect else 1.0

    return ScoreResult(
        score=rate if schema_ok else 0.0,
        sub_metrics={
            "parse_ok": True,
            "schema_ok": schema_ok,
            "schema_errors": errors[:5],
            "expect_total": len(expect),
            "expect_matched": matched,
            "expect_missed": missing,
        },
        tags=(),
        applicable=(),
    )
