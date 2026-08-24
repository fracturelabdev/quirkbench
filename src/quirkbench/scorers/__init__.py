"""次元固有の採点。**`Parsed` しか見ない**（生テキストに触れない）。

scorer は失敗型の語彙を持たない。`failures.py` の語彙のタグを append するだけ。

``SCORER_VERSION`` は採点ロジックを変えたら上げる。再採点は `scores.jsonl` への
追記だけで行い、レポートは gen_id ごとに version 最大の行を採るので、
**過去の採点も残り「scorer を直したら結果がどう動いたか」がそのまま追える**。
これは癖のプロファイリングという目的そのものに効く（FLB-QB-001 §10.2）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..parse import Parsed

SCORER_VERSION = 1


@dataclass(frozen=True)
class ScoreResult:
    """採点 1 件。``score`` は 0.0〜1.0。"""

    score: float
    sub_metrics: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    applicable: tuple[str, ...] = ()


class ScorerNotImplemented(NotImplementedError):
    """その ``score.kind`` の採点器がまだ無い。段階的に実装するため例外にする。"""


def _registry() -> dict[str, Callable[[Parsed, Any], ScoreResult]]:
    from .instruct import score_json_schema

    return {"json_schema": score_json_schema}


def score(parsed: Parsed, case: Any) -> ScoreResult:
    kind = case.score["kind"]
    scorer = _registry().get(kind)
    if scorer is None:
        raise ScorerNotImplemented(kind)
    return scorer(parsed, case)
