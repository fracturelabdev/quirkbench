"""`code-gen` 次元の採点（FLB-QB-001 §12）。

**スコアは 0 か 1**（all-or-nothing）。テストが全部通れば 1。
このとき §3 の ``score(model, case)``（反復の平均）が**そのまま `pass@1` になる**。

**部分点にしない。** テスト 10 個中 7 個で 0.7 とすると `pass@1` の定義から外れ、
§3 の残差が「pass@1 の残差」でなくなる。通過テスト数は ``sub_metrics`` に残して人が読む。

採点器は**サンドボックスを直接呼ばない**。``executor`` を必須で受け取る（§12.10）。
"""

from __future__ import annotations

import ast
from typing import Any

from .. import failures
from ..cases import Case
from ..embed import Embedder
from ..parse import Parsed
from . import Executor, ExecutorRequired, ScoreResult

#: 実行採点が付けうるタグ。宣言しておかないと「適用して 0 件」と
#: 「そもそも見ていない」が区別できない（§6-B）。
_APPLICABLE = (failures.EXEC_ERROR, failures.EXEC_TIMEOUT)


def count_asserts(source: str) -> int:
    """``check`` 内の assert 文の数。粒度の監視に使う（§12.2）。"""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return 0
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assert))


def score_pytest(
    parsed: Parsed, case: Case, *, executor: Executor | None, embedder: Embedder | None
) -> ScoreResult:
    """``embedder`` は受け取るが使わない（registry の型を 1 つに保つため・§12.10）。"""
    del embedder
    if executor is None:
        raise ExecutorRequired(f"{case.id}: 実行採点には executor が要る")

    spec: dict[str, Any] = case.score
    entry_point = str(spec["entry_point"])
    check_source = str(spec["test"])
    timeout_seconds = int(spec.get("timeout_seconds", 30))

    # 形式が壊れているコードは実行に届かない。ここで止めるのは節約ではなく、
    # **exec_error と format_broken が同時に付いて原因が読めなくなるのを避ける**ため。
    if parsed.syntax_error is not None:
        return ScoreResult(
            0.0,
            {
                "executed": False,
                "reason": "syntax_error",
                "assert_count": count_asserts(check_source),
            },
            applicable=_APPLICABLE,
        )

    outcome = executor.run(
        payload=parsed.payload,
        check_source=check_source,
        entry_point=entry_point,
        timeout_seconds=timeout_seconds,
        # キャッシュのキーに入る。**採点器はキャッシュを知らない**が、
        # 「この実行がどのテスト定義に対するものか」は採点器しか知らない（§12.9）。
        check_hash=case.check_hash,
    )
    return from_outcome(outcome, check_source=check_source)


def from_outcome(outcome: Any, *, check_source: str) -> ScoreResult:
    """実行結果 1 件を ScoreResult に畳む。**キャッシュ行からも同じ関数で作る**（§12.9）。"""
    if isinstance(outcome, dict):
        raw = outcome.get("verdict", "")
    else:
        raw = getattr(outcome, "verdict", "")
    verdict = str(raw)
    sub: dict[str, Any] = {
        "executed": True,
        "verdict": verdict,
        "assert_count": count_asserts(check_source),
        "returncode": getattr(outcome, "returncode", None),
        "marker_seen": getattr(outcome, "marker_seen", None),
        "workdir_entries": getattr(outcome, "workdir_entries", None),
        "workdir_bytes": getattr(outcome, "workdir_bytes", None),
        "workdir_scan": getattr(outcome, "workdir_scan", None),
        "detail": (getattr(outcome, "detail", "") or "")[:500],
    }
    tags: tuple[str, ...] = ()
    if verdict == "exec_timeout":
        tags = (failures.EXEC_TIMEOUT,)
    elif verdict == "exec_error":
        tags = (failures.EXEC_ERROR,)
    score = 1.0 if verdict == "pass" else 0.0
    return ScoreResult(score, sub, tags=tags, applicable=_APPLICABLE)
