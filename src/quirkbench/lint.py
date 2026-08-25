"""ケース定義のゲート（FLB-QB-001 §12.2）。

**規約は、書いた時点では効かない。** 初版は「1 ケース = 1 関数の契約」を規約としか
書いておらず、`cases.py` が実際に見ていたのは ``entry_point`` の数だけだった。
``assert_count`` をレポートに出すのも検出であってゲートではないので、
**崩れた粒度がスコアに入ったあとで見えるだけ**になる。

これは §11.4 で §11.2 の表について自分で書いたことと同じで、そちらには照合スクリプトを
足したのに、ケース側には足していなかった。

**しきい値は固定する。** 調整可能にすると、落ちたときにしきい値のほうを動かして通す。
それは検査を持っていないのと同じになる。
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass

from .cases import Case

#: assert 数の上下限。8 個束ねると pass@1 が 0.47 ずれる（§12.2 の実測表）
MIN_ASSERTS = 1
MAX_ASSERTS = 6

#: 次元内の assert 数のばらつき上限（最大 ÷ 最小）。個々が範囲内でも、
#: 束ね方が揃っていなければ残差が粒度を測る
MAX_ASSERT_SPREAD = 3.0

#: timeout の範囲。小さすぎる値はケース作者の癖を測り、
#: 大きすぎる値は非停止の検出を諦めている
MIN_TIMEOUT_SECONDS = 5
MAX_TIMEOUT_SECONDS = 120

#: ideate の要求件数の下限。1 案では diversity が原理的に 0 になり、
#: 合成スコアが常に 0 になる（§13.1）。`cases.py` の
#: ``max_tokens >= num_predict`` と同じ「原理的に発火しない検査」を弾く型の規約
MIN_IDEATE_N = 2


@dataclass(frozen=True)
class LintIssue:
    case_id: str
    check: str
    message: str


def _asserts(source: str) -> int:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return -1
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assert))


def _toplevel_defs(source: str, name: str) -> int:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return -1
    return sum(
        1
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
    )


def _lint_ideate(case: Case) -> list[LintIssue]:
    """`ideate` ケースの規約（FLB-QB-001 §13.7）。

    **`score.count` と `failure.count` の一致を機械で見るのがここの主目的。**
    ずれると採点器と失敗検出器が違う件数を見る（§13.2）が、
    どちらも例外を出さずに**それらしい数字を返す**ので、走らせても気づけない。
    """
    issues: list[LintIssue] = []
    spec = case.score
    score_count = spec.get("count")
    failure_count = case.failure.get("count")

    if failure_count is None:
        issues.append(
            LintIssue(case.id, "ideate_count_required", "failure.count が無いと案を切り出せない")
        )
    elif score_count != failure_count:
        issues.append(
            LintIssue(
                case.id,
                "ideate_count_agrees",
                f"score.count={score_count} と failure.count={failure_count} が違う。"
                "採点器と失敗検出器が別の件数を見る",
            )
        )

    if not str(spec.get("topic", "")).strip():
        issues.append(
            LintIssue(case.id, "ideate_topic", "score.topic が空だと埋め込みゲートが成立しない")
        )

    groups = spec.get("coverage_terms")
    if not isinstance(groups, list) or not groups:
        issues.append(
            LintIssue(case.id, "ideate_terms", "coverage_terms が空だと coverage が 0 固定")
        )
    elif any(not group for group in groups):
        issues.append(
            LintIssue(
                case.id,
                "ideate_terms",
                "coverage_terms に空グループがある。常に未被覆になり coverage の上限が 1 を下回る",
            )
        )

    n = (score_count or {}).get("n") if isinstance(score_count, dict) else None
    if n is not None and int(n) < MIN_IDEATE_N:
        issues.append(
            LintIssue(
                case.id,
                "ideate_n_range",
                f"count.n={n}。{MIN_IDEATE_N} 未満だと diversity が原理的に 0 になる",
            )
        )
    return issues


def lint(cases: list[Case]) -> list[LintIssue]:
    """静的な検査だけを行う。**実行を伴う検査は入れない** —

    参照解の実測時間に対する余裕（§12.2）は `qb score` 側で測る。
    実行が要る検査を CI の必須ゲートに入れると、ollama も macOS も無い
    `checks` ジョブで回せなくなる。
    """
    issues: list[LintIssue] = []
    by_dim: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for case in cases:
        spec = case.score
        if spec.get("kind") == "ideate":
            issues.extend(_lint_ideate(case))
            continue
        if spec.get("kind") != "pytest":
            continue
        entry = str(spec.get("entry_point", ""))
        test = str(spec.get("test", ""))
        reference = str(spec.get("reference", ""))

        count = _asserts(test)
        if count < 0:
            issues.append(LintIssue(case.id, "test_parses", "test が Python として読めない"))
        else:
            by_dim[case.dim].append((case.id, count))
            if not MIN_ASSERTS <= count <= MAX_ASSERTS:
                issues.append(
                    LintIssue(
                        case.id,
                        "assert_count",
                        f"assert が {count} 個。{MIN_ASSERTS}〜{MAX_ASSERTS} に収める "
                        f"（束ねすぎると pass@1 が粒度を測る）",
                    )
                )

        defs = _toplevel_defs(reference, entry)
        if defs < 0:
            issues.append(
                LintIssue(case.id, "reference_parses", "reference が Python として読めない")
            )
        elif defs != 1:
            issues.append(
                LintIssue(
                    case.id,
                    "reference_entry_point",
                    f"reference の top-level に {entry} の定義が {defs} 個。"
                    f"ちょうど 1 個でないと陽性対照が成立しない",
                )
            )

        timeout = int(spec.get("timeout_seconds", 30))
        if not MIN_TIMEOUT_SECONDS <= timeout <= MAX_TIMEOUT_SECONDS:
            issues.append(
                LintIssue(
                    case.id,
                    "timeout_range",
                    f"timeout_seconds={timeout}。"
                    f"{MIN_TIMEOUT_SECONDS}〜{MAX_TIMEOUT_SECONDS} に収める",
                )
            )

    for dim, entries in by_dim.items():
        if len(entries) < 2:
            continue
        counts = [n for _, n in entries if n > 0]
        if not counts:
            continue
        spread = max(counts) / min(counts)
        if spread > MAX_ASSERT_SPREAD:
            worst = max(entries, key=lambda item: item[1])[0]
            issues.append(
                LintIssue(
                    worst,
                    "assert_spread",
                    f"次元 {dim} の assert 数が {min(counts)}〜{max(counts)}（{spread:.1f} 倍）。"
                    f"{MAX_ASSERT_SPREAD:.0f} 倍以内に揃える",
                )
            )
    return issues
