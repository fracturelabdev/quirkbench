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

#: `contains` を許す最小の期待文字数（§15.1）。これ未満だと無関係な出力に
#: 偶然含まれて正解になる。**閾値ではなく禁止**にするのが要点 —
#: 選べる幅を残すと、落ちたときに `contains` に逃げる（§13.3 と同じ論法）
MIN_CONTAINS_LENGTH = 4

#: プロンプト文字数からトークン数を見積もる上限（§15.2）。
#: 実測: 日本語 0.961 / 0.940 / 0.935 tok/char、英語 0.200 / 0.193。
#: **日本語の最悪値の上に丸める。** 英語では 5 倍過大に見積もるが、
#: 過小に見積もって静かに切り詰められるより、過大で落ちるほうがよい
TOKENS_PER_CHAR = 1.0

#: プロンプトが `num_ctx` に対して占めてよい割合。応答のぶんを残す
MAX_PROMPT_RATIO = 0.75


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


def _lint_answer(case: Case) -> list[LintIssue]:
    """`exact` / `numeric` の規約（FLB-QB-001 §15.1）。"""
    issues: list[LintIssue] = []
    spec = case.score
    kind = spec.get("kind")

    if spec.get("expect") is None:
        issues.append(LintIssue(case.id, "answer_expect", f"score.kind={kind} には expect が要る"))
        return issues

    if kind == "exact":
        mode = str(spec.get("match", "equals"))
        if mode not in {"equals", "contains"}:
            issues.append(LintIssue(case.id, "exact_match_mode", f"未知の match {mode!r}"))
        elif mode == "contains":
            values = spec["expect"]
            values = values if isinstance(values, list) else [values]
            short = [str(v) for v in values if len(str(v).strip()) < MIN_CONTAINS_LENGTH]
            if short:
                issues.append(
                    LintIssue(
                        case.id,
                        "contains_too_short",
                        f"match=contains で期待値 {short} が {MIN_CONTAINS_LENGTH} 文字未満。"
                        "無関係な出力に偶然含まれて正解になる",
                    )
                )
    if kind == "numeric":
        try:
            float(spec["expect"])
        except (TypeError, ValueError):
            issues.append(
                LintIssue(case.id, "numeric_expect", "score.kind=numeric の expect が数値でない")
            )
        if float(spec.get("tolerance", 0.0)) < 0:
            issues.append(LintIssue(case.id, "numeric_tolerance", "tolerance が負"))
    return issues


def _lint_context(case: Case) -> list[LintIssue]:
    """プロンプトが `num_ctx` に収まるか（FLB-QB-001 §15.2）。

    **収まらないとプロンプトは黙って切り詰められる。** エラーも警告も出ない。
    実測では 7,821 文字を `num_ctx` 4096 に投げると 2,050 トークンしか入らず、
    モデルは「文中に記述はありません」と答えた。**測っているのは
    長文脈保持ではなく切り詰め**になる。
    """
    num_ctx = int(case.options.get("num_ctx", 0))
    if not num_ctx:
        return []
    estimated = len(case.prompt) * TOKENS_PER_CHAR
    budget = num_ctx * MAX_PROMPT_RATIO
    if estimated <= budget:
        return []
    return [
        LintIssue(
            case.id,
            "prompt_fits_context",
            f"プロンプト {len(case.prompt)} 文字（見積り {estimated:.0f} トークン）が "
            f"num_ctx={num_ctx} の {MAX_PROMPT_RATIO:.0%}（{budget:.0f}）を超える。"
            "収まらないとプロンプトは黙って切り詰められ、測るのは切り詰めになる",
        )
    ]


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
        # **長さの検査は全ケースに掛ける。** longctx だけに掛けると、
        # 他の次元でプロンプトを伸ばしたときに黙って切り詰められる
        issues.extend(_lint_context(case))
        if spec.get("kind") == "ideate":
            issues.extend(_lint_ideate(case))
            continue
        if spec.get("kind") in {"exact", "numeric"}:
            issues.extend(_lint_answer(case))
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
    issues.extend(_lint_tasks(cases))
    return issues


def _lint_tasks(cases: list[Case]) -> list[LintIssue]:
    """`task` が独立な観測の単位として使える形になっているかを見る（§17.1）。

    **数え間違いは黙って起きる。** `task` がずれても run は通り、
    レポートは出て、**観測数だけが実態と違う数字になる。**
    """
    issues: list[LintIssue] = []
    by_id = {case.id: case for case in cases}

    dims_of_task: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        dims_of_task[case.task].add(case.dim)
    for task, dims in sorted(dims_of_task.items()):
        if len(dims) > 1:
            for case in cases:
                if case.task == task:
                    issues.append(
                        LintIssue(
                            case.id,
                            "task_spans_dims",
                            f"task {task!r} が次元 {sorted(dims)} にまたがっている。"
                            "task は次元の中で閉じていなければ観測数を数えられない",
                        )
                    )

    for case in cases:
        if case.pair is None:
            continue
        other = by_id.get(case.pair)
        if other is None:
            # **宛先の無い pair は黙って無視される。** ja_penalty は対を見つけられず、
            # 母数から静かに消えるだけでエラーにならない（§14.7）
            issues.append(
                LintIssue(
                    case.id,
                    "pair_missing",
                    f"pair の宛先 {case.pair!r} が存在しない。ja_penalty から黙って落ちる",
                )
            )
            continue
        # **対訳は定義上「同じ問題を別の言語で聞いたもの」**（§14.7）。
        # task が違うなら、どちらかの宣言が間違っている
        if other.task != case.task:
            issues.append(
                LintIssue(
                    case.id,
                    "pair_task_mismatch",
                    f"対訳の相手 {case.pair!r} の task が {other.task!r} で一致しない。"
                    "対訳は同じ task でなければ ja_penalty と観測数が食い違う",
                )
            )
    return issues
