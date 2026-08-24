"""採点のオーケストレーション — 生成 1 件を採点行 1 件に変換する。

**採点は追記のみ。** `generations.jsonl` は不変に保つ。数時間かけた生成を
scorer の修正で書き換えることは絶対にしない（FLB-QB-001 §10.2）。

再採点の条件は ``(gen_id, scorer_version, check_hash)``。3 つとも同じなら
出る結果も同じなので追記しない。どれかが変われば新しい行を足す。

**レポートが採る行の規則**（S5 で実装する。ここで決めておかないと、レポートの
実装ごとに暗黙の挙動が決まってしまう）:

    現在のケース定義の ``check_hash`` に一致する行のうち ``scorer_version`` 最大。
    同値なら ``ts`` 最大。

``scorer_version`` だけでは決まらない。期待値を直すと**同一 gen_id・同一
scorer_version で check_hash 違いの行が正常運用で 2 行できる**ため。
現在のケース定義を唯一の正とする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import failures, scorers
from .cases import Case
from .clock import now_iso
from .parse import parse
from .store import RunStore


@dataclass
class ScoreSummary:
    total: int = 0
    scored: int = 0
    skipped: int = 0
    generation_errors: int = 0
    unsupported: dict[str, int] = field(default_factory=dict)
    missing_cases: dict[str, int] = field(default_factory=dict)


def score_generation(row: dict[str, Any], case: Case) -> dict[str, Any]:
    """生成 1 行を採点する。パースは 1 回だけ行い、採点も失敗判定もその結果を見る。"""
    parsed = parse(row.get("response") or "", case.failure)
    done_reason = row.get("done_reason")
    eval_count = row.get("eval_count")

    report = failures.detect(parsed, case.failure, done_reason=done_reason, eval_count=eval_count)
    result = scorers.score(parsed, case)
    report = report.merge(tags=result.tags, applicable=result.applicable)

    na_tags, na_applicable = failures.detect_non_attempt(
        score=result.score,
        eval_count=eval_count,
        min_tokens=case.failure.get("min_tokens"),
        truncated=done_reason == "length",
    )
    report = report.merge(tags=na_tags, applicable=na_applicable)

    return {
        "gen_id": row["gen_id"],
        "ts": now_iso(),
        "scorer_version": scorers.SCORER_VERSION,
        "check_hash": case.check_hash,
        "score": result.score,
        "sub_metrics": result.sub_metrics,
        "failure_tags": list(report.tags),
        "failure_applicable": list(report.applicable),
        "detected_lang": parsed.detected_lang,
        "details": report.details,
    }


def score_run(store: RunStore, cases: list[Case]) -> ScoreSummary:
    by_id = {case.id: case for case in cases}
    summary = ScoreSummary()

    existing, _ = store.scores()
    done = {
        (str(row.get("gen_id")), row.get("scorer_version"), row.get("check_hash"))
        for row in existing
    }

    rows, _ = store.generations()
    for row in rows:
        summary.total += 1
        if row.get("error"):
            summary.generation_errors += 1
            continue
        case_id = str(row.get("case_id"))
        case = by_id.get(case_id)
        if case is None:
            # ケースが消された / 絞り込みで外れた。黙って落とさず数える
            summary.missing_cases[case_id] = summary.missing_cases.get(case_id, 0) + 1
            continue
        score_key = (str(row.get("gen_id")), scorers.SCORER_VERSION, case.check_hash)
        if score_key in done:
            summary.skipped += 1
            continue
        try:
            store.append_score(score_generation(row, case))
        except scorers.ScorerNotImplemented as exc:
            kind = str(exc)
            summary.unsupported[kind] = summary.unsupported.get(kind, 0) + 1
            continue
        # 同じ gen_id の成功行が 2 行ある run（中間破損を挟んだ再生成など）で
        # 二重に採点しないよう、追記のたびに更新する
        done.add(score_key)
        summary.scored += 1
    return summary
