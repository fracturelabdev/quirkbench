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

from . import execcache, failures, scorers
from . import keys as _keys
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
    #: 実際にサンドボックスを起動した回数と、キャッシュで済んだ回数。
    #: S3 の完了条件（2 回目は実行 0 回）を機械で確かめるために持つ。
    exec_runs: int = 0
    exec_hits: int = 0


class _CachedExecutor:
    """実行のたびに ``exec_cache`` を引く実行器のラッパ（FLB-QB-001 §12.9）。

    **採点器から見ると普通の実行器**で、キャッシュの存在を知らない。
    キャッシュは実行の性質であって採点の性質ではないので、採点器に持ち込まない。
    """

    def __init__(self, inner: scorers.Executor, store: RunStore, fingerprint: str) -> None:
        self._inner = inner
        self._store = store
        self._fingerprint = fingerprint
        self._applied = inner.sandbox_applied
        self._cache = store.exec_cache(sandbox_applied=self._applied)
        self.hits = 0
        self.runs = 0

    @property
    def sandbox_applied(self) -> bool:
        return self._applied

    def run(
        self,
        *,
        payload: str,
        check_source: str,
        entry_point: str,
        timeout_seconds: int,
        check_hash: str = "",
    ) -> dict[str, Any]:
        key = _keys.exec_key(
            payload_sha256=_keys.payload_sha256(payload),
            check_hash=check_hash,
            fingerprint=self._fingerprint,
        )
        if key in self._cache:
            self.hits += 1
        else:
            self.runs += 1

        def _once() -> Any:
            return self._inner.run(
                payload=payload,
                check_source=check_source,
                entry_point=entry_point,
                timeout_seconds=timeout_seconds,
            )

        return execcache.resolve(
            exec_key=key,
            cache=self._cache,
            run_once=_once,
            write=lambda row: self._store.append_exec_cache(row, sandbox_applied=self._applied),
        )


def score_generation(
    row: dict[str, Any], case: Case, *, executor: scorers.Executor | None = None
) -> dict[str, Any]:
    """生成 1 行を採点する。パースは 1 回だけ行い、採点も失敗判定もその結果を見る。"""
    parsed = parse(row.get("response") or "", case.failure)
    done_reason = row.get("done_reason")
    eval_count = row.get("eval_count")

    report = failures.detect(parsed, case.failure, done_reason=done_reason, eval_count=eval_count)
    result = scorers.score(parsed, case, executor=executor)
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


def score_run(
    store: RunStore,
    cases: list[Case],
    *,
    executor: scorers.Executor | None = None,
    fingerprint: str = "",
) -> ScoreSummary:
    by_id = {case.id: case for case in cases}
    summary = ScoreSummary()
    if executor is not None:
        executor = _CachedExecutor(executor, store, fingerprint)

    existing, _ = store.scores()
    done = {
        (str(row.get("gen_id")), row.get("scorer_version"), row.get("check_hash"))
        for row in existing
        # **`error` を持つ score 行は `done` に入れない**（FLB-QB-001 §12.9 第 1 段）。
        # 入れると、一時的な workdir 作成失敗が永久スキップになる。
        # `store.completed_keys()` が生成側の `error` 行を除外しているのと揃える。
        if not row.get("error")
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
            store.append_score(score_generation(row, case, executor=executor))
        except scorers.ScorerNotImplemented as exc:
            kind = str(exc)
            summary.unsupported[kind] = summary.unsupported.get(kind, 0) + 1
            continue
        # 同じ gen_id の成功行が 2 行ある run（中間破損を挟んだ再生成など）で
        # 二重に採点しないよう、追記のたびに更新する
        done.add(score_key)
        summary.scored += 1
    if isinstance(executor, _CachedExecutor):
        summary.exec_runs = executor.runs
        summary.exec_hits = executor.hits
    return summary
