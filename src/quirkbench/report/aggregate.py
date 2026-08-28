"""数値の集計 — **残差と σ の計算はここだけ**（FLB-QB-001 §14）。

レンダリングを持たない。ここが Markdown を知ると、表形式を足すたびに
残差の計算が増える。

**読むのは 1 つの run だけ**（§14.1）。残差はケース内でモデル間の平均を引くので、
run をまたぐと**引き算の相手が別の測定条件で測られたもの**になる。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from .. import failures
from ..cases import Case
from ..store import RunStore

#: 「同じスコア」とみなす幅（§14.3）。厳密比較にすると浮動小数の最下位ビットで
#: 識別力ありと誤判定しうる。広げると意味のある差を潰す。
SAME_SCORE_EPSILON = 1e-9

#: この件数未満の識別力ありケースしか無い次元は、z 値に警告を付ける（§14.5）
MIN_CASES_FOR_TRUSTED_Z = 2


class InconsistentRun(RuntimeError):
    """1 つの run の中で測定条件がずれている（§14.1）。

    同じモデル名に複数の ``model_digest`` があるか、``ollama_version`` が
    複数あると、**差し引かれるのが難易度ではなく測定条件**になる。
    """


@dataclass(frozen=True)
class CaseStat:
    """ケース 1 件の集計。``by_model`` は反復の平均。"""

    case_id: str
    dim: str
    lang: str
    by_model: dict[str, float]
    #: 反復スコアの母集団標準偏差。「安定して弱い」と「たまに当たる」を分ける（§14.6）
    sd_by_model: dict[str, float]
    n_by_model: dict[str, int]
    discriminating: bool
    #: 識別力なしのとき、全モデルが満点だったか（`ceiling`）全滅だったか（`floor`）。
    #: **同じ「除外」でも次の手が違う**（§14.3）
    degenerate_kind: str | None
    #: 脱線ゲートが落とした案の割合（`ideate` のみ・§17.11）。
    #: **ゲートを持たない次元では空**。0 と書くと「落ちなかった」に読めるが、
    #: 実際は「そもそもゲートが無い」で意味が違う
    gate_drop_by_model: dict[str, float] = field(default_factory=dict)
    #: `diversity` の崖に落ちた生成の割合（`ideate` のみ・§19.8）。
    #: valid が 2 件未満だと **スコアは比例減ではなく 0 に落ちる**。
    #: ゲートが削った結果として起きるので、`gate_drop_by_model` と並べて読む
    floor_rate_by_model: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DimStat:
    """次元 1 つの集計。"""

    dim: str
    sd: float
    #: モデル → z 値。σ が 0 なら空（**0 を返さない**・§14.4）
    z_by_model: dict[str, float]
    residual_by_model: dict[str, float]
    #: 反復ばらつきの平均
    spread_by_model: dict[str, float]
    absolute_by_model: dict[str, float]
    #: 対訳ペアの日英差分。ペアが無ければ空（**0 を入れない**・§14.7）
    ja_penalty_by_model: dict[str, float]
    ja_pairs: int
    discriminating_cases: int
    excluded_cases: int

    @property
    def z_is_trusted(self) -> bool:
        return self.discriminating_cases >= MIN_CASES_FOR_TRUSTED_Z

    @property
    def measurable(self) -> bool:
        """σ が 0 でない = z が出せる。**「差が無い」ではなく「測れていない」**。"""
        return bool(self.z_by_model)


@dataclass(frozen=True)
class Perf:
    """1 モデルの性能。**行ごとの値を平均しない**（§14.9）。"""

    model: str
    tokens_per_second: float
    prompt_tokens_per_second: float
    ttft_ms_median: float
    load_ms_median: float


@dataclass
class Aggregation:
    run_id: str
    models: list[str] = field(default_factory=list)
    cases: list[CaseStat] = field(default_factory=list)
    dims: list[DimStat] = field(default_factory=list)
    perf: list[Perf] = field(default_factory=list)
    #: 失敗型 → モデル → (分子, 母数)。母数は ``failure_applicable``（§14.8）
    failure_counts: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict)
    #: モデル内の全次元平均からの偏差。**副指標**（§14.11）
    within_model_deviation: dict[str, dict[str, float]] = field(default_factory=dict)
    #: 採る採点行が 1 つも無かった生成。**黙って落とさない**（§14.2）
    unscored: int = 0
    total_generations: int = 0
    generation_errors: int = 0
    ollama_version: str = ""
    #: 比較ビューが使う seed（最小 seed・§14.10）
    compare_seed: int | None = None


def _pick_scores(
    generations: dict[str, dict[str, Any]], score_rows: list[dict[str, Any]], by_id: dict[str, Case]
) -> tuple[dict[str, dict[str, Any]], int]:
    """§9 の読み出し規則。

    > 現在のケース定義の ``check_hash`` に一致する行のうち ``scorer_version`` 最大。
    > 同値なら ``ts`` 最大。
    """
    best: dict[str, dict[str, Any]] = {}
    for row in score_rows:
        gen = generations.get(str(row.get("gen_id")))
        if gen is None:
            continue
        case = by_id.get(str(gen.get("case_id")))
        if case is None or row.get("check_hash") != case.check_hash:
            continue
        key = str(row["gen_id"])
        current = best.get(key)
        rank = (int(row.get("scorer_version") or 0), str(row.get("ts") or ""))
        if current is None or rank > (
            int(current.get("scorer_version") or 0),
            str(current.get("ts") or ""),
        ):
            best[key] = row
    return best, len(generations) - len(best)


def _check_consistency(generations: dict[str, dict[str, Any]]) -> str:
    digests: dict[str, set[str]] = {}
    versions: set[str] = set()
    for gen in generations.values():
        digests.setdefault(str(gen.get("model")), set()).add(str(gen.get("model_digest")))
        versions.add(str(gen.get("ollama_version")))
    drifted = {model: sorted(seen) for model, seen in digests.items() if len(seen) > 1}
    if drifted:
        raise InconsistentRun(
            f"同じ run の中でモデルの中身が変わっている: {drifted}。"
            "残差はケース内でモデル間の平均を引くので、混ざると差し引かれるのが難易度でなくなる"
        )
    if len(versions) > 1:
        raise InconsistentRun(f"同じ run の中で ollama のバージョンが複数ある: {sorted(versions)}")
    return next(iter(versions), "")


def _degenerate_kind(values: list[float]) -> str | None:
    """識別力なしの理由。満点なら ``ceiling``、全滅なら ``floor``、それ以外は ``flat``。"""
    if not values:
        return "flat"
    if all(v >= 1.0 - SAME_SCORE_EPSILON for v in values):
        return "ceiling"
    if all(v <= SAME_SCORE_EPSILON for v in values):
        return "floor"
    return "flat"


def _perf(generations: dict[str, dict[str, Any]], models: list[str]) -> list[Perf]:
    out: list[Perf] = []
    for model in models:
        rows = [g for g in generations.values() if g.get("model") == model]
        if not rows:
            continue
        # **総和で割る。** 行ごとに tok/s を出してから平均すると、
        # 短い生成ほど 1 件の重みが大きくなり、測っているのが
        # 「短い応答の速さ」になる（§14.9）
        eval_count = sum(int(g.get("eval_count") or 0) for g in rows)
        eval_ns = sum(int(g.get("eval_duration_ns") or 0) for g in rows)
        prompt_count = sum(int(g.get("prompt_eval_count") or 0) for g in rows)
        prompt_ns = sum(int(g.get("prompt_eval_duration_ns") or 0) for g in rows)
        # 中央値で出す。run の最初の 1 件だけモデルのロードが入る
        loads = [int(g.get("load_duration_ns") or 0) / 1e6 for g in rows]
        ttfts = [
            (int(g.get("load_duration_ns") or 0) + int(g.get("prompt_eval_duration_ns") or 0)) / 1e6
            for g in rows
        ]
        out.append(
            Perf(
                model=model,
                tokens_per_second=eval_count / (eval_ns / 1e9) if eval_ns else 0.0,
                prompt_tokens_per_second=prompt_count / (prompt_ns / 1e9) if prompt_ns else 0.0,
                ttft_ms_median=statistics.median(ttfts) if ttfts else 0.0,
                load_ms_median=statistics.median(loads) if loads else 0.0,
            )
        )
    return out


def aggregate(store: RunStore, cases: list[Case]) -> Aggregation:
    """1 つの run を集計する。**複数 run を結合しない**（§14.1）。"""
    by_id = {case.id: case for case in cases}
    raw, _ = store.generations()
    generations = {str(r["gen_id"]): r for r in raw if not r.get("error")}
    errors = len(raw) - len(generations)
    version = _check_consistency(generations)

    score_rows, _ = store.scores()
    picked, unscored = _pick_scores(generations, score_rows, by_id)

    models = sorted({str(g.get("model")) for g in generations.values()})
    result = Aggregation(
        run_id=store.run_id,
        models=models,
        unscored=unscored,
        total_generations=len(raw),
        generation_errors=errors,
        ollama_version=version,
    )
    seeds = [int(s) for g in generations.values() if (s := g.get("seed")) is not None]
    result.compare_seed = min(seeds) if seeds else None

    # --- ケース単位
    per_case: dict[str, dict[str, list[float]]] = {}
    # **ゲートが落とした件数も集める**（§17.11）。`sub_metrics` は今まで
    # 誰も読み返しておらず、**落とした案だけがレポートに出ていなかった** —
    # 「黙って落とした件数は必ず出す」を、ここだけ守れていなかった
    per_gate: dict[str, dict[str, list[float]]] = {}
    #: 崖に落ちた生成（§19.8）。**ゲートが削った結果として起きる**ので、
    #: ゲートの除外率と並べて読めるように同じ形で集める
    per_floor: dict[str, dict[str, list[float]]] = {}
    for gen_id, row in picked.items():
        gen = generations[gen_id]
        case_id = str(gen.get("case_id"))
        model = str(gen.get("model"))
        per_case.setdefault(case_id, {}).setdefault(model, []).append(
            float(row.get("score") or 0.0)
        )
        sub = row.get("sub_metrics") or {}
        rate = sub.get("offtopic_rate")
        if rate is not None:
            per_gate.setdefault(case_id, {}).setdefault(model, []).append(float(rate))
        floored = sub.get("diversity_floored")
        if floored is not None:
            per_floor.setdefault(case_id, {}).setdefault(model, []).append(float(bool(floored)))

    for case_id in sorted(per_case):
        case = by_id[case_id]
        by_model = {m: sum(v) / len(v) for m, v in per_case[case_id].items()}
        values = list(by_model.values())
        discriminating = bool(values) and (max(values) - min(values)) >= SAME_SCORE_EPSILON
        result.cases.append(
            CaseStat(
                case_id=case_id,
                dim=case.dim,
                lang=case.lang,
                by_model=by_model,
                sd_by_model={
                    m: statistics.pstdev(v) if len(v) > 1 else 0.0
                    for m, v in per_case[case_id].items()
                },
                n_by_model={m: len(v) for m, v in per_case[case_id].items()},
                discriminating=discriminating,
                degenerate_kind=None if discriminating else _degenerate_kind(values),
                gate_drop_by_model={
                    m: sum(v) / len(v) for m, v in per_gate.get(case_id, {}).items()
                },
                floor_rate_by_model={
                    m: sum(v) / len(v) for m, v in per_floor.get(case_id, {}).items()
                },
            )
        )

    # --- 次元単位
    for dim in sorted({c.dim for c in result.cases}):
        in_dim = [c for c in result.cases if c.dim == dim]
        usable = [c for c in in_dim if c.discriminating]
        residuals: dict[str, list[float]] = {}
        for case_stat in usable:
            mean = sum(case_stat.by_model.values()) / len(case_stat.by_model)
            for model, value in case_stat.by_model.items():
                residuals.setdefault(model, []).append(value - mean)
        flat = [x for values in residuals.values() for x in values]
        # **母集団標準偏差**（§14.4）。対象は手元にあるモデル全部で、標本ではない
        sd = statistics.pstdev(flat) if len(flat) > 1 else 0.0
        residual_mean = {m: sum(v) / len(v) for m, v in residuals.items()}
        result.dims.append(
            DimStat(
                dim=dim,
                sd=sd,
                # σ が 0 なら空にする。**0 を返すと「差が無い」と読める**が、
                # 実際は「測れていない」（§14.4）
                z_by_model=({m: v / sd for m, v in residual_mean.items()} if sd > 0 else {}),
                residual_by_model=residual_mean,
                spread_by_model=_mean_by_model(in_dim, lambda c: c.sd_by_model),
                absolute_by_model=_mean_by_model(in_dim, lambda c: c.by_model),
                ja_penalty_by_model=_ja_penalty(in_dim, by_id),
                ja_pairs=_ja_pairs(in_dim, by_id),
                discriminating_cases=len(usable),
                excluded_cases=len(in_dim) - len(usable),
            )
        )

    result.perf = _perf(generations, models)
    result.failure_counts = _failure_counts(picked, generations, models)
    result.within_model_deviation = _within_model(result)
    return result


def _mean_by_model(
    stats: list[CaseStat], pick: Any, /
) -> dict[str, float]:  # pick: (CaseStat) -> dict[str, float]
    acc: dict[str, list[float]] = {}
    for stat in stats:
        for model, value in pick(stat).items():
            acc.setdefault(model, []).append(value)
    return {m: sum(v) / len(v) for m, v in acc.items()}


def _ja_pairs(stats: list[CaseStat], by_id: dict[str, Case]) -> int:
    present = {s.case_id for s in stats}
    return sum(
        1
        for s in stats
        if s.lang == "ja" and by_id[s.case_id].pair and by_id[s.case_id].pair in present
    )


def _ja_penalty(stats: list[CaseStat], by_id: dict[str, Case]) -> dict[str, float]:
    """対訳ペアの日英差分（§4・§14.7）。**両側が同じ run にあるペアだけ。**

    **符号は `en − ja`。正の値が「日本語で落ちる」**（§4）。
    S5 の初版は逆に書いていた（§15.3）。"penalty" は正の値が罰であるべきで、
    `ja − en` だと読み手が毎回符号を反転させて読むことになる。
    """
    index = {s.case_id: s for s in stats}
    acc: dict[str, list[float]] = {}
    for stat in stats:
        if stat.lang != "ja":
            continue
        pair = by_id[stat.case_id].pair
        other = index.get(str(pair)) if pair else None
        if other is None:
            continue
        for model, value in stat.by_model.items():
            if model in other.by_model:
                # en − ja。stat が ja 側、other が en 側
                acc.setdefault(model, []).append(other.by_model[model] - value)
    return {m: sum(v) / len(v) for m, v in acc.items()}


def _failure_counts(
    picked: dict[str, dict[str, Any]], generations: dict[str, dict[str, Any]], models: list[str]
) -> dict[str, dict[str, tuple[int, int]]]:
    """失敗型 → モデル → (分子, 母数)。**母数は ``failure_applicable``**（§14.8）。"""
    numer: dict[str, dict[str, int]] = {}
    denom: dict[str, dict[str, int]] = {}
    for gen_id, row in picked.items():
        model = str(generations[gen_id].get("model"))
        for tag in row.get("failure_applicable") or []:
            denom.setdefault(str(tag), {}).setdefault(model, 0)
            denom[str(tag)][model] += 1
        for tag in row.get("failure_tags") or []:
            numer.setdefault(str(tag), {}).setdefault(model, 0)
            numer[str(tag)][model] += 1
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for tag in sorted(set(numer) | set(denom), key=lambda t: failures.PRIORITY.index(t)):
        out[tag] = {m: (numer.get(tag, {}).get(m, 0), denom.get(tag, {}).get(m, 0)) for m in models}
    return out


def _within_model(result: Aggregation) -> dict[str, dict[str, float]]:
    """モデル内の全次元平均からの偏差。**副指標**（§14.11・単体では信用できない）。"""
    out: dict[str, dict[str, float]] = {}
    for model in result.models:
        values = {
            d.dim: d.absolute_by_model[model] for d in result.dims if model in d.absolute_by_model
        }
        if not values:
            continue
        mean = sum(values.values()) / len(values)
        out[model] = {dim: value - mean for dim, value in values.items()}
    return out
