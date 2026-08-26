"""結論がケースの選び方に依存していないかを測る（FLB-QB-001 §17）。

**新しい生成を 1 回もしない。** `aggregate` は対象ケースを引数で受け、
`_pick_scores` が対象外のケースの生成行を捨てる（§14.2）。
したがって **`cases` から 1 件除いて呼び直せば、そのケースが存在しなかった場合の集計**が
そのまま得られる。既存の `scores.jsonl` を読み直すだけで済む。

**閾値を置かない。** 「区間が 0 をまたぐか」と「区間の幅」だけを出す。
どこからを問題と見るかは読み手が決める — ここで定数を決めると、
**その定数を動かすだけで「安定した」と言えてしまう。**
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..cases import Case
from ..store import RunStore
from .aggregate import Aggregation, aggregate

#: jackknife に最低限要るケース数。1 件だと抜いた時点で次元が消える
MIN_CASES_FOR_JACKKNIFE = 2


@dataclass(frozen=True)
class ModelStability:
    """1 次元 1 モデルの z が、ケースを 1 件抜くとどこまで動くか。"""

    model: str
    #: 全ケースで測った z（README に載る値）
    z_full: float
    #: 1 件抜きの再集計で得た z（抜いたケースの数だけある）
    replicates: tuple[float, ...]

    @property
    def z_min(self) -> float:
        return min(self.replicates)

    @property
    def z_max(self) -> float:
        return max(self.replicates)

    @property
    def width(self) -> float:
        return self.z_max - self.z_min

    @property
    def crosses_zero(self) -> bool:
        """区間が 0 をまたぐ = **符号をデータが決めていない**。

        端が 0 ちょうどの場合も「またぐ」に数える。
        z が 0 の複製は、その回に限れば偏りを検出できていない。
        **見逃すより多めに拾う側に倒す。**
        """
        return self.z_min <= 0.0 <= self.z_max


@dataclass(frozen=True)
class DimStability:
    dim: str
    #: ケースの件数（`id` の数）
    cases: int
    #: **独立な観測の件数**（`task` の種類）。§17.1 — `cases` と一致しない
    tasks: int
    #: この run に採点済みの生成があるか。**ケースが定義されていることとは別**
    observed: bool
    models: tuple[ModelStability, ...]

    @property
    def measurable(self) -> bool:
        return self.observed and self.cases >= MIN_CASES_FOR_JACKKNIFE and bool(self.models)

    @property
    def why_not(self) -> str | None:
        """測れなかった理由。**3 通りある**ので 1 つの文言に潰さない。

        潰すと嘘になる。実際 CI の煙テストで
        「ケースが 4 件なので抜けるものが無い」という文が出た —
        ケースは 4 件あり、無かったのは**その run の生成**だった。
        """
        if not self.observed:
            return "この run に採点済みの生成が無い"
        if self.cases < MIN_CASES_FOR_JACKKNIFE:
            return f"ケースが {self.cases} 件なので抜けるものが無い"
        if not self.models:
            return "次元内の σ が 0 なので z が出ていない"
        return None

    @property
    def max_width(self) -> float:
        """優先順位はこれで付ける。**n の違う次元どうしでは比べない**（§17.7）。"""
        return max((m.width for m in self.models), default=0.0)


@dataclass(frozen=True)
class CaseSpread:
    """ケース 1 件がどれだけモデルを分けたか。

    `discriminating` は `max - min >= 1e-9` で判定するので（§14.3）、
    **6 本中 5 本が 0.00 で 1 本だけ 0.20 のケースも「識別力あり」に通る。**
    定義は変えない（変えると公開済みの数字の意味が遡って動く）。
    **実効幅を併記して、読み手が floor を自分で見つけられるようにする**（§17.4）。
    """

    case_id: str
    dim: str
    task: str
    spread: float
    best: float
    worst: float


@dataclass
class Stability:
    run_id: str
    models: list[str] = field(default_factory=list)
    dims: list[DimStability] = field(default_factory=list)
    case_spreads: list[CaseSpread] = field(default_factory=list)
    #: jackknife に掛けられなかった次元（ケースが 1 件しかない）
    skipped: list[str] = field(default_factory=list)


def stability(store: RunStore, cases: list[Case]) -> Stability:
    full = aggregate(store, cases)
    result = Stability(run_id=store.run_id, models=list(full.models))
    result.case_spreads = _case_spreads(full, cases)

    by_dim: dict[str, list[Case]] = {}
    for case in cases:
        by_dim.setdefault(case.dim, []).append(case)

    full_z = {d.dim: dict(d.z_by_model) for d in full.dims}
    observed_dims = {d.dim for d in full.dims}

    for dim in sorted(by_dim):
        in_dim = by_dim[dim]
        tasks = len({c.task for c in in_dim})
        observed = dim in observed_dims
        if not observed or len(in_dim) < MIN_CASES_FOR_JACKKNIFE:
            # **黙って飛ばさない。** 飛ばしたことが出ないと、
            # 表に出ていない次元が「安定していた」と読まれる
            result.skipped.append(dim)
            result.dims.append(
                DimStability(dim=dim, cases=len(in_dim), tasks=tasks, observed=observed, models=())
            )
            continue

        reps: dict[str, list[float]] = {}
        for dropped in in_dim:
            subset = [c for c in cases if c.id != dropped.id]
            replicate = aggregate(store, subset)
            for stat in replicate.dims:
                if stat.dim != dim:
                    continue
                for model, value in stat.z_by_model.items():
                    reps.setdefault(model, []).append(value)

        models = tuple(
            ModelStability(
                model=model,
                z_full=full_z.get(dim, {}).get(model, 0.0),
                replicates=tuple(values),
            )
            for model, values in sorted(reps.items())
            if values
        )
        if not models:
            result.skipped.append(dim)
        result.dims.append(
            DimStability(dim=dim, cases=len(in_dim), tasks=tasks, observed=observed, models=models)
        )
    return result


def _case_spreads(full: Aggregation, cases: list[Case]) -> list[CaseSpread]:
    task_of = {case.id: case.task for case in cases}
    out: list[CaseSpread] = []
    for stat in full.cases:
        values = list(stat.by_model.values())
        if not values:
            continue
        out.append(
            CaseSpread(
                case_id=stat.case_id,
                dim=stat.dim,
                task=task_of.get(stat.case_id, "?"),
                spread=max(values) - min(values),
                best=max(values),
                worst=min(values),
            )
        )
    # **次元順に並べる。** 幅の降順に並べると、それ自体が順位表になる（§16.1）
    return sorted(out, key=lambda c: (c.dim, c.case_id))


def median_width(dim: DimStability) -> float:
    """次元内の幅の中央値。最大幅が 1 モデルの外れ値で決まっていないかを見る。"""
    widths = [m.width for m in dim.models]
    return statistics.median(widths) if widths else 0.0
