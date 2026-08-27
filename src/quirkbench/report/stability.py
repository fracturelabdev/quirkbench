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

#: jackknife に最低限要る**観測**数。1 件だと抜いた時点で次元が消える
MIN_TASKS_FOR_JACKKNIFE = 2

#: z を 0 と見なす幅。`aggregate.SAME_SCORE_EPSILON` と揃える。
#:
#: **これが無いと最下位ビットの符号で判定が変わる。** 実データで
#: `2.05e-17` という複製が出た — 数値としては 0 だが厳密比較では正なので、
#: 「0 をまたぐ」から漏れた。同じ次元の別モデルはたまたま負側に落ちて拾われており、
#: **拾えるかどうかが丸め誤差で決まっていた。**
Z_ZERO_EPSILON = 1e-9


@dataclass(frozen=True)
class ModelStability:
    """1 次元 1 モデルの z が、ケースを 1 件抜くとどこまで動くか。"""

    model: str
    #: 全ケースで測った z（README に載る値）
    z_full: float
    #: 1 件抜きの再集計で得た z。**抜いたケースの数と一致しなければならない**
    replicates: tuple[float, ...]
    #: z が出なかった複製の数。**0 でなければ幅も符号も意味を持たない**
    missing: int = 0

    # **`complete` が False のとき、下の 3 つは意味を持たない。**
    # 落ちないようにしてあるだけで、読んではいけない（`measurable` が False になる）
    @property
    def z_min(self) -> float:
        return min(self.replicates, default=0.0)

    @property
    def z_max(self) -> float:
        return max(self.replicates, default=0.0)

    @property
    def width(self) -> float:
        return self.z_max - self.z_min

    @property
    def claim_collapsed(self) -> bool:
        """**公開した主張が、観測 1 件に乗っている**か（§17.7 の読み方 2）。

        「区間が 0 をまたぐ」だけでは足りない — `z` が 0 付近のモデルは
        1 件抜けば当然どちらにも振れる。**崩れたと言えるのは、
        `z` が 0 から離れているのに区間が 0 をまたぐとき**である。

        「離れている」は**幅との相対**で見る。幅の半分より遠ければ、
        振れ幅では説明できない位置に `z` がある。

        **`z` 自体が 0 のときは主張が無いので、崩れようがない。**
        ここを見落とすと、幅 0・`z` 0 のモデルに対して
        「z=+0.00σ だが +0.00 … +0.00 に振れる」という、
        **振れていないのに振れると言う文**が出る（独立レビューで指摘された）。
        """
        if not self.crosses_zero or abs(self.z_full) <= Z_ZERO_EPSILON:
            return False
        return abs(self.z_full) >= self.width / 2

    @property
    def complete(self) -> bool:
        """抜いた数だけ複製が揃っているか。

        **揃っていないときの幅は 0 に寄る。** 1 件抜いて z が出なくなった次元は
        「その z がそのケース 1 件に依存している」という意味で**最も不安定**なのに、
        残った複製だけで幅を出すと **0.00 = 完全に安定**と表示される。
        **測っていないものを、最も安定した値として出してはいけない。**
        """
        return self.missing == 0 and bool(self.replicates)

    @property
    def crosses_zero(self) -> bool:
        """区間が 0 をまたぐ = **符号をデータが決めていない**。

        端が 0 ちょうどの場合も「またぐ」に数える。
        z が 0 の複製は、その回に限れば偏りを検出できていない。
        **見逃すより多めに拾う側に倒す。**
        """
        return self.z_min <= Z_ZERO_EPSILON and self.z_max >= -Z_ZERO_EPSILON


@dataclass(frozen=True)
class DimStability:
    dim: str
    #: **この run で採点された**ケースの件数。定義済みの件数ではない
    cases: int
    #: **独立な観測の件数**（`task` の種類）。§17.1 — `cases` と一致しない。
    #: **1 件抜きはこの単位で行うので、複製もこの数だけ要る**
    tasks: int
    #: この run に採点済みの生成があるか。**ケースが定義されていることとは別**
    observed: bool
    models: tuple[ModelStability, ...]

    @property
    def measurable(self) -> bool:
        return (
            self.observed
            and self.tasks >= MIN_TASKS_FOR_JACKKNIFE
            and bool(self.models)
            and all(m.complete for m in self.models)
        )

    @property
    def why_not(self) -> str | None:
        """測れなかった理由。**3 通りある**ので 1 つの文言に潰さない。

        潰すと嘘になる。実際 CI の煙テストで
        「ケースが 4 件なので抜けるものが無い」という文が出た —
        ケースは 4 件あり、無かったのは**その run の生成**だった。
        """
        if not self.observed:
            return "この run に採点済みの生成が無い"
        if self.tasks < MIN_TASKS_FOR_JACKKNIFE:
            return f"独立した観測が {self.tasks} 件なので抜けるものが無い"
        if not self.models:
            return "次元内の σ が 0 なので z が出ていない"
        incomplete = [m for m in self.models if not m.complete]
        if incomplete:
            worst = max(m.missing for m in incomplete)
            return f"{worst} 件の複製で z が出ない（この次元の z が特定のケースだけで決まっている）"
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
    #: jackknife に掛けられなかった次元。**理由は 4 通りある**（`DimStability.why_not`）。
    #: 未観測 / 独立した観測が 1 件 / 次元内の σ が 0 / 複製が揃わない。
    #: **1 通りしか書かないと、他の 3 つを説明したことにならない**
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
    # **その run で実際に採点されたケースだけを観測とみなす。**
    # 定義済みの件数で数えると、生成が無いケースを抜いた複製が**恒等**になり、
    # 幅が 0 に寄る。観測 1 件の次元が「完全に安定」と出る（独立レビューで指摘された）
    scored_ids = {stat.case_id for stat in full.cases}

    for dim in sorted(by_dim):
        in_dim = [c for c in by_dim[dim] if c.id in scored_ids]
        # **抜く単位は `task`。** §17.1 で「独立した観測の単位は task」と定めた以上、
        # ケース単位で抜くと ja/en の片割れが残り、**その観測は消えない**。
        # 消えないものを抜いても、答えたい問い（この観測が無かったらどうなるか）に届かない
        tasks = sorted({c.task for c in in_dim})
        observed = dim in observed_dims
        if not observed or len(tasks) < MIN_TASKS_FOR_JACKKNIFE:
            # **黙って飛ばさない。** 飛ばしたことが出ないと、
            # 表に出ていない次元が「安定していた」と読まれる
            result.skipped.append(dim)
            result.dims.append(
                DimStability(
                    dim=dim,
                    cases=len(in_dim),
                    tasks=len(tasks),
                    observed=observed,
                    models=(),
                )
            )
            continue

        reps: dict[str, list[float]] = {}
        # **z が出なかった複製を数える。** 黙って落とすと、複製が 1 本しか無い次元が
        # 幅 0.00 = 完全に安定として表示される（独立レビューで指摘された）
        seen_models = set(full_z.get(dim, {}))
        missing: dict[str, int] = {}
        for dropped_task in tasks:
            subset = [c for c in cases if not (c.dim == dim and c.task == dropped_task)]
            got = _replicate_z(aggregate(store, subset), dim)
            seen_models |= set(got)
            for model in seen_models:
                if model in got:
                    reps.setdefault(model, []).append(got[model])
                else:
                    missing[model] = missing.get(model, 0) + 1

        models = tuple(
            ModelStability(
                model=model,
                z_full=full_z.get(dim, {}).get(model, 0.0),
                replicates=tuple(reps.get(model, ())),
                missing=missing.get(model, 0),
            )
            for model in sorted(seen_models)
        )
        dim_stat = DimStability(
            dim=dim, cases=len(in_dim), tasks=len(tasks), observed=observed, models=models
        )
        # **測れなかったものは、理由が何であれ `skipped` に載せる。**
        # 載せ忘れると、表に出ない次元が「安定していた」と読まれる
        if not dim_stat.measurable:
            result.skipped.append(dim)
        result.dims.append(dim_stat)
    return result


def _replicate_z(replicate: Aggregation, dim: str) -> dict[str, float]:
    """1 件抜きの再集計から、**`dims` の z だけ**を取り出す。

    **複製の `Aggregation` は `dims` 以外どれも意味を持たない。**
    `aggregate` は生成行を絞らず `_pick_scores` だけが対象外ケースを捨てるので、
    複製では `unscored` が抜いたケースの生成件数だけ膨らみ（実測で 0 → 120）、
    `failure_counts` は母数が欠け、`perf` と `total_generations` は全件のまま残る。

    **型としては全部読めてしまう**ので、読める場所を関数 1 つに絞る。
    ここを経由しない読み方が増えたら、その時点で静かに壊れる（独立レビューで指摘された）。
    """
    for stat in replicate.dims:
        if stat.dim == dim:
            return dict(stat.z_by_model)
    return {}


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
