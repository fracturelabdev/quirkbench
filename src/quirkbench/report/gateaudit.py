"""脱線ゲートが効いているかを測る（FLB-QB-001 §19）。

**ゲートを直す装置ではない。** 直そうとして 2 度失敗している
（§5 のテンプレ案・§19.4 のアンカー案）ので、3 度目はやらない。
**測れないことを測って出す**という、`qb stability` と同じ形を採る。

**除外率だけでは足りない。** 除外 0 件は「脱線した案が無かった」とも
「ゲートが無情報だった」とも読める。**発火したかではなく、発火が正しかったか**を測る。

**判定の閾値を持たない**（`stability.py` と同じ規律）。何 % 以上を無情報と呼ぶかを
定数にすると、**その定数を動かすだけで「ゲートは効いている」と言えてしまう。**
出すのは数字と読み方の規則だけで、判断は読み手がする。

対照の作り方は §19.2 —
**別のお題への案を、このお題のゲートに掛ければ、それは定義上の脱線である。**
この判定に人手は要らないので、§18.2 が「実装者本人がラベルを付けるとバイアスで汚れる」
と言って止まっていた測定が、バイアス無しで通る。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..cases import Case
from ..embed import Embedder, cosine
from ..parse import parse
from ..scorers.ideate import OFFTOPIC_THRESHOLD, morph_reject
from ..store import RunStore

#: 運用点。**正例の誤除外をここまでに抑えた位置で、対照の見逃し率を読む。**
#:
#: **これは判定の閾値ではない。** 2 つの方式を同じ条件で比べるための固定で、
#: `stability.MIN_TASKS_FOR_JACKKNIFE` と同型。「何 % の見逃しを不合格とするか」は
#: **決めない** — そこを定数にすると、動かすだけで合格を作れる。
FALSE_REJECT_TARGET = 0.05

#: AUC を出すのに要る正例の下限。1 件では分布にならない
MIN_POSITIVES = 2


def auc(positives: list[float], controls: list[float]) -> float:
    """正例が対照より高く出る確率。**0.5 が無情報、1.0 が完全分離。**

    **タイは 0.5 で数える。** 落とすと、同値だらけの分布が完全分離にも
    完全逆転にも見える。

    **母数が空なら ``nan``。** 0 を返すと「完全逆転」と区別できず、
    0.5 を返すと「無情報」と区別できない。**未測定は未測定として出す**（§14.7）。
    """
    if not positives or not controls:
        return math.nan
    wins = 0.0
    for p in positives:
        for c in controls:
            if p > c:
                wins += 1.0
            elif p == c:
                wins += 0.5
    return wins / (len(positives) * len(controls))


def operating_point(
    positives: list[float], controls: list[float], *, false_reject: float = FALSE_REJECT_TARGET
) -> tuple[float, float, int]:
    """``(閾値, 対照の見逃し率, 実際に落とす正例の件数)``。

    **切り上げない。** 10 件で 5% は 0.5 件だが、1 件落とすと実際の誤除外は
    10% になり、**目標のほうを超える**。少ない母数では 1 件も落とさない。

    **閾値ちょうどの対照は見逃しに数える。** ゲートは ``< 閾値`` で落とすので、
    等しい値は通る。ここを落とす側に数えると、**見逃しを少なく見せる**ことになる。
    """
    if not positives:
        return (math.nan, math.nan, 0)
    ordered = sorted(positives)
    drop = int(len(ordered) * false_reject)
    threshold = ordered[drop]
    if not controls:
        return (threshold, math.nan, drop)
    leaked = sum(1 for c in controls if c >= threshold)
    return (threshold, leaked / len(controls), drop)


@dataclass(frozen=True)
class CaseGate:
    """1 ケースぶんのゲート検査。

    **`measurable` が False のとき、下の数字は読んではいけない。**
    落ちないように ``nan`` を入れてあるだけで、意味は持たない。
    """

    case_id: str
    dim: str
    task: str
    lang: str
    positives: int
    controls: int
    #: 対照に使ったケース。**空なら測っていない**
    control_cases: tuple[str, ...]
    auc: float
    threshold: float
    #: 主指標。運用点で対照が何割通るか
    control_leak: float
    #: 運用点で落ちる正例の件数（母数は `positives`）
    rejected: int
    #: 現行の `OFFTOPIC_THRESHOLD` が実際に落としている正例の割合
    current_reject_rate: float
    why_not: str | None = None

    @property
    def measurable(self) -> bool:
        return self.why_not is None


@dataclass
class GateAudit:
    run_id: str
    threshold_in_use: float = OFFTOPIC_THRESHOLD
    cases: list[CaseGate] = field(default_factory=list)
    #: 測れなかったケース。**黙って落とさない** — 表に出ないものは
    #: 「効いていた」と読まれる（`stability.skipped` と同じ理由）
    skipped: list[str] = field(default_factory=list)


def gate_audit(store: RunStore, cases: list[Case], embedder: Embedder) -> GateAudit:
    """既存の run を読み直して、ケースごとにゲートの分離能を出す。

    **新しい生成はしない**（`qb stability` と同じ・§17.2）。ただし
    **埋め込みは要る** — 案とお題の類似度を測り直すため。同じ run に対しては
    `embeddings.jsonl` のキャッシュが効くので、追加の計算はほとんど無い。
    """
    gated = [c for c in cases if c.score.get("kind") == "ideate"]
    result = GateAudit(run_id=store.run_id)
    if not gated:
        return result

    by_id = {c.id: c for c in gated}
    rows, _report = store.generations()
    items: dict[str, list[str]] = {c.id: [] for c in gated}
    for row in rows:
        case = by_id.get(str(row.get("case_id", "")))
        if case is None or row.get("error"):
            continue
        for item in parse(str(row.get("response", "")), case.failure).items:
            # **採点器と同じ第 1 段を通す**（§19.7）。ここで別の判定を持つと、
            # 測っているゲートと採点しているゲートが違うものになる
            if morph_reject(item, case.lang) is None:
                items[case.id].append(item)

    # **お題と案をまとめて 1 回で埋め込む。** バッチ構成は値を変えない（§13.4）
    texts: list[str] = []
    for case in gated:
        texts.append(str(case.score.get("topic", "")))
        texts.extend(items[case.id])
    # **重複は 1 本にまとめてから投げる。** 同じ案が別の生成に何度も現れるので、
    # まとめないと同じテキストを何度も埋め込むことになる
    unique = list(dict.fromkeys(texts))
    vectors = dict(zip(unique, embedder.embed(unique), strict=True))

    for case in sorted(gated, key=lambda c: c.id):
        result.cases.append(_audit_case(case, gated, items, vectors))
    result.skipped = [c.case_id for c in result.cases if not c.measurable]
    return result


def _controls_for(case: Case, gated: list[Case]) -> list[Case]:
    """対照に使えるケース。**同じ次元・同じ言語・別 `task`**（§19.2）。

    **言語をまたがない。** 日本語のお題に英語の案をぶつければ類似度は
    構造的に低く出て、**ゲートが実際より良く見える。**

    **対訳ペアを対照にしない。** 同じ `task` の別言語は「別のお題」ではない（§17.1）。
    `task` で切れば対訳は自動的に外れる。
    """
    return [
        other
        for other in gated
        if other.dim == case.dim and other.lang == case.lang and other.task != case.task
    ]


def _audit_case(
    case: Case,
    gated: list[Case],
    items: dict[str, list[str]],
    vectors: dict[str, list[float]],
) -> CaseGate:
    topic = str(case.score.get("topic", ""))
    topic_vector = vectors.get(topic)
    controls = _controls_for(case, gated)
    positives = [cosine(topic_vector, vectors[t]) for t in items[case.id]] if topic_vector else []
    control_scores: list[float] = []
    if topic_vector:
        for other in controls:
            control_scores.extend(cosine(topic_vector, vectors[t]) for t in items[other.id])

    why: str | None = None
    if len(positives) < MIN_POSITIVES:
        # **3 通りを 1 つの文言に潰さない**（`DimStability.why_not` と同じ理由）
        why = (
            "この run に採点済みの生成が無い"
            if not items[case.id]
            else f"形態フィルタを通った案が {len(positives)} 件しかない"
        )
    elif not controls:
        why = "同じ次元・同じ言語に別の task が無いので対照を作れない"
    elif not control_scores:
        why = "対照ケースにこの run の生成が無い"

    a = auc(positives, control_scores)
    threshold, leak, rejected = operating_point(positives, control_scores)
    current = (
        sum(1 for p in positives if p < OFFTOPIC_THRESHOLD) / len(positives)
        if positives
        else math.nan
    )
    return CaseGate(
        case_id=case.id,
        dim=case.dim,
        task=case.task,
        lang=case.lang,
        positives=len(positives),
        controls=len(control_scores),
        control_cases=tuple(sorted(c.id for c in controls)),
        auc=a,
        threshold=threshold,
        control_leak=leak,
        rejected=rejected,
        current_reject_rate=current,
        why_not=why,
    )
