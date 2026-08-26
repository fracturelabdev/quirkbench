"""`ideate` 次元の採点 — 「発想の質」ではなく**制約内多様性**を測る（FLB-QB-001 §5・§13）。

    ideate = valid_rate × coverage × diversity

**乗算にするのは、単独の指標を見ないため。** いずれかが 0 なら全体が 0 になるので、
**多様性だけが高い出力（支離滅裂）は上がらない** — これが初期案（ペア間距離の平均のみ）が
退化していた点で、実測で「支離滅裂 0.598 > 正常 0.528」が確認されている（§5）。

**この採点器は「発想が優秀か」を測っていない。** 測っているのは
constrained diversity で、最終的な質の判断は人が比較ビューで行う。

判定の 2 段構えについては §13.1 を見ること。要点だけ:

- **第 1 段（形態フィルタ）は決定的**。実測で誤検出ゼロ
- **第 2 段（埋め込みゲート）はマージンが +0.023 しかない**。だから
  第 1 段のみの ``valid_rate_morph`` を必ず併記して、後から寄与を切り出せるようにする
"""

from __future__ import annotations

from typing import Any

from ..cases import Case
from ..embed import Embedder, cosine, mean_pairwise_distance
from ..parse import Parsed, normalize
from ..textstats import char_class_of
from . import EmbedderRequired, Executor, ScoreResult

#: 案とお題の cos 類似度がこれ未満なら脱線とみなす。
#:
#: **ケース YAML から上書きできない**（§13.3）。ケースごとに動かせると、
#: 落ちたときに閾値のほうを動かして通すことになり、測っているのがモデルの癖では
#: なくケース作者の調整になる。§12.2 で assert 数について決めたのと同じ論法。
#:
#: 値の根拠は §5 の実測: 正例の最小 0.3851 / ジャンク 0.3623 / 無関係なお題 0.246〜0.334。
OFFTOPIC_THRESHOLD = 0.35

#: 形態フィルタの下限。正規化後にこれ未満の文字数なら案とみなさない
MIN_ITEM_CHARS = 2


def _morph_reject(item: str, lang: str) -> str | None:
    """形態フィルタ（第 1 段・§13.1）。落とす理由を返す。通れば ``None``。

    **`textstats.detect_language` を使わない。** あちらは
    ``MIN_LETTERS_FOR_LANG = 20`` の下限を持ち、店名のような短い文字列は
    必ず ``unknown`` を返す。案 1 件は原理的にこの下限を下回る。
    """
    text = normalize(item)
    if len(text) < MIN_ITEM_CHARS:
        return "too_short"
    stats = char_class_of(text)
    # 「数字・記号のみ」を除外語のリストで書かない。リストを持つと、
    # その網羅度がそのまま指標になる
    if stats.letters == 0:
        return "no_letters"
    if lang == "ja" and stats.kana + stats.kanji == 0:
        return "wrong_script"
    return None


def _fold(text: str) -> str:
    """被覆の照合用の正規化。**NFKC に加えて大文字小文字を畳む。**

    畳まないと英語で成立しない。実測（S6）: 英語版で 6 モデル中 4 モデルの
    ``coverage`` が **0.000** になった。原因は ``Cobblestone Cafe`` が
    語彙の ``cafe`` に一致しないことで、**測っていたのは大文字の使い方**だった。

    **`parse.normalize` 側は畳まない。** あちらは `instruct` の期待値比較にも
    使われ、``Ichiro Tanaka`` と ``ichiro tanaka`` を同じにしてはいけない。
    被覆は「その語に触れたか」を見るだけなので、ここだけ畳む。
    """
    return normalize(text).casefold()


def _coverage(items: list[str], groups: Any) -> tuple[float, list[int]]:
    """被覆したグループ数 ÷ 全グループ数。``(率, 未被覆グループの添字)``。

    照合対象は**valid な案だけ**（§13.1）。除外された屑を入れると、
    屑を大量に出すほど被覆が上がる。
    """
    if not isinstance(groups, list) or not groups:
        return 0.0, []
    haystack = _fold("\n".join(items))
    missed: list[int] = []
    hit = 0
    for index, group in enumerate(groups):
        words = group if isinstance(group, list) else [group]
        if any(_fold(str(word)) in haystack for word in words):
            hit += 1
        else:
            missed.append(index)
    return hit / len(groups), missed


def score_ideate(
    parsed: Parsed, case: Case, *, executor: Executor | None, embedder: Embedder | None
) -> ScoreResult:
    """``executor`` は受け取るが使わない（registry の型を 1 つに保つため・§12.10）。"""
    del executor
    if embedder is None:
        raise EmbedderRequired(f"{case.id}: 代理指標の採点には embedder が要る")

    spec: dict[str, Any] = case.score
    wanted = int((spec.get("count") or {})["n"])
    topic = str(spec.get("topic", ""))

    # 案の切り出しは failure.count と同一実装（§13.2）。`Parsed.items` がそれ。
    items = list(parsed.items)

    # --- 第 1 段: 形態フィルタ（決定的）
    morph_ok: list[str] = []
    rejected: dict[str, int] = {}
    for item in items:
        reason = _morph_reject(item, case.lang)
        if reason is None:
            morph_ok.append(item)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1

    # --- 第 2 段: 埋め込みゲート（補助・マージンが薄い）
    # お題を先頭に置いて 1 回のバッチで埋め込む。バッチ構成は値を変えない（§13.4）
    vectors = embedder.embed([topic, *morph_ok]) if morph_ok else []
    valid: list[str] = []
    valid_vectors: list[list[float]] = []
    offtopic = 0
    if vectors:
        topic_vector = vectors[0]
        for item, vector in zip(morph_ok, vectors[1:], strict=True):
            if cosine(topic_vector, vector) < OFFTOPIC_THRESHOLD:
                offtopic += 1
            else:
                valid.append(item)
                valid_vectors.append(vector)

    # --- 3 因子
    # 分母は**要求件数**。産出件数にすると、1 案だけ返して valid なら 1.0 になり、
    # 課題を回避したモデルが満点を取る（§13.1）。
    # 超過側はスコアで罰さない — 件数の逸脱は count_mismatch タグが持つ
    valid_rate = min(1.0, len(valid) / wanted) if wanted else 0.0
    valid_rate_morph = min(1.0, len(morph_ok) / wanted) if wanted else 0.0
    coverage, missed = _coverage(valid, spec.get("coverage_terms"))
    diversity = mean_pairwise_distance(valid_vectors)

    return ScoreResult(
        score=valid_rate * coverage * diversity,
        sub_metrics={
            "items_produced": len(items),
            "items_wanted": wanted,
            "valid_count": len(valid),
            "valid_rate": valid_rate,
            # 第 2 段を通さない値。埋め込みゲートの寄与を後から切り出すために
            # 必ず併記する（§13.1）。マージンが +0.023 しかないため
            "valid_rate_morph": valid_rate_morph,
            "coverage": coverage,
            "coverage_missed": missed,
            "diversity": diversity,
            # 脱線しやすさ自体が癖である（§5）。分母は形態フィルタを通った件数
            "offtopic_rate": offtopic / len(morph_ok) if morph_ok else 0.0,
            "morph_rejected": rejected,
        },
        # **失敗型のタグは付けない。** count_mismatch は failures.py が単独で所有する。
        # offtopic をタグにしないのは §13.1 の判断 — 案ごとの性質を生成単位の二値に
        # 畳むには第 2 の閾値が要り、+0.023 のマージンの上に積むと
        # 分布が測るのは検出器の不安定さになる
        tags=(),
        applicable=(),
    )
