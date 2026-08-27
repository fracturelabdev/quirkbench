"""ideate 次元の採点（FLB-QB-001 §5・§13）。

**埋め込みは stub。** 実モデルは CI に無く、また実モデルを使うと
「指標の性質を見るテスト」が「モデルの出来を見るテスト」になる。
ここで固定するのは§13.1 の操作的定義で、実測は手元で行う（§13.8 条件 1・7）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

import pytest

from quirkbench.cases import parse_case
from quirkbench.parse import parse
from quirkbench.scorers import EmbedderRequired
from quirkbench.scorers.ideate import OFFTOPIC_THRESHOLD, score_ideate

RAW_CASE = {
    "id": "idea",
    "dim": "ideate",
    "task": "t",
    "lang": "ja",
    "prompt": "5 つ考えてください",
    "failure": {"format": "none", "count": {"n": 5, "pattern": "numbered_list"}},
    "score": {
        "kind": "ideate",
        "count": {"n": 5, "pattern": "numbered_list"},
        "topic": "住宅街にある席数12の小さな喫茶店の名前",
        "coverage_terms": [["珈琲", "コーヒー"], ["街", "町"], ["小", "静"]],
    },
}
CASE = parse_case(RAW_CASE, Path("idea.yaml"))


class PlanarEmbedder:
    """本文を角度に写す埋め込み器。**多様性を意図した値に置ける**のがねらい。

    ``ANGLES`` に無い本文は、お題から遠い角度（= 脱線）に落とす。
    """

    model = "planar"
    ANGLES: ClassVar[dict[str, float]] = {"住宅街にある席数12の小さな喫茶店の名前": 0.0}

    @property
    def fingerprint(self) -> str:
        return "planar"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            angle = self.ANGLES.get(text, 1.4)
            out.append([math.cos(angle), math.sin(angle)])
        return out


def make_embedder(angles: dict[str, float]) -> PlanarEmbedder:
    embedder = PlanarEmbedder()
    embedder.ANGLES = {**PlanarEmbedder.ANGLES, **angles}
    return embedder


def numbered(items: list[str]) -> str:
    return "\n".join(f"{index + 1}. {item}" for index, item in enumerate(items))


def run(items: list[str], angles: dict[str, float] | None = None, case=CASE):
    text = numbered(items)
    embedder = make_embedder(angles or {item: 0.1 for item in items})
    return score_ideate(parse(text, case.failure), case, executor=None, embedder=embedder)


# ------------------------------------------------------------ 依存の強制


def test_missing_embedder_raises() -> None:
    """**既定でローカル計算に落とさない**（§13.6・完了条件 4）。

    落とすと、埋め込みモデル無しでもそれらしい数字が出る経路ができ、
    多様性を測っていない run が測ったものと同じ形で並ぶ。
    """
    with pytest.raises(EmbedderRequired, match="idea"):
        score_ideate(parse("1. あ", CASE.failure), CASE, executor=None, embedder=None)


# -------------------------------------------------- 第 1 段: 形態フィルタ


@pytest.mark.parametrize(
    ("item", "reason"),
    [
        ("あ", "too_short"),
        ("41", "no_letters"),
        ("1234567890", "no_letters"),
        ("!!!???", "no_letters"),
        ("Corner Coffee", "wrong_script"),
    ],
)
def test_morph_filter_rejects(item: str, reason: str) -> None:
    result = run([item, "街角の珈琲店", "静かな小窓", "路地の灯", "隣の喫茶"])
    assert result.sub_metrics["morph_rejected"].get(reason) == 1


def test_morph_filter_does_not_reject_normal_names() -> None:
    """**実測で誤検出ゼロ**（§5）。正常な店名を落としたら指標が壊れる。"""
    items = ["街角の珈琲店", "小さな窓辺", "静かな路地", "隣町のコーヒー", "灯りの喫茶"]
    result = run(items)
    assert result.sub_metrics["morph_rejected"] == {}
    assert result.sub_metrics["valid_rate_morph"] == 1.0


def test_morph_filter_does_not_use_detect_language() -> None:
    """短い和名は `detect_language` では必ず ``unknown`` になる（下限 20 文字）。

    そちらを使っていたら、正常な店名が全部 wrong_script になる。
    """
    from quirkbench.textstats import char_class_of, detect_language

    assert detect_language(char_class_of("街角の珈琲店")) == "unknown"
    assert run(["街角の珈琲店"] * 5).sub_metrics["morph_rejected"] == {}


# -------------------------------------------- 第 2 段: 埋め込みゲート


def test_offtopic_items_are_excluded_and_counted() -> None:
    items = ["街角の珈琲店", "小さな窓辺", "静かな路地", "隣町のコーヒー", "全然関係ない話題"]
    angles = {item: 0.1 for item in items[:4]}
    angles["全然関係ない話題"] = 1.5  # cos(1.5-0.0) ≈ 0.07 < 0.35
    result = run(items, angles)
    assert result.sub_metrics["valid_count"] == 4
    assert result.sub_metrics["offtopic_rate"] == pytest.approx(0.2)


def test_morph_value_is_reported_alongside() -> None:
    """**第 1 段のみの値を必ず併記する**（§13.1）。

    第 2 段のマージンは実測で +0.023 しかない。併記していないと、
    指標が埋め込みの不安定さを測り始めても再実行するまで分からない。
    """
    items = ["街角の珈琲店", "小さな窓辺", "静かな路地", "隣町のコーヒー", "無関係"]
    angles = {item: 0.1 for item in items[:4]}
    angles["無関係"] = 1.5
    result = run(items, angles)
    assert result.sub_metrics["valid_rate"] == pytest.approx(0.8)
    assert result.sub_metrics["valid_rate_morph"] == 1.0


def test_threshold_is_not_case_overridable() -> None:
    """閾値をケースが持つと、落ちたときに閾値を動かして通すことになる（§13.3）。"""
    raw = {**RAW_CASE, "score": {**RAW_CASE["score"], "offtopic_threshold": 0.99}}
    case = parse_case(raw, Path("x.yaml"))
    items = ["街角の珈琲店", "小さな窓辺", "静かな路地", "隣町のコーヒー", "灯りの喫茶"]
    # ケース側の指定は読まれないので、全件が既定の閾値で通る
    assert run(items, case=case).sub_metrics["valid_count"] == 5
    assert OFFTOPIC_THRESHOLD == 0.35


# -------------------------------------------------------------- 3 因子


def test_valid_rate_denominator_is_the_requested_count() -> None:
    """**産出件数を分母にしない**（§13.1）。

    1 案だけ返して valid なら 1.0 になり、課題を回避したモデルが満点を取る。
    """
    result = run(["街角の珈琲店"])
    assert result.sub_metrics["items_produced"] == 1
    assert result.sub_metrics["valid_rate"] == pytest.approx(0.2)


def test_overproduction_is_not_penalised_by_the_score() -> None:
    """超過は count_mismatch タグが持つ。スコアと失敗型に同じ役割を持たせない（§6）。"""
    items = [f"街角の珈琲店{i}" for i in range(7)]
    result = run(items)
    assert result.sub_metrics["valid_rate"] == 1.0
    assert result.tags == ()


def test_coverage_counts_groups_not_words() -> None:
    """1 グループ内はいずれか 1 語で被覆（OR）。"""
    items = ["街角の珈琲店", "街の店", "街道の店", "街路の店", "街区の店"]
    result = run(items)
    # 「珈琲」と「街」は当たり、3 つ目のグループ（小 / 静）が外れる
    assert result.sub_metrics["coverage"] == pytest.approx(2 / 3)
    assert result.sub_metrics["coverage_missed"] == [2]


def test_coverage_ignores_items_rejected_by_the_morph_filter() -> None:
    """**除外された案に被覆を稼がせない**（§13.1）。屑を大量に出すほど上がってしまう。

    落ちる案のほうに被覆語を持たせないと、valid で数えても items で数えても
    同じ値になり、**この規則を外しても気づけない**（変異検査で実際に生き残った）。
    """
    # 被覆語を持つ案を形態フィルタで落とすには、長さ下限を使うしかない
    # （数字・記号だけの案は、そもそも被覆語を持てない）
    items = ["街の店", "町の家", "路地の角", "窓の店", "珈"]
    result = run(items)
    assert result.sub_metrics["morph_rejected"] == {"too_short": 1}
    assert result.sub_metrics["valid_count"] == 4
    assert result.sub_metrics["coverage"] == pytest.approx(1 / 3)


def test_coverage_ignores_items_rejected_by_the_offtopic_gate() -> None:
    """**第 2 段で落ちた案にも被覆を稼がせない。**

    被覆語を持つ案を落とすには、形態フィルタ（数字・記号・長さ）では届かない。
    脱線ゲートなら、まともな日本語のまま除外できる。
    """
    items = ["街の店", "街の家", "街の角", "街の窓", "珈琲の静かな家"]
    angles = {item: 0.1 for item in items[:4]}
    angles["珈琲の静かな家"] = 1.5
    result = run(items, angles)
    assert result.sub_metrics["valid_count"] == 4
    # valid には「街」しか無い。items で数えると「珈琲」「静」まで当たって 1.0 になる
    assert result.sub_metrics["coverage"] == pytest.approx(1 / 3)


def test_coverage_matching_is_normalised() -> None:
    """全角半角の揺れで被覆を落とさない。"""
    raw = {**RAW_CASE, "score": {**RAW_CASE["score"], "coverage_terms": [["ｺｰﾋｰ"]]}}
    case = parse_case(raw, Path("n.yaml"))
    result = run(
        ["コーヒーの店", "コーヒーの家", "コーヒーの窓", "コーヒーの灯", "コーヒーの角"], case=case
    )
    assert result.sub_metrics["coverage"] == 1.0


def test_diversity_needs_two_valid_items() -> None:
    """1 案では 0.0（§13.1）。値が無いからではなく、多様性を満たしていないから。"""
    result = run(["街角の珈琲店"])
    assert result.sub_metrics["diversity"] == 0.0
    assert result.score == 0.0


def test_identical_items_score_low_not_zero_on_validity() -> None:
    """重複だらけ: valid は満点でも多様性で潰れる（§5 の実測パターン）。"""
    items = ["街角の小さな珈琲店"] * 5
    result = run(items)
    assert result.sub_metrics["valid_rate"] == 1.0
    assert result.sub_metrics["diversity"] == pytest.approx(0.0)
    assert result.score == pytest.approx(0.0)


# ---------------------------------------------- §5 の 4 パターンの順序


def _pattern_score(items: list[str], angles: dict[str, float]) -> float:
    return run(items, angles).score


def test_composite_orders_the_four_patterns() -> None:
    """**完了条件 7**（§13.8）: 正常 > 重複 > 支離滅裂・退化。

    §5 の数値の再現ではない。§5 は試作での測定で、実装は分岐を確定させたぶん
    値が動きうる。**順序が保たれることだけを条件にする。**
    """
    normal = ["街角の珈琲店", "小さな窓辺", "静かな路地", "隣町のコーヒー", "灯りの喫茶"]
    normal_angles = {item: 0.1 * index for index, item in enumerate(normal)}

    duplicated = ["街角の珈琲店", "街角の珈琲店", "街角の珈琲店", "街角の珈琲店", "街角の珈琲店"]

    incoherent = ["街角の珈琲店", "無関係A", "無関係B", "無関係C", "無関係D"]
    incoherent_angles = {"街角の珈琲店": 0.1}  # 残り 4 件は既定の 1.4 で脱線

    degenerate = ["41", "43", "49", "51", "53"]  # 実測: 0.5b が返した出力（§5）

    scores = {
        "normal": _pattern_score(normal, normal_angles),
        "duplicated": _pattern_score(duplicated, {"街角の珈琲店": 0.1}),
        "incoherent": _pattern_score(incoherent, incoherent_angles),
        "degenerate": _pattern_score(degenerate, {}),
    }
    assert scores["normal"] > scores["duplicated"]
    assert scores["duplicated"] >= scores["incoherent"]
    assert scores["degenerate"] == 0.0
    assert scores["incoherent"] == 0.0


def test_multiplication_kills_high_diversity_alone() -> None:
    """**多様性だけが高い出力は上がらない。** 初期案が退化していた点（§5）。"""
    junk = ["1234", "5678", "9012", "3456", "7890"]
    result = run(junk, {item: 0.3 * index for index, item in enumerate(junk)})
    assert result.sub_metrics["valid_count"] == 0
    assert result.score == 0.0


def test_scorer_owns_no_failure_vocabulary() -> None:
    """count_mismatch は failures.py が単独で所有する。"""
    result = run(["街角の珈琲店", "小さな窓辺"])
    assert result.tags == ()
    assert result.applicable == ()


def test_coverage_folds_case_for_english() -> None:
    """**畳まないと英語で成立しない**（§15.7）。

    実測: 英語版で 6 モデル中 4 モデルの coverage が 0.000 になった。
    原因は `Cobblestone Cafe` が語彙の `cafe` に一致しないことで、
    **測っていたのは大文字の使い方**だった。
    """
    raw = {
        **RAW_CASE,
        "lang": "en",
        "score": {**RAW_CASE["score"], "coverage_terms": [["cafe"], ["corner"]]},
    }
    case = parse_case(raw, Path("en.yaml"))
    items = [
        "Cobblestone Cafe",
        "The Quiet Corner",
        "Willow Nook",
        "Sunbeam Bistro",
        "Morning Mocha",
    ]
    result = run(items, case=case)
    assert result.sub_metrics["coverage"] == pytest.approx(1.0)


def test_case_folding_does_not_leak_into_expect_comparison() -> None:
    """**`parse.normalize` 側は畳まない。**

    `instruct` の期待値比較に使われるので、`Ichiro Tanaka` と
    `ichiro tanaka` を同じにしてはいけない。
    """
    from quirkbench.parse import normalize

    assert normalize("Ichiro Tanaka") != normalize("ichiro tanaka")
