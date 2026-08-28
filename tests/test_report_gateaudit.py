"""ゲートの分離能を測る装置のテスト（FLB-QB-001 §19.7）。

**この装置は判定の閾値を持たない。** 持たせると、その定数を動かすだけで
「ゲートは効いている」と言えてしまう（`stability.py` と同じ規律）。
テストもそれを前提に、**数字が正しく出ること**だけを見る。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

import pytest

from quirkbench.cases import parse_case
from quirkbench.report.gateaudit import (
    FALSE_REJECT_TARGET,
    auc,
    gate_audit,
    operating_point,
)
from quirkbench.store import RunStore


class TestAuc:
    def test_完全分離は_1(self) -> None:
        assert auc([0.8, 0.9], [0.1, 0.2]) == 1.0

    def test_完全逆転は_0(self) -> None:
        assert auc([0.1, 0.2], [0.8, 0.9]) == 0.0

    def test_同値だけなら_0_5(self) -> None:
        """**タイは 0.5 で数える。** 落とすと、同値だらけの分布が
        完全分離にも完全逆転にも見える。"""
        assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5

    def test_半分重なると_0_5付近(self) -> None:
        assert auc([0.3, 0.7], [0.3, 0.7]) == 0.5

    def test_母数が空なら_nan(self) -> None:
        """**0/0 を 0.5 と書かない。** 無情報と未測定は違う（§14.7）。"""
        assert auc([], [0.1]) != auc([], [0.1])  # nan は自分自身と等しくない
        assert auc([0.1], []) != auc([0.1], [])


class TestOperatingPoint:
    def test_誤除外を許す件数だけ下から切る(self) -> None:
        """20 件・目標 5% なら 1 件だけ落とせる。"""
        positives = [i / 100 for i in range(20)]  # 0.00 .. 0.19
        threshold, _leak, rejected = operating_point(positives, [0.0], false_reject=0.05)
        assert threshold == pytest.approx(0.01)
        assert rejected == 1

    def test_件数が少なければ_1件も落とさない(self) -> None:
        """**切り上げない。** 10 件で 5% は 0.5 件だが、
        1 件落とすと実際の誤除外は 10% になり、目標を超える。"""
        positives = [i / 10 for i in range(10)]
        threshold, _leak, rejected = operating_point(positives, [0.0], false_reject=0.05)
        assert rejected == 0
        assert threshold == pytest.approx(0.0)

    def test_見逃し率は閾値以上の対照の割合(self) -> None:
        positives = [0.5] * 20
        controls = [0.1, 0.2, 0.6, 0.7]
        _threshold, leak, _rejected = operating_point(positives, controls, false_reject=0.05)
        assert leak == pytest.approx(0.5)

    def test_境界の対照は見逃しに数える(self) -> None:
        """**閾値ちょうどは通る**（ゲートは `< 閾値` で落とす）。
        見逃しを少なく見せる側に倒さない。"""
        positives = [0.5] * 20
        _threshold, leak, _rejected = operating_point(positives, [0.5], false_reject=0.05)
        assert leak == 1.0

    def test_目標は既定値を持つ(self) -> None:
        assert FALSE_REJECT_TARGET == 0.05


# --------------------------------------------------------------- 統合


class PlanarEmbedder:
    """本文を角度に写す埋め込み器。**類似度を意図した値に置ける**のがねらい。"""

    ANGLES: ClassVar[dict[str, float]] = {}

    @property
    def fingerprint(self) -> str:
        return "planar"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [math.cos(self.ANGLES.get(t, 1.4)), math.sin(self.ANGLES.get(t, 1.4))] for t in texts
        ]


def ideate_case(case_id: str, task: str, lang: str, topic: str):
    return parse_case(
        {
            "id": case_id,
            "dim": "ideate",
            "task": task,
            "lang": lang,
            "prompt": "p",
            "failure": {"format": "none", "count": {"n": 3, "pattern": "numbered_list"}},
            "score": {
                "kind": "ideate",
                "count": {"n": 3, "pattern": "numbered_list"},
                "topic": topic,
                "coverage_terms": [["あ"]],
            },
        },
        Path(f"{case_id}.yaml"),
    )


def build_run(tmp_path, rows, *, run="r"):
    """``rows`` は ``(case_id, 応答本文)``。"""
    store = RunStore(tmp_path, run)
    store.dir.mkdir(parents=True, exist_ok=True)
    for index, (case_id, response) in enumerate(rows):
        store.append_generation(
            {
                "gen_id": f"g{index}",
                "model": "m",
                "case_id": case_id,
                "seed": 1000,
                "response": response,
                "error": None,
            }
        )
    return store


def numbered(*items: str) -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))


class TestGateAudit:
    def test_対照が無ければ測らない(self, tmp_path) -> None:
        """**タスクが 1 つの次元では対照を作れない。** 0 と書かず、理由を出す。"""
        case = ideate_case("ideate-a-ja", "a", "ja", "お題A")
        store = build_run(tmp_path, [("ideate-a-ja", numbered("あさひ", "ゆうひ", "よあけ"))])
        result = gate_audit(store, [case], PlanarEmbedder())
        (row,) = result.cases
        assert not row.measurable
        assert "別の task が無い" in row.why_not
        assert result.skipped == ["ideate-a-ja"]

    def test_対訳ペアは対照にならない(self, tmp_path) -> None:
        """同じ `task` の別言語は「別のお題」ではない（§17.1）。"""
        cases = [
            ideate_case("ideate-a-ja", "a", "ja", "お題A"),
            ideate_case("ideate-a-en", "a", "en", "topicA"),
        ]
        store = build_run(
            tmp_path,
            [
                ("ideate-a-ja", numbered("あさひ", "ゆうひ", "よあけ")),
                ("ideate-a-en", numbered("dawn", "dusk", "noon")),
            ],
        )
        result = gate_audit(store, cases, PlanarEmbedder())
        for row in result.cases:
            assert not row.measurable
            assert row.control_cases == ()

    def test_対照は言語をまたがない(self, tmp_path) -> None:
        """またぐと類似度が構造的に低く出て、**ゲートが実際より良く見える**。"""
        cases = [
            ideate_case("ideate-a-ja", "a", "ja", "お題A"),
            ideate_case("ideate-b-ja", "b", "ja", "お題B"),
            ideate_case("ideate-b-en", "b", "en", "topicB"),
        ]
        store = build_run(
            tmp_path,
            [
                ("ideate-a-ja", numbered("あさ", "いち", "うみ")),
                ("ideate-b-ja", numbered("かぜ", "きし", "くも")),
                ("ideate-b-en", numbered("xx", "yy", "zz")),
            ],
        )
        result = gate_audit(store, cases, PlanarEmbedder())
        target = next(c for c in result.cases if c.case_id == "ideate-a-ja")
        assert target.control_cases == ("ideate-b-ja",)

    def test_分離できていれば見逃しは_0(self, tmp_path) -> None:
        embedder = PlanarEmbedder()
        embedder.ANGLES = {
            "お題A": 0.0,
            "あさ": 0.05,
            "いち": 0.05,
            "うみ": 0.05,
            "えき": 0.05,
            "かぜ": 1.5,
            "きし": 1.5,
            "くも": 1.5,
            "けや": 1.5,
        }
        cases = [
            ideate_case("ideate-a-ja", "a", "ja", "お題A"),
            ideate_case("ideate-b-ja", "b", "ja", "お題B"),
        ]
        store = build_run(
            tmp_path,
            [
                ("ideate-a-ja", numbered("あさ", "いち", "うみ", "えき")),
                ("ideate-b-ja", numbered("かぜ", "きし", "くも", "けや")),
            ],
        )
        result = gate_audit(store, cases, embedder)
        target = next(c for c in result.cases if c.case_id == "ideate-a-ja")
        assert target.measurable
        assert target.auc == 1.0
        assert target.control_leak == 0.0

    def test_分離できていなければ見逃しが出る(self, tmp_path) -> None:
        """**正例と対照が同じ角度なら、どこに閾値を置いても分けられない。**"""
        embedder = PlanarEmbedder()
        embedder.ANGLES = dict.fromkeys(
            ["あさ", "いち", "うみ", "えき", "かぜ", "きし", "くも", "けや"], 0.5
        )
        embedder.ANGLES["お題A"] = 0.0
        cases = [
            ideate_case("ideate-a-ja", "a", "ja", "お題A"),
            ideate_case("ideate-b-ja", "b", "ja", "お題B"),
        ]
        store = build_run(
            tmp_path,
            [
                ("ideate-a-ja", numbered("あさ", "いち", "うみ", "えき")),
                ("ideate-b-ja", numbered("かぜ", "きし", "くも", "けや")),
            ],
        )
        result = gate_audit(store, cases, embedder)
        target = next(c for c in result.cases if c.case_id == "ideate-a-ja")
        assert target.auc == 0.5
        assert target.control_leak == 1.0

    def test_生成が無いケースは理由を分けて出す(self, tmp_path) -> None:
        cases = [
            ideate_case("ideate-a-ja", "a", "ja", "お題A"),
            ideate_case("ideate-b-ja", "b", "ja", "お題B"),
        ]
        store = build_run(tmp_path, [("ideate-b-ja", numbered("かぜ", "きし", "くも"))])
        result = gate_audit(store, cases, PlanarEmbedder())
        target = next(c for c in result.cases if c.case_id == "ideate-a-ja")
        assert target.why_not == "この run に採点済みの生成が無い"

    def test_ideate_以外の次元は対象外(self, tmp_path) -> None:
        other = parse_case(
            {
                "id": "reason-x-ja",
                "dim": "reason",
                "task": "x",
                "lang": "ja",
                "prompt": "p",
                "failure": {"format": "none"},
                "score": {"kind": "numeric", "expect": 1},
            },
            Path("r.yaml"),
        )
        store = build_run(tmp_path, [("reason-x-ja", "1")])
        assert gate_audit(store, [other], PlanarEmbedder()).cases == []
