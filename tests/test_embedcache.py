"""埋め込みキャッシュ（FLB-QB-001 §13.5）。

**確認プロトコルが無いことがここの要点。** 実行キャッシュ（`test_execcache.py`）は
「同じ入力から違う判定が出る」ことを前提に組んであるが、埋め込みは実測で決定的
（§13.4）なので、ここが守るのは**値の混在と再計算**だけである。
"""

from __future__ import annotations

import json

import pytest

from quirkbench import keys
from quirkbench.embedcache import CachedEmbedder, EmbedCountMismatch
from quirkbench.store import EMBEDDINGS, RunStore


class StubEmbedder:
    """呼ばれた本文を記録する埋め込み器。**同じ本文には同じ値**を返す。"""

    model = "stub-embed"

    def __init__(self, fingerprint: str = "f" * 64) -> None:
        self._fingerprint = fingerprint
        self.batches: list[list[str]] = []

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


class ShortEmbedder(StubEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return super().embed(texts)[:-1]


def test_first_call_computes_and_writes(tmp_path) -> None:
    inner = StubEmbedder()
    with RunStore(tmp_path, "r") as store:
        cached = CachedEmbedder(inner, store)
        assert cached.embed(["ab", "cde"]) == [[2.0, 1.0], [3.0, 1.0]]
        assert (cached.runs, cached.hits) == (2, 0)
    rows = [json.loads(line) for line in (tmp_path / "r" / EMBEDDINGS).read_text().splitlines()]
    assert len(rows) == 2
    assert {row["dims"] for row in rows} == {2}
    assert all(row["model"] == "stub-embed" for row in rows)


def test_second_invocation_computes_nothing(tmp_path) -> None:
    """**S4 の完了条件 3 そのもの**（§13.8）。2 回目は計算 0 本。"""
    inner = StubEmbedder()
    with RunStore(tmp_path, "r") as store:
        CachedEmbedder(inner, store).embed(["ab", "cde"])
    with RunStore(tmp_path, "r") as store:
        second = CachedEmbedder(StubEmbedder(), store)
        assert second.embed(["ab", "cde"]) == [[2.0, 1.0], [3.0, 1.0]]
        assert (second.runs, second.hits) == (0, 2)


def test_repeated_text_is_computed_once(tmp_path) -> None:
    """**同じ案を繰り返すのは癖として普通に起きる。**

    重複排除しないと同じ本文を 2 回計算する。
    """
    inner = StubEmbedder()
    with RunStore(tmp_path, "r") as store:
        cached = CachedEmbedder(inner, store)
        result = cached.embed(["同じ", "同じ", "違う"])
    assert result[0] == result[1]
    assert inner.batches == [["同じ", "違う"]]
    assert cached.runs == 2


def test_misses_are_batched_into_one_call(tmp_path) -> None:
    """未ヒットは 1 回のバッチにまとめる。バッチ構成は値を変えない（§13.4）。"""
    inner = StubEmbedder()
    with RunStore(tmp_path, "r") as store:
        cached = CachedEmbedder(inner, store)
        cached.embed(["a"])
        cached.embed(["a", "bb", "ccc"])
    assert inner.batches == [["a"], ["bb", "ccc"]]


def test_order_is_preserved(tmp_path) -> None:
    """**返す順が入力順**でないと、案とベクトルの対応が 1 つずれる。"""
    with RunStore(tmp_path, "r") as store:
        cached = CachedEmbedder(StubEmbedder(), store)
        cached.embed(["bb"])
        assert cached.embed(["ccc", "bb", "a"]) == [[3.0, 1.0], [2.0, 1.0], [1.0, 1.0]]


def test_short_response_is_rejected_not_zipped(tmp_path) -> None:
    """**黙って切り詰めない。** 切り詰めると対応が 1 つずれたまま数字が出る。"""
    with RunStore(tmp_path, "r") as store:
        cached = CachedEmbedder(ShortEmbedder(), store)
        with pytest.raises(EmbedCountMismatch, match="2 本を要求"):
            cached.embed(["a", "bb"])


def test_failure_writes_nothing(tmp_path) -> None:
    """失敗は例外であって値ではないので、書かれる経路が無い（§13.5）。

    ホストが落ちている間の失敗がキャッシュされると、直っても二度と計算されない。
    """

    class Broken(StubEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("ホストに届かない")

    with RunStore(tmp_path, "r") as store:
        cached = CachedEmbedder(Broken(), store)
        with pytest.raises(RuntimeError):
            cached.embed(["a"])
    assert not (tmp_path / "r" / EMBEDDINGS).exists()


def test_different_fingerprint_does_not_reuse(tmp_path) -> None:
    """**モデルが差し替わったら旧ベクトルを再利用しない**（§13.5）。"""
    with RunStore(tmp_path, "r") as store:
        CachedEmbedder(StubEmbedder(fingerprint="old"), store).embed(["a"])
    with RunStore(tmp_path, "r") as store:
        fresh = CachedEmbedder(StubEmbedder(fingerprint="new"), store)
        fresh.embed(["a"])
        assert fresh.runs == 1


def test_cache_is_first_wins(tmp_path) -> None:
    """後から書かれた行が先の行を上書きしない。"""
    store = RunStore(tmp_path, "r")
    store.dir.mkdir(parents=True)
    key = keys.embed_key(text_hash=keys.text_sha256("a"), fingerprint="f" * 64)
    for value in ([1.0], [9.0]):
        store.append_embedding({"embed_key": key, "vector": value})
    assert store.embeddings()[key] == [1.0]
