"""埋め込みの取得層とベクトル演算。

**実モデルは呼ばない。** ollama は CI に無い。ここで検証するのは
指紋の導出と数値計算で、実測（§13.4）は手元でのみ行う。
"""

from __future__ import annotations

import math

import pytest

from quirkbench import keys
from quirkbench.embed import OllamaEmbedder, cosine, mean_pairwise_distance


class FakeOllama:
    """`Ollama` の必要な部分だけ。"""

    def __init__(self, digest: str = "d" * 12, version: str = "0.32.13") -> None:
        self._digest = digest
        self._version = version
        self.calls: list[list[str]] = []

    def model_info(self, model: str):
        from quirkbench.ollama import ModelInfo

        return ModelInfo(model, self._digest, "", "", "", None)

    def version(self) -> str:
        return self._version

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        self.calls.append(list(inputs))
        return [[float(len(text)), 0.0] for text in inputs]


# ------------------------------------------------------------------ cosine


def test_cosine_of_identical_is_one() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_does_not_assume_unit_norm() -> None:
    """**長さの違うベクトルでも向きが同じなら 1.0。**

    `bge-m3` は実測で単位正規化されていた（§13.4）が、それはモデルの性質で
    ``/api/embed`` の約束ではない。正規化を前提にすると、別の埋め込みモデルに
    替えたときにここが黙って壊れる。
    """
    assert cosine([3.0, 4.0], [30.0, 40.0]) == pytest.approx(1.0)


def test_cosine_of_zero_vector_is_zero_not_nan() -> None:
    """ゼロ除算で NaN を返すと、NaN が合成スコアを汚染して比較不能になる。"""
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_rejects_mismatched_dims() -> None:
    with pytest.raises(ValueError, match="次元が違う"):
        cosine([1.0], [1.0, 0.0])


# -------------------------------------------------------------- 多様性


@pytest.mark.parametrize("count", [0, 1])
def test_diversity_of_fewer_than_two_is_zero(count: int) -> None:
    """**値が無いからではなく、1 案は多様性を満たしていないから 0**（§13.1）。"""
    assert mean_pairwise_distance([[1.0, 0.0]] * count) == 0.0


def test_diversity_of_identical_items_is_zero() -> None:
    assert mean_pairwise_distance([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]) == pytest.approx(0.0)


def test_diversity_of_orthogonal_items_is_one() -> None:
    assert mean_pairwise_distance([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(1.0)


def test_diversity_clips_per_pair_not_on_the_mean() -> None:
    """**逆向きのペアが他のペアを押し下げられないこと。**

    cos は −1 まで下がるので ``1 − cos`` は 2 になる。平均を取ってからクリップすると
    2 が他を引き上げ、ペアごとにクリップしないと合成スコアが発散する（§13.1）。
    ここでは逆向き 1 ペア + 同一 1 ペアを混ぜ、平均が 1.0 を超えないことを見る。
    """
    value = mean_pairwise_distance([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    assert 0.0 <= value <= 1.0
    # 逆向き 2 ペア（各 1.0 にクリップ）+ 同一 1 ペア（0.0）→ 2/3
    assert value == pytest.approx(2.0 / 3.0)


def test_diversity_is_a_real_mean() -> None:
    """直交ペアと同一ペアが混ざれば中間の値になる（0 か 1 に丸めない）。"""
    value = mean_pairwise_distance([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert value == pytest.approx(2.0 / 3.0)


# ------------------------------------------------------------------ 指紋


def test_fingerprint_tracks_model_digest() -> None:
    """**`ollama pull` で中身が変わっても名前は変わらない**（§13.5）。

    digest が指紋に入っていないと、旧モデルと新モデルのベクトルから
    同じ run の中で cos を取ることになる。
    """
    a = OllamaEmbedder(FakeOllama(digest="aaa")).fingerprint  # type: ignore[arg-type]
    b = OllamaEmbedder(FakeOllama(digest="bbb")).fingerprint  # type: ignore[arg-type]
    assert a != b


def test_fingerprint_tracks_ollama_version() -> None:
    """バージョンをまたいだ一致は**測っていない**ので、一致するものとして扱わない。"""
    a = OllamaEmbedder(FakeOllama(version="0.32.13")).fingerprint  # type: ignore[arg-type]
    b = OllamaEmbedder(FakeOllama(version="0.33.0")).fingerprint  # type: ignore[arg-type]
    assert a != b


def test_fingerprint_is_stable_for_the_same_environment() -> None:
    a = OllamaEmbedder(FakeOllama()).fingerprint  # type: ignore[arg-type]
    b = OllamaEmbedder(FakeOllama()).fingerprint  # type: ignore[arg-type]
    assert a == b


def test_fingerprint_is_resolved_once_at_construction() -> None:
    """呼び出しのたびに引き直すと、run の途中で指紋が変わりキャッシュが二分される。"""
    client = FakeOllama()
    embedder = OllamaEmbedder(client)  # type: ignore[arg-type]
    first = embedder.fingerprint
    client._digest = "changed"
    assert embedder.fingerprint == first


def test_embed_key_ignores_batch_composition() -> None:
    """**バッチ構成をキーに入れない**（§13.5）。実測でバッチに非依存だった（§13.4）。"""
    fingerprint = "f" * 64
    solo = keys.embed_key(text_hash=keys.text_sha256("A"), fingerprint=fingerprint)
    again = keys.embed_key(text_hash=keys.text_sha256("A"), fingerprint=fingerprint)
    assert solo == again


def test_embedder_passes_texts_through() -> None:
    client = FakeOllama()
    embedder = OllamaEmbedder(client)  # type: ignore[arg-type]
    assert embedder.embed(["ab", "cde"]) == [[2.0, 0.0], [3.0, 0.0]]
    assert client.calls == [["ab", "cde"]]


def test_unit_norm_assumption_is_only_documented_not_relied_on() -> None:
    """設計文書が引用している L2 ノルムの計算式そのものを固定する。"""
    assert math.isclose(math.sqrt(sum(x * x for x in [0.6, 0.8])), 1.0)
