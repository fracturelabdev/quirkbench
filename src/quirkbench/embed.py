"""埋め込みの取得とベクトル演算（FLB-QB-001 §13）。

**HTTP を行うのは `ollama.py` だけ**なので、ここは呼び出しの組み立てと
ベクトルの数値計算だけを持つ。キャッシュは `embedcache.py` にある。

判定（何を脱線と呼ぶか・何を valid と呼ぶか）はここに置かない。
`scorers/ideate.py` が持つ。ここは `textstats.py` と同じ「計量だけ」の層。
"""

from __future__ import annotations

import math
from typing import Protocol

from . import keys as _keys
from .ollama import Ollama

#: 既定の埋め込みモデル。実測（§13.4）はこのモデルに対して行った
DEFAULT_EMBED_MODEL = "bge-m3"


class Embedder(Protocol):
    """本文を埋め込むもの（``OllamaEmbedder``・``embedcache.CachedEmbedder``・テストの stub）。

    **`Executor` が `scorers/__init__.py` にあるのに対し、こちらは取得層に置く。**
    `embedcache` もこの型を実装する側なので、`scorers` に置くと
    ``scorers → embedcache → scorers`` の循環になる。

    ``**kwargs`` の逃げ道は作らない（§12.10）。作ると引数を足したときに
    実装漏れを型が捕まえられなくなる。
    """

    @property
    def fingerprint(self) -> str:
        """埋め込み器の同定。キャッシュのキーに入る（§13.5）。"""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """``texts`` と**同じ順・同じ本数**のベクトルを返す。"""
        ...


class OllamaEmbedder:
    """ollama の ``/api/embed`` を叩く埋め込み器。

    指紋は**構築時に 1 回だけ**求める。呼び出しのたびに ``/api/tags`` を
    引き直すと、run の途中でモデルが差し替わったときに**同じ run の中で
    指紋が変わり**、キャッシュが静かに二分される。差し替わりは
    `runner.py` の ``DigestDrift`` と同じく検出すべき事象であって、
    黙って吸収してよいものではない。
    """

    def __init__(self, client: Ollama, model: str = DEFAULT_EMBED_MODEL) -> None:
        self._client = client
        info = client.model_info(model)
        # **解決後の名前を使う**。``bge-m3`` と ``bge-m3:latest`` は同じモデルだが、
        # 書き方のまま指紋に入れると同じモデルが別キーになる（§13.5）
        self.model = info.name
        self.model_digest = info.digest
        self._fingerprint = _keys.embedder_fingerprint(
            model=info.name,
            model_digest=info.digest,
            ollama_version=client.version(),
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed(self.model, texts)


# ---------------------------------------------------------------- ベクトル演算


def cosine(a: list[float], b: list[float]) -> float:
    """cos 類似度。

    **単位正規化を前提にしない。** 実測では `bge-m3` の L2 ノルムは
    1.0000000193 だったが（§13.4）、それはそのモデルの性質であって
    ``/api/embed`` の約束ではない。別の埋め込みモデルに替えたときに、
    ここが黙って壊れる形にしない。
    """
    if len(a) != len(b):
        raise ValueError(f"次元が違う: {len(a)} と {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def mean_pairwise_distance(vectors: list[list[float]]) -> float:
    """全ペアの ``1 − cos`` の平均。**2 件未満は 0.0**（§13.1）。

    2 件未満を 0.0 にするのは値が無いからではなく、
    **「k 案出せ」に対して 1 案は多様性を満たしていない**から。

    **クリップはペアごとに掛ける。** 平均に掛けると、cos が負の 1 ペアが
    他のペアの値を押し下げられる。単位ベクトルの cos は理論上 −1 まで下がり、
    距離が 1 を超えると合成スコアが発散する。
    """
    if len(vectors) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for index, left in enumerate(vectors):
        for right in vectors[index + 1 :]:
            total += min(1.0, max(0.0, 1.0 - cosine(left, right)))
            pairs += 1
    return total / pairs
