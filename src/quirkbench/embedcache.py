"""埋め込みのキャッシュ（FLB-QB-001 §13.5）。

**`execcache.py` と役割が逆であることが、この短さの理由である。**

| | 実行キャッシュ | 埋め込みキャッシュ |
|---|---|---|
| 入力から値が一意に決まるか | 決まらない（時間依存） | **決まる**（§13.4 で実測） |
| 無いと何が壊れるか | **冪等性そのもの** | 何も壊れない。**時間だけ掛かる** |
| 確認プロトコル | 要る | **要らない** |

だから ``confirm`` に相当するものが無い。**手を抜いたのではなく、
同じ入力から違う値が出る余地が無いので「2 回目を採る」規則が意味を持たない。**

非終端の扱いも要らない。埋め込みの失敗は ``OllamaError`` という**例外**であって
値ではないので、失敗したときは単に書かずに伝播する。``execcache`` が
``_NOT_TERMINAL`` で守っていた「失敗をキャッシュして二度と計算されない」は、
ここでは構造上起こらない。
"""

from __future__ import annotations

from . import keys as _keys
from .embed import Embedder
from .store import RunStore


class EmbedCountMismatch(RuntimeError):
    """要求した本数と返ってきたベクトルの本数が違う。

    **黙って zip で切り詰めない。** 切り詰めると、案とベクトルの対応が
    1 つずれたまま多様性が計算され、**それらしい数字が出てしまう**。
    """


class CachedEmbedder:
    """キャッシュを引く埋め込み器。**採点器から見ると普通の埋め込み器**。

    ``_CachedExecutor`` が `scoring.py` にある（1 回の実行を包むだけ）のに対し、
    こちらを独立したモジュールに置くのは、**未ヒットだけをまとめて 1 回の
    バッチにする**という固有の処理を持つから。実測でバッチ構成が値を変えない
    ことを確かめてある（§13.4）ので、この束ね直しは値に影響しない。
    """

    def __init__(self, inner: Embedder, store: RunStore) -> None:
        self._inner = inner
        self._store = store
        self._fingerprint = inner.fingerprint
        self._cache = store.embeddings()
        #: 実際に計算した本数と、キャッシュで済んだ本数。
        #: S4 の完了条件（2 回目は計算 0 回）を機械で確かめるために持つ
        self.runs = 0
        self.hits = 0

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed(self, texts: list[str]) -> list[list[float]]:
        wanted = [
            _keys.embed_key(text_hash=_keys.text_sha256(text), fingerprint=self._fingerprint)
            for text in texts
        ]
        # 未ヒットを**重複排除してから**1 回で計算する。同じ本文が 2 度出る
        # （同じ案を繰り返すモデル）のは癖として普通に起きるので、
        # 排除しないと同じ本文を 2 回計算する
        missing: dict[str, str] = {}
        for key, text in zip(wanted, texts, strict=True):
            if key in self._cache:
                self.hits += 1
            elif key not in missing:
                missing[key] = text

        if missing:
            keys_missing = list(missing)
            vectors = self._inner.embed([missing[key] for key in keys_missing])
            if len(vectors) != len(keys_missing):
                raise EmbedCountMismatch(
                    f"{len(keys_missing)} 本を要求したが {len(vectors)} 本返ってきた"
                )
            for key, vector in zip(keys_missing, vectors, strict=True):
                self._store.append_embedding(
                    {
                        "embed_key": key,
                        "text_sha256": _keys.text_sha256(missing[key]),
                        "model": getattr(self._inner, "model", ""),
                        "dims": len(vector),
                        "vector": vector,
                    }
                )
                self._cache[key] = vector
                self.runs += 1

        return [self._cache[key] for key in wanted]
