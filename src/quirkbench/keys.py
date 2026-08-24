"""完了キーと各種ハッシュの導出。

resume の正しさはこのモジュールに集約する。導出が runner と store に散ると必ずずれる。

ハッシュを 3 つに分けているのは、**何が変わったら何をやり直すか**を分けるため:

- ``prompt_hash``  … 変われば再生成が要る（数時間）
- ``options_hash`` … 同上
- ``check_hash``   … 期待値・schema の修正。**再採点だけで済む**（数秒）

``check_hash`` を完了キーに入れてはいけない。期待 JSON のキーを 1 つ直しただけで
数百件の生成が飛ぶ。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, TypedDict

# seed は反復の軸なので options_hash に含めない（独立フィールドとして持つ）
_SEED_KEYS = frozenset({"seed"})


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """辞書の順序やスペースでハッシュが変わらないように正規化する。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def prompt_hash(prompt: str) -> str:
    """プロンプト本文のハッシュ。これが変われば生成をやり直す必要がある。"""
    return _sha256(prompt)


def options_hash(options: dict[str, Any]) -> str:
    """生成パラメータのハッシュ。seed は反復軸なので除く。"""
    return _sha256(canonical_json({k: v for k, v in options.items() if k not in _SEED_KEYS}))


def check_hash(score_spec: Any, failure_spec: Any) -> str:
    """採点手続きのハッシュ。**完了キーには入れない。**"""
    return _sha256(canonical_json({"score": score_spec, "failure": failure_spec}))


class KeyParts(TypedDict):
    """完了キーを構成する値。**キーに要素を足すならここに足す。**

    ``dict`` ではなく TypedDict にしているのは、要素を足したときに
    **組み立て側が型検査で落ちる**ようにするため。`store` と `runner` が
    独立に組み立てていた頃は、片方だけ古いまま静かに一致しなくなり、
    resume が全件ミスしてもテストが通った。
    """

    model: str
    model_digest: str
    case_id: str
    prompt_hash: str
    seed: int
    options_hash: str
    attempt: int


CompletionKey = tuple[str, str, str, str, int, str, int]

# キーを構成するフィールド名。JSONL の行から取り出すときの必須項目でもある
_KEY_FIELDS: tuple[str, ...] = (
    "model",
    "model_digest",
    "case_id",
    "prompt_hash",
    "seed",
    "options_hash",
    "attempt",
)


def key_parts_from_row(row: Mapping[str, Any]) -> KeyParts | None:
    """JSONL の 1 行から完了キーの構成値を取り出す。欠けていれば ``None``。

    **行からの取り出しもこのモジュールが持つ。** 呼び出し側で組み立てると、
    `KeyParts` に要素を足したときに片方だけ古いまま静かに一致しなくなる。
    ここに置けば、要素を足した時点でこの関数が型検査で落ちる。

    欠けた行に対して ``None`` 入りのキーを作らないのは、そのキーが何とも一致せず
    **黙って再生成されるだけで壊れた行が報告されない**から。呼び出し側が数える。
    """
    if any(row.get(field) is None for field in _KEY_FIELDS):
        return None
    return KeyParts(
        model=row["model"],
        model_digest=row["model_digest"],
        case_id=row["case_id"],
        prompt_hash=row["prompt_hash"],
        seed=row["seed"],
        options_hash=row["options_hash"],
        attempt=row["attempt"],
    )


def completion_key(parts: KeyParts) -> CompletionKey:
    """この組が既にあれば生成をスキップする。"""
    return (
        parts["model"],
        parts["model_digest"],
        parts["case_id"],
        parts["prompt_hash"],
        parts["seed"],
        parts["options_hash"],
        parts["attempt"],
    )


def gen_id(*, run_id: str, parts: KeyParts) -> str:
    """生成 1 件の決定的な ID。

    決定的にすることで、クラッシュ後の再実行が同じ ID を作り、重複排除が単純になる。
    ``attempt`` を含めないと、正当な再生成（エラー後のリトライ）が衝突する。

    **現状 ``attempt`` は常に 0。** 行単位のリトライはまだ実装しておらず
    （一時エラーの再試行は `ollama.py` の中で完結する）、この引数は行単位の
    再生成を入れるときのための予約。実態と違う保証を書かないためここに明記する。
    """
    key = completion_key(parts)
    return _sha256("|".join([run_id, *(str(part) for part in key)]))
