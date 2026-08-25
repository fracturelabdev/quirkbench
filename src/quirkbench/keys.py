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
import os
import platform
import sys
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


# ------------------------------------------------------- 実行結果のキャッシュ（S3）


def runner_fingerprint(
    *, sandbox_pkg_sha256: str, profile_sb_sha256: str, sandbox_applied: bool
) -> str:
    """実行環境の指紋（FLB-QB-001 §12.9）。

    **判定を変える入力だけを入れる。**

    - ``python_identity``: ``realpath(sys.executable)`` と ``sys.version`` の完全文字列。
      バージョン番号だけでは、uv 管理と system、arm64 と Rosetta が同じ ``3.12.x`` を
      名乗る。プロファイルは Python prefix を allow するので**どの prefix かは判定を変える**
    - ``os_build``: 将来の macOS が SBPL を黙って無視しても指紋が変わらないと、
      **隔離あり時代と隔離なし時代の判定が同一キーで混ざる**
    - ``sandbox_pkg_sha256``: ``sandbox/`` 配下全体。手で上げる版番号にしない —
      ``profile.sb`` を入れて ``_child.py`` を忘れたのと同じ形になる。
      **カナリア実装の変更もここで拾う**
    - ``sandbox_applied``: ``--unsafe-no-sandbox`` の判定を正常な run が再利用しない

    **``canary_verdict`` は入れない。** 値が動くと payload 横断で全 ``exec_key`` が
    変わる（部分無効化が無い）うえ、ゲートが「1 本でも落ちたら実行拒否」なので
    **キャッシュに書ける行の verdict は常に pass** になり、区別する情報を持たない。
    verdict は ``meta.json`` と score 行に記録する（§12.8）。

    **``reference_seconds`` も入れない。** §12.5 でタイムアウトを絶対値にしたので、
    測定値が判定に入らなくなった。
    """
    python_identity = f"{os.path.realpath(sys.executable)}|{sys.version}"
    os_build = f"{platform.system()}|{platform.release()}|{platform.machine()}"
    return _sha256(
        "|".join(
            [
                python_identity,
                os_build,
                sandbox_pkg_sha256,
                profile_sb_sha256,
                "1" if sandbox_applied else "0",
            ]
        )
    )


def exec_key(*, payload_sha256: str, check_hash: str, fingerprint: str) -> str:
    """実行結果キャッシュのキー（FLB-QB-001 §12.9）。

    ``payload_sha256`` は**抽出後のペイロード**（``Parsed.payload``）。生テキストにしない —
    抽出実装だけ直したときに、生テキストのハッシュだと古い実行結果が返る。

    ``scorer_version`` を**入れない**。scorer を直すたびに subprocess を回し直さない
    ためで、これがこのキャッシュの目的そのもの（§10.2）。
    """
    return _sha256("|".join([payload_sha256, check_hash, fingerprint]))


def payload_sha256(payload: str) -> str:
    """抽出後のペイロードのハッシュ。"""
    return _sha256(payload)


# ------------------------------------------------------- 埋め込みのキャッシュ（S4）


def text_sha256(text: str) -> str:
    """埋め込む本文のハッシュ。"""
    return _sha256(text)


def embedder_fingerprint(*, model: str, model_digest: str, ollama_version: str) -> str:
    """埋め込み器の指紋（FLB-QB-001 §13.5）。

    **実行器の指紋（``runner_fingerprint``）とは役割が違う。** あちらは
    「同じ入力から違う判定が出る」のを封じるための装置だが、埋め込みは実測で
    決定的だった（§13.4）ので、こちらが防ぐのは**値の混在**だけである。

    - ``model_digest`` … ``ollama pull`` でモデルの中身が差し替わっても
      **名前は変わらない**。digest が無いと、旧モデルのベクトルと新モデルの
      ベクトルから**同じ run の中で cos を取る**ことになる
    - ``ollama_version`` … §13.4 で「バージョンをまたいだ一致は測っていない」と
      書いた。**測っていないものを一致するものとして扱わない**。更新でキャッシュは
      全部無効になるが、無効化の代償は再計算だけで済む（実行キャッシュと違い、
      隔離環境も参照解も要らない）

    **``dims`` は入れない。** digest が同定するので冗長で、指紋に冗長な項目を
    入れると「何が値を変えるのか」の記述としての価値が落ちる。

    **``check_hash`` / ``scorer_version`` も入れない。** 埋め込みは採点手続きに
    依存しない。``topic`` や閾値を直しても各案のベクトルは同じで、ここを入れると
    閾値を 1 つ動かしただけで全ベクトルが無効になる（§10.2 と同じ論法）。
    """
    return _sha256("|".join([model, model_digest, ollama_version]))


def embed_key(*, text_hash: str, fingerprint: str) -> str:
    """埋め込みキャッシュのキー（FLB-QB-001 §13.5）。

    **バッチ構成を入れない。** 実測でバッチの有無・並び順に依存せず完全一致した
    （§13.4）ので、入れると同じ本文がバッチの組み方だけで別キーになり、
    キャッシュがほぼ効かなくなる。
    """
    return _sha256("|".join([text_hash, fingerprint]))
