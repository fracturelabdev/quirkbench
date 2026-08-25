"""実行結果のキャッシュと確認プロトコル（FLB-QB-001 §12.9）。

**冪等とは「入力が判定を一意に決める」ことではない。** 打ち切りは wall-clock に依存するので、
同じ入力から ``exec_timeout`` にも pass にもなりうる。それは保証できない。

**保証するのは「最初にキャッシュへ書けた値が、以降の唯一の判定になる」こと。**
終端は*確認プロトコルの完了*であって、*2 回の観測が一致したこと*ではない。

規約は 4 つ:

1. キャッシュに書くのは、確認プロトコルが終わったあとの **1 回だけ**
2. キャッシュにヒットしたら **実行しない**
3. 書き順は **cache → score**
4. **score はキャッシュの先勝ち行の関数**であって、今回の実行結果ではない

4 が無いと、キャッシュは先勝ち・score は後勝ち（``ts`` 最大）で**同じ行に逆向きに効き**、
同一ペイロードの別 ``gen_id`` が古い判定を見る。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

#: 確認プロトコルを掛ける verdict。**``exec_error`` には掛けない** —
#: ホスト由来を §12.6 で基盤失敗に分離したので、残る ``exec_error`` は
#: モデル側の決定的な振る舞いであり、再実行しても同じになる。
_RETRY_VERDICTS = frozenset({"exec_timeout", "infra"})

#: キャッシュにも score の ``done`` にも入れない verdict。
#: 基盤失敗は母数外（§12.6）で、書くと第 2 段が永久にその失敗を返し、
#: 第 1 段の除外が無意味になる。ホストが直っても二度と実行されない。
_NOT_TERMINAL = frozenset({"infra"})


class _Runner(Protocol):
    def __call__(self) -> Any: ...


def _as_row(outcome: Any) -> dict[str, Any]:
    if is_dataclass(outcome) and not isinstance(outcome, type):
        return asdict(outcome)
    if isinstance(outcome, dict):
        return dict(outcome)
    raise TypeError(f"実行結果を行にできない: {type(outcome)!r}")


def confirm(run_once: _Runner) -> tuple[dict[str, Any], int]:
    """確認プロトコル。``(終端した結果, 試行回数)`` を返す。

    - 1 回目が ``exec_timeout`` か基盤失敗なら、**同じ invocation 内でもう 1 回だけ**実行する
    - **2 回目が非 ``exec_timeout`` なら、その verdict をそのまま採る** —
      pass でも fail でも ``exec_error`` でも同じ。「2 回目が通ったら」ではない
    - 2 回とも ``exec_timeout`` なら終端

    初版は「2 回目が通った → その結果を採る」としか書いておらず、
    **2 回目が ``exec_error`` や通常の fail だったときの終端が未定義**だった。
    実装は自然に「3 回目」や「1 回目を採る」に流れる。
    """
    first = _as_row(run_once())
    if str(first.get("verdict")) not in _RETRY_VERDICTS:
        return first, 1
    second = _as_row(run_once())
    return second, 2


def is_terminal(row: dict[str, Any]) -> bool:
    """キャッシュに書いてよいか。基盤失敗は書かない。"""
    return str(row.get("verdict")) not in _NOT_TERMINAL


def resolve(
    *,
    exec_key: str,
    cache: dict[str, dict[str, Any]],
    run_once: _Runner,
    write: Any,
) -> dict[str, Any]:
    """キャッシュを引き、無ければ確認して書き、**先勝ち行を返す**。

    ``write`` は ``(row) -> None``。呼んだあと ``cache`` にも入れて、
    **同じ invocation 内の後続が同じ行を見る**ようにする（先勝ちの成立条件）。
    """
    hit = cache.get(exec_key)
    if hit is not None:
        return hit

    row, attempts = confirm(run_once)
    row = {**row, "exec_key": exec_key, "attempts": attempts}
    if not is_terminal(row):
        # 基盤失敗。キャッシュに書かず、次回また試す。
        return row
    write(row)
    cache[exec_key] = row
    return row
