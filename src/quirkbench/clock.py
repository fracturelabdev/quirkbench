"""時刻。**行に書き込む時刻の形式を 1 箇所に集める。**

`generations.jsonl` と `scores.jsonl` の ``ts`` は、レポートが同一 gen_id の
採点行を決着させるのに使う。2 箇所が独立に形式を持つと、片方だけ変えたときに
比較が黙って壊れる。
"""

from __future__ import annotations

import datetime as _dt


def now_iso() -> str:
    """UTC の秒精度 ISO8601。JSONL 行の ``ts`` はすべてこれ。"""
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
