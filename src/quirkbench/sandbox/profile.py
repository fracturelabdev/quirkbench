"""Seatbelt プロファイルの生成。

**パスは必ず ``realpath`` を通す**（FLB-QB-001 §10.4）。Seatbelt は解決済みパスで
照合するので、``/var/folders/…`` のような素の値を書くと write 許可が一切効かず
**全ケースが謎の失敗をする**。

``(allow default)`` からの deny 引き算で書く。``(deny default)`` から積むと、
uv 管理の Python 本体が読めずに即死する。

**``$HOME`` を deny したあとに、読めないと死ぬものを allow し直す。** SBPL は後の規則が勝つ。
実測で 2 つ見つかった:

1. **Python 本体**。この環境の Python は ``~/.local/share/uv/python/…`` にある
2. **``sandbox/`` パッケージ自身**。``_child.py`` は ``$HOME`` 配下の repo か
   site-packages に置かれるのが普通で、deny だけ書くと
   ``can't open file … [Errno 1] Operation not permitted`` で子が起動しない

2 は「実行器を workdir にコピーする」でも解ける。**採らなかった** —
コピー先は生成コードが書き込める場所なので、**信頼している実行器を
書き込み可能な領域に置くことになる**。read を 1 行戻すほうが境界として素直。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: プロファイルの内容を変えたら上げる。``profile_sb_sha256`` が指紋に入るので
#: 実際の無効化はハッシュで起きるが、人間が版を追えるように持つ。
PROFILE_VERSION = 1

_TEMPLATE = """(version 1)
(allow default)

;; --- 書き込み: workdir だけ ---
(deny file-write*)
(allow file-write* (subpath "{workdir}"))
(allow file-write* (literal "/dev/null") (literal "/dev/dtracehelper"))

;; --- 読み取り: $HOME を落とし、読めないと死ぬものだけ戻す ---
(deny file-read* (subpath "{home}"))
(allow file-read* (subpath "{py_prefix}"))
(allow file-read* (subpath "{runner_dir}"))
(allow file-read* (subpath "{workdir}"))

;; --- ネットワーク: 外向きすべて（127.0.0.1 と AF_UNIX を含む） ---
(deny network*)

;; --- 実行: Python 本体以外 ---
(deny process-exec*)
(allow process-exec (subpath "{py_prefix}"))
"""


def render(workdir: str) -> str:
    """``workdir`` 用のプロファイル本文を返す。"""
    return _TEMPLATE.format(
        workdir=os.path.realpath(workdir),
        home=os.path.realpath(os.path.expanduser("~")),
        py_prefix=os.path.realpath(sys.base_prefix),
        runner_dir=os.path.realpath(str(Path(__file__).parent)),
    )
