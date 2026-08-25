"""境界の自己診断（FLB-QB-001 §12.8）。

**陰性カナリアは「失敗すれば合格」ではない。** 対象が存在しなければ、境界が無くても
失敗する。実際 ``~/.aws/credentials`` は手元にも GitHub の ``macos-26`` runner にも無く、
**境界を全部外しても `ENOENT` で「合格」していた**。

だから 3 つを満たす形にする:

1. **親が sandbox の外で対象を実在させてから**子を起動する
2. 子は失敗を ``OSError.errno`` で受け、**``EPERM`` / ``EACCES`` を名指しで期待する**
3. **``ENOENT`` / ``ECONNREFUSED`` は不合格**（= 空振り）。成功も不合格（境界が破れている）

判定は 3 値で、**合格になるのは ``EPERM`` / ``EACCES`` を観測したときだけ**。

6〜8 は Seatbelt ではなく**親側の機構**（打ち切り・``env=`` の最小化・``stdin=DEVNULL``）を測る。
サンドボックスの ON/OFF では判別しないので、**対照実験は機構ごとに分ける**（§10.4 の実測表）。
"""

from __future__ import annotations

import errno
import os
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: カナリアの実装を変えたら上げる。``sandbox_pkg_sha256`` が指紋に入るので実際の
#: 無効化はハッシュで起きるが、人間が版を追えるように持つ。
CANARY_SUITE_VERSION = 1

#: 合格とみなす errno。これ以外は理由を問わず不合格。
_BOUNDARY_ERRNOS = frozenset({errno.EPERM, errno.EACCES})

#: 準備物の名前に入れる。``$HOME`` 配下に作るので、他と衝突しない固定 nonce を使う。
_NONCE = "quirkbench-canary"


@dataclass(frozen=True)
class CanaryResult:
    name: str
    passed: bool
    detail: str


def boundary_verdict(name: str, outcome: dict[str, object]) -> CanaryResult:
    """子が返した観測を 3 値で判定する。

    ``outcome`` は ``{"outcome": "oserror", "errno": 1}`` の形。
    """
    kind = outcome.get("outcome")
    if kind == "succeeded":
        return CanaryResult(name, False, "境界が破れている（操作が成功した）")
    if kind != "oserror":
        return CanaryResult(name, False, f"想定外の結果: {kind}")
    num = outcome.get("errno")
    if num in _BOUNDARY_ERRNOS:
        return CanaryResult(name, True, f"{errno.errorcode.get(int(num), '?')}({num})")
    label = errno.errorcode.get(int(num), "?") if isinstance(num, int) else "?"
    return CanaryResult(
        name, False, f"空振り: {label}({num})。対象が実在しないか、別の理由で失敗している"
    )


@dataclass
class Fixtures:
    """親が sandbox の外で用意する『実在する対象』。

    **これが無いとカナリアは検査になっていない。** 後始末は ``close()`` で行い、
    失敗しても run は続ける — 削除失敗を採点の失敗にすると基盤の都合が判定に混ざる。
    """

    home_dir: Path
    home_token: Path
    sibling_file: Path
    tcp_port: int
    unix_path: str
    _root: tempfile.TemporaryDirectory
    _tcp: socket.socket
    _unix: socket.socket

    @classmethod
    def create(cls) -> Fixtures:
        root = tempfile.TemporaryDirectory(prefix="qb-canary-")
        base = Path(os.path.realpath(root.name))
        sibling = base / "sibling"
        sibling.mkdir()
        sibling_file = sibling / "other-result"
        sibling_file.write_text("SIBLING-TOKEN", encoding="utf-8")

        home_dir = Path(os.path.realpath(os.path.expanduser("~"))) / f".{_NONCE}"
        home_dir.mkdir(exist_ok=True)
        home_token = home_dir / "token"
        home_token.write_text("HOME-TOKEN", encoding="utf-8")

        tcp = socket.socket()
        tcp.bind(("127.0.0.1", 0))
        tcp.listen(1)
        unix_path = str(base / "u.sock")
        unix = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        unix.bind(unix_path)
        unix.listen(1)

        return cls(
            home_dir=home_dir,
            home_token=home_token,
            sibling_file=sibling_file,
            tcp_port=tcp.getsockname()[1],
            unix_path=unix_path,
            _root=root,
            _tcp=tcp,
            _unix=unix,
        )

    def close(self) -> None:
        for sock in (self._tcp, self._unix):
            try:
                sock.close()
            except OSError:
                pass
        for path in (self.home_token, self.home_dir):
            try:
                path.unlink() if path.is_file() else path.rmdir()
            except OSError:
                pass
        try:
            self._root.cleanup()
        except OSError:
            pass
