"""カナリア 1 本を sandbox 内で実行し、観測を継承 fd に書く（FLB-QB-001 §12.8）。

**失敗したことではなく、失敗の理由（errno）を返す。** 親が ``EPERM`` / ``EACCES`` を
名指しで確かめられるようにするため。
"""

from __future__ import annotations

import errno
import json
import os
import socket
import subprocess
import sys
import time

_RESULT_FD = int(os.environ.pop("QB_RESULT_FD"))


def _emit(payload: dict[str, object]) -> None:
    os.write(_RESULT_FD, (json.dumps(payload, ensure_ascii=False) + "\n").encode())


def _classify(action) -> dict[str, object]:  # type: ignore[no-untyped-def]
    try:
        action()
    except OSError as exc:
        return {
            "outcome": "oserror",
            "errno": exc.errno,
            "name": errno.errorcode.get(exc.errno or 0, "?"),
        }
    except Exception as exc:  # 生成コードではなくカナリアなので型で分ける
        return {"outcome": "exception", "type": type(exc).__name__, "detail": str(exc)[:120]}
    return {"outcome": "succeeded", "errno": None}


def _run(name: str, args: list[str]) -> dict[str, object]:
    if name == "home_write":
        return _classify(lambda: open(os.path.join(args[0], "probe"), "w").write("x"))
    if name == "sibling_write":
        return _classify(lambda: open(args[0], "w").write("x"))
    if name == "home_read":
        seen: dict[str, str] = {}

        def _read() -> None:
            with open(args[0], encoding="utf-8") as handle:
                seen["value"] = handle.read()

        out = _classify(_read)
        out["read_back"] = seen.get("value")
        return out
    if name == "connect_tcp":
        return _classify(
            lambda: socket.create_connection(("127.0.0.1", int(args[0])), timeout=3).close()
        )
    if name == "connect_unix":

        def _connect() -> None:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect(args[0])
            sock.close()

        return _classify(_connect)
    if name == "exec_binary":
        return _classify(lambda: subprocess.run([args[0]], capture_output=True, timeout=5))
    if name == "env_leak":
        return {"outcome": "read", "secret_visible": "QB_CANARY_SECRET" in os.environ}
    if name == "stdin_eof":
        started = time.monotonic()
        data = sys.stdin.read()
        elapsed = round(time.monotonic() - started, 3)
        return {"outcome": "read", "bytes": len(data), "seconds": elapsed}
    if name == "spin":
        _emit({"canary": name, "marker": "started"})
        while True:
            pass
    return {"outcome": "unknown_canary"}


def main() -> None:
    name = sys.argv[1]
    result = _run(name, sys.argv[2:])
    result["canary"] = name
    _emit(result)


main()
