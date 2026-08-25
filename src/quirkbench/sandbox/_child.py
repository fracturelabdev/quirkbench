"""サンドボックス内で走る子。**このファイルは親から見て「実行される側」で、
生成コードと同じプロセスに載る。** だから判定に必要なことは
``solution.py`` を import する前にすべて済ませる（FLB-QB-001 §12.6）。

判定チャネルは**親から継承した fd**。``pass_fds`` は fd を継承させるだけで番号を
3 に固定しないので、**番号は ``QB_RESULT_FD`` で受け取り、読んだ直後に環境から消す**
（生成コードに番号を残さないため。実測で必要と分かった要件）。

書き込みは ``os.write`` の生呼び出しで行う。``os.fdopen`` で包むと、開始マーカーが
ユーザ空間のバッファに留まったまま ``SIGKILL`` で失われ、**判定の前提そのものが
判定より弱い経路に乗る**（§12.6）。
"""

from __future__ import annotations

import json
import os
import random
import resource
import sys
import traceback

#: 可変部分の上限。``RLIMIT_FSIZE`` は 1 ファイル単位なので、この fd もその網に掛かる。
MAX_DETAIL_BYTES = 64 * 1024

_RESULT_FD = int(os.environ.pop("QB_RESULT_FD"))


def _emit(payload: dict[str, object]) -> None:
    os.write(_RESULT_FD, (json.dumps(payload, ensure_ascii=False) + "\n").encode())


def _set_limits(cpu_seconds: int) -> None:
    """``preexec_fn`` を使わず自分で張る（fork 後のデッドロックを避ける・§10.4）。

    **soft/hard を両方有限にする。** soft 超過の ``SIGXCPU`` は捕捉・無視できるので、
    soft だけでは止まらない。メモリ系は macOS で設定できないので張らない（§10.4）。
    """
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _truncate(text: str) -> str:
    raw = text.encode("utf-8", "replace")
    if len(raw) <= MAX_DETAIL_BYTES:
        return text
    return raw[:MAX_DETAIL_BYTES].decode("utf-8", "replace") + "…[truncated]"


def main() -> None:
    workdir, cpu_seconds, entry_point = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    _set_limits(cpu_seconds)

    # **マーカーは import より前。** これがあれば「子は起動した」と言える。
    _emit({"marker": "started"})

    random.seed(0)
    sys.path.insert(0, workdir)
    try:
        solution = __import__("solution")
        check = __import__("check_module")
    except BaseException:  # SystemExit も拾う。sys.exit() を書く癖を握りつぶさない
        _emit({"verdict": "error", "stage": "import", "detail": _truncate(traceback.format_exc())})
        return

    candidate = getattr(solution, entry_point, None)
    if candidate is None:
        _emit({"verdict": "error", "stage": "entry_point", "detail": f"{entry_point} が無い"})
        return

    try:
        check.check(candidate)
    except AssertionError:
        _emit({"verdict": "fail", "stage": "assert", "detail": _truncate(traceback.format_exc())})
        return
    except BaseException:  # 生成コードが投げるものはすべて fail
        _emit({"verdict": "fail", "stage": "raise", "detail": _truncate(traceback.format_exc())})
        return

    _emit({"verdict": "pass"})


main()
