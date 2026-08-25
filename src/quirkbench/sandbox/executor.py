"""生成コードを隔離して実行する（FLB-QB-001 §10.4・§12.5・§12.6）。

**この体系で唯一の実質的なセキュリティ境界。** 採点器はここを直接呼ばず、
``executor`` を必須引数で受け取る（§12.10）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import profile as _profile

#: 実行器の挙動を変えたら上げる。実際のキャッシュ無効化は ``sandbox_pkg_sha256``
#: （パッケージ全体のハッシュ）で起きるので、これは人間が版を追うための番号。
EXECUTOR_VERSION = 1

#: 打ち切り由来の returncode。**どちらも ``exec_timeout`` に落とす**（§12.5）。
#: どちらが先に来るかは負荷で変わるので、分けると同じコードが実行のたびに違う型になる。
_KILL_RETURNCODES = frozenset({-9, -24})

#: workdir 走査の上限。子のタイムアウトの外（親側）なので、無いと事故で親が止まる。
_SCAN_MAX_ENTRIES = 10_000
_SCAN_MAX_SECONDS = 2.0

_CHILD = Path(__file__).with_name("_child.py")


@dataclass(frozen=True)
class ExecOutcome:
    """実行 1 回の結果。``verdict`` は pass / fail / exec_error / exec_timeout / infra。

    ``infra`` は**基盤失敗**で母数外（§12.6）。ほかは母数内。
    """

    verdict: str
    detail: str = ""
    returncode: int | None = None
    marker_seen: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    workdir_entries: int = 0
    workdir_bytes: int = 0
    workdir_scan: str = "complete"
    sub: dict[str, object] = field(default_factory=dict)


def _scan_workdir(root: Path) -> tuple[int, int, str]:
    """削除する前にファイル数と総サイズを測る。

    ``RLIMIT_FSIZE`` は 1 ファイルあたりなので、**小さいファイルを 10 万個作るコードは
    素通りする**。走査自体が新しい非停止経路にならないよう上限を掛ける。
    """
    import time

    deadline = time.monotonic() + _SCAN_MAX_SECONDS
    entries = 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            entries += 1
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
            if entries >= _SCAN_MAX_ENTRIES or time.monotonic() > deadline:
                return entries, total, "truncated"
    return entries, total, "complete"


def _classify(returncode: int, lines: list[dict[str, object]]) -> tuple[str, str, bool]:
    """**評価順序が判定を変える**（§12.5）。

    打ち切り由来の returncode を**マーカーの有無より先に**見る。逆にすると、
    ``SIGKILL`` でマーカーを取りこぼした無限ループが「マーカー無し = 基盤失敗 = 母数外」に
    落ち、**いちばん出やすい癖が母数から静かに消える**。
    """
    if returncode in _KILL_RETURNCODES:
        return (
            "exec_timeout",
            f"returncode={returncode}",
            any(line.get("marker") == "started" for line in lines),
        )

    marker_seen = any(line.get("marker") == "started" for line in lines)
    verdicts = [line for line in lines if "verdict" in line]
    if verdicts:
        last = verdicts[-1]
        kind = str(last.get("verdict"))
        detail = str(last.get("detail", ""))
        if kind == "error":
            return "exec_error", detail, marker_seen
        return kind, detail, marker_seen

    if marker_seen:
        # 子は起動したのに結果が無い。sys.exit() / os._exit() / open の差し替え。
        # **モデルの癖として母数に入れる**（§12.6）。
        return "exec_error", "開始マーカーはあるが結果が無い", True
    return "infra", "子が起動していない", False


class SandboxExecutor:
    """``sandbox-exec`` で隔離して実行する。darwin 専用。"""

    name = "sandbox"

    def __init__(self, *, use_sandbox: bool = True) -> None:
        #: ``False`` は ``--unsafe-no-sandbox`` 用。**結果は別ファイルに書く**（§12.9）。
        self.use_sandbox = use_sandbox

    @property
    def sandbox_applied(self) -> bool:
        return self.use_sandbox

    def run(
        self,
        *,
        payload: str,
        check_source: str,
        entry_point: str,
        timeout_seconds: int,
        check_hash: str = "",
        canary: str | None = None,
        canary_args: tuple[str, ...] = (),
        stdin: int = subprocess.DEVNULL,
        minimal_env: bool = True,
        extra_env: dict[str, str] | None = None,
    ) -> ExecOutcome:
        """1 回実行する。``canary`` を渡すと ``_child.py`` ではなくカナリアを走らせる。

        ``check_hash`` は受け取るが**使わない**。``Executor`` プロトコルを 1 つに保つため
        （§12.10）。キャッシュ層（``scoring._CachedExecutor``）だけがこれを読む。

        ``stdin`` と ``minimal_env`` を既定から変えられるのは、**カナリアの対照実験のため**
        （§10.4）。通常の採点経路では既定のまま使う。``stdin`` を fd で受けるのは、
        カナリア 8 の反証に**実際にブロックする源**（誰も書かず閉じないパイプ）が要るから。
        ``stdin=None`` にして親のものを継承させるだけでは、親の stdin が既に
        EOF なら**機構を外したのに 0 バイトが返り、判別しない**（実測で確認）。
        """
        del check_hash
        # workdir は mkdtemp。決定的パスにすると削除失敗時に次の実行が古い結果を読む。
        with tempfile.TemporaryDirectory(prefix="qb-exec-") as raw:
            workdir = Path(os.path.realpath(raw))
            return self._run_in(
                workdir,
                payload=payload,
                check_source=check_source,
                entry_point=entry_point,
                timeout_seconds=timeout_seconds,
                canary=canary,
                canary_args=canary_args,
                stdin=stdin,
                minimal_env=minimal_env,
                extra_env=extra_env,
            )

    def _run_in(
        self,
        workdir: Path,
        *,
        payload: str,
        check_source: str,
        entry_point: str,
        timeout_seconds: int,
        canary: str | None,
        canary_args: tuple[str, ...],
        stdin: int,
        minimal_env: bool,
        extra_env: dict[str, str] | None,
    ) -> ExecOutcome:
        if canary is None:
            (workdir / "solution.py").write_text(payload, encoding="utf-8")
            (workdir / "check_module.py").write_text(check_source, encoding="utf-8")
        prof = workdir / "profile.sb"
        prof.write_text(_profile.render(str(workdir)), encoding="utf-8")

        # 結果チャネルは**親が作る一時ファイル**。pipe だと 64KB でデッドロックし、
        # wall-clock で殺されて exec_timeout に誤分類される（§12.6）。
        result_fd, result_path = tempfile.mkstemp(prefix="qb-result-")
        try:
            return self._spawn(
                workdir,
                prof,
                result_fd,
                result_path,
                entry_point=entry_point,
                timeout_seconds=timeout_seconds,
                canary=canary,
                canary_args=canary_args,
                stdin=stdin,
                minimal_env=minimal_env,
                extra_env=extra_env,
            )
        finally:
            os.close(result_fd)
            try:
                os.unlink(result_path)
            except OSError:
                pass

    def _spawn(
        self,
        workdir: Path,
        prof: Path,
        result_fd: int,
        result_path: str,
        *,
        entry_point: str,
        timeout_seconds: int,
        canary: str | None,
        canary_args: tuple[str, ...],
        stdin: int,
        minimal_env: bool,
        extra_env: dict[str, str] | None,
    ) -> ExecOutcome:
        py = os.path.realpath(sys.executable)
        if canary is None:
            script = [str(_CHILD), str(workdir), str(timeout_seconds), entry_point]
        else:
            script = [str(Path(__file__).with_name("_canary_child.py")), canary, *canary_args]

        if minimal_env:
            env = {
                "PATH": "/usr/bin:/bin",
                "HOME": str(workdir),
                "TMPDIR": str(workdir),
                # 決定性のため（§12.9 第 3 段）
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        else:
            env = dict(os.environ)
        # pass_fds は fd 番号を 3 に固定しない。番号を env で渡す（実測で判明・§10.4）。
        env["QB_RESULT_FD"] = str(result_fd)
        if extra_env:
            env.update(extra_env)

        prefix = ["/usr/bin/sandbox-exec", "-f", str(prof)] if self.use_sandbox else []
        proc = subprocess.Popen(
            [*prefix, py, *script],
            pass_fds=(result_fd,),
            env=env,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            cwd=str(workdir),
        )
        try:
            out, err = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), 9)
            out, err = proc.communicate()

        lines = _read_result(result_path)
        if canary is not None:
            # **カナリアに verdict 分類を適用しない。** 分類は解答実行のための語彙で、
            # カナリアの観測（errno）は `sub` で受け取る。混ぜると `infra` に見える。
            verdict, detail = "canary", canary
            marker = any(line.get("marker") == "started" for line in lines)
        else:
            verdict, detail, marker = _classify(proc.returncode, lines)
        entries, total, scan = _scan_workdir(workdir)
        return ExecOutcome(
            verdict=verdict,
            detail=detail,
            returncode=proc.returncode,
            marker_seen=marker,
            stdout_tail=(out or b"").decode("utf-8", "replace")[-2000:],
            stderr_tail=(err or b"").decode("utf-8", "replace")[-2000:],
            workdir_entries=entries,
            workdir_bytes=total,
            workdir_scan=scan,
            sub=lines[-1] if lines else {},
        )


def _read_result(path: str) -> list[dict[str, object]]:
    """壊れた行は捨てる（§10.3 の規約を延長）。"""
    rows: list[dict[str, object]] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return rows
    return rows
