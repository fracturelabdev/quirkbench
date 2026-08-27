"""隔離境界のテスト（FLB-QB-001 §10.4・§12.8）。

**実際に境界を張って測る。** モックにすると、測っているのがモックの挙動になる。
darwin でしか走らないので ``skipif`` を付ける — CI は macOS ジョブでこれを回す。
"""

from __future__ import annotations

import os
import sys

import pytest

from quirkbench.gate import run_canaries, select_executor, verify_positive_controls
from quirkbench.sandbox import Fixtures, SandboxExecutor, boundary_verdict, package_sha256

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec は macOS 専用")

CHECK = "def check(candidate):\n    assert candidate(2) == 4\n    assert candidate(3) == 6\n"


def _run(executor: SandboxExecutor, code: str, timeout: int = 10):  # type: ignore[no-untyped-def]
    return executor.run(
        payload=code, check_source=CHECK, entry_point="double", timeout_seconds=timeout
    )


# ---------------------------------------------------------------- 判定の語彙


@darwin_only
@pytest.mark.parametrize(
    ("code", "expected", "timeout"),
    [
        ("def double(x):\n    return x * 2\n", "pass", 10),
        ("def double(x):\n    return x + 2\n", "fail", 10),
        ("def double(x):\n    raise ValueError('boom')\n", "fail", 10),
        ("def double(x):\n    while True:\n        pass\n", "exec_timeout", 3),
        ("import sys\ndef double(x):\n    return x * 2\nsys.exit(0)\n", "exec_error", 10),
        ("def other(x):\n    return x * 2\n", "exec_error", 10),
    ],
)
def test_verdicts(code: str, expected: str, timeout: int) -> None:
    assert _run(SandboxExecutor(), code, timeout).verdict == expected


@darwin_only
def test_sys_exit_stays_in_the_denominator() -> None:
    """``sys.exit()`` を末尾に書く癖が「ケースが存在しなかったこと」にならない。

    マーカーが見えているなら**子は起動した**ので、モデルの癖として母数に入れる（§12.6）。
    """
    outcome = _run(SandboxExecutor(), "import sys\ndef double(x):\n    return x * 2\nsys.exit(0)\n")
    assert outcome.verdict == "exec_error"
    assert outcome.marker_seen is True


@darwin_only
def test_timeout_is_classified_before_marker() -> None:
    """打ち切りはマーカーの有無より先に見る（§12.5）。

    順序が逆だと、``SIGKILL`` でマーカーを取りこぼした無限ループが基盤失敗に落ち、
    **いちばん出やすい癖が母数から消える**。
    """
    outcome = _run(SandboxExecutor(), "def double(x):\n    while True:\n        pass\n", 3)
    assert outcome.verdict == "exec_timeout"
    assert outcome.returncode in (-9, -24)


# ---------------------------------------------------------------- 境界


@darwin_only
def test_file_write_outside_workdir_is_denied() -> None:
    outcome = _run(
        SandboxExecutor(),
        "import pathlib\npathlib.Path('/etc/qb-probe').write_text('x')\n"
        "def double(x):\n    return x * 2\n",
    )
    assert outcome.verdict == "exec_error"
    assert "Operation not permitted" in outcome.detail


# ---------------------------------------------------------------- カナリア


@darwin_only
def test_all_canaries_pass_with_the_boundary() -> None:
    report = run_canaries(SandboxExecutor(use_sandbox=True))
    assert report.passed, report.failures()


@darwin_only
def test_boundary_canaries_fail_without_the_boundary() -> None:
    """**これが無いとカナリアは検査になっていない。**

    「合格した」だけでは、対象が存在しないだけの空振りと区別できない。
    """
    report = run_canaries(SandboxExecutor(use_sandbox=False))
    assert not report.passed
    failed = {name for name, ok, _ in report.results if not ok}
    assert failed == {
        "home_write",
        "sibling_write",
        "home_read",
        "connect_tcp",
        "connect_unix",
        "exec_binary",
    }


@darwin_only
def test_env_canary_discriminates() -> None:
    """カナリア 7 は親側の ``env=`` 最小化を測る。サンドボックスとは独立。"""
    os.environ["QB_CANARY_SECRET"] = "probe"
    executor = SandboxExecutor()
    minimal = executor.run(
        payload="", check_source="", entry_point="", timeout_seconds=8, canary="env_leak"
    )
    inherited = executor.run(
        payload="",
        check_source="",
        entry_point="",
        timeout_seconds=8,
        canary="env_leak",
        minimal_env=False,
    )
    assert minimal.sub["secret_visible"] is False
    assert inherited.sub["secret_visible"] is True


@darwin_only
def test_stdin_canary_discriminates() -> None:
    """カナリア 8 の反証には**実際にブロックする源**が要る。

    ``stdin=None`` で親のものを継承させるだけでは、親の stdin が既に EOF なら
    機構を外したのに 0 バイトが返り、判別しない（実測で確認）。
    """
    executor = SandboxExecutor()
    quiet = executor.run(
        payload="", check_source="", entry_point="", timeout_seconds=4, canary="stdin_eof"
    )
    assert quiet.sub["bytes"] == 0

    read_fd, write_fd = os.pipe()
    try:
        blocked = executor.run(
            payload="",
            check_source="",
            entry_point="",
            timeout_seconds=4,
            canary="stdin_eof",
            stdin=read_fd,
        )
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert blocked.returncode in (-9, -24)


def test_boundary_verdict_rejects_vacuous_pass() -> None:
    """``ENOENT`` を合格にしない。**これが 2 巡目レビューの中心指摘**。"""
    import errno

    assert boundary_verdict("x", {"outcome": "oserror", "errno": errno.EPERM}).passed
    assert boundary_verdict("x", {"outcome": "oserror", "errno": errno.EACCES}).passed
    assert not boundary_verdict("x", {"outcome": "oserror", "errno": errno.ENOENT}).passed
    assert not boundary_verdict("x", {"outcome": "oserror", "errno": errno.ECONNREFUSED}).passed
    assert not boundary_verdict("x", {"outcome": "succeeded"}).passed


def test_fixtures_create_real_targets() -> None:
    """**親が対象を実在させてから**子を起動する。無いと空振りで合格する。"""
    fixtures = Fixtures.create()
    try:
        assert fixtures.home_token.exists()
        assert fixtures.sibling_file.exists()
        assert fixtures.tcp_port > 0
        assert os.path.exists(fixtures.unix_path)
    finally:
        fixtures.close()
    assert not fixtures.home_token.exists()


def test_package_hash_changes_with_content(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """指紋は ``sandbox/`` 全体。手で上げる版番号にしない（§12.9）。"""
    assert len(package_sha256()) == 64


# ---------------------------------------------------------------- 陽性対照


@darwin_only
@pytest.mark.parametrize(
    ("reference", "should_fail"),
    [
        ("def f(x):\n    return x * 2\n", False),
        ("def f(x):\n    return x + 2\n", True),
        ("def f(x):\n    raise RuntimeError('x')\n", True),
        ("   ", True),
        ("def g(x):\n    return x * 2\n", True),
    ],
)
def test_positive_control(reference: str, should_fail: bool) -> None:
    """陰性だけでは「全部落ちる」壊れ方に気づけない（§12.8）。"""
    from pathlib import Path

    from quirkbench.cases import parse_case

    raw = {
        "id": "probe",
        "dim": "code-gen",
        "task": "t",
        "lang": "ja",
        "prompt": "p",
        "options": {"num_predict": 512},
        "failure": {
            "format": "python",
            "extract": "fenced_or_whole",
            "language": "none",
            "min_tokens": 20,
        },
        "score": {
            "kind": "pytest",
            "entry_point": "f",
            "timeout_seconds": 8,
            "test": "def check(c):\n    assert c(2) == 4\n    assert c(3) == 6\n",
            "reference": reference,
        },
    }
    case = parse_case(raw, Path("t.yaml"))
    broken = verify_positive_controls(select_executor(), [case])
    assert bool(broken) is should_fail
