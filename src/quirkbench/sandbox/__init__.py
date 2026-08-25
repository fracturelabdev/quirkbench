"""生成コードの隔離実行。**この体系で唯一の実質的なセキュリティ境界**（FLB-QB-001 §10.4）。

公開するのは実行器とカナリアだけ。採点器はここを直接 import せず、
``executor`` を必須引数で受け取る（§12.10）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .canary import CANARY_SUITE_VERSION, CanaryResult, Fixtures, boundary_verdict
from .executor import EXECUTOR_VERSION, ExecOutcome, SandboxExecutor
from .profile import PROFILE_VERSION, render

__all__ = [
    "CANARY_SUITE_VERSION",
    "EXECUTOR_VERSION",
    "PROFILE_VERSION",
    "CanaryResult",
    "ExecOutcome",
    "Fixtures",
    "SandboxExecutor",
    "boundary_verdict",
    "package_sha256",
    "render",
]


def package_sha256() -> str:
    """``sandbox/`` 配下の全 ``.py`` をパス順に連結したハッシュ。

    **手で上げる版番号にしない。** ``profile.sb`` を指紋に入れて ``_child.py`` を
    忘れたのと同じ形になる（§12.9）。パッケージ全体を取れば、**新しいファイルを
    足しても自動で入る**。親側（この ``__init__`` や ``executor``）の変更も拾う。
    """
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
