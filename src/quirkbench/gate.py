"""実行器を選ぶゲート（FLB-QB-001 §12.8・§12.10）。

**ゲートは実行器を選ぶ層に置く。** 採点器側に置くと、ubuntu の `checks` で採点ロジックを
回すためにテスト用の迂回が要り、**その迂回が §12.10 で禁じたスイッチそのものになる**。
テストは採点器を直接呼んで stub を注入する。

規約:

- カナリアが**実行できないこと**と**失敗すること**を同じ扱い（拒否）にする。
  「`sandbox-exec` が無いから skip」が最も起きやすい実装ミスで、
  その結果は**境界ゼロで LLM 生成コードを実行すること**になる
- ``sys.platform != "darwin"`` なら `code-gen` を明示的に拒否する
- **``--unsafe-no-sandbox`` は darwin でのみ受け付ける。** 非 darwin では黙って無視せず
  エラーで止める。この 2 つを同じフラグで実装すると、**ubuntu 上でフラグ 1 つ・
  境界ゼロで LLM 生成コードを実行できる**（文書化された迂回路になる）
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from typing import Any

from .sandbox import Fixtures, SandboxExecutor, boundary_verdict, render

#: 陰性カナリアの一覧。``kind`` は判定の仕方で、``boundary`` は errno を名指しで見る。
NEGATIVE_CANARIES = (
    ("home_write", "boundary"),
    ("sibling_write", "boundary"),
    ("home_read", "boundary"),
    ("connect_tcp", "boundary"),
    ("connect_unix", "boundary"),
    ("exec_binary", "boundary"),
    ("spin", "timeout"),
    ("env_leak", "env"),
    ("stdin_eof", "stdin"),
)


class SandboxUnavailable(RuntimeError):
    """境界を張れないので `code-gen` を走らせない。"""


@dataclass(frozen=True)
class GateReport:
    passed: bool
    results: tuple[tuple[str, bool, str], ...]

    @property
    def verdict(self) -> str:
        return "pass" if self.passed else "fail"

    def failures(self) -> list[tuple[str, bool, str]]:
        return [row for row in self.results if not row[1]]


def profile_sha256(workdir: str = "/tmp/qb-probe") -> str:
    """プロファイル本文のハッシュ。指紋に入る（§12.9）。"""
    return hashlib.sha256(render(workdir).encode()).hexdigest()


def run_canaries(executor: SandboxExecutor) -> GateReport:
    """陰性カナリア 9 本を回す（`spin` / `env_leak` / `stdin_eof` を含む）。

    **判定は 3 値。** 境界系は ``EPERM`` / ``EACCES`` を観測したときだけ合格で、
    ``ENOENT`` / ``ECONNREFUSED`` は**空振りとして不合格**にする。
    """
    fixtures = Fixtures.create()
    os.environ.setdefault("QB_CANARY_SECRET", "quirkbench-canary")
    args = {
        "home_write": (str(fixtures.home_dir),),
        "sibling_write": (str(fixtures.sibling_file),),
        "home_read": (str(fixtures.home_token),),
        "connect_tcp": (str(fixtures.tcp_port),),
        "connect_unix": (fixtures.unix_path,),
        "exec_binary": ("/usr/bin/true",),
    }
    rows: list[tuple[str, bool, str]] = []
    try:
        for name, kind in NEGATIVE_CANARIES:
            timeout = 3 if name == "spin" else 10
            outcome = executor.run(
                payload="",
                check_source="",
                entry_point="",
                timeout_seconds=timeout,
                canary=name,
                canary_args=args.get(name, ()),
            )
            rows.append(_verdict(name, kind, outcome))
    finally:
        fixtures.close()
    return GateReport(all(row[1] for row in rows), tuple(rows))


def _verdict(name: str, kind: str, outcome: object) -> tuple[str, bool, str]:
    sub = getattr(outcome, "sub", {}) or {}
    if kind == "boundary":
        result = boundary_verdict(name, sub)
        return (name, result.passed, result.detail)
    if kind == "timeout":
        rc = getattr(outcome, "returncode", None)
        ok = rc in (-9, -24) and bool(getattr(outcome, "marker_seen", False))
        return (name, ok, f"returncode={rc} marker={getattr(outcome, 'marker_seen', None)}")
    if kind == "env":
        visible = sub.get("secret_visible")
        return (name, visible is False, f"secret_visible={visible}")
    if kind == "stdin":
        ok = sub.get("outcome") == "read" and sub.get("bytes") == 0
        return (name, ok, f"bytes={sub.get('bytes')} seconds={sub.get('seconds')}")
    return (name, False, f"未知のカナリア種別: {kind}")


def verify_positive_controls(executor: SandboxExecutor, cases: list[Any]) -> list[tuple[str, str]]:
    """参照解が自分のテストを通ることを、**採点するケースの数だけ**確かめる（§12.8）。

    陰性カナリアだけでは「プロファイルが過剰に厳しくなる」方向の壊れ方に気づけない —
    全部が期待どおり失敗して合格を出し、**全モデルが 0 点になったものが
    「このモデル群はコードが書けない」というプロファイルとして出る**。

    **exec キャッシュを引かない・書かない。** 参照解のペイロードはケースごとに不変なので、
    除外しないと 2 回目以降は一度も実行されない（§12.9）。
    ``executor`` に生のものを渡すのはそのため。

    返すのは**落ちたケース**の ``(case_id, 説明)``。空なら全部通っている。
    """
    failed: list[tuple[str, str]] = []
    for case in cases:
        spec = case.score
        if spec.get("kind") != "pytest":
            continue
        reference = str(spec.get("reference", ""))
        if not reference.strip():
            failed.append((case.id, "reference が空。陽性対照が成立しない"))
            continue
        outcome = executor.run(
            payload=reference,
            check_source=str(spec.get("test", "")),
            entry_point=str(spec.get("entry_point", "")),
            timeout_seconds=int(spec.get("timeout_seconds", 30)),
        )
        if outcome.verdict != "pass":
            detail = (outcome.detail or "").strip().splitlines()[-1:] or [""]
            failed.append(
                (case.id, f"参照解が自分のテストを通らない: {outcome.verdict} {detail[0]}")
            )
    return failed


def select_executor(*, unsafe_no_sandbox: bool = False) -> SandboxExecutor:
    """`code-gen` 用の実行器を返す。**張れないなら返さない。**"""
    if sys.platform != "darwin":
        # 非 darwin では unsafe フラグを受け付けない。黙って無視もしない。
        if unsafe_no_sandbox:
            raise SandboxUnavailable(
                "--unsafe-no-sandbox は darwin でのみ使える。"
                "この環境には境界が無く、フラグ 1 つで生成コードが素で走ることになる"
            )
        raise SandboxUnavailable(
            f"code-gen は macOS 専用（sandbox-exec が要る）。この環境は {sys.platform}"
        )
    if unsafe_no_sandbox:
        return SandboxExecutor(use_sandbox=False)

    executor = SandboxExecutor(use_sandbox=True)
    report = run_canaries(executor)
    if not report.passed:
        detail = "; ".join(f"{n}: {d}" for n, _ok, d in report.failures())
        raise SandboxUnavailable(f"カナリアが期待どおりでない: {detail}")
    return executor
