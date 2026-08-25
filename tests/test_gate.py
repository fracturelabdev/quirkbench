"""実行器のゲート（FLB-QB-001 §12.8・§12.10）。

**プラットフォーム判定を実機に依存させない。** ここは「非 darwin でどう振る舞うか」の
検査なので、``sys.platform`` を差し替えて測る。実機の境界は test_sandbox.py。
"""

from __future__ import annotations

import pytest

from quirkbench import gate


def test_non_darwin_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux ユーザーに境界がそもそも存在しないので、明示的に拒否する。"""
    monkeypatch.setattr(gate.sys, "platform", "linux")
    with pytest.raises(gate.SandboxUnavailable, match="macOS 専用"):
        gate.select_executor()


def test_unsafe_flag_is_disabled_on_non_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    """**フラグ 1 つで境界ゼロの実行に落ちる経路を閉じる。**

    「非 darwin なら拒否」と「カナリアが落ちたら unsafe を明示すれば走る」を同じフラグで
    実装すると、ubuntu 上でフラグ 1 つ・境界ゼロで LLM 生成コードを実行できる。
    黙って無視するのではなくエラーで止める。
    """
    monkeypatch.setattr(gate.sys, "platform", "linux")
    with pytest.raises(gate.SandboxUnavailable, match="darwin でのみ"):
        gate.select_executor(unsafe_no_sandbox=True)


def test_gate_report_lists_failures() -> None:
    report = gate.GateReport(False, (("a", True, "ok"), ("b", False, "空振り")))
    assert report.verdict == "fail"
    assert [row[0] for row in report.failures()] == ["b"]


def test_profile_hash_is_stable() -> None:
    assert gate.profile_sha256("/tmp/x") == gate.profile_sha256("/tmp/x")
    assert gate.profile_sha256("/tmp/x") != gate.profile_sha256("/tmp/y")


def test_canary_list_covers_both_kinds() -> None:
    """境界系（Seatbelt）と親側の機構を、**別の判定で**測っている。"""
    kinds = {kind for _name, kind in gate.NEGATIVE_CANARIES}
    assert kinds == {"boundary", "timeout", "env", "stdin"}
    boundary = [n for n, k in gate.NEGATIVE_CANARIES if k == "boundary"]
    assert len(boundary) == 6
