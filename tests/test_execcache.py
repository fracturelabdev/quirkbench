"""実行結果のキャッシュと確認プロトコル（FLB-QB-001 §12.9）。

**サンドボックスを起動しない。** ここで測るのは規約であって境界ではないので、
実行器は呼び出し回数を数える stub にする（境界テストは test_sandbox.py）。
"""

from __future__ import annotations

from quirkbench.execcache import confirm, is_terminal, resolve


class _Sequence:
    """あらかじめ決めた verdict を順に返す。呼ばれた回数を数える。"""

    def __init__(self, *verdicts: str) -> None:
        self._verdicts = list(verdicts)
        self.calls = 0

    def __call__(self) -> dict[str, object]:
        verdict = self._verdicts[min(self.calls, len(self._verdicts) - 1)]
        self.calls += 1
        return {"verdict": verdict}


def test_pass_terminates_on_first_attempt() -> None:
    runner = _Sequence("pass")
    row, attempts = confirm(runner)
    assert row["verdict"] == "pass"
    assert attempts == 1
    assert runner.calls == 1


def test_fail_is_not_retried() -> None:
    """通常の fail は決定的。再実行してもコストが増えるだけ。"""
    runner = _Sequence("fail")
    assert confirm(runner)[1] == 1


def test_exec_error_is_not_retried() -> None:
    """ホスト由来を §12.6 で基盤失敗に分離したので、残る exec_error は決定的。"""
    runner = _Sequence("exec_error")
    assert confirm(runner)[1] == 1


def test_timeout_is_confirmed_by_one_rerun() -> None:
    runner = _Sequence("exec_timeout", "exec_timeout")
    row, attempts = confirm(runner)
    assert row["verdict"] == "exec_timeout"
    assert attempts == 2
    assert runner.calls == 2


def test_second_attempt_wins_whatever_it_is() -> None:
    """**「2 回目が通ったら」ではない。** 非 timeout ならその verdict を採る。

    初版は「2 回目が通った → その結果を採る」としか書いておらず、2 回目が
    ``exec_error`` や通常の fail のときの終端が未定義だった。
    """
    for second in ("pass", "fail", "exec_error"):
        runner = _Sequence("exec_timeout", second)
        row, attempts = confirm(runner)
        assert row["verdict"] == second
        assert attempts == 2


def test_infra_failure_is_not_terminal() -> None:
    """基盤失敗はキャッシュに書かない。書くと第 1 段の除外が無意味になる。"""
    assert is_terminal({"verdict": "pass"})
    assert is_terminal({"verdict": "exec_timeout"})
    assert not is_terminal({"verdict": "infra"})


def test_cache_hit_does_not_execute() -> None:
    runner = _Sequence("pass")
    cache = {"k": {"verdict": "fail", "exec_key": "k"}}
    row = resolve(exec_key="k", cache=cache, run_once=runner, write=lambda r: None)
    assert row["verdict"] == "fail"
    assert runner.calls == 0


def test_first_write_wins_within_one_invocation() -> None:
    """先勝ちが成立するのは、**確認が終わるまで書かない**から（§12.9）。"""
    written: list[dict[str, object]] = []
    cache: dict[str, dict[str, object]] = {}
    first = resolve(exec_key="k", cache=cache, run_once=_Sequence("pass"), write=written.append)
    second = resolve(exec_key="k", cache=cache, run_once=_Sequence("fail"), write=written.append)
    assert first["verdict"] == "pass"
    assert second["verdict"] == "pass"
    assert len(written) == 1


def test_timeout_is_written_once_not_twice() -> None:
    """attempt 1 の timeout を書くと、先勝ちが timeout を掴み score は pass になる。

    同じコードの別 ``gen_id`` が timeout を見る — §12.9 が潰したはずの順序依存。
    """
    written: list[dict[str, object]] = []
    row = resolve(
        exec_key="k",
        cache={},
        run_once=_Sequence("exec_timeout", "pass"),
        write=written.append,
    )
    assert row["verdict"] == "pass"
    assert [r["verdict"] for r in written] == ["pass"]
    assert written[0]["attempts"] == 2


def test_infra_is_never_cached() -> None:
    written: list[dict[str, object]] = []
    cache: dict[str, dict[str, object]] = {}
    resolve(exec_key="k", cache=cache, run_once=_Sequence("infra", "infra"), write=written.append)
    assert written == []
    assert cache == {}
