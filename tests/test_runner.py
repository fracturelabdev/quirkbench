"""実行計画。Ollama を叩かない部分だけを検証する。"""

from pathlib import Path

from quirkbench.cases import parse_case
from quirkbench.runner import host_info, plan_work

CASE_A = parse_case(
    {
        "id": "a",
        "dim": "reason",
        "task": "t",
        "lang": "ja",
        "prompt": "p",
        "score": {"kind": "exact"},
    },
    Path("a.yaml"),
)
CASE_B = parse_case(
    {
        "id": "b",
        "dim": "reason",
        "task": "t",
        "lang": "ja",
        "prompt": "q",
        "score": {"kind": "exact"},
    },
    Path("b.yaml"),
)


def test_plan_covers_every_combination():
    work = plan_work([CASE_A, CASE_B], ["m1", "m2"], repeats=3, base_seed=100)
    assert len(work) == 2 * 2 * 3


def test_seeds_are_derived_from_base():
    work = plan_work([CASE_A], ["m1"], repeats=3, base_seed=100)
    assert [seed for _, _, seed in work] == [100, 101, 102]


def test_work_is_grouped_by_model():
    """ケースごとにモデルを切り替えると毎回ロードが走り、load_duration も汚れる。"""
    work = plan_work([CASE_A, CASE_B], ["m1", "m2"], repeats=1, base_seed=0)
    models = [model for model, _, _ in work]
    assert models == ["m1", "m1", "m2", "m2"]


def test_host_info_has_expected_keys():
    info = host_info()
    assert {"platform", "machine", "python"} <= set(info)
