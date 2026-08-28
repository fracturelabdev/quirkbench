"""コマンドライン。異常系で人が読めるメッセージを出すことまで含めて確認する。"""

import json
from pathlib import Path

import pytest
import yaml

from quirkbench import cli
from quirkbench.store import RunStore
from tests.test_runner_integration import FakeOllama

CASE = {
    "id": "a",
    "dim": "reason",
    "task": "t",
    "lang": "ja",
    "prompt": "p",
    "score": {"kind": "exact"},
}


@pytest.fixture
def workspace(tmp_path: Path):
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "a.yaml").write_text(yaml.safe_dump(CASE), encoding="utf-8")
    return tmp_path


def _argv(workspace: Path, *extra: str) -> list[str]:
    return [
        "run",
        "--models",
        "m",
        "--cases",
        str(workspace / "cases"),
        "--runs",
        str(workspace / "runs"),
        "--repeats",
        "2",
        *extra,
    ]


def test_version_flag_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


def test_run_generates_and_reports(workspace, monkeypatch, capsys):
    monkeypatch.setattr(cli, "Ollama", lambda host: FakeOllama())
    assert cli.main(_argv(workspace)) == 0
    assert "生成 2" in capsys.readouterr().out


def test_second_run_skips(workspace, monkeypatch, capsys):
    monkeypatch.setattr(cli, "Ollama", lambda host: FakeOllama())
    cli.main(_argv(workspace))
    capsys.readouterr()
    cli.main(_argv(workspace))
    assert "スキップ 2" in capsys.readouterr().out


def test_status_reports_counts(workspace, monkeypatch, capsys):
    monkeypatch.setattr(cli, "Ollama", lambda host: FakeOllama())
    cli.main(_argv(workspace))
    capsys.readouterr()
    assert cli.main(["status", "--runs", str(workspace / "runs")]) == 0
    assert "有効 2" in capsys.readouterr().out


def test_status_on_missing_run_is_an_error(tmp_path, capsys):
    assert cli.main(["status", "--runs", str(tmp_path)]) == 1
    assert "存在しない" in capsys.readouterr().out


def test_dim_filter_that_matches_nothing_is_an_error(workspace, monkeypatch, capsys):
    monkeypatch.setattr(cli, "Ollama", lambda host: FakeOllama())
    assert cli.main(_argv(workspace, "--dims", "ideate")) == 2
    assert "エラー" in capsys.readouterr().err


def test_bad_case_directory_is_an_error(tmp_path, capsys):
    assert cli.main(["run", "--models", "m", "--cases", str(tmp_path / "none")]) == 2
    assert "エラー" in capsys.readouterr().err


def test_lock_conflict_is_reported(workspace, monkeypatch, capsys):
    monkeypatch.setattr(cli, "Ollama", lambda host: FakeOllama())
    held = RunStore(workspace / "runs", "main")
    held.__enter__()
    try:
        assert cli.main(_argv(workspace)) == 2
        assert "実行中" in capsys.readouterr().err
    finally:
        held.__exit__()


def test_digest_drift_is_reported(workspace, monkeypatch, capsys):
    monkeypatch.setattr(cli, "Ollama", lambda host: FakeOllama(digest="d1"))
    cli.main(_argv(workspace))
    monkeypatch.setattr(cli, "Ollama", lambda host: FakeOllama(digest="d2"))
    capsys.readouterr()
    assert cli.main(_argv(workspace)) == 2
    assert "digest" in capsys.readouterr().err


def test_digest_drift_can_be_overridden(workspace, monkeypatch):
    monkeypatch.setattr(cli, "Ollama", lambda host: FakeOllama(digest="d1"))
    cli.main(_argv(workspace))
    monkeypatch.setattr(cli, "Ollama", lambda host: FakeOllama(digest="d2"))
    assert cli.main(_argv(workspace, "--allow-digest-drift")) == 0


# ---------------------------------------------------------------- qb score


def _write_case(root: Path) -> None:
    (root / "instruct").mkdir(parents=True)
    (root / "instruct" / "c.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "c1",
                "dim": "instruct",
                "task": "t",
                "lang": "ja",
                "prompt": "p",
                "failure": {"format": "json", "extract": "fenced_or_first_object", "min_tokens": 4},
                "score": {
                    "kind": "json_schema",
                    "schema": {
                        "type": "object",
                        "required": ["age"],
                        "properties": {"age": {"type": "integer"}},
                    },
                    "expect": {"age": 42},
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _write_generation(runs: Path, response: str) -> None:
    run_dir = runs / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "generations.jsonl").write_text(
        json.dumps(
            {
                "gen_id": "g1",
                "case_id": "c1",
                "response": response,
                "done_reason": "stop",
                "eval_count": 20,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_score_command_writes_scores(tmp_path, capsys):
    cases, runs = tmp_path / "cases", tmp_path / "runs"
    _write_case(cases)
    _write_generation(runs, '{"age": 42}')

    rc = cli.main(["score", "--run", "r1", "--cases", str(cases), "--runs", str(runs)])
    assert rc == 0
    assert "採点 1" in capsys.readouterr().out

    rows = [json.loads(line) for line in (runs / "r1" / "scores.jsonl").read_text().splitlines()]
    assert rows[0]["score"] == 1.0


def test_score_command_is_idempotent(tmp_path, capsys):
    cases, runs = tmp_path / "cases", tmp_path / "runs"
    _write_case(cases)
    _write_generation(runs, '{"age": 42}')

    cli.main(["score", "--run", "r1", "--cases", str(cases), "--runs", str(runs)])
    capsys.readouterr()
    cli.main(["score", "--run", "r1", "--cases", str(cases), "--runs", str(runs)])
    assert "採点済みスキップ 1" in capsys.readouterr().out


def test_score_command_missing_run(tmp_path, capsys):
    cases = tmp_path / "cases"
    _write_case(cases)
    rc = cli.main(
        ["score", "--run", "nope", "--cases", str(cases), "--runs", str(tmp_path / "runs")]
    )
    assert rc == 2
    assert "存在しない" in capsys.readouterr().err


# ------------------------------------------------- lint-cases / score のゲート


def test_lint_cases_passes_on_bundled_cases(capsys) -> None:  # type: ignore[no-untyped-def]
    from quirkbench.cli import main

    assert main(["lint-cases", "--cases", "cases"]) == 0
    assert "指摘 0 件" in capsys.readouterr().out


def test_lint_cases_fails_on_broken_case(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    """**ゲートは落ちなければゲートではない。**"""
    from quirkbench.cli import main

    root = tmp_path / "cases" / "code-gen"
    root.mkdir(parents=True)
    (root / "bad.yaml").write_text(
        "id: bad\ndim: code-gen\nlang: ja\ntask: bad\nprompt: p\n"
        "options: {num_predict: 512}\n"
        "failure: {format: python, extract: fenced_or_whole, language: none, min_tokens: 20}\n"
        "score:\n  kind: pytest\n  entry_point: f\n  timeout_seconds: 2\n"
        "  test: |\n    def check(c):\n        assert c(1) == 1\n"
        "  reference: |\n    def f(x):\n        return x\n",
        encoding="utf-8",
    )
    assert main(["lint-cases", "--cases", str(tmp_path / "cases")]) == 1
    assert "timeout_range" in capsys.readouterr().err


def test_score_refuses_code_gen_on_non_darwin(tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """境界の無い環境で `code-gen` を採点しない。**黙って素で走らせない。**"""
    from quirkbench import gate
    from quirkbench.cli import main
    from quirkbench.store import RunStore

    monkeypatch.setattr(gate.sys, "platform", "linux")
    with RunStore(tmp_path / "runs", "r") as store:
        store.append_generation({"gen_id": "g", "case_id": "code-gen-two-sum-ja", "response": "x"})
    code = main(
        [
            "score",
            "--run",
            "r",
            "--runs",
            str(tmp_path / "runs"),
            "--cases",
            "cases",
            "--dims",
            "code-gen",
        ]
    )
    assert code == 2
    assert "macOS 専用" in capsys.readouterr().err


# ------------------------------------------------------------- report


def _write_profile_run(runs: Path) -> None:
    """2 モデル × 2 ケース。片方は全モデル満点で識別力なしにする。"""
    run_dir = runs / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    index = 0
    for model in ["m-a", "m-b"]:
        for case_id in ["c1", "c2"]:
            # c1 だけモデル間で応答を変えて識別力を持たせる。c2 は両モデルとも
            # 正解にして「全モデル満点（ceiling）」を作る
            wrong = case_id == "c1" and model == "m-b"
            response = '{"age": 1}' if wrong else '{"age": 42}'
            index += 1
            lines.append(
                json.dumps(
                    {
                        "gen_id": f"g{index}",
                        "model": model,
                        "model_digest": f"digest-{model}",
                        "ollama_version": "0.32.13",
                        "case_id": case_id,
                        "seed": 1000,
                        "response": response,
                        "done_reason": "stop",
                        "eval_count": 20,
                        "eval_duration_ns": 100_000_000,
                        "prompt_eval_count": 5,
                        "prompt_eval_duration_ns": 10_000_000,
                        "load_duration_ns": 1_000_000,
                        "wall_seconds": 0.1,
                    },
                    ensure_ascii=False,
                )
            )
    (run_dir / "generations.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_two_cases(cases: Path) -> None:
    cases.mkdir(parents=True, exist_ok=True)
    for case_id in ["c1", "c2"]:
        (cases / f"{case_id}.yaml").write_text(
            yaml.safe_dump(
                {
                    "id": case_id,
                    "dim": "instruct",
                    "task": "t",
                    "lang": "ja",
                    "prompt": f"p-{case_id}",
                    "failure": {"format": "json"},
                    "score": {
                        "kind": "json_schema",
                        "schema": {"type": "object"},
                        "expect": {"age": 42},
                    },
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )


def test_report_writes_both_files(tmp_path, capsys):
    cases, runs, out = tmp_path / "cases", tmp_path / "runs", tmp_path / "reports"
    _write_two_cases(cases)
    _write_profile_run(runs)
    cli.main(["score", "--run", "r1", "--cases", str(cases), "--runs", str(runs)])
    capsys.readouterr()

    rc = cli.main(
        ["report", "--run", "r1", "--cases", str(cases), "--runs", str(runs), "--out", str(out)]
    )
    assert rc == 0
    assert (out / "r1" / "report.md").exists()
    assert (out / "r1" / "compare.md").exists()


def test_report_names_the_non_discriminating_case(tmp_path, capsys):
    """**識別力なしは件数だけでなく ID と理由を出す**（§14.3）。"""
    cases, runs, out = tmp_path / "cases", tmp_path / "runs", tmp_path / "reports"
    _write_two_cases(cases)
    _write_profile_run(runs)
    cli.main(["score", "--run", "r1", "--cases", str(cases), "--runs", str(runs)])
    capsys.readouterr()

    cli.main(
        ["report", "--run", "r1", "--cases", str(cases), "--runs", str(runs), "--out", str(out)]
    )
    stdout = capsys.readouterr().out
    assert "識別力なし: c2（ceiling）" in stdout


def test_report_warns_when_z_rests_on_one_case(tmp_path, capsys):
    cases, runs, out = tmp_path / "cases", tmp_path / "runs", tmp_path / "reports"
    _write_two_cases(cases)
    _write_profile_run(runs)
    cli.main(["score", "--run", "r1", "--cases", str(cases), "--runs", str(runs)])
    capsys.readouterr()

    cli.main(
        ["report", "--run", "r1", "--cases", str(cases), "--runs", str(runs), "--out", str(out)]
    )
    assert "z は要注意" in capsys.readouterr().out


def test_report_warns_about_unscored_generations(tmp_path, capsys):
    """**採点し直し忘れに気づけるようにする**（§14.2）。"""
    cases, runs, out = tmp_path / "cases", tmp_path / "runs", tmp_path / "reports"
    _write_two_cases(cases)
    _write_profile_run(runs)

    rc = cli.main(
        ["report", "--run", "r1", "--cases", str(cases), "--runs", str(runs), "--out", str(out)]
    )
    assert rc == 0
    assert "採点行が見つからない生成が 4 件" in capsys.readouterr().out


def test_report_rejects_a_run_with_mixed_conditions(tmp_path, capsys):
    """**差し引かれるのが難易度でなくなる**（§14.1）。"""
    cases, runs, out = tmp_path / "cases", tmp_path / "runs", tmp_path / "reports"
    _write_two_cases(cases)
    _write_profile_run(runs)
    with (runs / "r1" / "generations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "gen_id": "drift",
                    "model": "m-a",
                    "model_digest": "SOMETHING-ELSE",
                    "ollama_version": "0.32.13",
                    "case_id": "c1",
                    "seed": 1000,
                }
            )
            + "\n"
        )
    rc = cli.main(
        ["report", "--run", "r1", "--cases", str(cases), "--runs", str(runs), "--out", str(out)]
    )
    assert rc == 2
    assert "モデルの中身が変わっている" in capsys.readouterr().err


def test_report_on_missing_run_is_an_error(tmp_path, capsys):
    cases = tmp_path / "cases"
    _write_two_cases(cases)
    rc = cli.main(
        ["report", "--run", "nope", "--cases", str(cases), "--runs", str(tmp_path / "runs")]
    )
    assert rc == 2
    assert "存在しない" in capsys.readouterr().err


# ------------------------------------------------------ gate-audit（§19.7）


def test_gate_audit_on_missing_run_is_an_error(tmp_path, capsys):
    assert (
        cli.main(
            [
                "gate-audit",
                "--run",
                "none",
                "--runs",
                str(tmp_path),
                "--cases",
                str(tmp_path / "cases"),
            ]
        )
        == 2
    )


def test_gate_audit_without_gated_cases_is_an_error(workspace, capsys):
    """**ゲートを持つケースが無い run で 0 を返さない。**

    返すと「測って問題なかった」と読める。
    """
    runs = workspace / "runs"
    (runs / "r").mkdir(parents=True)
    assert (
        cli.main(
            [
                "gate-audit",
                "--run",
                "r",
                "--runs",
                str(runs),
                "--cases",
                str(workspace / "cases"),
            ]
        )
        == 2
    )
    assert "ゲートを持つケースが無い" in capsys.readouterr().err
