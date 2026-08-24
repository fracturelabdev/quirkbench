"""resume の分岐そのものを守る。手動実行だけに頼らない。"""

from pathlib import Path

import pytest

from quirkbench.cases import parse_case
from quirkbench.ollama import ModelInfo
from quirkbench.runner import DigestDrift, run
from quirkbench.store import RunStore

CASE = parse_case(
    {"id": "a", "dim": "reason", "lang": "ja", "prompt": "p", "score": {"kind": "exact"}},
    Path("a.yaml"),
)


class FakeOllama:
    """Ollama の代わり。生成回数を数えるためだけに存在する。"""

    def __init__(self, digest="d1"):
        self.digest = digest
        self.generated = 0

    def version(self):
        return "0.0.0-fake"

    def model_info(self, model):
        return ModelInfo(model, self.digest, "1.5B", "Q4", "qwen2", 32768)

    def generate(self, model, prompt, options):
        self.generated += 1
        return {
            "response": f"out-{options['seed']}",
            "done_reason": "stop",
            "load_duration": 1,
            "prompt_eval_count": 2,
            "prompt_eval_duration": 3,
            "eval_count": 4,
            "eval_duration": 5,
            "total_duration": 6,
        }


def _run(tmp_path, client, **kwargs):
    with RunStore(tmp_path, "r") as store:
        return run(
            store=store,
            client=client,
            cases=[CASE],
            models=["m"],
            report=lambda _msg: None,
            **kwargs,
        )


def test_first_run_generates_everything(tmp_path: Path):
    client = FakeOllama()
    progress = _run(tmp_path, client, repeats=3)
    assert (progress.generated, progress.skipped, client.generated) == (3, 0, 3)


def test_resume_skips_completed_work(tmp_path: Path):
    """ここが壊れると、再開のたびに数時間分を回し直す。"""
    _run(tmp_path, FakeOllama(), repeats=3)
    client = FakeOllama()
    progress = _run(tmp_path, client, repeats=3)
    assert (progress.generated, progress.skipped, client.generated) == (0, 3, 0)


def test_resume_generates_only_the_new_repeats(tmp_path: Path):
    _run(tmp_path, FakeOllama(), repeats=3)
    client = FakeOllama()
    progress = _run(tmp_path, client, repeats=5)
    assert (progress.generated, progress.skipped, client.generated) == (2, 3, 2)


def test_failed_generation_is_retried_next_time(tmp_path: Path):
    """失敗を完了扱いにすると永久に再試行されなくなる。"""
    from quirkbench.ollama import OllamaError

    class Failing(FakeOllama):
        def generate(self, model, prompt, options):
            raise OllamaError("boom")

    first = _run(tmp_path, Failing(), repeats=2)
    assert (first.generated, first.failed) == (0, 2)
    client = FakeOllama()
    second = _run(tmp_path, client, repeats=2)
    assert (second.generated, client.generated) == (2, 2)


def test_digest_drift_aborts_by_default(tmp_path: Path):
    """黙って回し直すのも、黙って混ぜるのも駄目。"""
    _run(tmp_path, FakeOllama(digest="d1"), repeats=1)
    with pytest.raises(DigestDrift, match="allow-digest-drift"):
        _run(tmp_path, FakeOllama(digest="d2"), repeats=1)


def test_digest_drift_can_be_allowed_explicitly(tmp_path: Path):
    _run(tmp_path, FakeOllama(digest="d1"), repeats=1)
    progress = _run(tmp_path, FakeOllama(digest="d2"), repeats=1, allow_digest_drift=True)
    assert progress.generated == 1


def test_recorded_row_carries_reproduction_fields(tmp_path: Path):
    """後から数字を引用できるだけの情報が各行に残っていること。"""
    _run(tmp_path, FakeOllama(), repeats=1)
    rows, _ = RunStore(tmp_path, "r").generations()
    required = {
        "model_digest",
        "ollama_version",
        "num_ctx",
        "max_context",
        "options_hash",
        "seed",
        "quantization_level",
        "done_reason",
        "load_duration_ns",
        "eval_count",
    }
    assert required <= set(rows[0])


def test_prompt_body_is_not_duplicated_into_generations(tmp_path: Path):
    """longctx で 100MB 級に膨らむのを避けるため、本文は prompts.jsonl にだけ置く。"""
    _run(tmp_path, FakeOllama(), repeats=2)
    rows, _ = RunStore(tmp_path, "r").generations()
    assert "prompt" not in rows[0]
    assert rows[0]["prompt_hash"] == CASE.prompt_hash
