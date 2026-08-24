"""永続化と resume。壊れた行の扱いが要点。"""

from pathlib import Path

import pytest

from quirkbench.store import RunLocked, RunStore, read_jsonl


def _row(**overrides):
    base = dict(
        gen_id="g",
        model="m",
        model_digest="d",
        case_id="c",
        prompt_hash="p",
        seed=1,
        options_hash="o",
        attempt=0,
    )
    return {**base, **overrides}


def test_append_and_read_back(tmp_path: Path):
    with RunStore(tmp_path, "r") as store:
        for i in range(3):
            store.append_generation(_row(gen_id=f"g{i}", seed=i))
        rows, report = store.generations()
    assert len(rows) == 3
    assert report.total == 3
    assert not report.has_warning


def test_prompts_are_deduplicated(tmp_path: Path):
    """longctx は同じ長文が何十回も繰り返されるので、素直に書くと桁違いに膨らむ。"""
    with RunStore(tmp_path, "r") as store:
        store.append_prompt("h", "本文")
        store.append_prompt("h", "本文")
        rows, _ = read_jsonl(store.dir / "prompts.jsonl")
    assert len(rows) == 1


def test_corrupt_tail_is_dropped_silently(tmp_path: Path):
    """kill された瞬間の書きかけ。キーが復元できないので単に再実行されればよい。"""
    with RunStore(tmp_path, "r") as store:
        store.append_generation(_row())
        path = store.dir / "generations.jsonl"
        path.write_text(path.read_text() + '{"gen_id": "part')
        rows, report = store.generations()
    assert len(rows) == 1
    assert report.corrupt_tail
    assert not report.has_warning


def test_corrupt_middle_is_reported(tmp_path: Path):
    """途中の破損はディスク障害か二重起動の兆候。末尾と同じ扱いにしてはいけない。"""
    with RunStore(tmp_path, "r") as store:
        store.append_generation(_row(gen_id="a"))
        store.append_generation(_row(gen_id="b"))
        path = store.dir / "generations.jsonl"
        lines = path.read_text().split("\n")
        lines[0] = "{broken"
        path.write_text("\n".join(lines))
        rows, report = store.generations()
    assert report.corrupt_middle == 1
    assert report.has_warning
    assert len(rows) == 1


def test_completed_keys_exclude_failed_rows(tmp_path: Path):
    """失敗した生成を完了扱いにすると、永久に再試行されなくなる。"""
    with RunStore(tmp_path, "r") as store:
        store.append_generation(_row(gen_id="ok", seed=1))
        store.append_generation(_row(gen_id="ng", seed=2, error="boom"))
        completed, _ = store.completed_keys()
    assert ("m", "d", "c", "p", 1, "o", 0) in completed
    assert ("m", "d", "c", "p", 2, "o", 0) not in completed


def test_second_process_cannot_acquire_lock(tmp_path: Path):
    store = RunStore(tmp_path, "r")
    store.__enter__()
    try:
        with pytest.raises(RunLocked):
            RunStore(tmp_path, "r").__enter__()
    finally:
        store.__exit__()


def test_lock_is_released_on_exit(tmp_path: Path):
    with RunStore(tmp_path, "r"):
        pass
    with RunStore(tmp_path, "r"):
        pass  # 例外が出なければよい


def test_meta_roundtrip(tmp_path: Path):
    with RunStore(tmp_path, "r") as store:
        store.write_meta({"run_id": "r", "model_digests": {"m": "d"}})
        assert store.read_meta()["model_digests"]["m"] == "d"


def test_unreliable_fs_is_flagged(tmp_path: Path):
    """flock はネットワーク/同期 FS で信頼できないので、黙って走らせない。"""
    icloud = Path.home() / "Library" / "Mobile Documents" / "runs"
    assert RunStore(icloud, "r").warn_unreliable_fs() is not None
    assert RunStore(tmp_path, "r").warn_unreliable_fs() is None


def test_completed_keys_uses_the_shared_derivation(tmp_path):
    """完了キーを手で組み直さない。

    keys.completion_key に要素が 1 つ増えたとき、手書きのタプルは静かに一致
    しなくなり、resume が全件ミスして数時間ぶんを再生成する。キー自体は
    自己整合しているのでテストも通ってしまう（Opus の指摘）。
    """
    from quirkbench import keys

    row = {
        "model": "m",
        "model_digest": "d",
        "case_id": "c",
        "prompt_hash": "p",
        "seed": 7,
        "options_hash": "o",
        "attempt": 0,
    }
    with RunStore(tmp_path, "r") as store:
        store.append_generation({"gen_id": "g", **row})
        found, _ = store.completed_keys()
    assert found == {keys.completion_key(row)}


def test_rows_missing_key_fields_are_counted_not_silently_skipped(tmp_path):
    """完了キーの項目が欠けた行を、None 入りのキーにしない。

    None 入りのキーは何とも一致しないので**黙って再生成されるだけ**になり、
    壊れた行が何度でも湧いていることに気づけない。中間の破損行と同じ扱いで数える。
    """
    with RunStore(tmp_path, "r") as store:
        store.append_generation({"gen_id": "g1", "model": "m", "case_id": "c"})  # 大半が欠落
        store.append_generation(
            {
                "gen_id": "g2",
                "model": "m",
                "model_digest": "d",
                "case_id": "c",
                "prompt_hash": "p",
                "seed": 1,
                "options_hash": "o",
                "attempt": 0,
            }
        )
        found, report = store.completed_keys()
    assert len(found) == 1
    assert report.missing_fields == 1
    assert report.has_warning
