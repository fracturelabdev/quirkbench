"""完了キーの導出。ここがずれると resume が壊れ、数時間分を取りこぼす。"""

from quirkbench import keys


def test_options_hash_ignores_seed():
    """seed は反復の軸なので options の同一性に影響してはいけない。"""
    base = {"temperature": 0.8, "num_ctx": 4096}
    assert keys.options_hash({**base, "seed": 1}) == keys.options_hash({**base, "seed": 999})


def test_options_hash_reacts_to_real_change():
    assert keys.options_hash({"temperature": 0.8}) != keys.options_hash({"temperature": 0.2})


def test_canonical_json_is_order_independent():
    assert keys.canonical_json({"b": 1, "a": 2}) == keys.canonical_json({"a": 2, "b": 1})


def _key(**overrides):
    base = dict(
        model="m",
        model_digest="d",
        case_id="c",
        prompt_hash="p",
        seed=1,
        options_hash="o",
        attempt=0,
    )
    return {**base, **overrides}


def test_gen_id_is_deterministic():
    assert keys.gen_id(run_id="r", parts=_key()) == keys.gen_id(run_id="r", parts=_key())


def test_gen_id_changes_with_attempt():
    """attempt を含めないと、エラー後の正当な再生成が既存行と衝突する。"""
    assert keys.gen_id(run_id="r", parts=_key()) != keys.gen_id(run_id="r", parts=_key(attempt=1))


def test_gen_id_changes_with_digest():
    assert keys.gen_id(run_id="r", parts=_key()) != keys.gen_id(
        run_id="r", parts=_key(model_digest="x")
    )


def test_check_hash_is_not_part_of_completion_key():
    """採点手続きを直しても再生成しないことの担保。"""
    assert "check_hash" not in keys.completion_key.__code__.co_varnames
