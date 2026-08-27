"""ケース定義の検証。数時間の run が途中でケースの typo に当たって落ちないようにする。"""

from pathlib import Path

import pytest
import yaml

from quirkbench.cases import DEFAULT_OPTIONS, CaseError, load_cases, parse_case

VALID = {
    "id": "x",
    "dim": "reason",
    "task": "t",
    "lang": "ja",
    "prompt": "p",
    "score": {"kind": "exact"},
}


def test_valid_case_parses():
    case = parse_case(dict(VALID), Path("t.yaml"))
    assert case.id == "x"
    assert case.prompt_hash and case.options_hash and case.check_hash


def test_defaults_are_applied():
    case = parse_case(dict(VALID), Path("t.yaml"))
    assert case.options["num_ctx"] == DEFAULT_OPTIONS["num_ctx"]


def test_case_options_override_defaults():
    case = parse_case({**VALID, "options": {"num_ctx": 8192}}, Path("t.yaml"))
    assert case.options["num_ctx"] == 8192


@pytest.mark.parametrize(
    "override",
    [
        {"dim": "nope", "task": "t"},
        {"lang": "fr"},
        {"score": {"kind": "zzz"}},
        {"prompt": "   "},
        {"score": "not-a-mapping"},
    ],
)
def test_invalid_cases_are_rejected(override):
    raw = {**VALID, **override}
    with pytest.raises(CaseError):
        parse_case(raw, Path("t.yaml"))


def test_missing_required_key_is_rejected():
    raw = dict(VALID)
    del raw["score"]
    with pytest.raises(CaseError):
        parse_case(raw, Path("t.yaml"))


def test_translation_pair_shares_check_hash_but_not_prompt_hash():
    """日英対の差分を取るには、採点手続きが同じでプロンプトだけ違う必要がある。"""
    ja = parse_case({**VALID, "id": "a", "lang": "ja", "prompt": "日本語"}, Path("a.yaml"))
    en = parse_case({**VALID, "id": "b", "lang": "en", "prompt": "English"}, Path("b.yaml"))
    assert ja.check_hash == en.check_hash
    assert ja.prompt_hash != en.prompt_hash


def test_duplicate_ids_are_rejected(tmp_path: Path):
    """ID が重複すると resume のキーが衝突し、片方が他方を完了済みに見せる。"""
    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text(yaml.safe_dump(VALID), encoding="utf-8")
    with pytest.raises(CaseError, match=r"重複"):
        load_cases(tmp_path)


def test_empty_directory_is_rejected(tmp_path: Path):
    with pytest.raises(CaseError):
        load_cases(tmp_path)


# --------------------------------------------------- failure / score の検証
# 「宣言していない型は評価されない」ので、キーの typo は
# **静かに検査されない**という最悪の壊れ方をする。ここで落とす。

_BASE = {"id": "x", "dim": "instruct", "task": "t", "lang": "ja", "prompt": "p"}
_SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def _build(**overrides):
    raw = dict(_BASE)
    raw.setdefault("score", {"kind": "json_schema", "schema": _SCHEMA})
    raw.setdefault("failure", {"format": "json"})
    raw.update(overrides)
    return parse_case(raw, Path("x.yaml"))


def test_failure_typo_rejected():
    with pytest.raises(CaseError, match=r"未知の failure キー"):
        _build(failure={"format": "json", "min_token": 4})


def test_failure_unknown_format_rejected():
    with pytest.raises(CaseError, match=r"未知の failure\.format"):
        _build(failure={"format": "yaml"}, score={"kind": "exact"})


def test_failure_unknown_extract_rejected():
    with pytest.raises(CaseError, match=r"未知の failure\.extract"):
        _build(failure={"format": "json", "extract": "magic"})


def test_failure_unknown_language_rejected():
    with pytest.raises(CaseError, match=r"未知の failure\.language"):
        _build(failure={"format": "json", "language": "zh"})


def test_count_requires_n_and_pattern():
    with pytest.raises(CaseError, match=r"failure\.count"):
        _build(failure={"format": "none", "count": {"n": 5}}, score={"kind": "exact"})


def test_count_unknown_pattern_rejected():
    with pytest.raises(CaseError, match=r"未知の count\.pattern"):
        _build(
            failure={"format": "none", "count": {"n": 5, "pattern": "table"}},
            score={"kind": "exact"},
        )


def test_max_tokens_at_num_predict_rejected():
    """max_tokens >= num_predict だと overlong は原理的に発火しない。

    先に truncated になるので、母数だけ増えて分子が常に 0 になる。
    実際に既存のケース 3 件がこの状態だった（実測で発見）。
    """
    with pytest.raises(CaseError, match=r"overlong は原理的に発火しない"):
        _build(
            failure={"format": "none", "max_tokens": 300},
            options={"num_predict": 300},
            score={"kind": "exact"},
        )


def test_max_tokens_below_num_predict_allowed():
    case = _build(
        failure={"format": "none", "max_tokens": 100},
        options={"num_predict": 300},
        score={"kind": "exact"},
    )
    assert case.failure["max_tokens"] == 100


def test_kind_and_format_must_agree():
    """ずれていると採点器と失敗検出器が別のものを見る。"""
    with pytest.raises(CaseError, match=r"failure\.format は 'json'"):
        _build(failure={"format": "none"})


def test_json_schema_requires_schema():
    with pytest.raises(CaseError, match=r"schema が要る"):
        _build(score={"kind": "json_schema"})


def test_bad_schema_rejected_at_load():
    with pytest.raises(CaseError, match=r"未対応の schema キーワード"):
        _build(score={"kind": "json_schema", "schema": {"type": "object", "oneOf": []}})


def test_real_cases_all_validate():
    """同梱ケースが検証を通ることを実ファイルで確認する。"""
    cases = load_cases(Path("cases"))
    assert {c.id for c in cases} >= {"instruct-json-profile-ja", "instruct-json-profile-en"}
