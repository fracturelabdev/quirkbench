"""最小 JSON Schema 実装のテスト。

**最重要は「未対応キーワードを黙って通さない」こと。** 通すと、制約を書いたのに
検査されないままスコアが出る。ベンチマークとしては最悪の壊れ方になる。
"""

from __future__ import annotations

import pytest

from quirkbench import minischema as M

PROFILE = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
}


def test_unknown_keyword_rejected() -> None:
    with pytest.raises(M.SchemaError, match="未対応の schema キーワード"):
        M.check_spec({"type": "object", "patternProperties": {}})


def test_unknown_keyword_rejected_in_nested_property() -> None:
    with pytest.raises(M.SchemaError, match=r"\$\.properties\.a"):
        M.check_spec({"type": "object", "properties": {"a": {"multipleOf": 2}}})


def test_unknown_type_rejected() -> None:
    with pytest.raises(M.SchemaError, match="未知の type"):
        M.check_spec({"type": "dict"})


def test_non_mapping_rejected() -> None:
    with pytest.raises(M.SchemaError):
        M.check_spec(["type"])


def test_bad_required_rejected() -> None:
    with pytest.raises(M.SchemaError, match="required"):
        M.check_spec({"required": "name"})


def test_valid_spec_passes() -> None:
    M.check_spec(PROFILE)
    M.check_spec({"type": "array", "items": {"type": "string"}, "minItems": 1})
    M.check_spec({"type": "object", "additionalProperties": {"type": "number"}})


# ---------------------------------------------------------------------- 検証


def test_valid_value() -> None:
    assert M.validate({"name": "a", "age": 42}, PROFILE) == []


def test_missing_required() -> None:
    assert "必須キー" in M.validate({"name": "a"}, PROFILE)[0]


def test_bool_is_not_integer() -> None:
    """bool は int の派生。true を integer として通すと採点が甘くなる。"""
    assert M.validate({"name": "a", "age": True}, PROFILE)


def test_bool_is_not_number() -> None:
    assert M.validate(True, {"type": "number"})
    assert M.validate(True, {"type": "boolean"}) == []


def test_int_accepted_as_number() -> None:
    assert M.validate(42, {"type": "number"}) == []


def test_type_mismatch_stops_further_checks() -> None:
    """型が違う時点で以降の制約は意味を持たない。エラーを 1 件に絞る。"""
    assert len(M.validate("abc", {"type": "integer", "minimum": 5})) == 1


def test_union_type() -> None:
    assert M.validate(None, {"type": ["string", "null"]}) == []
    assert M.validate(1, {"type": ["string", "null"]})


@pytest.mark.parametrize(
    ("value", "schema", "ok"),
    [
        ("ab", {"type": "string", "minLength": 3}, False),
        ("abcd", {"type": "string", "maxLength": 3}, False),
        ("abc", {"type": "string", "minLength": 3, "maxLength": 3}, True),
        (5, {"type": "integer", "minimum": 6}, False),
        (5, {"type": "integer", "maximum": 4}, False),
        (5, {"type": "integer", "minimum": 1, "maximum": 10}, True),
        ("x", {"enum": ["a", "b"]}, False),
        ("a", {"enum": ["a", "b"]}, True),
        (7, {"const": 7}, True),
        (8, {"const": 7}, False),
        ([], {"type": "array", "minItems": 1}, False),
        ([1, 2], {"type": "array", "maxItems": 1}, False),
    ],
)
def test_constraints(value: object, schema: dict, ok: bool) -> None:
    assert (M.validate(value, schema) == []) is ok


def test_array_items_validated() -> None:
    schema = {"type": "array", "items": {"type": "string"}}
    assert M.validate(["a", "b"], schema) == []
    assert M.validate(["a", 1], schema)[0].startswith("$[1]")


def test_additional_properties_false() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "additionalProperties": False,
    }
    assert M.validate({"a": 1}, schema) == []
    assert "余分なキー" in M.validate({"a": 1, "b": 2}, schema)[0]


def test_additional_properties_schema() -> None:
    schema = {"type": "object", "properties": {}, "additionalProperties": {"type": "integer"}}
    assert M.validate({"x": 1}, schema) == []
    assert M.validate({"x": "s"}, schema)


def test_nested_object_path_reported() -> None:
    schema = {
        "type": "object",
        "properties": {"outer": {"type": "object", "properties": {"inner": {"type": "integer"}}}},
    }
    errors = M.validate({"outer": {"inner": "no"}}, schema)
    assert errors[0].startswith("$.outer.inner")
