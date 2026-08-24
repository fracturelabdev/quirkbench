"""JSON Schema の最小実装。

外部依存を増やさないための自前実装。**対応していないキーワードは黙って無視せず、
ケース読み込みの時点でエラーにする。** 無視すると「制約を書いたのに検査されていない」
状態になり、スコアが静かに水増しされる。ベンチマークではこれが最悪の壊れ方になる。
"""

from __future__ import annotations

from typing import Any

TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})

# 実装済みのキーワード。ここに無いものはケース読み込みで落とす。
#
# 同梱ケースが今使っているのは type / required / properties の 3 つだけで、
# 残り 10 は先回りしている。**投機的一般化であることは認めたうえで残している**:
# JSON Schema は閉じた既知の語彙で、各キーワードは 2〜4 行、すべて単体テスト済み。
# 削ると、選択肢問題（enum）や配列出力（items / minItems）のケースを 1 本書くたびに
# 検証器の改修が挟まる。**増やしたいのはケースのほう**なので、そこに摩擦を置かない。
KEYWORDS = frozenset(
    {
        "type",
        "required",
        "properties",
        "items",
        "enum",
        "const",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "additionalProperties",
    }
)


class SchemaError(ValueError):
    """schema の書き方が不正（ケースの誤り）。"""


def check_spec(schema: Any, path: str = "$") -> None:
    """schema 自体を検証する。ケース読み込み時に呼ぶ。"""
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: schema はマッピングでなければならない")
    unknown = set(schema) - KEYWORDS
    if unknown:
        raise SchemaError(
            f"{path}: 未対応の schema キーワード {sorted(unknown)}。"
            "無視すると検査されないまま通るので、minischema.py に実装するか使わない"
        )
    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        for name in names:
            if name not in TYPES:
                raise SchemaError(f"{path}: 未知の type {name!r}")
    for key in ("properties",):
        value = schema.get(key)
        if value is not None:
            if not isinstance(value, dict):
                raise SchemaError(f"{path}.{key}: マッピングでなければならない")
            for name, sub in value.items():
                check_spec(sub, f"{path}.{key}.{name}")
    items = schema.get("items")
    if items is not None:
        check_spec(items, f"{path}.items")
    extra = schema.get("additionalProperties")
    if isinstance(extra, dict):
        check_spec(extra, f"{path}.additionalProperties")
    required = schema.get("required")
    if required is not None and not (
        isinstance(required, list) and all(isinstance(item, str) for item in required)
    ):
        raise SchemaError(f"{path}.required: 文字列のリストでなければならない")


def _type_matches(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        # bool は int の派生。true を integer として通すと採点が甘くなる
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    return value is None


def validate(value: Any, schema: Any, path: str = "$") -> list[str]:
    """``value`` を検証し、違反の一覧を返す。空なら合格。"""
    errors: list[str] = []
    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        if not any(_type_matches(value, name) for name in names):
            errors.append(f"{path}: type が {declared} でない（実際は {type(value).__name__}）")
            return errors  # 型が違う時点で以降の制約は意味を持たない

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: const {schema['const']!r} と一致しない")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: enum {schema['enum']!r} に含まれない")

    if isinstance(value, dict):
        for name in schema.get("required") or []:
            if name not in value:
                errors.append(f"{path}: 必須キー {name!r} が無い")
        properties = schema.get("properties") or {}
        for name, sub in properties.items():
            if name in value:
                errors.extend(validate(value[name], sub, f"{path}.{name}"))
        extra = schema.get("additionalProperties")
        if extra is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}: 余分なキー {name!r}")
        elif isinstance(extra, dict):
            for name, item in value.items():
                if name not in properties:
                    errors.extend(validate(item, extra, f"{path}.{name}"))

    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                errors.extend(validate(item, items, f"{path}[{index}]"))
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: 要素が {schema['minItems']} 未満")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: 要素が {schema['maxItems']} 超")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: {schema['minLength']} 文字未満")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: {schema['maxLength']} 文字超")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {schema['minimum']} 未満")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {schema['maximum']} 超")

    return errors
