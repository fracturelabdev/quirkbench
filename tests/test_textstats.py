"""文字種・言語・反復・件数の計量。

**実測で見つけた誤判定を回帰として固定している。** どれも「モデルの癖ではなく
検出器の癖を測る」形の壊れ方で、FLB-QB-001 §6 が最大の危険として名指ししていたもの。
"""

from __future__ import annotations

import pytest

from quirkbench import textstats as T

# ------------------------------------------------------------------ 文字種


def test_nakaguro_is_not_kana() -> None:
    """・ を仮名に数えると箇条書きで仮名比率が跳ね、言語判定が壊れる。"""
    assert T._classify("・") == "other"
    assert T._classify("あ") == "kana"
    assert T._classify("ア") == "kana"
    assert T._classify("漢") == "kanji"
    assert T._classify("a") == "latin"
    assert T._classify("1") == "digit"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("これは日本語の文章です。しっかり長さがあります。", "ja"),
        ("This is a reasonably long English sentence for detection.", "en"),
        ("这是一段中文句子内容足够长可以用来判定语言种类了对吧", "zh"),
        ("短い", "unknown"),  # 20 文字未満には言語を付けない
    ],
)
def test_detect_language(text: str, expected: str) -> None:
    assert T.detect_language(T.char_class_of(text)) == expected


# -------------------------------------------------------------------- 反復


def test_repeat_detects_degenerate_loop() -> None:
    repeat = T.max_repeat("はい、承知しました。" * 5)
    assert repeat.runs >= 3
    assert "承知" in repeat.unit


def test_repeat_does_not_fire_on_numbered_list() -> None:
    """番号つき箇条書きは周期的に見えるが、各行の中身が違えば反復ではない。"""
    text = "1. 珈琲の木\n2. 街角の椅子\n3. 静かな窓辺\n4. 路地の灯\n5. 小さな隣人"
    assert T.max_repeat(text).span < 12


# -------------------------------------------------------------------- 件数


@pytest.mark.parametrize(
    ("text", "pattern", "count"),
    [
        ("1. あ\n2. い\n3. う", "numbered_list", 3),
        ("１．あ\n２．い", "numbered_list", 2),  # 全角
        ("(1) あ\n(2) い", "numbered_list", 2),
        ("- あ\n- い\n* う\n・え", "bullet_list", 4),
        ("あ\n\nい\n", "lines", 2),
    ],
)
def test_extract_items(text: str, pattern: str, count: int) -> None:
    assert len(T.extract_items(text, pattern)) == count


def test_extract_items_unknown_pattern() -> None:
    with pytest.raises(ValueError, match=r"未知の count\.pattern"):
        T.extract_items("x", "nope")


# ------------------------------------------------- レビュー由来の回帰（実測で再現済み）


def test_kanji_only_japanese_is_not_chinese() -> None:
    """仮名が無いことだけを中国語の証拠にしない。和風の店名は全漢字が普通。

    判定できないものは unknown にする（Codex の指摘）。
    """
    names = "1. 珈琲小径\n2. 街角十二席\n3. 灯窓珈琲\n4. 静席\n5. 住宅街喫茶"
    assert T.detect_language(T.char_class_of(names)) == "unknown"


def test_long_chinese_run_is_still_chinese() -> None:
    """漢字が長く連続していれば中国語と判定する。"""
    text = "这是一段中文句子内容足够长可以用来判定语言种类了对吧"
    assert T.detect_language(T.char_class_of(text)) == "zh"


def test_long_period_repetition_is_detected() -> None:
    """周期 40 文字を超える文のループを取り逃さない（Codex の指摘）。"""
    unit = (
        "これはとても長い反復文で、周期が四十文字を確実に超えるため"
        "検出されにくい文章として追加の語を入れます。"
    )
    assert len(unit) > T.MAX_REPEAT_PERIOD
    assert T.max_repeat(unit * 4).runs >= 3
