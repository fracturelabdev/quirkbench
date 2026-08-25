"""失敗型の語彙と汎用検出器（FLB-QB-001 §6）。

**語彙は 1 つ、生産者は 2 つ、パースは 1 回。**
汎用の型はここが判定し、次元固有の型（schema や k を知る必要があるもの）は
scorer が**同じ語彙のタグを append** する。

設計上の最大の危険は、9 型を同じ強度の機械ラベルとして並べると
**モデルの癖ではなく検出器の癖を測ってしまう**こと。3 つの原則で塞いでいる:

A. 単一ラベルをやめ、複数タグを許す。固定するのは原因の優先順位だけ
B. ケース YAML に判定手続きを宣言する。宣言の無い型は評価しない
C. 判定の固さで 3 階層に分け、主表に出すのは固いものだけ

**分布は「その型を適用したケース」を母数にする。** これが `applicable` の役目で、
これが無いと「宣言していない型が 0 件」と「適用して 0 件」が区別できない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .parse import Parsed

TRUNCATED = "truncated"
EMPTY = "empty"
INCOMPLETE = "incomplete"
FORMAT_BROKEN = "format_broken"
PREAMBLE = "preamble"
COUNT_MISMATCH = "count_mismatch"
OVERLONG = "overlong"
WRONG_LANGUAGE = "wrong_language"
ZH_LEAK = "zh_leak"
REPEATED = "repeated"
NON_ATTEMPT = "non_attempt"
#: 実行採点でのみ出る（§12.6）。どちらも**母数内**で、基盤失敗とは別物。
EXEC_ERROR = "exec_error"
EXEC_TIMEOUT = "exec_timeout"

# 原因の優先順位。タグはこの順で並べる（FLB-QB-001 §6-A）
PRIORITY: tuple[str, ...] = (
    EMPTY,
    TRUNCATED,
    INCOMPLETE,
    FORMAT_BROKEN,
    PREAMBLE,
    COUNT_MISMATCH,
    OVERLONG,
    WRONG_LANGUAGE,
    ZH_LEAK,
    REPEATED,
    # 実行の失敗は、形式の失敗より後に置く。format_broken なコードは
    # そもそも実行に届かないので、両方付くときは形式が原因。
    EXEC_TIMEOUT,
    EXEC_ERROR,
    NON_ATTEMPT,
)

# 判定の固さ。主表に出すのは hard / declared だけ（FLB-QB-001 §6-C）
TIER: dict[str, str] = {
    TRUNCATED: "hard",
    EMPTY: "hard",
    INCOMPLETE: "declared",
    FORMAT_BROKEN: "declared",
    PREAMBLE: "declared",
    COUNT_MISMATCH: "declared",
    OVERLONG: "declared",
    # 実行の事実そのもの。returncode と継承 fd の内容だけで決まる（§12.5・§12.6）
    EXEC_TIMEOUT: "hard",
    EXEC_ERROR: "hard",
    WRONG_LANGUAGE: "candidate",
    ZH_LEAK: "candidate",
    REPEATED: "candidate",
    NON_ATTEMPT: "candidate",
}

# repeated の閾値。同一 n-gram が 3 回以上連続し、かつその区間が 12 文字以上
REPEAT_MIN_RUNS = 3
REPEAT_MIN_SPAN = 12


@dataclass(frozen=True)
class FailureReport:
    """1 応答の失敗型。``applicable`` は分布の母数に入る型の集合。"""

    tags: tuple[str, ...] = ()
    applicable: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def merge(self, *, tags: tuple[str, ...], applicable: tuple[str, ...]) -> FailureReport:
        """scorer が付けたタグを取り込む。元のオブジェクトは変えない。"""
        return FailureReport(
            tags=_ordered(set(self.tags) | set(tags)),
            applicable=_ordered(set(self.applicable) | set(applicable)),
            details=dict(self.details),
        )


def _ordered(tags: set[str]) -> tuple[str, ...]:
    return tuple(tag for tag in PRIORITY if tag in tags)


def detect(
    parsed: Parsed,
    spec: dict[str, Any] | None,
    *,
    done_reason: str | None,
    eval_count: int | None,
) -> FailureReport:
    """汎用の失敗型を判定する。``spec`` はケースの ``failure`` ブロック。"""
    spec = spec or {}
    # format は Parsed から読む。spec から再導出すると、同じ spec が両者に渡される
    # ことだけが正しさの根拠になり、二重解釈の余地が残る
    fmt = parsed.fmt
    language = str(spec.get("language", "none"))
    count_spec = spec.get("count") or {}
    max_tokens = spec.get("max_tokens")
    max_chars = spec.get("max_chars")

    tags: set[str] = set()
    applicable: set[str] = {TRUNCATED, EMPTY}
    details: dict[str, Any] = {}

    truncated = done_reason == "length"
    if truncated:
        tags.add(TRUNCATED)

    if (eval_count is not None and eval_count == 0) or parsed.visible_chars == 0:
        # 中身が無いものに派生判定を掛けても、測るのは検出器の既定値でしかない
        tags.add(EMPTY)
        return FailureReport(_ordered(tags), _ordered(applicable), {"visible_chars": 0})

    if fmt != "none":
        applicable |= {FORMAT_BROKEN, PREAMBLE}
        # incomplete は done_reason == "stop" のときだけ発火する。打ち切られた行を
        # 母数に入れると、**原理的に発火しない行**が分母を膨らませ、切れやすい
        # モデルほど incomplete 率が低く見える
        if done_reason != "length":
            applicable.add(INCOMPLETE)
        if done_reason == "stop" and parsed.unclosed:
            # 打ち切られていないのに閉じていない。小モデルの主要な癖がここに出る
            tags.add(INCOMPLETE)
            details["unclosed"] = list(parsed.unclosed)
        if not parsed.format_ok:
            tags.add(FORMAT_BROKEN)
            details["format_error"] = parsed.json_error or parsed.syntax_error
        elif parsed.preamble:
            # 抽出は成功していて、その手前に非空テキストがある
            tags.add(PREAMBLE)
            details["preamble"] = parsed.preamble[:120]

    if count_spec:
        applicable.add(COUNT_MISMATCH)
        expected = int(count_spec["n"])
        details["item_count"] = len(parsed.items)
        if len(parsed.items) != expected:
            tags.add(COUNT_MISMATCH)

    if max_tokens is not None or max_chars is not None:
        applicable.add(OVERLONG)
        if max_tokens is not None and eval_count is not None and eval_count > int(max_tokens):
            tags.add(OVERLONG)
        # 長さは content_chars（フェンス記号を除いた本文）で測る。len(raw) だと
        # フェンスを付けるモデルだけが数文字ぶん不利になる
        if max_chars is not None and parsed.content_chars > int(max_chars):
            tags.add(OVERLONG)

    # 判定できたときだけ母数に入れる。**判定不能を別言語の証拠にしない。**
    # 短い応答（20 文字未満）や、仮名の無い短い漢字語の羅列は unknown になる。
    # unknown を「日本語ではない」と数えると、和風の店名を並べただけで
    # wrong_language が付き、測るのは言語逸脱ではなく表記スタイルになる。
    # 閾値はここに持たない。**言語の下限は parse.detect_language が単独で持つ**
    # （短い応答・仮名の無い短い漢字語の羅列は unknown になる）。2 箇所に閾値を
    # 置くと、片方だけ動かしたときに帯域まるごとが wrong_language になる。
    if language in {"ja", "en"}:
        details["detected_lang"] = parsed.detected_lang
        if parsed.detected_lang != "unknown":
            applicable |= {WRONG_LANGUAGE, ZH_LEAK}
            if parsed.detected_lang != language:
                tags.add(WRONG_LANGUAGE)
            if parsed.detected_lang == "zh":
                tags.add(ZH_LEAK)

    # 反復は散文の現象として測る。JSON / Python のペイロードは構造上あたりまえに
    # 繰り返すので、同じ閾値を当てると形式のあるケースだけ反復率が高く出る
    if fmt == "none":
        applicable.add(REPEATED)
        if parsed.repeat.runs >= REPEAT_MIN_RUNS and parsed.repeat.span >= REPEAT_MIN_SPAN:
            tags.add(REPEATED)
            details["repeat"] = {"runs": parsed.repeat.runs, "unit": parsed.repeat.unit[:40]}

    return FailureReport(_ordered(tags), _ordered(applicable), details)


def detect_non_attempt(
    *,
    score: float | None,
    eval_count: int | None,
    min_tokens: Any,
    truncated: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """スコアが出てから判定する型。``(tags, applicable)`` を返す。

    ``refused`` は廃止した。拒否語彙の一致だけで判定すると、短い正答
    （数値・はい/いいえ）を拒否と誤る（FLB-QB-001 §6-C）。
    """
    if min_tokens is None or score is None or eval_count is None:
        return (), ()
    if score == 0 and eval_count < int(min_tokens) and not truncated:
        return (NON_ATTEMPT,), (NON_ATTEMPT,)
    return (), (NON_ATTEMPT,)
