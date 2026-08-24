"""テキストの計量 — 文字種・言語・反復・件数。

`parse.py` から分けているのは関心が違うから。あちらは**構造の抽出**
（フェンス・均衡した括弧・ペイロードの切り出し）で、こちらは**文字列の計量**。
どちらも `Parsed` に集まるが、片方を直すときにもう片方を読む必要はない。

ここは判定をしない。**閾値を持つのは `detect_language` だけ**で、それ以外の
判定（何回以上の反復を `repeated` と呼ぶか等）は `failures.py` が持つ。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

COUNT_PATTERNS = frozenset({"numbered_list", "bullet_list", "lines"})

_NUMBERED_RE = re.compile(r"^[ \t]*[(（]?([0-9０-９]{1,3})[)）.．、:：]\s*(\S.*?)[ \t]*$", re.M)
# 和文の中黒は「・項目」と空白を置かないのが普通。ASCII の -*+ は空白を必須にする
# （ハイフンや強調記号を箇条書きと誤検出しないため）
_BULLET_RE = re.compile(r"^[ \t]*(?:[-*+]\s+|[・•‣]\s*)(\S.*?)[ \t]*$", re.M)

# 言語判定の下限。短い応答に言語ラベルを付けない（FLB-QB-001 §6 候補階層）。
# **この閾値はここにしか置かない。** 2 箇所に持つと、片方だけ動かしたときに
# 該当する長さの応答がまるごと wrong_language になる。
MIN_LETTERS_FOR_LANG = 20
JA_KANA_RATIO = 0.05
EN_LATIN_RATIO = 0.70
# 中国語と判定するのに必要な漢字の連続長。
# **仮名が無いことを中国語の証拠にしない。** 和風の店名は全漢字が普通で、
# 短い漢字語を並べただけの日本語を zh に落とすと、測るのは表記スタイルになる。
ZH_MIN_KANJI_RUN = 12

# 反復検出。短い周期は総当たり、長い周期は窓の衝突から候補を作る
MAX_REPEAT_PERIOD = 40
MAX_REPEAT_SCAN = 4000
_REPEAT_WINDOW = 16
_REPEAT_CANDIDATES = 12


@dataclass(frozen=True)
class CharClass:
    """本文の文字種の内訳。比率の分母は「文字」= 仮名 + 漢字 + ラテン。"""

    kana: int = 0
    kanji: int = 0
    latin: int = 0
    digit: int = 0
    other: int = 0
    max_kanji_run: int = 0

    @property
    def letters(self) -> int:
        return self.kana + self.kanji + self.latin

    def _ratio(self, count: int) -> float:
        return count / self.letters if self.letters else 0.0

    @property
    def kana_ratio(self) -> float:
        return self._ratio(self.kana)

    @property
    def kanji_ratio(self) -> float:
        return self._ratio(self.kanji)

    @property
    def latin_ratio(self) -> float:
        return self._ratio(self.latin)


@dataclass(frozen=True)
class Repeat:
    """連続反復の最大値。``span`` は反復区間の文字数。"""

    runs: int = 1
    span: int = 0
    unit: str = ""


# ------------------------------------------------------------------ 文字種


def _classify(ch: str) -> str:
    code = ord(ch)
    if code == 0x30FB:  # ・ は区切り記号。仮名に数えると箇条書きで比率が跳ねる
        return "other"
    if 0x3041 <= code <= 0x309F or 0x30A1 <= code <= 0x30FF or 0xFF66 <= code <= 0xFF9D:
        return "kana"
    if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
        return "kanji"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isdigit():
        return "digit"
    return "other"


def char_class_of(text: str) -> CharClass:
    counts = {"kana": 0, "kanji": 0, "latin": 0, "digit": 0, "other": 0}
    run = best_run = 0
    for ch in text:
        kind = _classify(ch)
        counts[kind] += 1
        if kind == "kanji":
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0
    return CharClass(**counts, max_kanji_run=best_run)


def detect_language(stats: CharClass) -> str:
    """文字種比から ja / en / zh を分類する。判断がつかなければ ``unknown``。

    ``zh`` を独立させているのは、対象が qwen2.5 である以上、英語漏れより
    **中国語漏れのほうが本体**だから（FLB-QB-001 §6）。ただし
    **仮名が無いことだけを中国語の証拠にしない** — 漢字の連続長を要求する。
    """
    if stats.letters < MIN_LETTERS_FOR_LANG:
        return "unknown"
    if stats.kana_ratio >= JA_KANA_RATIO:
        return "ja"
    if stats.latin_ratio >= EN_LATIN_RATIO:
        return "en"
    if (
        stats.kana == 0
        and stats.max_kanji_run >= ZH_MIN_KANJI_RUN
        and stats.kanji_ratio > stats.latin_ratio
    ):
        return "zh"
    return "unknown"


# -------------------------------------------------------------- 反復の検出


def _check_period(text: str, period: int) -> tuple[int, int, int]:
    """周期 ``period`` の連続反復を数える。``(runs, span, head)``。

    ``text[i] == text[i+period]`` が run 個続けば、周期区間の長さは run+period、
    反復回数は ``run // period + 1``。
    """
    best = (1, 0, 0)
    run = 0
    for index in range(len(text) - period):
        if text[index] == text[index + period]:
            run += 1
            reps = run // period + 1
            if reps > best[0]:
                best = (reps, run + period, index - run + 1)
        else:
            run = 0
    return best


def _long_period_candidates(text: str) -> list[int]:
    """長い周期の候補を、同一 16 文字窓の出現間隔から作る。

    総当たりを 40 文字までに抑えたまま、**長い文をループするモデルを取り逃さない**
    ため。総当たりを広げると 1 応答あたりの計算量が実用外になる。
    """
    first: dict[str, int] = {}
    counts: dict[int, int] = {}
    limit = len(text) // 2
    for index in range(len(text) - _REPEAT_WINDOW + 1):
        key = text[index : index + _REPEAT_WINDOW]
        if key in first:
            period = index - first[key]
            if MAX_REPEAT_PERIOD < period <= limit:
                counts[period] = counts.get(period, 0) + 1
        else:
            first[key] = index
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [period for period, _ in ranked[:_REPEAT_CANDIDATES]]


def max_repeat(text: str) -> Repeat:
    """連続して繰り返される最長の n-gram を返す。

    コーパス頻度は使わない。**同一 n-gram の連続反復だけ**を数える（FLB-QB-001 §6）。
    """
    text = text[:MAX_REPEAT_SCAN]
    periods = list(range(1, min(MAX_REPEAT_PERIOD, len(text) // 2) + 1))
    periods += _long_period_candidates(text)

    best = Repeat()
    for period in periods:
        runs, span, head = _check_period(text, period)
        if runs > best.runs or (runs == best.runs and span > best.span):
            best = Repeat(runs=runs, span=span, unit=text[head : head + period])
    return best


# ------------------------------------------------------------ 件数の数え方


def extract_items(text: str, pattern: str) -> tuple[str, ...]:
    if pattern == "numbered_list":
        return tuple(match.group(2) for match in _NUMBERED_RE.finditer(text))
    if pattern == "bullet_list":
        return tuple(match.group(1) for match in _BULLET_RE.finditer(text))
    if pattern == "lines":
        return tuple(line.strip() for line in text.split("\n") if line.strip())
    raise ValueError(f"未知の count.pattern: {pattern!r}")
