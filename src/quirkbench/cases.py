"""ケース定義の読み込みと検証。

ケースは YAML に宣言的に書く。コードに埋めない。
検証をここで済ませることで、数時間かかる run が「3 時間後にケースの typo で落ちる」
という壊れ方をしないようにする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import keys, minischema
from .parse import EXTRACTORS, FORMATS
from .textstats import COUNT_PATTERNS

DIMENSIONS = frozenset(
    {"instruct", "code-gen", "code-fix", "code-read", "extract", "reason", "longctx", "ideate"}
)
SCORE_KINDS = frozenset({"json_schema", "json_keys", "exact", "numeric", "pytest", "ideate"})
LANGUAGES = frozenset({"ja", "en"})

# failure ブロックで書けるキー。未知のキーは黙って無視せず落とす
FAILURE_KEYS = frozenset(
    {"format", "extract", "language", "count", "max_tokens", "min_tokens", "max_chars"}
)
# score.kind ごとに要求する failure.format。ずれていると採点と失敗判定が別物を見る
REQUIRED_FORMAT = {"json_schema": "json", "json_keys": "json", "pytest": "python"}

# 既定の生成パラメータ。ケース側の options で上書きできる。
# temperature を 0 にしないのは、**seed 間のばらつきそのものを測る**のが目的だから。
DEFAULT_OPTIONS: dict[str, Any] = {
    "temperature": 0.8,
    "top_p": 0.9,
    "num_predict": 512,
    "num_ctx": 4096,
}


class CaseError(ValueError):
    """ケース定義が不正。"""


@dataclass(frozen=True)
class Case:
    id: str
    dim: str
    lang: str
    prompt: str
    options: dict[str, Any]
    score: dict[str, Any]
    failure: dict[str, Any]
    # **独立な観測の単位**（§17.1）。同じ問題を ja/en で 2 回聞いても観測は 1 件で、
    # `id` の数と観測の数は一致しない。**推測させない** — `pair` と同じ理由で、
    # ID の命名規則から切り出すと、命名が揺れた瞬間に数え方が黙って狂う
    task: str
    # 対訳ケースの相手。S5 の ja_penalty（同一課題の日英差分）が読む。
    # ここで保持しておかないと、レポート側が ID の命名規則から推測することになる
    pair: str | None
    path: Path
    prompt_hash: str = field(compare=False, default="")
    options_hash: str = field(compare=False, default="")
    check_hash: str = field(compare=False, default="")


def _require(raw: dict[str, Any], key: str, path: Path) -> Any:
    if key not in raw:
        raise CaseError(f"{path}: 必須キー {key!r} が無い")
    return raw[key]


def parse_case(raw: dict[str, Any], path: Path) -> Case:
    case_id = str(_require(raw, "id", path))
    dim = str(_require(raw, "dim", path))
    lang = str(_require(raw, "lang", path))
    prompt = str(_require(raw, "prompt", path))
    task = str(_require(raw, "task", path))
    score = _require(raw, "score", path)

    if dim not in DIMENSIONS:
        raise CaseError(f"{path}: 未知の dim {dim!r}（既知: {sorted(DIMENSIONS)}）")
    if lang not in LANGUAGES:
        raise CaseError(f"{path}: 未知の lang {lang!r}（既知: {sorted(LANGUAGES)}）")
    if not isinstance(score, dict) or "kind" not in score:
        raise CaseError(f"{path}: score は kind を持つマッピングでなければならない")
    if score["kind"] not in SCORE_KINDS:
        raise CaseError(
            f"{path}: 未知の score.kind {score['kind']!r}（既知: {sorted(SCORE_KINDS)}）"
        )
    if not prompt.strip():
        raise CaseError(f"{path}: prompt が空")
    if not task.strip():
        raise CaseError(f"{path}: task が空")

    options = {**DEFAULT_OPTIONS, **(raw.get("options") or {})}
    failure = raw.get("failure") or {}
    _validate_failure(failure, options, path)
    _validate_score(score, failure, path)

    return Case(
        id=case_id,
        dim=dim,
        lang=lang,
        prompt=prompt,
        task=task,
        options=options,
        score=score,
        failure=failure,
        pair=raw.get("pair"),
        path=path,
        prompt_hash=keys.prompt_hash(prompt),
        options_hash=keys.options_hash(options),
        check_hash=keys.check_hash(score, failure),
    )


def _validate_failure(failure: dict[str, Any], options: dict[str, Any], path: Path) -> None:
    """failure ブロックの検証。

    **宣言していない型は評価されない**（分布の母数にも入らない）ので、
    キーの typo は「静かに検査されない」という最悪の壊れ方をする。ここで落とす。
    """
    if not isinstance(failure, dict):
        raise CaseError(f"{path}: failure はマッピングでなければならない")
    unknown = set(failure) - FAILURE_KEYS
    if unknown:
        raise CaseError(f"{path}: 未知の failure キー {sorted(unknown)}")

    fmt = failure.get("format", "none")
    if fmt not in FORMATS:
        raise CaseError(f"{path}: 未知の failure.format {fmt!r}（既知: {sorted(FORMATS)}）")
    extract = failure.get("extract")
    if extract is not None and extract not in EXTRACTORS:
        raise CaseError(f"{path}: 未知の failure.extract {extract!r}")
    language = failure.get("language", "none")
    if language not in LANGUAGES | {"none"}:
        raise CaseError(f"{path}: 未知の failure.language {language!r}")

    count = failure.get("count")
    if count is not None:
        if not isinstance(count, dict) or "n" not in count or "pattern" not in count:
            raise CaseError(f"{path}: failure.count は n と pattern を持つマッピング")
        if count["pattern"] not in COUNT_PATTERNS:
            raise CaseError(f"{path}: 未知の count.pattern {count['pattern']!r}")

    max_tokens = failure.get("max_tokens")
    if max_tokens is not None:
        num_predict = options.get("num_predict")
        if num_predict is not None and int(max_tokens) >= int(num_predict):
            raise CaseError(
                f"{path}: failure.max_tokens ({max_tokens}) が num_predict ({num_predict}) "
                "以上だと overlong は原理的に発火しない（先に truncated になる）。"
                "母数だけ増えて分子が常に 0 になるので、長さ制約は max_chars で書くか "
                "num_predict を上げる"
            )


def _validate_score(score: dict[str, Any], failure: dict[str, Any], path: Path) -> None:
    kind = score["kind"]
    required = REQUIRED_FORMAT.get(kind)
    if required is not None and failure.get("format") != required:
        raise CaseError(
            f"{path}: score.kind={kind!r} なら failure.format は {required!r} でなければならない。"
            "ずれていると採点器と失敗検出器が別のものを見る"
        )
    if kind == "json_schema":
        schema = score.get("schema")
        if schema is None:
            raise CaseError(f"{path}: score.kind=json_schema には schema が要る")
        try:
            minischema.check_spec(schema)
        except minischema.SchemaError as exc:
            raise CaseError(f"{path}: {exc}") from exc


def load_cases(
    root: Path,
    *,
    dims: set[str] | None = None,
    ids: set[str] | None = None,
) -> list[Case]:
    """``root`` 以下の YAML をすべて読み、検証して返す。

    ID の重複は**その場で落とす**。重複したまま走らせると resume のキーが衝突し、
    片方の結果がもう片方を完了済みに見せてしまう。
    """
    files = sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
    if not files:
        raise CaseError(f"{root} にケースが 1 件も無い")

    cases: list[Case] = []
    seen: dict[str, Path] = {}
    for path in files:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise CaseError(f"{path}: トップレベルがマッピングでない")
        case = parse_case(raw, path)
        if case.id in seen:
            raise CaseError(f"ケース ID {case.id!r} が重複している: {seen[case.id]} と {path}")
        seen[case.id] = path
        cases.append(case)

    if dims:
        cases = [c for c in cases if c.dim in dims]
    if ids:
        cases = [c for c in cases if c.id in ids]
    if not cases:
        raise CaseError("絞り込みの結果、ケースが 0 件になった")
    return cases
