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


def test_exact_unknown_score_key_rejected():
    """``exclude`` の typo は、囮を含む回答が静かに正解になる。"""
    with pytest.raises(CaseError, match=r"未知の score キー"):
        parse_case(
            {
                "id": "x",
                "dim": "longctx",
                "task": "t",
                "lang": "ja",
                "prompt": "p",
                "failure": {"format": "none"},
                "score": {
                    "kind": "exact",
                    "match": "contains",
                    "expect": "PINECONE-19",
                    "exlude": "SPRUCE-91",
                },
            },
            Path("x.yaml"),
        )


def test_exact_exclude_must_be_nonempty_strings():
    with pytest.raises(CaseError, match=r"score\.exclude"):
        parse_case(
            {
                "id": "x",
                "dim": "longctx",
                "task": "t",
                "lang": "ja",
                "prompt": "p",
                "failure": {"format": "none"},
                "score": {
                    "kind": "exact",
                    "match": "contains",
                    "expect": "PINECONE-19",
                    "exclude": ["SPRUCE-91", ""],
                },
            },
            Path("x.yaml"),
        )


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


def test_bundled_pytest_references_pass_their_checks():
    """参照解が自分のテストを通ることを、素の Python で確かめる。

    隔離の自己診断は macOS の ``qb score`` が持つ。
    ここは参照解とテストの食い違いだけを見る。CI に macOS は無い。
    """
    cases = load_cases(Path("cases"))
    checked = 0
    for case in cases:
        if case.score.get("kind") != "pytest":
            continue
        ref_ns: dict[str, object] = {}
        exec(case.score["reference"], ref_ns, ref_ns)
        entry = str(case.score["entry_point"])
        test_ns: dict[str, object] = {}
        exec(case.score["test"], test_ns, test_ns)
        check = test_ns["check"]
        assert callable(check)
        check(ref_ns[entry])
        checked += 1
    assert checked >= 1


def test_contains_expect_appears_exactly_once():
    """``contains`` の期待値がプロンプトに 1 回だけある。

    0 回なら正解が本文に無い。2 回以上なら、質問や囮が期待値を含んでいて
    そこを繰り返すだけで正解になる。
    """
    cases = load_cases(Path("cases"))
    seen = 0
    for case in cases:
        if case.score.get("kind") != "exact" or case.score.get("match") != "contains":
            continue
        expect = case.score["expect"]
        values = expect if isinstance(expect, list) else [expect]
        for value in values:
            assert case.prompt.count(str(value)) == 1, case.id
        seen += 1
    assert seen >= 1


def test_declared_expect_is_a_perfect_answer():
    """期待値そのものが、その採点器で満点になる。

    参照解の実行は上のテストが見る。こちらは JSON・数値・完全一致で、
    期待値の型や表記が採点器と食い違っていないことを見る。
    """
    from quirkbench.parse import parse
    from quirkbench.scorers import score

    cases = load_cases(Path("cases"))
    checked = 0
    for case in cases:
        kind = case.score.get("kind")
        raw = _perfect_answer(case.score)
        if raw is None or kind not in {"json_schema", "json_keys", "exact", "numeric"}:
            continue
        result = score(parse(raw, case.failure), case, executor=None, embedder=None)
        assert result.score == 1.0, case.id
        checked += 1
    assert checked >= 1


@pytest.mark.parametrize("case_id", ["instruct-shift-ja", "instruct-shift-en"])
def test_shift_rejects_keys_outside_the_four(case_id):
    """``shift`` はキーを 4 つに限る指示なので、余分なキーを足した答えは満点にしない。"""
    import json

    from quirkbench.parse import parse
    from quirkbench.scorers import score

    (case,) = [c for c in load_cases(Path("cases")) if c.id == case_id]
    answer = dict(case.score["expect"], extra=True)
    result = score(
        parse(json.dumps(answer, ensure_ascii=False), case.failure),
        case,
        executor=None,
        embedder=None,
    )
    assert result.score == 0.0
    assert result.sub_metrics["schema_ok"] is False


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("PINECONE-19", 1.0),
        ("第 3 診療所の当直符号は PINECONE-19 です。", 1.0),
        ("SPRUCE-91 または PINECONE-19", 0.0),
        ("PINECONE-19 または SPRUCE-91", 0.0),
        ("ＳＰＲＵＣＥ－９１ または PINECONE-19", 0.0),
        ("SPRUCE-91", 0.0),
    ],
)
def test_clinic_mid_rejects_a_listed_decoy(answer: str, expected: float) -> None:
    """第 3 診療所の符号を問う。正答だけなら正解、囮と並べたら不正解。

    ``contains`` のまま、文で返した正答は通す。``equals`` にすると
    出力の素っ気なさを測ることになる。囮の符号を含んだ回答だけを落とす。
    全角の囮も正規化してから見る。
    """
    from quirkbench.parse import parse
    from quirkbench.scorers import score

    (case,) = [c for c in load_cases(Path("cases")) if c.id == "longctx-clinic-mid-ja"]
    result = score(parse(answer, case.failure), case, executor=None, embedder=None)
    assert result.score == expected


def _run_check(case, fn) -> None:
    namespace: dict[str, object] = {}
    exec(case.score["test"], namespace, namespace)
    check = namespace["check"]
    assert callable(check)
    check(fn)


def test_latest_rejects_a_prefix_and_the_seeded_comparison():
    """最大は末尾と途中に置く。同点のあいだに小さい年を挟み、種まきした ``<`` を落とす。

    最大が 2 件目だと、先頭 2 件だけ比べる実装が通る。
    最大が端だけだと、先頭と末尾だけを比べる実装が通る。
    同点だけの並びだと、``<`` も先に現れたほうを残して通る。
    """
    cases = [c for c in load_cases(Path("cases")) if c.task == "latest"]
    assert len(cases) == 2

    def seeded(records):
        best_name, best_year = records[0]
        for name, year in records[1:]:
            if year < best_year:
                best_name, best_year = name, year
        return best_name

    def prefix(records):
        window = records[:2]
        best_name, best_year = window[0]
        for name, year in window[1:]:
            if year > best_year:
                best_name, best_year = name, year
        return best_name

    def later_tie(records):
        best_name, best_year = records[0]
        for name, year in records[1:]:
            if year >= best_year:
                best_name, best_year = name, year
        return best_name

    def ends_only(records):
        return max((records[0], records[-1]), key=lambda record: record[1])[0]

    for case in cases:
        for fn in (seeded, prefix, later_tie, ends_only):
            with pytest.raises(AssertionError):
                _run_check(case, fn)


def test_count_long_line_number_is_not_shared():
    """常に同じ行番号を書くモデルが、別の code-read まで通らない。"""
    cases = {c.id: c for c in load_cases(Path("cases"))}
    expect = cases["code-read-count-long-ja"].score["expect"]
    assert expect == cases["code-read-count-long-en"].score["expect"]
    shared = {
        cases["code-read-accumulate-ja"].score["expect"],
        cases["code-read-accumulate-en"].score["expect"],
        cases["code-read-off-by-one-ja"].score["expect"],
        cases["code-read-off-by-one-en"].score["expect"],
    }
    assert expect not in shared
    for case_id in ("code-read-count-long-ja", "code-read-count-long-en"):
        bug_line = next(
            line for line in cases[case_id].prompt.splitlines() if line.startswith(f"{expect}:")
        )
        assert ">" in bug_line
        assert "len(" in bug_line


@pytest.mark.parametrize("case_id", ["instruct-shift-ja", "instruct-shift-en"])
def test_shift_schema_can_pass_while_the_value_is_wrong(case_id):
    """schema を通る別の列挙値でも、expect と違えば満点にしない。"""
    import json

    from quirkbench.parse import parse
    from quirkbench.scorers import score

    (case,) = [c for c in load_cases(Path("cases")) if c.id == case_id]
    wrong_shift = "遅番" if case.lang == "ja" else "late"
    answer = dict(case.score["expect"], shift=wrong_shift)
    result = score(
        parse(json.dumps(answer, ensure_ascii=False), case.failure),
        case,
        executor=None,
        embedder=None,
    )
    assert result.sub_metrics["schema_ok"] is True
    assert result.score < 1.0
    assert "shift" in result.sub_metrics["expect_missed"]


@pytest.mark.parametrize("case_id", ["extract-sample-ja", "extract-sample-en"])
def test_sample_keeps_zero_and_a_negative_number(case_id):
    """0 と負数は値として残る。符号を落とすと温度は不一致になる。"""
    import json

    from quirkbench.parse import parse
    from quirkbench.scorers import score

    (case,) = [c for c in load_cases(Path("cases")) if c.id == case_id]
    expect = case.score["expect"]
    perfect = score(
        parse(json.dumps(expect, ensure_ascii=False), case.failure),
        case,
        executor=None,
        embedder=None,
    )
    assert perfect.score == 1.0
    assert expect["humidity"] == 0
    assert expect["temperature"] < 0

    dropped_sign = dict(expect, temperature=abs(expect["temperature"]))
    missed = score(
        parse(json.dumps(dropped_sign, ensure_ascii=False), case.failure),
        case,
        executor=None,
        embedder=None,
    )
    assert "temperature" in missed.sub_metrics["expect_missed"]


def test_clinic_mid_lines_are_diverse_and_long_enough():
    """符号が 2 つだけだと、先頭の囮を除いた残りが正答になり、中央を読まなくて済む。

    各行に別の符号を置く。第 3 診療所への言及は先頭にもあるが、正答の符号は中央の 1 行だけ。
    対訳は置かない。num_ctx の 65% から 75% に収める。
    """
    import re

    (case,) = [c for c in load_cases(Path("cases")) if c.id == "longctx-clinic-mid-ja"]
    assert case.pair is None
    assert case.lang == "ja"
    numbered = []
    for line in case.prompt.splitlines():
        text = line.strip()
        if text[:1].isdigit() and ". " in text:
            numbered.append(text.split(". ", 1)[1])
    assert len(numbered) >= 90
    assert len(set(numbered)) == len(numbered)
    assert case.prompt.count("PINECONE-19") == 1
    assert case.prompt.count("SPRUCE-91") == 1
    assert "特筆すべき事故の記載はなかった" not in case.prompt
    codes = [re.findall(r"[A-Z]+-\d+", body) for body in numbered]
    assert all(len(found) == 1 for found in codes)
    flat = [found[0] for found in codes]
    assert len(set(flat)) == len(flat)
    assert len(set(flat) - {"SPRUCE-91"}) > 1
    pine = next(i for i, body in enumerate(numbered) if "PINECONE-19" in body)
    spruce = next(i for i, body in enumerate(numbered) if "SPRUCE-91" in body)
    assert spruce < 5
    assert flat[spruce] == "SPRUCE-91"
    assert 20 < pine < len(numbered) - 20
    assert flat[pine] == "PINECONE-19"
    assert flat[-1] != "PINECONE-19"
    mentions = [i for i, body in enumerate(numbered) if "第 3 診療所" in body]
    assert len(mentions) >= 2
    assert pine in mentions
    assert spruce in mentions
    assert "PINECONE-19" not in numbered[spruce]
    duty = "第 3 診療所の当直符号は"
    assert sum(duty in body for body in numbered) == 1
    assert duty in numbered[pine]
    ratio = len(case.prompt) / case.options["num_ctx"]
    assert 0.65 <= ratio <= 0.75


@pytest.mark.parametrize(
    ("case_id", "needles", "copied_sample"),
    [
        (
            "extract-sample-ja",
            ["単位を付けない整数", "符号を残", "0 は 0", "「試料」という語を付けず"],
            "試料 B7",
        ),
        (
            "extract-sample-en",
            ["no unit", "minus sign", "write zero as 0", "without the word Sample"],
            "Sample B7",
        ),
    ],
)
def test_sample_prompt_states_sign_zero_and_unit(case_id, needles, copied_sample):
    """期待値は単位なしの整数。符号・ゼロ・単位と、sample が識別子だけであることを言う。

    本文は「試料 B7」「Sample B7」なので、識別子だけの指示が無いと
    試料名をそのまま入れた答えが完全一致で落ちる。
    """
    (case,) = [c for c in load_cases(Path("cases")) if c.id == case_id]
    import json

    from quirkbench.parse import parse
    from quirkbench.scorers import score

    for needle in needles:
        assert needle in case.prompt
    for key in ("temperature", "humidity", "weight"):
        assert isinstance(case.score["expect"][key], int)
        assert not isinstance(case.score["expect"][key], bool)
    copied = dict(case.score["expect"], sample=copied_sample)
    result = score(
        parse(json.dumps(copied, ensure_ascii=False), case.failure),
        case,
        executor=None,
        embedder=None,
    )
    assert "sample" in result.sub_metrics["expect_missed"]
    assert result.score < 1.0


def test_tool_shed_coverage_terms_reject_loose_substrings():
    """短い部分文字列は外し、日英で返却の言い回しが当たるようにする。"""
    from quirkbench.scorers.ideate import _coverage

    cases = {c.id: c for c in load_cases(Path("cases"))}
    en = cases["ideate-tool-shed-en"].score["coverage_terms"]
    ja = cases["ideate-tool-shed-ja"].score["coverage_terms"]
    en_words = [word for group in en for word in group]
    ja_words = [word for group in ja for word in group]
    assert "back" not in en_words
    assert "reserve" not in en_words
    assert "replace" not in en_words
    assert "壊" not in ja_words
    assert all(len(word) >= 4 for word in en_words)
    assert all(len(word) >= 2 for word in ja_words)

    for text in (
        "Keep a backup in the back of the shed.",
        "Give background information.",
        "Move the input backlog to the shelf.",
    ):
        _, missed = _coverage([text], en)
        assert 0 in missed, text
    for text in (
        "Return the drill the next day.",
        "Bring back the drill the next day.",
        "Give back what you borrowed.",
        "Bring it back tomorrow.",
        "Give it back tomorrow.",
        "Give it back.",
        "Bring it back.",
    ):
        _, missed = _coverage([text], en)
        assert 0 not in missed, text
    for text in (
        "Return every tool unbroken.",
        "Return every tool undamaged.",
    ):
        _, missed = _coverage([text], en)
        assert 0 not in missed, text
        assert 1 in missed, text
    _, missed = _coverage(["Report any damage before leaving."], en)
    assert 1 not in missed
    _, missed = _coverage(["Replace the hammer if it is broken."], en)
    assert 1 not in missed
    _, missed = _coverage(["Preserve the tools. They are irreplaceable."], en)
    assert 1 in missed
    assert 2 in missed

    for text in ("工具は返す。", "借りた工具は返さなければならない。", "工具は返すべきだ。"):
        _, missed = _coverage([text], ja)
        assert 0 not in missed, text
    _, missed = _coverage(["決壊や崩壊や破壊の話は規則ではない。"], ja)
    assert 1 in missed
    _, missed = _coverage(["壊したら修理する。"], ja)
    assert 1 not in missed


def _perfect_answer(spec: dict) -> str | None:
    """ケースの expect から、採点器が満点を付けるべき文字列を作る。作れなければ None。"""
    import json

    kind = spec.get("kind")
    expect = spec.get("expect")
    if kind == "json_keys" and isinstance(expect, dict):
        return json.dumps(expect, ensure_ascii=False)
    if kind == "json_schema" and isinstance(expect, dict):
        required = set((spec.get("schema") or {}).get("required") or [])
        # occupation のように、正解が一意でないキーは expect に入れない。
        # そのケースはここからは満点の形を復元できない
        if not required <= set(expect):
            return None
        return json.dumps(expect, ensure_ascii=False)
    if kind == "numeric":
        return str(expect)
    if kind == "exact":
        values = expect if isinstance(expect, list) else [expect]
        text = str(values[0])
        if spec.get("match") == "contains":
            return f"答えは {text} です"
        return text
    return None
