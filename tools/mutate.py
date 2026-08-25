#!/usr/bin/env python
"""ミューテーションテスト — ガードが**呼ばれている**ことを確認する。

カバレッジは「その行を通ったか」しか見ない。ガードを定数に置き換えても
テストが落ちなければ、そのガードには**呼び出しを検証するテストが無い**。
実際、`scorers/instruct.py` は行カバレッジ 100% のまま 1 件のガードが
無防備だった（S2 で発見）。

各変異は「この安全策を無効化したら」を 1 つずつ再現する。**全部落ちるのが正常。**
生き残りが出たら、テストを足すか——そのガードが冗長ならガードを消す
（S2 では言語判定の閾値が 2 箇所にあり、後者だった）。

    uv run python tools/mutate.py

対象文字列はソースと一致していなければならない。リファクタで「対象なし」が
出たら、**それは検証が効いていない**ということなので追随させる。
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "src" / "quirkbench"

#: 境界の変異は**別扱いにする**（FLB-QB-001 §12.10）。
#:
#: 理由は 2 つ。**① 走らせる場所が違う** — 境界テストは macOS でしか回らないので、
#: ubuntu の `checks` で回すと全部「落ちた」ことになり、変異が効いているのか
#: skip されただけなのか区別がつかない。**② 時間が桁違い** — 境界テストは
#: 実際に sandbox-exec を 20 回以上起動するので 1 変異あたり 30 秒を超える。
#: 通常の変異まで巻き込むと、検査そのものが回されなくなる。
FAST_ARGS = ["--deselect", "tests/test_sandbox.py"]
BOUNDARY_ARGS = ["tests/test_sandbox.py"]

M = [
    (
        "parse.py",
        "フェンス行を前置きから除く処理を無効化",
        'preamble = strip_fence_lines(raw[:span_start]).strip() if fmt != "none" else ""',
        'preamble = raw[:span_start].strip() if fmt != "none" else ""',
    ),
    (
        "parse.py",
        "反復検出でフェンス記号を除く処理を無効化",
        "content = strip_fence_lines(raw)",
        "content = raw",
    ),
    (
        "textstats.py",
        "・ を仮名として数える",
        '    if code == 0x30FB:  # ・ は区切り記号。仮名に数えると箇条書きで比率が跳ねる\n        return "other"\n',
        "",
    ),
    (
        "parse.py",
        "抽出候補の形式検証を無効化（第一候補を無条件に採る）",
        "        ok = json_error is None and syntax_error is None",
        "        ok = True",
    ),
    (
        "parse.py",
        "未閉じフェンスを応答末尾のブロックから取る",
        "    if source_block is not None and not source_block.closed:",
        "    if blocks and not blocks[-1].closed:",
    ),
    (
        "textstats.py",
        "zh 判定の漢字連続長の要件を外す",
        "        and stats.max_kanji_run >= ZH_MIN_KANJI_RUN\n",
        "",
    ),
    (
        "textstats.py",
        "長周期の反復候補を無効化",
        "    periods += _long_period_candidates(text)",
        "",
    ),
    (
        "failures.py",
        "empty の短絡を無効化",
        '        return FailureReport(_ordered(tags), _ordered(applicable), {"visible_chars": 0})',
        "        pass",
    ),
    (
        "failures.py",
        "format 検証の成否に関わらず preamble を付ける",
        "        elif parsed.preamble:",
        "        if parsed.preamble:",
    ),
    (
        "failures.py",
        "打ち切り行でも incomplete を母数に入れる",
        '        if done_reason != "length":\n            applicable.add(INCOMPLETE)',
        "        applicable.add(INCOMPLETE)",
    ),
    (
        "failures.py",
        "宣言の有無に関わらず count_mismatch を母数に入れる",
        "    if count_spec:\n        applicable.add(COUNT_MISMATCH)",
        "    applicable.add(COUNT_MISMATCH)\n    if count_spec:",
    ),
    (
        "failures.py",
        "overlong を len(raw) で測る",
        "        if max_chars is not None and parsed.content_chars > int(max_chars):",
        "        if max_chars is not None and len(parsed.raw) > int(max_chars):",
    ),
    (
        "textstats.py",
        "言語判定の 20 文字下限を撤廃",
        '    if stats.letters < MIN_LETTERS_FOR_LANG:\n        return "unknown"\n',
        "",
    ),
    (
        "failures.py",
        "unknown を別言語として数える",
        '        if parsed.detected_lang != "unknown":\n            applicable |= {WRONG_LANGUAGE, ZH_LEAK}\n            if parsed.detected_lang != language:',
        "        if True:\n            applicable |= {WRONG_LANGUAGE, ZH_LEAK}\n            if parsed.detected_lang != language:",
    ),
    (
        "failures.py",
        "構造つきペイロードにも repeated を適用",
        '    if fmt == "none":\n        applicable.add(REPEATED)',
        "    if True:\n        applicable.add(REPEATED)",
    ),
    (
        "failures.py",
        "non_attempt を常に発火させる",
        "    if score == 0 and eval_count < int(min_tokens) and not truncated:",
        "    if True:",
    ),
    (
        "minischema.py",
        "未対応キーワードの検出を無効化",
        "    if unknown:\n        raise SchemaError(",
        "    if False:\n        raise SchemaError(",
    ),
    (
        "minischema.py",
        "bool を integer として通す",
        "        return isinstance(value, int) and not isinstance(value, bool)",
        "        return isinstance(value, int)",
    ),
    (
        "scoring.py",
        "採点済みスキップを無効化",
        "        if score_key in done:",
        "        if False:",
    ),
    ("scoring.py", "採点後の重複排除キー更新を削除", "        done.add(score_key)\n", ""),
    (
        "scoring.py",
        "未実装 kind を黙って握りつぶす",
        "            summary.unsupported[kind] = summary.unsupported.get(kind, 0) + 1",
        "            pass",
    ),
    (
        "keys.py",
        "完了キーの構成から要素を 1 つ落とす",
        '        options_hash=row["options_hash"],\n',
        '        options_hash="",\n',
    ),
    (
        "cases.py",
        "max_tokens >= num_predict の検出を無効化",
        "        if num_predict is not None and int(max_tokens) >= int(num_predict):",
        "        if False:",
    ),
    (
        "cases.py",
        "score.kind と failure.format の整合検査を無効化",
        '    if required is not None and failure.get("format") != required:',
        "    if False:",
    ),
    (
        "scorers/instruct.py",
        "schema 不合格でも expect 一致率を返す",
        "        score=rate if schema_ok else 0.0,",
        "        score=rate,",
    ),
    (
        "scorers/instruct.py",
        "scorer が失敗型の語彙を持つ（format_broken を append）",
        '                "expect_matched": 0,\n            },\n        )',
        '                "expect_matched": 0,\n            },\n            tags=("format_broken",),\n            applicable=("format_broken",),\n        )',
    ),
    (
        "keys.py",
        "キー項目の欠落チェックを無効化（None 入りキーを作る）",
        "    if any(row.get(field) is None for field in _KEY_FIELDS):\n        return None",
        "    if False:\n        return None",
    ),
    (
        "store.py",
        "キー項目が欠けた行を黙って捨てる",
        "                report.missing_fields += 1\n",
        "                pass\n",
    ),
    # ---- S4: ideate の代理指標（FLB-QB-001 §13）
    (
        "scorers/ideate.py",
        "埋め込み器の必須化を外す（ローカル計算に落ちる経路が開く）",
        '    if embedder is None:\n        raise EmbedderRequired(f"{case.id}: 代理指標の採点には embedder が要る")\n',
        "",
    ),
    (
        "scorers/ideate.py",
        "脱線ゲートを無効化（閾値 0 で全件が通る）",
        "OFFTOPIC_THRESHOLD = 0.35",
        "OFFTOPIC_THRESHOLD = 0.0",
    ),
    (
        "scorers/ideate.py",
        "形態フィルタの「文字が無い」判定を外す（数字だけの案が valid になる）",
        '    if stats.letters == 0:\n        return "no_letters"\n',
        "",
    ),
    (
        "scorers/ideate.py",
        "形態フィルタの長さ下限を外す",
        '    if len(text) < MIN_ITEM_CHARS:\n        return "too_short"\n',
        "",
    ),
    (
        "scorers/ideate.py",
        "valid_rate の分母を産出件数にする（1 案返して満点が取れる）",
        "    valid_rate = min(1.0, len(valid) / wanted) if wanted else 0.0",
        "    valid_rate = len(valid) / len(items) if items else 0.0",
    ),
    (
        "scorers/ideate.py",
        "coverage の対象に除外された案を含める（屑で被覆を稼げる）",
        '    coverage, missed = _coverage(valid, spec.get("coverage_terms"))',
        '    coverage, missed = _coverage(items, spec.get("coverage_terms"))',
    ),
    (
        "scorers/ideate.py",
        "合成を乗算から平均にする（多様性だけ高い出力が上がる）",
        "        score=valid_rate * coverage * diversity,",
        "        score=(valid_rate + coverage + diversity) / 3,",
    ),
    (
        "embed.py",
        "多様性の 2 件未満の扱いを 1.0 にする",
        "    if len(vectors) < 2:\n        return 0.0",
        "    if len(vectors) < 2:\n        return 1.0",
    ),
    (
        "embed.py",
        "ペアごとのクリップを外す（逆向きのペアが平均を押し上げる）",
        "            total += min(1.0, max(0.0, 1.0 - cosine(left, right)))",
        "            total += 1.0 - cosine(left, right)",
    ),
    (
        "embed.py",
        "cos が単位正規化を前提にする（別の埋め込みモデルで黙って壊れる）",
        "    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))\n    return dot / norm if norm else 0.0",
        "    return dot",
    ),
    (
        "embedcache.py",
        "本数の食い違いを黙って許す（案とベクトルの対応が 1 つずれる）",
        "            if len(vectors) != len(keys_missing):",
        "            if False:",
    ),
    (
        "keys.py",
        "埋め込み指紋から model_digest を外す（差し替え後も旧ベクトルを再利用）",
        '    return _sha256("|".join([model, model_digest, ollama_version]))',
        '    return _sha256("|".join([model, ollama_version]))',
    ),
    (
        "lint.py",
        "score.count と failure.count の一致検査を外す",
        "    elif score_count != failure_count:",
        "    elif False:",
    ),
]

BOUNDARY = [
    (
        "gate.py",
        "カナリアの総合判定を常に合格にする",
        "    return GateReport(all(row[1] for row in rows), tuple(rows))",
        "    return GateReport(True, tuple(rows))",
    ),
    (
        "gate.py",
        "実行器をサンドボックス無しに差し替える",
        "    executor = SandboxExecutor(use_sandbox=True)",
        "    executor = SandboxExecutor(use_sandbox=False)",
    ),
    (
        "sandbox/canary.py",
        "空振り（ENOENT 等）も合格にする",
        "    if num in _BOUNDARY_ERRNOS:",
        "    if True:",
    ),
    (
        "sandbox/executor.py",
        "打ち切りの判定をマーカー判定より後ろに回す",
        "    if returncode in _KILL_RETURNCODES:",
        "    if False:",
    ),
]

boundary_mode = "--boundary" in sys.argv
targets = BOUNDARY if boundary_mode else M
pytest_args = BOUNDARY_ARGS if boundary_mode else FAST_ARGS
print(f"対象: {'境界' if boundary_mode else '通常'}（{len(targets)} 変異）\n")

# **kill されると finally は走らない。** 実際に背景実行を止めたときソースが
# 変異したまま残り、それを観測した。その瞬間に commit していれば、
# 安全策を無効化した状態が履歴に入っていた。番兵で次回に検出する。
SENTINEL = pathlib.Path(__file__).resolve().parent / ".mutate-in-progress"
if SENTINEL.exists():
    print(
        f"前回の実行が途中で止まっている: {SENTINEL.read_text().strip()}\n"
        f"そのファイルを git で戻してから {SENTINEL} を消すこと",
        file=sys.stderr,
    )
    sys.exit(2)

killed = survived = missing = 0
for name, label, old, new in targets:
    path = ROOT / name
    backup = path.read_bytes()
    text = backup.decode()
    if old not in text:
        print(f"  ?? {label:48} 変異対象なし")
        missing += 1
        continue
    # **復元は finally に置く。** 例外や Ctrl-C で抜けると、変異したままの
    # ソースが手元に残る。検証のための道具が壊し得るのは本末転倒なので構造で塞ぐ。
    try:
        SENTINEL.write_text(f"{path}（{label}）\n")
        path.write_bytes(text.replace(old, new, 1).encode())
        rc = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "-q",
                "-x",
                "--no-header",
                "-p",
                "no:cacheprovider",
                *pytest_args,
            ],
            capture_output=True,
            text=True,
        ).returncode
    finally:
        path.write_bytes(backup)
        assert path.read_bytes() == backup, f"{path} の復元に失敗した"
        SENTINEL.unlink(missing_ok=True)
    if rc != 0:
        print(f"  ○ {label:48} 落ちた")
        killed += 1
    else:
        print(f"  ✗ {label:48} 生き残った")
        survived += 1

print(f"\n殺した {killed} / 生き残り {survived} / 対象なし {missing}")
sys.exit(1 if survived or missing else 0)
