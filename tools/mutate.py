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

import importlib.util
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
        "lint.py",
        "task の検査そのものを呼ばない",
        "    issues.extend(_lint_tasks(cases))\n",
        "",
    ),
    (
        "lint.py",
        "task が次元をまたいでも通す",
        "        if len(dims) > 1:",
        "        if False:",
    ),
    (
        "lint.py",
        "対訳の task が食い違っても通す",
        "        if other.task != case.task:",
        "        if False:",
    ),
    (
        "report/aggregate.py",
        "ゲートが落とした割合を集めない（削られたことが誰にも見えない）",
        '        rate = (row.get("sub_metrics") or {}).get("offtopic_rate")',
        "        rate = None",
    ),
    (
        "report/aggregate.py",
        "ゲートを持たないケースにも 0 を入れる（測っていないのを 0 と書く）",
        "                gate_drop_by_model={\n                    m: sum(v) / len(v) for m, v in per_gate.get(case_id, {}).items()\n                },",
        "                gate_drop_by_model={m: 0.0 for m in by_model},",
    ),
    (
        "report/render_profile.py",
        "ゲートの表を出さない",
        "    out += _gate_table(agg, names)\n",
        "",
    ),
    (
        "report/render_profile.py",
        "ゲートを持たない run でも空の表を出す（測ったが 0 に読める）",
        "    if not gated:\n        return []",
        "    if False:\n        return []",
    ),
    (
        "report/stability.py",
        "z が 0 のモデルにも「振れる」と言う（振れていないのに）",
        "        if not self.crosses_zero or abs(self.z_full) <= Z_ZERO_EPSILON:",
        "        if not self.crosses_zero:",
    ),
    (
        "report/stability.py",
        "主張の大きさを振れ幅と比べない（0 付近のものまで名指しする）",
        "        return abs(self.z_full) >= self.width / 2",
        "        return True",
    ),
    (
        "report/stability.py",
        "0 判定の許容幅を外す（丸め誤差の符号で判定が変わる）",
        "        return self.z_min <= Z_ZERO_EPSILON and self.z_max >= -Z_ZERO_EPSILON",
        "        return self.z_min <= 0.0 <= self.z_max",
    ),
    (
        "report/stability.py",
        "採点されていないケースも観測に数える（恒等な複製が混ざる）",
        "        in_dim = [c for c in by_dim[dim] if c.id in scored_ids]",
        "        in_dim = list(by_dim[dim])",
    ),
    (
        "report/stability.py",
        "z が出なかった複製を黙って落とす（幅 0.00 = 安定に見せる）",
        "                else:\n                    missing[model] = missing.get(model, 0) + 1\n",
        "",
    ),
    (
        "report/stability.py",
        "複製が揃っていなくても測れたことにする",
        "            and all(m.complete for m in self.models)\n",
        "",
    ),
    (
        "report/stability.py",
        "測れなかった理由を 1 つに潰す（生成が無いのをケース不足と言う）",
        '            return "この run に採点済みの生成が無い"\n',
        "",
    ),
    (
        "report/stability.py",
        "生成の有無を見ずに測れることにする",
        "        observed = dim in observed_dims",
        "        observed = True",
    ),
    (
        "report/stability.py",
        "1 件抜きをやめる（全件のまま再集計する）",
        "            subset = [c for c in cases if not (c.dim == dim and c.task == dropped_task)]",
        "            subset = list(cases)",
    ),
    (
        "report/stability.py",
        "区間が 0 をまたぐ判定を無効化",
        "        return self.z_min <= Z_ZERO_EPSILON and self.z_max >= -Z_ZERO_EPSILON",
        "        return False",
    ),
    (
        "report/stability.py",
        "観測数をケース数で数える（task を無視）",
        "        tasks = sorted({c.task for c in in_dim})",
        "        tasks = [c.id for c in in_dim]",
    ),
    (
        "report/stability.py",
        "測れなかった次元を黙って落とす",
        "            result.skipped.append(dim)\n",
        "",
    ),
    (
        "report/stability.py",
        "観測 1 件でも jackknife を試みる",
        "MIN_TASKS_FOR_JACKKNIFE = 2",
        "MIN_TASKS_FOR_JACKKNIFE = 1",
    ),
    (
        "report/stability.py",
        "ケースの実効幅を最大値そのものにする",
        "                spread=max(values) - min(values),",
        "                spread=max(values),",
    ),
    (
        "report/stability.py",
        "ケース表を幅の降順に並べる（順位表にする）",
        "    return sorted(out, key=lambda c: (c.dim, c.case_id))",
        "    return sorted(out, key=lambda c: -c.spread)",
    ),
    (
        "report/stability.py",
        "全件の z を 0 に固定する",
        "                z_full=full_z.get(dim, {}).get(model, 0.0),",
        "                z_full=0.0,",
    ),
    (
        "report/render_stability.py",
        "0 をまたいだ印を出さない",
        '                        CROSS_MARK if model.crosses_zero else "",',
        '                        "",',
    ),
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
    # ---- S5: レポートと判断の関門（FLB-QB-001 §14）
    (
        "report/aggregate.py",
        "識別力なしのケースも残差に入れる（難易度の相殺が壊れる）",
        "        usable = [c for c in in_dim if c.discriminating]",
        "        usable = list(in_dim)",
    ),
    (
        "report/aggregate.py",
        "σ が 0 でも z を 0 として返す（「測れていない」が「差が無い」に化ける）",
        "                z_by_model=({m: v / sd for m, v in residual_mean.items()} if sd > 0 else {}),",
        "                z_by_model={m: 0.0 for m in residual_mean},",
    ),
    (
        "report/aggregate.py",
        "σ を標本標準偏差にする（n が小さいとき z が縮む）",
        "        sd = statistics.pstdev(flat) if len(flat) > 1 else 0.0",
        "        sd = statistics.stdev(flat) if len(flat) > 1 else 0.0",
    ),
    (
        "report/aggregate.py",
        "識別力の判定を厳密比較にする（浮動小数の最下位ビットで誤判定）",
        "        discriminating = bool(values) and (max(values) - min(values)) >= SAME_SCORE_EPSILON",
        "        discriminating = bool(values) and len(set(values)) > 1",
    ),
    (
        "report/aggregate.py",
        "ceiling と floor の区別をやめる（次の手が同じになる）",
        '    if all(v >= 1.0 - SAME_SCORE_EPSILON for v in values):\n        return "ceiling"\n',
        "",
    ),
    (
        "report/aggregate.py",
        "run 内の digest ずれを見逃す",
        "    if drifted:",
        "    if False:",
    ),
    (
        "report/aggregate.py",
        "ollama のバージョンずれを見逃す",
        "    if len(versions) > 1:",
        "    if False:",
    ),
    (
        "report/aggregate.py",
        "採点行の無い生成を数えない（採点し直し忘れが見えなくなる）",
        "    return best, len(generations) - len(best)",
        "    return best, 0",
    ),
    (
        "report/aggregate.py",
        "読み出しで scorer_version を見ない（古い採点を採りうる）",
        '        rank = (int(row.get("scorer_version") or 0), str(row.get("ts") or ""))',
        '        rank = (0, str(row.get("ts") or ""))',
    ),
    (
        "report/aggregate.py",
        "check_hash の一致を見ない（現在のケース定義と違う採点を採る）",
        '        if case is None or row.get("check_hash") != case.check_hash:',
        "        if case is None:",
    ),
    (
        "report/aggregate.py",
        "tok/s を行ごとの平均にする（短い応答の速さを測る）",
        "                tokens_per_second=eval_count / (eval_ns / 1e9) if eval_ns else 0.0,",
        "                tokens_per_second=statistics.mean(\n"
        "                    [\n"
        '                        int(g.get("eval_count") or 0) / (int(g["eval_duration_ns"]) / 1e9)\n'
        "                        for g in rows\n"
        '                        if g.get("eval_duration_ns")\n'
        "                    ]\n"
        "                )\n"
        "                if eval_ns\n"
        "                else 0.0,",
    ),
    (
        "report/aggregate.py",
        "load を平均にする（最初の 1 件のモデルロードに引きずられる）",
        "                load_ms_median=statistics.median(loads) if loads else 0.0,",
        "                load_ms_median=statistics.mean(loads) if loads else 0.0,",
    ),
    (
        "report/aggregate.py",
        "対訳の相手が run に無くても ja_penalty を出す",
        "        if other is None:\n            continue\n",
        "",
    ),
    (
        "report/render_profile.py",
        "z が 1 ケースで決まっている印を消す",
        '    mark = "" if dim.z_is_trusted else UNTRUSTED_MARK',
        '    mark = ""',
    ),
    (
        "report/render_profile.py",
        "母数 0 を 0/0 として出す（0% と読める）",
        '                cells.append(f"{numer} / {denom}" if denom else NA)',
        '                cells.append(f"{numer} / {denom}")',
    ),
    (
        "report/render_profile.py",
        "候補階層を主表に混ぜる",
        "        tags = [t for t in agg.failure_counts if failures.TIER.get(t) == tier]",
        "        tags = list(agg.failure_counts)",
    ),
    (
        "report/render_compare.py",
        "フェンスの長さを本文に合わせない（生出力のフェンスで表示が壊れる）",
        '    marker = "`" * max(3, longest + 1)',
        '    marker = "```"',
    ),
    (
        "report/render_compare.py",
        "seed で絞らない（比較の条件が揃わない）",
        '        rows = [r for r in rows if r.get("seed") is not None and int(r["seed"]) == seed]',
        "        rows = list(rows)",
    ),
    # ---- S6: 残り 5 次元と、S5 が持ち越した欠陥（FLB-QB-001 §15）
    (
        "scorers/answer.py",
        "numeric で最初の数値を採る（問題文の数値を書くモデルに点が入る）",
        "    extracted = numbers[-1] if numbers else None",
        "    extracted = numbers[0] if numbers else None",
    ),
    (
        "scorers/answer.py",
        "numeric の許容差を無視する",
        "    hit = extracted is not None and abs(extracted - expect) <= tolerance",
        "    hit = extracted is not None and extracted == expect",
    ),
    (
        "scorers/answer.py",
        "exact の contains を equals と同じにする（長文脈が出力の素っ気なさを測る）",
        '    hit = contains if mode == "contains" else equals',
        "    hit = equals",
    ),
    (
        "scorers/answer.py",
        "exact で正規化を掛けない（全角の揺れで不正解になる）",
        '    return unicodedata.normalize("NFKC", text).strip()',
        "    return text.strip()",
    ),
    (
        "scorers/answer.py",
        "bare_answer をスコアに入れる（reason が指示追従を測り始める）",
        '    return ScoreResult(\n        score=1.0 if hit else 0.0,\n        sub_metrics={\n            "extracted": extracted,',
        '    return ScoreResult(\n        score=1.0 if (hit and len(numbers) == 1) else 0.0,\n        sub_metrics={\n            "extracted": extracted,',
    ),
    (
        "scorers/extract.py",
        "json_keys で余分なキーを減点する（instruct と同じものを測り始める）",
        "        score=len(matched) / len(expect) if expect else 1.0,",
        "        score=(len(matched) / len(expect) if expect else 1.0)\n"
        "        * (1.0 if len(value) == len(expect) else 0.0),",
    ),
    (
        "scorers/extract.py",
        "json_keys が値の一致を見ない（キーがあるだけで正解になる）",
        "        if key in value and _equal(value[key], wanted):",
        "        if key in value:",
    ),
    (
        "lint.py",
        "contains の 4 文字下限を外す（短い答えが偶然一致する）",
        "            short = [str(v) for v in values if len(str(v).strip()) < MIN_CONTAINS_LENGTH]",
        "            short = []",
    ),
    (
        "lint.py",
        "プロンプトが num_ctx に収まるかの検査を外す（黙って切り詰められる）",
        "    if estimated <= budget:\n        return []",
        "    return []",
    ),
    (
        "lint.py",
        "長さの検査を longctx だけに掛ける（他の次元で伸ばすと切り詰められる）",
        "        issues.extend(_lint_context(case))",
        '        if case.dim == "longctx":\n            issues.extend(_lint_context(case))',
    ),
    (
        "runner.py",
        "プロンプトの切り詰めを実行時に見逃す",
        '        if error is None and used is not None and int(used) >= int(options["num_ctx"]):',
        "        if False:",
    ),
    (
        "report/aggregate.py",
        "ja_penalty の符号を逆にする（penalty が負で罰になる）",
        "                acc.setdefault(model, []).append(other.by_model[model] - value)",
        "                acc.setdefault(model, []).append(value - other.by_model[model])",
    ),
    (
        "scorers/ideate.py",
        "被覆の照合で大文字小文字を畳まない（英語で coverage が 0 に固定される）",
        "    return normalize(text).casefold()",
        "    return normalize(text)",
    ),
    (
        "report/render_profile.py",
        "モデル名の短縮で衝突を見ない（別ファミリの同サイズが同じ名前で並ぶ）",
        "    if len(set(short.values())) != len(models):\n        return {m: m for m in models}",
        "",
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


def write_source(path: pathlib.Path, data: bytes) -> None:
    """ソースを書き、**そのファイルのバイトコードキャッシュを消す**。

    **これが無いと、変異が `.pyc` に残ったまま検査が続く。**

    Python は `.pyc` の有効性を「ソースの mtime（**秒**）とサイズ」で判定する。
    この道具は同じファイルを 1 秒以内に何度も書き換えるので、
    **バイト数の変わらない変異**（`a - b` を `b - a` にする類）を書いて戻すと、
    mtime もサイズも変異前と一致し、**変異したバイトコードが有効なまま残る**。

    実際に S6 で踏んだ。`ja_penalty` の符号を直したあとに変異検査を回したところ、
    ソースは正しいのにテストだけが失敗し続けた。`inspect.getsource` は
    ファイルを読むので正しく見え、**実行されているのは古いバイトコード**だった。

    影響は「実行後に手元が壊れる」だけではない。**検査の最中に、前の変異の
    バイトコードで次の変異を判定しうる** — 撃墜・生き残りの判定そのものが狂う。
    """
    path.write_bytes(data)
    cache = pathlib.Path(importlib.util.cache_from_source(str(path)))
    cache.unlink(missing_ok=True)


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
        write_source(path, text.replace(old, new, 1).encode())
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
        write_source(path, backup)
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
