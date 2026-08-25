"""コマンドラインの入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .cases import CaseError, load_cases
from .embed import DEFAULT_EMBED_MODEL, OllamaEmbedder
from .gate import (
    SandboxUnavailable,
    profile_sha256,
    select_executor,
    verify_positive_controls,
)
from .keys import runner_fingerprint
from .lint import lint
from .ollama import DEFAULT_HOST, Ollama, OllamaError
from .report import InconsistentRun, aggregate, render_compare, render_profile
from .runner import DigestDrift, run
from .sandbox import package_sha256
from .scoring import score_run
from .store import RunLocked, RunStore


def _split(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qb", description="ローカル LLM の癖を測る")
    parser.add_argument("--version", action="version", version=f"quirkbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser(
        "run",
        help="生成を実行する（同じ --run 名で再実行すると続きから再開する）",
    )
    run_cmd.add_argument("--models", required=True, help="カンマ区切りのモデル名")
    run_cmd.add_argument("--run", default="main", help="run 名。同名なら resume する（既定: main）")
    run_cmd.add_argument("--cases", type=Path, default=Path("cases"), help="ケースのルート")
    run_cmd.add_argument("--runs", type=Path, default=Path("runs"), help="出力のルート")
    run_cmd.add_argument("--repeats", type=int, default=5, help="1 ケースあたりの反復数")
    run_cmd.add_argument("--base-seed", type=int, default=1000)
    run_cmd.add_argument("--dims", help="次元での絞り込み（カンマ区切り）")
    run_cmd.add_argument("--ids", help="ケース ID での絞り込み（カンマ区切り）")
    run_cmd.add_argument("--host", default=DEFAULT_HOST)
    run_cmd.add_argument(
        "--allow-digest-drift",
        action="store_true",
        help="run の途中でモデルの中身が変わっても続行する",
    )

    score_cmd = sub.add_parser(
        "score",
        help="生成済みの応答を採点する（追記のみ。生成は書き換えない）",
    )
    score_cmd.add_argument("--run", default="main")
    score_cmd.add_argument("--cases", type=Path, default=Path("cases"))
    score_cmd.add_argument("--runs", type=Path, default=Path("runs"))
    score_cmd.add_argument("--dims", help="次元での絞り込み（カンマ区切り）")
    score_cmd.add_argument("--ids", help="ケース ID での絞り込み（カンマ区切り）")
    score_cmd.add_argument("--host", default=DEFAULT_HOST, help="埋め込みに使う ollama")
    score_cmd.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"代理指標に使う埋め込みモデル（既定: {DEFAULT_EMBED_MODEL}）",
    )
    score_cmd.add_argument(
        "--unsafe-no-sandbox",
        action="store_true",
        help="隔離なしで生成コードを実行する（darwin 限定・結果は別ファイルに書く）",
    )

    lint_cmd = sub.add_parser(
        "lint-cases",
        help="ケース定義のゲート。1 件でも落ちたら停止する",
    )
    lint_cmd.add_argument("--cases", type=Path, default=Path("cases"))

    report_cmd = sub.add_parser(
        "report",
        help="プロファイル表と比較ビューを出す（1 つの run だけを読む）",
    )
    report_cmd.add_argument("--run", default="main")
    report_cmd.add_argument("--cases", type=Path, default=Path("cases"))
    report_cmd.add_argument("--runs", type=Path, default=Path("runs"))
    report_cmd.add_argument("--out", type=Path, default=Path("reports"))
    report_cmd.add_argument(
        "--all-seeds",
        action="store_true",
        help="比較ビューに全 seed を出す（既定は最小 seed の 1 本）",
    )

    status = sub.add_parser("status", help="run の進捗を表示する")
    status.add_argument("--run", default="main")
    status.add_argument("--runs", type=Path, default=Path("runs"))
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases, dims=_split(args.dims), ids=_split(args.ids))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    client = Ollama(args.host)

    with RunStore(args.runs, args.run) as store:
        progress = run(
            store=store,
            client=client,
            cases=cases,
            models=models,
            repeats=args.repeats,
            base_seed=args.base_seed,
            allow_digest_drift=args.allow_digest_drift,
        )
    print(
        f"\n合計 {progress.total} / 生成 {progress.generated} / "
        f"再開でスキップ {progress.skipped} / 失敗 {progress.failed}"
    )
    return 1 if progress.failed else 0


def _cmd_score(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases, dims=_split(args.dims), ids=_split(args.ids))
    store = RunStore(args.runs, args.run)
    if not store.dir.exists():
        print(f"run {args.run!r} は存在しない", file=sys.stderr)
        return 2
    # **ケースのゲートを先に通す。** 粒度が崩れたケースで採点すると、
    # 数字が出たあとで気づくことになる（§12.2）。
    issues = lint(cases)
    if issues:
        print("ケース定義が規約を満たしていない:", file=sys.stderr)
        for issue in issues:
            print(f"  {issue.case_id}: [{issue.check}] {issue.message}", file=sys.stderr)
        return 2

    executor = None
    fingerprint = ""
    canary_verdict = "not_run"
    if any(case.score.get("kind") == "pytest" for case in cases):
        # **ゲートは実行器を選ぶ層に置く**（§12.10）。採点器側に置くとテスト用の
        # 迂回が要り、その迂回が禁じたスイッチそのものになる。
        try:
            executor = select_executor(unsafe_no_sandbox=args.unsafe_no_sandbox)
        except SandboxUnavailable as exc:
            print(f"実行採点を拒否した: {exc}", file=sys.stderr)
            return 2
        canary_verdict = "skipped(unsafe)" if args.unsafe_no_sandbox else "pass"
        # 陽性対照。**キャッシュを通さない生の実行器で回す**（§12.8）。
        broken = verify_positive_controls(executor, cases)
        if broken:
            print("陽性対照が落ちた。採点を拒否する:", file=sys.stderr)
            for case_id, detail in broken:
                print(f"  {case_id}: {detail}", file=sys.stderr)
            return 2
        fingerprint = runner_fingerprint(
            sandbox_pkg_sha256=package_sha256(),
            profile_sb_sha256=profile_sha256(),
            sandbox_applied=executor.sandbox_applied,
        )

    # 代理指標の採点には ollama が要る。**既定でローカル計算に落とさない**（§13.6）ので、
    # ideate ケースがあるときだけ構築し、届かなければ OllamaError で採点ごと止まる。
    embedder = None
    if any(case.score.get("kind") == "ideate" for case in cases):
        embedder = OllamaEmbedder(Ollama(args.host), args.embed_model)

    with store:
        summary = score_run(
            store, cases, executor=executor, embedder=embedder, fingerprint=fingerprint
        )
        # meta は 1 回だけ読んで 1 回だけ書く。2 回に分けると、後の書き込みが
        # 先の書き込みを読まないまま上書きする形になる
        meta = store.read_meta()
        if executor is not None:
            meta["canary_verdict"] = canary_verdict
            meta["sandbox_applied"] = executor.sandbox_applied
            meta["runner_fingerprint"] = fingerprint
        if embedder is not None:
            meta["embed_model"] = embedder.model
            meta["embed_model_digest"] = embedder.model_digest
            meta["embedder_fingerprint"] = embedder.fingerprint
        if executor is not None or embedder is not None:
            store.write_meta(meta)
    print(
        f"生成 {summary.total} / 採点 {summary.scored} / "
        f"採点済みスキップ {summary.skipped} / 生成失敗 {summary.generation_errors}"
    )
    if executor is not None:
        print(
            f"サンドボックス起動 {summary.exec_runs} 回 / キャッシュヒット {summary.exec_hits} 回"
        )
    if embedder is not None:
        print(f"埋め込み計算 {summary.embed_runs} 本 / キャッシュヒット {summary.embed_hits} 本")
    # 黙って落とした件数は必ず出す。0 件と「対象外だった」は違う
    for kind, count in sorted(summary.unsupported.items()):
        print(f"  未実装の score.kind {kind!r}: {count} 件を採点していない")
    for case_id, count in sorted(summary.missing_cases.items()):
        print(f"  ケース定義が見つからない {case_id!r}: {count} 件を採点していない")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """**1 つの run だけを読む**（§14.1）。複数 run の結合はしない。

    残差はケース内でモデル間の平均を引くので、run をまたぐと
    **引き算の相手が別の測定条件で測られたもの**になる。
    """
    cases = load_cases(args.cases)
    store = RunStore(args.runs, args.run)
    if not store.dir.exists():
        print(f"run {args.run!r} は存在しない", file=sys.stderr)
        return 2

    agg = aggregate(store, cases)
    out_dir = args.out / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(render_profile.render(agg), encoding="utf-8")
    seed = render_compare.pick_seed(store)
    (out_dir / "compare.md").write_text(
        render_compare.render(store, cases, seed=seed, all_seeds=args.all_seeds),
        encoding="utf-8",
    )

    print(f"{out_dir / 'report.md'}")
    print(f"{out_dir / 'compare.md'}")
    excluded = [c for c in agg.cases if not c.discriminating]
    print(
        f"モデル {len(agg.models)} / ケース {len(agg.cases)}"
        f"（識別力なし {len(excluded)}）/ 生成 {agg.total_generations}"
    )
    # 黙って落とした件数は必ず出す（§14.2）
    if agg.unscored:
        print(f"  警告: 採点行が見つからない生成が {agg.unscored} 件。qb score を回したか")
    for stat in excluded:
        print(f"  識別力なし: {stat.case_id}（{stat.degenerate_kind}）")
    for dim in agg.dims:
        if not dim.measurable:
            print(f"  {dim.dim}: 識別力のあるケースが 0 件なので z を出していない")
        elif not dim.z_is_trusted:
            print(f"  {dim.dim}: 識別力のあるケースが {dim.discriminating_cases} 件。z は要注意")
    return 0


def _cmd_lint(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    issues = lint(cases)
    for issue in issues:
        print(f"{issue.case_id}: [{issue.check}] {issue.message}", file=sys.stderr)
    print(f"ケース {len(cases)} 件 / 指摘 {len(issues)} 件")
    return 1 if issues else 0


def _cmd_status(args: argparse.Namespace) -> int:
    store = RunStore(args.runs, args.run)
    if not store.dir.exists():
        print(f"run {args.run!r} は存在しない")
        return 1
    rows, report = store.generations()
    _, key_report = store.completed_keys()
    ok = [r for r in rows if not r.get("error")]
    print(f"run       : {args.run}")
    print(f"行数      : {report.total}（有効 {len(ok)} / 失敗 {len(rows) - len(ok)}）")
    if report.corrupt_tail:
        print("末尾      : 書きかけの行あり（次回に再実行される）")
    if report.has_warning:
        print(f"警告      : 途中に壊れた行が {report.corrupt_middle} 件")
    if key_report.missing_fields:
        # 黙って再生成されるだけだと、壊れた行が何度でも湧いていることに気づけない
        print(f"警告      : 完了キーの項目が欠けた行が {key_report.missing_fields} 件")
    for model in sorted({str(r.get("model")) for r in ok}):
        subset = [r for r in ok if r.get("model") == model]
        print(f"  {model:20} {len(subset):5} 件")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "score":
            return _cmd_score(args)
        if args.command == "report":
            return _cmd_report(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "lint-cases":
            return _cmd_lint(args)
    except (CaseError, OllamaError, RunLocked, DigestDrift, InconsistentRun) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n中断した。同じ --run 名で再実行すると続きから再開する", file=sys.stderr)
        return 130
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
