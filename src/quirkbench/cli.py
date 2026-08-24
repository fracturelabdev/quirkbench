"""コマンドラインの入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .cases import CaseError, load_cases
from .ollama import DEFAULT_HOST, Ollama, OllamaError
from .runner import DigestDrift, run
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
    with store:
        summary = score_run(store, cases)
    print(
        f"生成 {summary.total} / 採点 {summary.scored} / "
        f"採点済みスキップ {summary.skipped} / 生成失敗 {summary.generation_errors}"
    )
    # 黙って落とした件数は必ず出す。0 件と「対象外だった」は違う
    for kind, count in sorted(summary.unsupported.items()):
        print(f"  未実装の score.kind {kind!r}: {count} 件を採点していない")
    for case_id, count in sorted(summary.missing_cases.items()):
        print(f"  ケース定義が見つからない {case_id!r}: {count} 件を採点していない")
    return 0


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
        if args.command == "status":
            return _cmd_status(args)
    except (CaseError, OllamaError, RunLocked, DigestDrift) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n中断した。同じ --run 名で再実行すると続きから再開する", file=sys.stderr)
        return 130
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
