"""比較ビュー — 同一プロンプトに対する各モデルの生出力を並べる（FLB-QB-001 §14.10）。

**これが最終的な質の判断の場所である。** 機械が付けた失敗型は候補にすぎず、
`ideate` の「発想の質」に至っては機械では測っていない（§5）。
生出力を並べて人が読めるようにすることが、このツールの前提条件になっている。

**選択が結論を作らないようにする。** スコアが極端なものを選んで並べると、
並べた時点で結論が決まる。**最小 seed の 1 本**をケース × モデルで機械的に出す。
"""

from __future__ import annotations

from ..cases import Case
from ..store import RunStore

#: 1 応答あたりの上限。`longctx` は数十 KB になりうるので、比較ビューが読めなくなる
MAX_CHARS = 4000


def _fence(text: str) -> str:
    """本文をフェンスで囲む。**本文中のフェンスと衝突しない長さ**にする。

    生出力にはコードフェンスが普通に含まれる。``` で囲むと途中で閉じ、
    **以降の表示が壊れたまま「モデルの出力が壊れている」ように見える。**
    """
    longest = 0
    run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    marker = "`" * max(3, longest + 1)
    return f"{marker}\n{text}\n{marker}"


def render(store: RunStore, cases: list[Case], *, seed: int | None, all_seeds: bool = False) -> str:
    by_id = {case.id: case for case in cases}
    rows, _ = store.generations()
    rows = [r for r in rows if not r.get("error") and str(r.get("case_id")) in by_id]
    if not all_seeds and seed is not None:
        # **`r.get("seed") or -1` と書かない。** seed 0 は偽値なので -1 に化け、
        # **seed 0 の run だけ比較ビューが空になる**（既定の base-seed が 1000 なので
        # 実データでは起きず、テストでしか見つからない類の欠陥）
        rows = [r for r in rows if r.get("seed") is not None and int(r["seed"]) == seed]

    out: list[str] = [
        f"# 比較ビュー — run `{store.run_id}`",
        "",
        "**同一プロンプト・同一 seed に対する各モデルの生出力。**"
        + ("（全 seed）" if all_seeds else f"（seed = {seed}）"),
        "",
        "機械が付けた失敗型は候補にすぎない。**ここを読んで人が判断する。**",
        "",
    ]
    for case_id in sorted({str(r.get("case_id")) for r in rows}):
        case = by_id[case_id]
        out += [
            f"## `{case_id}`",
            "",
            f"- 次元 `{case.dim}` / 言語 `{case.lang}`"
            + (f" / 対訳 `{case.pair}`" if case.pair else ""),
            "",
            "**プロンプト**",
            "",
            _fence(case.prompt.strip()),
            "",
        ]
        subset = [r for r in rows if str(r.get("case_id")) == case_id]
        for row in sorted(subset, key=lambda r: (str(r.get("model")), int(r.get("seed", 0) or 0))):
            body = str(row.get("response") or "")
            clipped = len(body) > MAX_CHARS
            head = f"### `{row.get('model')}`"
            if all_seeds:
                head += f" — seed {row.get('seed')}"
            out += [
                head,
                "",
                f"`done_reason={row.get('done_reason')}` / "
                f"`eval_count={row.get('eval_count')}` / {row.get('wall_seconds')} 秒"
                + ("（**以下は先頭のみ**）" if clipped else ""),
                "",
                _fence(body[:MAX_CHARS].strip() or "(空)"),
                "",
            ]
    return "\n".join(out) + "\n"


def pick_seed(store: RunStore) -> int | None:
    """比較に使う seed。**最小の seed**（§14.10）。"""
    rows, _ = store.generations()
    seeds = [int(s) for r in rows if not r.get("error") and (s := r.get("seed")) is not None]
    return min(seeds) if seeds else None
