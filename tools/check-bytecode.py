"""ソースと .pyc のバイトコードが一致するか（FLB-QB-001 §15.5a）。

**mtime とサイズの比較では足りない。** 事故を起こした .pyc はどちらも一致していた
（だから Python がそれを有効と判断した）。

**`marshal.dumps` の再シリアライズ比較でも足りない。** marshal は参照の共有表を持ち、
**同じ内容の code オブジェクトでも直列化のバイト列が一致しない**。
Python 3.12 では偶然一致したが 3.14 では一致せず、CI だけが落ちた。
`co_filename` も比較に混ざる（import 機構が使ったパスと `compile` に渡すパスは同じとは限らない）。

**コード本体を再帰的に正準化して比べる。** ファイル名は**わざと外す** —
見たいのは「そのバイトコードがそのソースから出たものか」であって、
どのパスで作られたかではない。
"""

import importlib.util
import marshal
import pathlib
import sys
import types


def canon(code: types.CodeType):
    return (
        code.co_name,
        code.co_code,
        code.co_names,
        code.co_varnames,
        code.co_flags,
        code.co_argcount,
        tuple(canon(c) if isinstance(c, types.CodeType) else c for c in code.co_consts),
    )


def main() -> int:
    bad = []
    checked = 0
    for src in sorted(pathlib.Path("src").rglob("*.py")):
        cache = pathlib.Path(importlib.util.cache_from_source(str(src)))
        if not cache.exists():
            continue
        checked += 1
        try:
            cached = marshal.loads(cache.read_bytes()[16:])
            fresh = compile(src.read_bytes(), str(src), "exec", dont_inherit=True)
        except (ValueError, EOFError, SyntaxError) as exc:
            bad.append(f"{src}: 読めない（{exc}）")
            continue
        if canon(cached) != canon(fresh):
            bad.append(f"{src}: バイトコードがソースと違う")

    for line in bad:
        print("  " + line, file=sys.stderr)
    print(f"  pyc を検査: {checked} 件 / 不一致 {len(bad)} 件")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
