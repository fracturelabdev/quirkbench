"""ワークフローにステップ名の重複が無いか。

**編集がステップを複製した。** 新しい版を挿入したつもりで古い版が残り、
**両方が回って古い版だけが落ちた**。ログには古い版の中身が出るので、
「直したはずのものが直っていない」という見え方になり、切り分けが遠回りになる。

重複は目で数えないと気づけない。**機械で数える。**
"""

from __future__ import annotations

import collections
import pathlib
import sys

import yaml

WORKFLOWS = pathlib.Path(__file__).resolve().parent.parent / ".github" / "workflows"


def main() -> int:
    problems: list[str] = []
    checked = 0
    for path in sorted(WORKFLOWS.glob("*.yml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in (data.get("jobs") or {}).items():
            names = [s.get("name") for s in (job.get("steps") or []) if s.get("name")]
            checked += len(names)
            for name, count in collections.Counter(names).items():
                if count > 1:
                    problems.append(f"{path.name} の {job_name}: ステップ {name!r} が {count} 回")
    for line in problems:
        print(f"  NG  {line}", file=sys.stderr)
    print(f"  ステップを検査: {checked} 件 / 重複 {len(problems)} 件")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
