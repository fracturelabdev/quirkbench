"""run ディレクトリへの永続化。

**append-only。** 数時間かけた生成を書き換えることは絶対にしない。
再採点は ``scores.jsonl`` への追記だけで行い、``generations.jsonl`` は不変に保つ。

壊れた行の扱いを 2 つに分けているのが要点:

- **末尾の壊れた行** … kill された瞬間の書きかけ。黙って捨ててよい（キーが復元できないので
  該当の生成は単に再実行される）
- **途中の壊れた行** … ディスク障害か二重起動の兆候。**黙って捨ててはいけない**ので警告して数える
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import keys as _keys

GENERATIONS = "generations.jsonl"
PROMPTS = "prompts.jsonl"
SCORES = "scores.jsonl"
EXEC_CACHE = "exec_cache.jsonl"
#: ``--unsafe-no-sandbox`` の結果は**物理的に分ける**（FLB-QB-001 §12.9）。
#: キーに ``sandbox_applied`` を入れるだけでなくファイルを分けて、混ざりようがなくする。
EXEC_CACHE_UNSAFE = "exec_cache_unsafe.jsonl"
#: 埋め込みのキャッシュ（FLB-QB-001 §13.5）。実行キャッシュと違い
#: ``--unsafe-no-sandbox`` で分ける必要がない — 隔離の有無は埋め込みの値に関係しない。
EMBEDDINGS = "embeddings.jsonl"
META = "meta.json"
LOCK = ".lock"

# flock が信頼できない場所。ここに run ディレクトリを置かせない
_UNRELIABLE_FS_MARKERS = ("/Library/Mobile Documents/", "/Volumes/")


class StoreError(RuntimeError):
    pass


class RunLocked(StoreError):
    """同じ run ディレクトリを別プロセスが掴んでいる。"""


@dataclass
class ReadReport:
    """JSONL を読んだときの健全性。"""

    total: int = 0
    corrupt_tail: bool = False
    corrupt_middle: int = 0
    missing_fields: int = 0

    @property
    def has_warning(self) -> bool:
        return self.corrupt_middle > 0 or self.missing_fields > 0


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], ReadReport]:
    report = ReadReport()
    if not path.exists():
        return [], report

    raw_lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()  # 末尾の改行による空要素

    rows: list[dict[str, Any]] = []
    last_index = len(raw_lines) - 1
    for index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        report.total += 1
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == last_index:
                report.corrupt_tail = True
            else:
                report.corrupt_middle += 1
    return rows, report


class RunStore:
    """1 つの run ディレクトリ。``with`` で使うとロックを取る。"""

    def __init__(self, root: Path, run_id: str) -> None:
        self.run_id = run_id
        self.dir = root / run_id
        self._lock_fd: int | None = None
        self._seen_prompts: set[str] | None = None

    # -------------------------------------------------------------- ロック

    def warn_unreliable_fs(self) -> str | None:
        resolved = str(self.dir.resolve())
        for marker in _UNRELIABLE_FS_MARKERS:
            if marker in resolved:
                return (
                    f"{resolved} はネットワーク/同期ファイルシステム上にある。"
                    "flock による二重起動の防止が効かない可能性がある"
                )
        return None

    def __enter__(self) -> RunStore:
        self.dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.dir / LOCK
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = lock_path.read_text(encoding="utf-8", errors="replace").strip()
            os.close(fd)
            raise RunLocked(f"run {self.run_id!r} は既に実行中。{holder or '(情報なし)'}") from None
        # PID ファイルと違い flock はプロセス死亡時に自動解放されるが、
        # 誰が掴んでいるかを人が読めるように中身も書いておく
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} host={os.uname().nodename}\n".encode())
        os.fsync(fd)
        self._lock_fd = fd
        return self

    def __exit__(self, *exc: object) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    # ---------------------------------------------------------------- 書き

    def _append(self, name: str, row: dict[str, Any]) -> None:
        """1 行を単一 write + fsync で追記する。

        3,200 行程度なら fsync のコストは無視できる。ここでケチると
        「電源断で数時間分が消える」ほうの損失が大きい。
        """
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with open(self.dir / name, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def append_generation(self, row: dict[str, Any]) -> None:
        self._append(GENERATIONS, row)

    def append_score(self, row: dict[str, Any]) -> None:
        self._append(SCORES, row)

    def exec_cache(self, *, sandbox_applied: bool = True) -> dict[str, dict[str, Any]]:
        """``exec_key`` → 実行結果。**先勝ち**（append-only なので最初の行）。

        先勝ちが安全なのは「同一 ``exec_key`` に後から違う判定が来ない」が成り立つとき
        だけで、それを成り立たせるのが §12.9 の確認プロトコル
        （**確認が終わるまでキャッシュに書かない**）。
        """
        name = EXEC_CACHE if sandbox_applied else EXEC_CACHE_UNSAFE
        rows, _report = read_jsonl(self.dir / name)
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = str(row.get("exec_key", ""))
            if key and key not in out:
                out[key] = row
        return out

    def append_exec_cache(self, row: dict[str, Any], *, sandbox_applied: bool = True) -> None:
        """**確認プロトコルが終わったあとの 1 回だけ**呼ぶ（§12.9）。

        1 回目の ``exec_timeout`` を書いてはいけない。書くと先勝ちが timeout を掴み、
        score は 2 回目の pass を書くので、**同じコードの別 gen_id が timeout を見る**。
        """
        self._append(EXEC_CACHE if sandbox_applied else EXEC_CACHE_UNSAFE, row)

    def embeddings(self) -> dict[str, list[float]]:
        """``embed_key`` → ベクトル。**先勝ち**（append-only なので最初の行）。

        実行キャッシュと違い、先勝ちは「唯一の判定を確定させる」ためではない。
        埋め込みは実測で決定的（§13.4）なので、**同じキーの行は同じ値**である。
        先勝ちにしているのは、破損行や旧仕様の行が混ざったときに
        **どちらを採ったかが読む順で決まらない**ようにするため。
        """
        rows, _report = read_jsonl(self.dir / EMBEDDINGS)
        out: dict[str, list[float]] = {}
        for row in rows:
            key = str(row.get("embed_key", ""))
            vector = row.get("vector")
            if key and key not in out and isinstance(vector, list):
                out[key] = [float(value) for value in vector]
        return out

    def append_embedding(self, row: dict[str, Any]) -> None:
        self._append(EMBEDDINGS, row)

    def append_prompt(self, prompt_hash: str, text: str) -> None:
        """プロンプト本文を重複排除して保存する。

        longctx は 1 プロンプトが数十 KB になり、同じ本文が seed x モデルで何度も繰り返される。
        generations.jsonl に直接埋めると桁違いに膨らむ。
        """
        if self._seen_prompts is None:
            rows, _ = read_jsonl(self.dir / PROMPTS)
            self._seen_prompts = {str(r.get("prompt_hash")) for r in rows}
        if prompt_hash in self._seen_prompts:
            return
        self._append(PROMPTS, {"prompt_hash": prompt_hash, "text": text})
        self._seen_prompts.add(prompt_hash)

    def write_meta(self, meta: dict[str, Any]) -> None:
        path = self.dir / META
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def read_meta(self) -> dict[str, Any]:
        path = self.dir / META
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- 読み

    def generations(self) -> tuple[list[dict[str, Any]], ReadReport]:
        return read_jsonl(self.dir / GENERATIONS)

    def scores(self) -> tuple[list[dict[str, Any]], ReadReport]:
        return read_jsonl(self.dir / SCORES)

    def completed_keys(self) -> tuple[set[tuple], ReadReport]:
        """既に生成済みのキー集合。これに一致するものは再実行しない。

        ``check_hash`` はキーに含まれない（採点手続きを直しても再生成しないため）。
        """
        rows, report = self.generations()
        # **キーの組み立てを手で書かない。** keys.KeyParts に要素が 1 つ増えたとき、
        # 手書きのタプルは静かに一致しなくなり、resume が全件ミスして数時間ぶんを
        # 再生成する。自己整合しているのでテストも通ってしまう。
        keys_seen: set[_keys.CompletionKey] = set()
        for row in rows:
            if row.get("error"):
                continue
            parts = _keys.key_parts_from_row(row)
            if parts is None:
                # フィールドの欠けた行。中間の破損行と同じく黙って捨てずに数える
                report.missing_fields += 1
                continue
            keys_seen.add(_keys.completion_key(parts))
        return keys_seen, report
