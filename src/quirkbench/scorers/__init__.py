"""次元固有の採点。**`Parsed` しか見ない**（生テキストに触れない）。

scorer は失敗型の語彙を持たない。`failures.py` の語彙のタグを append するだけ。

``SCORER_VERSION`` は採点ロジックを変えたら上げる。再採点は `scores.jsonl` への
追記だけで行い、レポートは gen_id ごとに version 最大の行を採るので、
**過去の採点も残り「scorer を直したら結果がどう動いたか」がそのまま追える**。
これは癖のプロファイリングという目的そのものに効く（FLB-QB-001 §10.2）。

**dispatch は `Protocol` で型づける**（§12.10）。手書きの ``Callable`` 注釈だと、
mypy が照合するのは*注釈と実シグネチャの整合*だけで、**注釈を緩めれば静かに検査が消える**。
``executor`` をキーワード必須で持つ ``Scorer`` にして初めて
「``executor`` を受け取らない関数は registry に入らない」が型で言える。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..cases import Case
from ..parse import Parsed

SCORER_VERSION = 2


@dataclass(frozen=True)
class ScoreResult:
    """採点 1 件。``score`` は 0.0〜1.0。"""

    score: float
    sub_metrics: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    applicable: tuple[str, ...] = ()


class Executor(Protocol):
    """生成コードを隔離して実行するもの（``sandbox.SandboxExecutor`` か、テストの stub）。

    **採点器はサンドボックスを直接 import しない。** ここを必須引数で受け取ることで、
    「境界を通さずに実行する」経路を型で作れなくする。
    """

    @property
    def sandbox_applied(self) -> bool: ...

    def run(
        self,
        *,
        payload: str,
        check_source: str,
        entry_point: str,
        timeout_seconds: int,
        check_hash: str = "",
    ) -> Any:
        """``check_hash`` は**キャッシュ層だけが読む**。実行そのものには使わない。

        ``**kwargs`` の逃げ道を作らない。作ると「何を渡してもよい」ことになり、
        引数を 1 つ足したときに受け取り側の実装漏れを型が捕まえられなくなる —
        それは ``Callable`` に戻したのと同じ（§12.10）。
        """
        ...


class Scorer(Protocol):
    """採点器のシグネチャ。``executor`` は**キーワード必須**で、既定値を持たない。"""

    def __call__(self, parsed: Parsed, case: Case, *, executor: Executor | None) -> ScoreResult: ...


class ScorerNotImplemented(NotImplementedError):
    """その ``score.kind`` の採点器がまだ無い。段階的に実装するため例外にする。"""


class ExecutorRequired(RuntimeError):
    """実行採点なのに実行器が渡されていない。

    **既定値で直接実行に落とさない。** ``executor=None → 直接実行`` のフォールバックは、
    境界を消す最短経路になる（§12.10）。
    """


def _registry() -> dict[str, Scorer]:
    from .code_gen import score_pytest
    from .instruct import score_json_schema

    return {"json_schema": score_json_schema, "pytest": score_pytest}


def score(parsed: Parsed, case: Case, *, executor: Executor | None = None) -> ScoreResult:
    kind = case.score["kind"]
    scorer = _registry().get(kind)
    if scorer is None:
        raise ScorerNotImplemented(kind)
    return scorer(parsed, case, executor=executor)
