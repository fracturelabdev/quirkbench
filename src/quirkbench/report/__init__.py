"""レポート — 集計とレンダリングを分ける（FLB-QB-001 §10.1・§14）。

`aggregate.py` が数値だけを持つデータクラスを返し、`render_*.py` が Markdown にする。
混ぜると、**残差の計算をレポート形式ごとに書き直すことになる。**

**`aggregate` 関数をここから再輸出しない。** サブモジュール名と同じなので、
``import quirkbench.report.aggregate`` が**モジュールではなく関数**を返す。
実際 S6 のデバッグ中に踏み、原因の切り分けが 1 段遠回りになった。
呼び出し側は ``from .report.aggregate import aggregate`` と書く。
"""

from .aggregate import Aggregation, InconsistentRun

__all__ = ["Aggregation", "InconsistentRun"]
