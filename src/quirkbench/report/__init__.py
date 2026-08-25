"""レポート — 集計とレンダリングを分ける（FLB-QB-001 §10.1・§14）。

`aggregate.py` が数値だけを持つデータクラスを返し、`render_*.py` が Markdown にする。
混ぜると、**残差の計算をレポート形式ごとに書き直すことになる。**
"""

from .aggregate import Aggregation, InconsistentRun, aggregate

__all__ = ["Aggregation", "InconsistentRun", "aggregate"]
