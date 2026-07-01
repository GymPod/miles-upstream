"""Storage layer for the CI metric-history regression gate.

The :class:`MetricHistoryStore` contract, its record types, and the two backends:
offline :class:`SQLiteMetricHistoryStore` and hosted :class:`NeonMetricHistoryStore`.
"""

from tests.ci.metric_history.storage.neon_store import NEON_DATABASE_URL_ENV, NeonMetricHistoryStore
from tests.ci.metric_history.storage.sqlite_store import SQLiteMetricHistoryStore
from tests.ci.metric_history.storage.store import MetricHistoryStore, MetricSample, RunIdentity, RunProvenance

__all__ = [
    "MetricHistoryStore",
    "MetricSample",
    "RunIdentity",
    "RunProvenance",
    "SQLiteMetricHistoryStore",
    "NeonMetricHistoryStore",
    "NEON_DATABASE_URL_ENV",
]
