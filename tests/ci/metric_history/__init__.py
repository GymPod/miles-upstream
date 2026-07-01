"""Metric-history subsystem for the CI regression gate.

Public surface: the :class:`MetricHistoryStore` contract and its backends
(:class:`SQLiteMetricHistoryStore`, :class:`NeonMetricHistoryStore`), plus the
gate declaration marker :func:`register_ci_gate` and its parsed :class:`CiGateSpec`.
Gate evaluation lives in :mod:`tests.ci.metric_history.gate`; series reduction in
:mod:`tests.ci.metric_history.reducers`.
"""

from tests.ci.metric_history.register import CiGateSpec, register_ci_gate
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
    "CiGateSpec",
    "register_ci_gate",
]
