import logging

from .base import BACKEND_REGISTRY, TrackingManager
from .ci_history import RECORD_DIR_ENV, TARGET_METRIC_KEYS, CiHistoryBackend

# Registered here, not in base.py: base must never import a backend module, or
# it is a circular import (ci_history imports TrackingBackend from base). This
# package's __init__ is the entry point and already imports CiHistoryBackend, so
# every consumer that reaches TrackingManager through the package sees it.
BACKEND_REGISTRY["ci_history"] = (CiHistoryBackend, "ci_enable_metrics_capture")

logger = logging.getLogger(__name__)
_manager = TrackingManager()

__all__ = [
    "CiHistoryBackend",
    "RECORD_DIR_ENV",
    "TARGET_METRIC_KEYS",
    "finish_tracking",
    "init_tracking",
    "log",
]


def init_tracking(args, primary: bool = True, **kwargs):
    _manager.init(args, primary=primary, **kwargs)


def log(args, metrics, step_key: str):
    step = metrics.get(step_key)
    _manager.log(metrics, step=step, step_key=step_key)


def finish_tracking():
    _manager.finish()
