import logging

from .base import MlflowBackend, PrometheusBackend, TensorboardBackend, TrackingBackend, TrackingManager, WandbBackend
from .ci_history import RECORD_DIR_ENV, TARGET_METRIC_KEYS, CiHistoryBackend

# The full registry lives here, not base.py: base must never import a backend
# module (ci_history imports TrackingBackend from base -> circular). This
# __init__ is the one place that imports every backend, so it owns the registry.
BACKEND_REGISTRY: dict[str, tuple[type[TrackingBackend], str]] = {
    "wandb": (WandbBackend, "use_wandb"),
    "tensorboard": (TensorboardBackend, "use_tensorboard"),
    "mlflow": (MlflowBackend, "use_mlflow"),
    "prometheus": (PrometheusBackend, "use_prometheus"),
    "ci_history": (CiHistoryBackend, "ci_enable_metrics_capture"),
}

logger = logging.getLogger(__name__)
_manager = TrackingManager(BACKEND_REGISTRY)

__all__ = [
    "BACKEND_REGISTRY",
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
