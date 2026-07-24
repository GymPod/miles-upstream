from argparse import Namespace
from unittest.mock import MagicMock, call

import pytest
import requests
import sglang_router
from tests.ci.ci_register import register_cpu_ci

import miles.backends.sglang_utils.sglang_engine as sglang_engine_module
from miles.backends.sglang_utils.router_worker_client import (
    RouterWorkerRegistration,
    RouterWorkerRegistrationFailed,
    RouterWorkerSubmissionRejected,
    RouterWorkerSubmissionUnknown,
    RouterWorkerType,
)
from miles.backends.sglang_utils.sglang_engine import SGLangEngine

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

_WORKER_ID = "12345678-1234-5678-1234-567812345678"
_WORKER_URL = "http://127.0.0.1:30000"


def _engine() -> SGLangEngine:
    engine = SGLangEngine.__new__(SGLangEngine)
    engine.args = Namespace(rollout_external=False, use_miles_router=False)
    engine.node_rank = 0
    engine.router_ip = "127.0.0.1"
    engine.router_port = 3000
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000
    engine.worker_type = "regular"
    engine.disaggregation_bootstrap_port = None
    engine._is_registered_with_router = False
    engine._router_worker_id = None
    engine._router_registration_submission_unknown = False
    return engine


def test_registration_timeout_reconciles_same_worker_without_second_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_router, "__version__", "0.3.2")
    client = MagicMock()
    registration = RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL)
    client.submit_registration.return_value = registration
    client.wait_until_active.side_effect = [TimeoutError("pending"), None]
    engine = _engine()
    monkeypatch.setattr(engine, "_router_worker_client", MagicMock(return_value=client))

    with pytest.raises(TimeoutError, match="pending"):
        engine.register_with_router()
    assert engine._is_registered_with_router is False
    assert engine._router_worker_id == _WORKER_ID

    engine.register_with_router()

    assert client.submit_registration.call_count == 1
    assert client.wait_until_active.call_count == 2
    assert engine._is_registered_with_router is True


def test_unknown_submission_result_blocks_duplicate_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_router, "__version__", "0.3.2")
    client = MagicMock()
    client.submit_registration.side_effect = TimeoutError("response lost")
    engine = _engine()
    monkeypatch.setattr(engine, "_router_worker_client", MagicMock(return_value=client))

    with pytest.raises(RouterWorkerSubmissionUnknown, match="generation must be replaced"):
        engine.register_with_router()
    with pytest.raises(RouterWorkerSubmissionUnknown, match="generation must be replaced"):
        engine.register_with_router()

    assert client.submit_registration.call_count == 1
    assert engine._is_registered_with_router is False

    registration = RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL)
    client.find_registration_by_url.return_value = registration
    engine.process = Namespace(pid=123)
    kill_process_tree = MagicMock()
    monkeypatch.setattr(sglang_engine_module, "kill_process_tree", kill_process_tree)

    engine.shutdown()

    client.find_registration_by_url.assert_called_once_with(worker_url=_WORKER_URL)
    client.remove.assert_called_once_with(registration=registration)
    assert engine._router_registration_submission_unknown is False
    kill_process_tree.assert_called_once_with(123)


def test_router_0_2_unknown_submission_retains_url_identity_for_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_router, "__version__", "0.2.2")
    client = MagicMock()
    client.submit_registration.side_effect = TimeoutError("response lost")
    engine = _engine()
    engine.process = Namespace(pid=123)
    monkeypatch.setattr(engine, "_router_worker_client", MagicMock(return_value=client))
    kill_process_tree = MagicMock()
    monkeypatch.setattr(sglang_engine_module, "kill_process_tree", kill_process_tree)

    with pytest.raises(RouterWorkerSubmissionUnknown, match="generation must be replaced"):
        engine.register_with_router()
    with pytest.raises(RouterWorkerSubmissionUnknown, match="generation must be replaced"):
        engine.register_with_router()

    assert client.submit_registration.call_count == 1
    assert engine._router_worker_id == _WORKER_URL

    engine.shutdown()

    client.find_registration_by_url.assert_not_called()
    client.remove.assert_called_once_with(
        registration=RouterWorkerRegistration(worker_id=_WORKER_URL, url=_WORKER_URL)
    )
    kill_process_tree.assert_called_once_with(123)


def test_router_0_2_registration_uses_async_worker_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_router, "__version__", "0.2.2")
    client = MagicMock()
    registration = RouterWorkerRegistration(worker_id=_WORKER_URL, url=_WORKER_URL)
    client.submit_registration.return_value = registration
    engine = _engine()
    monkeypatch.setattr(engine, "_router_worker_client", MagicMock(return_value=client))

    engine.register_with_router()

    client.submit_registration.assert_called_once_with(
        worker_url=_WORKER_URL,
        worker_type=RouterWorkerType.REGULAR,
        bootstrap_port=None,
    )
    client.wait_until_active.assert_called_once_with(registration=registration)
    assert engine._is_registered_with_router is True


def test_router_0_2_1_registration_keeps_synchronous_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_router, "__version__", "0.2.1")
    response = MagicMock()
    post = MagicMock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    engine = _engine()
    router_worker_client = MagicMock()
    monkeypatch.setattr(engine, "_router_worker_client", router_worker_client)

    engine.register_with_router()

    post.assert_called_once_with(
        "http://127.0.0.1:3000/add_worker?url=http://127.0.0.1:30000",
        timeout=10.0,
    )
    response.raise_for_status.assert_called_once_with()
    router_worker_client.assert_not_called()
    assert engine._is_registered_with_router is True


def test_router_0_2_1_unknown_submission_blocks_retry_and_allows_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_router, "__version__", "0.2.1")
    remove_response = MagicMock()
    post = MagicMock(side_effect=[requests.Timeout("response lost"), remove_response])
    monkeypatch.setattr(requests, "post", post)
    engine = _engine()
    engine.process = Namespace(pid=123)
    kill_process_tree = MagicMock()
    monkeypatch.setattr(sglang_engine_module, "kill_process_tree", kill_process_tree)

    with pytest.raises(RouterWorkerSubmissionUnknown, match="generation must be replaced"):
        engine.register_with_router()
    with pytest.raises(RouterWorkerSubmissionUnknown, match="generation must be replaced"):
        engine.register_with_router()

    assert engine._router_worker_id == _WORKER_URL
    assert engine._is_registered_with_router is False

    engine.shutdown()

    assert post.call_args_list == [
        call(
            "http://127.0.0.1:3000/add_worker?url=http://127.0.0.1:30000",
            timeout=10.0,
        ),
        call(
            "http://127.0.0.1:3000/remove_worker?url=http://127.0.0.1:30000",
            timeout=10.0,
        ),
    ]
    remove_response.raise_for_status.assert_called_once_with()
    assert engine._router_worker_id is None
    kill_process_tree.assert_called_once_with(123)


@pytest.mark.parametrize(
    ("router_version", "worker_id"),
    [
        ("0.2.2", _WORKER_URL),
        ("0.3.2", _WORKER_ID),
    ],
)
def test_definitive_submission_rejection_remains_retryable(
    monkeypatch: pytest.MonkeyPatch,
    router_version: str,
    worker_id: str,
) -> None:
    monkeypatch.setattr(sglang_router, "__version__", router_version)
    client = MagicMock()
    registration = RouterWorkerRegistration(worker_id=worker_id, url=_WORKER_URL)
    client.submit_registration.side_effect = [
        RouterWorkerSubmissionRejected("registration rejected"),
        registration,
    ]
    engine = _engine()
    monkeypatch.setattr(engine, "_router_worker_client", MagicMock(return_value=client))

    with pytest.raises(RouterWorkerSubmissionRejected, match="registration rejected"):
        engine.register_with_router()

    assert engine._router_registration_submission_unknown is False
    assert engine._router_worker_id is None

    engine.register_with_router()

    assert client.submit_registration.call_count == 2
    client.wait_until_active.assert_called_once_with(registration=registration)
    assert engine._is_registered_with_router is True


def test_background_registration_failure_retains_identity_without_resubmission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_router, "__version__", "0.3.2")
    client = MagicMock()
    registration = RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL)
    client.submit_registration.return_value = registration
    client.wait_until_active.side_effect = [
        RouterWorkerRegistrationFailed("registration failed"),
        RouterWorkerRegistrationFailed("registration failed"),
    ]
    engine = _engine()
    monkeypatch.setattr(engine, "_router_worker_client", MagicMock(return_value=client))

    with pytest.raises(RouterWorkerRegistrationFailed, match="registration failed"):
        engine.register_with_router()

    assert engine._router_worker_id == _WORKER_ID
    assert engine._is_registered_with_router is False

    with pytest.raises(RouterWorkerRegistrationFailed, match="registration failed"):
        engine.register_with_router()

    assert client.submit_registration.call_count == 1
    assert client.wait_until_active.call_count == 2
    assert engine._is_registered_with_router is False


def test_router_0_2_1_registration_failure_remains_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_router, "__version__", "0.2.1")
    first_response = MagicMock()
    first_response.status_code = 400
    first_response.raise_for_status.side_effect = requests.HTTPError(
        "registration failed",
        response=first_response,
    )
    second_response = MagicMock()
    post = MagicMock(side_effect=[first_response, second_response])
    monkeypatch.setattr(requests, "post", post)
    engine = _engine()

    with pytest.raises(requests.HTTPError, match="registration failed"):
        engine.register_with_router()

    assert engine._is_registered_with_router is False

    engine.register_with_router()

    assert post.call_count == 2
    assert second_response.raise_for_status.call_count == 1
    assert engine._is_registered_with_router is True


def test_shutdown_kills_process_when_router_removal_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_router, "__version__", "0.3.2")
    client = MagicMock()
    client.remove.side_effect = RuntimeError("remove failed")
    engine = _engine()
    engine._is_registered_with_router = True
    engine._router_worker_id = _WORKER_ID
    engine.process = Namespace(pid=123)
    monkeypatch.setattr(engine, "_router_worker_client", MagicMock(return_value=client))
    kill_process_tree = MagicMock()
    monkeypatch.setattr(sglang_engine_module, "kill_process_tree", kill_process_tree)

    with pytest.raises(RuntimeError, match="remove failed"):
        engine.shutdown()

    kill_process_tree.assert_called_once_with(123)


def test_shutdown_removes_accepted_worker_before_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_router, "__version__", "0.3.2")
    client = MagicMock()
    engine = _engine()
    engine._router_worker_id = _WORKER_ID
    engine.process = Namespace(pid=123)
    monkeypatch.setattr(engine, "_router_worker_client", MagicMock(return_value=client))
    kill_process_tree = MagicMock()
    monkeypatch.setattr(sglang_engine_module, "kill_process_tree", kill_process_tree)

    engine.shutdown()

    client.remove.assert_called_once_with(
        registration=RouterWorkerRegistration(
            worker_id=_WORKER_ID,
            url=_WORKER_URL,
        )
    )
    assert engine._router_worker_id is None
    kill_process_tree.assert_called_once_with(123)
