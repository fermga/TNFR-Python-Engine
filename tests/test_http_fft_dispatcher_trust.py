"""Remote pickle decoding requires explicit trust; no live network is used."""

from __future__ import annotations

import base64
import pickle
from types import SimpleNamespace

import pytest

import tnfr.dynamics.fft_dispatchers as module

_RESPONSE_EVENTS: list[str] = []


def _response_marker() -> dict[str, int]:
    _RESPONSE_EVENTS.append("executed")
    return {"marker": 1}


class _ExecutableResponse:
    def __reduce__(self):
        return _response_marker, ()


class _Session:
    def __init__(self, response):
        self.response = response
        self.posts = 0
        self.before_response = lambda: None

    def post(self, url, **kwargs):
        self.posts += 1
        self.before_response()
        return SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: self.response
        )


@pytest.fixture(autouse=True)
def _isolated_requests(monkeypatch):
    _RESPONSE_EVENTS.clear()
    monkeypatch.setattr(module, "requests", SimpleNamespace())
    yield
    _RESPONSE_EVENTS.clear()


def _payload(value):
    return {"payload": base64.b64encode(pickle.dumps(value)).decode("ascii")}


@pytest.mark.parametrize("auth_token", [None, "test-only-placeholder"])
def test_remote_executable_response_is_refused_by_default(auth_token):
    session = _Session(_payload(_ExecutableResponse()))
    dispatcher = module.HTTPFFTDispatcher(
        "https://example.invalid",
        auth_token=auth_token,
        session_factory=lambda: session,
    )

    with pytest.raises(PermissionError, match="trusted server and transport"):
        dispatcher.dispatch("probe", {})

    assert _RESPONSE_EVENTS == []
    assert session.posts == 1


def test_explicit_opt_in_retains_existing_pickle_wire_values():
    expected = {"coefficients": (1, -2, 3), "raw": b"\x00\xff"}
    session = _Session(_payload(expected))
    dispatcher = module.HTTPFFTDispatcher(
        "https://example.invalid",
        session_factory=lambda: session,
        allow_unsafe_pickle=True,
    )

    assert dispatcher.dispatch("probe", {}) == expected
    assert session.posts == 1


def test_explicit_trust_allows_executable_pickle_and_is_not_a_sandbox():
    session = _Session(_payload(_ExecutableResponse()))
    dispatcher = module.HTTPFFTDispatcher(
        "https://example.invalid",
        session_factory=lambda: session,
        allow_unsafe_pickle=True,
    )

    assert dispatcher.dispatch("probe", {}) == {"marker": 1}
    assert _RESPONSE_EVENTS == ["executed"]


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", "false", []])
def test_malformed_trust_flag_rejects_before_session_creation(invalid):
    def unexpected_session():
        pytest.fail("invalid trust policy must not create a session")

    with pytest.raises(TypeError, match="must be a boolean"):
        module.HTTPFFTDispatcher(
            "https://example.invalid",
            session_factory=unexpected_session,
            allow_unsafe_pickle=invalid,
        )


def test_live_flag_cannot_enable_pickle_through_string_truthiness():
    session = _Session(_payload(_ExecutableResponse()))
    dispatcher = module.HTTPFFTDispatcher(
        "https://example.invalid", session_factory=lambda: session
    )
    dispatcher.allow_unsafe_pickle = "false"

    with pytest.raises(TypeError, match="must be a boolean"):
        dispatcher.dispatch("probe", {})

    assert session.posts == 0
    assert _RESPONSE_EVENTS == []


def test_trust_revoked_during_transport_is_checked_before_decoding():
    session = _Session(_payload(_ExecutableResponse()))
    dispatcher = module.HTTPFFTDispatcher(
        "https://example.invalid",
        session_factory=lambda: session,
        allow_unsafe_pickle=True,
    )
    session.before_response = lambda: setattr(dispatcher, "allow_unsafe_pickle", False)

    with pytest.raises(PermissionError):
        dispatcher.dispatch("probe", {})

    assert _RESPONSE_EVENTS == []


def test_direct_json_response_without_pickle_retains_existing_behavior():
    session = _Session({"status": "ready", "workers": 2})
    dispatcher = module.HTTPFFTDispatcher(
        "https://example.invalid", session_factory=lambda: session
    )

    assert dispatcher.dispatch("probe", {}) == {"status": "ready", "workers": 2}


def test_optional_requests_dependency_still_reports_unavailability(monkeypatch):
    monkeypatch.setattr(module, "requests", None)

    with pytest.raises(RuntimeError, match="requests is required"):
        module.HTTPFFTDispatcher("https://example.invalid")
