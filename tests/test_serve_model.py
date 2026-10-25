"""ModelServer readiness: the server that answers on the port must be ours.

A reply on the port proves only that something is bound to it. A second
local model given the same port, or a server left over from an earlier run,
answers first while the just-launched process dies on bind, and every later
request would silently go to the wrong model.
"""

import io

import pytest

import serve_model

OURS = "org/guesser"


class FakeProc:
    def __init__(self, exit_code=None):
        self.returncode = exit_code
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO("")
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class Reply:
    def __init__(self, ids=None, *, invalid=False):
        self._ids = ids or []
        self._invalid = invalid

    def json(self):
        if self._invalid:
            raise ValueError("not json")
        return {"data": [{"id": i} for i in self._ids]}


def refused():
    raise serve_model.requests.exceptions.RequestException("connection refused")


@pytest.fixture
def server_env(monkeypatch):
    """Script the HTTP replies and the subprocess; return the launched procs."""
    procs = []
    state = {"exit_code": None, "replies": []}

    def popen(*args, **kwargs):
        proc = FakeProc(exit_code=state["exit_code"])
        procs.append(proc)
        return proc

    def get(url, timeout=5):
        return state["replies"].pop(0)()

    monkeypatch.setattr(serve_model.subprocess, "Popen", popen)
    monkeypatch.setattr(serve_model.requests, "get", get)
    monkeypatch.setattr(serve_model.time, "sleep", lambda s: None)

    def start(replies, exit_code=None):
        state["replies"] = list(replies)
        state["exit_code"] = exit_code
        return serve_model.ModelServer(OURS, port=8000)

    start.procs = procs
    return start


def test_ready_once_our_model_answers_after_retries(server_env):
    server = server_env([refused, refused, lambda: Reply([OURS])])
    assert server.process is server_env.procs[0]
    assert not server.process.terminated
    assert server.base_url == "http://localhost:8000/v1"


def test_another_model_on_the_port_is_rejected_and_our_process_stopped(server_env):
    with pytest.raises(RuntimeError, match=r"8000 is already serving \['org/other'\]"):
        server_env([lambda: Reply(["org/other"])])
    assert server_env.procs[0].terminated


def test_a_non_json_service_on_the_port_counts_as_foreign(server_env):
    with pytest.raises(RuntimeError, match=r"serving \[\]"):
        server_env([lambda: Reply(invalid=True)])
    assert server_env.procs[0].terminated


def test_a_process_that_exits_before_ready_is_reported(server_env):
    with pytest.raises(RuntimeError, match="exited with code 1"):
        server_env([], exit_code=1)


def test_hosted_model_builds_an_api_client_and_no_server(server_env):
    server = serve_model.ModelServer("gpt-4.1")
    assert not hasattr(server, "process")
    assert server_env.procs == []


def test_local_model_name_must_have_an_org_prefix():
    with pytest.raises(ValueError, match="org/name"):
        serve_model.ModelServer("bare-name")
