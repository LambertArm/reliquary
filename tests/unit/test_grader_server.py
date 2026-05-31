"""Tests for the grader server (pool + dispatch + watchdog).

Spawns a real server with worker subprocesses (python -m
reliquary.environment.grader.worker — no runsc). The IPC contract
is exercised end-to-end over a real Unix socket.
"""

import asyncio
import os
import socket
import tempfile
import threading
import time
import json
import pytest


@pytest.fixture
def grader_server(tmp_path):
    """Spawn a real GraderServer with 2 workers (no sandbox)."""
    from reliquary.environment.grader.server import GraderServer

    sock_path = str(tmp_path / "grader.sock")
    server = GraderServer(
        socket_path=sock_path,
        pool_size=2,
        worker_argv=["python", "-m", "reliquary.environment.grader.worker"],
        eval_timeout_s=5.0,
        metrics_port=0,  # 0 → OS-assigned ephemeral port
    )
    server.start()
    deadline = time.time() + 5.0
    while not os.path.exists(sock_path) and time.time() < deadline:
        time.sleep(0.05)
    yield server
    server.stop()


def _request(sock_path: str, code: str, tests: list[str], timeout_s: float = 5.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10.0)
        s.connect(sock_path)
        req = {"req_id": "test-req", "code": code, "tests": tests, "timeout_s": timeout_s}
        s.sendall(json.dumps(req).encode() + b"\n")
        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                break
        return json.loads(buf.split(b"\n", 1)[0])


def test_server_grades_correct_code(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="def add(a,b): return a+b",
        tests=["assert add(1,2) == 3", "assert add(0,0) == 0"],
    )
    assert resp["status"] == "ok"
    assert resp["passed"] == 2
    assert resp["total"] == 2


def test_server_grades_incorrect_code(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="def add(a,b): return a-b",
        tests=["assert add(1,2) == 3"],
    )
    assert resp["status"] == "ok"
    assert resp["passed"] == 0
    assert resp["total"] == 1


def test_server_handles_concurrent_requests(grader_server):
    """Pool of 2 → 4 concurrent requests should all succeed."""
    results = []
    errors = []

    def submit():
        try:
            r = _request(
                grader_server.socket_path,
                code="def f(): return 1",
                tests=["assert f() == 1"],
            )
            results.append(r)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=submit) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=15.0)

    assert not errors, f"unexpected errors: {errors}"
    assert len(results) == 4
    assert all(r["passed"] == 1 and r["total"] == 1 for r in results)


def test_server_returns_timeout_status_for_infinite_loop(grader_server):
    """Wall-clock timeout enforced by the server, not the worker."""
    resp = _request(
        grader_server.socket_path,
        code="while True: pass",
        tests=["assert True"],
        timeout_s=1.0,
    )
    assert resp["status"] == "timeout"
    assert resp["passed"] == 0


def test_pool_recovers_after_timeout(grader_server):
    """After a timed-out worker is killed and respawned, the pool
    must serve the next request normally."""
    # First request: infinite loop → server kills + respawns the worker.
    bad = _request(
        grader_server.socket_path,
        code="while True: pass",
        tests=["assert True"],
        timeout_s=1.0,
    )
    assert bad["status"] == "timeout"
    # Second request must succeed using the respawned worker.
    good = _request(
        grader_server.socket_path,
        code="def f(): return 42",
        tests=["assert f() == 42"],
    )
    assert good["status"] == "ok"
    assert good["passed"] == 1


def test_server_returns_grader_error_on_invalid_json_request(grader_server):
    """Malformed JSON on the wire → server replies grader_error, no hang."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5.0)
        s.connect(grader_server.socket_path)
        s.sendall(b"{this is not json\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk: break
            buf += chunk
    resp = json.loads(buf.split(b"\n", 1)[0])
    assert resp["status"] == "grader_error"


def test_runsc_workers_get_unique_container_ids(monkeypatch, tmp_path):
    """Regression: under runsc each pool worker must get a UNIQUE container
    id. ``runsc run <id>`` refuses a duplicate id, so a shared id silently
    starts only 1 of N workers. The prod argv carries a placeholder that the
    server substitutes per slot."""
    from reliquary.environment.grader import server as srv

    captured: list[list[str]] = []

    class _FakeProc:
        pid = 4321
        stdin = None
        stdout = None
        def poll(self):
            return None

    def _fake_popen(argv, **kw):
        captured.append(list(argv))
        return _FakeProc()

    monkeypatch.setattr(srv.subprocess, "Popen", _fake_popen)

    s = srv.GraderServer(
        socket_path=str(tmp_path / "g.sock"),
        pool_size=3,
        worker_argv=["runsc", "--network=none", "run", "--bundle", "/b",
                     srv.GRADER_CONTAINER_ID_PLACEHOLDER],
        metrics_port=0,
    )
    for i in range(3):
        s._spawn_worker(i)

    container_ids = [argv[-1] for argv in captured]
    assert len(set(container_ids)) == 3, \
        f"container ids must be unique per worker, got {container_ids}"
    assert srv.GRADER_CONTAINER_ID_PLACEHOLDER not in container_ids, \
        "placeholder must be substituted before exec"


def test_runsc_respawn_force_deletes_stale_container(monkeypatch, tmp_path):
    """A SIGKILL'd runsc container can't clean its own state, so respawning
    the same slot must `runsc delete --force` the stale id first or the
    re-run fails with 'container already exists'."""
    from reliquary.environment.grader import server as srv

    deletes: list[list[str]] = []

    class _FakeProc:
        pid = 4321
        stdin = None
        stdout = None
        def poll(self):
            return None

    monkeypatch.setattr(srv.subprocess, "Popen", lambda argv, **kw: _FakeProc())

    def _fake_run(argv, **kw):
        deletes.append(list(argv))
        return None

    monkeypatch.setattr(srv.subprocess, "run", _fake_run)

    s = srv.GraderServer(
        socket_path=str(tmp_path / "g.sock"),
        pool_size=1,
        worker_argv=["runsc", "run", "--bundle", "/b",
                     srv.GRADER_CONTAINER_ID_PLACEHOLDER],
        metrics_port=0,
    )
    s._spawn_worker(0)          # initial: no stale container to clean
    assert deletes == []
    s._spawn_worker(0)          # respawn: must force-delete the slot's id
    assert deletes, "respawn must force-delete the stale container id"
    assert deletes[-1][:3] == ["runsc", "delete", "--force"]


def test_metrics_endpoint_exposes_eval_counter(grader_server):
    """Hit /metrics on the grader's loopback HTTP listener."""
    import urllib.request, time
    # Trigger one eval.
    _request(grader_server.socket_path, code="x=1", tests=["assert x==1"])
    time.sleep(0.1)
    try:
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{grader_server.metrics_port}/metrics", timeout=2.0,
        )
    except Exception as e:
        pytest.skip(f"metrics endpoint not reachable: {e}")
    body = resp.read().decode()
    assert "grader_eval_total" in body
