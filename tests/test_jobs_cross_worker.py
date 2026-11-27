"""Cross-worker job-registry regression test (Kanban 3eb5bf3b-9510-810a-a513-ec5628c2656f).

`jobs._jobs` was a bare in-process dict. The workspace runs
`AW_WORKSPACE_WORKERS=10` — ten OS processes, ten Python heaps, no request
stickiness — so `POST /provision/run` created a job on whichever worker
answered and the next `GET /testcases/jobs/<id>` landed on a different one and
returned `404 no such run job`. Measured live before the fix: 7 of 12
consecutive polls of one healthy job came back 404. The CLI read that as a
provisioning failure and exited non-zero, which the hourly "Architecture Test
Provisioning" task escalated to a full system-analyst agent — ~1.56M input
tokens in one 14h window, repeated for three weeks.

**"Two workers" here is two independently-loaded copies of `jobs.py`.** Each
gets its own module object and therefore its own `_jobs` dict, which is exactly
what a second uvicorn worker is from this module's point of view — the same
thing aw-app-devctl's `test_relay_multiworker.py` achieves with two relay
instances, except loading the module twice also covers the module-level state
that an instance-based test cannot reach.

Both copies share one store, and the suite runs twice over:

* **fake** — an in-memory stand-in for the sync redis client, injected by
  injecting a fake `ctx.state.kv` via `bind`. No service needed, so this arm runs everywhere,
  including this repo's own CI (which has neither a Redis nor an aw-workspace
  checkout). It exercises the real code path: both module copies go through
  `_mirror`/`_read_shared`/`_scan_shared`.
* **real** — an actual Redis, skipped when one isn't reachable. Same
  assertions against the genuine client, so a fake that has drifted from
  redis-py's behaviour cannot hide a break.

Mutation-tested: with `start()`'s `_mirror(...)` call removed (the fix),
test_job_started_on_one_worker_is_visible_from_another and
test_snapshot_spans_workers fail on both arms.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_JOBS_PY = _REPO_ROOT / "architecture_app" / "jobs.py"

# Isolate this run's keys from any real workspace sharing the same Redis. Set
# before anything reads it (the facade's key prefix resolves it per call).
os.environ["AW_WORKSPACE"] = f"architecture-jobs-test-{uuid.uuid4().hex[:8]}"
os.environ.pop("AW_ARCHITECTURE_JOBS_SHARED", None)


def _load_worker(name: str):
    """Load `jobs.py` under its own module name — a fresh `_jobs` dict, a fresh
    client cache, no shared Python state with any other copy."""
    spec = importlib.util.spec_from_file_location(name, _JOBS_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---- the two store backends -------------------------------------------------

class _FakeKv:
    """Async stand-in for ``ctx.state.kv`` — put(ttl required)/get/scan, TTLs
    honoured against the wall clock so expiry behaviour is really exercised."""

    def __init__(self) -> None:
        self.data: dict[str, tuple[Any, float]] = {}

    async def put(self, name, value, ttl):
        self.data[name] = (json.loads(json.dumps(value)), time.time() + ttl)

    async def get(self, name):
        entry = self.data.get(name)
        if entry is None or time.time() >= entry[1]:
            self.data.pop(name, None)
            return None
        return entry[0]

    async def scan(self, prefix=""):
        return [k for k in list(self.data)
                if k.startswith(prefix) and await self.get(k) is not None]

    async def delete(self, name):
        self.data.pop(name, None)


class _FakeCtx:
    def __init__(self, kv) -> None:
        self.state = type("State", (), {"kv": kv})()


class _DeadKv:
    """A store whose every call raises — must degrade, not escape."""

    async def put(self, *a, **k):
        raise ConnectionError("store is down")

    get = scan = delete = put


@pytest.fixture(scope="module")
def loop():
    """One event loop on its own thread — stands in for the workspace loop the
    app is activated on; run threads bridge onto it like they do in prod."""
    lp = asyncio.new_event_loop()
    t = threading.Thread(target=lp.run_forever, daemon=True)
    t.start()
    yield lp
    lp.call_soon_threadsafe(lp.stop)
    t.join(5)


def _real_redis_url() -> str:
    return next((os.environ[v] for v in ("AW_TEST_REDIS_URL", "AW_WORKSPACE_REDIS_URL",
                                         "AW_REDIS_URL") if os.environ.get(v)),
                "redis://127.0.0.1:6379/0")


def _real_ctx():
    """A real minimal ``AppContext`` granted ``state:own`` against a live
    Redis, or None when core's ``src`` or a Redis is unavailable (this repo's
    own CI has neither)."""
    try:
        import redis

        sys.path.insert(0, os.environ.get("AW_WORKSPACE_CONTAINER_DIR", "/opt/aw-workspace"))
        from src.apps.base import AppContext

        url = next((os.environ[v] for v in ("AW_TEST_REDIS_URL", "AW_WORKSPACE_REDIS_URL",
                                            "AW_REDIS_URL") if os.environ.get(v)),
                   "redis://127.0.0.1:6379/0")
        redis.Redis.from_url(url, socket_connect_timeout=2).ping()

        class _Journal:
            def record(self, *a, **kw):
                pass

        class _Runtime:
            journal = _Journal()

        return AppContext(runtime=_Runtime(), app_id="architecture", version="1.0.0",
                          granted_permissions=["state:own"], config={},
                          package_dir="/tmp/architecture")
    except Exception:
        return None


@pytest.fixture(params=["fake", "real"])
def workers(request, loop):
    """Two independent copies of `jobs.py` sharing one store — worker A and
    worker B. Yields `(a, b, kv)`; `kv` is the fake store or the real facade."""
    if request.param == "real":
        ctx = _real_ctx()
        if ctx is None:
            pytest.skip("no aw-workspace checkout + reachable Redis for the real arm")
    else:
        ctx = _FakeCtx(_FakeKv())
    a = _load_worker(f"jobs_worker_a_{request.param}")
    b = _load_worker(f"jobs_worker_b_{request.param}")
    for worker in (a, b):
        worker.bind(ctx, loop)

    yield a, b, ctx.state.kv

    if request.param == "real":
        async def _cleanup():
            for name in await ctx.state.kv.scan("run-"):
                await ctx.state.kv.delete(name)
        asyncio.run_coroutine_threadsafe(_cleanup(), loop).result(10)


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


# ---- the regression --------------------------------------------------------

def test_job_started_on_one_worker_is_visible_from_another(workers):
    """THE bug. Worker A starts a job; worker B — which has never heard of it —
    must answer for it rather than 404."""
    a, b, _ = workers
    job = a.start("/some/test_file.py", lambda fp: {"ok": True, "ran": fp})

    assert b._jobs == {}, "worker B must not share worker A's in-process dict"
    seen = b.get(job["id"])
    assert seen is not None, (
        "worker B returned None for a job worker A had just created — this is "
        "the 404 'no such run job' the CLI reads as a provisioning failure")
    assert seen["id"] == job["id"]
    assert seen["file_path"] == "/some/test_file.py"


def test_start_on_the_loop_thread_does_not_deadlock_and_still_mirrors(workers, loop):
    """`start()` is called from an async route, i.e. ON the bound loop, where
    a blocking bridge would deadlock. It must schedule the mirror write
    instead, and the job must still become visible to another worker."""
    a, b, _ = workers

    async def _from_route():
        return a.start("/some/test_loop.py", lambda fp: {"ok": True})

    job = asyncio.run_coroutine_threadsafe(_from_route(), loop).result(5)
    assert _wait_for(lambda: b.get(job["id"])), (
        "a job started on the loop thread never reached the shared store")


def test_a_still_queued_job_is_visible_from_another_worker(workers):
    """The `start()` mirror has to happen before the run thread does anything.

    `MAX_CONCURRENT` is 2, so a third job sits at `queued` — sometimes for
    minutes, since the slots ahead of it are cold pip installs. Mirroring only
    on the queued→running transition would leave exactly that window 404ing,
    which is the window a scheduled `autoprovision` poll is most likely to land
    in. Found by mutation-testing: removing `start()`'s own mirror left every
    other test in this file green.
    """
    a, b, _ = workers
    blocked = [threading.Event() for _ in range(a.MAX_CONCURRENT)]
    for gate in blocked:
        a.start("/hog/test_slot.py", lambda _fp, _g=gate: _g.wait(10) or {"ok": True})
    try:
        _wait_for(lambda: sum(1 for j in a.snapshot()
                              if j["status"] == "running") == a.MAX_CONCURRENT)
        queued = a.start("/some/test_waiting.py", lambda fp: {"ok": True})
        assert a.get(queued["id"])["status"] == "queued", (
            "setup failed — the job was expected to be waiting on the semaphore")

        seen = b.get(queued["id"])
        assert seen is not None, (
            "a queued job is invisible to other workers — a poll landing on one "
            "of them 404s on a job that has not even started yet")
        assert seen["status"] == "queued"
    finally:
        for gate in blocked:
            gate.set()


def test_result_of_a_finished_job_crosses_workers(workers):
    """Not just the id: the result the poller is actually waiting for."""
    a, b, _ = workers
    job = a.start("/some/test_file.py", lambda fp: {"ok": True, "provisioned": []})

    done = _wait_for(lambda: (b.get(job["id"]) or {}).get("status") == "done"
                     and b.get(job["id"]))
    assert done, "worker B never saw the job reach 'done'"
    assert done["result"] == {"ok": True, "provisioned": []}
    assert done["error"] is None
    assert done["finished_at"]


def test_a_failing_run_reports_its_error_across_workers(workers):
    a, b, _ = workers

    def _boom(_fp):
        raise RuntimeError("pip exploded")

    job = a.start("/some/test_file.py", _boom)
    done = _wait_for(lambda: (b.get(job["id"]) or {}).get("status") == "done"
                     and b.get(job["id"]))
    assert done, "worker B never saw the failing job finish"
    assert done["error"] == "RuntimeError: pip exploded"


def test_snapshot_spans_workers(workers):
    """A UI that reloaded onto another worker must still find the run."""
    a, b, _ = workers
    job_a = a.start("/a/test_one.py", lambda fp: {"ok": True})
    job_b = b.start("/b/test_two.py", lambda fp: {"ok": True})

    ids_from_b = {j["id"] for j in b.snapshot()}
    assert job_a["id"] in ids_from_b, "worker A's job is missing from B's snapshot"
    assert job_b["id"] in ids_from_b
    assert {j["id"] for j in a.snapshot()} == ids_from_b


def test_the_heartbeat_keeps_a_long_run_visible_past_the_live_ttl(workers, monkeypatch):
    """A running job's shared entry is short-lived on purpose — that is what
    makes a dead worker's claim expire instead of asserting "running" forever.
    The owning worker therefore has to renew it, or a provision that outlives
    one TTL window (they all do: cold pip is minutes, the TTL is 120s) goes
    invisible mid-run and the poll 404s on a healthy job.

    The real constants are too slow to observe, so they're shrunk here: a 1s
    TTL renewed every 0.2s, checked after more than two windows have passed.
    """
    a, b, _ = workers
    monkeypatch.setattr(a, "_LIVE_TTL_S", 1)
    monkeypatch.setattr(a, "_HEARTBEAT_INTERVAL_S", 0.2)

    release = threading.Event()
    job = a.start("/slow/test_thing.py", lambda _fp: release.wait(10) or {"ok": True})
    try:
        time.sleep(2.3)          # > 2 × the shrunk TTL
        seen = b.get(job["id"])
        assert seen is not None, (
            "the running job's shared entry expired mid-run — the owning "
            "worker is not renewing it")
        assert seen["status"] == "running"
    finally:
        release.set()


def test_own_job_is_served_from_the_local_dict(workers):
    """The worker that owns the run stays authoritative — it holds the live
    object, so it must not be reading its own state back out of the store."""
    a, _b, client = workers
    job = a.start("/some/test_file.py", lambda fp: {"ok": True})
    _wait_for(lambda: (a.get(job["id"]) or {}).get("status") == "done")

    # Corrupt the shared copy; the owner must be unaffected.
    asyncio.run_coroutine_threadsafe(
        client.put(job["id"], {"id": job["id"], "status": "bogus"}, 60),
        a._loop).result(5)
    assert a.get(job["id"])["status"] == "done"


def test_finished_jobs_are_shared_with_the_retention_window_as_a_ttl(workers):
    """The original objection to persisting this state was rows outliving what
    they describe. The shared copy answers it with a TTL, not with a reaper."""
    a, _b, client = workers
    job = a.start("/some/test_file.py", lambda fp: {"ok": True})
    _wait_for(lambda: (a.get(job["id"]) or {}).get("status") == "done")

    if hasattr(client, "data"):   # fake store — its own expiry bookkeeping
        ttl = client.data[job["id"]][1] - time.time()
    else:                         # real facade has no TTL read; probe the raw key
        import redis as _redis
        from src.apps.state_facade import kv_key
        raw = _redis.Redis.from_url(_real_redis_url())
        ttl = raw.ttl(kv_key("architecture", job["id"]))
    assert 0 < ttl <= a._RETAIN_SECONDS
    assert ttl > a._LIVE_TTL_S, (
        "a finished job must be retained for the full window, not the short "
        "live TTL — a late poller would otherwise get a 404 for the result")


# ---- degradation -----------------------------------------------------------

def test_an_unreachable_store_degrades_to_worker_local(loop):
    """Redis down must cost cross-worker visibility and nothing else: the job
    still runs, the owning worker still answers for it, no exception escapes."""
    ctx = _FakeCtx(_DeadKv())
    a = _load_worker("jobs_worker_a_dead")
    b = _load_worker("jobs_worker_b_dead")
    for worker in (a, b):
        worker.bind(ctx, loop)

    job = a.start("/some/test_file.py", lambda fp: {"ok": True})
    done = _wait_for(lambda: (a.get(job["id"]) or {}).get("status") == "done"
                     and a.get(job["id"]))
    assert done and done["result"] == {"ok": True}
    assert b.get(job["id"]) is None          # degraded, as before the fix
    assert b.snapshot() == []                # and not an exception


def test_degraded_facade_is_not_called_at_all(loop):
    """While ``ctx.state.degraded`` is set, a poll must not pay the bridge wait
    (the facade still issues ops during an outage; it only records the
    breaker): no kv call is made and the read returns at once."""
    calls: list[str] = []

    class _SlowKv:
        async def get(self, name):
            calls.append(name)
            await asyncio.sleep(0.5)

        scan = put = delete = get

    ctx = _FakeCtx(_SlowKv())
    ctx.state.degraded = True
    a = _load_worker("jobs_worker_a_degraded")
    a.bind(ctx, loop)

    t0 = time.monotonic()
    assert a.get("run-nope") is None
    assert a.snapshot() == []
    assert time.monotonic() - t0 < 1.0, "a degraded facade must short-circuit"
    assert calls == []

    ctx.state.degraded = False   # recovery: sharing resumes
    a._BRIDGE_TIMEOUT_S = 0.2
    a.get("run-nope")
    assert calls == ["run-nope"]


def test_sharing_can_be_turned_off(monkeypatch, loop):
    """`AW_ARCHITECTURE_JOBS_SHARED=0` is the field escape hatch."""
    a = _load_worker("jobs_worker_a_off")
    monkeypatch.setenv(a.SHARE_ENV, "0")
    a.bind(_FakeCtx(_FakeKv()), loop)
    assert a._call(lambda: a._ctx.state.kv.scan("")) is None
    monkeypatch.setenv(a.SHARE_ENV, "1")
    assert a._sharing_enabled() is True
