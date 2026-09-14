"""Children kept warm ahead of the panels, one per panel, and let go of honestly.

One render child held every live panel's renderer, and its workers shared one
interpreter: the compiled kernels release the GIL, but artist updates, chrome
drawing and the pickle of each published front do not, so four panels' Python
ran one at a time however many cores were idle.

A child is 2.3 s from spawn to its first front, so WHEN they are started is
the whole difference between the pool being free and being expensive: started
when a panel first needed one, four panels painted 2.4, 4.7, 7.0 and 7.1 s
after the ask -- one child's boot, repeated.  Kept warm ahead of the ask, the
same four paint in 0.35 s.

And a child is never shared.  A panel that finds no warm child waits for a
fresh one rather than joining a busy one, because two panels in one child put
their Python back on one interpreter, which is the ceiling the pool exists to
lift.

What is asserted here is that policy and the reclaiming that has to go with
it.  That a child can draw is the rest of this suite's business.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time

import pytest

from zlc_plot.render_process import (
    _CHILD_NAME_PREFIX,
    DEFAULT_RENDER_SETTLED_SPARES,
    DEFAULT_RENDER_SPARES,
    RenderProcessPool,
)


class _Member:
    """A render child that costs nothing, so the policy can be seen."""

    def __init__(self, name: str, *, host_retired=None) -> None:
        self.name = name
        self.hosts: list[object] = []
        self.host_retired = host_retired
        self.releases = 0
        self.closes = 0

    @property
    def host_count(self) -> int:
        return len(self.hosts)

    def build_host(self, tag: object) -> object:
        self.hosts.append(tag)
        return (self.name, tag)

    def retire_host(self) -> None:
        """What a child does when its frontend lets a Host go."""

        if self.hosts:
            self.hosts.pop()
        if self.host_retired is not None:
            self.host_retired()

    def release(self, timeout: float = 0.0) -> bool:
        self.releases += 1
        # A real child's last owner starts a shutdown and reports it as not
        # settled, which is what makes the caller close it.
        return False

    def close(self, timeout: float = 0.0) -> bool:
        self.closes += 1
        return True

    def _await_close(self, timeout: float) -> bool:
        return True


@pytest.fixture
def spawned(monkeypatch):
    made: list[_Member] = []
    lock = threading.Lock()

    def factory(name: str, *, host_retired=None) -> _Member:
        member = _Member(name, host_retired=host_retired)
        with lock:
            made.append(member)
        return member

    monkeypatch.setattr("zlc_plot.render_process.RenderProcess", factory)
    return made


@pytest.fixture
def gated(monkeypatch):
    """Children that come up only while the gate is open, so a boot can be seen."""

    made: list[_Member] = []
    gate = threading.Event()
    gate.set()

    def factory(name: str, *, host_retired=None) -> _Member:
        gate.wait(5.0)
        member = _Member(name, host_retired=host_retired)
        made.append(member)
        return member

    monkeypatch.setattr("zlc_plot.render_process.RenderProcess", factory)
    return made, gate


def _settled(pool: RenderProcessPool, warm: int, timeout: float = 5.0) -> None:
    """Wait for the pool's own threads to finish shaping it."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        members = list(pool._members)
        if not pool._starting and len(
            [member for member in members if member.host_count == 0]
        ) == warm:
            return
        time.sleep(0.005)
    raise AssertionError(
        f"warm={len([m for m in pool._members if m.host_count == 0])} "
        f"total={len(pool._members)} starting={len(pool._starting)}"
    )


def test_a_fresh_pool_stands_its_children_up_before_anyone_asks(spawned) -> None:
    """The whole point: a panel must never wait out a child's boot."""

    pool = RenderProcessPool("test", spares=4)
    _settled(pool, warm=4)
    assert len(spawned) == 4
    assert all(member.host_count == 0 for member in spawned)


def test_taking_a_warm_child_starts_its_replacement(spawned) -> None:
    """Warm again by the time the next panel asks, which is the request."""

    pool = RenderProcessPool("test", spares=2)
    _settled(pool, warm=2)
    pool.build_host("a")
    _settled(pool, warm=2)
    assert len(spawned) == 3
    assert [member.host_count for member in spawned] == [1, 0, 0]


def test_a_board_gets_one_child_per_panel(spawned) -> None:
    """Spread, never share: a warm child always beats a busy one."""

    pool = RenderProcessPool("test", spares=2)
    _settled(pool, warm=2)
    for tag in ("a", "b", "c", "d"):
        pool.build_host(tag)
        _settled(pool, warm=2)
    drawing = [member for member in spawned if member.host_count]
    assert len(drawing) == 4
    assert all(member.host_count == 1 for member in drawing)


def test_a_panel_with_no_warm_child_waits_for_a_fresh_one_and_never_shares(
    gated,
) -> None:
    """Out of warm children, a panel waits out a boot rather than doubling up.

    The replacement already on its way is the one it waits for: no second
    child is started for a wait one start already covers.
    """

    made, gate = gated
    pool = RenderProcessPool("test", spares=1)
    _settled(pool, warm=1)
    gate.clear()
    pool.build_host("a")
    assert len(pool._starting) == 1

    landed: list[object] = []
    asker = threading.Thread(target=lambda: landed.append(pool.build_host("b")))
    asker.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and pool._waiting < 1:
        time.sleep(0.005)
    time.sleep(0.05)
    assert asker.is_alive()
    assert pool._waiting == 1
    assert len(pool._starting) == 1
    assert [member.host_count for member in made] == [1]

    gate.set()
    asker.join(5.0)
    assert not asker.is_alive()
    _settled(pool, warm=1)
    assert landed == [(made[1].name, "b")]
    assert [member.host_count for member in made] == [1, 1, 0]
    assert pool._waiting == 0


def test_panels_arriving_together_each_get_a_child_of_their_own(spawned) -> None:
    """Four asks at once against one warm child: four children, one left warm."""

    pool = RenderProcessPool("test", spares=1)
    _settled(pool, warm=1)
    askers = [
        threading.Thread(target=pool.build_host, args=(tag,))
        for tag in ("a", "b", "c", "d")
    ]
    for asker in askers:
        asker.start()
    for asker in askers:
        asker.join(5.0)
    assert not any(asker.is_alive() for asker in askers)
    _settled(pool, warm=1)
    assert sorted(member.host_count for member in spawned) == [0, 1, 1, 1, 1]


def test_a_child_whose_panel_closed_is_warm_again(spawned) -> None:
    """And the surplus is let go, not kept for ever.

    A board that opens three panels and closes them again would otherwise
    hold three children plus the spares, for ever, with nothing drawing.
    """

    pool = RenderProcessPool("test", spares=2)
    _settled(pool, warm=2)
    for tag in ("a", "b", "c"):
        pool.build_host(tag)
        _settled(pool, warm=2)
    assert len(spawned) == 5

    for member in [item for item in spawned if item.host_count]:
        member.retire_host()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if len(pool._members) == 2 and not pool._starting:
            break
        time.sleep(0.005)
    assert len(pool._members) == 2
    # Told to go, not waited for: the reclaim can run on the very reader
    # thread a close would wait for.
    assert sum(member.releases for member in spawned) == 3
    assert sum(member.closes for member in spawned) == 0


def test_the_last_window_owner_lets_every_child_go(spawned) -> None:
    pool = RenderProcessPool("test", spares=3)
    _settled(pool, warm=3)
    pool.retain()
    assert pool.release(0.0) is True
    assert [member.releases for member in spawned] == [0, 0, 0]
    pool.release(0.0)
    assert [member.releases for member in spawned] == [1, 1, 1]


def test_closing_never_waits_for_a_child_that_is_still_starting(monkeypatch) -> None:
    """Two halves, and the window depends on both.

    A window's close must return within a Qt turn -- the console asserts
    fifty milliseconds -- and a child takes 2.3 s to start, so a close that
    joined the starting threads would hold the GUI for seconds whenever an
    operator shut a console while one was coming up.  And the late child
    must still be let go: the window between deciding to start one and that
    one existing is seconds wide, which is exactly the window a console is
    closed in.
    """

    made: list[_Member] = []
    allowed = threading.Event()

    def factory(name: str, *, host_retired=None) -> _Member:
        allowed.wait(5.0)
        member = _Member(name, host_retired=host_retired)
        made.append(member)
        return member

    monkeypatch.setattr("zlc_plot.render_process.RenderProcess", factory)
    pool = RenderProcessPool("test", spares=2)

    begun = time.monotonic()
    pool.release(0.0)
    assert time.monotonic() - begun < 0.05

    allowed.set()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and len(made) < 2:
        time.sleep(0.005)
    assert pool.close(10.0) is True
    assert len(made) == 2
    assert all(member.closes for member in made)


def test_closing_lets_a_waiting_panel_go(gated) -> None:
    """A panel waiting for a child must not outwait the window that asked."""

    made, gate = gated
    pool = RenderProcessPool("test", spares=1)
    _settled(pool, warm=1)
    gate.clear()
    pool.build_host("a")
    outcome: list[object] = []

    def ask() -> None:
        try:
            outcome.append(pool.build_host("b"))
        except RuntimeError as error:
            outcome.append(error)

    asker = threading.Thread(target=ask)
    asker.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and pool._waiting < 1:
        time.sleep(0.005)
    pool.release(0.0)
    asker.join(2.0)
    assert not asker.is_alive()
    assert isinstance(outcome[0], RuntimeError)
    gate.set()
    assert pool.close(5.0) is True


def test_a_pool_keeps_at_least_one_child_warm() -> None:
    with pytest.raises(ValueError):
        RenderProcessPool("test", spares=0)
    with pytest.raises(ValueError):
        RenderProcessPool("test", spares=2, settled_spares=3)
    with pytest.raises(ValueError):
        RenderProcessPool("test", spares=2, settled_spares=0)
    with pytest.raises(ValueError):
        RenderProcessPool("   ")


def test_a_board_that_has_arrived_stops_holding_a_board_in_reserve(
    spawned,
) -> None:
    """Four warm is an answer to "a whole board at once", asked once.

    A board arrives together and then GROWS one panel at a time, so the
    opening count is the wrong answer for the rest of the session: four
    more idle renderers is most of a gigabyte held against an operator who
    adds one panel.  Past the settled count the pool holds the settled
    count instead -- and comes back up when the board is closed, because
    the next board arrives the same way the first did.
    """

    pool = RenderProcessPool("test", spares=4, settled_spares=2)
    _settled(pool, warm=4)

    taken = []
    for tag in ("a", "b"):
        pool.build_host(tag)
        taken.append(tag)
        _settled(pool, warm=4)

    # The third panel is what makes this a board rather than an opening.
    pool.build_host("c")
    _settled(pool, warm=2)
    assert len([member for member in spawned if member.host_count]) == 3

    for member in [item for item in spawned if item.host_count]:
        member.retire_host()
    _settled(pool, warm=4)


def test_the_defaults_are_a_board_then_what_an_operator_adds() -> None:
    """Four warm, because a board is four cards; two once one is drawing,
    because a board grows one panel at a time."""

    assert DEFAULT_RENDER_SPARES == 4
    assert DEFAULT_RENDER_SETTLED_SPARES == 2
    assert DEFAULT_RENDER_SETTLED_SPARES < DEFAULT_RENDER_SPARES


def _child_report(connection) -> None:
    """What a render child sees: its BLAS thread bound and its commit charge."""

    import ctypes
    from ctypes import wintypes

    import numpy  # noqa: F401  -- the library whose thread pool is bounded

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.WinDLL("psapi").GetProcessMemoryInfo(
        ctypes.WinDLL("kernel32").GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    connection.send(
        (os.environ.get("OPENBLAS_NUM_THREADS"), counters.PrivateUsage / 2**20)
    )
    connection.close()


@pytest.mark.skipif(sys.platform != "win32", reason="commit charge is read through psapi")
def test_a_render_child_uses_one_blas_thread() -> None:
    """OpenBLAS commits a scratch buffer per thread it may use the moment it
    loads, and a child never multiplies a matrix: a child on a sixteen-core
    machine committed 1258 MB, about a gigabyte of it two thread pools.
    The bound has to be in the child's environment BEFORE numpy loads, and
    the product's bootstrap never runs in a child -- a package ``__main__``
    is not re-run by a spawned process -- so the child sets it itself, in
    the first module it imports, keyed on the name every child is spawned
    under.  Spawned the way the pool spawns, the child says so, and its
    commit charge says the library heard it.
    """

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_report, args=(child,), name=f"{_CHILD_NAME_PREFIX}test", daemon=True
    )
    process.start()
    child.close()
    try:
        assert parent.poll(120.0), "the child never reported"
        threads, private_mb = parent.recv()
    finally:
        process.join(30.0)
        if process.is_alive():
            process.terminate()
    assert threads == "1"
    # Numpy alone commits over five hundred megabytes with sixteen threads
    # and under a hundred with one; the ceiling leaves room for a smaller
    # machine's pool without admitting a full one.
    assert private_mb < 250.0, f"a child still committed {private_mb:.0f} MB"
