"""Children kept warm ahead of the panels, and let go of honestly.

One render child held every live panel's renderer, and its workers shared one
interpreter: the compiled kernels release the GIL, but artist updates, chrome
drawing and the pickle of each published front do not, so four panels' Python
ran one at a time however many cores were idle.

A child is 2.3 s from spawn to its first front, so WHEN they are started is
the whole difference between the pool being free and being expensive: started
when a panel first needed one, four panels painted 2.4, 4.7, 7.0 and 7.1 s
after the ask -- one child's boot, repeated.  Kept warm ahead of the ask, the
same four paint in 0.35 s.

What is asserted here is that policy and the reclaiming that has to go with
it.  That a child can draw is the rest of this suite's business.
"""

from __future__ import annotations

import threading
import time

import pytest

from zlc_plot.render_process import (
    DEFAULT_RENDER_LIMIT,
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
    """Spread before sharing: a warm child always beats a busy one."""

    pool = RenderProcessPool("test", spares=2, limit=8)
    _settled(pool, warm=2)
    for tag in ("a", "b", "c", "d"):
        pool.build_host(tag)
        _settled(pool, warm=2)
    drawing = [member for member in spawned if member.host_count]
    assert len(drawing) == 4
    assert all(member.host_count == 1 for member in drawing)


def test_a_board_past_the_ceiling_shares_a_child_again(spawned) -> None:
    """Memory is the whole cost, so the count is bounded.

    Past the ceiling a panel joins the least busy child rather than waiting
    for one that will never be started.
    """

    pool = RenderProcessPool("test", spares=1, limit=3)
    _settled(pool, warm=1)
    for tag in range(6):
        pool.build_host(tag)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and pool._starting:
        time.sleep(0.005)
    assert len(spawned) == 3
    assert sorted(member.host_count for member in spawned) == [2, 2, 2]


def test_a_child_whose_panel_closed_is_warm_again(spawned) -> None:
    """And the surplus is let go, not kept for ever.

    A board that opens three panels and closes them again would otherwise
    hold three children plus the spares, for ever, with nothing drawing.
    """

    pool = RenderProcessPool("test", spares=2, limit=8)
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


def test_a_pool_keeps_at_least_one_child_warm_and_holds_at_least_that_many() -> None:
    with pytest.raises(ValueError):
        RenderProcessPool("test", spares=0)
    with pytest.raises(ValueError):
        RenderProcessPool("test", spares=4, limit=2)
    with pytest.raises(ValueError):
        RenderProcessPool("   ")


def test_the_defaults_are_a_board_and_a_memory_ceiling() -> None:
    """Four warm, because a board is four cards; ten at most, because a
    child is two hundred megabytes."""

    assert DEFAULT_RENDER_SPARES == 4
    assert DEFAULT_RENDER_LIMIT >= DEFAULT_RENDER_SPARES
