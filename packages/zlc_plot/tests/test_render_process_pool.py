"""Panels that draw at the same time draw in different processes.

One render child held every live panel's renderer.  Its workers share one
interpreter, and while the compiled kernels release the GIL, the artist
updates, the chrome drawing and the pickle of each published front do not --
so four panels' Python ran one at a time however many cores were idle.

The pool is the fix and its whole content is a policy: spawn before sharing,
and never spawn what nobody asked for.  That policy is what is asserted here;
that a child can draw is the rest of this suite's business.
"""

from __future__ import annotations

import pytest

from zlc_plot.render_process import RenderProcessPool, default_render_process_count


class _Member:
    """A render child that costs nothing, so the policy can be seen."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.hosts: list[object] = []
        self.releases = 0
        self.closes = 0

    @property
    def host_count(self) -> int:
        return len(self.hosts)

    @property
    def alive(self) -> bool:
        return True

    @property
    def pid(self) -> int:
        return 1000 + len(self.name)

    def build_host(self, tag: object) -> object:
        self.hosts.append(tag)
        return (self.name, tag)

    def release(self, timeout: float = 0.0) -> bool:
        self.releases += 1
        return True

    def close(self, timeout: float = 0.0) -> bool:
        self.closes += 1
        return True

    def _await_close(self, timeout: float) -> bool:
        return True


@pytest.fixture
def spawned(monkeypatch):
    made: list[_Member] = []

    def factory(name: str) -> _Member:
        member = _Member(name)
        made.append(member)
        return member

    monkeypatch.setattr("zlc_plot.render_process.RenderProcess", factory)
    return made


def test_a_console_showing_one_panel_still_runs_one_child(spawned) -> None:
    """The pool is a ceiling, not a quota.

    A child is a whole renderer and about two hundred megabytes before it
    draws anything.  Spawning the cap up front would make every notebook and
    every one-panel window pay for panels it does not have.
    """

    pool = RenderProcessPool("test", size=4)
    assert not spawned
    pool.build_host("a")
    assert len(spawned) == 1


def test_panels_are_spread_before_they_are_shared(spawned) -> None:
    """Spread first, share second: the common board is one panel per child.

    Balanced from the start -- least-loaded among the members that exist --
    a four-panel board would have put all four in the first child, because
    a child with one host is still the least loaded when it is the only one.
    """

    pool = RenderProcessPool("test", size=4)
    for tag in ("a", "b", "c", "d"):
        pool.build_host(tag)
    assert [member.host_count for member in spawned] == [1, 1, 1, 1]


def test_a_board_larger_than_the_pool_balances_across_it(spawned) -> None:
    pool = RenderProcessPool("test", size=2)
    for tag in range(6):
        pool.build_host(tag)
    assert len(spawned) == 2
    assert [member.host_count for member in spawned] == [3, 3]


def test_a_panel_that_closes_frees_its_place(spawned) -> None:
    """Placement follows the children's live hosts, not the order of asks."""

    pool = RenderProcessPool("test", size=2)
    pool.build_host("a")
    pool.build_host("b")
    pool.build_host("c")
    assert [member.host_count for member in spawned] == [2, 1]
    spawned[0].hosts.clear()
    pool.build_host("d")
    assert [member.host_count for member in spawned] == [1, 1]


def test_the_last_window_owner_shuts_every_child_it_spawned(spawned) -> None:
    pool = RenderProcessPool("test", size=3)
    pool.build_host("a")
    pool.build_host("b")
    pool.retain()
    assert pool.release(0.0) is True
    assert [member.releases for member in spawned] == [0, 0]
    assert pool.release(0.0) is True
    assert [member.releases for member in spawned] == [1, 1]


def test_a_pool_nobody_drew_on_has_nothing_to_shut_down(spawned) -> None:
    pool = RenderProcessPool("test", size=4)
    assert pool.alive
    assert pool.release(0.0) is True
    assert pool.close(0.0) is True
    assert not spawned


def test_the_default_count_is_a_quarter_of_the_machine(monkeypatch) -> None:
    """Capped at four, and one on a machine too small to gain from more.

    A child is memory as much as a core, so the cap is not the core count:
    sixteen children on a sixteen-core workstation would be three gigabytes
    of renderer for a board that never shows sixteen panels.
    """

    for logical, expected in ((1, 1), (4, 1), (8, 2), (16, 4), (64, 4)):
        monkeypatch.setattr("zlc_plot.render_process.os.cpu_count",
                            lambda logical=logical: logical)
        assert default_render_process_count() == expected


def test_a_pool_refuses_to_be_smaller_than_one_child() -> None:
    with pytest.raises(ValueError):
        RenderProcessPool("test", size=0)
    with pytest.raises(ValueError):
        RenderProcessPool("   ")
