"""A causal group presents atomically: frame and derived occupancy, one shot.

The plane already guarantees the DATA is same-shot: ``build_front`` joins a
continuous component on the lineage roots reached through each publication's
``direct_parent_refs`` and holds the previous complete component when a
member lags.  What used to break was PRESENTATION -- each panel's widget put
its own render up the moment it landed, so two panels of one causal group
showed different shots whenever their render times differed, and Pause froze
whatever half-board happened to be visible.

Here the ports stage their renders and the board presents each batch only
when every member's render completed -- in one owner pass, whatever order
the renders arrived in, and regardless of Pause.
"""

from __future__ import annotations

from concurrent.futures import Future
import time
from types import SimpleNamespace

from zlc_runtime import LiveDatasetOutput
from zlc_runtime.plane import SignalDataPlane
from zlc_workbench.board import LiveBoard
from zlc_workbench.presentation import PlotPanelPort

from test_signal_front import _output


FRAME = "camera/frame"
OCCUPANCY = "occupancy/value"


def _exact_output(name: str, revision: int) -> LiveDatasetOutput:
    return _output(name, revision)


class _RenderHost:
    """A plot host double whose renders complete when the test says so."""

    def __init__(self, name: str) -> None:
        self.host_id = f"host-{name}"
        self.futures: list[Future] = []
        self.rendered: list[object] = []

    def update_data(self, plot_input: object, **_render: object) -> Future:
        self.rendered.append(plot_input)
        future: Future = Future()
        self.futures.append(future)
        return future

    def complete(self, index: int = -1) -> None:
        future = self.futures[index]
        front = SimpleNamespace(sequence=len(self.rendered))
        future.set_result(SimpleNamespace(value=None, front=front))


def _stage_on(host):
    def replace_host(plot_input, _value, _publication):
        return host, host.update_data(plot_input)

    return replace_host


class _Bench:
    """One plane with a camera producer and a derived occupancy signal."""

    def __init__(self) -> None:
        self.plane = SignalDataPlane()
        self.revision = 0
        self.outputs = {"frame": _exact_output("frame", 1)}
        self.node = SimpleNamespace(
            instance_id="camera",
            dataset_output_declarations=(self.outputs["frame"].declaration,),
            signal_key=lambda name: f"camera/{name}",
        )
        self.plane.reserve(self.node)
        # Deliberately NO set_front_signals here: declaring the coherent set
        # is the scheduler's job on tick.  A bench that declares by hand hides
        # a board that never does -- which is exactly how the same-shot suite
        # once passed while the real console presented skewed shots.
        self.publish_shot()  # revision 1 binds the derived route
        root = self.plane.latest_publication(FRAME)
        occupancy = _exact_output("occupancy", 1)
        self.derived_node = SimpleNamespace(
            instance_id="occupancy",
            dataset_output_declarations=(occupancy.declaration,),
            signal_key=lambda _name: OCCUPANCY,
        )
        self.derived_tap = self.plane.reserve_follow_processor(
            self.derived_node,
            source_name=FRAME,
            source_publication=root,
        )
        self.publish_derived()

    def publish_shot(self) -> None:
        self.revision += 1
        self.outputs["frame"] = _exact_output("frame", self.revision)
        self.plane.commit_live(self.node, self.outputs)
        self.plane.freeze()

    def publish_derived(self) -> None:
        root = self.plane.latest_publication(FRAME)
        self.plane.commit_processor(
            self.derived_node,
            {"occupancy": _exact_output("occupancy", self.revision)},
            source_publication=root,
        )

    def close(self) -> None:
        self.derived_tap.close()
        self.plane.close()


def _shot_of(port: PlotPanelPort) -> object | None:
    """The lineage root a panel currently shows -- the same-shot key.

    The frame publication IS a root; the derived publication names its exact
    consumed frame event in ``direct_parent_refs``.  Two panels show the same
    shot exactly when these resolve to one ``EventRef``.
    """

    publication = port.presented_publication()
    if publication is None:
        return None
    parents = tuple(publication.direct_parent_refs)
    return parents[0] if parents else publication.event_ref


def _bench_board(bench: _Bench):
    frame_host = _RenderHost("frame")
    occupancy_host = _RenderHost("occupancy")
    presents: list[tuple[str, object]] = []
    ports: list[PlotPanelPort] = []
    board = LiveBoard(
        bench.plane,
        lambda: tuple(ports),
        intervals=(100, 200, 400, 800),
    )
    frame_port = PlotPanelPort(
        "panel-frame",
        FRAME,
        display_interval_ms=100,
        submit_projection=board.submit_projection,
        replace_host=_stage_on(frame_host),
        present=lambda operation: presents.append(("panel-frame", operation.front)),
    )
    occupancy_port = PlotPanelPort(
        "panel-occupancy",
        OCCUPANCY,
        display_interval_ms=100,
        submit_projection=board.submit_projection,
        replace_host=_stage_on(occupancy_host),
        present=lambda operation: presents.append(("panel-occupancy", operation.front)),
    )
    ports.extend((frame_port, occupancy_port))
    return board, frame_host, occupancy_host, frame_port, occupancy_port, presents


def _assert_same_shot(frame_port, occupancy_port) -> None:
    assert _shot_of(frame_port) == _shot_of(occupancy_port)


def _wait_staged(count: int, *hosts: _RenderHost) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if all(len(host.futures) >= count for host in hosts):
            return
        time.sleep(0.001)
    raise AssertionError("presentation projection did not stage")


def test_mismatched_render_arrival_still_presents_the_group_as_one_shot() -> None:
    bench = _Bench()
    board, frame_host, occupancy_host, frame_port, occupancy_port, presents = (
        _bench_board(bench)
    )
    try:
        board.tick()
        _wait_staged(1, frame_host, occupancy_host)
        assert len(frame_host.futures) == len(occupancy_host.futures) == 1

        # The derived member lands FIRST; its partner is still rendering, so
        # nothing may present -- not even the member that is ready.
        occupancy_host.complete()
        board.commit()
        assert presents == []
        _assert_same_shot(frame_port, occupancy_port)

        frame_host.complete()
        board.commit()
        assert [name for name, _front in presents] == [
            "panel-frame",
            "panel-occupancy",
        ]
        _assert_same_shot(frame_port, occupancy_port)
        assert _shot_of(frame_port) is not None

        # Next shot, opposite arrival order: the frame lands first.
        presents.clear()
        bench.publish_shot()
        bench.publish_derived()
        board.tick()
        _wait_staged(2, frame_host, occupancy_host)
        assert len(frame_host.futures) == len(occupancy_host.futures) == 2
        first_shot = _shot_of(frame_port)
        frame_host.complete()
        board.commit()
        assert presents == []
        assert _shot_of(frame_port) == first_shot  # nobody ran ahead
        occupancy_host.complete()
        board.commit()
        assert len(presents) == 2
        _assert_same_shot(frame_port, occupancy_port)
        assert _shot_of(frame_port) != first_shot
    finally:
        board.close()
        bench.close()


def test_pause_leaves_every_group_member_on_the_same_shot() -> None:
    """Pause stops NEW shots; a batch already travelling still lands whole.

    The presenter keeps committing while paused (only the tick stops), so an
    in-flight group finishes presenting together and the frozen board shows
    exactly one shot.
    """

    bench = _Bench()
    board, frame_host, occupancy_host, frame_port, occupancy_port, presents = (
        _bench_board(bench)
    )
    try:
        board.tick()
        _wait_staged(1, frame_host, occupancy_host)
        frame_host.complete()
        occupancy_host.complete()
        board.commit()
        presents.clear()

        bench.publish_shot()
        bench.publish_derived()
        board.tick()
        _wait_staged(2, frame_host, occupancy_host)

        # Pause NOW: no more ticks.  One member has landed, one is mid-render.
        frame_host.complete()
        board.commit()  # the paused beat still commits
        assert presents == []
        _assert_same_shot(frame_port, occupancy_port)

        occupancy_host.complete()
        board.commit()
        assert len(presents) == 2
        _assert_same_shot(frame_port, occupancy_port)
    finally:
        board.close()
        bench.close()


def test_a_lagging_member_defers_its_partner_by_at_most_one_refresh() -> None:
    """Latest-only coalescing bounds staleness to one refresh.

    When the group's next shot is staged while the previous occupancy render
    is still queued, the host coalesces the old render away; the arbiter then
    abandons the whole superseded batch and the NEWER batch presents both
    members together -- the partner waited one refresh, never diverged.
    """

    bench = _Bench()
    board, frame_host, occupancy_host, frame_port, occupancy_port, presents = (
        _bench_board(bench)
    )
    try:
        board.tick()
        _wait_staged(1, frame_host, occupancy_host)
        # Shot 2 arrives before shot 1's occupancy render finished; the host
        # coalesces the queued render away, cancelling its future.
        bench.publish_shot()
        bench.publish_derived()
        frame_host.complete()  # shot 1's frame DID render
        assert occupancy_host.futures[0].cancel()
        board.tick()
        _wait_staged(2, frame_host, occupancy_host)
        assert len(frame_host.futures) == len(occupancy_host.futures) == 2

        board.commit()
        # The superseded batch was abandoned whole: shot 1 never half-showed.
        assert presents == []
        _assert_same_shot(frame_port, occupancy_port)

        frame_host.complete()
        occupancy_host.complete()
        board.commit()
        assert [name for name, _front in presents] == [
            "panel-frame",
            "panel-occupancy",
        ]
        _assert_same_shot(frame_port, occupancy_port)
    finally:
        board.close()
        bench.close()
