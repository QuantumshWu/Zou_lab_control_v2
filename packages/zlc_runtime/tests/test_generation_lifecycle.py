"""A node can be run more than once.

A producer generation belongs to ONE run, but the node that performs the run is
a reusable object.  ``seal_committed`` retains one finished result;
``begin_generation`` replaces it only when the next run starts.

``begin_generation`` is that decision, in one place.  Before it existed the
policy lived only inside NodeHost as a retire-then-reserve pair, so every other
caller -- including the entire domain package -- could only ``reserve``, and
could therefore only ever run once.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from zlc_data import AxisSpec, DatasetSchema
from zlc_runtime.dataset import DatasetCoverage, MonitorCoverage
from zlc_runtime.dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)
from zlc_runtime.plane import SignalDataPlane
from zlc_runtime.streams import StreamEndedEarly

from _snapshots import snapshot


class _Producer:
    """The smallest object the plane accepts as a producer."""

    instance_id = "probe"
    declaration = DatasetOutputDeclaration("frames", "test.frames")

    @property
    def dataset_output_declarations(self):
        return (self.declaration,)

    def signal_key(self, name: str) -> str:
        return f"@logic/{self.instance_id}/{name}"


def _publish(plane: SignalDataPlane, node: _Producer, revision: int) -> None:
    plane.commit_live(
        node,
        {"frames": _commit(node, revision=revision, total=1, origin=0)},
    )
    plane.seal_committed(node)


def _commit(
    node: _Producer,
    *,
    revision: int,
    total: int,
    origin: int,
) -> LiveDatasetOutput:
    event = snapshot("probe", revision)
    event_schema = event.block.schema
    repeat = event_schema.repeat_axis
    canonical = DatasetSchema(
        AxisSpec(
            repeat.axis_id,
            repeat.name,
            repeat.role,
            total,
            tuple(range(total)),
        ),
        event_schema.point_table,
        event_schema.grid_topology,
        event_schema.cell_schema,
    )
    return LiveDatasetOutput(
        node.declaration,
        event,
        DatasetCoverage(origin + 1, total),
        canonical_schema=canonical,
        cell_origin=(origin, 0),
    )


def _latest_value(plane: SignalDataPlane, node: _Producer) -> float:
    publication = plane.latest_publication(node.signal_key("frames"))
    assert publication is not None
    value = publication.value(node.signal_key("frames"))
    assert value is not None
    values = value.snapshot.block.values
    return float(np.asarray(values).reshape(-1)[0])


def test_retained_monitor_keeps_its_last_value_until_the_next_run() -> None:
    plane = SignalDataPlane()
    node = _Producer()
    signal = node.signal_key("frames")
    try:
        first = plane.begin_generation(node)
        plane.commit_live(
            node,
            {
                "frames": LiveDatasetOutput(
                    node.declaration,
                    snapshot("probe", 1),
                    MonitorCoverage(1, 1, retain_at_terminal=True),
                )
            },
        )
        assert plane.seal_committed(node)
        assert _latest_value(plane, node) == 1.0
        assert not plane.is_generation_live(signal)

        second = plane.begin_generation(node)
        assert second != first
        plane.commit_live(
            node,
            {
                "frames": LiveDatasetOutput(
                    node.declaration,
                    snapshot("probe", 2),
                    MonitorCoverage(1, 1, retain_at_terminal=True),
                )
            },
        )
        assert plane.seal_committed(node)
        assert _latest_value(plane, node) == 2.0
    finally:
        plane.close()


def test_transient_monitor_still_retires_at_terminal() -> None:
    plane = SignalDataPlane()
    node = _Producer()
    try:
        plane.begin_generation(node)
        plane.commit_live(
            node,
            {
                "frames": LiveDatasetOutput(
                    node.declaration,
                    snapshot("probe", 1),
                    MonitorCoverage(1, 1),
                )
            },
        )
        assert not plane.seal_committed(node)
        assert plane.latest_publication(node.signal_key("frames")) is None
    finally:
        plane.close()


def test_reserve_alone_can_never_run_the_same_node_twice() -> None:
    """The behaviour that made a second shot impossible, pinned deliberately.

    ``reserve`` is the low-level operation and it is RIGHT to refuse -- it must
    not silently discard a generation.  The defect was that it was the only
    operation available, so a caller had no way to say "the previous run is
    over" and every domain node could fire exactly once per process.
    """

    plane = SignalDataPlane()
    node = _Producer()
    try:
        plane.reserve(node)
        _publish(plane, node, 1)
        with pytest.raises(RuntimeError, match="already active"):
            plane.reserve(node)
    finally:
        plane.close()


def test_begin_generation_supersedes_a_finished_run_so_a_node_runs_again() -> None:
    plane = SignalDataPlane()
    node = _Producer()
    try:
        first = plane.begin_generation(node)
        _publish(plane, node, 1)
        assert _latest_value(plane, node) == 1.0

        second = plane.begin_generation(node)
        assert second != first, "a new run gets a new generation, it does not reuse the old one"
        _publish(plane, node, 2)
        assert _latest_value(plane, node) == 2.0
    finally:
        plane.close()


def test_a_node_runs_three_times_in_a_row() -> None:
    """Three shots in a row -- the acceptance the owner asked for by name."""

    plane = SignalDataPlane()
    node = _Producer()
    seen: list[float] = []
    try:
        for shot in (1, 2, 3):
            plane.begin_generation(node)
            _publish(plane, node, shot)
            seen.append(_latest_value(plane, node))
    finally:
        plane.close()
    assert seen == [1.0, 2.0, 3.0]


def test_a_finished_result_stays_readable_until_the_next_run_starts() -> None:
    """A sealed result remains until the next run replaces its generation.

    The caller reads its result after the run completes.  Superseding happens at
    the START of the next run, not at the end of the previous one.
    """

    plane = SignalDataPlane()
    node = _Producer()
    try:
        plane.begin_generation(node)
        _publish(plane, node, 7)
        assert _latest_value(plane, node) == 7.0
        assert _latest_value(plane, node) == 7.0
    finally:
        plane.close()


def test_begin_generation_on_an_untouched_generation_is_the_same_run() -> None:
    """Starting a run twice before it produces anything is not a new run."""

    plane = SignalDataPlane()
    node = _Producer()
    try:
        first = plane.begin_generation(node)
        # Nothing has been published, so this is the SAME run being started
        # again, not a second one: the plane has no notion of in-flight work and
        # reserve is idempotent while the generation is untouched.  Concurrency
        # is the caller state machine's job, which is why NodeHost has _active.
        assert plane.begin_generation(node) == first
    finally:
        plane.close()


def test_concurrent_generation_starters_share_one_installed_successor(
    monkeypatch,
) -> None:
    plane = SignalDataPlane()
    node = _Producer()
    start = threading.Barrier(3)
    withdraw_arrived = threading.Barrier(2)
    successor_returned = threading.Event()
    role_lock = threading.Lock()
    leader: list[int] = []
    results: list[object] = []
    errors: list[BaseException] = []
    real_withdraw = plane._withdraw_owner

    def gated_withdraw(owner_id: str):
        identity = threading.get_ident()
        with role_lock:
            if not leader:
                leader.append(identity)
        withdraw_arrived.wait(timeout=2.0)
        if identity != leader[0]:
            assert successor_returned.wait(2.0), (
                "first starter did not return its successor"
            )
        return real_withdraw(owner_id)

    monkeypatch.setattr(plane, "_withdraw_owner", gated_withdraw)
    try:
        plane.begin_generation(node)
        _publish(plane, node, 1)

        def start_successor() -> None:
            identity = threading.get_ident()
            try:
                start.wait(timeout=2.0)
                generation = plane.begin_generation(node)
                results.append(generation)
                if leader and identity == leader[0]:
                    successor_returned.set()
            except BaseException as error:
                errors.append(error)
                successor_returned.set()

        workers = tuple(
            threading.Thread(target=start_successor) for _index in range(2)
        )
        for worker in workers:
            worker.start()
        start.wait(timeout=2.0)
        for worker in workers:
            worker.join(2.0)

        assert all(not worker.is_alive() for worker in workers)
        assert not errors
        assert len(results) == 2
        assert results[0] == results[1]

        value = plane.commit_live(
            node,
            {"frames": _commit(node, revision=2, total=1, origin=0)},
        )[node.signal_key("frames")]
        assert value.snapshot.ref.stream_generation == results[0]
    finally:
        successor_returned.set()
        plane.close()


def test_reserve_stays_idempotent_before_anything_is_published() -> None:
    plane = SignalDataPlane()
    node = _Producer()
    try:
        assert plane.reserve(node) == plane.reserve(node)
    finally:
        plane.close()


def test_committed_seal_closes_follow_without_a_duplicate_full_event() -> None:
    plane = SignalDataPlane()
    node = _Producer()
    received = []
    try:
        plane.begin_generation(node)
        plane.commit_live(
            node,
            {"frames": _commit(node, revision=81, total=3, origin=0)},
        )
        plane.commit_live(
            node,
            {"frames": _commit(node, revision=81, total=3, origin=1)},
        )
        baseline, tap = plane.follow_publications(node.signal_key("frames"))
        assert baseline.event_ref.sequence == 2

        plane.commit_live(
            node,
            {"frames": _commit(node, revision=81, total=3, origin=2)},
        )
        first = tap.next(1.0)
        replayed = tap.next(1.0)
        publication = tap.next(1.0)
        assert [
            first.event_ref.sequence,
            replayed.event_ref.sequence,
            publication.event_ref.sequence,
        ] == [1, 2, 3]
        assert not hasattr(publication, "payload")
        received.append(
            plane.current_dataset(node.signal_key("frames"), publication)
        )
        assert received[0].block.values.reshape(-1).tolist() == [81.0] * 3
        assert plane.seal_committed(node)
        with pytest.raises(StreamEndedEarly):
            tap.next(0.0)
        latest = plane.latest_publication(node.signal_key("frames"))
        assert latest is publication
        assert latest.event_ref.sequence == 3
        assert not plane.is_generation_live(node.signal_key("frames"))
    finally:
        plane.close()


def test_sealed_commit_generation_is_superseded_by_the_next_run() -> None:
    plane = SignalDataPlane()
    node = _Producer()
    try:
        first = plane.begin_generation(node)
        plane.commit_live(
            node,
            {"frames": _commit(node, revision=1, total=1, origin=0)},
        )
        plane.seal_committed(node)
        assert plane.current_dataset(node.signal_key("frames")).block.values.item() == 1

        second = plane.begin_generation(node)
        assert second != first
        plane.commit_live(
            node,
            {"frames": _commit(node, revision=2, total=1, origin=0)},
        )
        plane.seal_committed(node)
        assert plane.current_dataset(node.signal_key("frames")).block.values.item() == 2
    finally:
        plane.close()


def test_current_dataset_rejects_a_same_sequence_publication_from_the_old_run() -> None:
    """A sequence number is meaningful only inside its exact stream generation."""

    plane = SignalDataPlane()
    node = _Producer()
    try:
        plane.begin_generation(node)
        plane.commit_live(
            node,
            {"frames": _commit(node, revision=1, total=1, origin=0)},
        )
        old_publication = plane.latest_publication(node.signal_key("frames"))
        assert old_publication is not None
        assert old_publication.event_ref.sequence == 1
        plane.seal_committed(node)

        plane.begin_generation(node)
        plane.commit_live(
            node,
            {"frames": _commit(node, revision=2, total=1, origin=0)},
        )
        current_publication = plane.latest_publication(node.signal_key("frames"))
        assert current_publication is not None
        assert current_publication.event_ref.sequence == 1
        assert (
            current_publication.event_ref.generation
            != old_publication.event_ref.generation
        )

        with pytest.raises(ValueError, match="another signal generation"):
            plane.current_dataset(node.signal_key("frames"), old_publication)
    finally:
        plane.close()
