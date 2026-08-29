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
from zlc_runtime.streams import SourceGenerationEnded, StreamEndedEarly

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


def test_a_live_generation_is_never_superseded_underneath_a_running_shot() -> None:
    """Two concurrent runs of one producer is a real error, not a restart.

    ``begin_generation`` replaces a FINISHED predecessor; a generation that
    has published and not ended is still running, and silently discarding
    it would drop the shot in flight.
    """

    plane = SignalDataPlane()
    node = _Producer()
    try:
        plane.begin_generation(node)
        # Committed but NOT sealed: the shot is in flight.
        plane.commit_live(
            node,
            {"frames": _commit(node, revision=1, total=1, origin=0)},
        )
        with pytest.raises(RuntimeError, match="already active"):
            plane.begin_generation(node)
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


def test_concurrent_generation_starters_share_one_installed_successor() -> None:
    plane = SignalDataPlane()
    node = _Producer()
    release_cancel = threading.Event()
    release_work = threading.Event()
    results: list[object] = []
    retired: list[frozenset[str]] = []
    errors: list[BaseException] = []

    class Dependent:
        def __init__(self, identity: str, *, busy: bool) -> None:
            self.instance_id = identity
            self.declaration = DatasetOutputDeclaration("value", "test.value")
            self.busy = busy
            self.entered = threading.Event()
            self.cancelled = threading.Event()
            self.cancel_count = 0
            self.wake = threading.Event()

        @property
        def dataset_output_declarations(self):
            return (self.declaration,)

        def signal_key(self, name: str) -> str:
            return f"@logic/{self.instance_id}/{name}"

        validate_processor_source = staticmethod(lambda _source: None)

        def evaluate_processor(self, *_args):
            if not self.busy:
                raise AssertionError("paused processor must not evaluate")
            self.entered.set()
            assert release_work.wait(2.0)
            return {"value": object()}

        accept_processor_result = staticmethod(
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("cancelled result must not be accepted")
            )
        )
        accept_processor_failure = staticmethod(
            lambda error: (_ for _ in ()).throw(error)
        )

        def accept_processor_cancelled(self) -> None:
            self.cancel_count += 1
            self.cancelled.set()
            if not self.busy:
                assert release_cancel.wait(2.0)

        def request_processor_owner_wake(self) -> None:
            self.wake.set()

    gate = Dependent("cleanup-gate", busy=False)
    busy = Dependent("cleanup-busy", busy=True)
    try:
        plane.begin_generation(node)
        plane.commit_live(
            node,
            {"frames": _commit(node, revision=1, total=1, origin=0)},
        )
        publication = plane.latest_publication(node.signal_key("frames"))
        assert publication is not None
        plane.attach_latest_only_processor(
            gate,
            source_name=node.signal_key("frames"),
            initial_publication=publication,
            paused=True,
        )
        plane.attach_latest_only_processor(
            busy,
            source_name=node.signal_key("frames"),
            initial_publication=publication,
        )
        assert busy.entered.wait(2.0)
        assert plane.seal_committed(node)

        def start_successor() -> None:
            try:
                results.append(plane.begin_generation(node))
            except BaseException as error:
                errors.append(error)

        def withdraw_dependent() -> None:
            try:
                retired.append(plane.retire(gate))
            except BaseException as error:
                errors.append(error)

        leader = threading.Thread(target=start_successor)
        follower = threading.Thread(target=start_successor)
        dependent_withdraw = threading.Thread(target=withdraw_dependent)
        leader.start()
        assert gate.cancelled.wait(2.0)
        follower.start()
        dependent_withdraw.start()
        follower.join(0.05)
        dependent_withdraw.join(0.05)
        assert follower.is_alive() and dependent_withdraw.is_alive()
        with pytest.raises(SourceGenerationEnded, match="retired"):
            plane.commit_live(
                node,
                {"frames": _commit(node, revision=2, total=1, origin=0)},
            )
        release_cancel.set()
        leader.join(2.0)
        follower.join(2.0)
        dependent_withdraw.join(2.0)

        assert not leader.is_alive() and not follower.is_alive()
        assert not dependent_withdraw.is_alive()
        assert not errors
        assert len(results) == 2
        assert results[0] == results[1]
        assert retired == [frozenset()]

        next_value = plane.commit_live(
            node,
            {"frames": _commit(node, revision=2, total=1, origin=0)},
        )[node.signal_key("frames")]
        next_publication = plane.latest_publication(node.signal_key("frames"))
        assert next_publication is not None
        assert next_value.snapshot.ref.stream_generation == results[0]

        with pytest.raises(RuntimeError, match="already active"):
            plane.attach_latest_only_processor(
                busy,
                source_name=node.signal_key("frames"),
                initial_publication=next_publication,
                paused=True,
            )

        release_work.set()
        assert busy.wake.wait(2.0)
        plane.freeze()
        assert busy.cancelled.is_set()
        assert gate.cancel_count == busy.cancel_count == 1

        plane.attach_latest_only_processor(
            busy,
            source_name=node.signal_key("frames"),
            initial_publication=next_publication,
            paused=True,
        )
    finally:
        release_cancel.set()
        release_work.set()
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


def test_an_armed_silent_generation_is_followable() -> None:
    """An armed chain that has not published yet accepts a follower.

    An externally triggered camera publishes nothing until something fires
    its triggers; a consumer that will itself cause the first frame (a scan
    firing its own pulse) must be able to subscribe to that silence.  The
    baseline is None exactly then, the first commit arrives through the
    tap, and retiring the generation stays loud.
    """

    from zlc_runtime.streams import SourceFailed

    plane = SignalDataPlane()
    node = _Producer()
    try:
        plane.begin_generation(node)
        baseline, tap = plane.follow_publications(
            node.signal_key("frames"), replay=False
        )
        assert baseline is None
        plane.commit_live(
            node,
            {"frames": _commit(node, revision=7, total=2, origin=0)},
        )
        first = tap.next(1.0)
        assert first.event_ref.sequence == 1
        plane.retire(node)
        with pytest.raises(SourceFailed):
            tap.next(1.0)
        # A retired generation leaves no state behind: following it
        # again is following an unknown signal.
        with pytest.raises(LookupError):
            plane.follow_publications(
                node.signal_key("frames"), replay=False
            )
        with pytest.raises(LookupError):
            plane.follow_publications("nobody:frames", replay=False)
    finally:
        plane.close()
