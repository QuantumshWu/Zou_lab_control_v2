"""A node can be run more than once.

A producer generation belongs to ONE run, but the node that performs the run is
a reusable object.  Something must decide when the previous generation ends, and
it cannot be ``publish_final``: the result has to stay readable afterwards,
which is precisely why nothing ever retired it and why a second run was
impossible.

``begin_generation`` is that decision, in one place.  Before it existed the
policy lived only inside NodeHost as a retire-then-reserve pair, so every other
caller -- including the entire domain package -- could only ``reserve``, and
could therefore only ever run once.
"""

from __future__ import annotations

import numpy as np
import pytest

from zlc_runtime.dataset_output import DatasetOutputDeclaration, FinalDatasetOutput
from zlc_runtime.plane import SignalDataPlane

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
    plane.publish_final(
        node,
        {"frames": FinalDatasetOutput(node.declaration, snapshot("probe", revision))},
    )


def _latest_value(plane: SignalDataPlane, node: _Producer) -> float:
    publication = plane.latest_publication(node.signal_key("frames"))
    assert publication is not None
    value = publication.value(node.signal_key("frames"))
    assert value is not None
    values = value.snapshot.block.values
    return float(np.asarray(values).reshape(-1)[0])


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
    """Why the generation cannot end at publish_final.

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


def test_reserve_stays_idempotent_before_anything_is_published() -> None:
    plane = SignalDataPlane()
    node = _Producer()
    try:
        assert plane.reserve(node) == plane.reserve(node)
    finally:
        plane.close()
