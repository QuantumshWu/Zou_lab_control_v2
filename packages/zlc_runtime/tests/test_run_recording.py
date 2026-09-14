"""What a run leaves behind, and what it refuses to let a disk cost it.

Commit is the one place every event of every node passes through, so it is
where recording belongs.  Two properties matter more than the writing: a run
that publishes nothing must leave nothing, and a run whose recording fails
must still be a run.
"""

from __future__ import annotations

from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest

from zlc_runtime.publication_store import PublicationReader
from zlc_runtime.dataset_output import DatasetOutputDeclaration
from zlc_runtime.host import NodeHost
from zlc_runtime.plane import SignalDataPlane
from zlc_runtime.recording import RunRecorder

from test_host import _finite_output, _wait


def _settled(recorder: RunRecorder, timeout: float = 10.0) -> RunRecorder:
    """Wait for what was handed over to reach the disk.

    The flush a terminal asks for is asked, not awaited -- the threads that
    reach a terminal are the ones polling the hosts, and on the console that
    is Qt's.  What the contract promises is that the tail lands shortly
    after, which is what this waits for.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if recorder.durable_events == recorder.events or recorder.failure:
            return recorder
        time.sleep(0.01)
    raise AssertionError(
        f"{recorder.durable_events} of {recorder.events} events reached disk"
    )


def _host(node, plane, wake, *, instance_id, recorder, outputs=()):
    """A hosted measurement that records where it is told to.

    Built here rather than through the lifecycle suite's helper: the whole
    subject is the argument that helper does not take, and a test that sets
    it afterwards would be asserting a field instead of a constructor.
    """

    return NodeHost(
        node,
        plane,
        wake.set,
        instance_id=instance_id,
        kind="measurement",
        dataset_output_declarations=outputs,
        recorder=recorder,
    )


def _recorded_run(tmp_path: Path, recorder: RunRecorder, shots: int = 5, *, alive=None):
    """``alive`` is asked with the host while the run is still standing."""

    declaration = DatasetOutputDeclaration("frame", "test.frame", recorded=True)
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        def execute(self, context):
            context.report_progress("capturing", current=shots, total=shots)
            for index in range(shots):
                context.commit_live(
                    {
                        "frame": _finite_output(
                            declaration,
                            value=float(index),
                            total=shots,
                            origin=index,
                            written=index + 1,
                        )
                    }
                )
            return {"status": "ok"}

    host = _host(
        Node(),
        plane,
        wake,
        instance_id="camera",
        outputs=(declaration,),
        recorder=recorder,
    )
    try:
        host.start()
        observation = _wait(host, wake)
        assert observation.phase == "done", observation
        if alive is not None:
            alive(host)
    finally:
        host.shutdown()
        plane.close()
    return recorder


def test_a_run_s_events_are_on_disk_while_it_lives_and_gone_when_it_is_retired(tmp_path) -> None:
    """The recording is the run's scratch: complete while the run stands,
    deleted with the run.  A retired run has no further use for it, and a
    folder per restart that nobody opens is what this used to leave."""

    root = tmp_path / "run"
    root.mkdir()
    recorder = RunRecorder(lambda: root, events_per_chunk=2)

    def while_alive(host) -> None:
        _settled(recorder)
        assert recorder.events == 5
        assert recorder.outputs == ("frame",)
        store = PublicationReader(root / "frame")
        assert store.events == 5
        values = store.values()
        assert [float(value) for value in values.reshape(5, -1)[:, 0]] == [
            0.0, 1.0, 2.0, 3.0, 4.0
        ]
        assert np.array_equal(store.snapshot(3).block.values, values[3])

    _recorded_run(tmp_path, recorder, alive=while_alive)
    assert recorder.failure is None
    assert not root.exists(), "a retired run's recording is deleted with it"


def test_a_run_that_publishes_nothing_leaves_nothing_behind(tmp_path) -> None:
    """Allocating a numbered run folder is itself a durable claim.

    A node configured, started and stopped without publishing has not made a
    run, and a folder saying it did is worse than no folder: it is a folder
    somebody has to open to find out it is empty.
    """

    asked = []

    def allocate() -> Path:
        asked.append(True)
        directory = tmp_path / "never"
        directory.mkdir()
        return directory

    wake = Event()
    plane = SignalDataPlane()

    class Node:
        def execute(self, context):
            context.report_progress("nothing to do")
            return {"status": "ok"}

    host = _host(
        Node(), plane, wake, instance_id="idle",
        recorder=RunRecorder(allocate),
    )
    try:
        host.start()
        _wait(host, wake)
    finally:
        host.shutdown()
        plane.close()
    assert not asked
    assert not (tmp_path / "never").exists()


def test_a_recording_that_cannot_be_written_does_not_stop_the_run(tmp_path) -> None:
    """Losing the recording is bad; losing the experiment is worse.

    The order is a decision, so it is asserted: the run completes, and the
    reason the recording stopped is kept where somebody can read it instead
    of disappearing into a bare except.
    """

    def allocate() -> Path:
        raise PermissionError("the data disk is read-only")

    recorder = RunRecorder(allocate)
    _recorded_run(tmp_path, recorder, shots=3)

    assert isinstance(recorder.failure, PermissionError)
    # ``events`` counts what was handed over, so the first shot is in it: the
    # disk is refused on the writing thread, after the producer has moved on.
    # What must be zero is what reached disk.
    assert recorder.durable_events == 0
    assert recorder.outputs == ()
    # And somebody is told, exactly once: a recording that stops itself and
    # says nothing is indistinguishable from a complete one.
    assert recorder.take_failure() is recorder.failure
    assert recorder.take_failure() is None


def test_an_output_that_does_not_ask_to_be_recorded_is_not(tmp_path) -> None:
    """The camera's finite output is its raw frames, and it must not land.

    Four and a half megabytes a frame: a thousand-event chunk is four
    gigabytes buffered before a byte reaches the disk.  So nothing is written
    unless the declaration says so, and the producer is the only thing that
    knows whether what it accumulates is the science or the pixels it read.
    """

    declaration = DatasetOutputDeclaration("frames", "test.frames")
    assert declaration.recorded is False
    wake = Event()
    plane = SignalDataPlane()
    asked = []

    class Node:
        def execute(self, context):
            context.report_progress("capturing", current=3, total=3)
            for index in range(3):
                context.commit_live(
                    {
                        "frames": _finite_output(
                            declaration, value=float(index), total=3,
                            origin=index, written=index + 1,
                        )
                    }
                )
            return {"status": "ok"}

    recorder = RunRecorder(lambda: asked.append(True) or tmp_path)
    host = _host(
        Node(), plane, wake, instance_id="camera",
        outputs=(declaration,), recorder=recorder,
    )
    try:
        host.start()
        assert _wait(host, wake).phase == "done"
    finally:
        host.shutdown()
        plane.close()
    assert recorder.events == 0
    assert recorder.failure is None
    assert not asked


def test_a_second_generation_of_one_host_is_recorded_too(tmp_path) -> None:
    """A host has as many generations as its source has.

    A processor whose source ends is refused into CANCELLED and a standing
    re-follow starts the SAME host again.  Closed at the first terminal, the
    recording would end there and say nothing about it; what a terminal owes
    is the buffered tail on disk, and the recording ends with the host.
    """

    root = tmp_path / "run"
    root.mkdir()
    declaration = DatasetOutputDeclaration("frame", "test.frame", recorded=True)
    wake = Event()
    plane = SignalDataPlane()
    recorder = RunRecorder(lambda: root, events_per_chunk=2)

    class Node:
        def execute(self, context):
            context.report_progress("capturing", current=1, total=1)
            context.commit_live(
                {
                    "frame": _finite_output(
                        declaration, value=1.0, total=1, origin=0, written=1
                    )
                }
            )
            return {"status": "ok"}

    host = _host(
        Node(), plane, wake, instance_id="camera",
        outputs=(declaration,), recorder=recorder,
    )
    try:
        for _generation in range(3):
            host.start()
            assert _wait(host, wake).phase == "done"
            # Every terminal leaves what it committed on disk, not just the
            # last one: this is what a flush buys over a close.
            _settled(recorder)
            assert PublicationReader(root / "frame").events == recorder.events
        assert recorder.events == 3
    finally:
        host.shutdown()
        plane.close()

    assert recorder.failure is None
    assert not root.exists(), "the recording ends with the host, and goes with it"


def test_a_recorder_needs_a_way_to_open_its_directory() -> None:
    with pytest.raises(TypeError):
        RunRecorder("not a callable")  # type: ignore[arg-type]
