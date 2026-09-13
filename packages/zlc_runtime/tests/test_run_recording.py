"""What a run leaves behind, and what it refuses to let a disk cost it.

Commit is the one place every event of every node passes through, so it is
where recording belongs.  Two properties matter more than the writing: a run
that publishes nothing must leave nothing, and a run whose recording fails
must still be a run.
"""

from __future__ import annotations

from pathlib import Path
from threading import Event

import numpy as np
import pytest

from zlc_runtime.publication_store import PublicationReader
from zlc_runtime.dataset_output import DatasetOutputDeclaration
from zlc_runtime.host import NodeHost
from zlc_runtime.plane import SignalDataPlane
from zlc_runtime.recording import RECORDING_DIRECTORY, RunRecorder

from test_host import _finite_output, _wait


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


def _recorded_run(tmp_path: Path, recorder: RunRecorder, shots: int = 5):
    declaration = DatasetOutputDeclaration("frame", "test.frame")
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
    finally:
        host.shutdown()
        plane.close()
    return recorder


def test_every_committed_event_is_on_disk_when_the_run_ends(tmp_path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    recorder = _recorded_run(tmp_path, RunRecorder(lambda: root, events_per_chunk=2))

    assert recorder.failure is None
    assert recorder.events == 5
    assert recorder.outputs == ("frame",)

    store = PublicationReader(root / RECORDING_DIRECTORY / "frame")
    assert store.events == 5
    values = store.values()
    assert [float(value) for value in values.reshape(5, -1)[:, 0]] == [
        0.0, 1.0, 2.0, 3.0, 4.0
    ]
    assert np.array_equal(store.snapshot(3).block.values,
                          values[3])


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
    assert recorder.events == 0
    assert recorder.outputs == ()


def test_a_monitor_is_not_recorded_because_it_keeps_no_history(tmp_path) -> None:
    """A monitor retains only its latest event, so there is none to write.

    It is also the expensive one: the monitor in a real run is the camera,
    eight megabytes a frame.  Recorded, one chunk is eight gigabytes and the
    console stalls for seconds buffering it -- measured, not feared.
    """

    from test_host import _monitor_output

    declaration = DatasetOutputDeclaration("frame", "test.frame")
    wake = Event()
    plane = SignalDataPlane()
    asked = []

    class Node:
        def execute(self, context):
            context.report_progress("watching")
            for index in range(4):
                context.commit_live({"frame": _monitor_output(declaration, index + 1)})
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


def test_a_recorder_needs_a_way_to_open_its_directory() -> None:
    with pytest.raises(TypeError):
        RunRecorder("not a callable")  # type: ignore[arg-type]
