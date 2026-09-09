"""What exists, offered to a window that must not know what any of it is.

The console could not answer "what can I put on a panel?".  Its Add Panel
button was connected to nothing and every panel was named by hand in the
composition root, so a signal that appeared while the experiment ran was
invisible unless someone had written its name down in advance.

Two properties are worth holding still:

* the projection carries only strings and bools.  A window handed live plane
  state would read it whenever it happened to paint and show a mixture of two
  instants;
* what is still arriving is offered first, because that is what an operator is
  most likely to want on screen.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_atom.nodes.scan.dataset import scan_dataset_schema
from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    SITE,
    AxisId,
    AxisSpec,
    DatasetSchema,
    DomainSpec,
    ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    LiveDatasetOutput,
    SignalDataPlane,
)
from zlc_workbench.logic import stable_signal_key
from zlc_workbench.session import ExperimentSession
from zlc_workbench.topology import SignalRow, format_signal_shape, project_signals
from pulse_fixtures import CAMERA_WINDOWS, PULSE_NAME, write_ordinary_pulse


@pytest.fixture
def plane():
    """A bare plane: the projection reads its descriptions and nothing else."""

    plane = SignalDataPlane()
    try:
        yield plane
    finally:
        plane.close()


def _finished_frames(plane, producer: str = "cm") -> str:
    """One stopped scan's ``frames`` Dataset; its signal name.

    Three source frames sit inside a single power coordinate, with fifty
    authored repeats. The first event and stopped run retain that full
    schema without needing the apparatus that would have acquired them.
    """

    declaration = DatasetOutputDeclaration("frames", "camera.frames")
    node = SimpleNamespace(
        instance_id=producer,
        dataset_output_declarations=(declaration,),
        signal_key=lambda name: stable_signal_key(producer, name),
    )
    repeat = AxisSpec(AxisId(f"{producer}.repeat"), "repeat", REPEAT, 1, (0,))
    point = AxisSpec(AxisId(f"{producer}.point"), "frame", READOUT_EVENT, 3, (0, 1, 2))
    site = AxisSpec(AxisId(f"{producer}.site"), "site", SITE, 3, (0, 1, 2))
    schema = DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec((3,), (point,), ((0, 1, 2),)),
        DomainSpec((3,), (site,)),
        ValueSchema.scalar(np.dtype("<f8")),
    )
    snapshot = owned_snapshot_from_arrays(schema, np.zeros((1, 3, 3)), 0)
    canonical = scan_dataset_schema(
        schema, ((135.0,),), (("power", "mVpp"),), run_repeats=50,
    )
    plane.begin_generation(node)
    plane.commit_live(
        node,
        {"frames": LiveDatasetOutput(
            declaration, snapshot, DatasetCoverage(3, 150), {}, canonical, (0, 0),
        )},
    )
    plane.seal_committed(node, cut_short=True)
    return node.signal_key("frames")


@pytest.fixture
def session(tmp_path):
    """The virtual apparatus, for the one test about a run still arriving."""

    write_ordinary_pulse(tmp_path)
    session = ExperimentSession.open(tmp_path, template="virtual")
    try:
        yield session
    finally:
        session.close()


def _measure(session, producer: str):
    session.load_pulse(PULSE_NAME)
    node = CameraMeasurementNode(
        camera=session.camera,
        request=CameraMeasurementRequest("camera", 0.02, None, 1, CAMERA_WINDOWS),
        signal_plane=session.signal_plane,
        producer=producer,
    )
    capture = node.prepare()
    session.fire(shots=1)
    capture.collect()
    return node


def test_a_finished_measurement_is_offerable_and_says_it_is_finished(plane) -> None:
    signal = _finished_frames(plane)
    rows = project_signals(plane)
    assert rows, "a run that produced data offered nothing to look at"
    row = next(row for row in rows if row.name == signal)
    description = next(item for item in plane.describe_signals() if item.name == signal)
    assert description.shape == (50, 3, 3)
    assert row.label == f"frames  [{format_signal_shape(description.schema)}]"
    assert row.label == (
        "frames  [(1 × 50) × (3 × 1) × (3)]"
    )
    scan_axis = description.schema.point_domain.axes[-1]
    assert (scan_axis.name, scan_axis.size, scan_axis.coordinates, scan_axis.unit) == (
        "power", 1, (135.0,), "mVpp",
    )
    assert row.producer == "cm"
    assert row.state == "finished"
    assert row.derived_from == ""
    assert not row.shown


def test_a_reserved_output_waits_until_its_first_publication() -> None:
    plane = SimpleNamespace(
        describe_signals=lambda: (
            SimpleNamespace(
                name="@logic/camera/frames",
                owner_id="camera",
                shape=None,
                schema=None,
                failure=None,
                live=True,
                source_name=None,
            ),
        )
    )
    (row,) = project_signals(plane)
    assert row.state == "waiting"
    assert row.label == "frames  [—]"


def test_a_live_monitor_is_offered_before_a_finished_run(session) -> None:
    """Ordering is a decision: what is still arriving is what you want first."""

    _measure(session, producer="done")
    session.load_pulse(PULSE_NAME)
    watching = CameraMeasurementNode(
        camera=session.camera,
        request=CameraMeasurementRequest("camera", 0.02, None, 0, CAMERA_WINDOWS),
        signal_plane=session.signal_plane,
        producer="watching",
    )
    monitor = watching.monitor()
    try:
        session.fire(shots=1)
        live_signal = watching.signal_key("frames")
        seen = 0
        deadline = time.monotonic() + 10.0
        while seen < CAMERA_WINDOWS and time.monotonic() < deadline:
            if monitor.poll() is None:
                time.sleep(0.001)
                continue
            seen += 1
            if seen < CAMERA_WINDOWS:
                assert session.signal_plane.latest_publication(live_signal) is None
        assert seen == CAMERA_WINDOWS
        rows = project_signals(session.signal_plane)
        states = [row.state for row in rows]
        assert states.index("live") < states.index("finished")
    finally:
        monitor.close()


def test_a_panel_already_showing_a_signal_says_so(plane) -> None:
    signal = _finished_frames(plane)
    rows = project_signals(plane, shown={signal})
    assert next(row for row in rows if row.name == signal).shown


def test_only_plain_values_cross(plane) -> None:
    """The rule that keeps a window from reading the plane directly."""

    _finished_frames(plane)
    for row in project_signals(plane):
        assert isinstance(row, SignalRow)
        assert isinstance(row.name, str)
        assert isinstance(row.label, str)
        assert isinstance(row.producer, str)
        assert isinstance(row.state, str)
        assert isinstance(row.derived_from, str)
        assert isinstance(row.shown, bool)
        with pytest.raises((AttributeError, TypeError)):
            row.name = "mutated"


def test_this_module_holds_no_domain_knowledge() -> None:
    """It names things for a person; it must not decide what they mean."""

    import ast

    import zlc_workbench.topology as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(
        name.startswith(("zlc_atom", "zlc_runtime", "PyQt5"))
        for name in imported
    ), imported
