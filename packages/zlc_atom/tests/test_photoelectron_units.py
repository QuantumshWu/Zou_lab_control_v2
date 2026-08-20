"""Counts are what the sensor says; photoelectrons are what they mean.

The conversion is the CAMERA's -- an offset and what one count is worth --
and it is written down on the DEVICE, beside its ROI and its exposure,
because it is a fact about that sensor rather than about any one run.  Read
back from a camera it would only be knowable once something is open, and a
form has to be able to offer the unit before anything has been armed.

So a measurement offers the two units its bound camera can answer for, with
that camera's own numbers in the option that uses them, and a bench that has
written none down cannot start a run in photoelectrons at all.  The
conversion itself is applied at the one read every frame comes through, so a
run is in one unit everywhere or in none.  It is affine, so it moves no
decision: what it buys is numbers a physicist can read, and what it costs is
float32 frames, which is why it is a choice and not a rule.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from zlc_atom.devices.camera.dcam import DcamCameraConfig
from zlc_atom.devices.camera.device_types import DCAM_CAMERA_SCHEMA
from zlc_atom.devices.camera.photoelectrons import (
    PHOTOELECTRONS,
    resolve_photoelectron_availability,
    stated_conversion,
)
from zlc_atom.install import create_installation
from zlc_atom.nodes.calibration.task import CalibrationTask
from zlc_atom.nodes.occupancy import OccupancyProcessor

from tests.fakes import FakePlane, camera_cycle_snapshot
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE
from test_installation_and_nodes import _calibration_request


def _task(request, directory: Path) -> CalibrationTask:
    installation = create_installation("virtual")
    return CalibrationTask(
        camera=installation.device("camera"),
        sequencer=installation.device("sequencer"),
        request=request,
        pulse_sequence=IMAGING_PULSE_RESOURCE.value,
        pulse_path=IMAGING_PULSE_RESOURCE.path,
        signal_plane=FakePlane(),
        artifact_directory=directory,
    )


def test_the_qcmos_states_the_conversion_its_configuration_gives_it() -> None:
    """Authored, not probed, and refused when only half of it is written down.

    The ORCA-Quest's datasheet numbers are what the device form starts at, so
    a bench that has one installed can ask for photoelectrons without hunting
    for them -- and clearing them is how an operator says this sensor states
    no conversion at all.
    """

    authored = DCAM_CAMERA_SCHEMA.project_values({})
    assert authored["offset_counts"] == pytest.approx(200.0)
    assert authored["electrons_per_count"] == pytest.approx(0.107)
    assert DCAM_CAMERA_SCHEMA.project_values(
        {"offset_counts": None, "electrons_per_count": None}
    )["electrons_per_count"] is None

    configured = DcamCameraConfig(
        offset_counts=authored["offset_counts"],
        electrons_per_count=authored["electrons_per_count"],
    )
    assert stated_conversion(
        configured.offset_counts,
        configured.electrons_per_count,
        camera="qCMOS camera",
    ) == (200.0, 0.107)
    assert DcamCameraConfig().offset_counts is None
    with pytest.raises(ValueError, match="both an offset"):
        DcamCameraConfig(offset_counts=200.0)


def test_the_camera_states_its_own_conversion() -> None:
    """The virtual sensor answers with the numbers it applies going the other way."""

    installation = create_installation("virtual")
    try:
        point = installation.device("camera").working_point()
        world = installation.world
        assert point.offset_counts == pytest.approx(world.offset_counts)
        assert point.electrons_per_count == pytest.approx(
            world.conversion_e_per_count
        )
        assert installation.device("camera").photoelectron_conversion == (
            pytest.approx(world.offset_counts),
            pytest.approx(world.conversion_e_per_count),
        )
    finally:
        installation.close()


def test_the_switch_is_available_only_when_the_camera_states_a_conversion() -> None:
    """A device fact, answered with the camera shut, with the reason it is not."""

    installation = create_installation("virtual")
    try:
        camera = installation.device("camera")
        assert resolve_photoelectron_availability({"camera": camera}) == {}
    finally:
        installation.close()

    refused = resolve_photoelectron_availability({"camera": None})
    assert set(refused) == {PHOTOELECTRONS}
    assert "states no photoelectron conversion" in refused[PHOTOELECTRONS]


def test_a_calibration_in_photoelectrons_reads_the_same_atoms(tmp_path: Path) -> None:
    """Affine: the levels move, the answers do not.

    Both runs see the same simulated world, so the sites and the judgements
    must agree while every level and threshold is scaled by the camera's own
    conversion.
    """

    request = _calibration_request(repeats=10)
    counts = _task(request, tmp_path).run()
    electrons = _task(
        replace(request, photoelectrons=True), tmp_path
    ).run()

    assert electrons.capture.frames[0].image.dtype == np.float32
    assert counts.capture.frames[0].image.dtype == np.uint16
    assert (
        electrons.calibration.site_map.n_sites
        == counts.calibration.site_map.n_sites
    )
    np.testing.assert_allclose(
        electrons.calibration.site_map.centers_xy,
        counts.calibration.site_map.centers_xy,
        atol=0.5,
    )

    installation = create_installation("virtual")
    try:
        offset = installation.world.offset_counts
        scale = installation.world.conversion_e_per_count
    finally:
        installation.close()
    box_counts = counts.calibration.select_model(
        counts.calibration.default_model_kind
    )
    box_electrons = electrons.calibration.select_model(
        electrons.calibration.default_model_kind
    )
    usable = np.asarray(box_counts.usable_sites, dtype=bool) & np.asarray(
        box_electrons.usable_sites, dtype=bool
    )
    assert usable.any()
    # A mean of counts becomes a mean of electrons by the same affine map.
    np.testing.assert_allclose(
        np.asarray(box_electrons.dark_mean)[usable],
        (np.asarray(box_counts.dark_mean)[usable] - offset) * scale,
        rtol=0.05,
        atol=0.5,
    )
    assert electrons.summary["models"]["box"]["fidelity"]["mean"] == pytest.approx(
        counts.summary["models"]["box"]["fidelity"]["mean"], abs=0.05
    )


def test_a_run_in_the_other_unit_is_refused(tmp_path: Path) -> None:
    """Not discovered in the data: refused where the two records meet."""

    calibration = _task(
        replace(_calibration_request(repeats=8), photoelectrons=True),
        tmp_path,
    ).run()
    processor = OccupancyProcessor(calibration.calibration)
    frames = np.asarray(
        [record.image for record in calibration.capture.frames[:3]],
        dtype=np.float32,
    ).reshape(1, 3, *calibration.capture.frames[0].image.shape)

    class _Source:
        run_record = {
            "parameters": {PHOTOELECTRONS: False},
            "device_snapshots": {"camera": {}},
        }
        snapshot = None

    with pytest.raises(ValueError, match="thresholds do not apply"):
        processor._validate_source_run_record(_Source())

    _Source.run_record = {
        "parameters": {PHOTOELECTRONS: True},
        "device_snapshots": {"camera": {}},
    }
    processor._validate_source_run_record(_Source())
    published = processor.process(
        camera_cycle_snapshot(frames),
    )
    assert np.asarray(published.occupied).shape[-1] == calibration.calibration.n_sites


def test_the_conversion_is_refused_when_the_camera_states_none() -> None:
    """A number nobody measured is not a number: the run stops instead."""

    from zlc_atom.nodes.camera_measurement.measurement import (
        CameraMeasurementNode,
        CameraMeasurementRequest,
    )

    installation = create_installation("virtual")
    try:
        node = CameraMeasurementNode(
            camera=installation.capability("camera.adapter"),
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.005,
                roi_xywh=None,
                repeat=1,
                frames_per_cycle=1,
                photoelectrons=True,
            ),
            signal_plane=FakePlane(),
            producer="units",
        )
        point = installation.device("camera").working_point()
        node._freeze_working_point(point)
        with pytest.raises(ValueError, match="states no photoelectron"):
            node._freeze_working_point(
                replace(point, offset_counts=None, electrons_per_count=None)
            )
    finally:
        installation.close()


def test_a_live_monitor_publishes_the_unit_the_run_asked_for() -> None:
    """The picture an operator watches, not just the one a run files away.

    The finite read converted and the monitor read did not, so a live panel
    showed counts while the same run's saved samples were electrons -- and a
    dark corner still read a few hundred, which is how this was found.  The
    two reads are one method now; this drives the monitor the way the console
    does and reads what reached the slot.
    """

    import time

    from zlc_atom.nodes.camera_measurement.measurement import (
        CameraMeasurementNode,
        CameraMeasurementRequest,
    )

    dark = 205  # counts: an ordinary dark corner, above the sensor's offset

    def _published(units: bool) -> np.ndarray:
        installation = create_installation("virtual")
        plane = FakePlane()
        try:
            node = CameraMeasurementNode(
                camera=installation.device("camera"),
                request=CameraMeasurementRequest(
                    camera_key="camera",
                    exposure_seconds=0.02,
                    roi_xywh=(0, 0, 16, 12),
                    repeat=0,
                    frames_per_cycle=1,
                    photoelectrons=units,
                ),
                signal_plane=plane,
                producer="monitor-units",
            )
            capture = node.monitor()
            camera = installation.device("camera")
            sensor = camera.working_point().sensor_shape_yx
            camera.trigger(4, frame=np.full(sensor, dark, dtype=np.uint16))
            deadline = time.monotonic() + 5.0
            signal = node.signal_key("frames")
            publication = None
            while publication is None and time.monotonic() < deadline:
                capture.poll()
                publication = plane.latest_publication(signal)
            capture.close()
            assert publication is not None, "the monitor published nothing"
            value = publication.value(signal)
            assert value is not None
            return np.asarray(value.snapshot.block.values[0, 0])
        finally:
            plane.close()
            installation.close()

    counts = _published(False)
    assert counts.dtype == np.uint16
    assert counts.min() == counts.max() == dark

    installation = create_installation("virtual")
    try:
        offset = installation.world.offset_counts
        scale = installation.world.conversion_e_per_count
    finally:
        installation.close()
    electrons = _published(True)
    assert electrons.dtype == np.float32
    np.testing.assert_allclose(
        electrons, np.float32((dark - offset) * scale), rtol=1e-6
    )


def test_the_camera_is_read_in_exactly_one_place() -> None:
    """One intake, mechanically: a capture may not reach the adapter itself.

    Both captures used to call ``read_frame_records`` directly and only one
    of them converted.  Nothing about that was visible in a review -- the
    conversion was there, in a method with a docstring calling itself THE
    conversion point -- so the rule is enforced on the source instead.
    """

    import ast
    from pathlib import Path

    import zlc_atom.nodes.camera_measurement.measurement as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_frame_records"
    ]
    assert len(reads) == 1, (
        "every frame this node takes must come through read_records; "
        f"found {len(reads)} calls to read_frame_records"
    )
    owner = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "read_records"
        and any(read in ast.walk(node) for read in reads)
    )
    assert owner.name == "read_records"


def test_saved_samples_and_the_preview_keep_the_unit_they_were_read_in(
    tmp_path: Path,
) -> None:
    """The file on disk is the numbers the fits ran on, not a re-quantised copy.

    The preview slot and the sample writer were handed the WORKING POINT's
    dtype -- the sensor's pixel format -- and cast every frame to it.  True of
    what a camera reads out; false of what this node holds once a run asks for
    photoelectrons: 0.535 e- was written as 0, and a pixel below the sensor's
    offset wrapped to 65535, so a saved folder could not reproduce the run it
    came from while the analysis it was saved beside was correct.
    """

    from zlc_data.figure_archive import read_archive, read_dataset

    result = _task(
        replace(
            _calibration_request(repeats=4),
            photoelectrons=True,
            save_frames=True,
        ),
        tmp_path,
    ).run()

    analysed = np.asarray(result.capture.frames[0].image)
    assert analysed.dtype == np.float32
    assert analysed.min() < 0.0, "a dark pixel below the offset is negative in electrons"

    samples = sorted((tmp_path / "calibration" / "frames").glob("sample_*.npz"))
    assert samples, "save_frames must leave the samples it paid for"
    for path in samples:
        info, arrays = read_archive(path)
        values = np.asarray(read_dataset(info, arrays, "data").block.values)
        assert values.dtype == np.float32, f"{path.name} was re-quantised"
        assert values.min() < 0.0
        assert not np.any(values == 65535), f"{path.name} wrapped a negative pixel"
