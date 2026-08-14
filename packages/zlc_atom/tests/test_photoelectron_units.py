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
from zlc_atom.devices.camera.units import (
    FRAME_UNITS,
    FrameUnit,
    frame_unit_choices,
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


def test_the_offered_units_are_what_the_bound_camera_says() -> None:
    """The option carries the camera's numbers, or the reason it cannot be taken."""

    installation = create_installation("virtual")
    try:
        offered = frame_unit_choices(installation.device("camera"))
    finally:
        installation.close()
    assert tuple(choice.value for choice in offered) == (
        FrameUnit.COUNTS.value,
        FrameUnit.PHOTOELECTRONS.value,
    )
    assert not any(choice.unavailable_reason for choice in offered)
    assert "0.107 e-/count" in offered[1].label
    assert "offset 200" in offered[1].label

    # A camera that states nothing still SHOWS the option -- one that vanishes
    # when a device is chosen is a moving target -- and says why not.
    silent = frame_unit_choices(None)
    assert not silent[0].unavailable_reason
    assert "states no photoelectron conversion" in silent[1].unavailable_reason
    assert "no conversion" in silent[1].label


def test_a_calibration_in_photoelectrons_reads_the_same_atoms(tmp_path: Path) -> None:
    """Affine: the levels move, the answers do not.

    Both runs see the same simulated world, so the sites and the judgements
    must agree while every level and threshold is scaled by the camera's own
    conversion.
    """

    request = _calibration_request(repeats=10)
    counts = _task(request, tmp_path).run()
    electrons = _task(
        replace(request, frame_units=FrameUnit.PHOTOELECTRONS), tmp_path
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
        replace(_calibration_request(repeats=8), frame_units=FrameUnit.PHOTOELECTRONS),
        tmp_path,
    ).run()
    processor = OccupancyProcessor(calibration.calibration)
    frames = np.asarray(
        [record.image for record in calibration.capture.frames[:3]],
        dtype=np.float32,
    ).reshape(1, 3, *calibration.capture.frames[0].image.shape)

    class _Source:
        run_record = {
            "parameters": {FRAME_UNITS: FrameUnit.COUNTS.value},
            "device_snapshots": {"camera": {}},
        }
        snapshot = None

    with pytest.raises(ValueError, match="thresholds do not apply"):
        processor._validate_source_run_record(_Source())

    _Source.run_record = {
        "parameters": {FRAME_UNITS: FrameUnit.PHOTOELECTRONS.value},
        "device_snapshots": {"camera": {}},
    }
    processor._validate_source_run_record(_Source())
    published = processor.process(
        camera_cycle_snapshot(frames),
        generation="unit-guard",
        revision=1,
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
                frame_units=FrameUnit.PHOTOELECTRONS,
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
