"""Occupancy reads each frame with the calibration that frame names; a calibration's API fields default in period order."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zlc_atom.devices.camera.contract import CameraFrameRecord
from zlc_atom.nodes import artifact_input_key, split_artifact_input_key
from zlc_atom.nodes.calibration import (
    CALIBRATION_ARTIFACT_CODEC,
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
)
from zlc_atom.nodes.calibration.logic_node import LOGIC_NODE as CALIBRATION_NODE
from zlc_atom.nodes.camera_measurement.measurement import frames_snapshot
from zlc_atom.nodes.occupancy.logic_node import LOGIC_NODE as OCCUPANCY_NODE
from zlc_atom.nodes.occupancy.processor import OccupancyProcessor
from zlc_atom.nodes.scan.dataset import scan_dataset_schema
from zlc_data import owned_snapshot_from_arrays
from zlc_pulse import api_bindings_in_period_order
from zlc_runtime import SignalValue

from pulse_fixture import calibration_request, pulse_sequence


def _calibration(threshold: float, *, sites: tuple[str, ...] = ("site-1",)) -> TrapCalibration:
    count = len(sites)
    return TrapCalibration(
        SiteMap(sites, np.asarray([[1.0, 1.0]] * count), [True] * count, [1.0] * count),
        (
            ReadoutModel(
                sites, [threshold] * count, [0.0] * count, [2.0 * threshold] * count,
                [True] * count, [1.0] * count,
            ),
        ),
        ReadoutModelKind.BOX,
        FrameContract((3, 3)),
    )


def _cycle(*levels: float):
    """One camera cycle of flat frames, one frame per level."""

    return frames_snapshot(
        (
            tuple(
                CameraFrameRecord(np.full((3, 3), level, dtype=np.float64), index)
                for index, level in enumerate(levels)
            ),
        ),
        producer="camera",
        generation="g",
        revision=1,
        value_unit="count",
    )


def test_each_frame_reads_with_the_calibration_it_names() -> None:
    """Frame 2 is judged by its own thresholds; the others by the shared
    ones.  The sites and the signals are the same in every frame -- only the
    verdict is read off a different calibration."""

    shared = _calibration(1.0)
    strict = _calibration(1e6)
    frames = _cycle(10.0, 10.0, 10.0)
    plain = OccupancyProcessor(shared).process(frames)
    assert plain.occupied.tolist() == [[[True], [True], [True]]]
    mixed = OccupancyProcessor(shared, calibration_by_frame={2: strict}).process(frames)
    assert mixed.occupied.tolist() == [[[True], [False], [True]]]
    assert mixed.artifacts["occupied"].expanded_validity().all(), "a stricter verdict is still a verdict"
    assert np.array_equal(mixed.counts, plain.counts)
    assert mixed.artifacts["counts"].block.schema == plain.artifacts["counts"].block.schema
    # A scan lays its frame axis innermost beneath the plan's rows and names a
    # repeated value once: the frames of every row are still 1, 2, 3.
    scanned_schema = scan_dataset_schema(
        frames.block.schema, ((1.0,), (3.0,), (1.0,)), (("power", "mW"),)
    )
    scanned = owned_snapshot_from_arrays(
        scanned_schema, np.tile(frames.block.values, (1, 3, 1, 1)), 1
    )
    judged = OccupancyProcessor(shared, calibration_by_frame={2: strict}).process(scanned)
    assert judged.occupied.tolist() == [[[True], [False], [True]] * 3]


def test_a_frames_calibration_must_read_the_same_sites_and_name_a_real_frame() -> None:
    with pytest.raises(ValueError, match="different sites"):
        OccupancyProcessor(
            _calibration(1.0),
            calibration_by_frame={2: _calibration(1.0, sites=("site-1", "site-2"))},
        )
    with pytest.raises(ValueError, match="counted from 1"):
        OccupancyProcessor(_calibration(1.0), calibration_by_frame={0: _calibration(1.0)})
    processor = OccupancyProcessor(_calibration(1.0), calibration_by_frame={3: _calibration(1e6)})
    with pytest.raises(ValueError, match="only 2 frame"):
        processor.process(_cycle(10.0, 10.0))
    # A frame's own calibration answers for its unit as the shared one does:
    # thresholds trained in photoelectrons do not judge frames read in counts.
    trained = replace(_calibration(1e6), report={"run_record": {"request": {"photoelectrons": True}}})
    counted = SignalValue("camera/frames", _cycle(10.0, 10.0), None)
    with pytest.raises(ValueError, match="frame 2's calibration was trained in photoelectrons"):
        OccupancyProcessor(_calibration(1.0), calibration_by_frame={2: trained}).evaluate(counted)


def test_the_logic_node_hands_each_frames_artifact_to_the_processor(tmp_path: Path) -> None:
    shared_path = tmp_path / "shared.json"
    strict_path = tmp_path / "strict.json"
    _calibration(1.0).save(shared_path)
    _calibration(1e6).save(strict_path)
    codec = CALIBRATION_ARTIFACT_CODEC
    assert "calibration_by_frame" in OCCUPANCY_NODE.build_argument_names
    processor = OCCUPANCY_NODE.instantiate(
        calibration=codec.resolve(shared_path),
        calibration_by_frame={2: codec.resolve(strict_path)},
        source_signal="camera/frames",
    )
    assert set(processor.calibration_by_frame) == {2}
    frames = _cycle(10.0, 10.0)
    record = processor.describe_run({"frames": SignalValue("camera/frames", frames, None)})
    assert record["parameters"]["calibration_path"] == str(shared_path.resolve())
    assert record["parameters"]["calibration_paths_by_frame"] == {"2": str(strict_path.resolve())}
    assert processor.process(frames).occupied.tolist() == [[[True], [False]]]
    with pytest.raises(TypeError, match="frame 2 calibration"):
        OCCUPANCY_NODE.instantiate(
            calibration=codec.resolve(shared_path),
            calibration_by_frame={2: object()},
            source_signal="camera/frames",
        )


def test_artifact_frame_keys_round_trip() -> None:
    assert artifact_input_key("calibration_path", 2) == "calibration_path[2]"
    assert split_artifact_input_key("calibration_path[2]") == ("calibration_path", 2)
    assert split_artifact_input_key("calibration_path") == ("calibration_path", None)
    for bad in ("calibration_path[0]", "calibration_path[x]", "[2]"):
        with pytest.raises(ValueError):
            split_artifact_input_key(bad)
    with pytest.raises(ValueError):
        artifact_input_key("calibration_path", 0)


def test_a_calibrations_api_fields_default_to_the_pulses_first_three_in_period_order() -> None:
    """Whatever order the bindings were declared in, the reference-before,
    readout and reference-after fields take the API parameters as the pulse
    plays them; with no pulse there is nothing to default."""

    sequence = pulse_sequence("imaging_template.json")
    shuffled = replace(sequence, bindings=tuple(reversed(sequence.bindings)))
    assert [binding.field_id for binding in shuffled.api_bindings] == [
        "duration:long_after", "duration:short", "duration:long_before",
    ]
    assert [binding.field_id for binding in api_bindings_in_period_order(shuffled)] == [
        "duration:long_before", "duration:short", "duration:long_after",
    ]
    defaults = CALIBRATION_NODE.resolve_defaults(
        {}, {"pulse_template": SimpleNamespace(value=shuffled)}
    )
    assert defaults == {
        "reference_before_field": "duration:long_before",
        "readout_field": "duration:short",
        "reference_after_field": "duration:long_after",
    }
    assert CALIBRATION_NODE.resolve_defaults({}, {}) == {}


def test_a_calibration_arms_its_camera_with_the_longest_window_and_shows_it() -> None:
    """The camera exposure is the calibration's own, derived from its two
    windows: the reference is the longest, so that is what the camera
    integrates.  The field is declared derived -- shown, never edited --
    and a request whose camera exposure would cut the reference window is
    refused."""

    from zlc_atom.nodes.calibration.task import camera_exposure_seconds

    assert camera_exposure_seconds(0.02, 0.005) == 0.02
    assert CALIBRATION_NODE.resolve_defaults(
        {"reference_exposure_seconds": 0.02, "readout_exposure_seconds": 0.005}, {}
    ) == {"camera_exposure_seconds": 0.02}
    assert CALIBRATION_NODE.resolve_defaults({"reference_exposure_seconds": ""}, {}) == {}
    field = next(
        field for field in CALIBRATION_NODE.authoring_schema.fields
        if field.name == "camera_exposure_seconds"
    )
    assert field.derived and not field.required

    request = calibration_request(repeats=3)
    assert request.to_dict()["camera_exposure_seconds"] == 0.02
    with pytest.raises(ValueError, match="cover the reference window"):
        replace(request, camera_exposure_seconds=0.01)
