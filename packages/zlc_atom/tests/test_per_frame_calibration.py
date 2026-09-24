"""Occupancy reads each frame with the calibration that frame names; a calibration's API fields default in period order."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from zlc_pulse import api_bindings_in_period_order
from zlc_pulse.codec import sequence_from_tree
from zlc_runtime import SignalValue


PULSES = Path(__file__).resolve().parent / "pulses"


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

    tree = json.loads((PULSES / "imaging_template.json").read_text(encoding="utf-8"))
    sequence = sequence_from_tree(tree)
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
