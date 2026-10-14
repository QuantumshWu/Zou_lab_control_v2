"""Occupancy reads each frame with the calibration that claims it; a calibration's API fields default in period order."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zlc_atom.devices.camera.contract import CameraFrameRecord
from zlc_atom.nodes import FrameClaim
from zlc_atom.nodes.calibration import (
    CALIBRATION_ARTIFACT_CODEC,
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
)
from zlc_atom.nodes.calibration.logic_node import LOGIC_NODE as CALIBRATION_NODE
from zlc_atom.nodes.camera.measurement import frames_snapshot
from zlc_atom.nodes.occupancy.logic_node import LOGIC_NODE as OCCUPANCY_NODE
from zlc_atom.nodes.occupancy.processor import CalibrationReading, OccupancyProcessor
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


def test_each_frame_reads_with_the_calibration_that_claims_it() -> None:
    """Frame 1 is judged by its own thresholds; the rest by the other file's.
    The sites and the signals are the same in every frame -- only the verdict
    is read off a different calibration."""

    shared = _calibration(1.0)
    strict = _calibration(1e6)
    frames = _cycle(10.0, 10.0, 10.0)
    plain = OccupancyProcessor(shared).process(frames)
    assert plain.occupied.tolist() == [[[True], [True], [True]]]
    readings = (CalibrationReading(shared, ("rest",)), CalibrationReading(strict, (1,)))
    mixed = OccupancyProcessor(readings).process(frames)
    assert mixed.occupied.tolist() == [[[True], [False], [True]]]
    assert mixed.artifacts["occupied"].expanded_validity().all(), "a stricter verdict is still a verdict"
    assert np.array_equal(mixed.counts, plain.counts)
    assert mixed.artifacts["counts"].block.schema == plain.artifacts["counts"].block.schema
    # A scan lays its frame axis innermost beneath the plan's rows and names a
    # repeated value once: the frames of every row are still 0, 1, 2.
    scanned_schema = scan_dataset_schema(
        frames.block.schema, ((1.0,), (3.0,), (1.0,)), (("power", "mW"),)
    )
    scanned = owned_snapshot_from_arrays(
        scanned_schema, np.tile(frames.block.values, (1, 3, 1, 1)), 1
    )
    judged = OccupancyProcessor(readings).process(scanned)
    assert judged.occupied.tolist() == [[[True], [False], [True]] * 3]


def test_last_and_rest_follow_the_frame_count_and_numbers_keep_their_place() -> None:
    """The same claims read a cycle that gained a frame the way they were
    meant: the last frame stays the strict one.  A number the cycle does not
    carry, a frame read twice or by nobody, refuses the run by name."""

    shared, strict = _calibration(1.0), _calibration(1e6)
    processor = OccupancyProcessor((
        CalibrationReading(strict, ("last",), Path("readout.json")),
        CalibrationReading(shared, ("rest",), Path("load.json")),
    ))
    assert processor.process(_cycle(10.0, 10.0)).occupied.tolist() == [[[True], [False]]]
    assert processor.process(_cycle(10.0, 10.0, 10.0)).occupied.tolist() == [[[True], [True], [False]]]
    # Without a frame axis a cycle is one frame: it is the last, and the first.
    single = frames_snapshot(
        ((CameraFrameRecord(np.full((3, 3), 10.0), 0),),),
        producer="camera", generation="g", revision=1, value_unit="count",
    )
    assert processor.process(single).occupied.tolist() == [[[False]]]

    beyond = OccupancyProcessor((
        CalibrationReading(shared, ("rest",), Path("load.json")),
        CalibrationReading(strict, (3,), Path("late.json")),
    ))
    with pytest.raises(ValueError, match=r"late names middle frame 3, but the source carries 2 frames \(0-1\)"):
        beyond.process(_cycle(10.0, 10.0))
    # In four frames 3 is the last frame, not a middle one: still absent.
    with pytest.raises(ValueError, match=r"late names middle frame 3"):
        beyond.process(_cycle(10.0, 10.0, 10.0, 10.0))
    assert beyond.process(_cycle(10.0, 10.0, 10.0, 10.0, 10.0)).occupied.tolist() == [
        [[True], [True], [True], [False], [True]]
    ]
    twice = OccupancyProcessor((
        CalibrationReading(strict, ("last",), Path("readout.json")),
        CalibrationReading(shared, (0, 2), Path("load.json")),
    ))
    # In three frames 2 is the last: load's middle 2 is absent, not a second reader.
    with pytest.raises(ValueError, match=r"load names middle frame 2"):
        twice.process(_cycle(10.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="frame 1 has no calibration"):
        twice.process(_cycle(10.0, 10.0, 10.0, 10.0))


def test_first_reads_frame_zero_whatever_the_count() -> None:
    """First is frame 0 of every cycle, the mirror of Last; a second entry
    reaching frame 0 -- by its number, or as the Last of a one-frame cycle --
    refuses the run by name and by how each reaches it."""

    shared, strict = _calibration(1.0), _calibration(1e6)
    processor = OccupancyProcessor((
        CalibrationReading(strict, ("first",), Path("load.json")),
        CalibrationReading(shared, ("rest",), Path("readout.json")),
    ))
    assert processor.process(_cycle(10.0, 10.0)).occupied.tolist() == [[[False], [True]]]
    assert processor.process(_cycle(10.0, 10.0, 10.0)).occupied.tolist() == [[[False], [True], [True]]]
    assert processor.process(_cycle(10.0)).occupied.tolist() == [[[False]]]
    ends = OccupancyProcessor((
        CalibrationReading(strict, ("first",), Path("load.json")),
        CalibrationReading(shared, ("last",), Path("readout.json")),
    ))
    assert ends.process(_cycle(10.0, 10.0)).occupied.tolist() == [[[False], [True]]]
    with pytest.raises(ValueError, match=r"frame 0 is read by both load \(first\) and readout \(last\)"):
        ends.process(_cycle(10.0))
    # 0 IS First: one claim, refused before any cycle arrives.
    with pytest.raises(ValueError, match="'first' is claimed by both load and readout"):
        OccupancyProcessor((
            CalibrationReading(strict, ("first",), Path("load.json")),
            CalibrationReading(shared, (0, "rest"), Path("readout.json")),
        ))


def test_calibrations_must_fit_together() -> None:
    with pytest.raises(ValueError, match="b names different sites"):
        OccupancyProcessor((
            CalibrationReading(_calibration(1.0), ("rest",), Path("a.json")),
            CalibrationReading(_calibration(1.0, sites=("site-1", "site-2")), (0,), Path("b.json")),
        ))
    with pytest.raises(ValueError, match="'rest' is claimed by both a and b"):
        OccupancyProcessor((
            CalibrationReading(_calibration(1.0), ("rest",), Path("a.json")),
            CalibrationReading(_calibration(1.0), ("rest",), Path("b.json")),
        ))
    with pytest.raises(ValueError, match="b reads no frame"):
        OccupancyProcessor((
            CalibrationReading(_calibration(1.0), ("rest",), Path("a.json")),
            CalibrationReading(_calibration(1.0), (), Path("b.json")),
        ))
    with pytest.raises(ValueError, match="a is listed twice"):
        OccupancyProcessor((
            CalibrationReading(_calibration(1.0), ("rest",), Path("a.json")),
            CalibrationReading(_calibration(1.0), (0,), Path("a.json")),
        ))
    with pytest.raises(ValueError, match="not a frame"):
        CalibrationReading(_calibration(1.0), (True,))
    assert OccupancyProcessor((
        CalibrationReading(_calibration(1.0), ("rest",), Path("a.json")),
        CalibrationReading(_calibration(2.0), (0,), Path("b.json")),
    )).model_kind is ReadoutModelKind.BOX
    with pytest.raises(ValueError, match="a has no uniform_psf readout model"):
        OccupancyProcessor(
            (
                CalibrationReading(_calibration(1.0), ("rest",), Path("a.json")),
                CalibrationReading(_calibration(1.0), (0,), Path("b.json")),
            ),
            model_kind=ReadoutModelKind.UNIFORM_PSF,
        )
    # Each calibration answers for its unit: thresholds trained in
    # photoelectrons do not judge frames read in counts.
    trained = replace(_calibration(1e6), report={"run_record": {"request": {"photoelectrons": True}}})
    counted = SignalValue("camera/frames", _cycle(10.0, 10.0), None)
    with pytest.raises(ValueError, match="trained was trained in photoelectrons"):
        OccupancyProcessor((
            CalibrationReading(_calibration(1.0), ("rest",), Path("plain.json")),
            CalibrationReading(trained, (1,), Path("trained.json")),
        )).evaluate(counted)


def test_the_logic_node_hands_each_claim_to_the_processor(tmp_path: Path) -> None:
    shared_path = tmp_path / "shared.json"
    strict_path = tmp_path / "strict.json"
    _calibration(1.0).save(shared_path)
    _calibration(1e6).save(strict_path)
    codec = CALIBRATION_ARTIFACT_CODEC
    assert "calibrations" in OCCUPANCY_NODE.build_argument_names
    processor = OCCUPANCY_NODE.instantiate(
        calibrations=(
            FrameClaim(codec.resolve(shared_path), ("rest",)),
            FrameClaim(codec.resolve(strict_path), ("last",)),
        ),
        source_signal="camera/frames",
    )
    frames = _cycle(10.0, 10.0)
    record = processor.describe_run({"frames": SignalValue("camera/frames", frames, None)})
    # What was asked for beside what it came to.
    assert record["parameters"]["calibrations"] == [
        {"path": str(shared_path.resolve()), "frames": ["rest"], "read_frames": [0]},
        {"path": str(strict_path.resolve()), "frames": ["last"], "read_frames": [1]},
    ]
    assert processor.process(frames).occupied.tolist() == [[[True], [False]]]
    with pytest.raises(TypeError, match="resolved calibration artifacts"):
        OCCUPANCY_NODE.instantiate(
            calibrations=(FrameClaim(object(), ("rest",)),),
            source_signal="camera/frames",
        )


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
    from zlc_pulse import PulseBinding, PulseBracket, PulseFieldRef

    count = PulseBinding(PulseFieldRef("bracket_count", bracket_id="early"), "count", source="api")
    first_period = shuffled.periods[0].period_id
    shuffled = replace(
        shuffled, bindings=(count, *shuffled.bindings),
        brackets=(*shuffled.brackets, PulseBracket("early", first_period, first_period, 1)),
    )
    assert count in api_bindings_in_period_order(shuffled)[:3]
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


def _claims_projection(entries, frames):
    """What the console projects for the calibrations editor."""

    from zlc_atom.nodes import check_claims, frame_problems, reach_frames

    names = tuple(Path(entry["path"]).stem for entry in entries)
    claimed = tuple(entry["frames"] for entry in entries)
    problems, reads = (), None
    try:
        check_claims(claimed, names)
    except ValueError as error:
        problems = (str(error),)
    if frames is not None and entries:
        problems = problems or frame_problems(claimed, len(frames), names, what="file")
        reach = reach_frames(claimed, len(frames))
        reads = tuple(reach.frames_of(index) for index in range(len(entries)))
    return {
        "artifact_claims": {"calibrations": {
            "label": "Calibrations", "entries": tuple(entries), "base_dir": "C:/data",
            "file_filter": "Calibration artifacts (*.json)",
            "files": tuple({"path": entry["path"], "summary": "20 ms · 1 sites · default Box"} for entry in entries),
            "reads": reads, "problems": problems,
        }},
        "source_frames": frames,
        "form_values": {"model_kind": "default"},
    }


def test_the_calibrations_editor_gives_frames_to_files_by_clicking(monkeypatch) -> None:
    from PyQt5 import QtCore, QtTest

    from zlc_atom.nodes.occupancy import editor as editor_module
    from zlc_ui.fluent import scaled_px, setting_label_width
    from zlc_ui.qt import ensure_qt_app

    app = ensure_qt_app(["zlc-calibrations-editor-test"])
    editor = OCCUPANCY_NODE.ui_contributions[0]()
    drafts = []
    editor.draft_changed.connect(drafts.append)
    grid = editor.grid

    def project(entries, frames=("0", "1", "2")):
        editor.update_projection(_claims_projection(entries, frames))
        app.processEvents()

    def header(column):
        return grid.model().headerData(column, QtCore.Qt.Horizontal)

    def state(row, column):
        return grid.model().index(row, column).data(QtCore.Qt.UserRole + 1)

    def press(row, column):
        index = grid.model().index(row, column)
        grid.scrollTo(index)
        app.processEvents()
        QtTest.QTest.mouseClick(grid.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                                grid.visualRect(index).center())
        app.processEvents()
        return drafts[-1]["artifact_inputs"]["calibrations"]

    def click(row, column):
        assert grid.model().index(row, column).flags() & QtCore.Qt.ItemIsSelectable, (row, column)
        return press(row, column)

    load = {"path": "C:/data/load.json", "frames": ("first",)}
    readout = {"path": "C:/data/readout.json", "frames": ("rest",)}
    try:
        editor.resize(900, 300)
        editor.show()
        project((load, readout))
        # The ends are named by what they are; the numbered columns are the
        # middle frames only, so no column repeats First or Last.
        assert [header(column) for column in range(5)] == ["First · 0", "Last · 2", "Rest", "", "1"]
        assert grid.selected_cells() == ((0, 0), (1, 2))
        assert state(1, 4) == "derived", "a middle frame read as rest"
        assert state(1, 1) == "derived", "the last frame read as rest"
        assert drafts == [], "showing claims is not an edit"
        assert "3 frames · each read once" in editor.summary.text()
        # The file names stand in the label column without setting it: the
        # first card starts on the model choice's edge, and a name wider than
        # the column is elided there instead of moving every field.
        editor.set_label_width(setting_label_width(editor.row_labels()))
        app.processEvents()

        def card_x():
            card = grid.viewport().mapTo(editor, grid.visualRect(grid.model().index(0, 0)).topLeft())
            return card.x() + scaled_px(2, minimum=1)

        edge = editor.model.mapTo(editor, QtCore.QPoint(0, 0)).x()
        assert abs(card_x() - edge) <= 1
        project((load, {"path": "C:/data/readout_with_a_long_name.json", "frames": ("rest",)}))
        assert "…" in grid.model().headerData(1, QtCore.Qt.Vertical)
        assert abs(card_x() - edge) <= 1 and editor.row_labels() == ("Readout model",)
        project((load, readout))
        # Last moves to the readout file; First taken here leaves the load file.
        claims = click(1, 1)
        assert claims == (load, {"path": readout["path"], "frames": ("last", "rest")})
        project(claims)
        claims = click(1, 0)
        assert claims == ({"path": load["path"], "frames": ()},
                          {"path": readout["path"], "frames": ("first", "last", "rest")})
        # A middle frame keeps its number while the ends follow the cycle: with
        # five frames the middle is 1-3; back at three frames, 2 is the LAST
        # frame, so the claim on middle frame 2 is absent -- never the last.
        middle = {"path": load["path"], "frames": (2, "first")}
        project((middle, readout), frames=("0", "1", "2", "3", "4"))
        assert [header(column) for column in range(7)] == [
            "First · 0", "Last · 4", "Rest", "", "1", "2", "3"]
        assert grid.selected_cells() == ((0, 0), (0, 5), (1, 2))
        project((middle, readout))
        assert header(5) == "2" and state(0, 5) == "warning"
        assert "load names middle frame 2" in editor.note.text()
        assert state(1, 1) == "derived", "the new last frame is read as rest, not as middle 2"
        assert click(0, 5)[0] == {"path": load["path"], "frames": ("first",)}
        # Each row ends, after a gap, with its own remove cell: never a claim.
        project((load, readout))
        remove = grid.model().columnCount() - 1
        assert header(remove - 1) is None
        assert not grid.model().index(0, remove).flags() & QtCore.Qt.ItemIsSelectable
        assert grid.model().index(0, remove).data(QtCore.Qt.ToolTipRole) == "Remove this calibration"
        before = grid.selected_cells()
        assert press(0, remove) == (readout,)
        assert grid.selected_cells() == before, "removing is not marking"
        # Clicking a file's name chooses another file for that row.
        monkeypatch.setattr(editor_module, "fluent_open_path", lambda *args: "C:/data/other.json")
        rows = grid.verticalHeader()
        QtTest.QTest.mouseClick(rows.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                                QtCore.QPoint(10, rows.sectionViewportPosition(1) + 10))
        assert drafts[-1]["artifact_inputs"]["calibrations"] == (
            load, {"path": "C:/data/other.json", "frames": ("rest",)})
        # The readout model, set once, is a segmented choice under the files.
        QtTest.QTest.mouseClick(editor.model.button("box"), QtCore.Qt.LeftButton)
        assert drafts[-1] == {"values": {"model_kind": "box"}}
        editor.add_file("C:/data/extra.json")
        assert drafts[-1]["artifact_inputs"]["calibrations"][-1] == {
            "path": "C:/data/extra.json", "frames": ()}, "Rest is taken; the new file starts with no frame"
        project(())
        assert not grid.isVisible() and editor.summary.text() == "No calibration yet"
        assert not editor.model_row.isVisible() and editor.add_button.isVisible()
        editor.add_file("C:/data/only.json")
        assert drafts[-1]["artifact_inputs"]["calibrations"] == (
            {"path": "C:/data/only.json", "frames": ("rest",)},)
    finally:
        editor.close()
        editor.deleteLater()
        app.processEvents()
