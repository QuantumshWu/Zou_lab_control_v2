"""Frame-claims artifacts through finalization, build arguments, the form and a saved layout."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zlc_atom.nodes import ArtifactInputSpec, FrameClaim
from zlc_atom.nodes.calibration import (
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
)
from zlc_atom.nodes.occupancy.logic_node import LOGIC_NODE as OCCUPANCY_NODE
from zlc_runtime import SignalDataPlane
from zlc_workbench.authoring_form import project_artifact_inputs
from zlc_workbench.console_layout import LayoutDocument, LayoutError, LogicLayoutEntry, resolve_layout
from zlc_workbench.logic import LogicCatalog, LogicDraft, build_arguments, finalize_logic_draft


def _calibration() -> TrapCalibration:
    sites = ("site-1",)
    return TrapCalibration(
        SiteMap(sites, np.asarray([[1.0, 1.0]]), [True], [1.0]),
        (ReadoutModel(sites, [1.0], [0.0], [2.0], [True], [1.0]),),
        ReadoutModelKind.BOX,
        FrameContract((3, 3)),
    )


@pytest.fixture
def workspace(tmp_path: Path):
    (tmp_path / "data").mkdir()
    return SimpleNamespace(root=tmp_path, data=tmp_path / "data")


def _saved(workspace, name: str) -> Path:
    path = workspace.data / name
    _calibration().save(path)
    return path


def _finalize(workspace, plane, claims, **kwargs):
    draft = LogicDraft({"model_kind": "default"}, "camera/frames", {}, {"calibrations": claims})
    return finalize_logic_draft(
        OCCUPANCY_NODE, draft,
        installation=SimpleNamespace(devices={}), signal_plane=plane, workspace=workspace, **kwargs,
    )


def test_finalization_resolves_each_file_and_hands_the_claims_to_the_build(workspace) -> None:
    load, readout = _saved(workspace, "load.json"), _saved(workspace, "readout.json")
    plane = SignalDataPlane()
    try:
        finalization = _finalize(workspace, plane, (
            {"path": str(load), "frames": [0]},
            {"path": "readout.json", "frames": ["rest"]},     # relative to the data folder
        ))
        assert finalization.can_start, finalization.issues
        claims = finalization.artifacts["calibrations"]
        assert [claim.value.path for claim in claims] == [load.resolve(), readout.resolve()]
        assert [claim.frames for claim in claims] == [("first",), ("rest",)], "frame 0 is stored as first"
        arguments = build_arguments(OCCUPANCY_NODE, signal_plane=plane, finalization=finalization)
        assert arguments["calibrations"] is claims
        assert all(isinstance(claim, FrameClaim) for claim in claims)
        # A file decoded once is taken from the last answer again.
        again = _finalize(workspace, plane, (
            {"path": str(load), "frames": [0]}, {"path": str(readout), "frames": ["rest"]},
        ), previous=finalization)
        assert again.artifacts["calibrations"][0].value is claims[0].value
    finally:
        plane.close()


def test_claims_are_refused_by_name_and_against_a_known_frame_count(workspace) -> None:
    load, readout = _saved(workspace, "load.json"), _saved(workspace, "readout.json")
    plane = SignalDataPlane()
    try:
        assert "Calibrations must have at least one file" in _finalize(workspace, plane, ()).issues
        twice = _finalize(workspace, plane, (
            {"path": str(load), "frames": ["rest"]}, {"path": str(load), "frames": [0]},
        ))
        assert "load is listed twice: give it every frame it reads in one row" in twice.issues
        both = _finalize(workspace, plane, (
            {"path": str(load), "frames": ["rest"]}, {"path": str(readout), "frames": ["rest"]},
        ))
        assert "'rest' is claimed by both load and readout" in both.issues
        claims = ({"path": str(load), "frames": [2]}, {"path": str(readout), "frames": ["rest"]})
        assert _finalize(workspace, plane, claims).can_start, "no count known: nothing to check yet"
        # Two frames have no middle frame 2: the claim is kept, and stops nothing.
        assert _finalize(workspace, plane, claims, source_frames=2).can_start
        assert _finalize(workspace, plane, claims, source_frames=4).can_start
        empty_row = _finalize(workspace, plane, ({"path": "", "frames": ["rest"]},))
        assert "Calibrations row 1 has no file yet" in empty_row.issues
        # A file that reads frames must be there; one kept without frames need not.
        gone = str(workspace.data / "gone.json")
        missing = _finalize(workspace, plane, ({"path": gone, "frames": ["rest"]},))
        assert missing.issues == (f"calibration.readout artifact “{gone}” is invalid: file not found",)
        # Every collision is said, and a row kept without frames hides none of them.
        hidden = _finalize(workspace, plane, (
            {"path": gone, "frames": []}, {"path": str(load), "frames": ["rest"]},
            {"path": str(readout), "frames": ["rest"]},
        ))
        assert hidden.issues == ("'rest' is claimed by both load and readout",)
        nothing = _finalize(workspace, plane, (
            {"path": str(load), "frames": []}, {"path": str(readout), "frames": []},
        ))
        assert nothing.issues == (
            "no file reads a frame: give one the frames it reads, or let one read the rest",)
        shape = _finalize(workspace, plane, "load.json")
        assert any("frame claims must be a list" in issue for issue in shape.issues)
    finally:
        plane.close()


def test_a_file_kept_without_frames_never_stops_start(workspace) -> None:
    """The user's case: Occupancy keeps calibration files that read no
    frame.  "Add calibration…" makes one whenever another file reads the
    rest; the draft starts, the editor says "1 file not read", and the run
    record lists the file.  So does a copy of a reading file, a file that
    has gone, and a row with no file at all -- and none of them is checked
    against the file that reads."""

    from PyQt5 import QtCore

    from zlc_atom.devices.camera.contract import CameraFrameRecord
    from zlc_atom.nodes.camera.measurement import frames_snapshot
    from zlc_runtime import SignalValue
    from zlc_ui.qt import ensure_qt_app
    from zlc_workbench.console import _claims_projection

    readout = _saved(workspace, "readout.json")
    other = workspace.data / "other_sites.json"
    TrapCalibration(
        SiteMap(("site-1", "site-2"), np.asarray([[1.0, 1.0], [2.0, 2.0]]), [True] * 2, [1.0] * 2),
        (ReadoutModel(("site-1", "site-2"), [1.0] * 2, [0.0] * 2, [2.0] * 2, [True] * 2, [1.0] * 2),),
        ReadoutModelKind.BOX,
        FrameContract((3, 3)),
    ).save(other)
    (spec,) = [s for s in OCCUPANCY_NODE.input_specs if isinstance(s, ArtifactInputSpec)]
    app = ensure_qt_app(["zlc-kept-calibration-test"])
    editor = OCCUPANCY_NODE.ui_contributions[0]()
    drafts = []
    editor.draft_changed.connect(drafts.append)
    plane = SignalDataPlane()

    def project(claims, finalization):
        projection = _claims_projection(spec, claims, finalization, ("0", "1", "2"), str(workspace.data))
        editor.update_projection({"artifact_claims": {"calibrations": projection},
                                  "source_frames": ("0", "1", "2"), "form_values": {"model_kind": "default"}})
        app.processEvents()
        return projection

    try:
        editor.add_file(str(readout))
        project(drafts[-1]["artifact_inputs"]["calibrations"], _finalize(workspace, plane, ()))
        editor.add_file(str(other))
        claims = drafts[-1]["artifact_inputs"]["calibrations"]
        assert claims[1] == {"path": str(other), "frames": ()}, "Rest is taken: the new file reads none"
        finalization = _finalize(workspace, plane, claims, source_frames=3)
        assert finalization.can_start, finalization.issues
        projection = project(claims, finalization)
        assert projection["problems"] == () and projection["unused"] == (1,)
        assert projection["files"][1]["summary"] and projection["files"][1]["unread"] == ""
        assert editor.summary.text() == "readout reads every frame · 1 file not read"

        gone = workspace.data / "gone.json"
        kept = (*claims, {"path": str(gone), "frames": ()}, {"path": str(readout), "frames": ()},
                {"path": "", "frames": ()})
        finalization = _finalize(workspace, plane, kept, source_frames=3)
        assert finalization.can_start, finalization.issues
        projection = project(kept, finalization)
        assert projection["unused"] == (1, 2, 3, 4)
        assert projection["files"][2] == {"path": str(gone.resolve()), "summary": "", "unread": "file not found"}
        assert projection["files"][4] is None
        assert editor.summary.text() == "readout reads every frame · 4 files not read"
        rows = editor.grid.model()
        assert "cannot be read: file not found" in rows.headerData(2, QtCore.Qt.Vertical, QtCore.Qt.ToolTipRole)
        assert "same file as readout" in rows.headerData(3, QtCore.Qt.Vertical, QtCore.Qt.ToolTipRole)

        arguments = build_arguments(OCCUPANCY_NODE, signal_plane=plane, finalization=finalization)
        processor = OCCUPANCY_NODE.instantiate(**arguments)
        frames = frames_snapshot(
            (tuple(CameraFrameRecord(np.full((3, 3), 10.0), index) for index in range(3)),),
            producer="camera", generation="g", revision=1, value_unit="count",
        )
        record = processor.describe_run({"frames": SignalValue("camera/frames", frames, None)})
        assert record["parameters"]["calibrations"] == [
            {"path": str(readout.resolve()), "frames": ["rest"], "read_frames": [0, 1, 2]},
            {"path": str(other.resolve()), "frames": [], "read_frames": []},
            {"path": str(gone.resolve()), "frames": [], "read_frames": []},
            {"path": str(readout.resolve()), "frames": [], "read_frames": []},
            {"path": None, "frames": [], "read_frames": []},
        ]
        assert processor.process(frames).occupied.shape == (1, 3, 1), "read with the one file that reads"
    finally:
        plane.close()
        editor.close()
        editor.deleteLater()
        app.processEvents()


def test_a_kept_file_that_cannot_be_read_is_read_at_the_boundaries_only(workspace) -> None:
    """A kept file that does not decode is answered with its error, and an
    edit carries that answer like a decoded one: it is read again only at a
    boundary (no last answer), which is where a fixed file is seen.  The
    error keeps no traceback -- the frames that raised it hold the last
    answer, so each answer would otherwise keep every one before it."""

    import gc
    import weakref

    readout = _saved(workspace, "readout.json")
    broken = workspace.data / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    reads = ({"path": str(readout), "frames": ["rest"]},)
    plane = SignalDataPlane()
    try:
        claims = (*reads, {"path": str(broken), "frames": []})
        first = _finalize(workspace, plane, claims, source_frames=3)
        assert first.can_start, first.issues
        error = first.artifacts["calibrations"][1].value.value
        assert isinstance(error, Exception)
        assert (error.__traceback__, error.__context__, error.__cause__) == (None, None, None)

        # Fixed on disk: an edit still carries the failed answer, a boundary reads it.
        _calibration().save(broken)
        edited = _finalize(workspace, plane, claims, source_frames=3, previous=first)
        assert edited.artifacts["calibrations"][1].value is first.artifacts["calibrations"][1].value
        fresh = _finalize(workspace, plane, claims, source_frames=3)
        assert isinstance(fresh.artifacts["calibrations"][1].value.value, TrapCalibration)

        # Every edit naming a new missing file: no answer keeps the one before it.
        answer = _finalize(workspace, plane, reads, source_frames=3)
        oldest = weakref.ref(answer)
        for index in range(5):
            gone = {"path": str(workspace.data / f"gone_{index}.json"), "frames": []}
            answer = _finalize(workspace, plane, (*reads, gone), source_frames=3, previous=answer)
            assert answer.can_start, answer.issues
        gc.collect()
        assert oldest() is None, "the first answer is still held by a later one"
    finally:
        plane.close()


def test_the_path_form_leaves_frame_claims_to_their_editor() -> None:
    (spec,) = [s for s in OCCUPANCY_NODE.input_specs if isinstance(s, ArtifactInputSpec)]
    assert spec.frame_claims
    assert project_artifact_inputs((spec,), base_dir="d").keys == ()


def test_a_saved_layout_keeps_the_claims_as_written() -> None:
    claims = ({"path": "load.json", "frames": [0]}, {"path": "readout.json", "frames": ["last", "rest"]},
              {"path": "spare.json", "frames": []})
    entry = LogicLayoutEntry(
        "occupancy", "occupancy", {"model_kind": "default"}, "", {}, {"calibrations": claims},
    )
    tree = LayoutDocument((), (entry,), ()).to_tree()
    assert tree["logic"][0]["artifact_inputs"]["calibrations"][1]["frames"] == ["last", "rest"]
    document = LayoutDocument.from_tree(tree)
    resolved = resolve_layout(
        document, catalog=LogicCatalog(), installation=SimpleNamespace(devices={}), panel_ids=(),
    )
    (binding,) = resolved.logic
    assert binding.draft.artifact_inputs == {"calibrations": (
        {"path": "load.json", "frames": ("first",)}, {"path": "readout.json", "frames": ("last", "rest")},
        {"path": "spare.json", "frames": ()},
    )}, "a file kept without frames is kept by the layout too"
    broken = LogicLayoutEntry(
        "occupancy", "occupancy", {}, "", {}, {"calibrations": [{"path": "a.json", "frames": ["second"]}]},
    )
    with pytest.raises(LayoutError, match="not a frame"):
        resolve_layout(
            LayoutDocument((), (broken,), ()), catalog=LogicCatalog(),
            installation=SimpleNamespace(devices={}), panel_ids=(),
        )
