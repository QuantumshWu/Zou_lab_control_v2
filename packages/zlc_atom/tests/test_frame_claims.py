"""Frame claims as written: one spelling, the draft value, and each row's name."""

from __future__ import annotations

import pytest

from zlc_atom.nodes import frame_claim_entries
from zlc_atom.nodes._framework.frames import canonical_frames, describe_frames, frame_identity


def test_claimed_frames_have_one_spelling_and_one_order() -> None:
    # Frame 0 IS the first frame: one claim, one spelling.
    assert canonical_frames(["rest", 2, "0", "LAST"]) == (2, "first", "last", "rest")
    assert canonical_frames(["rest", "last", 3, " First "]) == (3, "first", "last", "rest")
    assert describe_frames((0, 2, "last")) == "0, 2, last"
    assert describe_frames((1, "first", "rest")) == "1, first, rest"
    assert describe_frames(()) == "no frame"
    assert tuple(frame_identity(index, 4) for index in range(4)) == (0, 1, 2, "last")
    assert frame_identity(0, 1) == 0
    for bad in ([True], [-1], ["second"], [1.5], "rest"):
        with pytest.raises(ValueError):
            canonical_frames(bad)
    with pytest.raises(ValueError, match="frame 1 is listed twice"):
        canonical_frames([1, "1"])
    with pytest.raises(ValueError, match="frame first is listed twice"):
        canonical_frames(["first", "FIRST"])


def test_a_draft_value_is_rows_of_path_and_frames() -> None:
    assert frame_claim_entries([{"path": " a.json ", "frames": ["rest", 0]}]) == (
        {"path": "a.json", "frames": ("first", "rest")},
    )
    for bad in ("a.json", {"path": "a", "frames": []}, [{"path": "a"}],
                [{"path": 1, "frames": []}], [{"path": "a", "frames": [], "x": 1}]):
        with pytest.raises(ValueError):
            frame_claim_entries(bad)


def test_a_row_is_named_by_its_run_folder_and_told_apart_by_the_day() -> None:
    """Every Calibration run saves <run>/final/calibration.json, so a row is
    named by its run folder, not the file's stem every row would share; runs
    of one name on two days carry the day; any other file keeps its stem."""

    from zlc_atom.nodes import claim_names

    names = claim_names((
        {"path": "C:/ws/data/2026_10_13/load_cal/final/calibration.json", "frames": ("first",)},
        {"path": "C:/ws/data/2026_10_14/load_cal/final/calibration.json", "frames": (1,)},
        {"path": "C:/ws/data/2026_10_14/readout_cal/final/calibration.json", "frames": ("last",)},
        {"path": "C:/ws/calibrations/background_5ms.json", "frames": ("rest",)},
        {"path": "", "frames": (2,)},
    ))
    assert names == ("2026_10_13/load_cal", "2026_10_14/load_cal", "readout_cal", "background_5ms", "row 5")
    assert "calibration" not in names

    # Two runs that kept the task's own folder name, and files saved before
    # runs had a final folder: each is told apart by its day, never by a
    # folder that repeats the name or by the data root.
    assert claim_names((
        {"path": "C:/ws/data/2026_09_01/calibration/final/calibration.json", "frames": ("first",)},
        {"path": "C:/ws/data/2026_09_02/calibration/final/calibration.json", "frames": ("last",)},
        {"path": "C:/ws/data/2026_08_16/calibration/calibration.json", "frames": (1,)},
        {"path": "C:/ws/data/2026_08_08/calibration.json", "frames": ("rest",)},
    )) == ("2026_09_01/calibration", "2026_09_02/calibration", "2026_08_16/calibration", "2026_08_08/calibration")
