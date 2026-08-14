"""Saved work lands under the day it was taken, and never lands on itself."""

from __future__ import annotations

from datetime import date

import pytest

from zlc_durable import day_folder, day_folder_name, unique_path
from zlc_durable.workspace import DAY_FOLDER_PATTERN


def test_day_folder_name_is_zero_padded_and_matches_the_declared_pattern() -> None:
    assert day_folder_name(date(2026, 8, 5)) == "2026_08_05"
    assert day_folder_name(date(2026, 12, 31)) == "2026_12_31"
    for day in (date(2026, 1, 1), date(2026, 8, 5), date(1999, 10, 9)):
        assert DAY_FOLDER_PATTERN.fullmatch(day_folder_name(day))


def test_day_folder_creates_the_day_beneath_an_existing_root(tmp_path) -> None:
    folder = day_folder(tmp_path, date(2026, 8, 5))
    assert folder == tmp_path / "2026_08_05"
    assert folder.is_dir()
    # Idempotent: asking twice on the same day is the normal case.
    assert day_folder(tmp_path, date(2026, 8, 5)) == folder


def test_day_folder_refuses_a_save_root_that_does_not_exist(tmp_path) -> None:
    """A typo in the save root must not silently scatter data into a new tree."""

    with pytest.raises(NotADirectoryError):
        day_folder(tmp_path / "typo", date(2026, 8, 5))


def test_unique_path_never_returns_an_occupied_name(tmp_path) -> None:
    """Saving twice in one day must not overwrite the morning's data."""

    first = unique_path(tmp_path, "scan", ".npz")
    assert first.name == "scan.npz"
    first.write_bytes(b"")

    second = unique_path(tmp_path, "scan", ".npz")
    assert second.name == "scan-2.npz"
    second.write_bytes(b"")

    assert unique_path(tmp_path, "scan", ".npz").name == "scan-3.npz"


def test_unique_path_sanitises_a_name_that_would_escape_or_break_the_folder(tmp_path) -> None:
    assert unique_path(tmp_path, "../../etc/passwd", ".npz").parent == tmp_path
    assert unique_path(tmp_path, "MOT loading: 3 ms", ".npz").name == "MOT-loading-3-ms.npz"
    assert unique_path(tmp_path, "///", ".npz").name == "untitled.npz"


def test_unique_path_requires_a_dotted_suffix_and_a_real_folder(tmp_path) -> None:
    with pytest.raises(ValueError):
        unique_path(tmp_path, "scan", "npz")
    with pytest.raises(NotADirectoryError):
        unique_path(tmp_path / "absent", "scan", ".npz")


def test_a_run_folder_takes_a_free_name_and_is_created(tmp_path) -> None:
    """An empty suffix asks for the directory a run leaves everything in.

    Names are taken against files and folders alike: a calibration folder and
    a file somebody saved beside it can never collide, and the second run of
    a day never writes into the first one's folder.
    """

    first = unique_path(tmp_path, "calibration", "")
    assert first.is_dir() and first.name == "calibration"
    second = unique_path(tmp_path, "calibration", "")
    assert second.is_dir() and second.name == "calibration-2"
    # A file of the same stem takes the name too, so neither shadows the other.
    (tmp_path / "report").write_text("x", encoding="utf-8")
    assert unique_path(tmp_path, "report", "").name == "report-2"
    assert unique_path(tmp_path, "calibration", ".json").name == "calibration.json"
