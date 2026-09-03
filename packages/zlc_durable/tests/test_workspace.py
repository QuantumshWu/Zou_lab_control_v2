"""Saved work lands under the day it was taken, and never lands on itself."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date
from threading import Barrier

import pytest

from zlc_durable import day_folder, day_folder_path, unique_path
from zlc_durable.workspace import DAY_FOLDER_PATTERN, day_folder_name


def _commit_process_payload(arguments: tuple[str, int]) -> tuple[str, bytes]:
    folder, index = arguments
    payload = f"process-{index}".encode()
    path = unique_path(
        folder,
        "shot",
        ".npz",
        writer=lambda temporary: temporary.write_bytes(payload),
    )
    return path.name, path.read_bytes()


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

    first = unique_path(
        tmp_path,
        "scan",
        ".npz",
        writer=lambda temporary: temporary.write_bytes(b"first"),
    )
    assert first.name == "scan.npz"

    second = unique_path(
        tmp_path,
        "scan",
        ".npz",
        writer=lambda temporary: temporary.write_bytes(b"second"),
    )
    assert second.name == "scan-2.npz"
    third = unique_path(
        tmp_path,
        "scan",
        ".npz",
        writer=lambda temporary: temporary.write_bytes(b"third"),
    )
    assert third.name == "scan-3.npz"
    assert [path.read_bytes() for path in (first, second, third)] == [
        b"first",
        b"second",
        b"third",
    ]


def test_unique_file_allocation_does_not_collapse_under_concurrency(tmp_path) -> None:
    callers = 32
    barrier = Barrier(callers)

    def allocate(_: int):
        barrier.wait()
        return unique_path(
            tmp_path,
            "shot",
            ".npz",
            writer=lambda temporary: temporary.write_bytes(b"complete"),
        )

    with ThreadPoolExecutor(max_workers=callers) as executor:
        paths = tuple(executor.map(allocate, range(callers)))

    assert len(set(paths)) == callers
    assert all(path.read_bytes() == b"complete" for path in paths)


def test_unique_file_commit_is_process_safe(tmp_path) -> None:
    callers = 16
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = tuple(
            executor.map(
                _commit_process_payload,
                ((str(tmp_path), index) for index in range(callers)),
            )
        )

    names = [name for name, _ in results]
    assert len(set(names)) == callers
    assert {payload for _, payload in results} == {
        f"process-{index}".encode() for index in range(callers)
    }


def test_unique_path_sanitises_a_name_that_would_escape_or_break_the_folder(tmp_path) -> None:
    write = lambda temporary: temporary.write_bytes(b"data")
    escaped = unique_path(tmp_path, "../../etc/passwd", ".npz", writer=write)
    assert escaped.parent == tmp_path
    assert (
        unique_path(tmp_path, "MOT loading: 3 ms", ".npz", writer=write).name
        == "MOT-loading-3-ms.npz"
    )
    assert unique_path(tmp_path, "///", ".npz", writer=write).name == "untitled.npz"
    assert unique_path(tmp_path, "神芯", ".npz", writer=write).name == "神芯.npz"
    assert unique_path(tmp_path, "CON", ".json", writer=write).name == "_CON.json"


def test_unique_path_requires_a_dotted_suffix_and_a_real_folder(tmp_path) -> None:
    with pytest.raises(ValueError):
        unique_path(tmp_path, "scan", "npz", writer=lambda path: None)
    with pytest.raises(ValueError):
        unique_path(tmp_path, "scan", ".x/inside", writer=lambda path: None)
    with pytest.raises(TypeError, match="requires writer"):
        unique_path(tmp_path, "scan", ".npz")
    with pytest.raises(NotADirectoryError):
        unique_path(
            tmp_path / "absent",
            "scan",
            ".npz",
            writer=lambda path: None,
        )


def test_unique_file_writer_failure_publishes_nothing(tmp_path) -> None:
    def fail(temporary):
        temporary.write_bytes(b"partial")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        unique_path(tmp_path, "shot", ".npz", writer=fail)

    assert not tuple(tmp_path.glob("shot*.npz"))
    assert not tuple(tmp_path.glob(".shot.*.npz"))


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
    assert (
        unique_path(
            tmp_path,
            "calibration",
            ".json",
            writer=lambda temporary: temporary.write_text("{}", encoding="utf-8"),
        ).name
        == "calibration.json"
    )


def test_day_folder_path_names_the_day_without_making_it(tmp_path) -> None:
    """A form that shows today's folder must not create it, let alone flush it."""

    named = day_folder_path(tmp_path, date(2026, 8, 5))
    assert named == tmp_path / "2026_08_05"
    assert not named.exists()
    # Naming is pure; only making refuses a root that is not there.
    assert day_folder_path(tmp_path / "missing", date(2026, 8, 5)) == (
        tmp_path / "missing" / "2026_08_05"
    )
    with pytest.raises(NotADirectoryError):
        day_folder(tmp_path / "missing", date(2026, 8, 5))
    with pytest.raises(ValueError):
        day_folder_path("relative/root", date(2026, 8, 5))
    # The write-side twin makes exactly that path.
    assert day_folder(tmp_path, date(2026, 8, 5)) == named.resolve()
    assert named.is_dir()
