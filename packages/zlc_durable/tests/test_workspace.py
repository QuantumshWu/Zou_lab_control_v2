"""Saved work lands under the day it was taken, and never lands on itself."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date
from pathlib import Path
from threading import Barrier

import pytest

from zlc_durable import day_folder, day_folder_path, unique_path


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


def test_unique_path_never_returns_an_occupied_name(tmp_path, monkeypatch) -> None:
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

    # A FAT32/exFAT stick has no hard links: the name is claimed instead.
    def no_links(*_args, **_kwargs):
        raise OSError(1, "Incorrect function")

    monkeypatch.setattr("zlc_durable.durability.os.link", no_links)
    fourth = unique_path(
        tmp_path,
        "scan",
        ".npz",
        writer=lambda temporary: temporary.write_bytes(b"fourth"),
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "scan-2.npz", "scan-3.npz", "scan-4.npz", "scan.npz",
    ] and fourth.read_bytes() == b"fourth"

    # A move onto the claim that fails takes the empty claim with it: nothing
    # is published under the final name, and the temporary goes too.
    def refused(*_args, **_kwargs):
        raise OSError(13, "Access is denied")

    monkeypatch.setattr("zlc_durable.durability.os.replace", refused)
    with pytest.raises(OSError, match="denied"):
        unique_path(
            tmp_path,
            "scan",
            ".npz",
            writer=lambda temporary: temporary.write_bytes(b"fifth"),
        )
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "scan-2.npz", "scan-3.npz", "scan-4.npz", "scan.npz",
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


def test_a_flush_failure_after_publication_names_what_landed(tmp_path, monkeypatch) -> None:
    """The error after a publish says the artifact is there, and where.

    Once the link or the mkdir has happened the work is complete and visible;
    only its directory entry's durability is unconfirmed.  An error that named
    just the directory read as "cannot save", and an operator who retried got
    ``shot-2.json`` beside a ``shot.json`` that was already whole.
    """

    import zlc_durable.durability as durability

    real_flush = durability.flush_directory
    folder = tmp_path.resolve()

    def flush(directory):
        if Path(directory).resolve() == folder:
            raise durability.DirectoryDurabilityError("scripted flush failure")
        real_flush(directory)

    monkeypatch.setattr(durability, "flush_directory", flush)
    with pytest.raises(durability.DirectoryDurabilityError, match="scripted flush failure") as caught:
        unique_path(
            tmp_path,
            "shot",
            ".json",
            writer=lambda temporary: temporary.write_text('{"complete": true}', encoding="utf-8"),
        )
    shot = folder / "shot.json"
    assert caught.value.published == shot
    assert shot.read_text(encoding="utf-8") == '{"complete": true}'
    assert str(shot) in str(caught.value) and "published and visible" in str(caught.value)
    assert not tuple(folder.glob(".shot.*.json"))

    with pytest.raises(durability.DirectoryDurabilityError, match="scripted flush failure") as caught:
        unique_path(tmp_path, "calibration", "")
    run = folder / "calibration"
    assert caught.value.published == run and run.is_dir()
    assert str(run) in str(caught.value)


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
    # Naming is pure: a root that is not there is still named.
    assert day_folder_path(tmp_path / "missing", date(2026, 8, 5)) == (
        tmp_path / "missing" / "2026_08_05"
    )
    with pytest.raises(ValueError):
        day_folder_path("relative/root", date(2026, 8, 5))
