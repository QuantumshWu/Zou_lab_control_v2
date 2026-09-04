"""One spelling per unit, on disk as well as in memory.

Delete this file together with ``tools/migrate_units.py`` and
``bin/migrate_units.bat`` once every machine holding pulses has run it.
"""

from __future__ import annotations

import json
from pathlib import Path

from pulse_fixtures import ordinary_imaging_sequence
from zlc_durable import write_readable_json
from zlc_workbench.pulse_state import PulseEditorState, read_pulse, write_pulse
from zlc_workbench.tools.migrate_units import BACKUP_SUFFIX, migrate_file


def _write_current(path: Path) -> Path:
    """One pulse document exactly as this build saves it."""

    write_pulse(path, PulseEditorState(sequence=ordinary_imaging_sequence()))
    return path


def _write_spelled(path: Path, unit: str) -> Path:
    """The same document, with one period authored in ``unit``.

    Derived from what this build writes and then re-spelled, so the fixture
    cannot drift from the schema the way a hand-typed document would.  Only
    the ONE period changes, and its duration with it: re-labelling a 0.002 s
    period as microseconds does not spell it differently, it makes it two
    nanoseconds, which a 20 ns clock rightly refuses.
    """

    _write_current(path)
    tree = json.loads(path.read_text(encoding="utf-8"))
    periods = list(tree["periods"])
    periods[0] = dict(periods[0], duration=100, unit=unit)
    tree["periods"] = periods
    write_readable_json(path, tree)
    return path


def test_an_accepted_spelling_still_reads_and_is_rewritten_canonical(
    tmp_path: Path,
) -> None:
    """``us`` is not wrong, it is just not the name we store.

    It has to keep reading -- it is what a keyboard has -- so this is not a
    rescue.  It is so that what sits on disk and what a Save would write are
    the same bytes, and a Save that changes nothing shows no diff.
    """

    path = _write_spelled(tmp_path / "imaging.json", "us")
    before = path.read_bytes()

    # It opens perfectly beforehand; that is the point.
    assert read_pulse(path).sequence is not None

    outcome = migrate_file(path)

    assert outcome.verdict == "rewritten", outcome.detail
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["periods"][0]["unit"] == "µs"
    assert path.with_name(path.name + BACKUP_SUFFIX).read_bytes() == before


def test_only_the_spelling_changes(tmp_path: Path) -> None:
    path = _write_spelled(tmp_path / "imaging.json", "us")
    before = json.loads(path.read_text(encoding="utf-8"))

    migrate_file(path)

    after = json.loads(path.read_text(encoding="utf-8"))
    assert [period["duration"] for period in after["periods"]] == [
        period["duration"] for period in before["periods"]
    ]
    assert {key: value for key, value in after.items() if key != "periods"} == {
        key: value for key, value in before.items() if key != "periods"
    }


def test_a_current_pulse_is_left_alone_and_keeps_no_backup(tmp_path: Path) -> None:
    """A backup of a file nobody changed is litter in the operator's folder."""

    path = _write_current(tmp_path / "imaging.json")
    before = path.read_bytes()

    outcome = migrate_file(path)

    assert outcome.verdict == "already current"
    assert path.read_bytes() == before
    assert not path.with_name(path.name + BACKUP_SUFFIX).exists()


def test_a_backup_is_never_the_thing_migrated_next_time(tmp_path: Path) -> None:
    path = _write_spelled(tmp_path / "imaging.json", "us")
    migrate_file(path)

    backup = path.with_name(path.name + BACKUP_SUFFIX)
    assert backup.suffix != ".json", "the Open dialog must not offer it"
    assert migrate_file(backup).verdict == "not a pulse"
    # A second run refuses rather than writing over the original it kept.
    second = migrate_file(path)
    assert second.verdict == "REFUSED" and "backup already exists" in second.detail
