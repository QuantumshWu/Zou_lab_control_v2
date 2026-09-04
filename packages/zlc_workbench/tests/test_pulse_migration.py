"""The one-shot that lets a pulse survive the bracket rename.

Delete this file together with ``tools/migrate_pulses.py`` and
``bin/migrate_pulses.bat`` once every machine holding pulses has run it.

The old shape here is DERIVED by taking a document the product writes today
and undoing the rename that produced it.  A hand-typed old pulse would only
prove that this tool agrees with whatever the test author remembered.
"""

from __future__ import annotations

from pathlib import Path

from pulse_fixtures import ordinary_imaging_sequence
from zlc_durable import write_readable_json
from zlc_pulse import sequence_to_tree
from zlc_workbench.pulse_state import read_pulse, state_from_tree
from zlc_workbench.tools.migrate_pulses import BACKUP_SUFFIX, migrate_file


def _current_document() -> dict:
    """One pulse document exactly as this build writes it."""

    return dict(sequence_to_tree(ordinary_imaging_sequence()))


def _as_written_before_the_bracket_split(document: dict) -> dict:
    """The same pulse, in the shape ``663de9b`` renamed away from."""

    older = dict(document)
    older["repeat"] = older.pop("bracket")
    older.pop("run_repeats")
    return older


def _write(path: Path, document: dict) -> Path:
    write_readable_json(path, document)
    return path


def test_a_pulse_from_before_the_rename_opens_again(tmp_path: Path) -> None:
    current = _current_document()
    path = _write(
        tmp_path / "imaging.json", _as_written_before_the_bracket_split(current)
    )

    # The failure the operator reports, reproduced before anything is fixed.
    try:
        read_pulse(path)
    except ValueError as error:
        assert "repeat" in str(error)
    else:  # pragma: no cover - the migration would have nothing to do
        raise AssertionError("the old shape must not read as a current pulse")

    outcome = migrate_file(path)

    assert outcome.verdict == "migrated", outcome.detail
    assert read_pulse(path) == state_from_tree(current)


def test_the_original_is_kept_beside_the_file_it_came_from(tmp_path: Path) -> None:
    older = _as_written_before_the_bracket_split(_current_document())
    path = _write(tmp_path / "imaging.json", older)
    before = path.read_bytes()

    outcome = migrate_file(path)

    backup = path.with_name(path.name + BACKUP_SUFFIX)
    assert outcome.backup == backup
    assert backup.read_bytes() == before
    # Not a pulse name: the Open dialog must not offer it, and a second run
    # must not try to migrate the copy it just made.
    assert backup.suffix != ".json"
    assert migrate_file(backup).verdict == "not a pulse"


def test_a_current_pulse_is_not_rewritten(tmp_path: Path) -> None:
    path = _write(tmp_path / "imaging.json", _current_document())
    before = path.read_bytes()

    outcome = migrate_file(path)

    assert outcome.verdict == "already current"
    assert path.read_bytes() == before
    assert not path.with_name(path.name + BACKUP_SUFFIX).exists()


def test_a_field_this_tool_does_not_know_is_refused_not_dropped(tmp_path: Path) -> None:
    """Silently deleting what it cannot explain is the one unforgivable move.

    A migration that drops the fields it does not recognise turns "this file
    will not open" into "this file opens and has lost something", which is
    the failure nobody notices until the data is gone.
    """

    older = _as_written_before_the_bracket_split(_current_document())
    older["something_an_older_build_wrote"] = 7
    path = _write(tmp_path / "imaging.json", older)
    before = path.read_bytes()

    outcome = migrate_file(path)

    assert outcome.verdict == "REFUSED"
    assert "something_an_older_build_wrote" in outcome.detail
    assert path.read_bytes() == before
    assert not path.with_name(path.name + BACKUP_SUFFIX).exists()
