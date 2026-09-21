"""The calibrated set reaches the board, and the archive says which one did.

A Config parameter is applied by the SEQUENCER at load/Fire. Two things have
to be true for a bench to be believable: the board is holding the workspace's
current set without anyone doing anything, and the record of a run says which
set that was.  Neither is visible in a pulse file, which is the whole point.
"""

from __future__ import annotations

import math

import pytest

from zlc_atom.devices.sequencer import sequencer_archive_snapshot
from zlc_atom.devices.simulation import SimulationWorld
from zlc_atom.devices.simulation.sequencer import VirtualSequencer
from zlc_pulse import CURRENT_CONFIG_VALUES, read_config_values, write_config_values


def test_a_saved_set_round_trips_through_the_shared_grammar(tmp_path) -> None:
    path = tmp_path / CURRENT_CONFIG_VALUES
    write_config_values(
        path, {"trigger_delay": (40.0, "ns")}
    )
    entries = read_config_values(path)
    assert entries == {"trigger_delay": (40.0, "ns")}


def test_the_archive_records_the_set_that_was_in_force() -> None:
    """A config file is overwritten by the next calibration.

    So naming the pulse says nothing about which numbers played.  The board's
    own snapshot is where a dataset keeps what it was made with, beside the
    board facts it already keeps -- not in the pulse's section, because the
    pulse never held them.
    """

    sequencer = VirtualSequencer(world=SimulationWorld())
    sequencer.open()
    try:
        board = sequencer.describe()
        before = sequencer_archive_snapshot(
            description=board, config=sequencer.config_values()
        )
        assert before["config"] == {}

        sequencer.load_config_values(
            {"trigger_delay": (40.0, "ns")}, source="/bench/config_values/current.json"
        )
        after = sequencer_archive_snapshot(
            description=board,
            config=sequencer.config_values(),
            state=sequencer.snapshot(),
        )
        assert after["config"] == {"trigger_delay": [40.0, "ns"]}
        # And where it came from, so the file can be found again.
        assert after["state"]["config_source"] == "/bench/config_values/current.json"
    finally:
        sequencer.close()


def test_the_archive_carries_the_program_and_the_pulse_that_played() -> None:
    """Beside the board's facts: what it PLAYED, as timing, not as a file name.

    A record that only named the pulse's file said nothing once that file
    was edited, and named it by the document's own name -- "untitled" for
    any pulse drawn in the editor and never renamed on screen.
    """

    from zlc_pulse import resolve_api_parameters
    from tests.pulse_fixture import pulse_sequence

    sequencer = VirtualSequencer(world=SimulationWorld())
    sequencer.open()
    try:
        board = sequencer.describe()
        filled, program = sequencer.compile_pulse(
            resolve_api_parameters(pulse_sequence("imaging_template.json")),
            board.geometry,
            board.clock_hz,
        )
        snapshot = sequencer_archive_snapshot(
            description=board,
            program=program,
            source=filled,
            rows=((1, 2), (3, 4)),
            run_repeats=5,
            scan_repeats=2,
        )
        played = snapshot["program"]
        assert played["digest"] == program.digest
        assert played["duration_seconds"] == pytest.approx(program.duration_seconds)
        assert played["loops"] == [list(loop) for loop in program.loops]
        assert played["rows"] == [[1, 2], [3, 4]]
        assert (played["run_repeats"], played["scan_repeats"]) == (5, 2)
        document = snapshot["pulse"]
        assert document["name"] == filled.name
        assert [period["name"] for period in document["periods"]] == [
            period.name for period in filled.periods
        ]
        with pytest.raises(ValueError, match="give the program"):
            sequencer_archive_snapshot(description=board, rows=((1,),))
    finally:
        sequencer.close()


def test_the_archive_keeps_config_source_in_its_whitelist() -> None:
    """The snapshot drops any state key it was not told about, silently.

    A key added to the device and forgotten here vanishes from every run
    record with nothing failing, so the list is stated once and pinned.
    """

    state = {
        "opened": True,
        "loaded": True,
        "config_source": "/bench/current.json",
        "invented": "should not survive",
    }
    result = sequencer_archive_snapshot(state=state)["state"]
    assert result["config_source"] == "/bench/current.json"
    assert "invented" not in result


def test_session_init_does_not_read_an_unselected_config_file(tmp_path) -> None:
    """An old file cannot block Init; only explicit Config Load reads a file."""

    from zlc_workbench.session import ExperimentSession, Workspace

    (tmp_path / "pulses").mkdir()
    space = Workspace(tmp_path).prepare()
    current = space.config_values / CURRENT_CONFIG_VALUES
    old = '{"format":"zlc.pulse.config_values","name":"current","source":"hand","values":{}}'
    current.write_text(old, encoding="utf-8")

    session = ExperimentSession.open(workspace=tmp_path, template="virtual")
    try:
        assert session.sequencer.config_values() == {}
        assert session.sequencer.config_source == ""
        assert current.read_text(encoding="utf-8") == old
        chosen = space.config_values / "chosen.json"
        write_config_values(chosen, {"trigger_delay": (40.0, "ns")})
        session.sequencer.load_config_file(chosen)
        assert session.sequencer.config_values() == {"trigger_delay": (40.0, "ns")}
        assert session.sequencer.config_source == str(chosen)
    finally:
        session.close()


def test_a_workspace_with_no_set_is_silent(tmp_path) -> None:
    """Opening a workspace creates a directory, not an implicitly active file."""

    from zlc_workbench.session import ExperimentSession, Workspace

    (tmp_path / "pulses").mkdir()
    space = Workspace(tmp_path).prepare()
    seeded = space.config_values / CURRENT_CONFIG_VALUES
    assert space.config_values.is_dir()
    assert not seeded.exists()

    session = ExperimentSession.open(workspace=tmp_path, template="virtual")
    try:
        assert session.sequencer.config_values() == {}
    finally:
        session.close()
