"""The calibrated set reaches the board, and the archive says which one did.

A config parameter is filled by the SEQUENCER at compile.  So two things have
to be true for a bench to be believable: the board is holding the workspace's
current set without anyone doing anything, and the record of a run says which
set that was.  Neither is visible in a pulse file, which is the whole point.
"""

from __future__ import annotations

import math

import pytest

from zlc_atom.devices.sequencer import sequencer_archive_snapshot
from zlc_atom.devices.simulation import SimulationWorld, VirtualSequencer
from zlc_atom.pulse_values import CURRENT_CONFIG_VALUES, read_config_values, write_config_values


def test_a_saved_set_round_trips_through_the_shared_grammar(tmp_path) -> None:
    path = tmp_path / CURRENT_CONFIG_VALUES
    write_config_values(
        path, {"gate_delay": (40.0, "ns")}, name="today", source="calibration"
    )
    name, source, entries = read_config_values(path)
    assert (name, source) == ("today", "calibration")
    assert entries == {"gate_delay": (40.0, "ns")}


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
            {"gate_delay": (40.0, "ns")}, source="/bench/config_values/current.json"
        )
        after = sequencer_archive_snapshot(
            description=board,
            config=sequencer.config_values(),
            state=sequencer.snapshot(),
        )
        assert after["config"] == {"gate_delay": [40.0, "ns"]}
        # And where it came from, so the file can be found again.
        assert after["state"]["config_source"] == "/bench/config_values/current.json"
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


def test_a_session_hands_its_board_the_workspace_set(tmp_path, monkeypatch) -> None:
    """Loaded once, by the session, for everything that fires through it.

    A scan, a calibration and a bound Pulse Editor all reach the board through
    this session's devices, so this is the one place the file has to be read.
    """

    from zlc_workbench.session import ExperimentSession, Workspace

    (tmp_path / "pulses").mkdir()
    space = Workspace(tmp_path).prepare()
    write_config_values(
        space.config_values / CURRENT_CONFIG_VALUES,
        {"gate_delay": (40.0, "ns")},
        name="current",
    )

    session = ExperimentSession.open(workspace=tmp_path, template="virtual")
    try:
        assert session.sequencer.config_values() == {"gate_delay": (40.0, "ns")}
        assert session.sequencer.config_source.endswith(CURRENT_CONFIG_VALUES)
    finally:
        session.close()


def test_a_workspace_with_no_set_is_silent(tmp_path) -> None:
    """An empty seed is deliberate: a seeded delay is a wrong calibration.

    So the board simply holds nothing, every ordinary pulse still runs, and a
    pulse that needs a calibrated number is refused by name when it compiles.
    """

    from zlc_workbench.session import ExperimentSession, Workspace

    (tmp_path / "pulses").mkdir()
    space = Workspace(tmp_path).prepare()
    seeded = space.config_values / CURRENT_CONFIG_VALUES
    assert seeded.is_file()
    assert read_config_values(seeded)[2] == {}

    session = ExperimentSession.open(workspace=tmp_path, template="virtual")
    try:
        assert session.sequencer.config_values() == {}
    finally:
        session.close()
