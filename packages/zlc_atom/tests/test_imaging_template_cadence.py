"""The imaging template's odd-looking gaps are a designed trigger cadence.

``gap_0 = 0.1 ms`` and ``gap_1 = 15.1 ms`` read like typos -- they were
reported as suspected ones -- and they are not.  They make the camera's three
trigger windows START at a constant spacing:

    long_before + gap_0  =  20.0 + 0.1  =  20.1 ms
    short       + gap_1  =   5.0 + 15.1 =  20.1 ms

A readout-limited sensor wants its triggers on a fixed cadence; the gaps are
chosen so the SHORT window sits on the same 20.1 ms grid as the long ones,
which is why gap_1 must equal ``long + gap_0 - short`` exactly.  Anyone
"fixing" 15.1 to a rounder number breaks the cadence, and this test is where
they find that out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from zlc_pulse import nanoseconds_per

from zlc_atom.nodes.calibration.pulse import load_calibration_pulse_template
from tests.pulse_fixture import pulse_document


def test_the_three_camera_windows_share_one_trigger_cadence() -> None:
    tree = json.loads(pulse_document("imaging_template.json").decode("utf-8"))
    raw_periods = {str(period["period_id"]): period for period in tree["periods"]}
    periods = {
        str(period["period_id"]): (
            float(period["duration"])
            * float(nanoseconds_per(str(period["unit"])) / nanoseconds_per("s"))
        )
        for period in tree["periods"]
    }
    assert set(periods) == {
        "load", "long_before", "gap_0", "short", "gap_1", "long_after"
    }
    assert raw_periods["load"]["duration"] == 100
    # The canonical spelling, which is the one the registry shows and the
    # document therefore stores.  "us" still READS -- it is an accepted
    # alias -- but a saved pulse holds one name per unit.
    assert raw_periods["load"]["unit"] == "µs"
    assert all(
        period["unit"] == "s"
        for period_id, period in raw_periods.items()
        if period_id != "load"
    )

    first_interval = periods["long_before"] + periods["gap_0"]
    second_interval = periods["short"] + periods["gap_1"]
    assert first_interval == pytest.approx(second_interval), (
        "the two inter-trigger intervals must be equal: the camera fires on a "
        "fixed cadence, and the gaps exist to keep the short window on it"
    )
    assert periods["gap_1"] == pytest.approx(
        periods["long_before"] + periods["gap_0"] - periods["short"]
    ), "gap_1 is DERIVED, not free: long + gap_0 - short"


def test_calibration_accepts_the_complete_document_saved_by_the_pulse_editor(
    tmp_path: Path,
) -> None:
    tree = json.loads(pulse_document("imaging_template.json").decode("utf-8"))
    tree["editor"] = {
        "visible_ports": None,
        "scan_source": "",
        "scan_rows": [],
        "scan_source_dirty": False,
        "scan_repeats": 0,
    }
    path = tmp_path / "imaging_template.json"
    path.write_text(json.dumps(tree), encoding="utf-8")

    sequence = load_calibration_pulse_template(path)

    assert sequence.name == "imaging_template"
