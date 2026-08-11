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

import pytest

from zlc_atom.nodes import calibration_pulse_template_bytes


def test_the_three_camera_windows_share_one_trigger_cadence() -> None:
    tree = json.loads(calibration_pulse_template_bytes().decode("utf-8"))
    periods = {
        str(period["period_id"]): float(period["duration"])
        for period in tree["periods"]
    }
    assert set(periods) == {
        "load", "long_before", "gap_0", "short", "gap_1", "long_after"
    }
    for period in tree["periods"]:
        assert period["unit"] == "s"

    first_interval = periods["long_before"] + periods["gap_0"]
    second_interval = periods["short"] + periods["gap_1"]
    assert first_interval == pytest.approx(second_interval), (
        "the two inter-trigger intervals must be equal: the camera fires on a "
        "fixed cadence, and the gaps exist to keep the short window on it"
    )
    assert periods["gap_1"] == pytest.approx(
        periods["long_before"] + periods["gap_0"] - periods["short"]
    ), "gap_1 is DERIVED, not free: long + gap_0 - short"
