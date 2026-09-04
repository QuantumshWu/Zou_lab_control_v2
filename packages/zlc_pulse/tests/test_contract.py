from __future__ import annotations

import inspect

from zlc_pulse import PulseStreamer, RemotePulseStreamer, compile_sequence
from zlc_pulse.device import AppliedState
from zlc_pulse.schedule import trigger_times
from zlc_pulse.wire import pack_program, pack_scan_rows


def test_pure_function_signatures_match_contract() -> None:
    assert tuple(inspect.signature(compile_sequence).parameters) == (
        "sequence",
        "geom",
        "clock_hz",
        "slot_tick_scales",
    )
    assert tuple(inspect.signature(pack_program).parameters) == ("program", "params")
    assert tuple(inspect.signature(pack_scan_rows).parameters) == (
        "rows", "geom", "bank", "chunk",
    )
    assert tuple(inspect.signature(trigger_times).parameters) == (
        "prog", "channel", "table", "run_repeats", "scan_repeats",
    )


def test_applied_state_device_contract_signature() -> None:
    assert tuple(inspect.signature(PulseStreamer.load).parameters) == (
        "self", "prog", "source", "rows",
    )
    assert tuple(inspect.signature(PulseStreamer.fire).parameters) == (
        "self", "run_repeats", "scan_repeats",
    )
    assert tuple(inspect.signature(PulseStreamer.applied).parameters) == ("self",)
    assert AppliedState.__dataclass_params__.frozen is True
    assert tuple(inspect.signature(RemotePulseStreamer.load).parameters) == (
        "self", "prog", "source", "rows",
    )
    assert tuple(inspect.signature(RemotePulseStreamer.fire).parameters) == (
        "self", "run_repeats", "scan_repeats",
    )
    assert tuple(inspect.signature(RemotePulseStreamer.applied).parameters) == ("self",)


def test_the_remote_client_mirrors_the_config_value_surface() -> None:
    """One board, one answer to "which calibrated numbers am I playing".

    A method added to the local streamer and forgotten on the remote client
    fails only on the bench, hours after the change -- and the config surface
    is the one where a mismatch means the two ends disagree about what a
    pulse contains rather than merely erroring.
    """

    for streamer in (PulseStreamer, RemotePulseStreamer):
        assert tuple(inspect.signature(streamer.compile_pulse).parameters) == (
            "self", "sequence", "geom", "clock_hz", "slot_tick_scales",
        )
        assert tuple(inspect.signature(streamer.load_config_values).parameters) == (
            "self", "entries", "source",
        )
        assert tuple(inspect.signature(streamer.config_values).parameters) == ("self",)
        assert isinstance(streamer.config_source, property)
