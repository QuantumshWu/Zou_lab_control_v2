from __future__ import annotations

import inspect

from zlc_pulse import PulseStreamer, RemotePulseStreamer, compile_sequence
from zlc_pulse.device import AppliedState
from zlc_pulse.schedule import trigger_times
from zlc_pulse.wire import pack_program, pack_scan_rows


def test_pure_function_signatures_match_contract() -> None:
    assert tuple(inspect.signature(compile_sequence).parameters) == ("sequence", "geom", "clock_hz")
    assert tuple(inspect.signature(pack_program).parameters) == ("program", "params")
    assert tuple(inspect.signature(pack_scan_rows).parameters) == ("rows", "geom", "bank", "chunk")
    assert tuple(inspect.signature(trigger_times).parameters) == ("prog", "channel", "table")


def test_applied_state_device_contract_signature() -> None:
    assert tuple(inspect.signature(PulseStreamer.load).parameters) == ("self", "prog", "source")
    assert tuple(inspect.signature(PulseStreamer.applied).parameters) == ("self",)
    assert AppliedState.__dataclass_params__.frozen is True
    assert tuple(inspect.signature(RemotePulseStreamer.load).parameters) == ("self", "prog", "source")
    assert tuple(inspect.signature(RemotePulseStreamer.applied).parameters) == ("self",)
