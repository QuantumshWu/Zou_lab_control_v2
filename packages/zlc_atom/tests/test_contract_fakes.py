from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from zlc_pulse import load_streamer_config
from zlc_pulse.transport.memory import MemoryRegisterTransport

from tests.fakes import FakeNodeHost, FakePlane, FakePulseStreamer


def test_fake_plane_exposes_the_frozen_runtime_call_shapes() -> None:
    processor_signature = inspect.signature(FakePlane.publish_processor)
    assert processor_signature.parameters["source_publication"].kind is inspect.Parameter.KEYWORD_ONLY
    assert tuple(processor_signature.parameters) == ("self", "control", "outputs", "source_publication")
    assert tuple(inspect.signature(FakePlane.mark_changed).parameters) == ("self", "producer", "live_slot")
    assert tuple(inspect.signature(FakePlane.attach_latest_only_processor).parameters) == (
        "self",
        "node",
        "source_name",
        "initial_publication",
        "coherent",
    )
    assert not any(path.name == "signal_plane.py" for path in (Path(__file__).parents[1] / "src").rglob("*.py"))


def test_fake_node_host_executes_the_node_surface() -> None:
    class Node:
        def execute(self, context: object) -> object:
            return context

    host = FakeNodeHost(Node(), {"run": 1})
    host.start()
    assert host.poll()["state"] == "SUCCEEDED"
    assert host.result == {"run": 1}
    host.shutdown()


def test_fake_pulse_streamer_checks_geometry_fingerprint_on_open() -> None:
    # The board is the single source of geometry; a hand-built default would
    # pass here and mismatch the deployed bitstream on the bench.
    geom = load_streamer_config()["params"]
    transport = MemoryRegisterTransport(geom=geom)
    streamer = FakePulseStreamer(transport, geom, 100_000_000)
    streamer.open()
    streamer.load(object())
    streamer.fire()
    assert streamer.wait_done() is not None
    streamer.close()

    mismatch = MemoryRegisterTransport(layout_id=0)
    with pytest.raises(RuntimeError, match="geometry/layout mismatch"):
        FakePulseStreamer(mismatch, geom, 100_000_000).open()
