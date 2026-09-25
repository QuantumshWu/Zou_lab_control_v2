from __future__ import annotations

import inspect

from zlc_pulse import PulseStreamer, RemotePulseStreamer


def test_the_remote_client_mirrors_the_local_streamer() -> None:
    """One board, one surface, whichever end of the wire drives it.

    A method added to the local streamer and forgotten on the remote client
    fails only on the bench, hours after the change -- and the config surface
    is the one where a mismatch means the two ends disagree about what a
    pulse contains rather than merely erroring.
    """

    for name in ("load", "fire", "applied", "compile_pulse", "load_config_values", "config_values"):
        local = tuple(inspect.signature(getattr(PulseStreamer, name)).parameters)
        remote = tuple(inspect.signature(getattr(RemotePulseStreamer, name)).parameters)
        assert local == remote, name
    for streamer in (PulseStreamer, RemotePulseStreamer):
        assert isinstance(streamer.config_source, property)
