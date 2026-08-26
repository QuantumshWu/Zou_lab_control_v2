"""Common contract and binding for phase-only SLM devices."""

from .device import SlmAdapter, bind_slm, canonical_phase


def open_slm_control(session, device_key: str, window_ratio=None):
    """Load the concrete Qt editor only when the device card requests it.

    A lazy shim forwards a call, so it forwards ALL of it: a keyword it
    does not restate is a keyword its caller cannot use, and the failure
    lands as a swallowed TypeError from a device card rather than at the
    signature that dropped it.
    """

    from .editor import open_slm_control as open_control

    return open_control(session, device_key, window_ratio=window_ratio)


__all__ = ["SlmAdapter", "bind_slm", "canonical_phase", "open_slm_control"]
