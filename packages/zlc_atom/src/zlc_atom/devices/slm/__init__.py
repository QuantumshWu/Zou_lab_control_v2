"""Common contract and binding for phase-only SLM devices."""

from .device import SlmAdapter, bind_slm, canonical_phase

__all__ = ["SlmAdapter", "bind_slm", "canonical_phase"]
