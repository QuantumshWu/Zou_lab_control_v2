"""Errors owned by the plotting/presentation runtime."""

from __future__ import annotations


class ZLCPlotError(Exception):
    """Base class for presentation-layer errors."""


class RevisionError(ZLCPlotError, ValueError):
    """A live update violates the monotonic revision contract."""


__all__ = ["RevisionError", "ZLCPlotError"]
