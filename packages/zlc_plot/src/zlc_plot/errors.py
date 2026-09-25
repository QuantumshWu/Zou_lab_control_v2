"""Errors owned by the plotting/presentation runtime."""

from __future__ import annotations


class RevisionError(ValueError):
    """A live update violates the monotonic revision contract."""


__all__ = ["RevisionError"]
