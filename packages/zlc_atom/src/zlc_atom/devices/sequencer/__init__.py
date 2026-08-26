"""Generic sequencer capability and physical-device descriptors."""

from .device import SequencerDevice, sequencer_archive_snapshot

__all__ = [
    "SequencerDevice",
    "sequencer_archive_snapshot",
]
