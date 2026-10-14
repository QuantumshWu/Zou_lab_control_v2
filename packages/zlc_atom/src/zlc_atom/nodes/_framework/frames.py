"""Which frames of a cycle each entry of an assignment reads.

An occupancy reads a load frame and a readout frame with calibrations trained
on each.  The operator says it the way they think it -- THIS file reads frame
0, THAT one the rest -- so an assignment is a list of entries, each claiming
frames:

* frame 0, the first frame, or a positive middle-frame index (the numbering
  the camera, Frame Survival and the Calibration task all use);
* ``"last"``, whatever the last frame of the cycle is;
* ``"rest"``, every frame no entry names.

"last" and "rest" follow the frame count, so a cycle that gains or loses a
frame keeps meaning what was authored; a middle-frame index keeps its position
and becomes absent rather than taking over the last frame of a shorter cycle. A
frame is read by exactly one entry.  Only "rest" ranks below anything else:
every other collision -- two entries reaching one frame, a number the source
does not carry, a frame nobody reads -- is refused with the entries named,
never settled silently.  Nothing here ever edits what was authored: the same
claims are valid again once the frame count that suited them returns.

There is no "first": frame 0 is the first frame of every cycle, so a second
spelling of it would only be a second way to collide with itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

LAST = "last"
REST = "rest"
FRAME_NAMES = (LAST, REST)

FrameToken = int | str


@dataclass(frozen=True)
class FrameClaim:
    """One entry of an assignment: what it carries and the frames it claims."""

    value: object
    frames: tuple[FrameToken, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", canonical_frames(self.frames))


def frame_token(raw: object) -> FrameToken:
    """One claimed frame: a number from 0, ``"last"`` or ``"rest"``."""

    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in FRAME_NAMES:
            return text
        if text.isdigit():
            return int(text)
    elif isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return int(raw)
    raise ValueError(
        f"{raw!r} is not a frame: use a frame number from 0, 'last' or 'rest'"
    )


def frame_identity(index: int, count: int) -> FrameToken:
    """Author a displayed endpoint by identity, and a middle frame by index."""

    return LAST if index > 0 and index == count - 1 else index


def resolve_frame(token: FrameToken, count: int) -> int | None:
    """Resolve one frame, retaining an absent numbered claim in the draft."""

    if token == LAST:
        return count - 1 if count else None
    if isinstance(token, int):
        return token if token == 0 and count > 0 or 0 < token < count - 1 else None
    raise ValueError("one frame must be a number from 0 or 'last'")


def canonical_frames(raw: object) -> tuple[FrameToken, ...]:
    """Claimed frames in one order -- numbers ascending, then last, then rest."""

    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("frames must be a list of frame numbers, 'last' or 'rest'")
    tokens = [frame_token(item) for item in raw]
    repeated = sorted({str(token) for token in tokens if tokens.count(token) > 1})
    if repeated:
        raise ValueError(f"frame {', '.join(repeated)} is listed twice")
    numbers = sorted(token for token in tokens if isinstance(token, int))
    return (*numbers, *(name for name in FRAME_NAMES if name in tokens))


def describe_frames(frames: Sequence[FrameToken]) -> str:
    """The one text form of claimed frames: ``"0, 2, last"``."""

    return ", ".join(str(token) for token in frames) or "no frame"


def check_claims(
    frames_by_entry: Sequence[Sequence[object]],
    names: Sequence[str],
) -> tuple[tuple[FrameToken, ...], ...]:
    """What can be refused without knowing the frame count.

    Every entry claims at least one frame, and every frame number, "last"
    and "rest" belongs to at most one entry.
    """

    if len(frames_by_entry) != len(names):
        raise ValueError("every entry needs a name")
    claims = tuple(canonical_frames(frames) for frames in frames_by_entry)
    for name, frames in zip(names, claims):
        if not frames:
            raise ValueError(f"{name} reads no frame: choose its frames or remove it")
    owners: dict[FrameToken, str] = {}
    for name, frames in zip(names, claims):
        for token in frames:
            if token in owners:
                what = f"frame {token}" if isinstance(token, int) else repr(token)
                raise ValueError(f"{what} is claimed by both {owners[token]} and {name}")
            owners[token] = name
    return claims


@dataclass(frozen=True)
class FrameReach:
    """How one frame count reads an assignment, problems included."""

    count: int
    #: Per frame: every ``(entry, how)`` reaching it, ``how`` being the frame
    #: number, ``"last"`` or ``"rest"``.
    readers: tuple[tuple[tuple[int, str], ...], ...]
    #: ``(entry, number)`` for every claimed number the cycle does not have.
    absent: tuple[tuple[int, int], ...]

    def entry_of(self, frame: int) -> int | None:
        """The one entry reading ``frame``; None when none or several do."""

        readers = self.readers[frame]
        return readers[0][0] if len(readers) == 1 else None

    def frames_of(self, entry: int) -> tuple[int, ...]:
        """Every frame ``entry`` reaches, conflicts included."""

        return tuple(
            frame
            for frame, readers in enumerate(self.readers)
            if any(reader == entry for reader, _how in readers)
        )


def reach_frames(
    frames_by_entry: Sequence[Sequence[FrameToken]],
    count: int,
) -> FrameReach:
    """Which entries reach each of ``count`` frames, without refusing anything."""

    count = int(count)
    if count < 1:
        raise ValueError("a cycle has at least one frame")
    readers: list[list[tuple[int, str]]] = [[] for _ in range(count)]
    absent: list[tuple[int, int]] = []
    for entry, frames in enumerate(frames_by_entry):
        for token in frames:
            if token != REST:
                frame = resolve_frame(token, count)
                if frame is None:
                    absent.append((entry, token))
                else:
                    readers[frame].append((entry, str(token)))
    rest = [entry for entry, frames in enumerate(frames_by_entry) if REST in frames]
    for frame in range(count):
        if not readers[frame]:
            readers[frame].extend((entry, REST) for entry in rest)
    return FrameReach(count, tuple(_one_per_entry(found) for found in readers), tuple(absent))


def _one_per_entry(found: Sequence[tuple[int, str]]) -> tuple[tuple[int, str], ...]:
    """One entry reaching a frame twice (0 and last, in a one-frame cycle) is one reader."""

    first: dict[int, str] = {}
    for entry, how in found:
        first.setdefault(entry, how)
    return tuple(first.items())


def frame_problems(
    frames_by_entry: Sequence[Sequence[FrameToken]],
    count: int,
    names: Sequence[str],
    *,
    what: str = "entry",
) -> tuple[str, ...]:
    """Why ``count`` frames cannot be read this way, in operator words."""

    reach = reach_frames(frames_by_entry, count)
    span = "only frame 0" if count == 1 else f"0-{count - 1}"
    problems = [
        f"{names[entry]} names frame {number}, but the source carries "
        f"{count} frame{'s' if count != 1 else ''} ({span}); "
        "numbered middle frames cannot stand for the last frame"
        for entry, number in reach.absent
    ]
    for frame, readers in enumerate(reach.readers):
        if len(readers) > 1:
            problems.append(
                f"frame {frame} is read by both "
                + " and ".join(f"{names[entry]} ({how})" for entry, how in readers)
            )
    unread = [frame for frame, readers in enumerate(reach.readers) if not readers]
    if unread:
        frames = ", ".join(str(frame) for frame in unread)
        problems.append(
            f"frame{'s' if len(unread) > 1 else ''} {frames} "
            f"{'have' if len(unread) > 1 else 'has'} no {what}: give "
            f"{'them' if len(unread) > 1 else 'it'} to one, or let one read the rest"
        )
    return tuple(problems)


def assign_frames(
    frames_by_entry: Sequence[Sequence[FrameToken]],
    count: int,
    names: Sequence[str],
    *,
    what: str = "entry",
) -> tuple[int, ...]:
    """The one entry each of ``count`` frames is read by, or every reason why not."""

    problems = frame_problems(frames_by_entry, count, names, what=what)
    if problems:
        raise ValueError("; ".join(problems))
    reach = reach_frames(frames_by_entry, count)
    return tuple(reach.entry_of(frame) for frame in range(count))


__all__ = [
    "FRAME_NAMES",
    "FrameClaim",
    "FrameReach",
    "FrameToken",
    "LAST",
    "REST",
    "assign_frames",
    "canonical_frames",
    "check_claims",
    "describe_frames",
    "frame_problems",
    "frame_token",
    "frame_identity",
    "resolve_frame",
    "reach_frames",
]
