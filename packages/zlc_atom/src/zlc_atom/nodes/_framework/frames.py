"""Which frames of a cycle each entry of an assignment reads.

An occupancy reads a load frame and a readout frame with calibrations trained
on each.  The operator says it the way they think it -- THIS file reads the
first frame, THAT one the last, ANOTHER the rest -- so an assignment is a list
of entries, each claiming frames.  The two ends of a cycle are named by what
they ARE, the frames between them by their number:

* ``"first"``, the first frame of the cycle -- frame 0, in every cycle (a
  claim written as frame 0 IS this claim, and is stored as ``"first"``);
* ``"last"``, whatever the last frame of the cycle is;
* a positive number, a MIDDLE frame strictly between the first and the last
  (the numbering the camera, Frame Survival and the Calibration task use);
* ``"rest"``, every frame no entry names.

So a cycle that gains or loses a frame keeps meaning what was authored: the
ends follow the cycle, a middle number keeps its position, and a middle
number the shorter cycle no longer has between its ends becomes ABSENT -- it
never takes over the last frame.  A frame is read by exactly one entry.  Only
"rest" ranks below anything else: every other collision -- two entries
reaching one frame ("first" and "last" in a one-frame cycle), a middle number
the source does not carry, a frame nobody reads -- is refused with the
entries named, never settled silently.  One entry reaching a frame two ways
is one reader.  Nothing here ever edits what was authored: the same claims
are valid again once the frame count that suited them returns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

FIRST = "first"
LAST = "last"
REST = "rest"
FRAME_NAMES = (FIRST, LAST, REST)

FrameToken = int | str


@dataclass(frozen=True)
class FrameClaim:
    """One entry of an assignment: what it carries and the frames it claims."""

    value: object
    frames: tuple[FrameToken, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", canonical_frames(self.frames))


def frame_token(raw: object) -> FrameToken:
    """One claimed frame: a number from 0, ``"first"``, ``"last"`` or ``"rest"``."""

    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in FRAME_NAMES:
            return text
        if text.isdigit():
            return int(text)
    elif isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return int(raw)
    raise ValueError(
        f"{raw!r} is not a frame: use 'first', 'last', 'rest' or a frame number from 0"
    )


def frame_identity(index: int, count: int) -> FrameToken:
    """Author a displayed frame: an end by identity, a middle frame by index.

    Frame Survival's pair matrix writes its endpoints through this; frame 0
    stays the number 0 there (a pair's start), the last frame is ``"last"``.
    """

    return LAST if index > 0 and index == count - 1 else index


def resolve_frame(token: FrameToken, count: int) -> int | None:
    """The frame one claim reaches in a ``count``-frame cycle; None when it has none.

    A middle number reaches its frame only strictly between the ends; the
    claim stays in the draft and is reported absent otherwise.
    """

    if token == FIRST or token == 0:
        return 0 if count > 0 else None
    if token == LAST:
        return count - 1 if count > 0 else None
    if isinstance(token, int):
        return token if 0 < token < count - 1 else None
    raise ValueError("one frame must be 'first', 'last' or a frame number from 0")


def canonical_frames(raw: object) -> tuple[FrameToken, ...]:
    """Claimed frames in one order -- middle numbers ascending, then first, last, rest.

    Frame 0 is written ``"first"``: one claim, one spelling, so an entry
    naming both is one claim.
    """

    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("frames must be a list of 'first', 'last', 'rest' or frame numbers")
    written = [frame_token(item) for item in raw]
    repeated = sorted({str(token) for token in written if written.count(token) > 1})
    if repeated:
        raise ValueError(f"frame {', '.join(repeated)} is listed twice")
    tokens = list(dict.fromkeys(FIRST if token == 0 else token for token in written))
    numbers = sorted(token for token in tokens if isinstance(token, int))
    return (*numbers, *(name for name in FRAME_NAMES if name in tokens))


def describe_frames(frames: Sequence[FrameToken]) -> str:
    """The one text form of claimed frames: ``"2, first, last"``."""

    return ", ".join(str(token) for token in frames) or "no frame"


#: The folder a run keeps its finished artifact in: ``<run>/final/<file>``.
RUN_FINAL_FOLDER = "final"


def _file_name(path: Path) -> str:
    """A claimed file as the operator knows it.

    A run's own artifact is named by its RUN FOLDER: every Calibration run
    saves ``<run>/final/calibration.json``, so the file's own stem is the
    same word on every row and tells none of them apart -- the folder is
    the name the run was given.  Any other file is named by its stem.
    """

    if path.parent.name == RUN_FINAL_FOLDER and path.parent.parent.name:
        return path.parent.parent.name
    return path.stem


def _named_with_parent(path: Path) -> str:
    """The name one folder further up: ``<day>/<run>`` for a run's artifact,
    ``<folder>/<stem>`` for any other file."""

    named = path.parent.parent if path.parent.name == RUN_FINAL_FOLDER else path.parent
    return f"{named.parent.name}/{_file_name(path)}" if named.parent.name else _file_name(path)


def claim_names(entries: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """What each frame-claims row is called, wherever a row is named.

    The run folder of a run's artifact, the stem of any other file (every
    calibration is a .json, so the extension tells no row from another), or
    "row N" for a row with no file yet.  Two different files that would
    read alike -- runs of one name on two days -- are told apart by the
    folder above them.  The editor's rows, its summary and tooltips, the
    draft issues and the claims projection all name rows through here, so a
    row reads one way on screen and in every sentence about it.
    """

    paths = [Path(str(entry["path"])) if entry["path"] else None for entry in entries]
    short = [_file_name(path) if path is not None else "" for path in paths]
    names = [
        _named_with_parent(path)
        if path is not None and len({other for other, name in zip(paths, short)
                                     if other is not None and name == short[index]}) > 1
        else short[index]
        for index, path in enumerate(paths)
    ]
    return tuple(name or f"row {index + 1}" for index, name in enumerate(names))


def check_claims(
    frames_by_entry: Sequence[Sequence[object]],
    names: Sequence[str],
) -> tuple[tuple[FrameToken, ...], ...]:
    """What can be refused without knowing the frame count.

    Every entry claims at least one frame, and every middle number, "first",
    "last" and "rest" belongs to at most one entry.  "first" against "last"
    collides only in a one-frame cycle: that is :func:`frame_problems`'.
    """

    if len(frames_by_entry) != len(names):
        raise ValueError("every entry needs a name")
    claims = tuple(canonical_frames(frames) for frames in frames_by_entry)
    for name, frames in zip(names, claims):
        if not frames:
            raise ValueError(f"{name} reads no frame: choose its frames or remove it")
    owners: dict[FrameToken, int] = {}
    for entry, frames in enumerate(claims):
        for token in frames:
            if token in owners:
                what = f"frame {token}" if isinstance(token, int) else repr(token)
                raise ValueError(
                    f"{what} is claimed by both {names[owners[token]]} and {names[entry]}")
            owners[token] = entry
    return claims


@dataclass(frozen=True)
class FrameReach:
    """How one frame count reads an assignment, problems included."""

    count: int
    #: Per frame: every ``(entry, how)`` reaching it, ``how`` being the middle
    #: number, ``"first"``, ``"last"`` or ``"rest"``.
    readers: tuple[tuple[tuple[int, str], ...], ...]
    #: ``(entry, number)`` for every claimed middle number the cycle does not
    #: have between its ends.
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
            if token == REST:
                continue
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
    """One entry reaching a frame twice (first and last in a one-frame cycle) is one reader."""

    seen: dict[int, str] = {}
    for entry, how in found:
        seen.setdefault(entry, how)
    return tuple(seen.items())


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
        f"{names[entry]} names middle frame {number}, but the source carries "
        f"{count} frame{'s' if count != 1 else ''} ({span}); a numbered middle "
        "frame never stands for the first or the last"
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
    "FIRST",
    "FRAME_NAMES",
    "FrameClaim",
    "FrameReach",
    "FrameToken",
    "LAST",
    "REST",
    "assign_frames",
    "canonical_frames",
    "check_claims",
    "claim_names",
    "describe_frames",
    "frame_identity",
    "frame_problems",
    "frame_token",
    "reach_frames",
    "resolve_frame",
]
