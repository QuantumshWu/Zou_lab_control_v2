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

An entry claiming no frame is PARKED: it stays in the assignment, reads
nothing and is checked against nothing (:func:`parked_entries`) -- a file
the operator keeps at hand is not a file they must use.  Only an assignment
in which every entry is parked is refused.

A cycle that gains or loses a frame keeps meaning what was authored: the
ends follow the cycle, a middle number keeps its position, and a middle
number the shorter cycle no longer has between its ends is ABSENT -- it
never takes over the last frame, reads nothing in this cycle and is noted
(:func:`absent_notes`), not refused.  A frame is read by exactly one entry.
Only "rest" ranks below anything else: every other collision -- two entries
reaching one frame ("first" and "last" in a one-frame cycle), a frame nobody
reads -- is refused with the entries named, never settled silently.  One
entry reaching a frame two ways is one reader.  Nothing here ever edits what
was authored: an absent number reads its frame again once a frame count that
has it returns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

FIRST = "first"
LAST = "last"
REST = "rest"
FRAME_NAMES = (FIRST, LAST, REST)
#: The name of a cycle's frame axis -- the READOUT_EVENT Point axis the
#: camera and the Calibration task publish, and Occupancy keeps -- and so
#: what a Derive pick or statistic writes to name it, before data as after.
FRAME_AXIS_NAME = "frame"

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
    """The name with the nearest folder above it that adds a word:
    ``<day>/<run>`` for a run's artifact, ``<folder>/<stem>`` for any other
    file.  A folder named like the name itself (``calibration/calibration.json``)
    or the run's ``final`` folder tells nothing apart and is passed over."""

    base = _file_name(path)
    start = path.parent.parent.parent if path.parent.name == RUN_FINAL_FOLDER else path.parent
    for folder in (start, *start.parents):
        if folder.name and folder.name not in (base, RUN_FINAL_FOLDER):
            return f"{folder.name}/{base}"
    return base


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


def parked_entries(frames_by_entry: Sequence[Sequence[object]]) -> tuple[int, ...]:
    """The entries claiming no frame: kept where they were written, reading nothing."""

    return tuple(entry for entry, frames in enumerate(frames_by_entry) if not frames)


def claim_problems(
    frames_by_entry: Sequence[Sequence[object]],
    names: Sequence[str],
    *,
    what: str = "entry",
) -> tuple[str, ...]:
    """What can be refused without knowing the frame count, every reason at once.

    Every middle number, "first", "last" and "rest" belongs to at most one
    entry, and at least one entry reads a frame.  A parked entry claims
    nothing, so it neither collides nor hides a collision of two others.
    "first" against "last" collides only in a one-frame cycle: that is
    :func:`frame_problems`'.
    """

    if len(frames_by_entry) != len(names):
        raise ValueError("every entry needs a name")
    claims = tuple(canonical_frames(frames) for frames in frames_by_entry)
    owners: dict[FrameToken, list[int]] = {}
    for entry, frames in enumerate(claims):
        for token in frames:
            owners.setdefault(token, []).append(entry)
    problems = [
        f"{f'frame {token}' if isinstance(token, int) else repr(token)} is claimed by "
        + (f"both {names[entries[0]]} and {names[entries[1]]}" if len(entries) == 2 else
           ", ".join(names[entry] for entry in entries[:-1]) + f" and {names[entries[-1]]}")
        for token, entries in owners.items() if len(entries) > 1
    ]
    if claims and len(parked_entries(claims)) == len(claims):
        problems.append(
            f"no {what} reads a frame: give one the frames it reads, or let one read the rest")
    return tuple(problems)


def check_claims(
    frames_by_entry: Sequence[Sequence[object]],
    names: Sequence[str],
    *,
    what: str = "entry",
) -> tuple[tuple[FrameToken, ...], ...]:
    """The claims in their canonical form, or every :func:`claim_problems` reason."""

    problems = claim_problems(frames_by_entry, names, what=what)
    if problems:
        raise ValueError("; ".join(problems))
    return tuple(canonical_frames(frames) for frames in frames_by_entry)


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
    """Why ``count`` frames cannot be read this way, in operator words.

    A frame two entries reach, and a frame nobody reads.  An absent middle
    number is not one: it is :func:`absent_notes`'.
    """

    reach = reach_frames(frames_by_entry, count)
    problems = [
        f"frame {frame} is read by both "
        + " and ".join(f"{names[entry]} ({how})" for entry, how in readers)
        for frame, readers in enumerate(reach.readers) if len(readers) > 1
    ]
    unread = [frame for frame, readers in enumerate(reach.readers) if not readers]
    if unread:
        frames = ", ".join(str(frame) for frame in unread)
        problems.append(
            f"frame{'s' if len(unread) > 1 else ''} {frames} "
            f"{'have' if len(unread) > 1 else 'has'} no {what}: give "
            f"{'them' if len(unread) > 1 else 'it'} to one, or let one read the rest"
        )
    return tuple(problems)


def absent_notes(
    frames_by_entry: Sequence[Sequence[FrameToken]],
    count: int,
    names: Sequence[str],
) -> tuple[str, ...]:
    """Each claimed middle number a ``count``-frame cycle does not have between
    its ends: kept as written and reading nothing now, so it is said, not refused."""

    span = "only frame 0" if count == 1 else f"frames 0-{count - 1}"
    return tuple(
        f"{names[entry]}'s middle frame {number} is not in this source ({span}); "
        "it is kept for a cycle that has it"
        for entry, number in reach_frames(frames_by_entry, count).absent
    )


def assign_frames(
    frames_by_entry: Sequence[Sequence[FrameToken]],
    count: int,
    names: Sequence[str],
    *,
    what: str = "entry",
) -> tuple[int, ...]:
    """The one entry each of ``count`` frames is read by, or every reason why not.

    An absent middle number does not stop it: that entry reads the frames it
    still reaches, and its absent number reads nothing.
    """

    problems = frame_problems(frames_by_entry, count, names, what=what)
    if problems:
        raise ValueError("; ".join(problems))
    reach = reach_frames(frames_by_entry, count)
    return tuple(reach.entry_of(frame) for frame in range(count))


__all__ = [
    "FIRST",
    "FRAME_AXIS_NAME",
    "FRAME_NAMES",
    "FrameClaim",
    "FrameReach",
    "FrameToken",
    "LAST",
    "REST",
    "absent_notes",
    "assign_frames",
    "canonical_frames",
    "check_claims",
    "claim_problems",
    "claim_names",
    "describe_frames",
    "frame_identity",
    "frame_problems",
    "frame_token",
    "parked_entries",
    "reach_frames",
    "resolve_frame",
]
