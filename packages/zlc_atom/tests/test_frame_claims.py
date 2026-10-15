"""Frame claims: numbers from 0, first, last and rest, each frame read exactly once."""

from __future__ import annotations

import pytest

from zlc_atom.nodes import frame_claim_entries
from zlc_atom.nodes._framework.frames import (
    assign_frames,
    canonical_frames,
    check_claims,
    claim_problems,
    describe_frames,
    frame_identity,
    frame_problems,
    parked_entries,
    reach_frames,
)


def test_claimed_frames_have_one_spelling_and_one_order() -> None:
    # Frame 0 IS the first frame: one claim, one spelling.
    assert canonical_frames(["rest", 2, "0", "LAST"]) == (2, "first", "last", "rest")
    assert canonical_frames(["rest", "last", 3, " First "]) == (3, "first", "last", "rest")
    assert describe_frames((0, 2, "last")) == "0, 2, last"
    assert describe_frames((1, "first", "rest")) == "1, first, rest"
    assert describe_frames(()) == "no frame"
    assert tuple(frame_identity(index, 4) for index in range(4)) == (0, 1, 2, "last")
    assert frame_identity(0, 1) == 0
    for bad in ([True], [-1], ["second"], [1.5], "rest"):
        with pytest.raises(ValueError):
            canonical_frames(bad)
    with pytest.raises(ValueError, match="frame 1 is listed twice"):
        canonical_frames([1, "1"])
    with pytest.raises(ValueError, match="frame first is listed twice"):
        canonical_frames(["first", "FIRST"])


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_first_is_frame_zero_in_every_cycle(count: int) -> None:
    reach = reach_frames((("first",),), count)
    assert reach.readers[0] == ((0, "first"),)
    assert reach.frames_of(0) == (0,)
    if count > 1:
        assert assign_frames((("first",), ("rest",)), count, ("A", "B")) == (0, *(1,) * (count - 1))
        assert assign_frames((("first",), ("last",), ("rest",)), count, ("A", "B", "C"))[0] == 0


def test_first_and_frame_zero_are_one_frame() -> None:
    # Two entries: refused by name, saying how each reaches it.
    assert frame_problems((("first",), (0, "rest")), 3, ("A", "B")) == (
        "frame 0 is read by both A (first) and B (0)",
    )
    assert frame_problems(((0,), ("first", "rest")), 3, ("A", "B")) == (
        "frame 0 is read by both A (0) and B (first)",
    )
    # One entry naming it both ways is one reader.
    assert assign_frames(((0, "first"), ("rest",)), 3, ("A", "B")) == (0, 1, 1)
    assert reach_frames(((0, "first"), ("rest",)), 3).readers[0] == ((0, "0"),)


def test_first_and_last_meet_in_a_one_frame_cycle() -> None:
    assert assign_frames((("first", "last"),), 1, ("A",)) == (0,)
    assert reach_frames((("first", "last"),), 1).readers == (((0, "first"),),)
    assert frame_problems((("first",), ("last",)), 1, ("A", "B")) == (
        "frame 0 is read by both A (first) and B (last)",
    )
    # The same claims are fine once the cycle has two ends again.
    assert assign_frames((("first",), ("last",)), 2, ("A", "B")) == (0, 1)


@pytest.mark.parametrize("count, expected", [
    (1, (1,)),          # the only frame is the last: B's
    (2, (0, 1)),
    (3, (0, 2, 1)),     # rest takes what nobody named
    (5, (0, 2, 2, 2, 1)),
])
def test_last_and_rest_follow_the_count(count: int, expected: tuple[int, ...]) -> None:
    # A reads frame 0, B the last, C the rest.  At one frame A's 0 and B's
    # last reach the same frame, which is refused rather than settled.
    claims = ((0,), ("last",), ("rest",))
    if count == 1:
        assert frame_problems(claims, 1, ("A", "B", "C")) == (
            "frame 0 is read by both A (0) and B (last)",
        )
        assert assign_frames(((), ("last",), ("rest",)), 1, ("A", "B", "C")) == expected
        return
    assert assign_frames(claims, count, ("A", "B", "C")) == expected


def test_one_entry_may_reach_a_frame_twice() -> None:
    assert assign_frames(((0, "last"), ("rest",)), 1, ("A", "B")) == (0,)
    reach = reach_frames(((0, "last"), ("rest",)), 1)
    assert reach.readers == (((0, "0"),),), "how each reader wrote it"
    assert reach.frames_of(1) == ()


def test_every_problem_is_named_and_none_is_settled_silently() -> None:
    assert frame_problems(((5,),), 3, ("late",)) == (
        "late names middle frame 5, but the source carries 3 frames (0-2); "
        "a numbered middle frame never stands for the first or the last",
        "frames 0, 1, 2 have no entry: give them to one, or let one read the rest",
    )
    assert frame_problems(((0,),), 2, ("load",), what="calibration") == (
        "frame 1 has no calibration: give it to one, or let one read the rest",
    )
    # In three frames 2 is the LAST frame: a middle 2 is absent, never the last.
    with pytest.raises(ValueError, match="names middle frame 2"):
        assign_frames((("last",), (2,)), 3, ("A", "B"))
    # The same claims are fine again once the count that suits them returns.
    assert assign_frames((("last",), (2,), ("rest",)), 4, ("A", "B", "C")) == (2, 2, 1, 0)


def test_claims_are_checked_among_themselves_before_any_count() -> None:
    with pytest.raises(ValueError, match="'rest' is claimed by both A and B"):
        check_claims((("rest",), ("rest",)), ("A", "B"))
    with pytest.raises(ValueError, match="'first' is claimed by both A and B"):
        check_claims(((0,), (0, "last")), ("A", "B"))
    with pytest.raises(ValueError, match="'first' is claimed by both A and B"):
        check_claims((("first",), ("first", "rest")), ("A", "B"))
    # 0 IS First: written either way, it is one claim and one owner.
    with pytest.raises(ValueError, match="'first' is claimed by both A and B"):
        check_claims((("first",), (0, "last")), ("A", "B"))
    with pytest.raises(ValueError, match="'first' is claimed by both A and B"):
        check_claims(((0, "rest"), ("first",)), ("A", "B"))
    # One entry naming frame 0 both ways is one claim; First against Last
    # collides only in a one-frame cycle, so only a count refuses it.
    assert check_claims(((0, "first"), ("rest",)), ("A", "B")) == (("first",), ("rest",))
    assert check_claims((("first",), ("last",)), ("A", "B")) == (("first",), ("last",))
    # An entry claiming no frame is parked: kept, colliding with nothing and
    # hiding no collision of two others -- every collision is said at once.
    assert check_claims((("rest",), ()), ("A", "B")) == (("rest",), ())
    assert parked_entries((("rest",), (), ("first",), ())) == (1, 3)
    assert claim_problems(((), ("rest",), ("rest", 2), (0,), ("first", 2)), ("P", "A", "B", "C", "D")) == (
        "'rest' is claimed by both A and B",
        "frame 2 is claimed by both B and D",
        "'first' is claimed by both C and D",
    )
    assert claim_problems((("last",), ("last",), ("last",)), ("A", "B", "C")) == (
        "'last' is claimed by A, B and C",
    )
    # Only an assignment in which nothing reads is refused for parking.
    with pytest.raises(ValueError, match="no calibration reads a frame: give one the frames it reads, "
                                         "or let one read the rest"):
        check_claims(((), ()), ("A", "B"), what="calibration")
    assert claim_problems((), ()) == ()


def test_a_draft_value_is_rows_of_path_and_frames() -> None:
    assert frame_claim_entries([{"path": " a.json ", "frames": ["rest", 0]}]) == (
        {"path": "a.json", "frames": ("first", "rest")},
    )
    for bad in ("a.json", {"path": "a", "frames": []}, [{"path": "a"}],
                [{"path": 1, "frames": []}], [{"path": "a", "frames": [], "x": 1}]):
        with pytest.raises(ValueError):
            frame_claim_entries(bad)


def test_a_row_is_named_by_its_run_folder_and_told_apart_by_the_day() -> None:
    """Every Calibration run saves <run>/final/calibration.json, so a row is
    named by its run folder, not the file's stem every row would share; runs
    of one name on two days carry the day; any other file keeps its stem."""

    from zlc_atom.nodes import claim_names

    names = claim_names((
        {"path": "C:/ws/data/2026_10_13/load_cal/final/calibration.json", "frames": ("first",)},
        {"path": "C:/ws/data/2026_10_14/load_cal/final/calibration.json", "frames": (1,)},
        {"path": "C:/ws/data/2026_10_14/readout_cal/final/calibration.json", "frames": ("last",)},
        {"path": "C:/ws/calibrations/background_5ms.json", "frames": ("rest",)},
        {"path": "", "frames": (2,)},
    ))
    assert names == ("2026_10_13/load_cal", "2026_10_14/load_cal", "readout_cal", "background_5ms", "row 5")
    assert "calibration" not in names

    # Two runs that kept the task's own folder name, and files saved before
    # runs had a final folder: each is told apart by its day, never by a
    # folder that repeats the name or by the data root.
    assert claim_names((
        {"path": "C:/ws/data/2026_09_01/calibration/final/calibration.json", "frames": ("first",)},
        {"path": "C:/ws/data/2026_09_02/calibration/final/calibration.json", "frames": ("last",)},
        {"path": "C:/ws/data/2026_08_16/calibration/calibration.json", "frames": (1,)},
        {"path": "C:/ws/data/2026_08_08/calibration.json", "frames": ("rest",)},
    )) == ("2026_09_01/calibration", "2026_09_02/calibration", "2026_08_16/calibration", "2026_08_08/calibration")
