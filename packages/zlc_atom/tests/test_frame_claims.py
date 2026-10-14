"""Frame claims: numbers from 0, last and rest, each frame read exactly once."""

from __future__ import annotations

import pytest

from zlc_atom.nodes import frame_claim_entries
from zlc_atom.nodes._framework.frames import (
    assign_frames,
    canonical_frames,
    check_claims,
    describe_frames,
    frame_identity,
    frame_problems,
    reach_frames,
)


def test_claimed_frames_have_one_spelling_and_one_order() -> None:
    assert canonical_frames(["rest", 2, "0", "LAST"]) == (0, 2, "last", "rest")
    assert describe_frames((0, 2, "last")) == "0, 2, last"
    assert describe_frames(()) == "no frame"
    assert tuple(frame_identity(index, 4) for index in range(4)) == (0, 1, 2, "last")
    assert frame_identity(0, 1) == 0
    for bad in ([True], [-1], ["first"], [1.5], "rest"):
        with pytest.raises(ValueError):
            canonical_frames(bad)
    with pytest.raises(ValueError, match="frame 1 is listed twice"):
        canonical_frames([1, "1"])


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
    assert reach.readers == (((0, "0"),),)
    assert reach.frames_of(1) == ()


def test_every_problem_is_named_and_none_is_settled_silently() -> None:
    assert frame_problems(((5,),), 3, ("late",)) == (
        "late names frame 5, but the source carries 3 frames (0-2); "
        "numbered middle frames cannot stand for the last frame",
        "frames 0, 1, 2 have no entry: give them to one, or let one read the rest",
    )
    assert frame_problems(((0,),), 2, ("load",), what="calibration") == (
        "frame 1 has no calibration: give it to one, or let one read the rest",
    )
    with pytest.raises(ValueError, match="names frame 2"):
        assign_frames((("last",), (2,)), 3, ("A", "B"))
    # The same claims are fine again once the count that suits them returns.
    assert assign_frames((("last",), (2,), ("rest",)), 4, ("A", "B", "C")) == (2, 2, 1, 0)


def test_claims_are_checked_among_themselves_before_any_count() -> None:
    with pytest.raises(ValueError, match="'rest' is claimed by both A and B"):
        check_claims((("rest",), ("rest",)), ("A", "B"))
    with pytest.raises(ValueError, match="frame 0 is claimed by both A and B"):
        check_claims(((0,), (0, "last")), ("A", "B"))
    with pytest.raises(ValueError, match="B reads no frame"):
        check_claims((("rest",), ()), ("A", "B"))


def test_a_draft_value_is_rows_of_path_and_frames() -> None:
    assert frame_claim_entries([{"path": " a.json ", "frames": ["rest", 0]}]) == (
        {"path": "a.json", "frames": (0, "rest")},
    )
    for bad in ("a.json", {"path": "a", "frames": []}, [{"path": "a"}],
                [{"path": 1, "frames": []}], [{"path": "a", "frames": [], "x": 1}]):
        with pytest.raises(ValueError):
            frame_claim_entries(bad)
