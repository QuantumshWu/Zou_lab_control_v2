"""A site is a place that stands out in single shots OR in the average."""

from __future__ import annotations

import numpy as np

from zlc_atom.nodes.calibration.calibration import box_fits, detect_sites


def _lattice(
    *,
    dim_amplitude: float,
    bright_amplitude: float = 1100.0,
    loading: float = 0.5,
    shots: int = 120,
    seed: int = 5,
) -> tuple[np.ndarray, list[tuple[float, float]], set[int]]:
    """A triangular lattice like the bench's: rows 5.4px apart, columns 12.5.

    Three of its traps are made dim.  Every trap, dim or not, is loaded the
    same fraction of the shots, so the only thing that separates them is how
    bright one loaded shot IS.
    """

    rng = np.random.default_rng(seed)
    height, width = 76, 72
    centres: list[tuple[float, float]] = []
    for row in range(10):
        y = 10.0 + 5.4 * row
        offset = 0.0 if row % 2 == 0 else 6.2
        for column in range(3 if row % 2 == 0 else 4):
            centres.append((14.0 + 12.5 * column + offset, y))
    dim = {len(centres) - 1, len(centres) - 2, len(centres) - 5}
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    stack = rng.normal(120.0, 7.0, size=(shots, height, width))
    for index, (x, y) in enumerate(centres):
        spot = np.exp(-(((grid_x - x) ** 2 + (grid_y - y) ** 2) / 2.0))
        amplitude = dim_amplitude if index in dim else bright_amplitude
        stack[rng.random(shots) < loading] += amplitude * spot
    return stack, centres, dim


def _score(stack: np.ndarray, centres: list[tuple[float, float]]) -> tuple[int, int, int]:
    found = detect_sites(stack, spot_sigma=1.2)
    got = np.asarray(found.centers_xy)
    placed = np.asarray(centres)
    missed = sum(
        1
        for x, y in centres
        if float(np.min(np.hypot(got[:, 0] - x, got[:, 1] - y))) > 1.5
    )
    spurious = sum(
        1
        for x, y in got
        if float(np.min(np.hypot(placed[:, 0] - x, placed[:, 1] - y))) > 1.5
    )
    return found.n_sites, missed, spurious


def test_a_trap_too_dim_for_one_shot_is_still_found_in_the_average() -> None:
    """The bench case: a spot plainly visible in the report, with no circle.

    At this brightness a loaded shot does not reach the per-shot cut, so the
    trap collects no sightings however often it loads and counting sightings
    can never admit it.  Averaged over the run two of the three stand at six
    and seven sigma of their own background and are admitted; the third
    reaches 2.3 and is refused, which is the honest answer -- nothing in the
    run says it is there.

    The number that matters is the last one: no place without a trap is ever
    admitted, at any brightness.
    """

    stack, centres, dim = _lattice(dim_amplitude=160.0)
    found, missed, spurious = _score(stack, centres)
    assert spurious == 0
    assert missed == 1, "only the trap with no evidence at all is refused"
    assert found == len(centres) - 1


def test_neither_admission_invents_a_trap() -> None:
    """The other side of the same cut: no evidence is no site.

    Background alone must produce nothing -- both thresholds are set so that a
    whole image of it is not expected to yield even half a site -- and a
    lattice whose dim traps are a tenth of the brightness above still admits
    no place that has no trap in it, however many of those dim ones it loses.
    """

    import pytest

    rng = np.random.default_rng(17)
    background = rng.normal(120.0, 7.0, size=(120, 76, 72))
    with pytest.raises(ValueError, match="no detectable sites"):
        detect_sites(background, spot_sigma=1.2)

    stack, centres, _dim = _lattice(dim_amplitude=60.0)
    _found, _missed, spurious = _score(stack, centres)
    assert spurious == 0


def test_a_rarely_loaded_trap_is_still_found_by_its_sightings() -> None:
    """And the first admission still stands on its own.

    A trap loaded three percent of the time contributes almost nothing to the
    average -- this is exactly the case the sightings count was built for --
    while every loaded shot of it is unmistakable.
    """

    rng = np.random.default_rng(3)
    shots, height, width = 200, 74, 74
    centres = [(10.0 + 9.0 * column, 10.0 + 9.0 * row) for row in range(6) for column in range(6)]
    loading = np.linspace(0.70, 0.03, len(centres))
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    stack = rng.normal(100.0, 6.0, size=(shots, height, width))
    for index, (x, y) in enumerate(centres):
        spot = np.exp(-(((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * 1.2**2)))
        stack[rng.random(shots) < loading[index]] += 1200.0 * spot

    found, missed, spurious = _score(stack, centres)
    assert spurious == 0
    # The last of these loads three percent of the time: six sightings in two
    # hundred shots, against a per-frame false rate this run MEASURES rather
    # than assumes.  Thirty-five of thirty-six is where that lands.
    assert missed <= 1
    assert found >= len(centres) - 1

def test_a_place_that_cannot_be_measured_is_not_published() -> None:
    """The bench crash: a corner artefact, then no report at all.

    A bright thing against the border is a peak like any other, and sub-pixel
    refinement pulls its centre further out than the integer peak the margin
    checked -- measured here at (4.07, 1.92).  The readout then asked for a
    radius-6 box round it, could not have one, and the run died with
    "site center ... lies outside image" before writing a single picture.

    What cannot be measured is not a site, so the detector is told which box
    has to fit and drops the rest.  The four real traps are untouched.
    """

    rng = np.random.default_rng(4)
    height = width = 60
    shots = 80
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    stack = rng.normal(120.0, 7.0, size=(shots, height, width))
    traps = [(20.0, 20.0), (32.0, 20.0), (20.0, 32.0), (32.0, 32.0)]
    for x, y in traps:
        spot = np.exp(-(((grid_x - x) ** 2 + (grid_y - y) ** 2) / 2.0))
        stack[rng.random(shots) < 0.5] += 1100.0 * spot
    stack += 900.0 * np.exp(-(((grid_x - 4.0) ** 2 + (grid_y - 2.0) ** 2) / 2.0))

    unguarded = detect_sites(stack, spot_sigma=1.2)
    assert unguarded.n_sites == len(traps) + 1

    found = detect_sites(stack, spot_sigma=1.2, measurement_radius=6)
    assert found.n_sites == len(traps)
    assert all(
        box_fits((float(x), float(y)), 6, (height, width))
        for x, y in found.centers_xy
    )


def test_two_traps_are_two_hills_and_one_spot_is_one() -> None:
    """What makes two peaks two traps is a valley, not a distance.

    Every distance was wrong in one direction or the other: a lattice at five
    pixels merges under an exclusion chosen for a wide spot, and a single
    smeared spot is reported twice under one chosen for a narrow one.  The
    light itself answers it -- if the evidence between two peaks never falls
    to half the weaker of them they are one hill.

    Both arrays here have the same spot width and differ only in spacing.
    """

    def lattice(pitch: float) -> tuple[np.ndarray, list[tuple[float, float]]]:
        rng = np.random.default_rng(11)
        height = width = 70
        centres = [
            (12.0 + pitch * column, 12.0 + pitch * row)
            for row in range(5)
            for column in range(5)
        ]
        grid_y, grid_x = np.mgrid[0:height, 0:width]
        stack = rng.normal(120.0, 7.0, size=(150, height, width))
        for x, y in centres:
            spot = np.exp(-(((grid_x - x) ** 2 + (grid_y - y) ** 2) / 2.0))
            stack[rng.random(150) < 0.5] += 1100.0 * spot
        return stack, centres

    for pitch in (5.0, 11.0):
        stack, centres = lattice(pitch)
        found, missed, spurious = _score(stack, centres)
        assert (found, missed, spurious) == (len(centres), 0, 0), pitch


def test_a_coincidence_that_does_not_repeat_is_not_a_trap() -> None:
    """Pure background invents nothing, because nothing in it repeats.

    Every threshold before this one is a statement about a distribution, and a
    band-passed correlated picture keeps breaking those in the direction that
    invents traps -- measured, two "traps" on an empty picture.  A trap is in
    the first half of the run and in the second; a noise peak is not.
    """

    import pytest

    rng = np.random.default_rng(17)
    background = rng.normal(120.0, 7.0, size=(120, 76, 72))
    with pytest.raises(ValueError, match="no detectable sites"):
        detect_sites(background, spot_sigma=1.2)
