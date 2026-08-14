from __future__ import annotations

import pytest

from zlc_plot import DEFAULTS
from zlc_plot.layout import PANEL_SIZE_NAMES, resolve_surface


@pytest.mark.parametrize("preset", PANEL_SIZE_NAMES)
@pytest.mark.parametrize("dpr", (1.0, 1.25, 2.0))
def test_fixed_surface_presets_scale_only_physical_pixels(preset: str, dpr: float) -> None:
    base = resolve_surface(
        preset,
        "curve",
        device_pixel_ratio=1.0,
        layout=DEFAULTS.layout,
        style=DEFAULTS.style,
    )
    selected = resolve_surface(
        preset,
        "curve",
        device_pixel_ratio=dpr,
        layout=DEFAULTS.layout,
        style=DEFAULTS.style,
    )
    assert selected.logical_size == base.logical_size
    assert selected.raster_size == tuple(
        max(1, int(value * dpr + 0.5)) for value in base.logical_size
    )
    assert selected.axes == base.axes
    assert selected.device_pixel_ratio == dpr
    assert selected.dpi == pytest.approx(base.dpi * dpr)


def test_rolling_surface_requires_explicit_distribution_policy() -> None:
    with pytest.raises(TypeError):
        resolve_surface(
            "2x2",
            "rolling",
            layout=DEFAULTS.layout,
            style=DEFAULTS.style,
        )
    with pytest.raises(ValueError):
        resolve_surface(
            "2x2",
            "curve",
            rolling_side_distribution=True,
            layout=DEFAULTS.layout,
            style=DEFAULTS.style,
        )


def test_a_pulse_gets_the_smallest_preset_that_draws_it_legibly() -> None:
    """The size of a pulse plot is decided by its content, not by hand.

    pulse_row_min_px and pulse_period_min_px are the floors that make a channel
    row and a period readable.  Both lived here with nothing using them -- the
    rule that consumed them had been left behind in the migration, so every
    pulse was drawn at whatever size the caller happened to pick and two
    pulses could not be compared.
    """

    from zlc_plot.layout import DEFAULT_LAYOUT, PANEL_SIZE_NAMES, recommended_pulse_preset

    def _area(name: str) -> int:
        preset = DEFAULT_LAYOUT.preset(name)
        return preset.rows * preset.columns

    small = recommended_pulse_preset(4, 6)
    busy = recommended_pulse_preset(22, 6)
    busier = recommended_pulse_preset(22, 20)

    assert small in PANEL_SIZE_NAMES
    # More channels needs more height; more periods needs more width.
    assert _area(busy) > _area(small)
    assert _area(busier) >= _area(busy)

    # Nothing fits an extreme pulse, and the answer is the largest preset
    # rather than an illegible one.
    largest = max(PANEL_SIZE_NAMES, key=_area)
    assert recommended_pulse_preset(400, 400) == largest

    # The floors are the reason, so the chosen preset really does clear them.
    from zlc_plot.config import DEFAULTS
    from zlc_plot.kinds import PlotKind
    from zlc_plot.layout import resolve_surface

    plan = resolve_surface(busy, PlotKind.PULSE_TIMELINE, layout=DEFAULT_LAYOUT, style=DEFAULTS.style)
    box = next(item for item in plan.axes if item.role == "main").box
    width, height = plan.logical_size
    assert (box.bottom - box.top) * height >= 22 * DEFAULT_LAYOUT.pulse_row_min_px
    assert (box.right - box.left) * width >= 6 * DEFAULT_LAYOUT.pulse_period_min_px


def test_the_base_figure_is_the_instrument_it_is_modelled_on() -> None:
    """The margins are not free parameters; they are quoted, and they add up.

    Confocal-GUIv2 live_plot/plot_strategy.py declares the figure these panels
    descend from:

        self.fixed_data_px = (480, 360)
        # canvas area (700, 500)
        # left 220, 140
        self.margins_px    = (110, 110, 100, 40)   # (L, R, B, T)

    Those numbers are self-proving -- 110+480+110 = 700 and 100+360+40 = 500 --
    which is what makes a typo in them detectable at all.  This tree had
    carried (110, 96, 80, 70): 686x510, reproducing neither the reference nor
    anything else, with three of four values altered.  It survived every
    review because a margin looks like a taste, and nobody re-derived the
    canvas it is supposed to produce.  So the derivation is the test.
    """

    from zlc_plot.layout import DEFAULT_LAYOUT

    margins = DEFAULT_LAYOUT.panel_margins
    unit = DEFAULT_LAYOUT.panel_unit
    preset = DEFAULT_LAYOUT.preset("2x2")

    data_width = preset.columns * unit.width
    data_height = preset.rows * unit.height
    assert (data_width, data_height) == (480, 360), "the reference data box"

    # Horizontal: the reference's, exactly -- and symmetric, which is what
    # made the stray 96 detectable.  Its own comment says "left 220".
    assert margins.left == margins.right == 110
    assert margins.left + data_width + margins.right == 700, "the reference canvas width"

    # Vertical: deliberately NOT the reference's 100/40.  These panels carry a
    # title the reference had none of and tile into a grid, so the room is
    # spent differently.  Pinned anyway, because the point of this test is
    # that every one of these four numbers is a decision somebody can name.
    assert (margins.bottom, margins.top) == (80, 70)


def test_the_side_panel_splits_are_the_reference_ratios() -> None:
    """Checked in the same sweep that found the margins wrong; these are right.

    live_plot/plot_strategy.py:
        Live1DDis  layout_split([0.825, 0.15], [0.025])
        Live2DDis  layout_split([0.75, 0.1, 0.1], [0.025, 0.025])
    """

    from zlc_plot.layout import DEFAULT_LAYOUT

    rolling = DEFAULT_LAYOUT.rolling_split
    image = DEFAULT_LAYOUT.image_split
    assert (rolling.history, rolling.gap, rolling.distribution) == (0.825, 0.025, 0.15)
    assert (image.image, image.distribution, image.colorbar) == (0.75, 0.10, 0.10)
    assert (image.image_distribution_gap, image.distribution_colorbar_gap) == (0.025, 0.025)


def test_an_aspect_locked_image_keeps_its_strips_beside_it() -> None:
    """A wide preset must not open a hole between the image and its rails.

    Every width in the image split used to be a fraction of the REGION, but
    an aspect-locked image's width is set by the region's HEIGHT: the two
    coincide only where the preset is square, so on 2x4 the image drew at the
    left, the strips sat at the far right, and the surplus became a gap.
    """

    def boxes(preset: str, aspect: float | None) -> dict[str, tuple[float, float]]:
        plan = resolve_surface(
            preset,
            "image",
            image_aspect=aspect,
            layout=DEFAULTS.layout,
            style=DEFAULTS.style,
        )
        width = plan.raster_size[0]
        return {
            axes.role: (axes.box.left * width, axes.box.right * width)
            for axes in plan.axes
        }

    square = boxes("2x2", 1.0)
    wide = boxes("2x4", 1.0)
    # The same square image, and its strips in the same place beside it.
    for role, edges in square.items():
        assert wide[role] == pytest.approx(edges), role
    # The surplus stays where the image anchor leaves it: on the right.
    assert wide["colorbar"][1] < boxes("2x4", None)["colorbar"][1]

    # An image nothing locks still fills the region it is given.
    free = boxes("2x4", None)
    assert free["image"][1] > square["image"][1]
    gap = free["distribution"][0] - free["image"][1]
    assert gap == pytest.approx(
        (free["image"][1] - free["image"][0])
        * DEFAULTS.layout.image_split.image_distribution_gap
        / DEFAULTS.layout.image_split.image,
        rel=1e-9,
    )


def test_the_image_aspect_belongs_to_image_surfaces_only() -> None:
    with pytest.raises(ValueError, match="image_aspect"):
        resolve_surface(
            "2x2", "curve", image_aspect=1.0, layout=DEFAULTS.layout, style=DEFAULTS.style
        )
