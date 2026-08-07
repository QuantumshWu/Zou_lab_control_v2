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
