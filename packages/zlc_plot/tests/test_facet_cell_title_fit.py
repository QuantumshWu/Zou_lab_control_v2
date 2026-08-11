"""A facet cell title owns its cell's width plus the column gap, nothing more.

The surface plan states each cell's exclusive title room; the fitter shrinks
a wide title to that room and truncates at the readable floor.  Overlapping
neighbour titles read as one wrong label, which is why the rule is enforced
with measured text widths rather than hoped-for ones.
"""

from __future__ import annotations

import pytest

from zlc_plot import DEFAULTS
from zlc_plot.layout import (
    FacetTopology,
    _text_width_pt,
    fitted_facet_cell_title,
    resolve_surface,
)


def _facet_plan(cells: int, preset: str | None = None):
    return resolve_surface(
        preset,
        "facet_grid",
        FacetTopology(cells),
        layout=DEFAULTS.layout,
        style=DEFAULTS.style,
    )


def test_the_plan_states_each_cells_exclusive_title_room() -> None:
    plan = _facet_plan(4)
    typography = plan.facet_typography
    assert typography is not None
    assert typography.cell_title_max_width_pt > 0
    assert 0 < typography.cell_title_min_pt <= typography.cell_title_pt
    # More columns squeeze each cell, so the title room shrinks with them.
    wide = _facet_plan(2).facet_typography
    narrow = _facet_plan(9).facet_typography
    assert wide.cell_title_max_width_pt > narrow.cell_title_max_width_pt


def test_a_short_title_keeps_the_planned_size() -> None:
    typography = _facet_plan(4).facet_typography
    text, size = fitted_facet_cell_title("x=1", typography, DEFAULTS.style.fonts)
    assert text == "x=1"
    assert size == typography.cell_title_pt


def test_a_wide_title_shrinks_until_it_measures_inside_its_room() -> None:
    # Normal tier: the planned size sits above the floor, so there is a
    # shrink band before truncation.  Grow a label one character past its
    # room so it overflows by less than the whole band.
    typography = _facet_plan(4).facet_typography
    assert typography.cell_title_pt > typography.cell_title_min_pt
    fonts = DEFAULTS.style.fonts
    label = "da_bias_x="
    while (
        _text_width_pt(label, fonts.sans_serif, typography.cell_title_pt)
        <= typography.cell_title_max_width_pt
    ):
        label += "6"
    text, size = fitted_facet_cell_title(label, typography, fonts)
    assert text == label, "shrinking keeps the whole label"
    assert size < typography.cell_title_pt
    assert size >= typography.cell_title_min_pt
    measured = _text_width_pt(text, fonts.sans_serif, size)
    assert measured <= typography.cell_title_max_width_pt * 1.001


@pytest.mark.parametrize("label", ["site_row=3, site_column=12, da_bias_x=-256, da_bias_y=128 (calibrated against the reference run)"])
def test_an_absurd_title_is_truncated_at_the_readable_floor(label) -> None:
    typography = _facet_plan(9).facet_typography
    text, size = fitted_facet_cell_title(label, typography, DEFAULTS.style.fonts)
    assert size == typography.cell_title_min_pt, "the floor holds"
    assert text.endswith("\N{HORIZONTAL ELLIPSIS}")
    assert len(text) < len(label)
    measured = _text_width_pt(text, DEFAULTS.style.fonts.sans_serif, size)
    assert measured <= typography.cell_title_max_width_pt * 1.001
