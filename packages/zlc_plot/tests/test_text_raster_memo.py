"""The renderer's text memo is invisible: every string lands pixel-identical.

The memo replaces Agg's per-draw text rasterization with a blit of the
raster it produced the first time.  That is only admissible if no pixel can
tell -- unrotated and rotated strings, fractional anchors, translucent
colours, strings half off the canvas, and the paths the memo must leave
alone (mathtext, rotated text that is clipped or translucent) -- and if a
warm memo answers exactly what a cold one did.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox

from zlc_plot.rendering import _prepare_renderer

_WIDTH, _HEIGHT, _DPI = 320, 240, 200.0


def _renderer() -> RendererAgg:
    renderer = RendererAgg(_WIDTH, _HEIGHT, _DPI)
    renderer.clear()
    return renderer


def _gc(renderer: RendererAgg, colour, *, clip: Bbox | None = None):
    gc = renderer.new_gc()
    gc.set_foreground(colour)
    gc.set_alpha(colour[3])
    gc.set_antialiased(True)
    if clip is not None:
        gc.set_clip_rectangle(clip)
    return gc


_PROP = FontProperties(family=["DejaVu Sans"], size=9.0)
#: (text, x, y, angle, colour, memoized?)
_SCENE = (
    ("value (count)", 40.0, 200.0, 90.0, (0.1, 0.2, 0.3, 1.0), True),
    ("value (count)", 60.25, 30.75, 0.0, (0.1, 0.2, 0.3, 1.0), True),
    ("133.1", 100.6, 100.4, 0.0, (0.9, 0.1, 0.1, 0.5), True),
    ("(1209, 619)", 150.3, 120.9, 45.0, (0.0, 0.0, 0.0, 1.0), True),
    ("tilted", 200.0, 60.0, -30.0, (0.2, 0.7, 0.2, 1.0), True),
    ("tilted", 230.0, 90.0, -30.0, (0.2, 0.7, 0.2, 0.8), False),
    ("edge", 305.0, 12.0, 0.0, (0.0, 0.0, 0.0, 1.0), True),
    ("clipped edge", -12.0, 235.0, 90.0, (0.0, 0.0, 0.0, 1.0), True),
)


def _draw_scene(renderer: RendererAgg, shift: float = 0.0) -> np.ndarray:
    for text, x, y, angle, colour, _memoized in _SCENE:
        renderer.draw_text(
            _gc(renderer, colour), x + shift, y - shift, text, _PROP, angle
        )
    return np.asarray(renderer.buffer_rgba()).copy()


def test_memo_reproduces_a_pristine_renderer_cold_and_warm() -> None:
    pristine = _draw_scene(_renderer())

    fitted = _renderer()
    assert _prepare_renderer(fitted) is fitted
    cold = _draw_scene(fitted)
    np.testing.assert_array_equal(cold, pristine)

    memo = fitted._zlc_text_rasters
    fitted.clear()
    warm = _draw_scene(fitted)
    np.testing.assert_array_equal(warm, pristine)
    # Exactly the admissible strings entered, each once, and the second
    # pass added nothing: it was answered from the memo.
    expected_keys = {(t, a) for t, _x, _y, a, _c, memoized in _SCENE if memoized}
    assert {(key[0], key[2]) for key in memo} == expected_keys
    assert len(memo) == len(expected_keys)


def test_memo_moves_with_the_anchor_by_integer_pixels() -> None:
    fitted = _prepare_renderer(_renderer())
    pristine = _renderer()
    for shift in (0.0, 0.3, 0.5, 0.7, 13.0, 13.49, 13.51):
        fitted.clear()
        pristine.clear()
        np.testing.assert_array_equal(
            _draw_scene(fitted, shift), _draw_scene(pristine, shift), err_msg=f"shift {shift}"
        )


@pytest.mark.parametrize("angle", [0.0, 90.0])
def test_clipped_text_is_exact_and_rotated_clipped_text_is_not_memoized(
    angle: float,
) -> None:
    """A clip cutting through the string: unrotated replays it, rotated defers."""

    fitted = _prepare_renderer(_renderer())
    pristine = _renderer()
    clip = Bbox.from_extents(20.0, 20.0, 60.0, 150.0)
    for renderer in (fitted, pristine):
        for _pass in range(2):
            renderer.draw_text(
                _gc(renderer, (0.0, 0.0, 0.0, 1.0), clip=clip),
                30.0,
                140.0,
                "clipped through",
                _PROP,
                angle,
            )
    np.testing.assert_array_equal(
        np.asarray(fitted.buffer_rgba()), np.asarray(pristine.buffer_rgba())
    )
    assert bool(fitted._zlc_text_rasters) == (angle == 0.0)


@pytest.mark.parametrize("angle", [0.0, 90.0])
def test_math_text_takes_the_original_route(angle: float) -> None:
    fitted = _prepare_renderer(_renderer())
    pristine = _renderer()
    for renderer in (fitted, pristine):
        renderer.draw_text(
            _gc(renderer, (0.0, 0.0, 0.0, 1.0)),
            160.0,
            100.0,
            r"$\sigma_x = 1.5$",
            _PROP,
            angle,
            ismath=True,
        )
    np.testing.assert_array_equal(
        np.asarray(fitted.buffer_rgba()), np.asarray(pristine.buffer_rgba())
    )
    assert not fitted._zlc_text_rasters, "mathtext may not enter the memo"


def test_memo_is_bounded_and_forgets_rather_than_grows() -> None:
    from zlc_plot import rendering

    fitted = _prepare_renderer(_renderer())
    memo = fitted._zlc_text_rasters
    for index in range(rendering._TEXT_RASTER_MEMO_LIMIT + 5):
        fitted.draw_text(
            _gc(fitted, (0.0, 0.0, 0.0, 1.0)), 10.0, 100.0, f"{index}", _PROP, 0.0
        )
    assert len(memo) <= rendering._TEXT_RASTER_MEMO_LIMIT


def test_preparing_twice_installs_one_memo() -> None:
    fitted = _renderer()
    _prepare_renderer(fitted)
    memo = fitted._zlc_text_rasters
    draw = fitted.draw_text
    _prepare_renderer(fitted)
    assert fitted._zlc_text_rasters is memo
    assert fitted.draw_text is draw
