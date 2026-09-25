"""The stroke kernels answer one picture however their lanes are cut.

``raster_polylines`` and ``raster_error_bars`` split a lane into column
bands so a lane with few peers still fills the pool.  A band may not change
a pixel: every band replays every primitive in painter order over its own
columns, so the blend sequence any one pixel sees is the one the serial
kernel produced.  The polylines are held to the kernel's own one-lane,
one-band picture, bit for bit, at every band count -- including counts that
do not divide the lane evenly.  What that serial picture should BE is held
against Agg in test_stroke_kernel_vs_agg.py: a copy of the kernel written
here could only agree with it.  The error bars keep an independent
reference, an inclusion-exclusion of each bar's rectangles.
"""

from __future__ import annotations

import numpy as np

from zlc_plot import _raster_kernels as kernels


def _reference_error_bars(
    x, y_low, y_high, offsets, colours, widths, cap_widths, clips, out
):
    """Independent inclusion-exclusion of each bar's three rectangles."""

    height, width = out.shape[:2]
    for group in range(offsets.size - 1):
        clip_left = max(0, int(clips[group, 0]))
        clip_top = max(0, int(clips[group, 1]))
        clip_right = min(width, int(clips[group, 2]))
        clip_bottom = min(height, int(clips[group, 3]))
        if clip_right <= clip_left or clip_bottom <= clip_top:
            continue
        radius = max(0.5, float(widths[group]) * 0.5)
        cap_half = max(0.0, float(cap_widths[group]) * 0.5)
        alpha_code = float(colours[group, 3]) / 255.0
        for point in range(int(offsets[group]), int(offsets[group + 1])):
            px, low, high = float(x[point]), float(y_low[point]), float(y_high[point])
            if not all(map(np.isfinite, (px, low, high))):
                continue
            low, high = sorted((low, high))
            if high <= low:
                continue
            rectangles = [(px - radius, px + radius, low, high)]
            if cap_half > 0.0:
                rectangles += [(px - cap_half, px + cap_half, edge - radius, edge + radius) for edge in (low, high)]
            from itertools import combinations
            terms = []
            for size in range(1, len(rectangles) + 1):
                for subset in combinations(rectangles, size):
                    terms.append((1 if size % 2 else -1, (max(r[0] for r in subset),
                        min(r[1] for r in subset), max(r[2] for r in subset), min(r[3] for r in subset))))
            for column in range(max(clip_left, int(np.floor(min(r[0] for r in rectangles)))),
                                min(clip_right, int(np.ceil(max(r[1] for r in rectangles))))):
                for row in range(max(clip_top, int(np.floor(min(r[2] for r in rectangles)))),
                                 min(clip_bottom, int(np.ceil(max(r[3] for r in rectangles))))):
                    cover = sum(sign * max(0., min(column + 1., right) - max(column, left))
                                * max(0., min(row + 1., bottom) - max(row, top))
                                for sign, (left, right, top, bottom) in terms)
                    if cover > 0.0:
                        _blend(out, row, column, colours[group], alpha_code * min(1., cover))


def _blend(out, row, column, colour, alpha):
    inverse = 1.0 - alpha
    for channel in range(3):
        value = float(colour[channel]) * alpha + float(out[row, column, channel]) * inverse
        out[row, column, channel] = np.uint8(min(255.0, np.floor(value + 0.5)))
    out[row, column, 3] = np.uint8(255)


def _canvas(rng, height=48, width=96):
    return rng.integers(0, 256, (height, width, 4), dtype=np.uint8)


def _polyline_scene(rng):
    """Many translucent overlapping lines on one axes: one serial lane."""

    lines, points, width = 7, 9, 96
    vertices = np.empty((lines * points, 2))
    for line in range(lines):
        xs = np.linspace(3.0 + line, 90.0 - line, points) + rng.uniform(-0.3, 0.3, points)
        ys = 24.0 + 12.0 * np.sin(xs / 9.0 + line) + rng.uniform(-2.0, 2.0, points)
        vertices[line * points : (line + 1) * points, 0] = xs
        vertices[line * points : (line + 1) * points, 1] = ys
    vertices[2 * points + 4] = (np.nan, np.nan)  # a gap
    vertices[5 * points + 3, 0] = vertices[5 * points + 2, 0]  # a vertical step
    offsets = np.arange(0, (lines + 1) * points, points, dtype=np.int64)
    colours = rng.integers(0, 256, (lines, 4), dtype=np.uint8)
    colours[:, 3] = rng.integers(90, 256, lines)
    widths = rng.uniform(1.0, 6.0, lines)
    clips = np.broadcast_to(np.asarray((2, 3, width - 2, 45), dtype=np.int32), (lines, 4)).copy()
    lanes = np.asarray((0, lines), dtype=np.int64)
    return vertices, offsets, colours, widths, clips, lanes


def _error_bar_scene(rng):
    groups, points = 5, 30
    x = np.tile(np.linspace(4.0, 92.0, points), groups) + rng.uniform(-0.4, 0.4, groups * points)
    centre = 24.0 + rng.uniform(-8.0, 8.0, groups * points)
    spread = rng.uniform(0.2, 9.0, groups * points)
    y_low, y_high = centre - spread, centre + spread
    y_low[7], y_high[7] = y_high[7], y_low[7]  # reversed pair
    y_low[11] = np.nan  # dropped sample
    offsets = np.arange(0, (groups + 1) * points, points, dtype=np.int64)
    colours = rng.integers(0, 256, (groups, 4), dtype=np.uint8)
    colours[:, 3] = rng.integers(60, 256, groups)
    widths = rng.uniform(1.0, 5.0, groups)
    cap_widths = rng.uniform(0.0, 9.0, groups)
    cap_widths[1] = 0.0
    clips = np.broadcast_to(np.asarray((1, 2, 95, 46), dtype=np.int32), (groups, 4)).copy()
    lanes = np.asarray((0, groups), dtype=np.int64)
    return x, y_low, y_high, offsets, colours, widths, cap_widths, clips, lanes


_BAND_COUNTS = (1, 2, 3, 5, 8, 16)


def _polylines(canvas, vertices, offsets, colours, widths, clips, lanes, bands):
    out = canvas.copy()
    kernels.raster_polylines(
        kernels.readable(vertices),
        kernels.readable(offsets),
        kernels.readable(colours),
        kernels.readable(widths),
        kernels.readable(clips),
        kernels.readable(lanes),
        bands,
        out,
    )
    return out


def test_polylines_match_the_serial_kernel_at_every_band_count() -> None:
    rng = np.random.default_rng(3)
    vertices, offsets, colours, widths, clips, lanes = _polyline_scene(rng)
    canvas = _canvas(rng)
    scene = (vertices, offsets, colours, widths, clips)
    expected = _polylines(canvas, *scene, lanes, 1)
    assert (expected != canvas).any(), "the scene must paint something"
    for bands in _BAND_COUNTS[1:]:
        out = _polylines(canvas, *scene, lanes, bands)
        np.testing.assert_array_equal(out, expected, err_msg=f"bands={bands}")


def test_error_bars_match_the_serial_reference_at_every_band_count() -> None:
    rng = np.random.default_rng(5)
    x, y_low, y_high, offsets, colours, widths, cap_widths, clips, lanes = _error_bar_scene(rng)
    canvas = _canvas(rng)
    expected = canvas.copy()
    _reference_error_bars(x, y_low, y_high, offsets, colours, widths, cap_widths, clips, expected)
    assert (expected != canvas).any(), "the scene must paint something"
    for bands in _BAND_COUNTS:
        out = canvas.copy()
        kernels.raster_error_bars(
            kernels.readable(x),
            kernels.readable(y_low),
            kernels.readable(y_high),
            kernels.readable(offsets),
            kernels.readable(colours),
            kernels.readable(widths),
            kernels.readable(cap_widths),
            kernels.readable(clips),
            kernels.readable(lanes),
            bands,
            out,
        )
        np.testing.assert_array_equal(out, expected, err_msg=f"bands={bands}")


def test_disjoint_lanes_paint_their_own_boxes_only() -> None:
    """Two lanes with disjoint clips: each cell is the serial picture of its lines."""

    rng = np.random.default_rng(9)
    vertices, offsets, colours, widths, clips, _lanes = _polyline_scene(rng)
    # Lines 0-3 in the left half, 4-6 in the right half, as two lanes.
    clips[:4] = (2, 3, 46, 45)
    clips[4:] = (50, 3, 94, 45)
    vertices[: 4 * 9, 0] = np.clip(vertices[: 4 * 9, 0] * 0.5, 2.0, 46.0)
    vertices[4 * 9 :, 0] = np.clip(48.0 + vertices[4 * 9 :, 0] * 0.5, 50.0, 94.0)
    lanes = np.asarray((0, 4, 7), dtype=np.int64)
    canvas = _canvas(rng)
    scene = (vertices, offsets, colours, widths, clips)
    # The serial picture: every line in one lane, one band.
    expected = _polylines(canvas, *scene, np.asarray((0, 7), dtype=np.int64), 1)
    assert (expected != canvas).any(), "the scene must paint something"
    for bands in _BAND_COUNTS:
        out = _polylines(canvas, *scene, lanes, bands)
        np.testing.assert_array_equal(out, expected, err_msg=f"bands={bands}")
