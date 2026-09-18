"""The stroke kernels answer one picture however their lanes are cut.

``raster_polylines`` and ``raster_error_bars`` split a lane into column
bands so a lane with few peers still fills the pool.  A band may not change
a pixel: every band replays every primitive in painter order over its own
columns, so the blend sequence any one pixel sees is the one the serial
kernel produced.  The reference here IS that serial kernel -- the
one-lane, one-band algorithm kept in plain Python as the specification --
and the compiled kernels must reproduce it bit for bit at every band
count, including counts that do not divide the lane evenly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from zlc_plot import _raster_kernels as kernels

if not kernels.HAVE_NUMBA:  # pragma: no cover - the kernels are the subject
    pytest.skip("compiled stroke kernels are absent", allow_module_level=True)


def _clamped_line_integral(value):
    if value <= 0.0:
        return 0.0
    if value <= 1.0:
        return 0.5 * value * value
    return value - 0.5


def _slanted_cover(depth, grade):
    if grade <= 1.0e-9:
        return min(1.0, max(0.0, depth))
    half = 0.5 * grade
    return (_clamped_line_integral(depth + half) - _clamped_line_integral(depth - half)) / grade


def _disc_cover(px, py, x, y, radius):
    return min(1.0, max(0.0, radius + 0.5 - math.hypot(px - x, py - y)))


def _segment_cover(px, py, x0, y0, x1, y1, nxt, radius, cap_start, cap_end):
    """One piece's contribution: its band (intersection of the two slanted
    edges), a projecting cap at a polyline end, and the round join's disc
    for the wedge beyond this piece and before the next."""

    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    if length2 <= 0.0:
        return 0.0
    along = ((px - x0) * dx + (py - y0) * dy) / length2
    length = math.sqrt(length2)
    beyond = 0.0
    if along < 0.0:
        if not cap_start:
            return 0.0
        beyond = -along * length
    elif along > 1.0:
        if not cap_end:
            if nxt is not None:
                nx, ny = nxt[0] - x1, nxt[1] - y1
                next2 = nx * nx + ny * ny
                if next2 > 0.0 and ((px - x1) * nx + (py - y1) * ny) / next2 >= 0.0:
                    return 0.0
            return _disc_cover(px, py, x1, y1, radius)
        beyond = (along - 1.0) * length
    if beyond > radius + 0.5:
        return 0.0
    if abs(dy) > abs(dx):
        grade = abs(dx / dy)
        centre = x0 + (py - y0) * dx / dy
        half = radius * math.sqrt(1.0 + grade * grade)
        cover = _slanted_cover(px - (centre - half) + 0.5, grade) + _slanted_cover((centre + half) - px + 0.5, grade) - 1.0
    else:
        grade = abs(dy / dx)
        centre = y0 + (px - x0) * dy / dx
        half = radius * math.sqrt(1.0 + grade * grade)
        cover = _slanted_cover(py - (centre - half) + 0.5, grade) + _slanted_cover((centre + half) - py + 0.5, grade) - 1.0
    if cover <= 0.0:
        return 0.0
    if beyond > 0.0:
        cover *= min(1.0, radius + 0.5 - beyond)
    return cover


def _reference_polylines(vertices, offsets, colours, widths, clips, out):
    """The serial stroke: every pixel of the clip against every segment of
    the line, the most any segment gives it, blended once per line."""

    height, width = out.shape[:2]
    for line in range(offsets.size - 1):
        start, stop = int(offsets[line]), int(offsets[line + 1])
        if stop - start < 2:
            continue
        clip_left = max(0, int(clips[line, 0]))
        clip_top = max(0, int(clips[line, 1]))
        clip_right = min(width, int(clips[line, 2]))
        clip_bottom = min(height, int(clips[line, 3]))
        if clip_right <= clip_left or clip_bottom <= clip_top:
            continue
        radius = max(0.5, float(widths[line]) * 0.5)
        finite_segments = [
            (float(vertices[p, 0]), float(vertices[p, 1]), float(vertices[p + 1, 0]), float(vertices[p + 1, 1]))
            for p in range(start, stop - 1)
            if np.all(np.isfinite(vertices[p : p + 2]))
        ]
        # Agg snaps a rectilinear path of fewer than 1024 vertices to pixel centres.
        snap = stop - start < 1024 and all(x0 == x1 or y0 == y1 for x0, y0, x1, y1 in finite_segments)
        placed = vertices[start:stop].astype(float)
        if snap:
            offset = 0.5 if int(math.floor(float(widths[line]) + 0.5)) % 2 == 1 else 0.0
            placed = np.where(np.isfinite(placed), np.floor(placed + 0.5) + offset, placed)
        segments = []
        for point in range(start, stop - 1):
            x0, y0 = float(placed[point - start, 0]), float(placed[point - start, 1])
            x1, y1 = float(placed[point - start + 1, 0]), float(placed[point - start + 1, 1])
            if not all(map(np.isfinite, (x0, y0, x1, y1))):
                continue
            cap_start = point == start or not np.all(np.isfinite(vertices[point - 1]))
            cap_end = point + 2 >= stop or not np.all(np.isfinite(vertices[point + 2]))
            segments.append((x0, y0, x1, y1, bool(cap_start), bool(cap_end)))
        alpha_code = float(colours[line, 3]) / 255.0
        for column in range(clip_left, clip_right):
            px = column + 0.5
            for row in range(clip_top, clip_bottom):
                py = row + 0.5
                amount = 0.0
                for index, (x0, y0, x1, y1, cap_start, cap_end) in enumerate(segments):
                    nxt = None
                    if not cap_end and index + 1 < len(segments):
                        nxt = segments[index + 1][2:4]
                    amount += _segment_cover(px, py, x0, y0, x1, y1, nxt, radius, cap_start, cap_end)
                amount = min(1.0, amount)
                if amount > 1.0e-6:
                    _blend(out, row, column, colours[line], alpha_code * amount)


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


def test_polylines_match_the_serial_reference_at_every_band_count() -> None:
    rng = np.random.default_rng(3)
    vertices, offsets, colours, widths, clips, lanes = _polyline_scene(rng)
    canvas = _canvas(rng)
    expected = canvas.copy()
    _reference_polylines(vertices, offsets, colours, widths, clips, expected)
    assert (expected != canvas).any(), "the scene must paint something"
    for bands in _BAND_COUNTS:
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
    expected = canvas.copy()
    _reference_polylines(vertices, offsets, colours, widths, clips, expected)
    for bands in _BAND_COUNTS:
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
        np.testing.assert_array_equal(out, expected, err_msg=f"bands={bands}")


def test_stroke_bands_share_the_pool_without_a_kernel_thread_query() -> None:
    """A lone lane gets the pool's threads as bands, capped; a full pool of lanes gets one."""

    from numba import get_num_threads

    threads = int(get_num_threads())
    assert kernels.stroke_bands(1) == max(1, min(kernels._STROKE_BAND_LIMIT, threads))
    assert kernels.stroke_bands(threads) == 1
    assert kernels.stroke_bands(threads + 5) == 1
    assert kernels.stroke_bands(0) == 1
