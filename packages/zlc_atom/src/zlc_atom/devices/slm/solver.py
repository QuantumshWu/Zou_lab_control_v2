"""Continuous SLM targets, phase retrieval, and their file formats.

Pillow, unicodedata and scipy are all reached from inside the functions
that use them.  A logic node's descriptor names one of the file readers
here as its artifact codec, so DISCOVERING the nodes -- which a task
console does before it shows anything -- imports this module; at module
scope those three cost it a font stack and a solver stack for a target
nobody has authored and a pattern nobody has solved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from zlc_durable import atomic_write_file, strict_json_loads, write_readable_json

from .device import canonical_phase

_TARGET_FORMAT = "zlc.slm.target"
_TARGET_KEYS = frozenset(
    {"format", "shape", "intensity", "objective_kind"}
)
_SCIENCE_CONTEXT_FORMAT = "zlc.slm.science-context"
SCIENCE_CONTEXT_ARTIFACT_CONTRACT = _SCIENCE_CONTEXT_FORMAT
_SCIENCE_CONTEXT_MEMBERS = frozenset(
    {"pattern_phase_delta", "target_intensity", "metadata"}
)
_SCIENCE_CONTEXT_KEYS = frozenset(
    {
        "format",
        "objective_kind",
        "pupil",
        "system_correction",
        "command_receipt",
        "pattern_metadata",
        "operator_metadata",
    }
)
_OBJECTIVE_KINDS = frozenset({"auto", "spots", "image"})
_SYSTEM_CORRECTION_KINDS = frozenset(
    {"pupil_phase_map", "target_response_map"}
)
_OPERATOR_MODES = frozenset(
    {
        "defocus", "astig_oblique", "astig_vertical", "coma_y", "coma_x",
        "trefoil_y", "trefoil_x", "spherical",
    }
)
_PHASE_CODE_COUNT = 1 << 16

def _pair(value: object, name: str) -> tuple[int, int]:
    try:
        pair = tuple(int(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a two-integer pair") from error
    if len(pair) != 2 or any(item <= 0 for item in pair):
        raise ValueError(f"{name} must contain two positive integers")
    return pair

def _float_pair(value: object, name: str) -> tuple[float, float]:
    try:
        pair = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a two-number pair") from error
    if len(pair) != 2 or any(not np.isfinite(item) or item <= 0.0 for item in pair):
        raise ValueError(f"{name} must contain two finite positive numbers")
    return pair

def _scalar(value: object, name: str, *, nonnegative: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (result < 0.0 if nonnegative else result <= 0.0):
        qualifier = "non-negative" if nonnegative else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result

def _readonly(values: object) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype="<f4")
    return np.frombuffer(contiguous.tobytes(), dtype="<f4").reshape(
        contiguous.shape
    )

def validate_target(values: object) -> np.ndarray:
    """Return the sole target representation: finite non-negative intensity."""

    try:
        target = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise TypeError("target intensity must be a numeric array") from error
    if target.ndim != 2:
        raise ValueError("target intensity must be two-dimensional")
    if min(target.shape) < 2:
        raise ValueError("target intensity dimensions must each be at least two")
    if not np.all(np.isfinite(target)):
        raise ValueError("target intensity must be finite")
    if np.any(target < 0.0):
        raise ValueError("target intensity must be non-negative")
    return _readonly(target)


def _objective_kind(value: object) -> str:
    if type(value) is not str or value not in _OBJECTIVE_KINDS:
        raise ValueError("objective_kind must be 'auto', 'spots', or 'image'")
    return value

def _grid_indices(
    shape_yx: object,
    grid_shape_yx: object,
    spacing_yx: object | None,
) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    shape = _pair(shape_yx, "shape_yx")
    grid = _pair(grid_shape_yx, "grid_shape_yx")
    spacing = (
        tuple(max(1, shape[index] // (grid[index] + 1)) for index in range(2))
        if spacing_yx is None
        else _pair(spacing_yx, "spacing_yx")
    )
    spans = tuple((grid[index] - 1) * spacing[index] for index in range(2))
    if any(span >= shape[index] for index, span in enumerate(spans)):
        raise ValueError("grid does not fit inside target shape")
    starts = tuple((shape[index] - spans[index]) // 2 for index in range(2))
    axes = tuple(
        starts[index] + spacing[index] * np.arange(grid[index], dtype=int)
        for index in range(2)
    )
    return shape, axes[0], axes[1]

_REARRANGEMENT_MATCH = None


def _match_rearrangement(indptr, columns, costs, occupied, row_order):
    """Exact rectangular primal-dual matching; compiled lazily in preparation."""
    def radix_bin(key, last):
        value = np.uint64(key) ^ np.uint64(last)
        if value == 0:
            return 0
        result = 1
        for shift in (32, 16, 8, 4, 2, 1):
            if value >> shift:
                value >>= shift
                result += shift
        return result

    rows, width = len(indptr) - 1, len(occupied)
    # Keep original source IDs and neighbor order; inactive edges never enter
    # initialization or an augmenting path, so filter them only once.
    pointers = np.empty_like(indptr)
    active_columns = np.empty_like(columns)
    active_costs = np.empty_like(costs)
    pointers[0] = 0
    used = 0
    for row in range(rows):
        for edge in range(indptr[row], indptr[row + 1]):
            col = columns[edge]
            if occupied[col]:
                active_columns[used] = col
                active_costs[used] = costs[edge]
                used += 1
        pointers[row + 1] = used
    indptr, columns, costs = pointers, active_columns, active_costs
    infinity = np.iinfo(np.int64).max
    assignment = np.full(rows, -1, np.int64)
    owner = np.full(width, -1, np.int64)
    u, v = np.zeros(rows, np.int64), np.zeros(width, np.int64)
    distance, parent = np.empty(width, np.int64), np.empty(width, np.int64)
    settled, visited = np.empty(width, np.bool_), np.empty(width, np.int64)
    # Sixty-five radix buckets, each with free/matched heads. Queue storage
    # depends on source count, never on coordinate magnitude or cost range.
    heads = np.empty(130, np.int64)
    bucket = np.empty(width, np.int64)
    previous, following = np.empty(width, np.int64), np.empty(width, np.int64)
    for row in row_order:
        minimum = infinity
        for edge in range(indptr[row], indptr[row + 1]):
            if costs[edge] < minimum:
                minimum = costs[edge]
        if minimum == infinity:
            return assignment, False
        u[row] = minimum
        for edge in range(indptr[row], indptr[row + 1]):
            col = columns[edge]
            if owner[col] < 0 and costs[edge] == minimum:
                assignment[row], owner[col] = col, row
                break
    for root in row_order:
        if assignment[root] >= 0:
            continue
        heads[:] = -1
        bucket[:] = -1
        distance[:] = infinity
        parent[:] = -1
        settled[:] = False
        visits, row = 0, root
        row_distance, last = np.int64(0), np.int64(0)
        while True:
            for edge in range(indptr[row], indptr[row + 1]):
                col = columns[edge]
                if settled[col]:
                    continue
                trial = row_distance + costs[edge] - u[row] - v[col]
                if trial >= distance[col]:
                    continue
                old = bucket[col]
                if old >= 0:
                    before, after = previous[col], following[col]
                    if before < 0:
                        heads[old] = after
                    else:
                        following[before] = after
                    if after >= 0:
                        previous[after] = before
                distance[col], parent[col] = trial, row
                slot = 2 * radix_bin(trial, last) + int(owner[col] >= 0)
                head = heads[slot]
                previous[col], following[col] = -1, head
                if head >= 0:
                    previous[head] = col
                heads[slot], bucket[col] = col, slot
            if heads[0] < 0 and heads[1] < 0:
                number = 1
                while number <= 64 and heads[2 * number] < 0 and heads[2 * number + 1] < 0:
                    number += 1
                if number > 64:
                    return assignment, False
                minimum = infinity
                for status in range(2):
                    col = heads[2 * number + status]
                    while col >= 0:
                        minimum = min(minimum, distance[col])
                        col = following[col]
                last = minimum
                for status in range(2):
                    col = heads[2 * number + status]
                    heads[2 * number + status] = -1
                    while col >= 0:
                        after = following[col]
                        slot = 2 * radix_bin(distance[col], last) + status
                        head = heads[slot]
                        previous[col], following[col] = -1, head
                        if head >= 0:
                            previous[head] = col
                        heads[slot], bucket[col] = col, slot
                        col = after
            if heads[0] >= 0:
                sink, length = heads[0], last
                break
            col = heads[1]
            head = following[col]
            heads[1] = head
            if head >= 0:
                previous[head] = -1
            bucket[col], settled[col] = -1, True
            visited[visits] = col
            visits += 1
            row, row_distance = owner[col], last
        # Update old matched rows only; the free sink keeps v=0. This is the
        # rectangular dual condition lost by naive epsilon-scaling auctions.
        u[root] += length
        for index in range(visits):
            col = visited[index]
            change = length - distance[col]
            u[owner[col]] += change
            v[col] -= change
        col = sink
        while col >= 0:
            row = parent[col]
            old = assignment[row]
            assignment[row], owner[col] = col, row
            col = old
    return assignment, True


def prepare_rearrangement_geometry(
    source_yx: object, target_yx: object, *, shape_yx: tuple[int, int],
    matching_radii: object, max_step: int = 1, minimum_separation: float,
) -> dict[str, object]:
    """Prepare fixed native Fourier-grid geometry without observing occupancy."""
    shape = _pair(shape_yx, "shape_yx")
    if not np.array_equal(np.asarray(shape_yx), shape):
        raise ValueError("shape_yx must contain positive integers")
    sites = []
    for name, values in (("source_yx", source_yx), ("target_yx", target_yx)):
        array = np.asarray(values)
        if (array.ndim != 2 or array.shape[1] != 2 or not len(array)
                or array.dtype.kind not in "iuf" or not np.all(np.isfinite(array))
                or np.any(array != np.floor(array))):
            raise ValueError(f"{name} must be a nonempty matrix of integer Y,X sites")
        if np.any(array < 0) or np.any(array >= shape):
            raise ValueError(f"{name} lies outside the native SLM shape")
        if np.any(array > np.iinfo(np.int32).max):
            raise ValueError(f"{name} exceeds the native int32 coordinate range")
        if len(np.unique(array, axis=0)) != len(array):
            raise ValueError(f"{name} must contain unique sites")
        sites.append(np.frombuffer(
            np.ascontiguousarray(array, dtype=np.int32).tobytes(), dtype=np.int32,
        ).reshape(-1, 2))
    source, target = sites
    if len(source) < len(target):
        raise ValueError("source site count is smaller than target site count")
    radii = np.asarray(matching_radii, dtype=float)
    if (radii.ndim != 1 or not radii.size or not np.all(np.isfinite(radii))
            or np.any(radii < 0) or np.any(np.diff(radii) <= 0)):
        raise ValueError("matching_radii must be finite increasing nonnegative radii")
    step = _scalar(max_step, "max_step")
    if step != int(step):
        raise ValueError("max_step must be a positive integer number of Fourier bins")
    separation = _scalar(minimum_separation, "minimum_separation")
    adjacency, eligible = [], []
    for radius in radii:
        pointers, column_parts, cost_parts = [0], [], []
        allowed = np.zeros(len(source), dtype=bool)
        for point in target:
            delta = source.astype(np.int64) - point
            cols = np.flatnonzero(np.max(np.abs(delta), axis=1) <= radius)
            values = np.sum(delta[cols] ** 2, axis=1)
            column_parts.append(cols)
            cost_parts.append(values)
            pointers.append(pointers[-1] + len(cols))
            allowed[cols] = True
        columns = np.concatenate(column_parts).astype(np.int64)
        costs = np.concatenate(cost_parts).astype(np.int64)
        divisor = max(1, int(np.gcd.reduce(costs))) if costs.size else 1
        normalized = costs // divisor
        # At most n augmentations, each of reduced length <= n*max(cost).
        # Keep dual updates and tentative distances within exact int64 arithmetic.
        span = len(target) ** 2 + len(target) + 2
        if normalized.size and int(normalized.max()) > np.iinfo(np.int64).max // span:
            raise ValueError("matching costs exceed the exact int64 working range")
        adjacency.append(tuple(_frozen(array) for array in (
            np.asarray(pointers, np.int64), columns, normalized,
        )))
        eligible.append(_frozen(allowed))
    order = _frozen(np.argsort(np.sum((target - np.mean(target, axis=0)) ** 2, axis=1), kind="stable"))
    global _REARRANGEMENT_MATCH
    if _REARRANGEMENT_MATCH is None:
        from numba import njit  # noqa: PLC0415

        _REARRANGEMENT_MATCH = njit(cache=True, nogil=True)(_match_rearrangement)
        _REARRANGEMENT_MATCH(*adjacency[-1], np.ones(len(source), dtype=bool), order)
    return {
        "source_yx": source, "target_yx": target, "shape_yx": shape,
        "matching_radii": tuple(float(radius) for radius in radii),
        "adjacency": tuple(adjacency), "eligible": tuple(eligible), "row_order": order,
        "max_step": int(step), "minimum_separation": separation,
    }


def plan_rearrangement(prepared: Mapping[str, object], occupied: object) -> dict[str, object]:
    """Assign observed atoms; the emitted path still requires a clearance gate.

    Fractions describe spatial progress, not a physical clock. Unselected
    occupied sources are reported; this planner does not discard their atoms.
    """
    source, target = prepared["source_yx"], prepared["target_yx"]
    # Fix dtype/layout/mutability at this small input boundary: immutable or
    # strided occupancy input must not trigger a new JIT signature online.
    mask = np.array(occupied, copy=True, order="C")
    if mask.dtype.kind != "b" or mask.shape != (len(source),):
        raise ValueError("occupied must contain one boolean per source site")
    if np.count_nonzero(mask) < len(target):
        raise ValueError("not enough occupied source sites for the target")
    for radius, adjacency, eligible in zip(
        prepared["matching_radii"], prepared["adjacency"], prepared["eligible"],
    ):
        if np.count_nonzero(mask & eligible) < len(target):
            continue
        assignment, feasible = _REARRANGEMENT_MATCH(*adjacency, mask, prepared["row_order"])
        if feasible:
            break
    else:
        raise ValueError("no complete assignment within the authored matching radii")
    assignment = assignment.astype(np.intp, copy=False)
    unselected = mask.copy()
    unselected[assignment] = False
    start = source[assignment]
    delta = target.astype(np.int64) - start
    steps = int(np.ceil(np.max(np.abs(delta)) / prepared["max_step"]))
    fraction = np.linspace(0.0, 1.0, steps + 1) if steps else np.zeros(1)
    # Integer arithmetic gives exact half-up rounding even when a rational
    # progress value lies just below a tie after floating-point division.
    motion = (start[None] + (2 * np.arange(steps + 1)[:, None, None] * delta
              + steps) // (2 * steps)) if steps else start[None].copy()
    return {
        "assignment": _frozen(assignment),
        "unselected_occupied": _frozen(np.flatnonzero(unselected)),
        "motion_yx": _frozen(motion.astype(np.int32)),
        "fraction": _frozen(fraction), "matching_radius": radius,
    }


def rearrangement_clearance(motion_yx: object) -> float:
    """Exact minimum distance along every emitted straight frame-to-frame path."""
    from scipy.spatial.distance import pdist  # noqa: PLC0415

    motion = np.asarray(motion_yx, dtype=float)
    if (motion.ndim != 3 or motion.shape[2] != 2 or min(motion.shape[:2]) < 1
            or not np.all(np.isfinite(motion))):
        raise ValueError("motion_yx must contain finite frame, site, Y/X coordinates")
    minimum = float(pdist(motion[0], "sqeuclidean").min(initial=np.inf))
    for start, end in zip(motion[:-1], motion[1:]):
        start_squared = pdist(start, "sqeuclidean")
        velocity_squared = pdist(end - start, "sqeuclidean")
        dot = (pdist(end, "sqeuclidean") - start_squared - velocity_squared) * 0.5
        progress = np.divide(-dot, velocity_squared, out=np.zeros_like(dot),
                             where=velocity_squared > 0)
        np.clip(progress, 0.0, 1.0, out=progress)
        squared = start_squared + progress * (2 * dot + progress * velocity_squared)
        minimum = min(minimum, float(squared.min(initial=np.inf)))
    return float(np.sqrt(max(0.0, minimum)))


def preset_grid(
    shape_yx: object,
    grid_shape_yx: object,
    *,
    spacing_yx: object | None = None,
    intensity: float = 1.0,
) -> np.ndarray:
    shape, rows, columns = _grid_indices(shape_yx, grid_shape_yx, spacing_yx)
    target = np.zeros(shape, dtype=np.float32)
    target[np.ix_(rows, columns)] = _scalar(intensity, "intensity", nonnegative=True)
    return validate_target(target)

def preset_checkerboard(
    shape_yx: object,
    grid_shape_yx: object,
    *,
    spacing_yx: object | None = None,
    intensity: float = 1.0,
) -> np.ndarray:
    shape = _pair(shape_yx, "shape_yx")
    row_count, long_count = _pair(grid_shape_yx, "grid_shape_yx")
    if long_count < 2:
        raise ValueError("checkerboard long rows must contain at least two sites")
    spacing = (
        (
            max(1, shape[0] // (row_count + 1)),
            max(1, shape[1] // (2 * long_count)),
        )
        if spacing_yx is None
        else _pair(spacing_yx, "spacing_yx")
    )
    span_y = (row_count - 1) * spacing[0]
    span_x = 2 * (long_count - 1) * spacing[1]
    if span_y >= shape[0] or span_x >= shape[1]:
        raise ValueError("checkerboard does not fit inside target shape")
    rows = (shape[0] - span_y) // 2 + spacing[0] * np.arange(row_count)
    long_columns = (
        (shape[1] - span_x) // 2
        + 2 * spacing[1] * np.arange(long_count)
    )
    short_columns = long_columns[:-1] + spacing[1]
    target = np.zeros(shape, dtype=np.float32)
    level = _scalar(intensity, "intensity", nonnegative=True)
    for index, row in enumerate(rows):
        target[row, long_columns if index % 2 == 0 else short_columns] = level
    return validate_target(target)

def preset_gaussian(
    shape_yx: object,
    radius_yx: object,
    *,
    intensity: float = 1.0,
) -> np.ndarray:
    shape = _pair(shape_yx, "shape_yx")
    radius = _float_pair(radius_yx, "radius_yx")
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    exponent = -2.0 * (
        ((yy - shape[0] // 2) / radius[0]) ** 2
        + ((xx - shape[1] // 2) / radius[1]) ** 2
    )
    profile = np.exp(exponent)
    profile[exponent < -8.0] = 0.0
    return validate_target(
        _scalar(intensity, "intensity", nonnegative=True) * profile
    )

def preset_flat_top(
    shape_yx: object,
    radius_yx: object,
    *,
    intensity: float = 1.0,
    edge: float = 0.0,
) -> np.ndarray:
    shape = _pair(shape_yx, "shape_yx")
    radius = _float_pair(radius_yx, "radius_yx")
    if any(2.0 * radius[index] > shape[index] for index in range(2)):
        raise ValueError("flat top does not fit inside target shape")
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    radial = np.sqrt(
        ((yy - shape[0] // 2) / radius[0]) ** 2
        + ((xx - shape[1] // 2) / radius[1]) ** 2
    )
    width = _scalar(edge, "edge", nonnegative=True)
    if width == 0.0:
        profile = (radial <= 1.0).astype(np.float32)
    else:
        distance = (1.0 - radial) * min(radius)
        profile = np.clip(0.5 + distance / width, 0.0, 1.0).astype(np.float32)
    return validate_target(
        _scalar(intensity, "intensity", nonnegative=True) * profile
    )

_WINDOWS_CJK_FONTS = (
    "msyh.ttc",
    "msyhbd.ttc",
    "simhei.ttf",
    "Dengb.ttf",
    "Deng.ttf",
    "simsun.ttc",
)

def _missing_font_characters(font_path: Path, text: str) -> tuple[str, ...]:
    from PIL import ImageFont  # noqa: PLC0415

    font = ImageFont.truetype(
        str(font_path),
        size=64,
        layout_engine=ImageFont.Layout.BASIC,
    )
    notdef = font.getmask("\U0010ffff", mode="L")
    notdef_key = notdef.size, bytes(notdef)
    missing: list[str] = []
    for character in text:
        if character == " ":
            continue
        glyph = font.getmask(character, mode="L")
        if (glyph.size, bytes(glyph)) == notdef_key and character not in missing:
            missing.append(character)
    return tuple(missing)

def _text_font_path(font_path: str | Path | None, text: str) -> Path:
    if font_path is not None:
        selected = Path(font_path)
        if not selected.is_file():
            raise FileNotFoundError(f"text font does not exist: {selected}")
        try:
            missing = _missing_font_characters(selected, text)
        except OSError as error:
            raise ValueError(f"text font could not be loaded: {selected}") from error
        if missing:
            raise ValueError(
                f"text font does not contain requested characters "
                f"{''.join(missing)!r}: {selected}"
            )
        return selected
    font_root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    found = False
    for name in _WINDOWS_CJK_FONTS:
        candidate = font_root / name
        if not candidate.is_file():
            continue
        found = True
        try:
            missing = _missing_font_characters(candidate, text)
        except OSError:
            continue
        if not missing:
            return candidate
    if found:
        raise ValueError(
            "no supported Windows CJK font contains every requested character"
        )
    raise FileNotFoundError(
        "no supported Windows CJK font was found "
        f"({', '.join(_WINDOWS_CJK_FONTS)})"
    )

def _allowed_text_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        character == " "
        or "A" <= character <= "Z"
        or "a" <= character <= "z"
        or "0" <= character <= "9"
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )

def _rasterized_text(text: str, font_path: Path, size: int) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    font = ImageFont.truetype(
        str(font_path),
        size=size,
        layout_engine=ImageFont.Layout.BASIC,
    )
    glyphs: list[tuple[str, int, tuple[int, int, int, int]]] = []
    visible_index = 0
    for index, character in enumerate(text):
        if character == " ":
            continue
        box = font.getbbox(character, anchor="ls")
        origin = int(round(
            font.getlength(text[:index + 1]) - font.getlength(character)
        )) + visible_index
        glyphs.append((character, origin, box))
        visible_index += 1
    if not glyphs:
        return np.zeros((0, 0), dtype=bool)

    left = min(origin + box[0] for _, origin, box in glyphs)
    right = max(origin + box[2] for _, origin, box in glyphs)
    top = min(box[1] for _, _, box in glyphs)
    bottom = max(box[3] for _, _, box in glyphs)
    image = Image.new("L", (max(1, right - left), max(1, bottom - top)), 0)
    draw = ImageDraw.Draw(image)
    for character, origin, box in glyphs:
        glyph_image = Image.new(
            "L", (max(1, box[2] - box[0]), max(1, box[3] - box[1])), 0
        )
        ImageDraw.Draw(glyph_image).text(
            (-box[0], -box[1]),
            character,
            fill=255,
            font=font,
            anchor="ls",
        )
        if not np.any(np.asarray(glyph_image, dtype=np.uint8) >= 128):
            return np.zeros((0, 0), dtype=bool)
        draw.text(
            (origin - left, -top),
            character,
            fill=255,
            font=font,
            anchor="ls",
        )
    mask = np.asarray(image, dtype=np.uint8) >= 128
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[
        rows.min():rows.max() + 1,
        columns.min():columns.max() + 1,
    ]

def preset_text(
    shape_yx: object,
    text: object,
    *,
    spacing: int,
    atom_budget: int,
    intensity: float = 1.0,
    font_path: str | Path | None = None,
) -> np.ndarray:
    """Rasterize one centered line of Latin/CJK text into discrete sites."""

    import unicodedata  # noqa: PLC0415

    shape = _pair(shape_yx, "shape_yx")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFC", text).strip(" ")
    if not normalized:
        raise ValueError("text must contain at least one visible character")
    if any(not _allowed_text_character(character) for character in normalized):
        raise ValueError(
            "text may contain only ASCII letters, digits, CJK characters, "
            "and ordinary spaces"
        )
    try:
        pitch = int(spacing)
    except (TypeError, ValueError) as error:
        raise TypeError("spacing must be a positive integer") from error
    if isinstance(spacing, bool) or pitch != spacing or pitch <= 0:
        raise ValueError("spacing must be a positive integer")
    try:
        budget = int(atom_budget)
    except (TypeError, ValueError) as error:
        raise TypeError("atom_budget must be a positive integer") from error
    if isinstance(atom_budget, bool) or budget != atom_budget or budget <= 0:
        raise ValueError("atom_budget must be a positive integer")
    selected_font = _text_font_path(font_path, normalized)
    logical_shape = (
        (shape[0] - 1) // pitch + 1,
        (shape[1] - 1) // pitch + 1,
    )
    best: np.ndarray | None = None
    for size in range(1, 2 * logical_shape[0] + 2):
        try:
            mask = _rasterized_text(normalized, selected_font, size)
        except OSError as error:
            raise ValueError(
                f"text font could not be loaded: {selected_font}"
            ) from error
        if mask.size == 0:
            continue
        if mask.shape[0] > logical_shape[0] or mask.shape[1] > logical_shape[1]:
            break
        if int(np.count_nonzero(mask)) > budget:
            continue
        best = mask
    if best is None:
        raise ValueError("text does not fit the target shape and atom budget")

    rows, columns = np.nonzero(best)
    span_y = (best.shape[0] - 1) * pitch
    span_x = (best.shape[1] - 1) * pitch
    rows = (shape[0] - span_y) // 2 + pitch * rows
    columns = (shape[1] - span_x) // 2 + pitch * columns
    target = np.zeros(shape, dtype=np.float32)
    target[rows, columns] = _scalar(
        intensity, "intensity", nonnegative=True
    )
    return validate_target(target)

def imported_target(values: object) -> np.ndarray:
    target = validate_target(values)
    peak = float(np.max(target))
    if peak <= 0.0:
        raise ValueError("imported target must contain positive intensity")
    return _readonly(target / peak)

#: The default pupil and its shifted plane, per shape: a pure function of
#: the shape, built once.
_DEFAULT_PUPILS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


#: Elementwise passes over the full 1024x1272 plane (arctan2, the phase
#: projection) dominate a solve's fixed cost, and numpy ufuncs release the
#: GIL on arrays this large -- so row stripes on a small shared pool run
#: them in parallel with bit-identical results (the same libm call per
#: element).  Planes below the threshold run inline.
_STRIPE_THRESHOLD = 1 << 18
_STRIPE_COUNT = 4
#: One pool for every solve, made here rather than on first use: two solves
#: at once (the Editor's and a Feedback run's) each found none and each made
#: one.  An executor starts no thread until work is submitted to it.
_STRIPE_POOL = ThreadPoolExecutor(
    max_workers=_STRIPE_COUNT,
    thread_name_prefix="slm-solver-stripe",
)


def _stripes(operation: Callable[[slice], None], rows: int, size: int) -> None:
    if size < _STRIPE_THRESHOLD or rows < _STRIPE_COUNT:
        operation(slice(0, rows))
        return
    step = (rows + _STRIPE_COUNT - 1) // _STRIPE_COUNT
    futures = [
        _STRIPE_POOL.submit(operation, slice(start, min(rows, start + step)))
        for start in range(0, rows, step)
    ]
    for future in futures:
        future.result()


def _frozen(values: np.ndarray) -> np.ndarray:
    """Freeze a solver-owned fresh array in place -- no copy, no retyping."""

    values.setflags(write=False)
    return values


def _pupil(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.ogrid[-1.0:1.0:shape[0] * 1j, -1.0:1.0:shape[1] * 1j]
    return (xx * xx + yy * yy <= 0.9**2).astype(np.float32)

def _unit_phase(values: np.ndarray, epsilon: float) -> np.ndarray:
    magnitude = np.abs(values).astype(np.float32, copy=False)
    return np.divide(
        values,
        magnitude,
        out=np.ones_like(values, dtype=np.complex64),
        where=magnitude > epsilon,
    )

#: The spots solve stops once the simulated support intensity, divided by the
#: desired intensity site by site, has a max/min no larger than this.  One
#: owner for the number: the loop's early-stop gate and the re-check of the
#: canonical phase both read it, and a caller that needs a tighter answer
#: passes ``support_tolerance`` to ``solve_phase`` instead of editing here.
SPOT_SUPPORT_TOLERANCE = 1.01


def _support_intensity_ratio(
    magnitude: np.ndarray,
    desired: np.ndarray,
    epsilon: float,
) -> float:
    relative = np.square(magnitude, dtype=np.float32) / desired
    return float(np.max(relative) / max(float(np.min(relative)), epsilon))


def _image_metrics(
    far_field: np.ndarray,
    desired: np.ndarray,
    support: np.ndarray,
    epsilon: float,
) -> tuple[float, float, float, float]:
    power = np.square(np.abs(far_field), dtype=np.float32)
    expected = desired[support]
    relative = power[support] / np.maximum(expected, epsilon)
    relative /= max(
        float(np.sum(relative * expected) / np.sum(expected)), epsilon
    )
    relative_rms = float(
        np.sqrt(
            np.sum(expected * np.square(relative - 1.0)) / np.sum(expected)
        )
    )
    relative_image = np.zeros(desired.shape, dtype=np.float32)
    relative_image[support] = relative
    differences = []
    vertical = support[1:] & support[:-1]
    horizontal = support[:, 1:] & support[:, :-1]
    if np.any(vertical):
        differences.append((relative_image[1:] - relative_image[:-1])[vertical])
    if np.any(horizontal):
        differences.append((relative_image[:, 1:] - relative_image[:, :-1])[horizontal])
    roughness = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.concatenate(differences)
                    if differences
                    else np.zeros(1, dtype=np.float32)
                )
            )
        )
    )
    background = float(
        np.sum(power[~support]) / max(float(np.sum(power)), epsilon)
    )
    return (
        relative_rms,
        roughness,
        background,
        relative_rms + 0.05 * roughness + 0.10 * background,
    )

def _project_field(
    back: np.ndarray,
    pupil: np.ndarray,
    magnitude: np.ndarray,
    mask: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """``pupil * _unit_phase(back)`` composed in caller-owned buffers.

    This projection runs once per iteration over the full SLM plane, and the
    naive expression allocated three plane-sized temporaries per pass --
    measured at roughly 40% of a whole spots solve.  Same arithmetic, same
    order; ``back`` is consumed as scratch.
    """

    epsilon = np.float32(np.finfo(np.float32).eps)

    def stripe(rows: slice) -> None:
        np.abs(back[rows], out=magnitude[rows])
        np.greater(magnitude[rows], epsilon, out=mask[rows])
        np.divide(back[rows], magnitude[rows], out=back[rows], where=mask[rows])
        np.logical_not(mask[rows], out=mask[rows])
        np.copyto(back[rows], np.complex64(1.0), where=mask[rows])
        np.multiply(back[rows], pupil[rows], out=out[rows])

    _stripes(stripe, back.shape[0], back.size)
    return out


def _radial_transport_seed(
    desired: np.ndarray, pupil: np.ndarray
) -> np.ndarray | None:
    """Radial transport initial phase for a near-isotropic image target.

    For a target whose energy is closer to circular than elliptical, the
    exact 1-D radial energy match (annulus by annulus) beats the separable
    marginal construction, whose product-form caustic is only right for
    separable targets.  Returns None for strongly anisotropic targets.
    """

    shape = desired.shape
    height, width = shape
    target = desired.astype(np.float64)
    total = float(target.sum())
    if total <= 0.0:
        return None
    yy, xx = np.ogrid[:height, :width]
    cy = float((target.sum(axis=1) * np.arange(height)).sum() / total)
    cx = float((target.sum(axis=0) * np.arange(width)).sum() / total)
    var_y = float((target.sum(axis=1) * (np.arange(height) - cy) ** 2).sum() / total)
    var_x = float((target.sum(axis=0) * (np.arange(width) - cx) ** 2).sum() / total)
    if min(var_y, var_x) <= 0.0:
        return None
    # A separable target (a Gaussian, any product form) is served exactly by
    # the separable marginal construction -- the radial build only wins for
    # genuinely round, non-product shapes such as a flat-top disk.
    marginal_y = target.sum(axis=1)
    marginal_x = target.sum(axis=0)
    product = np.outer(marginal_y, marginal_x) / total
    product_error = float(np.abs(product - target).sum()) / total
    if product_error < 0.05:
        return None
    anisotropy = max(var_y / var_x, var_x / var_y) ** 0.5
    if anisotropy > 1.05:
        return None
    power = pupil.astype(np.float64)
    power = power * power
    source_total = float(power.sum())
    if source_total <= 0.0:
        return None
    sy = float((power.sum(axis=1) * np.arange(height)).sum() / source_total)
    sx = float((power.sum(axis=0) * np.arange(width)).sum() / source_total)
    source_rho = np.hypot(yy - sy, xx - sx)
    target_rho = np.hypot(yy - cy, xx - cx)
    bins = int(np.ceil(max(source_rho.max(), target_rho.max()))) + 1
    source_hist = np.bincount(
        source_rho.astype(np.int64).ravel(), weights=power.ravel(), minlength=bins
    )
    target_hist = np.bincount(
        target_rho.astype(np.int64).ravel(), weights=target.ravel(), minlength=bins
    )
    cumulative_source = np.cumsum(source_hist) / source_total
    cumulative_target = np.cumsum(target_hist) / total
    radius = np.arange(bins, dtype=np.float64)
    mapped = np.interp(cumulative_source, cumulative_target, radius)
    # A radial potential integrates the matched slope; the anisotropic FFT
    # pixel pitch is folded in through the geometric-mean extent, which is
    # exact for a square raster and a seed-grade approximation otherwise.
    scale = 2.0 * np.pi / float(np.sqrt(height * width))
    profile = np.cumsum(mapped) * scale
    profile -= profile[0]
    seed = np.interp(source_rho.ravel(), radius, profile).reshape(shape)
    seed += 2.0 * np.pi * (
        (cy - height / 2.0) * (yy - sy) / height
        + (cx - width / 2.0) * (xx - sx) / width
    )
    return seed.astype(np.float32)


def _mapping_seed(desired: np.ndarray, pupil: np.ndarray) -> np.ndarray:
    """Separable geometric-mapping initial phase for an image target.

    Marginal energy matching between the pupil illumination and the target
    intensity gives, per axis, the ray mapping u -> R(u); integrating the
    matched linear-phase slope yields a caustic seed that starts MRAF near
    the transport solution instead of a fixed quadratic guess.  The seed
    only decides where iteration starts; the quality metrics, the stop
    criteria and the returned contract are untouched.
    """

    shape = desired.shape
    seed = np.zeros(shape, dtype=np.float32)
    power = pupil.astype(np.float64)
    np.square(power, out=power)
    target = desired.astype(np.float64)
    for axis in (0, 1):
        other = 1 - axis
        source_marginal = power.sum(axis=other)
        target_marginal = target.sum(axis=other)
        n = shape[axis]
        source_total = float(source_marginal.sum())
        target_total = float(target_marginal.sum())
        if source_total <= 0.0 or target_total <= 0.0:
            continue
        cumulative_source = np.cumsum(source_marginal) / source_total
        cumulative_target = np.cumsum(target_marginal) / target_total
        mapped = np.interp(
            cumulative_source,
            cumulative_target,
            np.arange(n, dtype=np.float64),
        )
        slope = 2.0 * np.pi * (mapped - n / 2.0) / n
        profile = np.cumsum(slope)
        profile -= profile[n // 2]
        if axis == 0:
            seed += profile.astype(np.float32)[:, None]
        else:
            seed += profile.astype(np.float32)[None, :]
    return seed


def _canonical_unshifted_phase(field: np.ndarray) -> np.ndarray:
    phase = np.empty(field.shape, dtype=np.float32)

    def stripe(rows: slice) -> None:
        part = field[rows]
        np.arctan2(part.imag, part.real, out=phase[rows])
        np.add(
            phase[rows],
            np.float32(2.0 * np.pi),
            out=phase[rows],
            where=phase[rows] < 0.0,
        )
        np.minimum(
            phase[rows],
            np.nextafter(np.float32(2.0 * np.pi), np.float32(0.0)),
            out=phase[rows],
        )

    _stripes(stripe, field.shape[0], field.size)
    return phase

def _phase_snapshot(field: np.ndarray) -> np.ndarray:
    from scipy import fft  # noqa: PLC0415

    return _readonly(fft.fftshift(_canonical_unshifted_phase(field)))

def _cartesian_support(
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    site_rows, site_columns = np.nonzero(support)
    if site_rows.size > 256:
        return None
    rows = np.unique(site_rows)
    columns = np.unique(site_columns)
    # The exact separable transform forms the selected row/column envelope.
    # Beyond these small bounds a full FFT is both simpler and cheaper; in
    # particular, 256 unrelated diagonal sites must not create a 256x256 DFT.
    if columns.size > 64 or rows.size * columns.size > 4096:
        return None
    active = support[np.ix_(rows, columns)]
    return rows, columns, active

def solve_phase(
    target: object,
    *,
    pupil_amplitude: object | None = None,
    spot_optimizer_state: dict[str, object] | None = None,
    initial_phase: object | None = None,
    objective_kind: str = "auto",
    iterations: int | None = None,
    seed: int = 0,
    stop_requested: Callable[[], bool] | None = None,
    support_tolerance: float = SPOT_SUPPORT_TOLERANCE,
    minimum_iterations: int = 1,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve one target, optionally reusing caller-owned transient spot state.

    The caller clears that state whenever the authored input pupil changes.

    ``support_tolerance`` and ``minimum_iterations`` govern the spots
    early-stop gate only: the solve may declare itself done no earlier than
    ``minimum_iterations`` passes and only once the support max/min intensity
    ratio is within ``support_tolerance``.  A feedback loop that re-solves
    every candidate from a hot start needs both, because the default 1%
    gate let a 1-2 iteration solve leave a fresh ~0.2% rms intensity error
    pattern on the sites each candidate -- three times the loop's own
    per-step correction, injected as noise the controller then chased.
    """

    # The transforms are reached here, not at the top of this module.  This
    # file declares the SLM's targets, presets and file formats as well as
    # solving for phase, and a logic node's descriptor names one of those
    # file readers as its artifact codec -- so discovering the nodes, which
    # a task console does before it draws anything, used to import scipy.
    # That was half a second of every console's open for a transform that
    # runs when somebody actually solves a pattern.
    from scipy import fft, ndimage  # noqa: PLC0415

    tolerance = float(support_tolerance)
    if not np.isfinite(tolerance) or tolerance < 1.0:
        raise ValueError("support_tolerance must be a finite ratio >= 1")
    if isinstance(minimum_iterations, bool) or int(minimum_iterations) < 1:
        raise ValueError("minimum_iterations must be a positive integer")
    minimum_passes = int(minimum_iterations)
    desired = validate_target(target)
    if float(np.max(desired)) <= 0.0:
        raise ValueError("target must contain positive intensity")
    if spot_optimizer_state is not None and not isinstance(
        spot_optimizer_state, dict
    ):
        raise TypeError("spot_optimizer_state must be a dict or None")
    saved_state = dict(spot_optimizer_state) if spot_optimizer_state else None
    state_requested = spot_optimizer_state is not None
    seed_value = int(seed)
    if pupil_amplitude is None:
        cached_default = _DEFAULT_PUPILS.get(desired.shape)
        if cached_default is None:
            pupil = _frozen(_pupil(desired.shape))
            cached_default = (pupil, _frozen(fft.ifftshift(pupil)))
            _DEFAULT_PUPILS[desired.shape] = cached_default
        pupil, pupil_unshifted = cached_default
        pupil_source = "default"
    else:
        pupil_source = "provided"
        try:
            pupil = np.asarray(pupil_amplitude, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise TypeError("pupil_amplitude must be a numeric array") from error
        if pupil.shape != desired.shape:
            raise ValueError("pupil_amplitude shape must match the target shape")
        if not np.all(np.isfinite(pupil)):
            raise ValueError("pupil_amplitude must be finite")
        if np.any(pupil < 0.0):
            raise ValueError("pupil_amplitude must be non-negative")
        if not np.any(pupil > 0.0):
            raise ValueError("pupil_amplitude must contain positive amplitude")
        pupil = _readonly(pupil)
        pupil_unshifted = _frozen(fft.ifftshift(pupil))
    if not isinstance(objective_kind, str) or objective_kind not in {
        "auto",
        "spots",
        "image",
    }:
        raise ValueError("objective_kind must be 'auto', 'spots', or 'image'")
    if stop_requested is not None and not callable(stop_requested):
        raise TypeError("stop_requested must be callable or None")
    support = desired > 0.0
    if objective_kind == "auto":
        labels, components = ndimage.label(support)
        sizes = np.bincount(labels.ravel(), minlength=components + 1)[1:]
        largest = int(np.max(sizes, initial=0))
        continuous_component = max(64, int(np.ceil(0.001 * desired.size)))
        resolved_kind = (
            "image"
            if (
                largest > continuous_component
                or np.count_nonzero(support) > 0.02 * desired.size
            )
            else "spots"
        )
    else:
        resolved_kind = objective_kind
    method = "wgs-kim" if resolved_kind == "spots" else "mraf"
    if iterations is None:
        count = 80 if method == "wgs-kim" else 300
    else:
        if isinstance(iterations, bool) or int(iterations) <= 0:
            raise ValueError("iterations must be a positive integer or None")
        count = int(iterations)

    desired_unshifted = _frozen(fft.ifftshift(desired))
    support_unshifted = _frozen(desired_unshifted > 0.0)
    epsilon = np.finfo(np.float32).eps
    # One set of plane-sized scratch buffers for the whole solve: the
    # per-iteration projection reuses them instead of allocating ~30 MB of
    # temporaries per pass.
    plane_magnitude = np.empty(desired.shape, dtype=np.float32)
    plane_mask = np.empty(desired.shape, dtype=np.bool_)
    field_buffer = np.empty(desired.shape, dtype=np.complex64)
    plane_scratch = np.empty(desired.shape, dtype=np.complex64)

    transform = "fft"
    early_stopped = False
    stop_was_interior = False
    iterations_run = 0
    hot_start_used = False
    checked_result: np.ndarray | None = None
    checked_selected: np.ndarray | None = None
    state_status = "not-requested" if not state_requested else "created"
    support_yx: list[list[int]] | None = None
    fixed_phase: np.ndarray | None = None
    weights: np.ndarray
    if method == "wgs-kim":
        cartesian = _cartesian_support(support_unshifted)
        if cartesian is None:
            desired_spots = desired_unshifted[support_unshifted]
            constrained = np.zeros(desired.shape, dtype=np.complex64)
            if saved_state is not None:
                state_status = "support-changed"
        else:
            rows, columns, active = cartesian
            desired_spots = desired_unshifted[np.ix_(rows, columns)][active]
            row_angles = (
                (-2.0 * np.pi / desired.shape[0])
                * rows.astype(np.float64)[:, None]
                * np.arange(desired.shape[0], dtype=np.float64)[None, :]
            )
            row_forward = (
                np.exp(1j * row_angles) / np.sqrt(desired.shape[0])
            ).astype(np.complex64)
            row_backward = np.ascontiguousarray(row_forward.conj().T)
            column_angles = (
                (-2.0 * np.pi / desired.shape[1])
                * columns.astype(np.float64)[:, None]
                * np.arange(desired.shape[1], dtype=np.float64)[None, :]
            )
            column_forward = (
                np.exp(1j * column_angles) / np.sqrt(desired.shape[1])
            ).astype(np.complex64)
            column_forward_transposed = np.ascontiguousarray(column_forward.T)
            column_backward = np.ascontiguousarray(
                column_forward.conj()
            )
            constrained_selected = np.zeros(
                active.shape, dtype=np.complex64
            )
            transform = "selected-dft"
            support_yx = [
                [
                    int((rows[row_index] + desired.shape[0] // 2) % desired.shape[0]),
                    int((columns[column_index] + desired.shape[1] // 2) % desired.shape[1]),
                ]
                for row_index, column_index in np.argwhere(active)
            ]

        amplitude_spots = np.sqrt(desired_spots).astype(
            np.float32, copy=False
        )
        amplitude_spots /= np.linalg.norm(amplitude_spots)

        if saved_state is not None and transform == "selected-dft":
            if saved_state.get("objective_kind") != "spots":
                state_status = "objective-changed"
            elif saved_state.get("pupil_source") != pupil_source:
                state_status = "pupil-changed"
            elif saved_state.get("shape_yx") != list(desired.shape):
                state_status = "support-changed"
            else:
                try:
                    saved_support = np.asarray(
                        saved_state["support_yx"], dtype=np.int64
                    )
                    saved_fixed = np.asarray(
                        saved_state["fixed_farfield_phase"], dtype=np.float32
                    )
                    saved_weights = np.asarray(
                        saved_state["site_weights"], dtype=np.float32
                    )
                    saved_amplitudes = np.asarray(
                        saved_state["target_amplitudes"], dtype=np.float32
                    )
                except (KeyError, TypeError, ValueError):
                    state_status = "invalid"
                else:
                    site_count = len(amplitude_spots)
                    if not np.array_equal(saved_support, support_yx):
                        state_status = "support-changed"
                    elif (
                        saved_fixed.shape != (site_count,)
                        or saved_weights.shape != (site_count,)
                        or saved_amplitudes.shape != (site_count,)
                        or not np.all(np.isfinite(saved_fixed))
                        or not np.all(np.isfinite(saved_weights))
                        or not np.all(np.isfinite(saved_amplitudes))
                        or np.any(saved_weights <= 0.0)
                        or np.any(saved_amplitudes <= 0.0)
                    ):
                        state_status = "invalid"
                    else:
                        fixed_phase = np.exp(
                            np.complex64(1j) * saved_fixed
                        ).astype(np.complex64, copy=False)
                        weights = np.array(saved_weights, copy=True)
                        weights *= amplitude_spots / saved_amplitudes
                        weights /= max(float(np.linalg.norm(weights)), epsilon)
                        if stop_requested is not None and stop_requested():
                            raise InterruptedError("SLM phase solve stopped")
                        constrained_selected.fill(0.0)
                        constrained_selected[active] = weights * fixed_phase
                        np.matmul(
                            row_backward @ constrained_selected,
                            column_backward,
                            out=plane_scratch,
                        )
                        field = _project_field(
                            plane_scratch,
                            pupil_unshifted,
                            plane_magnitude,
                            plane_mask,
                            field_buffer,
                        )
                        hot_start_used = True
                        state_status = "reused"

        if not hot_start_used:
            if initial_phase is None:
                phase = np.random.default_rng(seed_value).uniform(
                    0.0, 2.0 * np.pi, desired.shape
                ).astype(np.float32)
            else:
                phase = np.array(
                    canonical_phase(initial_phase, desired.shape), copy=True
                )
            np.multiply(phase, np.complex64(1j), out=plane_scratch)
            np.exp(plane_scratch, out=plane_scratch)
            plane_scratch *= pupil
            field = fft.ifftshift(plane_scratch)
            weights = np.array(amplitude_spots, copy=True)

        selected: np.ndarray | None = None
        while iterations_run < count:
            if stop_requested is not None and stop_requested():
                raise InterruptedError("SLM phase solve stopped")
            if selected is None:
                if transform == "selected-dft":
                    selected_grid = (
                        row_forward @ (field @ column_forward_transposed)
                    )
                    selected = selected_grid[active]
                else:
                    far = fft.fft2(field, norm="ortho")
                    selected = far[support_unshifted]
            magnitude = np.abs(selected).astype(np.float32, copy=False)
            measured = magnitude / max(float(np.linalg.norm(magnitude)), epsilon)
            weights *= np.clip(
                amplitude_spots / np.maximum(measured, epsilon), 0.2, 5.0
            ) ** np.float32(0.8)
            weights /= max(float(np.linalg.norm(weights)), epsilon)
            current_phase = _unit_phase(selected, epsilon)
            if fixed_phase is None:
                selected_phase = current_phase
                if iterations_run + 1 == 12:
                    fixed_phase = np.array(current_phase, copy=True)
            else:
                selected_phase = fixed_phase
            constrained_values = weights * selected_phase
            if transform == "selected-dft":
                constrained_selected.fill(0.0)
                constrained_selected[active] = constrained_values
                np.matmul(
                    row_backward @ constrained_selected,
                    column_backward,
                    out=plane_scratch,
                )
                back = plane_scratch
            else:
                constrained.fill(0.0)
                constrained[support_unshifted] = constrained_values
                back = fft.ifft2(constrained, norm="ortho")
            field = _project_field(
                back, pupil_unshifted, plane_magnitude, plane_mask, field_buffer
            )
            iterations_run += 1
            selected = None

            gate_start = max(1 if hot_start_used else 12, minimum_passes)
            if iterations is None and iterations_run >= gate_start:
                if transform == "selected-dft":
                    selected_grid = (
                        row_forward @ (field @ column_forward_transposed)
                    )
                    selected = selected_grid[active]
                else:
                    far = fft.fft2(field, norm="ortho")
                    selected = far[support_unshifted]
                magnitude = np.abs(selected).astype(np.float32, copy=False)
                support_ratio = _support_intensity_ratio(
                    magnitude, desired_spots, epsilon
                )
                checked_result = None
                checked_selected = None
                if support_ratio <= tolerance:
                    candidate_phase = _canonical_unshifted_phase(field)
                    candidate_field = np.empty(
                        desired.shape, dtype=np.complex64
                    )
                    np.multiply(
                        candidate_phase,
                        np.complex64(1j),
                        out=candidate_field,
                    )
                    np.exp(candidate_field, out=candidate_field)
                    candidate_field *= pupil_unshifted
                    if transform == "selected-dft":
                        candidate_grid = (
                            row_forward
                            @ (candidate_field @ column_forward_transposed)
                        )
                        candidate_selected = candidate_grid[active]
                    else:
                        candidate_far = fft.fft2(candidate_field, norm="ortho")
                        candidate_selected = candidate_far[support_unshifted]
                    candidate_ratio = _support_intensity_ratio(
                        np.abs(candidate_selected).astype(
                            np.float32, copy=False
                        ),
                        desired_spots,
                        epsilon,
                    )
                    if candidate_ratio <= tolerance:
                        checked_result = _readonly(
                            fft.fftshift(candidate_phase)
                        )
                        checked_selected = candidate_selected
                        early_stopped = True
                        break
    else:
        if saved_state is not None:
            state_status = "objective-changed"
        if bool(np.all(support_unshifted)):
            raise ValueError(
                "image objective requires zero-valued pixels defining a noise region"
            )
        amplitude = np.sqrt(desired_unshifted).astype(
            np.float32, copy=False
        )
        amplitude /= np.linalg.norm(amplitude[support_unshifted])
        amplitude_spots = amplitude[support_unshifted]

        def build_field(seed_phase: np.ndarray) -> np.ndarray:
            np.multiply(seed_phase, np.complex64(1j), out=plane_scratch)
            np.exp(plane_scratch, out=plane_scratch)
            np.multiply(plane_scratch, pupil, out=plane_scratch)
            return fft.ifftshift(plane_scratch)

        def mraf_update(
            far: np.ndarray, weights: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            """One MRAF pass from an already-computed far field.

            ``far`` is consumed as scratch and ``weights`` is updated in
            place, exactly as the loop always did.
            """

            selected = far[support_unshifted]
            magnitude = np.abs(selected).astype(np.float32, copy=False)
            measured = magnitude / max(float(np.linalg.norm(magnitude)), epsilon)
            weights *= np.sqrt(
                np.clip(amplitude_spots / np.maximum(measured, epsilon), 0.2, 5.0)
            )
            weights /= max(float(np.linalg.norm(weights)), epsilon)
            current_power = float(np.sum(np.square(magnitude, dtype=np.float32)))
            # ``selected`` was copied out above, so the noise-region scaling
            # can run on ``far`` itself instead of a fresh full plane.
            far *= np.complex64(0.9)
            far[support_unshifted] = (
                weights
                * np.sqrt(max(current_power, epsilon))
                * _unit_phase(selected, epsilon)
            )
            back = fft.ifft2(far, norm="ortho")
            return (
                _project_field(
                    back,
                    pupil_unshifted,
                    plane_magnitude,
                    plane_mask,
                    field_buffer,
                ),
                weights,
            )

        image_seed = "authored"
        coarse_iterations = 0
        multigrid_seeded = False
        if initial_phase is not None:
            phase = np.array(canonical_phase(initial_phase, desired.shape), copy=True)
            field = build_field(phase)
        elif (
            iterations is None
            and min(desired.shape) >= 512
            and desired.shape[0] % 4 == 0
            and desired.shape[1] % 4 == 0
            and int(
                np.count_nonzero(desired >= 0.999 * float(np.max(desired)))
            )
            >= 64
        ):
            # Multigrid only where the interior-uniformity gate can stop the
            # polish: a stagnation-governed target (no flat interior) chases
            # the interpolation artifacts of the lifted seed instead of
            # stopping, and measured slower than solving single-grid.
            # Multigrid: converge the same MRAF at quarter resolution (a
            # sixteenth of the work per pass, through this same function),
            # lift the phase through its cosine/sine planes so wrapping
            # survives interpolation, and polish at full resolution.  The
            # result still ends at the full-resolution gates below.
            factor = 4
            coarse_desired = desired.reshape(
                desired.shape[0] // factor,
                factor,
                desired.shape[1] // factor,
                factor,
            ).mean(axis=(1, 3))
            coarse_pupil = pupil.reshape(
                desired.shape[0] // factor,
                factor,
                desired.shape[1] // factor,
                factor,
            ).mean(axis=(1, 3))
            coarse_phase, coarse_metadata = solve_phase(
                coarse_desired,
                pupil_amplitude=coarse_pupil,
                objective_kind="image",
                seed=seed_value,
                stop_requested=stop_requested,
            )
            coarse_iterations = int(coarse_metadata["iterations_run"])
            zoomed_cos = ndimage.zoom(np.cos(coarse_phase), factor, order=1)
            zoomed_sin = ndimage.zoom(np.sin(coarse_phase), factor, order=1)
            seed_phase = np.arctan2(zoomed_sin, zoomed_cos).astype(np.float32)
            image_seed = f"multigrid({coarse_metadata['image_seed']})"
            multigrid_seeded = True
            field = build_field(seed_phase)
        else:
            # Measured across horizons (12..300 iterations) the mapping seed
            # dominates a quadratic defocus seed (pi * (0.75 x^2 + y^2) over
            # the normalised aperture) on both apodized and hard
            # illumination: same iteration budget, better figure of merit
            # and better interior uniformity every time.  Kept as the
            # measurement it is, rather than as a function that reads like a
            # live alternative.
            radial = _radial_transport_seed(desired, pupil)
            if radial is not None:
                image_seed = "radial-transport"
                field = build_field(radial)
            else:
                image_seed = "mapping"
                field = build_field(_mapping_seed(desired, pupil))
        weights = np.array(amplitude_spots, copy=True)
        minimum_gate_iterations = 8 if multigrid_seeded else 24
        best_fom = float("inf")
        previous_fom = float("inf")
        best_field: np.ndarray | None = None
        stagnant_iterations = 0
        # The flat interior the operator asked for: pixels whose desired
        # intensity is within 0.1% of the peak.  Where that region is a real
        # area, its measured 95th/5th percentile ratio reaching 1% is the
        # image analogue of the spots support gate -- a physical stop that
        # does not depend on how fast the merit happens to be moving.
        strong_interior = desired_unshifted >= 0.999 * float(np.max(desired))
        interior_gate_usable = int(np.count_nonzero(strong_interior)) >= 64
        while iterations_run < count:
            if stop_requested is not None and stop_requested():
                raise InterruptedError("SLM phase solve stopped")
            far = fft.fft2(field, norm="ortho")
            # The stop metrics are a tracker, not the update: sampling them
            # every fourth iteration keeps the same best-so-far intent and
            # the same effective stagnation span (three stale CHECKS covers
            # the twelve iterations the per-iteration count required) at a
            # quarter of their full-plane cost.
            if iterations is None and iterations_run % 4 == 0:
                _relative_rms, _roughness, _background, fom = _image_metrics(
                    far, desired_unshifted, support_unshifted, epsilon
                )
                if best_field is None or fom < best_fom:
                    best_fom = fom
                    best_field = np.array(field, copy=True)
                if np.isfinite(previous_fom) and previous_fom - fom < 1e-4:
                    stagnant_iterations += 1
                else:
                    stagnant_iterations = 0
                previous_fom = fom
                interior_uniform = False
                if interior_gate_usable:
                    interior = np.square(
                        np.abs(far[strong_interior]).astype(
                            np.float32, copy=False
                        )
                    )
                    interior_uniform = float(
                        np.percentile(interior, 95)
                        / max(float(np.percentile(interior, 5)), epsilon)
                    ) <= 1.01
                # A target with a real flat interior converges when THAT
                # region is uniform to 1%; merit stagnation alone must not
                # declare success short of it (a slow tail kept improving the
                # interior well after the merit deltas fell under the
                # threshold).  Targets without such a region keep the
                # stagnation criterion; one that never reaches the gate runs
                # to the bounded iteration cap and says so.
                converged = (
                    interior_uniform
                    if interior_gate_usable
                    else stagnant_iterations >= 3
                )
                if (
                    iterations_run >= minimum_gate_iterations
                    and _relative_rms <= 0.005
                    and converged
                ):
                    if not interior_uniform:
                        field = best_field
                    early_stopped = True
                    stop_was_interior = interior_uniform
                    break
            field, weights = mraf_update(far, weights)
            iterations_run += 1

    if method == "wgs-kim":
        result = (
            checked_result
            if checked_result is not None
            else _phase_snapshot(field)
        )
    else:
        result = canonical_phase(fft.fftshift(np.angle(field)), desired.shape)

    if method == "wgs-kim" and transform == "selected-dft":
        if checked_selected is None:
            final_field = np.empty(desired.shape, dtype=np.complex64)
            np.multiply(
                fft.ifftshift(result),
                np.complex64(1j),
                out=final_field,
            )
            np.exp(final_field, out=final_field)
            final_field *= pupil_unshifted
            final_grid = (
                row_forward @ (final_field @ column_forward_transposed)
            )
            final_selected = final_grid[active]
        else:
            final_selected = checked_selected
        final_magnitude = np.abs(final_selected).astype(
            np.float32, copy=False
        )
        total_power = float(np.sum(np.square(pupil, dtype=np.float32)))
    else:
        final_field = pupil_unshifted * np.exp(
            np.complex64(1j) * fft.ifftshift(result)
        ).astype(np.complex64, copy=False)
        final = fft.fft2(final_field, norm="ortho")
        final_magnitude = np.abs(final[support_unshifted]).astype(
            np.float32, copy=False
        )
        total_power = float(np.sum(np.square(np.abs(final), dtype=np.float32)))
    support_ratio = (
        _support_intensity_ratio(
            final_magnitude, desired_unshifted[support_unshifted], epsilon
        )
        if method == "wgs-kim"
        else None
    )
    measured = np.square(final_magnitude, dtype=np.float32)
    measured /= max(float(np.sum(measured)), epsilon)
    expected = desired_unshifted[support_unshifted]
    expected /= float(np.sum(expected))
    error = float(np.sqrt(np.mean((measured - expected) ** 2)))
    efficiency = float(
        np.sum(np.square(final_magnitude, dtype=np.float32))
        / total_power
    )

    new_state: dict[str, object] = {}
    if (
        method == "wgs-kim"
        and transform == "selected-dft"
        and fixed_phase is not None
        and support_yx is not None
    ):
        new_state = {
            "objective_kind": "spots",
            "pupil_source": pupil_source,
            "shape_yx": list(desired.shape),
            "support_yx": support_yx,
            "fixed_farfield_phase": np.angle(fixed_phase).astype(float).tolist(),
            "site_weights": weights.astype(float).tolist(),
            "target_amplitudes": amplitude_spots.astype(float).tolist(),
        }
        if state_requested and saved_state is None:
            state_status = "created"
    elif state_requested and saved_state is None:
        state_status = (
            "not-ready"
            if method == "wgs-kim" and transform == "selected-dft"
            else "not-applicable"
        )
    if spot_optimizer_state is not None:
        spot_optimizer_state.clear()
        spot_optimizer_state.update(new_state)

    metadata = {
        "method": method,
        "objective_kind": resolved_kind,
        "pupil_source": pupil_source,
        "optimizer_state_status": state_status,
        "hot_start_used": hot_start_used,
        "transform": transform,
        "iterations": iterations_run,
        "iterations_run": iterations_run,
        "max_iterations": count,
        "early_stopped": early_stopped,
        "stop_reason": (
            "support-ratio"
            if early_stopped and method == "wgs-kim"
            else "interior-uniformity"
            if early_stopped and stop_was_interior
            else "fom-stagnation"
            if early_stopped
            else "iteration-limit"
        ),
        "seed": seed_value,
        "rms_intensity_error": error,
        "diffraction_efficiency": efficiency,
    }
    if method == "mraf":
        metadata["image_seed"] = image_seed
        metadata["coarse_iterations"] = coarse_iterations
    if support_ratio is not None:
        metadata["support_intensity_ratio"] = support_ratio
        metadata["support_tolerance"] = tolerance
        metadata["minimum_iterations"] = minimum_passes
    else:
        relative_rms, roughness, background, fom = _image_metrics(
            final, desired_unshifted, support_unshifted, epsilon
        )
        metadata.update(
            {
                "signal_pixels": int(np.count_nonzero(support_unshifted)),
                "noise_pixels": int(np.count_nonzero(~support_unshifted)),
                "signal_relative_rms": relative_rms,
                "signal_roughness": roughness,
                "background_power_fraction": background,
                "figure_of_merit": fom,
            }
        )
    return result, metadata

def save_target(
    path: str | Path,
    target: object,
    *,
    objective_kind: str,
) -> Path:
    intensity = validate_target(target)
    return write_readable_json(
        path,
        {
            "format": _TARGET_FORMAT,
            "shape": list(intensity.shape),
            "intensity": intensity.tolist(),
            "objective_kind": _objective_kind(objective_kind),
        },
    )

def load_target(path: str | Path) -> tuple[np.ndarray, str]:
    payload = strict_json_loads(Path(path).read_text(encoding="utf-8"), "target JSON")
    if not isinstance(payload, dict) or set(payload) != _TARGET_KEYS:
        raise ValueError(
            "target JSON fields differ from the strict intensity/objective format"
        )
    if payload["format"] != _TARGET_FORMAT:
        raise ValueError(f"unsupported target JSON format; expected {_TARGET_FORMAT!r}")
    shape = payload["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value < 2 for value in shape)
    ):
        raise ValueError("target JSON shape must be two integer dimensions")
    intensity = payload["intensity"]
    if (
        not isinstance(intensity, list)
        or len(intensity) != shape[0]
        or any(not isinstance(row, list) or len(row) != shape[1] for row in intensity)
        or any(type(value) not in (int, float) for row in intensity for value in row)
    ):
        raise ValueError("target JSON intensity must be a rectangular numeric matrix")
    target = validate_target(intensity)
    if list(target.shape) != shape:
        raise ValueError("target JSON shape differs from intensity")
    return target, _objective_kind(payload["objective_kind"])

def _metadata_json(metadata: Mapping[str, object]) -> str:
    if not isinstance(metadata, Mapping) or any(not isinstance(key, str) for key in metadata):
        raise TypeError("phase metadata must be a string-keyed mapping")
    return json.dumps(dict(metadata), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _json_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{name} must be a string-keyed mapping")
    encoded = _metadata_json(value)
    result = strict_json_loads(encoded, name)
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a JSON object")
    return result


def _system_correction(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    result = _json_object(value, "system correction reference")
    if set(result) != {
        "kind",
        "reference",
        "wavelength_nm",
        "pupil",
        "coordinate_system",
        "valid_region",
        "measurement_method",
    }:
        raise ValueError("system correction reference has the wrong fields")
    if result["kind"] not in _SYSTEM_CORRECTION_KINDS:
        raise ValueError(
            "system correction kind must be pupil_phase_map or target_response_map"
        )
    reference = result["reference"]
    if type(reference) is not str or not reference.strip():
        raise ValueError("system correction reference must be non-empty text")
    wavelength = result["wavelength_nm"]
    if (
        type(wavelength) not in (int, float)
        or not np.isfinite(wavelength)
        or wavelength <= 0
    ):
        raise ValueError("system correction wavelength_nm must be finite and positive")
    for key in ("coordinate_system", "valid_region", "measurement_method"):
        if type(result[key]) is not str or not result[key].strip():
            raise ValueError(f"system correction {key} must be non-empty text")
    return {
        "kind": result["kind"],
        "reference": reference,
        "wavelength_nm": float(wavelength),
        "pupil": _pupil_metadata(result["pupil"]),
        "coordinate_system": result["coordinate_system"],
        "valid_region": result["valid_region"],
        "measurement_method": result["measurement_method"],
    }


def _command_receipt(value: object) -> dict[str, object]:
    result = _json_object(value, "SLM command receipt")
    required = {
        "transport", "identity", "profile", "wavelength_nm", "flip_x",
        "flip_y", "correction_path", "correction_enabled",
        "mapping_revision", "outcome", "command_revision",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"SLM command receipt is missing {sorted(missing)!r}")
    wavelength_policy = {"dvi": True, "usb": True, "virtual": False}
    try:
        requires_wavelength = wavelength_policy[result["transport"]]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "SLM command receipt transport must be dvi, usb, or virtual"
        ) from error
    if result["outcome"] not in {"known-old", "known-new", "unknown"}:
        raise ValueError("SLM command receipt has an invalid outcome")
    for key in ("identity", "profile", "correction_path"):
        if type(result[key]) is not str:
            raise TypeError(f"SLM command receipt {key} must be text")
    for key in ("flip_x", "flip_y", "correction_enabled"):
        if type(result[key]) is not bool:
            raise TypeError(f"SLM command receipt {key} must be bool")
    for key in ("mapping_revision", "command_revision"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"SLM command receipt {key} must be a non-negative int")
    wavelength = result["wavelength_nm"]
    if requires_wavelength and wavelength is None:
        raise ValueError(
            "physical SLM command receipt wavelength_nm must be finite and positive"
        )
    if wavelength is not None and (
        type(wavelength) not in (int, float)
        or not np.isfinite(wavelength)
        or wavelength <= 0
    ):
        raise ValueError("SLM command receipt wavelength_nm is invalid")
    return result


def _pupil_metadata(value: object) -> dict[str, object]:
    result = _json_object(value, "pupil metadata")
    if set(result) != {"enabled", "center_xy", "diameter_xy"}:
        raise ValueError("pupil metadata has the wrong fields")
    if type(result["enabled"]) is not bool:
        raise TypeError("pupil enabled must be bool")
    for key, positive in (("center_xy", False), ("diameter_xy", True)):
        pair = result[key]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(item) not in (int, float) for item in pair)
            or any(not np.isfinite(item) or (positive and item <= 0) for item in pair)
        ):
            raise ValueError(f"pupil {key} must be a finite numeric pair")
    return result


def _operator_metadata(value: object) -> dict[str, object]:
    result = _json_object(value, "operator metadata")
    if set(result) != {
        "enabled", "carrier_waves_xy", "zernike_noll_waves_rms"
    }:
        raise ValueError("operator metadata has the wrong fields")
    carrier = result["carrier_waves_xy"]
    coefficients = result["zernike_noll_waves_rms"]
    if (
        type(result["enabled"]) is not bool
        or not isinstance(carrier, list)
        or len(carrier) != 2
        or any(
            type(item) not in (int, float)
            or not np.isfinite(item)
            or abs(item) > 1000.0
            for item in carrier
        )
        or not isinstance(coefficients, dict)
        or set(coefficients) - _OPERATOR_MODES
        or any(
            type(item) not in (int, float)
            or not np.isfinite(item)
            or abs(item) > 1000.0
            for item in coefficients.values()
        )
    ):
        raise ValueError("operator metadata is invalid")
    return result


def _phase_codes(values: object, shape_yx: tuple[int, int]) -> np.ndarray:
    shape = _pair(shape_yx, "shape_yx")
    source = np.asarray(values)
    if source.shape != shape or source.dtype.kind not in "iuf":
        canonical = canonical_phase(values, shape)
    elif (
        source.dtype == np.dtype("<f4")
        and np.all(np.isfinite(source))
        and np.all(source >= 0.0)
        and np.all(source < 2.0 * np.pi)
    ):
        canonical = source
    else:
        canonical = canonical_phase(source, shape)
    scaled = np.floor(
        canonical * np.float32(_PHASE_CODE_COUNT / (2.0 * np.pi))
        + np.float32(0.5)
    ).astype(np.uint32)
    return np.asarray(scaled & (_PHASE_CODE_COUNT - 1), dtype="<u2")


def freeze_pattern_phase(values: object, shape_yx: tuple[int, int]) -> np.ndarray:
    """Freeze a logical pattern on a uniform 16-bit circular phase grid."""

    shape = _pair(shape_yx, "shape_yx")
    codes = _phase_codes(values, shape)
    return _readonly(
        codes.astype(np.float32)
        * np.float32(2.0 * np.pi / _PHASE_CODE_COUNT)
    )


def _encoded_pattern_phase(values: object, shape: tuple[int, int]) -> np.ndarray:
    codes = _phase_codes(values, shape)
    delta = np.empty(shape, dtype="<u2")
    delta[:, 0] = codes[:, 0]
    np.subtract(codes[:, 1:], codes[:, :-1], out=delta[:, 1:], dtype=np.uint16)
    return delta


def _decoded_pattern_phase(values: object) -> np.ndarray:
    delta = np.asarray(values)
    if delta.dtype != np.dtype("<u2") or delta.ndim != 2 or min(delta.shape) < 2:
        raise ValueError(
            "science context pattern_phase_delta must be a uint16 matrix"
        )
    cumulative = np.add.accumulate(delta, axis=1, dtype=np.uint64)
    codes = np.asarray(cumulative & (_PHASE_CODE_COUNT - 1), dtype="<u2")
    return _readonly(
        codes.astype(np.float32)
        * np.float32(2.0 * np.pi / _PHASE_CODE_COUNT)
    )


def _pupil_geometry(
    shape_yx: tuple[int, int], pupil: Mapping[str, object]
) -> tuple[tuple[int, int], dict[str, object]]:
    shape = _pair(shape_yx, "shape_yx")
    pupil_values = _pupil_metadata(pupil)
    height, width = shape
    center_x, center_y = pupil_values["center_xy"]
    diameter_x, diameter_y = pupil_values["diameter_xy"]
    if not (
        0.0 <= center_x <= width - 1
        and 0.0 <= center_y <= height - 1
        and diameter_x <= 2.0 * width
        and diameter_y <= 2.0 * height
    ):
        raise ValueError("Science Context pupil lies outside SLM limits")
    return shape, pupil_values


def _pupil_coordinates(
    shape_yx: tuple[int, int], pupil: Mapping[str, object]
) -> tuple[tuple[int, int], dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    shape, pupil_values = _pupil_geometry(shape_yx, pupil)
    height, width = shape
    center_x, center_y = pupil_values["center_xy"]
    diameter_x, diameter_y = pupil_values["diameter_xy"]
    yy, xx = np.ogrid[:height, :width]
    zx = (xx - center_x) / (diameter_x / 2.0)
    zy = (yy - center_y) / (diameter_y / 2.0)
    radius_squared = zx * zx + zy * zy
    return shape, pupil_values, zx, zy, radius_squared


def science_pupil_fields(
    shape_yx: tuple[int, int], pupil: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    """Derive the frozen pupil planes from their semantic parameters."""

    shape, pupil_values, _zx, _zy, radius_squared = _pupil_coordinates(
        shape_yx, pupil
    )
    support = np.asarray(radius_squared <= 1.0, dtype=bool)
    amplitude = (
        np.exp(-radius_squared).astype(np.float32)
        if pupil_values["enabled"]
        else np.ones(shape, dtype=np.float32)
    )
    immutable_support = np.frombuffer(
        np.ascontiguousarray(support).tobytes(), dtype=np.bool_
    ).reshape(shape)
    return immutable_support, _readonly(amplitude)


def science_operator_wavefront(
    shape_yx: tuple[int, int],
    pupil: Mapping[str, object],
    operator_metadata: Mapping[str, object],
) -> np.ndarray:
    """Derive the frozen operator plane from pupil and operator parameters."""

    shape, _pupil_values, zx, zy, radius_squared = _pupil_coordinates(
        shape_yx, pupil
    )
    operator = _operator_metadata(operator_metadata)
    support = np.asarray(radius_squared <= 1.0, dtype=bool)
    wavefront = np.zeros(shape, dtype=np.float64)
    if operator["enabled"]:
        height, width = shape
        full_y, full_x = np.ogrid[
            -1.0:1.0:height * 1j, -1.0:1.0:width * 1j
        ]
        carrier_x, carrier_y = operator["carrier_waves_xy"]
        wavefront += np.pi * (carrier_x * full_x + carrier_y * full_y)
        modes = {
            "defocus": lambda: np.sqrt(3.0) * (2.0 * radius_squared - 1.0),
            "astig_oblique": lambda: 2.0 * np.sqrt(6.0) * zx * zy,
            "astig_vertical": lambda: np.sqrt(6.0) * (zx * zx - zy * zy),
            "coma_y": lambda: np.sqrt(8.0) * zy * (
                3.0 * radius_squared - 2.0
            ),
            "coma_x": lambda: np.sqrt(8.0) * zx * (
                3.0 * radius_squared - 2.0
            ),
            "trefoil_y": lambda: np.sqrt(8.0) * zy * (
                3.0 * zx * zx - zy * zy
            ),
            "trefoil_x": lambda: np.sqrt(8.0) * zx * (
                zx * zx - 3.0 * zy * zy
            ),
            "spherical": lambda: np.sqrt(5.0) * (
                6.0 * radius_squared * radius_squared
                - 6.0 * radius_squared
                + 1.0
            ),
        }
        for key, coefficient in operator["zernike_noll_waves_rms"].items():
            if coefficient:
                values = modes[key]()
                wavefront[support] += (
                    2.0 * np.pi * coefficient
                    * np.broadcast_to(values, shape)[support]
                )
    return canonical_phase(wavefront, shape)


def compose_science_phase(
    pattern_phase: object, operator_wavefront: object
) -> np.ndarray:
    pattern = np.asarray(pattern_phase)
    operator = np.asarray(operator_wavefront)
    if pattern.ndim != 2 or operator.shape != pattern.shape:
        raise ValueError("Science Context phase layers differ")
    # Both layers arrive canonical -- a frozen pattern, an operator plane --
    # and canonicalizing a canonical plane returns it unchanged, so the sum
    # is wrapped once rather than each full plane re-validated first.
    return canonical_phase(
        pattern.astype(np.float64) + operator.astype(np.float64),
        tuple(pattern.shape),
    )


def save_science_context(
    path: str | Path,
    pattern_phase: object,
    *,
    target_intensity: object,
    objective_kind: str,
    pupil: Mapping[str, object],
    system_correction: Mapping[str, object] | None,
    command_receipt: Mapping[str, object],
    pattern_metadata: Mapping[str, object],
    operator_metadata: Mapping[str, object],
) -> Path:
    values = np.asarray(pattern_phase)
    if values.ndim != 2:
        raise ValueError("Science Context Pattern must be two-dimensional")
    shape = tuple(values.shape)
    encoded_pattern = _encoded_pattern_phase(values, shape)
    target = validate_target(target_intensity)
    if target.shape != shape:
        raise ValueError("Science Context Target must match the phase shape")
    metadata = {
        "format": _SCIENCE_CONTEXT_FORMAT,
        "objective_kind": _objective_kind(objective_kind),
        "pupil": _pupil_metadata(pupil),
        "system_correction": _system_correction(system_correction),
        "command_receipt": _command_receipt(command_receipt),
        "pattern_metadata": _json_object(pattern_metadata, "Pattern metadata"),
        "operator_metadata": _operator_metadata(operator_metadata),
    }
    _pupil_geometry(shape, metadata["pupil"])
    encoded = _metadata_json(metadata)
    return atomic_write_file(
        path,
        lambda stream: np.savez_compressed(
            stream,
            pattern_phase_delta=encoded_pattern,
            target_intensity=target,
            metadata=np.asarray(encoded),
        ),
    )


def load_science_context(path: str | Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        members = tuple(archive.files)
        if (
            len(members) != len(_SCIENCE_CONTEXT_MEMBERS)
            or set(members) != _SCIENCE_CONTEXT_MEMBERS
        ):
            raise ValueError("science context NPZ has the wrong members")
        encoded_pattern = np.asarray(archive["pattern_phase_delta"])
        target = np.asarray(archive["target_intensity"])
        encoded = np.asarray(archive["metadata"])
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise ValueError("science context metadata must be scalar Unicode JSON")
        metadata = strict_json_loads(str(encoded.item()), "science context metadata")
    if not isinstance(metadata, dict) or set(metadata) != _SCIENCE_CONTEXT_KEYS:
        raise ValueError("science context metadata has the wrong fields")
    if metadata["format"] != _SCIENCE_CONTEXT_FORMAT:
        raise ValueError("unsupported science context format")
    pattern = _decoded_pattern_phase(encoded_pattern)
    shape = tuple(pattern.shape)
    if target.dtype != np.dtype("<f4") or target.shape != shape:
        raise ValueError(
            "science context target intensity must be a matching float32 matrix"
        )
    target = validate_target(target)
    normalized = {
        "objective_kind": _objective_kind(metadata["objective_kind"]),
        "pupil": _pupil_metadata(metadata["pupil"]),
        "system_correction": _system_correction(metadata["system_correction"]),
        "command_receipt": _command_receipt(metadata["command_receipt"]),
        "pattern_metadata": _json_object(
            metadata["pattern_metadata"], "Pattern metadata"
        ),
        "operator_metadata": _operator_metadata(metadata["operator_metadata"]),
    }
    support, amplitude = science_pupil_fields(shape, normalized["pupil"])
    operator = science_operator_wavefront(
        shape, normalized["pupil"], normalized["operator_metadata"]
    )
    phase = compose_science_phase(pattern, operator)
    return {
        "phase": phase,
        "pattern_phase": pattern,
        "operator_wavefront": operator,
        "pupil_amplitude": amplitude,
        "pupil_support": support,
        "target_intensity": target,
        **normalized,
    }


# A Fourier band is a statement about nonzero target coefficients, not a
# smaller SLM. All phase constraints below still run on every physical pixel.
_REARRANGEMENT_PHASE_CODE_CUDA = r'''
__device__ unsigned char phase_code(double command,unsigned int pixel){
 if(command<0)command+=6.2831853071795864769;
 if(command<0)command+=6.2831853071795864769;
 float a=fminf((float)command,6.28318500518798828125f);
 unsigned int p16=((unsigned int)floorf(a*10430.3783504704527f+0.5f))&65535;
 unsigned int hash=pixel+0x9e3779b9u;
 hash^=hash>>16;hash*=0x7feb352du;hash^=hash>>15;hash*=0x846ca68bu;hash^=hash>>16;
 // Exact p16/256 plus the same spatial threshold. A float-radian roundtrip
 // can spuriously turn an exact code 255 into zero at particular pixels.
 unsigned int carry=(((p16&255)<<16)+(hash>>8))>>24;
 return (unsigned char)(((p16>>8)+carry)&255);
}
'''


_REARRANGEMENT_CUDA = _REARRANGEMENT_PHASE_CODE_CUDA + r'''
#include <cuda_fp16.h>
#if HALF
typedef half real_t;
__device__ real_t cv(float a){return __float2half_rn(a);}
#else
typedef float real_t;
__device__ real_t cv(float a){return a;}
#endif
extern "C" __global__ void load_motion_frame(int* counter,const int* native_indices,const int* coarse_indices,
 const int* frequencies,const long long* offsets,const float2* coefficients,int* native_index,int* coarse_index,
 int* frequency,float2* current,int N,int K){
 __shared__ int frame;int t=threadIdx.x;
 if(!t){frame=counter[0];counter[0]=frame+1;}__syncthreads();
 unsigned long long row=(unsigned long long)frame*N;
 for(int j=t;j<N;j+=256){native_index[j]=native_indices[row+j];coarse_index[j]=coarse_indices[row+j];current[j]=coefficients[row+j];}
 long long begin=offsets[frame],end=offsets[frame+1];
 for(int j=t;j<K;j+=256)frequency[j]=j<end-begin?frequencies[begin+j]:0;
}
extern "C" __global__ void store_motion_frame(const int* counter,const unsigned char* codes,
 const float2* coefficients,const float2* actual,unsigned char* movie,float2* saved_coefficients,float2* saved_actual,int N,int area){
 int i=blockIdx.x*blockDim.x+threadIdx.x,frame=counter[0]-1;
 if(i<area)movie[(unsigned long long)frame*area+i]=codes[i];
 if(i<N){unsigned long long row=(unsigned long long)frame*N+i;saved_coefficients[row]=coefficients[i];saved_actual[row]=actual[i];}
}
extern "C" __global__ void scatter(const float2* c,const int* index,float2* spectrum,int N){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N){atomicAdd(&spectrum[index[i]].x,c[i].x);atomicAdd(&spectrum[index[i]].y,c[i].y);}}
extern "C" __global__ void gather(const float2* spectrum,const int* index,float2* c,int N){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)c[i]=spectrum[index[i]];}
extern "C" __global__ void pack_inverse(const float2* spectrum,real_t* packed,int H,int LY,int K){
 __shared__ float2 tile[32][33];
 int y=blockIdx.x*32+threadIdx.x,k=blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(y<H&&k+j<K)tile[threadIdx.y+j][threadIdx.x]=spectrum[(k+j)*LY+(y-H/2+LY)%LY];
 __syncthreads();
 int yy=blockIdx.x*32+threadIdx.y,kk=blockIdx.y*32+threadIdx.x;
 for(int j=0;j<32;j+=8)if(yy+j<H&&kk<K){float2 v=tile[threadIdx.x][threadIdx.y+j];int i=(yy+j)*K+kk;
 packed[i]=cv(v.x);packed[i+H*K]=cv(v.y);}
}
// Roots at opposite centered x coordinates are conjugate even for an asymmetric
// pupil. Reconstruct both physical pixels before applying their own amplitudes.
__device__ __forceinline__ float2 synthesis_pixel(const float* image,int y,int x,int H,int W,int P){
 int d=abs(x-W/2),r=y*2*P+d;float sign=x>=W/2?1.f:-1.f;
 return make_float2(image[r]-sign*image[r+H*2*P+P],image[r+H*2*P]+sign*image[r+P]);
}
__device__ __forceinline__ float2 pupil_projection(float2 field,float amplitude){
 float magnitude=hypotf(field.x,field.y);
 return magnitude>1e-20f?make_float2(field.x*(amplitude/magnitude),field.y*(amplitude/magnitude)):make_float2(amplitude,0);
}
__device__ __forceinline__ void pack_pair(real_t* packed,float2 plus,float2 minus,int y,int d,int H,int P){
 int r=y*2*P+d;
 packed[r]=cv(plus.x+minus.x);packed[r+P]=cv(plus.y-minus.y);
 packed[r+H*2*P]=cv(plus.y+minus.y);packed[r+H*2*P+P]=cv(minus.x-plus.x);
}
extern "C" __global__ void project(const float* image,const float* pupil,real_t* packed,int H,int W,int P){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H*P)return;int d=i%P,y=i/P,xp=W/2+d,xm=W/2-d;
 float2 plus=make_float2(0,0),minus=make_float2(0,0);
 if(xp<W)plus=pupil_projection(synthesis_pixel(image,y,xp,H,W,P),pupil[y*W+xp]);
 if(d>0&&xm>=0)minus=pupil_projection(synthesis_pixel(image,y,xm,H,W,P),pupil[y*W+xm]);
 pack_pair(packed,plus,minus,y,d,H,P);
}
extern "C" __global__ void pack_field(const float2* field,real_t* packed,int H,int W,int P){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H*P)return;int d=i%P,y=i/P,xp=W/2+d,xm=W/2-d;
 float2 plus=xp<W?field[y*W+xp]:make_float2(0,0);
 float2 minus=d>0&&xm>=0?field[y*W+xm]:make_float2(0,0);
 pack_pair(packed,plus,minus,y,d,H,P);
}
extern "C" __global__ void select_roots(const float2* bank,const int* frequencies,real_t* backward,real_t* forward,int NF,int K,int P){
 __shared__ float2 tile[32][33];
 int x=blockIdx.x*32+threadIdx.x,k=blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(x<P&&k+j<K){
  float2 v=k+j<NF?bank[frequencies[k+j]*P+x]:make_float2(0,0);tile[threadIdx.y+j][threadIdx.x]=v;
  if(backward){int r=(k+j)*2*P+x;backward[r]=cv(v.x);backward[r+P]=cv(v.y);}
 }
 __syncthreads();
 int xx=blockIdx.x*32+threadIdx.y,kk=blockIdx.y*32+threadIdx.x;
 for(int j=0;j<32;j+=8)if(xx+j<P&&kk<K){float2 v=tile[threadIdx.x][threadIdx.y+j];
  int out=(xx+j)*K+kk;
  forward[out]=cv(v.x);forward[out+P*K]=cv(v.y);}
}
extern "C" __global__ void pack_forward(const float* projected,float2* spectrum,int H,int LY,int K){
 if(blockIdx.y*32>=H){
  int yy=blockIdx.y*32-H/2+threadIdx.x,k=blockIdx.x*32+threadIdx.y;
  for(int j=0;j<32;j+=8)if(k+j<K&&yy<LY-H/2)spectrum[(k+j)*LY+yy]=make_float2(0,0);
  return;
 }
 __shared__ float2 tile[32][33];
 int k=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(k<K&&y+j<H){int r=(y+j)*K+k;
  tile[threadIdx.y+j][threadIdx.x]=make_float2(projected[r],projected[r+H*K]);}
 __syncthreads();
 int yy=blockIdx.y*32+threadIdx.x,kk=blockIdx.x*32+threadIdx.y;
 // Also clear the partial physical-height tile: odd H has padding here too.
 for(int j=0;j<32;j+=8)if(yy<LY&&kk+j<K)
  spectrum[(kk+j)*LY+(yy-H/2+LY)%LY]=yy<H?tile[threadIdx.x][threadIdx.y+j]:make_float2(0,0);
}
extern "C" __global__ void encode(const float* input,const float* pupil,const float* incident,float2* optical,
 unsigned char* codes,int H,int W,int P){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H*W)return;
 float2 field=P?synthesis_pixel(input,i/W,i%W,H,W,P):((const float2*)input)[i];
 float a=atan2f(field.y,field.x),s,c;
 unsigned char code=phase_code((double)a-(double)incident[i],(unsigned int)i);codes[i]=code;
 sincosf(code*.02454369260617025968f+incident[i],&s,&c);optical[i]=make_float2(pupil[i]*c,pupil[i]*s);
}
__device__ __forceinline__ float weight_ratio(float magnitude,float target,float brightness){
 return fminf(5.f,fmaxf(.2f,brightness*target/fmaxf(magnitude,1e-20f)));
}
extern "C" __global__ void anderson_begin(const float2* c,float2* phase,int* state,int N){
  int t=threadIdx.x;for(int j=t;j<N;j+=256){float2 a=c[j];double m=hypot((double)a.x,(double)a.y);
    phase[j]=m>0?make_float2((float)(a.x/m),(float)(a.y/m)):make_float2(1,0);}
  if(!t){state[0]=0;state[1]=0;}}
extern "C" __global__ void anderson_update(const float2* field,const float* target,float2* c,const float2* phase,
  float* gh,float* rh,double* candidate,int* state,int N,float exponent){
  __shared__ double sums[5][256],brightness,mx,mg,gamma0,gamma1,largest,normalizer;
  __shared__ int bad;
  int t=threadIdx.x,slot=state[1],count=state[0],prev=(slot+2)%3,older=(slot+1)%3;
  double se=0,sa=0;
  for(int j=t;j<N;j+=256){float2 e=field[j];se+=(double)e.x*e.x+(double)e.y*e.y;sa+=(double)target[j]*target[j];}
  sums[0][t]=se;sums[1][t]=sa;__syncthreads();
  for(int k=128;k;k/=2){if(t<k){sums[0][t]+=sums[0][t+k];sums[1][t]+=sums[1][t+k];}__syncthreads();}
  if(!t){brightness=sqrt(sums[0][0]/fmax(sums[1][0],1e-30));bad=0;}__syncthreads();
  double sx=0,sg=0;
  for(int j=t;j<N;j+=256){float2 a=c[j],e=field[j];
    float x=logf(fmaxf(hypotf(a.x,a.y),1e-20f));
    float gain=exponent*logf(weight_ratio(hypotf(e.x,e.y),target[j],(float)brightness));
    sx+=x;sg+=gain;}
  sums[0][t]=sx;sums[1][t]=sg;__syncthreads();
  for(int k=128;k;k/=2){if(t<k){sums[0][t]+=sums[0][t+k];sums[1][t]+=sums[1][t+k];}__syncthreads();}
  if(!t){mx=sums[0][0]/N;mg=sums[1][0]/N;}__syncthreads();
  for(int j=t;j<N;j+=256){float2 a=c[j],e=field[j];
    double x=(double)logf(fmaxf(hypotf(a.x,a.y),1e-20f))-mx;
    double f=(double)(exponent*logf(weight_ratio(hypotf(e.x,e.y),target[j],(float)brightness)))-mg;
    gh[slot*N+j]=(float)(x+f);rh[slot*N+j]=(float)f;}
  __syncthreads();
  double h00=0,h01=0,h11=0,b0=0,b1=0;
  if(count)for(int j=t;j<N;j+=256){double f=rh[slot*N+j],d0=f-rh[prev*N+j];
    double d1=count>1?(double)rh[prev*N+j]-rh[older*N+j]:0;
    h00+=d0*d0;h01+=d0*d1;h11+=d1*d1;b0+=d0*f;b1+=d1*f;}
  sums[0][t]=h00;sums[1][t]=h01;sums[2][t]=h11;sums[3][t]=b0;sums[4][t]=b1;__syncthreads();
  for(int k=128;k;k/=2){if(t<k)for(int q=0;q<5;++q)sums[q][t]+=sums[q][t+k];__syncthreads();}
  if(!t){
    gamma0=0;gamma1=0;
    if(count){double a=sums[0][0]*1.0001,b=sums[1][0],d=sums[2][0]*1.0001;
      if(!isfinite(a)||!isfinite(b)||!isfinite(d)||!isfinite(sums[3][0])||!isfinite(sums[4][0]))bad=1;
      else if(a>0&&d>0){double determinant=a*d-b*b;
        if(determinant>1e-12*a*d&&isfinite(determinant)){
          gamma0=(d*sums[3][0]-b*sums[4][0])/determinant;
          gamma1=(a*sums[4][0]-b*sums[3][0])/determinant;
        }else bad=1;
      }else if(a>0){gamma0=sums[3][0]/a;}
      else if(d>0){gamma1=sums[4][0]/d;}
      else bad=1;
      if(!isfinite(gamma0)||!isfinite(gamma1))bad=1;
    }
    if(bad){gamma0=0;gamma1=0;}
  }__syncthreads();
  for(int j=t;j<N;j+=256){double g=gh[slot*N+j],v=g;
    if(count)v-=gamma0*(g-gh[prev*N+j]);
    if(count>1)v-=gamma1*((double)gh[prev*N+j]-gh[older*N+j]);
    candidate[j]=v;if(!isfinite(v))atomicExch(&bad,1);}
  __syncthreads();
  double largest_local=-1.0e300;
  for(int j=t;j<N;j+=256){if(bad)candidate[j]=gh[slot*N+j];largest_local=fmax(largest_local,candidate[j]);}
  sums[0][t]=largest_local;__syncthreads();
  for(int k=128;k;k/=2){if(t<k)sums[0][t]=fmax(sums[0][t],sums[0][t+k]);__syncthreads();}
  if(!t)largest=sums[0][0];__syncthreads();
  double norm=0;for(int j=t;j<N;j+=256){double v=exp(candidate[j]-largest);candidate[j]=v;norm+=v*v;}
  sums[0][t]=norm;__syncthreads();
  for(int k=128;k;k/=2){if(t<k)sums[0][t]+=sums[0][t+k];__syncthreads();}
  if(!t)normalizer=rsqrt(sums[0][0]);__syncthreads();
  for(int j=t;j<N;j+=256){double a=candidate[j]*normalizer;float2 p=phase[j];c[j]=make_float2((float)(a*p.x),(float)(a*p.y));}
  if(!t){state[0]=bad?1:min(count+1,2);state[1]=(slot+1)%3;}
}
extern "C" __global__ void compact_delta(const float2* field,const int* positions,const float* target,float2* delta,int N){
 __shared__ float numerator[256],denominator[256];int t=threadIdx.x;float a=0,b=0;
 for(int j=t;j<N;j+=256){float2 e=field[positions[j]];float v=target[j];a+=v*hypotf(e.x,e.y);b+=v*v;}
 numerator[t]=a;denominator[t]=b;__syncthreads();
 for(int k=128;k;k/=2){if(t<k){numerator[t]+=numerator[t+k];denominator[t]+=denominator[t+k];}__syncthreads();}
 float scale=numerator[0]/fmaxf(denominator[0],1e-30f);
 for(int j=t;j<N;j+=256){int p=positions[j];float2 e=field[p];float m=hypotf(e.x,e.y),v=scale*target[j];
  delta[p]=m>0?make_float2(v*e.x/m-e.x,v*e.y/m-e.y):make_float2(v,0);}
}
extern "C" __global__ void compact_project(float2* field,const float2* correction,const float* pupil,float2* unit,int M){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M)return;
 float re=field[i].x+correction[i].x/M,im=field[i].y+correction[i].y/M,m=hypotf(re,im);
 float2 u=m>0?make_float2(re/m,im/m):make_float2(1,0);unit[i]=u;field[i]=make_float2(pupil[i]*u.x,pupil[i]*u.y);
}
extern "C" __global__ void fold_native(const float2* native,float2* small,int H,int W,int FH,int FW){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FH*FW)return;int x=i%FW,y=i/FW;float re=0,im=0;
 for(int ty=0;ty<H/FH;++ty)for(int tx=0;tx<W/FW;++tx){float2 a=native[(y+ty*FH)*W+x+tx*FW];re+=a.x;im+=a.y;}
 small[i]=make_float2(re,im);
}
extern "C" __global__ void source_encode(const float2* unit,const float2* carrier,const float* pupil,const float* incident,
 unsigned char* codes,float2* demod,int H,int W,int FH,int FW){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H*W)return;int x=i%W,y=i/W;
 float2 a=unit[(y%FH)*FW+x%FW],b=carrier[i];float re=a.x*b.x-a.y*b.y,im=a.x*b.y+a.y*b.x;
 int pixel=((y+H/2)%H)*W+(x+W/2)%W;
 unsigned char code=phase_code((double)atan2f(im,re)-(double)incident[pixel],(unsigned int)pixel);codes[pixel]=code;
 float s,c;sincosf(code*.02454369260617025968f+incident[pixel],&s,&c);
 demod[i]=make_float2(pupil[i]*(c*b.x+s*b.y),pupil[i]*(s*b.x-c*b.y));
}
extern "C" __global__ void clearance(const int* p,float* distances,int F,int N){
 int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y*blockDim.y+threadIdx.y;if(i>=N||j>=i)return;
 float best=1e30f;
 for(int f=0;f<F;++f){int a=(f*N+i)*2,b=(f*N+j)*2;float y=p[a]-p[b],x=p[a+1]-p[b+1],yy=0,xx=0;
  if(f+1<F){yy=p[a+N*2]-p[b+N*2]-y;xx=p[a+N*2+1]-p[b+N*2+1]-x;}
  float v=yy*yy+xx*xx,t=v>0?fminf(1,fmaxf(0,-(y*yy+x*xx)/v)):0;y+=t*yy;x+=t*xx;best=fminf(best,y*y+x*x);}
 distances[i*N+j]=best;
}
'''


def _prepare_rearrangement_gpu(geometry, pupil_amplitude, pupil_phase):
    """Prepare the one native Fourier model and optional sampling quadrature."""
    try:
        import cupy as cp  # noqa: PLC0415
        from cupy.cuda import cublas, cufft  # noqa: PLC0415
        from cuda.pathfinder import load_nvidia_dynamic_lib  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError('SLM GPU calculation requires pip install "zou-lab-control[slm-gpu]"') from error
    import ctypes  # noqa: PLC0415

    shape = geometry["shape_yx"]
    pupil = np.asarray(pupil_amplitude, np.float32)
    if (pupil.shape != shape or not np.all(np.isfinite(pupil))
            or np.any(pupil < 0) or not np.any(pupil > 0)):
        raise ValueError("pupil_amplitude must be finite, nonnegative and match the full SLM")
    incident = np.zeros(shape, np.float32) if pupil_phase is None else canonical_phase(pupil_phase, shape)
    stream = cp.cuda.Stream(non_blocking=True)
    library = (ctypes.WinDLL if os.name == "nt" else ctypes.CDLL)(load_nvidia_dynamic_lib("cublas").abs_path)
    gemm = library.cublasGemmEx
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    gemm.argtypes = [pointer, integer, integer, integer, integer, integer, pointer,
                    pointer, integer, integer, pointer, integer, integer,
                    pointer, pointer, integer, integer, integer, integer]
    gemm.restype = integer
    number = len(geometry["target_yx"])
    signed = 3 * (np.concatenate((geometry["source_yx"], geometry["target_yx"]))[:, 1] - shape[1] // 2)
    low, high = int(signed.min()), int(signed.max())
    capacity = (min(number, high - low + 1) + 15) // 16 * 16
    module = cp.RawModule(code="#define HALF 1\n" + _REARRANGEMENT_CUDA)
    names = ("load_motion_frame", "store_motion_frame", "scatter", "gather", "pack_inverse", "project", "pack_field", "select_roots",
             "pack_forward", "encode", "anderson_begin", "anderson_update", "compact_delta", "compact_project",
             "fold_native", "source_encode", "clearance")
    kernels = {name: module.get_function(name) for name in names}
    gpu = dict(cp=cp, cublas=cublas, cufft=cufft, stream=stream, module=module, kernels=kernels,
               library=library, gemm=gemm, shape=shape, number=number, low=low, high=high,
               pupil_cpu=pupil.copy(), incident_cpu=np.asarray(incident, np.float32).copy(),
               pupil_energy=float(np.sum(pupil.astype(np.float64) ** 2)), pupil_scale=float(np.max(pupil)),
               resources={}, output_pool=cp.cuda.PinnedMemoryPool(), weight_exponent=np.float32(.8))
    # Stride-two samples share the centered native coordinates only at these sizes.
    factors = (1, 2) if all(size % 4 == 0 for size in shape) else (1,)
    source, target = geometry["source_yx"], geometry["target_yx"]
    distance = int(np.max(np.maximum(source.max(axis=0) - target.min(axis=0),
                                     target.max(axis=0) - source.min(axis=0))))
    steps = max(1, int(np.ceil(min(max(geometry["matching_radii"]), distance) / geometry["max_step"])))
    gpu["motion_capacity"] = 3 * steps
    handles = []
    def close():
        stream.synchronize()
        for handle in handles:
            cublas.destroy(handle)
        handles.clear()
        if gpu:
            gpu["output_pool"].free_all_blocks()
        gpu.clear()
    gpu["close"] = close
    try:
        with stream:
            gpu["coefficients"] = cp.zeros(number, cp.complex64)
            gpu["frequencies"] = cp.zeros(capacity, cp.int32)
            for factor, precise in [(factor, False) for factor in factors] + [(1, True)]:
                dtype = cp.float32 if precise else cp.float16
                work_module = cp.RawModule(code="#define HALF 0\n" + _REARRANGEMENT_CUDA) if precise else module
                work_kernels = {name: work_module.get_function(name) for name in names} if precise else kernels
                h, w = (size // factor for size in shape)
                padded, ly = (w // 2 + 16) // 16 * 16, 3 * h
                if not precise:
                    roots = np.exp(2j * np.pi * np.arange(low, high + 1)[:, None]
                                   * np.arange(padded)[None] / (3 * w)).astype(np.complex64)
                    roots[:, w // 2 + 1:] = 0
                handle = cublas.create()
                handles.append(handle)
                cublas.setStream(handle, stream.ptr)
                work = dict(shape=(h, w), padded=padded, ly=ly, handle=handle,
                            typecode=0 if precise else 2, module=work_module, kernels=work_kernels,
                            bank=gpu["resources"][1]["bank"] if precise else cp.asarray(roots),
                            forward=cp.empty((2 * padded, capacity), dtype),
                            spectrum=cp.empty((capacity, ly), cp.complex64),
                            transformed=cp.empty((capacity, ly), cp.complex64),
                            field_gemm=cp.empty((2, h, 2 * padded), dtype),
                            projected=cp.empty((2, h, capacity), cp.float32),
                            actual=cp.empty(number, cp.complex64),
                            index=(gpu["resources"][1]["index"] if precise
                                   else cp.arange(number, dtype=cp.int32) % (16 * ly)),
                            frequencies=gpu["frequencies"],
                            alpha=np.asarray(1, np.float32), beta=np.asarray(0, np.float32),
                            plans={band: cufft.Plan1d(ly, cufft.CUFFT_C2C, band)
                                   for band in range(16, capacity + 1, 16)})
                if precise:
                    # The final decision uses FP32 roots/packing on the same finite
                    # Fourier map. Roots, emitted optical field and query indices
                    # belong to the same native geometry; only packing is precise.
                    work["optical"] = gpu["resources"][1]["optical"]
                    gpu["measurement"] = work
                else:
                    work.update(backward=cp.empty((capacity, 2 * padded), dtype),
                                packed=cp.empty((2, h, capacity), dtype),
                                image=cp.empty((2, h, 2 * padded), cp.float32),
                                pupil=cp.asarray(pupil[::factor, ::factor]
                                                 / np.float32(gpu["pupil_scale"]) * factor ** 2))
                    if factor == 1:
                        work.update(capacity=capacity, optical=cp.empty((h, w), cp.complex64),
                                    codes=cp.empty((h, w), cp.uint8),
                                    physical_pupil=cp.asarray(pupil),
                                    incident=cp.asarray(incident, cp.float32))
                    gpu["resources"][factor] = work
                for band in work["plans"]:
                    _rearrangement_select_roots(work, band)
                    if precise:
                        _rearrangement_propagate(gpu, work, band, "forward")
                    else:
                        _rearrangement_propagate(gpu, work, band, "roundtrip")
                        if factor == 1:
                            _rearrangement_propagate(gpu, work, band, "encode")
            gpu["aa_phase"] = cp.empty(number, cp.complex64)
            gpu["aa_g"] = cp.empty((3, number), cp.float32)
            gpu["aa_r"] = cp.empty((3, number), cp.float32)
            gpu["aa_candidate"] = cp.empty(number, cp.float64)
            gpu["aa_state"] = cp.zeros(2, cp.int32)
            motion_count = gpu["motion_capacity"]
            gpu["motion_codes"] = cp.empty((motion_count, *shape), cp.uint8)
            gpu["motion_coefficients"] = cp.zeros((motion_count, number), cp.complex64)
            gpu["motion_actual"] = cp.empty((motion_count, number), cp.complex64)
            gpu["motion_indices"] = {factor: cp.zeros((motion_count, number), cp.int32) for factor in factors}
            gpu["motion_frequencies"] = cp.zeros(motion_count * capacity, cp.int32)
            gpu["motion_offsets"] = cp.zeros(motion_count + 1, cp.int64)
            gpu["frame_index"] = cp.zeros(1, cp.int32)
            gpu["distances"] = cp.full((number, number), cp.inf, cp.float32)
            trial = cp.asarray(geometry["target_yx"][None])
            kernels["clearance"](((number + 15) // 16,) * 2, (16, 16),
                                 (trial, gpu["distances"], np.int32(1), np.int32(number)))
            float(cp.sqrt(cp.min(gpu["distances"])))
            stream.synchronize()
    except BaseException:
        close()
        raise
    return gpu


def _rearrangement_select_roots(work, band):
    """Pack this frame's fixed frequencies once in this workspace's precision."""
    padded = work["padded"]
    work["kernels"]["select_roots"](((padded + 31) // 32, (band + 31) // 32), (32, 8),
                                   (work["bank"], work["frequencies"], work.get("backward", np.uint64(0)), work["forward"],
                                    np.int32(band), np.int32(band), np.int32(padded)))


def _rearrangement_propagate(gpu, work, band, operation):
    """Centered finite Fourier propagation with the frame's already packed roots."""
    h, w = work["shape"]
    padded, ly, number = work["padded"], work["ly"], gpu["number"]
    kernels, cb = work["kernels"], gpu["cublas"]
    typecode = work["typecode"]
    alpha, beta = work["alpha"].ctypes.data, work["beta"].ctypes.data
    if operation != "forward":
        work["spectrum"][:band].fill(0)
        kernels["scatter"](((number + 255) // 256,), (256,),
                           (gpu["coefficients"], work["index"], work["spectrum"], np.int32(number)))
        work["plans"][band].fft(work["spectrum"][:band], work["transformed"][:band], gpu["cufft"].CUFFT_INVERSE)
        kernels["pack_inverse"](((h + 31) // 32, (band + 31) // 32), (32, 8),
                                (work["transformed"], work["packed"], np.int32(h), np.int32(ly), np.int32(band)))
        status = gpu["gemm"](work["handle"], 0, 0, 2 * padded, 2 * h, band, alpha,
                             work["backward"].data.ptr, typecode, 2 * padded,
                             work["packed"].data.ptr, typecode, band, beta,
                             work["image"].data.ptr, 0, 2 * padded,
                             cb.CUBLAS_COMPUTE_32F, cb.CUBLAS_GEMM_DEFAULT)
        if status:
            raise RuntimeError(f"SLM cuBLAS synthesis failed ({status})")
        if operation == "encode":
            kernels["encode"](((h * w + 255) // 256,), (256,),
                              (work["image"], work["physical_pupil"], work["incident"], work["optical"],
                               work["codes"], *map(np.int32, (h, w, padded))))
            return
        kernels["project"](((h * padded + 255) // 256,), (256,),
                           (work["image"], work["pupil"], work["field_gemm"],
                            np.int32(h), np.int32(w), np.int32(padded)))
    if operation == "forward":
        kernels["pack_field"](((h * padded + 255) // 256,), (256,),
                              (work["optical"], work["field_gemm"], np.int32(h), np.int32(w), np.int32(padded)))
    status = gpu["gemm"](work["handle"], 0, 0, band, 2 * h, 2 * padded, alpha,
                         work["forward"].data.ptr, typecode, band,
                         work["field_gemm"].data.ptr, typecode, 2 * padded, beta,
                         work["projected"].data.ptr, 0, band,
                         cb.CUBLAS_COMPUTE_32F, cb.CUBLAS_GEMM_DEFAULT)
    if status:
        raise RuntimeError(f"SLM cuBLAS analysis failed ({status})")
    kernels["pack_forward"](((band + 31) // 32, (ly + 31) // 32), (32, 8),
                            (work["projected"], work["spectrum"], np.int32(h), np.int32(ly), np.int32(band)))
    work["plans"][band].fft(work["spectrum"][:band], work["transformed"][:band], gpu["cufft"].CUFFT_FORWARD)
    kernels["gather"](((number + 255) // 256,), (256,),
                      (work["transformed"], work["index"], work["actual"], np.int32(number)))


def _rearrangement_amplitude_updates(gpu, work, band, count):
    """Depth-two amplitude updates, reset for this frame and resolution."""
    _rearrangement_select_roots(work, band)
    if not count:
        return
    number = np.int32(gpu["number"])
    gpu["kernels"]["anderson_begin"]((1,), (256,),
                                     (gpu["coefficients"], gpu["aa_phase"], gpu["aa_state"], number))
    for _ in range(count):
        _rearrangement_propagate(gpu, work, band, "roundtrip")
        gpu["kernels"]["anderson_update"]((1,), (256,),
            (work["actual"], gpu["amplitude"], gpu["coefficients"], gpu["aa_phase"],
             gpu["aa_g"], gpu["aa_r"], gpu["aa_candidate"], gpu["aa_state"], number, gpu["weight_exponent"]))


def _rearrangement_load_frame(gpu):
    """Stage one row; its incremented counter stays fixed until the next load."""
    native = gpu["resources"][1]
    coarse = gpu["resources"].get(2, native)
    gpu["kernels"]["load_motion_frame"]((1,), (256,),
        (gpu["frame_index"], gpu["motion_indices"][1], gpu["motion_indices"].get(2, gpu["motion_indices"][1]),
         gpu["motion_frequencies"], gpu["motion_offsets"], gpu["motion_coefficients"],
         native["index"], coarse["index"], gpu["frequencies"], gpu["coefficients"],
         np.int32(gpu["number"]), np.int32(native["capacity"])))


def _rearrangement_store_frame(gpu):
    """Copy accepted frame outputs; no block changes the row counter here."""
    area = int(np.prod(gpu["shape"]))
    gpu["kernels"]["store_motion_frame"](((area + 255) // 256,), (256,),
        (gpu["frame_index"], gpu["resources"][1]["codes"], gpu["coefficients"], gpu["measurement"]["actual"],
         gpu["motion_codes"], gpu["motion_coefficients"], gpu["motion_actual"], np.int32(gpu["number"]), np.int32(area)))


def _rearrangement_bind(gpu, points):
    """Build bulk trajectory metadata; the caller retains host inputs through transfer."""
    shape = np.asarray(gpu["shape"])
    scaled = 3 * (np.asarray(points, np.float64) - shape // 2)
    frequencies = np.rint(scaled).astype(np.int64)
    if np.max(abs(scaled - frequencies)) > 1e-8:
        raise ValueError("emitted sites must lie on the native third-bin grid")
    columns = frequencies[..., 1] - gpu["low"]
    present = np.zeros((len(points), gpu["high"] - gpu["low"] + 1), bool)
    present[np.arange(len(points))[:, None], columns] = True
    lookup = np.cumsum(present, axis=1, dtype=np.int32) - 1
    counts = present.sum(axis=1, dtype=np.int32)
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    selected = np.nonzero(present)[1].astype(np.int32)
    indices = {}
    for factor, work in gpu["resources"].items():
        packed = lookup[np.arange(len(points))[:, None], columns] * work["ly"] + frequencies[..., 0] % work["ly"]
        indices[factor] = packed.astype(np.int32)
    return indices, selected, counts, offsets


def _prepare_rearrangement_lattice(gpu, source_points, initial_codes):
    """Fold the actual encoded source field and the complete physical pupil."""
    from cupyx.scipy.fft import get_fft_plan  # noqa: PLC0415

    cp, shape = gpu["cp"], gpu["shape"]
    h, w = shape
    signed = np.asarray(source_points, np.int64) - np.asarray(shape) // 2
    tiles = tuple(int(np.gcd.reduce(np.r_[size, signed[:, axis] - signed[0, axis]]))
                  for axis, size in enumerate(shape))
    fh, fw = (size // tile for size, tile in zip(shape, tiles))
    residue = signed[0] % tiles
    modes = (signed - residue) // np.asarray(tiles)
    gy, gx = tiles
    yy, xx = np.ogrid[:h, :w]
    carrier = np.exp(2j * np.pi * (residue[0] * yy / h + residue[1] * xx / w)).astype(np.complex64)
    pupil = np.fft.ifftshift(gpu["pupil_cpu"])
    effective = pupil.astype(np.float64).reshape(gy, fh, gx, fw).sum(axis=(0, 2)).astype(np.float32)
    optical = gpu["pupil_cpu"] * np.exp(1j * (initial_codes.astype(np.float64) * (2 * np.pi / 256)
                                             + gpu["incident_cpu"]))
    initial = np.fft.ifftshift(optical) * carrier.conj()
    initial = initial.reshape(gy, fh, gx, fw).sum(axis=(0, 2)).astype(np.complex64)
    small = cp.asarray(initial)
    state = dict(shape=shape, fundamental=(fh, fw), carrier=cp.asarray(carrier), pupil=cp.asarray(pupil),
                 effective=cp.asarray(effective), incident=cp.asarray(gpu["incident_cpu"]),
                 initial=small.copy(), small=small, unit=cp.empty_like(small),
                 spectrum=cp.empty_like(small), delta=cp.empty_like(small), correction=cp.empty_like(small),
                 native=cp.empty(shape, cp.complex64), codes=cp.empty(shape, cp.uint8),
                 positions=cp.asarray(((modes[:, 0] % fh) * fw + modes[:, 1] % fw).astype(np.int32)),
                 target=cp.ones(len(source_points), cp.float32), actual=cp.empty(len(source_points), cp.complex64),
                 plan=get_fft_plan(small, axes=(-2, -1)))
    gpu["source"] = state
    _rearrangement_lattice_correct(gpu, 64)
    gpu["stream"].synchronize()
    gpu["stream"].begin_capture()
    _rearrangement_lattice_correct(gpu, 64)
    state["graph"] = gpu["stream"].end_capture()
    state["small"][:] = state["initial"]


def _rearrangement_lattice_correct(gpu, iterations):
    """Free-phase amplitude/dark projection, then native encoding and analysis."""
    cp, state, kernels = gpu["cp"], gpu["source"], gpu["kernels"]
    h, w = state["shape"]
    fh, fw = state["fundamental"]
    size, number = fh * fw, len(state["target"])
    compact_grid = ((size + 255) // 256,)
    for _ in range(iterations):
        state["plan"].fft(state["small"], state["spectrum"], gpu["cufft"].CUFFT_FORWARD)
        state["delta"].fill(0)
        kernels["compact_delta"]((1,), (256,), (state["spectrum"], state["positions"], state["target"],
                                               state["delta"], np.int32(number)))
        state["plan"].fft(state["delta"], state["correction"], gpu["cufft"].CUFFT_INVERSE)
        kernels["compact_project"](compact_grid, (256,), (state["small"], state["correction"], state["effective"],
                                                          state["unit"], np.int32(size)))
    if iterations == 0:
        state["correction"].fill(0)
        kernels["compact_project"](compact_grid, (256,), (state["small"], state["correction"], state["effective"],
                                                          state["unit"], np.int32(size)))
    kernels["source_encode"](((h * w + 255) // 256,), (256,),
                              (state["unit"], state["carrier"], state["pupil"], state["incident"],
                               state["codes"], state["native"], *map(np.int32, (h, w, fh, fw))))
    kernels["fold_native"](compact_grid, (256,),
                           (state["native"], state["small"], *map(np.int32, (h, w, fh, fw))))
    state["plan"].fft(state["small"], state["spectrum"], gpu["cufft"].CUFFT_FORWARD)
    cp.take(state["spectrum"], state["positions"], out=state["actual"])


def _rearrangement_endpoint(cp, points, shape, pupil, intensity, iterations, seed):
    """Prepare a fixed endpoint, retaining the coefficients that reproduce it."""
    height, width = shape
    iy, ix = cp.asarray((points - np.asarray(shape) // 2).T % np.asarray(shape)[:, None])
    amplitude = cp.sqrt(cp.asarray(intensity, dtype=cp.float32))
    amplitude /= cp.linalg.norm(amplitude)
    phase = cp.asarray(np.random.default_rng(seed).uniform(-np.pi, np.pi, len(points)).astype(np.float32))
    weight = amplitude.copy()
    spectrum = cp.zeros(shape, cp.complex64)
    illumination = cp.fft.ifftshift(cp.asarray(pupil))
    coefficient = weight * cp.exp(cp.complex64(1j) * phase)
    for iteration in range(iterations):
        spectrum.fill(0)
        spectrum[iy, ix] = coefficient
        back = cp.fft.ifft2(spectrum)
        field = illumination * cp.exp(cp.complex64(1j) * cp.angle(back))
        actual = cp.fft.fft2(field)[iy, ix]
        magnitude = cp.abs(actual)
        measured = magnitude / cp.maximum(cp.linalg.norm(magnitude), 1e-20)
        weight *= cp.clip(amplitude / cp.maximum(measured, 1e-20), .2, 5) ** cp.float32(.8)
        weight /= cp.linalg.norm(weight)
        if iteration < min(50, iterations // 2):
            phase = cp.angle(actual)
        coefficient = weight * cp.exp(cp.complex64(1j) * phase)
    spectrum.fill(0)
    spectrum[iy, ix] = coefficient
    pattern = cp.remainder(cp.angle(cp.fft.fftshift(cp.fft.ifft2(spectrum))), cp.float32(2 * np.pi))
    return cp.asnumpy(coefficient), cp.asnumpy(pattern)


def prepare_rearrangement(
    source_yx: object, target_yx: object, *, shape_yx: tuple[int, int],
    pupil_amplitude: object, matching_radii: object, minimum_separation: float,
    pupil_phase: object | None = None, source_intensities: object | None = None,
    target_intensities: object | None = None,
    endpoint_iterations: int = 150, seed: int = 0,
    endpoint_data: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Prepare the calibrated optical working point before observing occupancy.

    Apply initial_phase_codes before measuring occupancy in source order. The
    returned workspace is caller-owned and serial-use only. Changing either
    array, pupil or incident aberration requires preparing it again.
    endpoint_data may supply source/target_phase_codes and _coefficients for
    fixed calibrated endpoints. Their actual encoded fields are measured here;
    requested intensity weights remain authoritative, with one physical RMS
    brightness scale per endpoint. Supplied codes must meet the 1.01 support
    ratio. Short endpoint optimization is allowed, but an inaccurate generated
    target is not reused as a solved final frame.
    The caller closes prepared["close"]() after its last serial solve.
    No occupancy-dependent answer is prepared.
    """
    geometry = prepare_rearrangement_geometry(
        source_yx, target_yx, shape_yx=shape_yx, matching_radii=matching_radii,
        minimum_separation=minimum_separation,
    )
    shape = geometry["shape_yx"]
    if (isinstance(endpoint_iterations, bool) or int(endpoint_iterations) != endpoint_iterations
            or endpoint_iterations < 1):
        raise ValueError("endpoint_iterations must be a positive integer")
    intensities = []
    for name, values, points in (("source", source_intensities, geometry["source_yx"]),
                                 ("target", target_intensities, geometry["target_yx"])):
        value = np.ones(len(points), np.float32) if values is None else np.asarray(values, np.float32)
        if value.shape != (len(points),) or not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ValueError(f"{name}_intensities must be finite and positive, one per site")
        intensities.append(_readonly(value))
    if endpoint_data is not None:
        calibration = dict(source_yx=geometry["source_yx"], target_yx=geometry["target_yx"],
                           pupil_amplitude=np.asarray(pupil_amplitude),
                           pupil_phase=np.zeros(shape, np.float32) if pupil_phase is None else np.asarray(pupil_phase),
                           source_intensities=intensities[0], target_intensities=intensities[1])
        for name, actual in calibration.items():
            if name in endpoint_data and not np.array_equal(actual, endpoint_data[name]):
                raise ValueError(f"endpoint_data {name} does not match the prepared working point")
        for name, points in (("source", geometry["source_yx"]), ("target", geometry["target_yx"])):
            for suffix in ("phase_codes", "coefficients"):
                if f"{name}_{suffix}" not in endpoint_data:
                    raise ValueError(f"endpoint_data requires {name}_{suffix}")
            codes = np.asarray(endpoint_data[f"{name}_phase_codes"])
            coefficient = np.asarray(endpoint_data[f"{name}_coefficients"])
            if codes.dtype != np.uint8 or codes.shape != shape:
                raise ValueError(f"endpoint_data {name}_phase_codes must be a native uint8 raster")
            if (coefficient.shape != (len(points),) or not np.all(np.isfinite(coefficient))
                    or np.any(abs(coefficient) == 0)):
                raise ValueError(f"endpoint_data {name}_coefficients must be finite and nonzero, one per site")
    gpu = _prepare_rearrangement_gpu(geometry, pupil_amplitude, pupil_phase)
    try:
        cp, stream = gpu["cp"], gpu["stream"]
        native = gpu["resources"][1]
        coefficients, codes_list, fields, brightness, endpoint_ratios = [], [], [], [], []
        with stream:
            for index, (name, points, authored) in enumerate((
                    ("source", geometry["source_yx"], intensities[0]),
                    ("target", geometry["target_yx"], intensities[1]))):
                if endpoint_data is None:
                    coefficient, pattern = _rearrangement_endpoint(
                        cp, points, shape, gpu["pupil_cpu"], authored, int(endpoint_iterations), int(seed) + index,
                    )
                    latent = cp.exp(cp.complex64(1j) * cp.asarray(pattern, cp.float32))
                    gpu["kernels"]["encode"](((int(np.prod(shape)) + 255) // 256,), (256,),
                                             (latent, native["physical_pupil"], native["incident"], native["optical"],
                                              native["codes"], *map(np.int32, (*shape, 0))))
                    codes = native["codes"].get()
                else:
                    coefficient = np.asarray(endpoint_data[f"{name}_coefficients"], np.complex64).copy()
                    codes = np.asarray(endpoint_data[f"{name}_phase_codes"]).copy()
                optical = gpu["pupil_cpu"].astype(np.float64) * np.exp(
                    1j * (codes.astype(np.float64) * (2 * np.pi / 256) + gpu["incident_cpu"]))
                spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(optical)))
                actual = spectrum[tuple(points.T)]
                power = abs(actual) ** 2
                if not np.all(np.isfinite(power)) or np.any(power <= 0):
                    raise ValueError(f"prepared {name} has a zero or invalid bright-site field")
                relative = power / authored
                ratio = float(relative.max() / relative.min())
                if endpoint_data is not None and ratio > SPOT_SUPPORT_TOLERANCE:
                    raise ValueError(f"endpoint_data {name} codes exceed authored intensity ratio "
                                     f"{SPOT_SUPPORT_TOLERANCE:g}: {ratio:.6g}")
                coefficients.append(_frozen(np.asarray(coefficient, np.complex64)))
                codes_list.append(_frozen(codes))
                fields.append(_frozen(actual.astype(np.complex64)))
                brightness.append(float(np.sqrt(np.sum(power) / np.sum(authored, dtype=np.float64))))
                endpoint_ratios.append(ratio)
            _prepare_rearrangement_lattice(gpu, geometry["source_yx"], codes_list[0])
            target_amplitude = np.sqrt(intensities[1])
            gpu["amplitude"] = cp.asarray(target_amplitude / np.linalg.norm(target_amplitude))
            gpu["target_codes"] = cp.asarray(codes_list[1])
            gpu["target_coefficients"] = cp.asarray(coefficients[1])
            gpu["target_fields"] = cp.asarray(fields[1])
            gpu["graphs"] = {}
            coarse_updates = 3 if 2 in gpu["resources"] else 0
            native_updates = 2 if coarse_updates else 10
            # Warm the exact update operation before capture, including its reduction.
            _rearrangement_amplitude_updates(gpu, native, 16, 1)
            if coarse_updates:
                _rearrangement_amplitude_updates(gpu, gpu["resources"][2], 16, 1)
            stream.synchronize()
            for band in native["plans"]:
                stream.begin_capture()
                _rearrangement_load_frame(gpu)
                if coarse_updates:
                    _rearrangement_amplitude_updates(gpu, gpu["resources"][2], band, coarse_updates)
                _rearrangement_amplitude_updates(gpu, native, band, native_updates)
                _rearrangement_propagate(gpu, native, band, "encode")
                _rearrangement_select_roots(gpu["measurement"], band)
                _rearrangement_propagate(gpu, gpu["measurement"], band, "forward")
                _rearrangement_store_frame(gpu)
                gpu["graphs"][band] = stream.end_capture()
                gpu["frame_index"].fill(0)
                gpu["graphs"][band].launch(stream)
            stream.synchronize()
            # Prepare the library's size classes for default two-ramp movies.
            # Two owned blocks allow an earlier result to stay live during the
            # next serial call; no occupancy-dependent data is precomputed.
            area = int(np.prod(shape))
            size = 1 << (max(512, 5 * area) - 1).bit_length()
            maximum_size = 1 << (max(512, (2 + gpu["motion_capacity"]) * area) - 1).bit_length()
            free_blocks = 0
            while size <= maximum_size:
                blocks = [gpu["output_pool"].malloc(size), gpu["output_pool"].malloc(size)]
                del blocks
                free_blocks += 2
                size *= 2
            if gpu["output_pool"].n_free_blocks() != free_blocks:
                raise MemoryError("SLM output buffers could not all be reserved before occupancy")
        return {
            **geometry, "gpu": gpu, "close": gpu["close"],
            "source_intensities": intensities[0], "target_intensities": intensities[1],
            "source_coefficients": coefficients[0], "target_coefficients": coefficients[1],
            "source_field": fields[0], "target_field": fields[1],
            "source_brightness": brightness[0], "target_brightness": brightness[1],
            "endpoint_support_intensity_ratios": tuple(endpoint_ratios),
            "initial_phase_codes": codes_list[0], "target_phase_codes": codes_list[1],
            "initial_pattern_phase": freeze_pattern_phase(codes_list[0].astype(np.float64) * (2 * np.pi / 256), shape),
            "target_pattern_phase": freeze_pattern_phase(codes_list[1].astype(np.float64) * (2 * np.pi / 256), shape),
        }
    except BaseException:
        gpu["close"]()
        raise


def compute_rearrangement(
    prepared: dict[str, object], occupied: object, *, surplus_policy: str,
    ramp_frames: int = 2, iterations: int | None = None,
    support_tolerance: float = SPOT_SUPPORT_TOLERANCE, dark_tolerance: float = .01,
    require_converged: bool = True, stop_requested: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Observed occupancy through every native phase-code map, ready on host.

    Unselected source light fades before motion. Bright constraints use authored
    weights and one measured physical brightness scale per endpoint; zero means
    a dark constraint and -1 an absent slot. Integer waypoints are subdivided
    into native third-bin positions without changing their piecewise path.
    None selects 64 source projections and 3 coarse + 2 native motion updates
    (10 native when stride-two quadrature is unavailable), then at most eight
    safeguarded encoded-field proposals for each failing motion frame. Explicit
    iterations requests exactly that many updates per calculated frame, with no
    additional adaptive proposals. Coarse steps and actual-field corrections
    share depth-two centered-log amplitude mixing. Histories belong to one
    frame and resolution, and restart when corrective damping changes.
    Zero still encodes and checks the result.
    Prepared target codes are reused only at the exact final endpoint when they
    satisfy the requested authored-weight gate.
    Converged covers intensity and dark-site gates, not refresh dynamics,
    integrated background power, physical removal, or atom survival. No device
    is contacted. Output ownership survives subsequent calls to this workspace.
    """
    import time  # noqa: PLC0415

    started = time.perf_counter()
    if not prepared["gpu"]:
        raise RuntimeError("SLM rearrangement workspace is closed")
    if surplus_policy != "discard":
        raise ValueError("this target consumes selected atoms only; explicitly choose surplus_policy='discard'")
    if (isinstance(ramp_frames, bool) or int(ramp_frames) != ramp_frames or ramp_frames < 1
            or (iterations is not None and
                (isinstance(iterations, bool) or int(iterations) != iterations or iterations < 0))):
        raise ValueError("ramp_frames must be positive and iterations nonnegative integers")
    tolerance = float(support_tolerance)
    if not np.isfinite(tolerance) or tolerance < 1:
        raise ValueError("support_tolerance must be finite and >= 1")
    dark_limit = _scalar(dark_tolerance, "dark_tolerance", nonnegative=True)
    gpu = prepared["gpu"]
    # These required resets depend only on the prepared working point, so let
    # the device perform them while the CPU matches atoms. No warmup is added.
    with gpu["stream"]:
        gpu["frame_index"].fill(0)
        gpu["source"]["small"][:] = gpu["source"]["initial"]
    plan = plan_rearrangement(prepared, occupied)
    after_plan = time.perf_counter()
    assignment = plan["assignment"]
    path, fractions = plan["motion_yx"], plan["fraction"]
    if len(path) == 1:
        path = np.concatenate((path, path))
        fractions = np.asarray([0., 1.])
    substeps = np.asarray([1 / 3, 2 / 3, 1.])
    moving = ((1 - substeps)[None, :, None, None] * path[:-1, None]
              + substeps[None, :, None, None] * path[1:, None]).reshape(-1, len(assignment), 2)
    progress = ((1 - substeps)[None] * fractions[:-1, None]
                + substeps[None] * fractions[1:, None]).reshape(-1)
    ramp_frames = int(ramp_frames)
    motion_frames, frames = len(moving), ramp_frames + len(moving)
    shape = prepared["shape_yx"]
    cp, stream, number = gpu["cp"], gpu["stream"], gpu["number"]
    native, source = gpu["resources"][1], gpu["source"]
    source_count = len(prepared["source_yx"])
    pixels = frames * int(np.prod(shape))
    desired = np.full((frames, source_count), -1, np.float32)
    sites = np.full((frames, source_count, 2), -1., np.float64)
    source_amplitude = prepared["source_brightness"] * np.sqrt(prepared["source_intensities"])
    target_amplitude = prepared["target_brightness"] * np.sqrt(prepared["target_intensities"])
    for index in range(ramp_frames):
        mix = (index + 1) / ramp_frames
        desired[index] = (1 - mix) * source_amplitude
        desired[index, assignment] += mix * target_amplitude
        sites[index] = prepared["source_yx"]
    desired[ramp_frames:, :number] = target_amplitude
    sites[ramp_frames:, :number] = moving
    source_updates = 64 if iterations is None else int(iterations)
    total_updates = (5 if 2 in gpu["resources"] else 10) if iterations is None else int(iterations)
    coarse_updates = min(3, max(0, total_updates - 2)) if 2 in gpu["resources"] else 0
    native_updates = total_updates - coarse_updates
    iteration_counts = np.full(frames, total_updates, np.int32)
    iteration_counts[:ramp_frames] = source_updates
    with stream:
        positions = cp.asarray(path, cp.int32)
        gpu["kernels"]["clearance"](((number + 15) // 16,) * 2, (16, 16),
                                    (positions, gpu["distances"], np.int32(len(path)), np.int32(number)))
        clearance = float(cp.sqrt(cp.min(gpu["distances"])))
        if clearance < prepared["minimum_separation"]:
            raise ValueError(f"emitted trajectory clearance {clearance:g} is below {prepared['minimum_separation']:g}")
        memory = gpu["output_pool"].malloc(pixels)
        host = np.frombuffer(memory, np.uint8, count=pixels).reshape((frames, *shape))
        movie = gpu["motion_codes"][:motion_frames]
        coefficients = gpu["motion_coefficients"][:motion_frames]
        actual_gpu = gpu["motion_actual"][:motion_frames]
        source_fields_gpu = cp.empty((ramp_frames, source_count), cp.complex64)
        for index in range(ramp_frames):
            if stop_requested is not None and stop_requested():
                stream.synchronize()
                raise InterruptedError("SLM rearrangement stopped")
            source["target"].set(desired[index], stream=stream)
            if iterations is None:
                source["graph"].launch(stream)
            else:
                _rearrangement_lattice_correct(gpu, source_updates)
            source_fields_gpu[index] = source["actual"]
            source["codes"].get(out=host[index], stream=stream, blocking=True)
        source_fields = source_fields_gpu.get()
        post = source_fields[-1, assignment]
        if np.any(abs(post) == 0) or not np.all(np.isfinite(post)):
            raise RuntimeError("source removal produced an invalid bright-site field")
        phase = np.angle(post)[None] + progress[:, None] * np.angle(
            prepared["target_coefficients"] * post.conj())[None]
        coefficient_values = (abs(prepared["target_coefficients"])[None] * np.exp(1j * phase)).astype(np.complex64)
        indices, frequencies, counts, offsets = _rearrangement_bind(gpu, moving)
        # Retain these ordinary host arrays until the final blocking transfer.
        coefficients.set(coefficient_values, stream=stream)
        for factor, values in indices.items():
            gpu["motion_indices"][factor][:motion_frames].set(values, stream=stream)
        gpu["motion_frequencies"][:len(frequencies)].set(frequencies, stream=stream)
        gpu["motion_offsets"][:motion_frames + 1].set(offsets, stream=stream)
        reused_target = False
        for index in range(motion_frames):
            if stop_requested is not None and stop_requested():
                # Bulk host metadata stays alive until every queued read ends.
                stream.synchronize()
                raise InterruptedError("SLM rearrangement stopped")
            if (index == motion_frames - 1 and progress[index] == 1.
                    and np.array_equal(moving[index], prepared["target_yx"])
                    and prepared["endpoint_support_intensity_ratios"][1] <= tolerance):
                movie[index] = gpu["target_codes"]
                coefficients[index] = gpu["target_coefficients"]
                actual_gpu[index] = gpu["target_fields"]
                iteration_counts[ramp_frames + index] = 0
                gpu["frame_index"].fill(motion_frames)
                reused_target = True
                continue
            band = (int(counts[index]) + 15) // 16 * 16
            if iterations is None:
                gpu["graphs"][band].launch(stream)
            else:
                _rearrangement_load_frame(gpu)
                if coarse_updates:
                    _rearrangement_amplitude_updates(gpu, gpu["resources"][2], band, coarse_updates)
                _rearrangement_amplitude_updates(gpu, native, band, native_updates)
                _rearrangement_propagate(gpu, native, band, "encode")
                _rearrangement_select_roots(gpu["measurement"], band)
                _rearrangement_propagate(gpu, gpu["measurement"], band, "forward")
                _rearrangement_store_frame(gpu)
        motion_fields = actual_gpu.get()
        relative = abs(motion_fields / target_amplitude[None]) ** 2
        ratios = relative.max(axis=1) / relative.min(axis=1)
        failed = np.flatnonzero(~np.isfinite(ratios) | (ratios > tolerance))
        proposals_evaluated = 0
        if iterations is None and len(failed):
            # A round submits every currently failing frame before the single
            # host read. Accepted coefficients/codes remain untouched on rejection.
            # Reuse the coarse update kernel with this frame's actual-field history.
            proposals = cp.empty((len(failed), number), cp.complex64)
            proposal_fields = cp.empty_like(proposals)
            proposal_codes = cp.empty((len(failed), *shape), cp.uint8)
            phases = cp.empty_like(proposals)
            history_g = cp.empty((len(failed), 3, number), cp.float32)
            history_r = cp.empty_like(history_g)
            candidate = cp.empty((len(failed), number), cp.float64)
            history_state = cp.empty((len(failed), 2), cp.int32)
            for slot, index in enumerate(failed):
                gpu["kernels"]["anderson_begin"]((1,), (256,),
                    (coefficients[index], phases[slot], history_state[slot], np.int32(number)))
            gains = np.full(len(failed), float(gpu["weight_exponent"]), np.float64)
            residual = np.log(np.maximum(abs(motion_fields[failed]), 1e-30) / target_amplitude[None])
            residual -= residual.mean(axis=1, keepdims=True)
            merits = np.mean(residual ** 2, axis=1)
            for _ in range(8):
                if stop_requested is not None and stop_requested():
                    stream.synchronize()
                    raise InterruptedError("SLM rearrangement stopped")
                active = np.flatnonzero(~np.isfinite(ratios[failed]) | (ratios[failed] > tolerance))
                if not len(active):
                    break
                for slot in active:
                    index = failed[slot]
                    band = (int(counts[index]) + 15) // 16 * 16
                    gpu["frame_index"].fill(int(index))
                    _rearrangement_load_frame(gpu)
                    gpu["kernels"]["anderson_update"]((1,), (256,),
                        (actual_gpu[index], gpu["amplitude"], gpu["coefficients"], phases[slot],
                         history_g[slot], history_r[slot], candidate[slot], history_state[slot],
                         np.int32(number), np.float32(gains[slot])))
                    _rearrangement_select_roots(native, band)
                    _rearrangement_propagate(gpu, native, band, "encode")
                    _rearrangement_select_roots(gpu["measurement"], band)
                    _rearrangement_propagate(gpu, gpu["measurement"], band, "forward")
                    proposals[slot] = gpu["coefficients"]
                    proposal_fields[slot] = gpu["measurement"]["actual"]
                    proposal_codes[slot] = native["codes"]
                proposed = proposal_fields.get()
                for slot in active:
                    index = failed[slot]
                    field = proposed[slot]
                    values = abs(field / target_amplitude) ** 2
                    ratio = float(values.max() / values.min())
                    update = np.log(np.maximum(abs(field), 1e-30) / target_amplitude)
                    update -= update.mean()
                    merit = float(np.mean(update ** 2))
                    accepted = np.isfinite(ratio) and (merit < merits[slot] or ratio <= tolerance)
                    dot = float(update @ residual[slot])
                    if accepted:
                        coefficients[index] = proposals[slot]
                        movie[index] = proposal_codes[slot]
                        actual_gpu[index] = proposal_fields[slot]
                        motion_fields[index] = field
                        residual[slot], merits[slot], ratios[index] = update, merit, ratio
                    if not accepted or dot < 0:
                        gains[slot] *= .5
                        # Damping changes the map; do not mix its old residuals.
                        history_state[slot].fill(0)
                    proposals_evaluated += 1
                    iteration_counts[ramp_frames + index] += 1
        # Source prefixes are already complete; publish the motion suffix only
        # after every accepted correction has reached the prepared GPU rows.
        movie.get(out=host[ramp_frames:], stream=stream, blocking=True)
        stream.synchronize()
    actual = np.zeros((frames, source_count), np.complex64)
    actual[:ramp_frames] = source_fields
    actual[ramp_frames:, :number] = motion_fields
    positive = desired > 0
    intensity = abs(actual.astype(np.complex128)) ** 2
    relative = np.divide(intensity, desired.astype(np.float64) ** 2,
                         out=np.zeros(desired.shape), where=positive)
    high = np.max(relative, axis=1)
    low = np.min(np.where(positive, relative, np.inf), axis=1)
    ratio = np.divide(high, low, out=np.full(frames, np.inf), where=low > 0)
    minimum = np.min(np.where(positive, intensity, np.inf), axis=1)
    dark = np.max(np.where(desired == 0, intensity, 0), axis=1)
    dark_ratio = np.divide(dark, minimum, out=np.full(frames, np.inf), where=minimum > 0)
    converged = bool(np.all(np.isfinite(ratio) & (ratio <= tolerance) & (dark_ratio <= dark_limit)))
    if require_converged and not converged:
        raise RuntimeError(f"SLM sequence did not meet intensity ratio {tolerance:g} and dark/bright ratio "
                           f"{dark_limit:g}; worst bright {np.max(ratio):.6g}, dark {np.max(dark_ratio):.6g}")
    mean = np.sum(relative, axis=1) / np.count_nonzero(positive, axis=1)
    rms = np.sqrt(np.sum(np.where(positive, (relative / mean[:, None] - 1) ** 2, 0), axis=1)
                  / np.count_nonzero(positive, axis=1))
    kept = np.concatenate((source_fields[:, assignment], motion_fields))
    initial = prepared["source_field"][assignment]
    brightness = abs(kept / initial[None]) ** 2
    reference = np.zeros(actual.shape, np.complex64)
    reference[:ramp_frames] = np.exp(1j * np.angle(prepared["source_field"]))[None]
    reference[ramp_frames:, :number] = np.exp(1j * phase)
    phase_error = np.angle(actual * reference.conj())
    codes = np.frombuffer(memoryview(memory).toreadonly(), np.uint8, count=pixels).reshape((frames, *shape))
    return {
        **plan, "motion_yx": _frozen(np.concatenate((path[:1].astype(float), moving))),
        "fraction": _frozen(np.r_[0., progress]), "phase_codes": codes,
        "sites_yx": sites, "actual_fields": actual, "desired_amplitudes": desired,
        "ramp_frames": ramp_frames, "iterations": tuple(map(int, iteration_counts)),
        "clearance": clearance, "support_intensity_ratios": ratio, "converged": converged,
        "dark_intensity_ratios": dark_ratio, "dark_tolerance": dark_limit, "support_tolerance": tolerance,
        "intensity_relative_rms": rms,
        "center_sample_power_proxy": np.sum(np.where(positive, intensity, 0), axis=1) / (np.prod(shape) * gpu["pupil_energy"]),
        "brightness_minimum_to_initial": brightness.min(axis=1),
        "brightness_maximum_to_initial": brightness.max(axis=1),
        "brightness_mean_to_initial": brightness.mean(axis=1),
        "source_brightness": prepared["source_brightness"], "target_brightness": prepared["target_brightness"],
        "phase_error_rms_rad": np.sqrt(np.sum(np.where(positive, phase_error ** 2, 0), axis=1)
                                      / np.count_nonzero(positive, axis=1)),
        "phase_step_max_rad": np.max(abs(np.angle(kept * np.concatenate((initial[None], kept[:-1])).conj())), axis=1),
        "surplus_policy": surplus_policy, "phase_encoding": "uint8:2pi/256",
        "prepared_target_reused": reused_target, "encoded_correction_proposals": proposals_evaluated,
        "timing_ms": {"plan": (after_plan - started) * 1000, "total": (time.perf_counter() - started) * 1000},
    }
