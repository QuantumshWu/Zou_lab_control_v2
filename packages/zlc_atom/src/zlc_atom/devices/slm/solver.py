"""Continuous SLM targets, phase retrieval, and their file formats."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
import json
import os
import weakref
from pathlib import Path
from typing import Any
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import fft, ndimage
from zlc_durable import atomic_write_file, write_readable_json

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

#: Prepared solve inputs keyed by the identity of the caller's read-only
#: array.  A feedback run hands the SAME frozen target and pupil to every
#: candidate solve, and re-validating, re-copying and re-shifting a full
#: 1024x1272 plane cost ~25 ms per call.  Identity is verified through the
#: stored weak reference; writable arrays are never cached because the
#: caller could mutate them in place between calls.
_PREPARED_LIMIT = 8
_PREPARED_TARGETS: dict[int, tuple[object, tuple]] = {}
_PREPARED_PUPILS: dict[int, tuple[object, tuple]] = {}
_DEFAULT_PUPILS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


#: Elementwise passes over the full 1024x1272 plane (arctan2, the phase
#: projection) dominate a solve's fixed cost, and numpy ufuncs release the
#: GIL on arrays this large -- so row stripes on a small shared pool run
#: them in parallel with bit-identical results (the same libm call per
#: element).  Planes below the threshold run inline.
_STRIPE_THRESHOLD = 1 << 18
_STRIPE_COUNT = 4
_stripe_pool: ThreadPoolExecutor | None = None


def _stripes(operation: Callable[[slice], None], rows: int, size: int) -> None:
    global _stripe_pool
    if size < _STRIPE_THRESHOLD or rows < _STRIPE_COUNT:
        operation(slice(0, rows))
        return
    if _stripe_pool is None:
        _stripe_pool = ThreadPoolExecutor(
            max_workers=_STRIPE_COUNT,
            thread_name_prefix="slm-solver-stripe",
        )
    step = (rows + _STRIPE_COUNT - 1) // _STRIPE_COUNT
    futures = [
        _stripe_pool.submit(operation, slice(start, min(rows, start + step)))
        for start in range(0, rows, step)
    ]
    for future in futures:
        future.result()


def _frozen(values: np.ndarray) -> np.ndarray:
    """Freeze a solver-owned fresh array in place -- no copy, no retyping."""

    values.setflags(write=False)
    return values


def _prepared_cache_get(cache: dict, values: object) -> tuple | None:
    if not isinstance(values, np.ndarray):
        return None
    entry = cache.get(id(values))
    if entry is None:
        return None
    reference, prepared = entry
    if reference() is values:
        return prepared
    del cache[id(values)]
    return None


def _prepared_cache_put(cache: dict, values: object, prepared: tuple) -> None:
    if not isinstance(values, np.ndarray) or values.flags.writeable:
        return
    key = id(values)

    def _drop(_reference: object, *, _cache: dict = cache, _key: int = key) -> None:
        _cache.pop(_key, None)

    cache[key] = (weakref.ref(values, _drop), prepared)
    while len(cache) > _PREPARED_LIMIT:
        cache.pop(next(iter(cache)))


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
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve one target, optionally reusing caller-owned transient spot state.

    The caller clears that state whenever the authored input pupil changes.
    """

    prepared_target = _prepared_cache_get(_PREPARED_TARGETS, target)
    if prepared_target is None:
        desired = validate_target(target)
        if float(np.max(desired)) <= 0.0:
            raise ValueError("target must contain positive intensity")
    else:
        desired = prepared_target[0]
    if spot_optimizer_state is not None and not isinstance(
        spot_optimizer_state, dict
    ):
        raise TypeError("spot_optimizer_state must be a dict or None")
    saved_state = dict(spot_optimizer_state) if spot_optimizer_state else None
    state_requested = spot_optimizer_state is not None
    seed_value = int(seed)
    prepared_pupil = None
    if pupil_amplitude is None:
        cached_default = _DEFAULT_PUPILS.get(desired.shape)
        if cached_default is None:
            pupil = _frozen(_pupil(desired.shape))
            cached_default = (pupil, _frozen(fft.ifftshift(pupil)))
            _DEFAULT_PUPILS[desired.shape] = cached_default
        prepared_pupil = cached_default
        pupil = cached_default[0]
        pupil_source = "default"
    else:
        pupil_source = "provided"
        prepared_pupil = _prepared_cache_get(_PREPARED_PUPILS, pupil_amplitude)
        if prepared_pupil is not None:
            pupil = prepared_pupil[0]
            if pupil.shape != desired.shape:
                raise ValueError(
                    "pupil_amplitude shape must match the target shape"
                )
        else:
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

    if prepared_pupil is not None and len(prepared_pupil) > 1:
        pupil_unshifted = prepared_pupil[1]
    else:
        pupil_unshifted = _frozen(fft.ifftshift(pupil))
        if pupil_source == "provided":
            _prepared_cache_put(
                _PREPARED_PUPILS, pupil_amplitude, (pupil, pupil_unshifted)
            )
    if prepared_target is not None:
        desired_unshifted, support_unshifted = prepared_target[1:3]
    else:
        desired_unshifted = _frozen(fft.ifftshift(desired))
        support_unshifted = _frozen(desired_unshifted > 0.0)
        _prepared_cache_put(
            _PREPARED_TARGETS,
            target,
            (desired, desired_unshifted, support_unshifted),
        )
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

            gate_start = 1 if hot_start_used else 12
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
                if support_ratio <= 1.01:
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
                    if candidate_ratio <= 1.01:
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

def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result

def _constant(value: str) -> None:
    raise ValueError(f"JSON contains {value}")

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
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_constant,
    )
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
    result = json.loads(
        encoded, object_pairs_hook=_strict_object, parse_constant=_constant
    )
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
    shape = tuple(pattern.shape)
    return canonical_phase(
        canonical_phase(pattern, shape).astype(np.float64)
        + canonical_phase(operator, shape).astype(np.float64),
        shape,
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
        metadata = json.loads(
            str(encoded.item()),
            object_pairs_hook=_strict_object,
            parse_constant=_constant,
        )
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
