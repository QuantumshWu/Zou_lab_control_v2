"""Continuous SLM targets, phase retrieval, and their two file formats."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import fft, ndimage
from zlc_durable import atomic_write_file, write_readable_json

from .device import canonical_phase

_TARGET_FORMAT = "zlc.slm.target"
_TARGET_KEYS = frozenset({"format", "version", "shape", "intensity"})

def _pair(value: object, name: str) -> tuple[int, int]:
    try:
        pair = tuple(int(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a two-integer pair") from error
    if len(pair) != 2 or any(item <= 0 for item in pair):
        raise ValueError(f"{name} must contain two positive integers")
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
    starts = tuple((shape[index] - 1 - spans[index]) * 0.5 for index in range(2))
    axes = tuple(
        np.rint(starts[index] + spacing[index] * np.arange(grid[index])).astype(int)
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
    intensity_a: float = 1.0,
    intensity_b: float = 0.0,
) -> np.ndarray:
    shape, rows, columns = _grid_indices(shape_yx, grid_shape_yx, spacing_yx)
    levels = (
        _scalar(intensity_a, "intensity_a", nonnegative=True),
        _scalar(intensity_b, "intensity_b", nonnegative=True),
    )
    target = np.zeros(shape, dtype=np.float32)
    parity = np.indices((len(rows), len(columns))).sum(axis=0) & 1
    target[np.ix_(rows, columns)] = np.where(parity, levels[1], levels[0])
    return validate_target(target)

def _profile(distance_inside: np.ndarray, edge: object) -> np.ndarray:
    width = _scalar(edge, "edge", nonnegative=True)
    if width == 0.0:
        return (distance_inside >= 0.0).astype(np.float32)
    return np.clip((distance_inside + 0.5) / width, 0.0, 1.0).astype(np.float32)

def preset_rectangle(
    shape_yx: object,
    size_yx: object,
    *,
    intensity: float = 1.0,
    edge: float = 0.0,
) -> np.ndarray:
    shape, size = _pair(shape_yx, "shape_yx"), _pair(size_yx, "size_yx")
    if any(size[index] > shape[index] for index in range(2)):
        raise ValueError("rectangle does not fit inside target shape")
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    dy = 0.5 * size[0] - np.abs(yy - 0.5 * (shape[0] - 1))
    dx = 0.5 * size[1] - np.abs(xx - 0.5 * (shape[1] - 1))
    return validate_target(
        _scalar(intensity, "intensity", nonnegative=True) * _profile(np.minimum(dy, dx), edge)
    )

def preset_ellipse(
    shape_yx: object,
    radius_yx: object,
    *,
    intensity: float = 1.0,
    edge: float = 0.0,
) -> np.ndarray:
    shape = _pair(shape_yx, "shape_yx")
    radius = tuple(float(item) for item in _pair(radius_yx, "radius_yx"))
    if any(2.0 * radius[index] > shape[index] for index in range(2)):
        raise ValueError("ellipse does not fit inside target shape")
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    radial = np.sqrt(
        ((yy - 0.5 * (shape[0] - 1)) / radius[0]) ** 2
        + ((xx - 0.5 * (shape[1] - 1)) / radius[1]) ** 2
    )
    distance = (1.0 - radial) * min(radius)
    return validate_target(
        _scalar(intensity, "intensity", nonnegative=True) * _profile(distance, edge)
    )

def preset_ring(
    shape_yx: object,
    *,
    radius: float,
    width: float,
    intensity: float = 1.0,
    edge: float = 0.0,
) -> np.ndarray:
    shape = _pair(shape_yx, "shape_yx")
    ring_radius = _scalar(radius, "radius")
    ring_width = _scalar(width, "width")
    if ring_radius + 0.5 * ring_width > 0.5 * min(shape):
        raise ValueError("ring does not fit inside target shape")
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    radial = np.hypot(yy - 0.5 * (shape[0] - 1), xx - 0.5 * (shape[1] - 1))
    distance = 0.5 * ring_width - np.abs(radial - ring_radius)
    return validate_target(
        _scalar(intensity, "intensity", nonnegative=True) * _profile(distance, edge)
    )

def imported_target(values: object) -> np.ndarray:
    target = validate_target(values)
    peak = float(np.max(target))
    if peak <= 0.0:
        raise ValueError("imported target must contain positive intensity")
    return _readonly(target / peak)

def _pupil(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.ogrid[-1.0:1.0:shape[0] * 1j, -1.0:1.0:shape[1] * 1j]
    return (xx * xx + yy * yy <= 0.9**2).astype(np.float32)

def _forward(field: np.ndarray) -> np.ndarray:
    return fft.fftshift(fft.fft2(fft.ifftshift(field), norm="ortho"))


def _backward(field: np.ndarray) -> np.ndarray:
    return fft.fftshift(fft.ifft2(fft.ifftshift(field), norm="ortho"))

def solve_phase(
    target: object,
    *,
    initial_phase: object | None = None,
    iterations: int | None = None,
    seed: int = 0,
    stop_requested: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve one target; support topology privately selects weighted GS/MRAF."""

    desired = validate_target(target)
    if float(np.max(desired)) <= 0.0:
        raise ValueError("target must contain positive intensity")
    if stop_requested is not None and not callable(stop_requested):
        raise TypeError("stop_requested must be callable or None")
    pupil = _pupil(desired.shape)
    if initial_phase is None:
        phase = np.random.default_rng(int(seed)).uniform(0.0, 2.0 * np.pi, desired.shape).astype(np.float32)
    else:
        phase = np.array(canonical_phase(initial_phase, desired.shape), copy=True)
    field = pupil.astype(np.complex64) * np.exp(1j * phase).astype(np.complex64)
    support = desired > 0.0
    labels, components = ndimage.label(support)
    component_sizes = np.bincount(labels.ravel(), minlength=components + 1)[1:]
    largest = int(np.max(component_sizes, initial=0))
    method = "mraf" if largest > 4 or np.count_nonzero(support) > 0.02 * desired.size else "weighted-gs"
    if iterations is None:
        count = 300 if method == "mraf" else 80
    else:
        if isinstance(iterations, bool) or int(iterations) <= 0:
            raise ValueError("iterations must be a positive integer or None")
        count = int(iterations)
    amplitude = np.sqrt(desired).astype(np.float32)
    amplitude /= np.linalg.norm(amplitude[support])
    weights = np.array(amplitude[support], copy=True)
    epsilon = np.finfo(np.float32).eps

    for _iteration in range(count):
        if stop_requested is not None and stop_requested():
            raise InterruptedError("SLM phase solve stopped")
        far = _forward(field)
        phases = np.exp(1j * np.angle(far)).astype(np.complex64)
        if method == "weighted-gs":
            measured = np.abs(far[support]).astype(np.float32)
            measured /= max(float(np.linalg.norm(measured)), epsilon)
            ratio = amplitude[support] / np.maximum(measured, epsilon)
            weights *= np.sqrt(np.clip(ratio, 0.2, 5.0))
            weights /= max(float(np.linalg.norm(weights)), epsilon)
            constrained = np.zeros_like(far)
            constrained[support] = weights * phases[support]
        else:
            measured = np.abs(far[support]).astype(np.float32)
            measured /= max(float(np.linalg.norm(measured)), epsilon)
            ratio = amplitude[support] / np.maximum(measured, epsilon)
            weights *= np.sqrt(np.clip(ratio, 0.2, 5.0))
            weights /= max(float(np.linalg.norm(weights)), epsilon)
            current_power = float(np.sum(np.abs(far[support]) ** 2))
            constrained = far * np.complex64(0.9)
            constrained[support] = (
                weights * np.sqrt(max(current_power, epsilon)) * phases[support]
            )
        back = _backward(constrained)
        field = pupil.astype(np.complex64) * np.exp(1j * np.angle(back)).astype(np.complex64)

    result = canonical_phase(np.angle(field), desired.shape)
    final = _forward(pupil.astype(np.complex64) * np.exp(1j * result).astype(np.complex64))
    measured = np.abs(final[support]) ** 2
    measured /= max(float(np.sum(measured)), epsilon)
    expected = desired[support] / float(np.sum(desired[support]))
    error = float(np.sqrt(np.mean((measured - expected) ** 2)))
    efficiency = float(np.sum(np.abs(final[support]) ** 2) / np.sum(np.abs(final) ** 2))
    return result, {
        "method": method,
        "iterations": count,
        "seed": int(seed),
        "rms_intensity_error": error,
        "diffraction_efficiency": efficiency,
    }

def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result

def _constant(value: str) -> None:
    raise ValueError(f"JSON contains {value}")

def save_target(path: str | Path, target: object) -> Path:
    intensity = validate_target(target)
    return write_readable_json(
        path,
        {
            "format": _TARGET_FORMAT,
            "version": 1,
            "shape": list(intensity.shape),
            "intensity": intensity.tolist(),
        },
    )

def load_target(path: str | Path) -> np.ndarray:
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_constant,
    )
    if not isinstance(payload, dict) or set(payload) != _TARGET_KEYS:
        raise ValueError("target JSON has the wrong fields")
    if payload["format"] != _TARGET_FORMAT or type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("unsupported target JSON format")
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
    return target

def _metadata_json(metadata: Mapping[str, object]) -> str:
    if not isinstance(metadata, Mapping) or any(not isinstance(key, str) for key in metadata):
        raise TypeError("phase metadata must be a string-keyed mapping")
    return json.dumps(dict(metadata), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)

def save_phase(
    path: str | Path,
    phase: object,
    metadata: Mapping[str, object],
) -> Path:
    values = np.asarray(phase)
    if values.ndim != 2:
        raise ValueError("phase must be a two-dimensional array")
    radians = canonical_phase(values, tuple(values.shape))
    encoded = _metadata_json(metadata)
    return atomic_write_file(
        path,
        lambda stream: np.savez(stream, phase=radians, metadata=np.asarray(encoded)),
    )

def load_phase(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"phase", "metadata"}:
            raise ValueError("phase NPZ members must be exactly phase and metadata")
        raw_phase = np.asarray(archive["phase"])
        encoded = np.asarray(archive["metadata"])
        if raw_phase.dtype != np.dtype("<f4") or raw_phase.ndim != 2:
            raise ValueError("phase NPZ phase must be a little-endian float32 matrix")
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise ValueError("phase NPZ metadata must be scalar Unicode JSON")
        if not np.all(np.isfinite(raw_phase)) or np.any(raw_phase < 0.0) or np.any(raw_phase >= 2.0 * np.pi):
            raise ValueError("phase NPZ phase must contain canonical radians")
        phase = canonical_phase(raw_phase, tuple(raw_phase.shape))
        metadata = json.loads(
            str(encoded.item()),
            object_pairs_hook=_strict_object,
            parse_constant=_constant,
        )
    if not isinstance(metadata, dict):
        raise ValueError("phase NPZ metadata JSON must be an object")
    _metadata_json(metadata)
    return phase, metadata
