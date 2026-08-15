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

def _canonical_unshifted_phase(field: np.ndarray) -> np.ndarray:
    phase = np.angle(field).astype(np.float32, copy=False)
    np.add(
        phase,
        np.float32(2.0 * np.pi),
        out=phase,
        where=phase < 0.0,
    )
    np.minimum(
        phase,
        np.nextafter(np.float32(2.0 * np.pi), np.float32(0.0)),
        out=phase,
    )
    return phase

def _phase_snapshot(field: np.ndarray) -> np.ndarray:
    return _readonly(fft.fftshift(_canonical_unshifted_phase(field)))

def _cartesian_support(
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    rows, columns = np.nonzero(support)
    unique_rows = np.unique(rows)
    unique_columns = np.unique(columns)
    if (
        rows.size > 256
        or unique_rows.size * unique_columns.size != rows.size
        or not np.all(support[np.ix_(unique_rows, unique_columns)])
    ):
        return None
    return unique_rows, unique_columns

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
        pupil = _pupil(desired.shape)
        pupil_source = "default"
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
        pupil_source = "provided"
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

    pupil_unshifted = fft.ifftshift(pupil)
    desired_unshifted = fft.ifftshift(desired)
    support_unshifted = desired_unshifted > 0.0
    epsilon = np.finfo(np.float32).eps

    transform = "fft"
    early_stopped = False
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
            rows, columns = cartesian
            desired_spots = desired_unshifted[np.ix_(rows, columns)].ravel()
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
            transform = "selected-dft"
            support_yx = [
                [
                    int((row + desired.shape[0] // 2) % desired.shape[0]),
                    int((column + desired.shape[1] // 2) % desired.shape[1]),
                ]
                for row in rows
                for column in columns
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
                        values = (weights * fixed_phase).reshape(
                            len(rows), len(columns)
                        )
                        back = (row_backward @ values) @ column_backward
                        field = pupil_unshifted * _unit_phase(back, epsilon)
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
            field = fft.ifftshift(
                pupil.astype(np.complex64)
                * np.exp(1j * phase).astype(np.complex64, copy=False)
            )
            weights = np.array(amplitude_spots, copy=True)

        selected: np.ndarray | None = None
        while iterations_run < count:
            if stop_requested is not None and stop_requested():
                raise InterruptedError("SLM phase solve stopped")
            if selected is None:
                if transform == "selected-dft":
                    selected = (
                        row_forward @ (field @ column_forward_transposed)
                    ).ravel()
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
            constrained_values = (weights * selected_phase).reshape(
                selected.shape if transform == "fft" else (len(rows), len(columns))
            )
            if transform == "selected-dft":
                back = (row_backward @ constrained_values) @ column_backward
            else:
                constrained.fill(0.0)
                constrained[support_unshifted] = constrained_values
                back = fft.ifft2(constrained, norm="ortho")
            field = pupil_unshifted * _unit_phase(back, epsilon)
            iterations_run += 1
            selected = None

            gate_start = 1 if hot_start_used else 12
            if iterations is None and iterations_run >= gate_start:
                if transform == "selected-dft":
                    selected = (
                        row_forward @ (field @ column_forward_transposed)
                    ).ravel()
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
                        candidate_selected = (
                            row_forward
                            @ (candidate_field @ column_forward_transposed)
                        ).ravel()
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
        if initial_phase is None:
            phase = np.random.default_rng(seed_value).uniform(
                0.0, 2.0 * np.pi, desired.shape
            ).astype(np.float32)
        else:
            phase = np.array(canonical_phase(initial_phase, desired.shape), copy=True)
        field = fft.ifftshift(
            pupil.astype(np.complex64)
            * np.exp(1j * phase).astype(np.complex64, copy=False)
        )
        amplitude = np.sqrt(desired_unshifted).astype(
            np.float32, copy=False
        )
        amplitude /= np.linalg.norm(amplitude[support_unshifted])
        amplitude_spots = amplitude[support_unshifted]
        weights = np.array(amplitude_spots, copy=True)
        for _iteration in range(count):
            if stop_requested is not None and stop_requested():
                raise InterruptedError("SLM phase solve stopped")
            far = fft.fft2(field, norm="ortho")
            selected = far[support_unshifted]
            magnitude = np.abs(selected).astype(np.float32, copy=False)
            measured = magnitude / max(float(np.linalg.norm(magnitude)), epsilon)
            weights *= np.sqrt(
                np.clip(amplitude_spots / np.maximum(measured, epsilon), 0.2, 5.0)
            )
            weights /= max(float(np.linalg.norm(weights)), epsilon)
            current_power = float(np.sum(np.square(magnitude, dtype=np.float32)))
            constrained = far * np.complex64(0.9)
            constrained[support_unshifted] = (
                weights
                * np.sqrt(max(current_power, epsilon))
                * _unit_phase(selected, epsilon)
            )
            back = fft.ifft2(constrained, norm="ortho")
            field = pupil_unshifted * _unit_phase(back, epsilon)
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
            final_selected = (
                row_forward @ (final_field @ column_forward_transposed)
            ).ravel()
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
    support_ratio = _support_intensity_ratio(
        final_magnitude, desired_unshifted[support_unshifted], epsilon
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

    return result, {
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
        "stop_reason": "support-ratio" if early_stopped else "iteration-limit",
        "support_intensity_ratio": support_ratio,
        "seed": seed_value,
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
