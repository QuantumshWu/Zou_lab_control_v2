"""Headless fit catalogue and solver used by every presentation backend."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from numbers import Real
import threading
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares, minimize, minimize_scalar
from scipy.signal import find_peaks

from ._validation import finite_real as _finite_real
from ._validation import integer, text as _text
from .kinds import AxisRef

if TYPE_CHECKING:
    from ._fit_scene import FitOverlay


ArrayTuple = tuple[np.ndarray, ...]
Evaluator = Callable[..., np.ndarray]
Initializer = Callable[[ArrayTuple, np.ndarray], Sequence[float]]
CandidateInitializer = Callable[
    [ArrayTuple, np.ndarray], Sequence[Sequence[float]]
]
BoundsInitializer = Callable[
    [ArrayTuple, np.ndarray],
    Mapping[str, tuple[float | None, float | None]],
]
Jacobian = Callable[..., np.ndarray]

_FIT_RSS_TIE_RELATIVE = 1e-10


class FitCancelled(RuntimeError):
    pass


class FitDeadlineExceeded(TimeoutError):
    pass


class ParameterDomain(str, Enum):
    REAL = "real"
    POSITIVE = "positive"
    NONNEGATIVE = "nonnegative"
    PHASE_RADIANS = "phase_radians"


class UnitRelation(str, Enum):
    VALUE = "value"
    AXIS_0 = "axis_0"
    AXIS_1 = "axis_1"
    INVERSE_AXIS_0 = "inverse_axis_0"
    RADIAN = "radian"


class FitTarget(str, Enum):
    """Semantic projection a model is authored to fit."""

    SERIES = "series"
    HISTOGRAM = "histogram"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class FitParameterDisplay:
    """One fitted parameter in the units currently painted on the plot."""

    name: str
    label: str
    value: float
    standard_error: float | None = None
    unit: str = ""

    def __post_init__(self) -> None:
        for field_name in ("name", "label"):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), f"fit parameter {field_name}"),
            )
        value = _finite_real(self.value, "fit parameter value")
        object.__setattr__(self, "value", value)
        if self.standard_error is not None:
            error = _finite_real(
                self.standard_error,
                "fit parameter standard_error",
            )
            if error < 0.0:
                raise ValueError("fit parameter standard_error must be non-negative")
            object.__setattr__(self, "standard_error", error)
        object.__setattr__(
            self,
            "unit",
            _text(self.unit, "fit parameter unit", allow_empty=True),
        )


@dataclass(frozen=True, slots=True)
class FitParameterSpec:
    """One solver parameter and its renderer-facing display semantics.

    ``affine_point`` distinguishes absolute positions from spans or amplitudes.
    It applies only to parameter values; standard errors are always converted
    as differences and therefore never consume a unit offset.
    ``solver_unit_relation`` records the canonical unit used by the evaluator
    when it differs from the parameter's painted axis relation.
    """

    name: str
    unit_relation: UnitRelation
    domain: ParameterDomain = ParameterDomain.REAL
    display_label: str | None = None
    affine_point: bool = False
    solver_unit_relation: UnitRelation | None = None

    def __post_init__(self) -> None:
        name = _text(self.name, "fit parameter name")
        if not isinstance(self.unit_relation, UnitRelation):
            raise TypeError("unit_relation must be UnitRelation")
        if not isinstance(self.domain, ParameterDomain):
            raise TypeError("domain must be ParameterDomain")
        display_label = self.display_label
        if display_label is not None:
            display_label = _text(display_label, "fit parameter display_label")
        if not isinstance(self.affine_point, bool):
            raise TypeError("affine_point must be bool")
        solver_unit_relation = self.solver_unit_relation
        if solver_unit_relation is None:
            solver_unit_relation = self.unit_relation
        if not isinstance(solver_unit_relation, UnitRelation):
            raise TypeError("solver_unit_relation must be UnitRelation or None")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "display_label", display_label)
        object.__setattr__(self, "solver_unit_relation", solver_unit_relation)

    @property
    def bounds(self) -> tuple[float, float]:
        if self.domain is ParameterDomain.REAL:
            return -np.inf, np.inf
        if self.domain is ParameterDomain.POSITIVE:
            return float(np.nextafter(0.0, np.inf)), np.inf
        if self.domain is ParameterDomain.NONNEGATIVE:
            return 0.0, np.inf
        if self.domain is ParameterDomain.PHASE_RADIANS:
            return -np.pi, float(np.nextafter(np.pi, -np.inf))
        raise RuntimeError(self.domain)


@dataclass(frozen=True, slots=True)
class FitComponentSpec:
    """One named additive component of the presented model family."""

    component_id: str
    evaluator: Evaluator

    def __post_init__(self) -> None:
        component_id = _text(self.component_id, "fit component id")
        if not callable(self.evaluator):
            raise TypeError("fit component evaluator must be callable")
        object.__setattr__(self, "component_id", component_id)


@dataclass(frozen=True, slots=True)
class FitEllipseGlyphSpec:
    """Parameter references needed to paint a fitted center and two radii."""

    center_parameters: tuple[str, str]
    radius_parameters: tuple[str, str]

    def __post_init__(self) -> None:
        center_parameters = tuple(
            _text(name, "ellipse glyph center parameter")
            for name in self.center_parameters
        )
        if (
            len(center_parameters) != 2
            or center_parameters[0] == center_parameters[1]
        ):
            raise ValueError("ellipse glyph requires two distinct center parameters")
        radius_parameters = tuple(
            _text(name, "ellipse glyph radius parameter")
            for name in self.radius_parameters
        )
        if len(radius_parameters) != 2:
            raise ValueError("ellipse glyph requires one radius parameter per axis")
        object.__setattr__(self, "center_parameters", center_parameters)
        object.__setattr__(self, "radius_parameters", radius_parameters)


@dataclass(frozen=True, slots=True)
class FitPresentationSpec:
    """Backend-neutral metadata for additive curves or one ellipse glyph."""

    components: tuple[FitComponentSpec, ...] = ()
    ellipse_glyph: FitEllipseGlyphSpec | None = None

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if any(not isinstance(item, FitComponentSpec) for item in components):
            raise TypeError("fit presentation components must be FitComponentSpec values")
        component_ids = tuple(item.component_id for item in components)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("fit presentation component ids must be unique")
        if self.ellipse_glyph is not None and not isinstance(
            self.ellipse_glyph, FitEllipseGlyphSpec
        ):
            raise TypeError("ellipse_glyph must be FitEllipseGlyphSpec or None")
        if self.ellipse_glyph is not None and components:
            raise ValueError("ellipse glyph presentation cannot also define components")
        object.__setattr__(self, "components", components)


@dataclass(frozen=True, slots=True)
class FitModelSpec:
    """One fit model, including optional MathText presentation metadata."""

    model_id: str
    display_name: str
    independent_arity: int
    parameters: tuple[FitParameterSpec, ...]
    headline: str
    evaluator: Evaluator
    initializer: Initializer
    targets: tuple[FitTarget, ...]
    formula: str | None = None
    jacobian: Jacobian | None = None
    candidate_initializer: CandidateInitializer | None = None
    bounds_initializer: BoundsInitializer | None = None
    presentation: FitPresentationSpec = field(default_factory=FitPresentationSpec)
    coordinate_relations: tuple[UnitRelation, ...] | None = None
    default_for: tuple[FitTarget, ...] = ()
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        model_id = _text(self.model_id, "fit model id")
        display_name = _text(self.display_name, "fit model display name")
        independent_arity = integer(
            self.independent_arity,
            "fit model independent_arity",
        )
        if independent_arity not in (1, 2):
            raise ValueError("fit models support one or two independent axes")
        parameters = tuple(self.parameters)
        if not parameters or any(not isinstance(item, FitParameterSpec) for item in parameters):
            raise ValueError("fit model requires parameter specifications")
        names = tuple(item.name for item in parameters)
        if len(names) != len(set(names)):
            raise ValueError("fit parameter names must be unique")
        headline = _text(self.headline, "fit model headline")
        if headline not in names:
            raise ValueError(
                f"fit model headline must name a parameter: {headline!r}"
            )
        if not callable(self.evaluator) or not callable(self.initializer):
            raise TypeError("evaluator and initializer must be callable")
        targets = tuple(self.targets)
        if (
            not targets
            or any(not isinstance(target, FitTarget) for target in targets)
            or len(targets) != len(set(targets))
        ):
            raise ValueError("fit model targets must be unique FitTarget values")
        defaults = tuple(self.default_for)
        if (
            any(not isinstance(target, FitTarget) for target in defaults)
            or len(defaults) != len(set(defaults))
            or not set(defaults).issubset(targets)
        ):
            raise ValueError("default_for must be unique members of targets")
        capabilities = frozenset(str(value).strip() for value in self.capabilities)
        if any(not value for value in capabilities):
            raise ValueError("fit capabilities must be non-empty text")
        if self.candidate_initializer is not None and not callable(
            self.candidate_initializer
        ):
            raise TypeError("candidate_initializer must be callable or None")
        if self.bounds_initializer is not None and not callable(
            self.bounds_initializer
        ):
            raise TypeError("bounds_initializer must be callable or None")
        if self.jacobian is not None and not callable(self.jacobian):
            raise TypeError("jacobian must be callable or None")
        if not isinstance(self.presentation, FitPresentationSpec):
            raise TypeError("presentation must be FitPresentationSpec")
        coordinate_relations = self.coordinate_relations
        if coordinate_relations is None:
            coordinate_relations = (
                (UnitRelation.AXIS_0,)
                if self.independent_arity == 1
                else (UnitRelation.AXIS_0, UnitRelation.AXIS_1)
            )
        else:
            coordinate_relations = tuple(coordinate_relations)
        if len(coordinate_relations) != self.independent_arity or any(
            not isinstance(relation, UnitRelation)
            or relation not in {UnitRelation.AXIS_0, UnitRelation.AXIS_1}
            for relation in coordinate_relations
        ):
            raise ValueError(
                "coordinate_relations must identify one plot axis per independent axis"
            )
        ellipse_glyph = self.presentation.ellipse_glyph
        if ellipse_glyph is not None:
            if self.independent_arity != 2:
                raise ValueError("ellipse glyph presentation requires a two-axis model")
            referenced = {
                *ellipse_glyph.center_parameters,
                *ellipse_glyph.radius_parameters,
            }
            unknown = referenced - set(names)
            if unknown:
                raise ValueError(
                    f"ellipse glyph names unknown parameters: {sorted(unknown)}"
                )
        formula = self.formula
        if formula is not None:
            formula = _text(formula, "fit model formula")
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "independent_arity", independent_arity)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "headline", headline)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "formula", formula)
        object.__setattr__(self, "coordinate_relations", coordinate_relations)
        object.__setattr__(self, "default_for", defaults)
        object.__setattr__(self, "capabilities", capabilities)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.parameters)

    def parameter_index(self, name: str) -> int:
        parameter_name = _text(name, "fit parameter name")
        try:
            return self.parameter_names.index(parameter_name)
        except ValueError as error:
            raise ValueError(f"unknown fit parameter: {name!r}") from error

    def evaluate(self, coordinates: ArrayTuple, values: Sequence[float]) -> np.ndarray:
        coordinates = _coordinate_arrays(coordinates, self.independent_arity)
        parameters = np.asarray(values, dtype=np.float64).reshape(-1)
        if parameters.size != len(self.parameters):
            raise ValueError("fit parameter count does not match model")
        return np.asarray(self.evaluator(*coordinates, *parameters), dtype=np.float64)

    def evaluate_component(
        self,
        component_id: str,
        coordinates: ArrayTuple,
        values: Sequence[float],
    ) -> np.ndarray:
        requested = _text(component_id, "fit component id")
        component = next(
            (
                item
                for item in self.presentation.components
                if item.component_id == requested
            ),
            None,
        )
        if component is None:
            raise ValueError(f"unknown fit component: {component_id!r}")
        coordinates = _coordinate_arrays(coordinates, self.independent_arity)
        parameters = np.asarray(values, dtype=np.float64).reshape(-1)
        if parameters.size != len(self.parameters):
            raise ValueError("fit parameter count does not match model")
        return np.asarray(
            component.evaluator(*coordinates, *parameters),
            dtype=np.float64,
        )

    def evaluate_jacobian(
        self,
        coordinates: ArrayTuple,
        values: Sequence[float],
    ) -> np.ndarray:
        if self.jacobian is None:
            raise RuntimeError(f"fit model {self.model_id!r} has no analytic jacobian")
        coordinates = _coordinate_arrays(coordinates, self.independent_arity)
        parameters = np.asarray(values, dtype=np.float64).reshape(-1)
        if parameters.size != len(self.parameters):
            raise ValueError("fit parameter count does not match model")
        jacobian = np.asarray(
            self.jacobian(*coordinates, *parameters),
            dtype=np.float64,
        )
        expected = (coordinates[0].size, len(self.parameters))
        if jacobian.shape != expected:
            raise ValueError(
                f"fit model jacobian must have shape {expected}, got {jacobian.shape}"
            )
        return jacobian


class FitModelRegistry:
    def __init__(self, models: Sequence[FitModelSpec] = ()) -> None:
        self._lock = threading.RLock()
        self._models: dict[str, FitModelSpec] = {}
        for model in models:
            self.register(model)

    def register(self, model: FitModelSpec, *, replace: bool = False) -> None:
        if not isinstance(model, FitModelSpec):
            raise TypeError("model must be FitModelSpec")
        with self._lock:
            if model.model_id in self._models and not replace:
                raise ValueError(f"fit model already registered: {model.model_id}")
            for target in model.default_for:
                conflict = next(
                    (
                        registered.model_id
                        for registered in self._models.values()
                        if registered.model_id != model.model_id
                        and target in registered.default_for
                    ),
                    None,
                )
                if conflict is not None:
                    raise ValueError(
                        f"fit target {target.value!r} already has default model "
                        f"{conflict!r}"
                    )
            self._models[model.model_id] = model

    def get(self, model_id: str) -> FitModelSpec:
        selected = _text(model_id, "fit model id")
        with self._lock:
            try:
                return self._models[selected]
            except KeyError as error:
                raise ValueError(f"unknown fit model: {model_id!r}") from error

    def models(self) -> tuple[FitModelSpec, ...]:
        with self._lock:
            return tuple(self._models.values())

    def models_for(self, target: FitTarget) -> tuple[FitModelSpec, ...]:
        """Return compatible models with the target's authored default first."""

        if not isinstance(target, FitTarget):
            raise TypeError("target must be FitTarget")
        with self._lock:
            models = tuple(
                model for model in self._models.values() if target in model.targets
            )
        return tuple(
            sorted(models, key=lambda model: target not in model.default_for)
        )


@dataclass(frozen=True, slots=True)
class FitOptions:
    loss: str = "linear"
    max_nfev: int = 5000
    deadline_seconds: float | None = None

    def __post_init__(self) -> None:
        loss = _text(self.loss, "fit loss")
        if loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError("unsupported scipy least-squares loss")
        max_nfev = integer(self.max_nfev, "max_nfev")
        if max_nfev <= 0:
            raise ValueError("max_nfev must be a positive integer")
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "max_nfev", max_nfev)
        if self.deadline_seconds is not None:
            deadline = _finite_real(self.deadline_seconds, "deadline_seconds")
            if deadline <= 0:
                raise ValueError("deadline_seconds must be positive")
            object.__setattr__(self, "deadline_seconds", deadline)


@dataclass(frozen=True, slots=True)
class FitNumericTable:
    """Pure numeric fit columns shared by single fits and facet batches.

    ``source_revision`` identifies the data used by the solver.  The
    independent ``batch_revision`` identifies this publication and is the
    revision a downstream derived-data stream must advance monotonically.
    """

    parameter_names: tuple[str, ...]
    parameter_units: Mapping[str, str]
    parameter_values: Mapping[str, np.ndarray]
    parameter_errors: Mapping[str, np.ndarray]
    parameter_error_validity: Mapping[str, np.ndarray]
    success: np.ndarray
    sample_axis_name: str
    sample_coordinates: np.ndarray
    sample_unit: str
    sample_labels: tuple[str, ...] | None
    source_revision: int
    batch_revision: int

    def __post_init__(self) -> None:
        names = tuple(_text(name, "fit table parameter name") for name in self.parameter_names)
        if len(names) != len(set(names)):
            raise ValueError("fit table parameter names must be unique")
        units = dict(self.parameter_units)
        if set(units) != set(names):
            raise ValueError("fit table parameter units must match parameter names")
        units = {
            name: _text(units[name], f"fit table unit {name}", allow_empty=True)
            for name in names
        }
        values = _numeric_fit_columns(self.parameter_values, names, "values")
        errors = _numeric_fit_columns(self.parameter_errors, names, "errors")
        error_validity = _boolean_fit_columns(
            self.parameter_error_validity,
            names,
            "error validity",
        )
        success = np.asarray(self.success, dtype=np.bool_).reshape(-1)
        count = int(success.size)
        if any(column.size != count for column in (*values.values(), *errors.values())):
            raise ValueError("fit table columns must have equal length")
        if any(column.size != count for column in error_validity.values()):
            raise ValueError("fit table error-validity columns must have equal length")
        for name in names:
            value_column = values[name]
            error_column = errors[name]
            valid_error = error_validity[name]
            if np.any(success & ~np.isfinite(value_column)):
                raise ValueError(
                    f"fit table successful values for {name!r} must be finite"
                )
            if np.any(~success & np.isfinite(value_column)):
                raise ValueError(
                    f"fit table failed values for {name!r} must be NaN"
                )
            expected_error_validity = success & np.isfinite(error_column)
            if not np.array_equal(valid_error, expected_error_validity):
                raise ValueError(
                    f"fit table error validity for {name!r} disagrees with values"
                )
            if np.any(~valid_error & np.isfinite(error_column)):
                raise ValueError(
                    f"fit table invalid errors for {name!r} must be NaN"
                )
            if np.any(valid_error & (error_column < 0.0)):
                raise ValueError(
                    f"fit table valid errors for {name!r} cannot be negative"
                )
        success = _readonly(success)
        coordinates = np.asarray(self.sample_coordinates, dtype=np.float64).reshape(-1)
        if coordinates.size != count or not np.all(np.isfinite(coordinates)):
            raise ValueError("fit table sample coordinates must match success")
        labels = self.sample_labels
        if labels is not None:
            labels = tuple(_text(label, "fit table sample label") for label in labels)
            if len(labels) != count:
                raise ValueError("fit table sample labels must match coordinates")
        axis_name = _text(self.sample_axis_name, "fit table sample axis name", allow_empty=True)
        unit = _text(self.sample_unit, "fit table sample unit", allow_empty=True)
        if unit == "1":
            unit = ""
        source_revision = integer(self.source_revision, "fit table source_revision")
        batch_revision = integer(self.batch_revision, "fit table batch_revision")
        if source_revision < 0:
            raise ValueError("fit table source_revision must be non-negative")
        if batch_revision < 0:
            raise ValueError("fit table batch_revision must be non-negative")
        if labels is None and unit and not axis_name:
            raise ValueError("unit-bearing sample coordinates require an axis name")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "parameter_units", MappingProxyType(units))
        object.__setattr__(self, "parameter_values", MappingProxyType(values))
        object.__setattr__(self, "parameter_errors", MappingProxyType(errors))
        object.__setattr__(
            self,
            "parameter_error_validity",
            MappingProxyType(error_validity),
        )
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "sample_axis_name", axis_name)
        object.__setattr__(self, "sample_coordinates", _readonly(coordinates))
        object.__setattr__(self, "sample_unit", unit)
        object.__setattr__(self, "sample_labels", labels)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "batch_revision", batch_revision)


def _readonly(array: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(array)
    backing: object = source
    while isinstance(getattr(backing, "base", None), np.ndarray):
        backing = backing.base
    if not source.flags.writeable and isinstance(getattr(backing, "base", None), bytes):
        return source
    return np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(source.shape)


def _numeric_fit_columns(
    columns: Mapping[str, np.ndarray],
    names: tuple[str, ...],
    label: str,
) -> dict[str, np.ndarray]:
    if not isinstance(columns, Mapping) or set(columns) != set(names):
        raise ValueError(f"fit table parameter {label} must match parameter names")
    normalized: dict[str, np.ndarray] = {}
    for name in names:
        values = np.asarray(columns[name], dtype=np.float64).reshape(-1)
        normalized[name] = _readonly(values)
    return normalized


def _boolean_fit_columns(
    columns: Mapping[str, np.ndarray],
    names: tuple[str, ...],
    label: str,
) -> dict[str, np.ndarray]:
    if not isinstance(columns, Mapping) or set(columns) != set(names):
        raise ValueError(f"fit table parameter {label} must match parameter names")
    normalized: dict[str, np.ndarray] = {}
    for name in names:
        values = np.asarray(columns[name], dtype=np.bool_).reshape(-1)
        normalized[name] = _readonly(values)
    return normalized


def _make_fit_numeric_table(
    *,
    parameter_names: tuple[str, ...],
    parameter_units: Mapping[str, str],
    parameter_values: Mapping[str, np.ndarray],
    parameter_errors: Mapping[str, np.ndarray],
    success: np.ndarray,
    sample_axis_name: str,
    sample_coordinates: np.ndarray,
    sample_unit: str,
    sample_labels: tuple[str, ...] | None,
    source_revision: int,
    batch_revision: int,
) -> FitNumericTable:
    """Construct the one numeric-table shape for scalar and facet fits."""

    successful = np.asarray(success, dtype=np.bool_).reshape(-1)
    values = {
        name: np.where(
            successful,
            np.asarray(parameter_values[name], dtype=np.float64).reshape(-1),
            np.nan,
        )
        for name in parameter_names
    }
    errors = {
        name: np.asarray(parameter_errors[name], dtype=np.float64).reshape(-1)
        for name in parameter_names
    }
    error_validity = {
        name: successful & np.isfinite(errors[name])
        for name in parameter_names
    }
    errors = {
        name: np.where(error_validity[name], errors[name], np.nan)
        for name in parameter_names
    }
    return FitNumericTable(
        parameter_names=parameter_names,
        parameter_units=parameter_units,
        parameter_values=values,
        parameter_errors=errors,
        parameter_error_validity=error_validity,
        success=successful,
        sample_axis_name=sample_axis_name,
        sample_coordinates=sample_coordinates,
        sample_unit=sample_unit,
        sample_labels=sample_labels,
        source_revision=source_revision,
        batch_revision=batch_revision,
    )


@dataclass(frozen=True, slots=True, eq=False)
class RegularImageFitInput:
    """A Cartesian image without materialized per-pixel coordinate grids.

    Rows are identified by ``y_coordinates`` and columns by ``x_coordinates``;
    consequently ``observations.shape`` must be ``(len(y), len(x))``.  The
    image storage is retained as a read-only view rather than copied.  Callers
    must therefore not mutate its backing storage while a fit is running.
    """

    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    observations: np.ndarray
    valid_mask: np.ndarray | None = None
    selected_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.x_coordinates, dtype=np.float64).reshape(-1)
        y = np.asarray(self.y_coordinates, dtype=np.float64).reshape(-1)
        if x.size == 0 or y.size == 0:
            raise ValueError("regular image coordinates cannot be empty")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("regular image coordinates must be finite")
        for name, values in (("x", x), ("y", y)):
            differences = np.diff(values)
            if differences.size and not (
                np.all(differences > 0.0) or np.all(differences < 0.0)
            ):
                raise ValueError(
                    f"regular image {name} coordinates must be strictly monotonic"
                )

        image = np.asarray(self.observations)
        if image.ndim != 2 or image.shape != (y.size, x.size):
            raise ValueError(
                "regular image observations must have shape (len(y), len(x))"
            )
        if image.dtype.kind not in "fiu":
            raise TypeError("regular image observations must be real numeric values")
        image = image.view()
        image.setflags(write=False)

        valid = self.valid_mask
        if valid is not None:
            valid = np.asarray(valid, dtype=np.bool_)
            if valid.shape != image.shape:
                raise ValueError("regular image valid_mask must match observations")
            valid = valid.view()
            valid.setflags(write=False)

        indices = self.selected_indices
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            if indices.size != image.size:
                raise ValueError(
                    "regular image selected_indices must identify every image cell"
                )
            indices = indices.reshape(image.shape).view()
            indices.setflags(write=False)

        object.__setattr__(self, "x_coordinates", _readonly(x))
        object.__setattr__(self, "y_coordinates", _readonly(y))
        object.__setattr__(self, "observations", image)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "selected_indices", indices)


@dataclass(frozen=True, slots=True)
class FitResult:
    """Immutable fit output with an explicit covariance validity boundary."""

    model: FitModelSpec
    parameter_values: np.ndarray
    standard_errors: np.ndarray
    covariance: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    selected_indices: np.ndarray
    source_revision: int
    success: bool
    message: str
    reduced_chi_square: float
    covariance_valid: bool = True
    parameter_units: Mapping[str, str] = field(default_factory=dict)
    batch_revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model, FitModelSpec):
            raise TypeError("model must be FitModelSpec")
        parameters = np.asarray(self.parameter_values, dtype=np.float64).reshape(-1)
        errors = np.asarray(self.standard_errors, dtype=np.float64).reshape(-1)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        count = len(self.model.parameters)
        if parameters.shape != (count,) or errors.shape != (count,):
            raise ValueError("fit parameter/error shapes disagree with model")
        if covariance.shape != (count, count):
            raise ValueError("fit covariance shape disagrees with model")
        if not isinstance(self.covariance_valid, (bool, np.bool_)):
            raise TypeError("covariance_valid must be bool")
        covariance_valid = bool(self.covariance_valid)
        if covariance_valid:
            if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(errors)):
                raise ValueError("valid fit covariance and standard errors must be finite")
            if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-15):
                raise ValueError("valid fit covariance must be symmetric")
            diagonal = np.diag(covariance)
            if np.any(diagonal < 0.0) or np.any(errors < 0.0):
                raise ValueError("valid fit variances and standard errors cannot be negative")
            if not np.allclose(errors**2, diagonal, rtol=1e-10, atol=1e-15):
                raise ValueError("standard errors must agree with fit covariance")
        else:
            covariance = np.full((count, count), np.nan, dtype=np.float64)
            errors = np.full(count, np.nan, dtype=np.float64)
        fitted = np.asarray(self.fitted_values, dtype=np.float64).reshape(-1)
        residuals = np.asarray(self.residuals, dtype=np.float64).reshape(-1)
        if fitted.shape != residuals.shape:
            raise ValueError("fitted values and residuals must have equal shape")
        indices = np.asarray(self.selected_indices, dtype=np.int64).reshape(-1)
        if indices.shape != fitted.shape:
            raise ValueError("selected indices must identify every fitted observation")
        source_revision = integer(self.source_revision, "source_revision")
        if source_revision < 0:
            raise ValueError("source_revision must be non-negative")
        batch_revision = integer(self.batch_revision, "batch_revision")
        if batch_revision < 0:
            raise ValueError("batch_revision must be non-negative")
        object.__setattr__(self, "parameter_values", _readonly(parameters))
        object.__setattr__(self, "standard_errors", _readonly(errors))
        object.__setattr__(self, "covariance", _readonly(covariance))
        object.__setattr__(self, "fitted_values", _readonly(fitted))
        object.__setattr__(self, "residuals", _readonly(residuals))
        object.__setattr__(self, "selected_indices", _readonly(indices))
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "batch_revision", batch_revision)
        object.__setattr__(
            self,
            "message",
            _text(self.message, "fit result message", allow_empty=True),
        )
        if not isinstance(self.success, (bool, np.bool_)):
            raise TypeError("fit result success must be bool")
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(
            self,
            "reduced_chi_square",
            _finite_real(self.reduced_chi_square, "reduced_chi_square"),
        )
        object.__setattr__(self, "covariance_valid", covariance_valid)
        units = dict(self.parameter_units)
        unknown = set(units) - set(self.model.parameter_names)
        if unknown:
            raise ValueError(
                f"fit result units name unknown parameters: {sorted(unknown)}"
            )
        object.__setattr__(
            self,
            "parameter_units",
            MappingProxyType({
                name: _text(
                    units.get(name, ""),
                    f"fit parameter unit {name}",
                    allow_empty=True,
                )
                for name in self.model.parameter_names
            }),
        )

    @property
    def parameters(self) -> Mapping[str, float]:
        return MappingProxyType(
            dict(zip(self.model.parameter_names, map(float, self.parameter_values), strict=True))
        )

    @property
    def errors(self) -> Mapping[str, float]:
        return MappingProxyType(
            dict(zip(self.model.parameter_names, map(float, self.standard_errors), strict=True))
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.model.parameter_names

    @property
    def parameter_errors(self) -> Mapping[str, float]:
        return self.errors

    @property
    def parameter_error_validity(self) -> Mapping[str, bool]:
        valid = bool(self.success and self.covariance_valid)
        return MappingProxyType({name: valid for name in self.parameter_names})

    @property
    def sample_axis_name(self) -> str:
        return ""

    @property
    def sample_coordinates(self) -> np.ndarray:
        return _readonly(np.asarray((0.0,), dtype=np.float64))

    @property
    def sample_unit(self) -> str:
        return ""

    @property
    def sample_labels(self) -> None:
        return None

    @property
    def success_mask(self) -> np.ndarray:
        return _readonly(np.asarray((self.success,), dtype=np.bool_))

    @property
    def table(self) -> FitNumericTable:
        return _make_fit_numeric_table(
            parameter_names=self.parameter_names,
            parameter_units=self.parameter_units,
            parameter_values={
                name: np.asarray((float(self.parameter_values[index]),), dtype=np.float64)
                for index, name in enumerate(self.parameter_names)
            },
            parameter_errors={
                name: np.asarray((
                    float(self.standard_errors[index])
                    if self.covariance_valid
                    else np.nan,
                ), dtype=np.float64)
                for index, name in enumerate(self.parameter_names)
            },
            success=self.success_mask,
            sample_axis_name="",
            sample_coordinates=np.asarray((0.0,), dtype=np.float64),
            sample_unit="",
            sample_labels=None,
            source_revision=self.source_revision,
            batch_revision=self.batch_revision,
        )

    def with_parameter_units(self, units: Mapping[str, str]) -> "FitResult":
        return replace(self, parameter_units=units)


def _bimodal_classifier_metrics(
    result: FitResult,
    threshold: float | None = None,
) -> tuple[float, float, float, float]:
    """Return threshold, left/right correctness, and balanced fidelity."""

    if result.model.model_id != "bimodal_gaussian" or not result.success:
        raise ValueError("threshold classification requires a successful bimodal fit")
    values = result.parameters
    center = float(values["center"])
    separation = float(values["center_splitting"])
    left_mean = center - 0.5 * separation
    right_mean = center + 0.5 * separation
    left_sigma = max(abs(float(values["left_sigma"])), np.finfo(float).eps)
    right_sigma = max(abs(float(values["right_sigma"])), np.finfo(float).eps)

    def cdf(value: float, mean: float, sigma: float) -> float:
        return 0.5 * (
            1.0 + math.erf((value - mean) / (sigma * math.sqrt(2.0)))
        )

    def error(value: float) -> float:
        return 0.5 * (
            1.0 - cdf(value, left_mean, left_sigma)
            + cdf(value, right_mean, right_sigma)
        )

    if threshold is None:
        lower, upper = sorted((left_mean, right_mean))
        if upper <= lower:
            threshold = lower
        else:
            optimum = minimize_scalar(error, bounds=(lower, upper), method="bounded")
            threshold = float(
                optimum.x if optimum.success else 0.5 * (lower + upper)
            )
    else:
        threshold = float(threshold)
    left = cdf(threshold, left_mean, left_sigma)
    right = 1.0 - cdf(threshold, right_mean, right_sigma)
    return threshold, left, right, 0.5 * (left + right)


@dataclass(frozen=True, slots=True)
class FacetFitBatchResult:
    """Ordered fit results for every cell of one FacetGrid projection."""

    facet: AxisRef
    facet_values: tuple[Any, ...]
    model: FitModelSpec
    results: tuple[FitResult | None, ...]
    failure_messages: tuple[str | None, ...]
    source_revision: int
    overlays: tuple["FitOverlay", ...]
    parameter_units: Mapping[str, str] = field(default_factory=dict)
    sample_axis_name: str = ""
    sample_coordinates: np.ndarray | None = None
    sample_unit: str = ""
    sample_labels: tuple[str, ...] | None = None
    batch_revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.facet, AxisRef):
            raise TypeError("facet must be AxisRef")
        if not isinstance(self.model, FitModelSpec):
            raise TypeError("model must be FitModelSpec")
        values = tuple(self.facet_values)
        results = tuple(self.results)
        errors = tuple(self.failure_messages)
        overlays = tuple(self.overlays)
        if not values:
            raise ValueError("facet fit batch cannot be empty")
        if len(results) != len(values) or len(errors) != len(values):
            raise ValueError(
                "facet values, results and failure messages must have equal length"
            )
        if len(overlays) != len(values):
            raise ValueError("facet values and overlays must have equal length")
        from ._fit_scene import FitOverlay

        if any(not isinstance(overlay, FitOverlay) for overlay in overlays):
            raise TypeError("facet fit overlays must contain FitOverlay values")
        revision = integer(self.source_revision, "facet fit source_revision")
        if revision < 0:
            raise ValueError("facet fit source_revision must be non-negative")
        batch_revision = integer(self.batch_revision, "facet fit batch_revision")
        if batch_revision < 0:
            raise ValueError("facet fit batch_revision must be non-negative")
        normalized: list[str | None] = []
        for index, (result, error, overlay) in enumerate(
            zip(results, errors, overlays, strict=True)
        ):
            if result is not None:
                if not isinstance(result, FitResult):
                    raise TypeError("facet fit results must contain FitResult or None")
                if result.model != self.model or result.source_revision != revision:
                    raise ValueError("facet fit result model/revision mismatch")
                if error is not None:
                    raise ValueError(
                        "a facet fit cell cannot have both result and failure message"
                    )
                if overlay.facet_index != index:
                    raise ValueError("facet fit overlay index does not match its cell")
                normalized.append(None)
            else:
                if not isinstance(error, str) or not error.strip():
                    raise ValueError("a failed facet fit cell requires a failure message")
                if overlay.facet_index != index:
                    raise ValueError("facet fit overlay index does not match its cell")
                normalized.append(error.strip())
        object.__setattr__(self, "facet_values", values)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "failure_messages", tuple(normalized))
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "batch_revision", batch_revision)
        object.__setattr__(self, "overlays", overlays)
        names = self.model.parameter_names
        units = dict(self.parameter_units)
        unknown = set(units) - set(names)
        if unknown:
            raise ValueError(f"facet fit units name unknown parameters: {sorted(unknown)}")
        if not units:
            for overlay in overlays:
                if overlay.parameter_display:
                    units = {
                        parameter.name: parameter.unit
                        for parameter in overlay.parameter_display
                    }
                    break
        object.__setattr__(
            self,
            "parameter_units",
            MappingProxyType({
                name: _text(
                    units.get(name, ""),
                    f"fit parameter unit {name}",
                    allow_empty=True,
                )
                for name in names
            }),
        )
        axis_name = _text(
            self.sample_axis_name,
            "facet fit sample axis name",
            allow_empty=True,
        )
        raw_coordinates = self.sample_coordinates
        numeric_values = all(
            isinstance(value, (Real, np.number)) and not isinstance(value, (bool, np.bool_))
            for value in values
        )
        if raw_coordinates is None:
            if numeric_values:
                coordinates = np.asarray(values, dtype=np.float64)
                labels = None
            else:
                coordinates = np.arange(len(values), dtype=np.float64)
                labels = tuple(str(value) for value in values)
        else:
            coordinates = np.asarray(raw_coordinates, dtype=np.float64).reshape(-1)
            labels = None if self.sample_labels is None else tuple(self.sample_labels)
        if coordinates.shape != (len(values),) or not np.all(np.isfinite(coordinates)):
            raise ValueError("facet fit sample coordinates must match facet values")
        if labels is not None and len(labels) != len(values):
            raise ValueError("facet fit sample labels must match facet values")
        if numeric_values and labels is not None:
            raise ValueError("numeric facet coordinates cannot have sample labels")
        if not numeric_values:
            labels = tuple(
                _text(label, "facet fit sample label") for label in (labels or ())
            )
        sample_unit = _text(
            self.sample_unit,
            "facet fit sample unit",
            allow_empty=True,
        )
        if sample_unit == "1":
            sample_unit = ""
        if not numeric_values:
            sample_unit = ""
        if not axis_name:
            axis_name = self.facet.axis_id or self.facet.domain.value
        object.__setattr__(self, "sample_axis_name", axis_name)
        object.__setattr__(self, "sample_coordinates", _readonly(coordinates))
        object.__setattr__(self, "sample_unit", sample_unit)
        object.__setattr__(self, "sample_labels", labels)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.model.parameter_names

    @property
    def success(self) -> np.ndarray:
        values = np.asarray(
            tuple(result is not None and result.success for result in self.results),
            dtype=np.bool_,
        )
        values.setflags(write=False)
        return values

    @property
    def parameter_values(self) -> Mapping[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for index, name in enumerate(self.model.parameter_names):
            values = np.asarray(
                tuple(
                    np.nan
                    if fit is None or not fit.success
                    else float(fit.parameter_values[index])
                    for fit in self.results
                ),
                dtype=float,
            )
            values.setflags(write=False)
            result[name] = values
        return MappingProxyType(result)

    @property
    def parameter_errors(self) -> Mapping[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for index, name in enumerate(self.model.parameter_names):
            values = np.asarray(
                tuple(
                    np.nan
                    if fit is None or not fit.success or not fit.covariance_valid
                    else float(fit.standard_errors[index])
                    for fit in self.results
                ),
                dtype=np.float64,
            )
            values.setflags(write=False)
            result[name] = values
        return MappingProxyType(result)

    @property
    def parameter_error_validity(self) -> Mapping[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for name, errors in self.parameter_errors.items():
            values = np.asarray(self.success & np.isfinite(errors), dtype=np.bool_)
            values.setflags(write=False)
            result[name] = values
        return MappingProxyType(result)

    @property
    def table(self) -> FitNumericTable:
        return _make_fit_numeric_table(
            parameter_names=self.parameter_names,
            parameter_units=self.parameter_units,
            parameter_values=self.parameter_values,
            parameter_errors=self.parameter_errors,
            success=self.success,
            sample_axis_name=self.sample_axis_name,
            sample_coordinates=self.sample_coordinates,
            sample_unit=self.sample_unit,
            sample_labels=self.sample_labels,
            source_revision=self.source_revision,
            batch_revision=self.batch_revision,
        )


class FitEngine:
    def __init__(self, registry: FitModelRegistry | None = None) -> None:
        self.registry = registry or default_fit_registry()

    def fit(
        self,
        model: str | FitModelSpec,
        coordinates: Sequence[np.ndarray] | RegularImageFitInput,
        observations: np.ndarray | None = None,
        *,
        selected_indices: np.ndarray | None = None,
        data_revision: int = 0,
        initial: Mapping[str, float] | Sequence[float] | None = None,
        warm_start: Mapping[str, float] | Sequence[float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        options: FitOptions | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> FitResult:
        spec = self.registry.get(model) if isinstance(model, str) else model
        if not isinstance(spec, FitModelSpec):
            raise TypeError("model must be a registered id or FitModelSpec")
        opts = options or FitOptions()
        if isinstance(coordinates, RegularImageFitInput):
            if observations is not None:
                raise TypeError(
                    "observations belong inside RegularImageFitInput and must not "
                    "also be passed separately"
                )
            if selected_indices is not None:
                raise TypeError(
                    "regular-image selected_indices belong inside "
                    "RegularImageFitInput"
                )
            if "regular_image_radial" not in spec.capabilities:
                raise ValueError(
                    "this model does not declare regular-image radial capability"
                )
            from ._fit_radial import fit_regular_radial_image

            return fit_regular_radial_image(
                spec,
                coordinates,
                data_revision=data_revision,
                initial=initial,
                warm_start=warm_start,
                bounds=bounds,
                options=opts,
                cancelled=cancelled,
            )
        if observations is None:
            raise TypeError("observations are required for coordinate-array fitting")

        coords = _coordinate_arrays(tuple(coordinates), spec.independent_arity)
        values = np.asarray(observations, dtype=np.float64).reshape(-1)
        if any(item.shape != values.shape for item in coords):
            raise ValueError("coordinates and observations must have equal flattened shape")
        if selected_indices is None:
            indices = np.arange(values.size, dtype=np.int64)
        else:
            indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
            if indices.size != values.size:
                raise ValueError("selected_indices must match observations")
        finite = np.isfinite(values)
        for coordinate in coords:
            finite &= np.isfinite(coordinate)
        if not bool(np.all(finite)):
            coords = tuple(item[finite] for item in coords)
            values = values[finite]
            indices = indices[finite]
        if values.size <= len(spec.parameters):
            raise ValueError("fit requires more finite observations than parameters")
        default_bounds = (
            spec.bounds_initializer(coords, values)
            if spec.bounds_initializer is not None
            else None
        )
        lower, upper = _solver_bounds(spec, default_bounds, bounds)
        seeds = _initial_candidates(spec, coords, values, initial, warm_start)
        low_inside = np.nextafter(lower, upper)
        high_inside = np.nextafter(upper, lower)
        seeds = tuple(np.minimum(np.maximum(seed, low_inside), high_inside) for seed in seeds)
        start = time.monotonic()
        invalid_residual = np.finfo(np.float64).max ** 0.25

        def check() -> None:
            if cancelled is not None and cancelled():
                raise FitCancelled("fit cancelled")
            if opts.deadline_seconds is not None and time.monotonic() - start > opts.deadline_seconds:
                raise FitDeadlineExceeded("fit deadline exceeded")

        def residual(parameters: np.ndarray) -> np.ndarray:
            check()
            predicted = spec.evaluate(coords, parameters).reshape(-1)
            if predicted.shape != values.shape or not np.all(np.isfinite(predicted)):
                return np.full(values.shape, invalid_residual)
            return predicted - values

        def analytic_jacobian(parameters: np.ndarray) -> np.ndarray:
            check()
            jacobian = spec.evaluate_jacobian(coords, parameters)
            if not np.all(np.isfinite(jacobian)):
                raise FloatingPointError("analytic fit jacobian is non-finite")
            return jacobian

        successful: list[tuple[float, Any, np.ndarray, np.ndarray]] = []
        unsuccessful: list[tuple[float, Any, np.ndarray, np.ndarray]] = []
        last_error: Exception | None = None
        for seed in seeds:
            check()
            try:
                candidate = least_squares(
                    residual,
                    seed,
                    bounds=(lower, upper),
                    loss=opts.loss,
                    max_nfev=opts.max_nfev,
                    x_scale="jac",
                    jac=(analytic_jacobian if spec.jacobian is not None else "2-point"),
                )
                check()
                solver_residual = np.asarray(candidate.fun, dtype=np.float64).reshape(-1)
                if (
                    solver_residual.shape != values.shape
                    or not np.all(np.isfinite(solver_residual))
                    or bool(np.all(solver_residual == invalid_residual))
                ):
                    continue
                fitted_candidate = values + solver_residual
                if not np.all(np.isfinite(fitted_candidate)):
                    continue
                residual_candidate = -solver_residual
                rss = float(np.dot(residual_candidate, residual_candidate))
                if not math.isfinite(rss):
                    continue
                (successful if candidate.success else unsuccessful).append(
                    (rss, candidate, fitted_candidate, residual_candidate)
                )
            except (FitCancelled, FitDeadlineExceeded):
                raise
            except (ValueError, RuntimeError, FloatingPointError) as error:
                last_error = error
        candidates = successful or unsuccessful
        if not candidates:
            if last_error is not None:
                raise last_error
            raise RuntimeError("least-squares failed for every initializer")
        best = candidates[0]
        for candidate in candidates[1:]:
            scale = max(1.0, abs(best[0]))
            if candidate[0] < best[0] - _FIT_RSS_TIE_RELATIVE * scale:
                best = candidate
        _rss, solved, fitted, residuals = best
        degrees = max(values.size - solved.x.size, 1)
        reduced = float(np.dot(residuals, residuals) / degrees)
        covariance, covariance_valid = _covariance(solved.jac, reduced)
        errors = (
            np.sqrt(np.maximum(np.diag(covariance), 0.0))
            if covariance_valid
            else np.full(solved.x.size, np.nan, dtype=np.float64)
        )
        return FitResult(
            spec,
            solved.x,
            errors,
            covariance,
            fitted,
            residuals,
            indices,
            data_revision,
            bool(solved.success),
            solved.message,
            reduced,
            covariance_valid=covariance_valid,
        )



def _coordinate_arrays(coordinates: Sequence[np.ndarray], arity: int) -> ArrayTuple:
    arrays = tuple(np.asarray(item, dtype=np.float64).reshape(-1) for item in coordinates)
    if len(arrays) != arity:
        raise ValueError("coordinate arity does not match fit model")
    if arrays and any(item.shape != arrays[0].shape for item in arrays):
        raise ValueError("coordinate arrays must have equal shape")
    return arrays


def _solver_bounds(
    model: FitModelSpec,
    defaults: Mapping[str, tuple[float | None, float | None]] | None,
    requested: Mapping[str, tuple[float | None, float | None]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    names = set(model.parameter_names)
    for label, source in (("default bounds", defaults), ("bounds", requested)):
        unknown = set(source or ()) - names
        if unknown:
            raise ValueError(f"{label} name unknown parameters: {sorted(unknown)}")
    for parameter in model.parameters:
        low, high = parameter.bounds
        source = (
            requested
            if requested is not None and parameter.name in requested
            else defaults
        )
        if source is not None and parameter.name in source:
            source_low, source_high = source[parameter.name]
            if source_low is not None:
                low = max(low, float(source_low))
            if source_high is not None:
                high = min(high, float(source_high))
        if not low < high:
            raise ValueError(f"empty bounds for parameter {parameter.name!r}")
        lower.append(low)
        upper.append(high)
    return np.asarray(lower), np.asarray(upper)


def _initial_values(
    model: FitModelSpec,
    coordinates: ArrayTuple,
    values: np.ndarray,
    initial: Mapping[str, float] | Sequence[float] | None,
) -> np.ndarray:
    if initial is None:
        seed = np.asarray(model.initializer(coordinates, values), dtype=np.float64).reshape(-1)
    elif isinstance(initial, Mapping):
        unknown = set(initial) - set(model.parameter_names)
        if unknown:
            raise ValueError(f"initial values name unknown parameters: {sorted(unknown)}")
        defaults = dict(zip(model.parameter_names, model.initializer(coordinates, values), strict=True))
        defaults.update({key: float(value) for key, value in initial.items()})
        seed = np.asarray([defaults[name] for name in model.parameter_names], dtype=np.float64)
    else:
        seed = np.asarray(initial, dtype=np.float64).reshape(-1)
    if seed.shape != (len(model.parameters),) or not np.all(np.isfinite(seed)):
        raise ValueError("fit initializer returned invalid parameter values")
    return seed


def _initial_candidates(
    model: FitModelSpec,
    coordinates: ArrayTuple,
    values: np.ndarray,
    initial: Mapping[str, float] | Sequence[float] | None,
    warm_start: Mapping[str, float] | Sequence[float] | None = None,
) -> tuple[np.ndarray, ...]:
    if warm_start is not None:
        seeds = [_initial_values(model, coordinates, values, warm_start)]
        seeds.extend(_initial_candidates(model, coordinates, values, initial))
        unique: list[np.ndarray] = []
        seen: set[bytes] = set()
        for seed in seeds:
            key = seed.tobytes()
            if key not in seen:
                seen.add(key)
                unique.append(seed)
        return tuple(unique)
    if initial is not None or model.candidate_initializer is None:
        return (_initial_values(model, coordinates, values, initial),)
    candidates = tuple(model.candidate_initializer(coordinates, values))
    if not candidates:
        raise ValueError("fit initializer returned no candidates")
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for candidate in candidates:
        seed = _initial_values(model, coordinates, values, candidate)
        key = seed.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(seed)
    return tuple(unique)


def _covariance(
    jacobian: np.ndarray,
    reduced_chi_square: float,
) -> tuple[np.ndarray, bool]:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    parameter_count = jacobian.shape[1] if jacobian.ndim == 2 else 0
    invalid = np.full((parameter_count, parameter_count), np.nan, dtype=np.float64)
    reduced_chi_square = float(reduced_chi_square)
    if (
        jacobian.ndim != 2
        or parameter_count == 0
        or jacobian.shape[0] <= parameter_count
        or not np.all(np.isfinite(jacobian))
        or not math.isfinite(reduced_chi_square)
        or reduced_chi_square < 0.0
    ):
        return invalid, False
    try:
        _left, singular_values, right = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError:
        return invalid, False
    if singular_values.size != parameter_count or not np.all(np.isfinite(singular_values)):
        return invalid, False
    tolerance = (
        np.finfo(np.float64).eps
        * max(jacobian.shape)
        * float(singular_values[0])
    )
    if singular_values[-1] <= tolerance:
        return invalid, False
    inverse_information = (right.T / singular_values**2) @ right
    covariance = inverse_information * reduced_chi_square
    covariance = (covariance + covariance.T) / 2.0
    diagonal = np.diag(covariance)
    if not np.all(np.isfinite(covariance)) or np.any(diagonal < 0.0):
        return invalid, False
    return covariance, True


def _covariance_from_information(
    information: np.ndarray,
    reduced_chi_square: float,
    observation_count: int,
) -> tuple[np.ndarray, bool]:
    matrix = np.asarray(information, dtype=np.float64)
    count = matrix.shape[0] if matrix.ndim == 2 else 0
    invalid = np.full((count, count), np.nan, dtype=np.float64)
    if (
        matrix.shape != (count, count)
        or not count
        or observation_count <= count
        or not np.all(np.isfinite(matrix))
        or not math.isfinite(reduced_chi_square)
        or reduced_chi_square < 0.0
    ):
        return invalid, False
    norms = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    if np.any(norms <= np.finfo(np.float64).tiny):
        return invalid, False
    normalized = matrix / np.outer(norms, norms)
    try:
        values, vectors = np.linalg.eigh((normalized + normalized.T) / 2.0)
    except np.linalg.LinAlgError:
        return invalid, False
    tolerance = np.finfo(np.float64).eps * count * float(np.max(values))
    if values[0] <= tolerance:
        return invalid, False
    covariance = (vectors / values) @ vectors.T
    covariance *= float(reduced_chi_square) / np.outer(norms, norms)
    covariance = (covariance + covariance.T) / 2.0
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) < 0.0):
        return invalid, False
    return covariance, True


VALUE = UnitRelation.VALUE
AXIS_0 = UnitRelation.AXIS_0
AXIS_1 = UnitRelation.AXIS_1
INVERSE_AXIS_0 = UnitRelation.INVERSE_AXIS_0
RADIAN = UnitRelation.RADIAN
REAL = ParameterDomain.REAL
POSITIVE = ParameterDomain.POSITIVE
NONNEGATIVE = ParameterDomain.NONNEGATIVE
PHASE = ParameterDomain.PHASE_RADIANS


def _span(x: np.ndarray) -> float:
    return max(float(np.ptp(x)), np.finfo(np.float64).eps)


def _edge_offset(y: np.ndarray) -> float:
    count = max(1, min(y.size // 10, 20))
    return float(np.median(np.concatenate((y[:count], y[-count:]))))


def _peak_seed(coords: ArrayTuple, y: np.ndarray) -> tuple[float, float, float, float]:
    x = coords[0]
    offset = _edge_offset(y)
    delta = y - offset
    index = int(np.argmax(np.abs(delta)))
    amplitude = float(delta[index])
    if amplitude == 0:
        amplitude = float(np.ptp(y) or 1.0)
    center = float(x[index])
    return amplitude, offset, center, _span(x)


def _data_interval(values: np.ndarray) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    if low < high:
        return low, high
    padding = math.sqrt(np.finfo(np.float64).eps) * max(abs(low), 1.0)
    return low - padding, high + padding


def _value_range(values: np.ndarray) -> float:
    low, high = _data_interval(values)
    return high - low


def _lorentzian(x, center, fwhm, amplitude, offset):
    half_squared = (fwhm / 2.0) ** 2
    return amplitude * half_squared / ((x - center) ** 2 + half_squared) + offset


def _lorentzian_jacobian(x, center, fwhm, amplitude, offset):
    half = fwhm / 2.0
    half_squared = half**2
    delta = x - center
    denominator = delta**2 + half_squared
    denominator_squared = denominator**2
    return np.column_stack((
        amplitude * half_squared * 2.0 * delta / denominator_squared,
        amplitude * half * delta**2 / denominator_squared,
        half_squared / denominator,
        np.ones_like(x, dtype=float),
    ))


def _gaussian_offset(x, amplitude, offset, sigma, center):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset


def _gaussian_offset_jacobian(x, amplitude, offset, sigma, center):
    delta = x - center
    gaussian = np.exp(-0.5 * (delta / sigma) ** 2)
    return np.column_stack((
        gaussian,
        np.ones_like(x, dtype=float),
        amplitude * gaussian * delta**2 / sigma**3,
        amplitude * gaussian * delta / sigma**2,
    ))


def _histogram_gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _histogram_gaussian_jacobian(x, amplitude, center, sigma):
    delta = x - center
    gaussian = np.exp(-0.5 * (delta / sigma) ** 2)
    return np.column_stack((
        gaussian,
        amplitude * gaussian * delta / sigma**2,
        amplitude * gaussian * delta**2 / sigma**3,
    ))


def _bimodal_left(
    x,
    center,
    center_splitting,
    left_amplitude,
    left_sigma,
    _right_amplitude,
    _right_sigma,
):
    return _histogram_gaussian(
        x,
        left_amplitude,
        center - center_splitting / 2.0,
        left_sigma,
    )


def _bimodal_right(
    x,
    center,
    center_splitting,
    _left_amplitude,
    _left_sigma,
    right_amplitude,
    right_sigma,
):
    return _histogram_gaussian(
        x,
        right_amplitude,
        center + center_splitting / 2.0,
        right_sigma,
    )


def _bimodal_gaussian(
    x,
    center,
    center_splitting,
    left_amplitude,
    left_sigma,
    right_amplitude,
    right_sigma,
):
    parameters = (
        center,
        center_splitting,
        left_amplitude,
        left_sigma,
        right_amplitude,
        right_sigma,
    )
    return _bimodal_left(x, *parameters) + _bimodal_right(x, *parameters)


def _bimodal_gaussian_jacobian(
    x,
    center,
    center_splitting,
    left_amplitude,
    left_sigma,
    right_amplitude,
    right_sigma,
):
    left_center = center - center_splitting / 2.0
    right_center = center + center_splitting / 2.0
    left_delta = x - left_center
    right_delta = x - right_center
    left_gaussian = np.exp(-0.5 * (left_delta / left_sigma) ** 2)
    right_gaussian = np.exp(-0.5 * (right_delta / right_sigma) ** 2)
    left_center_derivative = left_amplitude * left_gaussian * left_delta / left_sigma**2
    right_center_derivative = right_amplitude * right_gaussian * right_delta / right_sigma**2
    return np.column_stack((
        left_center_derivative + right_center_derivative,
        -0.5 * left_center_derivative + 0.5 * right_center_derivative,
        left_gaussian,
        left_amplitude * left_gaussian * left_delta**2 / left_sigma**3,
        right_gaussian,
        right_amplitude * right_gaussian * right_delta**2 / right_sigma**3,
    ))


def _symmetric_lorentzian_doublet(x, center, common_fwhm, component_amplitude, offset, center_splitting):
    return _lorentzian(x, center - center_splitting / 2, common_fwhm, component_amplitude, offset) + _lorentzian(
        x, center + center_splitting / 2, common_fwhm, component_amplitude, 0.0
    )


def _symmetric_lorentzian_doublet_jacobian(
    x,
    center,
    common_fwhm,
    component_amplitude,
    offset,
    center_splitting,
):
    left = _lorentzian_jacobian(
        x,
        center - center_splitting / 2.0,
        common_fwhm,
        component_amplitude,
        offset,
    )
    right = _lorentzian_jacobian(
        x,
        center + center_splitting / 2.0,
        common_fwhm,
        component_amplitude,
        0.0,
    )
    return np.column_stack((
        left[:, 0] + right[:, 0],
        left[:, 1] + right[:, 1],
        left[:, 2] + right[:, 2],
        left[:, 3],
        -0.5 * left[:, 0] + 0.5 * right[:, 0],
    ))


def _damped_sine(x, amplitude, offset, baseband_frequency, decay_time, phase):
    return offset + amplitude * np.exp(-x / decay_time) * np.sin(
        2 * np.pi * baseband_frequency * x + phase
    )


def _damped_sine_jacobian(x, amplitude, offset, baseband_frequency, decay_time, phase):
    exponential = np.exp(-x / decay_time)
    argument = 2.0 * np.pi * baseband_frequency * x + phase
    sine = np.sin(argument)
    cosine = np.cos(argument)
    return np.column_stack((
        exponential * sine,
        np.ones_like(x, dtype=float),
        amplitude * exponential * cosine * 2.0 * np.pi * x,
        amplitude * exponential * sine * x / decay_time**2,
        amplitude * exponential * cosine,
    ))


def _exponential_decay(x, amplitude, offset, decay_time):
    return offset + amplitude * np.exp(-x / decay_time)


def _exponential_decay_jacobian(x, amplitude, offset, decay_time):
    exponential = np.exp(-x / decay_time)
    return np.column_stack((
        exponential,
        np.ones_like(x, dtype=float),
        amplitude * exponential * x / decay_time**2,
    ))


def _radial_gaussian_center(x, y, amplitude, offset, one_over_e_radius, center_x, center_y):
    radius_squared = (x - center_x) ** 2 + (y - center_y) ** 2
    return offset + amplitude * np.exp(-radius_squared / one_over_e_radius**2)


def _radial_gaussian_center_jacobian(
    x,
    y,
    amplitude,
    offset,
    one_over_e_radius,
    center_x,
    center_y,
):
    delta_x = x - center_x
    delta_y = y - center_y
    radius_squared = delta_x**2 + delta_y**2
    gaussian = np.exp(-radius_squared / one_over_e_radius**2)
    return np.column_stack((
        gaussian,
        np.ones_like(x, dtype=float),
        amplitude * gaussian * 2.0 * radius_squared / one_over_e_radius**3,
        amplitude * gaussian * 2.0 * delta_x / one_over_e_radius**2,
        amplitude * gaussian * 2.0 * delta_y / one_over_e_radius**2,
    ))


def _anisotropic_gaussian_center(
    x,
    y,
    amplitude,
    offset,
    radius_x,
    radius_y,
    center_x,
    center_y,
):
    exponent = (
        (x - center_x) ** 2 / radius_x**2
        + (y - center_y) ** 2 / radius_y**2
    )
    return offset + amplitude * np.exp(-exponent)


def _anisotropic_gaussian_center_jacobian(
    x,
    y,
    amplitude,
    offset,
    radius_x,
    radius_y,
    center_x,
    center_y,
):
    delta_x = x - center_x
    delta_y = y - center_y
    gaussian = np.exp(
        -(delta_x**2 / radius_x**2 + delta_y**2 / radius_y**2)
    )
    return np.column_stack((
        gaussian,
        np.ones_like(x, dtype=float),
        amplitude * gaussian * 2.0 * delta_x**2 / radius_x**3,
        amplitude * gaussian * 2.0 * delta_y**2 / radius_y**3,
        amplitude * gaussian * 2.0 * delta_x / radius_x**2,
        amplitude * gaussian * 2.0 * delta_y / radius_y**2,
    ))


def _init_lorentzian(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    return _lorentzian_candidates(coords, y)[0]


def _lorentzian_candidates(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Sequence[Sequence[float]]:
    x = coords[0]
    width = _span(x) / 4.0
    y_range = _value_range(y)
    low_y, high_y = float(np.min(y)), float(np.max(y))
    return (
        (float(x[np.argmax(y)]), width, y_range, low_y),
        (float(x[np.argmin(y)]), width, -y_range, high_y),
    )


def _lorentzian_bounds(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Mapping[str, tuple[float | None, float | None]]:
    x_low, x_high = _data_interval(coords[0])
    y_low, y_high = _data_interval(y)
    x_span = _span(coords[0])
    y_range = _value_range(y)
    width = x_span / 4.0
    return {
        "center": (x_low, x_high),
        "fwhm": (width / 10.0, width * 10.0),
        "amplitude": (-10.0 * y_range, 10.0 * y_range),
        "offset": (y_low - 10.0 * y_range, y_high + 10.0 * y_range),
    }


def _init_gaussian(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    amplitude, offset, center, span = _peak_seed(coords, y)
    return amplitude, offset, span / 6.0, center


def _init_histogram(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    x = coords[0]
    weights = np.maximum(y, 0)
    total = float(np.sum(weights))
    if total <= 0:
        center = float(np.mean(x))
        sigma = _span(x) / 6
    else:
        center = float(np.sum(x * weights) / total)
        sigma = max(float(np.sqrt(np.sum(weights * (x - center) ** 2) / total)), _span(x) / 1000)
    return max(float(np.max(y)), 0.0), center, sigma


def _init_bimodal(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    x = coords[0]
    midpoint = float((np.min(x) + np.max(x)) / 2)
    left = y[x <= midpoint]
    right = y[x > midpoint]
    left_x = x[x <= midpoint]
    right_x = x[x > midpoint]
    span = _span(x)
    lc = float(left_x[np.argmax(left)]) if left.size else midpoint - span / 4
    rc = float(right_x[np.argmax(right)]) if right.size else midpoint + span / 4
    la = max(float(np.max(left)) if left.size else float(np.max(y)), 0.0)
    ra = max(float(np.max(right)) if right.size else float(np.max(y)), 0.0)
    return (lc + rc) / 2, abs(rc - lc), la, span / 10, ra, span / 10


def _init_doublet(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    return _doublet_candidates(coords, y)[0]


def _doublet_candidates(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Sequence[Sequence[float]]:
    order = np.argsort(coords[0], kind="stable")
    x = coords[0][order]
    ordered_y = y[order]
    x_span = _span(x)
    y_range = _value_range(ordered_y)
    unique_x = np.unique(x)
    step = (
        float(np.median(np.abs(np.diff(unique_x))))
        if unique_x.size > 1
        else x_span
    )
    step = max(step, np.finfo(np.float64).eps)
    seeds: list[tuple[float, ...]] = []
    for sign in (1.0, -1.0):
        signed = sign * ordered_y
        peaks, properties = find_peaks(
            signed,
            width=1,
            prominence=y_range / 8.0,
        )
        if peaks.size == 0:
            continue
        strongest = peaks[np.argsort(signed[peaks])[::-1]][:4]
        widths = properties.get("widths", np.ones(peaks.size))
        width_by_peak = {
            int(peak): max(float(widths[index]) * step, step)
            for index, peak in enumerate(peaks)
        }
        first = int(strongest[0])
        for second_raw in strongest:
            second = int(second_raw)
            width = width_by_peak[first]
            seeds.append(
                (
                    float((x[first] + x[second]) / 2.0),
                    width,
                    sign * y_range,
                    float(np.min(ordered_y) if sign > 0.0 else np.max(ordered_y)),
                    float(abs(x[second] - x[first])),
                )
            )
    if seeds:
        return tuple(seeds)
    width = x_span / 8.0
    return (
        (
            float(x[np.argmax(ordered_y)]),
            width,
            y_range,
            float(np.min(ordered_y)),
            width * 2.0,
        ),
        (
            float(x[np.argmin(ordered_y)]),
            width,
            -y_range,
            float(np.max(ordered_y)),
            width * 2.0,
        ),
    )


def _doublet_bounds(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Mapping[str, tuple[float | None, float | None]]:
    x_low, x_high = _data_interval(coords[0])
    y_low, y_high = _data_interval(y)
    x_span = _span(coords[0])
    y_range = _value_range(y)
    width = float(_doublet_candidates(coords, y)[0][1])
    return {
        "center": (x_low, x_high),
        "common_fwhm": (width / 10.0, width * 10.0),
        "component_amplitude": (-10.0 * y_range, 10.0 * y_range),
        "offset": (y_low - 10.0 * y_range, y_high + 10.0 * y_range),
        "center_splitting": (0.0, 2.0 * x_span),
    }


def _init_damped_sine(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    x = coords[0]
    order = np.argsort(x)
    sx = x[order]
    sy = y[order]
    offset = float(np.mean(sy))
    amplitude = max(float(np.ptp(sy) / 2), np.finfo(np.float64).eps)
    spacing = float(np.median(np.diff(sx))) if sx.size > 1 else 1.0
    spectrum = np.abs(np.fft.rfft(sy - offset))
    frequencies = np.fft.rfftfreq(sy.size, d=max(abs(spacing), np.finfo(float).eps))
    frequency = float(frequencies[1 + np.argmax(spectrum[1:])]) if spectrum.size > 1 else 1 / _span(x)
    frequency = max(frequency, np.finfo(float).eps)
    decay_time = _span(x)
    origin = float(np.min(x))
    exponent = origin / decay_time
    if abs(exponent) < 700.0:
        amplitude *= math.exp(exponent)
    phase = float(
        (-2.0 * np.pi * frequency * origin + np.pi) % (2.0 * np.pi) - np.pi
    )
    return amplitude, offset, frequency, decay_time, phase


def _damped_sine_candidates(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Sequence[Sequence[float]]:
    amplitude, offset, frequency, decay_time, phase = _init_damped_sine(coords, y)
    return tuple(
        (
            amplitude,
            offset,
            frequency,
            decay_time,
            float((phase + shift + np.pi) % (2.0 * np.pi) - np.pi),
        )
        for shift in (-np.pi / 2.0, 0.0, np.pi / 2.0)
    )


def _damped_sine_bounds(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Mapping[str, tuple[float | None, float | None]]:
    amplitude, _offset, frequency, decay_time, _phase = _init_damped_sine(coords, y)
    y_low, y_high = _data_interval(y)
    return {
        "amplitude": (amplitude / 5.0, amplitude * 5.0),
        "offset": (y_low, y_high),
        "baseband_frequency": (frequency / 5.0, frequency * 5.0),
        "decay_time": (decay_time / 5.0, decay_time * 5.0),
    }


def _init_exponential(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    x = coords[0]
    offset = float(np.median(y[np.argsort(x)[-max(1, y.size // 10) :]]))
    decay_time = _span(x) / 3
    observed_amplitude = float(y[np.argmin(x)] - offset)
    exponent = float(np.min(x)) / decay_time
    amplitude = (
        observed_amplitude * math.exp(exponent)
        if abs(exponent) < 700.0
        else observed_amplitude
    )
    return amplitude or float(np.ptp(y) or 1.0), offset, decay_time


def _exponential_candidates(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Sequence[Sequence[float]]:
    amplitude, offset, decay_time = _init_exponential(coords, y)
    return (
        (amplitude, offset, decay_time),
        (-amplitude, offset, decay_time),
    )


def _exponential_bounds(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Mapping[str, tuple[float | None, float | None]]:
    amplitude, _offset, decay_time = _init_exponential(coords, y)
    y_low, y_high = _data_interval(y)
    y_range = _value_range(y)
    amplitude_limit = max(4.0 * y_range, 10.0 * abs(amplitude))
    return {
        "amplitude": (-amplitude_limit, amplitude_limit),
        "offset": (y_low - 10.0 * y_range, y_high + 10.0 * y_range),
        "decay_time": (decay_time / 10.0, decay_time * 10.0),
    }


def _init_radial(coords: ArrayTuple, values: np.ndarray) -> Sequence[float]:
    return _radial_seed(coords, values, 1.0)


def _radial_seed(
    coords: ArrayTuple,
    values: np.ndarray,
    sign: float,
) -> tuple[float, ...]:
    x, y = coords
    offset = float(np.median(values))
    weights = np.maximum(sign * (values - offset), 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        center_x, center_y = float(np.mean(x)), float(np.mean(y))
        radius = max(_span(x), _span(y)) / 4
    else:
        center_x = float(np.sum(x * weights) / total)
        center_y = float(np.sum(y * weights) / total)
        radius = max(
            float(np.sqrt(np.sum(weights * ((x - center_x) ** 2 + (y - center_y) ** 2)) / total)),
            np.finfo(float).eps,
        )
    amplitude = (
        float(np.max(values) - offset)
        if sign > 0.0
        else float(np.min(values) - offset)
    )
    if amplitude == 0.0:
        amplitude = sign * _value_range(values)
    return amplitude, offset, radius, center_x, center_y


def _radial_candidates(
    coords: ArrayTuple,
    values: np.ndarray,
) -> Sequence[Sequence[float]]:
    return (
        _radial_seed(coords, values, 1.0),
        _radial_seed(coords, values, -1.0),
    )


def _radial_bounds(
    coords: ArrayTuple,
    values: np.ndarray,
) -> Mapping[str, tuple[float | None, float | None]]:
    x_low, x_high = _data_interval(coords[0])
    y_low, y_high = _data_interval(coords[1])
    value_low, value_high = _data_interval(values)
    value_range = _value_range(values)
    radii = [float(seed[2]) for seed in _radial_candidates(coords, values)]
    radius_low = max(min(radii) / 10.0, np.finfo(np.float64).eps)
    radius_high = max(radii) * 10.0
    return {
        "amplitude": (-4.0 * value_range, 4.0 * value_range),
        "offset": (value_low - value_range, value_high + value_range),
        "one_over_e_radius": (radius_low, radius_high),
        "center_x": (x_low, x_high),
        "center_y": (y_low, y_high),
    }


def _anisotropic_seed(
    coords: ArrayTuple,
    values: np.ndarray,
    sign: float,
) -> tuple[float, ...]:
    x, y = coords
    offset = float(np.median(values))
    weights = np.maximum(sign * (values - offset), 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        center_x, center_y = float(np.mean(x)), float(np.mean(y))
        radius_x, radius_y = _span(x) / 4.0, _span(y) / 4.0
    else:
        center_x = float(np.sum(x * weights) / total)
        center_y = float(np.sum(y * weights) / total)
        radius_x = float(
            np.sqrt(np.sum(weights * (x - center_x) ** 2) / total)
        )
        radius_y = float(
            np.sqrt(np.sum(weights * (y - center_y) ** 2) / total)
        )
    epsilon = np.finfo(np.float64).eps
    radius_x = max(radius_x, epsilon)
    radius_y = max(radius_y, epsilon)
    amplitude = (
        float(np.max(values) - offset)
        if sign > 0.0
        else float(np.min(values) - offset)
    )
    if amplitude == 0.0:
        amplitude = sign * _value_range(values)
    return amplitude, offset, radius_x, radius_y, center_x, center_y


def _anisotropic_candidates(
    coords: ArrayTuple,
    values: np.ndarray,
) -> Sequence[Sequence[float]]:
    return (
        _anisotropic_seed(coords, values, 1.0),
        _anisotropic_seed(coords, values, -1.0),
    )


def _init_anisotropic(coords: ArrayTuple, values: np.ndarray) -> Sequence[float]:
    return _anisotropic_seed(coords, values, 1.0)


def _anisotropic_bounds(
    coords: ArrayTuple,
    values: np.ndarray,
) -> Mapping[str, tuple[float | None, float | None]]:
    x_low, x_high = _data_interval(coords[0])
    y_low, y_high = _data_interval(coords[1])
    value_low, value_high = _data_interval(values)
    value_range = _value_range(values)
    seeds = _anisotropic_candidates(coords, values)
    radius_x = max(float(seed[2]) for seed in seeds)
    radius_y = max(float(seed[3]) for seed in seeds)
    epsilon = np.finfo(np.float64).eps
    return {
        "amplitude": (-4.0 * value_range, 4.0 * value_range),
        "offset": (value_low - value_range, value_high + value_range),
        "radius_x": (max(radius_x / 10.0, epsilon), radius_x * 10.0),
        "radius_y": (max(radius_y / 10.0, epsilon), radius_y * 10.0),
        "center_x": (x_low, x_high),
        "center_y": (y_low, y_high),
    }


def builtin_fit_models() -> tuple[FitModelSpec, ...]:
    return (
        FitModelSpec(
            "lorentzian",
            "Lorentzian",
            1,
            (
                FitParameterSpec(
                    "center", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
                FitParameterSpec(
                    "fwhm", AXIS_0, POSITIVE, display_label=r"$\mathrm{FWHM}$"
                ),
                FitParameterSpec("amplitude", VALUE, display_label=r"$H$"),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
            ),
            "center",
            _lorentzian,
            _init_lorentzian,
            (FitTarget.SERIES,),
            formula=(
                r"$f(x)=H\frac{(\mathrm{FWHM}/2)^2}"
                r"{(x-x_0)^2+(\mathrm{FWHM}/2)^2}+B$"
            ),
            jacobian=_lorentzian_jacobian,
            candidate_initializer=_lorentzian_candidates,
            bounds_initializer=_lorentzian_bounds,
            default_for=(FitTarget.SERIES,),
        ),
        FitModelSpec(
            "gaussian_offset",
            "Gaussian with offset",
            1,
            (
                FitParameterSpec("amplitude", VALUE, display_label=r"$A$"),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
                FitParameterSpec(
                    "sigma", AXIS_0, POSITIVE, display_label=r"$\sigma$"
                ),
                FitParameterSpec(
                    "center", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
            ),
            "center",
            _gaussian_offset,
            _init_gaussian,
            (FitTarget.SERIES,),
            formula=r"$f(x)=A e^{-\frac{1}{2}((x-x_0)/\sigma)^2}+B$",
            jacobian=_gaussian_offset_jacobian,
        ),
        FitModelSpec(
            "histogram_gaussian",
            "Single Gaussian",
            1,
            (
                FitParameterSpec(
                    "amplitude", VALUE, NONNEGATIVE, display_label=r"$A$"
                ),
                FitParameterSpec(
                    "center", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
                FitParameterSpec(
                    "sigma", AXIS_0, POSITIVE, display_label=r"$\sigma$"
                ),
            ),
            "center",
            _histogram_gaussian,
            _init_histogram,
            (FitTarget.HISTOGRAM,),
            formula=r"$f(x)=A e^{-\frac{1}{2}((x-x_0)/\sigma)^2}$",
            jacobian=_histogram_gaussian_jacobian,
        ),
        FitModelSpec(
            "bimodal_gaussian",
            "Bimodal Gaussian",
            1,
            (
                FitParameterSpec(
                    "center", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
                FitParameterSpec(
                    "center_splitting",
                    AXIS_0,
                    NONNEGATIVE,
                    display_label=r"$\delta$",
                ),
                FitParameterSpec(
                    "left_amplitude", VALUE, NONNEGATIVE, display_label=r"$A_L$"
                ),
                FitParameterSpec(
                    "left_sigma", AXIS_0, POSITIVE, display_label=r"$\sigma_L$"
                ),
                FitParameterSpec(
                    "right_amplitude", VALUE, NONNEGATIVE, display_label=r"$A_R$"
                ),
                FitParameterSpec(
                    "right_sigma", AXIS_0, POSITIVE, display_label=r"$\sigma_R$"
                ),
            ),
            "center",
            _bimodal_gaussian,
            _init_bimodal,
            (FitTarget.HISTOGRAM,),
            formula=(
                r"$f(x)=A_L e^{-\frac{1}{2}((x-x_0+\delta/2)/\sigma_L)^2}"
                r"+A_R e^{-\frac{1}{2}((x-x_0-\delta/2)/\sigma_R)^2}$"
            ),
            jacobian=_bimodal_gaussian_jacobian,
            presentation=FitPresentationSpec(
                components=(
                    FitComponentSpec("left", _bimodal_left),
                    FitComponentSpec("right", _bimodal_right),
                ),
            ),
            default_for=(FitTarget.HISTOGRAM,),
        ),
        FitModelSpec(
            "symmetric_lorentzian_doublet",
            "Symmetric Lorentzian doublet",
            1,
            (
                FitParameterSpec(
                    "center", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
                FitParameterSpec(
                    "common_fwhm",
                    AXIS_0,
                    POSITIVE,
                    display_label=r"$\mathrm{FWHM}$",
                ),
                FitParameterSpec(
                    "component_amplitude", VALUE, display_label=r"$H$"
                ),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
                FitParameterSpec(
                    "center_splitting",
                    AXIS_0,
                    NONNEGATIVE,
                    display_label=r"$\delta$",
                ),
            ),
            "center",
            _symmetric_lorentzian_doublet,
            _init_doublet,
            (FitTarget.SERIES,),
            formula=(
                r"$f(x)=H[L(x;x_0-\delta/2)+L(x;x_0+\delta/2)]+B$"
            ),
            jacobian=_symmetric_lorentzian_doublet_jacobian,
            candidate_initializer=_doublet_candidates,
            bounds_initializer=_doublet_bounds,
        ),
        FitModelSpec(
            "damped_sine",
            "Damped sine",
            1,
            (
                FitParameterSpec(
                    "amplitude", VALUE, NONNEGATIVE, display_label=r"$A$"
                ),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
                FitParameterSpec(
                    "baseband_frequency",
                    INVERSE_AXIS_0,
                    POSITIVE,
                    display_label=r"$f$",
                ),
                FitParameterSpec(
                    "decay_time", AXIS_0, POSITIVE, display_label=r"$\tau$"
                ),
                FitParameterSpec("phase", RADIAN, PHASE, display_label=r"$\varphi$"),
            ),
            "decay_time",
            _damped_sine,
            _init_damped_sine,
            (FitTarget.SERIES,),
            formula=r"$f(t)=A\sin(2\pi f t+\varphi)e^{-t/\tau}+B$",
            jacobian=_damped_sine_jacobian,
            candidate_initializer=_damped_sine_candidates,
            bounds_initializer=_damped_sine_bounds,
        ),
        FitModelSpec(
            "exponential_decay",
            "Exponential decay",
            1,
            (
                FitParameterSpec("amplitude", VALUE, display_label=r"$A$"),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
                FitParameterSpec(
                    "decay_time", AXIS_0, POSITIVE, display_label=r"$\tau$"
                ),
            ),
            "decay_time",
            _exponential_decay,
            _init_exponential,
            (FitTarget.SERIES,),
            formula=r"$f(t)=Ae^{-t/\tau}+B$",
            jacobian=_exponential_decay_jacobian,
            candidate_initializer=_exponential_candidates,
            bounds_initializer=_exponential_bounds,
        ),
        FitModelSpec(
            "anisotropic_gaussian_center",
            "Anisotropic Gaussian center",
            2,
            (
                FitParameterSpec("amplitude", VALUE, display_label=r"$A$"),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
                FitParameterSpec(
                    "radius_x", AXIS_0, POSITIVE, display_label=r"$R_x$"
                ),
                FitParameterSpec(
                    "radius_y", AXIS_1, POSITIVE, display_label=r"$R_y$"
                ),
                FitParameterSpec(
                    "center_x", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
                FitParameterSpec(
                    "center_y",
                    AXIS_1,
                    display_label=r"$y_0$",
                    affine_point=True,
                ),
            ),
            "center_x",
            _anisotropic_gaussian_center,
            _init_anisotropic,
            (FitTarget.IMAGE,),
            formula=(
                r"$f(x,y)=Ae^{-((x-x_0)^2/R_x^2+(y-y_0)^2/R_y^2)}+B$"
            ),
            jacobian=_anisotropic_gaussian_center_jacobian,
            candidate_initializer=_anisotropic_candidates,
            bounds_initializer=_anisotropic_bounds,
            presentation=FitPresentationSpec(
                ellipse_glyph=FitEllipseGlyphSpec(
                    ("center_x", "center_y"),
                    ("radius_x", "radius_y"),
                ),
            ),
            coordinate_relations=(AXIS_0, AXIS_1),
        ),
        FitModelSpec(
            "radial_gaussian_center",
            "Radial Gaussian center",
            2,
            (
                FitParameterSpec("amplitude", VALUE, display_label=r"$A$"),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
                FitParameterSpec(
                    "one_over_e_radius", AXIS_0, POSITIVE, display_label=r"$R$"
                ),
                FitParameterSpec(
                    "center_x", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
                FitParameterSpec(
                    "center_y",
                    AXIS_1,
                    display_label=r"$y_0$",
                    affine_point=True,
                    solver_unit_relation=AXIS_0,
                ),
            ),
            "center_x",
            _radial_gaussian_center,
            _init_radial,
            (FitTarget.IMAGE,),
            formula=r"$f(x,y)=Ae^{-((x-x_0)^2+(y-y_0)^2)/R^2}+B$",
            jacobian=_radial_gaussian_center_jacobian,
            candidate_initializer=_radial_candidates,
            bounds_initializer=_radial_bounds,
            presentation=FitPresentationSpec(
                ellipse_glyph=FitEllipseGlyphSpec(
                    ("center_x", "center_y"),
                    ("one_over_e_radius", "one_over_e_radius"),
                ),
            ),
            coordinate_relations=(AXIS_0, AXIS_0),
            default_for=(FitTarget.IMAGE,),
            capabilities=frozenset({"regular_image_radial"}),
        ),
    )


def default_fit_registry() -> FitModelRegistry:
    return FitModelRegistry(builtin_fit_models())


__all__ = [
    "FitCancelled",
    "FitComponentSpec",
    "FitDeadlineExceeded",
    "FitEngine",
    "FacetFitBatchResult",
    "FitNumericTable",
    "FitModelRegistry",
    "FitModelSpec",
    "FitOptions",
    "FitParameterDisplay",
    "FitParameterSpec",
    "FitPresentationSpec",
    "FitEllipseGlyphSpec",
    "FitResult",
    "FitTarget",
    "ParameterDomain",
    "RegularImageFitInput",
    "UnitRelation",
    "builtin_fit_models",
    "default_fit_registry",
]
