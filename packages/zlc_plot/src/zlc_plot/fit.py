"""Headless fit catalogue and solver used by every presentation backend."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import math
import re
from numbers import Real
import threading
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares, minimize, minimize_scalar
from scipy.signal import find_peaks

from . import _fit_compiled as _compiled_fit
from ._validation import finite_real as _finite_real
from ._validation import integer, text as _text
from .kinds import AxisRef

if TYPE_CHECKING:
    from ._fit_scene import FitOverlay


ArrayTuple = tuple[np.ndarray, ...]

# Models whose x origin is the start of the window they are fitted over.
_DOMAIN_ANCHORED = "domain_anchored"
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
#: The smallest expected count a Poisson deviance may be taken at.  A model
#: that predicts nothing where something was counted is infinitely unlikely;
#: the floor turns that into "very unlikely" so a solver can walk away from it.
_COUNT_FLOOR = 1e-9

# Capability bits that route RegularImageFitInput to the separable
# stripe/BLAS solver in ``_fit_radial`` instead of coordinate expansion.
_REGULAR_IMAGE_CAPABILITIES = frozenset(
    {"regular_image_radial", "regular_image_separable"}
)
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


#: A symbol has to survive being typed into a one-line box and split on
#: commas and equals signs, so it is a bare word.
_SYMBOL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*")


def formula_symbols(formula: str) -> frozenset[str]:
    """Every symbol a rendered formula shows, as an operator would type it.

    The formula is LaTeX for Matplotlib's mathtext.  Reading it back is one
    rule, not a table of aliases: a command becomes its own name (\\tau is
    tau), a \\mathrm{} wrapper is only a font instruction, and braces group
    without meaning anything to a reader.  What is left is the words.

    Used to CHECK the declared symbols, never to invent them -- which is why
    it may be generous: extra tokens like frac or sin cost nothing, while a
    parameter whose symbol is absent from its own formula is caught.
    """

    text = str(formula)
    text = text.replace("$", " ")
    text = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\\([A-Za-z]+)", r" \1", text)
    text = re.sub(r"[{}]", " ", text)
    return frozenset(_SYMBOL_PATTERN.findall(text))


@dataclass(frozen=True, slots=True)
class FitParameterSpec:
    """One solver parameter and its renderer-facing display semantics.

    ``affine_point`` distinguishes absolute positions from spans or amplitudes.
    It applies only to parameter values; standard errors are always converted
    as differences and therefore never consume a unit offset.
    ``solver_unit_relation`` records the canonical unit used by the evaluator
    when it differs from the parameter's painted axis relation.

    ``symbol`` is what the operator TYPES, and it is the same thing the
    formula prints.  The model showed f(t)=A e^{-(t-t_0)/tau}+B and then
    asked for "amplitude" and "decay_time": two names for one parameter,
    one on screen and one in the box under it, and nothing on screen said
    the second existed.

    It is READ OFF the display label, by the same function that reads a
    formula.  A label is one symbol by construction, not an expression, so
    reading it is exact rather than a guess -- and a label that does not
    resolve to exactly one word is refused here instead of being guessed
    at.  Deriving it also means the symbol cannot be mistyped into
    disagreeing with the label beside it.

    The NAME stays the identity.  It is what the solver, the stored fit
    target, the saved panel and every report key on; only what the operator
    types and reads changes here.
    """

    name: str
    unit_relation: UnitRelation
    domain: ParameterDomain = ParameterDomain.REAL
    display_label: str | None = None
    affine_point: bool = False
    solver_unit_relation: UnitRelation | None = None
    symbol: str | None = None

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
        symbol = self.symbol
        if symbol is None and display_label is None:
            # Nothing is printed for this one, so there are not two
            # vocabularies to disagree: its own name is what an operator
            # would type.  A model that DOES print a formula still has to
            # write this name in it, which is where a real divergence is
            # caught.
            symbol = name
        if symbol is None and display_label is not None:
            found = formula_symbols(display_label)
            if len(found) != 1:
                raise ValueError(
                    f"fit parameter {name!r} has display label "
                    f"{display_label!r}, which reads as {sorted(found)} -- a "
                    "label is one symbol, so say which one with symbol="
                )
            symbol = next(iter(found))
        if symbol is not None:
            symbol = _text(symbol, "fit parameter symbol")
            if not _SYMBOL_PATTERN.fullmatch(symbol):
                raise ValueError(
                    f"fit parameter symbol {symbol!r} must be a bare word an "
                    "operator can type: letters, digits and underscores"
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "display_label", display_label)
        object.__setattr__(self, "solver_unit_relation", solver_unit_relation)
        object.__setattr__(self, "symbol", symbol)

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
    compiled_descriptor: _compiled_fit.CompiledFitDescriptor | None = None

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
        # WHAT THE FORMULA WRITES IS WHAT THE OPERATOR MAY TYPE.
        #
        # The formula on screen said A and tau while the box under it wanted
        # "amplitude" and "decay_time", and nothing on screen said so.  The
        # two vocabularies could only stay together by somebody remembering
        # to move both, so the model refuses to exist unless they agree.
        symbols = tuple(item.symbol for item in parameters)
        unnamed = tuple(
            item.name for item in parameters if not item.symbol
        )
        if unnamed:
            raise ValueError(
                "every fit parameter needs the symbol the formula prints; "
                f"these have none: {list(unnamed)}"
            )
        if len(symbols) != len(set(symbols)):
            raise ValueError(
                f"fit parameter symbols must be unique within a model: "
                f"{list(symbols)}"
            )
        if self.formula is not None:
            printed = formula_symbols(self.formula)
            absent = tuple(symbol for symbol in symbols if symbol not in printed)
            if absent:
                raise ValueError(
                    f"fit model {self.model_id!r} asks the operator for "
                    f"{list(absent)}, which its own formula never writes"
                )
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
        if self.compiled_descriptor is not None and not isinstance(
            self.compiled_descriptor,
            _compiled_fit.CompiledFitDescriptor,
        ):
            raise TypeError(
                "compiled_descriptor must be CompiledFitDescriptor or None"
            )
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

    @property
    def symbols(self) -> tuple[str, ...]:
        """What the operator may type, in the order the formula reads."""

        return tuple(str(item.symbol) for item in self.parameters)

    def parameter_for_symbol(self, symbol: str) -> FitParameterSpec | None:
        """The parameter an operator named, or None if this model has no such."""

        wanted = str(symbol)
        for item in self.parameters:
            if item.symbol == wanted:
                return item
        return None

    def parameter_index(self, name: str) -> int:
        parameter_name = _text(name, "fit parameter name")
        try:
            return self.parameter_names.index(parameter_name)
        except ValueError as error:
            raise ValueError(f"unknown fit parameter: {name!r}") from error

    def anchored_at(self, origin: float) -> "FitModelSpec":
        """Return this model with its x origin moved to ``origin``.

        A decay is written from an origin: ``A`` is the amplitude *at x=0*.
        Fitted over a window that opens at shot 1200 the very same curve needs
        ``A*exp(1200/tau)`` -- a number no solver can carry and no operator can
        read.  The origin belongs to the window being fitted, not to the world,
        so it is bound here and travels with the result: overlay, components
        and jacobian all evaluate through this spec and see the same anchor.
        """

        origin = float(origin)
        if self.independent_arity != 1:
            raise ValueError("only single-axis fit models can be anchored")
        if not math.isfinite(origin):
            raise ValueError("fit anchor must be finite")
        if origin == 0.0:
            return self

        def relative(coordinates: ArrayTuple) -> ArrayTuple:
            first, *rest = coordinates
            return (np.asarray(first, dtype=np.float64) - origin, *rest)

        def anchor(evaluate: Evaluator) -> Evaluator:
            def anchored(x, *values):
                return evaluate(np.asarray(x, dtype=np.float64) - origin, *values)

            return anchored

        presentation = self.presentation
        if presentation.components:
            presentation = replace(
                presentation,
                components=tuple(
                    replace(component, evaluator=anchor(component.evaluator))
                    for component in presentation.components
                ),
            )
        initializer = self.initializer
        candidate = self.candidate_initializer
        limits = self.bounds_initializer
        return replace(
            self,
            evaluator=anchor(self.evaluator),
            jacobian=None if self.jacobian is None else anchor(self.jacobian),
            initializer=lambda coordinates, observed: initializer(
                relative(coordinates), observed
            ),
            candidate_initializer=(
                None
                if candidate is None
                else lambda coordinates, observed: candidate(
                    relative(coordinates), observed
                )
            ),
            bounds_initializer=(
                None
                if limits is None
                else lambda coordinates, observed: limits(
                    relative(coordinates), observed
                )
            ),
            presentation=presentation,
        )

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
    #: Curve fits with more finite points than this iterate on an x-binned
    #: sufficient-statistics compression (bin means weighted by counts) and
    #: keep the final model evaluation, residuals and quality on the full
    #: data.  ``None`` solves every point exactly at any size.
    max_exact_points: int | None = 4096

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
        if self.max_exact_points is not None:
            max_exact = integer(self.max_exact_points, "max_exact_points")
            if max_exact <= 0:
                raise ValueError("max_exact_points must be a positive integer")
            object.__setattr__(self, "max_exact_points", max_exact)


def _readonly(array: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(array)
    backing: object = source
    while isinstance(getattr(backing, "base", None), np.ndarray):
        backing = backing.base
    if not source.flags.writeable and isinstance(getattr(backing, "base", None), bytes):
        return source
    return np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(source.shape)


class _DeferredFitData:
    """One shared loader for lazily materialized per-observation fit arrays.

    The regular-image solver retains only its fit input and the solved
    parameters; ``fitted_values``, ``residuals`` and ``selected_indices`` are
    computed on first access and cached, so accepting a fit never copies three
    full-image planes inside the presentation transaction.
    """

    __slots__ = ("_loader", "_lock", "_arrays")

    def __init__(
        self,
        loader: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        if not callable(loader):
            raise TypeError("deferred fit data requires a callable loader")
        self._loader: (
            Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]] | None
        ) = loader
        self._lock = threading.Lock()
        self._arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            if self._arrays is None:
                assert self._loader is not None
                fitted, residuals, indices = self._loader()
                fitted = _readonly(
                    np.asarray(fitted, dtype=np.float64).reshape(-1)
                )
                residuals = _readonly(
                    np.asarray(residuals, dtype=np.float64).reshape(-1)
                )
                indices = _readonly(
                    np.asarray(indices, dtype=np.int64).reshape(-1)
                )
                if fitted.shape != residuals.shape or indices.shape != fitted.shape:
                    raise ValueError(
                        "deferred fit arrays must have one shared shape"
                    )
                self._arrays = (fitted, residuals, indices)
                self._loader = None
            return self._arrays


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
    fixed_parameter_names: tuple[str, ...] = ()

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
        fixed_names = tuple(self.fixed_parameter_names)
        fixed_indices = tuple(self.model.parameter_index(name) for name in fixed_names)
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
            if fixed_indices:
                covariance[list(fixed_indices), :] = 0.0
                covariance[:, list(fixed_indices)] = 0.0
                errors[list(fixed_indices)] = 0.0
        deferred = (
            _FIT_RESULT_RAW["fitted_values"].__get__(self, type(self))
            if _FIT_RESULT_RAW
            else self.fitted_values
        )
        if isinstance(deferred, _DeferredFitData):
            raw_residuals = _FIT_RESULT_RAW["residuals"].__get__(self, type(self))
            raw_indices = _FIT_RESULT_RAW["selected_indices"].__get__(
                self, type(self)
            )
            if raw_residuals is not deferred or raw_indices is not deferred:
                raise TypeError(
                    "deferred fit arrays must share one _DeferredFitData value"
                )
        else:
            deferred = None
            fitted = np.asarray(self.fitted_values, dtype=np.float64).reshape(-1)
            residuals = np.asarray(self.residuals, dtype=np.float64).reshape(-1)
            if fitted.shape != residuals.shape:
                raise ValueError("fitted values and residuals must have equal shape")
            indices = np.asarray(self.selected_indices, dtype=np.int64).reshape(-1)
            if indices.shape != fitted.shape:
                raise ValueError(
                    "selected indices must identify every fitted observation"
                )
        source_revision = integer(self.source_revision, "source_revision")
        if source_revision < 0:
            raise ValueError("source_revision must be non-negative")
        batch_revision = integer(self.batch_revision, "batch_revision")
        if batch_revision < 0:
            raise ValueError("batch_revision must be non-negative")
        object.__setattr__(self, "parameter_values", _readonly(parameters))
        object.__setattr__(self, "standard_errors", _readonly(errors))
        object.__setattr__(self, "covariance", _readonly(covariance))
        if deferred is None:
            object.__setattr__(self, "fitted_values", _readonly(fitted))
            object.__setattr__(self, "residuals", _readonly(residuals))
            object.__setattr__(self, "selected_indices", _readonly(indices))
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "batch_revision", batch_revision)
        object.__setattr__(self, "fixed_parameter_names", fixed_names)
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
        return MappingProxyType({
            name: valid and name not in self.fixed_parameter_names
            for name in self.parameter_names
        })

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

    def _clone(self, **overrides: Any) -> "FitResult":
        """Copy this result while preserving still-deferred arrays.

        ``dataclasses.replace`` reads every field through the lazy accessors
        and would materialize the deferred arrays; this constructor-mirror
        passes the raw slot values through instead.
        """

        values: dict[str, Any] = {
            "model": self.model,
            "parameter_values": self.parameter_values,
            "standard_errors": self.standard_errors,
            "covariance": self.covariance,
            "fitted_values": _FIT_RESULT_RAW["fitted_values"].__get__(
                self, type(self)
            ),
            "residuals": _FIT_RESULT_RAW["residuals"].__get__(self, type(self)),
            "selected_indices": _FIT_RESULT_RAW["selected_indices"].__get__(
                self, type(self)
            ),
            "source_revision": self.source_revision,
            "success": self.success,
            "message": self.message,
            "reduced_chi_square": self.reduced_chi_square,
            "covariance_valid": self.covariance_valid,
            "parameter_units": self.parameter_units,
            "batch_revision": self.batch_revision,
            "fixed_parameter_names": self.fixed_parameter_names,
        }
        values.update(overrides)
        if set(overrides).issubset({"parameter_units", "batch_revision"}):
            batch_revision = integer(values["batch_revision"], "batch_revision")
            if batch_revision < 0:
                raise ValueError("batch_revision must be non-negative")
            units = dict(values["parameter_units"])
            unknown = set(units) - set(self.model.parameter_names)
            if unknown:
                raise ValueError(
                    f"fit result units name unknown parameters: {sorted(unknown)}"
                )
            parameter_units = MappingProxyType({
                name: _text(
                    units.get(name, ""),
                    f"fit parameter unit {name}",
                    allow_empty=True,
                )
                for name in self.model.parameter_names
            })
            clone = object.__new__(FitResult)
            for name in (
                "model",
                "parameter_values",
                "standard_errors",
                "covariance",
                "source_revision",
                "success",
                "message",
                "reduced_chi_square",
                "covariance_valid",
                "fixed_parameter_names",
            ):
                object.__setattr__(clone, name, values[name])
            for name in ("fitted_values", "residuals", "selected_indices"):
                _FIT_RESULT_RAW[name].__set__(clone, values[name])
            object.__setattr__(clone, "parameter_units", parameter_units)
            object.__setattr__(clone, "batch_revision", batch_revision)
            return clone
        return FitResult(**values)

    def with_parameter_units(self, units: Mapping[str, str]) -> "FitResult":
        return self._clone(parameter_units=units)

    def with_batch_revision(self, batch_revision: int) -> "FitResult":
        """Return a copy carrying one monotonic publication revision."""

        return self._clone(batch_revision=batch_revision)


_FIT_RESULT_RAW: dict[str, Any] = {}


def _install_lazy_fit_result_fields() -> None:
    """Route the per-observation FitResult fields through lazy accessors.

    The dataclass slot descriptors are preserved in ``_FIT_RESULT_RAW`` and
    replaced by properties that materialize a ``_DeferredFitData`` value on
    first read.  Construction, ``dataclasses.replace`` and the frozen
    ``__setattr__`` contract are unchanged.
    """

    lazy_fields = ("fitted_values", "residuals", "selected_indices")
    for name in lazy_fields:
        _FIT_RESULT_RAW[name] = FitResult.__dict__[name]

    def make_accessor(name: str) -> property:
        slot = _FIT_RESULT_RAW[name]

        def getter(self: FitResult) -> np.ndarray:
            value = slot.__get__(self, FitResult)
            if isinstance(value, _DeferredFitData):
                fitted, residuals, indices = value.arrays()
                _FIT_RESULT_RAW["fitted_values"].__set__(self, fitted)
                _FIT_RESULT_RAW["residuals"].__set__(self, residuals)
                _FIT_RESULT_RAW["selected_indices"].__set__(self, indices)
                value = slot.__get__(self, FitResult)
            return value

        def setter(self: FitResult, value: Any) -> None:
            slot.__set__(self, value)

        return property(getter, setter)

    for name in lazy_fields:
        setattr(FitResult, name, make_accessor(name))


_install_lazy_fit_result_fields()


def _fit_result_from_validated_batch_row(
    *,
    model: FitModelSpec,
    parameter_values: np.ndarray,
    standard_errors: np.ndarray,
    covariance: np.ndarray,
    fitted_values: np.ndarray,
    residuals: np.ndarray,
    selected_indices: np.ndarray,
    source_revision: int,
    success: bool,
    message: str,
    reduced_chi_square: float,
    covariance_valid: bool,
    parameter_units: Mapping[str, str],
    fixed_parameter_names: tuple[str, ...],
) -> FitResult:
    """Build one row after its entire compiled batch passed validation.

    The public constructor deliberately copies every incoming array to make an
    isolated immutable result.  Compiled output already owns one private batch
    backing, so copying and re-validating it 64 times costs more than the
    solve.  This factory accepts only read-only rows from a backing validated
    once by ``FitEngine`` and fills the same frozen slots directly.
    """

    arrays = (
        parameter_values,
        standard_errors,
        covariance,
        fitted_values,
        residuals,
        selected_indices,
    )
    if any(np.asarray(item).flags.writeable for item in arrays):
        raise RuntimeError("compiled fit result backing must be read-only")
    result = object.__new__(FitResult)
    object.__setattr__(result, "model", model)
    object.__setattr__(result, "parameter_values", parameter_values)
    object.__setattr__(result, "standard_errors", standard_errors)
    object.__setattr__(result, "covariance", covariance)
    _FIT_RESULT_RAW["fitted_values"].__set__(result, fitted_values)
    _FIT_RESULT_RAW["residuals"].__set__(result, residuals)
    _FIT_RESULT_RAW["selected_indices"].__set__(result, selected_indices)
    object.__setattr__(result, "source_revision", source_revision)
    object.__setattr__(result, "success", success)
    object.__setattr__(result, "message", message)
    object.__setattr__(result, "reduced_chi_square", reduced_chi_square)
    object.__setattr__(result, "covariance_valid", covariance_valid)
    object.__setattr__(result, "parameter_units", parameter_units)
    object.__setattr__(result, "batch_revision", 0)
    object.__setattr__(result, "fixed_parameter_names", fixed_parameter_names)
    return result


#: How many shots a fitted component must hold to be a POPULATION.  Below
#: this it is a handful of counts that a three-parameter curve can sit on
#: exactly -- which is what a two-state fit does with the first thirty shots
#: of a run that has not loaded an atom yet, and it reports a threshold and a
#: fidelity for it.  Measured: with a rare state at 2%, every fit before ~150
#: shots put one component on a single bin holding one count.
_CLASSIFIER_MINIMUM_COMPONENT_SHOTS = 4.0


def _bimodal_classifier_metrics(
    result: FitResult,
    threshold: float | None = None,
) -> tuple[float | None, float, float, float]:
    """Return threshold, left/right population fractions, and fidelity.

    The threshold is ``None`` when the fit does not describe two populations:
    asked where two states separate, the honest answer for one state and a
    stray count is that there is nowhere.  Every caller already treats a
    missing threshold as "no classifier", so the line, the label and the
    fidelity all disappear together until the shots arrive.
    """

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

    left_area = float(values["left_amplitude"]) * left_sigma
    right_area = float(values["right_amplitude"]) * right_sigma
    total_area = left_area + right_area
    if not math.isfinite(total_area) or total_area <= 0.0:
        return (None, float("nan"), float("nan"), float("nan"))
    left_weight = left_area / total_area
    right_weight = 1.0 - left_weight

    def error(value: float) -> float:
        return (
            left_weight * (1.0 - cdf(value, left_mean, left_sigma))
            + right_weight * cdf(value, right_mean, right_sigma)
        )
    if threshold is None:
        # The fitted curve's own total is the shot count, whatever the bins
        # are: the model IS counts per bin.
        shots = float(np.sum(np.asarray(result.fitted_values, dtype=float)))
        if (
            min(left_weight, right_weight) * shots
            < _CLASSIFIER_MINIMUM_COMPONENT_SHOTS
        ):
            return (None, float("nan"), float("nan"), float("nan"))
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
    left_correct = cdf(threshold, left_mean, left_sigma)
    right_correct = 1.0 - cdf(threshold, right_mean, right_sigma)
    left_fraction = (
        left_weight * cdf(threshold, left_mean, left_sigma)
        + right_weight * cdf(threshold, right_mean, right_sigma)
    )
    return (
        threshold,
        left_fraction,
        1.0 - left_fraction,
        left_weight * left_correct + right_weight * right_correct,
    )


@dataclass(frozen=True, slots=True)
class FacetFitBatchResult:
    """Ordered fit results for every cell of one FacetGrid projection."""

    facet: AxisRef | None
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
        if self.facet is not None and not isinstance(self.facet, AxisRef):
            raise TypeError("facet must be AxisRef or None")
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
                if (
                    result.model.model_id != self.model.model_id
                    or result.source_revision != revision
                ):
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
            axis_name = self.facet.axis_id
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
                    if (
                        fit is None
                        or not fit.success
                        or not fit.covariance_valid
                        or name in fit.fixed_parameter_names
                    )
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


class FitEngine:
    def __init__(self, registry: FitModelRegistry | None = None) -> None:
        self.registry = registry or default_fit_registry()
        self._compiled_context_lock = threading.RLock()
        self._compiled_contexts: OrderedDict[tuple[Any, ...], np.ndarray] = (
            OrderedDict()
        )

    def _compiled_context(
        self,
        descriptor: _compiled_fit.CompiledFitDescriptor,
        coordinates: ArrayTuple,
    ) -> np.ndarray:
        """Return one immutable coordinate plan from a small exact LRU."""

        digest = hashlib.blake2b(digest_size=20)
        digest.update(descriptor.cache_key.encode("utf-8"))
        signature: list[Any] = [descriptor.cache_key, len(coordinates)]
        for axis in coordinates:
            values = np.ascontiguousarray(axis, dtype=np.float64)
            signature.extend((values.shape, values.dtype.str))
            digest.update(memoryview(values).cast("B"))
        key = (*signature, digest.digest())
        with self._compiled_context_lock:
            cached = self._compiled_contexts.get(key)
            if cached is not None:
                self._compiled_contexts.move_to_end(key)
                return cached
        built = _readonly(
            np.asarray(
                descriptor.context_builder(coordinates),
                dtype=np.float64,
            )
        )
        if built.ndim != 2:
            raise ValueError("compiled fit context_builder must return a 2D array")
        with self._compiled_context_lock:
            existing = self._compiled_contexts.get(key)
            if existing is not None:
                self._compiled_contexts.move_to_end(key)
                return existing
            self._compiled_contexts[key] = built
            while len(self._compiled_contexts) > 16:
                self._compiled_contexts.popitem(last=False)
        return built

    def _compiled_descriptor(
        self,
        model: FitModelSpec,
    ) -> _compiled_fit.CompiledFitDescriptor | None:
        """Return the descriptor only for this registry's exact model.

        ``dataclasses.replace`` is intentionally a customization boundary: a
        caller replacing an evaluator or Jacobian must not accidentally keep
        running the compiled callbacks attached to the original built-in.
        """

        descriptor = model.compiled_descriptor
        if descriptor is None:
            return None
        try:
            registered = self.registry.get(model.model_id)
        except ValueError:
            return None
        return descriptor if registered is model else None

    def fit_batch(
        self,
        model: str | FitModelSpec,
        coordinates: Sequence[Sequence[np.ndarray] | RegularImageFitInput],
        observations: Sequence[np.ndarray | None],
        *,
        observation_sigmas: Sequence[np.ndarray | None] | None = None,
        selected_indices: Sequence[np.ndarray | None] | None = None,
        data_revisions: Sequence[int] | None = None,
        initial: Mapping[str, float] | Sequence[float] | None = None,
        warm_starts: Sequence[
            Mapping[str, float] | Sequence[float] | None
        ] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        options: FitOptions | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[tuple[FitResult | None, ...], tuple[str | None, ...]]:
        """Fit independent cells through one numerical owner.

        Built-ins with a compiled descriptor enter one independent batched
        solve.  Regular images use the shared separable batch owner, including
        a one-cell public fit.  A custom model, caller-replaced model or custom
        engine keeps the authored scalar implementation.  A failed cell is
        reported in the matching failure slot; it is never silently sent
        through a second numerical implementation.
        """

        spec = self.registry.get(model) if isinstance(model, str) else model
        if not isinstance(spec, FitModelSpec):
            raise TypeError("model must be a registered id or FitModelSpec")
        coordinate_items = tuple(coordinates)
        value_items = tuple(observations)
        count = len(coordinate_items)
        if count == 0:
            raise ValueError("fit batch cannot be empty")
        if len(value_items) != count:
            raise ValueError("fit batch observations must match coordinates")

        def sequence_or_default(
            values: Sequence[Any] | None,
            default: Any,
            name: str,
        ) -> tuple[Any, ...]:
            resolved = (default,) * count if values is None else tuple(values)
            if len(resolved) != count:
                raise ValueError(f"fit batch {name} must match coordinates")
            return resolved

        sigma_items = sequence_or_default(
            observation_sigmas,
            None,
            "observation_sigmas",
        )
        index_items = sequence_or_default(
            selected_indices,
            None,
            "selected_indices",
        )
        revision_items = sequence_or_default(data_revisions, 0, "data_revisions")
        warm_items = sequence_or_default(warm_starts, None, "warm_starts")

        descriptor = self._compiled_descriptor(spec)
        custom_fit = type(self).fit is not FitEngine.fit
        if descriptor is None or custom_fit:
            results: list[FitResult | None] = []
            failures: list[str | None] = []
            for cell in range(count):
                try:
                    result = self.fit(
                        spec,
                        coordinate_items[cell],
                        value_items[cell],
                        observation_sigma=sigma_items[cell],
                        selected_indices=index_items[cell],
                        data_revision=revision_items[cell],
                        initial=initial,
                        warm_start=warm_items[cell],
                        bounds=bounds,
                        options=options,
                        cancelled=cancelled,
                    )
                except FitCancelled:
                    raise
                except Exception as error:
                    results.append(None)
                    failures.append(str(error) or type(error).__name__)
                else:
                    results.append(result)
                    failures.append(None)
            return tuple(results), tuple(failures)

        if (
            spec.capabilities & _REGULAR_IMAGE_CAPABILITIES
            and all(
                isinstance(item, RegularImageFitInput)
                for item in coordinate_items
            )
        ):
            if any(item is not None for item in value_items):
                raise TypeError(
                    "observations belong inside RegularImageFitInput and must "
                    "not also be passed separately"
                )
            if any(item is not None for item in sigma_items):
                raise TypeError(
                    "regular-image fitting has no per-point sigma channel"
                )
            if any(item is not None for item in index_items):
                raise TypeError(
                    "regular-image selected_indices belong inside "
                    "RegularImageFitInput"
                )
            from ._fit_radial import fit_regular_separable_images

            return fit_regular_separable_images(
                spec,
                coordinate_items,  # type: ignore[arg-type]
                data_revisions=revision_items,
                initial=initial,
                warm_starts=warm_items,
                bounds=bounds,
                options=options or FitOptions(),
                cancelled=cancelled,
            )

        results: list[FitResult | None] = [None] * count
        failures: list[str | None] = [None] * count
        compiled_cells: list[int] = []
        compiled_coordinates: list[Sequence[np.ndarray]] = []
        compiled_observations: list[np.ndarray | None] = []
        compiled_sigmas: list[np.ndarray | None] = []
        compiled_indices: list[np.ndarray | None] = []
        compiled_revisions: list[int] = []
        compiled_warm: list[Mapping[str, float] | Sequence[float] | None] = []
        for cell, coordinate_item in enumerate(coordinate_items):
            if not isinstance(coordinate_item, RegularImageFitInput):
                compiled_cells.append(cell)
                compiled_coordinates.append(coordinate_item)
                compiled_observations.append(value_items[cell])
                compiled_sigmas.append(sigma_items[cell])
                compiled_indices.append(index_items[cell])
                compiled_revisions.append(revision_items[cell])
                compiled_warm.append(warm_items[cell])
                continue
            try:
                if value_items[cell] is not None:
                    raise TypeError(
                        "observations belong inside RegularImageFitInput and "
                        "must not also be passed separately"
                    )
                if sigma_items[cell] is not None:
                    raise TypeError(
                        "regular-image fitting has no per-point sigma channel"
                    )
                if index_items[cell] is not None:
                    raise TypeError(
                        "regular-image selected_indices belong inside "
                        "RegularImageFitInput"
                    )
                if not (spec.capabilities & _REGULAR_IMAGE_CAPABILITIES):
                    raise ValueError(
                        "this model does not declare a regular-image capability"
                    )
                results[cell] = self.fit(
                    spec,
                    coordinate_item,
                    data_revision=revision_items[cell],
                    initial=initial,
                    warm_start=warm_items[cell],
                    bounds=bounds,
                    options=options,
                    cancelled=cancelled,
                )
                continue
            except FitCancelled:
                raise
            except Exception as error:
                failures[cell] = str(error) or type(error).__name__
                continue
        if not compiled_cells:
            return tuple(results), tuple(failures)
        solved, failed = self._fit_compiled_batch(
            spec,
            descriptor,
            compiled_coordinates,
            compiled_observations,
            observation_sigmas=compiled_sigmas,
            selected_indices=compiled_indices,
            data_revisions=compiled_revisions,
            initial=initial,
            warm_starts=compiled_warm,
            bounds=bounds,
            options=options,
            cancelled=cancelled,
        )
        for local, cell in enumerate(compiled_cells):
            results[cell] = solved[local]
            failures[cell] = failed[local]
        return tuple(results), tuple(failures)

    def _fit_compiled_batch(
        self,
        model: FitModelSpec,
        descriptor: _compiled_fit.CompiledFitDescriptor,
        coordinates: Sequence[Sequence[np.ndarray] | RegularImageFitInput],
        observations: Sequence[np.ndarray | None],
        *,
        observation_sigmas: Sequence[np.ndarray | None],
        selected_indices: Sequence[np.ndarray | None],
        data_revisions: Sequence[int],
        initial: Mapping[str, float] | Sequence[float] | None,
        warm_starts: Sequence[
            Mapping[str, float] | Sequence[float] | None
        ],
        bounds: Mapping[str, tuple[float | None, float | None]] | None,
        options: FitOptions | None,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[tuple[FitResult | None, ...], tuple[str | None, ...]]:
        """Pack finite cells, solve equal-size buckets, and restore metadata."""

        opts = options or FitOptions()
        started = time.monotonic()

        def check() -> None:
            if cancelled is not None and cancelled():
                raise FitCancelled("fit cancelled")
            if (
                opts.deadline_seconds is not None
                and time.monotonic() - started > opts.deadline_seconds
            ):
                raise FitDeadlineExceeded("fit deadline exceeded")

        check()
        parameter_count = len(model.parameters)
        base_lower = np.asarray(
            [item.bounds[0] for item in model.parameters],
            dtype=np.float64,
        )
        base_upper = np.asarray(
            [item.bounds[1] for item in model.parameters],
            dtype=np.float64,
        )
        requested_lower, requested_upper = _solver_bounds(model, None, bounds)
        requested_mask = np.asarray(
            [item.name in (bounds or {}) for item in model.parameters],
            dtype=np.bool_,
        )
        fixed_names, free_indices = _fixed_parameter_partition(model, bounds)
        free_index = np.asarray(free_indices, dtype=np.int64)
        counted = model.targets == (FitTarget.HISTOGRAM,)

        results: list[FitResult | None] = [None] * len(coordinates)
        failures: list[str | None] = [None] * len(coordinates)
        prepared: dict[int, dict[str, Any]] = {}
        #: Finiteness of a coordinate OBJECT, by identity -- a tensor
        #: facet's cells share their axis arrays, and every cell re-scanned
        #: the same megabytes for NaNs.  True means proven all-finite.
        axis_all_finite: dict[int, bool] = {}
        for cell, coordinate_item in enumerate(coordinates):
            check()
            try:
                if isinstance(coordinate_item, RegularImageFitInput):
                    raise TypeError(
                        "regular-image inputs belong to the separable fit route"
                    )
                incoming = observations[cell]
                if incoming is None:
                    raise TypeError("observations are required for coordinate fitting")
                coords = _coordinate_arrays(
                    tuple(coordinate_item),
                    model.independent_arity,
                )
                values = np.asarray(incoming, dtype=np.float64).reshape(-1)
                if any(axis.shape != values.shape for axis in coords):
                    raise ValueError(
                        "coordinates and observations must have equal flattened shape"
                    )
                sigma = observation_sigmas[cell]
                sigma_array: np.ndarray | None = None
                if sigma is not None:
                    sigma_array = np.asarray(sigma, dtype=np.float64).reshape(-1)
                    if sigma_array.shape != values.shape:
                        raise ValueError("observation_sigma must match observations")
                    if counted:
                        raise ValueError(
                            "histogram targets weight themselves by counts; an "
                            "external sigma has no meaning there"
                        )
                indices_item = selected_indices[cell]
                if indices_item is None:
                    indices = np.arange(values.size, dtype=np.int64)
                else:
                    indices = np.asarray(indices_item, dtype=np.int64).reshape(-1)
                    if indices.shape != values.shape:
                        raise ValueError(
                            "selected_indices must match observations"
                        )
                finite = np.isfinite(values)
                for axis in coords:
                    proven = axis_all_finite.get(id(axis))
                    if proven is None:
                        proven = bool(np.all(np.isfinite(axis)))
                        axis_all_finite[id(axis)] = proven
                    if not proven:
                        finite &= np.isfinite(axis)
                if not bool(np.all(finite)):
                    coords = tuple(axis[finite] for axis in coords)
                    values = values[finite]
                    indices = indices[finite]
                    if sigma_array is not None:
                        sigma_array = sigma_array[finite]
                if values.size <= len(free_indices):
                    raise ValueError(
                        "fit requires more finite observations than free parameters"
                    )
                revision = integer(data_revisions[cell], "data_revision")
                if revision < 0:
                    raise ValueError("data_revision must be non-negative")

                origin = 0.0
                effective_model = model
                origin_axis = descriptor.coordinate_origin
                if origin_axis is not None:
                    if origin_axis >= len(coords):
                        raise ValueError(
                            "compiled coordinate origin exceeds model arity"
                        )
                    origin = float(np.min(coords[origin_axis]))
                    if origin_axis != 0 or model.independent_arity != 1:
                        raise ValueError(
                            "result-model anchoring currently requires axis 0"
                        )
                    effective_model = model.anchored_at(origin)

                solver_coords = coords
                solver_values = values
                weights: np.ndarray | None = None
                binned = False
                if sigma_array is not None:
                    usable = sigma_array[
                        np.isfinite(sigma_array) & (sigma_array > 0.0)
                    ]
                    if usable.size:
                        floor = float(np.min(usable))
                        bounded = np.where(
                            np.isfinite(sigma_array) & (sigma_array > 0.0),
                            sigma_array,
                            floor,
                        )
                        weights = 1.0 / bounded
                elif (
                    not counted
                    and bool(free_indices)
                    and model.independent_arity == 1
                    and opts.max_exact_points is not None
                    and values.size > 2 * opts.max_exact_points
                ):
                    compressed = _binned_curve_statistics(
                        coords[0],
                        values,
                        opts.max_exact_points,
                    )
                    if compressed is not None:
                        solver_coords, solver_values, weights = compressed
                        binned = True
                compiled_coords = solver_coords
                if descriptor.coordinate_origin is not None:
                    origin_axis = descriptor.coordinate_origin
                    compiled_coords = tuple(
                        axis - origin if index == origin_axis else axis
                        for index, axis in enumerate(solver_coords)
                    )
            except Exception as error:
                failures[cell] = str(error) or type(error).__name__
                continue

            # Authored and warm values are explicit public inputs.  Invalid
            # values are a caller error, not a reason to silently run cold.
            authored = (
                None
                if initial is None or not free_indices
                else _initial_values(
                    effective_model,
                    solver_coords,
                    solver_values,
                    initial,
                )
            )
            warm_item = warm_starts[cell]
            warm = (
                None
                if warm_item is None or not free_indices
                else _initial_values(
                    effective_model,
                    solver_coords,
                    solver_values,
                    warm_item,
                )
            )
            prepared[cell] = {
                "coords": coords,
                "values": values,
                "indices": indices,
                "solver_coords": solver_coords,
                "compiled_coords": compiled_coords,
                "solver_values": solver_values,
                "weights": weights,
                "binned": binned,
                "revision": revision,
                "origin": origin,
                "model": effective_model,
                "authored": authored,
                "warm": warm,
            }

        if not free_indices:
            fixed_values = np.asarray(requested_lower, dtype=np.float64)
            count = parameter_count
            empty_units = MappingProxyType({
                name: "" for name in model.parameter_names
            })
            for cell, item in prepared.items():
                check()
                try:
                    fitted = item["model"].evaluate(
                        item["coords"],
                        fixed_values,
                    ).reshape(-1)
                    if (
                        fitted.shape != item["values"].shape
                        or not np.all(np.isfinite(fitted))
                    ):
                        raise RuntimeError("fixed fit evaluation is non-finite")
                    residuals = item["values"] - fitted
                    if counted:
                        expected = np.maximum(fitted, _COUNT_FLOOR)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            logarithm = np.where(
                                item["values"] > 0.0,
                                item["values"]
                                * np.log(item["values"] / expected),
                                0.0,
                            )
                        deviance = 2.0 * np.maximum(
                            expected - item["values"] + logarithm,
                            0.0,
                        )
                        quality = np.copysign(
                            np.sqrt(deviance),
                            expected - item["values"],
                        )
                    elif item["weights"] is None:
                        quality = residuals
                    else:
                        quality = residuals * item["weights"]
                    parameters = _readonly(fixed_values)
                    errors = _readonly(np.zeros(count, dtype=np.float64))
                    covariance = _readonly(
                        np.zeros((count, count), dtype=np.float64)
                    )
                    fitted = _readonly(fitted)
                    residuals = _readonly(residuals)
                    indices = _readonly(item["indices"])
                    results[cell] = _fit_result_from_validated_batch_row(
                        model=item["model"],
                        parameter_values=parameters,
                        standard_errors=errors,
                        covariance=covariance,
                        fitted_values=fitted,
                        residuals=residuals,
                        selected_indices=indices,
                        source_revision=item["revision"],
                        success=True,
                        message="all parameters fixed",
                        reduced_chi_square=float(
                            np.dot(quality, quality) / item["values"].size
                        ),
                        covariance_valid=True,
                        parameter_units=empty_units,
                        fixed_parameter_names=fixed_names,
                    )
                    failures[cell] = None
                except Exception as error:
                    failures[cell] = str(error) or type(error).__name__
            return tuple(results), tuple(failures)

        buckets: dict[tuple[int, bytes], list[int]] = {}
        # One digest per coordinate OBJECT, not per cell: a tensor facet's
        # cells share the same axis arrays, and hashing the same quarter
        # megabyte forty times over was most of the batch's Python time.
        # Identity is only a fast path -- equal content in distinct objects
        # hashes separately and still lands in the same bucket.
        axis_digests: dict[int, bytes] = {}
        for cell, item in prepared.items():
            digest = hashlib.blake2b(digest_size=20)
            for axis in item["compiled_coords"]:
                known = axis_digests.get(id(axis))
                if known is None:
                    values = np.ascontiguousarray(axis, dtype=np.float64)
                    axis_hash = hashlib.blake2b(digest_size=20)
                    axis_hash.update(str(values.shape).encode("ascii"))
                    axis_hash.update(memoryview(values).cast("B"))
                    known = axis_hash.digest()
                    axis_digests[id(axis)] = known
                digest.update(known)
            key = (int(item["solver_values"].size), digest.digest())
            buckets.setdefault(key, []).append(cell)

        for bucket in buckets.values():
            check()
            items = [prepared[cell] for cell in bucket]
            shared_coordinates = tuple(
                np.ascontiguousarray(axis, dtype=np.float64)
                for axis in items[0]["compiled_coords"]
            )
            value_stack = np.stack([item["solver_values"] for item in items])
            weight_stack: np.ndarray | None = None
            if any(item["weights"] is not None for item in items):
                weight_stack = np.stack(
                    [
                        np.ones_like(item["solver_values"])
                        if item["weights"] is None
                        else item["weights"]
                        for item in items
                    ]
                )
            authored_flags = np.asarray(
                [item["authored"] is not None for item in items],
                dtype=np.bool_,
            )
            authored_stack = (
                None
                if not bool(np.any(authored_flags))
                else np.stack(
                    [
                        np.zeros(parameter_count, dtype=np.float64)
                        if item["authored"] is None
                        else item["authored"]
                        for item in items
                    ]
                )
            )
            warm_flags = np.asarray(
                [item["warm"] is not None for item in items],
                dtype=np.bool_,
            )
            warm_stack = (
                None
                if not bool(np.any(warm_flags))
                else np.stack(
                    [
                        np.zeros(parameter_count, dtype=np.float64)
                        if item["warm"] is None
                        else item["warm"]
                        for item in items
                    ]
                )
            )

            context = self._compiled_context(
                descriptor,
                shared_coordinates,
            )

            try:
                solve = (
                    _compiled_fit.solve_compiled_single
                    if len(bucket) == 1
                    else _compiled_fit.solve_compiled_batch
                )
                output = solve(
                    descriptor,
                    shared_coordinates,
                    value_stack[0] if len(bucket) == 1 else value_stack,
                    base_lower=base_lower,
                    base_upper=base_upper,
                    valid=None,
                    context=context,
                    requested_lower=requested_lower,
                    requested_upper=requested_upper,
                    requested_mask=requested_mask,
                    free_indices=free_index,
                    weights=(
                        None
                        if weight_stack is None
                        else weight_stack[0] if len(bucket) == 1 else weight_stack
                    ),
                    poisson=counted,
                    loss=opts.loss,
                    max_nfev=opts.max_nfev,
                    ftol=1.0e-8,
                    xtol=1.0e-8,
                    gtol=1.0e-8,
                    authored_seeds=(
                        None
                        if authored_stack is None
                        else authored_stack[0]
                        if len(bucket) == 1
                        else authored_stack
                    ),
                    use_authored=(
                        bool(authored_flags[0])
                        if len(bucket) == 1
                        else authored_flags
                    ),
                    warm_seeds=(
                        None
                        if warm_stack is None
                        else warm_stack[0]
                        if len(bucket) == 1
                        else warm_stack
                    ),
                    use_warm=(
                        bool(warm_flags[0]) if len(bucket) == 1 else warm_flags
                    ),
                    coordinates_are_canonical=(
                        descriptor.coordinate_origin is not None
                    ),
                )
                check()
            except (FitCancelled, FitDeadlineExceeded):
                raise
            except Exception as error:
                message = str(error) or type(error).__name__
                for cell in bucket:
                    failures[cell] = message
                continue

            fixed_index = tuple(
                index
                for index in range(parameter_count)
                if index not in free_indices
            )
            for local, cell in enumerate(bucket):
                item = prepared[cell]
                if int(output.status[local]) in {
                    _compiled_fit.STATUS_INVALID,
                    _compiled_fit.STATUS_NO_CANDIDATE,
                }:
                    continue
                covariance_valid = bool(output.covariance_valid[local])
                if item["binned"]:
                    fitted = item["model"].evaluate(
                        item["coords"],
                        output.parameters[local],
                    ).reshape(-1)
                    residuals = item["values"] - fitted
                    reduced = float(
                        np.dot(residuals, residuals)
                        / max(item["values"].size - len(free_indices), 1)
                    )
                    compiled_reduced = float(output.reduced_chi_square[local])
                    output.reduced_chi_square[local] = reduced
                    item["final_fitted"] = _readonly(fitted)
                    item["final_residuals"] = _readonly(residuals)
                    if (
                        fitted.shape != item["values"].shape
                        or not np.all(np.isfinite(fitted))
                        or not np.all(np.isfinite(residuals))
                    ):
                        item["final_error"] = (
                            "compiled fit returned invalid full-data arrays"
                        )
                    if covariance_valid:
                        if compiled_reduced > 0.0:
                            ratio = reduced / compiled_reduced
                            output.covariance[local] *= ratio
                            output.standard_errors[local] *= math.sqrt(ratio)
                        elif reduced > 0.0:
                            output.covariance_valid[local] = False
                            covariance_valid = False
                if not covariance_valid:
                    output.covariance[local].fill(np.nan)
                    output.standard_errors[local].fill(np.nan)
                    if fixed_index:
                        output.covariance[local, fixed_index, :] = 0.0
                        output.covariance[local, :, fixed_index] = 0.0
                        output.standard_errors[local, fixed_index] = 0.0

            statuses = np.asarray(output.status)
            active = ~np.isin(
                statuses,
                (
                    _compiled_fit.STATUS_INVALID,
                    _compiled_fit.STATUS_NO_CANDIDATE,
                ),
            )
            covariance_rows = active & np.asarray(output.covariance_valid)
            batch_error: str | None = None
            try:
                expected_points = value_stack.shape[1]
                if (
                    output.parameters.shape != (len(bucket), parameter_count)
                    or output.standard_errors.shape
                    != (len(bucket), parameter_count)
                    or output.covariance.shape
                    != (len(bucket), parameter_count, parameter_count)
                    or output.fitted_values.shape
                    != (len(bucket), expected_points)
                    or output.residuals.shape != (len(bucket), expected_points)
                    or output.reduced_chi_square.shape != (len(bucket),)
                ):
                    raise ValueError("compiled fit returned invalid batch shapes")
                if (
                    not np.all(np.isfinite(output.parameters[active]))
                    or not np.all(np.isfinite(output.fitted_values[active]))
                    or not np.all(np.isfinite(output.residuals[active]))
                    or not np.all(
                        np.isfinite(output.reduced_chi_square[active])
                    )
                    or np.any(output.reduced_chi_square[active] < 0.0)
                ):
                    raise ValueError("compiled fit returned invalid batch arrays")
                if bool(np.any(covariance_rows)):
                    covariance_values = output.covariance[covariance_rows]
                    error_values = output.standard_errors[covariance_rows]
                    diagonal = np.diagonal(
                        covariance_values,
                        axis1=1,
                        axis2=2,
                    )
                    if (
                        not np.all(np.isfinite(covariance_values))
                        or not np.all(np.isfinite(error_values))
                        or not np.allclose(
                            covariance_values,
                            np.swapaxes(covariance_values, 1, 2),
                            rtol=1e-12,
                            atol=1e-15,
                        )
                        or np.any(diagonal < 0.0)
                        or np.any(error_values < 0.0)
                        or not np.allclose(
                            error_values**2,
                            diagonal,
                            rtol=1e-10,
                            atol=1e-15,
                        )
                    ):
                        raise ValueError(
                            "compiled fit returned invalid batch covariance"
                        )
            except Exception as error:
                batch_error = str(error) or type(error).__name__

            for backing in (
                output.parameters,
                output.standard_errors,
                output.covariance,
                output.fitted_values,
                output.residuals,
                output.reduced_chi_square,
                output.covariance_valid,
                output.success,
                output.status,
                output.coordinate_origins,
            ):
                backing.setflags(write=False)
            empty_units = MappingProxyType({
                name: "" for name in model.parameter_names
            })
            for local, cell in enumerate(bucket):
                item = prepared[cell]
                status = int(output.status[local])
                if status in {
                    _compiled_fit.STATUS_INVALID,
                    _compiled_fit.STATUS_NO_CANDIDATE,
                }:
                    failures[cell] = _compiled_fit.termination_message(status)
                    continue
                if batch_error is not None or "final_error" in item:
                    failures[cell] = batch_error or item["final_error"]
                    continue
                parameters = output.parameters[local]
                errors = output.standard_errors[local]
                covariance = output.covariance[local]
                fitted = (
                    item["final_fitted"]
                    if item["binned"]
                    else output.fitted_values[local]
                )
                residuals = (
                    item["final_residuals"]
                    if item["binned"]
                    else output.residuals[local]
                )
                indices = _readonly(item["indices"])
                reduced = float(output.reduced_chi_square[local])
                covariance_valid = bool(output.covariance_valid[local])
                try:
                    if indices.shape != fitted.shape:
                        raise ValueError(
                            "selected indices must identify every fitted observation"
                        )
                    results[cell] = _fit_result_from_validated_batch_row(
                        model=item["model"],
                        parameter_values=parameters,
                        standard_errors=errors,
                        covariance=covariance,
                        fitted_values=fitted,
                        residuals=residuals,
                        selected_indices=indices,
                        source_revision=item["revision"],
                        success=bool(output.success[local]),
                        message=_compiled_fit.termination_message(status),
                        reduced_chi_square=reduced,
                        covariance_valid=covariance_valid,
                        parameter_units=empty_units,
                        fixed_parameter_names=fixed_names,
                    )
                except Exception as error:
                    failures[cell] = str(error) or type(error).__name__
                else:
                    failures[cell] = None
        return tuple(results), tuple(failures)

    def fit(
        self,
        model: str | FitModelSpec,
        coordinates: Sequence[np.ndarray] | RegularImageFitInput,
        observations: np.ndarray | None = None,
        *,
        observation_sigma: np.ndarray | None = None,
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
        descriptor = self._compiled_descriptor(spec)
        if isinstance(coordinates, RegularImageFitInput):
            if observations is not None:
                raise TypeError(
                    "observations belong inside RegularImageFitInput and must not "
                    "also be passed separately"
                )
            if observation_sigma is not None:
                raise TypeError(
                    "regular-image fitting has no per-point sigma channel"
                )
            if selected_indices is not None:
                raise TypeError(
                    "regular-image selected_indices belong inside "
                    "RegularImageFitInput"
                )
            if not (spec.capabilities & _REGULAR_IMAGE_CAPABILITIES):
                raise ValueError(
                    "this model does not declare a regular-image capability"
                )
            from ._fit_radial import fit_regular_separable_image

            return fit_regular_separable_image(
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

        if descriptor is not None:
            results, failures = self._fit_compiled_batch(
                spec,
                descriptor,
                (tuple(coordinates),),
                (observations,),
                observation_sigmas=(observation_sigma,),
                selected_indices=(selected_indices,),
                data_revisions=(data_revision,),
                initial=initial,
                warm_starts=(warm_start,),
                bounds=bounds,
                options=opts,
                cancelled=cancelled,
            )
            result = results[0]
            if result is None:
                raise ValueError(failures[0] or "compiled fit failed")
            return result

        coords = _coordinate_arrays(tuple(coordinates), spec.independent_arity)
        values = np.asarray(observations, dtype=np.float64).reshape(-1)
        if any(item.shape != values.shape for item in coords):
            raise ValueError("coordinates and observations must have equal flattened shape")
        sigma = None
        if observation_sigma is not None:
            sigma = np.asarray(observation_sigma, dtype=np.float64).reshape(-1)
            if sigma.shape != values.shape:
                raise ValueError("observation_sigma must match observations")
            if spec.targets == (FitTarget.HISTOGRAM,):
                raise ValueError(
                    "histogram targets weight themselves by counts; an external "
                    "sigma has no meaning there"
                )
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
            if sigma is not None:
                sigma = sigma[finite]
        if _DOMAIN_ANCHORED in spec.capabilities:
            spec = spec.anchored_at(float(np.min(coords[0])))
        counted_observations = spec.targets == (FitTarget.HISTOGRAM,)
        solver_coords, solver_values = coords, values
        weight_roots: np.ndarray | None = None
        binned_statistics = False
        if sigma is not None:
            # Known per-point uncertainty weights the residuals by 1/sigma.
            # A non-positive or non-finite sigma cannot weight anything, and
            # the boolean-rate endpoints (p in {0,1}) legitimately report a
            # zero sample spread: those points take the strongest honest
            # weight -- the smallest positive sigma present.  With no
            # positive sigma at all the data is spreadless and the fit is
            # the ordinary unweighted one.
            usable_sigma = sigma[np.isfinite(sigma) & (sigma > 0.0)]
            if usable_sigma.size:
                floor = float(np.min(usable_sigma))
                bounded = np.where(
                    np.isfinite(sigma) & (sigma > 0.0), sigma, floor
                )
                weight_roots = 1.0 / bounded
        elif (
            not counted_observations
            and spec.independent_arity == 1
            and opts.max_exact_points is not None
            and values.size > 2 * opts.max_exact_points
        ):
            compressed = _binned_curve_statistics(
                coords[0], values, opts.max_exact_points
            )
            if compressed is not None:
                solver_coords, solver_values, weight_roots = compressed
                binned_statistics = True
        start = time.monotonic()
        invalid_residual = np.finfo(np.float64).max ** 0.25

        def check() -> None:
            if cancelled is not None and cancelled():
                raise FitCancelled("fit cancelled")
            if (
                opts.deadline_seconds is not None
                and time.monotonic() - start > opts.deadline_seconds
            ):
                raise FitDeadlineExceeded("fit deadline exceeded")

        def poisson_deviance(
            predicted: np.ndarray,
            observed: np.ndarray,
        ) -> np.ndarray:
            expected = np.maximum(predicted, _COUNT_FLOOR)
            with np.errstate(divide="ignore", invalid="ignore"):
                logarithm = np.where(
                    observed > 0.0,
                    observed * np.log(observed / expected),
                    0.0,
                )
            deviance = 2.0 * np.maximum(
                expected - observed + logarithm,
                0.0,
            )
            return np.copysign(np.sqrt(deviance), expected - observed)

        default_bounds = (
            spec.bounds_initializer(solver_coords, solver_values)
            if spec.bounds_initializer is not None
            else None
        )
        lower, upper = _solver_bounds(spec, default_bounds, bounds)
        if spec.targets == (FitTarget.HISTOGRAM,):
            # A histogram cannot resolve a width finer than its own binning: a
            # component narrower than a bin describes one bar, not a
            # distribution, and on sparse data that is exactly what the best
            # fit becomes -- a spike standing on a single tall bin beside a
            # broad partner covering everything else.  Widths therefore start
            # at half a bin: they are the positive parameters measured along
            # the value axis, which is the sigmas and not a splitting or an
            # amplitude.
            steps = np.diff(np.unique(solver_coords[0]))
            step = float(np.median(steps)) if steps.size else 0.0
            if step > 0.0:
                floor = 0.5 * step
                for index, parameter in enumerate(spec.parameters):
                    if (
                        parameter.domain is ParameterDomain.POSITIVE
                        and parameter.unit_relation is UnitRelation.AXIS_0
                        and lower[index] < floor < upper[index]
                    ):
                        lower[index] = floor
        fixed_names, free_indices = _fixed_parameter_partition(spec, bounds)
        if values.size <= len(free_indices):
            raise ValueError("fit requires more finite observations than free parameters")
        if not free_indices:
            fitted = spec.evaluate(coords, lower).reshape(-1)
            if fitted.shape != values.shape or not np.all(np.isfinite(fitted)):
                raise RuntimeError("fixed fit evaluation is non-finite")
            residuals = values - fitted
            quality = (
                poisson_deviance(fitted, values)
                if counted_observations
                else residuals if weight_roots is None else residuals * weight_roots
            )
            count = len(spec.parameters)
            return FitResult(
                spec, lower, np.zeros(count), np.zeros((count, count)),
                fitted, residuals, indices, data_revision, True,
                "all parameters fixed", float(np.dot(quality, quality) / values.size),
                fixed_parameter_names=fixed_names,
            )
        free_index = np.asarray(free_indices, dtype=np.int64)

        def expand_parameters(free: Sequence[float]) -> np.ndarray:
            complete = lower.copy()
            complete[free_index] = free
            return complete

        complete_seeds = _initial_candidates(
            spec,
            solver_coords,
            solver_values,
            initial,
            warm_start,
        )
        low_inside = np.nextafter(lower, upper)
        high_inside = np.nextafter(upper, lower)
        prepared_seeds: list[np.ndarray] = []
        seen_seeds: set[bytes] = set()
        for complete_seed in complete_seeds:
            free_seed = np.asarray(complete_seed, dtype=np.float64)[free_index]
            free_seed = np.minimum(
                np.maximum(free_seed, low_inside[free_index]),
                high_inside[free_index],
            )
            key = free_seed.tobytes()
            if key not in seen_seeds:
                seen_seeds.add(key)
                prepared_seeds.append(free_seed)
        seeds = tuple(prepared_seeds)
        free_lower, free_upper = lower[free_index], upper[free_index]
        # A histogram's observations are COUNTS, and counts have their own
        # likelihood.  Fitted as though every bin were equally certain, a tall
        # peak outvotes a sparse one by the ratio of their heights, so the
        # best fit of a rare state over a common one puts BOTH components on
        # the common peak.  Dividing by sqrt(n) -- the obvious repair, and
        # what this did -- overshoots the other way: it makes an empty bin the
        # most certain measurement on the axis, and the cheapest way to
        # satisfy it is a component one bin wide standing on a single tall
        # bar.  Measured on a rare state at 2%: that spike beats the true
        # solution under sqrt(n) weighting (16.21 against 16.55) and loses to
        # it under the deviance by eight to one (228 against 29).
        #
        # The deviance IS the likelihood: 2*(mu - n + n*ln(n/mu)) per bin, the
        # quantity whose sum is what a Poisson maximum-likelihood fit
        # minimises, written as a signed square root so an ordinary
        # least-squares solver minimises it unchanged.

        def deviance_residual(predicted: np.ndarray) -> np.ndarray:
            """The Poisson deviance of each bin, signed, as a residual."""

            return poisson_deviance(predicted, solver_values)

        def residual(parameters: np.ndarray) -> np.ndarray:
            check()
            predicted = spec.evaluate(
                solver_coords,
                expand_parameters(parameters),
            ).reshape(-1)
            if predicted.shape != solver_values.shape or not np.all(
                np.isfinite(predicted)
            ):
                return np.full(solver_values.shape, invalid_residual)
            if counted_observations:
                return deviance_residual(predicted)
            delta = predicted - solver_values
            return delta if weight_roots is None else delta * weight_roots

        def analytic_jacobian(parameters: np.ndarray) -> np.ndarray:
            check()
            jacobian = spec.evaluate_jacobian(
                solver_coords,
                expand_parameters(parameters),
            )[:, free_index]
            if not np.all(np.isfinite(jacobian)):
                raise FloatingPointError("analytic fit jacobian is non-finite")
            if not counted_observations:
                return (
                    jacobian
                    if weight_roots is None
                    else jacobian * weight_roots[:, None]
                )
            # d/dmu of the signed root, by the chain rule on the expression
            # above: |mu - n| / (mu * root), which is positive everywhere.
            # Where the model already matches the data both are zero; the
            # limit there is 1/sqrt(mu), since the root behaves as
            # (mu - n)/sqrt(n) in that neighbourhood.
            predicted = spec.evaluate(
                solver_coords,
                expand_parameters(parameters),
            ).reshape(-1)
            expected = np.maximum(predicted, _COUNT_FLOOR)
            root = np.abs(deviance_residual(predicted))
            scale = np.where(
                root > _COUNT_FLOOR,
                np.abs(expected - solver_values)
                / (expected * np.maximum(root, _COUNT_FLOOR)),
                1.0 / np.sqrt(expected),
            )
            return jacobian * scale[:, None]

        successful: list[tuple[float, Any]] = []
        unsuccessful: list[tuple[float, Any]] = []
        last_error: Exception | None = None
        for seed_index, seed in enumerate(seeds):
            check()
            try:
                candidate = least_squares(
                    residual,
                    seed,
                    bounds=(free_lower, free_upper),
                    loss=opts.loss,
                    max_nfev=opts.max_nfev,
                    x_scale="jac",
                    jac=(analytic_jacobian if spec.jacobian is not None else "2-point"),
                )
                check()
                solver_residual = np.asarray(candidate.fun, dtype=np.float64).reshape(-1)
                if (
                    solver_residual.shape != solver_values.shape
                    or not np.all(np.isfinite(solver_residual))
                    or bool(np.all(solver_residual == invalid_residual))
                ):
                    continue
                # Candidates compete on the quantity being minimised.
                rss = float(np.dot(solver_residual, solver_residual))
                if not math.isfinite(rss):
                    continue
                (successful if candidate.success else unsuccessful).append(
                    (rss, candidate)
                )
                if (
                    warm_start is not None
                    and seed_index == 0
                    and candidate.success
                    and rss <= _FIT_RSS_TIE_RELATIVE
                ):
                    break
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
        _rss, solved = best
        # The model and data residual are needed only for the winner.  Solver
        # residuals may encode a likelihood (histograms), so evaluate rather
        # than trying to invert ``solved.fun``.
        solved_parameters = expand_parameters(solved.x)
        fitted = spec.evaluate(coords, solved_parameters).reshape(-1)
        if fitted.shape != values.shape or not np.all(np.isfinite(fitted)):
            raise RuntimeError("winning fit evaluation is non-finite")
        residuals = values - fitted
        degrees = max(values.size - len(free_indices), 1)
        if binned_statistics:
            # The solver minimised the binned statistics; the reported quality
            # is the full data's, from the evaluation above.
            _rss = float(np.dot(residuals, residuals))
        elif weight_roots is not None:
            # Sigma-weighted: the quality IS the chi-square, so the reduced
            # value and the parameter covariance carry the per-point sigmas.
            weighted = residuals * weight_roots
            _rss = float(np.dot(weighted, weighted))
        reduced = float(_rss / degrees)
        free_covariance, covariance_valid = _covariance(solved.jac, reduced)
        covariance, errors = _expand_fixed_covariance(
            len(spec.parameters),
            free_indices,
            free_covariance,
            covariance_valid,
        )
        return FitResult(
            spec,
            solved_parameters,
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
            fixed_parameter_names=fixed_names,
        )



def _coordinate_arrays(coordinates: Sequence[np.ndarray], arity: int) -> ArrayTuple:
    arrays = tuple(np.asarray(item, dtype=np.float64).reshape(-1) for item in coordinates)
    if len(arrays) != arity:
        raise ValueError("coordinate arity does not match fit model")
    if arrays and any(item.shape != arrays[0].shape for item in arrays):
        raise ValueError("coordinate arrays must have equal shape")
    return arrays


def _binned_curve_statistics(
    x: np.ndarray,
    values: np.ndarray,
    bins: int,
) -> tuple[ArrayTuple, np.ndarray, np.ndarray] | None:
    """X-binned means with count weights -- a curve's sufficient statistics.

    Weighted least squares on (bin mean x, bin mean y, sqrt(count)) agrees
    with the full-data solution up to second order in the model's curvature
    within one bin; at the default 4096 bins that error sits far below the
    solver's own tolerance for every registered curve model, including a
    damped sine with hundreds of periods across the span.  The caller keeps
    the final model evaluation, per-point residuals and quality numbers on
    the FULL data, so the compression only decides where the solver iterates.
    Returns None when the span is degenerate or the data barely compresses.
    """

    low = float(np.min(x))
    high = float(np.max(x))
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        return None
    scale = (bins - 1) / (high - low)
    codes = ((x - low) * scale).astype(np.int64)
    np.clip(codes, 0, bins - 1, out=codes)
    counts = np.bincount(codes, minlength=bins)
    used = np.flatnonzero(counts)
    if used.size < 8 or used.size * 2 > x.size:
        return None
    weights = counts[used].astype(np.float64)
    x_means = np.bincount(codes, weights=x, minlength=bins)[used] / weights
    y_means = np.bincount(codes, weights=values, minlength=bins)[used] / weights
    return (x_means,), y_means, np.sqrt(weights)


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
        exact = False
        source = (
            requested
            if requested is not None and parameter.name in requested
            else defaults
        )
        if source is not None and parameter.name in source:
            source_low, source_high = source[parameter.name]
            exact = (
                source is requested
                and source_low is not None
                and source_high is not None
                and float(source_low) == float(source_high)
            )
            if source_low is not None:
                low = max(low, float(source_low))
            if source_high is not None:
                high = min(high, float(source_high))
        if low > high or (low == high and not exact):
            raise ValueError(f"empty bounds for parameter {parameter.name!r}")
        lower.append(low)
        upper.append(high)
    return np.asarray(lower), np.asarray(upper)


def _fixed_parameter_partition(
    model: FitModelSpec,
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    fixed_names = {
        name
        for name, pair in (bounds or {}).items()
        if pair[0] is not None and pair[1] is not None and pair[0] == pair[1]
    }
    return (
        tuple(name for name in model.parameter_names if name in fixed_names),
        tuple(
            index
            for index, name in enumerate(model.parameter_names)
            if name not in fixed_names
        ),
    )


def _expand_fixed_covariance(
    parameter_count: int,
    free_indices: Sequence[int],
    free_covariance: np.ndarray,
    covariance_valid: bool,
) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    if not covariance_valid:
        return covariance, np.full(parameter_count, np.nan, dtype=np.float64)
    if free_indices:
        covariance[np.ix_(free_indices, free_indices)] = free_covariance
    return covariance, np.sqrt(np.maximum(np.diag(covariance), 0.0))


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


def _poisson_kernel_input(x, values) -> tuple[np.ndarray, np.ndarray]:
    """Fresh, writable, C-contiguous arrays: the one array type the compiled
    Poisson-Gaussian kernels are specialised for (a read-only view would be
    a second compilation of the same kernel)."""

    coords = np.array(np.reshape(x, (1, -1)), dtype=np.float64, order="C")
    return coords, np.array(values, dtype=np.float64)


def _histogram_poisson_gaussian(x, amplitude, rate, sigma):
    """The Poisson law extended to a real photon number through the Gamma
    function, ``p(u) = rate^u e^-rate / Gamma(u+1)`` on ``u >= 0``,
    convolved with the camera's Gaussian read noise:
    ``A / (sigma sqrt(2 pi)) int p(u) exp(-(x-u)^2 / 2 sigma^2) du``.  A
    smooth function of the bin centre like every other model; negative
    values (read noise below zero photons) are ordinary.  ``A`` is the
    counts times the bin width once the extended law carries unit mass
    (from about two photons up).  The quadrature has ONE implementation,
    the compiled kernel; the frozen anchors hold it to an independent one.
    (A NumPy twin evaluated over a pixel-value histogram cost forty cells'
    overlays 240 ms.)"""

    coords, parameters = _poisson_kernel_input(x, (amplitude, rate, sigma))
    return _compiled_fit._value_jacobian_poisson(coords, parameters)[0]


def _histogram_poisson_gaussian_jacobian(x, amplitude, rate, sigma):
    coords, parameters = _poisson_kernel_input(x, (amplitude, rate, sigma))
    return _compiled_fit._value_jacobian_poisson(coords, parameters)[1]


def _poisson_bimodal_left(
    x,
    left_rate,
    _rate_splitting,
    left_amplitude,
    left_sigma,
    _right_amplitude,
    _right_sigma,
):
    return _histogram_poisson_gaussian(x, left_amplitude, left_rate, left_sigma)


def _poisson_bimodal_right(
    x,
    left_rate,
    rate_splitting,
    _left_amplitude,
    _left_sigma,
    right_amplitude,
    right_sigma,
):
    return _histogram_poisson_gaussian(
        x, right_amplitude, left_rate + rate_splitting, right_sigma
    )


def _bimodal_poisson_gaussian(x, *parameters):
    coords, values = _poisson_kernel_input(x, parameters)
    return _compiled_fit._value_jacobian_poisson_bimodal(coords, values)[0]


def _bimodal_poisson_gaussian_jacobian(x, *parameters):
    coords, values = _poisson_kernel_input(x, parameters)
    return _compiled_fit._value_jacobian_poisson_bimodal(coords, values)[1]


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


def _bimodal_candidates(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Sequence[Sequence[float]]:
    """Seed a two-peak histogram fit from cuts through the distribution.

    Least squares finds the minimum it starts next to, so for a two-peak model
    the starting cut IS the fit.  Cutting the axis in half and taking the
    tallest bin on each side assumes the two peaks straddle the middle of the
    plotted range, and they usually do not: a distribution whose second state
    is rare sits entirely in the left third, and both halves of that cut then
    describe the same peak, whereupon the solver settles into two overlapping
    bells over one peak -- a fit that is visibly wrong beside an obviously
    better one, and stable, because from there no small step improves it.

    Cutting instead at deciles of the distribution itself -- and at the cut
    that best separates it -- always puts one candidate between the two peaks,
    wherever in the frame they sit.  The closest of those cuts is the start,
    and one solve from it is enough: measured over 200 two-state histograms,
    solving from all ten cuts and keeping the best answer was no better than
    solving from the best cut alone (41 misplaced peaks against 46) and eleven
    times slower -- 105 distribution panels in a calibration report is 1046
    least-squares solves against 105.
    """

    x = np.asarray(coords[0], dtype=np.float64).reshape(-1)
    counts = np.asarray(y, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="stable")
    x, counts = x[order], np.clip(counts[order], 0.0, None)
    span = _span(x)
    step = _histogram_step(x)
    total = float(counts.sum())
    if x.size < 3 or total <= 0.0:
        midpoint = float((np.min(x) + np.max(x)) / 2.0) if x.size else 0.0
        height = max(float(np.max(counts)) if counts.size else 0.0, 0.0)
        return ((midpoint, span / 2.0, height, span / 10.0, height, span / 10.0),)

    seeds: list[tuple[float, ...]] = []
    for split in _histogram_cuts(x, counts):
        left = x <= split
        right = ~left
        left_weight, right_weight = float(counts[left].sum()), float(counts[right].sum())
        if left_weight <= 0.0 or right_weight <= 0.0:
            continue
        left_center = float(np.sum(x[left] * counts[left]) / left_weight)
        right_center = float(np.sum(x[right] * counts[right]) / right_weight)
        if not right_center > left_center:
            continue
        left_sigma = max(
            math.sqrt(float(np.sum(counts[left] * (x[left] - left_center) ** 2) / left_weight)),
            step,
        )
        right_sigma = max(
            math.sqrt(
                float(np.sum(counts[right] * (x[right] - right_center) ** 2) / right_weight)
            ),
            step,
        )
        seeds.append(
            (
                (left_center + right_center) / 2.0,
                right_center - left_center,
                max(float(np.max(counts[left])), 0.0),
                left_sigma,
                max(float(np.max(counts[right])), 0.0),
                right_sigma,
            )
        )
    if not seeds:
        midpoint = float((np.min(x) + np.max(x)) / 2.0)
        height = max(float(np.max(counts)), 0.0)
        return ((midpoint, span / 2.0, height, span / 10.0, height, span / 10.0),)

    # The closest seed first, so a single-start caller gets the best of them.
    seeds.sort(
        key=lambda seed: float(np.sum((_bimodal_gaussian(x, *seed) - counts) ** 2))
    )
    return tuple(seeds)


def _init_bimodal(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    return _bimodal_candidates(coords, y)[0]


def _histogram_step(x: np.ndarray) -> float:
    """The histogram's bin pitch: the median distance between distinct bins."""

    unique_x = np.unique(x)
    span = _span(x)
    step = (
        float(np.median(np.abs(np.diff(unique_x))))
        if unique_x.size > 1
        else max(span, np.finfo(np.float64).eps)
    )
    return max(step, np.finfo(np.float64).eps)


def _histogram_cuts(x: np.ndarray, counts: np.ndarray) -> tuple[float, ...]:
    """The cuts a two-peak seed is tried from, on a sorted histogram.

    Deciles of the distribution itself, then the cut that most separates
    its two halves (see ``_bimodal_candidates`` for why the cuts are the
    distribution's and not the frame's).
    """

    total = float(counts.sum())
    cumulative = np.cumsum(counts / total)
    splits = [
        float(x[int(np.searchsorted(cumulative, fraction))])
        for fraction in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        if int(np.searchsorted(cumulative, fraction)) < x.size
    ]
    mass = np.cumsum(counts)[:-1]
    first_moment = np.cumsum(counts * x)
    left_mass = np.maximum(mass, np.finfo(np.float64).tiny)
    right_mass = np.maximum(total - mass, np.finfo(np.float64).tiny)
    left_mean = first_moment[:-1] / left_mass
    right_mean = (first_moment[-1] - first_moment[:-1]) / right_mass
    between = left_mass * right_mass * (right_mean - left_mean) ** 2
    if between.size and np.any(np.isfinite(between)):
        splits.append(float(x[int(np.argmax(np.nan_to_num(between, nan=-np.inf)))]))
    return tuple(dict.fromkeys(splits))


def _poisson_moments(
    x: np.ndarray, counts: np.ndarray, step: float
) -> tuple[float, float, float]:
    """(amplitude, rate, sigma) of one Poisson-Gaussian component from a
    histogram sorted by ``x``.

    The mean of a Poisson-Gaussian is its rate and its variance the rate plus
    the read-noise variance, so the histogram's location and width seed
    both: the rate as the mass-weighted mean of the bins between the
    quartiles, the total width as the quartile range over 1.349 -- not raw
    moments: a photon-count histogram carries hot-pixel and cosmic-ray
    spikes far from its peak, and the moments of that put the seed in a flat
    valley from which neither solver finds the peak.  The read noise is the
    width's excess over the rate and never under a bin (the solver floors
    it at half a bin, as it does every width on a histogram).  The
    amplitude is the mass times the bin: the model is a density.
    """

    total = float(counts.sum())
    if total <= 0.0:
        return 0.0, max(float(np.mean(x)), 0.25 * step), max(_span(x) / 6.0, step)
    cumulative = np.cumsum(counts)
    lower, upper = (
        float(x[min(int(np.searchsorted(cumulative, fraction * total)), x.size - 1)])
        for fraction in (0.25, 0.75)
    )
    core = (x >= lower) & (x <= upper)
    rate = max(float(np.sum(x[core] * counts[core]) / float(counts[core].sum())), 0.25 * step)
    width = (upper - lower) / 1.349
    sigma = max(math.sqrt(max(width * width - rate, 0.0)), step)
    return total * step, rate, sigma


def _sorted_histogram(coords: ArrayTuple, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(coords[0], dtype=np.float64).reshape(-1)
    counts = np.asarray(y, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="stable")
    return x[order], np.clip(counts[order], 0.0, None)


def _init_poisson_histogram(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    x, counts = _sorted_histogram(coords, y)
    return _poisson_moments(x, counts, _histogram_step(x))


def _init_poisson_bimodal(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    """The two-state Poisson-Gaussian seed from the same cuts as the Gaussian
    seed: each side's quartiles give its rate and read noise, and the
    closest cut (``_poisson_seed_distance``) is the one solved from, as the
    compiled seeder does."""

    x, counts = _sorted_histogram(coords, y)
    span = _span(x)
    step = _histogram_step(x)
    total = float(counts.sum())

    def fallback() -> tuple[float, ...]:
        midpoint = float((np.min(x) + np.max(x)) / 2.0) if x.size else 0.0
        return (
            max(midpoint - span / 4.0, 0.25 * step),
            span / 2.0,
            0.5 * total * step,
            max(span / 10.0, step),
            0.5 * total * step,
            max(span / 10.0, step),
        )

    if x.size < 3 or total <= 0.0:
        return fallback()
    seeds: list[tuple[float, ...]] = []
    for split in _histogram_cuts(x, counts):
        left = x <= split
        right = ~left
        if float(counts[left].sum()) <= 0.0 or float(counts[right].sum()) <= 0.0:
            continue
        left_amplitude, left_rate, left_sigma = _poisson_moments(
            x[left], counts[left], step
        )
        right_amplitude, right_rate, right_sigma = _poisson_moments(
            x[right], counts[right], step
        )
        if not right_rate > left_rate:
            continue
        seeds.append((
            left_rate,
            right_rate - left_rate,
            left_amplitude,
            left_sigma,
            right_amplitude,
            right_sigma,
        ))
    if not seeds:
        return fallback()
    return min(seeds, key=lambda seed: _poisson_seed_distance(x, counts, seed))


def _poisson_seed_distance(x: np.ndarray, counts: np.ndarray, seed: Sequence[float]) -> float:
    """How far a two-state seed is from the histogram, by each component's
    Gaussian approximation (variance ``rate + sigma^2``): one exponential
    per bin ranks the cuts and costs nothing per cell.  The same arithmetic
    as ``_fit_compiled._poisson_bimodal_score``, so both paths solve from
    the same cut."""

    left_rate, splitting, left_amplitude, left_sigma, right_amplitude, right_sigma = seed
    predicted = np.zeros_like(counts)
    for amplitude, rate, sigma in (
        (left_amplitude, left_rate, left_sigma),
        (right_amplitude, left_rate + splitting, right_sigma),
    ):
        variance = rate + sigma * sigma
        predicted += (
            amplitude / (math.sqrt(2.0 * math.pi) * math.sqrt(variance))
            * np.exp(-0.5 * (x - rate) ** 2 / variance)
        )
    return float(np.sum((predicted - counts) ** 2))


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
    # Anchored at the window start: the phase is measured from there.
    return amplitude, offset, frequency, decay_time, 0.0


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
    # The model is anchored at the start of this window, so the amplitude is
    # the height there -- no extrapolation back to a distant origin.
    amplitude = float(y[np.argmin(x)] - offset)
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
            compiled_descriptor=_compiled_fit.lorentzian_descriptor(),
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
            compiled_descriptor=_compiled_fit.gaussian_offset_descriptor(),
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
            compiled_descriptor=_compiled_fit.histogram_gaussian_descriptor(),
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
            compiled_descriptor=_compiled_fit.bimodal_gaussian_descriptor(),
        ),
        FitModelSpec(
            "histogram_poisson_gaussian",
            "Poisson-Gaussian",
            1,
            (
                FitParameterSpec(
                    "amplitude", VALUE, NONNEGATIVE, display_label=r"$A$"
                ),
                # A rate is a position on the axis, not a width: zero photons
                # is a value it may take (the model is then empty), and the
                # histogram's half-bin floor on positive widths is not for it.
                FitParameterSpec(
                    "rate",
                    AXIS_0,
                    NONNEGATIVE,
                    display_label=r"$\lambda$",
                    affine_point=True,
                ),
                FitParameterSpec(
                    "sigma", AXIS_0, POSITIVE, display_label=r"$\sigma$"
                ),
            ),
            "rate",
            _histogram_poisson_gaussian,
            _init_poisson_histogram,
            (FitTarget.HISTOGRAM,),
            formula=(
                r"$f(x)=\frac{A}{\sigma\sqrt{2\pi}}\int_0^{\infty}"
                r"\frac{\lambda^u e^{-\lambda}}{\Gamma(u+1)}"
                r"\,e^{-\frac{1}{2}((x-u)/\sigma)^2}\,du$"
            ),
            jacobian=_histogram_poisson_gaussian_jacobian,
            compiled_descriptor=(
                _compiled_fit.histogram_poisson_gaussian_descriptor()
            ),
        ),
        FitModelSpec(
            "bimodal_poisson_gaussian",
            "Bimodal Poisson-Gaussian",
            1,
            (
                FitParameterSpec(
                    "left_rate",
                    AXIS_0,
                    NONNEGATIVE,
                    display_label=r"$\lambda_L$",
                    affine_point=True,
                ),
                FitParameterSpec(
                    "rate_splitting",
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
            "rate_splitting",
            _bimodal_poisson_gaussian,
            _init_poisson_bimodal,
            (FitTarget.HISTOGRAM,),
            formula=(
                r"$f(x)=A_L P(x;\lambda_L,\sigma_L)+A_R P(x;\lambda_L+\delta,\sigma_R),"
                r"\ P(x;\lambda,\sigma)=\frac{1}{\sigma\sqrt{2\pi}}\int_0^{\infty}"
                r"\frac{\lambda^u e^{-\lambda}}{\Gamma(u+1)}"
                r"\,e^{-\frac{1}{2}((x-u)/\sigma)^2}\,du$"
            ),
            jacobian=_bimodal_poisson_gaussian_jacobian,
            presentation=FitPresentationSpec(
                components=(
                    FitComponentSpec("left", _poisson_bimodal_left),
                    FitComponentSpec("right", _poisson_bimodal_right),
                ),
            ),
            compiled_descriptor=(
                _compiled_fit.bimodal_poisson_gaussian_descriptor()
            ),
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
                r"$f(x)=H[L(x;x_0-\delta/2,\mathrm{FWHM})+L(x;x_0+\delta/2,\mathrm{FWHM})]+B$"
            ),
            jacobian=_symmetric_lorentzian_doublet_jacobian,
            candidate_initializer=_doublet_candidates,
            bounds_initializer=_doublet_bounds,
            compiled_descriptor=(
                _compiled_fit.symmetric_lorentzian_doublet_descriptor()
            ),
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
                FitParameterSpec("phase", RADIAN, PHASE, display_label=r"$\phi$"),
            ),
            "decay_time",
            _damped_sine,
            _init_damped_sine,
            (FitTarget.SERIES,),
            formula=(
                r"$f(t)=A\sin(2\pi f (t-t_0)+\phi)e^{-(t-t_0)/\tau}+B$"
            ),
            jacobian=_damped_sine_jacobian,
            candidate_initializer=_damped_sine_candidates,
            bounds_initializer=_damped_sine_bounds,
            capabilities=frozenset({_DOMAIN_ANCHORED}),
            compiled_descriptor=_compiled_fit.damped_sine_descriptor(),
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
            formula=r"$f(t)=A e^{-(t-t_0)/\tau}+B$",
            jacobian=_exponential_decay_jacobian,
            candidate_initializer=_exponential_candidates,
            bounds_initializer=_exponential_bounds,
            capabilities=frozenset({_DOMAIN_ANCHORED}),
            compiled_descriptor=_compiled_fit.exponential_decay_descriptor(),
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
                r"$f(x,y)=A e^{-((x-x_0)^2/R_x^2+(y-y_0)^2/R_y^2)}+B$"
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
            capabilities=frozenset({"regular_image_separable"}),
            compiled_descriptor=(
                _compiled_fit.anisotropic_gaussian_center_descriptor()
            ),
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
            formula=r"$f(x,y)=A e^{-((x-x_0)^2+(y-y_0)^2)/R^2}+B$",
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
            compiled_descriptor=_compiled_fit.radial_gaussian_center_descriptor(),
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
    "FitModelRegistry",
    "FitModelSpec",
    "FitOptions",
    "FitParameterDisplay",
    "FitParameterSpec",
    "formula_symbols",
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
