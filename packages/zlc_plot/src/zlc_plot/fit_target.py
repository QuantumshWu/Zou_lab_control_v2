"""Lightweight canonical grammar for Panel/Figure fit targets."""

from __future__ import annotations

from collections.abc import Mapping
import keyword

from ._validation import finite_real


def _parameter_values(
    values: Mapping[str, object],
    name: str,
) -> dict[str, float]:
    source = values.get(name)
    if source is None:
        return {}
    if not isinstance(source, Mapping):
        raise TypeError(f"fit {name} must be a parameter mapping")
    prepared: dict[str, float] = {}
    for parameter, value in source.items():
        if (
            not isinstance(parameter, str)
            or not parameter.isidentifier()
            or keyword.iskeyword(parameter)
        ):
            raise ValueError(f"fit {name} parameter names must be identifiers")
        prepared[parameter] = finite_real(
            value,
            f"fit {name} parameter {parameter!r}",
        )
    return prepared


def normalize_fit_target(target: Mapping[str, object] | None) -> dict[str, object]:
    """Return the strict canonical Panel/Figure fit document."""

    if target is None:
        return {}
    if not isinstance(target, Mapping):
        raise TypeError("fit target must be a mapping or None")
    values = dict(target)
    allowed = {
        "model",
        "selector_kind",
        "fixed",
        "initial",
        "bounds",
        "options",
        "fit_all_facets",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown canonical fit fields: {sorted(unknown)}")
    model = values.get("model")
    if model is None or not isinstance(model, str) or not model.strip():
        if values:
            raise ValueError("a non-empty fit target requires a model")
        return {}
    if model != model.strip():
        raise ValueError("fit model cannot have surrounding whitespace")
    result: dict[str, object] = {"model": model}

    selector = values.get("selector_kind")
    if selector is not None:
        if (
            not isinstance(selector, str)
            or not selector.strip()
            or selector != selector.strip()
        ):
            raise TypeError("fit selector_kind must be non-empty text or None")
        result["selector_kind"] = selector

    fixed = _parameter_values(values, "fixed")
    initial = _parameter_values(values, "initial")
    for parameter in fixed:
        initial.pop(parameter, None)
    if fixed:
        result["fixed"] = fixed
    if initial:
        result["initial"] = initial

    raw_bounds = values.get("bounds")
    if raw_bounds is not None:
        if not isinstance(raw_bounds, Mapping):
            raise TypeError("fit bounds must be a parameter mapping")
        bounds: dict[str, tuple[float | None, float | None]] = {}
        for parameter, pair in raw_bounds.items():
            if (
                not isinstance(parameter, str)
                or not parameter.isidentifier()
                or keyword.iskeyword(parameter)
            ):
                raise ValueError("fit bounds parameter names must be identifiers")
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError("each fit bound must contain low and high")
            low = None if pair[0] is None else finite_real(
                pair[0], f"fit lower bound {parameter!r}"
            )
            high = None if pair[1] is None else finite_real(
                pair[1], f"fit upper bound {parameter!r}"
            )
            if low is not None and high is not None and not low < high:
                raise ValueError(f"empty bounds for parameter {parameter!r}")
            if low is not None or high is not None:
                bounds[parameter] = (low, high)
        if bounds:
            result["bounds"] = bounds

    raw_options = values.get("options")
    if raw_options is not None:
        if not isinstance(raw_options, Mapping):
            raise TypeError("fit options must be a mapping")
        # FitOptions remains the one policy validator.  Import it only for the
        # uncommon advanced-options document, keeping ordinary PanelState
        # construction free of scipy.
        from .fit import FitOptions

        options = FitOptions(**dict(raw_options))
        result["options"] = {
            "loss": options.loss,
            "max_nfev": options.max_nfev,
            "deadline_seconds": options.deadline_seconds,
            "max_exact_points": options.max_exact_points,
        }
    all_facets = values.get("fit_all_facets", False)
    if not isinstance(all_facets, bool):
        raise TypeError("fit_all_facets must be bool")
    if all_facets:
        result["fit_all_facets"] = True
    return result


__all__ = ["normalize_fit_target"]
