from __future__ import annotations

import pytest

from zlc_plot import FitTarget
from zlc_plot.fit import FitModelSpec, builtin_fit_models
from zlc_plot.fit import FitParameterSpec, UnitRelation


def test_builtin_models_declare_a_parameter_headline() -> None:
    models = {model.model_id: model for model in builtin_fit_models()}
    assert {
        "lorentzian": "center",
        "gaussian_offset": "center",
        "histogram_gaussian": "center",
        "bimodal_gaussian": "center",
        "histogram_poisson_gaussian": "rate",
        "bimodal_poisson_gaussian": "rate_splitting",
        "symmetric_lorentzian_doublet": "center",
        "damped_sine": "decay_time",
        "exponential_decay": "decay_time",
        "anisotropic_gaussian_center": "center_x",
        "radial_gaussian_center": "center_x",
    } == {name: model.headline for name, model in models.items()}
    assert all(model.headline in model.parameter_names for model in models.values())


def test_fit_model_rejects_a_headline_that_is_not_a_parameter() -> None:
    with pytest.raises(ValueError, match="headline must name a parameter"):
        FitModelSpec(
            "invalid_headline",
            "Invalid headline",
            1,
            (FitParameterSpec("value", UnitRelation.VALUE),),
            "missing",
            lambda x, value: x * 0.0 + value,
            lambda coordinates, values: (0.0,),
            (FitTarget.SERIES,),
        )
