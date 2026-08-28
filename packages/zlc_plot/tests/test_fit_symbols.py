"""What a fit formula writes is what the operator may type.

The panel drew f(t)=A e^{-(t-t_0)/tau}+B and then asked, in the box under
it, for "amplitude" and "decay_time".  Two vocabularies for one set of
parameters, only one of them ever on screen, and the box's own hint said
"name=value" without saying which names -- so the only way to learn the
second vocabulary was to guess a word and read the refusal.

These pin the rule for every model there is and every model there will be.
"""

from __future__ import annotations

import pytest

from zlc_plot.fit import (
    FitParameterSpec,
    ParameterDomain,
    UnitRelation,
    builtin_fit_models,
    formula_symbols,
)


def test_every_parameter_is_written_in_its_own_formula() -> None:
    """The whole rule, over every model, as one assertion each."""

    models = builtin_fit_models()
    assert models, "there are no fit models to check"
    for model in models:
        assert model.formula, f"{model.model_id} has no formula to agree with"
        printed = formula_symbols(model.formula)
        for parameter in model.parameters:
            assert parameter.symbol, parameter.name
            assert parameter.symbol in printed, (
                "%s asks for %r, which its formula never writes: %s"
                % (model.model_id, parameter.symbol, model.formula)
            )


def test_no_two_parameters_of_one_model_share_a_symbol() -> None:
    """Two boxes with one name is a box the operator cannot address."""

    for model in builtin_fit_models():
        symbols = list(model.symbols)
        assert len(symbols) == len(set(symbols)), (model.model_id, symbols)


def test_a_model_refuses_to_disagree_with_its_own_formula() -> None:
    """The rule is enforced where a model is built, not just asserted here.

    Otherwise the two vocabularies stay together only while somebody
    remembers to move both -- which is how they came apart.
    """

    from zlc_plot.fit import FitModelSpec, FitTarget

    def evaluate(x, *values):
        return x

    def initialise(coords, values):
        return (1.0,)

    with pytest.raises(ValueError) as refused:
        FitModelSpec(
            "invented",
            "Invented",
            1,
            (
                FitParameterSpec(
                    "amplitude",
                    UnitRelation.VALUE,
                    display_label=r"$A$",
                ),
            ),
            "amplitude",
            evaluate,
            initialise,
            (FitTarget.SERIES,),
            # The formula writes Q; the parameter is offered as A.
            formula=r"$f(x)=Q$",
        )
    assert "never writes" in str(refused.value)
    assert "'A'" in str(refused.value)


def test_a_label_that_is_not_one_symbol_is_refused_not_guessed() -> None:
    """A display label is one symbol by construction, so reading it is exact."""

    assert (
        FitParameterSpec(
            "sigma_left", UnitRelation.VALUE, display_label=r"$\sigma_L$"
        ).symbol
        == "sigma_L"
    )
    assert (
        FitParameterSpec(
            "width", UnitRelation.VALUE, display_label=r"$\mathrm{FWHM}$"
        ).symbol
        == "FWHM"
    )
    # No label to read: nothing is printed, so there are not two
    # vocabularies and the name is what an operator would type.
    assert (
        FitParameterSpec("value", UnitRelation.VALUE).symbol == "value"
    )
    with pytest.raises(ValueError) as refused:
        FitParameterSpec(
            "both", UnitRelation.VALUE, display_label=r"$A+B$"
        )
    assert "one symbol" in str(refused.value)


def test_the_formula_reader_keeps_a_subscript_with_its_command() -> None:
    """``\\sigma_L`` is one symbol, not sigma and then L."""

    assert "sigma_L" in formula_symbols(r"$A_L e^{-x/\sigma_L}$")
    assert "FWHM" in formula_symbols(r"$\mathrm{FWHM}/2$")
    assert "x_0" in formula_symbols(r"$(x-x_0)^2$")
    # It is generous on purpose: it CHECKS declared symbols, never invents
    # them, so structural noise costs nothing.
    assert "frac" in formula_symbols(r"$\frac{a}{b}$")


def test_the_whole_vocabulary_survives_the_placeholder() -> None:
    """A hint that names half the parameters is the old silence, shortened.

    ``qt_form`` truncates a placeholder to 48 characters (the tooltip keeps
    the whole line), and the symbols are written first so the clip falls on
    the syntax example.  That only works while the vocabulary itself fits.
    """

    for model in builtin_fit_models():
        printed = ", ".join(model.symbols)
        assert len(printed) <= 48, (model.model_id, printed)


def test_an_amplitude_is_not_glued_to_its_exponential() -> None:
    """``Ae^{...}`` reads as one symbol, to a person and to the checker.

    Three formulas wrote it that way while the Gaussian beside them wrote
    ``A e^{...}``, so the same amplitude was legible in one model and not in
    the next.
    """

    for model in builtin_fit_models():
        assert "Ae^" not in (model.formula or ""), model.model_id
