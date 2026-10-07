"""What separates two populations from one, readable without a fit engine.

Two fitters decide by this one number and this one comparison: the plot's
bimodal fit (:mod:`zlc_plot.fit`) and the readout's two-state classification
(:mod:`zlc_atom.nodes.calibration.bimodal`).  The rule therefore has exactly
one owner, and the owner cannot be either of them -- it is this module, whose
whole content is the constant and :func:`decisive`.  Spelled once in each
fitter, it was ``>=`` in one and ``>`` in the other.

WHY A MODULE OF ITS OWN.  The readout's classification is deliberately
dependency-light -- it is the normal CDF and a threshold -- and it once
reached this number through the fit module, which then imported the compiled
engine with it: numba and llvmlite, half a second and a couple of hundred
megabytes.  Nothing noticed until the task console, which discovers its
logic nodes at startup and never renders a raster in its own process, was
found carrying the entire fit engine: that single edge, for a float, was the
largest block of a console's open.
"""

from __future__ import annotations

#: The evidence two populations must show over one before a two-population
#: fit keeps its own parameters: the BIC gain of the pair over the nested
#: single population, ten being Kass and Raftery's "very strong" (a Bayes
#: factor of about 150).  A loaded site clears it by hundreds; a dark site
#: whose one Gaussian the fitter split in two, 1.7 sigma apart, came in at
#: +4.6 and was reported as loaded.
DECISIVE_BIC_GAIN = 10.0


def decisive(gain: float, threshold: float = DECISIVE_BIC_GAIN) -> bool:
    """Whether a BIC gain is decisive evidence for two populations over one.

    It must EXCEED the threshold, as Kass and Raftery's "very strong" is a
    gain over ten.  A NaN gain decides nothing and is never decisive.
    """

    return gain > threshold


__all__ = ["DECISIVE_BIC_GAIN", "decisive"]
