"""What separates two populations from one, readable without a fit engine.

Two fitters decide by this one number: the plot's bimodal fit
(:mod:`zlc_plot.fit`) and the readout's two-state classification
(:mod:`zlc_atom.nodes.calibration.bimodal`).  It therefore has exactly one
owner, and the owner cannot be either of them -- it is this module, whose
whole content is the constant.

WHY A MODULE OF ITS OWN.  :mod:`zlc_plot.fit` is the solver: importing it
brings numba and llvmlite and the compiled engine's forty dispatchers,
half a second and a couple of hundred megabytes, into whatever process
asks.  The readout's classification is deliberately
dependency-light -- it is the normal CDF and a threshold -- and it reached
this number through the solver.  Nothing noticed until the task console,
which discovers its logic nodes at startup and never renders a raster in
its own process, was found carrying the entire fit engine: that single
edge, for a float, was the largest block of a console's open.
"""

from __future__ import annotations

#: The evidence two populations must show over one before a two-population
#: fit keeps its own parameters: the BIC gain of the pair over the nested
#: single population, ten being Kass and Raftery's "very strong" (a Bayes
#: factor of about 150).  A loaded site clears it by hundreds; a dark site
#: whose one Gaussian the fitter split in two, 1.7 sigma apart, came in at
#: +4.6 and was reported as loaded.
DECISIVE_BIC_GAIN = 10.0


__all__ = ["DECISIVE_BIC_GAIN"]
