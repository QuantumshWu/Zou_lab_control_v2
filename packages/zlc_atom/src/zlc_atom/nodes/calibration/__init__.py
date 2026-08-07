"""Calibration task and the readout mathematics it owns."""

from .bimodal import (
    BimodalFit,
    confidence_weighted_fidelity,
    fit_bimodal,
    gaussian,
    gaussian_fidelity,
    normal_cdf,
    optimal_gaussian_threshold,
    per_site_fidelity,
)
from .calibration import (
    AtomDetection,
    CalibrationResult,
    TrapCalibration,
    classify_threshold,
    detect,
    extract_box_signals,
    extract_psf_signals,
    signals,
)
from .logic_node import LOGIC_NODE
from .psf import gaussian_psf_kernel, normalized_psf_kernel
from .task import CalibrationRunResult, CalibrationTask

__all__ = [
    "BimodalFit",
    "AtomDetection",
    "CalibrationResult",
    "CalibrationRunResult",
    "CalibrationTask",
    "LOGIC_NODE",
    "TrapCalibration",
    "classify_threshold",
    "confidence_weighted_fidelity",
    "detect",
    "extract_box_signals",
    "extract_psf_signals",
    "fit_bimodal",
    "gaussian",
    "gaussian_fidelity",
    "gaussian_psf_kernel",
    "normal_cdf",
    "normalized_psf_kernel",
    "optimal_gaussian_threshold",
    "per_site_fidelity",
    "signals",
]
