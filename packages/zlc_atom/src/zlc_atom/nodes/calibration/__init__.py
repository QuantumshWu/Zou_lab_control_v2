"""Calibration task and the readout mathematics it owns."""

from .bimodal import (
    BimodalFit,
    fit_bimodal,
    gaussian_fidelity,
    normal_cdf,
    optimal_gaussian_threshold,
    per_site_fidelity,
)
from .artifact import CALIBRATION_ARTIFACT_CODEC
from .calibration import (
    AtomDetection,
    CalibrationResult,
    DEFAULT_READOUT_MODEL_CHOICE,
    FrameContract,
    READOUT_MODEL_CHOICES,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
    calibrate,
    classify_threshold,
    detect_sites,
    extract_box_signals,
    extract_psf_signals,
    readout_model_kind_from_choice,
)
from .logic_node import LOGIC_NODE
from .outputs import site_map_image_overlay
from .psf import gaussian_psf_kernel, normalized_psf_kernel
from .task import (
    CalibrationCapture,
    CalibrationRequest,
    CalibrationRunResult,
    CalibrationTask,
)

__all__ = [
    "BimodalFit",
    "AtomDetection",
    "CalibrationResult",
    "CalibrationCapture",
    "CalibrationRequest",
    "CalibrationRunResult",
    "CalibrationTask",
    "CALIBRATION_ARTIFACT_CODEC",
    "DEFAULT_READOUT_MODEL_CHOICE",
    "FrameContract",
    "LOGIC_NODE",
    "ReadoutModel",
    "ReadoutModelKind",
    "READOUT_MODEL_CHOICES",
    "SiteMap",
    "TrapCalibration",
    "calibrate",
    "classify_threshold",
    "detect_sites",
    "extract_box_signals",
    "extract_psf_signals",
    "fit_bimodal",
    "gaussian_fidelity",
    "gaussian_psf_kernel",
    "normal_cdf",
    "normalized_psf_kernel",
    "optimal_gaussian_threshold",
    "per_site_fidelity",
    "readout_model_kind_from_choice",
    "site_map_image_overlay",
]
