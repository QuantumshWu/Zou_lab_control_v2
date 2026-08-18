"""What a calibration found, in the numbers an operator judges it by.

A calibration writes a large artifact -- every site's centre, every model's
weights, every threshold -- and a folder of pictures.  Neither answers the
first question anyone asks of it: how well does it read out, and which model
should the next measurement use?  Those numbers were all computed and none of
them were stated; an operator had to open a picture and read them off the
annotations, per site, per model.

This is that statement.  One dictionary, written beside the pictures, holding
what each model achieves and what it cost -- and nothing that is not measured
here, so there is no second place for it to disagree with.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .calibration import CalibrationResult, ReadoutModelKind


def _plain(value: Any) -> Any:
    """One record as JSON holds it, with numpy scalars spelled as numbers."""

    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("summary keys must be strings")
            result[key] = _plain(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    raise TypeError(f"summary contains non-plain {type(value).__name__}")


def _numbers(values: object, mask: object | None = None) -> dict[str, float | None]:
    """One array of per-site values, summarised the way a report reads it."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool).reshape(-1)]
    array = array[np.isfinite(array)]
    if not array.size:
        return {"median": None, "mean": None, "min": None, "max": None}
    # min and max, not "worst" and "best": which end is which depends on the
    # quantity, and a fidelity's worst is its minimum while an error rate's
    # worst is its maximum.
    return {
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _separation(
    signals: np.ndarray,
    occupied: np.ndarray,
    valid: np.ndarray,
    usable: np.ndarray,
) -> np.ndarray:
    """How far apart the two populations sit, in units of their own width.

    The number the histograms show.  Fidelity saturates -- 0.993 and 0.999
    are both "nearly always right" -- while this keeps growing, so it is what
    says whether one model actually collects more of the light than another.
    """

    separation = np.full(signals.shape[1], np.nan, dtype=float)
    for site in range(signals.shape[1]):
        if not usable[site]:
            continue
        selected = valid[:, site] & np.isfinite(signals[:, site])
        dark = signals[selected & ~occupied[:, site], site]
        bright = signals[selected & occupied[:, site], site]
        if dark.size < 3 or bright.size < 3:
            continue
        width = float(
            np.sqrt(0.5 * (dark.std(ddof=1) ** 2 + bright.std(ddof=1) ** 2))
        )
        if width > 0.0:
            separation[site] = float(bright.mean() - dark.mean()) / width
    return separation


def readout_summary(
    result: CalibrationResult,
    *,
    run_chain: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The headline numbers of one calibration, model by model.

    ``run_chain`` is the same provenance a saved panel keeps, in the same
    shape: the run records of this measurement and everything it was derived
    from -- the request, the device snapshots, the pulse.  The numbers are
    only readable beside what produced them, and an operator opening the one
    file a calibration leaves for a person should not have to go and find
    the artifact to learn the exposure the camera was actually at.
    """

    if not isinstance(result, CalibrationResult):
        raise TypeError("readout_summary needs a CalibrationResult")
    calibration = result.calibration
    report = result.report
    site_map = calibration.site_map
    occupied = np.asarray(report["labels_occupied"], dtype=bool)
    valid = np.asarray(report["labels_valid"], dtype=bool)
    contract = calibration.frame_contract

    labelled = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        loading = np.where(labelled > 0, occupied.sum(axis=0) / labelled, np.nan)

    models: dict[str, Any] = {}
    for model in calibration.models:
        model_report = report["models"][model.kind.value]
        usable = np.asarray(model.usable_sites, dtype=bool)
        signals = np.asarray(model_report["short_signals"], dtype=float)
        dark_right = np.asarray(model_report["site_fidelity_dark"], dtype=float)
        bright_right = np.asarray(model_report["site_fidelity_bright"], dtype=float)
        entry: dict[str, Any] = {
            "usable_sites": int(np.count_nonzero(usable)),
            "held_out_shots": int(np.sum(model_report["site_n_test"])),
            "fidelity": _numbers(model_report["site_fidelity"], usable),
            # What the fidelity is made of: a dark shot called bright, and a
            # bright shot called dark, are different failures with different
            # causes -- stray light against loss during the readout.
            "error_rate": {
                "dark_called_bright": _numbers(1.0 - dark_right, usable),
                "bright_called_dark": _numbers(1.0 - bright_right, usable),
            },
            "separation": _numbers(
                _separation(signals, occupied, valid, usable), usable
            ),
            "threshold": _numbers(model.thresholds, usable),
            "dark_level": _numbers(model.dark_mean, usable),
            "bright_level": _numbers(model.bright_mean, usable),
            "training_shots": {
                "dark": _numbers(model_report["site_n_train_dark"], usable),
                "bright": _numbers(model_report["site_n_train_bright"], usable),
            },
        }
        if model.kind is ReadoutModelKind.BOX:
            entry["integration"] = {
                "box": f"{2 * model.integration_half_width + 1}"
                f"x{2 * model.integration_half_width + 1}",
                "reducer": model.reducer,
            }
        else:
            entry["integration"] = {
                "box": f"{2 * model.integration_half_width + 1}"
                f"x{2 * model.integration_half_width + 1}",
                "weights": "measured from this run's own loaded-minus-empty frames",
                # How much of the atom's light the box can reach at all: the
                # ceiling on what any weighting inside it can collect.
                "light_inside_the_box": _numbers(
                    model_report.get("psf_box_light_fraction", ()), usable
                ),
                "spot_sigma_px": _numbers(
                    np.mean(
                        np.asarray(model_report["psf_fit_sigma_xy"], dtype=float),
                        axis=1,
                    ),
                    usable & np.asarray(model_report["psf_fit_ok"], dtype=bool),
                ),
            }
        models[model.kind.value] = entry

    # Ranked by how often it is right, and where that ties -- fidelity
    # saturates, and three models all reading 100% of a well-loaded lattice
    # correctly is the normal case -- by how far apart it holds the two
    # populations, which does not saturate.
    ranked = sorted(
        (
            (
                entry["fidelity"]["mean"] if entry["fidelity"]["mean"] is not None else float("-inf"),
                entry["separation"]["median"] if entry["separation"]["median"] is not None else float("-inf"),
                name,
            )
            for name, entry in models.items()
        ),
        reverse=True,
    )
    return {
        "format": "calibration-summary.v1",
        "run": {
            "sites": int(site_map.n_sites),
            "samples": int(occupied.shape[0]),
            "labelled_shots": _numbers(labelled),
            "loading_rate": _numbers(loading),
            "frame_shape_yx": list(contract.image_shape),
            "roi_xywh": None if contract.roi_xywh is None else list(contract.roi_xywh),
            "binning_yx": list(contract.binning_yx),
            "readout_exposure_seconds": contract.exposure_seconds,
        },
        "models": models,
        # Which one the next measurement will use, and which one this run says
        # it should: a calibration that measured a better model and shipped
        # the other is worth being told about.
        "default_model": calibration.default_model_kind.value,
        "best_model": ranked[0][-1] if ranked else None,
        "run_chain": [_plain(record) for record in run_chain],
    }


def summary_lines(summary: Mapping[str, Any]) -> tuple[str, ...]:
    """The same numbers as lines, for a log or a progress report."""

    lines = [
        f"{summary['run']['sites']} sites over {summary['run']['samples']} samples, "
        f"loading {_percent(summary['run']['loading_rate']['median'])}"
    ]
    for name, entry in summary["models"].items():
        # A leading space is not allowed here: these lines are also reported
        # as node progress, and a progress message is canonical text.
        mark = "*" if name == summary.get("best_model") else "-"
        lines.append(
            f"{mark} {name:11s} fidelity {_percent(entry['fidelity']['mean'])} "
            f"(worst site {_percent(entry['fidelity']['min'])}), "
            f"dark->bright {_percent(entry['error_rate']['dark_called_bright']['mean'])}, "
            f"bright->dark {_percent(entry['error_rate']['bright_called_dark']['mean'])}, "
            f"separation {_value(entry['separation']['median'])}"
        )
    return tuple(lines)


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * float(value):.2f}%"


def _value(value: float | None) -> str:
    return "--" if value is None else f"{float(value):.2f}"


__all__ = ["readout_summary", "summary_lines"]
