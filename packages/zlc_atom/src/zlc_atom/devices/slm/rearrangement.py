"""Shared, headless SLM rearrangement parameters, target policy and playback."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from time import perf_counter, monotonic, time_ns

import numpy as np

from zlc_atom.authoring import AuthoringChoice, AuthoringField
from .solver import compute_rearrangement


REARRANGEMENT_FIELDS = (
    AuthoringField("target_rows", "int", "Target rows", 3, minimum=1),
    AuthoringField("target_columns", "int", "Target columns", 3, minimum=1),
    AuthoringField("frame_mode", "choice", "Frame count", "fixed",
                   choices=(AuthoringChoice("fixed", "Fixed frames"),
                            AuthoringChoice("camera_step", "Maximum camera step"))),
    AuthoringField("motion_frames", "int", "Total frames", 16, minimum=1, maximum=256,
                   enabled_when=("frame_mode", ("fixed",)),
                   description="Total displayed maps, including one initial map removing unused traps (when needed) and the destination; excludes the already displayed source. Movement follows without an additional hold."),
    AuthoringField("max_camera_step", "float", "Maximum camera step", 1., minimum=1e-9, unit="pixel",
                   enabled_when=("frame_mode", ("camera_step",)),
                   description="Maximum two-dimensional Euclidean movement per frame in original camera sensor pixels. The actual frame count is determined after occupancy and path planning."),
    AuthoringField("frame_rate_hz", "float", "Display frame rate", 60., minimum=1e-9, unit="Hz",
                   description="Requested display cadence, not a measured liquid-crystal response."),
    AuthoringField(
        "nominal_playback_seconds", "float", "Display duration", None,
        derived=True, unit="s",
        description=(
            "Total frame count divided by the SLM frame rate. Automatic frame count remains pending until occupancy and path planning. "
            "GPU computation, upload, preparation and any additional optical settling are excluded."
        ),
    ),
    AuthoringField("phase_method", "choice", "Phase method", "lpi",
                   choices=(AuthoringChoice("iterative", "Iterative"), AuthoringChoice("lpi", "LPI (phase interpolation)")),
                   description="LPI interpolates site phases, holds each frame's phases fixed during amplitude balancing, and records the iteration count. Both methods enforce the motion and final intensity tolerances."),
    AuthoringField("minimum_separation", "float", "Minimum trap distance (Fourier)", 15., minimum=0., unit="pixel",
                   description="Hard continuous separation of moving and stationary traps, including surplus atoms while they fade. Uses the same units as SLM Target grid spacing (for example, 25 gap); not camera pixels."),
    AuthoringField("motion_intensity_error_percent", "float", "Motion intensity tolerance (%)", 10., minimum=0.,
                   description="Weighted maximum/minimum intensity minus one for the initial removal and other intermediate maps. This is not a bound on absolute trap-depth change or atom loss."),
    AuthoringField("intensity_error_percent", "float", "Final intensity tolerance (%)", 1., minimum=0.,
                   description="Weighted maximum/minimum intensity minus one for the actual last map and offline endpoints. A single-map sequence uses this final tolerance."),
)


def rearrangement_defaults(values, resources):
    del resources
    if values.get("frame_mode", "fixed") == "camera_step":
        return {"nominal_playback_seconds": None}
    try:
        frames = int(values.get("motion_frames", 16))
        rate = float(values.get("frame_rate_hz", 60.))
    except (TypeError, ValueError):
        return {}
    if rate <= 0:
        return {}
    return {"nominal_playback_seconds": frames / rate}


def compact_target(initial_target, rows, columns):
    """Generate a central complete rectangular subset of the calibrated roster.

    This is the shared camera-guided target policy. The optical transition solver accepts
    arbitrary explicit endpoints and knows nothing about camera occupancy.
    """
    initial = np.asarray(initial_target)
    source = np.column_stack(np.nonzero(initial > 0))
    ys, xs = np.unique(source[:, 0]), np.unique(source[:, 1])
    if not 1 <= rows <= len(ys) or not 1 <= columns <= len(xs):
        raise ValueError(f"Target {rows} x {columns} does not fit the source roster")
    center = np.mean(source, axis=0)
    best = None
    for iy in range(len(ys)-rows+1):
        for ix in range(len(xs)-columns+1):
            yy, xx = np.meshgrid(ys[iy:iy+rows], xs[ix:ix+columns], indexing='ij')
            points = np.column_stack((yy.ravel(), xx.ravel()))
            if np.any(initial[points[:, 0], points[:, 1]] <= 0):
                continue
            score = float(np.sum((points.mean(axis=0)-center)**2))
            if best is None or score < best[0]:
                best = (score, points)
    if best is None:
        raise ValueError(f"The source has no complete {rows} x {columns} target rectangle")
    target = np.zeros_like(initial)
    points = best[1]
    target[points[:, 0], points[:, 1]] = initial[points[:, 0], points[:, 1]]
    indices = {tuple(point): i for i, point in enumerate(source)}
    return target, np.asarray([indices[tuple(point)] for point in points], dtype=np.intp)


def run_rearrangement(
    prepared, plan, *, slm, motion_frames, frame_interval, sampled=None,
    support_tolerance=1.01, motion_support_tolerance=1.10,
    stop_requested=None, before_play=None, timings=None, received_at_ns=None,
    player=None, outcome=None, retain_phase_sequence=True,
):
    """Compute and feed one prepared device sequence; the caller owns its lease.

    Task and Measurement share the same verified-prefix/error handling. Sequence
    preparation/release and any Pulse deadline stay with the caller. ``outcome``
    retains the actual numeric result and playback receipt when an error occurs.
    """
    timings = {} if timings is None else timings
    outcome = {} if outcome is None else outcome
    outcome.update(result=None, playback=None, playback_attempted=False,
                   finished_at=None, finished_wall_ns=None)
    started = perf_counter()
    timings["estimated_nominal_playback"] = motion_frames * frame_interval * 1000
    playback = None

    def check_stop():
        if stop_requested is not None and stop_requested():
            raise InterruptedError("SLM rearrangement stopped")

    with ExitStack() as resources:
        if player is None:
            player = resources.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="slm-play"))

        def play():
            began = perf_counter()
            if received_at_ns is not None:
                timings["sequence_call_after_before_frame"] = (time_ns()-received_at_ns)/1e6
            try:
                return slm.play_phase_sequence(stop_requested=stop_requested)
            finally:
                outcome["finished_wall_ns"] = time_ns()
                outcome["finished_at"] = monotonic()
                timings["sequence_play"] = (perf_counter()-began)*1000
                if received_at_ns is not None:
                    timings["camera_frame_to_sequence_return"] = (outcome["finished_wall_ns"]-received_at_ns)/1e6

        def frame_ready(index, codes):
            nonlocal playback
            check_stop()
            if playback is None:
                timings["first_verified_frame_ready"] = (perf_counter()-started)*1000
                if before_play is not None:
                    before_play()
                outcome["playback_attempted"] = True
                playback = player.submit(play)
            if playback.done():
                playback.result()
                raise RuntimeError("SLM playback ended before all frames were submitted")
            try:
                slm.submit_phase_frame(index, codes)
            except BaseException as admission_error:
                # A display/upload error may wake the producer as "cancelled";
                # the device's play result owns the original failure.
                if not playback.done():
                    try:
                        slm.cancel_phase_sequence()
                    except BaseException as cleanup:
                        admission_error.add_note(f"SLM cancellation also failed: {cleanup}")
                try:
                    playback.result()
                except BaseException as device_error:
                    if device_error is not admission_error:
                        device_error.add_note(f"Frame submission also failed: {admission_error}")
                    raise
                raise

        try:
            check_stop()
            if received_at_ns is not None:
                timings["compute_started_after_before_frame"] = (time_ns()-received_at_ns)/1e6
            compute_started = perf_counter()
            result = outcome["result"] = compute_rearrangement(
                prepared, plan, motion_frames=motion_frames, sampled=sampled,
                support_tolerance=support_tolerance, motion_support_tolerance=motion_support_tolerance,
                require_converged=False, frame_ready=frame_ready, stop_requested=stop_requested,
                retain_phase_sequence=retain_phase_sequence)
            timings["compute_and_feed"] = (perf_counter()-compute_started)*1000
            timings.update(("compute_"+name, float(value)) for name, value in result["timing_ms"].items())
            if not result.get("quality_accepted", result["converged"]):
                raise RuntimeError("The phase sequence did not pass its encoded-field quality checks; playback is stopped at its verified prefix. See the partial numeric report.")
            if playback is None:
                receipt = {"frame_count": 0, "played_frames": 0, "cancelled": False,
                           "noop": True, "authored_timing_completed": True,
                           "acknowledgment": "No new phase commanded"}
                outcome["finished_wall_ns"], outcome["finished_at"] = time_ns(), monotonic()
            else:
                receipt = playback.result()
            outcome["playback"] = receipt
            if (receipt["cancelled"] or receipt["played_frames"] != result["motion_frames"]
                    or not receipt["authored_timing_completed"]):
                raise RuntimeError("SLM sequence did not complete all authored frames and time slots")
            check_stop()
            return result, receipt
        except BaseException as error:
            if playback is not None:
                if not playback.done():
                    try:
                        slm.cancel_phase_sequence()
                    except BaseException as cleanup:
                        error.add_note(f"SLM cancellation also failed: {cleanup}")
                try:
                    outcome["playback"] = playback.result()
                except BaseException as cleanup:
                    if cleanup is not error:
                        error.add_note(f"SLM playback also failed: {cleanup}")
                if outcome["playback"] is None:
                    try:
                        outcome["playback"] = slm.last_command_receipt.get("sequence")
                    except BaseException as cleanup:
                        error.add_note(f"SLM partial receipt also failed: {cleanup}")
            raise
        finally:
            timings["online_rearrangement"] = (perf_counter()-started)*1000

__all__ = ["REARRANGEMENT_FIELDS", "rearrangement_defaults", "compact_target", "run_rearrangement"]
