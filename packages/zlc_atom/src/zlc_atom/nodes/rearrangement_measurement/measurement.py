"""Rearrangement actions inside the ordinary camera acquisition/publication."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import perf_counter, time_ns

import numpy as np

from zlc_atom.devices.slm.solver import (
    acquire_rearrangement, plan_rearrangement, sample_rearrangement, rearrangement_is_noop,
)
from zlc_atom.nodes.calibration.calibration import reads_photoelectrons, registration_order
from zlc_atom.nodes.camera.measurement import CameraMeasurementNode, photoelectron_frame
from zlc_atom.devices.slm.rearrangement import compact_target, run_rearrangement


class RearrangementMeasurement(CameraMeasurementNode):
    """Only frames are public; classification and SLM actions stay internal."""

    def __init__(self, *, camera, request, signal_plane, slm, slm_key, calibration,
                 calibration_path, science_context, science_context_path, target_intensity=None,
                 target_path=None, target_rows=3, target_columns=3, phase_method="lpi",
                 frame_mode="fixed", motion_frames=16, max_camera_step=1., frame_rate_hz=60.,
                 minimum_separation=15., intensity_tolerance=1.01, motion_intensity_tolerance=1.10):
        super().__init__(camera=camera, request=request, signal_plane=signal_plane,
                         producer="rearrangement_measurement", record_received=self._on_record)
        if request.frames_per_cycle < 2:
            raise ValueError("Rearrangement needs distinct first and last frames in each cycle")
        self.slm, self.slm_key = slm, slm_key
        self.calibration, self.science_context = calibration, science_context
        self.calibration_path, self.science_context_path = calibration_path, science_context_path
        if science_context["objective_kind"] != "spots" or tuple(science_context["phase"].shape) != tuple(slm.shape_yx):
            raise ValueError("Rearrangement requires a spots Science Context matching the selected SLM")
        source, weights, self._readout_order, affine = registration_order(
            calibration, science_context, science_context_path)
        target = (compact_target(science_context["target_intensity"], target_rows, target_columns)[0]
                  if target_intensity is None else np.asarray(target_intensity))
        if target.shape != tuple(slm.shape_yx):
            raise ValueError("End Target shape differs from the selected SLM")
        destination = np.column_stack(np.nonzero(target > 0)).astype(np.int32)
        self._source, self._destination = source, destination
        self._weights = (weights, target[tuple(destination.T)])
        # Translation is irrelevant to displacements; Calibration owns binning.
        self._sensor_linear = affine[:2] * np.asarray(calibration.frame_contract.binning_yx)[::-1]
        self.phase_method, self.frame_mode = phase_method, frame_mode
        self.motion_frames, self.max_camera_step = int(motion_frames), float(max_camera_step)
        self.frame_interval = 1. / float(frame_rate_hz)
        self.minimum_separation = float(minimum_separation)
        self.intensity_tolerance, self.motion_intensity_tolerance = intensity_tolerance, motion_intensity_tolerance
        self._configuration = dict(calibration_path=str(calibration_path), science_context_path=str(science_context_path),
            end_target_path=target_path, target_rows=target_rows, target_columns=target_columns,
            phase_method=phase_method, frame_mode=frame_mode, motion_frames=motion_frames,
            max_camera_step=max_camera_step, frame_rate_hz=frame_rate_hz, minimum_separation=minimum_separation,
            intensity_tolerance=intensity_tolerance, motion_intensity_tolerance=motion_intensity_tolerance,
            trigger_frame=0, restore_frame=request.frames_per_cycle - 1)
        self._movement = None
        self._sequence_prepared = False
        self._abort = Event()
        self._cycle_evidence = {}
        self._restored_at_ns = 0
        self._at_source = False

    def _freeze_working_point(self, point):
        super()._freeze_working_point(point)
        roi = (*point.roi_origin_yx[::-1], *point.roi_shape_yx[::-1])
        self._calibration = self.calibration.rebased(roi, point.binning_yx, point.frame_shape_yx)
        self._run_record["named_devices"]["slm"] = self.slm_key
        self._run_record["rearrangement"] = self._configuration

    def _camera_event_record(self, records):
        event = super()._camera_event_record(records)
        if self._cycle_evidence.get("cycle") == records[0].source_ordinal // self.frames_per_cycle:
            event["rearrangement"] = self._cycle_evidence
        return event

    def _stopped(self):
        return self._abort.is_set() or self._context.cancel_requested()

    def _prepare_sequence(self, count):
        self._sequence_prepared = True
        self.slm.prepare_phase_sequence(None, self.frame_interval, frame_count=count)

    def _restore_source(self):
        if self._sequence_prepared:
            self.slm.release_phase_sequence()
            self._sequence_prepared = False
        if not self._at_source:
            self.slm.apply_phase(self.science_context["phase"])
            if self.slm.last_command_receipt.get("outcome") != "known-new":
                raise RuntimeError("SLM source-phase restoration was not confirmed")
            self._restored_at_ns = time_ns()
            self._at_source = True

    def _rearrange(self, record):
        timings, outcome = self._timings, self._outcome
        began = perf_counter()
        image = record.image
        if reads_photoelectrons(self._calibration):
            image = photoelectron_frame(image, self.actual_working_point)
        detected = self._calibration.detect(image)
        usable = (self._calibration.site_map.valid_sites & self._calibration.select_model().usable_sites
                  & np.isfinite(detected.counts) & np.isfinite(detected.thresholds))
        selected = np.flatnonzero((detected.occupied & usable)[self._readout_order])
        timings["classification"] = (perf_counter() - began) * 1000
        began = perf_counter()
        plan = plan_rearrangement(self._prepared, selected)
        timings["matching"] = (perf_counter() - began) * 1000
        sampled, count = None, self.motion_frames
        if self.frame_mode == "camera_step":
            if rearrangement_is_noop(self._prepared, plan):
                count = 0
            else:
                sensor_path = np.asarray(plan["motion_yx"])[..., ::-1] @ self._sensor_linear
                sampled = sample_rearrangement(self._prepared, plan, maximum_step=self.max_camera_step,
                                                step_path=sensor_path)
                count = sampled["motion_frames"]
                self._prepare_sequence(count)
        def before_play():
            self._at_source = False
        result, playback = run_rearrangement(self._prepared, plan, slm=self.slm,
            motion_frames=count, frame_interval=self.frame_interval, sampled=sampled,
            support_tolerance=self.intensity_tolerance, motion_support_tolerance=self.motion_intensity_tolerance,
            stop_requested=self._stopped, timings=timings, received_at_ns=record.host_received_at_ns,
            player=self._workers, outcome=outcome, before_play=before_play)
        self._cycle_evidence = {"cycle": record.source_ordinal // self.frames_per_cycle,
            "source_indices": plan["source_indices"].tolist(), "target_indices": plan["target_indices"].tolist(),
            "frames": len(result["phase_codes"]), "played_frames": playback["played_frames"],
            "timing_ms": dict(timings), "initial_frame_ordinal": record.source_ordinal}
        # A continuous measurement retains neither movies nor old-cycle arrays.
        outcome.pop("result", None)

    def _on_record(self, record):
        if self._stopped():
            return  # Stop still drains camera data, never initiates new actions.
        position = record.source_ordinal % self.frames_per_cycle
        if position == 0:
            if self._movement is not None:
                self._movement.result()
            if record.host_received_at_ns < self._restored_at_ns:
                raise RuntimeError("The next camera cycle arrived before the initial SLM state was restored")
            self._timings, self._outcome = {}, {}
            self._movement = self._workers.submit(self._rearrange, record)
        elif self._movement is not None and self._movement.done():
            self._movement.result()
        if position == self.frames_per_cycle - 1:
            ended = self._outcome.get("finished_wall_ns")
            early = ended is None or ended > record.host_received_at_ns
            if early:
                self._abort.set()
                error = RuntimeError("The camera cycle ended before rearrangement completed")
                for cleanup in (self.slm.cancel_phase_sequence, self._movement.result, self._restore_source):
                    try:
                        cleanup()
                    except BaseException as detail:
                        error.add_note(f"Rearrangement cleanup: {detail}")
                raise error
            self._movement.result()
            self._restore_source()
            self._cycle_evidence["restored_at_ns"] = self._restored_at_ns
            if not self._stopped() and self.frame_mode == "fixed" and (self.repeat == 0 or record.source_ordinal + 1 < self.repeat * self.frames_per_cycle):
                self._prepare_sequence(self.motion_frames)

    def read_records(self, count, *, timeout, exact):
        if self._movement is not None and self._movement.done() and not self._stopped():
            self._movement.result()
        return super().read_records(count, timeout=timeout, exact=exact)

    def execute(self, context):
        self._context = context
        with acquire_rearrangement(source_yx=self._source, target_yx=self._destination, shape_yx=self.slm.shape_yx,
                pupil_amplitude=self.science_context["pupil_amplitude"],
                pupil_phase=-np.asarray(self.science_context["operator_wavefront"]),
                phase_center_yx=tuple(self.science_context["pupil"]["center_xy"])[::-1],
                source_intensities=self._weights[0], target_intensities=self._weights[1],
                minimum_separation=self.minimum_separation, method=self.phase_method,
                support_tolerance=self.intensity_tolerance,
                maximum_motion_frames=self.motion_frames if self.frame_mode == "fixed" else 16,
                endpoint_data={"source_phase": self.science_context["phase"]},
                stop_requested=context.cancel_requested) as (prepared, reused):
            self._prepared = prepared
            self._configuration["gpu_preparation_reused"] = reused
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="slm-acquire") as workers:
                self._workers = workers
                # Start both fixed workers before camera readiness, not on the
                # first photo or first displayed map of a time-critical cycle.
                released = Event()
                warm_workers = [workers.submit(released.wait) for _ in range(2)]
                released.set()
                for warm in warm_workers:
                    warm.result()
                error = None
                try:
                    self._restore_source()
                    if self.frame_mode == "fixed":
                        self._prepare_sequence(self.motion_frames)
                    return super().execute(context)
                except BaseException as caught:
                    error = caught
                    raise
                finally:
                    self._abort.set()
                    try:
                        if self._movement is not None:
                            if not self._movement.done():
                                self.slm.cancel_phase_sequence()
                            self._movement.result()
                    except BaseException as cleanup:
                        if error is not None:
                            if cleanup is not error: error.add_note(f"Rearrangement cleanup: {cleanup}")
                        elif not context.cancel_requested():
                            raise
                    finally:
                        try:
                            self._restore_source()
                        except BaseException as cleanup:
                            if error is None: raise
                            error.add_note(f"SLM restoration: {cleanup}")
                        self._prepared = None
