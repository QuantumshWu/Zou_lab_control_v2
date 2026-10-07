"""One operator-authored Pulse: photograph, rearrange, photograph and verify."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter, monotonic, time_ns
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from zlc_data import (AxisId, AxisSpec, COMPONENT, SCAN_POINT, SPATIAL_Y, SPATIAL_X,
                      DatasetSchema, DomainSpec, ValueSchema, ValidityContract, owned_snapshot_from_arrays)
from zlc_durable import atomic_write_file, atomic_write_text, write_readable_json
from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput, MonitorCoverage
from zlc_plot import (AxisRef, CurvePlot, ImagePlot, PlotLabels, ImageFrame, ImagePointOverlay, PointStatus,
                      IMAGE_POINT_OVERLAY_CONTRACT, IMAGE_POINT_OVERLAY_GEOMETRY_RECORD,
                      image_point_overlay_geometry)
from zlc_pulse import PulseSequence
from zlc_pulse.schedule import trigger_edge_ticks
from zlc_pulse.wire import STATUS_DONE, STATUS_RUNNING, STATUS_ERROR, STATUS_UNDERFLOW

from zlc_atom.data import snapshot_from_array
from zlc_atom.devices.slm.device import phase_from_codes
from zlc_atom.devices.camera.contract import CameraFrameRecord
from zlc_atom.devices.slm.solver import prepare_rearrangement, plan_rearrangement, compute_rearrangement
from zlc_atom.devices.sequencer import sequencer_archive_snapshot
from zlc_atom.nodes.calibration import TrapCalibration
from zlc_atom.nodes.calibration.calibration import reads_photoelectrons
from zlc_atom.nodes.calibration.pulse import resolve_pulse, arm_sequencer
from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode, CameraMeasurementRequest, frames_snapshot)
from zlc_atom.nodes.scan.source import check_cancelled, wait_for_report
from zlc_atom.nodes.slm_feedback.task import _register_target_sites, _plain_json

REARRANGEMENT_ARTIFACT_CONTRACT = "zlc.slm.rearrangement"
BEFORE_FRAME_OUTPUT = DatasetOutputDeclaration("before_frame", "slm-rearrangement.before-frame")
AFTER_FRAME_OUTPUT = DatasetOutputDeclaration("after_frame", "slm-rearrangement.after-frame")
BEFORE_OCCUPIED_OUTPUT = DatasetOutputDeclaration("before_occupied", IMAGE_POINT_OVERLAY_CONTRACT)
AFTER_OCCUPIED_OUTPUT = DatasetOutputDeclaration("after_occupied", IMAGE_POINT_OVERLAY_CONTRACT)
PHASE_OUTPUT = DatasetOutputDeclaration("phase", "slm-rearrangement.phase")
TRAJECTORY_OUTPUT = DatasetOutputDeclaration("trajectory", "slm-rearrangement.trajectory")
QUALITY_OUTPUT = DatasetOutputDeclaration("intensity_ratio", "slm-rearrangement.intensity-ratio")
OUTPUTS = (BEFORE_FRAME_OUTPUT, BEFORE_OCCUPIED_OUTPUT, AFTER_FRAME_OUTPUT,
           AFTER_OCCUPIED_OUTPUT, PHASE_OUTPUT, TRAJECTORY_OUTPUT, QUALITY_OUTPUT)


def pulse_timing(sequence, program, rows, before_period, after_period):
    """Selected Periods in the actual compiled, Config-filled one-shot program."""
    ids = tuple(p.period_id for p in sequence.periods)
    if before_period not in ids or after_period not in ids:
        raise ValueError("The imaging Period is absent from the applied Pulse")
    point = tuple(rows[0]) if rows else ()
    if len(rows) > 1:
        raise ValueError("One rearrangement requires one Pulse point, not a scan table")
    # The public walker accounts for nested loops; bounded replay catches an
    # imaging row repeated billions of times without expanding all repeats.
    visits = program.frame_visits(point, bracket_bodies=lambda span: 2)
    durations = program.resolved_durations(point)
    result = {}
    for key, identity in (("before", before_period), ("after", after_period)):
        row = ids.index(identity)
        starts = [tick for index, tick in visits if index == row]
        if len(starts) != 1:
            raise ValueError(f"{key.title()} imaging Period must play exactly once")
        result[key + "_start_seconds"] = starts[0] / program.clock_hz
        result[key + "_end_seconds"] = (starts[0] + durations[row]) / program.clock_hz
        result[key + "_period"] = {"id": identity, "name": sequence.periods[row].name or identity}
    if result["before_end_seconds"] > result["after_start_seconds"]:
        raise ValueError("Before imaging must finish before after imaging begins")
    # The operator identifies the imaging Period, not an SDK camera pin. A
    # negative lane delay can move an edge earlier; use the earliest possible
    # digital edge shift rather than silently assuming zero delay.
    result["earliest_lane_delay_seconds"] = min((0, *program.channel_delays)) / program.clock_hz
    result["available_gap_seconds"] = result["after_start_seconds"] - result["before_end_seconds"]
    result["program_digest"] = program.digest
    result["pulse_duration_seconds"] = program.frame_ticks(point) / program.clock_hz
    channels = tuple(channel for bit, channel in enumerate(program.channels) if not program.clk_enable & (1 << bit))
    table = np.asarray(rows, dtype=np.int64) if rows else None
    streams = trigger_edge_ticks(program, channels, table, run_repeats=1, scan_repeats=1,
                                 bracket_bodies=lambda span: 2)
    candidates = []
    for channel in channels:
        edges = streams[channel]
        rising = np.asarray(edges[::2], dtype=float) / program.clock_hz
        if (len(rising) == 2 and result["before_start_seconds"] <= rising[0] < result["before_end_seconds"]
                and result["after_start_seconds"] <= rising[1] < result["after_end_seconds"]):
            candidates.append({"channel": channel, "rising_seconds": rising.tolist()})
    if not candidates:
        raise ValueError("No digital lane has exactly two rising edges in the selected imaging Periods; inspect the camera trigger Pulse")
    result["imaging_trigger_candidates"] = candidates
    result["timing_basis"] = "Operator-selected Periods; earliest possible lane delay; not hardware camera-trigger timestamps"
    return result


def _registration_order(calibration, context, context_path):
    target = np.asarray(context["target_intensity"])
    points = np.column_stack(np.nonzero(target > 0)).astype(np.int32)
    if len(points) != calibration.n_sites:
        raise ValueError("Rearrangement needs one calibrated readout per authored source/target site")
    model = calibration.select_model()
    registered = _register_target_sites(
        calibration.site_map, target,
        {"science_context_path": str(context_path), "command_receipt": dict(context["command_receipt"])},
        frame_shape=calibration.frame_contract.image_shape,
        measurement_radius=model.integration_half_width)
    # Registration retains the measured coordinates verbatim when every site
    # is observed. Convert that existing result to a readout permutation only.
    original = {tuple(center): i for i, center in enumerate(calibration.site_map.centers_xy)}
    order = np.asarray([original[tuple(center)] for center in registered.centers_xy], dtype=np.intp)
    if len(set(order)) != len(points):
        raise ValueError("The calibrated readout does not map one-to-one to SLM sites")
    return points, target[points[:, 0], points[:, 1]], order


def compact_target(initial_target, rows, columns):
    """Generate a central complete rectangular subset of the calibrated roster.

    This is the atom Task's target policy. The optical transition solver accepts
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


class SlmRearrangementTask:
    """The concrete experiment owner; devices retain acquisition and playback."""
    instance_id = "slm_rearrangement"

    def __init__(self, *, camera, camera_key, sequencer, sequencer_key, slm, slm_key,
                 signal_plane, calibration, calibration_path, science_context,
                 science_context_path, target_rows=3, target_columns=3,
                 target_intensity=None, target_path=None,
                 pulse_sequence, pulse_path, before_period, after_period,
                 exposure_seconds=.005, motion_frames=16, frame_rate_hz=60.,
                 minimum_separation=4.5, intensity_tolerance=1.01,
                 save_phase_sequence=True, save_figure_artifact=None):
        self.camera, self.sequencer, self.slm = camera, sequencer, slm
        self.camera_key, self.sequencer_key, self.slm_key = camera_key, sequencer_key, slm_key
        self.signal_plane = signal_plane
        self.calibration, self.science_context = calibration, science_context
        self.context_path, self.calibration_path = Path(science_context_path).resolve(), Path(calibration_path).resolve()
        self.target_rows, self.target_columns = int(target_rows), int(target_columns)
        self.target_path = None if target_path is None else str(Path(target_path).resolve())
        self.sequence, self.pulse_path = pulse_sequence, Path(pulse_path).resolve()
        self.before_period, self.after_period = str(before_period), str(after_period)
        self.exposure_seconds = float(exposure_seconds)
        self.motion_frames = int(motion_frames)
        self.frame_rate_hz = float(frame_rate_hz)
        self.minimum_separation = float(minimum_separation)
        self.intensity_tolerance = float(intensity_tolerance)
        self.save_phase_sequence = bool(save_phase_sequence)
        self._save_figure_artifact = save_figure_artifact
        if not isinstance(pulse_sequence, PulseSequence):
            raise TypeError("Rearrangement requires an authored Pulse file")
        if not isinstance(calibration, TrapCalibration) or science_context.get("objective_kind") != "spots":
            raise ValueError("Rearrangement requires a calibrated spots Science Context")
        if tuple(np.asarray(science_context["phase"]).shape) != tuple(slm.shape_yx):
            raise ValueError("Science Context shape differs from the selected SLM")
        if not all(callable(getattr(slm, method, None)) for method in (
                "prepare_phase_sequence", "submit_phase_frame", "play_phase_sequence",
                "cancel_phase_sequence", "release_phase_sequence")):
            raise ValueError("The selected SLM does not offer phase-sequence playback")
        source, weights, order = _registration_order(calibration, science_context, self.context_path)
        if target_intensity is None:
            self.target_intensity, self.target_indices = compact_target(science_context['target_intensity'], self.target_rows, self.target_columns)
        else:
            self.target_intensity = np.asarray(target_intensity)
            if self.target_intensity.shape != tuple(slm.shape_yx):
                raise ValueError("End Target shape differs from the selected SLM")
            source_indices = {tuple(point): i for i, point in enumerate(source)}
            target_points = np.column_stack(np.nonzero(self.target_intensity > 0))
            self.target_indices = np.asarray([source_indices.get(tuple(point), -1) for point in target_points], dtype=np.intp)
        destination = np.column_stack(np.nonzero(self.target_intensity > 0)).astype(np.int32)
        self.points = (source, destination)
        self.weights = (weights, self.target_intensity[tuple(destination.T)])
        self.readout_order = order
        contract = calibration.frame_contract
        self.roi = contract.roi_xywh
        self._revision = 0
        self._snapshots, self._overlays, self._records, self._detections = {}, {}, [], []
        self._timings, self._playback, self._result, self._plan = {}, None, None, None
        self._device_snapshots = {}
        self._available_outputs, self._capture_evidence = set(), {}
        self._overlay_geometry = None

    @property
    def dataset_output_declarations(self):
        return OUTPUTS

    def _record(self):
        return {"named_devices": {"camera": self.camera_key, "sequencer": self.sequencer_key, "slm": self.slm_key},
                "pulse": {"path": str(self.pulse_path), "name": self.pulse_path.name},
                "science_context_path": str(self.context_path), "calibration_path": str(self.calibration_path),
                "end_target_path": self.target_path,
                "target_rows": self.target_rows, "target_columns": self.target_columns,
                "target_source_indices": self.target_indices.tolist(),
                "before_period": self.before_period, "after_period": self.after_period,
                "exposure_seconds": self.exposure_seconds, "motion_frames": self.motion_frames,
                "frame_rate_hz": self.frame_rate_hz,
                "minimum_separation": self.minimum_separation,
                "intensity_tolerance": self.intensity_tolerance}

    def _snapshot(self, context, declaration, values, **axes):
        return snapshot_from_array(values, producer=self.instance_id, signal=declaration.name,
            generation=str(getattr(context.generation, "value", context.generation)),
            revision=self._revision, **axes)

    def _publish(self, context, snapshots):
        event = {"capture_events": dict(self._capture_evidence), "device_snapshots": {**self._device_snapshots, "slm": {
            "identity": str(self.slm.identity), "shape_yx": list(self.slm.shape_yx),
            "command_revision": int(self.slm.command_revision), "mapping_revision": int(self.slm.mapping_revision),
            "command_receipt": dict(self.slm.last_command_receipt)}}}
        updates = {decl.name: value for decl,value in snapshots}
        self._available_outputs.update(updates)
        self._snapshots.update(updates)
        if self._overlay_geometry is not None:
            event[IMAGE_POINT_OVERLAY_GEOMETRY_RECORD] = self._overlay_geometry
        context.commit_live({decl.name: LiveDatasetOutput(decl, self._snapshots[decl.name],
                             MonitorCoverage(int(np.prod(self._snapshots[decl.name].block.values.shape[:2])),
                                             int(np.prod(self._snapshots[decl.name].block.values.shape[:2]))),
                             event_record=event)
                             for decl in OUTPUTS})

    def _prepare_outputs(self, context, node):
        """One frozen output vocabulary; future results have real invalidity."""
        point = node.actual_working_point
        shape = point.frame_shape_yx
        record = CameraFrameRecord(np.zeros(shape, np.float32 if node.reads_photoelectrons else point.dtype), 0)
        image = frames_snapshot(((record,),), producer=self.instance_id,
            generation=str(getattr(context.generation, "value", context.generation)),
            revision=self._revision, working_point=point, value_unit=node.frame_value_unit)
        image = owned_snapshot_from_arrays(image.block.schema, image.block.values, image.block.revision,
            validity=np.zeros((1,1), bool), stream_generation=image.ref.stream_generation)
        n = len(self.points[0])
        self._status_axis = AxisSpec(AxisId('slm_rearrangement.site'),'site',COMPONENT,n,tuple(float(i+1) for i in range(n)))
        for index,(frame,occ) in enumerate(((BEFORE_FRAME_OUTPUT,BEFORE_OCCUPIED_OUTPUT),(AFTER_FRAME_OUTPUT,AFTER_OCCUPIED_OUTPUT))):
            site = self._status_axis
            schema = DatasetSchema(image.block.schema.repeat_domain, image.block.schema.point_domain,
                DomainSpec((n,), (site,)), ValueSchema(ValidityContract.components(site.axis_id), np.dtype('?'), '1', name='occupied'))
            self._snapshots[frame.name] = image
            self._snapshots[occ.name] = owned_snapshot_from_arrays(schema, np.zeros((1,1,n),bool), image.block.revision,
                validity=np.zeros((1,1,n),bool), stream_generation=image.ref.stream_generation)
        roi = (*tuple(point.roi_origin_yx)[::-1], *tuple(point.roi_shape_yx)[::-1])
        cal = self.calibration.rebased(roi, point.binning_yx, shape)
        centers=cal.site_map.centers_xy[self.readout_order]
        ids=tuple(f'site_{i:04d}' for i in range(n))
        labels=tuple(str(i+1) for i in range(n))
        self._overlay_geometry = image_point_overlay_geometry(image,centers,ids,
            status_axis=self._status_axis,labels=labels,coordinates_are_indices=True)
        n = self.motion_frames+1
        frame = AxisSpec(AxisId('slm_rearrangement.frame'),'frame',SCAN_POINT,n,tuple(float(i) for i in range(n)))
        site = AxisSpec(AxisId('slm_rearrangement.source_site'),'site',COMPONENT,len(self.points[0]),
                        tuple(float(i+1) for i in range(len(self.points[0]))))
        self._snapshots[TRAJECTORY_OUTPUT.name] = self._snapshot(context,TRAJECTORY_OUTPUT,
            np.zeros((1,n,len(self.points[0]),2)),point_axes=(frame,),
            cell_axes=(site,AxisSpec(AxisId('slm_rearrangement.coordinate'),'coordinate',COMPONENT,2,(0.,1.),coordinate_labels=('x','y'))),
            validity=np.zeros((1,n,len(self.points[0]),2),bool),value_unit='1')
        n = self.motion_frames
        frame = AxisSpec(AxisId('slm_rearrangement.output_frame'),'frame',SCAN_POINT,n,tuple(float(i) for i in range(n)))
        self._snapshots[QUALITY_OUTPUT.name] = self._snapshot(context,QUALITY_OUTPUT,np.zeros((1,n)),
            point_axes=(frame,),validity=np.zeros((1,n),bool))
        self._snapshots[PHASE_OUTPUT.name] = self._snapshot(context,PHASE_OUTPUT,np.zeros((1,*self.slm.shape_yx),np.float32),
            cell_axes=(SPATIAL_Y,SPATIAL_X),value_unit='rad')
        # Overlay geometry is a fixed run fact, available before any invalid
        # future image is mounted. The first event must not create a host with
        # a missing status roster that only a later photograph can repair.
        context.set_run_record({**self._record(), IMAGE_POINT_OVERLAY_GEOMETRY_RECORD:self._overlay_geometry,
                                "device_snapshots":dict(self._device_snapshots)})

    def _publish_phase(self, context, phase):
        self._revision += 1
        snap = self._snapshot(context, PHASE_OUTPUT, np.asarray(phase)[None],
                              cell_axes=(SPATIAL_Y, SPATIAL_X), value_unit="rad")
        self._capture_evidence[PHASE_OUTPUT.name] = {"device_snapshots": {
            **self._device_snapshots, "slm": {
                "identity": str(self.slm.identity), "shape_yx": list(self.slm.shape_yx),
                "command_revision": self.slm.command_revision,
                "mapping_revision": self.slm.mapping_revision,
                "command_receipt": dict(self.slm.last_command_receipt)}}}
        self._publish(context, ((PHASE_OUTPUT, snap),))

    def _read_photo(self, context, node, cycle, index):
        self._revision += 1
        snap = frames_snapshot((cycle,), producer=self.instance_id,
            generation=str(getattr(context.generation, "value", context.generation)),
            revision=self._revision, working_point=node.actual_working_point, value_unit=node.frame_value_unit)
        point = node.actual_working_point
        roi = (*tuple(point.roi_origin_yx)[::-1], *tuple(point.roi_shape_yx)[::-1])
        cal = self.calibration.rebased(roi, point.binning_yx, point.frame_shape_yx)
        wanted = reads_photoelectrons(cal)
        if wanted is not None and wanted != node.reads_photoelectrons:
            raise ValueError("Camera counts/photoelectron unit differs from the selected Calibration")
        model = cal.select_model()
        detection = cal.detect(snap.block.values[0, 0])
        order = self.readout_order
        valid = (cal.site_map.valid_sites & model.usable_sites & np.isfinite(detection.counts)
                 & np.isfinite(detection.thresholds))[order]
        occupied = detection.occupied[order] & valid
        site = self._status_axis
        frame_decl, occ_decl = ((BEFORE_FRAME_OUTPUT, BEFORE_OCCUPIED_OUTPUT),
                               (AFTER_FRAME_OUTPUT, AFTER_OCCUPIED_OUTPUT))[index]
        status_schema = DatasetSchema(snap.block.schema.repeat_domain, snap.block.schema.point_domain,
            DomainSpec((site.size,), (site,)), ValueSchema(ValidityContract.components(site.axis_id), np.dtype('?'), '1', name='occupied'))
        status = owned_snapshot_from_arrays(status_schema, occupied[None,None], snap.block.revision,
            validity=valid[None,None], stream_generation=snap.ref.stream_generation)
        geometry = self._overlay_geometry
        self._capture_evidence[frame_decl.name] = {"device_snapshots": {**self._device_snapshots, "slm": {
            "identity":str(self.slm.identity),"shape_yx":list(self.slm.shape_yx),
            "command_revision":self.slm.command_revision,"mapping_revision":self.slm.mapping_revision,
            "command_receipt":dict(self.slm.last_command_receipt)}},"source_ordinal":cycle[0].source_ordinal}
        self._capture_evidence[frame_decl.name]["camera_source"] = {
            "signal":node.signal_key("frames"),"generation":str(getattr(node.generation,'value',node.generation)),
            "source_ordinal":cycle[0].source_ordinal}
        self._publish(context, ((frame_decl, snap), (occ_decl, status)))
        # Reports use the same geometry and exact validity as the live overlay.
        from zlc_plot.primitives import image_point_overlay_from_signal
        self._overlays[frame_decl.name] = image_point_overlay_from_signal(geometry, status, snap, revision=self._revision)
        self._detections.append({"counts": detection.counts[order], "occupied": occupied,
                                  "valid": valid, "thresholds": detection.thresholds[order]})
        self._records.append(cycle[0])
        return occupied, valid

    def _save(self, context, status, error=None):
        started = perf_counter()
        directory = context.run_directory
        data, figures = directory / "data", directory / "figures"
        data.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
        arrays = {"source_yx": self.points[0], "target_yx": self.points[1]}
        confirmed_phase = self.slm.last_commanded_phase
        if confirmed_phase is not None:
            arrays["last_confirmed_phase"] = confirmed_phase
        for index, value in enumerate(self._detections):
            prefix = ("before", "after")[index]
            arrays.update({prefix + "_" + name: item for name, item in value.items()})
            arrays[prefix + "_image"] = self._records[index].image
        if self._result is not None:
            for key in ("motion_yx", "fraction", "support_intensity_ratios", "sites_yx",
                        "brightness_minimum_to_initial", "brightness_maximum_to_initial",
                        "brightness_mean_to_initial", "phase_error_rms_rad", "phase_step_max_rad",
                        "pupil_phase_step_rms_rad", "discard_intensity_ratios", "field_projection_updates",
                        "frame_solve_ms", "frame_copy_ms", "frame_ready_ms",
                        "center_sample_power_proxy", "background_intensity_ratios", "desired_amplitudes",
                        "actual_fields", "active_sites"):
                if key in self._result: arrays[key] = self._result[key]
            if self.save_phase_sequence: arrays["phase_codes"] = self._result["phase_codes"]
        if self._plan is not None:
            for key in ("assigned_source_indices", "assigned_target_indices", "removed_source_indices",
                        "source_indices", "target_indices", "target_filled"):
                if key in self._plan:
                    arrays["planned_target_filled" if key == "target_filled" else key] = np.asarray(self._plan[key])
            if "motion_yx" in self._plan:
                arrays["planned_motion_yx"] = self._plan["motion_yx"]
        summary = {**self._record(), "status": status, "error": None if error is None else str(error),
                   "timing_ms": self._timings, "gpu": getattr(self, "_gpu_info", None),
                   "pulse_timing": getattr(self, "_pulse_timing", None), "playback": self._playback,
                   "device_snapshots": {**self._device_snapshots, "slm": {
                       "identity": str(self.slm.identity), "shape_yx": list(self.slm.shape_yx),
                       "command_revision": int(self.slm.command_revision), "mapping_revision": int(self.slm.mapping_revision),
                       "command_receipt": dict(self.slm.last_command_receipt)}},
                   "capture_events": self._capture_evidence,
                   "frame_records": [{"source_ordinal": r.source_ordinal, "host_received_at_ns": r.host_received_at_ns,
                     "timestamp_seconds": r.timestamp_seconds, "timestamp_microseconds": r.timestamp_microseconds}
                     for r in self._records]}
        # Inputs are frozen values already read before Start. References alone
        # would become ambiguous when an operator overwrites a Calibration or
        # Context file for the next run. Numeric pupil arrays are reconstructed
        # from these same semantic Context facts rather than saved twice.
        summary["calibration"] = self.calibration.to_dict()
        summary["science_context"] = {key:self.science_context[key] for key in (
            "objective_kind", "pupil", "operator_metadata", "system_correction", "pattern_metadata", "command_receipt")}
        arrays['source_pattern_phase']=self.science_context['pattern_phase']
        arrays['source_target_intensity']=self.science_context['target_intensity']
        arrays['generated_target_intensity']=self.target_intensity
        if self._detections:
            summary["before_occupied"] = int(self._detections[0]["occupied"].sum())
        if len(self._detections) == 2:
            after = self._target_detection()
            arrays.update({"after_target_" + key: value for key, value in after.items()})
            summary.update(target_sites=len(after["occupied"]), judged_target_sites=int(after["valid"].sum()),
                           filled_target_sites=int(after["occupied"].sum()),
                           verification_complete=bool(after["valid"].all()),
                           target_filling_fraction=(float(after["occupied"].mean()) if after["valid"].all() else None),
                           judged_target_filling_fraction=(float(after["occupied"][after["valid"]].mean()) if after["valid"].any() else None),
                           missing_target_indices=np.flatnonzero(after["valid"] & ~after["occupied"]).tolist(),
                           invalid_target_indices=np.flatnonzero(~after["valid"]).tolist())
        if self._plan is not None:
            summary["assigned_atoms"] = len(self._plan["assigned_source_indices"])
            summary["removed_atoms"] = len(self._plan["removed_source_indices"])
            summary["planning"] = {key:self._plan[key] for key in
                ("total_distance", "euclidean_lower_bound", "detour_ratio", "routing") if key in self._plan}
            summary["unfilled_target_indices"] = np.setdiff1d(
                np.arange(len(self.points[1])), self._plan["assigned_target_indices"]).tolist()
        if self._result is not None:
            summary["motion_diagnostics"] = {key: self._result[key] for key in
                ("maximum_step", "recommended_motion_frames", "clearance", "fade_clearance",
                 "surplus_stationary_clearance", "release_verified", "recommended_release_hold_frames",
                 "discard_reference_limit", "discard_converged", "converged",
                 "noop", "fade_frames", "emitted_frame_count") if key in self._result}
        encoded = json.dumps(_plain_json(summary), allow_nan=False, separators=(",", ":"))
        arrays["metadata"] = np.asarray(encoded)
        archive = atomic_write_file(data / "rearrangement.npz", lambda f: np.savez(f, **arrays))
        context.register_artifact("artifact_path" if status == "completed" else "partial_data", archive,
            role="final" if status == "completed" else "checkpoint", contract_id=REARRANGEMENT_ARTIFACT_CONTRACT)
        # File rendering starts only after the second image or an actual failure.
        writer = self._save_figure_artifact
        if writer is None:
            from zlc_plot import save_figure_artifact as writer
        for name, snap in self._snapshots.items():
            if name not in self._available_outputs: continue
            if name in {BEFORE_OCCUPIED_OUTPUT.name, AFTER_OCCUPIED_OUTPUT.name}: continue
            if name == TRAJECTORY_OUTPUT.name:
                extent = self._result["motion_yx"].reshape(-1,2)
                if not len(extent): extent = self.points[1]
                pad = max(1., .08*float(np.ptp(extent,axis=0).max()))
                y0,x0 = np.maximum(0,np.floor(extent.min(axis=0)-pad).astype(int))
                y1,x1 = np.minimum(self.slm.shape_yx,np.ceil(extent.max(axis=0)+pad+1).astype(int))
                source_image = snapshot_from_array(
                    self.science_context["target_intensity"][None,y0:y1,x0:x1],
                    producer=self.instance_id, signal="source_target",
                    generation=str(getattr(context.generation,"value",context.generation)), revision=self._revision,
                    cell_axes=(
                        AxisSpec(AxisId("slm_rearrangement.path_y"),"Fourier y",SPATIAL_Y,int(y1-y0),tuple(float(i) for i in range(y0,y1))),
                        AxisSpec(AxisId("slm_rearrangement.path_x"),"Fourier x",SPATIAL_X,int(x1-x0),tuple(float(i) for i in range(x0,x1)))),
                    value_unit="1")
                path = self._result["motion_yx"][:, :, ::-1].transpose(1,0,2)
                ids = self._plan["source_indices"]
                overlay = None if not len(ids) else ImagePointOverlay(
                    revision=self._revision, coordinates=self.points[0][ids, ::-1],
                    point_ids=tuple(str(i) for i in ids), labels=tuple(f"#{i+1}" for i in ids),
                    static_statuses=tuple(PointStatus.OCCUPIED for _ in ids), paths_xy=path)
                plot_input = source_image if overlay is None else ImageFrame(source_image,overlay)
                axes = source_image.block.schema.cell_domain.axes
                spec = ImagePlot(x=AxisRef.cell_data(str(axes[1].axis_id)),
                                 y=AxisRef.cell_data(str(axes[0].axis_id)),
                                 labels=PlotLabels(title="Commanded trap paths", x="Fourier x (pixel)", y="Fourier y (pixel)"))
                written = writer(figures / "trajectory_2d.png", plot_input=plot_input, spec=spec,
                    parameters={"show_image":False, "side_distribution":False, "show_colorbar":False},
                    size="4x4", source={"task":self.instance_id, "report":"trajectory_2d", "run_record":self._record()})
                if hasattr(written,"result"): written=written.result()
                png,npz=written
                context.register_artifact("trajectory_2d_figure",npz,role="figure",contract_id="zlc.figure")
                context.register_artifact("trajectory_2d_preview",png,role="preview")
                continue
            image = name in {BEFORE_FRAME_OUTPUT.name, AFTER_FRAME_OUTPUT.name, PHASE_OUTPUT.name}
            plot_input = ImageFrame(snap, self._overlays[name]) if name in self._overlays else snap
            spec = (ImagePlot(x=AxisRef.cell_data(str(snap.block.schema.cell_domain.axes[1].axis_id)),
                              y=AxisRef.cell_data(str(snap.block.schema.cell_domain.axes[0].axis_id)),
                              labels=PlotLabels(title=name.replace('_', ' '))) if image else
                    CurvePlot(x=AxisRef.point(str(snap.block.schema.point_domain.axes[0].axis_id)),
                              labels=PlotLabels(title=name.replace('_', ' '))))
            written = writer(figures / f"{name}.png", plot_input=plot_input, spec=spec,
                parameters={}, size="4x4", source={"task": self.instance_id, "report": name, "run_record": {**self._record(),
                    "device_snapshots": self._capture_evidence.get(name, {}).get("device_snapshots",summary["device_snapshots"])}})
            if hasattr(written, "result"): written = written.result()
            png, npz = written
            context.register_artifact(name + "_figure", npz, role="figure", contract_id="zlc.figure")
            context.register_artifact(name + "_preview", png, role="preview")
        self._timings["save_and_render"] = (perf_counter()-started)*1000
        summary["timing_ms"] = dict(self._timings)
        context.register_artifact("summary", write_readable_json(directory/"summary.json", _plain_json(summary)), role="summary")
        lines = ["SLM Rearrangement", f"Status: {status}", f"Pulse: {self.pulse_path}",
                 f"Movement frames: {self.motion_frames}; nominal rate: {self.frame_rate_hz:g} Hz"]
        if hasattr(self, "_pulse_timing"):
            lines.append(f"Available between imaging Periods: {self._pulse_timing['available_gap_seconds']:.6f} s")
        lines.extend(f"{name.replace('_',' ')}: {value:.3f} ms" for name,value in self._timings.items())
        for key in ("before_occupied", "filled_target_sites", "judged_target_sites", "target_sites", "target_filling_fraction"):
            if key in summary: lines.append(f"{key.replace('_',' ')}: {summary[key]}")
        for key, value in summary.get("motion_diagnostics", {}).items():
            lines.append(f"{key.replace('_', ' ')}: {value}")
        lines.extend(["Playback acknowledgments are transport facts, not optical vblank measurements.",
                      "Target filling is not individual-atom identity tracking.",
                      "compute/feed and device playback overlap inside online_rearrangement; do not add overlapping windows.",
                      "compute_callback is bounded queue/backpressure time, not numerical solving."])
        if error is not None: lines.append(f"Error: {error}")
        context.register_artifact("summary_text", atomic_write_text(directory/"summary.txt", '\n'.join(lines)+'\n'), role="summary")
        return archive

    def _target_detection(self):
        """Read known destination sites without inventing a new calibration."""
        n = len(self.points[1])
        result = {"counts": np.full(n, np.nan), "thresholds": np.full(n, np.nan),
                  "occupied": np.zeros(n, bool), "valid": np.zeros(n, bool)}
        if len(self._detections) < 2:
            return result
        known = self.target_indices >= 0
        for key in result:
            result[key][known] = self._detections[1][key][self.target_indices[known]]
        return result

    def execute(self, context):
        self.instance_id = context.instance_id
        self._timings = {}; prepared = capture = None
        self._snapshots, self._overlays, self._records, self._detections = {}, {}, [], []
        self._available_outputs, self._capture_evidence = set(), {}
        self._device_snapshots, self._overlay_geometry = {}, None
        self._playback, self._result, self._plan = None, None, None
        playback_attempted = False
        run_started = perf_counter()
        try:
            check_cancelled(context)
            context.report_progress("Detecting GPU and preparing the source optical working point")
            start = perf_counter()
            prepared = prepare_rearrangement(shape_yx=self.slm.shape_yx,
                source_yx=self.points[0], target_yx=self.points[1],
                pupil_amplitude=self.science_context["pupil_amplitude"],
                pupil_phase=-np.asarray(self.science_context["operator_wavefront"]),
                source_intensities=self.weights[0], target_intensities=self.weights[1],
                minimum_separation=self.minimum_separation,
                support_tolerance=self.intensity_tolerance,
                maximum_motion_frames=self.motion_frames,
                endpoint_data={"source_phase": self.science_context["phase"]},
                stop_requested=context.cancel_requested)
            self._timings["gpu_prepare"] = (perf_counter()-start)*1000
            self._gpu_info = prepared["gpu_info"]
            context.report_progress(f"GPU ready: {self._gpu_info.get('device_name','CUDA')}; establishing source phase")
            start = perf_counter()
            applied_source = self.slm.apply_phase(prepared["initial_phase"])
            if not np.array_equal(applied_source, prepared["initial_phase"]):
                raise RuntimeError("SLM did not apply the prepared source phase")
            if self.slm.last_command_receipt.get("outcome") != "known-new":
                raise RuntimeError("SLM source-phase outcome was not confirmed")
            self._timings["source_apply"] = (perf_counter()-start)*1000
            pulse = resolve_pulse(self.sequence, path=self.pulse_path, sequencer=self.sequencer, api_values={})
            start = perf_counter(); arm_sequencer(self.sequencer, pulse)
            self._timings["pulse_load"] = (perf_counter()-start)*1000
            loaded = self.sequencer.applied()
            self._pulse_timing = pulse_timing(loaded.source, loaded.program, loaded.rows, self.before_period, self.after_period)
            self._frame_interval = 1/self.frame_rate_hz
            settle = float(self.slm.last_command_receipt.get("settle_seconds", 0.))
            nominal = self.motion_frames*self._frame_interval + max(0., settle-self._frame_interval)
            self._timings["estimated_nominal_playback"] = nominal*1000
            photoelectron = reads_photoelectrons(self.calibration)
            node = CameraMeasurementNode(camera=self.camera,
                request=CameraMeasurementRequest(camera_key=self.camera_key, exposure_seconds=self.exposure_seconds,
                    roi_xywh=self.roi, repeat=2, frames_per_cycle=1, photoelectrons=bool(photoelectron)),
                signal_plane=self.signal_plane, producer=f"{self.instance_id}/camera")
            start = perf_counter(); capture = node.prepare(should_stop=context.cancel_requested)
            self._timings["camera_arm"] = (perf_counter()-start)*1000
            self._device_snapshots["camera"] = dict(node.run_record["device_snapshots"]["camera"])
            actual = node.actual_working_point
            self._prepare_outputs(context, node)
            self._publish_phase(context, applied_source)
            context.report_progress("Acquiring before photograph from the authored Pulse", current=0, total=2)
            firing_started = monotonic()
            fire_wall = time_ns()
            start = perf_counter(); execution = self.sequencer.fire(run_repeats=1, scan_repeats=1)
            self._timings["pulse_fire_request"] = (perf_counter()-start)*1000
            self._pulse_timing = pulse_timing(execution.source, execution.program, execution.rows, self.before_period, self.after_period)
            self._device_snapshots["sequencer"] = sequencer_archive_snapshot(applied=execution)
            deadline = firing_started + self._pulse_timing["after_start_seconds"] + self._pulse_timing["earliest_lane_delay_seconds"]
            camera_timeout = capture.timeout
            capture.timeout = max(camera_timeout, firing_started+self._pulse_timing['before_end_seconds']-monotonic()+camera_timeout)
            playback_finished_wall = None
            playback_finished_at = None

            def photograph(cycle, index):
                nonlocal playback_finished_wall, playback_finished_at, playback_attempted
                # Camera receive is independent: callback dispatch time is not
                # the exposure timestamp of a queued second photograph.
                node._commit_direct_cycle(cycle, index)
                start = perf_counter()
                mask, valid = self._read_photo(context, node, cycle, index)
                self._timings[("before", "after")[index]+"_readout"] = (perf_counter()-start)*1000
                if index == 1:
                    self._timings["after_frame_available_after_fire"] = (cycle[0].host_received_at_ns-fire_wall)/1e6
                    if playback_finished_wall is None or cycle[0].host_received_at_ns < playback_finished_wall:
                        raise RuntimeError("The verification frame reached the camera queue before SLM playback completed")
                    return
                self._timings["before_frame_available_after_fire"] = (cycle[0].host_received_at_ns-fire_wall)/1e6
                context.report_progress(f"Computing {self.motion_frames} motion frames for {len(self.points[1])} targets")
                start = perf_counter()
                plan = self._plan = plan_rearrangement(prepared, np.flatnonzero(mask & valid))
                self._timings['matching']=(perf_counter()-start)*1000
                online_started = start
                playback = None
                # The existing Task worker computes; one worker waits on the
                # device's existing play operation. Mapping/display ownership
                # remains in the SLM server, including on the same machine.
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="slm-play") as player:
                    def play():
                        nonlocal playback_finished_wall, playback_finished_at
                        began = perf_counter()
                        try:
                            return self.slm.play_phase_sequence(stop_requested=context.cancel_requested)
                        finally:
                            playback_finished_wall = time_ns()
                            playback_finished_at = monotonic()
                            self._timings["sequence_play_and_final_settle"] = (perf_counter()-began)*1000

                    def frame_ready(index, codes):
                        nonlocal playback, playback_attempted
                        check_cancelled(context)
                        if playback is None:
                            remaining = deadline-monotonic()
                            if remaining < nominal:
                                raise RuntimeError(f"Only {remaining:.6g}s remain before verification; nominal playback needs {nominal:.6g}s. Edit the Pulse gap.")
                            began = perf_counter()
                            self.slm.prepare_phase_sequence(None, 1/self.frame_rate_hz, frame_count=self.motion_frames)
                            self._timings["sequence_prepare"] = (perf_counter()-began)*1000
                            self._timings["first_verified_frame_ready"] = (began-online_started)*1000
                            playback_attempted = True
                            playback = player.submit(play)
                            context.report_progress("Computing and playing verified SLM frames")
                        if playback.done():
                            playback.result()  # Preserve an actual device error.
                            raise RuntimeError("SLM playback ended before all frames were submitted")
                        try:
                            self.slm.submit_phase_frame(index, codes)
                        except BaseException as admission_error:
                            # A real display/upload failure can wake a blocked
                            # queue producer as "cancelled". The play result,
                            # not that wake-up symptom, owns the device error.
                            if not playback.done():
                                try:
                                    self.slm.cancel_phase_sequence()
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
                        self._result = compute_rearrangement(prepared, plan,
                            motion_frames=self.motion_frames,
                            support_tolerance=self.intensity_tolerance,
                            require_converged=False, frame_ready=frame_ready,
                            stop_requested=context.cancel_requested)
                        self._timings["compute_and_feed"] = (perf_counter()-online_started)*1000
                        for name, value in self._result["timing_ms"].items():
                            self._timings["compute_"+name] = float(value)
                        if not self._result["converged"]:
                            raise RuntimeError("The phase sequence did not pass its encoded-field quality checks; playback is stopped at its verified prefix. See the partial numeric report.")
                        if playback is not None:
                            self._playback = playback.result()
                    except BaseException as error:
                        if playback is not None:
                            if not playback.done():
                                try:
                                    self.slm.cancel_phase_sequence()
                                except BaseException as cleanup:
                                    error.add_note(f"SLM cancellation also failed: {cleanup}")
                            try:
                                self._playback = playback.result()
                            except BaseException as cleanup:
                                if cleanup is not error:
                                    error.add_note(f"SLM playback also failed: {cleanup}")
                        raise
                    finally:
                        self._timings["online_rearrangement"] = (perf_counter()-online_started)*1000
                if not len(self._result["phase_codes"]):
                    self._playback = {"frame_count": 0, "played_frames": 0, "cancelled": False,
                                      "noop": True, "acknowledgment": "No new phase commanded"}
                    playback_finished_wall = time_ns()
                    context.report_progress("SLM target held; acquiring verification photograph", current=1, total=2)
                    capture.timeout = max(camera_timeout,
                        firing_started+self._pulse_timing['after_end_seconds']-monotonic()+camera_timeout)
                    return
                if self._playback["cancelled"] or self._playback["played_frames"] != len(self._result["phase_codes"]):
                    raise RuntimeError("SLM sequence stopped before its target frame")
                self._timings["verification_deadline_margin"] = (deadline-playback_finished_at)*1000
                if playback_finished_at > deadline:
                    raise RuntimeError("SLM playback missed the conservative after-imaging deadline; verification is not accepted")
                context.report_progress("SLM target held; acquiring verification photograph", current=1, total=2)
                # This Task knows the next trigger is intentionally later in
                # the same authored Pulse. Retain the camera's timeout AFTER
                # that scheduled window rather than timing out inside the gap.
                capture.timeout = max(camera_timeout,
                    firing_started+self._pulse_timing['after_end_seconds']-monotonic()+camera_timeout)

            result = capture.collect(commit_cycle=photograph, retain_cycles=False)
            report = wait_for_report(self.sequencer, context)
            if report.fault or not report.status & STATUS_DONE or report.status & (STATUS_RUNNING|STATUS_ERROR|STATUS_UNDERFLOW):
                raise RuntimeError(f"Pulse did not complete successfully: {report.fault or report.status}")
            if result is None or result.cycle_count != 2:
                raise RuntimeError("The authored Pulse did not deliver exactly two photographs")
            self._timings["pulse_elapsed"] = report.elapsed_seconds*1000
            self._timings["pulse_report_retrieval_delay"] = report.report_delay_seconds*1000
            self._timings["experiment_before_save"] = (perf_counter()-run_started)*1000
            self._revision += 1
            path = self._result["motion_yx"]
            full_path = np.zeros((self.motion_frames+1, len(self.points[0]), 2), dtype=np.float64)
            valid_path = np.zeros(full_path.shape[:-1], bool)
            if len(self._plan["source_indices"]):
                full_path[:len(path), self._plan["source_indices"]] = path
                valid_path[:len(path), self._plan["source_indices"]] = True
            path = full_path
            frame_axis = AxisSpec(AxisId("slm_rearrangement.frame"), "frame", SCAN_POINT,
                                 len(path), tuple(float(i) for i in range(len(path))))
            site_axis = AxisSpec(AxisId("slm_rearrangement.source_site"), "site", COMPONENT,
                                path.shape[1], tuple(float(i+1) for i in range(path.shape[1])))
            coordinate_axis = AxisSpec(AxisId('slm_rearrangement.coordinate'),'coordinate',COMPONENT,2,(0.,1.),coordinate_labels=('x','y'))
            trajectory = self._snapshot(context, TRAJECTORY_OUTPUT, path[None,:,:,::-1],
                                         point_axes=(frame_axis,), cell_axes=(site_axis,coordinate_axis),
                                         validity=np.repeat(valid_path[None,...,None],2,axis=-1),value_unit="1")
            ratio = self._result["support_intensity_ratios"]
            ratio_valid = np.ones(self.motion_frames, bool) if len(ratio) else np.zeros(self.motion_frames, bool)
            if not len(ratio): ratio = np.zeros(self.motion_frames)
            quality = self._snapshot(context, QUALITY_OUTPUT, ratio[None],
                point_axes=(AxisSpec(AxisId("slm_rearrangement.output_frame"), "frame", SCAN_POINT,
                 self.motion_frames, tuple(float(i) for i in range(self.motion_frames))),), validity=ratio_valid[None])
            self._publish(context, ((TRAJECTORY_OUTPUT, trajectory), (QUALITY_OUTPUT, quality)))
            expected_final = (phase_from_codes(self._result["phase_codes"][-1], self.slm.shape_yx)
                              if len(self._result["phase_codes"]) else prepared["initial_phase"])
            confirmed = self.slm.last_commanded_phase
            if self.slm.last_command_receipt.get("outcome") != "known-new" or confirmed is None or not np.array_equal(confirmed, expected_final):
                raise RuntimeError("The SLM did not confirm the final target phase")
            self._publish_phase(context, confirmed)
            context.seal_terminal()
            context.report_progress("Saving photographs, phase sequence, verification and timing report")
            archive = self._save(context, "completed")
            verification = self._target_detection()
            return {"artifact_path": str(archive), "target_filling_fraction":
                    float(verification["occupied"].mean()) if verification["valid"].all() else None}
        except BaseException as error:
            if playback_attempted and self._playback is None:
                self._playback = self.slm.last_command_receipt.get("sequence")
            try:
                self.sequencer.safe()
            except BaseException as cleanup: error.add_note(f"Pulse SAFE also failed: {cleanup}")
            try:
                self._save(context, "stopped" if context.cancel_requested() else "failed", error)
            except BaseException as cleanup: error.add_note(f"Partial report also failed: {cleanup}")
            raise
        finally:
            active_error = sys.exception()
            cleanup_errors = []
            if capture is not None:
                if not capture.closed:
                    capture.stopped = True
                    try: capture.close()
                    except BaseException as cleanup: cleanup_errors.append(cleanup)
                # The Task's own before/after snapshots and archive now own
                # the evidence; its private acquisition is no longer a source.
                try: self.signal_plane.retire(capture.node)
                except BaseException as cleanup: cleanup_errors.append(cleanup)
            for cleanup in (self.slm.release_phase_sequence, None if prepared is None else prepared["close"]):
                if cleanup is not None:
                    try: cleanup()
                    except BaseException as error: cleanup_errors.append(error)
            # The durable archive and preview snapshots now own the results;
            # a stopped/completed node must not retain a whole pinned movie.
            self._result = None
            if active_error is not None:
                for error in cleanup_errors: active_error.add_note(f"Cleanup also failed: {error}")
            elif cleanup_errors:
                first, *rest = cleanup_errors
                for error in rest: first.add_note(f"Cleanup also failed: {error}")
                raise first


__all__ = ["SlmRearrangementTask", "pulse_timing"]
