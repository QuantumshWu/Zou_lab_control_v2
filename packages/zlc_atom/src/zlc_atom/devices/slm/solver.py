"""Continuous SLM targets, phase retrieval, and their file formats.

Pillow, unicodedata and scipy are all reached from inside the functions
that use them.  A logic node's descriptor names one of the file readers
here as its artifact codec, so DISCOVERING the nodes -- which a task
console does before it shows anything -- imports this module; at module
scope those three cost it a font stack and a solver stack for a target
nobody has authored and a pattern nobody has solved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import atexit
import json
import os
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from typing import Any

import numpy as np
from zlc_durable import atomic_write_file, strict_json_loads, write_readable_json

from .device import canonical_phase

_TARGET_FORMAT = "zlc.slm.target"
_TARGET_KEYS = frozenset(
    {"format", "shape", "intensity", "objective_kind"}
)
_SCIENCE_CONTEXT_FORMAT = "zlc.slm.science-context"
SCIENCE_CONTEXT_ARTIFACT_CONTRACT = _SCIENCE_CONTEXT_FORMAT
_SCIENCE_CONTEXT_MEMBERS = frozenset(
    {"pattern_phase_delta", "target_intensity", "metadata"}
)
_SCIENCE_CONTEXT_KEYS = frozenset(
    {
        "format",
        "objective_kind",
        "pupil",
        "system_correction",
        "command_receipt",
        "pattern_metadata",
        "operator_metadata",
    }
)
_OBJECTIVE_KINDS = frozenset({"auto", "spots", "image"})
_SYSTEM_CORRECTION_KINDS = frozenset(
    {"pupil_phase_map", "target_response_map"}
)
_OPERATOR_MODES = frozenset(
    {
        "defocus", "astig_oblique", "astig_vertical", "coma_y", "coma_x",
        "trefoil_y", "trefoil_x", "spherical",
    }
)
_PHASE_CODE_COUNT = 1 << 16

def _pair(value: object, name: str) -> tuple[int, int]:
    try:
        pair = tuple(int(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a two-integer pair") from error
    if len(pair) != 2 or any(item <= 0 for item in pair):
        raise ValueError(f"{name} must contain two positive integers")
    return pair

def _float_pair(value: object, name: str) -> tuple[float, float]:
    try:
        pair = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a two-number pair") from error
    if len(pair) != 2 or any(not np.isfinite(item) or item <= 0.0 for item in pair):
        raise ValueError(f"{name} must contain two finite positive numbers")
    return pair

def _scalar(value: object, name: str, *, nonnegative: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (result < 0.0 if nonnegative else result <= 0.0):
        qualifier = "non-negative" if nonnegative else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result

def _readonly(values: object) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype="<f4")
    return np.frombuffer(contiguous.tobytes(), dtype="<f4").reshape(
        contiguous.shape
    )

def validate_target(values: object) -> np.ndarray:
    """Return the sole target representation: finite non-negative intensity."""

    try:
        target = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise TypeError("target intensity must be a numeric array") from error
    if target.ndim != 2:
        raise ValueError("target intensity must be two-dimensional")
    if min(target.shape) < 2:
        raise ValueError("target intensity dimensions must each be at least two")
    if not np.all(np.isfinite(target)):
        raise ValueError("target intensity must be finite")
    if np.any(target < 0.0):
        raise ValueError("target intensity must be non-negative")
    return _readonly(target)


def _objective_kind(value: object) -> str:
    if type(value) is not str or value not in _OBJECTIVE_KINDS:
        raise ValueError("objective_kind must be 'auto', 'spots', or 'image'")
    return value

def _grid_indices(
    shape_yx: object,
    grid_shape_yx: object,
    spacing_yx: object | None,
) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    shape = _pair(shape_yx, "shape_yx")
    grid = _pair(grid_shape_yx, "grid_shape_yx")
    spacing = (
        tuple(max(1, shape[index] // (grid[index] + 1)) for index in range(2))
        if spacing_yx is None
        else _pair(spacing_yx, "spacing_yx")
    )
    spans = tuple((grid[index] - 1) * spacing[index] for index in range(2))
    if any(span >= shape[index] for index, span in enumerate(spans)):
        raise ValueError("grid does not fit inside target shape")
    starts = tuple((shape[index] - spans[index]) // 2 for index in range(2))
    axes = tuple(
        starts[index] + spacing[index] * np.arange(grid[index], dtype=int)
        for index in range(2)
    )
    return shape, axes[0], axes[1]

def prepare_rearrangement_geometry(
    source_yx: object, target_yx: object, *, shape_yx: tuple[int, int],
    minimum_separation: float,
) -> dict[str, object]:
    """Prepare native endpoint geometry and Euclidean bottleneck edge costs."""
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
    from scipy.sparse import csr_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import maximum_bipartite_matching  # noqa: PLC0415
    from scipy.spatial.distance import pdist  # noqa: PLC0415

    shape = _pair(shape_yx, "shape_yx")
    if not np.array_equal(np.asarray(shape_yx), shape):
        raise ValueError("shape_yx must contain positive integers")
    sites = []
    for name, values in (("source_yx", source_yx), ("target_yx", target_yx)):
        array = np.asarray(values)
        if (array.ndim != 2 or array.shape[1] != 2 or not len(array)
                or array.dtype.kind not in "iuf" or not np.all(np.isfinite(array))
                or np.any(array != np.floor(array))):
            raise ValueError(f"{name} must be a nonempty matrix of integer Y,X sites")
        if np.any(array < 0) or np.any(array >= shape):
            raise ValueError(f"{name} lies outside the native SLM shape")
        if np.any(array > np.iinfo(np.int32).max):
            raise ValueError(f"{name} exceeds the native int32 coordinate range")
        if len(np.unique(array, axis=0)) != len(array):
            raise ValueError(f"{name} must contain unique sites")
        sites.append(np.frombuffer(np.ascontiguousarray(array, dtype=np.int32).tobytes(), np.int32).reshape(-1, 2))
    source, target = sites
    distance = np.linalg.norm(target[:, None].astype(float) - source[None], axis=-1)
    # Import and initialize the established assignment implementation before imaging.
    linear_sum_assignment(np.zeros((1, 1)))
    maximum_bipartite_matching(csr_matrix([[True]]), perm_type="column")
    pdist(np.zeros((1, 2)), metric="sqeuclidean")
    return {
        "source_yx": source, "target_yx": target, "shape_yx": shape,
        "assignment_costs": _frozen(distance),
        "minimum_separation": _scalar(minimum_separation, "minimum_separation", nonnegative=True),
    }


def plan_rearrangement(prepared: Mapping[str, object], available_source_indices: object) -> dict[str, object]:
    """Minimize the longest path, subject to continuous same-time clearance.

    Threshold matching gives an exact assignment lower bound. Conflict-directed
    reassignment and waypoint/wait repair are bounded searches; the returned
    maximum is globally optimal only when it reaches that lower bound.
    """
    from heapq import heappop, heappush  # noqa: PLC0415
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
    from scipy.sparse import csr_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import maximum_bipartite_matching  # noqa: PLC0415
    from scipy.spatial.distance import pdist  # noqa: PLC0415

    source, target = prepared["source_yx"], prepared["target_yx"]
    indices = np.asarray(available_source_indices)
    if not indices.size:
        indices = np.empty(0, np.intp)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= len(source))):
        raise ValueError("available_source_indices must contain integer indices into the prepared source roster")
    available = np.unique(indices).astype(np.intp, copy=False)
    costs = np.asarray(prepared["assignment_costs"])[:, available]
    cardinality = min(costs.shape)
    separation = float(prepared["minimum_separation"])
    endpoints = [("initial occupied sources", source[available])]
    if cardinality == len(target):
        endpoints.append(("required target sites", target))
    for name, points in endpoints:
        # All occupied sources coexist before surplus fade. All target sites
        # are mandatory only when there are enough atoms: a sparse target
        # subset must not be rejected because two unused endpoints are close.
        clearance = float(np.sqrt(np.min(pdist(points, metric="sqeuclidean"), initial=np.inf)))
        if clearance < separation:
            raise ValueError(f"{name} clearance {clearance:g} is below the authored minimum separation {separation:g}")
    radii = np.unique(costs)

    def assignment(blocked):
        if not cardinality:
            return 0., np.empty(0, np.intp), np.empty(0, np.intp)
        allowed = np.ones(costs.shape, bool)
        for row, column in blocked:
            allowed[row, column] = False
        # Every member of the smaller roster must be filled. Start at its
        # nearest-neighbour lower bound: a translated400-site array needs one
        # sparse matching, not binary probes through dense long-distance graphs.
        nearest = np.min(np.where(allowed, costs, np.inf), axis=1 if costs.shape[0] <= costs.shape[1] else 0)
        lo = int(np.searchsorted(radii, np.max(nearest)))
        hi, stride = lo, 1
        while hi < len(radii):
            matching = maximum_bipartite_matching(
                csr_matrix(allowed & (costs <= radii[hi])), perm_type="column")
            if np.count_nonzero(matching >= 0) == cardinality:
                break
            lo, hi, stride = hi + 1, min(len(radii), hi + stride), stride * 2
        if hi == len(radii):
            hi -= 1
            if lo > hi:
                return None
            matching = maximum_bipartite_matching(csr_matrix(allowed), perm_type="column")
            if np.count_nonzero(matching >= 0) != cardinality:
                return None
        while lo < hi:
            middle = (lo + hi) // 2
            matching = maximum_bipartite_matching(
                csr_matrix(allowed & (costs <= radii[middle])), perm_type="column")
            if np.count_nonzero(matching >= 0) == cardinality:
                hi = middle
            else:
                lo = middle + 1
        # Only a tie break inside the exact bottleneck radius. Squared edge
        # length favours coordinated short moves over long relay shortcuts.
        rows, columns = linear_sum_assignment(
            np.where(allowed & (costs <= radii[lo]), costs ** 2, np.inf))
        order = np.argsort(columns)
        return float(radii[lo]), rows[order], columns[order]

    search_limited = False

    def repair(motion, initial_nearest):
        nonlocal search_limited
        start, destination = motion[0], motion[-1]
        velocity = destination - start
        moving = np.any(destination != start, axis=1)
        speeds = np.linalg.norm(velocity, axis=1)
        first, second = np.triu_indices(len(start), 1)
        # An immutable site is an obstacle for the whole straight path: no
        # departure delay can fix that collision. Reassignment/waypoints below
        # can still move it, so this only skips an impossible waiting search.
        static_collision = np.any((initial_nearest < separation ** 2)
                                  & (~moving[:, None] | ~moving[None, :]))
        schedules = ([] if static_collision else
                     [(float(speeds.max(initial=0.)), 0, frozenset(), np.zeros(len(start)), initial_nearest)])
        seen_schedules = {frozenset()}
        unsafe_delays = {}
        best_wait, best_wait_distance = None, np.inf
        for _ in range(32):
            if not schedules:
                break
            distance, _, constraints, begins, closest = heappop(schedules)
            if np.min(closest) >= separation ** 2:
                if distance < best_wait_distance:
                    knots = np.unique(np.r_[0., begins[moving], begins[moving] + 1.])
                    progress = np.where(knots[:, None] >= begins[None] + 1., 1.,
                                        np.clip(knots[:, None] - begins[None], 0., 1.))
                    candidate = start[None] + progress[..., None] * velocity
                    # The compact per-pair calculation and the final shared
                    # trajectory gate must agree; never relax the hard limit.
                    if rearrangement_clearance(candidate) >= separation:
                        best_wait, best_wait_distance = candidate, distance
                continue
            i, j = np.unravel_index(np.argmin(closest), closest.shape)
            pair = (min(i, j), max(i, j))
            if pair not in unsafe_delays:
                a, b = pair
                unsafe_delays[pair] = _rearrangement_unsafe_delay(
                    start[a], destination[a], start[b], destination[b], separation)
            forbidden = unsafe_delays[pair]
            if forbidden is None:
                continue
            i, j = pair
            for origin, follower, lag in ((i, j, forbidden[1]), (j, i, -forbidden[0])):
                if not np.isfinite(lag):
                    continue
                # Boundaries of the open unsafe interval are safe in exact
                # arithmetic. Absolute-coordinate interpolation needs more
                # room than one ULP of a small delay. Convert a conservative
                # float64 arithmetic bound to time; do not lower clearance.
                scale = max(1., float(np.max(abs(start[[i, j]]))), float(np.max(abs(destination[[i, j]]))))
                roundoff_time = 64 * np.finfo(float).eps * scale / min(speeds[i], speeds[j])
                lag = float(np.nextafter(lag + roundoff_time, np.inf))
                old_edges = {(a, b): value for a, b, value in constraints}
                old_edges[origin, follower] = max(lag, old_edges.get((origin, follower), -np.inf))
                child = frozenset((a, b, value) for (a, b), value in old_edges.items())
                if child in seen_schedules:
                    continue
                seen_schedules.add(child)
                updated = begins.copy()
                successors = [[] for _ in start]
                for a, b, value in child:
                    successors[a].append((b, value))
                pending, valid = [origin], True
                while pending and valid:
                    a = pending.pop()
                    for b, value in successors[a]:
                        required = updated[a] + value
                        if required > updated[b]:
                            # Parent constraints are feasible. A newly positive
                            # cycle must include the new/tightened edge, hence
                            # would increase its origin. Zero/negative cycles
                            # remain valid, unlike a DAG-only precedence rule.
                            if b == origin:
                                valid = False
                                break
                            updated[b] = required
                            pending.append(b)
                if not valid:
                    continue
                changed = updated != begins
                affected = changed[first] | changed[second]
                a, b = first[affected], second[affected]
                child_closest = closest.copy()
                if len(a):
                    delta = updated[b] - updated[a]
                    u, v = np.clip(delta, 0., 1.)[:, None], np.clip(-delta, 0., 1.)[:, None]
                    # For each pair these four vertices describe its entire
                    # held/moving interaction, including fractional overlap.
                    # Each pair has its own three segments; the existing
                    # continuous formula is the only distance implementation.
                    paired_motion = np.stack((
                        np.concatenate((start[a], start[b])),
                        np.concatenate((start[a] + velocity[a] * u, start[b] + velocity[b] * v)),
                        np.concatenate((destination[a] - velocity[a] * v, destination[b] - velocity[b] * u)),
                        np.concatenate((destination[a], destination[b]))))
                    values, _, _ = _rearrangement_pair_metrics(
                        paired_motion, np.arange(len(a)), np.arange(len(a)) + len(a))
                    child_closest[a, b], child_closest[b, a] = values, values
                knots = np.unique(np.r_[0., updated[moving], updated[moving] + 1.])
                active = (knots[:-1, None] >= updated) & (knots[:-1, None] < updated + 1.)
                cost = float(np.sum(np.diff(knots) * np.max(active * speeds, axis=1, initial=0.)))
                heappush(schedules, (cost, len(seen_schedules), child, updated, child_closest))
        search_limited |= bool(schedules)
        if best_wait is not None:
            return best_wait, "straight paths with continuous departure delays"
        fractions = np.array([0., 1.])
        nearest = initial_nearest.copy()
        for _ in range(min(16, len(start))):
            before_count = np.count_nonzero(nearest < separation ** 2)
            i, j = np.unravel_index(np.argmin(nearest), nearest.shape)
            _, segments, local_times = _rearrangement_pair_metrics(motion, (i,), (j,))
            interval = int(segments[0])
            local_time = float(local_times[0])
            when = fractions[interval] + local_time * (fractions[interval + 1] - fractions[interval])
            if not 0 < when < 1:
                break
            vertex = interval if local_time == 0. else interval + 1 if local_time == 1. else None
            if vertex in (0, len(motion) - 1):
                break
            middle = motion[interval] + local_time * (motion[interval + 1] - motion[interval])
            direction = middle[i] - middle[j]
            if np.linalg.norm(direction) < 1e-12:
                velocity = (motion[interval + 1, i] - motion[interval, i]
                            - motion[interval + 1, j] + motion[interval, j])
                direction = np.asarray([-velocity[1], velocity[0]])
            if np.linalg.norm(direction) < 1e-12:
                break
            direction /= np.linalg.norm(direction)
            candidates = []
            for which, sign in ((i, 1), (j, -1)):
                old_row_count = np.count_nonzero(nearest[which] < separation ** 2)
                unaffected_minimum = min(
                    float(np.min(nearest[:which, :which], initial=np.inf)),
                    float(np.min(nearest[:which, which + 1:], initial=np.inf)),
                    float(np.min(nearest[which + 1:, which + 1:], initial=np.inf)))
                for side in (1., -1.):
                    for margin in (1.5, 2., 3.):
                        waypoint = middle.copy()
                        anchor = middle[j] if which == i else middle[i]
                        waypoint[which] = anchor + sign * side * margin * separation * direction
                        if np.any(waypoint < 0) or np.any(waypoint >= prepared["shape_yx"]):
                            continue
                        candidate = motion.copy() if vertex is not None else np.insert(
                            motion, interval + 1, waypoint, axis=0)
                        if vertex is not None:
                            candidate[vertex] = waypoint
                        # Other paths are unchanged: inserting their linear
                        # interpolation does not alter any pair distance.
                        changed, _, _ = _rearrangement_pair_metrics(candidate, (which,))
                        after_count = before_count + 2 * (np.count_nonzero(changed[0] < separation ** 2) - old_row_count)
                        after_minimum = min(unaffected_minimum, float(np.min(changed[0], initial=np.inf)))
                        if (after_count < before_count or
                                (after_count == before_count and after_minimum > np.min(nearest))):
                            length = float(np.max(np.linalg.norm(
                                np.diff(candidate, axis=0), axis=-1).sum(axis=0), initial=0.))
                            candidates.append((after_count, length, candidate, which, changed[0]))
            if not candidates:
                break
            _, _, motion, which, changed = min(candidates, key=lambda item: (item[0], item[1]))
            nearest[which], nearest[:, which] = changed, changed
            if vertex is None:
                fractions = np.insert(fractions, interval + 1, when)
            if np.min(nearest) >= separation ** 2:
                return motion, "joint assignment and waypoint clearance repair"
        else:
            search_limited = True
        return None

    lower_bound, rows, columns = assignment(())
    queue = [(lower_bound, 0, frozenset(), rows, columns)]
    seen = {frozenset()}
    repaired_assignments = {}
    best, best_score, evaluated = None, (np.inf, np.inf, np.inf), 0
    # Branch on both members of the closest conflicting assignment pair. No
    # source is locked merely because its start coincides with a target.
    # Bounds order the search by max length, not total length or agent count.
    while queue and evaluated < 128:
        bound, _, blocked, rows, columns = heappop(queue)
        if bound > best_score[0]:
            break
        evaluated += 1
        start = source[available[columns]].astype(float)
        destination = target[rows].astype(float)
        motion = np.stack((start, destination))
        displacement = destination - start
        if np.all(displacement == displacement[:1]):
            # Initial occupied endpoints were checked above. Exact common
            # translation keeps every pair distance unchanged.
            nearest, clear = None, True
        else:
            nearest, _, _ = _rearrangement_pair_metrics(motion)
            clear = np.min(nearest, initial=np.inf) >= separation ** 2
        if clear:
            repaired = motion, "bottleneck assignment; simultaneous straight paths"
        else:
            # Different forbidden-edge nodes may yield the same exact mapping.
            # Routing depends on that mapping, not on which unused assignment
            # edges were forbidden. Reuse it, but retain both search nodes.
            mapping = columns.tobytes(), rows.tobytes()
            if mapping not in repaired_assignments:
                repaired_assignments[mapping] = repair(motion, nearest)
            repaired = repaired_assignments[mapping]
        if repaired is not None:
            vertices, route = repaired
            lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=-1)
            maximum = float(np.max(lengths.sum(axis=0), initial=0.))
            parallel_distance = float(np.max(lengths, axis=1, initial=0.).sum())
            score = maximum, parallel_distance, len(vertices)
            if score < best_score:
                best, best_score = (rows, columns, vertices, route), score
            # Reaching the geometric bound alone need not minimize waiting.
            # Straight parallel distance has the same lower bound, so only
            # when both reach it is further equal-length search redundant.
            if best_score[0] <= np.nextafter(lower_bound, np.inf) and best_score[1] <= np.nextafter(lower_bound, np.inf):
                break
        if clear:
            continue
        i, j = np.unravel_index(np.argmin(nearest), nearest.shape)
        for which in (i, j):
            child = blocked | {(int(rows[which]), int(columns[which]))}
            if child in seen:
                continue
            seen.add(child)
            match = assignment(child)
            if match is not None and match[0] <= best_score[0]:
                heappush(queue, (match[0], len(seen), child, match[1], match[2]))
    if best is None:
        raise ValueError("no valid schedule found at the authored minimum separation")
    ordered_targets, columns, motion, route = best
    # Simplify the winning joint path, not the sampler: every replacement
    # segment passes the same continuous clearance gate. Other segments are
    # unchanged, and triangle inequality cannot increase any path length or
    # the sum of maximum segment travel. This removes redundant near-coincident
    # events without merging times by a tolerance or weakening separation.
    removed_vertices, vertex = 0, 1
    while vertex < len(motion) - 1:
        if rearrangement_clearance(motion[[vertex - 1, vertex + 1]]) >= separation:
            motion = np.delete(motion, vertex, axis=0)
            removed_vertices += 1
            vertex = max(1, vertex - 1)
        else:
            vertex += 1
    if removed_vertices:
        route += "; checked waypoint shortcuts"
    motion, fractions, _ = _rearrangement_motion_timing(motion)
    ordered_sources = available[columns]
    removed = available[~np.isin(available, ordered_sources)]
    order = np.argsort(ordered_targets)
    assigned, target_indices = ordered_sources[order], ordered_targets[order]
    filled = np.zeros(len(target), bool)
    filled[target_indices] = True
    lengths = np.linalg.norm(np.diff(motion, axis=0), axis=-1)
    best_score = (float(np.max(lengths.sum(axis=0), initial=0.)),
                  float(np.max(lengths, axis=1, initial=0.).sum()), len(motion))
    distance = float(lengths.sum())
    return {
        "assigned_source_indices": _frozen(assigned),
        "assigned_target_indices": _frozen(target_indices),
        "removed_source_indices": _frozen(removed),
        "source_indices": _frozen(ordered_sources), "target_indices": _frozen(ordered_targets),
        "target_filled": _frozen(filled),
        "initial_occupied_count": len(available), "selected_count": len(assigned),
        "motion_yx": _frozen(motion.astype(np.float64)),
        "fraction": _frozen(fractions), "total_distance": distance,
        "maximum_path_length": best_score[0], "maximum_path_lower_bound": lower_bound,
        "optimality_gap": max(0., best_score[0] - lower_bound),
        "parallel_travel_distance": best_score[1],
        "shortcut_vertices_removed": removed_vertices,
        "routing": route, "assignment_candidates": evaluated,
        "search_budget_exhausted": bool((search_limited or (queue and evaluated >= 128))
                                        and best_score[1] > np.nextafter(lower_bound, np.inf)),
    }


def _rearrangement_motion_timing(motion):
    """Retain corners, remove global idle knots, and bound the common speed.

    Reparameterizing a complete simultaneous segment does not change any
    same-time pair clearance. A local detour must receive time proportional
    to its actual travel, not its old nearly-zero collision-time interval.
    """
    distances = np.max(np.linalg.norm(np.diff(motion, axis=0), axis=-1), axis=1, initial=0.)
    if not np.any(distances):
        keep = np.array([0, len(motion) - 1])
        return motion[keep], np.array([0., 1.]), keep
    keep = np.r_[0, np.flatnonzero(distances > 0) + 1]
    cumulative = np.r_[0., np.cumsum(distances[distances > 0])]
    return motion[keep], cumulative / cumulative[-1], keep


def _rearrangement_unsafe_delay(start_i, end_i, start_j, end_j, separation):
    """Open unsafe interval of delta=start_time_j-start_time_i, or None.

    Both straight moves last one unit, with endpoints held before/after.
    During motion, ||r+a*u-b*v||<d intersects the unit (u,v) square in a
    convex set; projecting delta=u-v gives an interval. Endpoint holds add
    rays attached to its four edges. Extremes lie on an edge or where the
    moving relative line is tangent to the separation circle.
    """
    from math import sqrt  # noqa: PLC0415

    rx, ry = float(start_i[0] - start_j[0]), float(start_i[1] - start_j[1])
    ax, ay = float(end_i[0] - start_i[0]), float(end_i[1] - start_i[1])
    bx, by = float(end_j[0] - start_j[0]), float(end_j[1] - start_j[1])
    distance = float(separation)
    if distance == 0:
        return None
    boundary = []
    lower_unbounded = upper_unbounded = False
    # (relative start, direction, delta scale/offset, unbounded side).
    for x, y, dx, dy, scale, offset, lower in (
        (rx, ry, -bx, -by, -1., 0., True),       # i held at start
        (rx + ax, ry + ay, -bx, -by, -1., 1., False),  # i held at end
        (rx, ry, ax, ay, 1., 0., False),         # j held at start
        (rx - bx, ry - by, ax, ay, 1., -1., True),  # j held at end
    ):
        speed = dx * dx + dy * dy
        if speed:
            cross = x * dy - y * dx
            discriminant = distance * distance * speed - cross * cross
            if discriminant <= 0:
                continue
            center = -(x * dx + y * dy) / speed
            radius = sqrt(discriminant) / speed
            lo, hi = max(0., center - radius), min(1., center + radius)
            if lo >= hi:
                continue
        elif x * x + y * y < distance * distance:
            lo, hi = 0., 1.
        else:
            continue
        boundary.extend((scale * lo + offset, scale * hi + offset))
        lower_unbounded |= lower
        upper_unbounded |= not lower
    vx, vy = ax - bx, ay - by
    speed = vx * vx + vy * vy
    cross_b = vx * by - vy * bx
    if speed and cross_b:
        cross_r = vx * ry - vy * rx
        radius = distance * sqrt(speed)
        for sign in (-1., 1.):
            delta = (sign * radius - cross_r) / cross_b
            when = -(vx * (rx + bx * delta) + vy * (ry + by * delta)) / speed
            if max(0., delta) <= when <= min(1., delta + 1.):
                boundary.append(delta)
    if not boundary:
        return None
    lo = -np.inf if lower_unbounded else min(boundary)
    hi = np.inf if upper_unbounded else max(boundary)
    return (lo, hi) if lo < hi else None


def _rearrangement_pair_metrics(motion, source_indices=None, target_indices=None):
    """One continuous same-time pair-distance calculation for gates and routing."""
    size = motion.shape[1]
    selected = np.arange(size) if source_indices is None else np.asarray(source_indices, np.intp)
    paired = None if target_indices is None else np.asarray(target_indices, np.intp)
    minimum = np.full((len(selected), size) if paired is None else (len(selected),), np.inf)
    intervals = np.zeros_like(minimum, np.intp)
    times = np.zeros_like(minimum)
    for index in range(max(1, len(motion) - 1)):
        start = motion[index]
        end = motion[min(index + 1, len(motion) - 1)]
        delta = end - start
        if paired is None:
            y, x = (start[selected, None, axis] - start[None, :, axis] for axis in (0, 1))
            vy, vx = (delta[selected, None, axis] - delta[None, :, axis] for axis in (0, 1))
        else:
            y, x = (start[selected, axis] - start[paired, axis] for axis in (0, 1))
            vy, vx = (delta[selected, axis] - delta[paired, axis] for axis in (0, 1))
        speed = vy ** 2 + vx ** 2
        at = np.divide(-(y * vy + x * vx), speed,
                       out=np.zeros_like(speed), where=speed > 0)
        at = np.clip(at, 0., 1.)
        squared = (y + at * vy) ** 2 + (x + at * vx) ** 2
        if paired is None:
            squared[np.arange(len(selected)), selected] = np.inf
        else:
            squared[selected == paired] = np.inf
        update = squared < minimum
        minimum[update], intervals[update], times[update] = squared[update], index, at[update]
    return minimum, intervals, times


def rearrangement_clearance(motion_yx: object) -> float:
    """Exact minimum distance along every emitted straight frame-to-frame path."""
    motion = np.asarray(motion_yx, dtype=float)
    if (motion.ndim != 3 or motion.shape[2] != 2 or len(motion) < 1
            or not np.all(np.isfinite(motion))):
        raise ValueError("motion_yx must contain finite frame, site, Y/X coordinates")
    if motion.shape[1] < 2:
        return np.inf
    initial, _, _ = _rearrangement_pair_metrics(motion[:1])
    upper = float(np.min(initial))
    if not np.any(motion[1:] != motion[:1]):
        return float(np.sqrt(upper))
    displacement = motion - motion[:1]
    translation = np.mean(displacement, axis=1, keepdims=True)
    residual = displacement - translation
    radius = np.max(np.linalg.norm(residual, axis=-1), axis=0)
    # The common translation cancels in every pair. A site's residual stays
    # inside this vertex-radius bound along each continuous linear segment.
    # Bound roundoff of the two subtractions/two-component norm outward;
    # this is arithmetic error, not a physical clearance tolerance.
    epsilon = np.finfo(np.float64).eps
    gamma = 8 * epsilon / (1 - 8 * epsilon)
    radius += gamma * np.sqrt(2.) * (np.max(abs(motion), axis=(0, 2))
                                   + np.max(abs(motion[0]), axis=1) + np.max(abs(translation)))
    lower = np.nextafter(np.sqrt(initial) - radius[:, None] - radius[None], -np.inf)
    first, second = np.nonzero(np.triu(lower <= np.nextafter(np.sqrt(upper), np.inf), 1))
    if len(first):
        squared, _, _ = _rearrangement_pair_metrics(motion, first, second)
        upper = min(upper, float(np.min(squared)))
    return float(np.sqrt(upper))


def preset_grid(
    shape_yx: object,
    grid_shape_yx: object,
    *,
    spacing_yx: object | None = None,
    intensity: float = 1.0,
) -> np.ndarray:
    shape, rows, columns = _grid_indices(shape_yx, grid_shape_yx, spacing_yx)
    target = np.zeros(shape, dtype=np.float32)
    target[np.ix_(rows, columns)] = _scalar(intensity, "intensity", nonnegative=True)
    return validate_target(target)

def preset_checkerboard(
    shape_yx: object,
    grid_shape_yx: object,
    *,
    spacing_yx: object | None = None,
    intensity: float = 1.0,
) -> np.ndarray:
    shape = _pair(shape_yx, "shape_yx")
    row_count, long_count = _pair(grid_shape_yx, "grid_shape_yx")
    if long_count < 2:
        raise ValueError("checkerboard long rows must contain at least two sites")
    spacing = (
        (
            max(1, shape[0] // (row_count + 1)),
            max(1, shape[1] // (2 * long_count)),
        )
        if spacing_yx is None
        else _pair(spacing_yx, "spacing_yx")
    )
    span_y = (row_count - 1) * spacing[0]
    span_x = 2 * (long_count - 1) * spacing[1]
    if span_y >= shape[0] or span_x >= shape[1]:
        raise ValueError("checkerboard does not fit inside target shape")
    rows = (shape[0] - span_y) // 2 + spacing[0] * np.arange(row_count)
    long_columns = (
        (shape[1] - span_x) // 2
        + 2 * spacing[1] * np.arange(long_count)
    )
    short_columns = long_columns[:-1] + spacing[1]
    target = np.zeros(shape, dtype=np.float32)
    level = _scalar(intensity, "intensity", nonnegative=True)
    for index, row in enumerate(rows):
        target[row, long_columns if index % 2 == 0 else short_columns] = level
    return validate_target(target)

def preset_gaussian(
    shape_yx: object,
    radius_yx: object,
    *,
    intensity: float = 1.0,
) -> np.ndarray:
    shape = _pair(shape_yx, "shape_yx")
    radius = _float_pair(radius_yx, "radius_yx")
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    exponent = -2.0 * (
        ((yy - shape[0] // 2) / radius[0]) ** 2
        + ((xx - shape[1] // 2) / radius[1]) ** 2
    )
    profile = np.exp(exponent)
    profile[exponent < -8.0] = 0.0
    return validate_target(
        _scalar(intensity, "intensity", nonnegative=True) * profile
    )

def preset_flat_top(
    shape_yx: object,
    radius_yx: object,
    *,
    intensity: float = 1.0,
    edge: float = 0.0,
) -> np.ndarray:
    shape = _pair(shape_yx, "shape_yx")
    radius = _float_pair(radius_yx, "radius_yx")
    if any(2.0 * radius[index] > shape[index] for index in range(2)):
        raise ValueError("flat top does not fit inside target shape")
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    radial = np.sqrt(
        ((yy - shape[0] // 2) / radius[0]) ** 2
        + ((xx - shape[1] // 2) / radius[1]) ** 2
    )
    width = _scalar(edge, "edge", nonnegative=True)
    if width == 0.0:
        profile = (radial <= 1.0).astype(np.float32)
    else:
        distance = (1.0 - radial) * min(radius)
        profile = np.clip(0.5 + distance / width, 0.0, 1.0).astype(np.float32)
    return validate_target(
        _scalar(intensity, "intensity", nonnegative=True) * profile
    )

_WINDOWS_CJK_FONTS = (
    "msyh.ttc",
    "msyhbd.ttc",
    "simhei.ttf",
    "Dengb.ttf",
    "Deng.ttf",
    "simsun.ttc",
)

def _missing_font_characters(font_path: Path, text: str) -> tuple[str, ...]:
    from PIL import ImageFont  # noqa: PLC0415

    font = ImageFont.truetype(
        str(font_path),
        size=64,
        layout_engine=ImageFont.Layout.BASIC,
    )
    notdef = font.getmask("\U0010ffff", mode="L")
    notdef_key = notdef.size, bytes(notdef)
    missing: list[str] = []
    for character in text:
        if character == " ":
            continue
        glyph = font.getmask(character, mode="L")
        if (glyph.size, bytes(glyph)) == notdef_key and character not in missing:
            missing.append(character)
    return tuple(missing)

def _text_font_path(font_path: str | Path | None, text: str) -> Path:
    if font_path is not None:
        selected = Path(font_path)
        if not selected.is_file():
            raise FileNotFoundError(f"text font does not exist: {selected}")
        try:
            missing = _missing_font_characters(selected, text)
        except OSError as error:
            raise ValueError(f"text font could not be loaded: {selected}") from error
        if missing:
            raise ValueError(
                f"text font does not contain requested characters "
                f"{''.join(missing)!r}: {selected}"
            )
        return selected
    font_root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    found = False
    for name in _WINDOWS_CJK_FONTS:
        candidate = font_root / name
        if not candidate.is_file():
            continue
        found = True
        try:
            missing = _missing_font_characters(candidate, text)
        except OSError:
            continue
        if not missing:
            return candidate
    if found:
        raise ValueError(
            "no supported Windows CJK font contains every requested character"
        )
    raise FileNotFoundError(
        "no supported Windows CJK font was found "
        f"({', '.join(_WINDOWS_CJK_FONTS)})"
    )

def _allowed_text_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        character == " "
        or "A" <= character <= "Z"
        or "a" <= character <= "z"
        or "0" <= character <= "9"
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )

def _rasterized_text(text: str, font_path: Path, size: int) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    font = ImageFont.truetype(
        str(font_path),
        size=size,
        layout_engine=ImageFont.Layout.BASIC,
    )
    glyphs: list[tuple[str, int, tuple[int, int, int, int]]] = []
    visible_index = 0
    for index, character in enumerate(text):
        if character == " ":
            continue
        box = font.getbbox(character, anchor="ls")
        origin = int(round(
            font.getlength(text[:index + 1]) - font.getlength(character)
        )) + visible_index
        glyphs.append((character, origin, box))
        visible_index += 1
    if not glyphs:
        return np.zeros((0, 0), dtype=bool)

    left = min(origin + box[0] for _, origin, box in glyphs)
    right = max(origin + box[2] for _, origin, box in glyphs)
    top = min(box[1] for _, _, box in glyphs)
    bottom = max(box[3] for _, _, box in glyphs)
    image = Image.new("L", (max(1, right - left), max(1, bottom - top)), 0)
    draw = ImageDraw.Draw(image)
    for character, origin, box in glyphs:
        glyph_image = Image.new(
            "L", (max(1, box[2] - box[0]), max(1, box[3] - box[1])), 0
        )
        ImageDraw.Draw(glyph_image).text(
            (-box[0], -box[1]),
            character,
            fill=255,
            font=font,
            anchor="ls",
        )
        if not np.any(np.asarray(glyph_image, dtype=np.uint8) >= 128):
            return np.zeros((0, 0), dtype=bool)
        draw.text(
            (origin - left, -top),
            character,
            fill=255,
            font=font,
            anchor="ls",
        )
    mask = np.asarray(image, dtype=np.uint8) >= 128
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[
        rows.min():rows.max() + 1,
        columns.min():columns.max() + 1,
    ]

def preset_text(
    shape_yx: object,
    text: object,
    *,
    spacing: int,
    atom_budget: int,
    intensity: float = 1.0,
    font_path: str | Path | None = None,
) -> np.ndarray:
    """Rasterize one centered line of Latin/CJK text into discrete sites."""

    import unicodedata  # noqa: PLC0415

    shape = _pair(shape_yx, "shape_yx")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFC", text).strip(" ")
    if not normalized:
        raise ValueError("text must contain at least one visible character")
    if any(not _allowed_text_character(character) for character in normalized):
        raise ValueError(
            "text may contain only ASCII letters, digits, CJK characters, "
            "and ordinary spaces"
        )
    try:
        pitch = int(spacing)
    except (TypeError, ValueError) as error:
        raise TypeError("spacing must be a positive integer") from error
    if isinstance(spacing, bool) or pitch != spacing or pitch <= 0:
        raise ValueError("spacing must be a positive integer")
    try:
        budget = int(atom_budget)
    except (TypeError, ValueError) as error:
        raise TypeError("atom_budget must be a positive integer") from error
    if isinstance(atom_budget, bool) or budget != atom_budget or budget <= 0:
        raise ValueError("atom_budget must be a positive integer")
    selected_font = _text_font_path(font_path, normalized)
    logical_shape = (
        (shape[0] - 1) // pitch + 1,
        (shape[1] - 1) // pitch + 1,
    )
    best: np.ndarray | None = None
    for size in range(1, 2 * logical_shape[0] + 2):
        try:
            mask = _rasterized_text(normalized, selected_font, size)
        except OSError as error:
            raise ValueError(
                f"text font could not be loaded: {selected_font}"
            ) from error
        if mask.size == 0:
            continue
        if mask.shape[0] > logical_shape[0] or mask.shape[1] > logical_shape[1]:
            break
        if int(np.count_nonzero(mask)) > budget:
            continue
        best = mask
    if best is None:
        raise ValueError("text does not fit the target shape and atom budget")

    rows, columns = np.nonzero(best)
    span_y = (best.shape[0] - 1) * pitch
    span_x = (best.shape[1] - 1) * pitch
    rows = (shape[0] - span_y) // 2 + pitch * rows
    columns = (shape[1] - span_x) // 2 + pitch * columns
    target = np.zeros(shape, dtype=np.float32)
    target[rows, columns] = _scalar(
        intensity, "intensity", nonnegative=True
    )
    return validate_target(target)

def imported_target(values: object) -> np.ndarray:
    target = validate_target(values)
    peak = float(np.max(target))
    if peak <= 0.0:
        raise ValueError("imported target must contain positive intensity")
    return _readonly(target / peak)

#: The default pupil and its shifted plane, per shape: a pure function of
#: the shape, built once.
_DEFAULT_PUPILS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


#: Elementwise passes over the full 1024x1272 plane (arctan2, the phase
#: projection) dominate a solve's fixed cost, and numpy ufuncs release the
#: GIL on arrays this large -- so row stripes on a small shared pool run
#: them in parallel with bit-identical results (the same libm call per
#: element).  Planes below the threshold run inline.
_STRIPE_THRESHOLD = 1 << 18
_STRIPE_COUNT = 4
#: One pool for every solve, made here rather than on first use: two solves
#: at once (the Editor's and a Feedback run's) each found none and each made
#: one.  An executor starts no thread until work is submitted to it.
_STRIPE_POOL = ThreadPoolExecutor(
    max_workers=_STRIPE_COUNT,
    thread_name_prefix="slm-solver-stripe",
)


def _stripes(operation: Callable[[slice], None], rows: int, size: int) -> None:
    if size < _STRIPE_THRESHOLD or rows < _STRIPE_COUNT:
        operation(slice(0, rows))
        return
    step = (rows + _STRIPE_COUNT - 1) // _STRIPE_COUNT
    futures = [
        _STRIPE_POOL.submit(operation, slice(start, min(rows, start + step)))
        for start in range(0, rows, step)
    ]
    for future in futures:
        future.result()


def _frozen(values: np.ndarray) -> np.ndarray:
    """Freeze a solver-owned fresh array in place -- no copy, no retyping."""

    values.setflags(write=False)
    return values


def _pupil(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.ogrid[-1.0:1.0:shape[0] * 1j, -1.0:1.0:shape[1] * 1j]
    return (xx * xx + yy * yy <= 0.9**2).astype(np.float32)

def _unit_phase(values: np.ndarray, epsilon: float) -> np.ndarray:
    magnitude = np.abs(values).astype(np.float32, copy=False)
    return np.divide(
        values,
        magnitude,
        out=np.ones_like(values, dtype=np.complex64),
        where=magnitude > epsilon,
    )

#: The spots solve stops once the simulated support intensity, divided by the
#: desired intensity site by site, has a max/min no larger than this.  One
#: owner for the number: the loop's early-stop gate and the re-check of the
#: canonical phase both read it, and a caller that needs a tighter answer
#: passes ``support_tolerance`` to ``solve_phase`` instead of editing here.
SPOT_SUPPORT_TOLERANCE = 1.01
SPOT_BACKGROUND_TOLERANCE = .05


def _support_intensity_ratio(
    magnitude: np.ndarray,
    desired: np.ndarray,
    epsilon: float,
) -> float:
    relative = np.square(magnitude, dtype=np.float32) / desired
    return float(np.max(relative) / max(float(np.min(relative)), epsilon))


def _image_metrics(
    far_field: np.ndarray,
    desired: np.ndarray,
    support: np.ndarray,
    epsilon: float,
) -> tuple[float, float, float, float]:
    power = np.square(np.abs(far_field), dtype=np.float32)
    expected = desired[support]
    relative = power[support] / np.maximum(expected, epsilon)
    relative /= max(
        float(np.sum(relative * expected) / np.sum(expected)), epsilon
    )
    relative_rms = float(
        np.sqrt(
            np.sum(expected * np.square(relative - 1.0)) / np.sum(expected)
        )
    )
    relative_image = np.zeros(desired.shape, dtype=np.float32)
    relative_image[support] = relative
    differences = []
    vertical = support[1:] & support[:-1]
    horizontal = support[:, 1:] & support[:, :-1]
    if np.any(vertical):
        differences.append((relative_image[1:] - relative_image[:-1])[vertical])
    if np.any(horizontal):
        differences.append((relative_image[:, 1:] - relative_image[:, :-1])[horizontal])
    roughness = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.concatenate(differences)
                    if differences
                    else np.zeros(1, dtype=np.float32)
                )
            )
        )
    )
    background = float(
        np.sum(power[~support]) / max(float(np.sum(power)), epsilon)
    )
    return (
        relative_rms,
        roughness,
        background,
        relative_rms + 0.05 * roughness + 0.10 * background,
    )

def _project_field(
    back: np.ndarray,
    pupil: np.ndarray,
    magnitude: np.ndarray,
    mask: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """``pupil * _unit_phase(back)`` composed in caller-owned buffers.

    This projection runs once per iteration over the full SLM plane, and the
    naive expression allocated three plane-sized temporaries per pass --
    measured at roughly 40% of a whole spots solve.  Same arithmetic, same
    order; ``back`` is consumed as scratch.
    """

    epsilon = np.float32(np.finfo(np.float32).eps)

    def stripe(rows: slice) -> None:
        np.abs(back[rows], out=magnitude[rows])
        np.greater(magnitude[rows], epsilon, out=mask[rows])
        np.divide(back[rows], magnitude[rows], out=back[rows], where=mask[rows])
        np.logical_not(mask[rows], out=mask[rows])
        np.copyto(back[rows], np.complex64(1.0), where=mask[rows])
        np.multiply(back[rows], pupil[rows], out=out[rows])

    _stripes(stripe, back.shape[0], back.size)
    return out


def _radial_transport_seed(
    desired: np.ndarray, pupil: np.ndarray
) -> np.ndarray | None:
    """Radial transport initial phase for a near-isotropic image target.

    For a target whose energy is closer to circular than elliptical, the
    exact 1-D radial energy match (annulus by annulus) beats the separable
    marginal construction, whose product-form caustic is only right for
    separable targets.  Returns None for strongly anisotropic targets.
    """

    shape = desired.shape
    height, width = shape
    target = desired.astype(np.float64)
    total = float(target.sum())
    if total <= 0.0:
        return None
    yy, xx = np.ogrid[:height, :width]
    cy = float((target.sum(axis=1) * np.arange(height)).sum() / total)
    cx = float((target.sum(axis=0) * np.arange(width)).sum() / total)
    var_y = float((target.sum(axis=1) * (np.arange(height) - cy) ** 2).sum() / total)
    var_x = float((target.sum(axis=0) * (np.arange(width) - cx) ** 2).sum() / total)
    if min(var_y, var_x) <= 0.0:
        return None
    # A separable target (a Gaussian, any product form) is served exactly by
    # the separable marginal construction -- the radial build only wins for
    # genuinely round, non-product shapes such as a flat-top disk.
    marginal_y = target.sum(axis=1)
    marginal_x = target.sum(axis=0)
    product = np.outer(marginal_y, marginal_x) / total
    product_error = float(np.abs(product - target).sum()) / total
    if product_error < 0.05:
        return None
    anisotropy = max(var_y / var_x, var_x / var_y) ** 0.5
    if anisotropy > 1.05:
        return None
    power = pupil.astype(np.float64)
    power = power * power
    source_total = float(power.sum())
    if source_total <= 0.0:
        return None
    sy = float((power.sum(axis=1) * np.arange(height)).sum() / source_total)
    sx = float((power.sum(axis=0) * np.arange(width)).sum() / source_total)
    source_rho = np.hypot(yy - sy, xx - sx)
    target_rho = np.hypot(yy - cy, xx - cx)
    bins = int(np.ceil(max(source_rho.max(), target_rho.max()))) + 1
    source_hist = np.bincount(
        source_rho.astype(np.int64).ravel(), weights=power.ravel(), minlength=bins
    )
    target_hist = np.bincount(
        target_rho.astype(np.int64).ravel(), weights=target.ravel(), minlength=bins
    )
    cumulative_source = np.cumsum(source_hist) / source_total
    cumulative_target = np.cumsum(target_hist) / total
    radius = np.arange(bins, dtype=np.float64)
    mapped = np.interp(cumulative_source, cumulative_target, radius)
    # A radial potential integrates the matched slope; the anisotropic FFT
    # pixel pitch is folded in through the geometric-mean extent, which is
    # exact for a square raster and a seed-grade approximation otherwise.
    scale = 2.0 * np.pi / float(np.sqrt(height * width))
    profile = np.cumsum(mapped) * scale
    profile -= profile[0]
    seed = np.interp(source_rho.ravel(), radius, profile).reshape(shape)
    seed += 2.0 * np.pi * (
        (cy - height / 2.0) * (yy - sy) / height
        + (cx - width / 2.0) * (xx - sx) / width
    )
    return seed.astype(np.float32)


def _mapping_seed(desired: np.ndarray, pupil: np.ndarray) -> np.ndarray:
    """Separable geometric-mapping initial phase for an image target.

    Marginal energy matching between the pupil illumination and the target
    intensity gives, per axis, the ray mapping u -> R(u); integrating the
    matched linear-phase slope yields a caustic seed that starts MRAF near
    the transport solution instead of a fixed quadratic guess.  The seed
    only decides where iteration starts; the quality metrics, the stop
    criteria and the returned contract are untouched.
    """

    shape = desired.shape
    seed = np.zeros(shape, dtype=np.float32)
    power = pupil.astype(np.float64)
    np.square(power, out=power)
    target = desired.astype(np.float64)
    for axis in (0, 1):
        other = 1 - axis
        source_marginal = power.sum(axis=other)
        target_marginal = target.sum(axis=other)
        n = shape[axis]
        source_total = float(source_marginal.sum())
        target_total = float(target_marginal.sum())
        if source_total <= 0.0 or target_total <= 0.0:
            continue
        cumulative_source = np.cumsum(source_marginal) / source_total
        cumulative_target = np.cumsum(target_marginal) / target_total
        mapped = np.interp(
            cumulative_source,
            cumulative_target,
            np.arange(n, dtype=np.float64),
        )
        slope = 2.0 * np.pi * (mapped - n / 2.0) / n
        profile = np.cumsum(slope)
        profile -= profile[n // 2]
        if axis == 0:
            seed += profile.astype(np.float32)[:, None]
        else:
            seed += profile.astype(np.float32)[None, :]
    return seed


def _canonical_unshifted_phase(field: np.ndarray) -> np.ndarray:
    phase = np.empty(field.shape, dtype=np.float32)

    def stripe(rows: slice) -> None:
        part = field[rows]
        np.arctan2(part.imag, part.real, out=phase[rows])
        np.add(
            phase[rows],
            np.float32(2.0 * np.pi),
            out=phase[rows],
            where=phase[rows] < 0.0,
        )
        np.minimum(
            phase[rows],
            np.nextafter(np.float32(2.0 * np.pi), np.float32(0.0)),
            out=phase[rows],
        )

    _stripes(stripe, field.shape[0], field.size)
    return phase

def _cartesian_support(
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    site_rows, site_columns = np.nonzero(support)
    if site_rows.size > 256:
        return None
    rows = np.unique(site_rows)
    columns = np.unique(site_columns)
    # The exact separable transform forms the selected row/column envelope.
    # Beyond these small bounds a full FFT is both simpler and cheaper; in
    # particular, 256 unrelated diagonal sites must not create a 256x256 DFT.
    if columns.size > 64 or rows.size * columns.size > 4096:
        return None
    active = support[np.ix_(rows, columns)]
    return rows, columns, active


def _spot_background_projection(field, pupil, support, amplitude, weights, *,
                                iterations, minimum_iterations, tolerance, reference,
                                restore_scale, stop_requested):
    """Weighted GS with full-field background and natural main-lobe bounds."""
    from scipy import fft  # noqa: PLC0415

    shape = np.asarray(field.shape)
    psf = np.square(abs(fft.fft2(pupil, norm="ortho", workers=4)), dtype=np.float32)
    psf /= psf[0, 0]
    centered = fft.fftshift(psf)
    offsets = np.argwhere(_psf_main_lobe(centered, tuple(shape // 2))) - shape // 2
    psf_values = psf[tuple((offsets % shape).T)]
    profile = np.zeros(field.shape, np.float32)
    for point, weight in zip(np.argwhere(support), amplitude * amplitude, strict=True):
        positions = (point + offsets) % shape
        np.maximum.at(profile, tuple(positions.T), weight * psf_values)
    inside, outside = profile > 0, profile == 0
    magnitude = np.empty(field.shape, np.float32)
    mask = np.empty(field.shape, bool)
    cap = np.empty(field.shape, np.float32)
    scratch = np.empty(field.shape, np.complex64)
    candidate_field = np.empty(field.shape, np.complex64)
    far = fft.fft2(field, norm="ortho", workers=4)
    phase = _unit_phase(far[support], np.finfo(np.float32).eps)
    if reference is None:
        reference = float(np.min(abs(far[support].astype(np.complex128)) ** 2))
    updates = 0
    if restore_scale is not None and iterations:
        far[support] = restore_scale * weights * phase
        field = _project_field(fft.ifft2(far, norm="ortho", workers=4), pupil, magnitude, mask, scratch)
        far = fft.fft2(field, norm="ortho", workers=4)
        updates = 1

    def measure(spectrum):
        nonlocal phase
        bright = spectrum[support].astype(np.complex128)
        # A static target prescribes intensity, not a transport phase path.
        # Preserve this iterate's actual phase while correcting its amplitude.
        phase = _unit_phase(bright, np.finfo(np.float32).eps).astype(np.complex64)
        scale = float(np.sum(abs(bright) * amplitude) / np.sum(amplitude.astype(float) ** 2))
        np.multiply(profile, np.float32(scale * scale), out=cap)
        np.maximum(cap, np.float32(.005 * reference), out=cap)
        np.absolute(spectrum, out=magnitude)
        np.square(magnitude, out=magnitude)
        relative = abs(bright) ** 2 / amplitude.astype(float) ** 2
        background = float(np.max(magnitude, where=outside, initial=0) / reference)
        np.divide(magnitude, cap, out=magnitude)
        envelope = float(np.max(magnitude, where=inside, initial=0))
        ratio = float(relative.max() / relative.min())
        return ratio, background, envelope, scale

    while True:
        if stop_requested is not None and stop_requested():
            raise InterruptedError("SLM phase solve stopped")
        ratio, background, envelope, scale = measure(far)
        converged = ratio <= tolerance and background <= .01
        if (converged and updates >= minimum_iterations) or updates == iterations:
            result = freeze_pattern_phase(fft.fftshift(_canonical_unshifted_phase(field)), tuple(shape))
            np.multiply(fft.ifftshift(result), np.complex64(1j), out=candidate_field)
            np.exp(candidate_field, out=candidate_field)
            candidate_field *= pupil
            candidate_far = fft.fft2(candidate_field, norm="ortho", workers=4)
            ratio, background, envelope, scale = measure(candidate_far)
            converged = ratio <= tolerance and background <= .01
            if converged or updates == iterations:
                return result, candidate_far[support], phase, updates, {
                    "converged": bool(converged), "support_intensity_ratio": ratio,
                    "background_intensity_ratio": background,
                    "main_lobe_envelope_ratio": envelope,
                    "background_reference_intensity": reference,
                }
            field, far = candidate_field, candidate_far
        # Retain WGS compensation: desired trap weights and the spectrum
        # amplitudes needed to produce them are not interchangeable.
        measured = abs(far[support]).astype(np.float32)
        measured /= max(float(np.linalg.norm(measured)), np.finfo(np.float32).eps)
        weights *= np.clip(amplitude / np.maximum(measured, np.finfo(np.float32).eps), .2, 5) ** np.float32(.8)
        weights /= np.linalg.norm(weights)
        np.absolute(far, out=magnitude)
        np.maximum(magnitude, np.float32(1e-20), out=magnitude)
        np.sqrt(cap, out=cap)
        np.divide(cap, magnitude, out=cap)
        np.minimum(cap, np.float32(1), out=cap)
        far *= cap
        far[support] = scale * weights * phase
        field = _project_field(fft.ifft2(far, norm="ortho", workers=4), pupil, magnitude, mask, scratch)
        far = fft.fft2(field, norm="ortho", workers=4)
        updates += 1


def solve_phase(
    target: object,
    *,
    pupil_amplitude: object | None = None,
    spot_optimizer_state: dict[str, object] | None = None,
    initial_phase: object | None = None,
    objective_kind: str = "auto",
    iterations: int | None = None,
    seed: int = 0,
    stop_requested: Callable[[], bool] | None = None,
    support_tolerance: float = SPOT_SUPPORT_TOLERANCE,
    minimum_iterations: int = 1,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve one target, optionally reusing caller-owned transient spot state.

    The caller clears that state whenever the authored input pupil changes.

    ``support_tolerance`` and ``minimum_iterations`` govern the spots
    early-stop gate only: the solve may declare itself done no earlier than
    ``minimum_iterations`` passes and only once the support max/min intensity
    ratio is within ``support_tolerance``.  A feedback loop that re-solves
    every candidate from a hot start needs both, because the default 1%
    gate let a 1-2 iteration solve leave a fresh ~0.2% rms intensity error
    pattern on the sites each candidate -- three times the loop's own
    per-step correction, injected as noise the controller then chased.
    """

    # The transforms are reached here, not at the top of this module.  This
    # file declares the SLM's targets, presets and file formats as well as
    # solving for phase, and a logic node's descriptor names one of those
    # file readers as its artifact codec -- so discovering the nodes, which
    # a task console does before it draws anything, used to import scipy.
    # That was half a second of every console's open for a transform that
    # runs when somebody actually solves a pattern.
    from scipy import fft, ndimage  # noqa: PLC0415

    tolerance = float(support_tolerance)
    if not np.isfinite(tolerance) or tolerance < 1.0:
        raise ValueError("support_tolerance must be a finite ratio >= 1")
    if isinstance(minimum_iterations, bool) or int(minimum_iterations) < 1:
        raise ValueError("minimum_iterations must be a positive integer")
    minimum_passes = int(minimum_iterations)
    desired = validate_target(target)
    if float(np.max(desired)) <= 0.0:
        raise ValueError("target must contain positive intensity")
    if spot_optimizer_state is not None and not isinstance(
        spot_optimizer_state, dict
    ):
        raise TypeError("spot_optimizer_state must be a dict or None")
    saved_state = dict(spot_optimizer_state) if spot_optimizer_state else None
    state_requested = spot_optimizer_state is not None
    seed_value = int(seed)
    if pupil_amplitude is None:
        cached_default = _DEFAULT_PUPILS.get(desired.shape)
        if cached_default is None:
            pupil = _frozen(_pupil(desired.shape))
            cached_default = (pupil, _frozen(fft.ifftshift(pupil)))
            _DEFAULT_PUPILS[desired.shape] = cached_default
        pupil, pupil_unshifted = cached_default
        pupil_source = "default"
    else:
        pupil_source = "provided"
        try:
            pupil = np.asarray(pupil_amplitude, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise TypeError("pupil_amplitude must be a numeric array") from error
        if pupil.shape != desired.shape:
            raise ValueError("pupil_amplitude shape must match the target shape")
        if not np.all(np.isfinite(pupil)):
            raise ValueError("pupil_amplitude must be finite")
        if np.any(pupil < 0.0):
            raise ValueError("pupil_amplitude must be non-negative")
        if not np.any(pupil > 0.0):
            raise ValueError("pupil_amplitude must contain positive amplitude")
        pupil = _readonly(pupil)
        pupil_unshifted = _frozen(fft.ifftshift(pupil))
    if not isinstance(objective_kind, str) or objective_kind not in {
        "auto",
        "spots",
        "image",
    }:
        raise ValueError("objective_kind must be 'auto', 'spots', or 'image'")
    if stop_requested is not None and not callable(stop_requested):
        raise TypeError("stop_requested must be callable or None")
    support = desired > 0.0
    if objective_kind == "auto":
        labels, components = ndimage.label(support)
        sizes = np.bincount(labels.ravel(), minlength=components + 1)[1:]
        largest = int(np.max(sizes, initial=0))
        continuous_component = max(64, int(np.ceil(0.001 * desired.size)))
        resolved_kind = (
            "image"
            if (
                largest > continuous_component
                or np.count_nonzero(support) > 0.02 * desired.size
            )
            else "spots"
        )
    else:
        resolved_kind = objective_kind
    method = "wgs-kim" if resolved_kind == "spots" else "mraf"
    if iterations is None:
        count = 128 if method == "wgs-kim" else 300
    else:
        if isinstance(iterations, bool) or int(iterations) <= 0:
            raise ValueError("iterations must be a positive integer or None")
        count = int(iterations)

    desired_unshifted = _frozen(fft.ifftshift(desired))
    support_unshifted = _frozen(desired_unshifted > 0.0)
    epsilon = np.finfo(np.float32).eps
    # One set of plane-sized scratch buffers for the whole solve: the
    # per-iteration projection reuses them instead of allocating ~30 MB of
    # temporaries per pass.
    plane_magnitude = np.empty(desired.shape, dtype=np.float32)
    plane_mask = np.empty(desired.shape, dtype=np.bool_)
    field_buffer = np.empty(desired.shape, dtype=np.complex64)
    plane_scratch = np.empty(desired.shape, dtype=np.complex64)

    transform = "fft"
    early_stopped = False
    stop_was_interior = False
    iterations_run = 0
    hot_start_used = False
    checked_result: np.ndarray | None = None
    checked_selected: np.ndarray | None = None
    state_status = "not-requested" if not state_requested else "created"
    support_yx: list[list[int]] | None = None
    fixed_phase: np.ndarray | None = None
    background_reference = None
    restore_scale = None
    spot_quality: dict[str, object] = {}
    weights: np.ndarray
    if method == "wgs-kim":
        cartesian = _cartesian_support(support_unshifted)
        if cartesian is None:
            desired_spots = desired_unshifted[support_unshifted]
            constrained = np.zeros(desired.shape, dtype=np.complex64)
            support_yx = ((np.argwhere(support_unshifted) + np.asarray(desired.shape) // 2) % desired.shape).tolist()
        else:
            rows, columns, active = cartesian
            desired_spots = desired_unshifted[np.ix_(rows, columns)][active]
            row_angles = (
                (-2.0 * np.pi / desired.shape[0])
                * rows.astype(np.float64)[:, None]
                * np.arange(desired.shape[0], dtype=np.float64)[None, :]
            )
            row_forward = (
                np.exp(1j * row_angles) / np.sqrt(desired.shape[0])
            ).astype(np.complex64)
            row_backward = np.ascontiguousarray(row_forward.conj().T)
            column_angles = (
                (-2.0 * np.pi / desired.shape[1])
                * columns.astype(np.float64)[:, None]
                * np.arange(desired.shape[1], dtype=np.float64)[None, :]
            )
            column_forward = (
                np.exp(1j * column_angles) / np.sqrt(desired.shape[1])
            ).astype(np.complex64)
            column_forward_transposed = np.ascontiguousarray(column_forward.T)
            column_backward = np.ascontiguousarray(
                column_forward.conj()
            )
            constrained_selected = np.zeros(
                active.shape, dtype=np.complex64
            )
            transform = "selected-dft"
            support_yx = [
                [
                    int((rows[row_index] + desired.shape[0] // 2) % desired.shape[0]),
                    int((columns[column_index] + desired.shape[1] // 2) % desired.shape[1]),
                ]
                for row_index, column_index in np.argwhere(active)
            ]

        amplitude_spots = np.sqrt(desired_spots).astype(
            np.float32, copy=False
        )
        amplitude_spots /= np.linalg.norm(amplitude_spots)

        if saved_state is not None:
            if saved_state.get("objective_kind") != "spots":
                state_status = "objective-changed"
            elif saved_state.get("pupil_source") != pupil_source:
                state_status = "pupil-changed"
            elif saved_state.get("shape_yx") != list(desired.shape):
                state_status = "support-changed"
            else:
                try:
                    saved_support = np.asarray(
                        saved_state["support_yx"], dtype=np.int64
                    )
                    saved_fixed = np.asarray(
                        saved_state["fixed_farfield_phase"], dtype=np.float32
                    )
                    saved_weights = np.asarray(
                        saved_state["site_weights"], dtype=np.float32
                    )
                    saved_amplitudes = np.asarray(
                        saved_state["target_amplitudes"], dtype=np.float32
                    )
                    background_reference = float(saved_state["background_reference_intensity"])
                except (KeyError, TypeError, ValueError):
                    state_status = "invalid"
                else:
                    site_count = len(amplitude_spots)
                    if not np.array_equal(saved_support, support_yx):
                        state_status = "support-changed"
                    elif (
                        saved_fixed.shape != (site_count,)
                        or saved_weights.shape != (site_count,)
                        or saved_amplitudes.shape != (site_count,)
                        or not np.all(np.isfinite(saved_fixed))
                        or not np.all(np.isfinite(saved_weights))
                        or not np.all(np.isfinite(saved_amplitudes))
                        or np.any(saved_weights <= 0.0)
                        or np.any(saved_amplitudes <= 0.0)
                        or not np.isfinite(background_reference) or background_reference <= 0
                    ):
                        state_status = "invalid"
                    else:
                        if initial_phase is None:
                            raise ValueError("reusing spot_optimizer_state requires the actual initial_phase")
                        fixed_phase = np.exp(
                            np.complex64(1j) * saved_fixed
                        ).astype(np.complex64, copy=False)
                        weights = np.array(saved_weights, copy=True)
                        weights *= amplitude_spots / saved_amplitudes
                        weights /= max(float(np.linalg.norm(weights)), epsilon)
                        if stop_requested is not None and stop_requested():
                            raise InterruptedError("SLM phase solve stopped")
                        if not np.array_equal(amplitude_spots, saved_amplitudes):
                            if transform == "selected-dft":
                                constrained_selected.fill(0)
                                constrained_selected[active] = weights * fixed_phase
                                np.matmul(row_backward @ constrained_selected, column_backward, out=plane_scratch)
                            else:
                                constrained.fill(0)
                                constrained[support_unshifted] = weights * fixed_phase
                                plane_scratch[:] = fft.ifft2(constrained, norm="ortho")
                            baseline = _project_field(plane_scratch, pupil_unshifted, plane_magnitude, plane_mask, field_buffer)
                            baseline_bright = fft.fft2(baseline, norm="ortho")[support_unshifted]
                            restore_scale = float(np.sum((baseline_bright * fixed_phase.conj()).real * amplitude_spots))
                        phase = fft.ifftshift(canonical_phase(initial_phase, desired.shape))
                        np.multiply(phase, np.complex64(1j), out=field_buffer)
                        np.exp(field_buffer, out=field_buffer)
                        field_buffer *= pupil_unshifted
                        field = field_buffer
                        hot_start_used = True
                        state_status = "reused"

        if not hot_start_used:
            background_reference = None
            weights = np.array(amplitude_spots, copy=True)
            if initial_phase is None:
                # Start from the desired virtual spots, as the GPU endpoint
                # solver does, rather than a full plane of random speckle.
                seed_phase = np.random.default_rng(seed_value).uniform(-np.pi, np.pi, len(weights)).astype(np.float32)
                seed_values = weights * np.exp(np.complex64(1j) * seed_phase)
                if transform == "selected-dft":
                    constrained_selected.fill(0)
                    constrained_selected[active] = seed_values
                    np.matmul(row_backward @ constrained_selected, column_backward, out=plane_scratch)
                else:
                    constrained.fill(0)
                    constrained[support_unshifted] = seed_values
                    plane_scratch[:] = fft.ifft2(constrained, norm="ortho")
                field = _project_field(plane_scratch, pupil_unshifted, plane_magnitude, plane_mask, field_buffer)
            else:
                phase = np.array(
                    canonical_phase(initial_phase, desired.shape), copy=True
                )
                np.multiply(phase, np.complex64(1j), out=plane_scratch)
                np.exp(plane_scratch, out=plane_scratch)
                plane_scratch *= pupil
                field = fft.ifftshift(plane_scratch)

        selected: np.ndarray | None = None
        while not hot_start_used and iterations_run < count:
            if stop_requested is not None and stop_requested():
                raise InterruptedError("SLM phase solve stopped")
            if selected is None:
                if transform == "selected-dft":
                    selected_grid = (
                        row_forward @ (field @ column_forward_transposed)
                    )
                    selected = selected_grid[active]
                else:
                    far = fft.fft2(field, norm="ortho")
                    selected = far[support_unshifted]
            magnitude = np.abs(selected).astype(np.float32, copy=False)
            measured = magnitude / max(float(np.linalg.norm(magnitude)), epsilon)
            weights *= np.clip(
                amplitude_spots / np.maximum(measured, epsilon), 0.2, 5.0
            ) ** np.float32(0.8)
            weights /= max(float(np.linalg.norm(weights)), epsilon)
            current_phase = _unit_phase(selected, epsilon)
            if fixed_phase is None:
                selected_phase = current_phase
                if iterations_run + 1 == 12:
                    fixed_phase = np.array(current_phase, copy=True)
            else:
                selected_phase = fixed_phase
            constrained_values = weights * selected_phase
            if transform == "selected-dft":
                constrained_selected.fill(0.0)
                constrained_selected[active] = constrained_values
                np.matmul(
                    row_backward @ constrained_selected,
                    column_backward,
                    out=plane_scratch,
                )
                back = plane_scratch
            else:
                constrained.fill(0.0)
                constrained[support_unshifted] = constrained_values
                back = fft.ifft2(constrained, norm="ortho")
            field = _project_field(
                back, pupil_unshifted, plane_magnitude, plane_mask, field_buffer
            )
            iterations_run += 1
            selected = None
            if iterations_run >= 12:
                if transform == "selected-dft":
                    selected = (row_forward @ (field @ column_forward_transposed))[active]
                else:
                    selected = fft.fft2(field, norm="ortho")[support_unshifted]
                if _support_intensity_ratio(abs(selected), desired_spots, epsilon) <= tolerance:
                    break

        checked_result, checked_selected, fixed_phase, updates, spot_quality = _spot_background_projection(
            field, pupil_unshifted, support_unshifted, amplitude_spots, weights,
            iterations=count - iterations_run,
            minimum_iterations=(0 if hot_start_used and minimum_passes == 1
                                and np.array_equal(amplitude_spots, saved_amplitudes)
                                else max(0, minimum_passes - iterations_run)),
            tolerance=tolerance, reference=background_reference, restore_scale=restore_scale,
            stop_requested=stop_requested)
        iterations_run += updates
        early_stopped = bool(spot_quality["converged"] and iterations_run < count)
        background_reference = spot_quality["background_reference_intensity"]
        if iterations is None and not spot_quality["converged"]:
            raise RuntimeError("SLM phase solve did not meet full-field optical constraints: "
                               f"bright ratio {spot_quality['support_intensity_ratio']:.6g} (limit {tolerance:g}), "
                               f"background {spot_quality['background_intensity_ratio']:.6g}, "
                               f"main-lobe envelope {spot_quality['main_lobe_envelope_ratio']:.6g} (diagnostic)")
    else:
        if saved_state is not None:
            state_status = "objective-changed"
        if bool(np.all(support_unshifted)):
            raise ValueError(
                "image objective requires zero-valued pixels defining a noise region"
            )
        amplitude = np.sqrt(desired_unshifted).astype(
            np.float32, copy=False
        )
        amplitude /= np.linalg.norm(amplitude[support_unshifted])
        amplitude_spots = amplitude[support_unshifted]

        def build_field(seed_phase: np.ndarray) -> np.ndarray:
            np.multiply(seed_phase, np.complex64(1j), out=plane_scratch)
            np.exp(plane_scratch, out=plane_scratch)
            np.multiply(plane_scratch, pupil, out=plane_scratch)
            return fft.ifftshift(plane_scratch)

        def mraf_update(
            far: np.ndarray, weights: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            """One MRAF pass from an already-computed far field.

            ``far`` is consumed as scratch and ``weights`` is updated in
            place, exactly as the loop always did.
            """

            selected = far[support_unshifted]
            magnitude = np.abs(selected).astype(np.float32, copy=False)
            measured = magnitude / max(float(np.linalg.norm(magnitude)), epsilon)
            weights *= np.sqrt(
                np.clip(amplitude_spots / np.maximum(measured, epsilon), 0.2, 5.0)
            )
            weights /= max(float(np.linalg.norm(weights)), epsilon)
            current_power = float(np.sum(np.square(magnitude, dtype=np.float32)))
            # ``selected`` was copied out above, so the noise-region scaling
            # can run on ``far`` itself instead of a fresh full plane.
            far *= np.complex64(0.9)
            far[support_unshifted] = (
                weights
                * np.sqrt(max(current_power, epsilon))
                * _unit_phase(selected, epsilon)
            )
            back = fft.ifft2(far, norm="ortho")
            return (
                _project_field(
                    back,
                    pupil_unshifted,
                    plane_magnitude,
                    plane_mask,
                    field_buffer,
                ),
                weights,
            )

        image_seed = "authored"
        coarse_iterations = 0
        multigrid_seeded = False
        if initial_phase is not None:
            phase = np.array(canonical_phase(initial_phase, desired.shape), copy=True)
            field = build_field(phase)
        elif (
            iterations is None
            and min(desired.shape) >= 512
            and desired.shape[0] % 4 == 0
            and desired.shape[1] % 4 == 0
            and int(
                np.count_nonzero(desired >= 0.999 * float(np.max(desired)))
            )
            >= 64
        ):
            # Multigrid only where the interior-uniformity gate can stop the
            # polish: a stagnation-governed target (no flat interior) chases
            # the interpolation artifacts of the lifted seed instead of
            # stopping, and measured slower than solving single-grid.
            # Multigrid: converge the same MRAF at quarter resolution (a
            # sixteenth of the work per pass, through this same function),
            # lift the phase through its cosine/sine planes so wrapping
            # survives interpolation, and polish at full resolution.  The
            # result still ends at the full-resolution gates below.
            factor = 4
            coarse_desired = desired.reshape(
                desired.shape[0] // factor,
                factor,
                desired.shape[1] // factor,
                factor,
            ).mean(axis=(1, 3))
            coarse_pupil = pupil.reshape(
                desired.shape[0] // factor,
                factor,
                desired.shape[1] // factor,
                factor,
            ).mean(axis=(1, 3))
            coarse_phase, coarse_metadata = solve_phase(
                coarse_desired,
                pupil_amplitude=coarse_pupil,
                objective_kind="image",
                seed=seed_value,
                stop_requested=stop_requested,
            )
            coarse_iterations = int(coarse_metadata["iterations_run"])
            zoomed_cos = ndimage.zoom(np.cos(coarse_phase), factor, order=1)
            zoomed_sin = ndimage.zoom(np.sin(coarse_phase), factor, order=1)
            seed_phase = np.arctan2(zoomed_sin, zoomed_cos).astype(np.float32)
            image_seed = f"multigrid({coarse_metadata['image_seed']})"
            multigrid_seeded = True
            field = build_field(seed_phase)
        else:
            # Measured across horizons (12..300 iterations) the mapping seed
            # dominates a quadratic defocus seed (pi * (0.75 x^2 + y^2) over
            # the normalised aperture) on both apodized and hard
            # illumination: same iteration budget, better figure of merit
            # and better interior uniformity every time.  Kept as the
            # measurement it is, rather than as a function that reads like a
            # live alternative.
            radial = _radial_transport_seed(desired, pupil)
            if radial is not None:
                image_seed = "radial-transport"
                field = build_field(radial)
            else:
                image_seed = "mapping"
                field = build_field(_mapping_seed(desired, pupil))
        weights = np.array(amplitude_spots, copy=True)
        minimum_gate_iterations = 8 if multigrid_seeded else 24
        best_fom = float("inf")
        previous_fom = float("inf")
        best_field: np.ndarray | None = None
        stagnant_iterations = 0
        # The flat interior the operator asked for: pixels whose desired
        # intensity is within 0.1% of the peak.  Where that region is a real
        # area, its measured 95th/5th percentile ratio reaching 1% is the
        # image analogue of the spots support gate -- a physical stop that
        # does not depend on how fast the merit happens to be moving.
        strong_interior = desired_unshifted >= 0.999 * float(np.max(desired))
        interior_gate_usable = int(np.count_nonzero(strong_interior)) >= 64
        while iterations_run < count:
            if stop_requested is not None and stop_requested():
                raise InterruptedError("SLM phase solve stopped")
            far = fft.fft2(field, norm="ortho")
            # The stop metrics are a tracker, not the update: sampling them
            # every fourth iteration keeps the same best-so-far intent and
            # the same effective stagnation span (three stale CHECKS covers
            # the twelve iterations the per-iteration count required) at a
            # quarter of their full-plane cost.
            if iterations is None and iterations_run % 4 == 0:
                _relative_rms, _roughness, _background, fom = _image_metrics(
                    far, desired_unshifted, support_unshifted, epsilon
                )
                if best_field is None or fom < best_fom:
                    best_fom = fom
                    best_field = np.array(field, copy=True)
                if np.isfinite(previous_fom) and previous_fom - fom < 1e-4:
                    stagnant_iterations += 1
                else:
                    stagnant_iterations = 0
                previous_fom = fom
                interior_uniform = False
                if interior_gate_usable:
                    interior = np.square(
                        np.abs(far[strong_interior]).astype(
                            np.float32, copy=False
                        )
                    )
                    interior_uniform = float(
                        np.percentile(interior, 95)
                        / max(float(np.percentile(interior, 5)), epsilon)
                    ) <= 1.01
                # A target with a real flat interior converges when THAT
                # region is uniform to 1%; merit stagnation alone must not
                # declare success short of it (a slow tail kept improving the
                # interior well after the merit deltas fell under the
                # threshold).  Targets without such a region keep the
                # stagnation criterion; one that never reaches the gate runs
                # to the bounded iteration cap and says so.
                converged = (
                    interior_uniform
                    if interior_gate_usable
                    else stagnant_iterations >= 3
                )
                if (
                    iterations_run >= minimum_gate_iterations
                    and _relative_rms <= 0.005
                    and converged
                ):
                    if not interior_uniform:
                        field = best_field
                    early_stopped = True
                    stop_was_interior = interior_uniform
                    break
            field, weights = mraf_update(far, weights)
            iterations_run += 1

    if method == "wgs-kim":
        result = checked_result
    else:
        result = canonical_phase(fft.fftshift(np.angle(field)), desired.shape)

    if method == "wgs-kim":
        final_magnitude = np.abs(checked_selected).astype(
            np.float32, copy=False
        )
        total_power = float(np.sum(np.square(pupil, dtype=np.float32)))
    else:
        final_field = pupil_unshifted * np.exp(
            np.complex64(1j) * fft.ifftshift(result)
        ).astype(np.complex64, copy=False)
        final = fft.fft2(final_field, norm="ortho")
        final_magnitude = np.abs(final[support_unshifted]).astype(
            np.float32, copy=False
        )
        total_power = float(np.sum(np.square(np.abs(final), dtype=np.float32)))
    support_ratio = float(spot_quality["support_intensity_ratio"]) if method == "wgs-kim" else None
    measured = np.square(final_magnitude, dtype=np.float32)
    measured /= max(float(np.sum(measured)), epsilon)
    expected = desired_unshifted[support_unshifted]
    expected /= float(np.sum(expected))
    error = float(np.sqrt(np.mean((measured - expected) ** 2)))
    efficiency = float(
        np.sum(np.square(final_magnitude, dtype=np.float32))
        / total_power
    )

    new_state: dict[str, object] = {}
    if (
        method == "wgs-kim"
        and fixed_phase is not None
        and support_yx is not None
    ):
        new_state = {
            "objective_kind": "spots",
            "pupil_source": pupil_source,
            "shape_yx": list(desired.shape),
            "support_yx": support_yx,
            "fixed_farfield_phase": np.angle(fixed_phase).astype(float).tolist(),
            "site_weights": weights.astype(float).tolist(),
            "target_amplitudes": amplitude_spots.astype(float).tolist(),
            "background_reference_intensity": background_reference,
        }
        if state_requested and saved_state is None:
            state_status = "created"
    elif state_requested and saved_state is None:
        state_status = (
            "not-ready"
            if method == "wgs-kim" and transform == "selected-dft"
            else "not-applicable"
        )
    if spot_optimizer_state is not None:
        spot_optimizer_state.clear()
        spot_optimizer_state.update(new_state)

    metadata = {
        "method": method,
        "objective_kind": resolved_kind,
        "pupil_source": pupil_source,
        "optimizer_state_status": state_status,
        "hot_start_used": hot_start_used,
        "transform": transform,
        "iterations": iterations_run,
        "iterations_run": iterations_run,
        "max_iterations": count,
        "early_stopped": early_stopped,
        "stop_reason": (
            "full-field-optical-constraints"
            if early_stopped and method == "wgs-kim"
            else "interior-uniformity"
            if early_stopped and stop_was_interior
            else "fom-stagnation"
            if early_stopped
            else "iteration-limit"
        ),
        "seed": seed_value,
        "rms_intensity_error": error,
        "diffraction_efficiency": efficiency,
    }
    if method == "mraf":
        metadata["image_seed"] = image_seed
        metadata["coarse_iterations"] = coarse_iterations
    if support_ratio is not None:
        metadata.update(spot_quality)
        metadata["quality_transform"] = "full-fft"
        metadata["refinement"] = "weighted-gs-with-bounded-background"
        metadata["phase_encoding"] = "uint16:2pi/65536"
        metadata["support_intensity_ratio"] = support_ratio
        metadata["support_tolerance"] = tolerance
        metadata["minimum_iterations"] = minimum_passes
    else:
        relative_rms, roughness, background, fom = _image_metrics(
            final, desired_unshifted, support_unshifted, epsilon
        )
        metadata.update(
            {
                "signal_pixels": int(np.count_nonzero(support_unshifted)),
                "noise_pixels": int(np.count_nonzero(~support_unshifted)),
                "signal_relative_rms": relative_rms,
                "signal_roughness": roughness,
                "background_power_fraction": background,
                "figure_of_merit": fom,
            }
        )
    return result, metadata

def save_target(
    path: str | Path,
    target: object,
    *,
    objective_kind: str,
) -> Path:
    intensity = validate_target(target)
    return write_readable_json(
        path,
        {
            "format": _TARGET_FORMAT,
            "shape": list(intensity.shape),
            "intensity": intensity.tolist(),
            "objective_kind": _objective_kind(objective_kind),
        },
    )

def load_target(path: str | Path) -> tuple[np.ndarray, str]:
    payload = strict_json_loads(Path(path).read_text(encoding="utf-8"), "target JSON")
    if not isinstance(payload, dict) or set(payload) != _TARGET_KEYS:
        raise ValueError(
            "target JSON fields differ from the strict intensity/objective format"
        )
    if payload["format"] != _TARGET_FORMAT:
        raise ValueError(f"unsupported target JSON format; expected {_TARGET_FORMAT!r}")
    shape = payload["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value < 2 for value in shape)
    ):
        raise ValueError("target JSON shape must be two integer dimensions")
    intensity = payload["intensity"]
    if (
        not isinstance(intensity, list)
        or len(intensity) != shape[0]
        or any(not isinstance(row, list) or len(row) != shape[1] for row in intensity)
        or any(type(value) not in (int, float) for row in intensity for value in row)
    ):
        raise ValueError("target JSON intensity must be a rectangular numeric matrix")
    target = validate_target(intensity)
    if list(target.shape) != shape:
        raise ValueError("target JSON shape differs from intensity")
    return target, _objective_kind(payload["objective_kind"])

def _metadata_json(metadata: Mapping[str, object]) -> str:
    if not isinstance(metadata, Mapping) or any(not isinstance(key, str) for key in metadata):
        raise TypeError("phase metadata must be a string-keyed mapping")
    return json.dumps(dict(metadata), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _json_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{name} must be a string-keyed mapping")
    encoded = _metadata_json(value)
    result = strict_json_loads(encoded, name)
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a JSON object")
    return result


def _system_correction(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    result = _json_object(value, "system correction reference")
    if set(result) != {
        "kind",
        "reference",
        "wavelength_nm",
        "pupil",
        "coordinate_system",
        "valid_region",
        "measurement_method",
    }:
        raise ValueError("system correction reference has the wrong fields")
    if result["kind"] not in _SYSTEM_CORRECTION_KINDS:
        raise ValueError(
            "system correction kind must be pupil_phase_map or target_response_map"
        )
    reference = result["reference"]
    if type(reference) is not str or not reference.strip():
        raise ValueError("system correction reference must be non-empty text")
    wavelength = result["wavelength_nm"]
    if (
        type(wavelength) not in (int, float)
        or not np.isfinite(wavelength)
        or wavelength <= 0
    ):
        raise ValueError("system correction wavelength_nm must be finite and positive")
    for key in ("coordinate_system", "valid_region", "measurement_method"):
        if type(result[key]) is not str or not result[key].strip():
            raise ValueError(f"system correction {key} must be non-empty text")
    return {
        "kind": result["kind"],
        "reference": reference,
        "wavelength_nm": float(wavelength),
        "pupil": _pupil_metadata(result["pupil"]),
        "coordinate_system": result["coordinate_system"],
        "valid_region": result["valid_region"],
        "measurement_method": result["measurement_method"],
    }


def _command_receipt(value: object) -> dict[str, object]:
    result = _json_object(value, "SLM command receipt")
    required = {
        "transport", "identity", "profile", "wavelength_nm", "flip_x",
        "flip_y", "correction_path", "correction_enabled",
        "mapping_revision", "outcome", "command_revision",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"SLM command receipt is missing {sorted(missing)!r}")
    wavelength_policy = {"dvi": True, "usb": True, "virtual": False}
    try:
        requires_wavelength = wavelength_policy[result["transport"]]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "SLM command receipt transport must be dvi, usb, or virtual"
        ) from error
    if result["outcome"] not in {"known-old", "known-new", "unknown"}:
        raise ValueError("SLM command receipt has an invalid outcome")
    for key in ("identity", "profile", "correction_path"):
        if type(result[key]) is not str:
            raise TypeError(f"SLM command receipt {key} must be text")
    for key in ("flip_x", "flip_y", "correction_enabled"):
        if type(result[key]) is not bool:
            raise TypeError(f"SLM command receipt {key} must be bool")
    for key in ("mapping_revision", "command_revision"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"SLM command receipt {key} must be a non-negative int")
    wavelength = result["wavelength_nm"]
    if requires_wavelength and wavelength is None:
        raise ValueError(
            "physical SLM command receipt wavelength_nm must be finite and positive"
        )
    if wavelength is not None and (
        type(wavelength) not in (int, float)
        or not np.isfinite(wavelength)
        or wavelength <= 0
    ):
        raise ValueError("SLM command receipt wavelength_nm is invalid")
    return result


def _pupil_metadata(value: object) -> dict[str, object]:
    result = _json_object(value, "pupil metadata")
    if set(result) != {"enabled", "center_xy", "diameter_xy"}:
        raise ValueError("pupil metadata has the wrong fields")
    if type(result["enabled"]) is not bool:
        raise TypeError("pupil enabled must be bool")
    for key, positive in (("center_xy", False), ("diameter_xy", True)):
        pair = result[key]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(item) not in (int, float) for item in pair)
            or any(not np.isfinite(item) or (positive and item <= 0) for item in pair)
        ):
            raise ValueError(f"pupil {key} must be a finite numeric pair")
    return result


def _operator_metadata(value: object) -> dict[str, object]:
    result = _json_object(value, "operator metadata")
    if set(result) != {
        "enabled", "carrier_waves_xy", "zernike_noll_waves_rms"
    }:
        raise ValueError("operator metadata has the wrong fields")
    carrier = result["carrier_waves_xy"]
    coefficients = result["zernike_noll_waves_rms"]
    if (
        type(result["enabled"]) is not bool
        or not isinstance(carrier, list)
        or len(carrier) != 2
        or any(
            type(item) not in (int, float)
            or not np.isfinite(item)
            or abs(item) > 1000.0
            for item in carrier
        )
        or not isinstance(coefficients, dict)
        or set(coefficients) - _OPERATOR_MODES
        or any(
            type(item) not in (int, float)
            or not np.isfinite(item)
            or abs(item) > 1000.0
            for item in coefficients.values()
        )
    ):
        raise ValueError("operator metadata is invalid")
    return result


def _phase_codes(values: object, shape_yx: tuple[int, int]) -> np.ndarray:
    shape = _pair(shape_yx, "shape_yx")
    source = np.asarray(values)
    if source.shape != shape or source.dtype.kind not in "iuf":
        canonical = canonical_phase(values, shape)
    elif (
        source.dtype == np.dtype("<f4")
        and np.all(np.isfinite(source))
        and np.all(source >= 0.0)
        and np.all(source < 2.0 * np.pi)
    ):
        canonical = source
    else:
        canonical = canonical_phase(source, shape)
    scaled = np.floor(
        canonical * np.float32(_PHASE_CODE_COUNT / (2.0 * np.pi))
        + np.float32(0.5)
    ).astype(np.uint32)
    return np.asarray(scaled & (_PHASE_CODE_COUNT - 1), dtype="<u2")


def freeze_pattern_phase(values: object, shape_yx: tuple[int, int]) -> np.ndarray:
    """Freeze a logical pattern on a uniform 16-bit circular phase grid."""

    shape = _pair(shape_yx, "shape_yx")
    codes = _phase_codes(values, shape)
    return _readonly(
        codes.astype(np.float32)
        * np.float32(2.0 * np.pi / _PHASE_CODE_COUNT)
    )


def _encoded_pattern_phase(values: object, shape: tuple[int, int]) -> np.ndarray:
    codes = _phase_codes(values, shape)
    delta = np.empty(shape, dtype="<u2")
    delta[:, 0] = codes[:, 0]
    np.subtract(codes[:, 1:], codes[:, :-1], out=delta[:, 1:], dtype=np.uint16)
    return delta


def _decoded_pattern_phase(values: object) -> np.ndarray:
    delta = np.asarray(values)
    if delta.dtype != np.dtype("<u2") or delta.ndim != 2 or min(delta.shape) < 2:
        raise ValueError(
            "science context pattern_phase_delta must be a uint16 matrix"
        )
    cumulative = np.add.accumulate(delta, axis=1, dtype=np.uint64)
    codes = np.asarray(cumulative & (_PHASE_CODE_COUNT - 1), dtype="<u2")
    return _readonly(
        codes.astype(np.float32)
        * np.float32(2.0 * np.pi / _PHASE_CODE_COUNT)
    )


def _pupil_geometry(
    shape_yx: tuple[int, int], pupil: Mapping[str, object]
) -> tuple[tuple[int, int], dict[str, object]]:
    shape = _pair(shape_yx, "shape_yx")
    pupil_values = _pupil_metadata(pupil)
    height, width = shape
    center_x, center_y = pupil_values["center_xy"]
    diameter_x, diameter_y = pupil_values["diameter_xy"]
    if not (
        0.0 <= center_x <= width - 1
        and 0.0 <= center_y <= height - 1
        and diameter_x <= 2.0 * width
        and diameter_y <= 2.0 * height
    ):
        raise ValueError("Science Context pupil lies outside SLM limits")
    return shape, pupil_values


def _pupil_coordinates(
    shape_yx: tuple[int, int], pupil: Mapping[str, object]
) -> tuple[tuple[int, int], dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    shape, pupil_values = _pupil_geometry(shape_yx, pupil)
    height, width = shape
    center_x, center_y = pupil_values["center_xy"]
    diameter_x, diameter_y = pupil_values["diameter_xy"]
    yy, xx = np.ogrid[:height, :width]
    zx = (xx - center_x) / (diameter_x / 2.0)
    zy = (yy - center_y) / (diameter_y / 2.0)
    radius_squared = zx * zx + zy * zy
    return shape, pupil_values, zx, zy, radius_squared


def science_pupil_fields(
    shape_yx: tuple[int, int], pupil: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    """Derive the frozen pupil planes from their semantic parameters."""

    shape, pupil_values, _zx, _zy, radius_squared = _pupil_coordinates(
        shape_yx, pupil
    )
    support = np.asarray(radius_squared <= 1.0, dtype=bool)
    amplitude = (
        np.exp(-radius_squared).astype(np.float32)
        if pupil_values["enabled"]
        else np.ones(shape, dtype=np.float32)
    )
    immutable_support = np.frombuffer(
        np.ascontiguousarray(support).tobytes(), dtype=np.bool_
    ).reshape(shape)
    return immutable_support, _readonly(amplitude)


def science_operator_wavefront(
    shape_yx: tuple[int, int],
    pupil: Mapping[str, object],
    operator_metadata: Mapping[str, object],
) -> np.ndarray:
    """Derive the frozen operator plane from pupil and operator parameters."""

    shape, _pupil_values, zx, zy, radius_squared = _pupil_coordinates(
        shape_yx, pupil
    )
    operator = _operator_metadata(operator_metadata)
    support = np.asarray(radius_squared <= 1.0, dtype=bool)
    wavefront = np.zeros(shape, dtype=np.float64)
    if operator["enabled"]:
        height, width = shape
        full_y, full_x = np.ogrid[
            -1.0:1.0:height * 1j, -1.0:1.0:width * 1j
        ]
        carrier_x, carrier_y = operator["carrier_waves_xy"]
        wavefront += np.pi * (carrier_x * full_x + carrier_y * full_y)
        modes = {
            "defocus": lambda: np.sqrt(3.0) * (2.0 * radius_squared - 1.0),
            "astig_oblique": lambda: 2.0 * np.sqrt(6.0) * zx * zy,
            "astig_vertical": lambda: np.sqrt(6.0) * (zx * zx - zy * zy),
            "coma_y": lambda: np.sqrt(8.0) * zy * (
                3.0 * radius_squared - 2.0
            ),
            "coma_x": lambda: np.sqrt(8.0) * zx * (
                3.0 * radius_squared - 2.0
            ),
            "trefoil_y": lambda: np.sqrt(8.0) * zy * (
                3.0 * zx * zx - zy * zy
            ),
            "trefoil_x": lambda: np.sqrt(8.0) * zx * (
                zx * zx - 3.0 * zy * zy
            ),
            "spherical": lambda: np.sqrt(5.0) * (
                6.0 * radius_squared * radius_squared
                - 6.0 * radius_squared
                + 1.0
            ),
        }
        for key, coefficient in operator["zernike_noll_waves_rms"].items():
            if coefficient:
                values = modes[key]()
                wavefront[support] += (
                    2.0 * np.pi * coefficient
                    * np.broadcast_to(values, shape)[support]
                )
    return canonical_phase(wavefront, shape)


def compose_science_phase(
    pattern_phase: object, operator_wavefront: object
) -> np.ndarray:
    pattern = np.asarray(pattern_phase)
    operator = np.asarray(operator_wavefront)
    if pattern.ndim != 2 or operator.shape != pattern.shape:
        raise ValueError("Science Context phase layers differ")
    # Both layers arrive canonical -- a frozen pattern, an operator plane --
    # and canonicalizing a canonical plane returns it unchanged, so the sum
    # is wrapped once rather than each full plane re-validated first.
    return canonical_phase(
        pattern.astype(np.float64) + operator.astype(np.float64),
        tuple(pattern.shape),
    )


def save_science_context(
    path: str | Path,
    pattern_phase: object,
    *,
    target_intensity: object,
    objective_kind: str,
    pupil: Mapping[str, object],
    system_correction: Mapping[str, object] | None,
    command_receipt: Mapping[str, object],
    pattern_metadata: Mapping[str, object],
    operator_metadata: Mapping[str, object],
) -> Path:
    values = np.asarray(pattern_phase)
    if values.ndim != 2:
        raise ValueError("Science Context Pattern must be two-dimensional")
    shape = tuple(values.shape)
    encoded_pattern = _encoded_pattern_phase(values, shape)
    target = validate_target(target_intensity)
    if target.shape != shape:
        raise ValueError("Science Context Target must match the phase shape")
    metadata = {
        "format": _SCIENCE_CONTEXT_FORMAT,
        "objective_kind": _objective_kind(objective_kind),
        "pupil": _pupil_metadata(pupil),
        "system_correction": _system_correction(system_correction),
        "command_receipt": _command_receipt(command_receipt),
        "pattern_metadata": _json_object(pattern_metadata, "Pattern metadata"),
        "operator_metadata": _operator_metadata(operator_metadata),
    }
    _pupil_geometry(shape, metadata["pupil"])
    encoded = _metadata_json(metadata)
    return atomic_write_file(
        path,
        lambda stream: np.savez_compressed(
            stream,
            pattern_phase_delta=encoded_pattern,
            target_intensity=target,
            metadata=np.asarray(encoded),
        ),
    )


def load_science_context(path: str | Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        members = tuple(archive.files)
        if (
            len(members) != len(_SCIENCE_CONTEXT_MEMBERS)
            or set(members) != _SCIENCE_CONTEXT_MEMBERS
        ):
            raise ValueError("science context NPZ has the wrong members")
        encoded_pattern = np.asarray(archive["pattern_phase_delta"])
        target = np.asarray(archive["target_intensity"])
        encoded = np.asarray(archive["metadata"])
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise ValueError("science context metadata must be scalar Unicode JSON")
        metadata = strict_json_loads(str(encoded.item()), "science context metadata")
    if not isinstance(metadata, dict) or set(metadata) != _SCIENCE_CONTEXT_KEYS:
        raise ValueError("science context metadata has the wrong fields")
    if metadata["format"] != _SCIENCE_CONTEXT_FORMAT:
        raise ValueError("unsupported science context format")
    pattern = _decoded_pattern_phase(encoded_pattern)
    shape = tuple(pattern.shape)
    if target.dtype != np.dtype("<f4") or target.shape != shape:
        raise ValueError(
            "science context target intensity must be a matching float32 matrix"
        )
    target = validate_target(target)
    normalized = {
        "objective_kind": _objective_kind(metadata["objective_kind"]),
        "pupil": _pupil_metadata(metadata["pupil"]),
        "system_correction": _system_correction(metadata["system_correction"]),
        "command_receipt": _command_receipt(metadata["command_receipt"]),
        "pattern_metadata": _json_object(
            metadata["pattern_metadata"], "Pattern metadata"
        ),
        "operator_metadata": _operator_metadata(metadata["operator_metadata"]),
    }
    support, amplitude = science_pupil_fields(shape, normalized["pupil"])
    operator = science_operator_wavefront(
        shape, normalized["pupil"], normalized["operator_metadata"]
    )
    phase = compose_science_phase(pattern, operator)
    return {
        "phase": phase,
        "pattern_phase": pattern,
        "operator_wavefront": operator,
        "pupil_amplitude": amplitude,
        "pupil_support": support,
        "target_intensity": target,
        **normalized,
    }


# A Fourier band is a statement about nonzero target coefficients, not a
# smaller SLM. All phase constraints below still run on every physical pixel.
_REARRANGEMENT_PHASE_CODE_CUDA = r'''
__device__ unsigned char phase_code(double command,unsigned int pixel){
 if(command<0)command+=6.2831853071795864769;
 if(command<0)command+=6.2831853071795864769;
 float a=fminf((float)command,6.28318500518798828125f);
 unsigned int p16=((unsigned int)floorf(a*10430.3783504704527f+0.5f))&65535;
 unsigned int hash=pixel+0x9e3779b9u;
 hash^=hash>>16;hash*=0x7feb352du;hash^=hash>>15;hash*=0x846ca68bu;hash^=hash>>16;
 // Exact p16/256 plus the same spatial threshold. A float-radian roundtrip
 // can spuriously turn an exact code 255 into zero at particular pixels.
 unsigned int carry=(((p16&255)<<16)+(hash>>8))>>24;
 return (unsigned char)(((p16>>8)+carry)&255);
}
'''


_REARRANGEMENT_CUDA = _REARRANGEMENT_PHASE_CODE_CUDA + r'''
#include <cuda_fp16.h>
#if HALF
typedef half real_t;
__device__ real_t cv(float a){return __float2half_rn(a);}
#else
typedef float real_t;
__device__ real_t cv(float a){return a;}
#endif
extern "C" __global__ void load_motion_frame(int* counter,const int* native_indices,const int* coarse_indices,
 const double2* frequencies,const long long* offsets,const float2* coefficients,int* native_index,int* coarse_index,
 const float* amplitudes,double2* frequency,float2* current,float* target,int N,int K,int warm_previous,int preserve_phase){
 __shared__ int frame;int t=threadIdx.x;
 if(!t){frame=counter[0];counter[0]=frame+1;}__syncthreads();
 unsigned long long row=(unsigned long long)frame*N;
 for(int j=t;j<N;j+=256){
  native_index[j]=native_indices[row+j];coarse_index[j]=coarse_indices[row+j];target[j]=amplitudes[row+j];
  float2 c=coefficients[row+j];
  if(warm_previous&&frame>0){float previous=amplitudes[row-N+j];
   if(previous>0){float2 prior=coefficients[row-N+j];float scale=target[j]/previous;
    if(preserve_phase){scale*=hypotf(prior.x,prior.y)/fmaxf(hypotf(c.x,c.y),1e-30f);}
    else c=prior;c.x*=scale;c.y*=scale;}}
  current[j]=target[j]>0?c:make_float2(0,0);}
 long long begin=offsets[frame],end=offsets[frame+1];
 for(int j=t;j<K;j+=256)frequency[j]=j<end-begin?frequencies[begin+j]:make_double2(0,0);
}
extern "C" __global__ void store_motion_frame(const int* counter,const unsigned char* codes,
 const float2* coefficients,const float2* actual,unsigned char* movie,float2* saved_coefficients,float2* saved_actual,int N,int area){
 int i=blockIdx.x*blockDim.x+threadIdx.x,frame=counter[0]-1;
 if(i<area)movie[(unsigned long long)frame*area+i]=codes[i];
 if(i<N){unsigned long long row=(unsigned long long)frame*N+i;saved_coefficients[row]=coefficients[i];saved_actual[row]=actual[i];}
}
extern "C" __global__ void phase_step_rms(const unsigned char* movie,const float* initial,
 const float* pupil,float* result,int area,double energy){
 __shared__ double sum[256];int t=threadIdx.x,frame=blockIdx.x;double value=0;
 for(int i=t;i<area;i+=256){
  float before=frame?movie[(unsigned long long)(frame-1)*area+i]*.02454369260617025968f:initial[i];
  float difference=movie[(unsigned long long)frame*area+i]*.02454369260617025968f-before;
  difference=atan2f(sinf(difference),cosf(difference));
  value+=(double)pupil[i]*pupil[i]*difference*difference;
 }
 sum[t]=value;__syncthreads();
 for(int stride=128;stride;stride/=2){if(t<stride)sum[t]+=sum[t+stride];__syncthreads();}
 if(!t)result[frame]=sqrt(sum[0]/energy);
}
extern "C" __global__ void scatter(const float2* c,const int* index,float2* spectrum,int N){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N){atomicAdd(&spectrum[index[i]].x,c[i].x);atomicAdd(&spectrum[index[i]].y,c[i].y);}}
extern "C" __global__ void gather(const float2* spectrum,const int* index,float2* c,int N){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)c[i]=spectrum[index[i]];}
extern "C" __global__ void pack_inverse(const float2* spectrum,const double2* y_roots,real_t* packed,int H,int LY,int K){
 __shared__ float2 tile[32][33];
 int y=blockIdx.x*32+threadIdx.x,k=blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(y<H&&k+j<K){
  float2 v=spectrum[(k+j)*LY+(y-H/2+LY)%LY];double2 root=y_roots[(k+j)*H+y];
  double c=root.x,s=root.y;
  tile[threadIdx.y+j][threadIdx.x]=make_float2(v.x*c-v.y*s,v.x*s+v.y*c);}
 __syncthreads();
 int yy=blockIdx.x*32+threadIdx.y,kk=blockIdx.y*32+threadIdx.x;
 for(int j=0;j<32;j+=8)if(yy+j<H&&kk<K){float2 v=tile[threadIdx.x][threadIdx.y+j];int i=(yy+j)*K+kk;
 packed[i]=cv(v.x);packed[i+H*K]=cv(v.y);}
}
// Roots at opposite centered x coordinates are conjugate even for an asymmetric
// pupil. Reconstruct both physical pixels before applying their own amplitudes.
__device__ __forceinline__ float2 synthesis_pixel(const float* image,int y,int x,int H,int W,int P){
 int d=abs(x-W/2),r=y*2*P+d;float sign=x>=W/2?1.f:-1.f;
 return make_float2(image[r]-sign*image[r+H*2*P+P],image[r+H*2*P]+sign*image[r+P]);
}
__device__ __forceinline__ float2 pupil_projection(float2 field,float amplitude){
 float magnitude=hypotf(field.x,field.y);
 return magnitude>1e-20f?make_float2(field.x*(amplitude/magnitude),field.y*(amplitude/magnitude)):make_float2(amplitude,0);
}
__device__ __forceinline__ void pack_pair(real_t* packed,float2 plus,float2 minus,int y,int d,int H,int P){
 int r=y*2*P+d;
 packed[r]=cv(plus.x+minus.x);packed[r+P]=cv(plus.y-minus.y);
 packed[r+H*2*P]=cv(plus.y+minus.y);packed[r+H*2*P+P]=cv(minus.x-plus.x);
}
extern "C" __global__ void project(const float* image,const float* pupil,real_t* packed,int H,int W,int P){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H*P)return;int d=i%P,y=i/P,xp=W/2+d,xm=W/2-d;
 float2 plus=make_float2(0,0),minus=make_float2(0,0);
 if(xp<W)plus=pupil_projection(synthesis_pixel(image,y,xp,H,W,P),pupil[y*W+xp]);
 if(d>0&&xm>=0)minus=pupil_projection(synthesis_pixel(image,y,xm,H,W,P),pupil[y*W+xm]);
 pack_pair(packed,plus,minus,y,d,H,P);
}
extern "C" __global__ void pack_field(const float2* field,real_t* packed,int H,int W,int P){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H*P)return;int d=i%P,y=i/P,xp=W/2+d,xm=W/2-d;
 float2 plus=xp<W?field[y*W+xp]:make_float2(0,0);
 float2 minus=d>0&&xm>=0?field[y*W+xm]:make_float2(0,0);
 pack_pair(packed,plus,minus,y,d,H,P);
}
extern "C" __global__ void select_roots(const double2* frequencies,real_t* backward,real_t* forward,
 double2* y_roots,int NF,int K,int P,int H,int W){
 __shared__ float2 tile[32][33];
 int x=blockIdx.x*32+threadIdx.x,k=blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(x<P&&k+j<K){
  double s=0,c=0;
  if(k+j<NF&&x<=W/2)sincos(6.2831853071795864769*frequencies[k+j].x*x/W,&s,&c);
  float2 v=make_float2(c,s);tile[threadIdx.y+j][threadIdx.x]=v;
  if(backward){int r=(k+j)*2*P+x;backward[r]=cv(v.x);backward[r+P]=cv(v.y);}
 }
 for(int j=0;j<32;j+=8)if(x<H&&k+j<K){
  double s,c;
  sincos(6.2831853071795864769*frequencies[k+j].y*(x-H/2)/H,&s,&c);
  y_roots[(k+j)*H+x]=make_double2(c,s);
 }
 __syncthreads();
 int xx=blockIdx.x*32+threadIdx.y,kk=blockIdx.y*32+threadIdx.x;
 for(int j=0;j<32;j+=8)if(xx+j<P&&kk<K){float2 v=tile[threadIdx.x][threadIdx.y+j];
  int out=(xx+j)*K+kk;
  forward[out]=cv(v.x);forward[out+P*K]=cv(v.y);}
}
extern "C" __global__ void pack_forward(const float* projected,const double2* y_roots,float2* spectrum,int H,int LY,int K){
 __shared__ float2 tile[32][33];
 int k=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(k<K&&y+j<H){int r=(y+j)*K+k;double2 root=y_roots[k*H+y+j];
  double c=root.x,s=root.y;
  float re=projected[r],im=projected[r+H*K];
  tile[threadIdx.y+j][threadIdx.x]=make_float2(re*c+im*s,im*c-re*s);}
 __syncthreads();
 int yy=blockIdx.y*32+threadIdx.x,kk=blockIdx.x*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(yy<LY&&kk+j<K)
  spectrum[(kk+j)*LY+(yy-H/2+LY)%LY]=tile[threadIdx.x][threadIdx.y+j];
}
__device__ __forceinline__ float2 encoded_pixel(float2 field,const float* pupil,const float* incident,unsigned char* codes,int i){
 float a=atan2f(field.y,field.x),s,c;
 unsigned char code=phase_code((double)a-(double)incident[i],(unsigned int)i);codes[i]=code;
 sincosf(code*.02454369260617025968f+incident[i],&s,&c);
 return make_float2(__fmul_rn(pupil[i],c),__fmul_rn(pupil[i],s));
}
extern "C" __global__ void encode(const float* input,const float* pupil,const float* incident,float2* optical,
 unsigned char* codes,int H,int W,int P){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H*W)return;
 float2 field=P?synthesis_pixel(input,i/W,i%W,H,W,P):((const float2*)input)[i];
 optical[i]=encoded_pixel(field,pupil,incident,codes,i);
}
extern "C" __global__ void encode_pack(const float* image,const float* pupil,const float* incident,real_t* packed,
 unsigned char* codes,int H,int W,int P){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H*P)return;int d=i%P,y=i/P,xp=W/2+d,xm=W/2-d;
 float2 plus=make_float2(0,0),minus=make_float2(0,0);
 if(xp<W)plus=encoded_pixel(synthesis_pixel(image,y,xp,H,W,P),pupil,incident,codes,y*W+xp);
 if(d>0&&xm>=0)minus=encoded_pixel(synthesis_pixel(image,y,xm,H,W,P),pupil,incident,codes,y*W+xm);
 pack_pair(packed,plus,minus,y,d,H,P);
}
extern "C" __global__ void field_project(const unsigned char* input,float2* field,const float* delta,
 const float2* previous_base,const float2* previous_corrected,const float* pupil,const float* incident,
 unsigned char* codes,int H,int W,int P,int mode){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H*W)return;float re,im,s,c;unsigned char code;
 if(mode==0){code=input[i];}
 else if(mode==1){
  float2 e=field[i],a=previous_corrected[i],b=previous_base[i];
  double correction=atan2((double)a.y*b.x-(double)a.x*b.y,(double)a.x*b.x+(double)a.y*b.y);
  code=phase_code((double)atan2f(e.y,e.x)+correction-(double)incident[i],i);
 }
 else{
  float2 e=field[i];re=e.x;im=e.y;
  float2 d=synthesis_pixel(delta,i/W,i%W,H,W,P);re+=d.x/(H*W);im+=d.y/(H*W);
  code=phase_code((double)atan2f(im,re)-(double)incident[i],i);
 }
 codes[i]=code;sincosf(code*.02454369260617025968f+incident[i],&s,&c);
 field[i]=make_float2(pupil[i]*c,pupil[i]*s);
}
__device__ __forceinline__ float weight_ratio(float magnitude,float target,float brightness){
 return fminf(5.f,fmaxf(.2f,brightness*target/fmaxf(magnitude,1e-20f)));
}
extern "C" __global__ void anderson_begin(const float2* c,float2* phase,int* state,int N){
  int t=threadIdx.x;for(int j=t;j<N;j+=256){float2 a=c[j];double m=hypot((double)a.x,(double)a.y);
    phase[j]=m>0?make_float2((float)(a.x/m),(float)(a.y/m)):make_float2(1,0);}
  if(!t){state[0]=0;state[1]=0;}}
__device__ void anderson_apply(const float2* field,const float* target,float2* c,const float2* phase,
  float* gh,float* rh,double* candidate,int* state,int N,float exponent){
  __shared__ double sums[5][256],brightness,mx,mg,gamma0,gamma1,largest,normalizer;
  __shared__ int bad,active;
  int t=threadIdx.x,slot=state[1],count=state[0],prev=(slot+2)%3,older=(slot+1)%3;
  double se=0,sa=0;
  double count_local=0;
  for(int j=t;j<N;j+=256)if(target[j]>0){float2 e=field[j];se+=(double)e.x*e.x+(double)e.y*e.y;sa+=(double)target[j]*target[j];++count_local;}
  sums[0][t]=se;sums[1][t]=sa;sums[2][t]=count_local;__syncthreads();
  for(int k=128;k;k/=2){if(t<k)for(int q=0;q<3;++q)sums[q][t]+=sums[q][t+k];__syncthreads();}
  if(!t){brightness=sqrt(sums[0][0]/fmax(sums[1][0],1e-30));active=(int)sums[2][0];bad=0;}__syncthreads();
  double sx=0,sg=0;
  for(int j=t;j<N;j+=256)if(target[j]>0){float2 a=c[j],e=field[j];
    float x=logf(fmaxf(hypotf(a.x,a.y),1e-20f));
    float gain=exponent*logf(weight_ratio(hypotf(e.x,e.y),target[j],(float)brightness));
    sx+=x;sg+=gain;}
  sums[0][t]=sx;sums[1][t]=sg;__syncthreads();
  for(int k=128;k;k/=2){if(t<k){sums[0][t]+=sums[0][t+k];sums[1][t]+=sums[1][t+k];}__syncthreads();}
  if(!t){mx=sums[0][0]/active;mg=sums[1][0]/active;}__syncthreads();
  for(int j=t;j<N;j+=256){if(target[j]<=0){gh[slot*N+j]=rh[slot*N+j]=0;continue;}float2 a=c[j],e=field[j];
    double x=(double)logf(fmaxf(hypotf(a.x,a.y),1e-20f))-mx;
    double f=(double)(exponent*logf(weight_ratio(hypotf(e.x,e.y),target[j],(float)brightness)))-mg;
    gh[slot*N+j]=(float)(x+f);rh[slot*N+j]=(float)f;}
  __syncthreads();
  double h00=0,h01=0,h11=0,b0=0,b1=0;
  if(count)for(int j=t;j<N;j+=256)if(target[j]>0){double f=rh[slot*N+j],d0=f-rh[prev*N+j];
    double d1=count>1?(double)rh[prev*N+j]-rh[older*N+j]:0;
    h00+=d0*d0;h01+=d0*d1;h11+=d1*d1;b0+=d0*f;b1+=d1*f;}
  sums[0][t]=h00;sums[1][t]=h01;sums[2][t]=h11;sums[3][t]=b0;sums[4][t]=b1;__syncthreads();
  for(int k=128;k;k/=2){if(t<k)for(int q=0;q<5;++q)sums[q][t]+=sums[q][t+k];__syncthreads();}
  if(!t){
    gamma0=0;gamma1=0;
    if(count){double a=sums[0][0]*1.0001,b=sums[1][0],d=sums[2][0]*1.0001;
      if(!isfinite(a)||!isfinite(b)||!isfinite(d)||!isfinite(sums[3][0])||!isfinite(sums[4][0]))bad=1;
      else if(a>0&&d>0){double determinant=a*d-b*b;
        if(determinant>1e-12*a*d&&isfinite(determinant)){
          gamma0=(d*sums[3][0]-b*sums[4][0])/determinant;
          gamma1=(a*sums[4][0]-b*sums[3][0])/determinant;
        }else bad=1;
      }else if(a>0){gamma0=sums[3][0]/a;}
      else if(d>0){gamma1=sums[4][0]/d;}
      else bad=1;
      if(!isfinite(gamma0)||!isfinite(gamma1))bad=1;
    }
    if(bad){gamma0=0;gamma1=0;}
  }__syncthreads();
  for(int j=t;j<N;j+=256){if(target[j]<=0){candidate[j]=0;continue;}double g=gh[slot*N+j],v=g;
    if(count)v-=gamma0*(g-gh[prev*N+j]);
    if(count>1)v-=gamma1*((double)gh[prev*N+j]-gh[older*N+j]);
    candidate[j]=v;if(!isfinite(v))atomicExch(&bad,1);}
  __syncthreads();
  double largest_local=-1.0e300;
  for(int j=t;j<N;j+=256)if(target[j]>0){if(bad)candidate[j]=gh[slot*N+j];largest_local=fmax(largest_local,candidate[j]);}
  sums[0][t]=largest_local;__syncthreads();
  for(int k=128;k;k/=2){if(t<k)sums[0][t]=fmax(sums[0][t],sums[0][t+k]);__syncthreads();}
  if(!t)largest=sums[0][0];__syncthreads();
  double norm=0;for(int j=t;j<N;j+=256)if(target[j]>0){double v=exp(candidate[j]-largest);candidate[j]=v;norm+=v*v;}
  sums[0][t]=norm;__syncthreads();
  for(int k=128;k;k/=2){if(t<k)sums[0][t]+=sums[0][t+k];__syncthreads();}
  if(!t)normalizer=rsqrt(sums[0][0]);__syncthreads();
  for(int j=t;j<N;j+=256){if(target[j]<=0){c[j]=make_float2(0,0);continue;}double a=candidate[j]*normalizer;float2 p=phase[j];c[j]=make_float2((float)(a*p.x),(float)(a*p.y));}
  if(!t){state[0]=bad?1:min(count+1,2);state[1]=(slot+1)%3;}
}
extern "C" __global__ void anderson_update(const float2* field,const float* target,float2* c,const float2* phase,
  float* gh,float* rh,double* candidate,int* state,int N,float exponent){
  anderson_apply(field,target,c,phase,gh,rh,candidate,state,N,exponent);
}
extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(unsigned long long,unsigned int);
extern "C" __global__ void focal_profile(const long long* indices,const float* values,float* profile,int N){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)atomicMax((unsigned int*)(profile+indices[i]),__float_as_uint(values[i]));
}
extern "C" __global__ void focal_cap(float2* spectrum,const float* profile,
  double* maxima,const double* control,int area){
  __shared__ double values[2][256];int t=threadIdx.x,i=blockIdx.x*256+t;double bg=0,excess=0;
  if(i<area){float2 e=spectrum[i];double v=(double)e.x*e.x+(double)e.y*e.y;
    float envelope=profile[i];
    double cap=fmax(.005*control[4],control[6]*control[6]*envelope);
    if(envelope>0)excess=v/fmax(cap,1e-30);else bg=v;
    double gain=fmin(1.,sqrt(cap/fmax(v,1e-30)));
    spectrum[i]=make_float2((float)(gain*e.x),(float)(gain*e.y));}
  values[0][t]=bg;values[1][t]=excess;__syncthreads();
  for(int k=128;k;k/=2){if(t<k){values[0][t]=fmax(values[0][t],values[0][t+k]);values[1][t]=fmax(values[1][t],values[1][t+k]);}__syncthreads();}
  if(!t){maxima[blockIdx.x]=values[0][0];maxima[gridDim.x+blockIdx.x]=values[1][0];}
}
extern "C" __global__ void focal_prepare(const float2* field,const float* amplitude,
 const float2* phase,double* control,float2* delta,double* result,int N){
 __shared__ double sums[5][256],scale;__shared__ int bad;
 int t=threadIdx.x;if(!t)bad=0;__syncthreads();
 double inf=__longlong_as_double(0x7ff0000000000000LL),lo=inf,hi=0,error=0,sf=0,sa=0;
 for(int j=t;j<N;j+=256){double a=amplitude[j];if(a<=0)continue;
   float2 e=field[j],p=phase[j];double re=e.x,im=e.y,power=re*re+im*im,v=power/(a*a);
   if(!isfinite(v))atomicExch(&bad,1);lo=fmin(lo,v);hi=fmax(hi,v);
   if(control[5]){sf+=a*(re*p.x+im*p.y);error=fmax(error,fabs(atan2(im*p.x-re*p.y,re*p.x+im*p.y)));}
   else sf+=a*sqrt(power);
   sa+=a*a;}
 sums[0][t]=lo;sums[1][t]=hi;sums[2][t]=error;sums[3][t]=sf;sums[4][t]=sa;__syncthreads();
 for(int k=128;k;k/=2){if(t<k){sums[0][t]=fmin(sums[0][t],sums[0][t+k]);
   sums[1][t]=fmax(sums[1][t],sums[1][t+k]);sums[2][t]=fmax(sums[2][t],sums[2][t+k]);
   sums[3][t]+=sums[3][t+k];sums[4][t]+=sums[4][t+k];}__syncthreads();}
 if(!t){result[0]=bad||sums[0][0]<=0?inf:sums[1][0]/sums[0][0];result[2]=sums[2][0];
   scale=sums[3][0]/fmax(sums[4][0],1e-30);control[6]=scale;}
 __syncthreads();
 for(int j=t;j<N;j+=256){float2 e=field[j],p=phase[j];double a=amplitude[j];
   result[5+2*j]=e.x;result[6+2*j]=e.y;
   if(a<=0){delta[j]=make_float2(0,0);continue;}
   if(!control[5]){double m=fmax(hypot((double)e.x,(double)e.y),1e-20);p=make_float2(e.x/m,e.y/m);}
   delta[j]=make_float2((float)(scale*a*p.x),(float)(scale*a*p.y));}
}
extern "C" __global__ void focal_residual(const float2* field,const float* amplitude,float2* delta,int N,int area){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<N&&amplitude[j]>0){
   delta[j].x-=field[j].x/area;delta[j].y-=field[j].y/area;}
}
extern "C" __global__ void focal_check(unsigned long long handle,const double* maxima,double* control,
 double* result,int blocks){
 __shared__ double values[2][256];int t=threadIdx.x;double bg=0,excess=0;
 for(int j=t;j<blocks;j+=256){bg=fmax(bg,maxima[j]);excess=fmax(excess,maxima[blocks+j]);}
 values[0][t]=bg;values[1][t]=excess;__syncthreads();
 for(int k=128;k;k/=2){if(t<k){values[0][t]=fmax(values[0][t],values[0][t+k]);values[1][t]=fmax(values[1][t],values[1][t+k]);}__syncthreads();}
 if(!t){double background=values[0][0]/control[4];
   unsigned int more=(!(result[0]<=control[2]&&background<=.01&&result[2]<=.01&&values[1][0]<=1.05)
                       ||control[0]<control[7])&&control[0]<control[1];
   control[3]=more;result[1]=background;result[3]=control[0];result[4]=values[1][0];
   if(more)control[0]+=1;cudaGraphSetConditional(handle,more);}
}
extern "C" __global__ void focal_encode(const float2* background,const float* delta,
 const float* pupil,const float* incident,float2* optical,unsigned char* codes,const double* control,
 int H,int W,int P){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H*W||control[3]==0)return;
 float2 e=background[i],d=synthesis_pixel(delta,i/W,i%W,H,W,P);
 e.x=(e.x+d.x)/(H*W);e.y=(e.y+d.y)/(H*W);
 optical[i]=encoded_pixel(e,pupil,incident,codes,i);
}
'''


_REARRANGEMENT_KEEPERS: dict[int, dict[str, object]] = {}
_REARRANGEMENT_KEEPERS_CHANGED = Condition()


def _rearrangement_keeper_dispose(keeper):
    keeper["stop"].set()
    thread = keeper.get("thread")
    if thread is not None and thread.ident is not None:
        thread.join()
    cp, stream = keeper["cp"], keeper.get("stream")
    if stream is not None:
        failure = None
        try:
            with cp.cuda.Device(keeper["device"]):
                stream.synchronize()
        except BaseException as error:
            failure = error
        finally:
            keeper.pop("scratch", None)
            keeper.pop("stream", None)
        try:
            with cp.cuda.Device(keeper["device"]):
                cp.get_default_memory_pool().free_all_blocks(stream=stream)
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                failure.add_note(f"GPU warming arena cleanup failed: {error}")
        if failure is not None:
            raise failure


def _rearrangement_keeper_acquire(cp):
    """One bounded heater per device, retained only by prepared workspaces."""
    device = cp.cuda.Device().id
    with _REARRANGEMENT_KEEPERS_CHANGED:
        while (keeper := _REARRANGEMENT_KEEPERS.get(device)) is not None:
            # Creation and final disposal happen outside this registry lock.
            if keeper["users"] and "thread" in keeper:
                if keeper["error"] is not None:
                    raise RuntimeError("SLM GPU warming failed") from keeper["error"]
                keeper["users"] += 1
                return keeper
            _REARRANGEMENT_KEEPERS_CHANGED.wait()
        keeper = dict(cp=cp, device=device, users=1, active=0, lock=Lock(), stop=Event(), error=None)
        _REARRANGEMENT_KEEPERS[device] = keeper
    try:
        stream = keeper["stream"] = cp.cuda.Stream(non_blocking=True, priority=0)
        size = max(64 * 1024 * 1024, 2 * int(cp.cuda.runtime.getDeviceProperties(device)["l2CacheSize"]))
        with stream:
            keeper["scratch"] = cp.zeros(size // 4, cp.float32)
            kernel = cp.RawKernel(r'''extern "C" __global__ void keep(float* p,int n){
                int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]+=.000001f;
            }''', "keep")
            kernel(((size // 4 + 255) // 256,), (256,), (keeper["scratch"], np.int32(size // 4)))
            stream.synchronize()

        def run():
            try:
                with cp.cuda.Device(device), stream:
                    while not keeper["stop"].wait(.020):
                        with keeper["lock"]:
                            if keeper["active"] or keeper["stop"].is_set():
                                continue
                            kernel(((size // 4 + 255) // 256,), (256,), (keeper["scratch"], np.int32(size // 4)))
                            stream.synchronize()
            except BaseException as error:
                import logging  # noqa: PLC0415

                with keeper["lock"]:
                    keeper["error"] = error
                    keeper["stop"].set()
                logging.getLogger(__name__).exception("SLM GPU warming stopped")
                error.__traceback__ = None  # the logged traceback must not retain the scratch array

        with _REARRANGEMENT_KEEPERS_CHANGED:
            thread = Thread(target=run, name=f"slm-gpu-warm-{device}", daemon=True)
            thread.start()
            keeper["thread"] = thread
            _REARRANGEMENT_KEEPERS_CHANGED.notify_all()
        return keeper
    except BaseException as error:
        try:
            _rearrangement_keeper_dispose(keeper)
        except BaseException as cleanup_error:
            error.add_note(f"GPU warming cleanup failed: {cleanup_error}")
        finally:
            with _REARRANGEMENT_KEEPERS_CHANGED:
                del _REARRANGEMENT_KEEPERS[device]
                _REARRANGEMENT_KEEPERS_CHANGED.notify_all()
        raise


def _rearrangement_keeper_release(keeper):
    with _REARRANGEMENT_KEEPERS_CHANGED:
        keeper["users"] -= 1
        if keeper["users"]:
            return
    try:
        _rearrangement_keeper_dispose(keeper)
    finally:
        with _REARRANGEMENT_KEEPERS_CHANGED:
            del _REARRANGEMENT_KEEPERS[keeper["device"]]
            _REARRANGEMENT_KEEPERS_CHANGED.notify_all()


@contextmanager
def _rearrangement_gpu_active(keeper, *, check_error=True):
    """Pause shared warming without serializing independent scientific work."""
    with keeper["lock"]:
        if check_error and keeper["error"] is not None:
            raise RuntimeError("SLM GPU warming failed") from keeper["error"]
        keeper["active"] += 1
    try:
        with keeper["cp"].cuda.Device(keeper["device"]):
            yield
    finally:
        with keeper["lock"]:
            keeper["active"] -= 1


def _prepare_rearrangement_gpu(geometry, pupil_amplitude, pupil_phase, stop_requested, method):
    """Prepare the one native Fourier model and optional sampling quadrature."""
    try:
        import cupy as cp  # noqa: PLC0415
        from cupy.cuda import cublas, cufft  # noqa: PLC0415
        from cupyx.scipy.fft import get_fft_plan  # noqa: PLC0415
        from cuda.pathfinder import load_nvidia_dynamic_lib  # noqa: PLC0415
        from cuda.bindings import runtime as cuda  # noqa: PLC0415
    except (ImportError, OSError) as error:
        import sys  # noqa: PLC0415

        raise RuntimeError(f"Cannot import CuPy/CUDA libraries with Python {sys.executable}: "
                           f"{type(error).__name__}: {error}. Run bin\\install_slm_gpu.bat "
                           "for this product/interpreter, or install zou-lab-control[slm-gpu] using this interpreter.") from error
    import ctypes  # noqa: PLC0415

    shape = geometry["shape_yx"]
    pupil = np.asarray(pupil_amplitude, np.float32)
    if (pupil.shape != shape or not np.all(np.isfinite(pupil))
            or np.any(pupil < 0) or not np.any(pupil > 0)):
        raise ValueError("pupil_amplitude must be finite, nonnegative and match the full SLM")
    incident = np.zeros(shape, np.float32) if pupil_phase is None else canonical_phase(pupil_phase, shape)
    keeper = _rearrangement_keeper_acquire(cp)
    gpu = None
    try:
        with _rearrangement_gpu_active(keeper):
            stream = cp.cuda.Stream(non_blocking=True, priority=-1)
            library = (ctypes.WinDLL if os.name == "nt" else ctypes.CDLL)(load_nvidia_dynamic_lib("cublas").abs_path)
            gemm = library.cublasGemmEx
            pointer, integer = ctypes.c_void_p, ctypes.c_int
            gemm.argtypes = [pointer, integer, integer, integer, integer, integer, pointer,
                            pointer, integer, integer, pointer, integer, integer,
                            pointer, pointer, integer, integer, integer, integer]
            gemm.restype = integer
            set_workspace = library.cublasSetWorkspace_v2
            set_workspace.argtypes = [pointer, pointer, ctypes.c_size_t]
            set_workspace.restype = integer
            number = len(geometry["source_yx"])
            maximum_selected = min(number, len(geometry["target_yx"]))
            # Unselected integer sites share one band per source X. Each selected
            # trajectory can add at most one(X, fractional-Y) band per frame.
            source_columns = len(np.unique(geometry["source_yx"][:, 1]))
            motion_bands = (min(number, source_columns + maximum_selected) + 15) // 16 * 16
            if method == "lpi":
                target_columns = len(np.unique(geometry["target_yx"][:, 1]))
                motion_bands = max(motion_bands, (target_columns + 15) // 16 * 16)
            capacity = motion_bands
            module = cp.RawModule(code="#define HALF 1\n" + _REARRANGEMENT_CUDA)
            names = ("load_motion_frame", "store_motion_frame", "phase_step_rms", "scatter", "gather", "pack_inverse", "project", "pack_field", "select_roots",
                     "pack_forward", "encode", "encode_pack", "field_project", "anderson_begin", "anderson_update", "focal_profile", "focal_prepare", "focal_cap", "focal_check", "focal_residual", "focal_encode")
            kernels = {name: module.get_function(name) for name in names}
            from scipy.fft import fft2, fftshift, ifftshift  # noqa: PLC0415

            psf = abs(fftshift(fft2(ifftshift(pupil)))) ** 2
            gpu = dict(cp=cp, cublas=cublas, cufft=cufft, cuda=cuda, stream=stream, module=module, kernels=kernels, graphs={}, background_graphs={},
                       library=library, gemm=gemm, shape=shape, number=number,
                       pupil_cpu=pupil.copy(), incident_cpu=np.asarray(incident, np.float32).copy(),
                       psf_support=_psf_main_lobe(psf, tuple(np.asarray(shape) // 2)),
                       pupil_energy=float(np.sum(pupil.astype(np.float64) ** 2)), pupil_scale=float(np.max(pupil)),
                       keeper=keeper, resources={}, output_pool=cp.cuda.PinnedMemoryPool(), weight_exponent=np.float32(.8), method=method)
            offsets = np.argwhere(gpu["psf_support"]) - np.asarray(shape) // 2
            lower, upper = offsets.min(axis=0) - 1, offsets.max(axis=0) + 1
            oversampling = 4
            oversampled = fftshift(fft2(pupil, s=tuple(oversampling * np.asarray(shape))))
            center = oversampling * np.asarray(shape) // 2
            slices = tuple(slice(int(a), int(b) + 1)
                           for a, b in zip(center + oversampling * lower, center + oversampling * upper))
            profile = np.asarray(abs(oversampled[slices]) ** 2 / abs(oversampled[tuple(center)]) ** 2, np.float32)
            del oversampled
            profile *= _psf_main_lobe(profile, tuple(-lower * oversampling))
            yy, xx = np.mgrid[lower[0]:upper[0] + 1, lower[1]:upper[1] + 1]
            gpu.update(psf_profile=profile, psf_profile_origin=lower, psf_oversampling=oversampling,
                       psf_neighborhood=np.stack((yy.ravel(), xx.ravel()), axis=1))
            # Stride-two samples share the centered native coordinates only at these sizes.
            factors = (1, 2) if all(size % 4 == 0 for size in shape) else (1,)
            handles = []
            def close():
                if not gpu:
                    return
                try:
                    with _rearrangement_gpu_active(keeper, check_error=False):
                        stream.synchronize()
                        _rearrangement_free_graphs(gpu)
                        for handle in handles:
                            cublas.destroy(handle)
                        handles.clear()
                        gpu["output_pool"].free_all_blocks()
                finally:
                    gpu.clear()
                    try:
                        with cp.cuda.Device(keeper["device"]):
                            cp.get_default_memory_pool().free_all_blocks(stream=stream)
                    finally:
                        _rearrangement_keeper_release(keeper)
            gpu["close"] = close
            with stream:
                # All handles use this one stream, so their scratch can be shared.
                # User-owned scratch prevents cuBLAS adding per-call allocation
                # nodes, which are also illegal inside a conditional graph.
                gpu["blas_workspace"] = cp.empty(4 * 1024 * 1024, cp.uint8)
                site_capacity = max(number, len(geometry["target_yx"])) if method == "lpi" else number
                gpu["coefficients"] = cp.zeros(site_capacity, cp.complex64)
                gpu["frequencies"] = cp.zeros((capacity, 2), cp.float64)
                for factor, precise in [(factor, False) for factor in factors] + [(1, True)]:
                    if stop_requested is not None and stop_requested():
                        raise InterruptedError("SLM rearrangement preparation stopped")
                    dtype = cp.float32 if precise else cp.float16
                    work_capacity = capacity if precise else motion_bands
                    work_module = cp.RawModule(code="#define HALF 0\n" + _REARRANGEMENT_CUDA) if precise else module
                    work_kernels = {name: work_module.get_function(name) for name in names} if precise else kernels
                    h, w = (size // factor for size in shape)
                    padded, ly = (w // 2 + 16) // 16 * 16, h
                    handle = cublas.create()
                    handles.append(handle)
                    cublas.setStream(handle, stream.ptr)
                    status = set_workspace(handle, gpu["blas_workspace"].data.ptr, gpu["blas_workspace"].nbytes)
                    if status:
                        raise RuntimeError(f"SLM cuBLAS workspace setup failed ({status})")
                    work = dict(shape=(h, w), padded=padded, ly=ly, handle=handle,
                                typecode=0 if precise else 2, module=work_module, kernels=work_kernels,
                                forward=cp.empty((2 * padded, work_capacity), dtype),
                                y_roots=cp.empty((work_capacity, h), cp.complex128),
                                spectrum=cp.empty((work_capacity, ly), cp.complex64),
                                transformed=cp.empty((work_capacity, ly), cp.complex64),
                                field_gemm=cp.empty((2, h, 2 * padded), dtype),
                                projected=cp.empty((2, h, work_capacity), cp.float32),
                                actual=cp.empty(site_capacity, cp.complex64),
                                index=(gpu["resources"][1]["index"] if precise
                                       else cp.arange(site_capacity, dtype=cp.int32) % (16 * ly)),
                                frequencies=gpu["frequencies"],
                                alpha=np.asarray(1, np.float32), beta=np.asarray(0, np.float32),
                                plans={band: cufft.Plan1d(ly, cufft.CUFFT_C2C, band)
                                       for band in range(16, work_capacity + 1, 16)})
                    if precise:
                        # The final decision uses FP32 roots/packing on the same finite
                        # Fourier map. Roots, emitted optical field and query indices
                        # belong to the same native geometry; only packing is precise.
                        work["optical"] = gpu["resources"][1]["optical"]
                        gpu["measurement"] = work
                        work.update(backward=cp.empty((work_capacity, 2 * padded), cp.float32),
                                    packed=cp.empty((2, h, work_capacity), cp.float32),
                                    image=cp.empty((2, h, 2 * padded), cp.float32))
                        if method == "lpi":
                            work.update(physical_pupil=gpu["resources"][1]["physical_pupil"],
                                        incident=gpu["resources"][1]["incident"],
                                        codes=gpu["resources"][1]["codes"])
                    else:
                        work.update(backward=cp.empty((work_capacity, 2 * padded), dtype),
                                    packed=cp.empty((2, h, work_capacity), dtype),
                                    image=cp.empty((2, h, 2 * padded), cp.float32),
                                    pupil=cp.asarray(pupil[::factor, ::factor]
                                                     / np.float32(gpu["pupil_scale"]) * factor ** 2))
                        if factor == 1:
                            work.update(capacity=work_capacity, optical=cp.empty((h, w), cp.complex64),
                                        codes=cp.empty((h, w), cp.uint8),
                                        physical_pupil=cp.asarray(pupil),
                                        incident=cp.asarray(incident, cp.float32))
                        gpu["resources"][factor] = work
                    for band in work["plans"]:
                        _rearrangement_select_roots(work, band)
                        if precise:
                            if method == "lpi":
                                _rearrangement_propagate(gpu, work, band, "encode")
                            _rearrangement_propagate(gpu, work, band, "forward")
                        else:
                            _rearrangement_propagate(gpu, work, band, "roundtrip")
                            if factor == 1:
                                _rearrangement_propagate(gpu, work, band, "encode")
                gpu["previous_base"] = cp.empty(shape, cp.complex64)
                gpu["previous_corrected"] = cp.empty(shape, cp.complex64)
                gpu["current_base"] = cp.empty(shape, cp.complex64)
                gpu["aa_phase"] = cp.empty(site_capacity, cp.complex64)
                gpu["focal_profile"] = cp.zeros(shape, cp.float32)
                profile_capacity = site_capacity * len(gpu["psf_neighborhood"])
                gpu["focal_profile_indices"] = cp.empty(profile_capacity, cp.int64)
                gpu["focal_profile_values"] = cp.empty(profile_capacity, cp.float32)
                gpu["focal_delta"] = cp.empty(site_capacity, cp.complex64)
                gpu["focal_amplitude"] = cp.zeros(site_capacity, cp.float32)
                gpu["focal_phase"] = cp.ones(site_capacity, cp.complex64)
                gpu["focal_spectrum"] = cp.empty(shape, cp.complex64)
                gpu["focal_background"] = cp.empty(shape, cp.complex64)
                gpu["focal_plan"] = get_fft_plan(gpu["focal_spectrum"], axes=(0, 1), value_type="C2C")
                gpu["focal_maxima"] = cp.empty(2 * ((int(np.prod(shape)) + 255) // 256), cp.float64)
                gpu["focal_control"] = cp.asarray([0., 0., 1., 0., 1., 0., 1., 0.])
                gpu["focal_result"] = cp.empty(5 + 2 * site_capacity, cp.float64)
                gpu["focal_work"] = {**gpu["measurement"], "number": site_capacity,
                                     "coefficients": gpu["focal_delta"]}
                gpu["aa_g"] = cp.empty((3, site_capacity), cp.float32)
                gpu["aa_r"] = cp.empty((3, site_capacity), cp.float32)
                gpu["aa_candidate"] = cp.empty(site_capacity, cp.float64)
                gpu["aa_state"] = cp.zeros(2, cp.int32)
                gpu["amplitude"] = cp.ones(site_capacity, cp.float32)
                gpu["frame_index"] = cp.zeros(1, cp.int32)
                staging = gpu["output_pool"].malloc(int(np.prod(shape)))
                gpu["host_frame"] = np.frombuffer(staging, np.uint8, count=int(np.prod(shape))).reshape(shape)
                gpu["motion_capacity"] = 0
                stream.synchronize()
            return gpu
    except BaseException as error:
        try:
            if gpu is not None and "close" in gpu:
                gpu["close"]()
            else:
                _rearrangement_keeper_release(keeper)
        except BaseException as cleanup_error:
            error.add_note(f"GPU preparation cleanup failed: {cleanup_error}")
        raise


def _rearrangement_select_roots(work, band):
    """Cache X roots in workspace precision and Y carriers in double precision."""
    padded, height = work["padded"], work["shape"][0]
    work["kernels"]["select_roots"](((max(padded, height) + 31) // 32, (band + 31) // 32), (32, 8),
                                   (work["frequencies"], work.get("backward", np.uint64(0)), work["forward"],
                                    work["y_roots"], *map(np.int32, (band, band, padded, height, work["shape"][1]))))


def _rearrangement_propagate(gpu, work, band, operation):
    """Centered finite Fourier propagation with the frame's already packed roots."""
    h, w = work["shape"]
    padded, ly, number = work["padded"], work["ly"], work.get("number", gpu["number"])
    kernels, cb = work["kernels"], gpu["cublas"]
    typecode = work["typecode"]
    alpha, beta = work["alpha"].ctypes.data, work["beta"].ctypes.data
    if operation != "forward":
        work["spectrum"][:band].fill(0)
        kernels["scatter"](((number + 255) // 256,), (256,),
                           (work.get("coefficients", gpu["coefficients"]), work["index"], work["spectrum"], np.int32(number)))
        work["plans"][band].fft(work["spectrum"][:band], work["transformed"][:band], gpu["cufft"].CUFFT_INVERSE)
        kernels["pack_inverse"](((h + 31) // 32, (band + 31) // 32), (32, 8),
                                (work["transformed"], work["y_roots"], work["packed"], np.int32(h), np.int32(ly), np.int32(band)))
        status = gpu["gemm"](work["handle"], 0, 0, 2 * padded, 2 * h, band, alpha,
                             work["backward"].data.ptr, typecode, 2 * padded,
                             work["packed"].data.ptr, typecode, band, beta,
                             work["image"].data.ptr, 0, 2 * padded,
                             cb.CUBLAS_COMPUTE_32F, cb.CUBLAS_GEMM_DEFAULT)
        if status:
            raise RuntimeError(f"SLM cuBLAS synthesis failed ({status})")
        if operation == "backward":
            return
        if operation == "correct":
            native = gpu["resources"][1]
            gpu["kernels"]["field_project"](((h * w + 255) // 256,), (256,),
                (native["codes"], native["optical"], work["image"], native["optical"], native["optical"],
                 native["physical_pupil"], native["incident"], native["codes"], *map(np.int32, (h, w, padded, 2))))
            return
        if operation == "encode":
            kernels["encode"](((h * w + 255) // 256,), (256,),
                              (work["image"], work["physical_pupil"], work["incident"], work["optical"],
                               work["codes"], *map(np.int32, (h, w, padded))))
            return
        if operation == "encode_forward":
            # The encoded field is consumed immediately by measurement. Pack
            # it directly instead of writing and rereading a full complex raster.
            kernels["encode_pack"](((h * padded + 255) // 256,), (256,),
                (work["image"], work["physical_pupil"], work["incident"], work["field_gemm"],
                 work["codes"], *map(np.int32, (h, w, padded))))
        else:
            kernels["project"](((h * padded + 255) // 256,), (256,),
                               (work["image"], work["pupil"], work["field_gemm"],
                                np.int32(h), np.int32(w), np.int32(padded)))
    if operation == "forward":
        kernels["pack_field"](((h * padded + 255) // 256,), (256,),
                              (work["optical"], work["field_gemm"], np.int32(h), np.int32(w), np.int32(padded)))
    status = gpu["gemm"](work["handle"], 0, 0, band, 2 * h, 2 * padded, alpha,
                         work["forward"].data.ptr, typecode, band,
                         work["field_gemm"].data.ptr, typecode, 2 * padded, beta,
                         work["projected"].data.ptr, 0, band,
                         cb.CUBLAS_COMPUTE_32F, cb.CUBLAS_GEMM_DEFAULT)
    if status:
        raise RuntimeError(f"SLM cuBLAS analysis failed ({status})")
    kernels["pack_forward"](((band + 31) // 32, (ly + 31) // 32), (32, 8),
                            (work["projected"], work["y_roots"], work["spectrum"], np.int32(h), np.int32(ly), np.int32(band)))
    work["plans"][band].fft(work["spectrum"][:band], work["transformed"][:band], gpu["cufft"].CUFFT_FORWARD)
    kernels["gather"](((number + 255) // 256,), (256,),
                      (work["transformed"], work["index"], work["actual"], np.int32(number)))


def _rearrangement_amplitude_updates(gpu, work, band, count):
    """Depth-two amplitude updates, reset for this frame and resolution."""
    _rearrangement_select_roots(work, band)
    if not count:
        return
    number = np.int32(gpu["number"])
    gpu["kernels"]["anderson_begin"]((1,), (256,),
                                     (gpu["coefficients"], gpu["aa_phase"], gpu["aa_state"], number))
    for _ in range(count):
        _rearrangement_propagate(gpu, work, band, "roundtrip")
        gpu["kernels"]["anderson_update"]((1,), (256,),
            (work["actual"], gpu["amplitude"], gpu["coefficients"], gpu["aa_phase"],
             gpu["aa_g"], gpu["aa_r"], gpu["aa_candidate"], gpu["aa_state"], number, gpu["weight_exponent"]))


def _rearrangement_load_frame(gpu, *, warm_previous=False, preserve_phase=False):
    """Stage one row; its incremented counter stays fixed until the next load."""
    native = gpu["resources"][1]
    coarse = gpu["resources"].get(2, native)
    gpu["kernels"]["load_motion_frame"]((1,), (256,),
        (gpu["frame_index"], gpu["motion_indices"][1], gpu["motion_indices"].get(2, gpu["motion_indices"][1]),
         gpu["motion_frequencies"], gpu["motion_offsets"], gpu["motion_coefficients"],
         native["index"], coarse["index"], gpu["motion_amplitudes"], gpu["frequencies"], gpu["coefficients"],
         gpu["amplitude"], np.int32(gpu["number"]), np.int32(native["capacity"]),
         np.int32(warm_previous), np.int32(preserve_phase)))


def _rearrangement_store_frame(gpu):
    """Copy accepted frame outputs; no block changes the row counter here."""
    area = int(np.prod(gpu["shape"]))
    gpu["kernels"]["store_motion_frame"](((area + 255) // 256,), (256,),
        (gpu["frame_index"], gpu["resources"][1]["codes"], gpu["coefficients"], gpu["measurement"]["actual"],
         gpu["motion_codes"], gpu["motion_coefficients"], gpu["motion_actual"], np.int32(gpu["number"]), np.int32(area)))


def _rearrangement_bind(gpu, points):
    """Build bulk trajectory metadata; the caller retains host inputs through transfer."""
    shape = np.asarray(gpu["shape"])
    signed = np.asarray(points, np.float64) - shape // 2
    integer_y = np.floor(signed[..., 0]).astype(np.int64)
    # Each band has one X frequency and one fractional Y carrier. Its remaining
    # Y frequencies are native integer FFT bins, for any authored frame count.
    bands = np.stack((signed[..., 1], signed[..., 0] - integer_y), axis=-1)
    # Group all frames once, retaining each frame's lexicographic band order
    # and local inverse indices, rather than setting up one unique per frame.
    frame_count, sites_per_frame = bands.shape[:2]
    frame_ids = np.repeat(np.arange(frame_count), sites_per_frame)
    flat = bands.reshape(-1, 2)
    order = np.lexsort((flat[:, 1], flat[:, 0], frame_ids))
    ordered_frames, ordered_bands = frame_ids[order], flat[order]
    starts = np.r_[True, (ordered_frames[1:] != ordered_frames[:-1])
                   | np.any(ordered_bands[1:] != ordered_bands[:-1], axis=1)]
    inverse = np.empty_like(order)
    inverse[order] = np.cumsum(starts) - 1
    counts = np.bincount(ordered_frames[starts], minlength=frame_count).astype(np.int32)
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    selected = ordered_bands[starts]
    lookup = inverse.reshape(frame_count, sites_per_frame) - offsets[:-1, None]
    indices = {}
    for factor, work in gpu["resources"].items():
        packed = lookup * work["ly"] + integer_y % work["ly"]
        indices[factor] = packed.astype(np.int32)
    return indices, selected, counts, offsets




def _rearrangement_endpoint(cp, points, shape, pupil, intensity, iterations, seed, stop_requested):
    """Prepare independent WGS-Kim coefficients for an endpoint geometry."""
    iy, ix = cp.asarray((points - np.asarray(shape) // 2).T % np.asarray(shape)[:, None])
    amplitude = cp.sqrt(cp.asarray(intensity, dtype=cp.float32))
    amplitude /= cp.linalg.norm(amplitude)
    phase = cp.asarray(np.random.default_rng(seed).uniform(-np.pi, np.pi, len(points)).astype(np.float32))
    weight = amplitude.copy()
    spectrum = cp.zeros(shape, cp.complex64)
    illumination = cp.fft.ifftshift(cp.asarray(pupil))
    coefficient = weight * cp.exp(cp.complex64(1j) * phase)
    for iteration in range(iterations):
        if stop_requested is not None and stop_requested():
            raise InterruptedError("SLM rearrangement preparation stopped")
        spectrum.fill(0)
        spectrum[iy, ix] = coefficient
        back = cp.fft.ifft2(spectrum)
        field = illumination * cp.exp(cp.complex64(1j) * cp.angle(back))
        actual = cp.fft.fft2(field)[iy, ix]
        magnitude = cp.abs(actual)
        measured = magnitude / cp.maximum(cp.linalg.norm(magnitude), 1e-20)
        weight *= cp.clip(amplitude / cp.maximum(measured, 1e-20), .2, 5) ** cp.float32(.8)
        weight /= cp.linalg.norm(weight)
        if iteration < 12:
            phase = cp.angle(actual)
        coefficient = weight * cp.exp(cp.complex64(1j) * phase)
    spectrum.fill(0)
    spectrum[iy, ix] = coefficient
    pattern = cp.remainder(cp.angle(cp.fft.fftshift(cp.fft.ifft2(spectrum))), cp.float32(2 * np.pi))
    return cp.asnumpy(coefficient), cp.asnumpy(pattern)


def _rearrangement_cuda_result(result):
    if result[0]:
        raise RuntimeError(f"SLM CUDA graph operation failed: {result[0].name}")
    return result[1] if len(result) > 1 else None


def _rearrangement_free_graphs(gpu):
    for graph, executable in gpu["background_graphs"].values():
        _rearrangement_cuda_result(gpu["cuda"].cudaGraphExecDestroy(executable))
        _rearrangement_cuda_result(gpu["cuda"].cudaGraphDestroy(graph))
    gpu["background_graphs"].clear()
    gpu["graphs"].clear()


def _rearrangement_motion_capacity(gpu, count, stop_requested):
    """Grow only movie-dependent arrays and bindings; fixed optics stay prepared."""
    if count <= gpu["motion_capacity"]:
        return
    cp, stream, number = gpu["cp"], gpu["stream"], gpu["number"]
    native = gpu["resources"][1]
    with _rearrangement_gpu_active(gpu["keeper"]), stream:
        stream.synchronize()
        gpu["motion_capacity"] = 0
        # Only the baseline graph addresses movie-sized arrays. The full-field
        # loop uses fixed optical buffers and survives a change in authored N.
        gpu["graphs"].clear()
        gpu["motion_codes"] = cp.empty((count, *gpu["shape"]), cp.uint8)
        gpu["motion_coefficients"] = cp.zeros((count, number), cp.complex64)
        gpu["motion_amplitudes"] = cp.ones((count, number), cp.float32)
        gpu["motion_actual"] = cp.empty((count, number), cp.complex64)
        gpu["motion_indices"] = {factor: cp.zeros((count, number), cp.int32) for factor in gpu["resources"]}
        gpu["motion_frequencies"] = cp.zeros((count * native["capacity"], 2), cp.float64)
        gpu["motion_offsets"] = cp.zeros(count + 1, cp.int64)
        gpu["phase_step_rms"] = cp.empty(count, cp.float32)
        gpu["motion_coefficients"][:] = gpu["coefficients"][:number][None]
        gpu["motion_amplitudes"][:] = gpu["amplitude"][:number][None]
        if gpu["method"] == "iterative":
            coarse_updates = 3 if 2 in gpu["resources"] else 0
            native_updates = 2 if coarse_updates else 10
            _rearrangement_amplitude_updates(gpu, native, 16, 1)
            if coarse_updates:
                _rearrangement_amplitude_updates(gpu, gpu["resources"][2], 16, 1)
            stream.synchronize()
            for band in native["plans"]:
                if stop_requested is not None and stop_requested():
                    raise InterruptedError("SLM rearrangement preparation stopped")
                stream.begin_capture()
                _rearrangement_load_frame(gpu, warm_previous=True)
                if coarse_updates:
                    _rearrangement_amplitude_updates(gpu, gpu["resources"][2], band, coarse_updates)
                _rearrangement_amplitude_updates(gpu, native, band, native_updates)
                _rearrangement_propagate(gpu, native, band, "encode")
                _rearrangement_select_roots(gpu["measurement"], band)
                _rearrangement_propagate(gpu, gpu["measurement"], band, "forward")
                _rearrangement_store_frame(gpu)
                gpu["graphs"][band] = stream.end_capture()
                gpu["frame_index"].fill(0)
                gpu["graphs"][band].launch(stream)
        if gpu["background_graphs"]:
            stream.synchronize()
            gpu["motion_capacity"] = count
            return
        # Both methods share one full-field constraint loop. All its buffers,
        # FFT plans and launches are prepared before an acquisition can fire.
        work, cuda, cufft = gpu["focal_work"], gpu["cuda"], gpu["cufft"]
        area = int(np.prod(gpu["shape"]))
        grid = ((area + 255) // 256,)
        gpu["focal_control"].set(np.array([0., 0., 1., 0., 1., 0., 1., 0.]), stream=stream)
        gpu["kernels"]["focal_profile"]((1,), (256,),
            (gpu["focal_profile_indices"], gpu["focal_profile_values"], gpu["focal_profile"], np.int32(0)))
        gpu["focal_plan"].fft(native["optical"], gpu["focal_spectrum"], cufft.CUFFT_FORWARD)
        gpu["focal_plan"].fft(gpu["focal_spectrum"], gpu["focal_background"], cufft.CUFFT_INVERSE)

        def evaluate(band, handle):
            _rearrangement_propagate(gpu, work, band, "forward")
            gpu["kernels"]["focal_prepare"]((1,), (256,),
                (work["actual"], gpu["focal_amplitude"], gpu["focal_phase"], gpu["focal_control"],
                 gpu["focal_delta"], gpu["focal_result"], np.int32(work["number"])))
            gpu["focal_plan"].fft(native["optical"], gpu["focal_spectrum"], cufft.CUFFT_FORWARD)
            gpu["kernels"]["focal_cap"](grid, (256,),
                (gpu["focal_spectrum"], gpu["focal_profile"], gpu["focal_maxima"],
                 gpu["focal_control"], np.int32(area)))
            gpu["kernels"]["focal_check"]((1,), (256,),
                (np.uint64(int(handle)), gpu["focal_maxima"], gpu["focal_control"],
                 gpu["focal_result"], np.int32(grid[0])))

        for band in work["plans"]:
            if stop_requested is not None and stop_requested():
                raise InterruptedError("SLM rearrangement preparation stopped")
            _rearrangement_select_roots(work, band)
            _rearrangement_propagate(gpu, work, band, "backward")
            graph = _rearrangement_cuda_result(cuda.cudaGraphCreate(0))
            try:
                handle = _rearrangement_cuda_result(cuda.cudaGraphConditionalHandleCreate(graph, 1, 1))
                # Decide before entering the update loop. A converged result
                # must not pay for an unused inverse transform and propagation.
                _rearrangement_cuda_result(cuda.cudaStreamBeginCaptureToGraph(
                    stream.ptr, graph, None, None, 0,
                    cuda.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
                try:
                    evaluate(band, handle)
                    frontier = cuda.cudaStreamGetCaptureInfo(stream.ptr)
                    _rearrangement_cuda_result(frontier)
                    dependencies = list(frontier[4])[:frontier[5]]
                finally:
                    _rearrangement_cuda_result(cuda.cudaStreamEndCapture(stream.ptr))
                node = cuda.cudaGraphNodeParams()
                node.type = cuda.cudaGraphNodeType.cudaGraphNodeTypeConditional
                node.conditional.handle = handle
                node.conditional.type = cuda.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile
                node.conditional.size = 1
                _rearrangement_cuda_result(cuda.cudaGraphAddNode(graph, dependencies, len(dependencies), node))
                _rearrangement_cuda_result(cuda.cudaStreamBeginCaptureToGraph(
                    stream.ptr, node.conditional.phGraph_out[0], None, None, 0,
                    cuda.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
                try:
                    gpu["focal_plan"].fft(gpu["focal_spectrum"], gpu["focal_background"], cufft.CUFFT_INVERSE)
                    # The background projection changed the field at bright
                    # coordinates too. Measure that field before replacing it;
                    # adding a residual from the pre-clipped field double-counts
                    # the amplitude change and progressively loses bright power.
                    _rearrangement_propagate(gpu, {**work, "optical": gpu["focal_background"]}, band, "forward")
                    gpu["kernels"]["focal_residual"](((work["number"] + 255) // 256,), (256,),
                        (work["actual"], gpu["focal_amplitude"], gpu["focal_delta"],
                         np.int32(work["number"]), np.int32(area)))
                    _rearrangement_propagate(gpu, work, band, "backward")
                    gpu["kernels"]["focal_encode"](grid, (256,),
                        (gpu["focal_background"], work["image"], native["physical_pupil"], native["incident"],
                         native["optical"], native["codes"], gpu["focal_control"],
                         *map(np.int32, (*gpu["shape"], work["padded"]))))
                    evaluate(band, handle)
                finally:
                    _rearrangement_cuda_result(cuda.cudaStreamEndCapture(stream.ptr))
                executable = _rearrangement_cuda_result(cuda.cudaGraphInstantiate(graph, 0))
            except BaseException:
                _rearrangement_cuda_result(cuda.cudaGraphDestroy(graph))
                raise
            gpu["background_graphs"][band] = (graph, executable)
            _rearrangement_cuda_result(cuda.cudaGraphLaunch(executable, stream.ptr))
        stream.synchronize()
        gpu["motion_capacity"] = count


def _rearrangement_background_projection(
    gpu, points, desired, codes, reference_intensity, tolerance, *,
    prescribed_phase=None, iterations=128, stop_requested=None, warm_start=False,
):
    """Native-grid correction; denser full-field audit remains independent.

    Keep low background free, bound each main lobe by the actual pupil's PSF,
    and impose bright complex fields with the exact fractional operator.
    This is an additional optical constraint, not a claim that ordinary WGS or
    LPI guarantees no ghosts. Every decision is made after logical encoding.
    """
    cp, stream, native = gpu["cp"], gpu["stream"], gpu["resources"][1]
    shape = np.asarray(gpu["shape"])
    number, area = len(points), int(np.prod(shape))
    active = np.asarray(desired) > 0
    from scipy.ndimage import map_coordinates  # noqa: PLC0415

    profile_pixels, profile_values = [], []
    for point, amplitude in zip(np.asarray(points)[active], np.asarray(desired)[active]):
        pixels = np.floor(point).astype(np.int64) + gpu["psf_neighborhood"]
        coordinates = (pixels - point - gpu["psf_profile_origin"]) * gpu["psf_oversampling"]
        values = map_coordinates(gpu["psf_profile"], coordinates.T, order=1, mode="constant", cval=0., prefilter=False)
        positive = values > 0
        pixels = (pixels[positive] - shape // 2) % shape
        profile_pixels.append(pixels[:, 0] * shape[1] + pixels[:, 1])
        profile_values.append(np.asarray(amplitude * amplitude * values[positive], np.float32))
    profile_pixels = np.concatenate(profile_pixels)
    profile_values = np.concatenate(profile_values)
    gpu["focal_profile"].fill(0)
    gpu["focal_profile_indices"][:len(profile_pixels)].set(profile_pixels, stream=stream)
    gpu["focal_profile_values"][:len(profile_values)].set(profile_values, stream=stream)
    gpu["kernels"]["focal_profile"](((len(profile_pixels) + 255) // 256,), (256,),
        (gpu["focal_profile_indices"], gpu["focal_profile_values"], gpu["focal_profile"], np.int32(len(profile_pixels))))
    indices, frequencies, counts, _ = _rearrangement_bind(gpu, np.asarray(points)[None])
    work = gpu["focal_work"]
    band = (int(counts[0]) + 15) // 16 * 16
    work["index"].fill(0)
    work["index"][:number].set(indices[1][0], stream=stream)
    gpu["frequencies"][:len(frequencies)].set(frequencies, stream=stream)
    _rearrangement_select_roots(work, band)
    grid = ((area + 255) // 256,)
    gpu["focal_amplitude"].fill(0)
    gpu["focal_amplitude"][:number].set(np.asarray(desired, np.float32), stream=stream)
    if prescribed_phase is not None:
        gpu["focal_phase"][:number].set(np.exp(1j * np.asarray(prescribed_phase, np.float64)).astype(np.complex64), stream=stream)
    gpu["focal_control"].set(np.array([0., iterations, tolerance, 0., reference_intensity,
                                      float(prescribed_phase is not None), 1., float(warm_start)]), stream=stream)
    gpu["kernels"]["field_project"](grid, (256,),
        (codes, native["optical"], native["image"], native["optical"], native["optical"],
         native["physical_pupil"], native["incident"], native["codes"],
         *map(np.int32, (*shape, native["padded"], 0))))
    gpu["current_base"][:] = native["optical"]
    if warm_start:
        # Restore the fresh geometry's bright scale once in the warm seed;
        # this is not a fixed-power constraint on the ensuing iterations.
        # Otherwise repeated warm starts inherit an ever dimmer signal subspace.
        _rearrangement_propagate(gpu, work, band, "forward")
        gpu["kernels"]["focal_prepare"]((1,), (256,),
            (work["actual"], gpu["focal_amplitude"], gpu["focal_phase"], gpu["focal_control"],
             gpu["focal_delta"], gpu["focal_result"], np.int32(work["number"])))
        gpu["kernels"]["field_project"](grid, (256,),
            (native["codes"], native["optical"], native["image"], gpu["previous_base"], gpu["previous_corrected"],
             native["physical_pupil"], native["incident"], native["codes"],
             *map(np.int32, (*shape, native["padded"], 1))))
        _rearrangement_propagate(gpu, work, band, "forward")
        gpu["kernels"]["focal_residual"](((work["number"] + 255) // 256,), (256,),
            (work["actual"], gpu["focal_amplitude"], gpu["focal_delta"], np.int32(work["number"]), np.int32(1)))
        _rearrangement_propagate(gpu, work, band, "correct")
    if stop_requested is not None and stop_requested():
        raise InterruptedError("SLM rearrangement stopped")
    executable = gpu["background_graphs"][band][1]
    _rearrangement_cuda_result(gpu["cuda"].cudaGraphLaunch(executable, stream.ptr))
    result = gpu["focal_result"].get(stream=stream)
    if stop_requested is not None and stop_requested():
        raise InterruptedError("SLM rearrangement stopped")
    gpu["previous_base"][:] = gpu["current_base"]
    gpu["previous_corrected"][:] = native["optical"]
    field = (result[5:5 + 2 * number:2] + 1j * result[6:5 + 2 * number:2]).astype(np.complex64)
    return field, float(result[1]), float(result[2]), int(result[3]), float(result[4])


def _rearrangement_balance_endpoint(gpu, points, intensities, coefficients, iterations, tolerance, stop_requested):
    """Balance one fixed-phase endpoint with the existing encoded-field update."""
    import time  # noqa: PLC0415

    started = time.perf_counter()
    cp, stream, native = gpu["cp"], gpu["stream"], gpu["resources"][1]
    work = gpu["measurement"]
    count, original_count = len(points), gpu["number"]
    indices, frequencies, counts, _ = _rearrangement_bind(gpu, np.asarray(points)[None])
    band = (int(counts[0]) + 15) // 16 * 16
    coefficient = np.asarray(coefficients, np.complex64).copy()
    coefficient /= np.linalg.norm(coefficient)
    amplitude = np.sqrt(np.asarray(intensities, np.float32))
    amplitude /= np.linalg.norm(amplitude)
    try:
        with _rearrangement_gpu_active(gpu["keeper"]), stream:
            gpu["number"] = count
            native["index"][:count].set(indices[1][0], stream=stream)
            gpu["frequencies"][:len(frequencies)].set(frequencies, stream=stream)
            gpu["coefficients"][:count].set(coefficient, stream=stream)
            gpu["amplitude"][:count].set(amplitude, stream=stream)
            _rearrangement_select_roots(work, band)
            gpu["kernels"]["anderson_begin"]((1,), (256,),
                (gpu["coefficients"], gpu["aa_phase"], gpu["aa_state"], np.int32(count)))
            for updates in range(iterations + 1):
                if stop_requested is not None and stop_requested():
                    raise InterruptedError("SLM endpoint preparation stopped")
                _rearrangement_propagate(gpu, work, band, "encode_forward")
                field = work["actual"][:count].get(stream=stream)
                relative = abs(field.astype(np.complex128)) ** 2 / intensities
                ratio = float(relative.max() / relative.min())
                if ratio <= tolerance or updates == iterations:
                    break
                gpu["kernels"]["anderson_update"]((1,), (256,),
                    (work["actual"], gpu["amplitude"], gpu["coefficients"], gpu["aa_phase"],
                     gpu["aa_g"], gpu["aa_r"], gpu["aa_candidate"], gpu["aa_state"],
                     np.int32(count), gpu["weight_exponent"]))
            coefficient = gpu["coefficients"][:count].get(stream=stream)
            codes = native["codes"].get(stream=stream)
            stream.synchronize()
    finally:
        gpu["number"] = original_count
    return {"coefficients": _frozen(coefficient), "field": _frozen(field), "phase_codes": _frozen(codes),
            "phase": _frozen(codes.astype(np.float64) * (2 * np.pi / 256)), "ratio": ratio,
            "iterations": updates, "timing_ms": (time.perf_counter() - started) * 1000}


def prepare_rearrangement(
    source_yx: object, target_yx: object, *, shape_yx: tuple[int, int],
    pupil_amplitude: object, minimum_separation: float,
    pupil_phase: object | None = None, source_intensities: object | None = None,
    target_intensities: object | None = None,
    endpoint_iterations: int = 150, seed: int = 0,
    endpoint_data: Mapping[str, object] | None = None,
    maximum_motion_frames: int = 16,
    support_tolerance: float = SPOT_SUPPORT_TOLERANCE,
    stop_requested: Callable[[], bool] | None = None,
    method: str = "iterative", phase_center_yx: object | None = None,
) -> dict[str, object]:
    """Prepare one source optical state and explicit destination geometry.

    A supplied source_phase stays exact for initial application. LPI prepares
    an independent WGS target, then balances its fixed coefficient phases with
    the shared encoded-field amplitude update before occupancy is known.
    Both methods use the same fractional Fourier map and logical phase encoder.
    maximum_motion_frames is an initial capacity, not an online frame limit.
    The caller owns and closes this serial workspace.
    """
    if stop_requested is not None and stop_requested():
        raise InterruptedError("SLM rearrangement preparation stopped")
    tolerance = float(support_tolerance)
    if method not in ("iterative", "lpi"):
        raise ValueError("method must be 'iterative' or 'lpi'")
    if not np.isfinite(tolerance) or tolerance < 1:
        raise ValueError("support_tolerance must be finite and >= 1")
    if (isinstance(maximum_motion_frames, bool) or int(maximum_motion_frames) != maximum_motion_frames
            or maximum_motion_frames < 1):
        raise ValueError("maximum_motion_frames must be a positive integer")
    if (isinstance(endpoint_iterations, bool) or int(endpoint_iterations) != endpoint_iterations
            or endpoint_iterations < 1):
        raise ValueError("endpoint_iterations must be a positive integer")
    geometry = prepare_rearrangement_geometry(
        source_yx, target_yx, shape_yx=shape_yx, minimum_separation=minimum_separation,
    )
    shape, source = geometry["shape_yx"], geometry["source_yx"]
    phase_center = np.asarray(np.asarray(shape) // 2 if phase_center_yx is None else phase_center_yx, np.float64)
    if phase_center.shape != (2,) or not np.all(np.isfinite(phase_center)):
        raise ValueError("phase_center_yx must be a finite Y,X pair")
    intensities = []
    for name, values, points in (("source", source_intensities, source),
                                 ("target", target_intensities, geometry["target_yx"])):
        value = np.ones(len(points), np.float32) if values is None else np.asarray(values, np.float32)
        if value.shape != (len(points),) or not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ValueError(f"{name}_intensities must be finite and positive, one per site")
        intensities.append(_readonly(value))
    if endpoint_data is not None:
        if "source_phase" in endpoint_data:
            canonical_phase(endpoint_data["source_phase"], shape)
        elif "source_phase_codes" in endpoint_data:
            codes = np.asarray(endpoint_data["source_phase_codes"])
            if codes.dtype != np.uint8 or codes.shape != shape:
                raise ValueError("endpoint_data source_phase_codes must be a native uint8 raster")
        else:
            raise ValueError("endpoint_data requires source_phase or source_phase_codes")
        if "source_coefficients" in endpoint_data:
            coefficient = np.asarray(endpoint_data["source_coefficients"])
            if (coefficient.shape != (len(source),) or not np.all(np.isfinite(coefficient))
                    or np.any(abs(coefficient) == 0)):
                raise ValueError("endpoint_data source_coefficients must be finite and nonzero, one per site")
    gpu = _prepare_rearrangement_gpu(geometry, pupil_amplitude, pupil_phase, stop_requested, method)
    try:
        cp, stream = gpu["cp"], gpu["stream"]
        native = gpu["resources"][1]
        with _rearrangement_gpu_active(gpu["keeper"]), stream:
            coefficient = None
            if endpoint_data is None:
                coefficient, pattern = _rearrangement_endpoint(
                    cp, source, shape, gpu["pupil_cpu"], intensities[0],
                    int(endpoint_iterations), int(seed), stop_requested,
                )
                latent = cp.exp(cp.complex64(1j) * cp.asarray(pattern, cp.float32))
            elif "source_phase" in endpoint_data:
                phase = canonical_phase(endpoint_data["source_phase"], shape)
                latent = cp.exp(cp.complex64(1j) * cp.asarray(phase + gpu["incident_cpu"], cp.float32))
            else:
                latent = None
            if latent is not None:
                gpu["kernels"]["encode"](((int(np.prod(shape)) + 255) // 256,), (256,),
                    (latent, native["physical_pupil"], native["incident"], native["optical"],
                     native["codes"], *map(np.int32, (*shape, 0))))
                codes = native["codes"].get()
            else:
                codes = np.asarray(endpoint_data["source_phase_codes"]).copy()
            if endpoint_data is None or "source_phase" not in endpoint_data:
                phase = codes.astype(np.float64) * (2 * np.pi / 256)
            optical = gpu["pupil_cpu"].astype(np.float64) * np.exp(
                1j * (phase.astype(np.float64) + gpu["incident_cpu"]))
            spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(optical)))
            actual = spectrum[tuple(source.T)]
            power = abs(actual) ** 2
            if not np.all(np.isfinite(power)) or np.any(power <= 0):
                raise ValueError("prepared source has a zero or invalid bright-site field")
            relative = power / intensities[0]
            ratio = float(relative.max() / relative.min())
            if method == "iterative" and ratio > tolerance:
                raise ValueError(f"source field exceeds authored intensity ratio {tolerance:g}: {ratio:.6g}")
            if endpoint_data is not None:
                coefficient = endpoint_data.get("source_coefficients")
            if coefficient is None:
                coefficient = actual / np.linalg.norm(actual)
            coefficient = np.array(coefficient, np.complex64, copy=True)
            coefficient /= np.linalg.norm(coefficient)
            gpu["coefficients"][:len(source)] = cp.asarray(coefficient)
            amplitude = np.sqrt(intensities[0])
            gpu["amplitude"][:len(source)] = cp.asarray(amplitude / np.linalg.norm(amplitude))
            gpu["initial_phase"] = cp.asarray(phase, cp.float32)
            _rearrangement_motion_capacity(gpu, int(maximum_motion_frames), stop_requested)
            target_phase = target_field = target_coefficient = None
            source_synthesis = source_reconstructed = None
            source_reconstruction_ratio = None
            source_endpoint_iterations, source_endpoint_ms = 0, 0.
            target_ratio = None
            target_endpoint_iterations, target_endpoint_ms = 0, 0.
            if method == "lpi":
                import time  # noqa: PLC0415

                auxiliary = _rearrangement_balance_endpoint(
                    gpu, source, intensities[0], actual, int(endpoint_iterations), tolerance, stop_requested)
                source_synthesis, source_reconstructed = auxiliary["coefficients"], auxiliary["field"]
                source_reconstruction_ratio = auxiliary["ratio"]
                source_endpoint_iterations, source_endpoint_ms = auxiliary["iterations"], auxiliary["timing_ms"]
                if source_reconstruction_ratio > tolerance or not np.isfinite(source_reconstruction_ratio):
                    raise ValueError(f"fixed-phase source synthesis exceeds authored intensity ratio {tolerance:g}: {source_reconstruction_ratio:.6g}")
                target_started = time.perf_counter()
                phase_iterations = min(12, int(endpoint_iterations))
                endpoint_coefficients, _ = _rearrangement_endpoint(
                    cp, geometry["target_yx"], shape, gpu["pupil_cpu"], intensities[1],
                    phase_iterations, int(seed) + 1, stop_requested)
                endpoint = _rearrangement_balance_endpoint(
                    gpu, geometry["target_yx"], intensities[1],
                    endpoint_coefficients,
                    int(endpoint_iterations), tolerance, stop_requested)
                target_coefficient, target_field, target_phase = endpoint["coefficients"], endpoint["field"], endpoint["phase"]
                target_ratio = endpoint["ratio"]
                target_endpoint_iterations = phase_iterations + endpoint["iterations"]
                target_endpoint_ms = (time.perf_counter() - target_started) * 1000
                if target_ratio > tolerance or not np.isfinite(target_ratio):
                    raise ValueError(f"fixed-phase target endpoint exceeds authored intensity ratio {tolerance:g}: {target_ratio:.6g}")
                gpu["coefficients"][:len(source)].set(coefficient, stream=stream)
                gpu["amplitude"][:len(source)].set(amplitude / np.linalg.norm(amplitude), stream=stream)
        free_memory, total_memory = cp.cuda.runtime.memGetInfo()
        return {
            **geometry, "gpu": gpu, "close": gpu["close"], "method": method,
            "phase_center_yx": _frozen(phase_center),
            "target_phase": None if target_phase is None else _frozen(target_phase),
            "target_field": None if target_field is None else _frozen(target_field.astype(np.complex64)),
            "target_synthesis_coefficients": target_coefficient,
            "target_support_intensity_ratio": target_ratio,
            "target_endpoint_iterations": target_endpoint_iterations, "target_endpoint_ms": target_endpoint_ms,
            "source_synthesis_coefficients": source_synthesis, "source_reconstruction_field": source_reconstructed,
            "source_reconstruction_intensity_ratio": source_reconstruction_ratio,
            "source_endpoint_iterations": source_endpoint_iterations, "source_endpoint_ms": source_endpoint_ms,
            "maximum_motion_frames": int(maximum_motion_frames),
            "gpu_info": {
                "device_name": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
                "device_memory_free_bytes": free_memory, "device_memory_total_bytes": total_memory,
                "host_output_staging_bytes": int(np.prod(shape)),
                "warm_buffer_bytes": int(gpu["keeper"]["scratch"].nbytes),
                "warm_interval_seconds": .020,
            },
            "source_intensities": intensities[0], "target_intensities": intensities[1],
            "source_coefficients": _frozen(coefficient), "source_field": _frozen(actual.astype(np.complex64)),
            "source_brightness": float(np.sqrt(np.sum(power) / np.sum(intensities[0], dtype=np.float64))),
            "source_support_intensity_ratio": ratio,
            "initial_phase_codes": _frozen(codes), "initial_phase": _frozen(phase),
        }
    except BaseException as error:
        try:
            gpu["close"]()
        except BaseException as cleanup_error:
            error.add_note(f"GPU preparation cleanup failed: {cleanup_error}")
        raise


_REARRANGEMENT_RESIDENT = None
_REARRANGEMENT_RESIDENT_LOCK = Lock()


def _rearrangement_inputs_equal(left, right):
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _rearrangement_inputs_equal(left[key], right[key]) for key in left)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def _close_resident_rearrangement():
    """Release the idle process-owned workspace, never an active daemon run."""
    global _REARRANGEMENT_RESIDENT
    if not _REARRANGEMENT_RESIDENT_LOCK.acquire(blocking=False):
        return
    try:
        previous, _REARRANGEMENT_RESIDENT = _REARRANGEMENT_RESIDENT, None
        if previous is not None:
            previous[1]["close"]()
    finally:
        _REARRANGEMENT_RESIDENT_LOCK.release()


@contextmanager
def acquire_rearrangement(source_yx, target_yx, **kwargs):
    """Exclusively borrow the one resident optical workspace for a whole run.

    Unlike explicit prepare_rearrangement, leaving this context keeps a healthy
    workspace warm. The caller must finish its workers and diagnostics before
    leaving; returned movies remain independently owned by that caller.
    """
    from copy import deepcopy  # noqa: PLC0415
    from inspect import signature  # noqa: PLC0415

    global _REARRANGEMENT_RESIDENT
    if not _REARRANGEMENT_RESIDENT_LOCK.acquire(blocking=False):
        raise RuntimeError("The resident SLM rearrangement workspace is already in use")
    prepared = None
    try:
        bound = signature(prepare_rearrangement).bind(source_yx, target_yx, **kwargs)
        bound.apply_defaults()
        inputs = dict(bound.arguments)
        stop = inputs.pop("stop_requested")
        capacity = inputs.pop("maximum_motion_frames")
        separation = _scalar(inputs.pop("minimum_separation"), "minimum_separation", nonnegative=True)
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError("maximum_motion_frames must be a positive integer")
        if stop is not None and stop():
            raise InterruptedError("SLM rearrangement preparation stopped")
        reused = False
        if _REARRANGEMENT_RESIDENT is not None:
            previous_inputs, prepared, device = _REARRANGEMENT_RESIDENT
            gpu = prepared["gpu"]
            if gpu["keeper"]["error"] is not None:
                raise RuntimeError("SLM GPU warming failed") from gpu["keeper"]["error"]
            reused = (device == gpu["cp"].cuda.Device().id
                      and _rearrangement_inputs_equal(previous_inputs, inputs))
            if not reused:
                _REARRANGEMENT_RESIDENT = None
                prepared["close"]()
                prepared = None
        if not reused:
            # Keep only independent optical inputs, never a run/plan/callback.
            saved_inputs = deepcopy(inputs)
            prepared = prepare_rearrangement(*bound.args, **bound.kwargs)
            _REARRANGEMENT_RESIDENT = (saved_inputs, prepared, prepared["gpu"]["keeper"]["device"])
            # Register after CUDA imports, before their exit-time teardown.
            atexit.unregister(_close_resident_rearrangement)
            atexit.register(_close_resident_rearrangement)
        else:
            prepared["minimum_separation"] = separation
            _rearrangement_motion_capacity(prepared["gpu"], int(capacity), stop)
            prepared["maximum_motion_frames"] = prepared["gpu"]["motion_capacity"]
        yield prepared, reused
        if prepared["gpu"]["keeper"]["error"] is not None:
            raise RuntimeError("SLM GPU warming failed") from prepared["gpu"]["keeper"]["error"]
    except BaseException as error:
        # SDK, classification, quality and Stop failures do not poison optics.
        # These are the existing CUDA-library errors and our own checked calls.
        gpu_fault = (type(error).__module__.startswith(("cupy_backends.cuda.", "cupy.cuda.", "cuda.bindings."))
                     or str(error).startswith(("SLM CUDA graph operation failed:", "SLM cuBLAS ")))
        if prepared is not None and (gpu_fault or prepared["gpu"].get("keeper", {}).get("error") is not None):
            _REARRANGEMENT_RESIDENT = None
            try:
                prepared["close"]()
            except BaseException as cleanup:
                error.add_note(f"Resident SLM GPU cleanup failed: {cleanup}")
        raise
    finally:
        _REARRANGEMENT_RESIDENT_LOCK.release()


def sample_rearrangement(
    prepared: Mapping[str, object], plan: Mapping[str, object], *, motion_frames: int | None = None,
    maximum_step: float | None = None, step_path: object | None = None,
) -> dict[str, object]:
    """Choose N, then share one source-removal map and waypoint-safe movement.

    Auto only chooses N; giving that N explicitly returns identical positions
    and phase progress. step_path supplies the same knots in the caller's
    distance units (the Task uses native camera sensor pixels).
    """
    source, target = prepared["source_yx"], prepared["target_yx"]
    selected = np.asarray(plan["source_indices"], np.intp)
    destinations = np.asarray(plan["target_indices"], np.intp)
    removed = np.asarray(plan["removed_source_indices"], np.intp)
    path = np.asarray(plan["motion_yx"], np.float64)
    fractions = np.asarray(plan["fraction"], np.float64)
    if (path.ndim != 3 or path.shape[1:] != (len(selected), 2)
            or not np.array_equal(path[0], source[selected])
            or not np.array_equal(path[-1], target[destinations])):
        raise ValueError("plan endpoints do not match the prepared source roster and target")
    if not len(selected):
        # Empty occupancy does not issue a source-removal map. In particular,
        # Auto must not validate an unplayed fade that Fixed/compute skips.
        return {
            "motion_frames": 0, "motion_yx": _frozen(path[:1]),
            "fraction": _frozen(np.zeros(1)), "sites_yx": _frozen(np.empty((0, len(source), 2))),
            "movement_fraction": _frozen(np.empty(0)), "fade_frames": 0,
            "clearance": np.inf, "fade_clearance": np.inf, "surplus_stationary_clearance": np.inf,
            "maximum_step": 0., "recommended_motion_frames": 0,
        }
    if np.any(np.diff(fractions) <= 0) or fractions[0] != 0 or fractions[-1] != 1:
        raise ValueError("plan fractions must increase from zero to one")
    original_shape = path.shape
    path, fractions, keep = _rearrangement_motion_timing(path)
    faded = np.flatnonzero(~np.isin(np.arange(len(source)), selected))
    spatial_motion = bool(np.any(path != path[:1]))
    fade = 1 if len(faded) else 0
    widths = np.diff(fractions)
    def allocate(count):
        # These ceilings include every highest-priority width/n allocation
        # needed by N. Remove only the excess lowest-priority allocations;
        # there are fewer than len(widths), instead of looping N times.
        subdivisions = np.maximum(1, np.ceil(count * widths)).astype(np.int64)
        for _ in range(int(subdivisions.sum()) - count):
            priority = np.divide(widths, subdivisions - 1,
                                 out=np.full(len(widths), np.inf), where=subdivisions > 1)
            last = len(widths) - 1 - int(np.argmin(priority[::-1]))
            subdivisions[last] -= 1
        return subdivisions

    automatic = maximum_step is not None
    if automatic:
        if motion_frames is not None:
            raise ValueError("choose motion_frames or maximum_step, not both")
        maximum_step = _scalar(maximum_step, "maximum_step")
        if maximum_step <= 0:
            raise ValueError("maximum_step must be positive")
        coordinates = np.asarray(plan["motion_yx"], np.float64) if step_path is None else np.asarray(step_path, np.float64)
        if coordinates.shape != original_shape or not np.all(np.isfinite(coordinates)):
            raise ValueError("step_path must contain the plan knots in the chosen distance units")
        coordinates = coordinates[keep]
        distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=-1)
        longest = float(np.max(np.sum(distances, axis=0), initial=0.))
        count = (fade + max(len(widths), int(np.ceil(longest / maximum_step)))
                 if spatial_motion else max(1, fade))
        required = np.maximum(1, np.ceil(np.max(distances, axis=1, initial=0.) / maximum_step)).astype(np.int64)
        if spatial_motion and np.any(required > 1):
            # All required increments have priority >= this value. Counting
            # every such increment bounds N directly; binary search chooses
            # the smallest N under the same allocation used by Fixed.
            priority = float(np.min(widths[required > 1] / (required[required > 1] - 1)))
            lo = count - fade
            hi = max(lo, int(np.maximum(1, np.ceil(widths / priority)).sum()) + len(widths))
            while lo < hi:
                middle = (lo + hi) // 2
                if np.all(allocate(middle) >= required):
                    hi = middle
                else:
                    lo = middle + 1
            count = fade + lo
    else:
        if (motion_frames is None or isinstance(motion_frames, bool)
                or int(motion_frames) != motion_frames or motion_frames < 1):
            raise ValueError("motion_frames must be a positive integer")
        count = int(motion_frames)
    fade_frames = min(fade, count)
    if spatial_motion and count <= fade_frames:
        raise ValueError("motion_frames must leave a movement frame after the source-removal map")
    if spatial_motion:
        moving_count = count - fade_frames
        if moving_count < len(widths):
            raise ValueError(f"motion_frames must include {fade_frames + len(widths)} maps to preserve all path waypoints")
        # The same deterministic allocation in both modes keeps every corner:
        # no frame-to-frame chord can cut through a planned clearance boundary.
        subdivisions = allocate(moving_count)
        progress = np.concatenate((np.zeros(fade_frames), *(
            np.linspace(a, b, int(n) + 1)[1:]
            for a, b, n in zip(fractions[:-1], fractions[1:], subdivisions, strict=True))))
    else:
        progress = np.zeros(count)
    segment = np.clip(np.searchsorted(fractions, progress, side="right") - 1, 0, len(path) - 2)
    mix = (progress - fractions[segment]) / (fractions[segment + 1] - fractions[segment])
    moving = path[segment] + mix[:, None, None] * (path[segment + 1] - path[segment])
    actual_path = np.concatenate((path[:1], moving))
    clearance = rearrangement_clearance(actual_path)
    if clearance < prepared["minimum_separation"]:
        raise ValueError(f"emitted trajectory clearance {clearance:g} is below {prepared['minimum_separation']:g}")
    fade_clearance = stationary_clearance = np.inf
    if len(faded):
        # Empty but still illuminated traps also participate in the transition.
        stationary = np.broadcast_to(source[faded], (fade_frames + 1, len(faded), 2))
        fade_clearance = rearrangement_clearance(np.concatenate((actual_path[:fade_frames + 1], stationary), axis=1))
        if fade_clearance < prepared["minimum_separation"]:
            raise ValueError(f"trajectory clearance during surplus fade {fade_clearance:g} "
                             f"is below {prepared['minimum_separation']:g}")
    if len(removed):
        stationary = np.broadcast_to(source[removed], (len(actual_path), len(removed), 2))
        stationary_clearance = rearrangement_clearance(np.concatenate((actual_path, stationary), axis=1))
    sites = np.broadcast_to(source, (count, len(source), 2)).astype(np.float64).copy()
    sites[:, selected] = moving
    velocity = np.max(np.linalg.norm(np.diff(path, axis=0), axis=-1)
                      / np.diff(fractions)[:, None], initial=0.)
    return {
        "motion_frames": count,
        "motion_yx": _frozen(actual_path), "fraction": _frozen(np.arange(count + 1) / count),
        "sites_yx": _frozen(sites), "movement_fraction": _frozen(progress), "fade_frames": fade_frames,
        "clearance": clearance, "fade_clearance": fade_clearance,
        "surplus_stationary_clearance": stationary_clearance,
        "maximum_step": float(np.max(np.linalg.norm(np.diff(actual_path, axis=0), axis=-1), initial=0.)),
        "recommended_motion_frames": max(1, int(np.ceil(velocity)) + fade_frames),
    }


def rearrangement_is_noop(prepared: Mapping[str, object], plan: Mapping[str, object]) -> bool:
    """One decision for device admission and computation of an existing plan."""
    selected = np.asarray(plan["source_indices"], np.intp)
    if not len(selected):
        return True
    path = np.asarray(plan["motion_yx"])
    return (len(selected) == len(prepared["source_yx"]) and not np.any(path != path[:1])
            and np.array_equal(prepared["source_intensities"][selected],
                               prepared["target_intensities"][np.asarray(plan["target_indices"], np.intp)]))


def compute_rearrangement(
    prepared: dict[str, object], plan: Mapping[str, object], *,
    motion_frames: int = 16, iterations: int | None = None,
    support_tolerance: float = SPOT_SUPPORT_TOLERANCE,
    motion_support_tolerance: float = 1.10,
    require_converged: bool = True, stop_requested: Callable[[], bool] | None = None,
    frame_ready: Callable[[int, np.ndarray], None] | None = None,
    sampled: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Emit N native maps with one continuous coefficient trajectory.

    Both methods remove unused beams in one source-position map, then follow the
    same sampled fractional path, with N total maps. Both use the same full
    focal-plane projection and actual-code acceptance. LPI constrains actual
    bright optical phases along the endpoint interpolation; Iterative lets
    those phases follow the current field. Empty/no-change plans hold
    the exact input phase and return an empty sequence. frame_ready receives each
    immutable host map in order; its wait/error stays on this same solve path.
    Intermediate maps use motion_support_tolerance; the actual last map uses
    support_tolerance, including a one-map sequence. Neither an extinguished
    synthesis coefficient nor uniform bright centers proves a dark background.
    The caller must stop playback on a later failure.
    """
    import time  # noqa: PLC0415

    started = time.perf_counter()
    if not prepared["gpu"]:
        raise RuntimeError("SLM rearrangement workspace is closed")
    if stop_requested is not None and stop_requested():
        raise InterruptedError("SLM rearrangement stopped")
    if (isinstance(motion_frames, bool) or int(motion_frames) != motion_frames or motion_frames < 0
            or (motion_frames == 0 and not rearrangement_is_noop(prepared, plan))
            or (iterations is not None and
                (isinstance(iterations, bool) or int(iterations) != iterations or iterations < 0))):
        raise ValueError("motion_frames must be positive and iterations nonnegative integers")
    motion_frames = int(motion_frames)
    method = prepared["method"]
    tolerance = float(support_tolerance)
    motion_tolerance = float(motion_support_tolerance)
    if not np.isfinite(tolerance) or tolerance < 1 or not np.isfinite(motion_tolerance) or motion_tolerance < 1:
        raise ValueError("support_tolerance and motion_support_tolerance must be finite and >= 1")
    source, target, shape = prepared["source_yx"], prepared["target_yx"], prepared["shape_yx"]
    source_indices = np.asarray(plan["source_indices"], np.intp)
    target_indices = np.asarray(plan["target_indices"], np.intp)
    removed = np.asarray(plan["removed_source_indices"], np.intp)
    path = np.asarray(plan["motion_yx"], np.float64)
    if (path.ndim != 3 or path.shape[1:] != (len(source_indices), 2)
            or not np.array_equal(path[0], source[source_indices])
            or not np.array_equal(path[-1], target[target_indices])):
        raise ValueError("plan endpoints do not match the prepared source roster and target")
    source_count, selected_count = len(source), len(source_indices)
    metadata = {
        "motion_support_tolerance": motion_tolerance,
        "method": method, "phase_center_yx": prepared["phase_center_yx"],
        "source_field": prepared["source_field"], "target_field": prepared["target_field"],
        "target_phase": prepared["target_phase"], "target_requested_intensities": prepared["target_intensities"],
        "target_synthesis_coefficients": prepared["target_synthesis_coefficients"],
        "source_synthesis_coefficients": prepared["source_synthesis_coefficients"],
        "source_reconstruction_field": prepared["source_reconstruction_field"],
        "source_coefficient_basis": "fixed-source-phase-synthesis" if method == "lpi" else "source-endpoint-coefficients",
        "target_coefficient_basis": "independent-target-phase-synthesis" if method == "lpi" else "authored-amplitude-source-phase",
        "field_phase_reference": "fft-center", "phase_interpolation": "shortest-modulo-2pi" if method == "lpi" else "adaptive-field-phase",
        "quality_scope": "bright-sites-and-full-native-background-with-natural-main-lobe-envelope",
    }
    if rearrangement_is_noop(prepared, plan):
        return {
            **plan, **metadata, "noop": True, "phase_codes": _frozen(np.empty((0, *shape), np.uint8)),
            "motion_yx": _frozen(path[:1]), "fraction": _frozen(np.zeros(1)),
            "movement_fraction": np.empty(0),
            "frame_support_tolerances": np.empty(0),
            "sites_yx": np.empty((0, source_count, 2)), "actual_fields": np.empty((0, source_count), np.complex64),
            "desired_amplitudes": np.empty((0, source_count)), "active_sites": np.empty((0, source_count), bool),
            "motion_frames": 0, "fade_frames": 0, "iterations": (), "emitted_frame_count": 0,
            "verified_frame_reuses": 0,
            "frame_solve_ms": np.empty(0), "frame_copy_ms": np.empty(0), "frame_ready_ms": np.empty(0),
            "clearance": rearrangement_clearance(path), "fade_clearance": np.inf,
            "surplus_stationary_clearance": np.inf, "maximum_step": 0., "recommended_motion_frames": 0,
            "support_intensity_ratios": np.empty(0), "background_intensity_ratios": np.empty(0),
            "discard_intensity_ratios": np.empty(0), "discard_reference_limit": .01,
            "discard_converged": True, "field_projection_updates": (),
            "brightness_minimum_to_initial": np.empty(0), "brightness_maximum_to_initial": np.empty(0),
            "brightness_mean_to_initial": np.empty(0), "phase_change_from_initial_rms_rad": np.empty(0),
            "focal_phase_error_rms_rad": np.empty(0),
            "phase_step_max_rad": np.empty(0), "center_sample_power_proxy": np.empty(0),
            "pupil_phase_step_rms_rad": np.empty(0),
            "converged": True, "quality_accepted": True, "quality_evaluated": True,
            "retained_intensity_ratios": np.empty(0),
            "support_tolerance": tolerance, "phase_encoding": "uint8:2pi/256",
            "release_verified": False, "recommended_release_hold_frames": 0,
            "timing_ms": {"prepare_frame_state": 0., "solve": 0., "copy": 0.,
                          "callback": 0., "first_frame_ready": 0., "first_frame_solve": 0., "first_frame_copy": 0.,
                          "total": (time.perf_counter() - started) * 1000},
        }
    if sampled is None:
        sampled = sample_rearrangement(prepared, plan, motion_frames=motion_frames)
    elif sampled["motion_frames"] != motion_frames or len(sampled["sites_yx"]) != motion_frames:
        raise ValueError("sampled trajectory must match motion_frames")
    actual_path, sites = sampled["motion_yx"], sampled["sites_yx"]
    frame_tolerances = np.full(motion_frames, motion_tolerance)
    frame_tolerances[-1] = tolerance
    metadata["frame_support_tolerances"] = _frozen(frame_tolerances)
    progress = sampled["movement_fraction"]
    clearance = sampled["clearance"]
    faded = np.flatnonzero(~np.isin(np.arange(source_count), source_indices))
    fade_frames = sampled["fade_frames"]
    fade_clearance, stationary_clearance = sampled["fade_clearance"], sampled["surplus_stationary_clearance"]
    maximum_step, recommended = sampled["maximum_step"], sampled["recommended_motion_frames"]
    initial = prepared["source_field"]
    initial_amplitude = abs(initial).astype(np.float64)
    desired = np.broadcast_to(initial_amplitude, (motion_frames, source_count)).copy()
    destination_amplitude = prepared["source_brightness"] * np.sqrt(prepared["target_intensities"][target_indices])
    # Optical weights can change even when no trap changes position. Keep that
    # authored transition separate from the geometrical movement fraction.
    amplitude_fraction = progress if np.any(path != path[:1]) else np.arange(1, motion_frames + 1) / motion_frames
    amplitude_progress = (np.minimum(1., np.arange(1, motion_frames + 1) / fade_frames)
                          if fade_frames else amplitude_fraction)
    desired[:, source_indices] = ((1 - amplitude_progress[:, None]) * initial_amplitude[source_indices]
                                  + amplitude_progress[:, None] * destination_amplitude)
    if fade_frames:
        fade = np.maximum(0., 1 - np.arange(1, motion_frames + 1) / fade_frames)
        desired[:, faded] *= fade[:, None]
    positive = desired > 0
    normalized = (desired / np.linalg.norm(desired, axis=1, keepdims=True)).astype(np.float32)
    phase = np.angle(initial)
    if method == "lpi":
        endpoint_coefficient = prepared["target_synthesis_coefficients"][target_indices]
        # A subset keeps these prescribed phases. Its amplitudes are an
        # initial guess, corrected by the same gate as every emitted map;
        # do not block the first map on a separate, unplayed endpoint solve.
        source_amplitude = abs(prepared["source_synthesis_coefficients"]).astype(np.float64)
        start_coefficient = prepared["source_synthesis_coefficients"][source_indices]
        if fade_frames:
            if np.any(path != path[:1]):
                # This is an initial guess, not a separately displayed endpoint.
                # The actual fade frames own their fixed-phase amplitude solve.
                start_coefficient = start_coefficient * np.sqrt(
                    prepared["target_intensities"][target_indices] / prepared["source_intensities"][source_indices])
                start_coefficient /= np.linalg.norm(start_coefficient)
            else:
                start_coefficient = endpoint_coefficient
        metadata.update(start_synthesis_coefficients=_frozen(start_coefficient))
        spectrum_amplitude = np.broadcast_to(source_amplitude, desired.shape).copy()
        moving_amplitude = ((1 - amplitude_fraction[:, None]) * abs(start_coefficient)
                            + amplitude_fraction[:, None] * abs(endpoint_coefficient))
        spectrum_amplitude[:, source_indices] = (
            (1 - amplitude_progress[:, None]) * source_amplitude[source_indices]
            + amplitude_progress[:, None] * moving_amplitude) if fade_frames else moving_amplitude
        if fade_frames:
            spectrum_amplitude[:, faded] *= fade[:, None]
        gauge_offset = (prepared["phase_center_yx"] - np.asarray(shape) // 2) / shape
        source_gauge = 2 * np.pi * np.sum((source[source_indices] - np.asarray(shape) // 2) * gauge_offset, axis=1)
        target_gauge = 2 * np.pi * np.sum((target[target_indices] - np.asarray(shape) // 2) * gauge_offset, axis=1)
        # Interpolate measured focal phases, not the synthesis coefficient
        # phases: phase-only projection makes those two quantities different.
        phase_delta = np.angle(prepared["target_field"][target_indices] * np.exp(1j * target_gauge)
                               * (initial[source_indices] * np.exp(1j * source_gauge)).conj())
        # A retained trap can stay in place while its phase changes to the
        # independently optimized final array. The exact whole-array no-op
        # already returned above; it must not suppress this optical transition.
        phase_progress = amplitude_fraction
        if not np.any(path != path[:1]) and fade_frames and motion_frames > fade_frames:
            phase_progress = np.maximum(0., np.arange(1, motion_frames + 1) - fade_frames) / (motion_frames - fade_frames)
        spectrum_phase = np.broadcast_to(phase, desired.shape).copy()
        current_gauge = 2 * np.pi * np.sum(
            (sites[:, source_indices] - np.asarray(shape) // 2) * gauge_offset, axis=-1)
        spectrum_phase[:, source_indices] = (phase[source_indices] + source_gauge
                                            + phase_progress[:, None] * phase_delta - current_gauge)
        coefficient_values = (spectrum_amplitude * np.exp(1j * spectrum_phase)).astype(np.complex64)
        coefficient_values /= np.linalg.norm(coefficient_values, axis=1, keepdims=True)
    else:
        coefficient_values = (abs(prepared["source_coefficients"])[None]
                              * (desired / initial_amplitude[None]) * np.exp(1j * phase)[None]).astype(np.complex64)
        coefficient_values /= np.linalg.norm(coefficient_values, axis=1, keepdims=True)
    gpu = prepared["gpu"]
    cp, stream, number = gpu["cp"], gpu["stream"], gpu["number"]
    native = gpu["resources"][1]
    _rearrangement_motion_capacity(gpu, motion_frames, stop_requested)
    prepared["maximum_motion_frames"] = gpu["motion_capacity"]
    total_updates = (5 if 2 in gpu["resources"] else 10) if iterations is None else int(iterations)
    coarse_updates = min(3, max(0, total_updates - 2)) if 2 in gpu["resources"] else 0
    native_updates = total_updates - coarse_updates
    iteration_counts = np.full(motion_frames, total_updates if method == "iterative" else 0, np.int32)
    pixels = motion_frames * int(np.prod(shape))
    indices, frequencies, counts, offsets = _rearrangement_bind(gpu, sites)
    with _rearrangement_gpu_active(gpu["keeper"]), stream:
        gpu["frame_index"].fill(0)
        coefficients = gpu["motion_coefficients"][:motion_frames]
        movie = gpu["motion_codes"][:motion_frames]
        actual_gpu = gpu["motion_actual"][:motion_frames]
        coefficients.set(coefficient_values, stream=stream)
        gpu["motion_amplitudes"][:motion_frames].set(normalized, stream=stream)
        for factor, values in indices.items():
            gpu["motion_indices"][factor][:motion_frames].set(values, stream=stream)
        gpu["motion_frequencies"][:len(frequencies)].set(frequencies, stream=stream)
        gpu["motion_offsets"][:motion_frames + 1].set(offsets, stream=stream)
        host = np.empty((motion_frames, *shape), np.uint8)
        codes = np.frombuffer(memoryview(host).toreadonly(), np.uint8, count=pixels).reshape(host.shape)
        fields = np.empty((motion_frames, number), np.complex64)
        background_limits = np.zeros(motion_frames)
        phase_errors = np.zeros(motion_frames)
        envelope_ratios = np.zeros(motion_frames)
        projection_updates = np.zeros(motion_frames, np.int32)
        baseline_updates = np.zeros(motion_frames, np.int32)
        frame_solve_ms, frame_copy_ms, frame_ready_ms = (np.zeros(motion_frames) for _ in range(3))
        callback_ms, emitted_count, verified_reuses = 0., 0, 0
        reference_intensity = float(np.min(abs(initial.astype(np.complex128)) ** 2))
        publication_open = True
        after_prepare = time.perf_counter()
        for index in range(motion_frames):
            frame_started = time.perf_counter()
            tolerance = float(frame_tolerances[index])
            if stop_requested is not None and stop_requested():
                stream.synchronize()
                raise InterruptedError("SLM rearrangement stopped")
            band = (int(counts[index]) + 15) // 16 * 16
            same_positions = (index and np.array_equal(sites[index], sites[index - 1])
                              and np.array_equal(positive[index], positive[index - 1]))
            mask = positive[index]
            reuse_verified = False
            if (same_positions and np.array_equal(coefficient_values[index], coefficient_values[index - 1])
                    and background_limits[index - 1] <= .01 and phase_errors[index - 1] <= .01
                    and envelope_ratios[index - 1] <= 1.05):
                relative = abs(fields[index - 1, mask].astype(np.complex128)) ** 2 / desired[index, mask] ** 2
                reuse_verified = bool(relative.max() / relative.min() <= tolerance)
            if reuse_verified:
                movie[index] = movie[index - 1]
                fields[index] = fields[index - 1]
                actual_gpu[index] = actual_gpu[index - 1]
                coefficients[index] = coefficients[index - 1]
                background_limits[index] = background_limits[index - 1]
                phase_errors[index] = phase_errors[index - 1]
                envelope_ratios[index] = envelope_ratios[index - 1]
                iteration_counts[index] = 0
                verified_reuses += 1
            else:
                gpu["frame_index"].fill(index)
                if method == "lpi":
                    # LPI supplies a single synthesis as the initial guess.
                    # The common full-field solve owns brightness, actual phase
                    # and background together; do not balance twice.
                    _rearrangement_load_frame(gpu, preserve_phase=True)
                    _rearrangement_select_roots(gpu["measurement"], band)
                    _rearrangement_propagate(gpu, gpu["measurement"], band, "encode")
                    movie[index] = native["codes"]
                    coefficients[index] = gpu["coefficients"][:number]
                    baseline_updates[index] = 0
                elif iterations is None:
                    gpu["graphs"][band].launch(stream)
                    baseline_updates[index] = total_updates
                else:
                    _rearrangement_load_frame(gpu, warm_previous=True)
                    if coarse_updates:
                        _rearrangement_amplitude_updates(gpu, gpu["resources"][2], band, coarse_updates)
                    _rearrangement_amplitude_updates(gpu, native, band, native_updates)
                    _rearrangement_propagate(gpu, native, band, "encode")
                    _rearrangement_select_roots(gpu["measurement"], band)
                    _rearrangement_propagate(gpu, gpu["measurement"], band, "forward")
                    _rearrangement_store_frame(gpu)
                    baseline_updates[index] = total_updates
                field, background_limits[index], phase_errors[index], updates, envelope_ratios[index] = _rearrangement_background_projection(
                    gpu, sites[index], desired[index], movie[index], reference_intensity, tolerance,
                    prescribed_phase=np.angle(coefficient_values[index]) if method == "lpi" else None,
                    iterations=128 if iterations is None else int(iterations),
                    stop_requested=stop_requested, warm_start=index > 0)
                fields[index] = field
                actual_gpu[index].set(field, stream=stream)
                movie[index] = native["codes"]
                projection_updates[index] = updates
                iteration_counts[index] = baseline_updates[index] + updates
            relative = abs(fields[index, mask].astype(np.complex128)) ** 2 / desired[index, mask] ** 2
            bright_ratio = float(relative.max() / relative.min())
            valid = (np.isfinite(bright_ratio) and bright_ratio <= tolerance
                     and background_limits[index] <= .01 and phase_errors[index] <= .01
                     and envelope_ratios[index] <= 1.05)
            publication_open = publication_open and valid
            if require_converged and not valid:
                raise RuntimeError(f"SLM {method} frame {index} did not meet optical constraints; "
                                   f"bright {bright_ratio:.6g} (limit {tolerance:g}), "
                                   f"background/initial {background_limits[index]:.6g} (limit 0.01), "
                                   f"focal phase error {phase_errors[index]:.6g} rad (limit 0.01), "
                                   f"main-lobe envelope {envelope_ratios[index]:.6g} (limit 1.05)")
            frame_solve_ms[index] = (time.perf_counter() - frame_started) * 1000
            copy_started = time.perf_counter()
            movie[index].get(out=gpu["host_frame"], stream=stream, blocking=True)
            host[index] = gpu["host_frame"]
            frame_copy_ms[index] = (time.perf_counter() - copy_started) * 1000
            frame_ready_ms[index] = (time.perf_counter() - started) * 1000
            if stop_requested is not None and stop_requested():
                raise InterruptedError("SLM rearrangement stopped")
            if frame_ready is not None and publication_open:
                # This callback consumes an already copied host map. Network
                # backpressure/long holds are idle time for this GPU owner.
                keeper = gpu["keeper"]
                with keeper["lock"]:
                    keeper["active"] -= 1
                callback_started = time.perf_counter()
                try:
                    frame_ready(index, codes[index])
                finally:
                    with keeper["lock"]:
                        keeper["active"] += 1
                callback_ms += (time.perf_counter() - callback_started) * 1000
                if keeper["error"] is not None:
                    raise RuntimeError("SLM GPU warming failed") from keeper["error"]
                emitted_count += 1
    intensity = abs(fields.astype(np.complex128)) ** 2
    relative = np.divide(intensity, desired ** 2, out=np.zeros(desired.shape), where=positive)
    ratios = relative.max(axis=1) / np.min(np.where(positive, relative, np.inf), axis=1)
    selected_fields = fields[:, source_indices]
    retained_relative = relative[:, source_indices]
    retained_ratios = retained_relative.max(axis=1) / retained_relative.min(axis=1)
    brightness = abs(selected_fields / initial[source_indices][None]) ** 2
    phase_change = np.angle(selected_fields * initial[source_indices].conj()[None])
    phase_step = np.angle(selected_fields * np.concatenate((initial[source_indices][None], selected_fields[:-1])).conj())
    with _rearrangement_gpu_active(gpu["keeper"]):
        accepted_coefficients = coefficients.get(stream=stream)
    metadata.update(endpoint_field=_frozen(fields[-1, source_indices]),
                    endpoint_synthesis_coefficients=_frozen(accepted_coefficients[-1, source_indices]),
                    endpoint_support_intensity_ratio=float(retained_ratios[-1]),
                    endpoint_iterations=int(iteration_counts[-1]), endpoint_balance_ms=0.)
    total_ms = (time.perf_counter() - started) * 1000
    prepare_ms, copy_ms = (after_prepare - started) * 1000, float(frame_copy_ms.sum())
    return {
        **plan, **sampled, **metadata, "noop": False, "phase_codes": codes,
        "actual_fields": _frozen(fields), "desired_amplitudes": _frozen(desired), "active_sites": _frozen(positive),
        "desired_spectrum_coefficients": _frozen(coefficient_values),
        # These are the sparse synthesis initial guesses, not a reconstruction
        # of the full-field corrected hologram.
        "synthesis_coefficients": _frozen(accepted_coefficients),
        "motion_frames": motion_frames, "iterations": tuple(map(int, iteration_counts)),
        "emitted_frame_count": emitted_count, "verified_frame_reuses": verified_reuses,
        "frame_solve_ms": frame_solve_ms, "frame_copy_ms": frame_copy_ms, "frame_ready_ms": frame_ready_ms,
        "quality_evaluated": True, "quality_accepted": bool(publication_open), "converged": bool(publication_open),
        "support_tolerance": tolerance, "discard_reference_limit": .01, "discard_converged": None,
        "support_intensity_ratios": _frozen(ratios), "retained_intensity_ratios": _frozen(retained_ratios),
        "quality_scope": "bright-sites-and-full-native-background-with-natural-main-lobe-envelope",
        "background_limit_ratios": _frozen(background_limits), "background_reference_intensity": reference_intensity,
        "focal_phase_error_max_rad": _frozen(phase_errors),
        "main_lobe_envelope_ratios": _frozen(envelope_ratios),
        "background_intensity_ratios": np.empty(0), "discard_intensity_ratios": np.empty(0),
        "field_projection_updates": tuple(map(int, projection_updates)),
        "brightness_minimum_to_initial": _frozen(brightness.min(axis=1)),
        "brightness_maximum_to_initial": _frozen(brightness.max(axis=1)),
        "brightness_mean_to_initial": _frozen(brightness.mean(axis=1)),
        "center_sample_power_proxy": _frozen(intensity[:, source_indices].sum(axis=1)
            / (np.prod(shape) * gpu["pupil_energy"])),
        "phase_change_from_initial_rms_rad": _frozen(np.sqrt(np.mean(phase_change ** 2, axis=1))),
        "focal_phase_error_rms_rad": np.empty(0),
        "phase_step_max_rad": _frozen(np.max(abs(phase_step), axis=1)),
        "pupil_phase_step_rms_rad": np.empty(0), "phase_encoding": "uint8:2pi/256",
        "release_verified": False, "recommended_release_hold_frames": 1 if len(removed) else 0,
        "timing_ms": {"prepare_frame_state": prepare_ms, "solve": total_ms - prepare_ms - copy_ms - callback_ms,
                      "copy": copy_ms, "callback": callback_ms, "first_frame_ready": float(frame_ready_ms[0]),
                      "first_frame_solve": float(frame_solve_ms[0]), "first_frame_copy": float(frame_copy_ms[0]),
                      "total": total_ms},
    }


def _psf_main_lobe(psf, center):
    """Connected one-percent main lobe of this numeric pupil's single trap."""
    from scipy.ndimage import label  # noqa: PLC0415

    components, _ = label(psf >= .01 * psf[center], structure=np.ones((3, 3), bool))
    return components == components[center]


def _rearrangement_focal_peaks(intensity, maxima, sites, psf_support):
    """Recognize one main peak per target, never an entire bright rectangle.

    The FFT plane is periodic. Equal neighboring samples form one plateau;
    every other local maximum remains eligible as unwanted light, including
    a second peak inside a target's natural-PSF neighborhood.
    """
    sites = np.asarray(sites, np.float64)
    shape = np.asarray(intensity.shape)
    offsets = np.argwhere(psf_support) - shape // 2
    # Include the two neighboring integer samples of a fractional center.
    lower, upper = offsets.min(axis=0) - 1, offsets.max(axis=0) + 1
    yy, xx = np.mgrid[lower[0]:upper[0] + 1, lower[1]:upper[1] + 1]
    neighborhood = np.stack((yy.ravel(), xx.ravel()), axis=1)
    peak_yx = np.full((len(sites), 2), np.nan)
    peak_intensity = np.full(len(sites), np.nan)
    target_region = np.zeros(intensity.shape, bool)
    for index, site in enumerate(sites):
        points = np.floor(site).astype(np.int64) + neighborhood
        delta = points - site
        support_index = (np.rint(delta).astype(np.int64) + shape // 2) % shape
        supported = psf_support[tuple(support_index.T)]
        points = points[supported] % shape
        target_region[tuple(points.T)] = True
        # A candidate belongs to its closest authored target, not whichever
        # target happens to be visited first.
        distance = points[:, None] - sites[None]
        distance -= np.rint(distance / shape) * shape
        nearest = np.argmin(np.sum(distance ** 2, axis=-1), axis=1)
        points = points[nearest == index]
        allowed = maxima[tuple(points.T)]
        if not np.any(allowed):
            continue
        candidates = points[allowed]
        distance = candidates - site
        distance -= np.rint(distance / shape) * shape
        chosen = int(np.argmin(np.sum(distance ** 2, axis=1)))
        point = candidates[chosen]
        value = intensity[tuple(point)]
        # Only recognized main peaks need plateau identities. Do not label
        # every tiny noise peak across the million-pixel focal plane.
        pending, members = [tuple(point)], []
        while pending:
            y, x = pending.pop()
            if not maxima[y, x] or intensity[y, x] != value:
                continue
            maxima[y, x] = False
            members.append((y, x))
            pending.extend(((y + dy) % shape[0], (x + dx) % shape[1])
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        # Exact half-pixel peaks straddling the FFT edge are one plateau.
        members = np.asarray(members)
        relative = (members - point).astype(np.float64)
        relative -= np.rint(relative / shape) * shape
        peak_yx[index] = (point + relative.mean(axis=0)) % shape
        peak_intensity[index] = intensity[tuple(point)]
    if np.any(maxima):
        flat = int(np.argmax(np.where(maxima, intensity, -np.inf)))
        unwanted_yx = np.asarray(np.unravel_index(flat, intensity.shape), np.float64)
        unwanted_intensity = float(intensity.flat[flat])
    else:
        unwanted_yx, unwanted_intensity = np.full(2, np.nan), 0.0
    return peak_yx, peak_intensity, unwanted_yx, unwanted_intensity, target_region


def rearrangement_diagnostics(
    prepared: Mapping[str, object], result: Mapping[str, object], *,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Measure delivered maps with the shared physical-pupil Fourier model.

    Called after playback/second photo, not a hidden LPI publication gate.
    Fields use the FFT-center phase reference; phase_center_yx only specifies
    the LPI interpolation gauge. Phase change from the initial field is distinct
    from LPI focal-field error relative to its prescribed synthesis phase.
    Fourfold zero padding resolves peaks between native Fourier samples; it
    does not resize the SLM pupil. No optical response or survival is inferred.
    """
    import time  # noqa: PLC0415

    started = time.perf_counter()
    gpu = prepared["gpu"]
    if not gpu:
        raise RuntimeError("SLM rearrangement workspace is closed")
    count, source_count = len(result["phase_codes"]), len(prepared["source_yx"])
    source_indices = np.asarray(result["source_indices"], np.intp)
    removed = np.asarray(result["removed_source_indices"], np.intp)
    fields = np.asarray(result["actual_fields"])
    discard = np.array(result["discard_intensity_ratios"], copy=True)
    pupil_steps = np.asarray(result["pupil_phase_step_rms_rad"])
    if discard.shape != (count,):
        discard = np.zeros(count)
    if fields.shape != (count, source_count):
        cp, stream, native = gpu["cp"], gpu["stream"], gpu["resources"][1]
        _rearrangement_motion_capacity(gpu, count, stop_requested)
        fields = np.empty((count, source_count), np.complex64)
        query = {**gpu["measurement"], "number": source_count}
        area = int(np.prod(prepared["shape_yx"]))
        with _rearrangement_gpu_active(gpu["keeper"]), stream:
            gpu["motion_codes"][:count].set(np.asarray(result["phase_codes"]), stream=stream)
            for index, sites in enumerate(result["sites_yx"]):
                if stop_requested is not None and stop_requested():
                    raise InterruptedError("SLM rearrangement diagnostics stopped")
                gpu["kernels"]["field_project"](((area + 255) // 256,), (256,),
                    (gpu["motion_codes"][index], native["optical"], native["image"],
                     native["optical"], native["optical"], native["physical_pupil"], native["incident"],
                     native["codes"], *map(np.int32, (*prepared["shape_yx"], native["padded"], 0))))
                indices, frequencies, counts, _ = _rearrangement_bind(gpu, np.asarray(sites)[None])
                band = (int(counts[0]) + 15) // 16 * 16
                query["index"][:source_count].set(indices[1][0], stream=stream)
                gpu["frequencies"][:len(frequencies)].set(frequencies, stream=stream)
                _rearrangement_select_roots(query, band)
                _rearrangement_propagate(gpu, query, band, "forward")
                fields[index] = query["actual"][:source_count].get(stream=stream)
    if pupil_steps.shape != (count,):
        cp, stream, native = gpu["cp"], gpu["stream"], gpu["resources"][1]
        _rearrangement_motion_capacity(gpu, count, stop_requested)
        area = int(np.prod(prepared["shape_yx"]))
        with _rearrangement_gpu_active(gpu["keeper"]), stream:
            gpu["motion_codes"][:count].set(np.asarray(result["phase_codes"]), stream=stream)
            gpu["kernels"]["phase_step_rms"]((count,), (256,),
                (gpu["motion_codes"], gpu["initial_phase"], native["physical_pupil"], gpu["phase_step_rms"],
                 np.int32(area), np.float64(gpu["pupil_energy"])))
            pupil_steps = gpu["phase_step_rms"][:count].get(stream=stream)
            stream.synchronize()
    desired, positive = np.asarray(result["desired_amplitudes"]), np.asarray(result["active_sites"])
    intensity = abs(fields.astype(np.complex128)) ** 2
    relative = np.divide(intensity, desired ** 2, out=np.zeros(desired.shape), where=positive)
    low = np.min(np.where(positive, relative, np.inf), axis=1) if count else np.empty(0)
    ratios = np.divide(relative.max(axis=1), low, out=np.full(count, np.inf), where=low > 0) if count else np.empty(0)
    selected_fields = fields[:, source_indices]
    initial = prepared["source_field"][source_indices]
    brightness = abs(selected_fields / initial[None]) ** 2
    selected_intensity = intensity[:, source_indices]
    # Identity-indexed samples cannot describe background: a moved identity
    # loses its old coordinate, while an inactive identity can coincide with a
    # legitimate new target. Inspect the actual complete focal plane instead.
    peak_intensity = np.zeros(count)
    peak_yx = np.full((count, 2), np.nan)
    target_peak_yx = np.full((count, source_count, 2), np.nan)
    target_peak_intensity = np.full((count, source_count), np.nan)
    source_site_intensity = np.empty((count, source_count))
    source_site_is_active = np.zeros((count, source_count), bool)
    source_reference = abs(np.asarray(prepared["source_field"], np.complex128)) ** 2
    initial_reference = float(np.min(source_reference))
    if count:
        from cupyx.scipy.ndimage import maximum_filter  # noqa: PLC0415

        cp, stream, native = gpu["cp"], gpu["stream"], gpu["resources"][1]
        shape = np.asarray(prepared["shape_yx"])
        source_sites = np.asarray(prepared["source_yx"], np.int64)
        area = int(np.prod(shape))
        sampling = int(gpu["psf_oversampling"])
        focal_shape = sampling * shape
        focal_center = focal_shape // 2
        psf_support = np.zeros(tuple(focal_shape), bool)
        profile = gpu["psf_profile"]
        start = focal_center + sampling * gpu["psf_profile_origin"]
        psf_support[tuple(slice(int(a), int(a + n)) for a, n in zip(start, profile.shape))] = profile > 0
        source_pixels = (source_sites - shape // 2) * sampling + focal_center
        halo_offsets = np.indices((4 * sampling + 1,) * 2).reshape(2, -1).T - 2 * sampling
        for index, codes in enumerate(result["phase_codes"]):
            if stop_requested is not None and stop_requested():
                raise InterruptedError("SLM rearrangement diagnostics stopped")
            with _rearrangement_gpu_active(gpu["keeper"]), stream:
                native["codes"].set(np.asarray(codes), stream=stream)
                gpu["kernels"]["field_project"](((area + 255) // 256,), (256,),
                    (native["codes"], native["optical"], native["image"], native["optical"], native["optical"],
                     native["physical_pupil"], native["incident"], native["codes"],
                     *map(np.int32, (*shape, native["padded"], 0))))
                # The centered physical pupil is embedded contiguously. Moving
                # that block only adds a common Fourier phase ramp, not power.
                focal = cp.fft.fftshift(cp.fft.fft2(native["optical"], s=tuple(focal_shape)))
                power = cp.abs(focal) ** 2
                maxima = ((power > 0) & (power == maximum_filter(power, size=3, mode="wrap"))).get(stream=stream)
                focal_intensity = power.get(stream=stream)
            active_points = np.asarray(result["sites_yx"])[index, positive[index]]
            active_pixels = (active_points - shape // 2) * sampling + focal_center
            recognized, values, peak_yx[index], peak_intensity[index], target_region = _rearrangement_focal_peaks(
                focal_intensity, maxima, active_pixels, psf_support)
            peak_yx[index] = (peak_yx[index] - focal_center) / sampling + shape // 2
            target_peak_yx[index, positive[index]] = (recognized - focal_center) / sampling + shape // 2
            target_peak_intensity[index, positive[index]] = values
            source_site_intensity[index] = focal_intensity[tuple(source_pixels.T)]
            source_site_is_active[index] = np.any(np.all(source_sites[:, None] == active_points[None], axis=-1), axis=1)
            if len(removed):
                halo = (source_pixels[removed, None] + halo_offsets[None]) % focal_shape
                halo_ratios = focal_intensity[tuple(halo.reshape(-1, 2).T)].reshape(len(removed), -1) / source_reference[removed, None]
                # An old identity landing under a current main lobe is not a
                # dark-region violation. Additional peaks there remain visible
                # in the independent whole-plane peak diagnosis above.
                excluded = target_region[tuple(halo.reshape(-1, 2).T)].reshape(len(removed), -1)
                discard[index] = float(np.max(np.where(excluded, 0, halo_ratios)))
    background = peak_intensity / np.min(selected_intensity, axis=1) if count else np.empty(0)
    phase_change = np.angle(selected_fields * initial.conj()[None])
    phase_step = np.angle(selected_fields * np.concatenate((initial[None], selected_fields[:-1])).conj())
    focal_phase_error = np.empty(0)
    if result["method"] == "lpi" and count:
        prescribed = np.asarray(result["desired_spectrum_coefficients"])[:, source_indices]
        focal_phase_error = np.sqrt(np.mean(np.angle(selected_fields * prescribed.conj()) ** 2, axis=1))
    retained_relative = relative[:, source_indices]
    retained_ratios = (retained_relative.max(axis=1) / retained_relative.min(axis=1)) if count else np.empty(0)
    fading_ratios = np.full(count, np.nan)
    fading = ~np.isin(np.arange(source_count), source_indices)
    for index in range(count):
        fading_active = fading & positive[index]
        if np.any(fading_active):
            values = relative[index, fading_active]
            fading_ratios[index] = values.max() / values.min()
    discarded_ok = bool(np.all(discard[max(0, result["fade_frames"] - 1):] <= SPOT_BACKGROUND_TOLERANCE))
    return {
        "actual_fields": _frozen(fields),
        "support_intensity_ratios": _frozen(ratios),
        "all_active_support_intensity_ratios": _frozen(ratios),
        "retained_intensity_ratios": _frozen(retained_ratios),
        "fading_intensity_ratios": _frozen(fading_ratios),
        "background_intensity_ratios": _frozen(background), "discard_intensity_ratios": _frozen(discard),
        "background_peak_yx": _frozen(peak_yx), "background_peak_intensity": _frozen(peak_intensity),
        "background_peak_to_initial_ratios": _frozen(peak_intensity / initial_reference),
        "background_initial_reference_intensity": initial_reference,
        "background_peak_tolerance": SPOT_BACKGROUND_TOLERANCE,
        "target_peak_yx": _frozen(target_peak_yx), "target_peak_intensity": _frozen(target_peak_intensity),
        "source_site_intensities": _frozen(source_site_intensity),
        "source_site_intensity_ratios": _frozen(source_site_intensity / source_reference[None]),
        "source_site_is_active": _frozen(source_site_is_active),
        "background_scope": "unclaimed-local-peaks-full-focal-plane; 4x-zero-padded",
        "peak_position_basis": "native-Fourier-pixels; quarter-pixel-samples; equal-plateau-centroid",
        "peak_assignment": "one-per-current-target; natural-pupil-PSF-center-component-at-1pct; nearest-target",
        "brightness_minimum_to_initial": _frozen(brightness.min(axis=1) if count else np.empty(0)),
        "brightness_maximum_to_initial": _frozen(brightness.max(axis=1) if count else np.empty(0)),
        "brightness_mean_to_initial": _frozen(brightness.mean(axis=1) if count else np.empty(0)),
        "center_sample_power_proxy": _frozen(selected_intensity.sum(axis=1) / (np.prod(prepared["shape_yx"]) * gpu["pupil_energy"])),
        "phase_change_from_initial_rms_rad": _frozen(np.sqrt(np.mean(phase_change ** 2, axis=1)) if count else np.empty(0)),
        "focal_phase_error_rms_rad": _frozen(focal_phase_error),
        "phase_step_max_rad": _frozen(np.max(abs(phase_step), axis=1) if count else np.empty(0)),
        "pupil_phase_step_rms_rad": _frozen(pupil_steps),
        "quality_evaluated": True, "discard_converged": discarded_ok,
        "converged": bool(result["converged"] and np.all(np.isfinite(retained_ratios)
                          & (retained_ratios <= result["frame_support_tolerances"]))
                          and np.all(peak_intensity <= SPOT_BACKGROUND_TOLERANCE * initial_reference)
                          and np.all(np.isfinite(target_peak_intensity[positive]))),
        "field_phase_reference": "fft-center", "publication_gate": False, "propagation": "native-fp32-fractional-fourier",
        "diagnostics_ms": (time.perf_counter() - started) * 1000,
    }
