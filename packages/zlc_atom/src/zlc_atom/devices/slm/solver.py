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
from hashlib import sha256
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
    from scipy.sparse import csr_array  # noqa: PLC0415
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
    maximum_bipartite_matching(csr_array([[True]]), perm_type="column")
    # Photograph-independent clearances: every source (held during the fade) and every target site.
    source_clearance, target_clearance = (float(np.sqrt(np.min(pdist(points, metric="sqeuclidean"), initial=np.inf)))
                                          for points in sites)
    return {
        "source_yx": source, "target_yx": target, "shape_yx": shape,
        "assignment_costs": _frozen(distance),
        "source_clearance": source_clearance, "target_clearance": target_clearance,
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
    from scipy.sparse import csr_array  # noqa: PLC0415
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
    endpoints = [("initial occupied sources",
                  float(np.sqrt(np.min(pdist(source[available], metric="sqeuclidean"), initial=np.inf))))]
    if cardinality == len(target):
        endpoints.append(("required target sites", prepared["target_clearance"]))
    for name, clearance in endpoints:
        # All occupied sources coexist before surplus fade. All target sites
        # are mandatory only when there are enough atoms: a sparse target
        # subset must not be rejected because two unused endpoints are close.
        if clearance < separation:
            raise ValueError(f"{name} clearance {clearance:g} is below the authored minimum separation {separation:g}")
    radii = np.unique(costs)

    def perfect(edges):
        # Whether these edges fill the smaller roster (CSR built straight from the mask).
        rows, columns = np.nonzero(edges)
        graph = csr_array((np.ones(len(rows), bool), columns, np.searchsorted(rows, np.arange(len(edges) + 1))),
                          shape=edges.shape)
        return np.count_nonzero(maximum_bipartite_matching(graph, perm_type="column") >= 0) == cardinality

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
            if perfect(allowed & (costs <= radii[hi])):
                break
            lo, hi, stride = hi + 1, min(len(radii), hi + stride), stride * 2
        if hi == len(radii):
            hi -= 1
            if lo > hi or not perfect(allowed):
                return None
        while lo < hi:
            middle = (lo + hi) // 2
            if perfect(allowed & (costs <= radii[middle])):
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
    # Every segment at once (a single frame is one held segment); a pair's
    # minimum is its first closest segment.
    start = motion[:max(1, len(motion) - 1)]
    delta = motion[1:] - motion[:-1] if len(motion) > 1 else np.zeros_like(start)
    if paired is None:
        y, x = (start[..., axis][:, selected, None] - start[:, None, :, axis] for axis in (0, 1))
        vy, vx = (delta[..., axis][:, selected, None] - delta[:, None, :, axis] for axis in (0, 1))
    else:
        y, x = (start[..., axis][:, selected] - start[..., axis][:, paired] for axis in (0, 1))
        vy, vx = (delta[..., axis][:, selected] - delta[..., axis][:, paired] for axis in (0, 1))
    speed = vy ** 2 + vx ** 2
    at = np.divide(-(y * vy + x * vx), speed,
                   out=np.zeros_like(speed), where=speed > 0)
    at = np.clip(at, 0., 1.)
    squared = (y + at * vy) ** 2 + (x + at * vx) ** 2
    if paired is None:
        squared[:, np.arange(len(selected)), selected] = np.inf
    else:
        squared[:, selected == paired] = np.inf
    intervals = np.argmin(squared, axis=0)
    minimum, times = (np.take_along_axis(values, intervals[None], 0)[0] for values in (squared, at))
    return minimum, intervals, times


def rearrangement_clearance(motion_yx: object) -> float:
    """Exact minimum distance along every emitted straight frame-to-frame path."""
    motion = np.asarray(motion_yx, dtype=float)
    if (motion.ndim != 3 or motion.shape[2] != 2 or len(motion) < 1
            or not np.all(np.isfinite(motion))):
        raise ValueError("motion_yx must contain finite frame, site, Y/X coordinates")
    if motion.shape[1] < 2:
        return np.inf
    first, second = np.triu_indices(motion.shape[1], 1)
    initial, _, _ = _rearrangement_pair_metrics(motion[:1], first, second)
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
    lower = np.nextafter(np.sqrt(initial) - radius[first] - radius[second], -np.inf)
    # Nor can a pair come closer than the boxes holding the two whole paths;
    # this bound only skips pairs, so its margin is generous roundoff room.
    low, high = motion.min(axis=0), motion.max(axis=0)
    gap = np.maximum(np.maximum(low[first] - high[second], low[second] - high[first]), 0.)
    box = np.hypot(gap[:, 0], gap[:, 1]) - 1e-9 * max(1., float(np.max(abs(motion))))
    reach = np.nextafter(np.sqrt(upper), np.inf)
    near = (lower <= reach) & (box <= reach)
    if np.any(near):
        squared, _, _ = _rearrangement_pair_metrics(motion, first[near], second[near])
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
__device__ __forceinline__ unsigned char dithered_code(unsigned int p16,unsigned int pixel){
 unsigned int hash=pixel+0x9e3779b9u;
 hash^=hash>>16;hash*=0x7feb352du;hash^=hash>>15;hash*=0x846ca68bu;hash^=hash>>16;
 // Exact p16/256 plus the same spatial threshold. A float-radian roundtrip
 // can spuriously turn an exact code 255 into zero at particular pixels.
 unsigned int carry=(((p16&255)<<16)+(hash>>8))>>24;
 return (unsigned char)(((p16>>8)+carry)&255);
}
__device__ unsigned char phase_code(double command,unsigned int pixel){
 if(command<0)command+=6.2831853071795864769;
 if(command<0)command+=6.2831853071795864769;
 float a=fminf((float)command,6.28318500518798828125f);
 return dithered_code(((unsigned int)floorf(a*10430.3783504704527f+0.5f))&65535,pixel);
}
'''


_REARRANGEMENT_CUDA = _REARRANGEMENT_PHASE_CODE_CUDA + r'''
#include <cuda_fp16.h>
#define CODE_RADIANS .02454369260617025968f
extern "C" __global__ void phase_step_rms(const unsigned char* movie,const float* initial,
 const float* pupil,float* result,int area,double energy){
 __shared__ double sum[256];int t=threadIdx.x,frame=blockIdx.x;double value=0;
 for(int i=t;i<area;i+=256){
  float before=frame?movie[(unsigned long long)(frame-1)*area+i]*CODE_RADIANS:initial[i];
  float difference=movie[(unsigned long long)frame*area+i]*CODE_RADIANS-before;
  difference=atan2f(sinf(difference),cosf(difference));
  value+=(double)pupil[i]*pupil[i]*difference*difference;
 }
 sum[t]=value;__syncthreads();
 for(int stride=128;stride;stride/=2){if(t<stride)sum[t]+=sum[t+stride];__syncthreads();}
 if(!t)result[frame]=sqrt(sum[0]/energy);
}
// exp(2 pi i f k / L) for f = whole + part. The integer product is reduced
// exactly, so FP64 (1/32 rate on T400-class parts) only splits each frequency once.
__device__ __forceinline__ float2 root(int whole,float part,int k,int L){
 float t=(float)((whole*k)%L)/L+part*k/L,s,c;sincospif(2.f*(t-rintf(t)),&s,&c);return make_float2(c,s);
}
// Each map evaluates only its roster points. Y is summed directly per point:
// a band holds one to a few points, so an FFT over all Y bins is wasted work.
extern "C" __global__ void point_roots(const double* fy,float2* roots,int H){
 __shared__ int whole;__shared__ float part;int y=blockIdx.x*blockDim.x+threadIdx.x,p=blockIdx.y;
 if(!threadIdx.x){double n=floor(fy[p]);whole=(int)n;part=(float)(fy[p]-n);}
 __syncthreads();
 if(y<H)roots[(unsigned long long)p*H+y]=root(whole,part,y-H/2,H);
}
// The analysis product is band-major (both planes of a band, H rows each): coalesced rows per point.
extern "C" __global__ void project_points(const float* projected,const int* point_band,const float2* roots,
 float2* actual,int H){
 __shared__ float sr[256],si[256];int p=blockIdx.x,t=threadIdx.x,k=point_band[p];float re=0,im=0;
 if(k>=0)for(int y=t;y<H;y+=256){float a=projected[k*2*H+y],b=projected[k*2*H+H+y];float2 r=roots[(unsigned long long)p*H+y];
  re+=a*r.x+b*r.y;im+=b*r.x-a*r.y;}
 sr[t]=re;si[t]=im;__syncthreads();
 for(int s=128;s;s>>=1){if(t<s){sr[t]+=sr[t+s];si[t]+=si[t+s];}__syncthreads();}
 if(!t)actual[p]=make_float2(sr[0],si[0]);
}
// The analysis product without cuBLAS: row i of band k sums field[i][x]*cos_k(x)+field[i][P+x]*sin_k(x)
// over x, cos and sin read from the synthesis roots. A warp sums TM rows for KC bands, each lane four x
// per step, then halves its partial sums across the warp until each lane holds TM*KC/32 of them. The NB
// warps of a block share their rows (one field read from memory, met again in L1) and take every NB-th
// group of KC bands.
__device__ __forceinline__ float2 half_pair(unsigned int bits){return __half22float2(*reinterpret_cast<half2*>(&bits));}
template<int TM,int KC>
__device__ __forceinline__ void analysis_rows(const half* __restrict__ field,const half2* __restrict__ roots,
 float* __restrict__ projected,int rows,int P,int K,int NB){
 constexpr int N=TM*KC;static_assert(N%32==0,"a warp's sums must fill its lanes");
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,row0=(blockIdx.x*(blockDim.x>>5)/NB+warp/NB)*TM;
 if(row0>=rows)return;
 for(int k0=warp%NB*KC;k0<K;k0+=NB*KC){
  float v[N];
  #pragma unroll
  for(int j=0;j<N;j++)v[j]=0;
  for(int c=lane*4;c<P;c+=128){
   float a[TM][8];
   #pragma unroll
   for(int r=0;r<TM;r++){const half* f=field+(unsigned long long)(row0+r)*2*P+c;
    uint2 u=*reinterpret_cast<const uint2*>(f),w=*reinterpret_cast<const uint2*>(f+P);
    float2 p0=half_pair(u.x),p1=half_pair(u.y),q0=half_pair(w.x),q1=half_pair(w.y);
    a[r][0]=p0.x;a[r][1]=q0.x;a[r][2]=p0.y;a[r][3]=q0.y;a[r][4]=p1.x;a[r][5]=q1.x;a[r][6]=p1.y;a[r][7]=q1.y;}
   #pragma unroll
   for(int k=0;k<KC;k++)if(k0+k<K){
    uint4 b=*reinterpret_cast<const uint4*>(roots+(unsigned long long)(k0+k)*P+c);
    float2 b0=half_pair(b.x),b1=half_pair(b.y),b2=half_pair(b.z),b3=half_pair(b.w);
    #pragma unroll
    for(int r=0;r<TM;r++){float s=v[r*KC+k];
     s=fmaf(a[r][0],b0.x,s);s=fmaf(a[r][1],b0.y,s);s=fmaf(a[r][2],b1.x,s);s=fmaf(a[r][3],b1.y,s);
     s=fmaf(a[r][4],b2.x,s);s=fmaf(a[r][5],b2.y,s);s=fmaf(a[r][6],b3.x,s);s=fmaf(a[r][7],b3.y,s);
     v[r*KC+k]=s;}
   }
  }
  // Lane bit m keeps the upper or lower half and adds its partner's: lane l ends with sums l*N/32 on.
  #pragma unroll
  for(int s=0;s<5;s++){const int m=16>>s,n=N>>(s+1);bool upper=lane&m;
   #pragma unroll
   for(int j=0;j<n;j++){float send=upper?v[j]:v[j+n],keep=upper?v[j+n]:v[j];
    v[j]=keep+__shfl_xor_sync(0xffffffffu,send,m);}}
  #pragma unroll
  for(int j=0;j<N/32;j++){int o=j+N/32*lane,k=k0+o%KC;if(k<K)projected[(unsigned long long)k*rows+row0+o/KC]=v[j];}
 }
}
__device__ __forceinline__ void pack_pair(half* packed,float2 plus,float2 minus,int y,int d,int H,int P){
 int r=y*2*P+d;
 packed[r]=__float2half_rn(plus.x+minus.x);packed[r+P]=__float2half_rn(plus.y-minus.y);
 packed[r+H*2*P]=__float2half_rn(plus.y+minus.y);packed[r+H*2*P+P]=__float2half_rn(minus.x-plus.x);
}
// Twice the displayed pixel (y, x), read back from pack_pair's sums and differences.
__device__ __forceinline__ float2 packed_pixel(const half* packed,int y,int x,int H,int W,int P){
 int d=abs(x-W/2),r=y*2*P+d;float sign=x>=W/2?1.f:-1.f;
 return make_float2(__half2float(packed[r])-sign*__half2float(packed[r+H*2*P+P]),
                    __half2float(packed[r+H*2*P])+sign*__half2float(packed[r+P]));
}
// X roots of a map's bands: half2 (cos, sin) per band and centered x for the
// synthesis, and the transposed cos then sin planes for the analysis product.
extern "C" __global__ void select_roots(const double* frequencies,half2* roots,half* forward,int K,int P,int W){
 __shared__ float2 tile[32][33];__shared__ int whole[32];__shared__ float part[32];
 int x=blockIdx.x*32+threadIdx.x,k=blockIdx.y*32+threadIdx.y;
 if(!threadIdx.y){int j=blockIdx.y*32+threadIdx.x;double f=j<K?frequencies[j]:0,n=floor(f);
  whole[threadIdx.x]=(int)n;part[threadIdx.x]=(float)(f-n);}
 __syncthreads();
 for(int j=0;j<32;j+=8)if(x<P&&k+j<K){
  float2 v=x<=W/2?root(whole[threadIdx.y+j],part[threadIdx.y+j],x,W):make_float2(0,0);
  tile[threadIdx.y+j][threadIdx.x]=v;roots[(k+j)*P+x]=__float22half2_rn(v);
 }
 __syncthreads();
 int xx=blockIdx.x*32+threadIdx.y,kk=blockIdx.y*32+threadIdx.x;
 for(int j=0;j<32;j+=8)if(xx+j<P&&kk<K){float2 v=tile[threadIdx.x][threadIdx.y+j];
  int out=(xx+j)*K+kk;
  forward[out]=__float2half_rn(v.x);forward[out+P*K]=__float2half_rn(v.y);}
}
__device__ __forceinline__ float2 displayed_pixel(unsigned char code,const float* pupil,const float* incident,int i){
 float s,c;sincosf(code*CODE_RADIANS+incident[i],&s,&c);
 return make_float2(__fmul_rn(pupil[i],c),__fmul_rn(pupil[i],s));
}
// atan2 in turns: octant reduction and an odd minimax polynomial (error below
// 1e-8 turn), without atan2f's IEEE division and reciprocal.
__device__ __forceinline__ float field_turns(float2 f){
 float ax=fabsf(f.x),ay=fabsf(f.y),t=__fdividef(fminf(ax,ay),fmaxf(ax,ay));
 if(!(t<=1.f))t=0;  // both zero
 float s=t*t,p=fmaf(s,-.0040540580f,.0218612288f);p=fmaf(p,s,-.0559098861f);p=fmaf(p,s,.0964200441f);
 p=fmaf(p,s,-.1390853351f);p=fmaf(p,s,.1994653599f);p=fmaf(p,s,-.3332985605f);p=fmaf(p,s,.9999993329f);
 float r=p*t*.15915494309189533577f;
 if(ay>ax)r=.25f-r;if(f.x<0)r=.5f-r;return f.y<0?-r:r;
}
// exp(2 pi i q/65536) of a 16-bit phase q: quadrant from its top bits, minimax
// polynomials on [-pi/4, pi/4] (error 1e-7) and a single type conversion, which
// runs at quarter rate on these parts.
__device__ __forceinline__ float2 rotor16(unsigned int q){
 q+=8192;float x=(float)((int)(q&16383)-8192)*9.5873799242852576e-5f,s=x*x;
 float sn=fmaf(x*s,fmaf(s,fmaf(s,-1.9515295891e-4f,8.3321608736e-3f),-1.6666654611e-1f),x);
 float cs=fmaf(s*s,fmaf(s,fmaf(s,2.443315711809948e-5f,-1.388731625493765e-3f),4.166664568298827e-2f),fmaf(s,-.5f,1.f));
 unsigned int k=(q>>14)&3;float c=k&1?-sn:cs,t=k&1?cs:sn;return k&2?make_float2(-c,-t):make_float2(c,t);
}
// One evaluation on every physical pixel. A thread owns the conjugate pixels
// (y, W/2+-d) of ROWS rows: the roster points of each band, summed over their Y
// roots, give the band's row coefficients; the band X roots then synthesize both
// pixels (roots at opposite centered x are conjugate even for an asymmetric
// pupil; each pixel gets its own amplitude). mode 0 keeps the displayed codes
// (measure), 1 encodes the phase of the synthesized field, 2 moves the codes by
// the damped phase-gradient correction of the synthesized residual r. With the
// displayed field e=A*exp(i*phase),
// delta_phase=gain*Im(conj(e)*r) is the gradient of the roster residual. It is
// the Gauss-Newton step only with the roster Gram matrix replaced by its
// diagonal sum(A^2)/2: it moves the field at every residual site by
// gain*sum(A^2)/2 times its residual when the sites are PSF widths apart, and
// ignores the cross-talk between sites and the conjugate term. Phases are whole
// 1/65536 turns (code<<8 plus the incident phase), quantized by dithered_code;
// amplitude is half precision: four bytes per pixel. The displayed pair is
// packed for the analysis product.
extern "C" __global__ void evaluate(const float2* coefficients,const int* band_start,const int* band_point,
 const float2* point_roots,const half2* roots,const half* amplitude,const unsigned short* incident,
 unsigned char* codes,half* packed,float gain,int mode,int sites,int H,int W,int P,int K){
 extern __shared__ float2 row[];
 int d=blockIdx.x*blockDim.x+threadIdx.x,y0=blockIdx.y*ROWS;
 for(int j=threadIdx.x;mode&&j<ROWS*K;j+=blockDim.x){
  int k=j/ROWS,y=min(y0+j%ROWS,H-1);float2 s=make_float2(0,0);
  for(int q=band_start[k];q<band_start[k+1];q++){int p=band_point[q];
   float2 a=coefficients[p],r=point_roots[(unsigned long long)p*H+y];s.x+=a.x*r.x-a.y*r.y;s.y+=a.x*r.y+a.y*r.x;}
  row[j]=s;
 }
 __syncthreads();
 if(d>=P)return;
 float A[ROWS]={},B[ROWS]={},C[ROWS]={},D[ROWS]={};
 for(int k=0;mode&&k<K;k++){float2 x=__half22float2(roots[k*P+d]);
  #pragma unroll
  for(int r=0;r<ROWS;r++){float2 c=row[k*ROWS+r];A[r]+=c.x*x.x;B[r]+=c.x*x.y;C[r]+=c.y*x.x;D[r]+=c.y*x.y;}}
 gain*=coefficients[sites].x;
 #pragma unroll
 for(int r=0;r<ROWS;r++){
  int y=y0+r;float2 field[2]={make_float2(A[r]-D[r],C[r]+B[r]),make_float2(A[r]+D[r],C[r]-B[r])},shown[2]={};
  for(int side=0;side<2&&y<H;side++){
   int x=side?W/2-d:W/2+d;if(x<0||x>=W||(side&&!d))continue;
   int i=y*W+x;unsigned int n=incident[i],code;float a=__half2float(amplitude[i]);
   if(mode==1)code=dithered_code((__float2int_rd(fmaf(field_turns(field[side]),65536.f,.5f))-n)&65535,i);
   else{code=codes[i];
    if(mode==2){float2 e=rotor16((code<<8)+n);
     code=dithered_code(((code<<8)+__float2int_rd(fmaf(gain*a*(e.x*field[side].y-e.y*field[side].x),
                                                       10430.378350470453f,.5f)))&65535,i);}}
   if(mode)codes[i]=code;
   float2 e=rotor16((code<<8)+n);shown[side]=make_float2(a*e.x,a*e.y);
  }
  if(y<H)pack_pair(packed,shown[0],shown[1],y,d,H,P);
 }
}
extern "C" __global__ void display(const unsigned char* codes,const float* pupil,const float* incident,float2* optical,int area){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<area)optical[i]=displayed_pixel(codes[i],pupil,incident,i);
}
extern "C" __global__ void encode_phase(const float* phase,unsigned char* codes,int area){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<area)codes[i]=phase_code((double)phase[i],(unsigned int)i);
}
// Full focal-plane Kim-WGS on the unshifted plane; index holds each trap's pixel.
// One warp: Kim weights, the cap (stray times the weakest trap) and the trap values.
extern "C" __global__ void plane_traps(float2* focal,const int* index,const float* goal,float* weight,float2* unit,
 float* limit,int n,int locked,float exponent,float stray){
 int t=threadIdx.x;float relative=0,total=0,low=3.4e38f,sum=0;
 for(int j=t;j<n;j+=32){float2 f=focal[index[j]];float m=hypotf(f.x,f.y);relative+=m/goal[j];total+=goal[j];low=fminf(low,m);}
 for(int o=16;o;o>>=1){relative+=__shfl_xor_sync(~0u,relative,o);total+=__shfl_xor_sync(~0u,total,o);
  low=fminf(low,__shfl_xor_sync(~0u,low,o));}
 float mean=relative/n;
 for(int j=t;j<n;j+=32){float2 f=focal[index[j]];float m=hypotf(f.x,f.y);
  weight[j]*=powf(fminf(5.f,fmaxf(.2f,mean*goal[j]/m)),exponent);sum+=weight[j];
  if(!locked)unit[j]=make_float2(f.x/m,f.y/m);}
 for(int o=16;o;o>>=1)sum+=__shfl_xor_sync(~0u,sum,o);
 for(int j=t;j<n;j+=32){float w=weight[j]*total/sum;weight[j]=w;focal[index[j]]=make_float2(mean*w*unit[j].x,mean*w*unit[j].y);}
 if(!t)*limit=stray*low;
}
extern "C" __global__ void plane_cap(float2* focal,const bool* near,const float* limit,int area){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=area||near[i])return;
 float2 f=focal[i];float m=hypotf(f.x,f.y),l=*limit;if(m>l)focal[i]=make_float2(f.x*(l/m),f.y*(l/m));
}
extern "C" __global__ void plane_unit(float2* field,const float* pupil,int area){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=area)return;
 float2 g=field[i];float m=hypotf(g.x,g.y),a=pupil[i];field[i]=m>0?make_float2(a*g.x/m,a*g.y/m):make_float2(a,0);
}
// Per-map tracking on the device, one 128-thread block each. The roster holds
// meta[0] retained traps, then meta[1] dark points whose residual is the field
// above their cap amplitude (all of it for a zero cap; none for an infinite
// cap, a point under a retained trap) and whose gate is a field amplitude; its
// last slots are reserved for watched stray peaks (select_watch).
// scalars: phase window, correction damping, uniformity gate. Reductions run in
// FP32 over warp shuffles: FP64 runs at 1/32 of the FP32 rate on these parts.
__device__ float block_reduce(float value,float* shared,bool maximum){
 for(int o=16;o;o>>=1){float other=__shfl_xor_sync(0xffffffffu,value,o);value=maximum?fmaxf(value,other):value+other;}
 __syncthreads();if(!(threadIdx.x&31))shared[threadIdx.x>>5]=value;__syncthreads();
 value=shared[0];for(int w=1;w<(blockDim.x>>5);w++)value=maximum?fmaxf(value,shared[w]):value+shared[w];return value;
}
__device__ __forceinline__ float2 dark_excess(float2 e,float cap){
 float a=hypotf(e.x,e.y),f=a>cap?1.f-cap/a:0.f;return make_float2(f*e.x,f*e.y);
}
extern "C" __global__ void track_load(const float2* c,float2* cancel,const float* cap,const int* meta,
 float2* coefficients,int sites){
 int t=threadIdx.x,n=meta[0],m=meta[1];
 for(int k=t;k<m;k+=blockDim.x)if(isinf(cap[k]))cancel[k]=make_float2(0,0);
 __syncthreads();
 for(int j=t;j<sites;j+=blockDim.x)coefficients[j]=j<n?c[j]:j<n+m?cancel[j-n]:make_float2(0,0);
 if(!t)coefficients[sites]=make_float2(1,0);
}
// Kim-weighted WGS step: weights clip(mean/relative,.2,5)^exponent, focal phase
// kept within the window of the reference (rotor=gauge*exp(-i*reference)), the
// dark points' cancelling coefficients, then the next encode's roster values.
// Coefficients that already meet the map's gates keep their amplitudes and the
// dark points' cancellation, so the next encode shows the same map (for very
// few traps the phase-only response amplifies a Kim update until it diverges);
// a focal phase beyond the window only turns its coefficient back by the
// excess, so the steering goes on.
extern "C" __global__ void track_sweep(const float2* field,const float* goal,const float2* rotor,const float* cap,
 const float* gate,const float* scalars,const int* meta,float2* c,float2* cancel,float2* accepted,float2* coefficients,
 int sites,float exponent){
 __shared__ float shared[32];
 int t=threadIdx.x,n=meta[0],m=meta[1];
 float relative=0,inverse=0,high=0,low=-3.4e38f,over=-1,power=0,limit=scalars[0];
 for(int j=t;j<n;j+=blockDim.x){float2 b=field[j],a=c[j];float e=hypotf(b.x,b.y),r=e/goal[j];
  relative+=r;inverse+=hypotf(a.x,a.y)/e;high=fmaxf(high,r);low=fmaxf(low,-r);}
 for(int k=t;k<m;k+=blockDim.x){float2 e=field[n+k];over=fmaxf(over,hypotf(e.x,e.y)-gate[k]);}
 high=block_reduce(high,shared,true);low=-block_reduce(low,shared,true);over=block_reduce(over,shared,true);
 if(high*high<=scalars[2]*low*low&&over<=0){
  for(int j=t;j<sites;j+=blockDim.x){float2 x=make_float2(0,0);
   if(j<n){float2 b=field[j],a=c[j],q=rotor[j];float deviation=atan2f(b.x*q.y+b.y*q.x,b.x*q.x-b.y*q.y),s,co;
    sincosf(fminf(limit,fmaxf(-limit,deviation))-deviation,&s,&co);
    accepted[j]=a;x=c[j]=make_float2(a.x*co-a.y*s,a.x*s+a.y*co);}
   else if(j<n+m)x=cancel[j-n];
   coefficients[j]=x;}
  if(!t)coefficients[sites]=make_float2(1,0);
  return;
 }
 float mean=block_reduce(relative,shared,false)/n,gain=block_reduce(inverse,shared,false)/n;
 for(int j=t;j<n;j+=blockDim.x){float2 b=field[j],a=c[j];
  float w=hypotf(a.x,a.y)*powf(fminf(5.f,fmaxf(.2f,mean*goal[j]/hypotf(b.x,b.y))),exponent);power+=w*w;}
 float inverse_norm=rsqrtf(block_reduce(power,shared,false));
 for(int j=t;j<n;j+=blockDim.x){float2 b=field[j],a=c[j],r=rotor[j],s;
  float w=hypotf(a.x,a.y)*powf(fminf(5.f,fmaxf(.2f,mean*goal[j]/hypotf(b.x,b.y))),exponent);
  float deviation=atan2f(b.x*r.y+b.y*r.x,b.x*r.x-b.y*r.y);
  float phase=atan2f(b.y,b.x)-(deviation-fminf(limit,fmaxf(-limit,deviation))),scale=w*inverse_norm;
  sincosf(phase,&s.y,&s.x);accepted[j]=a;c[j]=coefficients[j]=make_float2(scale*s.x,scale*s.y);}
 for(int k=t;k<m;k+=blockDim.x){float2 e=dark_excess(field[n+k],cap[k]),x=cancel[k];
  x=make_float2((x.x-e.x*gain)*inverse_norm,(x.y-e.y*gain)*inverse_norm);cancel[k]=coefficients[n+k]=x;}
 for(int j=n+m+t;j<sites;j+=blockDim.x)coefficients[j]=make_float2(0,0);
 if(!t)coefficients[sites]=make_float2(1,0);
}
// Correction residual: retained traps equalized (an amplitude applied along their
// own phase), dark points pushed down to their caps, the last `watch` roster slots
// (watched stray peaks; unused slots read a zero field) down to watch_cap of the
// weakest retained trap's amplitude. A step's measured residual change is the
// roster Gram matrix, conjugate coupling included, times that step, so while the
// map is outside its gates (a watched peak's is watch_gate), correction c (fixed
// in the captured graph) is Anderson-mixed with the map's last MEMORY steps: the
// damped residual minus the least-squares combination of the earlier steps and
// residual changes (rings: MEMORY+1 residuals, MEMORY steps of 2*sites floats;
// one pass forms the normal equations, regularized at the FP32 rounding of their
// trace). Inside them the residual is mostly quantization noise, which
// the mixing would amplify for few traps: the plain damped step polishes. Values
// are normalized to unit maximum for half precision, the scale stored after the
// roster for the correction.
extern "C" __global__ void track_residual(const float2* field,const float* goal,const float* cap,const float* gate,
 const float* scalars,const int* meta,float2* coefficients,int sites,int watch,float watch_cap,float watch_gate,
 int c,float* history,float* steps){
 __shared__ float shared[32],mix[MEMORY];
 int t=threadIdx.x,n=meta[0],m=meta[1],length=2*sites;float damping=scalars[1];
 float power=0,high=0,low=-3.4e38f,weakest=-3.4e38f,over=-1,largest=0;
 for(int j=t;j<n;j+=blockDim.x){float2 b=field[j];float g=goal[j],e=b.x*b.x+b.y*b.y,r=e/(g*g);
  power+=r;high=fmaxf(high,r);low=fmaxf(low,-r);weakest=fmaxf(weakest,-e);}
 float level=sqrtf(block_reduce(power,shared,false)/n);
 high=block_reduce(high,shared,true);low=-block_reduce(low,shared,true);
 weakest=sqrtf(-block_reduce(weakest,shared,true));
 float limit=watch_cap*weakest;
 float* now=history+(c%(MEMORY+1))*length;
 for(int j=t;j<sites;j+=blockDim.x){float2 b=field[j],r=make_float2(0,0);
  if(j<n)r.x=level*goal[j]-hypotf(b.x,b.y);
  else if(j<n+m||j>=sites-watch){r=dark_excess(b,j<n+m?cap[j-n]:limit);r.x=-r.x;r.y=-r.y;
   over=fmaxf(over,hypotf(b.x,b.y)-(j<n+m?gate[j-n]:watch_gate*weakest));}
  now[2*j]=r.x;now[2*j+1]=r.y;}
 over=block_reduce(over,shared,true);
 int used=high<=scalars[2]*low&&over<=0?0:min(c,MEMORY);
 // Residual change a is history[c-used+a+1]-history[c-used+a].
 float A[MEMORY][MEMORY]={},B[MEMORY]={};
 for(int i=t;i<length;i+=blockDim.x){float d[MEMORY];
  #pragma unroll
  for(int a=0;a<MEMORY;a++)if(a<used)d[a]=history[(c-used+a+1)%(MEMORY+1)*length+i]-history[(c-used+a)%(MEMORY+1)*length+i];
  #pragma unroll
  for(int a=0;a<MEMORY;a++)if(a<used){B[a]+=d[a]*now[i];
   #pragma unroll
   for(int b=a;b<MEMORY;b++)if(b<used)A[a][b]+=d[a]*d[b];}}
 #pragma unroll
 for(int a=0;a<MEMORY;a++)if(a<used){B[a]=block_reduce(B[a],shared,false);
  #pragma unroll
  for(int b=a;b<MEMORY;b++)if(b<used)A[b][a]=A[a][b]=block_reduce(A[a][b],shared,false);}
 if(!t){  // symmetric positive definite: Gaussian elimination without pivoting; unused rows are identity
  float trace=A[0][0];for(int a=1;a<MEMORY;a++)trace+=A[a][a];
  for(int a=0;a<MEMORY;a++)A[a][a]=a<used?A[a][a]+1e-6f*trace:1.f;
  for(int a=0;a<MEMORY;a++)for(int b=a+1;b<MEMORY;b++){float f=A[a][a]>0?A[b][a]/A[a][a]:0;
   for(int k=a;k<MEMORY;k++)A[b][k]-=f*A[a][k];B[b]-=f*B[a];}
  for(int a=MEMORY-1;a>=0;a--){float x=B[a];for(int k=a+1;k<MEMORY;k++)x-=A[a][k]*mix[k];mix[a]=A[a][a]>0?x/A[a][a]:0;}
 }
 __syncthreads();
 // Each element's oldest step is read before this step overwrites its ring slot.
 float* step=steps+c%MEMORY*length;
 for(int i=t;i<length;i+=blockDim.x){float s=damping*now[i];
  for(int a=0;a<used;a++){const float *a1=history+(c-used+a+1)%(MEMORY+1)*length,*a0=history+(c-used+a)%(MEMORY+1)*length;
   s-=mix[a]*(steps[(c-used+a)%MEMORY*length+i]+damping*(a1[i]-a0[i]));}
  step[i]=s;}
 __syncthreads();
 for(int j=t;j<sites;j+=blockDim.x)largest=fmaxf(largest,hypotf(step[2*j],step[2*j+1]));
 float scale=block_reduce(largest,shared,true);
 if(!(scale>0))scale=1;
 for(int j=t;j<sites;j+=blockDim.x){float2 b=field[j];float x=step[2*j]/scale;
  if(j<n){float e=hypotf(b.x,b.y);coefficients[j]=make_float2(x*b.x/e,x*b.y/e);}
  else coefficients[j]=make_float2(x,step[2*j+1]/scale);}
 if(!t)coefficients[sites]=make_float2(scale,0);
}
// Ghost watch of the displayed map before a listed correction. Each 2x2 block of
// the packed displayed field (packed_pixel reads twice a pixel), every pixel turned
// by exp(-2*pi*i*(cy*y/H+cx*x/W)) in exact integer turns, is summed into an
// H/2 x WB plane, zero padded from W/2 to WB columns. Its FFT samples the focal
// field at fy=cy+ky*H/(2*(H/2)) and fx=cx+kx*W/(2*WB) native pixels, folded to
// +-H/4 and +-W/4 around the window centre (cy, cx), times the block response
// exp(i*pi*(u/H+v/W))*cos(pi*u/H)*cos(pi*v/W) of the offsets u=fy-cy, v=fx-cx.
// Thread 0 restarts found=[count, threshold, entries, relisted count, bins]: the
// threshold is detection times the weakest of the meta[0] retained trap intensities.
extern "C" __global__ void watch_display(const half* field,float2* plane,int H,int W,int P,int WB,int cy,int cx,
 const float2* actual,const int* meta,float detection,float* found){
 int i=blockIdx.x*blockDim.x+threadIdx.x,*counts=(int*)found;
 if(!i){float low=3.4e38f;for(int j=0;j<meta[0];j++){float2 a=actual[j];low=fminf(low,a.x*a.x+a.y*a.y);}
  counts[0]=counts[2+4*CANDIDATES]=0;found[1]=detection*low;}
 if(i<BINS)counts[3+4*CANDIDATES+i]=0;
 if(i>=H/2*WB)return;
 int y=i/WB*2,x=i%WB*2;float2 s=make_float2(0,0);
 for(int a=0;a<2;a++)for(int b=0;b<2&&x+b<W;b++){float2 p=packed_pixel(field,y+a,x+b,H,W,P),e=make_float2(1,0);
  if(cy|cx)sincospif(-2.f*((float)(cy*(y+a)%H)/H+(float)(cx*(x+b)%W)/W),&e.y,&e.x);  // an axis-centred window needs no turn
  s.x+=p.x*e.x-p.y*e.y;s.y+=p.x*e.y+p.y*e.x;}
 plane[i]=make_float2(.5f*s.x,.5f*s.y);
}
__device__ __forceinline__ void watch_frequency(int index,int HB,int WB,int* ky,int* kx,float* response){
 *ky=index/WB;*kx=index%WB;if(*ky>=(HB+1)/2)*ky-=HB;if(*kx>=WB/2)*kx-=WB;
 *response=cospif(*ky/(2.f*HB))*cospif(*kx/(2.f*WB));
}
// Local maxima of |F|^2 whose block-compensated value exceeds the threshold and
// that may lie more than exclusion px from every one of the meta[0] retained traps:
// a maximum lies anywhere within half a sample of its sample, so only a sample
// whose whole cell is within exclusion px of a trap is left out (a cancelled dark
// point pushes its light right next to it; distances over the window's alias
// periods H/2, W/2). They are listed as (index, compensated |F|^2, re, im)
// and counted in quarter-octave bins above the threshold. When more than
// CANDIDATES pass, the overflow launch (a block per SM) relists only the strongest
// whole bins that fit: the listed set never depends on the order of the atomics.
__device__ void list_peak(int i,const float2* F,int H,int W,int HB,int WB,int cy,int cx,float* found,float exclusion,
 const double* point_fy,const int* point_band,const double* frequencies,const int* meta,int overflow,int lowest){
 int* counts=(int*)found;int y=i/WB,x=i%WB,ky,kx,k;float r;
 float2 a=F[i];float v=a.x*a.x+a.y*a.y;
 watch_frequency(i,HB,WB,&ky,&kx,&r);if(v<=found[1]*r*r)return;
 for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
  float2 b=F[((y+dy+HB)%HB)*WB+(x+dx+WB)%WB];if(b.x*b.x+b.y*b.y>v)return;}
 float fy=(float)ky*H/(2*HB)+cy,fx=(float)kx*W/(2*WB)+cx;
 for(int p=0;p<meta[0];p++){float dy=fy-(float)point_fy[p],dx=fx-(float)frequencies[point_band[p]];
  dy-=.5f*H*rintf(dy/(.5f*H));dx-=.5f*W*rintf(dx/(.5f*W));
  dy=fabsf(dy)+.25f*H/HB;dx=fabsf(dx)+.25f*W/WB;
  if(dy*dy+dx*dx<=exclusion*exclusion)return;}
 v/=r*r;int bin=min(BINS-1,(int)(4.f*log2f(v/found[1])));
 if(overflow){if(bin<lowest)return;k=atomicAdd(counts+2+4*CANDIDATES,1);}
 else{atomicAdd(counts+3+4*CANDIDATES+bin,1);k=atomicAdd(counts,1);}
 if(k<CANDIDATES){float* e=found+2+4*k;e[0]=__int_as_float(i);e[1]=v;e[2]=a.x;e[3]=a.y;}
}
extern "C" __global__ void peaks(const float2* F,int H,int W,int HB,int WB,int cy,int cx,float* found,float exclusion,
 const double* point_fy,const int* point_band,const double* frequencies,const int* meta,int overflow){
 __shared__ int lowest;const int* counts=(const int*)found;
 if(overflow){
  if(counts[0]<=CANDIDATES)return;
  if(!threadIdx.x){int n=0;lowest=BINS;while(lowest&&n+counts[3+4*CANDIDATES+lowest-1]<=CANDIDATES)n+=counts[3+4*CANDIDATES+--lowest];}
  __syncthreads();
 }
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<HB*WB;i+=gridDim.x*blockDim.x)
  list_peak(i,F,H,W,HB,WB,cy,cx,found,exclusion,point_fy,point_band,frequencies,meta,overflow,lowest);
}
// Vertex of the parabola through |F| at the sample before, at and after a maximum, in samples.
__device__ __forceinline__ float vertex(float2 before,float2 at,float2 after){
 float b=hypotf(before.x,before.y),a=hypotf(at.x,at.y),c=hypotf(after.x,after.y),curvature=b-2*a+c;
 return curvature<0?fminf(.5f,fmaxf(-.5f,.5f*(b-c)/curvature)):0.f;
}
// One block of 1024 threads: the K strongest listed maxima (ties by position)
// that are at least separation px from every stronger one become the reserved
// roster points slot+j, located between the samples by parabolas through |F|. A
// point joins the band of the nearest roster column, else of an earlier watched
// point, within half a pixel of its X frequency, else the next free one of the
// `bands` reserved bands after band; a point left without a band is skipped and
// an unused slot gets none. Y frequencies, bands, new band X frequencies and the
// fields (block response removed, the sparse model's centred reference) are
// written, then every site's band CSR is rebuilt in site order (the record's
// own when nothing is watched). The roots are made afterwards by point_roots
// and select_roots.
extern "C" __global__ void select_watch(const float2* F,const float* found,int H,int W,int HB,int WB,int cy,int cx,
 int K,int bands,float separation,int band,int slot,int sites,double* point_fy,int* point_band,int* band_point,
 int* band_start,double* frequencies,float2* actual){
 __shared__ float value[CANDIDATES],py[CANDIDATES],px[CANDIDATES];
 __shared__ int column[CANDIDATES],order[CANDIDATES],chosen,used;
 const int* counts=(const int*)found;
 int t=threadIdx.x,count=counts[0]>CANDIDATES?counts[2+4*CANDIDATES]:counts[0],total=band+bands;
 for(int c=t;c<count;c+=blockDim.x){
  int i=__float_as_int(found[2+4*c]),y=i/WB,x=i%WB,ky,kx;float r,nearest=.5f;watch_frequency(i,HB,WB,&ky,&kx,&r);
  value[c]=found[3+4*c];order[c]=-1;column[c]=-1;
  py[c]=(ky+vertex(F[(y+HB-1)%HB*WB+x],F[i],F[(y+1)%HB*WB+x]))*H/(2.f*HB)+cy;
  px[c]=(kx+vertex(F[y*WB+(x+WB-1)%WB],F[i],F[y*WB+(x+1)%WB]))*W/(2.f*WB)+cx;
  for(int k=0;k<band;k++)if(band_start[k+1]>band_start[k]){float d=fabsf((float)frequencies[k]-px[c]);
   if(d<nearest){nearest=d;column[c]=k;}}
 }
 __syncthreads();
 for(int c=t;c<count;c+=blockDim.x){int rank=0;bool kept=true;
  for(int q=0;q<count;q++)if(value[q]>value[c]||(value[q]==value[c]&&(py[q]<py[c]||(py[q]==py[c]&&px[q]<px[c])))){
   float dy=py[q]-py[c],dx=px[q]-px[c];rank++;kept&=dy*dy+dx*dx>=separation*separation;}
  if(kept)order[rank]=c;
 }
 __syncthreads();
 // Bands in strength order by thread 0, from shared memory only (value now holds the new band X frequencies).
 if(!t){int j=0;used=0;
  for(int rank=0;rank<count&&j<K;rank++){int c=order[rank],b;float nearest=.5f;if(c<0)continue;b=column[c];
   for(int k=0;column[c]<0&&k<used;k++){float d=fabsf(value[k]-px[c]);if(d<nearest){nearest=d;b=band+k;}}
   if(b<0&&used<bands){b=band+used;value[used++]=px[c];}
   if(b>=0){column[c]=b;order[j++]=c;}}
  chosen=j;}
 __syncthreads();
 for(int j=t;j<K;j+=blockDim.x){
  if(j>=chosen){point_band[slot+j]=-1;actual[slot+j]=make_float2(0,0);continue;}
  int c=order[j],ky,kx;float r,s,co;watch_frequency(__float_as_int(found[2+4*c]),HB,WB,&ky,&kx,&r);
  // exp(2*pi*i*(fy*(H/2)/H+fx*(W/2)/W)) moves the FFT origin to the centre; the
  // block response adds exp(i*pi*(u/H+v/W)). Exact integer turns, in half turns.
  int ty=(ky*(2*(H/2)-1))%(4*HB),tx=(kx*(2*(W/2)-1))%(4*WB),uy=cy*2*(H/2)%(2*H),ux=cx*2*(W/2)%(2*W);
  sincospif((float)(ty<0?ty+4*HB:ty)/(2*HB)+(float)(tx<0?tx+4*WB:tx)/(2*WB)+(float)uy/H+(float)ux/W,&s,&co);
  float re=found[4+4*c]/r,im=found[5+4*c]/r;
  point_fy[slot+j]=py[c];point_band[slot+j]=column[c];actual[slot+j]=make_float2(re*co-im*s,re*s+im*co);
 }
 for(int k=t;k<used;k+=blockDim.x)frequencies[band+k]=value[k];
 for(int k=t;k<=total;k+=blockDim.x)band_start[k]=0;
 __syncthreads();
 for(int p=t;p<sites;p+=blockDim.x)if(point_band[p]>=0)atomicAdd(band_start+point_band[p]+1,1);
 __syncthreads();
 if(!t)for(int k=1;k<=total;k++)band_start[k]+=band_start[k-1];
 __syncthreads();
 for(int p=t;p<sites;p+=blockDim.x)if(point_band[p]>=0){int q=band_start[point_band[p]];
  for(int o=0;o<p;o++)q+=point_band[o]==point_band[p];band_point[q]=p;}
}
'''
# The analysis product's warp kernels (rows per warp, bands per pass), measured
# against cuBLAS per GPU and band count (_rearrangement_analysis_choices).
_ANALYSIS_KERNELS = ((4, 8), (8, 8), (2, 16), (4, 16), (2, 32))
_REARRANGEMENT_CUDA += "".join(
    f'extern "C" __global__ void analysis_{rows}_{bands}(const half* field,const half2* roots,float* projected,'
    f'int R,int P,int K,int NB){{analysis_rows<{rows},{bands}>(field,roots,projected,R,P,K,NB);}}\n'
    for rows, bands in _ANALYSIS_KERNELS)
_REARRANGEMENT_KERNELS = ("phase_step_rms", "point_roots", "project_points", "select_roots", "evaluate",
                          "display", "encode_phase", "plane_traps", "plane_cap", "plane_unit",
                          "track_load", "track_sweep", "track_residual", "watch_display", "peaks", "select_watch",
                          *(f"analysis_{rows}_{bands}" for rows, bands in _ANALYSIS_KERNELS))
_EVALUATE_ROWS = 8  # focal rows per evaluate block (ROWS)


_REARRANGEMENT_KEEPERS: dict[int, dict[str, object]] = {}
_REARRANGEMENT_KEEPERS_CHANGED = Condition()
# Fraction of the idle time the heater keeps the GPU busy. One 64 MB pass every
# 20 ms (about 10% on the lab's T400) let the driver drop the SM clock to about
# 690 MHz during an exposure, so a shot's first maps ran 2-4 times slower than
# on a busy GPU; the heater now runs back to back until a computation starts.
_REARRANGEMENT_WARM_DUTY = 1.


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
            from time import perf_counter  # noqa: PLC0415

            try:
                with cp.cuda.Device(device), stream:
                    pause = .020
                    while not keeper["stop"].wait(pause):
                        with keeper["lock"]:
                            if keeper["active"] or keeper["stop"].is_set():
                                pause = .020
                                continue
                            start = perf_counter()
                            kernel(((size // 4 + 255) // 256,), (256,), (keeper["scratch"], np.int32(size // 4)))
                            stream.synchronize()
                            pause = (perf_counter() - start) * (1 / _REARRANGEMENT_WARM_DUTY - 1)
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


# Per-map tracking of one continuous optical state (ARCHITECTURE_DESIGN §8.4).
_TRACK_KIM_EXPONENT = .8      # Kim weight exponent; weights clipped to [.2, 5] per update
_TRACK_RELEASE_WINDOW = .8    # rad around the reference focal phase: release and next maps, unmoving plans
_TRACK_WINDOW = .32           # rad around the reference focal phase: transport and last maps
# Surplus-atom sites on the removal map and the next one, relative to that
# atom's own initial trap; after these two maps they are ordinary focal points
# for the ghost watch. The liquid crystal lowers a trap over milliseconds,
# adiabatically for the atom, so only a dark site lets it go; a single-point
# cancellation is well posed, while forcing a whole sub-PSF neighbourhood dark
# piles the light into a bright ring just outside it.
_TRACK_DARK_LIMIT = .002
# Empty unselected sources hold no atom: on those two maps they only stay below
# this fraction of their initial trap. Forcing every non-retained lattice site
# fully dark pushes the lattice intermodulation onto the sites just outside the
# array and costs the main lobe its power.
_TRACK_EMPTY_LIMIT = .1
# Fixed per-map schedule chosen offline, never an online early exit: (Kim-WGS
# sweeps, phase-gradient corrections of the displayed codes, correction
# damping), chosen on replays scored by independent propagation of the
# delivered codes; their measured scope per plan class is stated in
# ARCHITECTURE_DESIGN §8.4. "first" is the first map whose roster drops the
# dark points and, without surplus atoms, both the fade map and the first map
# that moves. With at least _TRACK_MANY[0] retained traps a correction
# converges faster, and transport maps are "many" except the first
# _TRACK_MANY[1] after "first" (a block of atoms still settling), the last
# _TRACK_MANY[2], and maps whose steered reference turns a retained trap by
# more than _TRACK_TURN rad since the previous map (a large phase rotation over
# few maps). A plan is "crowded" with at least _TRACK_CROWDED surplus atoms
# per retained trap (darkening many bright sites pushes light into stray peaks
# during the corrections), with fewer than _TRACK_FEW retained traps (a
# correction moves them more than its estimate), and when no retained trap
# moves: the retained traps then stay on lattice sites beside the surplus atoms,
# which take their lattice intermodulation. Its removal and next maps (and
# "dark_last") take the crowded schedules, and its transport maps that are not
# "many" are "crowded transport". A map that moves a retained trap farther than
# _TRACK_LEAP native pixels since the previous map (few maps for a long path) is
# a "leap" (a partially filled last one "leap_last"); the removal and next maps
# keep their own kinds whatever their step. A filled last map shows the prepared
# destination; "dark_last", a removal/next map that is also the last (surplus
# atoms imply a filled target), corrects that destination. A crowded one darkens
# up to hundreds of bright sites, so it is watched every four corrections and
# keeps eight after the last watch for its uniformity. A partially filled last
# map runs its sweeps, capped full-plane iterations (its empty target sites
# collect lattice intermodulation, which the cap spreads; _TRACK_VACANT_PLANE[0]
# with uniform requested intensities, [1] with non-uniform ones or in a crowded
# plan), one measurement and its corrections.
# A plan in which no retained trap moves (a filled target that only loses its
# surplus atoms, or one that only changes weights) keeps every reference at its
# source phase: turned on the spot, the phase set half way to the destination
# puts the retained traps' lattice intermodulation onto the traps themselves,
# and a turn of up to 2.7 rad within one map nearly empties a trap in the
# liquid-crystal transition. Its maps keep the darkened maps'
# _TRACK_RELEASE_WINDOW, so the phases can follow changing requested weights
# while the references stay. After its "first" map it repeats the previous map
# ("hold": no evaluation) while the requested weights stay unchanged, and its
# last map is refined like a partially filled one, except that with two maps
# the removal map that is also the last continues from the removal map and with
# one it corrects the prepared destination (its transition from the photograph
# then turns the traps to the destination phases at once; liquid-crystal
# transitions are not gated). With fewer than _TRACK_FEW retained traps a
# correction moves them more than its Gram-diagonal estimate (two traps: twice,
# through the conjugate term), so maps without dark points scale the damping
# by retained/_TRACK_FEW.
_TRACK_SCHEDULE = {"release": (4, 5, .8), "next": (1, 4, 1.), "crowded release": (4, 12, .8),
                   "crowded next": (1, 10, 1.), "first": (2, 6, 1.), "transport": (1, 5, 1.),
                   "crowded transport": (1, 8, 1.), "many": (1, 2, 1.), "hold": (0, 0, 1.), "leap": (1, 6, 1.),
                   "dark_last": (0, 15, .8), "crowded dark_last": (0, 20, .8), "last": (1, 4, 1.),
                   "leap_last": (2, 5, 1.)}
_TRACK_MANY = (40, 1, 2)
_TRACK_TURN = .18
_TRACK_FEW = 4
_TRACK_CROWDED = .7
_TRACK_LEAP = 1.5
_TRACK_VACANT_PLANE = (8, 16)
# Phase-only encoding puts the intermodulation of the retained traps on other
# lattice sites, vacated ones included. Every intermediate map except a "hold"
# map, and "dark_last" (the other last maps are the prepared destination or a
# full-plane refinement), is watched at a fixed cost: one FFT of the
# 2x2-binned displayed field lists the local maxima above a detection fraction
# of the weakest retained trap's intensity that may lie farther than
# _TRACK_WATCH_EXCLUSION from every retained trap (a sample whose whole cell,
# 0.99375 native pixels wide along X on the lab's plane, lies within it is left
# out); the _TRACK_WATCH strongest, _TRACK_WATCH_SEPARATION apart and located
# between the samples, are written into the reserved roster slots. A watched
# point shares the band of a roster column or of another point within half a
# pixel, else takes one of _TRACK_WATCH_BANDS reserved bands (none left: it is
# skipped), so the corrections evaluate a fixed band+_TRACK_WATCH_BANDS
# whatever the number of points; an unused slot reads a zero field. The
# corrections push each point down to its cap; points between the detection and
# the cap only catch light moving onto them. Cancelled peaks move their light
# onto other sites, and darkening many bright sites (the crowded removal map,
# "dark_last") raises stray peaks during the corrections, so each kind is
# watched before fixed corrections (0: right after its sweeps or measurement),
# the Anderson memory restarting at each watch. No host decision within a map.
# An intermediate map's stray light is its strongest watched point on the final
# codes over the weakest retained trap, a lower bound of the focal plane's (a
# peak that grows after the last watch, beyond the watched points or below the
# detection on the binned samples is not seen); a "hold" map repeats the
# previous map's. The last map's is measured on its delivered codes once it has
# been handed over: the strongest local maximum farther than
# _TRACK_WATCH_EXCLUSION from its traps on the twofold zero-padded focal plane
# (_rearrangement_stray; preparation measured the prepared destination, which a
# filled last map of a moving plan shows). Up to _TRACK_WATCH_GATE
# (intermediate, last map) it is inside the map's gates.
# The binned plane holds the H/2 x W/2 native pixels around the sources' and
# targets' bounding-box centre rounded to whole _TRACK_WATCH_STEP: a near-axis
# array keeps the axis-centred window the schedule was tuned with, and half a
# step changes the 2x2 block response by under 0.1%. A geometry that does not
# fit it is refused at preparation. A peak outside the window (intermodulation a
# few lattice periods beyond an array that reaches the window's edge) folds back
# into it, so it is neither watched nor reported online; the post-playback
# diagnostic covers the whole plane. When more than _TRACK_WATCH_CANDIDATES
# maxima pass, the strongest of _TRACK_WATCH_BINS quarter-octave strength bins
# that fit are listed.
# _TRACK_WATCH_LIMITS: (cap, detection, corrections watched before) per
# watched kind, cap and detection as fractions of the weakest retained trap's
# intensity.
_TRACK_WATCH = 64
_TRACK_WATCH_BANDS = 8
_TRACK_WATCH_LIMITS = {"release": (.04, .025, (0,)), "next": (.03, .02, (0, 1)),
                       "crowded release": (.04, .025, (0, 2, 5, 8)), "crowded next": (.03, .02, (0, 1, 4, 7)),
                       "first": (.03, .02, (0, 3)), "transport": (.03, .02, (0, 2)),
                       "crowded transport": (.03, .02, (0, 2, 5)), "many": (.03, .02, (0,)),
                       "leap": (.03, .02, (0, 4)), "dark_last": (.025, .015, (5,)),
                       "crowded dark_last": (.025, .015, (0, 4, 8, 12))}
_TRACK_WATCH_GATE = (.2, .05)
_TRACK_WATCH_SEPARATION = 2.
_TRACK_WATCH_EXCLUSION = 2.5  # native Fourier pixels around a retained trap (its sampled main lobe)
_TRACK_WATCH_STEP = 16        # native Fourier pixels
# The CUDA array sizes below are compiled in from these constants (_prepare_rearrangement_gpu).
_TRACK_WATCH_CANDIDATES = 2048  # listed local maxima per map, the size of select_watch's shared arrays
_TRACK_WATCH_BINS = 64
_TRACK_ANDERSON = 3           # memory of the Anderson-mixed corrections (track_residual)
# A prepared array (destination, or a source without endpoint_data) is solved
# before the photograph: sparse Kim-WGS from random phases, then the start with
# the most light near its traps is refined by Kim-WGS on the full native focal
# plane, where light farther than _ARRAY_STRAY_RADIUS from every trap is capped
# at _ARRAY_STRAY of the weakest trap. Phase-only intermodulation of a lattice
# lands on lattice sites; the cap spreads it, while cancelling chosen peaks only
# moves them. The last _ARRAY_LOCKED iterations keep the trap phases so that the
# weights converge for 1 to 400 traps.
_ARRAY_SEEDS = 4
_ARRAY_PLANE_ITERATIONS = 50
_ARRAY_LOCKED = 20
_ARRAY_STRAY = .01
_ARRAY_STRAY_RADIUS = 3.      # native Fourier pixels around a trap that belong to it


def _rearrangement_cuda_imports():
    """CuPy, its cuBLAS module and the loader of NVIDIA's libraries, or a message naming the GPU installer."""
    try:
        import cupy as cp  # noqa: PLC0415
        from cupy.cuda import cublas  # noqa: PLC0415
        from cuda.pathfinder import load_nvidia_dynamic_lib  # noqa: PLC0415
    except (ImportError, OSError) as error:
        import sys  # noqa: PLC0415

        raise RuntimeError(f"Cannot import CuPy/CUDA libraries with Python {sys.executable}: "
                           f"{type(error).__name__}: {error}. Run bin\\install_slm_gpu.bat "
                           "for this product/interpreter, or install zou-lab-control[slm-gpu] using this interpreter.") from error
    return cp, cublas, load_nvidia_dynamic_lib


def _rearrangement_blas(load_library):
    """cuBLAS and its GemmEx and SetWorkspace entry points."""
    import ctypes  # noqa: PLC0415

    library = (ctypes.WinDLL if os.name == "nt" else ctypes.CDLL)(load_library("cublas").abs_path)
    gemm = library.cublasGemmEx
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    gemm.argtypes = [pointer, integer, integer, integer, integer, integer, pointer,
                     pointer, integer, integer, pointer, integer, integer,
                     pointer, pointer, integer, integer, integer, integer]
    gemm.restype = integer
    set_workspace = library.cublasSetWorkspace_v2
    set_workspace.argtypes = [pointer, pointer, ctypes.c_size_t]
    set_workspace.restype = integer
    return library, gemm, set_workspace


def _rearrangement_module(cp):
    """The rearrangement CUDA module; CuPy keeps its compiled binary on disk."""
    return cp.RawModule(code=_REARRANGEMENT_CUDA, options=tuple(f"-D{name}={value}" for name, value in (
        ("ROWS", _EVALUATE_ROWS), ("MEMORY", _TRACK_ANDERSON), ("CANDIDATES", _TRACK_WATCH_CANDIDATES),
        ("BINS", _TRACK_WATCH_BINS))))


def _prepare_rearrangement_gpu(geometry, pupil_amplitude, pupil_phase, stop_requested):
    """Prepare the one native fractional-Fourier workspace and its band sizes."""
    cp, cublas, load_library = _rearrangement_cuda_imports()
    shape = geometry["shape_yx"]
    pupil = np.asarray(pupil_amplitude, np.float32)
    if (pupil.shape != shape or not np.all(np.isfinite(pupil))
            or np.any(pupil < 0) or not np.any(pupil > 0)):
        raise ValueError("pupil_amplitude must be finite, nonnegative and match the full SLM")
    incident = np.zeros(shape, np.float32) if pupil_phase is None else canonical_phase(pupil_phase, shape)
    # Every trap and dark site lies on or between sources and targets: they must fit the ghost watch's window.
    signed = np.concatenate((geometry["source_yx"], geometry["target_yx"])) - np.asarray(shape) // 2
    centre = np.rint((signed.min(axis=0) + signed.max(axis=0)) / (2 * _TRACK_WATCH_STEP)).astype(int) * _TRACK_WATCH_STEP
    reach, room = np.max(abs(signed - centre), axis=0), np.asarray(shape) / 4 - _TRACK_WATCH_EXCLUSION
    if np.any(reach > room):
        raise ValueError(f"SLM rearrangement sources and targets lie up to {reach.tolist()} native Fourier pixels "
                         f"(Y, X) from the ghost-watch window centre {centre.tolist()}; the window holds +-{room.tolist()}")
    keeper = _rearrangement_keeper_acquire(cp)
    gpu = None
    try:
        with _rearrangement_gpu_active(keeper):
            stream = cp.cuda.Stream(non_blocking=True, priority=-1)
            library, gemm, set_workspace = _rearrangement_blas(load_library)
            source, target = geometry["source_yx"], geometry["target_yx"]
            number = len(source)
            # A map roster is the selected traps plus at most every unselected
            # source: stationary sources share one band per source X, and each
            # selected trajectory adds at most one (X, fractional-Y) band. A
            # prepared array (source or destination) has one band per column.
            # The last _TRACK_WATCH sites and _TRACK_WATCH_BANDS bands are reserved for watched peaks.
            sites = max(number, len(target)) + _TRACK_WATCH
            selected_bands = min(number, len(np.unique(source[:, 1])) + min(number, len(target)))
            capacity = (max(selected_bands, len(np.unique(target[:, 1]))) + 7) // 8 * 8 + _TRACK_WATCH_BANDS
            # evaluate keeps 8*ROWS B of row coefficients per band in shared memory, up to capacity bands.
            properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            shared = properties["sharedMemPerBlockOptin"]
            if 8 * _EVALUATE_ROWS * capacity > shared:
                raise ValueError(f"SLM rearrangement needs {capacity} Fourier bands (distinct source and target "
                                 f"columns, {_TRACK_WATCH_BANDS} for the ghost watch); this GPU's shared memory holds "
                                 f"at most {shared // (8 * _EVALUATE_ROWS)}")
            module = _rearrangement_module(cp)
            from cupyx.scipy.fft import get_fft_plan  # noqa: PLC0415
            from scipy.fft import fft2, fftshift, ifftshift  # noqa: PLC0415

            psf = abs(fftshift(fft2(ifftshift(pupil)))) ** 2
            h, w = shape
            padded = (w // 2 + 16) // 16 * 16
            gpu = dict(cp=cp, cublas=cublas, stream=stream, module=module,
                       kernels={name: module.get_function(name) for name in _REARRANGEMENT_KERNELS},
                       library=library, gemm=gemm, shape=shape, padded=padded,
                       capacity=capacity, pupil_cpu=pupil.copy(), incident_cpu=np.asarray(incident, np.float32).copy(),
                       psf_support=_psf_main_lobe(psf, tuple(np.asarray(shape) // 2)),
                       pupil_energy=float(np.sum(pupil.astype(np.float64) ** 2)),
                       keeper=keeper, output_pool=cp.cuda.PinnedMemoryPool(), watch_centre=tuple(map(int, centre)),
                       processors=properties["multiProcessorCount"],
                       # Diagnostic temporaries (movie, fourfold focal planes and their plan; the prepared
                       # destination's twofold plane), released after each use, also on Stop or failure.
                       diagnostic_pool=cp.cuda.MemoryPool(),
                       alpha=np.asarray(1, np.float32), beta=np.asarray(0, np.float32))
            # The loaded function is shared by every workspace, so it always opts in to the device maximum.
            gpu["kernels"]["evaluate"].max_dynamic_shared_size_bytes = shared
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
            gpu.update(psf_profile=profile, psf_profile_origin=lower, psf_oversampling=oversampling)
            handles = []

            def close():
                if not gpu:
                    return
                try:
                    with _rearrangement_gpu_active(keeper, check_error=False):
                        stream.synchronize()
                        for handle in handles:
                            cublas.destroy(handle)
                        handles.clear()
                        gpu["output_pool"].free_all_blocks()
                        gpu["diagnostic_pool"].free_all_blocks()
                finally:
                    gpu.clear()
                    try:
                        with cp.cuda.Device(keeper["device"]):
                            cp.get_default_memory_pool().free_all_blocks(stream=stream)
                    finally:
                        _rearrangement_keeper_release(keeper)
            gpu["close"] = close
            with stream:
                # User-owned scratch keeps cuBLAS from allocating inside each call.
                gpu["blas_workspace"] = cp.empty(4 * 1024 * 1024, cp.uint8)
                handle = cublas.create()
                handles.append(handle)
                cublas.setStream(handle, stream.ptr)
                status = set_workspace(handle, gpu["blas_workspace"].data.ptr, gpu["blas_workspace"].nbytes)
                if status:
                    raise RuntimeError(f"SLM cuBLAS workspace setup failed ({status})")
                gpu.update(
                    handle=handle,
                    # Half-precision Fourier roots and packed fields halve the dominant
                    # memory traffic; every sum accumulates in FP32.
                    forward=cp.empty((2 * padded, capacity), cp.float16),
                    roots=cp.empty((capacity, padded, 2), cp.float16),
                    field=cp.empty((2, h, 2 * padded), cp.float16),
                    projected=cp.empty((capacity, 2, h), cp.float32),
                    # Roster values, then one normalization of a correction residual.
                    coefficients=cp.zeros(sites + 1, cp.complex64), actual=cp.zeros(sites, cp.complex64),
                    point_roots=cp.zeros((sites, h), cp.complex64),
                    # Tracking state: coefficients carried between maps, the dark
                    # points' cancelling coefficients, and the last encoded ones.
                    track_c=cp.zeros(sites, cp.complex64), track_cancel=cp.zeros(sites, cp.complex64),
                    track_accepted=cp.zeros(sites, cp.complex64),
                    # Anderson rings of one map's correction residuals and steps.
                    track_history=cp.zeros((_TRACK_ANDERSON + 1, 2 * sites), cp.float32),
                    track_steps=cp.zeros((_TRACK_ANDERSON, 2 * sites), cp.float32),
                    # Ghost watch: the binned focal plane (width padded to 256 pixels) and its local
                    # maxima [count, threshold, (index, value, re, im)..., relisted count, strength bins].
                    watch_plane=cp.empty((h // 2, -(-w // 256) * 128), cp.complex64),
                    watch_found=cp.zeros(3 + 4 * _TRACK_WATCH_CANDIDATES + _TRACK_WATCH_BINS, cp.float32),
                    optical=cp.empty((h, w), cp.complex64), codes=cp.zeros((h, w), cp.uint8),
                    physical_pupil=cp.asarray(pupil), incident=cp.asarray(incident, cp.float32),
                    # An evaluation reads 4 bytes per pixel: half amplitude, incident phase in 1/65536 turns.
                    amplitude=cp.asarray(pupil.astype(np.float16)),
                    incident_turns=cp.asarray((np.rint(np.mod(np.asarray(incident, np.float64) / (2 * np.pi), 1) * 65536)
                                               % 65536).astype(np.uint16)),
                    # A map's codes leave the device on their own stream from a
                    # snapshot while the next map computes: one slot per map in flight.
                    copy_stream=cp.cuda.Stream(non_blocking=True), frame_slots=cp.empty((2, h, w), cp.uint8))
                choices, _ = _rearrangement_analysis_choices(gpu, range(8, capacity + 1, 8), stop_requested)
                gpu["analysis"] = {band: _rearrangement_analysis_launcher(gpu, entry["choice"], band)
                                   for band, entry in choices.items()}
                # cuFFT plans of the workspace's own: a cached plan would keep its work area after close.
                gpu["watch_plan"] = get_fft_plan(gpu["watch_plane"], axes=(0, 1), value_type="C2C")
                gpu["plane_plan"] = get_fft_plan(gpu["optical"], axes=(0, 1), value_type="C2C")
                gpu["stray_plan"] = get_fft_plan(gpu["optical"], shape=(2 * h, 2 * w), axes=(0, 1), value_type="C2C")
                # One map's roster and tracking inputs live in one record: one
                # host-to-device copy binds a map to the captured graphs.
                gpu["record"] = cp.zeros(_rearrangement_record_views(None, sites, capacity), cp.uint8)
                gpu.update(_rearrangement_record_views(gpu["record"], sites, capacity))
                # One prepare-time evaluation (roster upload, propagation, roster
                # download) is one graph launch for every band size.
                host = gpu["output_pool"].malloc((2 * sites + 1) * 8)
                gpu["host_values"] = np.frombuffer(host, np.complex64, count=sites + 1)
                gpu["host_fields"] = np.frombuffer(host, np.complex64, count=sites, offset=(sites + 1) * 8)
                gpu["host_memory"] = host
                gpu["host_values"][:] = 0
                gpu["graphs"], gpu["track_graphs"] = {}, {}
                runtime = cp.cuda.runtime
                segments = {segment for kind in _TRACK_SCHEDULE for complete in (False, True)
                            for segment in _rearrangement_segments(kind, complete)}
                for band in range(8, capacity - _TRACK_WATCH_BANDS + 1, 8):
                    if stop_requested is not None and stop_requested():
                        raise InterruptedError("SLM rearrangement preparation stopped")
                    _rearrangement_roots(gpu, band)
                    for operation in ("encode", "measure"):
                        # Compile and plan outside capture, then record the same work.
                        _rearrangement_propagate(gpu, band, operation)
                        stream.begin_capture(runtime.streamCaptureModeThreadLocal)
                        try:
                            # Pinned-host copies are graph nodes; CuPy's array copies refuse capture.
                            runtime.memcpyAsync(gpu["coefficients"].data.ptr, host.ptr, (sites + 1) * 8,
                                                runtime.memcpyHostToDevice, stream.ptr)
                            _rearrangement_propagate(gpu, band, operation)
                            runtime.memcpyAsync(host.ptr + (sites + 1) * 8, gpu["actual"].data.ptr, sites * 8,
                                                runtime.memcpyDeviceToHost, stream.ptr)
                        finally:
                            graph = stream.end_capture()
                        graph.upload(stream)
                        gpu["graphs"][band, operation] = graph
                    # A whole map of the fixed schedule is one launch on the bound record; its
                    # watched evaluations run band+_TRACK_WATCH_BANDS bands, launched first outside capture.
                    _rearrangement_propagate(gpu, band + _TRACK_WATCH_BANDS, "measure")
                    for segment in segments:
                        stream.begin_capture(runtime.streamCaptureModeThreadLocal)
                        try:
                            _rearrangement_track_map(gpu, band, *segment)
                        finally:
                            graph = stream.end_capture()
                        graph.upload(stream)
                        gpu["track_graphs"][(band, *segment)] = graph
                gpu["codes"].fill(0)
                # Warm one output allocation without keeping a writable frame.
                staging = gpu["output_pool"].malloc(h * w)
                del staging
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


def _rearrangement_roots(gpu, band, first=0):
    """Cache the bound record's roots: the Y roots of the points from site first on and the X roots of band bands."""
    h, padded, sites = gpu["shape"][0], gpu["padded"], gpu["actual"].size
    gpu["kernels"]["point_roots"](((h + 255) // 256, sites - first), (256,),
                                  (gpu["point_fy"][first:], gpu["point_roots"][first:], np.int32(h)))
    gpu["kernels"]["select_roots"](((padded + 31) // 32, (band + 31) // 32), (32, 8),
                                   (gpu["frequencies"], gpu["roots"], gpu["forward"],
                                    *map(np.int32, (band, padded, gpu["shape"][1]))))


def _rearrangement_propagate(gpu, band, operation):
    """One evaluation of the complete native aperture at the roster points.

    encode displays the phase of the synthesized coefficients; correct moves the
    displayed codes by the phase-gradient step of the synthesized residual; measure
    keeps the displayed codes. Each measures the actual encoded field."""
    h, w = gpu["shape"]
    padded, sites = gpu["padded"], gpu["actual"].size
    kernels = gpu["kernels"]
    rows = _EVALUATE_ROWS
    kernels["evaluate"](((padded + 127) // 128, (h + rows - 1) // rows), (128,),
                        (gpu["coefficients"], gpu["band_start"], gpu["band_point"], gpu["point_roots"], gpu["roots"],
                         gpu["amplitude"], gpu["incident_turns"], gpu["codes"], gpu["field"],
                         np.float32(2 / gpu["pupil_energy"]),
                         *map(np.int32, (("measure", "encode", "correct").index(operation), sites, h, w, padded, band))),
                        shared_mem=rows * band * 8)
    gpu["analysis"][band]()
    kernels["project_points"]((sites,), (256,), (gpu["projected"], gpu["point_band"], gpu["point_roots"],
                                                  gpu["actual"], np.int32(h)))


# The analysis product (the packed field against the X roots of each band) is
# cuBLAS's or a warp kernel's, whichever is faster on this GPU for the band
# count. cuBLAS multiplies the half field on tensor cores where the GPU has them;
# the lab's T400 has none, and there it took 0.57 ms at 8 to 16 bands and 2.2 ms
# from 40, more than the evaluate kernel, while a warp kernel reads the field
# from memory once. A choice is measured once per GPU, driver, cuBLAS, CUDA
# source and panel size and kept in _ANALYSIS_CACHE beside the checkout (in the
# user's local cache when installed): bin\warm_slm_gpu.bat measures every band
# count up to _ANALYSIS_WARM_BANDS, and a preparation measures any it still
# lacks. A candidate whose single launch is 1.5 times the fastest one's is
# dropped; the others are timed _ANALYSIS_TRIALS times in turn, about 1 ms each
# in one CUDA graph like a map's, and the lowest median wins.
_ANALYSIS_CACHE = "slm_gpu_cache"
_ANALYSIS_TRIALS = 5
_ANALYSIS_WARM_BANDS = 256


def _rearrangement_analysis_launcher(gpu, name, band):
    """A launch of one analysis product of band bands into gpu["projected"]."""
    h, padded = gpu["shape"][0], gpu["padded"]
    if name == "cublas":
        cb, gemm = gpu["cublas"], gpu["gemm"]
        arguments = (gpu["handle"], 1, 1, 2 * h, band, 2 * padded, gpu["alpha"].ctypes.data,
                     gpu["field"].data.ptr, 2, 2 * padded, gpu["forward"].data.ptr, 2, band, gpu["beta"].ctypes.data,
                     gpu["projected"].data.ptr, 0, 2 * h, cb.CUBLAS_COMPUTE_32F, cb.CUBLAS_GEMM_DEFAULT)

        def launch():
            status = gemm(*arguments)
            if status:
                raise RuntimeError(f"SLM cuBLAS analysis failed ({status})")
        return launch
    rows, bands = map(int, name.split("_")[1:])
    # Up to four warps share a block's rows; with fewer band groups a block holds more rows.
    warps = min(-(-band // bands), 4)
    groups = max(1, 4 // warps)
    kernel, grid, block = gpu["kernels"][name], (-(-(2 * h // rows) // groups),), (32 * warps * groups,)
    arguments = (gpu["field"], gpu["roots"], gpu["projected"], *map(np.int32, (2 * h, padded, band, warps)))
    return lambda: kernel(grid, block, arguments)


def _analysis_cache_file():
    here = Path(__file__).resolve()
    root = here.parents[6]
    if here.parent == root / "packages" / "zlc_atom" / "src" / "zlc_atom" / "devices" / "slm":
        return root / _ANALYSIS_CACHE / "analysis.json"
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "zou_lab_control" / _ANALYSIS_CACHE / "analysis.json"


def _analysis_cache_read(path, key):
    try:
        cache = strict_json_loads(path.read_text(encoding="utf-8"), "SLM analysis choices")
    except (OSError, ValueError):
        return {}
    entry = cache.get(key) if isinstance(cache, dict) else None
    return dict(entry) if isinstance(entry, dict) else {}


def _rearrangement_analysis_choices(gpu, bands, stop_requested=None):
    """Each band count's analysis product {band: {"choice", "us"}} and the band counts measured now."""
    cp, stream, h = gpu["cp"], gpu["stream"], gpu["shape"][0]
    runtime = cp.cuda.runtime
    properties = runtime.getDeviceProperties(cp.cuda.Device().id)
    key = " | ".join((properties["name"].decode(), f"sm_{properties['major']}{properties['minor']}",
                      f"{properties['multiProcessorCount']} SMs", f"driver {runtime.driverGetVersion()}",
                      f"cuBLAS {gpu['cublas'].getVersion(gpu['handle'])}", f"field {2 * h}x{2 * gpu['padded']}",
                      f"source {sha256(_REARRANGEMENT_CUDA.encode()).hexdigest()[:16]}"))
    names = ["cublas", *(f"analysis_{rows}_{bands}" for rows, bands in _ANALYSIS_KERNELS if 2 * h % rows == 0)]
    path = _analysis_cache_file()
    known = _analysis_cache_read(path, key)
    missing = [band for band in bands if not (isinstance(known.get(str(band)), dict)
                                              and known[str(band)].get("choice") in names)]
    if missing:
        start, end = cp.cuda.Event(), cp.cuda.Event()

        def captured(launch, repeats):
            stream.begin_capture(runtime.streamCaptureModeThreadLocal)
            try:
                for _ in range(repeats):
                    launch()
            finally:
                graph = stream.end_capture()
            graph.upload(stream)
            return graph

        def timed(graph):
            start.record(stream)
            graph.launch(stream)
            end.record(stream)
            end.synchronize()
            return cp.cuda.get_elapsed_time(start, end) * 1e3

        # An idle GPU lowers its clock: keep it busy before the first candidate is timed.
        busy = captured(_rearrangement_analysis_launcher(gpu, "cublas", missing[0]), 10)
        for _ in range(20):
            timed(busy)
        for band in missing:
            if stop_requested is not None and stop_requested():
                raise InterruptedError("SLM rearrangement preparation stopped")
            launches = {name: _rearrangement_analysis_launcher(gpu, name, band) for name in names}
            single = {}
            for name, launch in launches.items():
                launch()  # loads the kernel and lets cuBLAS plan outside capture, as a preparation does
                single[name] = timed(captured(launch, 1))
            fastest = min(single.values())
            graphs = {}
            for name, launch in launches.items():
                if single[name] <= 1.5 * fastest:
                    repeats = max(1, min(20, round(1000 / max(single[name], 1))))
                    graphs[name] = captured(launch, repeats), repeats
            trials = {name: [] for name in graphs}
            for _ in range(_ANALYSIS_TRIALS):
                for name, (graph, repeats) in graphs.items():
                    trials[name].append(timed(graph) / repeats)
            us = {**single, **{name: float(np.median(times)) for name, times in trials.items()}}
            known[str(band)] = {"choice": min(trials, key=us.__getitem__),
                                "us": {name: round(us[name], 1) for name in names}}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                cache = strict_json_loads(path.read_text(encoding="utf-8"), "SLM analysis choices")
            except (OSError, ValueError):
                cache = {}
            cache = cache if isinstance(cache, dict) else {}
            cache[key] = {**_analysis_cache_read(path, key), **{str(band): known[str(band)] for band in missing}}
            write_readable_json(path, cache)
        except OSError:
            pass  # a read-only location only measures again next time
    return {band: known[str(band)] for band in bands}, missing


@contextmanager
def _rearrangement_analysis_workspace(shape_yx, capacity):
    """The compiled module, a cuBLAS handle and the analysis buffers of one panel, without a preparation."""
    cp, cublas, load_library = _rearrangement_cuda_imports()
    library, gemm, set_workspace = _rearrangement_blas(load_library)
    h, w = shape_yx
    padded = (w // 2 + 16) // 16 * 16
    pool = cp.cuda.MemoryPool()
    stream = cp.cuda.Stream(non_blocking=True)
    with cp.cuda.using_allocator(pool.malloc), stream:
        module = _rearrangement_module(cp)
        gpu = dict(cp=cp, cublas=cublas, library=library, gemm=gemm, stream=stream, shape=(h, w), padded=padded,
                   kernels={name: module.get_function(name) for name in _REARRANGEMENT_KERNELS},
                   blas_workspace=cp.empty(4 * 1024 * 1024, cp.uint8),
                   field=cp.zeros((2, h, 2 * padded), cp.float16), roots=cp.zeros((capacity, padded, 2), cp.float16),
                   forward=cp.zeros((2 * padded, capacity), cp.float16), projected=cp.zeros((capacity, 2, h), cp.float32),
                   alpha=np.asarray(1, np.float32), beta=np.asarray(0, np.float32))
        gpu["handle"] = cublas.create()
        try:
            cublas.setStream(gpu["handle"], stream.ptr)
            status = set_workspace(gpu["handle"], gpu["blas_workspace"].data.ptr, gpu["blas_workspace"].nbytes)
            if status:
                raise RuntimeError(f"SLM cuBLAS workspace setup failed ({status})")
            yield gpu
            stream.synchronize()
        finally:
            cublas.destroy(gpu["handle"])
            gpu.clear()
            pool.free_all_blocks()


def warm_rearrangement_gpu(shape_yx, maximum_bands=_ANALYSIS_WARM_BANDS):
    """Compile the rearrangement CUDA module and choose the analysis product of
    every band count up to maximum_bands for one panel, as a first preparation
    would; the report names each choice. bin/warm_slm_gpu.bat runs it."""
    bands = range(8, maximum_bands + 1, 8)
    with _rearrangement_analysis_workspace(shape_yx, bands[-1]) as gpu:
        choices, measured = _rearrangement_analysis_choices(gpu, bands)
        cp = gpu["cp"]
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
    h, w = shape_yx
    lines = [f"{name}: SLM rearrangement CUDA module compiled; analysis product of {h} x {w} panels "
             f"for {len(choices)} band counts ({len(measured)} measured now) in {_analysis_cache_file()}",
             f"{'bands':>6}  {'choice':<14}{'us':>9}{'cuBLAS us':>11}"]
    for band, entry in choices.items():
        lines.append(f"{band:>6}  {entry['choice']:<14}{entry['us'][entry['choice']]:>9.1f}{entry['us']['cublas']:>11.1f}")
    return "\n".join(lines)


def _rearrangement_record_views(buffer, sites, capacity):
    """Typed views of one map record (its byte size when buffer is None)."""
    views, offset = {}, 0
    for name, dtype, count in (("point_fy", "f8", sites), ("frequencies", "f8", capacity), ("rotor", "c8", sites),
                               ("goal", "f4", sites), ("cap", "f4", sites), ("gate", "f4", sites), ("scalars", "f4", 3),
                               ("band_start", "i4", capacity + 1), ("point_band", "i4", sites),
                               ("band_point", "i4", sites), ("meta", "i4", 2)):
        size = np.dtype(dtype).itemsize * count
        if buffer is not None:
            views[name] = buffer[offset:offset + size].view(dtype)
        offset += size
    return views if buffer is not None else offset


def _rearrangement_record(views, bands, signed_y, frequencies, band):
    """Write one roster: band and signed Y frequency of each point, points grouped by band, band X frequencies."""
    count = len(bands)
    order = np.argsort(bands, kind="stable")
    views["band_start"][:] = count
    views["band_start"][:band + 1] = np.searchsorted(bands[order], np.arange(band + 1))
    views["point_band"][:] = -1
    views["point_band"][:count] = bands
    views["point_fy"][:] = 0
    views["point_fy"][:count] = signed_y
    views["band_point"][:] = 0
    views["band_point"][:count] = order
    views["frequencies"][:] = 0
    views["frequencies"][:len(frequencies)] = frequencies


def _rearrangement_roster(gpu, bands, signed_y, frequencies, band):
    """Bind one prepare-time roster to the captured evaluations."""
    row = np.zeros(gpu["record"].size, np.uint8)
    _rearrangement_record(_rearrangement_record_views(row, gpu["actual"].size, gpu["capacity"]),
                          bands, signed_y, frequencies, band)
    gpu["record"].set(row, stream=gpu["stream"])
    _rearrangement_roots(gpu, band)


def _rearrangement_segments(kind, complete):
    """Captured graphs (sweeps, corrections, measure first, ghost watch (cap, detection, gate, corrections watched
    before) or ()) that make one map of this kind."""
    sweeps, corrections, _ = _TRACK_SCHEDULE[kind]
    if kind == "hold":
        return []
    if kind in ("last", "leap_last"):
        # A filled last map is measured as prepared; a refined one (partially
        # filled, or of a plan in which no retained trap moves) is refined on the
        # full focal plane between its sweeps and its corrections.
        return [(0, 0, True, ())] if complete else [(sweeps, 0, False, ()), (0, corrections, True, ())]
    # Surplus atoms imply a filled target: "dark_last" corrects the prepared destination.
    cap, detection, before = _TRACK_WATCH_LIMITS[kind]
    last = kind.endswith("dark_last")
    return [(sweeps, corrections, last, (cap, detection, _TRACK_WATCH_GATE[last], before))]


def _rearrangement_watch(gpu, band, detection):
    """Enqueue the ghost watch of the displayed map (see _TRACK_WATCH) into the reserved roster slots and bands."""
    h, w = gpu["shape"]
    kernels, plane, found = gpu["kernels"], gpu["watch_plane"], gpu["watch_found"]
    sites, centre = gpu["actual"].size, gpu["watch_centre"]
    blocks = ((plane.size + 255) // 256,)
    kernels["watch_display"](blocks, (256,), (
        gpu["field"], plane, *map(np.int32, (h, w, gpu["padded"], plane.shape[1], *centre)), gpu["actual"], gpu["meta"],
        np.float32(detection), found))
    gpu["watch_plan"].fft(plane, plane, -1)
    for overflow, grid, block in ((0, blocks, 256), (1, (gpu["processors"],), 1024)):
        kernels["peaks"](grid, (block,), (plane, *map(np.int32, (h, w, *plane.shape, *centre)), found,
                                          np.float32(_TRACK_WATCH_EXCLUSION), gpu["point_fy"], gpu["point_band"],
                                          gpu["frequencies"], gpu["meta"], np.int32(overflow)))
    kernels["select_watch"]((1,), (1024,), (
        plane, found, *map(np.int32, (h, w, *plane.shape, *centre, _TRACK_WATCH, _TRACK_WATCH_BANDS)),
        np.float32(_TRACK_WATCH_SEPARATION),
        *map(np.int32, (band, sites - _TRACK_WATCH, sites)), gpu["point_fy"], gpu["point_band"], gpu["band_point"],
        gpu["band_start"], gpu["frequencies"], gpu["actual"]))
    _rearrangement_roots(gpu, band + _TRACK_WATCH_BANDS, sites - _TRACK_WATCH)


def _rearrangement_track_map(gpu, band, sweeps, corrections, measure, watch):
    """Enqueue one map on the bound record: roots, then a measurement or WGS sweeps, then corrections, each listed
    one preceded by the ghost watch (cap, detection, gate, corrections watched before; () unwatched)."""
    kernels, sites = gpu["kernels"], np.int32(gpu["actual"].size)
    _rearrangement_roots(gpu, band)
    kernels["track_load"]((1,), (128,), (gpu["track_c"], gpu["track_cancel"], gpu["cap"], gpu["meta"],
                                         gpu["coefficients"], sites))
    if measure:
        _rearrangement_propagate(gpu, band, "measure")
    for _ in range(sweeps):
        _rearrangement_propagate(gpu, band, "encode")
        kernels["track_sweep"]((1,), (128,), (gpu["actual"], gpu["goal"], gpu["rotor"], gpu["cap"], gpu["gate"],
                                              gpu["scalars"], gpu["meta"], gpu["track_c"], gpu["track_cancel"],
                                              gpu["track_accepted"], gpu["coefficients"], sites,
                                              np.float32(_TRACK_KIM_EXPONENT)))
    cap, detection, gate, before = watch or (1., 1., 1., ())
    evaluated, watched = band, 0
    for correction in range(corrections):
        if correction in before:
            _rearrangement_watch(gpu, band, detection)
            evaluated, watched = band + _TRACK_WATCH_BANDS, correction
        kernels["track_residual"]((1,), (128,), (
            gpu["actual"], gpu["goal"], gpu["cap"], gpu["gate"], gpu["scalars"], gpu["meta"], gpu["coefficients"], sites,
            np.int32(_TRACK_WATCH), *np.sqrt(np.float32((cap, gate))), np.int32(correction - watched),
            gpu["track_history"], gpu["track_steps"]))
        _rearrangement_propagate(gpu, evaluated, "correct")


def _rearrangement_evaluate(gpu, band, values, operation):
    """Upload roster coefficients, evaluate once and read the roster fields (prepare-time solves)."""
    count, host = len(values), gpu["host_values"]
    host[:count] = values
    host[count:] = 0
    gpu["graphs"][band, operation].launch(gpu["stream"])
    gpu["stream"].synchronize()
    return gpu["host_fields"][:count].astype(np.complex128)


def _rearrangement_bind(points, shape):
    """Group each frame's points by X frequency once for a whole sequence.

    One band carries one X root; every point keeps its own Y frequency, so
    points of one column share a band whatever their fractional Y position.
    Returns each point's band, its signed Y frequency, the band X frequencies of
    all frames, and their per-frame counts and offsets."""
    signed = np.asarray(points, np.float64) - np.asarray(shape) // 2
    frame_count, sites_per_frame = signed.shape[:2]
    frame_ids = np.repeat(np.arange(frame_count), sites_per_frame)
    x = signed[..., 1].reshape(-1)
    order = np.lexsort((x, frame_ids))
    ordered_frames, ordered_x = frame_ids[order], x[order]
    starts = np.ones(len(x), bool)
    starts[1:] = (ordered_frames[1:] != ordered_frames[:-1]) | (ordered_x[1:] != ordered_x[:-1])
    inverse = np.empty_like(order)
    inverse[order] = np.cumsum(starts) - 1
    counts = np.bincount(ordered_frames[starts], minlength=frame_count).astype(np.int32)
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    lookup = (inverse.reshape(frame_count, sites_per_frame) - offsets[:-1, None]).astype(np.int32)
    return lookup, signed[..., 0], ordered_x[starts], counts, offsets


def _rearrangement_near(cp, points, shape):
    """Native focal pixels within _ARRAY_STRAY_RADIUS of integer sites."""
    reach = int(_ARRAY_STRAY_RADIUS)
    disc = np.argwhere(np.hypot(*np.mgrid[-reach:reach + 1, -reach:reach + 1]) <= _ARRAY_STRAY_RADIUS) - reach
    near = cp.zeros(tuple(shape), bool)
    near[tuple(cp.asarray(np.clip(np.asarray(points, np.int64)[:, None] + disc, 0,
                                  np.asarray(shape) - 1).reshape(-1, 2)).T)] = True
    return near


def _rearrangement_focal_quality(gpu, points):
    """Power fraction near the traps of the displayed codes on the native focal grid."""
    cp = gpu["cp"]
    with gpu["plane_plan"]:
        intensity = cp.fft.fftshift(abs(cp.fft.fft2(gpu["physical_pupil"] * cp.exp(
            1j * (gpu["codes"] * np.float32(2 * np.pi / 256) + gpu["incident"])))) ** 2)
    near = _rearrangement_near(cp, points, gpu["shape"])
    return float(cp.sum(intensity[near]) / cp.sum(intensity))


def _rearrangement_stray(gpu, points):
    """Stray light of the gates on the displayed codes: the strongest local maximum of the twofold zero-padded
    focal plane farther than _TRACK_WATCH_EXCLUSION from every integer trap site, over the weakest trap. It
    runs after the last map has been handed over, on the workspace's own plan and memory pool."""
    from cupyx.scipy.ndimage import maximum_filter  # noqa: PLC0415

    cp, shape = gpu["cp"], np.asarray(gpu["shape"])
    area, plane = int(np.prod(shape)), 2 * shape
    sites = (np.asarray(points, np.int64) - shape // 2) * 2 + plane // 2
    reach = int(np.ceil(2 * _TRACK_WATCH_EXCLUSION))
    disc = np.argwhere(np.hypot(*np.mgrid[-reach:reach + 1, -reach:reach + 1]) <= 2 * _TRACK_WATCH_EXCLUSION) - reach
    gpu["kernels"]["display"](((area + 255) // 256,), (256,),
                              (gpu["codes"], gpu["physical_pupil"], gpu["incident"], gpu["optical"], np.int32(area)))
    with gpu["stray_plan"]:
        power = cp.abs(cp.fft.fftshift(cp.fft.fft2(gpu["optical"], s=tuple(plane)))) ** 2
    maxima = (power > 0) & (power == maximum_filter(power, size=3, mode="wrap"))
    maxima[tuple(cp.asarray((sites[:, None] + disc).reshape(-1, 2) % plane).T)] = False
    return float(cp.max(cp.where(maxima, power, 0)) / cp.min(power[tuple(cp.asarray(sites).T)]))


def _rearrangement_plane_wgs(gpu, points, amplitudes, iterations, locked, stop_requested=None):
    """Kim-WGS of the displayed codes on the full native focal plane (see _ARRAY_*); re-encodes the codes.
    Enqueued on the current stream without host synchronization. The focal plane stays unshifted (the
    trap and near indices move instead), both transforms run in place, and each other pass over the
    plane is one kernel."""
    from cupyx.scipy import fft  # noqa: PLC0415

    cp, kernels, (h, w) = gpu["cp"], gpu["kernels"], gpu["shape"]
    near = cp.fft.ifftshift(_rearrangement_near(cp, points, gpu["shape"]))
    sites = (np.asarray(points, np.int64) - np.asarray(gpu["shape"]) // 2) % np.asarray(gpu["shape"])
    index = cp.asarray((sites[:, 0] * w + sites[:, 1]).astype(np.int32))
    goal = cp.asarray(amplitudes / np.mean(amplitudes), cp.float32)
    weight, unit, limit = goal.copy(), cp.zeros(len(goal), cp.complex64), cp.zeros(1, cp.float32)
    field, area = gpu["optical"], np.int32(h * w)
    blocks = ((h * w + 255) // 256,)
    kernels["display"](blocks, (256,), (gpu["codes"], gpu["physical_pupil"], gpu["incident"], field, area))
    for iteration in range(iterations):
        if stop_requested is not None and stop_requested():
            raise InterruptedError("SLM rearrangement preparation stopped")
        fft.fft2(field, overwrite_x=True, plan=gpu["plane_plan"])
        kernels["plane_traps"]((1,), (32,), (field, index, goal, weight, unit, limit, np.int32(len(goal)),
                                             np.int32(iteration >= iterations - locked),
                                             np.float32(_TRACK_KIM_EXPONENT), np.float32(np.sqrt(_ARRAY_STRAY))))
        kernels["plane_cap"](blocks, (256,), (field, near, limit, area))
        fft.ifft2(field, overwrite_x=True, norm="forward", plan=gpu["plane_plan"])
        kernels["plane_unit"](blocks, (256,), (field, gpu["physical_pupil"], area))
    command = cp.mod(cp.angle(field) - gpu["incident"], np.float32(2 * np.pi)).astype(cp.float32)
    kernels["encode_phase"](blocks, (256,), (command, gpu["codes"], area))


def _rearrangement_solve_array(gpu, points, intensities, iterations, seed, stop_requested):
    """Solve one complete static array before the photograph (see _ARRAY_*).

    Sparse Kim-WGS of `iterations` sweeps from each of _ARRAY_SEEDS random
    phases; the start with the most light near its traps is refined on the
    full focal plane. Its codes remain displayed in the workspace."""
    bands, signed_y, frequencies, counts, _ = _rearrangement_bind(np.asarray(points, np.float64)[None], gpu["shape"])
    band = (int(counts[0]) + 7) // 8 * 8
    _rearrangement_roster(gpu, bands[0], signed_y[0], frequencies, band)
    amplitude = np.sqrt(np.asarray(intensities, np.float64))
    amplitude /= np.linalg.norm(amplitude)
    random = np.random.default_rng(seed)
    start, start_near = None, -1.
    for _ in range(_ARRAY_SEEDS):
        coefficient = amplitude * np.exp(1j * random.uniform(-np.pi, np.pi, len(amplitude)))
        for _ in range(iterations):
            if stop_requested is not None and stop_requested():
                raise InterruptedError("SLM rearrangement preparation stopped")
            field = _rearrangement_evaluate(gpu, band, coefficient, "encode")
            relative = abs(field) / amplitude
            weight = abs(coefficient) * np.clip(relative.mean() / relative, .2, 5) ** _TRACK_KIM_EXPONENT
            coefficient = weight / np.linalg.norm(weight) * np.exp(1j * np.angle(field))
        near = _rearrangement_focal_quality(gpu, points)
        if near > start_near:
            start, start_near = gpu["codes"].copy(), near
    gpu["codes"][...] = start
    _rearrangement_plane_wgs(gpu, points, amplitude, _ARRAY_PLANE_ITERATIONS, _ARRAY_LOCKED, stop_requested)
    field = _rearrangement_evaluate(gpu, band, amplitude, "measure")
    relative = abs(field) ** 2 / amplitude ** 2
    return {"codes": gpu["codes"].copy(), "field": _frozen(field.astype(np.complex64)),
            "ratio": float(relative.max() / relative.min())}


def prepare_rearrangement(
    source_yx: object, target_yx: object, *, shape_yx: tuple[int, int],
    pupil_amplitude: object, minimum_separation: float,
    pupil_phase: object | None = None, source_intensities: object | None = None,
    target_intensities: object | None = None,
    endpoint_iterations: int = 60, seed: int = 0,
    endpoint_data: Mapping[str, object] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    phase_center_yx: object | None = None,
) -> dict[str, object]:
    """Prepare one source optical state, the destination phases and the GPU workspace.

    A supplied source_phase (or native codes) stays exact for initial application;
    otherwise the source array is solved like the destination. The complete
    destination array is solved independently (endpoint_iterations sparse
    Kim-WGS sweeps per random start, then the full focal plane); its focal
    phases steer the retained traps during transport and its codes are the last
    map when every target is filled. The caller owns and closes this serial
    workspace.
    """
    if stop_requested is not None and stop_requested():
        raise InterruptedError("SLM rearrangement preparation stopped")
    if (isinstance(endpoint_iterations, bool) or int(endpoint_iterations) != endpoint_iterations
            or endpoint_iterations < 1):
        raise ValueError("endpoint_iterations must be a positive integer")
    geometry = prepare_rearrangement_geometry(
        source_yx, target_yx, shape_yx=shape_yx, minimum_separation=minimum_separation,
    )
    shape, source, target = geometry["shape_yx"], geometry["source_yx"], geometry["target_yx"]
    phase_center = np.asarray(np.asarray(shape) // 2 if phase_center_yx is None else phase_center_yx, np.float64)
    if phase_center.shape != (2,) or not np.all(np.isfinite(phase_center)):
        raise ValueError("phase_center_yx must be a finite Y,X pair")
    intensities = []
    for name, values, points in (("source", source_intensities, source), ("target", target_intensities, target)):
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
    gpu = _prepare_rearrangement_gpu(geometry, pupil_amplitude, pupil_phase, stop_requested)
    try:
        cp, stream = gpu["cp"], gpu["stream"]
        with _rearrangement_gpu_active(gpu["keeper"]), stream:
            if endpoint_data is None:
                _rearrangement_solve_array(gpu, source, intensities[0], int(endpoint_iterations), int(seed), stop_requested)
                codes = gpu["codes"].get(stream=stream)
                phase = codes.astype(np.float64) * (2 * np.pi / 256)
            elif "source_phase" in endpoint_data:
                phase = canonical_phase(endpoint_data["source_phase"], shape)
                gpu["kernels"]["encode_phase"](((int(np.prod(shape)) + 255) // 256,), (256,),
                    (cp.asarray(phase, cp.float32), gpu["codes"], np.int32(np.prod(shape))))
                codes = gpu["codes"].get(stream=stream)
            else:
                codes = np.asarray(endpoint_data["source_phase_codes"]).copy()
                phase = codes.astype(np.float64) * (2 * np.pi / 256)
            # The exact displayed source, independent of the sparse GPU model.
            optical = gpu["pupil_cpu"].astype(np.float64) * np.exp(
                1j * (np.asarray(phase, np.float64) + gpu["incident_cpu"]))
            actual = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(optical)))[tuple(source.T)]
            power = abs(actual) ** 2
            if not np.all(np.isfinite(power)) or np.any(power <= 0):
                raise ValueError("prepared source has a zero or invalid bright-site field")
            relative = power / intensities[0]
            destination = _rearrangement_solve_array(
                gpu, target, intensities[1], int(endpoint_iterations), int(seed) + 1, stop_requested)
            # The filled last map's stray light, measured once on the still displayed destination.
            destination_stray = _rearrangement_stray(gpu, target)
            gpu["initial_phase"] = cp.asarray(phase, cp.float32)
            gpu["target_codes"] = destination["codes"]
        free_memory, total_memory = cp.cuda.runtime.memGetInfo()
        return {
            **geometry, "gpu": gpu, "close": gpu["close"],
            "phase_center_yx": _frozen(phase_center),
            "target_field": destination["field"], "target_support_intensity_ratio": destination["ratio"],
            "target_stray_intensity_ratio": destination_stray,
            "gpu_info": {
                "device_name": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
                "device_memory_free_bytes": free_memory, "device_memory_total_bytes": total_memory,
                "host_output_frame_bytes": int(np.prod(shape)),
                "warm_buffer_bytes": int(gpu["keeper"]["scratch"].nbytes),
                "warm_interval_seconds": .020,
            },
            "source_intensities": intensities[0], "target_intensities": intensities[1],
            "source_field": _frozen(actual.astype(np.complex64)),
            "source_brightness": float(np.sqrt(np.sum(power) / np.sum(intensities[0], dtype=np.float64))),
            "source_support_intensity_ratio": float(relative.max() / relative.min()),
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
        separation = _scalar(inputs.pop("minimum_separation"), "minimum_separation", nonnegative=True)
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
        yield prepared, reused
        if prepared["gpu"]["keeper"]["error"] is not None:
            raise RuntimeError("SLM GPU warming failed") from prepared["gpu"]["keeper"]["error"]
    except BaseException as error:
        # SDK, classification, quality and Stop failures do not poison optics.
        # These are the existing CUDA-library errors and our own checked calls.
        gpu_fault = (type(error).__module__.startswith(("cupy_backends.cuda.", "cupy.cuda.", "cuda.bindings."))
                     or str(error).startswith("SLM cuBLAS "))
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
        # Nothing moves before the fade ends: that is the whole source roster.
        fade_clearance = prepared["source_clearance"]
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
    motion_frames: int = 16,
    support_tolerance: float = SPOT_SUPPORT_TOLERANCE,
    motion_support_tolerance: float = 1.10,
    require_converged: bool = True, stop_requested: Callable[[], bool] | None = None,
    frame_ready: Callable[[int, np.ndarray], None] | None = None,
    sampled: Mapping[str, object] | None = None,
    retain_phase_sequence: bool = True,
) -> dict[str, object]:
    """Emit N native maps that track one continuous optical state.

    Every map starts from the previous map's coefficients: Kim-weighted WGS
    sweeps keep each retained trap's focal phase inside a window around its
    steered reference, which moves with the trap's progress from the actual
    source phase to the prepared destination phase (one common offset, the one
    that makes the largest turn smallest, removed) when the traps move; damped
    phase-gradient corrections of the displayed codes, Anderson-mixed while the
    map is outside its gates, then equalize it. Every intermediate map except a
    repeated one, and a removal map that is also the last, watches its strongest
    stray peaks before the corrections its kind lists (_TRACK_WATCH_LIMITS), and
    the corrections hold them at a fraction of the weakest retained trap. The
    numbers of sweeps and corrections are a fixed schedule per map kind
    (_TRACK_SCHEDULE) chosen offline, the kind following from the plan alone
    (role, retained and surplus counts, step and phase turn per map): each map
    is one record upload and one graph launch (a refined last map two, around
    its full-plane iterations; a repeated map none), enqueued one map ahead of
    the map being published, and published as soon as the GPU has finished it.
    The first map is enqueued from its own state alone; when sampled is None
    (Fixed) and that map holds every site at its source (a fade), the
    trajectory is sampled and validated while the GPU runs it, and nothing is
    published if that fails. The removal map and the next one (the removal map
    alone when it is also the last) also darken the surplus-atom sites (below
    _TRACK_DARK_LIMIT of their initial traps) and keep empty sources dim. A
    filled last map of a moving plan is the prepared destination; a partially
    filled one, and the last map of a plan in which no retained trap moves, is
    refined on the full focal plane. A plan in which no retained trap moves
    keeps the source phases, and between its first map after the darkened maps
    and its last it repeats the previous map ("hold": no evaluation and no
    watch; it reports the map it repeats) while the requested weights stay
    unchanged. Otherwise only roster points and the watch's binned plane are
    evaluated, with the sparse fractional Fourier model of the complete
    aperture; once the last map has been handed over, its stray light is
    measured on the twofold zero-padded focal plane of its delivered codes (a
    filled last map of a moving plan reports the prepared destination's), and
    the full focal plane is diagnosed after playback. The read-back roster
    fields report every map against its gates (motion_support_tolerance, the
    actual last map support_tolerance, surplus-atom sites on the darkened maps,
    stray light within _TRACK_WATCH_GATE of the weakest retained trap) in
    gate_misses: a map outside its gates is still published and reported,
    require_converged raises after the sequence, and only a non-finite field
    stops publication. Trap power relative to the initial traps is reported
    (brightness_*_to_initial), not gated. Empty/no-change plans hold the exact
    input phase and return an empty sequence. frame_ready receives each
    immutable host map in order; maps are retained only when requested. The
    caller must stop playback on a failure.
    """
    import time  # noqa: PLC0415

    started = time.perf_counter()
    if not prepared["gpu"]:
        raise RuntimeError("SLM rearrangement workspace is closed")
    if stop_requested is not None and stop_requested():
        raise InterruptedError("SLM rearrangement stopped")
    if (isinstance(motion_frames, bool) or int(motion_frames) != motion_frames or motion_frames < 0
            or (motion_frames == 0 and not rearrangement_is_noop(prepared, plan))):
        raise ValueError("motion_frames must be a positive integer")
    motion_frames = int(motion_frames)
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
        "motion_support_tolerance": motion_tolerance, "support_tolerance": tolerance,
        "phase_center_yx": prepared["phase_center_yx"], "field_phase_reference": "fft-center",
        "source_field": prepared["source_field"], "target_field": prepared["target_field"],
        "target_requested_intensities": prepared["target_intensities"],
        "phase_trajectory": "tracked-steered-to-destination", "phase_encoding": "uint8:2pi/256",
        "quality_scope": "retained-trap-uniformity-surplus-atom-sites-and-stray-peaks",
        "surplus_site_limit": _TRACK_DARK_LIMIT, "stray_intensity_limits": _TRACK_WATCH_GATE,
    }
    if rearrangement_is_noop(prepared, plan):
        return {
            **plan, **metadata, "noop": True, "phase_codes": () if retain_phase_sequence else None,
            "motion_yx": _frozen(path[:1]), "fraction": _frozen(np.zeros(1)),
            "movement_fraction": np.empty(0), "frame_support_tolerances": np.empty(0),
            "sites_yx": np.empty((0, source_count, 2)), "actual_fields": np.empty((0, source_count), np.complex64),
            "desired_amplitudes": np.empty((0, source_count)), "active_sites": np.empty((0, source_count), bool),
            "motion_frames": 0, "fade_frames": 0, "iterations": (), "correction_updates": (), "watched_stray_peaks": (),
            "emitted_frame_count": 0, "frame_solve_ms": np.empty(0), "frame_ready_ms": np.empty(0),
            "clearance": rearrangement_clearance(path), "fade_clearance": np.inf,
            "surplus_stationary_clearance": np.inf, "maximum_step": 0., "recommended_motion_frames": 0,
            "support_intensity_ratios": np.empty(0), "retained_intensity_ratios": np.empty(0),
            "surplus_site_intensity_ratios": np.empty(0), "stray_intensity_ratios": np.empty(0),
            "gate_misses": np.empty(0, bool), "synthesis_coefficients": np.empty((0, 0), np.complex64),
            "brightness_minimum_to_initial": np.empty(0), "brightness_maximum_to_initial": np.empty(0),
            "brightness_mean_to_initial": np.empty(0), "phase_change_from_initial_rms_rad": np.empty(0),
            "phase_step_max_rad": np.empty(0),
            "converged": True, "quality_accepted": True, "quality_evaluated": True,
            "timing_ms": {"prepare_frame_state": 0., "callback": 0., "first_frame_ready": 0.,
                          "total": (time.perf_counter() - started) * 1000},
        }
    if sampled is not None and (sampled["motion_frames"] != motion_frames or len(sampled["sites_yx"]) != motion_frames):
        raise ValueError("sampled trajectory must match motion_frames")
    frame_tolerances = np.full(motion_frames, motion_tolerance)
    frame_tolerances[-1] = tolerance
    metadata["frame_support_tolerances"] = _frozen(frame_tolerances)
    # Unselected sources fade on the first map, which holds every site at its
    # source: that map needs no sampled trajectory.
    fade_frames = int(selected_count < source_count)
    if sampled is None and not fade_frames:
        sampled = sample_rearrangement(prepared, plan, motion_frames=motion_frames)
    initial = prepared["source_field"].astype(np.complex128)
    initial_amplitude = abs(initial)
    destination_amplitude = prepared["source_brightness"] * np.sqrt(prepared["target_intensities"][target_indices])
    moving = bool(np.any(path != path[:1]))
    # Every map evaluates its retained traps. The removal map and the next one
    # (the removal map alone when it is also the last) also evaluate the
    # unselected sources after them: surplus-atom sites first, darkened to the
    # gate, then empty sources, only held below their cap. A point within
    # _ARRAY_STRAY_RADIUS of a retained trap of that map belongs to the trap
    # and is left alone. Later maps leave those columns out.
    darkened = min(fade_frames + 1, motion_frames) if len(removed) else 0
    empty = np.ones(source_count, bool)
    empty[source_indices] = empty[removed] = False
    dark = np.concatenate((removed, np.flatnonzero(empty)))
    surplus = np.arange(len(dark)) < len(removed)
    initial_surplus = abs(initial[removed]) ** 2
    # Trap phases are compared about the optical axis (phase_center_yx): in the
    # FFT-centre gauge of the computed fields a moving trap gains coordinate phase.
    offset = (prepared["phase_center_yx"] - np.asarray(shape) // 2) / np.asarray(shape)
    start = initial[source_indices] * np.exp(
        2j * np.pi * np.sum((source[source_indices] - np.asarray(shape) // 2) * offset, axis=-1))
    source_phase = np.angle(start)
    destination = prepared["target_field"][target_indices].astype(np.complex128) * np.exp(
        2j * np.pi * np.sum((target[target_indices] - np.asarray(shape) // 2) * offset, axis=-1))
    # Each retained trap rotates along its own shortest branch to the destination
    # phase after one common phase, which leaves every intensity unchanged. The
    # common phase centres the smallest arc that holds every turn (opposite the
    # widest empty arc), so the largest turn and every per-map step are smallest.
    rotation = np.angle(destination * np.exp(-1j * source_phase))
    ordered = np.sort(rotation)
    gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
    widest = int(np.argmax(gaps))
    common = float(np.angle(np.exp(1j * (ordered[widest] + gaps[widest] / 2 + np.pi))))
    rotation = np.angle(np.exp(1j * (rotation - common)))
    complete = selected_count == len(target)
    crowded = ("crowded " if selected_count < _TRACK_FEW or len(removed) >= _TRACK_CROWDED * selected_count
               or not moving else "")
    # A filled last map of a moving plan, after its darkened maps, shows the prepared destination.
    shown = complete and moving and darkened < motion_frames
    largest_turn = np.max(abs(rotation))
    # A refined last map's capped full-plane iterations: more with non-uniform requests or in a crowded plan.
    refinement = _TRACK_VACANT_PLANE[bool(crowded or np.ptp(destination_amplitude))]

    def frame_state(sites, progress):
        """Desired amplitudes, darkened-site caps and gates, gauge, steering, steps over _TRACK_LEAP and bound
        rosters of the first len(sites) maps."""
        maps = np.arange(1, len(sites) + 1)
        # Phases follow the movement to the destination phases. A plan in which no
        # retained trap moves keeps them at the source phases (see _TRACK_SCHEDULE);
        # its weights change over the maps after the fade instead.
        progressed = (progress.copy() if moving else
                      np.maximum(0., maps - fade_frames) / max(1, motion_frames - fade_frames))
        progressed[motion_frames - 1:] = 1.
        steering = progressed if moving else np.zeros(len(maps))
        # Optical weights can change even when no trap changes position. Keep that
        # authored transition separate from the geometrical movement fraction.
        # Without a fade the retained traps leave their measured source fields;
        # with one their requested weights move with the phases from the source
        # request to the destination request: a jump on the removal map would step
        # the trap depths there, which its fixed schedule does not absorb.
        desired = np.zeros((len(sites), source_count))
        if fade_frames:
            requested = prepared["source_intensities"][source_indices]
            desired[:, source_indices] = prepared["source_brightness"] * np.sqrt(
                requested + progressed[:, None] * (prepared["target_intensities"][target_indices] - requested))
        else:
            desired[:, source_indices] = ((1 - progressed[:, None]) * initial_amplitude[source_indices]
                                          + progressed[:, None] * destination_amplitude)
        kept = sites[:darkened, source_indices]
        free = np.array([np.min(np.hypot(*np.moveaxis(source[dark][:, None] - points, -1, 0)), axis=1, initial=np.inf)
                         for points in kept]).reshape(len(kept), len(dark)) > _ARRAY_STRAY_RADIUS
        cap = np.where(free, np.where(surplus, 0., np.sqrt(_TRACK_EMPTY_LIMIT) * abs(initial[dark])), np.inf)
        gate = np.where(free & surplus, np.sqrt(_TRACK_DARK_LIMIT) * abs(initial[dark]), np.inf)
        dark_roster = np.concatenate((kept, np.broadcast_to(source[dark], (len(kept), len(dark), 2))), axis=1)
        gauge = np.exp(2j * np.pi * np.sum((sites[:, source_indices] - np.asarray(shape) // 2) * offset, axis=-1))
        leap = np.max(np.linalg.norm(np.diff(sites[:, source_indices], axis=0, prepend=source[None, source_indices]),
                                     axis=-1), axis=1) > _TRACK_LEAP
        return desired, free, cap, gate, gauge, steering, leap, (
            _rearrangement_bind(dark_roster, shape), _rearrangement_bind(sites[darkened:, source_indices], shape))

    def kind(index):
        if index < darkened:
            return crowded + ("dark_last" if index == motion_frames - 1 else
                              "release" if index < fade_frames else "next")
        if index == motion_frames - 1:
            return "leap_last" if leap[index] else "last"
        if leap[index]:
            return "leap"
        if index in (darkened, fade_frames):
            return "first"
        if not moving:
            return "hold" if np.array_equal(desired[index], desired[index - 1]) else "transport"
        return ("many" if selected_count >= _TRACK_MANY[0]
                and max(darkened, fade_frames) + _TRACK_MANY[1] < index < motion_frames - 1 - _TRACK_MANY[2]
                and largest_turn * (steering[index] - steering[index - 1]) <= _TRACK_TURN else crowded + "transport")

    gpu = prepared["gpu"]
    cp, stream = gpu["cp"], gpu["stream"]
    area = int(np.prod(shape))
    fields = np.full((motion_frames, source_count), np.nan + 0j, np.complex64)
    synthesis = np.full((motion_frames, selected_count), np.nan + 0j, np.complex64)
    ratios, surplus_ratios, stray = np.zeros(motion_frames), np.zeros(motion_frames), np.zeros(motion_frames)
    evaluations, corrections = np.zeros(motion_frames, np.int32), np.zeros(motion_frames, np.int32)
    watched = np.zeros(motion_frames, np.int32)
    frame_solve_ms, frame_ready_ms = np.zeros(motion_frames), np.zeros(motion_frames)
    callback_ms, emitted_count = 0., 0
    retained_frames = [] if retain_phase_sequence else None
    transfer_frames = retain_phase_sequence or frame_ready is not None
    with _rearrangement_gpu_active(gpu["keeper"]), stream:
        runtime, pool, copy = cp.cuda.runtime, gpu["output_pool"], gpu["copy_stream"]
        roster, record_size = gpu["actual"].size, gpu["record"].size
        # One pinned block: the start coefficients, every map's record, and the
        # read-back roster fields and encoded coefficients of every map.
        pinned = pool.malloc(8 * roster * (1 + 2 * motion_frames) + motion_frames * record_size)
        carried = np.frombuffer(pinned, np.complex64, count=roster)
        readback = np.frombuffer(pinned, np.complex64, count=2 * motion_frames * roster,
                                 offset=carried.nbytes).reshape(2, motion_frames, roster)
        rows = np.frombuffer(pinned, np.uint8, count=motion_frames * record_size,
                             offset=carried.nbytes + readback.nbytes).reshape(motion_frames, record_size)
        rows[:] = 0
        carried[:] = 0
        carried[:selected_count] = initial[source_indices] / np.linalg.norm(initial[source_indices])
        runtime.memcpyAsync(gpu["track_c"].data.ptr, carried.ctypes.data, carried.nbytes,
                            runtime.memcpyHostToDevice, stream.ptr)
        runtime.memsetAsync(gpu["track_cancel"].data.ptr, 0, gpu["track_cancel"].nbytes, stream.ptr)
        begun, ended, copied = [None] * motion_frames, [None] * motion_frames, [None] * motion_frames
        outputs, pending = [None] * motion_frames, []

        def retire():
            nonlocal callback_ms, emitted_count
            index = pending.pop(0)
            ended[index].synchronize()
            frame_solve_ms[index] = cp.cuda.get_elapsed_time(begun[index], ended[index])
            count = selected_count + (len(dark) if index < darkened else 0)
            field = readback[0, index, :count].astype(np.complex128)
            if not np.all(np.isfinite(field)):
                raise RuntimeError(f"SLM frame {index} has a non-finite encoded field; publication stopped")
            fields[index, source_indices] = field[:selected_count]
            fields[index, dark[:count - selected_count]] = field[selected_count:]
            relative = abs(field[:selected_count]) ** 2 / desired[index, source_indices] ** 2
            ratios[index] = relative.max() / relative.min()
            if index < darkened:
                surplus_ratios[index] = np.max(abs(field[selected_count:selected_count + len(removed)]) ** 2
                                               / initial_surplus, where=free[index, :len(removed)], initial=0.)
            peaks = readback[0, index, roster - _TRACK_WATCH:].astype(np.complex128)
            watched[index] = np.count_nonzero(peaks)
            stray[index] = np.max(abs(peaks) ** 2, initial=0.) / np.min(abs(field[:selected_count]) ** 2)
            if index < motion_frames - 1:
                synthesis[index] = readback[1, index, :selected_count]
            frame = None
            if transfer_frames:
                copied[index].synchronize()
                frame = np.frombuffer(memoryview(outputs[index]).toreadonly(), np.uint8, count=area).reshape(shape)
                outputs[index] = None
            if retained_frames is not None:
                retained_frames.append(frame)
            frame_ready_ms[index] = (time.perf_counter() - started) * 1000
            if stop_requested is not None and stop_requested():
                raise InterruptedError("SLM rearrangement stopped")
            if frame_ready is not None:
                # This callback consumes an already copied host map while the
                # next map runs. Network backpressure/long holds are idle time.
                keeper = gpu["keeper"]
                with keeper["lock"]:
                    keeper["active"] -= 1
                callback_started = time.perf_counter()
                try:
                    frame_ready(index, frame)
                finally:
                    with keeper["lock"]:
                        keeper["active"] += 1
                callback_ms += (time.perf_counter() - callback_started) * 1000
                if keeper["error"] is not None:
                    raise RuntimeError("SLM GPU warming failed") from keeper["error"]
                emitted_count += 1

        try:
            # The first map needs only its own state: a Fixed sequence's
            # sampling, and every later map's state, follow while the GPU runs it.
            desired, free, cap, gate, gauge, steering, leap, bindings = (
                frame_state(source[None].astype(np.float64), np.zeros(1)) if sampled is None else
                frame_state(sampled["sites_yx"][:1], np.asarray(sampled["movement_fraction"][:1], np.float64)))
            after_prepare = time.perf_counter()
            for index in range(motion_frames):
                if stop_requested is not None and stop_requested():
                    raise InterruptedError("SLM rearrangement stopped")
                name, dark_map = kind(index), index < darkened
                bands, signed_y, frequencies, counts, offsets = bindings[not dark_map]
                local = index if dark_map else index - darkened
                count = int(counts[local])
                band = (count + 7) // 8 * 8
                views = _rearrangement_record_views(rows[index], roster, gpu["capacity"])
                _rearrangement_record(views, bands[local], signed_y[local],
                                      frequencies[int(offsets[local]):int(offsets[local]) + count], band)
                views["goal"][:selected_count] = desired[index, source_indices]
                views["rotor"][:selected_count] = gauge[index] * np.exp(
                    -1j * (source_phase + steering[index] * rotation))
                if dark_map:
                    views["cap"][:len(dark)], views["gate"][:len(dark)] = cap[index], gate[index]
                views["scalars"][:] = (_TRACK_RELEASE_WINDOW if dark_map or not moving else _TRACK_WINDOW,
                                       _TRACK_SCHEDULE[name][2]
                                       * (1. if dark_map else min(1., selected_count / _TRACK_FEW)),
                                       frame_tolerances[index])
                views["meta"][:] = (selected_count, len(dark) if dark_map else 0)
                begun[index], ended[index] = cp.cuda.Event(), cp.cuda.Event()
                begun[index].record(stream)
                runtime.memcpyAsync(gpu["record"].data.ptr, rows[index].ctypes.data, record_size,
                                    runtime.memcpyHostToDevice, stream.ptr)
                if index == motion_frames - 1 and shown or name.endswith("dark_last") and (moving or not index):
                    # The prepared destination itself, its phases shifted by the
                    # common offset in whole codes, which keeps its dither.
                    cp.add(gpu["target_codes"], np.uint8(int(np.rint(-common * 128 / np.pi)) % 256), out=gpu["codes"])
                for segment_index, segment in enumerate(_rearrangement_segments(name, complete and moving)):
                    if segment_index and name in ("last", "leap_last"):
                        # Empty target sites of a partial fill, or the source
                        # phases kept when no retained trap moves, collect lattice
                        # intermodulation; capped full-plane iterations spread it.
                        _rearrangement_plane_wgs(gpu, target[target_indices], destination_amplitude, refinement, 0)
                    gpu["track_graphs"][(band, *segment)].launch(stream)
                    evaluations[index] += segment[0] + segment[1] + segment[2]
                    corrections[index] += segment[1]
                if transfer_frames:
                    # Maps before index - 1 are retired, so this slot's download is done.
                    slot = gpu["frame_slots"][index % 2]
                    runtime.memcpyAsync(slot.data.ptr, gpu["codes"].data.ptr, area, runtime.memcpyDeviceToDevice, stream.ptr)
                    snapped, copied[index] = cp.cuda.Event(disable_timing=True), cp.cuda.Event(disable_timing=True)
                    snapped.record(stream)
                    copy.wait_event(snapped)
                    outputs[index] = np.frombuffer(pool.malloc(area), np.uint8, count=area).reshape(shape)
                    runtime.memcpyAsync(outputs[index].ctypes.data, slot.data.ptr, area, runtime.memcpyDeviceToHost, copy.ptr)
                    copied[index].record(copy)
                for row, values in ((0, gpu["actual"]), (1, gpu["track_accepted"])):
                    runtime.memcpyAsync(readback[row, index].ctypes.data, values.data.ptr, roster * 8,
                                        runtime.memcpyDeviceToHost, stream.ptr)
                ended[index].record(stream)
                ended[index].done  # submit now (WDDM batches launches until a query)
                pending.append(index)
                if sampled is None:
                    sampled = sample_rearrangement(prepared, plan, motion_frames=motion_frames)
                # Retire the previous map while this one runs, and this one too
                # if the GPU already finished it (the host is the slower side).
                while len(pending) > 1 or pending and ended[pending[0]].done:
                    retire()
                if not index and motion_frames > 1:  # one map: its own state is the whole state
                    desired, free, cap, gate, gauge, steering, leap, bindings = frame_state(
                        sampled["sites_yx"], np.asarray(sampled["movement_fraction"], np.float64))
                    if pending and ended[0].done:
                        retire()
            while pending:
                retire()
            # The last map's stray light on the delivered codes, measured once it has been handed over.
            stray[-1] = (prepared["target_stray_intensity_ratio"] if shown
                         else _rearrangement_stray(gpu, target[target_indices]))
        except BaseException:
            stream.synchronize()  # pinned outputs and records must outlive pending copies
            copy.synchronize()
            raise
    positive = desired > 0
    selected_fields = fields[:, source_indices].astype(np.complex128)
    brightness = abs(selected_fields / initial[source_indices][None]) ** 2
    stray_limits = np.full(motion_frames, _TRACK_WATCH_GATE[0])
    stray_limits[-1] = _TRACK_WATCH_GATE[1]
    gate_misses = (ratios > frame_tolerances) | (surplus_ratios > _TRACK_DARK_LIMIT) | (stray > stray_limits)
    if require_converged and np.any(gate_misses):
        missed = np.flatnonzero(gate_misses)
        raise RuntimeError(f"SLM maps {missed.tolist()} did not meet optical constraints; "
                           f"bright {np.round(ratios[missed], 6).tolist()} (limits {frame_tolerances[missed].tolist()}), "
                           f"surplus-atom site/initial trap {np.round(surplus_ratios[missed], 6).tolist()} "
                           f"(limit {_TRACK_DARK_LIMIT:g}), stray peak/weakest trap {np.round(stray[missed], 4).tolist()} "
                           f"(limits {stray_limits[missed].tolist()})")
    intensity = abs(fields.astype(np.complex128)) ** 2
    relative = np.divide(intensity, desired ** 2, out=np.zeros(desired.shape), where=positive)
    support = relative.max(axis=1) / np.min(np.where(positive, relative, np.inf), axis=1)
    axis_fields = selected_fields * gauge
    phase_change = np.angle(axis_fields * start.conj()[None])
    phase_step = np.angle(axis_fields * np.concatenate((start[None], axis_fields[:-1])).conj())
    total_ms = (time.perf_counter() - started) * 1000
    return {
        **plan, **sampled, **metadata, "noop": False,
        "phase_codes": tuple(retained_frames) if retained_frames is not None else None,
        "actual_fields": _frozen(fields), "desired_amplitudes": _frozen(desired), "active_sites": _frozen(positive),
        "synthesis_coefficients": _frozen(synthesis), "destination_phase_offset_rad": float(common),
        "endpoint_field": _frozen(fields[-1, source_indices]),
        "endpoint_support_intensity_ratio": float(ratios[-1]),
        "motion_frames": motion_frames, "iterations": tuple(map(int, evaluations)),
        "correction_updates": tuple(map(int, corrections)), "watched_stray_peaks": tuple(map(int, watched)),
        "emitted_frame_count": emitted_count, "frame_solve_ms": frame_solve_ms, "frame_ready_ms": frame_ready_ms,
        "quality_evaluated": True, "quality_accepted": not gate_misses.any(), "converged": not gate_misses.any(),
        "gate_misses": _frozen(gate_misses),
        "support_intensity_ratios": _frozen(support), "retained_intensity_ratios": _frozen(ratios),
        "surplus_site_intensity_ratios": _frozen(surplus_ratios), "stray_intensity_ratios": _frozen(stray),
        "brightness_minimum_to_initial": _frozen(brightness.min(axis=1)),
        "brightness_maximum_to_initial": _frozen(brightness.max(axis=1)),
        "brightness_mean_to_initial": _frozen(brightness.mean(axis=1)),
        "phase_change_from_initial_rms_rad": _frozen(np.sqrt(np.mean(phase_change ** 2, axis=1))),
        "phase_step_max_rad": _frozen(np.max(abs(phase_step), axis=1)),
        "timing_ms": {"prepare_frame_state": (after_prepare - started) * 1000, "callback": callback_ms,
                      "first_frame_ready": float(frame_ready_ms[0]), "total": total_ms},
    }


def _psf_main_lobe(psf, center):
    """Connected one-percent main lobe of this numeric pupil's single trap."""
    from scipy.ndimage import label  # noqa: PLC0415

    components, _ = label(psf >= .01 * psf[center], structure=np.ones((3, 3), bool))
    return components == components[center]


def _rearrangement_focal_peaks(intensity, maxima, sites, psf_support, exclusion):
    """Recognize one main peak per target, never an entire bright rectangle,
    and the strongest local maximum farther than exclusion samples from every
    target: the stray light of the gates (nearer maxima shape the trap itself).

    The FFT plane is periodic. Equal neighboring samples form one plateau.
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
    reach = int(np.ceil(exclusion))
    yy, xx = np.mgrid[-reach - 1:reach + 2, -reach - 1:reach + 2]
    disc = np.floor(sites).astype(np.int64)[:, None] + np.stack((yy.ravel(), xx.ravel()), axis=1)
    within = np.sum((disc - sites[:, None]) ** 2, axis=-1) <= exclusion ** 2
    maxima[tuple((disc[within] % shape).T)] = False
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

    Called after playback/second photo, not a publication gate. Fields use the
    FFT-center phase reference; trap phase changes stay compute_rearrangement's,
    about the optical axis. Fourfold zero padding resolves peaks between native
    Fourier samples; it does not resize the SLM pupil. No optical response or
    survival is inferred. Only full-plane results are returned: per map the
    stray light of the gates (strongest local maximum farther than
    _TRACK_WATCH_EXCLUSION from every current retained trap over the weakest of
    them) and focal_plane_accepted, its own verdict against _TRACK_WATCH_GATE;
    compute_rearrangement's results and verdicts stay its own.
    """
    import time  # noqa: PLC0415

    started = time.perf_counter()
    gpu = prepared["gpu"]
    if not gpu:
        raise RuntimeError("SLM rearrangement workspace is closed")
    phase_codes = result["phase_codes"]
    if phase_codes is None:
        raise ValueError("SLM phase diagnostics require a retained phase sequence")
    count, source_count = len(phase_codes), len(prepared["source_yx"])
    source_indices = np.asarray(result["source_indices"], np.intp)
    removed = np.asarray(result["removed_source_indices"], np.intp)
    fields = np.asarray(result["actual_fields"])
    discard = np.zeros(count)
    pupil_steps = np.empty(0)
    positive = np.asarray(result["active_sites"])
    selected_intensity = abs(fields[:, source_indices].astype(np.complex128)) ** 2
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
    if count:
        from cupyx.scipy.fft import get_fft_plan  # noqa: PLC0415
        from cupyx.scipy.ndimage import maximum_filter  # noqa: PLC0415

        cp, stream, pool = gpu["cp"], gpu["stream"], gpu["diagnostic_pool"]
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
        movie = steps = plan = power = None
        try:
            with _rearrangement_gpu_active(gpu["keeper"]), stream, cp.cuda.using_allocator(pool.malloc):
                movie = cp.empty((count, *prepared["shape_yx"]), cp.uint8)
                for index, codes in enumerate(phase_codes):
                    movie[index].set(np.asarray(codes), stream=stream)
                steps = cp.empty(count, cp.float32)
                gpu["kernels"]["phase_step_rms"]((count,), (256,),
                    (movie, gpu["initial_phase"], gpu["physical_pupil"], steps,
                     np.int32(area), np.float64(gpu["pupil_energy"])))
                pupil_steps = steps.get(stream=stream)
                del movie, steps
                plan = get_fft_plan(gpu["optical"], shape=tuple(focal_shape), axes=(0, 1), value_type="C2C")
            for index, codes in enumerate(phase_codes):
                if stop_requested is not None and stop_requested():
                    raise InterruptedError("SLM rearrangement diagnostics stopped")
                with _rearrangement_gpu_active(gpu["keeper"]), stream, cp.cuda.using_allocator(pool.malloc), plan:
                    gpu["codes"].set(np.asarray(codes), stream=stream)
                    gpu["kernels"]["display"](((area + 255) // 256,), (256,),
                        (gpu["codes"], gpu["physical_pupil"], gpu["incident"], gpu["optical"], np.int32(area)))
                    # The centered physical pupil is embedded contiguously. Moving
                    # that block only adds a common Fourier phase ramp, not power.
                    power = cp.abs(cp.fft.fftshift(cp.fft.fft2(gpu["optical"], s=tuple(focal_shape)))) ** 2
                    maxima = ((power > 0) & (power == maximum_filter(power, size=3, mode="wrap"))).get(stream=stream)
                    focal_intensity = power.get(stream=stream)
                active_points = np.asarray(result["sites_yx"])[index, positive[index]]
                active_pixels = (active_points - shape // 2) * sampling + focal_center
                recognized, values, peak_yx[index], peak_intensity[index], target_region = _rearrangement_focal_peaks(
                    focal_intensity, maxima, active_pixels, psf_support, _TRACK_WATCH_EXCLUSION * sampling)
                peak_yx[index] = (peak_yx[index] - focal_center) / sampling + shape // 2
                target_peak_yx[index, positive[index]] = (recognized - focal_center) / sampling + shape // 2
                target_peak_intensity[index, positive[index]] = values
                source_site_intensity[index] = focal_intensity[tuple(source_pixels.T)]
                source_site_is_active[index] = np.any(np.all(source_sites[:, None] == active_points[None], axis=-1),
                                                      axis=1)
                if len(removed):
                    halo = (source_pixels[removed, None] + halo_offsets[None]) % focal_shape
                    halo_ratios = (focal_intensity[tuple(halo.reshape(-1, 2).T)].reshape(len(removed), -1)
                                   / source_reference[removed, None])
                    # An old identity landing under a current main lobe is not a
                    # dark-region violation. Additional peaks there remain visible
                    # in the independent whole-plane peak diagnosis above.
                    excluded = target_region[tuple(halo.reshape(-1, 2).T)].reshape(len(removed), -1)
                    discard[index] = float(np.max(np.where(excluded, 0, halo_ratios)))
        finally:
            # Stopped or failed diagnostics release them too: the resident workspace keeps none.
            movie = steps = plan = power = None
            pool.free_all_blocks()
    # The same stray light as the gates (_TRACK_WATCH_GATE of the current
    # weakest retained trap, beyond _TRACK_WATCH_EXCLUSION), on the full plane.
    background = peak_intensity / np.min(selected_intensity, axis=1) if count else np.empty(0)
    limits = np.full(count, _TRACK_WATCH_GATE[0])
    limits[-1:] = _TRACK_WATCH_GATE[1]
    # The removal map is what lets surplus atoms go; later maps are reported.
    discarded_ok = bool(np.all(discard[:result["fade_frames"] if len(removed) else 0] <= SPOT_BACKGROUND_TOLERANCE))
    return {
        "background_intensity_ratios": _frozen(background), "discard_intensity_ratios": _frozen(discard),
        "background_peak_yx": _frozen(peak_yx), "background_peak_intensity": _frozen(peak_intensity),
        "target_peak_yx": _frozen(target_peak_yx), "target_peak_intensity": _frozen(target_peak_intensity),
        "source_site_intensities": _frozen(source_site_intensity),
        "source_site_intensity_ratios": _frozen(source_site_intensity / source_reference[None]),
        "source_site_is_active": _frozen(source_site_is_active),
        "background_scope": f"local-peaks-beyond-{_TRACK_WATCH_EXCLUSION:g}px-of-current-traps-full-focal-plane; "
                            "4x-zero-padded; over-weakest-current-trap",
        "peak_position_basis": "native-Fourier-pixels; quarter-pixel-samples; equal-plateau-centroid",
        "peak_assignment": "one-per-current-target; natural-pupil-PSF-center-component-at-1pct; nearest-target",
        "center_sample_power_proxy": _frozen(selected_intensity.sum(axis=1) / (np.prod(prepared["shape_yx"]) * gpu["pupil_energy"])),
        "pupil_phase_step_rms_rad": _frozen(pupil_steps), "discard_converged": discarded_ok,
        # Judged like the gates, against the whole focal plane instead of the watched points.
        "focal_plane_accepted": bool(discarded_ok and np.all(background <= limits)
                                     and np.all(np.isfinite(target_peak_intensity[positive]))),
        "publication_gate": False, "propagation": "native-fp32-fractional-fourier",
        "diagnostics_ms": (time.perf_counter() - started) * 1000,
    }
