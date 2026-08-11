"""Backend-neutral pointer routing for the PlotSession facade."""

from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING, Any
import math

import numpy as np

from ._axis_transform import AxisTransform
from ._gesture_engine import (
    _ColorGesture,
    _ColorLimitDrag,
    _PanGesture,
    _PointerGesture,
    _SelectorGesture,
    area_drag_handle,
    pan_rectangle,
    range_endpoint_hit,
)
from ._session_state import _LiveFitRequest, SelectionChange
from ._selector_scene import ColorLimitCandidate
from .parameters import RenderEffect
from .selectors import (
    CrosshairPoint,
    DragHandle,
    NumericRange,
    RectangleRange,
    SelectorKind,
    SelectorState,
)
from .specs import FacetGridPlot, ImagePlot
from .kinds import AxisRef

if TYPE_CHECKING:
    from .session import PlotSession


class GestureSessionMixin:

    def _pointer_state(self) -> SelectorState | None:
        gesture = self._gesture
        if not isinstance(gesture, _SelectorGesture):
            return None
        pointer = self._selector_controller.candidate_state()
        return (
            pointer
            if pointer is not None and pointer.kind is gesture.kind
            else None
        )

    def _pointer_threshold_active(self) -> bool:
        pointer = self._pointer_state()
        return bool(
            pointer is not None
            and self._projected._is_histogram_plot()
            and pointer.kind is SelectorKind.THRESHOLD
        )

    def _event_canonical(
        self,
        event: Any,
        *,
        captured_transform: AxisTransform | None = None,
    ) -> CrosshairPoint | None:
        assert self._renderer is not None
        if captured_transform is None:
            return None
        point = captured_transform.canonical_point(
            event,
            self._renderer.figure.canvas,
        )
        if point is None:
            return None
        return (
            CrosshairPoint(point.x, point.x)
            if self._pointer_threshold_active()
            else point
        )

    def _is_double_click(self, event: Any, button: int) -> bool:
        """Normalize native, browser-detail, and timed middle/right clicks."""

        explicit = bool(getattr(event, "dblclick", False))
        gui_event = getattr(event, "guiEvent", None)
        detail = getattr(gui_event, "detail", 0)
        try:
            browser_double = int(detail) >= 2
        except (TypeError, ValueError):
            browser_double = False
        if explicit or browser_double:
            self._click_history.pop(button, None)
            return True
        if button not in (2, 3):
            return False
        try:
            x = float(getattr(event, "x"))
            y = float(getattr(event, "y"))
        except (TypeError, ValueError):
            self._click_history.pop(button, None)
            return False
        if not np.all(np.isfinite((x, y))):
            self._click_history.pop(button, None)
            return False
        now = monotonic()
        previous = self._click_history.get(button)
        interval = self._defaults.interaction.double_click_interval_ms / 1000.0
        radius = self._defaults.interaction.double_click_radius_px
        if previous is not None:
            then, previous_x, previous_y = previous
            if now - then <= interval and math.hypot(
                x - previous_x,
                y - previous_y,
            ) <= radius:
                self._click_history.pop(button, None)
                return True
        self._click_history[button] = (now, x, y)
        return False

    def _on_button_press(
        self,
        event: Any,
        *,
        interaction_transform: AxisTransform | None = None,
    ) -> None:
        button = getattr(event, "button", None)
        if button not in (1, 2, 3):
            return
        button = int(button)
        assert self._renderer is not None
        if interaction_transform is None:
            return
        event_axes = self._axis_for_transform(interaction_transform)
        event.inaxes = event_axes
        is_double = self._is_double_click(event, button)
        self._cancel_gesture()
        if isinstance(self._spec, FacetGridPlot):
            focus_index = self._facet_focus_index
            if focus_index is None:
                if button == 1 and is_double:
                    facet_index = self._renderer.facet_index_for_axes(event_axes)
                    if facet_index is not None:
                        self.focus_facet(facet_index)
                return
            if button == 1 and is_double and event_axes is self._renderer.primary_axes:
                self.show_facet_overview()
                return
        if button == 2:
            if event_axes is not self._renderer.primary_axes:
                return
            if is_double:
                self._zoom_to_selection_or_reset()
                return
            origin = interaction_transform.display_point(
                event,
                self._renderer.figure.canvas,
            )
            if origin is None:
                return
            self._gesture = _PanGesture(
                event_axes,
                interaction_transform,
                origin,
                NumericRange(*interaction_transform.x_limits),
                NumericRange(*interaction_transform.y_limits),
            )
            return
        point = self._event_canonical(
            event,
            captured_transform=interaction_transform,
        )
        if point is None:
            return
        if button == 3:
            if event_axes is not self._renderer.primary_axes:
                return
            self._update_pointer_crosshair(
                point,
                clear=is_double,
            )
            return

        if event_axes in self._renderer.axes.get("distribution", ()):
            self._begin_color_limit_pointer(
                point,
                event_axes,
                interaction_transform,
            )
            return
        hit = self._hit_selector(
            point,
            event_axes,
            event=event,
            transform=interaction_transform,
        )
        if hit is None:
            state = self._start_pointer_selection(point, event_axes)
            if state is None:
                return
            handle = DragHandle.NEW
        else:
            state, handle = hit
        selector_axes = self._selector_axes(state)
        if selector_axes is None:
            return
        try:
            self._selector_controller.state(state.kind)
        except KeyError:
            draft = state
        else:
            draft = None
        self._selector_controller.pointer_down(
            state.kind,
            point.x,
            point.y,
            handle=handle,
            draft=draft,
        )
        self._gesture = _SelectorGesture(
            selector_axes,
            interaction_transform,
            state.kind,
        )
        with self._renderer.raster_transaction():
            if hit is None:
                self._render_current(RenderEffect.OVERLAY)
            self._renderer.begin_selector_gesture(state.kind)


    def _start_pointer_selection(
        self,
        point: CrosshairPoint,
        event_axes: Any,
    ) -> SelectorState | None:
        assert self._renderer is not None
        if event_axes is not self._renderer.primary_axes:
            return None
        value = RectangleRange(
            NumericRange(point.x, point.x),
            NumericRange(point.y, point.y),
        )
        return SelectorState(
            SelectorKind.AREA,
            value,
            facet_index=self._focused_facet_index,
        )

    def _update_pointer_crosshair(
        self,
        point: CrosshairPoint,
        *,
        clear: bool,
    ) -> None:
        state = self._projected._selector_state_or_none(SelectorKind.CROSSHAIR)
        if clear:
            if state is not None:
                self.remove_selector(SelectorKind.CROSSHAIR)
            return
        self._install_selector_state(SelectorState(
            SelectorKind.CROSSHAIR,
            point,
            facet_index=self._focused_facet_index,
        ))

    def _begin_color_limit_pointer(
        self,
        point: CrosshairPoint,
        event_axes: Any,
        transform: AxisTransform,
    ) -> bool:
        assert self._renderer is not None
        # The SEMANTIC spec decides: a focused FacetGrid image cell shows
        # the standalone image's distribution, and its guides must drag the
        # same way.  Gating on the outer spec left that surface inert.
        if not isinstance(self._semantic_spec, ImagePlot):
            return False
        distribution = self._renderer.axes.get("distribution", ())
        if event_axes not in distribution:
            return False
        bounds = self._color_selector_domain()
        value = self._current_color_limit_value()
        tolerance = max(
            bounds.span * self._defaults.interaction.selector_hit_radius_fraction,
            1.0e-12,
        )
        hit = range_endpoint_hit(value, point.x, tolerance)
        if hit is None:
            return True
        _score, handle = hit
        minimum_span = min(
            bounds.span,
            max(value.span * 1.0e-12, bounds.span * 1.0e-12, 1.0e-12),
        )
        self._gesture = _ColorGesture(
            event_axes,
            transform,
            _ColorLimitDrag(
                original=value,
                candidate=value,
                handle=handle,
                origin=point.x,
                bounds=bounds,
                minimum_span=minimum_span,
            ),
        )
        candidate = self._display_color_limit_candidate()
        assert candidate is not None
        with self._renderer.raster_transaction():
            self._renderer.begin_color_limit_gesture(candidate)
        return True

    def _area_drag_handle(
        self,
        state: SelectorState,
        axes: Any,
        event: Any | None,
        transform: AxisTransform | None = None,
    ) -> tuple[float, DragHandle] | None:
        pixel_x = None if event is None else getattr(event, "x", None)
        pixel_y = None if event is None else getattr(event, "y", None)
        if pixel_x is None or pixel_y is None:
            return None
        displayed = (
            self._projected._display_selector_state(state)
            if self._view is not None
            else state
        )
        if not isinstance(displayed.value, RectangleRange):
            return None
        mouse = np.asarray((float(pixel_x), float(pixel_y)), dtype=float)

        pixel_ratio = self.surface_plan.device_pixel_ratio
        center_radius = (
            self._defaults.interaction.selector_center_radius_px * pixel_ratio
        )
        handle_radius = (
            self._defaults.interaction.selector_handle_radius_px * pixel_ratio
        )
        canvas = axes.figure.canvas

        def pixel_point(coordinate: tuple[float, float]) -> np.ndarray:
            shown = (
                axes.transData.transform(coordinate)
                if transform is None
                else transform.display_to_pixel(
                    coordinate[0],
                    coordinate[1],
                    canvas,
                )
            )
            return np.asarray(shown, dtype=float)

        return area_drag_handle(
            displayed.value,
            mouse,
            pixel_point,
            center_radius=center_radius,
            handle_radius=handle_radius,
        )

    def _on_scroll(
        self,
        event: Any,
        *,
        interaction_transform: AxisTransform | None = None,
    ) -> None:
        direction = getattr(event, "button", None)
        if direction not in ("up", "down"):
            return
        if interaction_transform is None:
            return
        self._cancel_gesture()
        assert self._renderer is not None
        if (
            isinstance(self._spec, FacetGridPlot)
            and self._facet_focus_index is None
        ):
            return
        axes = self._axis_for_transform(interaction_transform)
        if axes is not self._renderer.primary_axes:
            return
        # Zoom against the axes' CURRENT limits, not the painted snapshot the
        # frontend sampled: a queued tick must compound onto the previous one
        # instead of re-deriving the same stale target and short-circuiting.
        try:
            inverted = axes.transData.inverted().transform(
                (float(event.x), float(event.y))
            )
            center_x, center_y = float(inverted[0]), float(inverted[1])
        except Exception:
            return
        if not (math.isfinite(center_x) and math.isfinite(center_y)):
            return
        zoom_factor = self._defaults.interaction.wheel_zoom_factor
        # ``step`` carries accumulated wheel ticks (positive = up); one tick
        # reproduces the historical single-notch factor exactly, a coalesced
        # burst compounds it.
        ticks = float(getattr(event, "step", 0.0))
        if ticks == 0.0:
            ticks = 1.0 if direction == "up" else -1.0
        factor = zoom_factor ** ticks
        x_axes = NumericRange(*sorted(map(float, axes.get_xlim())))
        y_axes = NumericRange(*sorted(map(float, axes.get_ylim())))
        zoomed_x = NumericRange(
            center_x + (x_axes.low - center_x) * factor,
            center_x + (x_axes.high - center_x) * factor,
        )
        zoomed_y = y_axes
        if isinstance(self._projected._semantic_spec(), ImagePlot):
            zoomed_y = NumericRange(
                center_y + (y_axes.low - center_y) * factor,
                center_y + (y_axes.high - center_y) * factor,
            )
        self.set_viewport(self._viewport_x_from_axes(zoomed_x), zoomed_y)

    def _zoom_to_selection_or_reset(self) -> None:
        """Zoom the viewport to the current range selection, or go home.

        A committed AREA selector wins over an X_RANGE one; X_RANGE keeps the
        current y limits.  Degenerate spans and selector-free plots restore
        the complete home view.  The selector itself is kept.  The viewport
        is display-space, so the display projection applies here (pulse maps
        it back to source units through ``_viewport_x_to_axes``).
        """

        assert self._renderer is not None
        for kind in (SelectorKind.AREA, SelectorKind.X_RANGE):
            state = self._projected._selector_state_or_none(kind)
            if state is None or state.facet_index != self._focused_facet_index:
                continue
            displayed = (
                self._projected._display_selector_state(state)
                if self._view is not None
                else self._special_display_selector_state(state)
            )
            value = displayed.value
            if kind is SelectorKind.AREA:
                assert isinstance(value, RectangleRange)
                if value.x.span <= 0.0 or value.y.span <= 0.0:
                    break
                self.set_viewport(value.x, value.y)
                return
            assert isinstance(value, NumericRange)
            if value.span <= 0.0:
                break
            current_y = NumericRange(
                *sorted(map(float, self._renderer.primary_axes.get_ylim()))
            )
            self.set_viewport(value, current_y)
            return
        self.reset_viewport()

    def _hit_selector(
        self,
        point: CrosshairPoint,
        event_axes: Any,
        *,
        event: Any | None = None,
        transform: AxisTransform | None = None,
    ) -> tuple[SelectorState, DragHandle] | None:
        x_bounds = self._selector_x_bounds(transform)
        y_bounds = self._selector_y_bounds(transform)
        hit_fraction = self._defaults.interaction.selector_hit_radius_fraction
        tx = max(x_bounds.span * hit_fraction, 1e-12)
        ty = max(y_bounds.span * hit_fraction, 1e-12)
        candidates: list[
            tuple[int, float, int, SelectorState, DragHandle]
        ] = []
        for order, state in enumerate(self._resolved_selector_snapshot().states):
            if self._selector_axes(state) is not event_axes:
                continue
            if state.facet_index != self._focused_facet_index:
                continue
            value = state.value
            score = float("inf")
            handle = DragHandle.BODY
            if state.kind is SelectorKind.X_RANGE and isinstance(value, NumericRange):
                endpoint = range_endpoint_hit(value, point.x, tx)
                if endpoint is not None:
                    score, handle = endpoint
                elif value.low <= point.x <= value.high:
                    score, handle = 0.9, DragHandle.BODY
            elif state.kind is SelectorKind.AREA and isinstance(value, RectangleRange):
                area_hit = self._area_drag_handle(
                    state,
                    event_axes,
                    event,
                    transform,
                )
                if area_hit is not None:
                    score, handle = area_hit
            elif state.kind is SelectorKind.THRESHOLD:
                score = (
                    abs(point.x - float(value)) / tx
                    if self._projected._is_histogram_plot()
                    else abs(point.y - float(value)) / ty
                )
            if score <= 1.0:
                kind_priority = (
                    0 if state.kind is SelectorKind.THRESHOLD else 1
                )
                candidates.append(
                    (
                        kind_priority,
                        score,
                        order,
                        state,
                        handle,
                    )
                )
        if not candidates:
            return None
        _priority, _score, _order, state, handle = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        return state, handle

    def _coordinate_bounds(self, ref: AxisRef) -> NumericRange:
        # The 1-D canonical domain carries the same extremes as the broadcast
        # coordinate tensor without materializing megapixels per call.
        view = self._projected._view
        if view is not None:
            domain = np.asarray(view._resolve(ref).domain_canonical, dtype=float)
            finite = domain[np.isfinite(domain)]
            if finite.size:
                return NumericRange(float(np.min(finite)), float(np.max(finite)))
        values = np.asarray(self._projected._coordinate(ref).canonical, dtype=float)
        finite = values[np.isfinite(values)]
        return NumericRange(float(np.min(finite)), float(np.max(finite)))

    def _value_bounds(self) -> NumericRange:
        values = np.asarray(self._projected._value_quantity().canonical, dtype=float)
        finite = values[np.isfinite(values)]
        return NumericRange(float(np.min(finite)), float(np.max(finite)))

    def _selector_x_bounds(
        self,
        transform: AxisTransform | None = None,
    ) -> NumericRange:
        if transform is not None:
            return NumericRange(*transform.canonical_x_limits)
        if self._view is None:
            assert self._renderer is not None
            return NumericRange(
                *sorted(map(float, self._renderer.primary_axes.get_xlim()))
            )
        source = self._projected._x_selector_source()
        return (
            self._coordinate_bounds(source)
            if isinstance(source, AxisRef)
            else self._value_bounds()
        )

    def _color_selector_domain(self) -> NumericRange:
        assert self._renderer is not None
        axes = self._renderer.axes.get("distribution", ())
        if not axes:
            raise RuntimeError("color selector requires a side distribution")
        low, high = sorted(map(float, axes[0].get_ylim()))
        if self._view is not None:
            quantity = self._projected._value_quantity()
            low = self._projected._display_scalar_to_canonical(low, quantity)
            high = self._projected._display_scalar_to_canonical(high, quantity)
        return NumericRange(*sorted((low, high)))

    def _current_color_limit_value(self) -> NumericRange:
        assert self._renderer is not None
        low, high = sorted(self._renderer.resolved_color_limits())
        if self._view is not None:
            quantity = self._projected._value_quantity()
            low = self._projected._display_scalar_to_canonical(low, quantity)
            high = self._projected._display_scalar_to_canonical(high, quantity)
        return NumericRange(*sorted((low, high)))

    def _display_color_limit_candidate(self) -> ColorLimitCandidate | None:
        gesture = self._gesture
        if not isinstance(gesture, _ColorGesture):
            return None
        value = gesture.drag.candidate
        if self._view is not None:
            value = self._projected._canonical_range_to_display(
                value,
                self._projected._value_quantity(),
            )
        return ColorLimitCandidate(value)

    def _selector_y_bounds(
        self,
        transform: AxisTransform | None = None,
    ) -> NumericRange:
        if transform is not None:
            if self._projected._is_histogram_plot():
                pointer = self._pointer_state()
                if pointer is not None and pointer.kind is SelectorKind.THRESHOLD:
                    return NumericRange(*transform.canonical_x_limits)
            return NumericRange(*transform.canonical_y_limits)
        pointer = self._pointer_state()
        if (
            pointer is not None
            and self._projected._is_histogram_plot()
            and pointer.kind is SelectorKind.THRESHOLD
        ):
            return self._selector_x_bounds()
        if self._view is None:
            assert self._renderer is not None
            return NumericRange(
                *sorted(map(float, self._renderer.primary_axes.get_ylim()))
            )
        assert self._renderer is not None
        visible = NumericRange(
            *sorted(map(float, self._renderer.primary_axes.get_ylim()))
        )
        if self._projected._is_histogram_plot():
            return visible
        source = self._projected._y_ref_or_value()
        return self._projected._display_range_to_canonical(visible, source)

    def _on_motion(self, event: Any) -> None:
        gesture = self._gesture
        if gesture is None:
            return
        if isinstance(gesture, _PanGesture):
            if gesture.lane_due(
                "pan",
                self._defaults.interaction.pointer_update_interval_ms,
            ):
                self._update_pan(event, gesture)
            return
        point = self._event_canonical(
            event,
            captured_transform=gesture.transform,
        )
        if point is None:
            self._cancel_gesture()
            return
        if isinstance(gesture, _ColorGesture):
            gesture.drag = gesture.drag.moved(point.x)
            candidate = self._display_color_limit_candidate()
            assert candidate is not None
            assert self._renderer is not None
            with self._renderer.raster_transaction():
                self._renderer.preview_color_limit_candidate(candidate)
                # The recolor rides the ONE continuous-gesture cadence pan
                # uses.  It once had its own slower lane because a preview
                # step cost a background recapture and a colorbar redraw;
                # now it is an RGBA recompose, and throttling it to 10 Hz
                # was itself the drag lag the throttle guarded against.
                if gesture.lane_due(
                    "raster",
                    self._defaults.interaction.pointer_update_interval_ms,
                ):
                    self._renderer.preview_color_limits(
                        candidate.value.low,
                        candidate.value.high,
                    )
                    self._render_current(RenderEffect.OVERLAY, schedule_fit=False)
            return
        if self._selector_controller.active_gesture is None:
            self._cancel_gesture()
            return
        current = self._selector_controller.candidate_state()
        if current is None:
            self._cancel_gesture()
            return
        updated = self._selector_controller.pointer_move(
            point.x,
            point.y,
            x_bounds=self._selector_x_bounds(gesture.transform),
            y_bounds=self._selector_y_bounds(gesture.transform),
        )
        if updated is not None and updated != current:
            assert self._renderer is not None
            with self._renderer.raster_transaction():
                self._renderer.preview_selector(self._painted_selector_state(updated))
            self._emit_selection(SelectionChange.UPDATED, updated)

    def _on_button_release(self, event: Any) -> None:
        gesture = self._gesture
        if gesture is None:
            return
        if getattr(event, "button", None) == 2:
            if not isinstance(gesture, _PanGesture):
                return
            try:
                self._update_pan(event, gesture, render=False)
            finally:
                self._clear_gesture(gesture)
            if gesture.candidate is not None:
                self._set_viewport_state(gesture.candidate)
            return
        if isinstance(gesture, _PanGesture):
            return
        if isinstance(gesture, _ColorGesture):
            self._finish_color_gesture(event, gesture)
            return
        self._finish_selector_gesture(event, gesture)

    def _finish_color_gesture(
        self,
        event: Any,
        gesture: _ColorGesture,
    ) -> None:
        point = self._event_canonical(
            event,
            captured_transform=gesture.transform,
        )
        if point is None:
            self._cancel_gesture()
            return
        gesture.drag = gesture.drag.moved(point.x)
        candidate = self._display_color_limit_candidate()
        assert candidate is not None
        self._clear_gesture(gesture)
        try:
            if gesture.drag.changed:
                self.set_color_limits(
                    candidate.value.low,
                    candidate.value.high,
                    fixed=True,
                )
            else:
                self._render_current(
                    RenderEffect.BASE_GEOMETRY,
                    schedule_fit=False,
                )
        except Exception:
            self._render_current(
                RenderEffect.BASE_GEOMETRY,
                schedule_fit=False,
            )
            raise

    def _finish_selector_gesture(
        self,
        event: Any,
        gesture: _SelectorGesture,
    ) -> None:
        if self._selector_controller.active_gesture is None:
            self._cancel_gesture()
            return
        try:
            point = self._event_canonical(
                event,
                captured_transform=gesture.transform,
            )
            if point is None:
                self._cancel_gesture()
                return
            candidate = self._selector_controller.finish_gesture(
                point.x,
                point.y,
                x_bounds=self._selector_x_bounds(gesture.transform),
                y_bounds=self._selector_y_bounds(gesture.transform),
            )
        except Exception:
            self._cancel_gesture()
            raise
        self._clear_gesture(gesture)
        if candidate is None:
            return
        if not self._is_degenerate_selector(candidate):
            stored = self._install_selector_state(
                candidate,
                emit_change=False,
                finished_gesture=gesture,
            )
            self._emit_selection(SelectionChange.COMMITTED, stored)
            return
        committed = self._projected._selector_state_or_none(candidate.kind)
        if committed is not None:
            self.remove_selector(committed.kind)
            return
        self._render_current(RenderEffect.OVERLAY, schedule_fit=False)

    def _on_key_press(self, event: Any) -> None:
        if str(getattr(event, "key", "")).lower() != "escape":
            return
        self._cancel_gesture()
        if (
            isinstance(self._spec, FacetGridPlot)
            and self._renderer is not None
            and self._facet_focus_index is not None
        ):
            self.show_facet_overview()

    @staticmethod
    def _is_degenerate_selector(state: SelectorState) -> bool:
        if state.kind is SelectorKind.X_RANGE:
            assert isinstance(state.value, NumericRange)
            return state.value.span <= 0.0
        if state.kind is SelectorKind.AREA:
            assert isinstance(state.value, RectangleRange)
            return state.value.x.span <= 0.0 or state.value.y.span <= 0.0
        return False

    @classmethod
    def _require_stable_selector(cls, state: SelectorState) -> None:
        if not cls._is_degenerate_selector(state):
            return
        if state.kind is SelectorKind.AREA:
            raise ValueError("area selector requires non-degenerate x and y ranges")
        raise ValueError(f"{state.kind.value} selector requires a non-degenerate range")

    @staticmethod
    def _fit_request_uses_selector(
        request: _LiveFitRequest,
        kind: SelectorKind,
    ) -> bool:
        if request.selector_kind is not None:
            return request.selector_kind is kind
        return kind in {SelectorKind.AREA, SelectorKind.X_RANGE}

    def _update_pan(
        self,
        event: Any,
        gesture: _PanGesture,
        *,
        render: bool = True,
    ) -> bool:
        assert self._renderer is not None
        point = gesture.transform.display_point(event, self._renderer.figure.canvas)
        if point is None:
            return False
        image_like = isinstance(self._projected._semantic_spec(), ImagePlot)
        moved = pan_rectangle(
            gesture.origin,
            point,
            gesture.x,
            gesture.y,
            image_like=image_like,
        )
        if moved is None:
            return False
        selected = RectangleRange(self._viewport_x_from_axes(moved.x), moved.y)
        current = gesture.candidate or self._projected.viewport
        if current is not None and np.allclose(
            (current.x.low, current.x.high, current.y.low, current.y.high),
            (selected.x.low, selected.x.high, selected.y.low, selected.y.high),
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            return False
        gesture.candidate = selected
        if render:
            self._render_current(
                RenderEffect.AXIS_TRANSFORM | RenderEffect.OVERLAY,
                schedule_fit=False,
            )
        return True

    def _clear_gesture(self, gesture: _PointerGesture) -> None:
        if self._gesture is not gesture:
            return
        self._gesture = None
        if isinstance(gesture, _PanGesture) or self._renderer is None:
            return
        self._renderer.end_selector_gesture()

    def _cancel_gesture(self) -> SelectorState | None:
        gesture = self._gesture
        if gesture is None:
            return None
        cancelled = None
        try:
            if isinstance(gesture, _SelectorGesture):
                cancelled = self._selector_controller.lost_pointer_capture()
        finally:
            self._clear_gesture(gesture)
        if (
            isinstance(gesture, _PanGesture)
            and gesture.candidate is not None
            and self._renderer is not None
            and not self._closed
        ):
            self._render_current(
                RenderEffect.AXIS_TRANSFORM | RenderEffect.OVERLAY,
                schedule_fit=False,
            )
        restore_committed = isinstance(gesture, _ColorGesture) or cancelled is not None
        if restore_committed and self._renderer is not None and not self._closed:
            color_gesture = isinstance(gesture, _ColorGesture)
            self._render_current(
                RenderEffect.BASE_GEOMETRY if color_gesture else RenderEffect.OVERLAY,
                schedule_fit=False,
            )
        return cancelled
