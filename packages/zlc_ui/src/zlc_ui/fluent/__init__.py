"""Lazy exports for the reusable Qt control family."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "ensure_qt_app": ("zlc_ui.qt", "ensure_qt_app"),
    "InfoPane": ("zlc_ui.fluent.info_pane", "InfoPane"),
    "PublishedItemsLegend": ("zlc_ui.fluent.published_items", "PublishedItemsLegend"),
}
for _name in (
    "window_pad", "resolve_fluent_auto_scale", "frameless_content_top_margin",
    "ensure_fluent_scale",
    "set_fluent_scale", "scaled_px", "screen_fit_window_size",
    "center_window_on_primary_screen", "release_window", "retain_window",
    "fluent_font_size", "fluent_text_width", "popup_gap", "fluent_widget_stylesheet",
    "fluent_scrollbar_thickness", "fluent_scrollbar_stylesheet", "status_dot_stylesheet",
    "stroke_card_border", "signals_blocked", "format_compact_number",
    "align_to_resolution", "FluentStatusDot", "FluentLabel", "muted_note_label",
    "FluentFrame", "FluentPopup", "show_fluent_popup_for_anchor",
    "FluentSettingsPopupAnchor", "FluentGroupBox", "FluentButton", "FluentLineEdit",
    "FluentReadoutEdit", "FluentReadoutMultiline", "FluentPathEdit",
    "FluentSectionLabel", "setting_label_width", "FluentSettingRow", "fluent_message",
    "retire_pending_widgets",
    "fluent_confirm", "FluentComboBox", "FluentTreeComboBox", "FluentTabWidget",
    "FluentSwitch", "fluent_switch_width", "fluent_spinbox_stylesheet", "FluentSpinBox", "FluentInputDialog",
    "FluentCodeEdit", "FluentDoubleSpinBox", "FluentCheckBox", "FluentScrollArea",
    "LinkedScrollPanes", "apply_fluent_scrollbars", "FluentWindow", "bind_body_close", "launch_qt_window", "launch_fluent_window",
    "open_fluent_window",
    "Metrics", "measure_text_width", "ElidedLabel", "FluentStatusStrip", "FluentFormGrid",
):
    _EXPORTS[_name] = ("zlc_ui.fluent.fluent", _name)
for _name in (
    "ACCENT", "API_VIOLET", "API_VIOLET_DARK", "BG", "CARD_PAD", "CARD_TITLE_PX",
    "COMBO_TRI_SIZE", "COMBO_WIDTH", "DIVIDER", "EDIT_PADDING_H", "FLUENT_SCALE_MAX",
    "FLUENT_SCALE_MIN", "FONT", "FONT_SIZE", "GREEN", "GRAPHITE", "GREY", "HINT",
    "HOVER", "MUTED_LABEL_STYLE", "ORANGE", "ORANGE_DARK", "ORANGE_TINT", "PADDING_H",
    "PADDING_V", "PLACEHOLDER", "RADIUS", "RED", "STEP_WIDTH", "SURFACE", "TEXT",
    "TITLE_LEFT_INSET", "WINDOW_PAD", "WINDOW_SCREEN_FRACTION", "YELLOW",
):
    _EXPORTS[_name] = ("zlc_ui.fluent.style", _name)
for _name in (
    "choice_state", "grouped_choice_items", "choice_tree_groups", "fill_grouped_choice_combo",
    "read_editable_combo", "coerce_choice_labels",
):
    _EXPORTS[_name] = ("zlc_ui.fluent.choice_picker", _name)

__all__ = tuple(sorted(_EXPORTS))


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


del _name
