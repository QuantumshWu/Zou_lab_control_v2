"""Discoverable frame-survival processor descriptor."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DatasetInputSpec,
    LogicNodeDescriptor,
    NodeKind,
)

from .processor import SURVIVAL_OUTPUTS, FrameSurvivalProcessor, _checked_pairs


def _validate(values):
    if not values["publish_all"]:
        _checked_pairs(tuple((row["start"], row["end"]) for row in values["pairs"]))

FRAME_SURVIVAL_SCHEMA = AuthoringSchema((
    AuthoringField("publish_all", "bool", "Publish all", True),
    AuthoringField("pairs", "rows", "Survival pairs", (), columns=(
        AuthoringField("start", "int", "Start frame", 0, required=True, minimum=0),
        AuthoringField("end", "int", "End frame", 1, required=True, minimum=0),
    )),
), validator=_validate)


def _build(*, source_signal: str, **values: object) -> FrameSurvivalProcessor:
    authored = FRAME_SURVIVAL_SCHEMA.project_values(values)
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    return FrameSurvivalProcessor(
        source_signal=selected_source,
        pairs=None if authored["publish_all"] else tuple(
            (row["start"], row["end"]) for row in authored["pairs"]),
    )


def _editor_factory(parent=None):
    from dataclasses import replace
    from PyQt5 import QtCore, QtWidgets
    from zlc_data import READOUT_EVENT
    from zlc_ui.fluent import ACCENT, FluentButton, FluentLabel
    from zlc_ui.form import FluentParameterForm, FormChoice, FormSpec
    from .processor import _forward_pairs

    class SurvivalForm(QtWidgets.QWidget):
        draft_changed = QtCore.pyqtSignal(object)
        managed_fields = ("publish_all", "pairs")

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            header = QtWidgets.QHBoxLayout()
            header.addWidget(FluentLabel("Start frame → End frame"))
            header.addStretch(1)
            self.all_button = FluentButton("Publish all", color=ACCENT)
            header.addWidget(self.all_button)
            layout.addLayout(header)
            self.form = FluentParameterForm(FormSpec(()))
            layout.addWidget(self.form)
            self.form.changed.connect(self._changed)
            self.all_button.clicked.connect(lambda: self.draft_changed.emit(
                {"values": {"publish_all": True, "pairs": ()}}))

        def _changed(self, key):
            self.draft_changed.emit({"values": {
                "publish_all": False, "pairs": self.form.read_value(key),
            }})

        def update_projection(self, projection):
            values = projection["form_values"]
            schema = projection.get("source_schema")
            axes = () if schema is None else tuple(
                axis for axis in schema.point_domain.axes if axis.role == READOUT_EVENT)
            axis = axes[0] if len(axes) == 1 else None
            all_pairs = _forward_pairs(axis.size) if axis is not None else ()
            rows = (tuple({"start": start, "end": end} for start, end in all_pairs)
                    if values["publish_all"] else tuple(values["pairs"]))
            choices = (() if axis is None else tuple(
                FormChoice(str(axis.coordinate_at(index)), index) for index in range(axis.size)))
            missing = sorted({value for row in rows for value in row.values()
                              if value is not None and (axis is None or value >= axis.size)})
            choices += tuple(FormChoice(f"Unavailable: {value}", value) for value in missing)
            used = {(row["start"], row["end"]) for row in rows}
            next_pair = next((pair for pair in all_pairs if pair not in used), (None, None))
            field = next(field for field in projection["form_spec"].fields if field.key == "pairs")
            columns = tuple(replace(column, kind="choice", minimum=None, maximum=None,
                                    choices=choices, default=default,
                                    unavailable_reason="Waiting for source frames" if not choices else "")
                            for column, default in zip(field.columns, next_pair))
            self.form.reconcile(FormSpec((replace(field, columns=columns),)), {"pairs": rows})
            self.form.widget_for("pairs").add_button.setEnabled(next_pair[0] is not None)

        def set_mutation_enabled(self, enabled):
            self.setEnabled(enabled)

    return SurvivalForm(parent)


LOGIC_NODE = LogicNodeDescriptor(
    "frame_survival",
    NodeKind.PROCESSOR,
    FRAME_SURVIVAL_SCHEMA,
    input_specs=(DatasetInputSpec("occupied", None, "exact"),),
    outputs=SURVIVAL_OUTPUTS,
    ui_contributions=(_editor_factory,),
    build=_build,
)

__all__ = ["FRAME_SURVIVAL_SCHEMA", "LOGIC_NODE"]
