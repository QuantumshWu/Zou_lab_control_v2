"""Qt projection of the headless simple-form contract.

Only this closed form registry knows how a field kind maps to a widget.  The
form owns no Apply workflow, revision, repository, run, domain object, or
hardware access; it only reads and writes an exact keyed draft.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import re
import sys
from types import MappingProxyType

from PyQt5 import QtCore, QtWidgets

from zlc_data.units import (
    DEFAULT_UNITS,
    UnitError,
    format_quantity,
)

from .form import (
    FormFieldProps,
    FormSpec,
    parse_number_text,
)

from ..fluent import (
    FluentComboBox,
    FluentCycleComboBox,
    FluentDoubleSpinBox,
    FluentLineEdit,
    FluentPathEdit,
    FluentSettingRow,
    FluentSwitch,
    FluentTreeComboBox,
    fluent_integer_box,
    fluent_unit_picker,
    fluent_switch_width,
    scaled_px,
    setting_label_width,
    signals_blocked,
)
from ..fluent.choice_picker import (
    coerce_choice_labels,
    fill_grouped_choice_combo,
    read_editable_combo,
)


_INT_TEXT = re.compile(r"[+-]?\d+")


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class FormRuntimeContext:
    """Qt-only live providers used by dynamic form fields."""

    choice_names: Callable[[str], object] | None = None
    choice_sources: Callable[[], object] | None = None
    choice_metadata: Callable[[], object] | None = None
    choice_labels: Callable[[], object] | None = None
    choice_state_labels: tuple[str, str, str] | None = None

    def names_for(self, key: str) -> tuple[str, ...]:
        if not callable(self.choice_names):
            return ()
        return tuple(str(value) for value in self.choice_names(str(key)))

    @staticmethod
    def _mapping(provider) -> Mapping[str, object]:
        if not callable(provider):
            return _empty_mapping()
        value = provider()
        if not isinstance(value, Mapping):
            raise TypeError("dynamic form provider must return a mapping")
        return value

    def sources(self) -> Mapping[str, object]:
        return self._mapping(self.choice_sources)

    def metadata(self) -> Mapping[str, object]:
        return self._mapping(self.choice_metadata)

    def labels(self) -> Mapping[str, object]:
        return coerce_choice_labels(self.choice_labels)

def _value_error(field: FormFieldProps, message: str) -> ValueError:
    return ValueError(f"field {field.key!r}: {message}")


def _connect_change(signal, on_change: Callable[[], None]) -> None:
    signal.connect(lambda *_args: on_change())


def being_edited(widget: QtWidgets.QWidget) -> bool:
    """Is the operator typing in this widget right now?

    THE question every projected editor asks before writing, in this form
    and outside it: a page that writes a projected count into its own spin
    box asks it here rather than keeping a private answer.

    Live editing means every keystroke round-trips through the owner and
    comes back as a new projection, and reconcile writes projections into
    the widgets.  Written into the box the operator is inside, that is not
    an update, it is a fight: type "0." into a number and the round trip
    normalises it to 0, writes back "0", and the decimal point the operator
    just pressed is gone -- a value they never typed, installed under their
    cursor.  Clearing an optional field is worse: the value becomes None,
    the Auto switch takes it, and the box is DISABLED mid-word.

    So the rule is one sentence for controls that can hold a partial draft:
    while the operator is inside one, its value, Auto switch and enabled state
    are theirs.  A choice activation is already a complete typed edit, so
    reconcile deliberately does not use this focus guard for choice fields.
    Composite editors are checked by ancestry, because focus sits on the inner
    spin box, not the host.
    """

    focused = QtWidgets.QApplication.focusWidget()
    if focused is None:
        return False
    return focused is widget or widget.isAncestorOf(focused)


class FormWidgetHandler(ABC):
    """The complete typed lifecycle for one field kind."""

    @abstractmethod
    def normalize(self, field: FormFieldProps, value: object) -> object:
        """Validate/coerce without touching a widget, for atomic population."""

    @abstractmethod
    def build(
        self,
        field: FormFieldProps,
        value: object,
        on_change: Callable[[], None],
        context: FormRuntimeContext | None = None,
    ) -> QtWidgets.QWidget:
        """Construct, seed, and wire one widget."""

    @abstractmethod
    def read(self, field: FormFieldProps, widget: QtWidgets.QWidget) -> object:
        """Read one typed value without evaluating free text."""

    @abstractmethod
    def write(
        self,
        field: FormFieldProps,
        widget: QtWidgets.QWidget,
        value: object,
    ) -> None:
        """Write one already validated value."""

    @abstractmethod
    def is_empty(self, field: FormFieldProps, widget: QtWidgets.QWidget) -> bool:
        """Report whether a required value is absent."""

    @abstractmethod
    def refresh(
        self,
        field: FormFieldProps,
        widget: QtWidgets.QWidget,
        context: FormRuntimeContext | None = None,
    ) -> None:
        """Refresh presentation options while preserving a legal selection."""

class _StaticHandler(FormWidgetHandler):
    def refresh(self, field, widget, context=None) -> None:
        del field, widget, context


def _blank_placeholder(field: FormFieldProps) -> str:
    """What a blank-or-number edit says while it is blank."""

    if not field.blank_allowed:
        return ""
    return "(required)" if field.required else "(optional)"


def _install_validator(field: FormFieldProps, widget: FluentLineEdit) -> None:
    """The owner's bounds on a blank-or-number edit.

    Called when the edit is built AND whenever the owner re-declares the
    field it keeps: the validator and the leave-time clamp both hold the
    bounds, and a widget kept across a widened range with the old ones
    judged a legal 50 as unfinished and clamped it back to the old 10 on
    Return.
    """

    if field.kind == "int":
        widget.set_numeric_validator("int", bottom=field.minimum, top=field.maximum)
    elif field.kind == "float" and field.unit and field.unit != "1":
        # A blank-or-number edit still holds a QUANTITY, and the RF bounds
        # -- the fields that carry dBm -- are all of them: they default to
        # None so the instrument's own limit stands.  It takes digits like
        # every other numeric field and gets the same picker beside it,
        # which is where its scale is said.
        widget.set_quantity_validator(
            field.unit, bottom=field.minimum, top=field.maximum
        )
    else:
        widget.set_numeric_validator(
            "float", bottom=field.minimum, top=field.maximum
        )


def _blank_or_number_edit(field: FormFieldProps) -> FluentLineEdit:
    widget = FluentLineEdit()
    widget.setMinimumWidth(scaled_px(120, minimum=96))
    widget.setPlaceholderText(_blank_placeholder(field))
    _install_validator(field, widget)
    return widget


class _TextHandler(_StaticHandler):
    def normalize(self, field: FormFieldProps, value: object) -> str:
        if value is None:
            # Whether a field may be empty is declared by the field, exactly
            # once, the way ``blank_allowed`` declares it for the scalars.
            # Reading it off ``default`` -- the CURRENT value -- made the same
            # field accept None while its title was automatic and reject it
            # the moment an operator had typed one, so switching a label back
            # to Auto raised out of a Qt slot and aborted the process.
            if field.required:
                raise _value_error(field, "value must be str")
            return ""
        if not isinstance(value, str):
            raise _value_error(field, "value must be str")
        return value

    def build(self, field, value, on_change, context=None):
        del context
        edit = FluentLineEdit()
        edit.setMinimumWidth(scaled_px(160, minimum=120))
        edit.setPlaceholderText(field.description[:48])
        edit.setToolTip(field.description)
        self.write(field, edit, value)
        # As it is TYPED.  Every other kind in this registry is live -- a
        # number on textChanged, a switch on toggled, a choice on activated --
        # and text alone waited for Return or a click elsewhere, so the same
        # popup answered two different rules depending on which row you were
        # in.  What makes this safe is the focus rule in reconcile(): the
        # value comes straight back as a new projection, and a form that
        # wrote it into the box you are typing in would eat the half-typed
        # character it normalised away.
        _connect_change(edit.textChanged, on_change)
        return edit

    def read(self, field, widget):
        del field
        return widget.text()

    def write(self, field, widget, value):
        widget.setText(self.normalize(field, value))

    def is_empty(self, field, widget):
        del field
        return not widget.text().strip()


class _IntHandler(_StaticHandler):
    """A Python int has no width, and neither does the box that holds one.

    A required integer is a whole-number box bounded only where its owner
    bounded it -- Qt's own integer spin stops at signed 31 bits and cannot
    say "no bound", so a form that used it invented one, and a legal 2**40
    overflowed the control that was supposed to hold it.  An optional
    integer is a blank-or-number edit, because a spin box is a number
    editor and cannot hold the vacancy.
    """

    @staticmethod
    def _configure_spin(field: FormFieldProps, widget: FluentDoubleSpinBox) -> None:
        widget.setRange(
            -sys.float_info.max if field.minimum is None else field.minimum,
            sys.float_info.max if field.maximum is None else field.maximum,
        )
        widget.setValueUnit(field.unit)

    def normalize(self, field: FormFieldProps, value: object) -> int | None:
        if value is None:
            if field.blank_allowed:
                return None
            raise _value_error(field, "value cannot be None")
        if not isinstance(value, int) or isinstance(value, bool):
            raise _value_error(field, "value must be int")
        if field.minimum is not None and value < field.minimum:
            raise _value_error(field, f"value is below {field.minimum}")
        if field.maximum is not None and value > field.maximum:
            raise _value_error(field, f"value is above {field.maximum}")
        return value

    def build(self, field, value, on_change, context=None):
        del context
        if not field.blank_allowed:
            widget = fluent_integer_box()
            self._configure_spin(field, widget)
            self.write(field, widget, value)
            _connect_change(widget.valueChanged, on_change)
        else:
            widget = _blank_or_number_edit(field)
            self.write(field, widget, value)
            _connect_change(widget.textChanged, on_change)
        widget.setToolTip(field.description)
        return widget

    def read(self, field, widget):
        if isinstance(widget, FluentDoubleSpinBox):
            if not widget.hasAcceptableInput():
                raise _value_error(field, "value is not a base-10 integer")
            # The decimal, not the double: the box holds the integer exactly
            # and its Qt shadow is an approximation past 2**53.
            return self.normalize(field, int(widget.decimalValue()))
        text = widget.text().strip()
        if not text:
            return self.normalize(field, None)
        if _INT_TEXT.fullmatch(text) is None:
            raise _value_error(field, "value is not a base-10 integer")
        return self.normalize(field, int(text, 10))

    def write(self, field, widget, value):
        prepared = self.normalize(field, value)
        if isinstance(widget, FluentDoubleSpinBox):
            if prepared is None:
                raise _value_error(field, "numeric spin cannot represent None")
            widget.setValue(prepared)
        else:
            widget.setText("" if prepared is None else str(prepared))

    def is_empty(self, field, widget):
        if isinstance(widget, FluentDoubleSpinBox):
            return False
        return not widget.text().strip()


class _NumberHandler(_StaticHandler):
    """Lossless ``int | float`` editor for owner contracts that accept both.

    Unlike a float spin box, entering ``1`` remains the integer ``1`` and an
    existing ``1.0`` remains a float on an untouched round trip.  Numeric API
    values need this distinction because their canonical lineage retains the
    authored numeric token even though hardware binding later validates it.
    """

    def normalize(
        self,
        field: FormFieldProps,
        value: object,
    ) -> int | float | None:
        if value is None:
            if field.blank_allowed:
                return None
            raise _value_error(field, "value cannot be None")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _value_error(field, "value must be an int or float")
        if isinstance(value, float) and not math.isfinite(value):
            raise _value_error(field, "value must be finite")
        if field.minimum is not None and value < field.minimum:
            raise _value_error(field, f"value is below {field.minimum}")
        if field.maximum is not None and value > field.maximum:
            raise _value_error(field, f"value is above {field.maximum}")
        return value

    def build(self, field, value, on_change, context=None):
        del context
        widget = _blank_or_number_edit(field)
        widget.setToolTip(field.description)
        self.write(field, widget, value)
        _connect_change(widget.textChanged, on_change)
        return widget

    def read(self, field, widget):
        if not widget.text().strip():
            return self.normalize(field, None)
        try:
            value = parse_number_text(widget.text(), field.key)
        except (TypeError, ValueError) as exc:
            raise _value_error(field, str(exc)) from exc
        return self.normalize(field, value)

    def write(self, field, widget, value):
        prepared = self.normalize(field, value)
        widget.setText(
            ""
            if prepared is None
            else str(prepared)
            if isinstance(prepared, int)
            else repr(prepared)
        )

    def is_empty(self, field, widget):
        del field
        return not widget.text().strip()


class _FloatHandler(_StaticHandler):
    @staticmethod
    def _configure_spin(field: FormFieldProps, widget: FluentDoubleSpinBox) -> None:
        # The box invents no bound: a side the owner left None is Qt's whole
        # double line, which the box reads as "none".
        widget.setRange(
            -sys.float_info.max if field.minimum is None else float(field.minimum),
            sys.float_info.max if field.maximum is None else float(field.maximum),
        )
        # The field said what its number is IN.  Nothing read it before, so
        # every box in this project showed a bare repr and refused a prefix.
        widget.setValueUnit(field.unit)

    def normalize(self, field: FormFieldProps, value: object) -> float | None:
        if value is None:
            if field.blank_allowed:
                return None
            raise _value_error(field, "value cannot be None")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _value_error(field, "value must be a real number")
        result = float(value)
        if not math.isfinite(result):
            raise _value_error(field, "value must be finite")
        if field.minimum is not None and result < field.minimum:
            raise _value_error(field, f"value is below {field.minimum}")
        if field.maximum is not None and result > field.maximum:
            raise _value_error(field, f"value is above {field.maximum}")
        return result

    def build(self, field, value, on_change, context=None):
        del context
        if not field.blank_allowed:
            # QDoubleSpinBox owns one real number only.  Optional numbers
            # remain a typed blank-or-number edit instead of smuggling None
            # through an out-of-domain float and a textual sentinel.
            widget = FluentDoubleSpinBox()
            self._configure_spin(field, widget)
            self.write(field, widget, value)
            _connect_change(widget.valueChanged, on_change)
        else:
            widget = _blank_or_number_edit(field)
            self.write(field, widget, value)
            _connect_change(widget.textChanged, on_change)
        widget.setToolTip(field.description)
        return widget

    def read(self, field, widget):
        if isinstance(widget, FluentDoubleSpinBox):
            if not widget.hasAcceptableInput():
                raise _value_error(field, "value is not a finite decimal number")
            return self.normalize(field, float(widget.value()))
        text = widget.text().strip()
        if not text:
            return self.normalize(field, None)
        try:
            number = _in_value_unit(widget, float(text), field.unit)
        except UnitError as error:
            raise _value_error(field, str(error)) from error
        except ValueError as error:
            raise _value_error(field, "value is not a finite decimal number") from error
        return self.normalize(field, number)

    def write(self, field, widget, value):
        prepared = self.normalize(field, value)
        if isinstance(widget, FluentDoubleSpinBox):
            if prepared is None:
                raise _value_error(field, "numeric spin cannot represent None")
            widget.setValue(prepared)
        else:
            widget.setText(
                ""
                if prepared is None
                else format_quantity(_in_shown_unit(widget, prepared, field.unit), "1")
            )

    def is_empty(self, field, widget):
        if isinstance(widget, FluentDoubleSpinBox):
            return False
        return not widget.text().strip()


class _BoolHandler(_StaticHandler):
    def normalize(self, field: FormFieldProps, value: object) -> bool:
        if not isinstance(value, bool):
            raise _value_error(field, "value must be bool")
        return value

    def build(self, field, value, on_change, context=None):
        del context
        widget = FluentSwitch("")
        widget.setToolTip(field.description)
        self.write(field, widget, value)
        _connect_change(widget.toggled, on_change)
        return widget

    def read(self, field, widget):
        return self.normalize(field, bool(widget.isChecked()))

    def write(self, field, widget, value):
        widget.setChecked(self.normalize(field, value))

    def is_empty(self, field, widget):
        del field, widget
        return False


class _ChoiceHandler(FormWidgetHandler):
    """The popup rows, and behind one of them a lazy sub-domain.

    The sub-domain is DATA -- every coordinate of an axis -- read one
    position at a time, so nothing here walks it when the widget already
    answers the question: what the widget shows at its position IS a value
    of the domain, and asking the domain to confirm it read the whole axis,
    on every beat, for a value that had not moved.
    """

    def normalize(self, field: FormFieldProps, value: object) -> object:
        if value is None:
            return None
        choice = field.choice_for(value)
        if choice is not None:
            return choice.value
        cycle = field.cycle_choice_for(value)
        if cycle is not None:
            return cycle[1]
        raise _value_error(field, "value is not one of the typed choices")

    @staticmethod
    def _shows(widget: FluentComboBox, value: object) -> bool:
        """Whether the widget's current row already holds ``value``."""

        if widget.currentIndex() < 0:
            return value is None
        return _same_typed_value(widget.currentData(), value)

    @staticmethod
    def _fill_cycle(field: FormFieldProps, widget: FluentComboBox) -> None:
        """Install the lazy sub-domain without touching the item list.

        A cycle is a DOMAIN -- every coordinate an axis offers to a Scope
        pin -- so it grows with the data: one more shot, one more entry.
        Rebuilding the popup for that closed an open dropdown and ate the
        click that was already on its way, once per shot, on every panel
        whose axis was still filling.
        """

        if field.cycle_choices is None:
            return
        if not isinstance(widget, FluentCycleComboBox):
            raise TypeError("cycle choices require FluentCycleComboBox")
        widget.setCycleChoices(field.cycle_label, field.cycle_choices)

    @staticmethod
    def _fill(field: FormFieldProps, widget: FluentComboBox) -> None:
        widget.clear()
        for choice in field.choices:
            widget.addItem(choice.label, choice.value)
        _ChoiceHandler._fill_cycle(field, widget)

    def build(self, field, value, on_change, context=None):
        del context
        widget = (
            FluentCycleComboBox()
            if field.cycle_choices is not None
            else FluentComboBox()
        )
        self._fill(field, widget)
        self.write(field, widget, value)
        widget.setToolTip(field.unavailable_reason or field.description)
        _connect_change(widget.activated, on_change)
        return widget

    def read(self, field, widget):
        if widget.currentIndex() < 0:
            return None
        if isinstance(widget, FluentCycleComboBox) and widget.isCycleSelected():
            # One read, at the position the widget holds: the domain is what
            # the field installed, so the value there needs no confirming.
            return widget.currentData()
        return self.normalize(field, widget.currentData())

    def write(self, field, widget, value):
        if value is None:
            widget.setCurrentIndex(-1)
            return
        choice = field.choice_for(value)
        if choice is not None:
            widget.setCurrentIndex(
                next(index for index, item in enumerate(field.choices) if item is choice)
            )
            return
        if not isinstance(widget, FluentCycleComboBox):
            raise TypeError("cycle value requires FluentCycleComboBox")
        if widget.isCycleSelected() and self._shows(widget, value):
            return
        cycle = field.cycle_choice_for(value)
        if cycle is None:
            raise _value_error(field, "value is not one of the typed choices")
        widget.setCyclePosition(cycle[0])

    def is_empty(self, field, widget):
        del field
        return widget.currentIndex() < 0

    def refresh(self, field, widget, context=None):
        del context
        current = self.read(field, widget)
        desired = tuple((choice.label, choice.value) for choice in field.choices)
        if field.cycle_choices is not None:
            desired = (*desired, (field.cycle_label, None))
        existing = tuple(
            (widget.itemText(index), widget.itemData(index))
            for index in range(widget.count())
        )
        if existing != desired:
            self._fill(field, widget)
            self.write(field, widget, current)
        widget.setToolTip(field.unavailable_reason or field.description)


class _PathHandler(_StaticHandler):
    def normalize(self, field, value):
        if value is None:
            return ""
        if not isinstance(value, str):
            raise _value_error(field, "value must be a path string")
        return value

    def build(self, field, value, on_change, context=None):
        del context
        picker = FluentPathEdit(
            self.normalize(field, value),
            mode=field.path_mode,
            caption=f"Choose {field.label}",
            file_filter=field.file_filter,
            base_dir=field.base_dir,
            refreshable=field.refreshable,
        )
        picker.setToolTip(field.description)
        _connect_change(picker.changed, on_change)
        return picker

    def read(self, field, widget):
        return self.normalize(field, widget.text())

    def write(self, field, widget, value):
        widget.setText(self.normalize(field, value))

    def is_empty(self, field, widget):
        del field
        return not widget.text().strip()


class _KeyedChoiceHandler(FormWidgetHandler):
    def normalize(self, field, value):
        if value is None:
            return ""
        if not isinstance(value, str):
            raise _value_error(field, "value must be a keyed choice string")
        return value

    @staticmethod
    def _fill(field, widget, current, context):
        runtime = context or FormRuntimeContext()
        fill_grouped_choice_combo(
            widget,
            names=runtime.names_for(field.key),
            sources=runtime.sources(),
            metadata=runtime.metadata(),
            labels=runtime.labels(),
            state_labels=runtime.choice_state_labels,
            current=current,
            none_label="Off" if not field.required else None,
            empty_source_label="Unresolved",
        )

    def build(self, field, value, on_change, context=None):
        combo = FluentTreeComboBox()
        self._fill(field, combo, self.normalize(field, value), context)
        combo.setToolTip(field.description)
        _connect_change(combo.activated, on_change)
        return combo

    def read(self, field, widget):
        return self.normalize(field, read_editable_combo(widget))

    def write(self, field, widget, value):
        del field
        widget.select_choice_key("" if value is None else str(value))

    def is_empty(self, field, widget):
        del field
        return not read_editable_combo(widget)

    def refresh(self, field, widget, context=None):
        current = read_editable_combo(widget)
        self._fill(field, widget, current, context)


FORM_WIDGET_HANDLERS: Mapping[str, FormWidgetHandler] = MappingProxyType(
    {
        "text": _TextHandler(),
        "int": _IntHandler(),
        "float": _FloatHandler(),
        "number": _NumberHandler(),
        "choice": _ChoiceHandler(),
        "bool": _BoolHandler(),
        "path": _PathHandler(),
        "keyed_choice": _KeyedChoiceHandler(),
    }
)


def _widget_family(field: FormFieldProps) -> str:
    """Concrete control family required by one declaration.

    A numeric family names its UNIT: the picker mounted beside the control
    is built for that unit's ladder, so a field re-declared in another unit
    is another row, not a re-configured one.
    """

    prefix = "auto:" if field.automatic else ""
    if field.kind in {"int", "float"} and not field.blank_allowed:
        return f"{prefix}{field.kind}-spin:{field.unit}"
    if field.kind in {"int", "float", "number"}:
        return f"{prefix}line-edit:{field.unit}"
    if field.kind == "text":
        return prefix + "line-edit"
    if field.kind == "choice":
        return prefix + (
            "choice-cycle" if field.cycle_choices is not None else "choice"
        )
    if field.kind == "bool":
        return "bool"
    if field.kind == "path":
        return f"path:{field.path_mode}:{field.file_filter}:{field.base_dir}:{field.refreshable}"
    if field.kind == "keyed_choice":
        return "keyed-choice"
    raise ValueError(f"unsupported form field kind: {field.kind!r}")


def _row_label_in(field: FormFieldProps, unit: str) -> str:
    """The row's label, naming ``unit`` beside it -- or nothing, for "".

    A row whose unit has other spellings carries a picker, and the picker
    says the unit; the label saying it too was the same fact in two places,
    one of which went stale the moment the other was changed.  So such a row
    is labelled with no unit at all, and a row with no picker keeps the
    owner's unit in its label, because there is nothing else to say it.
    """

    label = field.label.strip()
    symbol = str(unit).strip()
    if symbol and symbol != "1":
        label = f"{label} ({symbol})"
    if field.required:
        label = f"{label} *"
    return label


def _shown_unit_of(widget: object, unit: str | None) -> str:
    """Which spelling this widget is being read in right now."""

    asked = getattr(widget, "shownUnit", None)
    shown = str(asked()).strip() if callable(asked) else ""
    return shown or (str(unit or "").strip() or "1")


def _in_value_unit(widget: object, number: float, unit: str | None) -> float:
    """A number typed on screen, said in the unit its owner declared."""

    shown = _shown_unit_of(widget, unit)
    owner = str(unit or "").strip() or "1"
    return float(number) if shown == owner else float(
        DEFAULT_UNITS.convert(number, shown, owner)
    )


def _in_shown_unit(widget: object, number: float, unit: str | None) -> float:
    """A number the owner holds, said in the unit on screen."""

    shown = _shown_unit_of(widget, unit)
    owner = str(unit or "").strip() or "1"
    return float(number) if shown == owner else float(
        DEFAULT_UNITS.convert(number, owner, shown)
    )


def _automatic_label(field: FormFieldProps, checked: bool) -> str:
    return f"{'Auto' if checked else 'Manual'} {field.row_label}"


def _form_label_width(fields) -> int:
    """The label column: wide enough for every text label AND every switch.

    An automatic field's label IS a :class:`FluentSwitch`, so its column need
    is the switch's own painted width -- asked from the one switch-width
    authority, never re-derived here with a padding constant.  Two independent
    formulas drifted apart and the switch's track painted underneath the
    editor beside it.
    """

    fields = tuple(fields)
    width = setting_label_width(field.row_label for field in fields)
    for field in fields:
        if field.automatic:
            width = max(
                width,
                fluent_switch_width(_automatic_label(field, False)),
                fluent_switch_width(_automatic_label(field, True)),
            )
    return width


def _same_typed_value(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


#: What a widget answers when it holds nothing readable.
_NO_VALUE = object()


def _widget_value(
    handler: FormWidgetHandler,
    field: FormFieldProps,
    widget: QtWidgets.QWidget,
) -> object:
    try:
        return handler.read(field, widget)
    except (TypeError, ValueError):
        return _NO_VALUE


def _widget_has_value(
    handler: FormWidgetHandler,
    field: FormFieldProps,
    widget: QtWidgets.QWidget,
    value: object,
) -> bool:
    return _same_typed_value(_widget_value(handler, field, widget), value)


def _seed(field: FormFieldProps, prepared: object) -> object:
    """What a new control is built holding: the prepared value, or, for an
    automatic field on Auto, the declared default -- Auto holds no value,
    and a number box has to be built holding some number."""

    if prepared is None and field.automatic:
        return field.default
    return prepared


def _reconfigure_widget(
    old_field: FormFieldProps,
    field: FormFieldProps,
    widget: QtWidgets.QWidget,
) -> None:
    """Apply changed presentation constraints to one compatible control.

    Whether the control is ENABLED is the form's decision, made in one
    place from the field, its Auto switch and its controller; nothing here
    touches it.
    """

    widget.setToolTip(field.unavailable_reason or field.description)
    if isinstance(widget, FluentLineEdit):
        if field.kind == "text":
            widget.setPlaceholderText(field.description[:48])
        else:
            widget.setPlaceholderText(_blank_placeholder(field))
            _install_validator(field, widget)
    elif isinstance(widget, FluentDoubleSpinBox):
        handler = _IntHandler if field.kind == "int" else _FloatHandler
        handler._configure_spin(field, widget)
    elif isinstance(widget, FluentComboBox):
        if old_field.choices != field.choices:
            _ChoiceHandler._fill(field, widget)
        elif (
            old_field.cycle_choices != field.cycle_choices
            or old_field.cycle_label != field.cycle_label
        ):
            # The choices the operator picks from are the same; only the
            # coordinates behind the Scope action moved.
            _ChoiceHandler._fill_cycle(field, widget)


#: ``(before, after)``: the widgets a host places before a row's control,
#: and the ``(widget, stretch)`` pairs it places after it.
RowCells = tuple[
    tuple[QtWidgets.QWidget, ...], tuple[tuple[QtWidgets.QWidget, int], ...]
]


class FluentParameterForm(QtWidgets.QWidget):
    """Thin exact-key form built from one ordered :class:`FormSpec`."""

    changed = QtCore.pyqtSignal(str)
    #: One field asked for its file to be re-read.  The form knows only which
    #: key; what re-reading means belongs to whoever owns the file.
    refresh_requested = QtCore.pyqtSignal(str)
    #: (key, symbol): the operator chose to READ this row in another
    #: spelling of its unit.  The value did not move; whoever shows a second
    #: number for the same field -- a device's current reading -- follows.
    shown_unit_changed = QtCore.pyqtSignal(str, str)
    #: The operator is DONE with this row: Return, or the focus left it.
    #: ``changed`` says every keystroke; a host that acts on a finished
    #: name -- renaming what everything else calls a binding -- waits for
    #: this one.
    committed = QtCore.pyqtSignal(str)

    def _connect_editor_signals(self, key: str, widget: QtWidgets.QWidget) -> None:
        signal = getattr(widget, "refresh_requested", None)
        if signal is not None:
            signal.connect(lambda key=key: self.refresh_requested.emit(key))
        finished = getattr(widget, "editingFinished", None)
        if finished is not None:
            finished.connect(lambda key=key: self.committed.emit(key))

    @staticmethod
    def _dependency_map(spec: FormSpec) -> dict[str, list[str]]:
        declared = set(spec.keys)
        dependents: dict[str, list[str]] = {}
        for field in spec.fields:
            if field.enabled_when is None:
                continue
            controller = field.enabled_when[0]
            if controller not in declared:
                raise KeyError(
                    f"field {field.key!r} is enabled by {controller!r}, "
                    "which this form does not declare"
                )
            dependents.setdefault(controller, []).append(field.key)
        return dependents

    def __init__(
        self,
        spec: FormSpec,
        values: Mapping[str, object] | None = None,
        parent=None,
        *,
        runtime: FormRuntimeContext | None = None,
        label_width: int | None = None,
        row_cells: Callable[[FormFieldProps], RowCells] | None = None,
    ) -> None:
        if not isinstance(spec, FormSpec):
            raise TypeError("spec must be FormSpec")
        if label_width is not None and (
            isinstance(label_width, bool)
            or not isinstance(label_width, int)
            or label_width <= 0
        ):
            raise ValueError("label_width must be a positive integer or None")
        super().__init__(parent)
        self._spec = spec
        self._runtime = runtime or FormRuntimeContext()
        self._label_width = label_width
        #: What a host puts beside each row's control, ``(before, after)``.
        #: The row is assembled ONCE, with everything in it, before it is
        #: shown.  A host that needs more than label and control -- a device
        #: control's current reading, live switch, Apply and status -- says
        #: so here, and never takes a built row apart: a row dismantled
        #: after the form had shown it painted with half its cells.
        self._row_cells = row_cells or (lambda field: ((), ()))
        self._fields = {field.key: field for field in spec.fields}
        self._widgets: dict[str, QtWidgets.QWidget] = {}
        self._handlers: dict[str, FormWidgetHandler] = {}
        self._rows: dict[str, QtWidgets.QWidget] = {}
        #: The unit picker beside each numeric row, by field key.
        self._unit_pickers: dict[str, object] = {}
        #: What occupies each row's control column, by field key: the editor
        #: itself, or the editor and its picker side by side.  ONE owner of
        #: that fact -- a host that rebuilds a row from ``widget_for`` alone
        #: leaves the picker orphaned in the row, painted wherever it was.
        self._cells: dict[str, QtWidgets.QWidget] = {}
        self._auto_switches: dict[str, FluentSwitch] = {}
        self._dependents = self._dependency_map(spec)

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(scaled_px(6, minimum=4))
        self._layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        label_width = self._label_width or _form_label_width(spec.fields)
        # Built ONCE, with the values it opens on.  A form that was built on
        # each field's default and then populated wrote every control twice
        # -- and refused a valid incoming value whenever the default it never
        # meant to show was not one: a required field with no default could
        # not be constructed with the value it was given.
        prepared = self._prepare_population(
            spec.default_values() if values is None else values, spec
        )
        for field in spec.fields:
            handler = FORM_WIDGET_HANDLERS[field.kind]
            widget = handler.build(
                field,
                _seed(field, prepared[field.key]),
                lambda key=field.key: self.changed.emit(key),
                self._runtime,
            )
            self._connect_editor_signals(field.key, widget)
            self._install_row(
                field,
                widget,
                *self._make_row(
                    field,
                    widget,
                    label_width,
                    automatic_checked=prepared[field.key] is None,
                ),
            )
            self._layout.addWidget(self._rows[field.key])
        # THE SLACK GOES BELOW THE ROWS, not between them.  Without this a
        # form shorter than its host shares the surplus height across its
        # rows, so the PITCH depends on the row COUNT: adding one control
        # moved every control already on screen upward, the lowest by the
        # most.  A row's position must depend on the rows above it and
        # nothing else.
        self._layout.addStretch(1)

        # A field that depends on another follows it while the form is being
        # edited, so the operator SEES that a folder belongs to "saved frames"
        # before choosing it -- rather than the row appearing out of nowhere
        # once they have.  Indexed BY the field that decides, so editing
        # anything else costs nothing: a form of thirty rows would otherwise
        # re-read every dependency on every keystroke.
        # Through the form's own change signal rather than each widget's: a
        # switch, combo and text edit all reach one dependency path.  Connect
        # once even when the initial schema has no dependencies; reconcile may
        # introduce them later.
        self.changed.connect(self._controller_changed)
        self._project_enabled_all()

    def _controller_changed(self, key: str) -> None:
        for dependent in self._dependents.get(str(key), ()):
            self._project_enabled(dependent)

    # ------------------------------------------------------- enabled state
    #
    # ONE formula.  Whether a control may be edited is decided by the field
    # (unavailable), its Auto switch (Auto holds no value to edit) and the
    # field that governs it (enabled_when), together -- and every path that
    # can change any of those projects the result through here.  Written as
    # three partial formulas in construction, populate and reconcile, the
    # same values enabled a dependent through one entry and disabled it
    # through another, and a governed field's Auto state was overwritten by
    # its controller.

    def _controller_value(self, key: str) -> object:
        """What the governing field holds; None while it holds nothing
        readable -- Auto, or text the operator is still typing.  Either is
        "not one of the enabling values", and neither may raise: this runs
        inside the form's own change slot."""

        automatic = self._auto_switches.get(key)
        if automatic is not None and automatic.isChecked():
            return None
        try:
            return self._handlers[key].read(self._fields[key], self._widgets[key])
        except (TypeError, ValueError):
            return None

    def _enabled(self, key: str, *, editing: bool = False) -> bool:
        if editing:
            # The operator is inside it; a projection never takes a control
            # away under their cursor.
            return True
        field = self._fields[key]
        if field.unavailable:
            return False
        automatic = self._auto_switches.get(key)
        if automatic is not None and automatic.isChecked():
            return False
        if field.enabled_when is None:
            return True
        controller, enabling = field.enabled_when
        current = self._controller_value(controller)
        return any(
            type(current) is type(value) and current == value for value in enabling
        )

    def _project_enabled(self, key: str, *, editing: bool = False) -> None:
        """Enable the EDITOR, never the cell around it: a unit picker beside
        a disabled number still chooses how the number is read."""

        self._widgets[key].setEnabled(self._enabled(key, editing=editing))
        automatic = self._auto_switches.get(key)
        if automatic is not None:
            automatic.setEnabled(not self._fields[key].unavailable)

    def _project_enabled_all(self, *, editing_key: str | None = None) -> None:
        for key in self._spec.keys:
            self._project_enabled(key, editing=key == editing_key)

    def _unit_picker(self, field, widget):
        """The shared picker, for a row whose unit has more than one spelling.

        Only for a widget that can be READ in another one: a count and a bare
        number have no ladder, and a picker beside them would offer a choice
        that changes nothing.
        """

        if not hasattr(widget, "setShownUnit"):
            return None
        picker = fluent_unit_picker(field.unit or "", self)
        if picker is not None:
            picker.unit_picked.connect(
                lambda symbol, key=field.key: self._shown_unit_picked(key, symbol)
            )
        return picker

    def _shown_unit_picked(self, key: str, symbol: str) -> None:
        """Show this row in another unit: the value is untouched."""

        widget = self._widgets.get(key)
        if widget is None:
            return
        try:
            widget.setShownUnit(symbol)
        except UnitError:
            return
        self.shown_unit_changed.emit(key, symbol)

    def shown_unit_for(self, key: str) -> str:
        """The spelling this row is read in right now; "" when it has none."""

        asked = getattr(self.widget_for(key), "shownUnit", None)
        return str(asked()).strip() if callable(asked) else ""

    def _label_for(self, field: FormFieldProps, *, picked: bool | None = None) -> str:
        """The row's label: with the owner's unit, unless a picker says it."""

        if picked is None:
            picked = field.key in self._unit_pickers
        return _row_label_in(field, "") if picked else field.row_label

    def cell_for(self, key: str) -> QtWidgets.QWidget:
        """What sits in this row's control column: editor, or editor + picker.

        ``widget_for`` is the editor -- what is read, written and enabled.
        A host that lays a row out for itself must place THIS, or the picker
        the form built beside the editor is left behind in the row.
        """

        return self._cells.get(key, self.widget_for(key))

    def _make_row(self, field, widget, label_width, *, automatic_checked: bool):
        """Build one row around ``widget`` and hand back its parts.

        Nothing is INSTALLED here: the row, its cell, its picker and its
        Auto switch are returned together and recorded by the caller once
        the row they replace is gone.  Recording the new cell and picker
        while the old row was still to be retired let the retirement erase
        them, and the form then answered ``unit_picker_for`` with None for a
        picker that was on screen.
        """

        automatic = None
        picker = self._unit_picker(field, widget)
        cell = widget
        if picker is not None:
            holder = QtWidgets.QWidget(self)
            holder.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Preferred,
            )
            beside = QtWidgets.QHBoxLayout(holder)
            beside.setContentsMargins(0, 0, 0, 0)
            beside.setSpacing(scaled_px(6, minimum=4))
            beside.addWidget(widget, 1)
            beside.addWidget(picker, 0)
            cell = holder
        label = self._label_for(field, picked=picker is not None)
        if field.automatic:
            automatic = FluentSwitch("", self)
            with signals_blocked(automatic):
                automatic.setChecked(automatic_checked)
            automatic.setText(_automatic_label(field, automatic_checked))
            automatic.toggled.connect(
                lambda checked, key=field.key: self._automatic_toggled(
                    key, checked
                )
            )
            label = automatic
        before, after = self._row_cells(field)
        row = FluentSettingRow(
            label,
            cell,
            label_width=label_width,
            before=before,
            after=after,
            parent=self,
        )
        return row, automatic, cell, picker

    def _install_row(self, field, widget, row, automatic, cell, picker) -> None:
        """Record every part of one built row, in the one place that does."""

        key = field.key
        self._widgets[key] = widget
        self._handlers[key] = FORM_WIDGET_HANDLERS[field.kind]
        self._rows[key] = row
        self._cells[key] = cell
        if picker is not None:
            self._unit_pickers[key] = picker
        if automatic is not None:
            self._auto_switches[key] = automatic

    def _retire_row(self, key: str) -> None:
        """Take one row off the form and forget every part of it."""

        row = self._rows.pop(key)
        self._layout.removeWidget(row)
        # Reconcile may run while an Edit page is visible.  An unparented
        # QWidget becomes a transient native window; hide the retired row
        # and retain this form as QObject owner until DeferredDelete.
        row.hide()
        row.deleteLater()
        self._widgets.pop(key, None)
        self._handlers.pop(key, None)
        self._auto_switches.pop(key, None)
        self._cells.pop(key, None)
        self._unit_pickers.pop(key, None)

    def _automatic_toggled(self, key: str, automatic: bool) -> None:
        field = self._fields[key]
        widget = self._widgets[key]
        self._set_automatic_label(key)
        self._project_enabled(key)
        if not automatic and self._handlers[key].is_empty(field, widget):
            if field.kind == "choice" and field.choices:
                self._handlers[key].write(field, widget, field.choices[0].value)
            elif field.kind in {"int", "float", "number"}:
                value = (
                    field.minimum
                    if field.minimum is not None
                    else field.maximum
                    if field.maximum is not None and field.maximum < 0
                    else 0
                )
                self._handlers[key].write(field, widget, value)
        self.changed.emit(key)

    def _set_automatic_label(self, key: str) -> None:
        field = self._fields[key]
        switch = self._auto_switches[key]
        row = self._rows[key]
        label = _automatic_label(field, switch.isChecked())
        width = self._label_width or _form_label_width(self._spec.fields)
        if switch.text() != label or switch.width() != width:
            row.set_label(label, width=width)

    @property
    def spec(self) -> FormSpec:
        return self._spec

    @property
    def keys(self) -> tuple[str, ...]:
        return self._spec.keys

    def minimum_content_width(self) -> int:
        """Width required by the widest current label/editor row."""

        return max(
            (row.layout().minimumSize().width() for row in self._rows.values()),
            default=0,
        )

    def widget_for(self, key: str) -> QtWidgets.QWidget:
        try:
            return self._widgets[key]
        except KeyError as exc:
            raise KeyError(f"unknown form field key: {key!r}") from exc

    def unit_picker_for(self, key: str):
        """The unit picker beside this row, or None when it has no ladder."""

        return self._unit_pickers.get(key)

    def auto_switch_for(self, key: str) -> FluentSwitch:
        try:
            return self._auto_switches[key]
        except KeyError as exc:
            raise KeyError(f"form field {key!r} does not declare Auto") from exc

    def is_empty(self, key: str) -> bool:
        field = self._field_for(key)
        automatic = self._auto_switches.get(key)
        if automatic is not None and automatic.isChecked():
            return False
        return self._handlers[key].is_empty(field, self._widgets[key])

    def read_value(self, key: str) -> object:
        """Read one edited leaf without promoting it to a whole-form snapshot."""
        field = self._field_for(key)
        handler, widget = self._handlers[key], self._widgets[key]
        automatic = self._auto_switches.get(key)
        if automatic is not None and automatic.isChecked():
            return None
        if (
            field.required
            and handler.is_empty(field, widget)
            and not field.required_choice_unavailable
        ):
            raise _value_error(field, "required value is empty")
        try:
            return handler.read(field, widget)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("field "):
                raise
            raise _value_error(field, str(exc)) from exc

    def read_all(self) -> dict[str, object]:
        return {field.key: self.read_value(field.key) for field in self._spec.fields}

    def read_draft(self) -> dict[str, object]:
        """Every box as it stands, vacancies included.

        A draft read answers "what has the operator written", not "hand me a
        complete record": a required box still empty reads as a vacancy
        (None) instead of refusing, so filling the OTHER boxes still
        commits.  What IS written is validated exactly as strictly as ever
        -- a typed value that does not parse or breaks its bounds refuses
        here, not later.  Completeness stays the build step's law.
        """

        values: dict[str, object] = {}
        for field in self._spec.fields:
            handler, widget = self._handlers[field.key], self._widgets[field.key]
            automatic = self._auto_switches.get(field.key)
            if automatic is not None and automatic.isChecked():
                values[field.key] = None
                continue
            if handler.is_empty(field, widget):
                values[field.key] = None
                continue
            values[field.key] = self.read_value(field.key)
        return values

    def populate(self, values: Mapping[str, object]) -> None:
        """Atomically populate every exact key without emitting edit signals.

        All coercion and bound/choice checks finish before the first widget is
        changed, so an invalid full state cannot leave a partially updated form.
        """

        prepared = self._prepare_population(values)
        widgets = tuple(self._widgets[key] for key in self._spec.keys)
        automatic = tuple(self._auto_switches.values())
        with signals_blocked(*widgets, *automatic):
            for field in self._spec.fields:
                switch = self._auto_switches.get(field.key)
                selected = switch is not None and prepared[field.key] is None
                if not selected:
                    self._handlers[field.key].write(
                        field, self._widgets[field.key], prepared[field.key]
                    )
                if switch is not None:
                    switch.setChecked(selected)
                    self._set_automatic_label(field.key)
        self._project_enabled_all()

    def validate_population(self, values: Mapping[str, object]) -> None:
        """Validate one exact owner projection without mutating any widget."""

        self._prepare_population(values)

    def adopt_projection(
        self,
        spec: FormSpec,
        values: Mapping[str, object],
    ) -> bool:
        """Adopt metadata when widgets already show the exact projection."""

        # A cycle's coordinates are DATA -- they lengthen as the axis
        # fills -- so a form is not a different form for having more of
        # them.  Compared as structure, every shot refused adoption and
        # sent the whole form through reconcile.
        metadata = tuple(
            name
            for name in FormFieldProps.__dataclass_fields__
            if name not in ("default", "cycle_choices")
        )
        if len(spec.fields) != len(self._spec.fields) or any(
            any(
                getattr(current, name) != getattr(incoming, name)
                for name in metadata
            )
            for current, incoming in zip(
                self._spec.fields,
                spec.fields,
                strict=True,
            )
        ):
            return False
        # The coordinates move in BEFORE the values are judged: a value and
        # the vocabulary it is judged against come from the same moment.
        # Judged against the vocabulary already installed, a coordinate the
        # axis had just grown was "not one of the typed choices", and the
        # refusal was an exception rather than the False that hands the
        # projection to reconcile.
        for current, incoming in zip(self._spec.fields, spec.fields, strict=True):
            if current.cycle_choices != incoming.cycle_choices:
                _ChoiceHandler._fill_cycle(incoming, self._widgets[incoming.key])
        exact = self._require_exact_values(values, spec)
        for field in spec.fields:
            automatic = self._auto_switches.get(field.key)
            selected = automatic is not None and exact[field.key] is None
            if automatic is not None and automatic.isChecked() != selected:
                return False
            if selected:
                continue
            handler = FORM_WIDGET_HANDLERS[field.kind]
            widget = self._widgets[field.key]
            shown = _widget_value(handler, field, widget)
            # What the widget shows is a value of its own vocabulary, so a
            # projection that repeats it is accepted on that one read; only
            # a value that differs is normalized, which for a lazy axis
            # means walking it.
            if shown is _NO_VALUE or not (
                _same_typed_value(shown, exact[field.key])
                or _same_typed_value(shown, handler.normalize(field, exact[field.key]))
            ):
                return False
        self._spec = spec
        self._fields = {field.key: field for field in spec.fields}
        self._dependents = self._dependency_map(spec)
        return True

    def _prepare_population(
        self,
        values: Mapping[str, object],
        spec: FormSpec | None = None,
    ) -> dict[str, object]:
        """Every value normalized by its field, before any widget is touched.

        An automatic field given None is on Auto: it holds no value, so
        there is nothing to normalize and nothing for its editor to refuse.
        """

        spec = self._spec if spec is None else spec
        exact = self._require_exact_values(values, spec)
        return {
            field.key: (
                None
                if field.automatic and exact[field.key] is None
                else FORM_WIDGET_HANDLERS[field.kind].normalize(
                    field, exact[field.key]
                )
            )
            for field in spec.fields
        }

    def write_all(self, values: Mapping[str, object]) -> None:
        self.populate(values)

    def _layout_order(self) -> list[str]:
        """The keys in the order the layout is holding them right now."""

        rows = {id(row): key for key, row in self._rows.items()}
        order: list[str] = []
        for index in range(self._layout.count()):
            item = self._layout.itemAt(index)
            widget = None if item is None else item.widget()
            key = None if widget is None else rows.get(id(widget))
            if key is not None:
                order.append(key)
        return order

    def reconcile(
        self,
        spec: FormSpec,
        values: Mapping[str, object],
    ) -> None:
        """Keyed-diff a new declaration into this stable form owner.

        Same-key controls with a compatible concrete widget family are updated
        in place.  New keys create one row, removed keys destroy one row,
        reordering only moves existing rows, and a true widget-family change
        replaces only that key's row.  All values are normalized before the
        first QWidget mutation.
        """

        if not isinstance(spec, FormSpec):
            raise TypeError("spec must be FormSpec")
        if not isinstance(values, Mapping):
            raise TypeError("form values must be a mapping")
        incoming = {key: values[key] for key in values}
        supplied = set(incoming)
        expected = set(spec.keys)
        if supplied != expected:
            missing = sorted(repr(key) for key in expected - supplied)
            extra = sorted(repr(key) for key in supplied - expected)
            raise ValueError(
                f"form values must have exact keys; missing={missing}, extra={extra}"
            )
        new_handlers = {
            field.key: FORM_WIDGET_HANDLERS[field.kind]
            for field in spec.fields
        }
        prepared = self._prepare_population(incoming, spec)
        new_dependents = self._dependency_map(spec)

        old_fields = self._fields
        # WHICH row the operator is inside, asked once, before anything
        # is decided.  The retained loop already refused to write over
        # them; every other path here could still take the widget away
        # underneath their cursor -- rebuild it because its family
        # changed, or re-insert it because some other row was added.  A
        # row that disappears mid-word is the same defect as a value that
        # changes mid-word, and it is the one the operator sees as the
        # whole form collapsing.
        #
        # A key that has LEFT the spec is not that case.  Its row is kept
        # from being rebuilt, never from being removed: nothing owns the
        # field any more, so a row that stayed behind would edit nothing.
        editing_key = next(
            (
                key
                for key, widget in self._widgets.items()
                # A choice has no partial draft: activating a row completes
                # the edit.  Focus returns to the collapsed combo while the
                # owner projects the resulting vocabulary, so treating that
                # focus like an unfinished text cursor lets a choice-domain
                # refill reset the visible value to its first row (Reduced)
                # and then suppresses the accepted value that should restore
                # it.
                if self._fields[key].kind != "choice"
                and being_edited(widget)
            ),
            None,
        )
        replacements: dict[
            str,
            tuple[QtWidgets.QWidget, QtWidgets.QWidget, FluentSwitch | None],
        ] = {}
        label_width = self._label_width or _form_label_width(spec.fields)
        old_label_width = self._label_width or _form_label_width(
            self._spec.fields
        )
        for field in spec.fields:
            old_field = old_fields.get(field.key)
            if (
                old_field is not None
                and _widget_family(old_field) == _widget_family(field)
            ):
                continue
            if field.key == editing_key:
                # Its family changed while they are typing in it.  The
                # new widget lands the moment they leave; a projection
                # is never worth the words they are still writing.
                continue
            handler = new_handlers[field.key]
            widget = handler.build(
                field,
                _seed(field, prepared[field.key]),
                lambda key=field.key: self.changed.emit(key),
                self._runtime,
            )
            self._connect_editor_signals(field.key, widget)
            replacements[field.key] = (
                widget,
                *self._make_row(
                    field,
                    widget,
                    label_width,
                    automatic_checked=prepared[field.key] is None,
                ),
            )

        retained_widgets = tuple(
            self._widgets[field.key]
            for field in spec.fields
            if field.key not in replacements and field.key in self._widgets
        )
        retained_switches = tuple(
            self._auto_switches[field.key]
            for field in spec.fields
            if field.key not in replacements and field.key in self._auto_switches
        )
        # Updates are held only while rows are built, removed or moved.
        # Re-enabling updates repaints the WHOLE form whether or not
        # anything changed, and a control re-projected every beat repainted
        # every beat: a projection that changes no row repaints no row.
        desired_keys = set(spec.keys)
        structural = bool(replacements) or any(
            key not in desired_keys for key in self._rows
        ) or [key for key in self._layout_order() if key in self._rows] != [
            field.key for field in spec.fields if field.key in self._rows
        ]
        if structural:
            self.setUpdatesEnabled(False)
        try:
            with signals_blocked(*retained_widgets, *retained_switches):
                for field in spec.fields:
                    if field.key in replacements:
                        continue
                    old_field = old_fields[field.key]
                    widget = self._widgets[field.key]
                    handler = new_handlers[field.key]
                    _reconfigure_widget(old_field, field, widget)
                    if field.kind == "keyed_choice":
                        # Its legal keys come from the live runtime context,
                        # not FormSpec.  Put the new choice domain into the
                        # retained tree before writing the owner's new key;
                        # doing this afterwards silently cleared a signal that
                        # only existed in the incoming domain.
                        handler.refresh(field, widget, self._runtime)
                    selected = (
                        field.key in self._auto_switches
                        and prepared[field.key] is None
                    )
                    # Choice edits are atomic.  Their owner projection must
                    # always win after a domain refill even though the combo
                    # still owns keyboard focus from the activation click.
                    editing = (
                        field.kind != "choice" and being_edited(widget)
                    )
                    if (
                        not editing
                        and not selected
                        and not _widget_has_value(
                            handler, field, widget, prepared[field.key]
                        )
                    ):
                        handler.write(field, widget, prepared[field.key])
                    automatic = self._auto_switches.get(field.key)
                    if automatic is not None and not editing:
                        automatic.setChecked(selected)
                        automatic.setText(_automatic_label(field, selected))

            replaced_keys = set(replacements)
            for key in tuple(self._rows):
                if key in desired_keys and key not in replaced_keys:
                    continue
                # No exemption for the focused row here.  A focused key
                # that is still in the spec never reaches this loop --
                # the replacement loop skips it, so the clause above
                # keeps it.  Exempting it a second time could therefore
                # only retain a row whose key had LEFT the spec, while
                # _fields and _handlers below are rebuilt from the new
                # spec alone: the next keystroke fired the build-time
                # changed(key), read_value raised KeyError out of a Qt
                # slot, and PyQt aborted the process with no traceback.
                self._retire_row(key)

            # The old row is gone; only now are the new row's parts recorded,
            # so the retirement above cannot erase what it did not build.
            for key, parts in replacements.items():
                field = next(field for field in spec.fields if field.key == key)
                self._install_row(field, *parts)

            # Only the rows whose POSITION changed.  This used to be
            # "any key was replaced or the key list differs", so adding
            # one row at the end pulled every surviving row out of the
            # layout and put it back -- which is most of what the
            # operator sees as the form collapsing and rebuilding.
            #
            # Compare LIKE WITH LIKE.  A row built above is already in
            # ``self._rows`` but not yet in the layout, so comparing the
            # layout against the whole wanted list said "order changed"
            # for every append -- the very case the paragraph above says
            # is handled.  The question is whether the rows that ARE
            # placed are in the right relative order; the ones that are
            # not placed get inserted below regardless.
            placed = [
                key for key in self._layout_order() if key in self._rows
            ]
            in_layout = set(placed)
            wanted = [field.key for field in spec.fields if field.key in self._rows]
            order_changed = placed != [key for key in wanted if key in in_layout]
            for index, field in enumerate(spec.fields):
                if field.key not in self._rows:
                    continue
                row = self._rows[field.key]
                if isinstance(row, FluentSettingRow):
                    automatic = self._auto_switches.get(field.key)
                    label = (
                        _automatic_label(field, automatic.isChecked())
                        if automatic is not None
                        else self._label_for(field)
                    )
                    old_field = old_fields.get(field.key)
                    if (
                        old_label_width != label_width
                        or old_field is None
                        or old_field.row_label != field.row_label
                    ):
                        row.set_label(label, width=label_width)
                if order_changed or field.key not in in_layout:
                    # removeWidget on a row that is not in the layout is a
                    # no-op, so a fresh row simply lands at its index.
                    self._layout.removeWidget(row)
                    self._layout.insertWidget(index, row)
                    if self.isVisible() and not row.isVisible():
                        # A row inserted while this form is visible stays
                        # hidden until the next layout activation, and a
                        # hidden child is an EMPTY layout item measuring
                        # zero height.  Anyone sizing a popup right after
                        # reconcile would pin a height with the new rows
                        # missing -- and the buttons row then painted over
                        # the form until the popup was reopened.
                        row.show()

            self._spec = spec
            self._fields = {field.key: field for field in spec.fields}
            self._handlers = new_handlers
            self._dependents = new_dependents
        finally:
            if structural:
                self.setUpdatesEnabled(True)
        self._project_enabled_all(editing_key=editing_key)

    def refresh(self) -> None:
        """Refresh every handler, preserving legal selections and edit silence."""

        widgets = tuple(self._widgets[key] for key in self._spec.keys)
        with signals_blocked(*widgets):
            for field in self._spec.fields:
                self._handlers[field.key].refresh(
                    field,
                    self._widgets[field.key],
                    self._runtime,
                )
                if field.key in self._auto_switches:
                    self._set_automatic_label(field.key)
        self._project_enabled_all()

    def _field_for(self, key: str) -> FormFieldProps:
        try:
            return self._fields[key]
        except KeyError as exc:
            raise KeyError(f"unknown form field key: {key!r}") from exc

    def _require_exact_values(
        self, values: Mapping[str, object], spec: FormSpec | None = None
    ) -> dict[str, object]:
        spec = self._spec if spec is None else spec
        if not isinstance(values, Mapping):
            raise TypeError("form values must be a mapping")
        supplied = set(values.keys())
        expected = set(spec.keys)
        if supplied != expected:
            missing = sorted(repr(key) for key in expected - supplied)
            extra = sorted(repr(key) for key in supplied - expected)
            raise ValueError(
                f"form values must have exact keys; missing={missing}, extra={extra}"
            )
        return {key: values[key] for key in spec.keys}


__all__ = [
    "FluentParameterForm",
    "FormRuntimeContext",
]
