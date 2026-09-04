"""Human acceptance demo for the pure pulse-editor views.

Everything in this example is fake data.  The log is the presenter seam: it
receives plain signal payloads and prints the same event to stdout.
"""

from __future__ import annotations

import argparse
import os

from PyQt5 import QtCore, QtWidgets

# Everything an outside host may use, from the top-level facade and nowhere
# else.  This demo is the tutorial for that surface, so it is written under the
# same rule the surface enforces: no submodule reaches, no widget classes.
from zlc_ui import (
    ConnectionChoiceVM,
    ConnectionVM,
    DelayRowVM,
    FieldVM,
    FormChoice,
    PeriodVM,
    PortRowVM,
    ScanPageRecord,
    ScheduleVM,
    TargetPortRecord,
    TargetWidthRule,
    VALIDATOR_FLOAT,
    VALIDATOR_INT,
    ensure_qt_app,
    open_pulse_editor,
)


def _field(
    text: str,
    *,
    binding: str = "",
    number: int = 0,
    editable: bool = True,
    validator_kind: str | None = None,
    validator_lo: float = -100,
    validator_hi: float = 100,
) -> FieldVM:
    kind = validator_kind or ("int" if text.lstrip("-").isdigit() else "float")
    return FieldVM(text, editable=editable, binding_kind=binding, binding_number=number,
                   binding_tooltip=f"fake {binding} parameter" if binding else "",
                   validator_kind=kind,
                   validator_lo=validator_lo, validator_hi=validator_hi,
                   resolution=1 if kind == "int" else 0,
                   allow_any=False)


def fake_schedule() -> ScheduleVM:
    digital_labels = (
        "cooling", "cooling_pgc", "repump", "probe", "pushout", "state_pre",
        "trig", "coil", "grey_cooling", "trap", "UV", "emCCD", "microwave",
        "address", "cooling_shutter", "repump_shutter", "probe_shutter", "bias",
    )
    digital_ports = tuple(
        PortRowVM(
            f"ch{index:02d}",
            "digital",
            label,
            f"ch{index:02d}",
            visible=index < 4,
        )
        for index, label in enumerate(digital_labels)
    )
    dac_ports = (
        PortRowVM(
            "da_dipole", "dac", "da_dipole", "da_dipole",
            width=10, lo=-512, hi=511, visible=False,
        ),
        PortRowVM(
            "da_bias_y", "dac", "da_bias_y", "da_bias_y",
            width=10, lo=-512, hi=511, visible=True,
        ),
        PortRowVM(
            "da_bias_x", "dac", "da_bias_x", "da_bias_x",
            width=10, lo=-512, hi=511, visible=False,
        ),
        PortRowVM(
            "da_bias_z", "dac", "da_bias_z", "da_bias_z",
            width=10, lo=-512, hi=511, visible=False,
        ),
    )
    ports = (*digital_ports, *dac_ports)
    periods = (
        PeriodVM(
            "p1", "", _field("s0", binding="scan", number=1), "ns", ("ns", "us"),
            digital=tuple((port.key, port.key == "ch00") for port in ports[:4]),
            analog=(
                (
                    "da_bias_y",
                    "ramp",
                    _field(
                        "s1", binding="scan", number=2,
                        validator_kind="int", validator_lo=-512, validator_hi=511,
                    ),
                ),
            ),
        ),
        PeriodVM(
            "p2", "", _field("1000", binding="api", number=1), "ns", ("ns", "us"),
            digital=tuple((port.key, False) for port in ports[:4]),
            analog=(
                (
                    "da_bias_y",
                    "hold",
                    _field(
                        "0", editable=False,
                        validator_kind="int", validator_lo=-512, validator_hi=511,
                    ),
                ),
            ),
        ),
    )
    return ScheduleVM(
        document_generation=1,
        revision=1,
        document_name="pulse_20260801_080148 (new)*",
        clock_text="50 MHz · 20 ns",
        total_text="2 us",
        total_tooltip="fake total duration",
        period_count=2,
        visible_text="5/22",
        summary_text="5/22 ports visible | 2 periods | step 20 ns | 2e+03 ns | 1 pulses | scan 2 slots × 16 pts",
        ports=ports,
        periods=periods,
        analog_mode_choices=(
            FormChoice("Edge", "edge"),
            FormChoice("Ramp", "ramp"),
            FormChoice("Hold", "hold"),
        ),
        bracket=None,
        run_repeats=0,
        delay_rows=(
            DelayRowVM("ch00", _field("0"), "ns", (("ns", 1.0), ("us", 1000.0))),
            DelayRowVM("ch01", _field("0"), "ns", (("ns", 1.0), ("us", 1000.0))),
            DelayRowVM("ch02", _field("0", binding="api", number=2), "ns", (("ns", 1.0), ("us", 1000.0))),
            DelayRowVM("ch03", _field("0"), "ns", (("ns", 1.0), ("us", 1000.0))),
            DelayRowVM(
                "da_bias_y",
                _field("0", validator_kind="int", validator_lo=0, validator_hi=100000),
                "ns",
                (("ns", 1.0), ("us", 1000.0)),
            ),
        ),
        scan_summary_text="2 slots · 16 pts",
        min_bracket_count=2,
        default_bracket_count=2,
    )


def _surface(text: str, color: str) -> QtWidgets.QWidget:
    # A plain Qt label, standing in for the canvas a drawing package builds.
    # Deliberately not one of this package's own controls: a drawn panel comes
    # from outside, and the point of the fake is to show it crossing.
    widget = QtWidgets.QLabel(text)
    widget.setAlignment(QtCore.Qt.AlignCenter)
    widget.setMinimumSize(360, 180)
    widget.setStyleSheet(f"background: {color}; color: #323130; border: 1px solid #E1DFDD;")
    return widget


class _FakeHost:
    """What a drawing package hands over: a host that owns its own widget.

    The seam is here on purpose.  zlc_ui may not import the package that draws
    -- a canvas would carry matplotlib into the GUI layer -- and an outside
    host may not hold a QWidget, so what crosses is the object that OWNS the
    widget, and the window is the only side that unwraps it.
    """

    def __init__(self, widget, logical_size) -> None:
        self._widget = widget
        self.logical_size = logical_size

    def qt_widget(self):
        return self._widget


def populate(editor) -> None:
    editor.set_title("PulseGUI - pulse_20260801_080148 (new)*")
    # The fake editor is dirty, but no device is running, so the header status
    # remains neutral grey.
    editor.set_status_color("idle")
    editor.set_summary("5/22 ports visible | 2 periods | step 20 ns | 2e+03 ns | 1 pulses | repeat | scan 2 slots × 16 pts")

    def remember(name: str):
        def handler(*payload) -> None:
            text = f"{name}{payload!r}"
            print(text, flush=True)
        return handler

    schedule = editor
    schedule_vm = fake_schedule()
    schedule.set_schedule(schedule_vm)

    schedule.set_connection(ConnectionVM(
        choices=(
            ConnectionChoiceVM("Virtual (sim)", "virtual"),
            ConnectionChoiceVM("Remote server", "remote", endpoint_editable=True),
            ConnectionChoiceVM("Offline (edit only)", "offline"),
        ),
        selected="offline",
        endpoint="127.0.0.1:18861",
        status="Offline (edit only)",
    ))
    scan = editor
    scan.set_scan_repeats_range(0, 0)
    scan.set_scan_page(ScanPageRecord(
        slots_text="2 bound scan slots · fake table ready",
        table_text="0.0  1.0\n0.5  1.5\n1.0  2.0",
        source_text="scan_table = [[0.0, 1.0], [0.5, 1.5]]",
        source_dirty=False,
        repeats=0,
        busy=False,
        progress_text="held at point 2/16: fake_value = 0.5",
        progress_polling=False,
    ))
    target = editor
    target.set_target_width_rules(TargetWidthRule(1, 1, 1), TargetWidthRule(1, 2, 4))
    target.set_target_ports((
        TargetPortRecord("digital_1", "digital", "Laser gate", ("d0",), lane_order=(0,)),
        TargetPortRecord("da_bias_y", "dac", "da_bias_y", ("ch38", "ch37"), "da_bias_y_clock", "da_clk1", (0, 1)),
    ), True, "Offline fake target · changes are draft-only until Apply")
    preview = editor
    preview.set_preview_size_names(("1x1", "2x2", "4x4"))
    preview.show_preview_placeholder("Fake zlc_plot QWidget mount point — renderer supplied by presenter")
    preview.show_preview(_FakeHost(
        _surface("fake timeline canvas\n(no renderer in zlc_ui)", "#EAF4FF"),
        (520, 240),
    ))

    signal_views = (
        (schedule.document_name_committed, "document_name_committed"),
        (schedule.period_name_committed, "period_name_committed"),
        (schedule.duration_committed, "duration_committed"),
        (schedule.digital_committed, "digital_committed"),
        (schedule.analog_committed, "analog_committed"),
        (schedule.binding_cycle_requested, "binding_cycle_requested"),
        (schedule.move_period_requested, "move_period_requested"),
        (schedule.bracket_committed, "bracket_committed"),
        (schedule.run_repeats_committed, "run_repeats_committed"),
        (schedule.visible_ports_committed, "visible_ports_committed"),
        (scan.scan_hold_requested, "scan_hold_requested"),
        (scan.scan_step_requested, "scan_step_requested"),
        (scan.scan_source_edited, "scan_source_edited"),
        (scan.scan_run_requested, "scan_run_requested"),
        (target.target_apply_requested, "target_apply_requested"),
        (target.target_feedback_requested, "target_feedback_requested"),
        (preview.preview_include_off_toggled, "preview_include_off_toggled"),
        (preview.preview_save_requested, "preview_save_requested"),
    )
    for signal, name in signal_views:
        signal.connect(remember(name))
    QtCore.QTimer.singleShot(0, scan.scan_hold_requested.emit)
    QtCore.QTimer.singleShot(0, preview.preview_save_requested.emit)


def create_window(
    argv: list[str] | None = None,
    *,
    window_ratio: float | None = None,
):
    """Open the editor the way an outside host does, and fill it with fakes.

    One call for the window, one object back.  The demo never names a widget
    class and never reaches a page: what it holds is the handle, which is the
    whole of what zlc_ui offers.
    """

    ensure_qt_app(["zlc-ui-pulse-demo", *(argv or [])])
    editor = open_pulse_editor(title="PulseGUI@Zou lab", window_ratio=window_ratio)
    populate(editor)
    return editor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    app = ensure_qt_app(["zlc-ui-pulse-demo", *(argv or [])])
    window = create_window(argv)
    app.processEvents()
    one_shot = args.once or os.environ.get("ZLC_UI_PULSE_ONESHOT") == "1" or app.platformName().strip().lower() == "offscreen"
    if one_shot:
        # A non-blocking demo normally lives inside app.exec_().  In a
        # one-shot/offscreen process there is no event-loop shutdown path, so
        # close it explicitly before Python tears down the frameless shell.
        window.close()
        app.processEvents()
        return 0
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
