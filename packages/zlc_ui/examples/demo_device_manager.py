"""Human acceptance demo for the plain-data DeviceManager view."""

from __future__ import annotations

import argparse
import os

# The facade and nothing else: this file is the tutorial for the surface an
# outside host may use, so it is written under the rule that surface enforces.
from zlc_ui import (
    FormFieldProps,
    FormSpec,
    ensure_qt_app,
    open_device_control,
    open_device_manager,
)


def populate(view) -> None:
    view.set_device_choices(
        (
            ("Fake sensor", "sensor.fake", "sensor"),
            ("Fake camera", "camera.fake", "camera"),
            ("Fake DAC", "rf.fake", "rf"),
        )
    )
    view.set_devices(
        (
            ("sensor-1", "input", "sensor.fake", "sensor"),
            ("camera-1", "imaging", "camera.fake", "camera"),
        )
    )
    spec = FormSpec(
        (
            FormFieldProps("count", "int", "Count", default=4, minimum=0, maximum=99),
            FormFieldProps("enabled", "bool", "Enabled", default=True),
        )
    )
    view.set_form_spec("sensor-1", spec, (("count", 4), ("enabled", True)))
    view.set_form_spec("camera-1", spec, (("count", 2), ("enabled", True)))
    view.set_loaded_devices(
        (
            ("sensor-1", "input", "sensor.fake"),
            ("camera-1", "imaging", "camera.fake"),
        )
    )

    controls = {}

    def open_control(instance_id: str) -> None:
        existing = controls.get(instance_id)
        if existing is not None:
            existing.restore()
            return
        control = open_device_control(
            title=f"{instance_id} control",
            spec=spec,
            values=(
                ("count", 4 if instance_id == "sensor-1" else 2),
                ("enabled", True),
            ),
        )
        controls[instance_id] = control
        control.field_committed.connect(
            lambda key, name=instance_id, handle=control: print(
                f"device_control_committed{(name, key, handle.read_values())!r}",
                flush=True,
            )
        )
        control.closed.connect(lambda name=instance_id: controls.pop(name, None))
        control.show_status("Offline fake device · controls are local demo data", "idle")

    view.device_open_requested.connect(open_control)
    view._demo_controls = controls

    def close_controls() -> None:
        for control in tuple(controls.values()):
            control.close()

    view.closed.connect(close_controls)

    def remember(name: str):
        def handler(*payload) -> None:
            print(f"{name}{payload!r}", flush=True)

        return handler

    view.device_add_requested.connect(remember("device_add_requested"))
    view.device_remove_requested.connect(remember("device_remove_requested"))
    view.role_committed.connect(remember("role_committed"))
    view.type_picked.connect(remember("type_picked"))
    view.parameter_committed.connect(remember("parameter_committed"))
    view.save_requested.connect(remember("save_requested"))
    view.show_status("Offline fake devices · edit only", "idle")
    print("device_demo_ready", flush=True)


def create_window(
    argv: list[str] | None = None,
    *,
    window_ratio: float | None = None,
):
    """Open the device manager the way an outside host does, with fakes."""

    ensure_qt_app(["zlc-ui-device-demo", *(argv or [])])
    view = open_device_manager(title="DeviceManager@Zou lab", window_ratio=window_ratio)
    populate(view)
    return view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    app = ensure_qt_app(["zlc-ui-device-demo", *(argv or [])])
    window = create_window(argv)
    app.processEvents()
    if args.once or os.environ.get("ZLC_UI_DEVICE_ONESHOT") == "1" or app.platformName().strip().lower() == "offscreen":
        window.close()
        app.processEvents()
        return 0
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
