from __future__ import annotations
import time
import test_console_presenter as T
from test_console_presenter import presenter, session  # noqa: F401
from zlc_workbench.logic import stable_signal_key


def test_probe(presenter, session) -> None:
    from PyQt5 import QtCore, QtGui
    from zlc_plot.backends import ensure_qt5_application
    app = ensure_qt5_application()
    camera_id = presenter.add_logic(
        "camera_measurement", node_id="roll",
        values={"exposure_seconds": 0.002, "repeat": 0, "frames_per_cycle": 1},
        device_keys={"camera": "camera"}, open_editor=False,
    )
    session.load_pulse(T.PULSE_NAME)
    assert presenter.start_logic(camera_id)
    signal = stable_signal_key(camera_id, "frames")
    pub = None
    deadline0 = time.monotonic() + 20.0
    while pub is None and time.monotonic() < deadline0:
        session.fire(shots=1); presenter.beat()
        pub = session.signal_plane.latest_publication(signal)
        time.sleep(0.005)
    snapshot = pub.value(signal).snapshot
    binding = presenter.add_panel(signal, snapshot, kind="rolling")
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and (
        binding.host is None or binding.accepted_surface is None
    ):
        presenter.beat(); app.processEvents(); time.sleep(0.005)
    assert binding.host is not None
    # window smaller than the run: the visible window must SLIDE
    binding.host.set_parameter("window", 4).result(timeout=20)
    binding.host.set_parameter("trailing", 12).result(timeout=20)
    for _ in range(80):
        session.fire(shots=1)
        presenter.beat(); presenter.poll_logic(); app.processEvents(); time.sleep(0.006)
    widget = binding.host.qt_widget(); widget.resize(520, 380); widget.show()
    for _ in range(60):
        session.fire(shots=1)
        presenter.beat(); presenter.poll_logic(); app.processEvents(); time.sleep(0.006)
    f = widget.presented_front
    ax = f.interaction.axes[0]
    print("window=", binding.host.describe_display().result(timeout=20).value.display_state.values.get("window"),
          " xlim=", getattr(ax, "x_limits", None))
