from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_card_extension_enters_the_console_card_surface() -> None:
    extension = ROOT / "examples" / "synthetic_card.py"
    demo = (ROOT / "examples" / "demo_console.py").read_text(encoding="utf-8")
    assert extension.is_file()
    assert "class SyntheticCardView" in extension.read_text(encoding="utf-8")
    assert "SyntheticCardView(f\"Synthetic surface {index}\")" in demo
    # It reaches the card as the thing a HOST draws with, not as a widget the
    # demo mounts itself: the outside hands over the host and nothing else.
    assert "console.show_panel(" in demo
    assert "def qt_widget(self):" in demo


def test_demo_console_once_echoes_a_view_signal() -> None:
    environment = dict(__import__("os").environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = __import__("subprocess").run(
        [__import__("sys").executable, "examples/demo_console.py", "--once"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "console_demo_ready" in completed.stdout
    # And every intent the port raises has an answer: a demo that wires some of
    # them shows a window whose other controls do nothing.
    demo = (ROOT / "examples" / "demo_console.py").read_text(encoding="utf-8")
    for name in (
        "panel_signal_picked", "panel_size_picked", "panel_update_ms_picked",
        "panel_title_committed", "panel_remove_requested", "panel_edit_requested",
        "logic_start_requested", "logic_stop_requested", "logic_edit_requested",
        "logic_remove_requested",
    ):
        assert name in demo, name
