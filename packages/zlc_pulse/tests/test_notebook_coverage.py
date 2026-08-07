from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
import re

import zlc_pulse


ROOT = Path(__file__).resolve().parents[1]
_BOARD_SIGNALS = ("cool" + "ing", "em" + "CCD", "da_" + "dipole", "da_bias" + "_y", "da_bias" + "_x", "da_bias" + "_z")


def _notebook() -> dict:
    return json.loads((ROOT / "notebooks/usage.ipynb").read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [cell for cell in notebook["cells"] if cell.get("cell_type") == "code" and _source(cell).strip()]


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _name_nodes(source: str) -> set[str]:
    tree = ast.parse(source)
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def test_tutorial_cells_are_small_explanatory_and_visible() -> None:
    notebook = _notebook()
    cells = notebook["cells"]
    executable = _code_cells(notebook)
    assert executable
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = _source(cell)
        if not source.strip():
            continue
        tree = ast.parse(source, filename=cell["id"])
        print_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
        ]
        assert len(source.splitlines()) <= 25, cell["id"]
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)], cell["id"]
        assert print_calls, cell["id"]
        assert index > 0 and cells[index - 1].get("cell_type") == "markdown", cell["id"]
    assert executable[-1]["id"] == "hardware-direct"


def test_tutorial_uses_every_retained_package_export_in_a_lesson() -> None:
    code = "\n".join(_source(cell) for cell in _code_cells(_notebook()))
    names = _name_nodes(code)
    missing = [name for name in zlc_pulse.__all__ if name not in names]
    assert missing == []


def test_tutorial_has_named_lessons_for_runtime_methods_and_readbacks() -> None:
    notebook = _notebook()
    by_id = {cell["id"]: _source(cell) for cell in _code_cells(notebook)}
    for cell_id in (
        "compile", "hardware-applied", "hardware-finite", "hardware-slots",
        "hardware-scan", "hardware-stop", "hardware-open", "hardware-load",
        "hardware-forever", "hardware-direct",
    ):
        assert cell_id in by_id
    hardware = "\n".join(by_id[cell_id] for cell_id in by_id if cell_id.startswith("hardware-"))
    for method in (
        "remote.open()", "remote.load(", "remote.applied(", "remote.fire()", "remote.wait_done(",
        "remote.write_slots(", "remote.write_scan_table(", "remote.cursor(",
        "remote.fire(forever=True)", "remote.snapshot(", "remote.safe(", "remote.close(",
    ):
        assert method in hardware, method
    for field in (
        "status_first", "status_second", "cursor_first", "cursor_second", "status",
        "underflow", "tail_elapsed", "clock_enable_words", "stable",
    ):
        assert field in hardware, field
    assert hardware.index("remote.fire(forever=True)") < hardware.index("remote.safe()")
    stop = by_id["hardware-stop"]
    assert stop.index("remote.safe()") < stop.index("remote.close()")
    assert "run_server.bat" in by_id["hardware-open"] and "18861" in by_id["hardware-open"]


def test_final_hardware_cell_is_the_static_scope_program_and_first_fire() -> None:
    executable = _code_cells(_notebook())
    final = _source(executable[-1])
    assert executable[-1]["id"] == "hardware-direct"
    assert "connect('127.0.0.1', 18861" in final
    assert "scope_program.ticks != (0, 50, 100)" in final
    assert "scope_program.slot_count != 0" in final
    assert "'F15', 'M13'" in final
    assert final.index("remote.load(") < final.index("remote.fire(forever=True)")
    assert "remote.write_slots(" not in final and "remote.write_scan_table(" not in final


def test_final_hardware_cell_compiles_the_waveform_seen_on_the_connected_pins() -> None:
    notebook = _notebook()
    namespace: dict[str, object] = {}
    for cell in _code_cells(notebook):
        if cell["id"] == "hardware-open":
            break
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(_source(cell), f"<notebook:{cell['id']}>", "exec"), namespace, namespace)

    class FakeRemote:
        program = None
        forever = False
        def open(self): pass
        def load(self, program, **_kwargs): self.program = program
        def fire(self, *, forever=False): self.forever = forever
        def snapshot(self): return {"firing": True, "forever": self.forever, "status": 2}
        def disconnect(self): pass

    fake = FakeRemote()
    namespace["connect"] = lambda *_args, **_kwargs: fake
    final = next(cell for cell in _code_cells(notebook) if cell["id"] == "hardware-direct")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_source(final), "<notebook:hardware-direct>", "exec"), namespace, namespace)

    program = fake.program
    target = namespace["hardware_target"]
    lanes = target.raw_lanes
    pin_lanes = {pin: lane for lane, pin in target.package_pins.items()}
    pin_bits = tuple(lanes.index(pin_lanes[pin]) for pin in ("F15", "M13"))
    assert program.ticks == (0, 50, 100) and program.slot_count == 0
    assert all(program.masks[0] & (1 << bit) for bit in pin_bits)
    assert program.masks[1:] == (0, 0)
    assert len(program.bus_segments) == 2 * namespace["hardware_geometry"].bus_count
    assert {segment.stop_value for segment in program.bus_segments} == {0, 1023}
    assert fake.forever is True


def test_notebook_guard_covers_the_exact_eleven_device_methods() -> None:
    contract = (ROOT / "docs/contract.md").read_text(encoding="utf-8")
    section = contract.split("## XDC board mapping", 1)[0]
    pulse = section.split("class PulseStreamer:", 1)[1]
    methods = set(re.findall(r"^    ([a-z][a-z0-9_]*)\(", pulse, re.MULTILINE))
    expected = {
        "open", "close", "load", "write_slots", "write_scan_table", "fire",
        "wait_done", "cursor", "safe", "snapshot", "applied",
    }
    assert methods & expected == expected
    notebook = _notebook()
    hardware = "\n".join(
        _source(cell) for cell in _code_cells(notebook) if cell["id"].startswith("hardware-")
    )
    for method in expected:
        assert f"remote.{method}(" in hardware, method


def test_notebook_offline_prefix_executes_without_hardware(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    namespace: dict[str, object] = {}
    output = io.StringIO()
    ran: list[str] = []
    for cell in _code_cells(_notebook()):
        if cell["id"] == "hardware-open":
            break
        with contextlib.redirect_stdout(output):
            exec(compile(_source(cell), f"<notebook:{cell['id']}>", "exec"), namespace, namespace)
        ran.append(cell["id"])
    assert ran[-2:] == ["hardware-config", "hardware-program"]


def test_board_signal_names_are_confined_to_the_xdc() -> None:
    roots = (
        ROOT / "src", ROOT / "tests", ROOT / "notebooks", ROOT / "examples", ROOT / "docs",
        ROOT / "README.md", ROOT / "fpga" / "README.md", ROOT / "fpga" / "board_config" / "README.md",
        ROOT / "fpga" / "pulse_streamer" / "README.md", ROOT / "fpga" / "run_server.bat",
    )
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                path for path in root.rglob("*")
                if path.is_file() and not {".ipynb_checkpoints", "__pycache__"}.intersection(path.parts)
                and path.name != "goal-archive.md"
                and path.suffix != ".pyc"
            )
    hits = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits.extend(f"{path}: {signal}" for signal in _BOARD_SIGNALS if signal in text)
    assert hits == []


def test_run_server_reports_connection_and_listener_state() -> None:
    launcher = (ROOT / "fpga/run_server.bat").read_text(encoding="utf-8")
    remote = (ROOT / "src/zlc_pulse/remote.py").read_text(encoding="utf-8")
    assert "HARDWARE CONNECTED" in launcher and "RPC LISTENING" in launcher
    assert "JTAG-to-AXI" in launcher and "JTAG-to-AXI" in remote
    assert "hardware CONNECTED" in remote and "RPC LISTENING" in remote
    assert "client CONNECTED" in remote and "outputs are SAFE" in remote
    assert "SERVER ADDRESS" in remote and "CLIENT ENDPOINT" in remote
    assert "remote_server same-computer endpoint" in launcher and "127.0.0.1" in launcher
    assert "echo Server listen bind:" in launcher and "CLIENT CONNECT EXAMPLE" in remote
    assert 'resolve_tools.bat" vivado' in launcher and "/optional" in launcher
    assert "--uart-port explicitly" in launcher
    assert "0.0.0.0 is listen-only" in remote and "same_computer_client=" in remote
    assert "DEFAULT_CLIENT_IDLE_TIMEOUT" in remote and "RemoteBusyError" in remote
    assert "ZLC_PS_SERVER_BACKEND" not in launcher and "%ZLC_FORWARD_ARGS%" in launcher
    assert "default=\"auto\"" in remote and "resolve_backend" in remote
