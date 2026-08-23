"""Run one command from the installed product manifest: ``zlc <command>``."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
import argparse
import inspect
import os
from pathlib import Path
import sys

from . import DISTRIBUTION_NAME, entry_specs


_COMMAND_GROUP = "zou_lab_control.commands"
_EVIDENCE_GROUP = "zou_lab_control.evidence"


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = entry_specs(_COMMAND_GROUP)
    if not arguments or arguments[0] in ("-h", "--help"):
        print("usage: zlc <command> [options]")
        print("commands: " + ", ".join(commands))
        return 0 if arguments else 2
    name, rest = arguments[0], arguments[1:]
    spec = commands.get(name)
    if spec is None:
        print(f"unknown command {name!r}; expected one of {', '.join(commands)}")
        return 2
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError(f"invalid product command entry point {spec!r}")
    entry = getattr(import_module(module_name), attribute)
    takes_argv = bool(inspect.signature(entry).parameters)
    result = entry(rest) if takes_argv else entry()
    return int(result or 0)


def _pytest_process(names: tuple[str, ...], paths: tuple[object, ...]) -> int:
    """Run one package/Qt lifetime in its own installed Python process."""

    import subprocess
    import tempfile

    script = r"""
from importlib import import_module
from pathlib import Path
import sys
import zou_lab_control
print(f'root={Path(zou_lab_control.__file__).resolve()}')
for name in sys.argv[1].split(','):
    module = import_module(name)
    print(f'tested={Path(module.__file__).resolve()}')
import pytest
raise SystemExit(pytest.main(['-q', *sys.argv[2:]]))
"""
    with tempfile.TemporaryDirectory(prefix="zlc-evidence-") as folder:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    ",".join(names),
                    *(str(path) for path in paths),
                ],
                cwd=folder,
                env=os.environ.copy(),
                timeout=900,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"pytest process exceeded 900 s: {paths}")
            return 124
    return int(result.returncode)


def _pytest_items(names: tuple[str, ...], path: Path) -> int:
    """Collect one Qt-heavy file, then give every item a fresh process."""

    import subprocess
    import tempfile

    script = r"""
from importlib import import_module
from pathlib import Path
import sys
import zou_lab_control
print(f'root={Path(zou_lab_control.__file__).resolve()}')
for name in sys.argv[1].split(','):
    module = import_module(name)
    print(f'tested={Path(module.__file__).resolve()}')
import pytest
raise SystemExit(pytest.main(['--collect-only', '-q', sys.argv[2]]))
"""
    with tempfile.TemporaryDirectory(prefix="zlc-collect-") as folder:
        try:
            collected = subprocess.run(
                [sys.executable, "-c", script, ",".join(names), str(path)],
                cwd=folder,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"pytest collection exceeded 120 s: {path}")
            return 124
    if collected.returncode != 0:
        print(collected.stdout, end="")
        print(collected.stderr, end="", file=sys.stderr)
        return int(collected.returncode)
    items = tuple(
        f"{path}::{line.split('::', 1)[1]}"
        for line in collected.stdout.splitlines()
        if "::" in line and not line.startswith("tested=")
    )
    if not items:
        print(f"no pytest items collected from {path}")
        return 2
    for item in items:
        result = _pytest_process(names, (item,))
        if result != 0:
            return result
    return 0


def evidence(argv: list[str] | None = None) -> int:
    """Execute one installed software lane, or identify a manual-only lane."""

    lanes = entry_specs(_EVIDENCE_GROUP)
    parser = argparse.ArgumentParser(prog="zlc evidence")
    parser.add_argument("lane", nargs="?", choices=tuple(lanes))
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--list", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.list or arguments.lane is None:
        for name in lanes:
            boundary = "manual only" if name in {"real_screen", "hardware"} else "automated"
            print(f"{name}: {boundary}")
        return 0 if arguments.list else 2

    lane = arguments.lane
    if lane in {"real_screen", "hardware"}:
        print(
            f"{lane}: NOT EXECUTED automatically; follow the installed-product "
            "runbook in README.md and the device-specific package README."
        )
        return 2
    if arguments.repo is None:
        print("automated evidence requires --repo pointing at the source/test tree")
        return 2
    repo = arguments.repo.expanduser().resolve()
    if not (repo / "pyproject.toml").is_file():
        print(f"not a ZLC source/test tree: {repo}")
        return 2
    if repo in Path(__file__).resolve().parents:
        print("evidence must run from the installed wheel, not the checkout")
        return 2

    product_path = Path(__file__).resolve()
    try:
        product = distribution(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        print(f"evidence requires an installed {DISTRIBUTION_NAME} wheel")
        return 2
    installed_files = {
        Path(product.locate_file(item)).resolve()
        for item in (product.files or ())
    }
    if product_path not in installed_files:
        print(f"evidence runner is not owned by installed {DISTRIBUTION_NAME}: {product_path}")
        return 2

    os.environ["ZLC_TEST_INSTALLED"] = "1"
    os.environ["PYTHONPATH"] = ""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["MPLBACKEND"] = "Agg"

    from zlc_workbench.tools.check_environment import check

    ownership_problems = check()
    if ownership_problems:
        for problem in ownership_problems:
            print(f"FAIL  {problem}")
        return 1

    if lane == "notebook_offline":
        import zlc_plot
        import zlc_workbench

        print(f"product={Path(__file__).resolve()}")
        print(f"tested={Path(zlc_workbench.__file__).resolve()}")
        print(f"tested={Path(zlc_plot.__file__).resolve()}")
        import tempfile
        import json
        import nbformat
        from nbclient import NotebookClient

        notebook = repo / "packages/zlc_workbench/notebooks/usage.ipynb"
        document = nbformat.read(notebook, as_version=4)
        with tempfile.TemporaryDirectory(prefix="zlc-notebook-evidence-") as folder:
            kernel_root = Path(folder) / "kernels" / "zlc-fresh"
            kernel_root.mkdir(parents=True)
            (kernel_root / "kernel.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            sys.executable,
                            "-m",
                            "ipykernel_launcher",
                            "-f",
                            "{connection_file}",
                        ],
                        "display_name": "ZLC fresh evidence",
                        "language": "python",
                    }
                ),
                encoding="utf-8",
            )
            previous_jupyter_path = os.environ.get("JUPYTER_PATH")
            os.environ["JUPYTER_PATH"] = folder
            NotebookClient(
                document,
                timeout=120,
                kernel_name="zlc-fresh",
                resources={"metadata": {"path": folder}},
            ).execute()
            if previous_jupyter_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous_jupyter_path
        provenance = next(
            cell for cell in document.cells if cell.get("id") == "provenance"
        )
        output = "".join(
            item.get("text", "")
            for item in provenance.get("outputs", ())
            if item.get("output_type") == "stream"
        )
        expected = (
            f"product bootstrap: {Path(sys.modules['zou_lab_control'].__file__).resolve()}",
            f"workbench layer: {Path(zlc_workbench.__file__).resolve()}",
        )
        if any(line not in output for line in expected):
            raise RuntimeError(
                "notebook kernel provenance differs from the fresh evidence environment"
            )
        print("notebook_offline: PASS")
        return 0

    if lane == "software":
        groups_list: list[tuple[tuple[str, ...], tuple[object, ...]]] = []
        itemized: list[tuple[tuple[str, ...], Path]] = []
        for name in entry_specs("zou_lab_control.layers"):
            tests = repo / f"packages/{name}/tests"
            if name != "zlc_workbench":
                groups_list.append(((name,), (tests,)))
                continue
            for path in sorted(tests.glob("test_*.py")):
                if path.name == "test_task_console_app.py":
                    itemized.append(((name,), path))
                else:
                    groups_list.append(((name,), (path,)))
        groups = tuple(groups_list)
    elif lane == "gui_offscreen":
        itemized = [
            (
                ("zlc_workbench",),
                repo / "packages/zlc_workbench/tests/test_task_console_app.py",
            )
        ]
        groups = (
            (("zlc_ui",), (repo / "packages/zlc_ui/tests",)),
            (("zlc_plot",), (repo / "packages/zlc_plot/tests/test_qt_widget.py",)),
            (("zlc_plot",), (repo / "packages/zlc_plot/tests/test_semantic_ui.py",)),
            (("zlc_atom",), (repo / "packages/zlc_atom/tests/test_slm_editor.py",)),
            (("zlc_workbench",), (repo / "packages/zlc_workbench/tests/test_console_presenter.py",)),
            (("zlc_workbench",), (repo / "packages/zlc_workbench/tests/test_device_manager.py",)),
            (("zlc_workbench",), (repo / "packages/zlc_workbench/tests/test_pulse_editor.py",)),
            (("zlc_workbench",), (repo / "packages/zlc_workbench/tests/test_viewer.py",)),
            (("zlc_workbench",), (repo / "packages/zlc_workbench/tests/test_gui_seam.py",)),
            (("zlc_workbench",), (repo / "packages/zlc_workbench/tests/test_windows.py",)),
            (("zlc_workbench",), (repo / "packages/zlc_workbench/tests/test_editor_named_behaviours.py",)),
        )
    else:
        itemized = []
        groups = (
            (
                ("zlc_atom", "zlc_workbench"),
                (
                    repo / "packages/zlc_workbench/tests/test_guard_a_virtual_chain.py",
                    repo / "packages/zlc_workbench/tests/test_guard_b_task_console_interaction.py",
                    repo / "packages/zlc_workbench/tests/test_guard_c_save_semantics.py",
                    repo / "packages/zlc_atom/tests/test_real_runtime_integration.py",
                    repo / "packages/zlc_atom/tests/test_temperature_chain.py",
                    str(repo / "packages/zlc_atom/tests/test_slm_feedback_task.py")
                    + "::test_virtual_feedback_recovers_missing_sites_and_retains_best_candidate",
                    str(repo / "packages/zlc_atom/tests/test_seamless_scan_node.py")
                    + "::test_the_board_advanced_scan_recovers_the_planted_trap_loss",
                    str(repo / "packages/zlc_atom/tests/test_stepped_scan_node.py")
                    + "::test_scanning_the_bias_dacs_finds_the_planted_mot_optimum",
                ),
            ),
        )
    status = 0
    for names, paths in groups:
        result = _pytest_process(names, paths)
        if result != 0:
            status = result
            break
    if status == 0:
        for names, path in itemized:
            result = _pytest_items(names, path)
            if result != 0:
                status = result
                break
    print(f"{lane}: {'PASS' if status == 0 else 'FAIL'}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
