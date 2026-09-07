"""Every layer and command belongs to the one installed product manifest."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

from zlc_workbench.tools.check_environment import OWNED, check
from zou_lab_control import entry_specs


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_checkout_bootstrap_loads_all_layers_from_a_neutral_directory(
    tmp_path: Path,
) -> None:
    script = r"""
from pathlib import Path
import zou_lab_control
print(zou_lab_control.__file__)
root = Path(zou_lab_control.__file__).resolve().parent.parent
for name in zou_lab_control.entry_specs("zou_lab_control.layers"):
    module = __import__(name)
    origin = Path(module.__file__).resolve()
    print(origin)
    assert root in origin.parents
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment.pop("ZLC_TEST_INSTALLED", None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_every_package_resolves_to_its_own_product() -> None:
    problems = check()
    assert problems == [], "\n".join(problems)


def test_product_manifest_owns_all_commands_and_layers() -> None:
    assert set(entry_specs("zou_lab_control.layers")) == set(OWNED) == {
        "zlc_data", "zlc_durable", "zlc_runtime", "zlc_plot", "zlc_ui",
        "zlc_pulse", "zlc_atom", "zlc_workbench",
    }
    # The manifest names the layers twice -- the entry-point group the
    # bootstrap reads and the setuptools search list the wheel is built
    # from -- and the two must be one list.
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    where = manifest["tool"]["setuptools"]["packages"]["find"]["where"]
    assert set(where) == {"."} | {
        f"packages/{name}/src" for name in entry_specs("zou_lab_control.layers")
    }
    assert set(entry_specs("zou_lab_control.commands")) == {
        "capture", "check", "device_manager", "evidence", "figure_viewer", "fpga",
        "pulse_editor", "pulse_server", "slm_server", "task_console",
        "warm_numba",
    }
    assert set(entry_specs("zou_lab_control.evidence")) == {
        "software", "gui_offscreen", "virtual_vertical", "notebook_offline",
        "real_screen", "hardware",
    }
    from importlib import import_module

    for spec in entry_specs("zou_lab_control.commands").values():
        module_name, attribute = spec.split(":", 1)
        assert callable(getattr(import_module(module_name), attribute))


def test_a_command_that_takes_no_arguments_refuses_them(monkeypatch, capsys) -> None:
    """Arguments a command cannot see are a usage error, not silence.

    ``zlc check --not-an-option`` ran the environment check and reported
    success: the dispatcher dropped whatever an argv-less entry could not
    take, so a typo was indistinguishable from the command it meant.
    """

    import sys
    import types

    from zou_lab_control import __main__ as product_entry

    calls: list[str] = []
    module = types.ModuleType("zlc_test_argvless_command")
    module.main = lambda: calls.append("ran") or 0
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        product_entry,
        "entry_specs",
        lambda group: (
            {"quiet": f"{module.__name__}:main"}
            if group == product_entry._COMMAND_GROUP
            else {}
        ),
    )

    assert product_entry.main(["quiet", "--not-an-option"]) == 2
    assert calls == []
    output = capsys.readouterr().out
    assert "quiet takes no arguments" in output and "--not-an-option" in output

    assert product_entry.main(["quiet"]) == 0
    assert calls == ["ran"]


def test_manual_evidence_never_prepares_an_automated_lane(monkeypatch, capsys) -> None:
    from zou_lab_control.__main__ import evidence

    monkeypatch.setenv("QT_QPA_PLATFORM", "windows")
    monkeypatch.setenv("MPLBACKEND", "QtAgg")
    for lane in ("real_screen", "hardware"):
        assert evidence([lane]) == 2
    assert os.environ["QT_QPA_PLATFORM"] == "windows"
    assert os.environ["MPLBACKEND"] == "QtAgg"
    assert capsys.readouterr().out.count("NOT EXECUTED") == 2


def test_automated_evidence_forces_installed_offscreen_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from zou_lab_control import __main__ as product_entry
    from zlc_workbench.tools import check_environment

    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='evidence-source'\n")
    owned = Path(product_entry.__file__).resolve()
    monkeypatch.setattr(
        product_entry,
        "distribution",
        lambda _name: SimpleNamespace(
            files=(owned,),
            locate_file=lambda item: item,
        ),
    )
    monkeypatch.setattr(check_environment, "check", lambda: [])
    monkeypatch.setattr(product_entry, "_pytest_process", lambda _names, _paths: 0)
    monkeypatch.setenv("QT_QPA_PLATFORM", "windows")
    monkeypatch.setenv("MPLBACKEND", "QtAgg")
    monkeypatch.setenv("ZLC_TEST_INSTALLED", "0")
    monkeypatch.setenv("PYTHONPATH", "source-test-path")

    assert product_entry.evidence(
        ["virtual_vertical", "--repo", str(repo)]
    ) == 0
    assert os.environ["ZLC_TEST_INSTALLED"] == "1"
    assert os.environ["PYTHONPATH"] == ""
    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"
    assert os.environ["MPLBACKEND"] == "Agg"
