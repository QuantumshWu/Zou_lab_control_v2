"""Every layer and command belongs to the one installed product manifest."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from zlc_workbench.tools.check_environment import OWNED, check
from zou_lab_control import entry_specs


def test_every_package_resolves_to_its_own_product() -> None:
    problems = check()
    assert problems == [], "\n".join(problems)


def test_product_manifest_owns_all_commands_and_layers() -> None:
    assert set(entry_specs("zou_lab_control.layers")) == set(OWNED) == {
        "zlc_data", "zlc_durable", "zlc_runtime", "zlc_plot", "zlc_ui",
        "zlc_pulse", "zlc_atom", "zlc_workbench",
    }
    assert set(entry_specs("zou_lab_control.commands")) == {
        "capture", "check", "device_manager", "evidence", "figure_viewer", "fpga",
        "pulse_editor", "pulse_server", "slm_server", "task_console",
    }
    assert set(entry_specs("zou_lab_control.evidence")) == {
        "software", "gui_offscreen", "virtual_vertical", "notebook_offline",
        "real_screen", "hardware",
    }
    from importlib import import_module

    for spec in entry_specs("zou_lab_control.commands").values():
        module_name, attribute = spec.split(":", 1)
        assert callable(getattr(import_module(module_name), attribute))


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
