"""An archive explains itself, and outlives the code that wrote it.

Nothing in the six packages persisted anything: not a frame, not a device
setting, not the pulse that drove the run.  Whatever was taken at the bench was
lost when the process exited.

The two properties that matter beyond "it round-trips":

* a reader needs numpy and the standard library, nothing of ours.  An archive
  whose metadata is pickled Python objects can only be opened by the code that
  wrote it and stops opening the day a class is renamed.
* saving twice in one day does not overwrite the morning.
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import Future
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from zlc_atom.devices.camera.contract import CameraFrameRecord
from zlc_atom.nodes.calibration import (
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
)
from zlc_atom.nodes.calibration.outputs import calibration_final_outputs
from zlc_atom.nodes import discover_logic_nodes
from zlc_data import StreamGenerationId
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImageFrame,
    ImagePlot,
)
from zlc_runtime import SignalPublication, SignalValue
from zlc_runtime.streams import EventRef, StreamId

from zlc_workbench.archive import FIGURE_SCHEMA, read_archive, write_figure
from zlc_workbench.calibration_report import (
    create_calibration_report_surface,
    export_calibration_report,
    load_calibration_report_page,
)
from zlc_workbench.task_reports import default_task_report_registry
from zlc_workbench.prepared_panel import PreparedPanelSurface
from zlc_workbench.panel_state import PanelState


def _sections() -> dict:
    return {
        "provenance": {
            "node": "cm",
            "captured_at": "2026-08-05 12:00:00",
            "devices": {"camera": {"exposure_seconds": 0.02, "gain": 1.0}},
            "acquisition_parameters": {"repeat": 3, "frames_per_cycle": 3},
        },
        "pulse": {"name": "calibration", "camera_windows": 3},
    }


def test_a_figure_lands_under_the_day_it_was_taken(tmp_path) -> None:
    path = write_figure(
        tmp_path,
        "MOT loading",
        arrays={"frames": np.zeros((2, 3, 4))},
        sections=_sections(),
        when=date(2026, 8, 5),
    )
    assert path.parent.name == "2026_08_05"
    assert path.parent.parent == tmp_path
    assert path.name == "MOT-loading.npz"


def test_saving_twice_in_one_day_keeps_both(tmp_path) -> None:
    first = write_figure(tmp_path, "scan", arrays={"y": np.arange(3.0)}, sections={}, when=date(2026, 8, 5))
    second = write_figure(tmp_path, "scan", arrays={"y": np.arange(3.0) + 1}, sections={}, when=date(2026, 8, 5))
    assert first != second
    assert first.exists() and second.exists()
    assert np.asarray(read_archive(first)[1]["y"])[0] == 0.0


def test_the_record_comes_back_whole(tmp_path) -> None:
    frames = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    path = write_figure(tmp_path, "shot", arrays={"frames": frames}, sections=_sections())

    info, arrays = read_archive(path)
    assert info["schema"] == FIGURE_SCHEMA
    assert info["name"] == "shot"
    np.testing.assert_array_equal(arrays["frames"], frames)

    provenance = info["sections"]["provenance"]
    assert provenance["devices"]["camera"]["exposure_seconds"] == 0.02
    assert provenance["acquisition_parameters"]["repeat"] == 3
    assert info["sections"]["pulse"]["camera_windows"] == 3


def test_an_archive_opens_with_numpy_alone(tmp_path) -> None:
    """The durability property: a reader must not need any zlc_* package.

    Run in a fresh interpreter that imports nothing of ours, with allow_pickle
    left at its safe default, exactly as a stranger would open it years later.
    """

    path = write_figure(
        tmp_path,
        "outlives-us",
        arrays={"frames": np.arange(6.0).reshape(2, 3)},
        sections=_sections(),
    )

    reader = (
        "import json, sys, numpy as np\n"
        f"archive = np.load(r'{path}')\n"
        "info = json.loads(str(archive['info']))\n"
        "assert info['sections']['provenance']['devices']['camera']['gain'] == 1.0\n"
        "assert archive['frames'].shape == (2, 3)\n"
        "assert not [m for m in sys.modules if m.startswith('zlc_')]\n"
        "print('readable')\n"
    )
    completed = subprocess.run([sys.executable, "-c", reader], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "readable" in completed.stdout


def test_arrays_may_not_hide_inside_the_info_document(tmp_path) -> None:
    """An array in the JSON would be unreadable and unmeasurable; say so."""

    with pytest.raises(TypeError, match="arrays belong beside info"):
        write_figure(
            tmp_path,
            "wrong",
            arrays={"y": np.arange(3.0)},
            sections={"panel": {"data": np.arange(3.0)}},
        )


def test_an_interrupted_save_does_not_destroy_the_previous_archive(tmp_path, monkeypatch) -> None:
    good = write_figure(tmp_path, "keep", arrays={"y": np.ones(3)}, sections={}, when=date(2026, 8, 5))
    before = good.read_bytes()

    import zlc_workbench.archive as archive_module

    def explode(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(archive_module, "atomic_write_bytes", explode)
    with pytest.raises(OSError):
        write_figure(tmp_path, "keep", arrays={"y": np.zeros(3)}, sections={}, when=date(2026, 8, 5))

    assert good.read_bytes() == before


def test_an_array_in_device_state_is_refused_with_the_reason() -> None:
    """It used to be stringified before the refusal could fire.

    Provenance ran its own plainifier first, which turned anything it did not
    recognise into str(value) -- so a device reporting an array had it silently
    truncated by numpy's repr, and the archive's message telling the author to
    pass it as an array instead could never be reached.
    """

    import numpy as np
    import pytest

    from zlc_workbench.archive import _jsonable

    with pytest.raises(TypeError, match="arrays belong beside info"):
        _jsonable({"camera": {"lookup": np.arange(4)}})


def _calibration_outputs():
    site_ids = ("site_0000", "site_0001", "site_0002")
    sample = np.linspace(-3.0, 3.0, 80)
    readout = np.column_stack(
        (
            np.where(sample < 0.0, sample - 2.0, sample + 2.0),
            np.where(sample < 0.0, sample - 1.0, sample + 3.0),
            np.where(sample < 0.0, sample - 3.0, sample + 1.0),
        )
    )
    calibration = TrapCalibration(
        SiteMap(
            site_ids,
            np.asarray(((2.0, 3.0), (5.0, 4.0), (8.0, 6.0))),
            np.asarray((True, True, False)),
            np.asarray((1.0, 0.9, 0.2)),
        ),
        (
            ReadoutModel(
                site_ids,
                np.asarray((0.0, 1.0, 2.0)),
                np.asarray((True, True, False)),
                np.asarray((0.99, 0.96, 0.2)),
            ),
        ),
        ReadoutModelKind.BOX,
        FrameContract((8, 10)),
    )
    image = np.zeros((8, 10), dtype=np.uint16)
    cycle = tuple(
        CameraFrameRecord(image, index, host_received_at_ns=index + 1)
        for index in range(3)
    )
    return calibration_final_outputs(
        calibration=calibration,
        capture_cycles=(cycle,) * len(readout),
        report={
            "reference_average": np.arange(80, dtype=float).reshape((8, 10)),
            "labels_valid": np.ones(readout.shape, dtype=bool),
            "models": {"box": {"short_signals": readout}},
        },
        generation="report-test",
        run_record={"task": "calibration"},
    )


def _calibration_publication(outputs) -> SignalPublication:
    first = next(iter(outputs.values()))
    run_record = dict(first.run_record or {})
    signals = {
        f"@logic/calibration/{name}": SignalValue(
            f"@logic/calibration/{name}",
            output.snapshot,
            None,
            False,
            run_record,
        )
        for name, output in outputs.items()
    }
    generation = first.snapshot.ref.stream_generation
    assert isinstance(generation, StreamGenerationId)
    return SignalPublication(
        EventRef(StreamId("calibration"), generation, 7),
        signals,
        object(),
        run_record=run_record,
    )


def test_calibration_report_is_three_typed_shared_plot_pages(tmp_path) -> None:
    publication = _calibration_publication(_calibration_outputs())
    descriptor = next(
        value
        for value in discover_logic_nodes()
        if value.api_name == "calibration"
    )
    report = default_task_report_registry().build(
        descriptor.task_reports[0],
        descriptor,
        publication,
    )
    assert report.adapter_id == "calibration.report.v1"
    assert report.publication is publication
    pages = report.pages

    assert tuple(page.key for page in pages) == (
        "site_map",
        "fidelity",
        "distribution",
    )
    assert isinstance(pages[0].plot_input, ImageFrame)
    assert isinstance(pages[0].spec, ImagePlot)
    assert pages[0].plot_input.overlay.point_ids == (
        "site_0000",
        "site_0001",
        "site_0002",
    )
    assert tuple(status.value for status in pages[0].plot_input.overlay.statuses) == (
        "unknown",
        "unknown",
        "invalid",
    )
    assert isinstance(pages[1].spec, CurvePlot)
    assert pages[1].spec.x == AxisRef.point_rows()
    assert pages[1].snapshot.block.schema.point_table.columns[0].values == (
        "site_0000",
        "site_0001",
        "site_0002",
    )
    assert isinstance(pages[2].spec, FacetGridPlot)
    assert isinstance(pages[2].spec.cell, HistogramPlot)
    assert pages[2].facet_thresholds == (0.0, 1.0, None)
    assert pages[2].fit_model == "bimodal_gaussian"

    files = export_calibration_report(tmp_path / "report", pages)
    assert tuple(path.name for path in files.images) == (
        "site_map.png",
        "fidelity.png",
        "distribution.png",
    )
    assert tuple(path.name for path in files.archives) == (
        "site_map.npz",
        "fidelity.npz",
        "distribution.npz",
    )
    assert all(path.stat().st_size > 0 for path in (*files.images, *files.archives))
    for archive in files.archives:
        info, arrays = read_archive(archive)
        assert info["sections"]["calibration_report"]["page"] == archive.stem
        assert "data" in info["sections"]["dataset"]
        assert arrays["data"].size

    loaded = tuple(load_calibration_report_page(path) for path in files.archives)
    assert tuple(page.key for page in loaded) == (
        "site_map",
        "fidelity",
        "distribution",
    )
    assert isinstance(loaded[0].plot_input, ImageFrame)
    site_ids = (
        "site_0000",
        "site_0001",
        "site_0002",
    )
    assert loaded[0].plot_input.overlay.point_ids == site_ids
    assert loaded[1].snapshot.block.schema.point_table.columns[0].values == site_ids
    assert loaded[2].facet_thresholds == (0.0, 1.0, None)
    for page in loaded:
        surface = create_calibration_report_surface(page)
        try:
            surface.wait(timeout=10.0)
            assert surface.host.front is not None
        finally:
            surface.close()


def test_prepared_panel_poll_reads_callback_cache_without_resolving_futures() -> None:
    """GUI polling observes completion; only the future callback resolves it."""

    class ProbeFuture(Future):
        def __init__(self) -> None:
            super().__init__()
            self.result_calls = 0

        def result(self, timeout=None):
            self.result_calls += 1
            return super().result(timeout=timeout)

    descriptions = tuple(ProbeFuture() for _ in range(3))
    surface = PreparedPanelSurface(
        SimpleNamespace(),
        PanelState("signal", "image", "2x2", 400, "Prepared"),
        SimpleNamespace(),
        (),
        descriptions,
    )
    assert surface.descriptions_ready is False
    assert surface.failure() is None
    assert tuple(item.result_calls for item in descriptions) == (0, 0, 0)

    expected = (object(), object(), object())
    for future, value in zip(descriptions, expected, strict=True):
        future.set_result(SimpleNamespace(value=value))
    assert surface.descriptions_ready is True
    assert surface.description_values() == expected
    assert tuple(item.result_calls for item in descriptions) == (1, 1, 1)
