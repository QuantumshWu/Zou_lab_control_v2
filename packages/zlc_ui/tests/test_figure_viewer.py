from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO_ROOT = ROOT.parents[1]


def _run_qt(code: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        ""
        if environment.get("ZLC_TEST_INSTALLED") == "1"
        else os.pathsep.join((str(REPO_ROOT), str(SRC)))
    )
    environment["QT_QPA_PLATFORM"] = "offscreen"
    verified = """
import zou_lab_control
import zlc_ui.figure_viewer.view as tested_module
print(zou_lab_control.__file__)
print(tested_module.__file__)
""" + code
    completed = subprocess.run(
        [sys.executable, "-c", verified], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_figure_viewer_mount_reconcile_and_open_intent() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_ui.figure_viewer import FigureViewerHandle, FigureViewerView
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(["figure-test"])
view = FigureViewerView(); view.set_info((("Summary", (("Name", "fake"),)), ("Flow", ())))
assert view.info_pane.path_edit._filter == "Saved figure archives (*.npz)"
view.set_panel_sizes(('2x2',), '2x2')
view.set_grid_cell_kinds(('curve', 'image', 'histogram'))
view.set_panel_kinds((('image', 'Image'), ('curve', 'Curve')))
view.add_panel('saved-panel-1', 'Camera frame')
view.set_panel_datasets('saved-panel-1', (('data', 'Camera frame'),), 'data')
state = {
    'signal': 'data', 'kind': 'image', 'cell_kind': '', 'size': '2x2',
    'interval_ms': 400, 'title': 'Camera frame', 'semantic': {}, 'display': {},
    'fit': {}, 'overlay_signal': '',
}
surface = {'semantic': (), 'display': (), 'fit': (), 'data_structure': (), 'data_scope': ()}
view.set_panel_projection('saved-panel-1', state, surface)
first = QtWidgets.QLabel("first"); second = QtWidgets.QLabel("second")
view.set_panel_surface('saved-panel-1', first); view.set_panel_surface('saved-panel-1', second)
assert view._cards['saved-panel-1'].surface is second and first.parentWidget() is None
view.resize(1200, 700); view.show(); app.processEvents()
assert view._dataset_bar.height() == view._dataset_bar.sizeHint().height()
assert view.board._cards['saved-panel-1'] is view._cards['saved-panel-1']
assert view.scroll.isVisible() and not view._placeholder.isVisible()

# Showing which file is open must not re-ask for it to be opened.
committed = []; view.path_committed.connect(committed.append)
view.set_path("D:/data/2026_08_05/run.npz")
assert view.info_pane.path_edit.text() == "D:/data/2026_08_05/run.npz"
assert committed == []

handle = FigureViewerHandle(None, view)
tree = (("fit @3", (("roi @2", (("camera @1", ()),)),)),)
handle.set_lineage_tree(tree)
assert view._lineage_tree == tree
flow = view.info_pane._tree_tabs["Flow"]
assert flow.topLevelItemCount() == 1
assert flow.topLevelItem(0).text(0) == "fit @3"
assert flow.topLevelItem(0).child(0).text(0) == "roi @2"
assert flow.topLevelItem(0).child(0).child(0).text(0) == "camera @1"

projection = {
    'state': state,
    'parameter_surface': surface,
    'signal_options': (('this archive', (('Camera frame', 'data'),)),),
    'overlay_signal_options': (),
    'live': False,
}
view.open_panel_editor(
    'saved-panel-1', projection, 'Edit · Camera frame'
)
editor = view._editors['saved-panel-1']
assert view.tabs.currentWidget() is editor
assert 'interval_ms' not in editor.panel_form.spec.keys
assert editor.snapshot_group.isHidden()
assert editor.interaction_group.isHidden()
assert editor.producer_group.isHidden()
assert editor.save_group.isHidden()
"""
    )


def test_figure_viewer_reuses_task_console_permanent_navigation_tabs() -> None:
    _run_qt(
        """
from zlc_ui.console import TaskConsoleView
from zlc_ui.figure_viewer import FigureViewerView
from zlc_ui.fluent import FluentTabWidget
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['shared-tabs'])
figure = FigureViewerView()
console = TaskConsoleView()
assert type(figure.info_pane.info_tabs) is FluentTabWidget
assert type(console.tabs) is FluentTabWidget
assert type(figure.info_pane.info_tabs) is type(console.tabs)
assert [figure.info_pane.info_tabs.tabText(i) for i in range(figure.info_pane.info_tabs.count())] == [
    'Plot', 'Measurement', 'Device', 'Flow', 'Raw'
]
assert [console.tabs.tabText(i) for i in range(console.tabs.count())] == ['Monitor', 'Logic']
"""
    )


def test_figure_viewer_demo_smoke() -> None:
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    command = (
        "import runpy, sys, zou_lab_control, zlc_ui; "
        "print(zou_lab_control.__file__); print(zlc_ui.__file__); "
        "sys.argv=['demo_figure_viewer.py', '--once']; "
        "runpy.run_path('examples/demo_figure_viewer.py', run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    # What the demo shows now is a HOST filling the page through the handle;
    # the File field's own intent is checked against the widget above, which is
    # where a widget may be poked at all.
    assert "filled: 2 datasets" in completed.stdout


def test_the_info_pane_width_is_stable_across_loaded_content() -> None:
    """Loading metadata must not move the Viewer split or resize the window."""

    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.figure_viewer import FigureViewerView
app = ensure_qt_app(['pane-width'])
view = FigureViewerView(); view.resize(1200, 700); view.show(); app.processEvents()
pane = view.info_pane
bar = pane.info_tabs.tabBar()

assert bar.count() == 5
assert not pane.info_tabs.cornerWidget().isVisible(), 'the tab bar overflowed its own pane'
assert sum(bar.tabRect(i).width() for i in range(bar.count())) >= bar.natural_width(), (
    'a tab title was elided to fit'
)

long_label = 'required_external_trigger_interval_seconds'
before = pane.width()
pane.set_tabs((('Device', ((long_label, '0.02'),)),))
assert pane.width() == before, 'loaded labels changed the fixed Viewer split'
"""
    )
