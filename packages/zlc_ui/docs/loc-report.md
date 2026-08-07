# LOC report

Measured 2026-08-03 with PowerShell `Get-Content | Measure-Object -Line`.
These are physical Python source lines; current-package `__init__.py` files are
included, while generated `__pycache__` files, tests, examples, and docs are
excluded.

| Current package | Files | Lines | v1 corresponding source | v1 files | v1 lines |
|---|---:|---:|---|---:|---:|
| package root (`__init__`, `qt`) | 2 | 79 | new package entry point | — | — |
| `fluent` | 6 | 3,491 | `zlc_frontend/qt_widgets/{fluent,style,choice_picker,published_items,info_pane}.py` | 5 | 3,505 |
| `form` | 3 | 1,434 | `zlc_frontend/form.py` + `qt_widgets/form.py` | 2 | 1,407 |
| `board` | 2 | 156 | `zlc_frontend/board_layout.py` | 1 | 161 |
| `graph` | 4 | 685 | `zlc_frontend/{flow_graph,shape_text}.py` + `qt_widgets/flow_graph_view.py` | 3 | 704 |
| `concurrency` | 2 | 288 | `zlc_frontend/qt_widgets/owner_wake.py` | 1 | 285 |
| `console` | 6 | 534 | selected UI files in `zlc_workbench/task_console/` (`panel_card`, `panel_board`, `logic_node_row`, `published_signal_row`, `logic_node_editor`) | 5 | 1,482 |
| `device_manager` | 2 | 167 | `zlc_workbench/device_manager/window.py` | 1 | 749 |
| `pulse` | 10 | 1,729 | pure Qt/view-model seam extracted from `pulse_editor` | selected pure pieces | 4,600–4,900 (survey estimate) |
| `figure_viewer` | 2 | 96 | pure shell extracted from `figure_viewer/window.py` | 1 | 357 |
| **Total** | **39** | **8,659** | selected comparison sources | **20+** | **13,250–13,550** |

The v1 task-console files are mixed view/runtime/domain boundaries, so their
1,482 lines are a reference surface rather than a migration target.  The v1
device-manager package is 1,622 lines including `controller.py` and
`editor_session.py`; only its UI window is compared above.  Those domain-side
parts intentionally remain outside `zlc_ui`.

Reference tree (read-only):
`C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v1_claude\Zou_lab_control_v1`.

The pulse row is intentionally shorter than the survey's 4.6–4.9k pure-view
estimate: this cut transfers the public interaction contract and Qt skeleton,
while leaving the reference presenter projection, controller, archive IO, and
plot/runtime composition out of this package. The resulting 1,729 lines are
the compact acceptance seam, not a line-for-line copy of the mixed 2,892-line
schedule page.
