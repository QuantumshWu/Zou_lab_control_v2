from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_plot_and_role_axis_data_exchange_real_owned_snapshot() -> None:
    """Both distributions remain importable and exchange one real snapshot."""

    repo_root = Path(__file__).resolve().parents[1]
    role_axis_root = repo_root.parent / "zlc_data" / "src"
    role_axis_package = role_axis_root / "zlc_data" / "__init__.py"
    assert role_axis_package.is_file(), (
        "the sibling role-axis zlc-data checkout is required for the namespace guard"
    )

    probe = """
import numpy as np
import zlc_data
import zlc_plot

repeat = zlc_data.AxisSpec(zlc_data.AxisId("repeat"), "repeat", zlc_data.REPEAT, 1, (0,))
point = zlc_data.PointColumn(
    zlc_data.AxisId("x"), "x", zlc_data.SCAN_POINT,
    zlc_data.PointColumn.NUMERIC, (0.0, 1.0), "mV",
)
table = zlc_data.PointTable(2, (point,))
value_axis = zlc_data.AxisSpec(zlc_data.AxisId("value"), "value", zlc_data.COMPONENT, 1, (0,))
cell = zlc_data.ValueSchema((value_axis,), zlc_data.ValidityContract.value(), np.dtype("float64"), "V")
schema = zlc_data.DatasetSchema(repeat, table, None, cell)
snapshot = zlc_data.owned_snapshot_from_arrays(schema=schema, values=np.ones((1, 2, 1)), revision=0)
assert isinstance(snapshot, zlc_data.OwnedSnapshot)
session = zlc_plot.PlotSession(snapshot, zlc_plot.CurvePlot(zlc_plot.AxisRef.point("x")))
session.close()
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(repo_root / "src"), str(role_axis_root))
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "zlc_plot and role-axis zlc_data did not remain independently importable:\n"
        + completed.stdout
        + completed.stderr
    )
