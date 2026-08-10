"""Guards for zlc_data ownership, import purity, and installation identity."""

from __future__ import annotations

import json
import importlib
import os
from importlib import metadata
from pathlib import Path
import re
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PACKAGE = (ROOT / "src" / "zlc_data").resolve()
#: Raised deliberately for the names siblings were ALREADY using through
#: submodules.  A sibling reaching past this list is the same surface with the
#: declaration skipped -- nobody can then see how wide it really is -- so what
#: zlc_runtime, zlc_plot and zlc_atom depend on is declared here.
MAX_PUBLIC_NAMES = 70
EXPECTED_PUBLIC_API = frozenset(
    {
        # Returned from the withdrawn lists the moment a sibling needed them
        # for something this facade did not otherwise offer: the selection
        # vocabulary, because two packages publish and consume selections
        # across a wire and were matching an enum by string VALUE rather than
        # by identity; and the snapshot manifest pair, because a figure
        # archive IS a snapshot plus its manifest, so whoever writes one needs
        # both and was reaching into the submodule to get them.
        "IndexSelection",
        "SelectionChange",
        "is_intrinsically_immutable_array",
        "resolve_selection_indices",
        "snapshot_from_manifest",
        "snapshot_manifest",
        "canonical_text",
        "digest_text",
        "exact_mapping",
        "expand_component_validity",
        "finite_real",
        "integer",
        "nonnegative_integer",
        "positive_integer",
        "AxisId",
        "AxisRoleId",
        "AxisSourceRef",
        "AxisSpec",
        "BlockId",
        "COMPONENT",
        "CellValidity",
        "ComponentValidity",
        "CoordinateFrameId",
        "DataBlock",
        "DatasetComponentValidity",
        "DatasetRevision",
        "DatasetRevisionRef",
        "DatasetSchema",
        "GridTopology",
        "INVALID",
        "Invalid",
        "OwnedSnapshot",
        "PointColumn",
        "PointTable",
        "READOUT_EVENT",
        "REPEAT",
        "SCAN_POINT",
        "SITE",
        "SPATIAL_X",
        "SPATIAL_Y",
        "Selection",
        "StreamGenerationId",
        "VALID",
        "Valid",
        "ValidityContract",
        "ValidityMode",
        "Value",
        "ValuePayloadContract",
        "ValueSchema",
        "NPZFormatError",
        "__version__",
        "compact_dataset_validity",
        "expand_dataset_validity",
        "expand_snapshot_validity",
        "load_npz",
        "materialize_derived_dataset",
        "materialize_value_dataset",
        "materialize_scalar_dataset",
        "owned_snapshot_from_arrays",
        "save_npz",
    }
)

#: Withdrawn from the facade and reachable only through their module.
#:
#: A name comes BACK the moment a sibling package needs it: reaching into a
#: submodule for something this facade does not name is the same surface with
#: the declaration skipped, and nobody can then see how wide it really is.
#: expand_component_validity returned for exactly that reason -- zlc_runtime
#: was already using it.
WITHDRAWN_PUBLIC_API = {
    "zlc_data.axis": {
        "CoordinateScalar",
        "HISTOGRAM_BIN",
        "SCALAR",
        "SCALAR_AXIS",
        "SPECTRAL",
    },
    "zlc_data.schema": {"ResolvedPointRows"},
    "zlc_data.selection": {
        "CoordinateRangeSelection",
        "IndexRangeSelection",
    },
    "zlc_data.value": {
        "dataset_cell_value",
        "expand_value_validity",
    },
}


def _import_report() -> dict[str, object]:
    # Through the one entry, the way every consumer of this checkout arrives.
    # A bare subprocess finds whatever is INSTALLED, and eight of these layers
    # are also installed from their standalone repositories -- so the probe
    # would faithfully report a copy nobody is editing.  The snapshot is taken
    # after the entry, so what it measures is still "what does importing
    # zlc_data pull in", and nothing else.
    code = """
import importlib
import json
import pkgutil
import sys

sys.path.insert(0, ENTRY)
import zou_lab_control_v2

before = set(sys.modules)
import zlc_data
discovered = sorted(
    module.name
    for module in pkgutil.walk_packages(
        zlc_data.__path__, zlc_data.__name__ + "."
    )
)
for module_name in discovered:
    importlib.import_module(module_name)
new_modules = sorted(set(sys.modules) - before)
print(json.dumps({
    "file": str(zlc_data.__file__),
    "version": zlc_data.__version__,
    "discovered": discovered,
    "new_modules": new_modules,
}))
"""
    entry = ROOT.parents[1]
    preamble = "ENTRY = {0!r}\n".format(str(entry))
    completed = subprocess.run(
        [sys.executable, "-c", preamble + code],
        cwd=ROOT,
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_import_is_pure_and_resolves_to_this_distribution():
    report = _import_report()
    discovered = report["discovered"]
    loaded = report["new_modules"]
    assert isinstance(discovered, list)
    assert isinstance(loaded, list)
    loaded_set = set(loaded)
    assert set(discovered) <= loaded_set
    stdlib = set(sys.stdlib_module_names)
    allowed_roots = stdlib | {"numpy", "zlc_data"}
    unexpected = sorted(
        {
            name.split(".", 1)[0]
            for name in loaded
            if name.split(".", 1)[0] not in allowed_roots
        }
    )
    assert unexpected == []

    forbidden_roots = {
        "matplotlib",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "zlc_neutral_atom",
        "zlc_plot",
        "zlc_storage",
    }
    forbidden = sorted(
        {
            name.split(".", 1)[0]
            for name in loaded
            if name.split(".", 1)[0] in forbidden_roots
            or (
                name.split(".", 1)[0].startswith("zlc_")
                and name.split(".", 1)[0] != "zlc_data"
            )
        }
    )
    assert forbidden == []
    assert Path(str(report["file"])).resolve().parent == EXPECTED_PACKAGE


def test_version_and_installation_path_are_owned_by_this_checkout():
    import zlc_data

    assert zlc_data.__version__ == "0.1.0"
    assert metadata.version("zlc-data") == zlc_data.__version__
    assert Path(zlc_data.__file__).resolve().parent == EXPECTED_PACKAGE


def test_top_level_public_api_is_the_explicit_allow_list():
    import zlc_data

    assert set(zlc_data.__all__) == EXPECTED_PUBLIC_API
    assert all(hasattr(zlc_data, name) for name in EXPECTED_PUBLIC_API)


def test_top_level_public_namespace_stays_under_the_mechanical_cap():
    import zlc_data

    public_names = {
        name
        for name in dir(zlc_data)
        if not name.startswith("_")
        and not isinstance(getattr(zlc_data, name), types.ModuleType)
    }
    assert len(public_names) <= MAX_PUBLIC_NAMES
    assert public_names == EXPECTED_PUBLIC_API - {"__version__"}


def _contract_public_api() -> set[str]:
    contract = (ROOT / "docs" / "contract.md").read_text(encoding="utf-8")
    match = re.search(
        r"## Public API allow-list\s+.*?```json\s+(.*?)\s+```",
        contract,
        flags=re.DOTALL,
    )
    assert match is not None
    names = json.loads(match.group(1))
    assert isinstance(names, list)
    assert all(isinstance(name, str) for name in names)
    return set(names)


def test_top_level_public_api_matches_the_contract_allow_list():
    import zlc_data

    assert set(zlc_data.__all__) == _contract_public_api()


def test_withdrawn_names_are_module_scoped_but_still_usable():
    import zlc_data

    for module_name, names in WITHDRAWN_PUBLIC_API.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name)
            assert not hasattr(zlc_data, name)


def test_cross_package_contract_states_unit_and_live_ownership():
    contract = (ROOT / "docs" / "contract.md").read_text(encoding="utf-8")

    assert "## 单位与 live 的归属" in contract
    assert "canonical unit" in contract
    assert "does not own a unit registry" in contract
    assert "latest-only live ingress" in contract
    assert "zlc_plot" in contract
