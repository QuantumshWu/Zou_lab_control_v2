"""Exhaustive fate audit: every plot kind x every axis x every offered fate.

For each combination it records what the editor OFFERS and what applying
that offer actually PRODUCES -- including what happened to the axes the
operator did not touch.
"""
import glob
import sys
from dataclasses import replace

ROOT = r"C:\Users\eadri\WorkCode\zlc_v2_h3d"
for p in sorted(glob.glob(ROOT + r"\packages\*\src")):
    sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg", force=True)
import numpy as np

from zlc_data import (
    AxisId, AxisSpec, COMPONENT, DatasetSchema, PRIMARY_INDEX, PointColumn,
    PointTable, REPEAT, SCAN_POINT, SITE, ValidityContract, ValueSchema,
)
from zlc_plot import PlotKind
from zlc_plot.semantics import (
    FATE_PREFIX, SemanticVacancy, describe_semantics, fate_field_name,
    is_scope_fate, scope_coordinate_from_fate, updated_spec,
)
from zlc_plot.specs import semantic_spec
from zlc_plot._kinds import default_spec
from zlc_plot._kinds import HANDLERS


def survival_schema():
    repeat = AxisSpec(AxisId("sv.repeat"), "repeat", REPEAT, 4, tuple(range(4)))
    pair = AxisSpec(AxisId("sv.pair"), "pair", COMPONENT, 3,
                    coordinate_labels=("0-1", "0-2", "1-2"))
    site = AxisSpec(AxisId("sv.site"), "site", SITE, 5, tuple(range(5)))
    cell = ValueSchema((pair, site),
                       ValidityContract.components(pair.axis_id, site.axis_id),
                       np.dtype("<f8"), "1")
    return DatasetSchema(repeat, PointTable(1, ()), None, cell)


def camera_schema():
    repeat = AxisSpec(AxisId("cm.repeat"), "repeat", REPEAT, 2, (0, 1))
    ys = AxisSpec(AxisId("cm.spatial-y"), "spatial-y", SITE, 96, tuple(range(96)))
    xs = AxisSpec(AxisId("cm.spatial-x"), "spatial-x", SITE, 128, tuple(range(128)))
    cell = ValueSchema((ys, xs),
                       ValidityContract.components(ys.axis_id, xs.axis_id),
                       np.dtype("<f8"), "1")
    frame = PointColumn(AxisId("cm.frame"), "frame", SCAN_POINT, PointColumn.NUMERIC,
                        (0.0, 1.0, 2.0))
    return DatasetSchema(repeat, PointTable(3, (frame,)), None, cell)


def scan_schema():
    repeat = AxisSpec(AxisId("sc.repeat"), "repeat", REPEAT, 3, (0, 1, 2))
    site = AxisSpec(AxisId("sc.site"), "site", SITE, 6, tuple(range(6)))
    cell = ValueSchema((site,), ValidityContract.components(site.axis_id),
                       np.dtype("<f8"), "1")
    fx = PointColumn(AxisId("sc.field.x"), "field.x", SCAN_POINT,
                     PointColumn.NUMERIC, tuple(float(v) for v in range(5)) * 4)
    fy = PointColumn(AxisId("sc.field.y"), "field.y", SCAN_POINT,
                     PointColumn.NUMERIC,
                     tuple(float(v) for v in range(4) for _ in range(5)))
    return DatasetSchema(repeat, PointTable(20, (fx, fy)), None, cell)


def indexed_schema():
    repeat = AxisSpec(AxisId("oc.repeat"), "repeat", REPEAT, 1, (0,))
    site = AxisSpec(AxisId("oc.site"), "site", SITE, 4, tuple(range(4)))
    cell = ValueSchema((site,), ValidityContract.components(site.axis_id),
                       np.dtype("<f8"), "1")
    shots = PointColumn(AxisId("zlc_data.primary-index"), "source index",
                        PRIMARY_INDEX, PointColumn.NUMERIC, tuple(range(6)))
    return DatasetSchema(repeat, PointTable(6, (shots,)), None, cell)


SCHEMAS = {
    "survival (repeat x [pair x site])": survival_schema(),
    "camera (repeat x frame x [y x x])": camera_schema(),
    "scan (repeat x [field.x,field.y] x site)": scan_schema(),
    "indexed (shots x site)": indexed_schema(),
}


def fate_table(schema, spec):
    d = describe_semantics(schema, spec)
    return {name: d.field(name).value for _ref, name in d.fate_rows}


def _short(name):
    return name.replace("fate:", "").replace("point_coordinate:", "").replace("data:", "")


def show(value):
    if is_scope_fate(value):
        return f"={scope_coordinate_from_fate(value)}"
    return str(value)


for schema_label, schema in SCHEMAS.items():
    print("=" * 78)
    print("SCHEMA:", schema_label)
    kinds = [h.kind for h in HANDLERS if h.admits(schema)]
    print("  admissible kinds:", [k.value for k in kinds])
    for kind in kinds:
        if kind is PlotKind.PULSE_TIMELINE:
            continue
        spec = default_spec(schema, kind)
        if spec is None:
            print(f"\n  [{kind.value}] NO DEFAULT SPEC")
            continue
        d = describe_semantics(schema, spec)
        declared = [f.name for f in d.fields]
        print(f"\n  [{kind.value}] default = {type(semantic_spec(spec)).__name__}"
              f" | rows={len(d.fate_rows)}"
              f" | reduction offered: {'reduction' in declared}")
        base = fate_table(schema, spec)
        for _ref, name in d.fate_rows:
            field = d.field(name)
            roles = [show(v) for v, _l in field.choices if not is_scope_fate(v)]
            pins = sum(1 for v, _l in field.choices if is_scope_fate(v))
            outcomes = {}
            for value, _label in field.choices:
                if value == field.value:
                    continue
                tag = "scope" if is_scope_fate(value) else show(value)
                try:
                    candidate = updated_spec(schema, spec, name, value)
                except SemanticVacancy as vac:
                    verdict = f"VACANCY({vac.role})"
                except Exception as error:
                    verdict = f"ERROR {type(error).__name__}"
                else:
                    after = fate_table(schema, candidate)
                    moved = {
                        _short(key): (show(base[key]), show(after[key]))
                        for key in base
                        if key in after and base[key] != after[key] and key != name
                    }
                    verdict = f"also {moved}" if moved else "clean"
                outcomes.setdefault((tag, verdict), 0)
                outcomes[(tag, verdict)] += 1
            summary = "; ".join(
                f"{tag}->{verdict}" + (f" x{n}" if n > 1 else "")
                for (tag, verdict), n in outcomes.items()
            )
            print(f"    {field.label:<12} now={show(field.value):<8}"
                  f" roles={roles} pins={pins}")
            if summary:
                print(f"       {summary}")
