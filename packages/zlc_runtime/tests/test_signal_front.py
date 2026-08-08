from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValueSchema,
)

from zlc_runtime.dataset import MonitorCoverage
from zlc_runtime.front import build_front
from zlc_runtime.plane import (
    DerivedSignalOutput,
    SignalDataPlane,
    SignalPublication,
    SignalValue,
)
from zlc_runtime.streams import EventRef, StreamId
from zlc_runtime.dataset_output import LiveDatasetOutput, DatasetOutputDeclaration


def _output(name: str, revision: int) -> LiveDatasetOutput:
    repeat = AxisSpec(AxisId(f"{name}.repeat"), "repeat", REPEAT, 1, (0,))
    point = AxisSpec(AxisId(f"{name}.point"), "point", SCAN_POINT, 1, (0,))
    schema = DatasetSchema(
        repeat,
        PointTable(
            1,
            (
                PointColumn(
                    point.axis_id,
                    point.name,
                    point.role,
                    PointColumn.NUMERIC,
                    point.coordinates,
                ),
            ),
        ),
        None,
        ValueSchema.scalar(np.dtype("float64"), "count"),
    )
    block = DataBlock(
        BlockId(f"{name}-{revision}"),
        DatasetRevision(revision),
        np.asarray([[[float(revision)]]], dtype=np.float64),
        CellValidity(np.ones((1, 1), dtype=np.bool_)),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId(f"{name}-generation")),
        block,
    )
    return LiveDatasetOutput(
        DatasetOutputDeclaration(name, f"test.{name}"),
        snapshot,
        MonitorCoverage(1, 1),
    )


def _publication(
    stream: str,
    generation: str,
    sequence: int,
    name: str,
    parents: tuple[SignalPublication, ...] = (),
) -> SignalPublication:
    output = _output(name.replace("/", "_"), sequence)
    value = SignalValue(name, output.snapshot, output.coverage, transient=True)
    return SignalPublication(
        EventRef(StreamId(stream), StreamGenerationId(generation), sequence),
        {name: value},
        object(),
        tuple(parent.event_ref for parent in parents),
    )


def _state(
    owner: str,
    generation: str,
    kind: str,
    names: tuple[str, ...],
    publication: SignalPublication | None,
    source_name: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        owner_id=owner,
        generation=StreamGenerationId(generation),
        kind=kind,
        output_names=names,
        source_name=source_name,
        publication=publication,
        failure=None,
        terminal=False,
        retired=False,
    )


def test_build_front_is_transitive_and_falls_back_as_one_family() -> None:
    root = _publication("camera", "g1", 1, "camera/frame")
    roi = _publication("roi", "g2", 1, "roi/value", (root,))
    fit = _publication("fit", "g3", 1, "fit/value", (roi,))
    parents = {root: (), roi: (root,), fit: (roi,)}
    states = [
        _state("camera", "g1", "producer", ("camera/frame",), root),
        _state("roi", "g2", "continuous", ("roi/value",), roi, "camera/frame"),
        _state("fit", "g3", "continuous", ("fit/value",), fit, "roi/value"),
    ]

    first = build_front(states, {"camera/frame", "roi/value", "fit/value"}, None, parents.__getitem__)
    assert first.names() == ("camera/frame", "roi/value", "fit/value")
    assert first.continuous_group("fit/value") == frozenset(
        {"camera/frame", "roi/value", "fit/value"}
    )

    root2 = _publication("camera", "g1", 2, "camera/frame")
    roi2 = _publication("roi", "g2", 2, "roi/value", (root2,))
    states[0].publication = root2
    states[1].publication = roi2
    states[2].publication = None
    parents.update({root2: (), roi2: (root2,)})
    held = build_front(
        states,
        {"camera/frame", "roi/value", "fit/value"},
        first,
        parents.__getitem__,
    )
    assert [held.publication(name).event_ref.sequence for name in held.names()] == [1, 1, 1]

    fit2 = _publication("fit", "g3", 2, "fit/value", (roi2,))
    states[2].publication = fit2
    parents[fit2] = (roi2,)
    recovered = build_front(
        states,
        {"camera/frame", "roi/value", "fit/value"},
        held,
        parents.__getitem__,
    )
    assert [recovered.publication(name).event_ref.sequence for name in recovered.names()] == [2, 2, 2]
    roots = {
        next(iter(_root_refs(recovered.publication(name), parents)))
        for name in recovered.names()
    }
    assert len(roots) == 1


def _root_refs(
    publication: SignalPublication,
    parents: dict[SignalPublication, tuple[SignalPublication, ...]],
) -> set[EventRef]:
    pending = [publication]
    roots: set[EventRef] = set()
    while pending:
        current = pending.pop()
        direct = parents[current]
        if direct:
            pending.extend(direct)
        else:
            roots.add(current.event_ref)
    return roots


def test_plane_front_keeps_weak_parent_payload_alive() -> None:
    plane = SignalDataPlane()
    output = _output("frame", 1)
    state = {"frame": output}
    node = SimpleNamespace(
        instance_id="camera",
        dataset_output_declarations=(output.declaration,),
        signal_key=lambda name: f"camera/{name}",
    )
    slot = SimpleNamespace(
        freeze_live_outputs=lambda: dict(state),
        close=lambda: None,
        notification_failure=None,
    )
    try:
        plane.reserve(node)
        plane.attach(node, slot)
        plane.set_front_signals({"camera/frame", "roi/value", "fit/value"})
        plane.mark_changed(node, slot)
        first_front = plane.freeze()
        root = first_front.publication("camera/frame")
        assert root is not None

        roi_generation = plane.bind_continuous_derived(
            "roi",
            source_name="camera/frame",
            expected_source_generation=root.event_ref.generation,
            output_names=("roi/value",),
        )
        assert plane.publish_continuous_derived(
            "roi",
            roi_generation,
            (root,),
            {"roi/value": DerivedSignalOutput(_output("roi", 1).snapshot)},
        )
        roi = plane.latest_publication("roi/value")
        assert roi is not None
        fit_generation = plane.bind_continuous_derived(
            "fit",
            source_name="roi/value",
            expected_source_generation=roi.event_ref.generation,
            output_names=("fit/value",),
        )
        assert plane.publish_continuous_derived(
            "fit",
            fit_generation,
            (roi,),
            {"fit/value": DerivedSignalOutput(_output("fit", 1).snapshot)},
        )
        first_front = plane.freeze()
        first_fit = first_front.publication("fit/value")
        assert first_fit is not None
        root_reference = weakref.ref(root)
        roi_reference = weakref.ref(roi)
        fit_reference = weakref.ref(first_fit)
        assert first_front.continuous_group("fit/value") == frozenset(
            {"camera/frame", "roi/value", "fit/value"}
        )

        state["frame"] = _output("frame", 2)
        plane.mark_changed(node, slot)
        held_front = plane.freeze()
        root2 = plane.latest_publication("camera/frame")
        assert root2 is not None
        assert held_front.value("camera/frame").snapshot.ref.revision.value == 1
        assert plane.publish_continuous_derived(
            "roi",
            roi_generation,
            (root2,),
            {"roi/value": DerivedSignalOutput(_output("roi", 2).snapshot)},
        )
        roi2 = plane.latest_publication("roi/value")
        assert roi2 is not None
        assert plane.publish_continuous_derived(
            "fit",
            fit_generation,
            (roi2,),
            {"fit/value": DerivedSignalOutput(_output("fit", 2).snapshot)},
        )
        second_front = plane.freeze()
        assert [
            second_front.publication(name).event_ref.sequence
            for name in second_front.names()
        ] == [2, 2, 2]

        # The live states and current front now retain only generation 2.  Keep
        # the old leaf as the sole deliberate strong reference while asking the
        # plane to resolve its exact parent chain.
        del held_front
        del first_front
        del root
        del roi
        gc.collect()
        assert root_reference() is not None
        assert roi_reference() is not None
        assert fit_reference() is first_fit
        old_roi = plane.direct_parent_publications(first_fit)[0]
        old_root = plane.direct_parent_publications(old_roi)[0]
        assert old_roi is roi_reference()
        assert old_root is root_reference()

        del old_root
        del old_roi
        del first_fit
        gc.collect()
        assert root_reference() is None
        assert roi_reference() is None
        assert fit_reference() is None
    finally:
        plane.close()
