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
    DomainSpec,
    OwnedSnapshot,
    SCALAR_DOMAIN,
    StreamGenerationId,
    ValueSchema,
)

from zlc_runtime.dataset import MonitorCoverage
from zlc_runtime.front import build_front
from zlc_runtime.plane import (
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
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec((1,), (point,), ((0,),)),
        SCALAR_DOMAIN,
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
    value = SignalValue(name, output.snapshot, output.coverage)
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
    coherent: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        owner_id=owner,
        generation=StreamGenerationId(generation),
        kind=kind,
        output_names=names,
        source_name=source_name,
        coherent=coherent,
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
        _state("roi", "g2", "processor", ("roi/value",), roi, "camera/frame"),
        _state("fit", "g3", "processor", ("fit/value",), fit, "roi/value"),
    ]

    first = build_front(states, {"camera/frame", "roi/value", "fit/value"}, None, parents.__getitem__)
    assert first.names() == ("camera/frame", "roi/value", "fit/value")
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


def test_a_presentation_paced_follower_never_holds_its_source() -> None:
    """A coherent=False route keeps lineage but joins no same-shot component.

    A panel's accepted-fit signal only advances AFTER its source presents:
    letting it hold the source's front selection deadlocks the whole
    component (the source waits for the fit that waits for the source's
    next presentation).  The follower's own consumers simply see its latest
    value, one shot behind by nature.
    """

    root = _publication("camera", "g1", 1, "camera/frame")
    fit = _publication("fit", "g3", 1, "fit/value", (root,))
    parents = {root: (), fit: (root,)}
    states = [
        _state("camera", "g1", "producer", ("camera/frame",), root),
        _state(
            "fit", "g3", "processor", ("fit/value",), fit,
            "camera/frame", coherent=False,
        ),
    ]
    requested = {"camera/frame", "fit/value"}

    first = build_front(states, requested, None, parents.__getitem__)
    # The camera advances two shots; the follower's fit is still for shot 1.
    root3 = _publication("camera", "g1", 3, "camera/frame")
    states[0].publication = root3
    parents[root3] = ()
    flowing = build_front(states, requested, first, parents.__getitem__)
    assert flowing.publication("camera/frame").event_ref.sequence == 3
    assert flowing.publication("fit/value").event_ref.sequence == 1

    # The same topology WITH coherence is the occupancy contract: the source
    # is held at the follower's shot instead of running ahead.
    states[1] = _state(
        "fit", "g3", "processor", ("fit/value",), fit, "camera/frame"
    )
    held = build_front(states, requested, first, parents.__getitem__)
    assert held.publication("camera/frame").event_ref.sequence == 1


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
    roi_tap = None
    fit_tap = None
    try:
        plane.begin_generation(node)
        plane.set_front_signals({"camera/frame", "roi/value", "fit/value"})
        plane.commit_live(node, state)
        first_front = plane.freeze()
        root = first_front.publication("camera/frame")
        assert root is not None

        roi_node = SimpleNamespace(
            instance_id="roi",
            dataset_output_declarations=(_output("roi", 1).declaration,),
            signal_key=lambda _name: "roi/value",
        )
        roi_tap = plane.reserve_follow_processor(
            roi_node,
            source_name="camera/frame",
            source_publication=root,
        )
        assert roi_tap.next(0.0).event_ref == root.event_ref
        plane.commit_processor(
            roi_node,
            {"roi": _output("roi", 1)},
            source_publication=root,
        )
        roi = plane.latest_publication("roi/value")
        assert roi is not None
        fit_node = SimpleNamespace(
            instance_id="fit",
            dataset_output_declarations=(_output("fit", 1).declaration,),
            signal_key=lambda _name: "fit/value",
        )
        fit_tap = plane.reserve_follow_processor(
            fit_node,
            source_name="roi/value",
            source_publication=roi,
        )
        assert fit_tap.next(0.0).event_ref == roi.event_ref
        plane.commit_processor(
            fit_node,
            {"fit": _output("fit", 1)},
            source_publication=roi,
        )
        first_front = plane.freeze()
        first_fit = first_front.publication("fit/value")
        assert first_fit is not None
        root_reference = weakref.ref(root)
        roi_reference = weakref.ref(roi)
        fit_reference = weakref.ref(first_fit)
        state["frame"] = _output("frame", 2)
        plane.commit_live(node, state)
        held_front = plane.freeze()
        root2 = plane.latest_publication("camera/frame")
        assert root2 is not None
        assert held_front.value("camera/frame").snapshot.ref.revision.value == 1
        plane.commit_processor(
            roi_node,
            {"roi": _output("roi", 2)},
            source_publication=root2,
        )
        roi2 = plane.latest_publication("roi/value")
        assert roi2 is not None
        plane.commit_processor(
            fit_node,
            {"fit": _output("fit", 2)},
            source_publication=roi2,
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
        if fit_tap is not None:
            fit_tap.close()
        if roi_tap is not None:
            roi_tap.close()
        plane.close()


def test_a_companion_that_never_spoke_in_its_generation_is_excused() -> None:
    """After a restart, frames flow without the overlay -- never neither.

    A processor re-reserved under a new generation carries its output name
    before its first commit.  Counting that silent name as a leaf made the
    whole component incoherent, and the fallback then popped the BASE too
    (its generation had changed): a camera panel with an occupancy overlay
    froze, silently and indefinitely, the moment a pulse restart re-reserved
    the overlay before its first new-generation shot.

    The distinction is per generation: a processor between two shots of ONE
    generation (current publication None, previous front carrying its last)
    is COMPUTING, and the hold-together contract above still stands -- the
    transitive-fallback test pins that half.
    """

    root = _publication("camera", "g1", 1, "camera/frame")
    occupancy = _publication("occ", "g2", 1, "occ/sites", (root,))
    parents = {root: (), occupancy: (root,)}
    states = [
        _state("camera", "g1", "producer", ("camera/frame",), root),
        _state("occ", "g2", "processor", ("occ/sites",), occupancy, "camera/frame"),
    ]
    first = build_front(
        states, {"camera/frame", "occ/sites"}, None, parents.__getitem__
    )
    assert first.names() == ("camera/frame", "occ/sites")

    root2 = _publication("camera", "g3", 1, "camera/frame")
    parents[root2] = ()
    restarted = [
        _state("camera", "g3", "producer", ("camera/frame",), root2),
        _state("occ", "g4", "processor", ("occ/sites",), None, "camera/frame"),
    ]
    resumed = build_front(
        restarted, {"camera/frame", "occ/sites"}, first, parents.__getitem__
    )
    # The new frame is on screen; the old generation's rings are not.
    assert resumed.publication("camera/frame") is root2
    assert resumed.publication("occ/sites") is None

    # And the first new-generation commit joins atomically.
    occupancy2 = _publication("occ", "g4", 1, "occ/sites", (root2,))
    parents[occupancy2] = (root2,)
    restarted[1].publication = occupancy2
    joined = build_front(
        restarted, {"camera/frame", "occ/sites"}, resumed, parents.__getitem__
    )
    assert joined.publication("occ/sites") is occupancy2
