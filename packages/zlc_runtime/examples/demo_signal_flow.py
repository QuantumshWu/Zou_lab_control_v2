"""Headless signal-flow timeline for the zlc-runtime acceptance surface."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from threading import Event
import time

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

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
from zlc_runtime import NodeHost, SignalDataPlane
from zlc_runtime.dataset import MonitorCoverage
from zlc_runtime.dataset_output import DatasetOutputDeclaration, LiveDatasetOutput


def _snapshot(name: str, revision: int) -> OwnedSnapshot:
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
    return OwnedSnapshot(
        block.ref(StreamGenerationId(f"{name}-generation-{revision}")),
        block,
    )


def _live(declaration: DatasetOutputDeclaration, revision: int) -> LiveDatasetOutput:
    return LiveDatasetOutput(
        declaration,
        _snapshot(declaration.name, revision),
        MonitorCoverage(1, 1, 0, False),
    )


class _CameraSlot:
    notification_failure = None

    def __init__(self, state: dict[str, LiveDatasetOutput]) -> None:
        self._state = state

    def freeze_live_outputs(self):
        return dict(self._state)

    def close(self) -> None:
        pass


class _Camera:
    instance_id = "camera"

    def __init__(self, declaration: DatasetOutputDeclaration) -> None:
        self.dataset_output_declarations = (declaration,)

    def signal_key(self, name: str) -> str:
        return f"camera/{name}"


def _drive_until(
    plane: SignalDataPlane,
    wake: Event,
    signal_name: str,
    revision: int,
) :
    deadline = time.monotonic() + 3.0
    while True:
        front = plane.freeze()
        value = front.value(signal_name)
        if value is not None and value.snapshot.ref.revision.value == revision:
            return front
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 or not wake.wait(remaining):
            raise TimeoutError(f"{signal_name} did not reach revision {revision}")
        wake.clear()


class _RoiNode:
    kind = "reactive"
    instance_id = "roi"
    input_signal = "camera/frame"

    def __init__(self, declaration: DatasetOutputDeclaration) -> None:
        self.dataset_output_declarations = (declaration,)
        self._declaration = declaration

    def evaluate(self, source):
        return {
            "roi": _live(
                self._declaration,
                source.snapshot.ref.revision.value,
            )
        }


def run(frame_count: int) -> None:
    camera_declaration = DatasetOutputDeclaration("frame", "demo.camera.frame")
    roi_declaration = DatasetOutputDeclaration("roi", "demo.roi")
    state = {"frame": _live(camera_declaration, 1)}
    camera = _Camera(camera_declaration)
    slot = _CameraSlot(state)
    plane = SignalDataPlane()
    wake = Event()
    host = NodeHost(_RoiNode(roi_declaration), plane, wake.set)

    try:
        plane.reserve(camera)
        plane.attach(camera, slot)
        plane.mark_changed(camera, slot)
        plane.set_front_signals({"camera/frame", "@logic/roi/roi"})
        plane.freeze()
        host.start()

        print("zlc-runtime signal flow")
        for revision in range(1, frame_count + 1):
            if revision > 1:
                state["frame"] = _live(camera_declaration, revision)
                plane.mark_changed(camera, slot)
            front = _drive_until(plane, wake, "@logic/roi/roi", revision)
            source_publication = front.publication("camera/frame")
            roi_publication = front.publication("@logic/roi/roi")
            assert source_publication is not None and roi_publication is not None
            print(
                f"frame={revision} "
                f"root_seq={source_publication.event_ref.sequence} "
                f"roi_parent={roi_publication.direct_parent_refs[0].sequence}"
            )
    finally:
        if host.running:
            host.cancel("demo complete")
        host.shutdown()
        plane.retire(camera)
        plane.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="emit one timeline frame")
    parser.add_argument("--frames", type=int, default=3, help="number of fake frames")
    args = parser.parse_args()
    frame_count = 1 if args.once else args.frames
    if frame_count <= 0:
        parser.error("--frames must be positive")
    run(frame_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
