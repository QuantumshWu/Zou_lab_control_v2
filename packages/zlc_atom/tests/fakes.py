"""Shared test doubles whose public surfaces are frozen external contracts,
and the one import reader the source-structure tests share."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    DomainSpec,
    OwnedSnapshot,
    READOUT_EVENT,
    REPEAT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    ValidityContract,
    ValueSchema,
)
from zlc_runtime import SignalDataPlane as RuntimeSignalDataPlane
from zlc_pulse.device import DoneReport, SafeReadback

from zlc_atom.data import snapshot_from_array
from zlc_atom.devices.simulation.camera import VirtualCamera, VirtualCameraConfig
from zlc_atom.nodes import discover_logic_nodes


def camera_cycle_snapshot(
    cycles: Sequence[Sequence[Any]],
    *,
    producer: str = "camera",
    signal: str = "frames",
    generation: str = "test",
    revision: int = 1,
) -> OwnedSnapshot:
    """Author one camera-shaped publication: (cycles) x (frames) x (y, x).

    The same structure ``camera_measurement`` publishes: a cycle's frames are
    POINTS of the acquisition, because each fires at a different place in the
    pulse.  Offline tests hold raw arrays or adapter records, and a raw array
    cannot say which of its axes is the point axis -- so a test that wants to
    hand frames to a consumer has to say it here, once, the way the producer
    would.
    """

    images = np.stack(
        [
            np.stack(
                [
                    np.asarray(getattr(frame, "image", frame))
                    for frame in cycle
                ],
                axis=0,
            )
            for cycle in cycles
        ],
        axis=0,
    )
    return snapshot_from_array(
        images,
        producer=producer,
        signal=signal,
        point_axes=(
            AxisSpec(
                AxisId(f"{producer}.{signal}.frame"),
                "frame",
                READOUT_EVENT,
                int(images.shape[1]),
                tuple(range(int(images.shape[1]))),
            ),
        ),
        cell_axes=(SPATIAL_Y, SPATIAL_X),
        generation=generation,
        revision=revision,
    )


def scan_source_schema(*, shots: int) -> DatasetSchema:
    """A scan source publishing ``shots`` per event over five sites."""

    repeat = AxisSpec(AxisId("shot"), "repeat", REPEAT, shots, tuple(range(shots)))
    event = AxisSpec(AxisId("event"), "event", READOUT_EVENT, 1, (0,))
    site = AxisSpec(AxisId("site"), "site", SITE, 5, tuple(range(5)))
    return DatasetSchema(
        DomainSpec((shots,), (repeat,), (tuple(range(shots)),)),
        DomainSpec((1,), (event,), ((0,),)),
        DomainSpec((site.size,), (site,)),
        ValueSchema(ValidityContract.components(site.axis_id), np.dtype("<f8"), "1"),
    )


#: The source tree the import-structure tests read.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def module_name(path: Path) -> str:
    """The dotted module one file under ``src`` is imported as."""

    return ".".join(path.resolve().relative_to(SOURCE_ROOT).with_suffix("").parts)


def imported_modules(path: Path) -> set[str]:
    """Every module one source file imports, relative spellings resolved.

    A relative import reaches exactly as far as an absolute one -- a sibling
    is ``from ..other import x`` -- so a guard that reads only absolute ones
    guards nothing.  ``from X import a`` may name a module too, so both
    readings are collected and matched.
    """

    package = module_name(path).rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if not node.level:
            target = node.module
        else:
            parts = package.split(".")
            base = ".".join(parts[: len(parts) - node.level + 1])
            target = f"{base}.{node.module}" if node.module else base
        found.add(target)
        found.update(f"{target}.{alias.name}" for alias in node.names)
    return found


class FakePlane(RuntimeSignalDataPlane):
    """Instrumented runtime plane; every method retains the frozen signature."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[Any, ...], Mapping[str, Any]]] = []

    def begin_generation(self, producer: object):
        self.calls.append(("begin_generation", (producer,), {}))
        return super().begin_generation(producer)

    def retire(self, producer: object):
        self.calls.append(("retire", (producer,), {}))
        return super().retire(producer)

    def cancel_latest_only_processor(self, control: object) -> bool:
        self.calls.append(("cancel_latest_only_processor", (control,), {}))
        return super().cancel_latest_only_processor(control)  # type: ignore[arg-type]


def running_slm_server(adapter: object) -> tuple[Any, threading.Thread]:
    """The SLM server over ``adapter`` on a loopback port, serving on a thread.

    The caller shuts the server down, closes it and joins the thread.
    """

    from zlc_atom.devices.slm.hamamatsu_x15213.remote import _open_slm_server

    server = _open_slm_server(adapter, "127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    return server, worker


#: The scripted source's frame: small, because only its VALUE is under test.
SCRIPTED_FRAME_SHAPE_YX = (4, 4)

#: What the scripted bench publishes to make the signal live before a scan
#: starts.  A scan drains the backlog before it applies its first point, so
#: this value must never appear among the shots a scan kept.
SCRIPTED_SEED_VALUE = 999


class ScriptedScanBench:
    """A real sequencer that also plays the SOURCE's part, on a script.

    The question a scan test has to ask -- WHICH publications did it keep --
    cannot be asked of frames that differ only by noise, and cannot be asked
    at all while a source runs on its own wall clock.  So exactly the two
    facts that distinguish one source family from another are scripted, and
    nothing else is:

    * every frame is a constant image whose value is that publication's
      index, so a kept shot can be named in an assertion;
    * one ``fire`` produces exactly ``publications_per_fire`` of them,
      matching the selected hardware-table segment.

    Everything else is the production path: the real virtual board compiles,
    loads, writes its scan table and fires; the real camera adapter, the real
    ``camera_measurement`` monitor and the real signal plane carry the frames.
    """

    def __init__(
        self,
        sequencer: object,
        plane: object,
        *,
        publications_per_fire: int,
        exposure_seconds: float = 0.001,
    ) -> None:
        self._sequencer = sequencer
        self._plane = plane
        self.publications_per_fire = int(publications_per_fire)
        if self.publications_per_fire < 1:
            raise ValueError("a fire produces at least one publication")
        self.camera = VirtualCamera(
            VirtualCameraConfig(
                frame_shape_yx=SCRIPTED_FRAME_SHAPE_YX,
                exposure_seconds=exposure_seconds,
            ),
            frame_source=lambda exposure: np.zeros(
                SCRIPTED_FRAME_SHAPE_YX, dtype="<u2"
            ),
        )
        descriptor = {
            value.api_name: value for value in discover_logic_nodes()
        }["camera_measurement"]
        self._node = descriptor.instantiate(
            camera=self.camera,
            camera_key="scripted",
            signal_plane=plane,
            repeat=0,
            frames_per_cycle=1,
            exposure_seconds=exposure_seconds,
            # These frames ARE the assertions: each carries the ordinal the
            # scan is checked against, so this bench reads the counts the
            # scripted camera writes rather than electrons derived from them.
            photoelectrons=False,
        )
        self.monitor = self._node.monitor()
        self.signal_name = self._node.signal_key("frames")
        self.published: list[int] = []
        self.loads = 0
        self.loaded_loop_counts: list[int] = []
        self.loaded_sources: list[object | None] = []
        self._loaded_program = None
        self.scan_tables: list[np.ndarray] = []
        self.fired_repeats: list[tuple[int, int]] = []
        self._next_value = 0
        self._loaded_rows: tuple[tuple[int, ...], ...] = ()

    # ------------------------------------------------------- the source

    def publish(self, value: int) -> None:
        """One frame, one monitor cycle, one materialised publication."""

        image = np.full(SCRIPTED_FRAME_SHAPE_YX, int(value), dtype="<u2")
        self.camera.trigger(1, frame=image)
        deadline = time.monotonic() + 1.0
        while self.monitor.poll() is None:
            if time.monotonic() >= deadline:
                raise RuntimeError("the scripted camera did not produce its frame")
            time.sleep(0.001)
        self.published.append(int(value))
        self._plane.freeze()

    def close(self) -> None:
        self.monitor.close()

    # --------------------------------------------- the sequencer surface

    def describe(self) -> object:
        return self._sequencer.describe()

    # The config surface, forwarded like everything else: the sequence a
    # caller loads must be the one the board filled, or the double proves a
    # behaviour the product does not have.
    def load_config_values(self, entries, *, source: str = "") -> None:
        self._sequencer.load_config_values(entries, source=source)

    def load_config_file(self, path) -> None:
        self._sequencer.load_config_file(path)

    def config_values(self) -> dict:
        return self._sequencer.config_values()

    def compile_pulse(self, sequence, geom, clock_hz):
        return self._sequencer.compile_pulse(sequence, geom, clock_hz)

    @property
    def config_source(self) -> str:
        return self._sequencer.config_source

    def load(
        self,
        prog: object,
        *,
        source: object | None = None,
        rows: object = (),
    ) -> None:
        self.loads += 1
        self.loaded_loop_counts.append(int(prog.loops[0][2]) if prog.loops else 1)
        self.loaded_sources.append(source)
        self._loaded_program = prog
        normalized = tuple(tuple(row) for row in rows)
        self._loaded_rows = normalized
        if normalized:
            self.scan_tables.append(np.asarray(normalized))
        self._sequencer.load(prog, source=source, rows=normalized)

    def fire(self, *, run_repeats: int, scan_repeats: int = 1):
        self.fired_repeats.append((int(run_repeats), int(scan_repeats)))
        execution = self._sequencer.fire(
            run_repeats=run_repeats,
            scan_repeats=scan_repeats,
        )
        for _ in range(self.publications_per_fire):
            self.publish(self._next_value)
            self._next_value += 1
        return execution

    def wait_done(self, timeout: float | None = None) -> DoneReport | None:
        return self._sequencer.wait_done(timeout)

    def snapshot(self) -> Mapping[str, object]:
        # A scan that is still waiting for a report asks the board whether
        # it is still firing; the answer is the real board's.
        return self._sequencer.snapshot()

    def applied(self):
        return self._sequencer.applied()

    def safe(self) -> SafeReadback:
        return self._sequencer.safe()


__all__ = [
    "SCRIPTED_SEED_VALUE",
    "SOURCE_ROOT",
    "FakePlane",
    "running_slm_server",
    "ScriptedScanBench",
    "imported_modules",
    "module_name",
    "scan_source_schema",
]
