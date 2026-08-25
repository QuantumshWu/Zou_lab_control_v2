"""Explicit Logic Node lifecycle contracts over the canonical signal plane."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from threading import Event
import time

import pytest

from zlc_data import AxisSpec, DatasetSchema
from zlc_runtime.dataset import DatasetCoverage, MonitorCoverage
from zlc_runtime.dataset_output import DatasetOutputDeclaration, LiveDatasetOutput
from zlc_runtime.host import LogicNodeObservation, NodeHost
from zlc_runtime.plane import SignalDataPlane, SignalValue

from _snapshots import snapshot as _snapshot


def _monitor_output(
    declaration: DatasetOutputDeclaration,
    revision: int,
    *,
    value: float | None = None,
) -> LiveDatasetOutput:
    return LiveDatasetOutput(
        declaration,
        _snapshot(declaration.name, revision, value=value),
        MonitorCoverage(1, 1),
    )


def _finite_output(
    declaration: DatasetOutputDeclaration,
    *,
    value: float,
    total: int,
    origin: int,
    written: int,
) -> LiveDatasetOutput:
    event = _snapshot(declaration.name, origin + 1, value=value)
    schema = event.block.schema
    repeat = schema.repeat_axis
    canonical = DatasetSchema(
        AxisSpec(
            repeat.axis_id,
            repeat.name,
            repeat.role,
            total,
            tuple(range(total)),
        ),
        schema.point_table,
        schema.grid_topology,
        schema.cell_schema,
    )
    return LiveDatasetOutput(
        declaration,
        event,
        DatasetCoverage(written, total),
        canonical_schema=canonical,
        cell_origin=(origin, 0),
    )


def _wait(host: NodeHost, wake: Event) -> LogicNodeObservation:
    deadline = time.monotonic() + 3.0
    while not host.terminal and time.monotonic() < deadline:
        host.poll()
        wake.wait(0.01)
        wake.clear()
    observation = host.poll()
    assert observation.terminal, observation
    return observation


def _host(
    node: object,
    plane: SignalDataPlane,
    wake: Event,
    *,
    instance_id: str,
    kind: str,
    outputs: tuple[DatasetOutputDeclaration, ...] = (),
    source: str | None = None,
    delivery: str | None = None,
    artifacts: dict[str, str] | None = None,
    task_name: str | None = None,
) -> NodeHost:
    return NodeHost(
        node,
        plane,
        wake.set,
        instance_id=instance_id,
        kind=kind,
        dataset_output_declarations=outputs,
        input_signal=source,
        input_delivery=delivery,
        required_artifacts={} if artifacts is None else artifacts,
        task_name=task_name,
    )


def test_constructor_uses_only_the_explicit_descriptor_contract() -> None:
    declaration = DatasetOutputDeclaration("declared", "test.declared")

    class MisleadingNode:
        instance_id = "wrong"
        source_signal = "@logic/wrong/source"
        dataset_output_declarations = (
            DatasetOutputDeclaration("wrong", "test.wrong"),
        )

    plane = SignalDataPlane()
    wake = Event()
    try:
        with pytest.raises(TypeError):
            NodeHost(MisleadingNode(), plane)
        host = _host(
            MisleadingNode(),
            plane,
            wake,
            instance_id="right",
            kind="measurement",
            outputs=(declaration,),
        )
        assert host.instance_id == "right"
        assert host.dataset_output_declarations == (declaration,)
        with pytest.raises(KeyError):
            host.signal_key("wrong")
        host.shutdown()
        with pytest.raises(ValueError, match="input_delivery"):
            _host(
                object(),
                plane,
                wake,
                instance_id="processor",
                kind="processor",
                outputs=(declaration,),
                source="@logic/source/value",
            )
    finally:
        plane.close()


def test_measurement_commits_live_then_runtime_seals_and_clears_progress() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        def execute(self, context):
            assert context.instance_id == "camera"
            context.report_progress("capturing", current=1, total=1)
            context.commit_live(
                {
                    "frame": _finite_output(
                        declaration,
                        value=7.0,
                        total=1,
                        origin=0,
                        written=1,
                    )
                }
            )
            current = context.current_dataset("frame")
            assert float(current.block.values[0, 0, 0]) == 7.0
            return {"status": "ok"}

    host = _host(
        Node(),
        plane,
        wake,
        instance_id="camera",
        kind="measurement",
        outputs=(declaration,),
    )
    try:
        host.start()
        assert host.observation.running and host.observation.phase == "running"
        observation = _wait(host, wake)
        assert observation.phase == "done"
        assert observation.progress is None
        assert host.final_result == {"status": "ok"}
        signal = host.signal_key("frame")
        assert not plane.is_generation_live(signal)
        assert float(plane.current_dataset(signal).block.values[0, 0, 0]) == 7.0
    finally:
        host.shutdown()
        plane.close()


def test_task_operator_input_is_exactly_answered_or_stopped(tmp_path: Path) -> None:
    declaration = DatasetOutputDeclaration("review", "test.operator-review")
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        def execute(self, context):
            context.report_progress("waiting for operator")
            context.commit_live({"review": _monitor_output(declaration, 1)})
            return dict(
                context.request_operator_input(
                    "manual-value",
                    title="Set the device",
                    message="Enter the observed value",
                    payload={"field": "voltage"},
                )
            )

    def waiting(host: NodeHost) -> object:
        deadline = time.monotonic() + 3.0
        while host.operator_request is None and time.monotonic() < deadline:
            wake.wait(0.01)
            wake.clear()
        request = host.operator_request
        assert request is not None
        return request

    host = _host(
        Node(),
        plane,
        wake,
        instance_id="interactive-task",
        kind="task",
        outputs=(declaration,),
    )
    try:
        host.start(run_root=tmp_path, input_summary={})
        request = waiting(host)
        assert request.kind == "manual-value"
        assert request.payload == {"field": "voltage"}
        with pytest.raises(RuntimeError, match="does not match"):
            host.submit_operator_input("stale", {"value": 3.0})
        host.submit_operator_input(request.request_id, {"value": 3.0})
        assert _wait(host, wake).phase == "done"
        assert host.final_result == {"value": 3.0}
        assert host.operator_request is None
        host.start(run_root=tmp_path, input_summary={})
        next_request = waiting(host)
        assert next_request.request_id != request.request_id
        host.cancel("operator stopped the review")
        assert _wait(host, wake).phase == "cancelled"
        assert host.operator_request is None
    finally:
        host.shutdown()
        plane.close()


def test_finite_commits_do_not_materialize_the_canonical_dataset(
    monkeypatch,
) -> None:
    declaration = DatasetOutputDeclaration("scan", "test.scan")
    wake = Event()
    plane = SignalDataPlane()
    calls: list[object] = []
    current_dataset = plane.current_dataset

    def observed(*args, **kwargs):
        calls.append((args, kwargs))
        return current_dataset(*args, **kwargs)

    monkeypatch.setattr(plane, "current_dataset", observed)

    class Node:
        def execute(self, context):
            for index in range(3):
                context.commit_live(
                    {
                        "scan": _finite_output(
                            declaration,
                            value=float(index + 1),
                            total=3,
                            origin=index,
                            written=index + 1,
                        )
                    }
                )

    host = _host(
        Node(),
        plane,
        wake,
        instance_id="growing-scan",
        kind="measurement",
        outputs=(declaration,),
    )
    try:
        host.start()
        assert _wait(host, wake).phase == "done"
        assert calls == []
        assert current_dataset(host.signal_key("scan")).block.values[:, 0, 0].tolist() == [
            1.0,
            2.0,
            3.0,
        ]
    finally:
        host.shutdown()
        plane.close()


def test_measurement_and_task_terminal_contracts_fail_loudly(tmp_path: Path) -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    cases = (
        (
            "measurement",
            (declaration,),
            {},
            lambda _context: None,
            "without a live Dataset commit",
        ),
        (
            "task",
            (),
            {},
            lambda _context: {"ok": True},
            "without reporting progress",
        ),
        (
            "task",
            (),
            {"artifact_path": "test.artifact"},
            lambda context: (
                context.report_progress("saving"),
                {"artifact_path": tmp_path / "missing.json"},
            )[1],
            "did not register required final artifact",
        ),
        (
            "task",
            (),
            {"artifact_path": "test.artifact"},
            lambda context: (
                context.report_progress("saving"),
                {"artifact_path": tmp_path},
            )[1],
            "did not register required final artifact",
        ),
    )
    for index, (kind, outputs, artifacts, execute, message) in enumerate(cases):
        wake = Event()
        plane = SignalDataPlane()
        node = type("Node", (), {"execute": staticmethod(execute)})()
        host = _host(
            node,
            plane,
            wake,
            instance_id=f"bad-{index}",
            kind=kind,
            outputs=outputs,
            artifacts=artifacts,
        )
        try:
            if kind == "task":
                host.start(run_root=tmp_path, input_summary={"case": index})
            else:
                host.start()
            observation = _wait(host, wake)
            assert observation.phase == "failed"
            assert message in (observation.error or "")
            assert observation.progress is None
            assert not host.final_result_resolved
        finally:
            host.shutdown()
            plane.close()


def test_task_requires_and_preserves_an_existing_declared_artifact(tmp_path: Path) -> None:
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        def execute(self, context):
            artifact = context.run_directory / "result.json"
            artifact.write_text("{}", encoding="utf-8")
            context.register_artifact(
                "artifact_path",
                artifact,
                role="final",
                contract_id="test.artifact",
            )
            context.report_progress("saved")
            return {"artifact_path": artifact}

    host = _host(
        Node(),
        plane,
        wake,
        instance_id="artifact-task",
        kind="task",
        artifacts={"artifact_path": "test.artifact"},
    )
    try:
        assert list(tmp_path.iterdir()) == []
        host.start(run_root=tmp_path, input_summary={"shots": 3})
        assert _wait(host, wake).phase == "done"
        artifact = host.artifacts[0]
        assert host.final_result == {"artifact_path": artifact.path}
        document = json.loads((host.run_directory / "run.json").read_text())
        assert document["status"]["state"] == "completed"
        assert document["input"] == {"shots": 3}
        assert document["artifacts"] == [{
            "name": artifact.name,
            "path": artifact.relative_path,
            "role": artifact.role,
            "contract_id": artifact.contract_id,
            "size_bytes": artifact.size_bytes,
        }]
    finally:
        host.shutdown()
        plane.close()


def test_task_failure_keeps_registered_process_artifacts_and_error(
    tmp_path: Path,
) -> None:
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        def execute(self, context):
            def save_partial(status, error):
                assert status == "failed"
                assert str(error) == "fit exploded"
                assert threading.current_thread().name.startswith("node-feedback")
                report = context.run_directory / "partial-figure.npz"
                report.write_bytes(b"partial figure")
                context.register_artifact(
                    "partial_figure", report, role="figure"
                )

            context.register_partial_exit_writer(save_partial)
            process = context.run_directory / "process"
            process.mkdir()
            checkpoint = process / "candidate.json"
            checkpoint.write_text('{"candidate": 1}', encoding="utf-8")
            context.register_artifact(
                "candidate-1",
                checkpoint,
                role="checkpoint",
            )
            context.report_progress("measured candidate", current=1, total=3)
            raise ValueError("fit exploded")

    host = _host(
        Node(),
        plane,
        wake,
        instance_id="feedback",
        kind="task",
        task_name="slm-feedback",
    )
    try:
        host.start(run_root=tmp_path, input_summary={"shots": 100})
        observation = _wait(host, wake)
        assert observation.phase == "failed"
        assert host.run_directory is not None
        assert host.run_directory.name == "slm-feedback"
        assert host.artifacts[0].path.read_text(encoding="utf-8") == '{"candidate": 1}'
        document = json.loads((host.run_directory / "run.json").read_text())
        assert document["status"]["state"] == "failed"
        assert document["error"]["type"].endswith("ValueError")
        assert document["error"]["message"] == "fit exploded"
        assert {item["role"] for item in document["artifacts"]} == {
            "checkpoint",
            "figure",
        }
        assert (host.run_directory / "partial-figure.npz").is_file()
    finally:
        host.shutdown()
        plane.close()


def test_task_stop_keeps_run_and_process_artifact(tmp_path: Path) -> None:
    running = Event()
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        def execute(self, context):
            def save_partial(status, _error):
                assert status == "stopped"
                report = context.run_directory / "stop-report.npz"
                report.write_bytes(b"stopped report")
                context.register_artifact(
                    "stop_report", report, role="figure"
                )

            context.register_partial_exit_writer(save_partial)
            checkpoint = context.run_directory / "partial.json"
            checkpoint.write_text("{}", encoding="utf-8")
            context.register_artifact("partial", checkpoint, role="process")
            context.report_progress("waiting")
            running.set()
            while not context.cancel_requested():
                time.sleep(0.001)

    host = _host(
        Node(),
        plane,
        wake,
        instance_id="long-task",
        kind="task",
    )
    try:
        host.start(run_root=tmp_path, input_summary={})
        assert running.wait(2.0)
        host.cancel("operator Stop")
        assert _wait(host, wake).phase == "cancelled"
        assert host.artifacts[0].path.is_file()
        document = json.loads((host.run_directory / "run.json").read_text())
        assert document["status"]["state"] == "stopped"
        assert document["status"]["stop_reason"] == "operator Stop"
        assert document["error"] is None
        assert {item["name"] for item in document["artifacts"]} == {
            "partial",
            "stop_report",
        }
    finally:
        host.shutdown()
        plane.close()


@pytest.mark.parametrize("observer", ("none", "panel-reads", "exact-processor"))
def test_stop_seals_the_same_partial_dataset_independent_of_display(
    observer: str,
) -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    derived_declaration = DatasetOutputDeclaration("seen", "test.seen")
    committed = Event()
    wake = Event()
    plane = SignalDataPlane()
    processor: NodeHost | None = None
    processor_seen = Event()

    class Node:
        def execute(self, context):
            context.commit_live(
                {
                    "frame": _finite_output(
                        declaration,
                        value=3.0,
                        total=2,
                        origin=0,
                        written=1,
                    )
                }
            )
            committed.set()
            while not context.cancel_requested():
                time.sleep(0.001)

    host = _host(
        Node(),
        plane,
        wake,
        instance_id="partial-camera",
        kind="measurement",
        outputs=(declaration,),
    )
    try:
        host.start()
        assert committed.wait(2.0)
        signal = host.signal_key("frame")
        assert host.running
        assert plane.latest_publication(signal) is not None

        if observer == "panel-reads":
            for _ in range(5):
                front = plane.freeze()
                assert front.value(signal) is not None
                displayed = plane.current_dataset(signal)
                assert displayed.block.values[:, 0, 0].tolist() == [3.0, 0.0]
                assert displayed.expanded_validity()[:, 0, 0].tolist() == [
                    True,
                    False,
                ]
        elif observer == "exact-processor":

            class Processor:
                def evaluate(self, value: SignalValue):
                    assert value.name == signal
                    assert value.snapshot.block.values[:, 0, 0].tolist() == [3.0]
                    processor_seen.set()
                    return {
                        "seen": _monitor_output(
                            derived_declaration,
                            1,
                            value=3.0,
                        )
                    }

            processor = _host(
                Processor(),
                plane,
                wake,
                instance_id="partial-observer",
                kind="processor",
                outputs=(derived_declaration,),
                source=signal,
                delivery="exact",
            )
            processor.start()
            assert processor_seen.wait(2.0)

        host.cancel("operator stop")
        observation = _wait(host, wake)
        assert observation.phase == "cancelled"
        assert observation.progress is None
        assert not host.final_result_resolved
        partial = plane.current_dataset(signal)
        assert partial.block.values[:, 0, 0].tolist() == [3.0, 0.0]
        assert partial.expanded_validity()[:, 0, 0].tolist() == [True, False]
        if processor is not None:
            assert _wait(processor, wake).phase == "done"
    finally:
        if processor is not None:
            processor.shutdown()
        host.shutdown()
        plane.close()


def test_stop_reports_partial_seal_failure_instead_of_cancellation(
    monkeypatch,
) -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    committed = Event()
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        def execute(self, context):
            context.commit_live(
                {
                    "frame": _finite_output(
                        declaration,
                        value=3.0,
                        total=2,
                        origin=0,
                        written=1,
                    )
                }
            )
            committed.set()
            while not context.cancel_requested():
                time.sleep(0.001)

    host = _host(
        Node(),
        plane,
        wake,
        instance_id="failed-partial-seal",
        kind="measurement",
        outputs=(declaration,),
    )
    try:
        host.start()
        assert committed.wait(2.0)

        def fail_seal(_node, *, cut_short=False):
            assert cut_short is True
            raise RuntimeError("seal exploded")

        monkeypatch.setattr(plane, "seal_committed", fail_seal)
        host.cancel("operator stop")
        observation = _wait(host, wake)
        assert observation.phase == "failed"
        assert observation.error == (
            "stopped partial Dataset could not be sealed: "
            "RuntimeError: seal exploded"
        )
        assert observation.progress is None
        assert not host.final_result_resolved
        assert not plane.is_generation_live(host.signal_key("frame"))
    finally:
        host.shutdown()
        plane.close()


@pytest.mark.parametrize(
    ("partial", "phase"),
    ((False, "failed"), (True, "done")),
)
def test_successful_partial_exact_output_requires_explicit_terminal_intent(
    partial: bool,
    phase: str,
) -> None:
    declaration = DatasetOutputDeclaration("history", "test.history")
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        def execute(self, context):
            context.commit_live(
                {
                    "history": _finite_output(
                        declaration,
                        value=1.0,
                        total=2,
                        origin=0,
                        written=1,
                    )
                }
            )
            context.seal_terminal(partial=partial)

    host = _host(
        Node(),
        plane,
        wake,
        instance_id=f"partial-{partial}",
        kind="measurement",
        outputs=(declaration,),
    )
    try:
        host.start()
        observation = _wait(host, wake)
        assert observation.phase == phase
        if partial:
            snapshot = plane.current_dataset(host.signal_key("history"))
            assert snapshot.expanded_validity()[:, 0, 0].tolist() == [True, False]
        else:
            assert "coverage is incomplete" in (observation.error or "")
    finally:
        host.shutdown()
        plane.close()


def test_task_can_accept_a_stop_inside_its_terminal_commit(tmp_path: Path) -> None:
    running = Event()
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        def execute(self, context):
            running.set()
            while not context.cancel_requested():
                time.sleep(0.001)
            context.report_progress("retaining best candidate")
            context.seal_terminal(accept_stop=True)
            return {"accepted_stop": True}

    host = _host(
        Node(),
        plane,
        wake,
        instance_id="accept-stop",
        kind="task",
    )
    try:
        host.start(run_root=tmp_path, input_summary={})
        assert running.wait(2.0)
        host.cancel("accept best")
        observation = _wait(host, wake)
        assert observation.phase == "done"
        assert observation.progress is None
        assert host.final_result == {"accepted_stop": True}
        document = json.loads((host.run_directory / "run.json").read_text())
        assert document["status"]["state"] == "stopped"
    finally:
        host.shutdown()
        plane.close()


class _Source:
    def __init__(self, instance_id: str, declaration: DatasetOutputDeclaration) -> None:
        self.instance_id = instance_id
        self.dataset_output_declarations = (declaration,)

    def signal_key(self, output_name: str) -> str:
        return f"@logic/{self.instance_id}/{output_name}"


@pytest.mark.parametrize("delivery", ("exact", "latest"))
def test_terminal_processor_always_receives_runtime_current_dataset(delivery: str) -> None:
    source_declaration = DatasetOutputDeclaration("frame", "test.frame")
    derived_declaration = DatasetOutputDeclaration("derived", "test.derived")
    source = _Source(f"source-{delivery}", source_declaration)
    plane = SignalDataPlane()
    plane.begin_generation(source)
    plane.commit_live(
        source,
        {
            "frame": _finite_output(
                source_declaration,
                value=1.0,
                total=2,
                origin=0,
                written=1,
            )
        },
    )
    plane.commit_live(
        source,
        {
            "frame": _finite_output(
                source_declaration,
                value=2.0,
                total=2,
                origin=1,
                written=2,
            )
        },
    )
    plane.seal_committed(source)
    seen: list[tuple[int, list[float]]] = []

    class Processor:
        def evaluate(self, value: SignalValue):
            seen.append(
                (
                    value.snapshot.block.schema.repeat_axis.size,
                    value.snapshot.block.values[:, 0, 0].tolist(),
                )
            )
            return {"derived": _monitor_output(derived_declaration, 1)}

    wake = Event()
    host = _host(
        Processor(),
        plane,
        wake,
        instance_id=f"processor-{delivery}",
        kind="processor",
        outputs=(derived_declaration,),
        source=source.signal_key("frame"),
        delivery=delivery,
    )
    try:
        host.start()
        assert _wait(host, wake).phase == "done"
        assert seen == [(2, [1.0, 2.0])]
        result = plane.current_dataset(host.signal_key("derived"))
        assert result.block.values.reshape(-1).tolist() == [1.0]
        assert not plane.is_generation_live(host.signal_key("derived"))
    finally:
        host.shutdown()
        plane.close()


def test_exact_processor_receives_each_event_chunk_not_cumulative_history() -> None:
    source_declaration = DatasetOutputDeclaration("frame", "test.frame")
    derived_declaration = DatasetOutputDeclaration("derived", "test.derived")
    source = _Source("exact-source", source_declaration)
    plane = SignalDataPlane()
    plane.begin_generation(source)
    plane.commit_live(
        source,
        {
            "frame": _finite_output(
                source_declaration,
                value=4.0,
                total=3,
                origin=0,
                written=1,
            )
        },
    )
    plane.commit_live(
        source,
        {
            "frame": _finite_output(
                source_declaration,
                value=6.0,
                total=3,
                origin=1,
                written=2,
            )
        },
    )
    seen: list[tuple[int, float, tuple[int, int] | None]] = []

    class Processor:
        def evaluate(self, value: SignalValue):
            seen.append(
                (
                    value.snapshot.block.schema.repeat_axis.size,
                    float(value.snapshot.block.values[0, 0, 0]),
                    value.cell_origin,
                )
            )
            return {
                "derived": _finite_output(
                    derived_declaration,
                    value=float(len(seen)),
                    total=3,
                    origin=len(seen) - 1,
                    written=len(seen),
                )
            }

    wake = Event()
    host = _host(
        Processor(),
        plane,
        wake,
        instance_id="exact-processor",
        kind="processor",
        outputs=(derived_declaration,),
        source=source.signal_key("frame"),
        delivery="exact",
    )
    try:
        host.start()
        deadline = time.monotonic() + 2.0
        while len(seen) < 2 and time.monotonic() < deadline:
            host.poll()
            time.sleep(0.001)
        partial = plane.current_dataset(host.signal_key("derived"))
        assert partial.block.values[:, 0, 0].tolist() == [1.0, 2.0, 0.0]
        assert partial.expanded_validity()[:, 0, 0].tolist() == [
            True,
            True,
            False,
        ]
        plane.commit_live(
            source,
            {
                "frame": _finite_output(
                    source_declaration,
                    value=9.0,
                    total=3,
                    origin=2,
                    written=3,
                )
            },
        )
        plane.seal_committed(source)
        assert _wait(host, wake).phase == "done"
        assert seen == [
            (1, 4.0, (0, 0)),
            (1, 6.0, (1, 0)),
            (1, 9.0, (2, 0)),
        ]
        derived = plane.current_dataset(host.signal_key("derived"))
        assert derived.block.values[:, 0, 0].tolist() == [1.0, 2.0, 3.0]
        publication = plane.latest_publication(host.signal_key("derived"))
        source_publication = plane.latest_publication(source.signal_key("frame"))
        assert publication is not None and source_publication is not None
        assert plane.direct_parent_publications(publication) == (
            source_publication,
        )
    finally:
        host.shutdown()
        plane.close()


def test_an_exact_processor_starts_on_an_armed_silent_source() -> None:
    """The bench flow behind a scan over a derived signal.

    Camera armed, pulse stopped: the source generation is live with ZERO
    publications.  The processor must genuinely start -- reserving its own
    derived generation, which is what a scan's armed-source gate reads --
    and the first frame, fired later by the scan's own pulse, must be its
    first exact input.
    """

    source_declaration = DatasetOutputDeclaration("frame", "test.frame")
    derived_declaration = DatasetOutputDeclaration("derived", "test.derived")
    source = _Source("armed-source", source_declaration)
    plane = SignalDataPlane()
    plane.begin_generation(source)
    seen: list[float] = []

    class Processor:
        def evaluate(self, value: SignalValue):
            seen.append(float(value.snapshot.block.values[0, 0, 0]))
            return {
                "derived": _finite_output(
                    derived_declaration,
                    value=float(len(seen)),
                    total=2,
                    origin=len(seen) - 1,
                    written=len(seen),
                )
            }

    wake = Event()
    host = _host(
        Processor(),
        plane,
        wake,
        instance_id="armed-processor",
        kind="processor",
        outputs=(derived_declaration,),
        source=source.signal_key("frame"),
        delivery="exact",
    )
    try:
        host.start()
        assert host.observation.phase == "running"
        # The derived generation is live BEFORE any source frame exists:
        # this is the fact a scan's armed-source gate reads.
        assert plane.is_generation_live(host.signal_key("derived"))
        assert seen == []
        for origin, value in enumerate((4.0, 6.0)):
            plane.commit_live(
                source,
                {
                    "frame": _finite_output(
                        source_declaration,
                        value=value,
                        total=2,
                        origin=origin,
                        written=origin + 1,
                    )
                },
            )
        plane.seal_committed(source)
        assert _wait(host, wake).phase == "done"
        assert seen == [4.0, 6.0]
    finally:
        host.shutdown()
        plane.close()
