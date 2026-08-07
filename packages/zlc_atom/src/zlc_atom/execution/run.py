"""Small safety-shaped finite run engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
import time
import uuid
from typing import Any, Callable, Protocol, runtime_checkable

from .ports import BoundDevice, DeviceBroker, SafetyInterrupt
from .resources import ResourceArbiter, ResourceBusy, ResourceClaim, ResourceKey, ResourceLease


class RunState(str, Enum):
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@runtime_checkable
class RunHandleLike(Protocol):
    def snapshot(self) -> RunSnapshot: ...
    def cancel(self, reason: str = "cancelled") -> bool: ...
    def result(self, timeout: float | None = None) -> Any: ...


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    state: RunState
    phase: str
    error: str | None = None


class SafetyInterruptError(RuntimeError):
    pass


class RunDevice:
    """Run-scoped execution proxy; retaining it after safety is harmless."""

    __slots__ = ("_context", "_binding")

    def __init__(self, context: "RunContext", binding: BoundDevice) -> None:
        self._context = context
        self._binding = binding

    @property
    def key(self) -> ResourceKey:
        return self._binding.key

    def execute(self, command: object) -> object:
        return self._context._execute_bound_device(self._binding, command)


class RunContext:
    def __init__(
        self,
        handle: "RunHandle",
        broker: DeviceBroker | None,
        devices: tuple[BoundDevice, ...],
        interrupts: tuple[SafetyInterrupt, ...] = (),
    ) -> None:
        self.handle = handle
        self.broker = broker
        self._pending_devices = tuple(devices)
        self.devices: dict[object, BoundDevice] = {}
        self._device_lease = None
        self._hardware_condition = threading.Condition(threading.RLock())
        self._execution_enabled = False
        self._execution_sealed = False
        self._hardware_revoked = False
        self._hardware_inflight = 0
        self._interrupt_operations = tuple(interrupts)
        self._interrupt_pending = False
        self._interrupt_dispatched = False

    @property
    def cancel_requested(self) -> bool:
        return self.handle.cancel_requested

    def checkpoint(self) -> None:
        if self.cancel_requested:
            raise RuntimeError("run cancellation requested")

    def _bind_devices(self) -> None:
        if self._device_lease is not None or self.devices:
            raise RuntimeError("Run devices are already bound")
        if self._pending_devices and self.broker is None:
            raise RuntimeError("Run devices require a DeviceBroker")
        if self.broker is not None:
            self._device_lease = self.broker.open_run(self.handle.run_id, self._pending_devices)
        self.devices = {device.key: device for device in self._pending_devices}
        self._pending_devices = ()

    def _execute_bound_device(self, binding: BoundDevice, command: object) -> object:
        with self._hardware_condition:
            if (
                not self._execution_enabled
                or self._execution_sealed
                or self._hardware_revoked
                or self.cancel_requested
            ):
                raise RuntimeError("Run has no active execution hardware capability")
            if self.devices.get(binding.key) is not binding:
                raise RuntimeError("device binding does not belong to this Run")
            lease = self._device_lease
            if lease is None:
                raise RuntimeError("Run has no device lease")
            self._hardware_inflight += 1
        try:
            return lease.execute(binding, command)
        finally:
            with self._hardware_condition:
                self._hardware_inflight -= 1
                self._hardware_condition.notify_all()

    def _run_interrupts(self, interrupts: tuple[SafetyInterrupt, ...]) -> None:
        errors: list[BaseException] = []
        with self._hardware_condition:
            self._interrupt_dispatched = True
        try:
            for item in interrupts:
                with self._hardware_condition:
                    if self._hardware_revoked or self._device_lease is None:
                        continue
                    binding = self.devices.get(item.key)
                    lease = self._device_lease
                    if binding is None:
                        errors.append(RuntimeError(f"Run does not own interrupt device {item.key}"))
                        continue
                    self._hardware_inflight += 1
                try:
                    lease.interrupt(binding, item.operation)
                except BaseException as error:
                    errors.append(error)
                finally:
                    with self._hardware_condition:
                        self._hardware_inflight -= 1
                        self._hardware_condition.notify_all()
        finally:
            with self._hardware_condition:
                self._interrupt_pending = False
                self._hardware_condition.notify_all()
        if errors:
            raise SafetyInterruptError(
                "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            ) from errors[0]

    def _seal_execution_on_cancel(self) -> None:
        with self._hardware_condition:
            self._execution_sealed = True
            self._execution_enabled = False
            if self.cancel_requested and self._interrupt_operations and not self._interrupt_dispatched:
                self._interrupt_pending = True
            self._hardware_condition.notify_all()

    def _enable_execution(self) -> bool:
        with self._hardware_condition:
            if self._hardware_revoked:
                raise RuntimeError("Run hardware was already revoked")
            if self.cancel_requested or self._execution_sealed:
                self._execution_sealed = True
                return False
            self._execution_enabled = True
            return True

    def _revoke_hardware(self) -> None:
        with self._hardware_condition:
            self._execution_enabled = False
            self._execution_sealed = True
            while self._interrupt_pending:
                self._hardware_condition.wait()
            self._hardware_revoked = True
            while self._hardware_inflight:
                self._hardware_condition.wait()
            lease = self._device_lease
            self._device_lease = None
            self.devices.clear()
            self._pending_devices = ()
        if lease is not None:
            lease.revoke()

    def device(self, key: object) -> RunDevice:
        try:
            return RunDevice(self, self.devices[key])
        except KeyError as exc:
            raise KeyError(f"run does not own device {key!r}") from exc


@dataclass(frozen=True)
class RunPlan:
    name: str
    resource_claims: tuple[ResourceClaim, ...]
    preflight: Callable[[RunContext], Any]
    execute: Callable[[RunContext, Any], Any]
    cleanup: Callable[[RunContext, Any, BaseException | None], None]
    finalize: Callable[[Any, Any], Any]
    bound_devices: tuple[BoundDevice, ...] = ()
    interrupt_operations: tuple[SafetyInterrupt, ...] = ()
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        claims = tuple(self.resource_claims)
        if any(not isinstance(claim, ResourceClaim) for claim in claims):
            raise TypeError("resource_claims must contain ResourceClaim values")
        object.__setattr__(self, "resource_claims", claims)
        devices = tuple(self.bound_devices)
        if any(not isinstance(device, BoundDevice) for device in devices):
            raise TypeError("bound_devices must contain BoundDevice values")
        if len({device.key for device in devices}) != len(devices):
            raise ValueError("bound_devices must have unique ResourceKeys")
        claimed_device_keys = {
            claim.key
            for claim in claims
            if claim.key.segments[0] == "device"
        }
        if {device.key for device in devices} != claimed_device_keys:
            raise ValueError("every device claim requires exactly one BoundDevice")
        object.__setattr__(self, "bound_devices", devices)
        interrupts = tuple(self.interrupt_operations)
        if any(not isinstance(item, SafetyInterrupt) for item in interrupts):
            raise TypeError("interrupt_operations must contain SafetyInterrupt values")
        if len(set(interrupts)) != len(interrupts):
            raise ValueError("interrupt_operations cannot contain duplicates")
        if any(item.key not in {device.key for device in devices} for item in interrupts):
            raise ValueError("interrupt operation must target a bound device")
        object.__setattr__(self, "interrupt_operations", interrupts)
        for field, value in (
            ("preflight", self.preflight),
            ("execute", self.execute),
            ("cleanup", self.cleanup),
            ("finalize", self.finalize),
        ):
            if not callable(value):
                raise TypeError(f"{field} must be callable")


class RunHandle:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._condition = threading.Condition()
        self._state = RunState.RUNNING
        self._phase = "starting"
        self._error: BaseException | None = None
        self._result: Any = _MISSING
        self._cancel_requested = False
        self._interrupt_callback: Callable[[], object] | None = None
        self._interrupt_error: BaseException | None = None

    @property
    def cancel_requested(self) -> bool:
        with self._condition:
            return self._cancel_requested

    def snapshot(self) -> RunSnapshot:
        with self._condition:
            return RunSnapshot(self.run_id, self._state, self._phase, None if self._error is None else str(self._error))

    def cancel(self, reason: str = "cancelled") -> bool:
        del reason
        with self._condition:
            if self._state in {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}:
                return False
            self._cancel_requested = True
            self._state = RunState.CANCELLING
            self._condition.notify_all()
        callback = self._interrupt_callback
        if callback is not None:
            try:
                callback()
            except BaseException as exc:
                with self._condition:
                    self._interrupt_error = exc
                    self._condition.notify_all()
        return True

    def result(self, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while self._state not in {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}:
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("run result timed out")
                    self._condition.wait(remaining)
            if self._state is RunState.FAILED and self._error is not None:
                raise self._error
            if self._state is RunState.CANCELLED:
                raise RuntimeError("run was cancelled")
            return self._result

    def _set_phase(self, phase: str) -> None:
        with self._condition:
            self._phase = str(phase)

    def _set_interrupt_callback(self, callback: Callable[[], object] | None) -> None:
        self._interrupt_callback = callback


_MISSING = object()


class _RunController:
    def __init__(self, *, arbiter: ResourceArbiter | None = None, broker: DeviceBroker | None = None) -> None:
        self.arbiter = arbiter or ResourceArbiter()
        self.broker = broker

    def start(self, plan: RunPlan) -> RunHandle:
        run_id = uuid.uuid4().hex
        acquisition = self.arbiter.acquire_all(run_id, plan.resource_claims)
        if isinstance(acquisition, tuple):
            raise RuntimeError(f"run resources are busy: {acquisition}")
        handle = RunHandle(run_id)
        context = RunContext(handle, self.broker, plan.bound_devices, plan.interrupt_operations)
        if self.broker is not None and plan.interrupt_operations:
            def interrupt() -> None:
                context._run_interrupts(plan.interrupt_operations)
            handle._set_interrupt_callback(interrupt)
        thread = threading.Thread(target=self._worker, args=(handle, plan, acquisition, context), daemon=True)
        thread.start()
        return handle

    def _worker(self, handle: RunHandle, plan: RunPlan, lease: ResourceLease, context: RunContext) -> None:
        prepared = _MISSING
        error: BaseException | None = None
        result: Any = _MISSING
        try:
            context._bind_devices()
            context._enable_execution()
            handle._set_phase("preflight")
            prepared = plan.preflight(context)
            context.checkpoint()
            handle._set_phase("execute")
            executed = plan.execute(context, prepared)
            context.checkpoint()
            handle._set_phase("finalize")
            result = plan.finalize(prepared, executed)
            context.checkpoint()
        except BaseException as exc:
            error = exc
        finally:
            cleanup_error: BaseException | None = None
            try:
                handle._set_phase("cleanup")
                context._seal_execution_on_cancel()
                plan.cleanup(context, None if prepared is _MISSING else prepared, error)
            except BaseException as exc:
                cleanup_error = exc
            finally:
                try:
                    context._revoke_hardware()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
                finally:
                    try:
                        lease.release()
                    except BaseException as exc:
                        cleanup_error = cleanup_error or exc
            terminal_error = error or cleanup_error
            with handle._condition:
                handle._error = terminal_error
                handle._result = _MISSING if terminal_error is not None or handle.cancel_requested else result
                if handle.cancel_requested:
                    handle._state = RunState.CANCELLED
                elif terminal_error is not None:
                    handle._state = RunState.FAILED
                else:
                    handle._state = RunState.SUCCEEDED
                handle._phase = "done"
                handle._condition.notify_all()


def run_plan(plan: RunPlan, *, arbiter: ResourceArbiter | None = None, broker: DeviceBroker | None = None) -> RunHandle:
    return _RunController(arbiter=arbiter, broker=broker).start(plan)


__all__ = ["RunContext", "RunDevice", "RunHandle", "RunHandleLike", "RunPlan", "RunSnapshot", "RunState", "SafetyInterruptError", "run_plan"]
