"""A Tektronix oscilloscope as a waveform source, over its SCPI programmer interface.

The DPO, MSO, MDO and TDS families share one command vocabulary for what
this needs: the horizontal scale, a channel's vertical scale, a single
sequence acquisition, and the waveform transfer preamble plus ``CURVe?``.
The driver is written against a four-verb link so the transport is the
only thing a test or a virtual bench stands in for; the vocabulary, the
read-back discipline and the scaling arithmetic run as shipped.

One record is one acquisition: the driver arms a single sequence, waits
for the scope to trigger and stop, then transfers every configured
channel's curve and scales it to volts with the preamble the scope gave for
that channel.  Whatever the scope is set to trigger on is what it triggers
on -- the trigger is the operator's on the front panel, and a measurement
that wants every shot puts the sequencer's edge on it.

The two knobs an operator reaches for -- time per division and each
channel's volts per division -- are tunable through the ordinary device
form, read back from the scope after every write because a scope snaps to
its own 1-2-5 steps, and frozen for the length of any capture.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Protocol
from uuid import uuid4

import numpy as np

from zlc_atom.authoring import AuthoringField, TunableField
from zlc_atom.devices.rf.rigol_dg4000 import (
    VisaResources,
    identity_fields,
    probeable_resources,
    visa_resources,
)
from zlc_atom.devices.waveform.contract import (
    WaveformAcquisitionMode,
    WaveformCaptureTerminalRecord,
    WaveformOutput,
    WaveformRecord,
    WaveformWorkingPoint,
)


class ScopeLink(Protocol):
    """The whole transport surface a SCPI oscilloscope needs."""

    def write(self, command: str) -> None: ...

    def query(self, command: str) -> str: ...

    def query_int16(self, command: str) -> np.ndarray: ...

    def close(self) -> None: ...


class VisaScopeLink:
    """A pyvisa resource behind the four-verb link."""

    def __init__(self, resource: str, *, timeout_seconds: float = 5.0) -> None:
        if not isinstance(resource, str) or not resource.strip():
            raise ValueError("VISA resource name is required")
        self._resource = visa_resources().open_resource(resource.strip())
        self._resource.timeout = int(float(timeout_seconds) * 1000.0)
        self._resource.read_termination = "\n"
        self._resource.write_termination = "\n"

    def write(self, command: str) -> None:
        self._resource.write(command)

    def query(self, command: str) -> str:
        return str(self._resource.query(command))

    def query_int16(self, command: str) -> np.ndarray:
        return np.asarray(
            self._resource.query_binary_values(
                command, datatype="h", is_big_endian=True, container=np.array
            ),
            dtype=np.int16,
        )

    def close(self) -> None:
        self._resource.close()


_IDENTITY_VENDOR = "TEKTRONIX"
PROBE_TIMEOUT_SECONDS = 1.0


def is_tektronix(identity: str) -> bool:
    """Whether this ``*IDN?`` answer is a scope this driver can drive."""

    fields = identity_fields(identity)
    return len(fields) >= 2 and _IDENTITY_VENDOR in fields[0].upper()


def discover_tek_scopes(
    resources: VisaResources | None = None,
    *,
    timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
) -> tuple[tuple[str, str], ...]:
    """Every Tektronix scope attached, as ``(resource, identity)`` pairs.

    Found the way a signal generator is found: open each listed resource,
    ask the one universal question, close it.  What does not answer is
    passed over.
    """

    manager = visa_resources() if resources is None else resources
    milliseconds = max(1, int(float(timeout_seconds) * 1000.0))
    found: list[tuple[str, str]] = []
    for name in probeable_resources(manager.list_resources()):
        try:
            session = manager.open_resource(name, open_timeout=milliseconds)
        except Exception:
            continue
        try:
            session.timeout = milliseconds
            identity = str(session.query("*IDN?")).strip()
        except Exception:
            continue
        finally:
            try:
                session.close()
            except Exception:
                pass
        if is_tektronix(identity):
            found.append((name, identity))
    return tuple(found)


@dataclass(frozen=True)
class TekScopeConfig:
    """Where the scope is, and which of its channels a record carries."""

    resource: str
    channels: tuple[int, ...] = (1,)
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        resource = str(self.resource).strip()
        if not resource:
            raise ValueError("VISA resource name is required")
        channels = tuple(int(channel) for channel in self.channels)
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must name at least one channel by its positive number")
        if len(set(channels)) != len(channels):
            raise ValueError("channels must be distinct")
        if float(self.timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be positive")
        object.__setattr__(self, "resource", resource)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))


TIME_PER_DIV_FIELD = "time_per_div"


def volts_per_div_field(channel: int) -> str:
    return f"ch{int(channel)}_volts_per_div"


#: How often an armed worker asks the scope whether the sequence has
#: finished.  Between asks it is a Stop that waits, not the scope.
_ACQUISITION_POLL_SECONDS = 0.005


class TekScopeWaveformSource:
    """One single-sequence acquisition per record, every configured channel a column."""

    def __init__(self, config: TekScopeConfig, *, link: ScopeLink | None = None) -> None:
        self.config = config
        self._link: ScopeLink = (
            VisaScopeLink(config.resource, timeout_seconds=config.timeout_seconds)
            if link is None
            else link
        )
        try:
            identity = self._link.query("*IDN?").strip()
            if not is_tektronix(identity):
                raise RuntimeError(
                    f"{config.resource} answered *IDN? with {identity!r}, which is "
                    "not a Tektronix oscilloscope"
                )
            self._identity = identity
            self._link.write(":HEADer OFF")
            self._link.write(":VERBose OFF")
            self._link.write(":DATa:ENCdg RIBinary")
            self._link.write(":DATa:WIDth 2")
            record_length = int(float(self._link.query(":HORizontal:RECOrdlength?")))
            self._link.write(":DATa:STARt 1")
            self._link.write(f":DATa:STOP {record_length}")
        except BaseException:
            self._link.close()
            raise
        self._condition = threading.Condition()
        self._device_session_id = uuid4().hex
        self._settings_epoch = 0
        self._queue: deque[WaveformRecord] = deque()
        self._armed = False
        self._accepting = False
        self._expected: int | None = None
        self._buffer_record_count = 1
        self._next_ordinal = 0
        self._produced_count = 0
        self._worker: threading.Thread | None = None
        self._worker_stop: threading.Event | None = None
        self._worker_error: BaseException | None = None
        self._terminal: WaveformCaptureTerminalRecord | None = None

    # ------------------------------------------------------------- readback
    @property
    def timeout(self) -> float:
        return self.config.timeout_seconds

    @property
    def identity(self) -> str:
        fields = identity_fields(self._identity)
        serial = fields[2] if len(fields) > 2 and fields[2] else self.config.resource
        return f"tek-scope:{serial}"

    def _query_float(self, command: str) -> float:
        return float(self._link.query(command).strip())

    def _select(self, channel: int) -> None:
        self._link.write(f":DATa:SOUrce CH{int(channel)}")

    def working_point(self) -> WaveformWorkingPoint:
        with self._condition:
            time_per_div = self._query_float(":HORizontal:SCAle?")
            self._select(self.config.channels[0])
            interval = self._query_float(":WFMOutpre:XINcr?")
            points = int(float(self._link.query(":WFMOutpre:NR_Pt?")))
            settings: dict[str, object] = {
                "resource": self.config.resource,
                "identity": self._identity,
                TIME_PER_DIV_FIELD: time_per_div,
                "record_length": points,
            }
            for channel in self.config.channels:
                settings[volts_per_div_field(channel)] = self._query_float(
                    f":CH{channel}:SCAle?"
                )
        return WaveformWorkingPoint(
            WaveformAcquisitionMode.EXTERNAL_TRIGGERED,
            interval,
            points,
            (
                WaveformOutput(
                    "voltage",
                    "V",
                    tuple(f"CH{channel}" for channel in self.config.channels),
                    tuple(range(len(self.config.channels))),
                ),
            ),
            settings,
        )

    # -------------------------------------------------------------- knobs
    def tunable_fields(self) -> tuple[TunableField, ...]:
        with self._condition:
            fields = [
                TunableField(
                    metadata=AuthoringField(
                        TIME_PER_DIV_FIELD,
                        "float",
                        "Time per division",
                        None,
                        minimum=1e-12,
                        maximum=1e3,
                        unit="s",
                    ),
                    current=self._query_float(":HORizontal:SCAle?"),
                    live_write=True,
                    dependency_group=(TIME_PER_DIV_FIELD,),
                )
            ]
            for channel in self.config.channels:
                name = volts_per_div_field(channel)
                fields.append(
                    TunableField(
                        metadata=AuthoringField(
                            name,
                            "float",
                            f"CH{channel} volts per division",
                            None,
                            minimum=1e-6,
                            maximum=1e3,
                            unit="V",
                        ),
                        current=self._query_float(f":CH{channel}:SCAle?"),
                        live_write=True,
                        dependency_group=(name,),
                    )
                )
        return tuple(fields)

    def tunable_values(self) -> dict[str, float]:
        return {
            field.metadata.name: float(field.current) for field in self.tunable_fields()
        }

    def settings_provenance(self) -> dict[str, object]:
        with self._condition:
            return {
                "device_session_id": self._device_session_id,
                "settings_epoch": self._settings_epoch,
            }

    def tune(self, name: str, value: float) -> float:
        """Write one knob and answer with what the scope snapped it to."""

        selected = str(name)
        requested = float(value)
        if not np.isfinite(requested) or requested <= 0.0:
            raise ValueError(f"{selected} must be finite and positive")
        if selected == TIME_PER_DIV_FIELD:
            command, readback = ":HORizontal:SCAle", ":HORizontal:SCAle?"
        else:
            channel = next(
                (
                    channel
                    for channel in self.config.channels
                    if volts_per_div_field(channel) == selected
                ),
                None,
            )
            if channel is None:
                raise ValueError(
                    f"this scope has no tunable field {selected!r}; it offers "
                    f"{TIME_PER_DIV_FIELD!r} and "
                    + ", ".join(repr(volts_per_div_field(c)) for c in self.config.channels)
                )
            command, readback = f":CH{channel}:SCAle", f":CH{channel}:SCAle?"
        with self._condition:
            if self._armed:
                raise RuntimeError("scope settings cannot change while a capture is armed")
            self._link.write(f"{command} {requested:.9g}")
            actual = self._query_float(readback)
            self._settings_epoch += 1
            return actual

    # ------------------------------------------------------------ capture
    def arm(
        self, records: int | None, *, buffer_record_count: int, timeout: float
    ) -> None:
        buffer_count = int(buffer_record_count)
        if buffer_count <= 0:
            raise ValueError("buffer_record_count must be positive")
        expected = None if records is None else int(records)
        if expected is not None and (expected <= 0 or buffer_count != expected):
            raise ValueError("a finite arm buffers exactly the records it expects")
        with self._condition:
            if self._armed:
                raise RuntimeError("the scope is already armed")
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("the previous scope acquisition worker is still running")
            # The vertical scaling of every channel, read once: the knobs
            # are frozen for the whole capture, so the preamble cannot
            # change under a record.
            scaling: list[tuple[int, float, float, float]] = []
            for channel in self.config.channels:
                self._select(channel)
                scaling.append(
                    (
                        channel,
                        self._query_float(":WFMOutpre:YMUlt?"),
                        self._query_float(":WFMOutpre:YOFf?"),
                        self._query_float(":WFMOutpre:YZEro?"),
                    )
                )
            points = int(float(self._link.query(":WFMOutpre:NR_Pt?")))
            self._queue.clear()
            self._armed = True
            self._accepting = True
            self._expected = expected
            self._buffer_record_count = buffer_count
            self._next_ordinal = 0
            self._produced_count = 0
            self._worker_error = None
            self._terminal = None
            stop = threading.Event()
            worker = threading.Thread(
                target=self._acquire,
                args=(stop, tuple(scaling), points, float(timeout)),
                name="zlc-tek-scope-acquisition",
                daemon=True,
            )
            self._worker_stop = stop
            self._worker = worker
            try:
                worker.start()
            except BaseException:
                self._worker = None
                self._worker_stop = None
                self._armed = False
                self._accepting = False
                raise

    def _acquire(
        self,
        stop: threading.Event,
        scaling: tuple[tuple[int, float, float, float], ...],
        points: int,
        timeout: float,
    ) -> None:
        del timeout
        try:
            while not stop.is_set():
                with self._condition:
                    if not self._accepting:
                        break
                    self._link.write(":ACQuire:STOPAfter SEQuence")
                    self._link.write(":ACQuire:STATE RUN")
                # The scope triggers when its trigger says so; until then
                # only a Stop is worth waking for.
                while not stop.is_set():
                    with self._condition:
                        if not self._accepting:
                            break
                        state = self._link.query(":ACQuire:STATE?").strip()
                    if state in ("0", "STOP"):
                        break
                    time.sleep(_ACQUISITION_POLL_SECONDS)
                else:
                    break
                with self._condition:
                    if stop.is_set() or not self._accepting:
                        break
                    columns = []
                    for channel, multiplier, offset, zero in scaling:
                        self._select(channel)
                        raw = self._link.query_int16(":CURVe?")
                        if raw.size != points:
                            raise RuntimeError(
                                f"CH{channel} returned {raw.size} points of a "
                                f"{points}-point record"
                            )
                        columns.append(
                            (raw.astype(np.float32) - np.float32(offset))
                            * np.float32(multiplier)
                            + np.float32(zero)
                        )
                    record = WaveformRecord(
                        np.stack(columns, axis=1),
                        self._next_ordinal,
                        time.time_ns(),
                    )
                    self._next_ordinal += 1
                    self._produced_count += 1
                    while len(self._queue) >= self._buffer_record_count:
                        self._queue.popleft()
                    self._queue.append(record)
                    if self._expected is not None and self._produced_count >= self._expected:
                        self._accepting = False
                    self._condition.notify_all()
        except BaseException as error:  # noqa: BLE001 -- surfaced to the reader of records
            with self._condition:
                self._worker_error = error
                self._accepting = False
                self._condition.notify_all()
        finally:
            with self._condition:
                if self._worker is threading.current_thread():
                    self._worker = None
                    self._worker_stop = None
                self._condition.notify_all()

    def read_records(
        self, n: int, *, timeout: float, exact: bool
    ) -> list[WaveformRecord]:
        requested = int(n)
        if requested <= 0:
            raise ValueError("n must be positive")
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while len(self._queue) < requested:
                if self._worker_error is not None:
                    raise RuntimeError("the scope acquisition worker failed") from self._worker_error
                running = self._worker is not None and self._worker.is_alive()
                if not self._armed or (not running and not self._accepting):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            if exact and len(self._queue) < requested:
                raise TimeoutError("the scope did not deliver the requested exact records")
            count = requested if exact else min(requested, len(self._queue))
            return [self._queue.popleft() for _ in range(count)]

    def finish_record_capture(self) -> WaveformCaptureTerminalRecord:
        with self._condition:
            if self._terminal is not None:
                return self._terminal
            if not self._armed:
                self._terminal = WaveformCaptureTerminalRecord(0, True, True, True)
                return self._terminal
            self._accepting = False
            worker = self._worker
            stop = self._worker_stop
            if stop is not None:
                stop.set()
            self._condition.notify_all()
        if worker is not None:
            worker.join(timeout=self.config.timeout_seconds + 1.0)
        with self._condition:
            if worker is not None and worker.is_alive():
                raise RuntimeError("the scope acquisition worker did not join")
            self._armed = False
            self._terminal = WaveformCaptureTerminalRecord(
                self._produced_count, True, not self._queue, True
            )
            self._condition.notify_all()
            if self._worker_error is not None:
                raise RuntimeError("the scope acquisition worker failed") from self._worker_error
            return self._terminal

    def capture_state(self) -> bool:
        with self._condition:
            return self._armed

    def close(self) -> None:
        if self.capture_state():
            self.finish_record_capture()
        with self._condition:
            self._queue.clear()
        self._link.close()


__all__ = [
    "PROBE_TIMEOUT_SECONDS",
    "ScopeLink",
    "TIME_PER_DIV_FIELD",
    "TekScopeConfig",
    "TekScopeWaveformSource",
    "VisaScopeLink",
    "discover_tek_scopes",
    "is_tektronix",
    "volts_per_div_field",
]
