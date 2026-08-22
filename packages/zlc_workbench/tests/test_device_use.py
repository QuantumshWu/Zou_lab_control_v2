"""Device-use admission and maintenance are one key-scoped truth."""

from __future__ import annotations

import pytest

from zlc_workbench.device_use import (
    DeviceClaim,
    DeviceUseBusy,
    DeviceUseCoordinator,
)


def _claim(key: str, device: object) -> DeviceClaim:
    return DeviceClaim("device", key, device)


def test_same_device_key_conflicts_even_when_objects_differ() -> None:
    coordinator = DeviceUseCoordinator()
    first = coordinator.acquire_command(
        object(),
        "first command",
        (_claim("camera", object()),),
    )

    with pytest.raises(DeviceUseBusy, match="first command"):
        coordinator.acquire_command(
            object(),
            "replacement command",
            (_claim("camera", object()),),
        )

    assert first.release() is True
    coordinator.assert_idle()


def test_maintenance_blocks_new_logic_and_commands_for_its_keys() -> None:
    coordinator = DeviceUseCoordinator()
    barrier = coordinator.begin_maintenance(
        object(),
        "camera maintenance",
        ("camera",),
    )

    with pytest.raises(DeviceUseBusy, match="camera maintenance"):
        coordinator.prepare_logic(
            object(),
            "new camera logic",
            (_claim("camera", object()),),
            stop=lambda _reason: None,
            superseded=lambda: None,
        )
    with pytest.raises(DeviceUseBusy, match="camera maintenance"):
        coordinator.acquire_command(
            object(),
            "new camera command",
            (_claim("camera", object()),),
        )

    assert barrier.release() is True
    coordinator.assert_idle()


def test_maintenance_stops_existing_logic_and_waits_for_lease_release() -> None:
    coordinator = DeviceUseCoordinator()
    stopped: list[str] = []
    reservation = coordinator.prepare_logic(
        object(),
        "camera measurement",
        (_claim("camera", object()),),
        stop=stopped.append,
        superseded=lambda: None,
    )
    lease = reservation.commit()

    barrier = coordinator.begin_maintenance(
        object(),
        "camera maintenance",
        ("camera",),
    )

    assert stopped == ["camera maintenance needs camera"]
    assert barrier.waiting_for == ("camera measurement",)
    assert lease.release() is True
    barrier.wait(timeout=0.1)
    assert barrier.waiting_for == ()
    assert barrier.release() is True
    coordinator.assert_idle()


def test_existing_command_refuses_maintenance_without_leaving_a_barrier() -> None:
    coordinator = DeviceUseCoordinator()
    command = coordinator.acquire_command(
        object(),
        "camera control",
        (_claim("camera", object()),),
    )

    with pytest.raises(DeviceUseBusy, match="camera control"):
        coordinator.begin_maintenance(
            object(),
            "camera maintenance",
            ("camera",),
        )

    assert command.release() is True
    barrier = coordinator.begin_maintenance(
        object(),
        "camera maintenance",
        ("camera",),
    )
    assert barrier.release() is True
    coordinator.assert_idle()
