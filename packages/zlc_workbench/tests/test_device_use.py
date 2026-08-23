"""Device-use admission and maintenance are one key-scoped truth."""

from __future__ import annotations

import pytest

from zlc_workbench.device_use import (
    DeviceClaim,
    DeviceUseBusy,
    DeviceUseCoordinator,
)


def _claim(
    key: str,
    device: object,
    protected_fields: tuple[str, ...] = (),
    *,
    exclusive: bool = True,
) -> DeviceClaim:
    return DeviceClaim("device", key, device, protected_fields, exclusive)


def test_field_policy_projects_logic_claims_and_dependency_closure() -> None:
    coordinator = DeviceUseCoordinator()
    reservation = coordinator.prepare_logic(
        object(),
        "camera measurement",
        (_claim("camera", object(), ("exposure_seconds", "roi_x")),),
        stop=lambda _reason: None,
        superseded=lambda: None,
    )
    assert coordinator.field_policy(
        "camera", ("exposure_seconds", "gain_db")
    ) == (0, (), {"exposure_seconds": (), "gain_db": ()})

    lease = reservation.commit()
    revision, owners, policy = coordinator.field_policy(
        "camera",
        ("exposure_seconds", "gain_db", "roi_width", "roi_height"),
        dependency_groups=(
            ("roi_x", "roi_width"),
            ("roi_width", "roi_height"),
        ),
    )
    assert revision == 1
    assert owners == ("camera measurement",)
    assert policy == {
        "exposure_seconds": ("camera measurement",),
        "gain_db": (),
        "roi_width": ("camera measurement",),
        "roi_height": ("camera measurement",),
    }

    assert lease.release() is True
    assert coordinator.field_policy("camera", ("exposure_seconds",)) == (
        2,
        (),
        {"exposure_seconds": ()},
    )


def test_field_command_atomically_rejects_stale_risk_and_claimed_fields() -> None:
    coordinator = DeviceUseCoordinator()
    device = object()
    first = coordinator.prepare_logic(
        object(),
        "first camera run",
        (_claim("camera", device, ("exposure_seconds",)),),
        stop=lambda _reason: None,
        superseded=lambda: None,
    ).commit()
    accepted_revision, owners, _policy = coordinator.field_policy(
        "camera", ("gain_db",)
    )
    assert owners == ("first camera run",)
    first.release()
    second = coordinator.prepare_logic(
        object(),
        "second camera run",
        (_claim("camera", device, ("exposure_seconds",)),),
        stop=lambda _reason: None,
        superseded=lambda: None,
    ).commit()

    with pytest.raises(DeviceUseBusy, match="owners changed"):
        coordinator.acquire_field_command(
            object(),
            "stale gain write",
            _claim("camera", device, ("gain_db",)),
            expected_owner_revision=accepted_revision,
            allow_while_logic=True,
        )

    revision, _owners, _policy = coordinator.field_policy(
        "camera", ("gain_db", "exposure_seconds")
    )
    with pytest.raises(DeviceUseBusy, match="second camera run"):
        coordinator.acquire_field_command(
            object(),
            "claimed exposure write",
            _claim("camera", device, ("exposure_seconds",)),
            expected_owner_revision=revision,
            allow_while_logic=True,
        )
    gain_owner = object()
    gain = coordinator.acquire_field_command(
        gain_owner,
        "accepted gain write",
        _claim("camera", device, ("gain_db",)),
        expected_owner_revision=revision,
        allow_while_logic=True,
    )
    with pytest.raises(DeviceUseBusy, match="accepted gain write"):
        coordinator.prepare_logic(
            object(),
            "third camera run",
            (_claim("camera", device),),
            stop=lambda _reason: None,
            superseded=lambda: None,
        )
    gain.release()
    second.release()
    coordinator.assert_idle()


def test_runtime_selected_field_protection_does_not_change_exclusive_admission() -> None:
    coordinator = DeviceUseCoordinator()
    camera = object()
    measurement = coordinator.prepare_logic(
        object(),
        "camera measurement",
        (_claim("camera", camera, ("exposure_seconds",)),),
        stop=lambda _reason: None,
        superseded=lambda: None,
    ).commit()
    scan = coordinator.prepare_logic(
        object(),
        "stepped scan",
        (
            _claim(
                "camera",
                camera,
                ("gain_db",),
                exclusive=False,
            ),
            _claim("sequencer", object(), ("program",)),
        ),
        stop=lambda _reason: None,
        superseded=lambda: None,
    ).commit()
    _revision, owners, policy = coordinator.field_policy(
        "camera", ("exposure_seconds", "gain_db")
    )
    assert owners == ("camera measurement", "stepped scan")
    assert policy == {
        "exposure_seconds": ("camera measurement",),
        "gain_db": ("stepped scan",),
    }
    scan.release()
    measurement.release()
    coordinator.assert_idle()


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
