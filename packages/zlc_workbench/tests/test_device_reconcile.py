from __future__ import annotations

from dataclasses import dataclass

import pytest

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.install import (
    DeviceCatalogSnapshot,
    DeviceTypeDescriptor,
    InstalledLeaf,
)
from zlc_atom.install.configuration import DeviceInstanceConfig, InstallationConfig
from zlc_workbench.session import ExperimentSession


@dataclass
class _Device:
    key: str
    serial: int


def _catalog(events: list[str]) -> DeviceCatalogSnapshot:
    serial = 0

    def factory(context, key, _config):
        nonlocal serial
        serial += 1
        device = _Device(key, serial)
        return InstalledLeaf(
            key,
            "test.base",
            device,
            {},
            closer=lambda key=key: events.append(f"close:{key}"),
        )

    def dependent(context, key, _config):
        nonlocal serial
        serial += 1
        assert "base" in context.devices
        device = _Device(key, serial)
        return InstalledLeaf(
            key,
            "test.dependent",
            device,
            {},
            closer=lambda key=key: events.append(f"close:{key}"),
        )

    return DeviceCatalogSnapshot(
        (
            DeviceTypeDescriptor(
                "test.base",
                "test",
                AuthoringSchema(()),
                (),
                factory=factory,
            ),
            DeviceTypeDescriptor(
                "test.dependent",
                "test",
                AuthoringSchema(()),
                (),
                dependencies=("test.base",),
                factory=dependent,
            ),
        ),
        (),
    )


def _device(key: str, *, role: str | None = None, value: int = 0, dependent=False):
    return DeviceInstanceConfig(
        key,
        role or key,
        "test.dependent" if dependent else "test.base",
        {"value": value},
    )


def test_reconcile_reuses_unchanged_leaf_and_only_builds_added_device(tmp_path):
    events: list[str] = []
    catalog = _catalog(events)
    initial = InstallationConfig((_device("base"),))
    session = ExperimentSession.from_config(tmp_path, initial, catalog=catalog)
    original = session.installation.device("base")

    wanted = InstallationConfig(
        (_device("base", role="renamed"), _device("other"))
    )
    plan = session.plan_device_reconcile(wanted)
    assert plan.retained_keys == ("base",)
    assert plan.build_keys == ("other",)

    session.reconcile_devices(plan)

    assert session.installation.device("base") is original
    assert set(session.installation.devices) == {"base", "other"}
    assert events == []
    session.close()
    assert events == ["close:other", "close:base"]


def test_reconcile_parameter_change_rebuilds_only_that_leaf(tmp_path):
    events: list[str] = []
    catalog = _catalog(events)
    initial = InstallationConfig((_device("base"), _device("other")))
    session = ExperimentSession.from_config(tmp_path, initial, catalog=catalog)
    original_base = session.installation.device("base")
    original_other = session.installation.device("other")

    wanted = InstallationConfig((_device("base", value=2), _device("other")))
    session.reconcile_devices(session.plan_device_reconcile(wanted))

    assert session.installation.device("base") is not original_base
    assert session.installation.device("other") is original_other
    assert events == ["close:base"]
    session.close()


def test_close_device_also_closes_factory_dependants_and_does_not_reopen(tmp_path):
    events: list[str] = []
    catalog = _catalog(events)
    config = InstallationConfig((_device("base"), _device("consumer", dependent=True)))
    session = ExperimentSession.from_config(tmp_path, config, catalog=catalog)

    plan = session.plan_device_reconcile(config, close_keys=frozenset({"base"}))
    assert set(plan.affected_keys) == {"base", "consumer"}
    assert plan.build_keys == ()
    session.reconcile_devices(plan)

    assert session.installation.devices == {}
    assert events == ["close:consumer", "close:base"]
    session.close()


def test_failed_close_remains_reachable_for_retry(tmp_path):
    events: list[str] = []
    catalog = _catalog(events)
    config = InstallationConfig((_device("base"),))
    session = ExperimentSession.from_config(tmp_path, config, catalog=catalog)
    leaf = session.installation.devices["base"]

    attempts = 0

    def flaky_close():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("still open")
        events.append("close:base")

    object.__setattr__(leaf, "closer", flaky_close)
    plan = session.plan_device_reconcile(config, close_keys=frozenset({"base"}))
    with pytest.raises(ExceptionGroup, match="installation close failed"):
        session.reconcile_devices(plan)

    assert session.installation.device("base") is leaf.device
    session.reconcile_devices(
        session.plan_device_reconcile(config, close_keys=frozenset({"base"}))
    )
    assert session.installation.devices == {}
    assert events == ["close:base"]
    session.close()


def test_virtual_world_boundary_preserves_independent_physical_leaf(tmp_path):
    from zlc_atom.devices.simulation.device_types import DEVICE_TYPES

    events: list[str] = []

    def physical_factory(_context, key, _config):
        return InstalledLeaf(
            key,
            "test.physical",
            _Device(key, 1),
            {},
            closer=lambda: events.append("close:physical"),
        )

    physical = DeviceTypeDescriptor(
        "test.physical",
        "test",
        AuthoringSchema(()),
        (),
        factory=physical_factory,
    )
    virtual = next(item for item in DEVICE_TYPES if item.type_id == "camera.virtual")
    catalog = DeviceCatalogSnapshot((physical, virtual), ())
    physical_config = DeviceInstanceConfig("real", "real", physical.type_id, {})
    initial = InstallationConfig((physical_config,))
    session = ExperimentSession.from_config(tmp_path, initial, catalog=catalog)
    original = session.installation.device("real")
    virtual_config = DeviceInstanceConfig(
        "sim",
        "sim",
        virtual.type_id,
        virtual.authoring_schema.project_values({}),
    )

    with_virtual = InstallationConfig((physical_config, virtual_config))
    add_plan = session.plan_device_reconcile(with_virtual)
    assert add_plan.world_changed
    assert add_plan.retained_keys == ("real",)
    assert add_plan.build_keys == ("sim",)
    session.reconcile_devices(add_plan)
    assert session.installation.device("real") is original
    assert events == []

    remove_plan = session.plan_device_reconcile(initial)
    assert remove_plan.world_changed
    assert remove_plan.retained_keys == ("real",)
    session.reconcile_devices(remove_plan)
    assert session.installation.device("real") is original
    assert session.installation.world is None
    assert events == []
    session.close()
    assert events == ["close:physical"]


def test_invalid_target_is_rejected_before_any_live_device_closes(tmp_path):
    events: list[str] = []
    catalog = _catalog(events)
    initial = InstallationConfig((_device("base"),))
    session = ExperimentSession.from_config(tmp_path, initial, catalog=catalog)
    original = session.installation.device("base")
    invalid = InstallationConfig(initial.devices, simulation={"seed": 2})

    with pytest.raises(ValueError, match="simulation requires a virtual device"):
        session.plan_device_reconcile(invalid)

    assert session.installation.device("base") is original
    assert events == []
    session.close()


def test_world_and_physical_change_rebuilds_physical_dependants(tmp_path):
    from zlc_atom.devices.simulation.device_types import DEVICE_TYPES

    events: list[str] = []
    base_catalog = _catalog(events)
    virtual = next(item for item in DEVICE_TYPES if item.type_id == "camera.virtual")
    catalog = DeviceCatalogSnapshot((*base_catalog.available, virtual), ())
    sim = DeviceInstanceConfig(
        "sim",
        "sim",
        virtual.type_id,
        virtual.authoring_schema.project_values({}),
    )
    initial = InstallationConfig(
        (_device("base"), _device("consumer", dependent=True), sim),
        simulation={"seed": 1},
    )
    session = ExperimentSession.from_config(tmp_path, initial, catalog=catalog)
    wanted = InstallationConfig(
        (_device("base", value=2), _device("consumer", dependent=True), sim),
        simulation={"seed": 2},
    )

    plan = session.plan_device_reconcile(wanted)
    assert set(plan.build_keys) == {"base", "consumer", "sim"}
    assert plan.retained_keys == ()
    session.close()


def test_operational_close_can_continue_after_last_virtual_leaf(tmp_path):
    from zlc_atom.devices.simulation.device_types import DEVICE_TYPES

    events: list[str] = []

    def physical_factory(_context, key, _config):
        return InstalledLeaf(
            key,
            "test.physical",
            _Device(key, 1),
            {},
            closer=lambda: events.append("close:physical"),
        )

    physical = DeviceTypeDescriptor(
        "test.physical", "test", AuthoringSchema(()), (), factory=physical_factory
    )
    virtual = next(item for item in DEVICE_TYPES if item.type_id == "camera.virtual")
    catalog = DeviceCatalogSnapshot((physical, virtual), ())
    config = InstallationConfig(
        (
            DeviceInstanceConfig("real", "real", physical.type_id, {}),
            DeviceInstanceConfig(
                "sim",
                "sim",
                virtual.type_id,
                virtual.authoring_schema.project_values({}),
            ),
        ),
        simulation={"seed": 3},
    )
    session = ExperimentSession.from_config(tmp_path, config, catalog=catalog)

    session.reconcile_devices(
        session.plan_device_reconcile(config, close_keys=frozenset({"sim"}))
    )
    effective = session.installation_config
    session.reconcile_devices(
        session.plan_device_reconcile(
            effective,
            close_keys=frozenset({"real"}),
        )
    )
    assert session.installation.devices == {}
    assert events == ["close:physical"]
    session.close()


def test_partial_close_projects_effective_config_and_retries_dependencies(tmp_path):
    events: list[str] = []
    dependent_attempts = 0

    def leaf_factory(type_id, *, dependency_key=None, closer=None):
        def factory(context, key, _config):
            if dependency_key is not None:
                assert dependency_key in context.devices
            return InstalledLeaf(
                key,
                type_id,
                _Device(key, 1),
                {},
                closer=closer or (lambda key=key: events.append(f"close:{key}")),
            )

        return factory

    def close_dependent():
        nonlocal dependent_attempts
        dependent_attempts += 1
        if dependent_attempts == 1:
            raise RuntimeError("dependent still owns its dependency")
        events.append("close:dependent")

    base = DeviceTypeDescriptor(
        "test.base", "test", AuthoringSchema(()), (),
        factory=leaf_factory("test.base"),
    )
    dependent = DeviceTypeDescriptor(
        "test.dependent", "test", AuthoringSchema(()), (),
        dependencies=(base.type_id,),
        factory=leaf_factory(
            "test.dependent", dependency_key="base", closer=close_dependent
        ),
    )
    grand = DeviceTypeDescriptor(
        "test.grand", "test", AuthoringSchema(()), (),
        dependencies=(dependent.type_id,),
        factory=leaf_factory("test.grand", dependency_key="dependent"),
    )
    catalog = DeviceCatalogSnapshot((base, dependent, grand), ())
    config = InstallationConfig(
        (
            DeviceInstanceConfig("base", "base", base.type_id, {}),
            DeviceInstanceConfig("dependent", "dependent", dependent.type_id, {}),
            DeviceInstanceConfig("grand", "grand", grand.type_id, {}),
        )
    )
    session = ExperimentSession.from_config(tmp_path, config, catalog=catalog)

    with pytest.raises(BaseExceptionGroup, match="installation close failed"):
        session.reconcile_devices(
            session.plan_device_reconcile(config, close_keys=frozenset({"base"}))
        )

    assert set(session.installation.devices) == {"base", "dependent"}
    assert tuple(
        item.instance_id for item in session.installation_config.devices
    ) == ("base", "dependent")
    assert events == ["close:grand"]
    effective = session.installation_config
    session.reconcile_devices(
        session.plan_device_reconcile(
            effective,
            close_keys=frozenset({"base"}),
        )
    )
    assert session.installation.devices == {}
    assert events == ["close:grand", "close:dependent", "close:base"]
    session.close()


def test_composition_recovery_remains_session_owned_until_it_closes(
    tmp_path,
    monkeypatch,
):
    import zlc_atom.install as install_module
    from zlc_atom.install import InstallationCompositionError, InstallationRecovery

    events: list[str] = []
    catalog = _catalog(events)
    initial = InstallationConfig((_device("base"), _device("other")))
    session = ExperimentSession.from_config(tmp_path, initial, catalog=catalog)
    wanted = InstallationConfig((_device("base", value=2), _device("other")))
    attempts = 0

    def close_recovery():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("replacement vendor handle is still open")
        events.append("close:replacement")

    recovery = InstallationRecovery(
        (
            InstalledLeaf(
                "base",
                "test.base",
                _Device("base", 99),
                {},
                closer=close_recovery,
            ),
        )
    )

    def fail_create(*_args, **_kwargs):
        raise InstallationCompositionError(
            "replacement composition failed",
            (RuntimeError("factory admission failed"),),
            recovery,
        )

    monkeypatch.setattr(install_module, "create_installation", fail_create)
    with pytest.raises(InstallationCompositionError):
        session.reconcile_devices(session.plan_device_reconcile(wanted))

    assert set(session.installation.devices) == {"other"}
    assert len(session._recovery_installations) == 1
    with pytest.raises(BaseExceptionGroup, match="recovery close failed"):
        session.close()
    assert "close:other" not in events
    session.close()
    assert events == ["close:base", "close:replacement", "close:other"]
    assert session._recovery_installations == []
