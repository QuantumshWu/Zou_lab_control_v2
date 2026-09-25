from __future__ import annotations

import sys
import types

import pytest

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.execution import (
    DeviceBroker,
    PhysicalDeviceIdentity,
    ResourceKey,
    bind_verified_device,
)
from zlc_atom.install import (
    CAPABILITY_TYPES,
    DeviceCatalogSnapshot,
    DeviceSpec,
    DeviceTypeDescriptor,
    InstalledLeaf,
    Installation,
    InstallationCompositionError,
    create_installation,
    discover_device_catalog,
    preflight_installation,
)
from zlc_atom.install.configuration import DeviceInstanceConfig, InstallationConfig


def test_one_failed_close_keeps_only_that_leaf_open() -> None:
    closed: list[str] = []
    attempts = 0

    def close_dependent() -> None:
        nonlocal attempts
        attempts += 1
        closed.append(f"dependent-{attempts}")
        if attempts == 1:
            raise RuntimeError("dependent refused its close")

    installation = Installation(
        {
            "base": InstalledLeaf(
                "base",
                "test.base",
                object(),
                {},
                closer=lambda: closed.append("base"),
            ),
            "dependent": InstalledLeaf(
                "dependent",
                "test.dependent",
                object(),
                {},
                closer=close_dependent,
            ),
            "later": InstalledLeaf(
                "later",
                "test.later",
                object(),
                {},
                closer=lambda: closed.append("later"),
            ),
        },
        world=None,
    )
    with pytest.raises(ExceptionGroup, match="installation close failed"):
        installation.close()
    assert closed == ["later", "dependent-1", "base"]
    assert tuple(installation.devices) == ("dependent",)

    installation.close()
    assert closed == ["later", "dependent-1", "base", "dependent-2"]


def _bound_test_leaf(
    broker: DeviceBroker,
    key: str,
    physical_id: str,
    close,
) -> InstalledLeaf:
    binding, proof = bind_verified_device(
        broker,
        key=ResourceKey.parse(f"device/{key}"),
        identity_probe=lambda: PhysicalDeviceIdentity(physical_id),
        capability_probe=dict,
    )
    broker.claim(binding)
    return InstalledLeaf(
        key,
        "test.bound",
        object(),
        dict(proof.snapshot),
        binding=binding,
        closer=close,
    )


def test_installation_unbinds_only_after_a_leaf_really_closes() -> None:
    broker = DeviceBroker()
    attempts = 0

    def close() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("vendor handle is still active")

    leaf = _bound_test_leaf(broker, "camera", "camera:serial-1", close)
    installation = Installation(
        {"camera": leaf},
        world=None,
        broker=broker,
    )
    assert leaf.physical_identity == PhysicalDeviceIdentity("camera:serial-1")

    with pytest.raises(ExceptionGroup, match="installation close failed"):
        installation.close()
    assert set(installation.devices) == {"camera"}
    with pytest.raises(RuntimeError, match="already bound"):
        _bound_test_leaf(broker, "other", "camera:serial-1", lambda: None)

    installation.close()
    assert installation.devices == {}
    assert attempts == 2
    replacement = _bound_test_leaf(
        broker,
        "camera-again",
        "camera:serial-1",
        lambda: None,
    )
    assert broker.unbind(replacement.binding) is True


def test_installation_transfers_unchanged_leaf_ownership_once_and_by_revision() -> None:
    broker = DeviceBroker()
    world = object()
    closed: list[str] = []
    retained = _bound_test_leaf(
        broker,
        "camera",
        "camera:serial-1",
        lambda: closed.append("camera"),
    )
    source = Installation(
        {"camera": retained},
        world=world,
        broker=broker,
    )
    target = Installation(
        {
            "new": InstalledLeaf(
                "new",
                "test.new",
                object(),
                {},
                closer=lambda: closed.append("new"),
            )
        },
        world=world,
        broker=broker,
    )

    with pytest.raises(RuntimeError, match="source installation ownership revision"):
        source.transfer_leaves_to(
            target,
            ("camera",),
            source_revision=1,
            target_revision=0,
        )
    assert source.devices["camera"] is retained
    assert "camera" not in target.devices

    # A target that already holds the key -- as a device or as a remembered
    # failure -- refuses before the source lets go.  The check used to need
    # the key in BOTH, so a same-named leaf was overwritten on the target and
    # lost by the source, and nobody closed it.
    for taken in (
        Installation(
            {
                "camera": InstalledLeaf(
                    "camera",
                    "test.other",
                    object(),
                    {},
                    closer=lambda: closed.append("other camera"),
                )
            },
            world=world,
            broker=broker,
        ),
        Installation(
            {},
            world=world,
            failures={"camera": RuntimeError("did not start")},
            broker=broker,
        ),
    ):
        with pytest.raises(ValueError, match=r"already owns keys \['camera'\]"):
            source.transfer_leaves_to(
                taken,
                ("camera",),
                source_revision=source.revision,
                target_revision=taken.revision,
            )
        assert source.devices["camera"] is retained
        assert source.revision == 0 and taken.revision == 0
        taken.close()
    assert closed == ["other camera"]
    closed.clear()

    moved = source.transfer_leaves_to(
        target,
        ("camera",),
        source_revision=source.revision,
        target_revision=target.revision,
    )
    assert moved == ("camera",)
    assert source.revision == 1 and target.revision == 1
    assert source.devices == {}
    assert target.devices["camera"] is retained

    source.close()
    assert closed == [], "the old installation no longer owns the retained leaf"
    target.close()
    assert closed == ["new", "camera"], "successor reverse-close order is preserved"
    replacement = _bound_test_leaf(
        broker,
        "camera-again",
        "camera:serial-1",
        lambda: None,
    )
    assert broker.unbind(replacement.binding) is True


def test_only_world_independent_leaves_can_transfer_to_a_new_world() -> None:
    broker = DeviceBroker()
    first_world = object()
    second_world = object()

    def physical_factory(_context, key, _values):
        return InstalledLeaf(key, "test.physical", object(), {})

    def virtual_factory(_context, key, _values):
        return InstalledLeaf(key, "test.virtual", object(), {})

    physical = DeviceTypeDescriptor(
        "test.physical",
        "test",
        AuthoringSchema(()),
        (),
        factory=physical_factory,
    )
    virtual = DeviceTypeDescriptor(
        "test.virtual",
        "test",
        AuthoringSchema(()),
        (),
        factory=virtual_factory,
        world_config=lambda _values: object(),
    )
    source = create_installation(
        (DeviceSpec("physical", physical.type_id), DeviceSpec("virtual", virtual.type_id)),
        world=first_world,
        broker=broker,
        catalog=DeviceCatalogSnapshot((physical, virtual), ()),
    )
    assert source.devices["physical"].world_affinity is None
    assert source.devices["virtual"].world_affinity is first_world
    target = Installation({}, world=second_world, broker=broker)

    source.transfer_leaves_to(
        target,
        ("physical",),
        source_revision=source.revision,
        target_revision=target.revision,
    )
    with pytest.raises(RuntimeError, match="crosses worlds"):
        source.transfer_leaves_to(
            target,
            ("virtual",),
            source_revision=source.revision,
            target_revision=target.revision,
        )
    source.close()
    target.close()


def test_preflight_resolves_world_and_topology_without_running_a_factory() -> None:
    from zlc_atom.devices.simulation import SimulationWorld, SimulationWorldConfig

    calls: list[str] = []

    def world_config(_values):
        calls.append("world")
        return SimulationWorldConfig()

    def factory(_context, key, _values):
        calls.append("factory")
        return InstalledLeaf(key, "test.preflight", object(), {})

    descriptor = DeviceTypeDescriptor(
        "test.preflight",
        "test",
        AuthoringSchema(()),
        (),
        factory=factory,
        world_config=world_config,
    )
    blueprint = preflight_installation(
        (DeviceSpec("virtual", descriptor.type_id),),
        simulation={},
        catalog=DeviceCatalogSnapshot((descriptor,), ()),
    )
    assert calls == ["world"]
    assert isinstance(blueprint.world, SimulationWorld)
    assert blueprint.specs[0].key == "virtual"

    installation = create_installation(blueprint)
    assert calls == ["world", "factory"]
    assert installation.devices["virtual"].world_affinity is blueprint.world
    installation.close()


def test_template_default_simulation_does_not_conflict_with_explicit_world() -> None:
    from zlc_atom.devices.simulation import SimulationWorld

    world = SimulationWorld()
    installation = create_installation("virtual", world=world)
    assert installation.world is world
    installation.close()


def test_factory_admission_rejects_foreign_binding_and_recovery_owns_what_stays_open() -> None:
    """A rejected candidate that cannot close ends the composition.

    Every other leaf is closed on its own -- one refusal stops nothing --
    and recovery owns exactly the leaves that stayed open: the rejected
    ``bad`` and a ``good`` that refused its first close.
    """

    broker = DeviceBroker()
    foreign = DeviceBroker()
    closed: list[str] = []
    accepted: list[InstalledLeaf] = []
    rejected: list[InstalledLeaf] = []
    attempts = {"good": 0, "bad": 0}

    def retrying_close(name: str) -> None:
        attempts[name] += 1
        closed.append(f"{name}-{attempts[name]}")
        if attempts[name] == 1:
            raise RuntimeError(f"{name} cleanup failed")

    def good_factory(_context, key, _values):
        leaf = InstalledLeaf(
            key,
            "test.good",
            object(),
            {},
            closer=lambda: retrying_close("good"),
        )
        accepted.append(leaf)
        return leaf

    def bad_factory(_context, key, _values):
        binding, proof = bind_verified_device(
            foreign,
            key=ResourceKey.parse(f"device/{key}"),
            identity_probe=lambda: PhysicalDeviceIdentity("foreign:device"),
            capability_probe=dict,
        )

        leaf = InstalledLeaf(
            key,
            "test.bad",
            object(),
            dict(proof.snapshot),
            binding=binding,
            closer=lambda: retrying_close("bad"),
        )
        rejected.append(leaf)
        return leaf

    good = DeviceTypeDescriptor(
        "test.good",
        "test",
        AuthoringSchema(()),
        (),
        factory=good_factory,
    )
    bad = DeviceTypeDescriptor(
        "test.bad",
        "test",
        AuthoringSchema(()),
        (),
        factory=bad_factory,
    )
    with pytest.raises(InstallationCompositionError) as captured:
        create_installation(
            (DeviceSpec("good", good.type_id), DeviceSpec("bad", bad.type_id)),
            world=object(),
            broker=broker,
            catalog=DeviceCatalogSnapshot((good, bad), ()),
        )
    assert isinstance(captured.value.exceptions[0], RuntimeError)
    assert "unknown" in str(captured.value.exceptions[0])
    assert [str(error) for error in captured.value.exceptions[1:]] == [
        "good cleanup failed",
        "bad cleanup failed",
    ]
    assert closed == ["bad-1", "good-1"]
    recovery = captured.value.recovery
    assert recovery.leaves == (accepted[0], rejected[0])
    assert foreign.verify_capability(rejected[0].binding).binding is rejected[0].binding
    recovery.close()
    assert closed == ["bad-1", "good-1", "bad-2", "good-2"]
    assert recovery.leaves == ()
    with pytest.raises(RuntimeError, match="unknown"):
        foreign.verify_capability(rejected[0].binding)


@pytest.mark.parametrize(
    ("returned_key", "returned_type", "match"),
    (("wrong", "test.strict", "leaf key"), ("strict", "wrong", "leaf type")),
)
def test_factory_leaf_logical_identity_must_match_its_spec(
    returned_key: str,
    returned_type: str,
    match: str,
) -> None:
    closed: list[str] = []

    def factory(_context, _key, _values):
        return InstalledLeaf(
            returned_key,
            returned_type,
            object(),
            {},
            closer=lambda: closed.append("closed"),
        )

    descriptor = DeviceTypeDescriptor(
        "test.strict",
        "test",
        AuthoringSchema(()),
        (),
        factory=factory,
    )
    installation = create_installation(
        (DeviceSpec("strict", descriptor.type_id),),
        world=object(),
        catalog=DeviceCatalogSnapshot((descriptor,), ()),
    )
    assert match in str(installation.failures["strict"])
    assert closed == ["closed"]
    installation.close()


def test_factory_binding_resource_key_must_match_the_logical_leaf_key() -> None:
    broker = DeviceBroker()
    closed: list[str] = []
    bindings = []

    def factory(context, key, _values):
        binding, proof = bind_verified_device(
            context.broker,
            key=ResourceKey.parse("device/someone-else"),
            identity_probe=lambda: PhysicalDeviceIdentity("strict:physical"),
            capability_probe=dict,
        )
        bindings.append(binding)
        return InstalledLeaf(
            key,
            "test.strict-binding",
            object(),
            dict(proof.snapshot),
            binding=binding,
            closer=lambda: closed.append("closed"),
        )

    descriptor = DeviceTypeDescriptor(
        "test.strict-binding",
        "test",
        AuthoringSchema(()),
        (),
        factory=factory,
    )
    installation = create_installation(
        (DeviceSpec("strict", descriptor.type_id),),
        world=object(),
        broker=broker,
        catalog=DeviceCatalogSnapshot((descriptor,), ()),
    )
    assert "expected device/strict" in str(installation.failures["strict"])
    assert closed == ["closed"]
    with pytest.raises(RuntimeError, match="unknown"):
        broker.verify_capability(bindings[0])
    installation.close()


def test_discovery_automatically_collects_a_synthetic_leaf_without_graph_changes(monkeypatch) -> None:
    calls: list[str] = []
    descriptor_module_name = "tests._synthetic_device_types"

    def factory(context, key, _values):
        binding, _proof = bind_verified_device(
            context.broker,
            key=ResourceKey.parse(f"device/{key}"),
            identity_probe=lambda: PhysicalDeviceIdentity(f"synthetic:{key}"),
            capability_probe=dict,
        )
        return InstalledLeaf(
            key,
            "test.synthetic",
            object(),
            {},
            binding=binding,
            closer=lambda: calls.append("closed"),
        )

    descriptor = DeviceTypeDescriptor(
        "test.synthetic",
        "test",
        AuthoringSchema(()),
        (),
        factory=factory,
    )
    module = types.ModuleType(descriptor_module_name)
    module.DEVICE_TYPES = (descriptor,)
    monkeypatch.setitem(sys.modules, descriptor_module_name, module)
    monkeypatch.setattr(
        "zlc_atom.install.discovery._modules",
        lambda: (descriptor_module_name,),
    )

    installation = create_installation((DeviceSpec("synthetic", "test.synthetic"),))
    assert installation.device("synthetic") is not None
    installation.close()
    assert calls == ["closed"]


def test_duplicate_device_keys_are_rejected_before_world_or_factory_side_effects() -> None:
    made: list[int] = []
    closed: list[int] = []
    world_calls: list[int] = []

    def world_config(_values):
        world_calls.append(1)
        return None

    def factory(_context, key, _values):
        ordinal = len(made)
        made.append(ordinal)
        return InstalledLeaf(
            key,
            "test.duplicate-key",
            object(),
            {},
            closer=lambda: closed.append(ordinal),
        )

    descriptor = DeviceTypeDescriptor(
        "test.duplicate-key",
        "test",
        AuthoringSchema(()),
        (),
        factory=factory,
        world_config=world_config,
    )
    catalog = DeviceCatalogSnapshot((descriptor,), ())
    # On the old implementation both factories ran, the second leaf replaced
    # the first in the dict, and close could only see leaf 1.
    with pytest.raises(ValueError, match="duplicate device key"):
        create_installation(
            (
                DeviceSpec("same", descriptor.type_id),
                DeviceSpec("same", descriptor.type_id),
            ),
            catalog=catalog,
        )
    assert world_calls == []
    assert made == []
    assert closed == []


def test_device_specs_and_config_documents_deep_own_nested_parameters() -> None:
    parameters = {"camera": {"gain": [1.0, 2.0]}}
    spec = DeviceSpec("camera", "camera.virtual", parameters)
    configured = DeviceInstanceConfig(
        "camera",
        "camera",
        "camera.virtual",
        parameters,
    )

    parameters["camera"]["gain"][0] = 99.0
    assert spec.config["camera"]["gain"] == (1.0, 2.0)
    assert configured.parameters["camera"]["gain"] == (1.0, 2.0)
    with pytest.raises(TypeError):
        spec.config["camera"]["gain"][0] = 99.0
    with pytest.raises(TypeError):
        configured.parameters["camera"]["gain"][0] = 99.0

    document = configured.to_dict()
    document["parameters"]["camera"]["gain"][0] = 99.0
    assert configured.parameters["camera"]["gain"] == (1.0, 2.0)
    specs = InstallationConfig((configured,)).specs()
    specs[0]["config"]["camera"]["gain"][0] = 99.0
    assert configured.parameters["camera"]["gain"] == (1.0, 2.0)


def test_installation_rejects_wrong_capability_instances_and_uses_one_registry() -> None:
    assert DeviceBroker.CAPABILITY_TYPES is CAPABILITY_TYPES

    def bad_factory(_context, key, _values):
        return InstalledLeaf(
            key,
            "test.bad-capability",
            object(),
            {"camera.adapter": object()},
        )

    descriptor = DeviceTypeDescriptor(
        "test.bad-capability",
        "test",
        AuthoringSchema(()),
        ("camera.adapter",),
        factory=bad_factory,
    )
    installation = create_installation(
        (DeviceSpec("bad", "test.bad-capability"),),
        catalog=DeviceCatalogSnapshot((descriptor,), ()),
    )
    assert isinstance(installation.failures["bad"], TypeError)
    assert "wrong type" in str(installation.failures["bad"])
    installation.close()


def test_discovered_installation_capabilities_match_declared_types() -> None:
    installation = create_installation("virtual")
    try:
        descriptors = {
            descriptor.type_id: descriptor
            for descriptor in discover_device_catalog().available
        }
        for key, leaf in installation.devices.items():
            descriptor = descriptors[leaf.type_id]
            assert set(descriptor.capabilities) <= set(leaf.capabilities)
            for token in descriptor.capabilities:
                assert isinstance(leaf.capabilities[token], CAPABILITY_TYPES[token])
    finally:
        installation.close()
