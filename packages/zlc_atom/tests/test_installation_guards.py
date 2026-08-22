from __future__ import annotations

import sys
import threading
import types

import pytest

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.execution import (
    DeviceBroker,
    DeviceIdentityEvidenceKind,
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


def test_a_failed_device_close_is_retried_before_installation_is_terminal() -> None:
    attempts = 0

    def close() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("camera is still owned")

    installation = Installation(
        {
            "camera": InstalledLeaf(
                "camera", "test.camera", object(), {}, closer=close
            )
        },
        world=None,
    )
    with pytest.raises(ExceptionGroup, match="installation close failed"):
        installation.close()
    installation.close()
    assert attempts == 2


def test_close_failure_keeps_every_earlier_possible_dependency_open() -> None:
    closed: list[str] = []
    attempts = 0

    def close_dependent() -> None:
        nonlocal attempts
        attempts += 1
        closed.append(f"dependent-{attempts}")
        if attempts == 1:
            raise RuntimeError("dependent still uses base")

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
    assert closed == ["later", "dependent-1"]
    assert tuple(installation.devices) == ("base", "dependent")

    installation.close()
    assert closed == ["later", "dependent-1", "dependent-2", "base"]


def _bound_test_leaf(
    broker: DeviceBroker,
    key: str,
    physical_id: str,
    close,
) -> InstalledLeaf:
    binding, proof = bind_verified_device(
        broker,
        key=ResourceKey.parse(f"device/{key}"),
        identity_probe=lambda: PhysicalDeviceIdentity(
            physical_id,
            DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
        ),
        capability_probe=dict,
    )
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
    assert leaf.physical_identity == PhysicalDeviceIdentity(
        "camera:serial-1",
        DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
    )

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


def test_new_world_preflight_can_borrow_and_retain_independent_physical_leaf() -> None:
    from zlc_atom.devices.simulation import SimulationWorldConfig

    broker = DeviceBroker()
    old_world = object()

    def physical_factory(_context, key, _values):
        return InstalledLeaf(key, "test.physical", object(), {})

    physical = DeviceTypeDescriptor(
        "test.physical",
        "test",
        AuthoringSchema(()),
        (),
        factory=physical_factory,
    )
    owner = create_installation(
        (DeviceSpec("physical", physical.type_id),),
        world=old_world,
        broker=broker,
        catalog=DeviceCatalogSnapshot((physical,), ()),
    )

    def virtual_factory(context, key, _values):
        assert context.devices["physical"] is owner.devices["physical"]
        return InstalledLeaf(key, "test.virtual", object(), {})

    virtual = DeviceTypeDescriptor(
        "test.virtual",
        "test",
        AuthoringSchema(()),
        (),
        dependencies=(physical.type_id,),
        factory=virtual_factory,
        world_config=lambda _values: SimulationWorldConfig(),
    )
    blueprint = preflight_installation(
        (DeviceSpec("virtual", virtual.type_id),),
        simulation={},
        borrowed_from=owner,
        borrowed_revision=owner.revision,
        catalog=DeviceCatalogSnapshot((virtual,), ()),
    )
    assert blueprint.world is not old_world
    successor = create_installation(blueprint)
    owner.transfer_leaves_to(
        successor,
        ("physical",),
        source_revision=owner.revision,
        target_revision=successor.revision,
    )
    assert set(successor.devices) == {"physical", "virtual"}
    owner.close()
    successor.close()


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


def test_preflight_owns_transitive_dependency_closure() -> None:
    def descriptor(type_id: str, dependencies=()):
        return DeviceTypeDescriptor(
            type_id,
            "test",
            AuthoringSchema(()),
            (),
            dependencies=tuple(dependencies),
            factory=lambda _context, key, _values, tid=type_id: InstalledLeaf(
                key, tid, object(), {}
            ),
        )

    base = descriptor("test.base")
    child = descriptor("test.child", (base.type_id,))
    grandchild = descriptor("test.grandchild", (child.type_id,))
    blueprint = preflight_installation(
        (
            DeviceSpec("grandchild", grandchild.type_id),
            DeviceSpec("child", child.type_id),
            DeviceSpec("base", base.type_id),
        ),
        world=object(),
        catalog=DeviceCatalogSnapshot((base, child, grandchild), ()),
    )
    assert tuple(spec.key for spec in blueprint.specs) == (
        "base",
        "child",
        "grandchild",
    )
    assert blueprint.dependent_keys((base.type_id,)) == {
        "child",
        "grandchild",
    }


def test_successor_factory_can_borrow_retained_dependency_without_owning_it() -> None:
    broker = DeviceBroker()
    world = object()
    closed: list[str] = []
    retained = _bound_test_leaf(
        broker,
        "camera",
        "camera:serial-1",
        lambda: closed.append("retained"),
    )
    owner = Installation(
        {"camera": retained},
        world=world,
        broker=broker,
    )
    observed: list[object] = []

    def factory(context, key, _values):
        observed.append(context.devices["camera"])
        return InstalledLeaf(
            key,
            "test.consumer",
            object(),
            {},
            closer=lambda: closed.append("consumer"),
        )

    descriptor = DeviceTypeDescriptor(
        "test.consumer",
        "test",
        AuthoringSchema(()),
        (),
        dependencies=(retained.type_id,),
        factory=factory,
    )
    successor = create_installation(
        (DeviceSpec("consumer", descriptor.type_id),),
        world=world,
        broker=broker,
        borrowed_from=owner,
        borrowed_revision=owner.revision,
        catalog=DeviceCatalogSnapshot((descriptor,), ()),
    )
    assert observed == [retained]
    assert set(successor.devices) == {"consumer"}
    assert "camera" not in successor.devices

    successor.close()
    assert closed == ["consumer"]
    with pytest.raises(RuntimeError, match="already bound"):
        _bound_test_leaf(broker, "duplicate", "camera:serial-1", lambda: None)
    owner.close()
    assert closed == ["consumer", "retained"]
    replacement = _bound_test_leaf(
        broker,
        "camera-again",
        "camera:serial-1",
        lambda: None,
    )
    assert broker.unbind(replacement.binding) is True


def test_successor_build_pins_borrowed_owner_against_close_and_transfer() -> None:
    broker = DeviceBroker()
    world = object()
    entered = threading.Event()
    release = threading.Event()
    retained = InstalledLeaf("base", "test.base", object(), {})
    owner = Installation(
        {"base": retained},
        world=world,
        broker=broker,
    )
    target = Installation({}, world=world, broker=broker)

    def factory(_context, key, _values):
        entered.set()
        assert release.wait(2.0)
        return InstalledLeaf(key, "test.consumer", object(), {})

    descriptor = DeviceTypeDescriptor(
        "test.consumer",
        "test",
        AuthoringSchema(()),
        (),
        dependencies=(retained.type_id,),
        factory=factory,
    )
    catalog = DeviceCatalogSnapshot((descriptor,), ())
    result: list[Installation] = []
    failures: list[BaseException] = []

    def build() -> None:
        try:
            result.append(
                create_installation(
                    (DeviceSpec("consumer", descriptor.type_id),),
                    world=world,
                    borrowed_from=owner,
                    borrowed_revision=owner.revision,
                    catalog=catalog,
                )
            )
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=build)
    thread.start()
    assert entered.wait(2.0)
    with pytest.raises(RuntimeError, match="leaves are borrowed"):
        owner.close()
    with pytest.raises(RuntimeError, match="leaves are borrowed"):
        owner.transfer_leaves_to(
            target,
            ("base",),
            source_revision=owner.revision,
            target_revision=target.revision,
        )
    release.set()
    thread.join(2.0)
    assert not thread.is_alive() and not failures
    assert set(result[0].devices) == {"consumer"}
    result[0].close()
    owner.close()
    target.close()


def test_borrowed_owner_and_revision_validation_precede_factory_side_effects() -> None:
    broker = DeviceBroker()
    world = object()
    retained = _bound_test_leaf(
        broker,
        "base",
        "base:serial-1",
        lambda: None,
    )
    owner = Installation(
        {"base": retained},
        world=world,
        broker=broker,
    )
    calls: list[str] = []

    def factory(_context, key, _values):
        calls.append(key)
        return InstalledLeaf(key, "test.consumer", object(), {})

    descriptor = DeviceTypeDescriptor(
        "test.consumer",
        "test",
        AuthoringSchema(()),
        (),
        factory=factory,
    )
    catalog = DeviceCatalogSnapshot((descriptor,), ())
    arguments = dict(world=world, catalog=catalog)
    try:
        with pytest.raises(ValueError, match="requires borrowed_from"):
            create_installation(
                (), borrowed_revision=owner.revision, **arguments
            )
        with pytest.raises(TypeError, match="Installation"):
            create_installation(
                (), borrowed_from=object(), borrowed_revision=0, **arguments
            )
        with pytest.raises(RuntimeError, match="revision changed"):
            create_installation(
                (), borrowed_from=owner, borrowed_revision=1, **arguments
            )
        with pytest.raises(ValueError, match="duplicate device spec"):
            create_installation(
                (DeviceSpec("base", descriptor.type_id),),
                borrowed_from=owner,
                borrowed_revision=owner.revision,
                **arguments,
            )
        blueprint = preflight_installation(
            (),
            world=world,
            borrowed_from=owner,
            borrowed_revision=owner.revision,
            catalog=catalog,
        )
        with pytest.raises(RuntimeError, match="borrowed installation broker"):
            create_installation(
                blueprint,
                broker=DeviceBroker(),
            )
        assert calls == []
    finally:
        owner.close()


def test_factory_admission_rejects_foreign_binding_and_rolls_back_prior_leaves() -> None:
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
            identity_probe=lambda: PhysicalDeviceIdentity(
                "foreign:device",
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            ),
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
        dependencies=(good.type_id,),
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
    assert closed == ["bad-1", "good-1"]
    assert captured.value.recovery.leaves == (accepted[0], rejected[0])
    assert foreign.verify_capability(rejected[0].binding).binding is rejected[0].binding
    captured.value.recovery.close()
    assert closed == ["bad-1", "good-1", "bad-2", "good-2"]
    assert captured.value.recovery.leaves == ()
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
            identity_probe=lambda: PhysicalDeviceIdentity(
                "strict:physical",
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            ),
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
            identity_probe=lambda: PhysicalDeviceIdentity(
                f"synthetic:{key}",
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            ),
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
    installation = None
    try:
        installation = create_installation(
            (
                DeviceSpec("same", descriptor.type_id),
                DeviceSpec("same", descriptor.type_id),
            ),
            catalog=catalog,
        )
    except ValueError as error:
        assert "duplicate device key" in str(error)
    else:
        # On the old implementation both factories ran, the second leaf
        # replaced the first in the dict, and close could only see leaf 1.
        installation.close()
        pytest.fail(
            "duplicate key was accepted: "
            f"world_calls={world_calls}, made={made}, closed={closed}"
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
