from __future__ import annotations

import sys
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
    create_installation,
    discover_device_catalog,
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
