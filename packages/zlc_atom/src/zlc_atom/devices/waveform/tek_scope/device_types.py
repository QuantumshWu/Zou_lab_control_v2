"""The Tektronix scope this bench can install, and how to find one."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.visa import identity_fields
from zlc_atom.devices.waveform.binding import bind_waveform_source
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from .source import TekScopeConfig, TekScopeWaveformSource, discover_tek_scopes


#: Where the scope is and which channels a record carries, by number.
TEK_SCOPE_SCHEMA = AuthoringSchema(
    (
        AuthoringField("resource", "str", "VISA resource", "", required=True),
        AuthoringField("channels", "str", "Channels (e.g. 1,2)", "1", required=True),
        AuthoringField("timeout_seconds", "float", "Timeout (s)", 5.0, minimum=0.1),
    )
)


def scope_channels(text: object) -> tuple[int, ...]:
    """The channel numbers an operator typed, e.g. ``"1, 3"``."""

    parts = [part.strip() for part in str(text).split(",")]
    try:
        return tuple(int(part) for part in parts if part)
    except ValueError as error:
        raise ValueError(
            f"channels must be channel numbers separated by commas, not {text!r}"
        ) from error


def _factory(context, key: str, values: dict) -> InstalledLeaf:
    authored = TEK_SCOPE_SCHEMA.project_values(values)
    config = TekScopeConfig(
        resource=str(authored["resource"]),
        channels=scope_channels(authored["channels"]),
        timeout_seconds=float(authored["timeout_seconds"]),
    )
    source = TekScopeWaveformSource(config)
    return bind_waveform_source(
        context, key, source, source.identity, "waveform.tek_scope"
    )


def _discover() -> tuple[DeviceInstanceConfig, ...]:
    def named(resource: str, identity: str) -> str:
        fields = identity_fields(identity)
        tail = fields[2] if len(fields) > 2 and fields[2] else "".join(
            c if c.isalnum() else "_" for c in resource
        )
        return f"scope_{tail}"

    return tuple(
        DeviceInstanceConfig(
            instance_id=named(resource, identity),
            role=named(resource, identity),
            type_id="waveform.tek_scope",
            parameters=TEK_SCOPE_SCHEMA.project_values({"resource": resource}),
        )
        for resource, identity in discover_tek_scopes()
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "waveform.tek_scope",
        "waveform",
        TEK_SCOPE_SCHEMA,
        ("waveform.source",),
        factory=_factory,
        discover=_discover,
    ),
)

__all__ = ["DEVICE_TYPES", "TEK_SCOPE_SCHEMA", "scope_channels"]
