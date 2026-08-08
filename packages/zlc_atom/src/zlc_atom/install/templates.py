"""Named starting points for an apparatus.

A template is a convenience, not a constraint: devices are installed one by one,
so any mixture is legal by construction.  A real sequencer with a virtual camera
is not a special mode -- it is what you get by picking each independently, which
is exactly how hardware bring-up works.  You verify the pulse timing against a
camera you trust, then the imaging against a sequencer you trust, and only then
both at once.  Without that, a first light session can only be attempted all at
once, and a failure has nowhere to be bisected.

The two templates below are the ends of that spectrum; everything between them
is a list of DeviceSpec the caller writes, or a saved configuration reopened.
"""

from .configuration import DeviceInstanceConfig, InstallationConfig
from .discovery import DeviceCatalogSnapshot
from .graph import DeviceSpec

_INSTALLATION_TEMPLATES = {
    "virtual": (
        DeviceSpec("camera", "camera.virtual"),
        DeviceSpec("sequencer", "sequencer.virtual"),
    ),
    "hardware": (
        DeviceSpec("camera", "camera.dcam"),
        DeviceSpec("sequencer", "sequencer.hardware"),
    ),
}


def installation_template_names() -> tuple[str, ...]:
    """Stable names offered by apparatus authoring surfaces."""

    return tuple(_INSTALLATION_TEMPLATES)


def installation_config_from_template(
    catalog: DeviceCatalogSnapshot,
    name: str,
) -> InstallationConfig:
    """Resolve one template against exactly the supplied catalog snapshot."""

    if not isinstance(catalog, DeviceCatalogSnapshot):
        raise TypeError("catalog must be DeviceCatalogSnapshot")
    try:
        specs = _INSTALLATION_TEMPLATES[str(name)]
    except KeyError as error:
        raise KeyError(f"unknown installation template {name!r}") from error
    by_type = {descriptor.type_id: descriptor for descriptor in catalog.available}
    missing = tuple(spec.type_id for spec in specs if spec.type_id not in by_type)
    if missing:
        raise KeyError(
            f"installation template {name!r} needs unavailable device types: "
            f"{sorted(set(missing))}"
        )
    return InstallationConfig(
        tuple(
            DeviceInstanceConfig(
                instance_id=spec.key,
                role=spec.key,
                type_id=spec.type_id,
                parameters=by_type[spec.type_id].authoring_schema.freeze(spec.config),
            )
            for spec in specs
        )
    )


__all__ = ["installation_config_from_template", "installation_template_names"]
