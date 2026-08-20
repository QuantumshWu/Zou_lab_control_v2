"""Canonical phase contract and installation binding for SLM adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from zlc_atom.install.descriptors import InstalledLeaf


_TWO_PI = 2.0 * np.pi
_MAX_WRAPPED_PHASE = np.nextafter(np.float32(_TWO_PI), np.float32(0.0))


def _shape(shape_yx: object) -> tuple[int, int]:
    if (
        not isinstance(shape_yx, tuple)
        or len(shape_yx) != 2
        or any(type(value) is not int or value <= 0 for value in shape_yx)
    ):
        raise TypeError("SLM shape_yx must be a two-positive-integer tuple")
    return shape_yx


def canonical_phase(radians: object, shape_yx: tuple[int, int]) -> np.ndarray:
    """Return one immutable owned phase snapshot in wrapped float32 radians."""

    shape = _shape(shape_yx)
    source = np.asarray(radians)
    if source.shape != shape:
        raise ValueError(
            f"SLM phase shape {source.shape!r} differs from device shape {shape!r}"
        )
    if source.dtype.kind not in "iuf":
        raise TypeError("SLM phase must contain real numeric values")
    values = np.asarray(source, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("SLM phase must contain only finite values")
    wrapped = np.asarray(np.remainder(values, _TWO_PI), dtype=np.float32)
    # float32(2*pi) rounds above the mathematical upper bound.  Clamp that
    # single rounding case so the public interval stays strictly [0, 2*pi)
    # and canonicalizing an already-canonical snapshot is idempotent.
    wrapped = np.minimum(wrapped, _MAX_WRAPPED_PHASE)
    # An ndarray backed by immutable bytes cannot be made writable again by a
    # caller, unlike an owning array with only its WRITEABLE flag cleared.
    return np.frombuffer(
        np.ascontiguousarray(wrapped).tobytes(),
        dtype=np.float32,
    ).reshape(shape)


@runtime_checkable
class SlmAdapter(Protocol):
    """The complete device-independent surface of one phase-only SLM."""

    @property
    def identity(self) -> str: ...

    @property
    def shape_yx(self) -> tuple[int, int]: ...

    def apply_phase(self, radians: object) -> np.ndarray: ...

    @property
    def last_commanded_phase(self) -> np.ndarray | None: ...

    @property
    def command_revision(self) -> int: ...

    @property
    def mapping_revision(self) -> int: ...

    @property
    def last_command_receipt(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...


def bind_slm(
    context: object,
    key: str,
    slm: SlmAdapter,
    type_id: str,
) -> InstalledLeaf:
    """Bind one adapter through the installation's existing device broker."""

    from zlc_atom.execution import (
        DeviceIdentityEvidenceKind,
        PhysicalDeviceIdentity,
        ResourceKey,
        bind_verified_device,
    )
    from zlc_atom.install.descriptors import InstalledLeaf

    if not isinstance(slm, SlmAdapter):
        raise TypeError("slm must implement the canonical SlmAdapter contract")
    try:
        identity = slm.identity
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or identity != identity.strip()
        ):
            raise ValueError("SLM identity must be non-empty text without surrounding space")
        shape = _shape(slm.shape_yx)
        observed_phase = slm.last_commanded_phase
        if observed_phase is not None:
            observed = np.asarray(observed_phase)
            canonical = canonical_phase(observed, shape)
            if observed.flags.writeable:
                raise ValueError("SLM last_commanded_phase must be an immutable snapshot")
            if not np.array_equal(observed, canonical):
                raise ValueError(
                    "SLM last_commanded_phase must already be canonical wrapped radians"
                )
        command_revision = slm.command_revision
        mapping_revision = slm.mapping_revision
        if type(command_revision) is not int or command_revision < 0:
            raise ValueError("SLM command_revision must be a non-negative integer")
        if type(mapping_revision) is not int or mapping_revision < 0:
            raise ValueError("SLM mapping_revision must be a non-negative integer")
        receipt = dict(slm.last_command_receipt)
        if receipt.get("outcome") not in {"known-old", "known-new", "unknown"}:
            raise ValueError("SLM command receipt has an invalid outcome")
        if receipt.get("command_revision") != command_revision:
            raise ValueError("SLM command receipt revision differs from device truth")
        if receipt.get("mapping_revision") != mapping_revision:
            raise ValueError("SLM command receipt mapping differs from device truth")
        if (observed_phase is None) != (receipt["outcome"] == "unknown"):
            raise ValueError("SLM phase knowledge differs from its command receipt")
        binding, proof = bind_verified_device(
            context.broker,
            key=ResourceKey.parse(f"device/{key}"),
            identity_probe=lambda: PhysicalDeviceIdentity(
                identity,
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            ),
            capability_probe=lambda: {"slm.phase": slm},
        )
    except BaseException:
        slm.close()
        raise
    return InstalledLeaf(
        key,
        type_id,
        slm,
        dict(proof.snapshot),
        binding=binding,
        closer=slm.close,
    )


__all__ = ["SlmAdapter", "bind_slm", "canonical_phase"]
