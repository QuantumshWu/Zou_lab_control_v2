"""Small resource and physical-identity contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ResourceKey:
    segments: tuple[str, ...]

    def __post_init__(self) -> None:
        segments = tuple(str(segment) for segment in self.segments)
        if not segments or any(not segment or "/" in segment for segment in segments):
            raise ValueError("resource key segments must be non-empty and slash-free")
        object.__setattr__(self, "segments", segments)

    @classmethod
    def parse(cls, value: str) -> "ResourceKey":
        if not isinstance(value, str) or not value:
            raise ValueError("resource key must be non-empty text")
        return cls(tuple(value.split("/")))

    def __str__(self) -> str:
        return "/".join(self.segments)


@dataclass(frozen=True, order=True)
class PhysicalDeviceIdentity:
    stable_device_identity: str

    def __post_init__(self) -> None:
        if not self.stable_device_identity:
            raise ValueError("stable_device_identity must be non-empty")


__all__ = [
    "PhysicalDeviceIdentity",
    "ResourceKey",
]
