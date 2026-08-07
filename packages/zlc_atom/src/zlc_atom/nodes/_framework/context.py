"""Four-capability application context injected into a node."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable


class ApplicationContext:
    """The deliberately narrow context surface: device/input/run/plane."""

    __slots__ = ("_devices", "_inputs", "_start_run", "_signal_plane")

    def __init__(self, *, devices: Mapping[object, object], inputs: Mapping[str, object], start_run: Callable[..., object], signal_plane: object) -> None:
        self._devices = dict(devices)
        self._inputs = dict(inputs)
        self._start_run = start_run
        self._signal_plane = signal_plane

    def device(self, key: object) -> object:
        try:
            return self._devices[key]
        except KeyError as exc:
            raise KeyError(f"no device is installed at {key!r}") from exc

    def input(self, name: str) -> object:
        try:
            return self._inputs[name]
        except KeyError as exc:
            raise KeyError(f"node input {name!r} is not bound") from exc

    def start_run(self, plan: object, **kwargs: Any) -> object:
        return self._start_run(plan, **kwargs)

    @property
    def signal_plane(self) -> object:
        return self._signal_plane


__all__ = ["ApplicationContext"]
