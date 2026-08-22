"""Minimal, headless atom device and logic-node package.

The top-level module is deliberately boring: importing :mod:`zlc_atom` does
not discover devices, import optional runtime/pulse packages, or import a GUI.
Subpackages own their contracts and are imported explicitly by composition
roots.
"""

__all__: tuple[str, ...] = ()
