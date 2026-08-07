# zlc_atom package contract

The package-level facade is intentionally limited to the installation identity
probe.  Devices, nodes, and authoring types remain available through their
explicit submodule paths; importing `zlc_atom` does not discover or re-export
them.

```python
__all__ = ("__version__",)
```

`__version__` is retained so callers and package guards can prove which
editable checkout supplied the import.  The allow-list used by tests lives in
`tests/test_import_boundaries.py`, not in the distributable package.
