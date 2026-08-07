# Vendored migration notes

The role-axis data contract is maintained by the sibling `../zlc_data`
repository and is installed as the `zlc-data` dependency.  This directory is
kept only as a marker so an obsolete vendored copy cannot be reintroduced.

This directory intentionally contains no vendored package and no `sys.path`
hook.  `zlc_plot` consumes `zlc_data.OwnedSnapshot` directly; presentation
units and latest-only transport live in `zlc_plot` itself.
