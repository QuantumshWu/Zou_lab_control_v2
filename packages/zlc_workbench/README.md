# zlc_workbench

The composition root: the only package allowed to know all the others.

Presenters live here, wiring `zlc_ui`'s mute views to `zlc_runtime`, `zlc_atom`
and `zlc_plot`. So do the application entry points and the cross-package
end-to-end tests.

Nothing else may. A domain rule, a rendering decision or a signal mechanism that
appears in this package is misplaced, and belongs to whichever package owns that
subject. This is a constitutional limit, not a style preference: the whole point
of the split was that each subject has exactly one home.

The notebook and the GUI drive the **same** session facade. If those two ever
grow separate paths, the split has failed.

## Check the environment first

```bash
python -m zlc_workbench.tools.check_environment
```

Run it from anywhere except the workspace root. Three separate incidents in this
project came from an import that succeeded while the wrong code ran — a monolith
installed under the same names, an uninstalled package resolving to an empty
namespace because the current directory happened to sit beside it, an editable
install pointing at a deleted copy. None of them raised. This asserts that every
package resolves to the repo that owns it, and that the retired names are gone.
