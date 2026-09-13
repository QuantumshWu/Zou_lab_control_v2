"""What a process must NOT be carrying by the time its window is up.

Every entry in here is a second an operator waited for work their process
never did.  They arrived the same way each time: something that is needed
to DO a job got imported in order to READ a declaration, or to ask whether
a job applied at all.  One import of IPython, to be told there was no
notebook.  One float taken from a solver, so a console that renders nothing
in its own process carried numba and llvmlite.  The fixes are in the
modules; these are the statements that keep them fixed, written where the
whole application is composed rather than inside the layer that slipped.

They are deliberately end-to-end: a new edge anywhere in any layer fails
them, which a rule about one module's import list would not.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGES = REPO_ROOT / "packages"

#: The engines.  A console composes the experiment, edits it and shows it;
#: the pictures are drawn in render children, which import these in
#: parallel, warm, before any panel asks.  Matplotlib is among them: the
#: console builds the plot STYLE to hand to those children and paints no
#: artist of its own, and it used to import the renderer and scan the font
#: list to construct that style.  ``scipy`` is not here, and only because
#: the VIRTUAL bench's simulated panel holds a startup hologram -- 15 ms to
#: solve and 0.39 s to import the transforms with; a real bench's console
#: has no such device and no scipy.
FOREIGN_TO_A_CONSOLE = ("numba", "llvmlite", "IPython", "matplotlib",
                        "zlc_plot.fit", "zlc_plot.raster", "zlc_plot.session")


def _sources() -> tuple[Path, ...]:
    paths = tuple(
        path
        for package in sorted(PACKAGES.iterdir())
        if (package / "src").is_dir()
        for path in (package / "src").rglob("*.py")
    )
    assert len(paths) > 100, f"source scan found only {len(paths)} files"
    return paths


def test_nothing_imports_ipython_to_ask_whether_there_is_one() -> None:
    """A shell that could be running this has already imported IPython.

    Which makes ``sys.modules`` the exact answer, and an import the one way
    of asking that costs 0.64 s to hear "no" -- paid by every window an
    operator opened from a plain process: console, device manager, pulse
    editor, figure viewer.  Two packages install the hook a Jupyter kernel
    needs, and both had asked the expensive way.

    So the rule is not "never import IPython": a notebook view legitimately
    does, inside the method that only ever runs in a notebook.  The rule is
    that IMPORTING it and ASKING it for the running shell cannot be the
    same file, because a file that has to ask is a file that does not know,
    and one that does not know must read ``sys.modules`` instead.
    """

    offenders = []
    for path in _sources():
        source = path.read_text(encoding="utf-8")
        if "get_ipython" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Call):
                called = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
                names = [
                    argument.value
                    for argument in node.args[:1]
                    if called in {"__import__", "import_module"}
                    and isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                ]
            else:
                continue
            offenders.extend(
                f"{path.relative_to(PACKAGES)}:{node.lineno}: imports {name} "
                f"in a file that asks get_ipython()"
                for name in names
                if name == "IPython" or name.startswith("IPython.")
            )
    assert offenders == [], chr(10).join(offenders)


def test_opening_the_console_leaves_the_engines_to_the_render_children() -> None:
    """The GUI process draws no raster and solves no fit, so it holds neither.

    It used to hold both, because discovering the logic nodes imported each
    node's implementation to read its descriptor and one of those reached a
    constant through the fit engine: 1.31 s and a couple of hundred
    megabytes of numba, llvmlite and scipy, in a process whose panels are
    all drawn somewhere else.
    """

    script = f'''
import sys
import tempfile
from pathlib import Path

import zou_lab_control  # noqa: F401
sys.path.insert(0, str(Path({str(REPO_ROOT)!r}) / "packages" / "zlc_workbench" / "tests"))
from pulse_fixtures import write_ordinary_pulse
from zlc_workbench.apps.task_console import create_window
from zlc_workbench.session import Workspace

root = Path(tempfile.mkdtemp(prefix="zlc-weight-"))
Workspace(root).prepare()
write_ordinary_pulse(root)
window = create_window(workspace=root, template="virtual", window_ratio=0.25)
carried = [
    name for name in {FOREIGN_TO_A_CONSOLE!r}
    if name in sys.modules
    or any(item.startswith(name + ".") for item in list(sys.modules))
]
window.close()
assert not carried, carried
'''
    environment = dict(__import__("os").environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
