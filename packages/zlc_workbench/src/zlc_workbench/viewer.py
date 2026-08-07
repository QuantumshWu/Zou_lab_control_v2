"""Opening a saved figure and reading what it was.

An archive already carries everything: the datasets with their axes, the
apparatus state, what was asked of it, which pulse drove it, which panel showed
what.  Until now it carried them for nobody -- `read_archive` returned a nested
JSON document, and reading a run six months later meant reading that document.

Two jobs, kept apart:

* :func:`describe_archive` projects the document into labelled rows.  It is
  pure, takes no Qt and no session, and is what makes "what was the apparatus
  doing" answerable in a notebook as well as a window.
* :class:`FigureViewerPresenter` connects those rows and a rebuilt dataset to a
  mute view.

Both stop short of interpretation.  Rows are named for the subject they came
from and are not renamed into something friendlier, because a record whose
labels drift from the ones the producing package used is a record you cannot
grep for.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .archive import read_archive, read_dataset


__all__ = ["ArchiveDescription", "FigureViewerPresenter", "describe_archive"]


Rows = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ArchiveDescription:
    """One saved figure, as tabs of labelled rows."""

    name: str
    schema: str
    #: (title, rows) pairs, in the order the view shows them.
    tabs: tuple[tuple[str, Rows], ...]
    #: Which datasets can be reopened, as (key, label) in saved order.  The
    #: key is the archive's own name for it; the label is what it IS, taken
    #: from the panel record that was saved beside it.
    datasets: tuple[tuple[str, str], ...]

    @property
    def dataset_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _label in self.datasets)


def describe_archive(
    info: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> ArchiveDescription:
    """Project one archive's info document into rows a person can read."""

    sections = info.get("sections", {})
    provenance = sections.get("provenance", {})
    panels = sections.get("panel", {})

    def _label(key: str) -> str:
        entry = panels.get(key)
        if not isinstance(entry, Mapping):
            return key
        title = str(entry.get("title") or "").strip()
        signal = str(entry.get("signal") or "").strip()
        if title and signal and title != signal:
            return f"{title} — {signal}"
        return title or signal or key

    keys = tuple(sections.get("dataset", {}))
    datasets = tuple((key, _label(key)) for key in keys)
    return ArchiveDescription(
        name=str(info.get("name", "")),
        schema=str(info.get("schema", "")),
        datasets=datasets,
        tabs=(
            ("Plot", _plot_rows(sections, arrays, keys)),
            ("Measurement", _measurement_rows(sections, provenance)),
            ("Device", _device_rows(provenance)),
            ("Flow", _flow_rows(sections, provenance)),
            ("Raw", _flatten(sections)),
        ),
    )


def _plot_rows(
    sections: Mapping[str, Any],
    arrays: Mapping[str, Any],
    datasets: Sequence[str],
) -> Rows:
    """What is in the file, and what each panel was showing."""

    rows: list[tuple[str, str]] = []
    for name, array in sorted(arrays.items()):
        shape = "x".join(str(size) for size in getattr(array, "shape", ()))
        dtype = getattr(getattr(array, "dtype", None), "name", "")
        reopenable = "" if name in datasets else "  (array only)"
        rows.append((name, f"{shape} {dtype}{reopenable}".strip()))
    for panel, entry in sorted(sections.get("panel", {}).items()):
        if isinstance(entry, Mapping):
            title = entry.get("title") or panel
            rows.append((f"panel {panel}", f"{title} — {entry.get('signal', '')}"))
    return tuple(rows)


def _measurement_rows(
    sections: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Rows:
    """What was asked of the apparatus, and what drove it."""

    rows: list[tuple[str, str]] = []
    for node, record in sorted(provenance.items()):
        if not isinstance(record, Mapping):
            continue
        rows.append((node, str(record.get("layer", ""))))
        captured = record.get("captured_at")
        if captured:
            rows.append((f"{node} captured", str(captured)))
        for key in ("acquisition_parameters", "source_acquisition_parameters"):
            for label, value in sorted(dict(record.get(key, {})).items()):
                rows.append((f"{node}.{label}", _text(value)))
        fingerprint = record.get("calibration_fingerprint")
        if fingerprint:
            rows.append((f"{node} calibration", str(fingerprint)))
    for label, value in sorted(dict(sections.get("pulse", {})).items()):
        rows.append((f"pulse.{label}", _text(value)))
    return tuple(rows)


def _device_rows(provenance: Mapping[str, Any]) -> Rows:
    """The apparatus, exactly as each device reported itself."""

    rows: list[tuple[str, str]] = []
    for node, record in sorted(provenance.items()):
        if not isinstance(record, Mapping):
            continue
        for role, state in sorted(dict(record.get("devices", {})).items()):
            rows.append((f"{node}.{role}", ""))
            if isinstance(state, Mapping):
                for label, value in sorted(state.items()):
                    rows.append((f"    {label}", _text(value)))
            else:
                rows.append(("    state", _text(state)))
    return tuple(rows)


def _flow_rows(
    sections: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Rows:
    """Who fed whom: the chain from apparatus to the panel on screen.

    This is the question a saved figure is least able to answer on its own and
    the one most often asked -- "where did this number come from" -- so it gets
    its own reading rather than being spelled out across three other tabs.
    """

    rows: list[tuple[str, str]] = []
    signal_to_panel: dict[str, str] = {}
    for panel, entry in sorted(sections.get("panel", {}).items()):
        if isinstance(entry, Mapping) and entry.get("signal"):
            signal_to_panel[str(entry["signal"])] = str(entry.get("title") or panel)

    for node, record in sorted(provenance.items()):
        if not isinstance(record, Mapping):
            continue
        upstream = record.get("consumes") or []
        devices = sorted(dict(record.get("devices", {})))
        source = ", ".join(str(item) for item in upstream) or ", ".join(devices) or "—"
        shown = [
            f"{signal}→{panel}"
            for signal, panel in signal_to_panel.items()
            if f"/{node}/" in signal
        ]
        rows.append((node, f"{source}  ⇒  {', '.join(shown) or 'not shown'}"))

    for signal, panel in sorted(signal_to_panel.items()):
        if not any(f"/{node}/" in signal for node in provenance):
            rows.append((panel, f"{signal}  (no record of what produced it)"))
    return tuple(rows)


def _flatten(value: Any, prefix: str = "") -> Rows:
    """Every leaf of the document, so nothing is hidden by the other tabs."""

    if isinstance(value, Mapping):
        rows: list[tuple[str, str]] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            rows.extend(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return tuple(rows)
    return ((prefix, _text(value)),)


def _text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(item) for item in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class FigureViewerPresenter:
    """Connects a saved-figure view to archives on disk."""

    def __init__(
        self,
        view: object,
        *,
        make_host: Callable[[Any, str], Any],
        edit_figure: Callable[[Any, str], object] | None = None,
    ) -> None:
        self.view = view
        self._make_host = make_host
        #: Opening the plot's own controls is Qt work this asks for rather
        #: than does, the same way the console asks for it.
        self._edit_figure = edit_figure
        #: What the operator called the figure, if they renamed it.
        self.figure_title = ""
        self.path: Path | None = None
        self.description: ArchiveDescription | None = None
        #: The archive as read, kept so switching datasets does not re-read the
        #: file: an archive is immutable once written.
        self._info: Mapping[str, Any] = {}
        self._arrays: Mapping[str, Any] = {}
        self.dataset = ""
        self._host: Any = None
        self._connect()

    def _connect(self) -> None:
        self.view.path_committed.connect(self.open)
        self.view.dataset_picked.connect(self.show_dataset)
        self.view.save_image_requested.connect(self.save_image)
        # The figure is a panel, so the decisions a panel carries are answered.
        self.view.figure_size_picked.connect(self.resize_figure)
        # NOT echoed back to the card: it already shows what was typed, and
        # writing it back made the card re-raise the commit -- a loop that
        # ended the process with no traceback, which is what a Qt signal cycle
        # looks like from outside.
        self.view.figure_title_committed.connect(self.rename_figure)
        self.view.figure_edit_requested.connect(self.edit_figure)
        self.view.close_requested.connect(self.close)

    def open(self, path: str) -> ArchiveDescription | None:
        """Read one archive and show it, or say plainly why it cannot be read.

        A file that cannot be opened is the normal case here -- an operator
        types a path, picks the wrong file, opens something from another tool --
        so it is answered, not raised.
        """

        try:
            info, arrays = read_archive(path)
            description = describe_archive(info, arrays)
        except Exception as error:
            self.view.set_status(f"cannot read {Path(path).name}: {error}", error=True)
            return None

        # Resolved, not stored as spelled.  Where an archive IS is an absolute
        # fact; how a caller happened to write it is not.  A relative path came
        # straight through to unique_path, which rightly refuses one -- out of
        # a Qt slot, so Save image ended the process instead of writing a file.
        self.path = Path(path).resolve()
        self.description = description
        # Kept so switching datasets does not re-read the file; an archive is
        # immutable once written, so there is nothing to re-read.
        self._info, self._arrays = info, arrays
        self.view.set_title(description.name or self.path.stem)
        self.view.set_path(str(self.path))
        self.view.set_info(description.tabs)
        self.view.set_datasets(
            description.datasets,
            description.dataset_keys[0] if description.datasets else "",
        )
        self._show_dataset(info, arrays, description)
        return description

    def _show_dataset(
        self,
        info: Mapping[str, Any],
        arrays: Mapping[str, Any],
        description: ArchiveDescription,
    ) -> None:
        if not description.datasets:
            self._mount(None)
            self.view.set_status(
                "opened; its arrays were saved without axes, so it cannot be replotted"
            )
            return
        self.show_dataset(description.dataset_keys[0])

    def show_dataset(self, name: str) -> bool:
        """Draw one of the archive's datasets.

        An archive holds one per panel that was on screen when it was saved, so
        a viewer that only ever draws the first hides the rest -- which reads
        exactly like an archive that kept only one.
        """

        if not self._info or not name:
            return False
        # The plot is titled with what the dataset IS, not the archive's key
        # for it: "panel-2" tells an operator nothing about what they opened.
        label = dict(self.description.datasets).get(str(name), str(name)) if self.description else str(name)
        try:
            snapshot = read_dataset(self._info, self._arrays, str(name))
            host = self._make_host(snapshot, label)
        except Exception as error:
            self._mount(None)
            self.view.set_status(f"cannot draw {name}: {error}", error=True)
            return False
        self._mount(host)
        self.dataset = str(name)
        total = len(self.description.datasets) if self.description else 1
        position = "" if total <= 1 else f"  ({total} datasets in this file)"
        self.view.set_status(f"showing {name}{position}")
        return True

    def rename_figure(self, text: str) -> None:
        """Remember what the operator called this figure."""

        self.figure_title = str(text)

    def resize_figure(self, size: str) -> bool:
        """The card and the picture inside it have to agree.

        A card resized around a figure that stayed 2x2 is a big card with a
        small picture in it -- the same rule the console keeps for its panels.
        """

        if self._host is None:
            return False
        set_size = getattr(self._host, "set_size", None)
        if not callable(set_size):
            return False
        try:
            result = set_size(str(size))
            if hasattr(result, "result"):
                result.result()
        except Exception as error:
            self.view.set_status(f"cannot resize: {error}", error=True)
            return False
        return True

    def edit_figure(self) -> object | None:
        """Open the plot's own controls, which belong to the plotting package."""

        if self._host is None or self._edit_figure is None:
            return None
        return self._edit_figure(self._host, self.dataset or "figure")

    def save_image(self) -> str:
        """Write the figure as it is drawn, beside the archive it came from."""

        from zlc_durable import unique_path

        if self._host is None or self.path is None:
            self.view.set_status("there is no figure to save", error=True)
            return ""
        target = unique_path(
            self.path.parent, f"{self.path.stem}-{self.dataset or 'figure'}", ".png"
        )
        save = getattr(self._host, "save", None)
        if not callable(save):
            self.view.set_status("this figure cannot save itself", error=True)
            return ""
        try:
            result = save(target)
            if hasattr(result, "result"):
                result.result()
        except Exception as error:
            self.view.set_status(f"cannot save: {error}", error=True)
            return ""
        self.view.set_status(f"saved {target.name}")
        return str(target)

    def _mount(self, host: Any) -> None:
        previous, self._host = self._host, host
        self.view.show_figure(host)
        if previous is not None:
            previous.close()

    def close(self) -> None:
        self._mount(None)
