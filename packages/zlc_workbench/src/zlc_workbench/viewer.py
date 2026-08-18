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

from zlc_plot import DEFAULTS

from .archive import read_archive, read_dataset
from .panel_save import (
    restore_panel_plot_annotations,
    restore_panel_plot_input,
)
from .panel_state import PanelState, apply_panel_fit
from .plot_annotations import (
    PanelPlotAnnotations,
)


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
    def _label(key: str) -> str:
        entry = _panel_record(sections, key)
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


def _panel_record(
    sections: Mapping[str, Any],
    dataset: str,
) -> Mapping[str, Any] | None:
    """Return the panel record which explicitly names ``dataset``."""

    panel = sections.get("panel", {})
    if not isinstance(panel, Mapping):
        return None
    if str(panel.get("dataset", "")) == str(dataset):
        state = panel.get("state")
        return state if isinstance(state, Mapping) else panel
    record = panel.get(str(dataset))
    return record if isinstance(record, Mapping) else None


def _panel_state(
    sections: Mapping[str, Any],
    dataset: str,
) -> PanelState | None:
    panel = sections.get("panel", {})
    if not isinstance(panel, Mapping) or str(panel.get("dataset", "")) != str(dataset):
        return None
    document = panel.get("state")
    if not isinstance(document, Mapping):
        return None
    return PanelState(
        signal=str(document.get("signal", "")),
        kind=str(document.get("kind", "")),
        cell_kind=str(document.get("cell_kind", "")),
        size=str(document.get("size", "")),
        interval_ms=int(document.get("interval_ms", 500)),
        title=str(document.get("title", "")),
        semantic=dict(document.get("semantic", {})),
        display=dict(document.get("display", {})),
        fit=dict(document.get("fit", {})),
        overlay_signal=str(document.get("overlay_signal", "")),
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
    for dataset in datasets:
        entry = _panel_record(sections, dataset)
        if isinstance(entry, Mapping):
            title = entry.get("title") or dataset
            rows.append(
                (f"panel {dataset}", f"{title} — {entry.get('signal', '')}")
            )
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
        make_host: Callable[[Any, str, PanelState | None], Any],
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
        self.panel_state: PanelState | None = None
        self.plot_annotations = PanelPlotAnnotations()
        self._host: Any = None
        self.view.set_panel_sizes(
            DEFAULTS.layout.size_names, DEFAULTS.layout.default_preset
        )
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
        # fact; how a caller happened to write it is not.  Derived saves must
        # keep landing beside this file even if the process cwd later changes.
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

        Panel Save Fig writes one dataset; notebook-created figure archives may
        still contain several.  The viewer therefore keeps the dataset choice
        explicit without implying that TaskConsole saves the whole board.
        """

        if not self._info or not name:
            return False
        # The plot is titled with what the dataset IS, not the archive's key
        # for it: "panel-2" tells an operator nothing about what they opened.
        label = dict(self.description.datasets).get(str(name), str(name)) if self.description else str(name)
        host: Any = None
        try:
            snapshot = read_dataset(self._info, self._arrays, str(name))
            plot_input = restore_panel_plot_input(
                self._info,
                self._arrays,
                str(name),
                snapshot,
            )
            panel_state = _panel_state(self._info.get("sections", {}), str(name))
            annotations = restore_panel_plot_annotations(self._info, str(name))
            host = self._make_host(
                plot_input,
                label,
                panel_state,
            )
            fit_error = self._configure_host(host, panel_state, annotations)
        except Exception as error:
            if host is not None:
                close = getattr(host, "close", None)
                if callable(close):
                    close()
            self._mount(None)
            self.view.set_status(f"cannot draw {name}: {error}", error=True)
            return False
        self._mount(host)
        self.dataset = str(name)
        self.panel_state = panel_state
        self.plot_annotations = annotations
        total = len(self.description.datasets) if self.description else 1
        position = "" if total <= 1 else f"  ({total} datasets in this file)"
        suffix = "" if fit_error is None else f"; saved fit was not restored: {fit_error}"
        self.view.set_status(f"showing {name}{position}{suffix}")
        return True

    @staticmethod
    def _await(operation: object) -> object:
        return operation.result() if hasattr(operation, "result") else operation

    @classmethod
    def _configure_host(
        cls,
        host: object,
        state: PanelState | None,
        annotations: PanelPlotAnnotations | None = None,
    ) -> Exception | None:
        """Apply the saved panel decisions through the plot host's public API."""

        selected_annotations = (
            PanelPlotAnnotations() if annotations is None else annotations
        )
        # The host was built from this same record and already holds the
        # appearance it accepts; re-sending the whole saved bag is how names
        # authored under another kind reached a vocabulary that never
        # declared them.  Only what this call adds travels.
        display: dict[str, object] = {}
        thresholds: tuple[float | None, ...] | None = None
        if not selected_annotations.empty:
            display["threshold_classifier"] = True
            thresholds = selected_annotations.classifier_thresholds
        configuration: dict[str, object] = {"parameters": display}
        if state is not None:
            configuration["size"] = state.size
        if thresholds is not None:
            configuration["classifier_thresholds"] = thresholds
        cls._await(host.configure(**configuration))

        if state is None:
            return None

        # One authority decides what this panel's fit record means; a viewer
        # host is static, so it AWAITS the analysis rather than presenting it.
        operation = apply_panel_fit(host, state, live=False)
        if operation is not None:
            try:
                cls._await(operation)
            except Exception as error:
                # The dataset and authored plot state remain useful when a fit
                # model is no longer available or this archived fit was never
                # executable.  Report that one omission without discarding the
                # otherwise exact figure.
                return error
        return None

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
        try:
            result = self._host.configure(size=str(size))
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
        save = getattr(self._host, "save", None)
        if not callable(save):
            self.view.set_status("this figure cannot save itself", error=True)
            return ""

        def write(temporary: Path) -> None:
            result = save(temporary)
            if hasattr(result, "result"):
                result.result()

        try:
            target = unique_path(
                self.path.parent,
                f"{self.path.stem}-{self.dataset or 'figure'}",
                ".png",
                writer=write,
            )
        except Exception as error:
            self.view.set_status(f"cannot save: {error}", error=True)
            return ""
        self.view.set_status(f"saved {target.name}")
        return str(target)

    def _mount(self, host: Any) -> None:
        previous, self._host = self._host, host
        if host is None:
            self.panel_state = None
        self.view.show_figure(host)
        if previous is not None:
            previous.close()

    def close(self) -> None:
        self._mount(None)
