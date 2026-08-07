"""A saved figure, republished as the signals it was drawn from.

v1's viewer is worth inheriting whole: it does not build a bespoke window for
a saved file.  It turns the file into a PRODUCER -- one that declares the same
outputs a live one does -- and then the viewer is a real board with those
signals on it.  Everything the console already knows how to do comes free:
another panel on the same data under a different kind, the signal picker,
re-wiring, fitting, saving again.  A bespoke viewer has to be taught each of
those separately, and had been taught none.

Nothing here interprets the data.  The archive already carries each dataset as
a snapshot plus the manifest that says what it is; this only gives that pair a
name and a producer identity so the plane will accept it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from zlc_data import snapshot_from_manifest
from zlc_runtime import DatasetOutputDeclaration, FinalDatasetOutput, SignalDataPlane


__all__ = ["ArchiveSession", "LoadedFigure"]

#: What a republished dataset claims to be.  A saved figure is not a live
#: measurement and does not pretend to be one: the contract says where it came
#: from, so a processor that would only accept live data can refuse it.
LOADED_CONTRACT = "zlc.loaded-figure"


class LoadedFigure:
    """One saved archive, as a producer of the datasets it stored."""

    def __init__(self, name: str, datasets: Mapping[str, Any]) -> None:
        self.instance_id = f"figure/{name}"
        self._snapshots = dict(datasets)
        self._declarations = tuple(
            DatasetOutputDeclaration(name=key, contract_id=LOADED_CONTRACT)
            for key in self._snapshots
        )

    @classmethod
    def from_archive(
        cls,
        name: str,
        info: Mapping[str, Any],
        arrays: Mapping[str, Any],
    ) -> "LoadedFigure":
        """Read every dataset the archive stored, by its own manifest."""

        manifests = info.get("sections", {}).get("dataset", {})
        datasets = {
            str(key): snapshot_from_manifest(manifest, arrays)
            for key, manifest in manifests.items()
        }
        return cls(name, datasets)

    # -- the producer contract the plane routes on ------------------------

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return self._declarations

    def signal_key(self, output_name: str) -> str:
        return f"@figure/{self.instance_id.split('/', 1)[1]}/{output_name}"

    # -- publishing -------------------------------------------------------

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return tuple(self._snapshots)

    def publish_into(self, plane: SignalDataPlane) -> tuple[str, ...]:
        """Put every stored dataset on the plane, once and finally.

        FINAL because that is what a saved file IS: it will not deliver again.
        A live producer publishes generation after generation; this one has
        exactly one to give, and saying so is what lets a panel treat it the
        same way without waiting for a second that never comes.
        """

        if not self._snapshots:
            return ()
        # A generation first, even for a file: the plane routes on "which run
        # of which owner", and a saved figure is exactly one run of one owner.
        # Publishing without opening it is how a producer that never started
        # asks to be believed.
        plane.begin_generation(self)
        published = plane.publish_final(
            self,
            {
                declaration.name: FinalDatasetOutput(
                    declaration=declaration,
                    snapshot=self._snapshots[declaration.name],
                )
                for declaration in self._declarations
            },
        )
        return tuple(published)


class ArchiveSession:
    """What a console needs when there is no experiment behind it.

    A console asks its session for two things: the plane it draws from and the
    folder it writes into.  A saved archive can answer both -- the plane it
    fills itself, and the folder is where the file already lives -- so the
    whole console works over a file with no devices, no board and no pulse.
    """

    def __init__(self, path: str | Path, info: Mapping[str, Any], arrays: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.signal_plane = SignalDataPlane()
        self.figure = LoadedFigure.from_archive(self.path.stem, info, arrays)
        self.signals = self.figure.publish_into(self.signal_plane)

    def day_folder(self) -> Path:
        """Beside the archive: a figure saved from one belongs with it."""

        return self.path.parent

    def close(self) -> None:
        close = getattr(self.signal_plane, "close", None)
        if callable(close):
            close()
