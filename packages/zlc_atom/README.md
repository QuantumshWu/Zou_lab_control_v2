# zlc_atom

`zlc_atom` is the headless atom-device and logic-node package. It contains the
camera/sequencer contracts and the camera-measurement, occupancy, and
calibration nodes. Virtual implementations live only under
`devices/simulation/`: `VirtualCamera` satisfies the same runtime-checkable
`CameraAdapter` contract as DCAM/Pylon, and `VirtualSequencer` is a
`SequencerDevice` over the same pulse-device surface as hardware.

The calibration mathematics under `nodes/calibration/` is headless and has no
Qt or plotting imports. Calibration consumes a project-owned
`zlc.pulse.v1` JSON document through the public `zlc_pulse` codec/resolution
API; there is no Python pulse-module format or second resolver.

Node results also expose `zlc-data` role-axis `OwnedSnapshot` artifacts, so
repeat, site, and readout-event meaning is carried by `DatasetSchema` rather
than inferred from array shape.

## Leaf pattern

Add a measurement or processor by creating its package plus a
`logic_node.py` that exports `LOGIC_NODE`. Discovery is derived from those
files, so the composition framework and UI do not need edits. A device type is
similarly declared in a discovered `device_types.py` module with a factory and
authoring schema.

The current device set is deliberately closed to camera and sequencer
capabilities. Virtual, DCAM, and Pylon camera adapters implement the same
`CameraAdapter` contract; Workbench resolves any compatible named instance such as
`camera` or `mot_camera` instead of hard-coding an instance name.

## Leaf tutorial

A new device leaf stays discoverable and self-contained:

```python
from zlc_atom.authoring import AuthoringSchema
from zlc_atom.install import DeviceTypeDescriptor, InstalledLeaf

def factory(context, key, values):
    device = build_device(context, values)
    return InstalledLeaf(key, "example.device", device, {})

DEVICE_TYPE = DeviceTypeDescriptor(
    "example.device", "example", AuthoringSchema(()), (), factory=factory
)
```

Place that descriptor in a `device_types.py` below `src/zlc_atom/devices/` and
the rglob discovery test will collect it without editing the graph. Logic
leaves follow the same pattern with a `logic_node.py` exporting `LOGIC_NODE`.
Factories must return declared capability instances; startup failures are
reported per leaf in `Installation.failures`, while independent leaves remain
usable and close in reverse order.

## Execution layers

The runtime has one direction of responsibility:

| Layer | Owns | Does not own |
| --- | --- | --- |
| Device | camera/sequencer protocols, buffers, trigger routing, virtual imaging world | calibration policy or analysis |
| Measurement | arm/read/finish observation and publication of camera frames | pulse selection or `sequencer.load/fire` |
| Task | pulse resolution, sequencer load/fire, repeated capture, progress, typed preview/final publication, calibration, and artifact creation | reusable readout mathematics or report rendering |
| Processor | consume a frames signal plus calibration and derive counts/occupied/rate with lineage | excitation or camera control |

For a manually controlled experiment, the notebook calls `resolve_pulse`,
`sequencer.load`, and `sequencer.fire` around a pure camera measurement. For
an automated experiment, `CalibrationTask.run()` owns that whole sequence and
returns a saved calibration artifact. Calibration discovers its site count and
centers from acquired images; it accepts no authored grid rows, columns, or
site count. It publishes a live `capture_preview` and one typed six-dataset
FINAL sibling bundle; the Workbench report adapter renders that bundle without
putting a calibration object or report blob on the signal plane.
`OccupancyProcessor` consumes an explicit frames signal plus the typed saved
calibration and selects `default`, `box`, `psf`, or `uniform_psf` readout.

The supported headless product path discovers the three logic descriptors and
hosts them through the real runtime plane: virtual Calibration writes a plain
workspace JSON, Camera Measurement publishes finite or `Repeat = 0` infinite
frames, and Occupancy consumes the frames key plus JSON path. Virtual and
physical cameras differ only below the adapter boundary.

## Executable integration path

Run from the monorepo after importing the root bootstrap before any `zlc_*`
package, so the checkout under test cannot be confused with an older editable
installation:

```powershell
python -c "import zou_lab_control_v2; import zlc_atom; print(zlc_atom.__file__)"
pytest -q
```

The formal virtual guard uses the real `zlc_runtime.SignalDataPlane`, catalog,
descriptor and `NodeHost` paths. Finite processors drain ordered `FollowTap`
events or consume a retained final snapshot once; infinite sources expose only
latest data and no loss telemetry. `zlc_pulse` remains a separate protocol
package, and the virtual sequencer has no alternate analysis path or
`if virtual` branch.

## Current package boundary

Real camera adapters remain under `devices/camera/`; all virtual camera,
sequencer, shared-world geometry, and virtual descriptors remain under
`devices/simulation/`. Installation descriptors expose only operator-owned
settings. Logic descriptors own their authoring schema, typed artifact/resource
inputs, live/final outputs, and task preview/report declarations; Workbench and
Qt consume those contracts rather than rebuilding them.
