# zlc_atom

`zlc_atom` is the headless atom-device and logic-node skeleton.  The first
release intentionally contains only camera/sequencer devices and the
camera-measurement, occupancy, and calibration nodes.  Virtual devices use the
same adapter and node paths as hardware; only the lowest device protocol is
faked.

The calibration mathematics under `nodes/calibration/` is headless and has no
Qt, plotting, or device imports. The `src/zlc_atom` package does not import
`zlc_pulse`; only the project-owned Python pulse definitions under `pulses/`
may use that separate compiler package.

Node results also expose `zlc-data` role-axis `OwnedSnapshot` artifacts, so
repeat, site, and readout-event meaning is carried by `DatasetSchema` rather
than inferred from array shape.

## Leaf pattern

Add a measurement or processor by creating its package plus a
`logic_node.py` that exports `LOGIC_NODE`. Discovery is derived from those
files, so the composition framework and UI do not need edits. A device type is
similarly declared in a discovered `device_types.py` module with a factory and
authoring schema.

The initial device set is deliberately closed to `camera` and `sequencer`;
the pylon camera adapter and all other device/node leaves are postponed until
this skeleton has been exercised through several correction rounds.

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
| Task | pulse resolution, sequencer load/fire, repeated capture, calibration, final report publication | reusable readout mathematics |
| Processor | consume a frames signal plus calibration and derive counts/occupied/rate with lineage | excitation or camera control |

For a manually controlled experiment, the notebook calls `resolve_pulse`,
`sequencer.load`, and `sequencer.fire` around a pure camera measurement. For
an automated experiment, `CalibrationTask.run()` owns that whole sequence and
returns the calibration/report product; its short-shot publication can be
passed directly to `OccupancyProcessor`.

The supported offline acceptance path is in
[`notebooks/usage.ipynb`](notebooks/usage.ipynb): virtual installation,
camera measurement, calibration, and occupancy all run without hardware or a
GUI.

## Executable integration path

Install the three local packages in editable mode from the workspace parent,
then install this package:

```powershell
python -m pip install -e ..\zlc_data -e ..\zlc_runtime -e ..\zlc_pulse -e .
pytest -q
python -m jupyter nbconvert --to notebook --execute notebooks/usage.ipynb --inplace
```

The notebook uses the real `zlc_runtime.SignalDataPlane` and executes both
paths: explicit user-owned pulse load/fire plus finite/monitor observation,
then one-call `CalibrationTask` orchestration followed by occupancy lineage.
It also runs the frozen oracle calibration with the required 29/360 error
count. `zlc_pulse` remains a separate protocol package; the virtual sequencer
has no alternate analysis path or `if virtual` branch.

## Migration line-count report

The report below counts physical Python lines as of 2026-08-04. Reference paths
are relative to the read-only `Zou_lab_control_v1_claude/Zou_lab_control_v1`
tree. A smaller count is intentional where `zlc_atom` retains only the closed
camera/sequencer and three-logic-node scope; it is not a claim of a full v1
feature migration.

| zlc_atom module | lines | reference bottom | lines |
| --- | ---: | --- | ---: |
| `nodes/calibration/bimodal.py` | 264 | `logic_nodes/readout/bimodal.py` | 254 |
| `nodes/calibration/calibration.py` | 873 | `logic_nodes/readout/calibration/calibration.py` | 1,053 |
| `nodes/calibration/psf.py` | 51 | `logic_nodes/readout/calibration/psf.py` | — |
| `nodes/calibration/task.py` | 221 | `logic_nodes/readout/calibration/task.py` | 311 |
| `nodes/_framework/camera_measurement.py` | 226 | `logic_nodes/camera_measurement/signal_source.py` | 469 |
| `nodes/camera_measurement/measurement.py` | 8 | public facade | — |
| `nodes/occupancy/processor.py` | 167 | `logic_nodes/readout/occupancy/processor.py` | 276 |
| `devices/camera/dcam.py` | 692 | `devices/camera/dcam.py` | 750 |
| `devices/camera/_dcam_driver.py` | 447 | `devices/camera/_dcam_driver.py` | 511 |
| `devices/camera/_owner_lane.py` | 65 | `devices/camera/_owner_lane.py` | 80 |
| `devices/camera/virtual.py` | 261 | `devices/simulation/apparatus.py` | 2,131 |
| `devices/camera/world.py` | 113 | `devices/simulation/apparatus.py` | 2,131 |
| `devices/sequencer/virtual.py` | 165 | `devices/simulation/apparatus.py` | 2,131 |
| `execution/ports.py` | 276 | `runtime/ports.py` | 734 |
| `execution/resources.py` | 129 | `runtime/resources.py` | 302 |
| `execution/run.py` | 335 | `runtime/run.py` | 1,197 |
| `install/graph.py` | 156 | `installation_runtime.py` | 328 |

The DCAM trio is a direct reference-stack migration with only validator/import
adaptation. The virtual apparatus is split into the camera-owned `world.py`,
one asynchronous camera adapter, and one sequencer adapter so imaging physics
has a single owner. The calibration math is likewise colocated with its task;
the shared observation primitive is framework infrastructure used by both the
public measurement facade and automated tasks. The execution layer is the
requested fail-closed minimum (run lease, cancellation sealing, broker-only
identity, and resource arbiter), not the reference application's larger
session/UI runtime. These choices are covered by frozen-oracle tests, runtime
integration tests, and the import/grep guards in `tests/`.
