from __future__ import annotations

import builtins
import importlib
import inspect
from pathlib import Path
import re

import zlc_pulse
from zlc_pulse import remote, wire
from zlc_pulse.remote import REMOTE_METHODS, RemotePulseStreamer


#: Raised deliberately, each time for a capability that was MISSING:
#: MINIMUM_REPEAT_COUNT, because the editor invented its own answer to "what
#: is the smallest count that IS a repeat" and got it wrong; and the three
#: codec names, because a PulseSequence had no persisted form at all.
#:
#: 26 -> 34 for the eight the editor was reaching into submodules for, each
#: for a capability the facade did not otherwise offer: TIME_UNIT_CHOICES /
#: TIME_UNIT_TO_NS (which units a duration may use, and what they are worth in
#: ticks -- the editor cannot offer a unit box without them); DEFAULT_HOST /
#: DEFAULT_PORT (where a board answers, written in five places across four
#: packages before it had one); the three scan names (what columns a bound
#: pulse has, a starting table, and whether a table is legal -- the Scan page
#: is those three questions); and MemoryRegisterTransport, which is how a
#: board is driven with no board present.
#:
#: Every one is DECLARED here and taught in the tutorial.  A sibling package
#: reaching into a submodule for something this list does not name is the same
#: surface with the declaration skipped, which is worse than a wide facade --
#: nobody can see how wide it really is.
#: 34 -> 36 for the scan crossing.  A table is written in the units the editor
#: shows and held in the units the wire uses, and until now the difference was
#: the author's problem: the template offered offset-binary codes and device
#: ticks for fields whose own boxes are signed codes and microseconds.  Naming
#: both directions makes the conversion a place instead of an instruction.
MAX_PUBLIC_NAMES = 38
EXPECTED_PUBLIC_NAMES = (
    "PulseStreamer",
    "RemotePulseStreamer",
    "connect",
    "serve",
    "PulseSequence",
    "PulsePeriod",
    "AnalogStep",
    "PulsePortSpec",
    "PulseTarget",
    "PulseSlot",
    "PulseFieldRef",
    "OutputDelay",
    "MINIMUM_REPEAT_COUNT",
    "PULSE_TREE_FORMAT",
    "sequence_from_tree",
    "sequence_to_tree",
    "RepeatRegion",
    "compile_sequence",
    "pulse_target_from_xdc",
    "load_streamer_config",
    "UartRegisterTransport",
    "VivadoAxiRegisterTransport",
    "MemoryRegisterTransport",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "TIME_UNIT_CHOICES",
    "TIME_UNIT_TO_NS",
    "align_to_grid",
    "resolve_scan_point",
    "scan_columns_for",
    "scan_table_template",
    "validate_scan_table",
    "scan_rows_to_wire",
    "scan_rows_from_wire",
    "RemoteError",
    "UartError",
    "BackendResolutionError",
    "__version__",
)


def test_package_exports_are_the_final_surface_and_not_builtin_shadowing() -> None:
    names = tuple(zlc_pulse.__all__)
    assert names == EXPECTED_PUBLIC_NAMES
    # Derived, not re-typed: the listing above IS the count.  Raised twice.
    # For MINIMUM_REPEAT_COUNT -- the smallest count that is a repeat is a fact
    # of the pulse model that anyone authoring one needs, and leaving it
    # private meant the editor invented its own answer and got it wrong.  And
    # for align_to_grid: this package could say whether a value was ON the
    # clock grid but not WHERE the grid is, so an editor had nothing to round
    # with and refused the operator mid-keystroke instead.  And for
    # resolve_scan_point, because holding one point of a scan is playing an
    # ORDINARY pulse that carries that row -- with no way to say so, an editor
    # said it as a scan table of length one instead, and the board it steered
    # into that corner stopped applying its DAC segments.
    assert len(names) == len(set(names)) == len(EXPECTED_PUBLIC_NAMES)
    assert len(names) <= MAX_PUBLIC_NAMES
    seen_ids: dict[int, str] = {}
    for name in names:
        value = getattr(zlc_pulse, name)
        if isinstance(value, int) and -5 <= value <= 256:
            continue
        previous = seen_ids.setdefault(id(value), name)
        assert previous == name, (previous, name)
        builtin_value = getattr(builtins, name, None)
        if builtin_value is not None:
            assert value is not builtin_value


def test_removed_implementation_types_stay_in_their_own_submodules() -> None:
    """Implementation, not surface -- and the list shrinks when that changes.

    MemoryRegisterTransport left it: it was listed as a test transport, and it
    stopped being one when the shipped window grew a "virtual" mode built on
    it.  The real command sequence with the wire substituted is a capability
    an operator uses, not a fixture, and a name returns to the facade the
    moment a sibling needs it for something the facade does not otherwise do.
    """

    for name in (
        "CompiledProgram",
        "RegisterTransport",
        "PulseRemoteServer",
        "StreamerParams",
        "AppliedState",
        "DoneReport",
        "SafeReadback",
        "trigger_times",
    ):
        assert not hasattr(zlc_pulse, name), name
    assert importlib.import_module("zlc_pulse.compile").CompiledProgram
    assert importlib.import_module("zlc_pulse.transport").RegisterTransport
    assert importlib.import_module("zlc_pulse.wire").StreamerParams
    assert importlib.import_module("zlc_pulse.device").AppliedState
    assert importlib.import_module("zlc_pulse.schedule").trigger_times
    assert not hasattr(wire, "PulseFieldRef")
    assert hasattr(remote, "PulseRemoteServer")


def test_contract_export_list_matches_package_export_list_in_both_directions() -> None:
    contract = (Path(__file__).resolve().parents[1] / "docs" / "contract.md").read_text(encoding="utf-8")
    block = contract.split("__all__ = (", 1)[1].split(")\n```", 1)[0]
    names = tuple(re.findall(r'"([A-Za-z_]\w*)"', block))
    assert names == EXPECTED_PUBLIC_NAMES
    assert set(names) == set(zlc_pulse.__all__)


def test_remote_adds_only_connection_management_methods() -> None:
    methods = {
        name
        for name, value in inspect.getmembers(RemotePulseStreamer, inspect.isfunction)
        if name in set(REMOTE_METHODS) | {"disconnect", "__enter__", "__exit__"}
    }
    assert methods - set(REMOTE_METHODS) == {"disconnect", "__enter__", "__exit__"}


def test_source_line_budget_excludes_only_the_rtl_engine_model() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    files = tuple(source_root.rglob("*.py"))
    counted = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in files if path.name != "engine_model.py")
    excluded = next(path for path in files if path.name == "engine_model.py")
    assert excluded.name == "engine_model.py"
    # X0.5 deliberately migrates the v1 SAFE/LOAD protocol and keeps the
    # audit helpers in the device layer; retain a small explicit cap for that
    # behavior instead of compressing safety code for a line-count game.
    # Raised deliberately, twice, each time for a capability that was missing
    # rather than for room to sprawl: 5_700 -> 5_800 for describe(), because a
    # board that cannot state its own ports, pins and clock forces every client
    # to guess them from local files; 5_800 -> 6_000 for the scan model, which
    # owns what a legal scan table is and which a window would otherwise write
    # for itself -- that is how a DAC column once inherited a duration's range
    # and every point came back clamped; 6_000 -> 6_060 for trigger_windows and
    # the two program methods over it, so a device twin can ASK how many camera
    # windows a program opens and how long each is, instead of keeping its own
    # fifty-line copy of the rising-edge rule the board plays in hardware;
    # 6_060 -> 6_100 for the affine slot-operand narrowing, which was written
    # only in the RTL mirror -- so the host predicted a schedule the board would
    # not play for any duration slot past 25 signed bits, validated it, and
    # accepted it; 6_100 -> 6_130 for the content-name width, stated here as
    # well as in zlc_data because this package imports neither it nor anything
    # else beyond numpy and pyserial; 6_130 -> 6_155 for DoneReport.fault, so
    # that what counts as a bad shot is decided where the report is defined --
    # every caller used to throw the report away, and a shot that errored,
    # underran, or never finished looked exactly like a clean one; 6_155 ->
    # 6_180 for the memory transport modelling COMMAND as the rising edge the
    # RTL detects, so a host that forgets to zero between commands fails here
    # instead of only on hardware; 6_180 -> 6_200 for the scan columns saying
    # what the wire actually reads -- ticks for a duration, offset-binary for a
    # DAC -- instead of nanoseconds and signed codes; 6_200 -> 6_240 for
    # validate_scan_table, because what a legal table is was decided in three
    # places and the file-load path skipped the width check the other two made;
    # 6_240 -> 6_440 for the pulse codec, because a PulseSequence had no
    # persisted form AT ALL: an editor could change one and had nowhere to put
    # the change, so its Save button was wired to a refusal.  A pulse's own
    # tree belongs to the package that owns the model, not to whoever happens
    # to be writing a file -- but it stays a submodule, since a serialiser is
    # not something a person learning to drive a board needs to be taught.
    # 6_465 -> 6_490 for the eight names the editor was reaching into
    # submodules for: the facade grew the import lines, not the code behind
    # them.  Nothing new was written -- what changed is that the surface now
    # says how wide it is.
    # 6_490 -> 6_515 for endpoint.py, which is the same three constants moved
    # out of the server module and given their own.  Publishing them from
    # there meant the package imported the SERVER to read a port, so
    # "python -m zlc_pulse.remote" loaded that module twice and Python said so
    # on every server start.  A constant must not oblige anyone to load a
    # socket server to read it.
    # 7_230 -> 7_310 for resolve_scan_point: a held scan point is an ordinary
    # pulse carrying that row's numbers, and nothing here could say that, so
    # the editor said it as a one-point scan table looping forever -- a state
    # the board is never otherwise asked for, and one where it stopped
    # re-applying the DAC segments while the digital edges kept playing.
    # 7_310 -> 7_330 corrects that change's shipped line count; the whole-tree
    # cap below already covered the same implementation and does not move.
    # 7_200 -> 7_230 for align_to_grid: v1 could say where the clock grid is
    # (time_value_per_tick) and this package could only say whether you were on
    # it, so an editor had nothing to round WITH and refused the operator
    # instead -- with a blocking dialog, while they were still typing.
    # 7_170 -> 7_200 for stating which lines can LOSE things.  The strobe
    # verify-and-retry is sound only where a timeout means nothing is still in
    # flight -- true of a UART frame, false of a Vivado TCL that may execute
    # after its deadline -- so each transport declares its loss semantics and
    # the machinery runs only where the declaration holds.  Found by review:
    # the courtesy window would have made every JTAG fire time out spuriously,
    # and the verify race could have doubled one.
    # 7_110 -> 7_170 for resolving a lost FIRE acknowledgement by observation
    # instead of by guessing.  Blind resending is two shots and blind failing
    # killed every sixth On Pulse on a line losing a byte per hundred; the
    # board itself knows whether it fired -- accepting FIRE consumes LOADED
    # and raises RUNNING -- so the status register arbitrates, and only a
    # verified-idle board is strobed again.  SAFE and LOAD are idempotent in
    # effect and simply ride the resending line.
    # 6_950 -> 7_110 for surviving a noisy line in all three of its fault
    # shapes, found by review and by the rig in the same afternoon: a LOST
    # reply resends the frame with a fresh time slice per attempt (one shared
    # deadline had made the first resend implementation inert -- attempt one
    # waited the whole budget out); a CORRUPTED reply is dropped at extraction
    # with a one-byte resync, so it neither kills the write (the rig's
    # "method=fire FrameError: CRC mismatch") nor misaligns the frames behind
    # it; a REJECTED request (ST_CRC_FAIL) is provably unexecuted, so even a
    # command strobe may go again, while a strobe whose acknowledgement is
    # simply missing still refuses to guess.  Reads retry the same way, which
    # is what stopped Stop from stalling five seconds per lost byte.
    # 6_830 -> 6_950 for resending a frame the board never answered.  Its
    # bridge is a single-frame state machine: one mis-sampled stop bit makes it
    # abandon the frame it was reading, and that frame is never acknowledged.
    # Every frame already carried a SEQ and every acknowledgement carried it
    # back -- the field was used only to assert equality, so one lost frame in
    # a load of ten failed the whole load.  Command strobes are excluded and
    # now go on their own, after the data they act on is acknowledged.
    # 6_780 -> 6_830 for a UART deadline that scales with the transfer, and a
    # timeout that says what was in flight.  One constant covered a register
    # read and a whole register image alike, so it was either too generous to
    # report a dead link quickly or too tight for a slow host -- and "UART
    # reply timed out" was the same sentence whether the board answered
    # nothing, answered half, or answered in bytes that never formed a frame.
    # 6_740 -> 6_780 for playing a table N times without sending it N times:
    # the RTL has no sweep register, so a finite scan is N*len(rows) points and
    # which row a point takes is decided when a bank is refilled.
    # 6_700 -> 6_740 for logging a POLLED answer only when it changes.  A poll
    # is a question, not an event: wait_done is asked every 10 ms by a client
    # that owns its own poll loop, and printing a line per ask buried the run
    # in four hundred identical ones.
    # 6_650 -> 6_700 for the separation between where a generated sweep is
    # SEEDED and what the board will actually accept.  They were one pair of
    # numbers, so the only thing that knew a slot cannot exceed a 25-bit
    # operand was the device -- which said so in ticks, at load time, after
    # compiling.  A column now carries its own legal range, the template
    # prints it, and a table is refused in the unit its author wrote it in.
    # 6_515 -> 6_580 for the scan crossing: a scan column now describes itself
    # in the unit the EDITOR shows for the field it scans, and carries the
    # scale and offset that reach the wire, so scan_rows_to_wire/from_wire can
    # be one place instead of an instruction to the person writing the table.
    # Before this the template offered offset-binary codes and device ticks for
    # fields whose own boxes are signed codes and microseconds -- one output,
    # two number systems, and nothing on screen saying which.
    # 6_440 -> 6_465 for whole_pulse_repeat, because "does this bracket replace
    # the outer level" was worked out by whoever happened to be reading the
    # document -- the preview did, a presentation helper that had lost its
    # caller did, and the path that actually fires did not, so a bracket around
    # the whole pulse was drawn faithfully and then run forever anyway.
    assert counted <= 7_330
    # The whole-tree cap moves with it, for the same reason: two numbers
    # describing one budget must not drift apart.  It sits a little above the
    # cap plus the excluded model, which is what "the rest of the tree, plus
    # the RTL mirror" means.
    # Moves with the cap above, for the reason stated there: two numbers
    # describing one budget must not drift apart.
    assert counted + len(excluded.read_text(encoding="utf-8").splitlines()) < 8_490


def test_build_tools_live_in_the_fpga_submodule_not_package_exports() -> None:
    build = importlib.import_module("zlc_pulse.fpga")
    for name in ("emit_geometry_vh", "emit_geom_tcl", "estimate_resources", "solve_capacity", "check_config_capacity"):
        assert name in build.__all__
        assert name not in zlc_pulse.__all__
        assert not hasattr(zlc_pulse, name)


def test_host_compiled_program_does_not_expose_rtl_repeat_register() -> None:
    compiled = importlib.import_module("zlc_pulse.compile").CompiledProgram
    assert not hasattr(compiled, "repeat_from_index")
