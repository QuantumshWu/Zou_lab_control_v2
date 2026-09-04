@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem ============================================================================
rem  Double-click this file to turn one spare GND output pin into a real digital
rem  pulse channel, and to migrate every pulse file it can find onto the board
rem  that results.  It does the whole host side of the change:
rem
rem    1. fpga\board_config\streamer_config.json  -- INSERT the lane after the
rem       last digital one, renumber the DAC lanes after it, bump channel_count
rem    2. fpga\board_config\board.xdc             -- the pin's GNDn port becomes
rem       the new signal
rem    3. fpga\pulse_streamer\...top.v            -- the port declaration, drop
rem       its tie-low, drive it from out_final[<new index>], and move the
rem       da_clk assigns that the renumbering shifted
rem    4. fpga\pulse_streamer\sim\tb_t_ff.v       -- it names the removed port,
rem       and a named connection to a port that is gone will not compile
rem    5. fpga\pulse_streamer\zlc_geometry.vh     -- regenerated (the LAYOUT
rem       fingerprint the host's connect-check verifies lives in here)
rem    6. every pulse *.json it finds -- retargeted BY PORT KEY, .bak kept
rem
rem  WHY IT INSERTS RATHER THAN APPENDS.  The RTL derives delay eligibility from
rem  POSITION: only the leading num_delay_ch = channel_count - buses*(width+1)
rem  channels get an event FIFO, and the slot->channel map is the identity.
rem  Append the lane at the end and that count still grows by one, so the new
rem  FIFO goes to a DAC bus bit while the new channel silently gets none.  This
rem  measures that run before and after, and refuses rather than break it.
rem
rem  Then it re-derives the host target from the manifest, the XDC and the top
rem  together, reloads every migrated pulse, and prints the resource estimate.
rem
rem  AFTERWARDS YOU STILL HAVE TO BUILD:  run bin\build_and_program.bat.  The
rem  host refuses to open a board whose LAYOUT_ID disagrees with the config, so
rem  until the bitstream is rebuilt and flashed the console will not connect.
rem
rem  Defaults to pgc_1D on P19.  For another pin:
rem     add_pulse_channel.bat <signal_name> <PACKAGE_PIN>
rem  Extra arguments are directories outside this checkout to migrate as well:
rem     add_pulse_channel.bat pgc_1D P19 D:\lab\my_workspace
rem ============================================================================

if /I "%~1"=="--inner" goto zlc_inner
rem shift does NOT rewrite %*, so the caller's arguments cross the re-entry in a
rem variable the inner pass inherits.
set "ZLC_ARGV=%*"
call "%~f0" --inner
set "ZLC_STATUS=%ERRORLEVEL%"
echo.
if "%ZLC_STATUS%"=="0" (
  echo ZLC add-channel: DONE.  Now run bin\build_and_program.bat to build and
  echo flash the bitstream -- the board cannot be opened until you do.
) else if "%ZLC_STATUS%"=="1" (
  echo ZLC add-channel: REFUSED -- nothing was written.  Read the reason above.
) else if "%ZLC_STATUS%"=="3" (
  echo ZLC add-channel: the board change is IN, but one or more PULSE FILES were
  echo NOT migrated -- they are named above with the reason, and the board will
  echo refuse them until they are.  Do not skip this.
) else (
  echo ZLC add-channel failed with code %ZLC_STATUS% -- read the messages above.
)
echo You can close this window, or press any key to exit.
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_STATUS%

:zlc_inner
for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
set "FPGA_DIR=%ZLC_HOME%\packages\zlc_pulse\fpga"

call "%FPGA_DIR%\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 exit /b 2

rem The work is Python -- editing JSON, XDC, Verilog and pulse documents from a
rem batch file would be its own bug farm.  It lives after the marker below and
rem is extracted to a temporary file, so this stays ONE file to pull.
set "ZLC_PY_SCRIPT=%TEMP%\zlc_add_channel_%RANDOM%%RANDOM%.py"
set "ZLC_PY_LINE="
for /f "delims=:" %%A in ('findstr /n /b /c:"### ZLC-EMBEDDED-PYTHON ###" "%~f0"') do set "ZLC_PY_LINE=%%A"
if not defined ZLC_PY_LINE (
  echo add_pulse_channel: cannot find its own embedded script -- is this file intact?
  exit /b 2
)
more +%ZLC_PY_LINE% "%~f0" > "%ZLC_PY_SCRIPT%"
if not exist "%ZLC_PY_SCRIPT%" (
  echo add_pulse_channel: could not write %ZLC_PY_SCRIPT%
  exit /b 2
)

echo Checkout: %ZLC_HOME%
echo.
%ZLC_PY_CMD% "%ZLC_PY_SCRIPT%" "%ZLC_HOME%" %ZLC_ARGV%
set "ZLC_RC=%ERRORLEVEL%"
del "%ZLC_PY_SCRIPT%" >nul 2>&1
exit /b %ZLC_RC%

### ZLC-EMBEDDED-PYTHON ###
"""Turn a spare GND pin into a digital pulse channel, and migrate the pulses.

A lane's facts live in four artefacts that are cross-checked rather than derived
from one another: the board manifest, the XDC, the RTL top, and the bitstream
(whose layout fingerprint the host verifies when it opens the board).  This
edits the three the host owns, regenerates the geometry header, and then
re-derives the target through the product's own validator, so a half-applied
change cannot be left behind.

The lane goes in AFTER THE LAST DIGITAL ONE, not at the end.  The RTL derives
delay eligibility from position -- "the board lays the real TTL outputs FIRST,
so the delay-eligible set is the contiguous leading NUM_DELAY_CH channels and
the slot->channel map is the identity" -- and NUM_DELAY_CH is computed as
channel_count - bus_count*(bus_width+1).  Appending would grow that count while
leaving the new channel outside the run it names, handing its event FIFO to a
DAC bus bit and leaving the new output with no delay at all, silently, with the
connect fingerprint reading green.  So the run is measured before and after.

A pulse file carries its own target, and each period's state vector is
POSITIONAL over that target's lanes, so both the extra lane and the renumbering
make every stored pulse unreadable to the new board.  The migration re-lays each
period BY PORT KEY, which is why it stays correct however the lanes move.
"""

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

CONFIG = "packages/zlc_pulse/fpga/board_config/streamer_config.json"
XDC = "packages/zlc_pulse/fpga/board_config/board.xdc"
TOP = "packages/zlc_pulse/fpga/pulse_streamer/zlc_pulse_streamer_top.v"
BENCH = "packages/zlc_pulse/fpga/pulse_streamer/sim/tb_t_ff.v"
HEADER = "packages/zlc_pulse/fpga/pulse_streamer/zlc_geometry.vh"
PULSE_FORMAT = "zlc.pulse"
LANE_LINE = '"logical_signal"'
OUT_FINAL = re.compile(r"\bout_final\[(?P<index>\d+)\]")
LANE_INDEX = re.compile(r'"index":\s*(?P<index>\d+)')


class Refused(Exception):
    """A precondition the operator has to resolve; nothing has been written."""


def say(message):
    print(message, flush=True)


def port_shape(port):
    """What a pulse's timing depends on, less the display label."""

    return (
        port.kind,
        len(port.lanes),
        port.bus_index,
        port.width,
        port.encoding,
        port.safe_value,
        port.latch_clock,
    )


def delay_run(lanes):
    """The leading run of digital lanes, and any digital lane outside it."""

    run = 0
    for lane in lanes:
        if lane["electrical_role"] != "digital":
            break
        run += 1
    outside = [
        lane["index"] for lane in lanes[run:] if lane["electrical_role"] == "digital"
    ]
    return run, outside


def expected_delay_channels(params, channel_count):
    """What the RTL will believe the delay-eligible run is."""

    return max(
        0,
        int(channel_count)
        - int(params["bus_count"]) * (int(params["bus_width"]) + 1),
    )


def edited_manifest(text, signal, pin):
    """The manifest with one digital lane inserted, as text, keeping its layout."""

    document = json.loads(text)
    lanes = document["board"]["lanes"]
    params = document["params"]
    count = int(params["channel_count"])
    if count != len(lanes):
        raise Refused(
            "the manifest is already inconsistent: params.channel_count is %d but "
            "board.lanes holds %d entries" % (count, len(lanes))
        )
    if [lane["index"] for lane in lanes] != list(range(count)):
        raise Refused("board.lanes indices are not exactly 0..%d" % (count - 1))
    for lane in lanes:
        if signal in (lane["logical_signal"], lane["rtl_port"]):
            raise Refused(
                "%r is already lane %d in the manifest -- has this already run?"
                % (signal, lane["index"])
            )
        if lane["package_pin"].upper() == pin.upper():
            raise Refused(
                "pin %s already drives %r (lane %d)"
                % (pin, lane["logical_signal"], lane["index"])
            )
    run, outside = delay_run(lanes)
    if outside:
        raise Refused(
            "this board already breaks the RTL's delay-eligibility invariant: "
            "digital lane(s) %s sit after the leading run of %d.  Fix the lane "
            "order first." % (", ".join(map(str, outside)), run)
        )
    if expected_delay_channels(params, count) != run:
        raise Refused(
            "num_delay_ch would be %d but the leading digital run is %d, so the "
            "RTL's identity delay map does not describe this board.  Fix that "
            "before adding a channel."
            % (expected_delay_channels(params, count), run)
        )
    new_index = run

    marker = '"channel_count": %d,' % count
    if text.count(marker) != 1:
        raise Refused(
            "cannot place params.channel_count in the manifest text (%d matches "
            "for %r)" % (text.count(marker), marker)
        )

    def renumber(match):
        value = int(match.group("index"))
        return '"index": %d' % (value + 1 if value >= new_index else value)

    lines = text.splitlines(keepends=True)
    first_moved = None
    for position, line in enumerate(lines):
        if LANE_LINE not in line:
            continue
        match = LANE_INDEX.search(line)
        if match is None:
            continue
        if int(match.group("index")) >= new_index:
            lines[position] = LANE_INDEX.sub(renumber, line, count=1)
            if first_moved is None:
                first_moved = position
    if first_moved is None:
        raise Refused("cannot find the lane the new one goes before")
    template = lines[first_moved]
    body = template.rstrip("\r\n")
    ending = template[len(body):] or "\n"
    indent = template[: len(template) - len(template.lstrip())]
    lines.insert(
        first_moved,
        '%s{"index": %d, "logical_signal": "%s", "rtl_port": "%s", '
        '"package_pin": "%s", "electrical_role": "digital"},%s'
        % (indent, new_index, signal, signal, pin, ending),
    )
    edited = "".join(lines).replace(marker, '"channel_count": %d,' % (count + 1), 1)

    after = json.loads(edited)
    fresh = after["board"]["lanes"]
    if [lane["index"] for lane in fresh] != list(range(count + 1)):
        raise Refused("the edited manifest's lane indices are not 0..%d" % count)
    if int(after["params"]["channel_count"]) != count + 1:
        raise Refused("the edited manifest did not take the new channel count")
    new_run, new_outside = delay_run(fresh)
    if new_outside or new_run != run + 1:
        raise Refused(
            "the edited manifest's leading digital run is %d with %s outside it, "
            "not %d with none" % (new_run, new_outside or "none", run + 1)
        )
    if expected_delay_channels(after["params"], count + 1) != new_run:
        raise Refused(
            "after the edit num_delay_ch would be %d but the digital run is %d"
            % (expected_delay_channels(after["params"], count + 1), new_run)
        )
    if fresh[new_index]["logical_signal"] != signal:
        raise Refused("the new lane did not land at index %d" % new_index)
    return edited, new_index


def edited_xdc(text, signal, pin):
    """The XDC with the pin's spare GND port renamed, and the port it replaced."""

    pattern = re.compile(
        r"(PACKAGE_PIN\s+" + re.escape(pin) + r"\b[^\n]*?\[\s*get_ports\s+)"
        r"(?P<port>[A-Za-z_][A-Za-z0-9_]*)(\s*\])",
        re.I,
    )
    match = pattern.search(text)
    if match is None:
        raise Refused(
            "pin %s is not constrained to a plain port in board.xdc -- add the "
            "constraint by hand first" % pin
        )
    placeholder = match.group("port")
    if re.fullmatch(r"GND\d*", placeholder, re.I) is None:
        raise Refused(
            "pin %s currently drives %r, which is not a spare GND output; pick a "
            "pin that is one, or free this one first" % (pin, placeholder)
        )
    return text[: match.start("port")] + signal + text[match.end("port"):], placeholder


def edited_top(text, signal, placeholder, new_index):
    """The top with the spare output driven, and the shifted da_clk assigns moved."""

    declaration = re.compile(r"\boutput\s+wire\s+" + re.escape(placeholder) + r"\b")
    if len(declaration.findall(text)) != 1:
        raise Refused(
            "expected exactly one 'output wire %s' in the RTL top" % placeholder
        )
    tie = re.compile(r"\s*assign\s+" + re.escape(placeholder) + r"\s*=\s*1'b0\s*;")
    if len(tie.findall(text)) != 1:
        raise Refused(
            "expected exactly one tie-low 'assign %s = 1'b0;' in the RTL top"
            % placeholder
        )
    edited = declaration.sub("output wire " + signal, text, count=1)
    edited = tie.sub("", edited, count=1)

    # Every out_final index at or after the insertion moved up by one.  One
    # pass, so no two lanes can collide on the way.  bus_out_final is a
    # different word and is deliberately not matched: a DAC data bit is
    # addressed by bus and bit, not by lane index, so it does not move.
    def shift(match):
        value = int(match.group("index"))
        return "out_final[%d]" % (value + 1 if value >= new_index else value)

    edited = OUT_FINAL.sub(shift, edited)

    lines = edited.splitlines(keepends=True)
    last = None
    for position, line in enumerate(lines):
        if "assign" in line and OUT_FINAL.search(line):
            last = position
    if last is None:
        raise Refused("cannot find the channel assign block in the RTL top")
    body = lines[last].rstrip("\r\n")
    ending = lines[last][len(body):] or "\n"
    indent = lines[last][: len(lines[last]) - len(lines[last].lstrip())]
    lines[last] = body + ending + "%sassign %s = out_final[%d];%s" % (
        indent,
        signal,
        new_index,
        ending,
    )
    edited = "".join(lines)
    if re.search(r"\b" + re.escape(placeholder) + r"\b", edited):
        raise Refused(
            "%r still appears in the RTL top after the edit -- it is named "
            "somewhere this cannot rewrite" % placeholder
        )
    return edited


def edited_bench(text, signal, placeholder):
    """The top's testbench, which names the port and must follow it."""

    pattern = re.compile(r"\b" + re.escape(placeholder) + r"\b")
    if not pattern.search(text):
        return None
    return pattern.sub(signal, text)


def retargeted(document, target, codec):
    """One pulse document re-laid onto ``target``, by port key."""

    sequence_tree, editor = codec["split"](document)
    old = codec["from_tree"](sequence_tree)
    if old.target == target:
        return None
    missing = [port.key for port in old.target.ports if port.key not in target.by_key]
    if missing:
        raise Refused(
            "drives port(s) the new board does not have: " + ", ".join(missing)
        )
    changed = [
        port.key
        for port in old.target.ports
        if port_shape(target.by_key[port.key]) != port_shape(port)
    ]
    if changed:
        raise Refused("port(s) changed shape, not just position: " + ", ".join(changed))
    lane_of = {lane: index for index, lane in enumerate(target.raw_lanes)}
    was_lane_of = {lane: index for index, lane in enumerate(old.target.raw_lanes)}
    periods = []
    for period in old.periods:
        states = [0] * len(target.raw_lanes)
        for port in old.target.ports:
            if port.kind != "digital":
                continue
            level = period.states[was_lane_of[port.lanes[0]]]
            states[lane_of[target.by_key[port.key].lanes[0]]] = int(level)
        periods.append(replace(period, states=tuple(states)))
    migrated = replace(old, target=target, periods=tuple(periods))
    tree = dict(codec["to_tree"](migrated))
    if editor:
        tree["editor"] = dict(editor)

    # Prove it before it is written: every port's level in every period is what
    # it was, and any channel the pulse never knew sits safe.
    check = codec["from_tree"](codec["split"](tree)[0])
    if check.target != target:
        raise Refused("the migrated target does not equal the board's")
    if len(check.periods) != len(old.periods):
        raise Refused("the migration changed the number of periods")
    for before, after in zip(old.periods, check.periods):
        if (before.period_id, before.duration, before.unit) != (
            after.period_id,
            after.duration,
            after.unit,
        ):
            raise Refused("the migration changed period %r" % before.period_id)
        for port in old.target.ports:
            if port.kind != "digital":
                continue
            was = int(before.states[was_lane_of[port.lanes[0]]])
            now = int(after.states[lane_of[target.by_key[port.key].lanes[0]]])
            if was != now:
                raise Refused(
                    "the migration changed %s in period %r"
                    % (port.key, before.period_id)
                )
        unsafe = [
            port.key
            for port in target.ports
            if port.kind == "digital"
            and port.key not in old.target.by_key
            and int(after.states[lane_of[port.lanes[0]]]) != 0
        ]
        if unsafe:
            raise Refused(
                "a new channel is not safe in period %r: %s"
                % (before.period_id, ", ".join(unsafe))
            )
    return tree


def pulse_documents(roots, codec):
    """Every pulse document under these roots."""

    found = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            resolved = path.resolve()
            if resolved in seen or ".git" in resolved.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if PULSE_FORMAT not in text:
                continue
            try:
                document = codec["parse"](text)
            except Exception:
                continue
            if getattr(document, "get", lambda _key: None)("format") != PULSE_FORMAT:
                continue
            seen.add(resolved)
            found.append((path, document))
    return found


def main(argv):
    if not argv:
        say("  usage: add_pulse_channel <checkout> [signal] [pin] [extra dirs...]")
        return 2
    root = Path(argv[0]).resolve()
    signal = (argv[1] if len(argv) > 1 else "").strip() or "pgc_1D"
    pin = (argv[2] if len(argv) > 2 else "").strip() or "P19"
    extra = [Path(item) for item in argv[3:] if item.strip()]
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", signal) is None:
        say("  REFUSED: %r is not a Verilog identifier" % signal)
        return 1
    if re.fullmatch(r"[A-Za-z]+[0-9]+", pin) is None:
        say("  REFUSED: %r does not look like a package pin" % pin)
        return 1

    sys.path.insert(0, str(root))
    try:
        import zou_lab_control  # noqa: F401 -- binds this checkout's packages
        from zlc_durable import write_readable_json
        from zlc_pulse import (
            pulse_target_from_xdc,
            sequence_from_tree,
            sequence_to_tree,
        )
        from zlc_pulse.codec import parse_pulse_tree_json, split_pulse_document_tree
    except Exception as error:
        say("  cannot import this checkout: %s: %s" % (type(error).__name__, error))
        return 2
    codec = {
        "parse": parse_pulse_tree_json,
        "split": split_pulse_document_tree,
        "from_tree": sequence_from_tree,
        "to_tree": sequence_to_tree,
    }

    paths = {
        "manifest": root / CONFIG,
        "xdc": root / XDC,
        "top": root / TOP,
        "bench": root / BENCH,
        "header": root / HEADER,
    }
    for path in paths.values():
        if not path.is_file():
            say("  REFUSED: %s is missing -- is %s a ZLC checkout?" % (path, root))
            return 1

    manifest_text = paths["manifest"].read_text(encoding="utf-8")
    standing = next(
        (
            lane
            for lane in json.loads(manifest_text)["board"]["lanes"]
            if lane["logical_signal"] == signal
        ),
        None,
    )
    # Re-runnable on purpose: once the board change is in, this becomes the
    # migration tool, so more pulse directories can be swept without undoing
    # anything.  A pulse file the operator keeps outside the checkout is the
    # ordinary case for that, and an unmigrated pulse is invisible in the
    # editor -- it lists the OPENED pulse's ports, not the board's.
    already = standing is not None and standing["package_pin"].upper() == pin.upper()
    if already:
        say(
            "%s is already lane %d on %s -- the board change is in, so this run "
            "only migrates pulse files." % (signal, standing["index"], pin)
        )
        edits = []
        new_index = int(standing["index"])
    else:
        say("Adding %s on pin %s." % (signal, pin))
        try:
            xdc_text = paths["xdc"].read_text(encoding="utf-8")
            top_text = paths["top"].read_text(encoding="utf-8")
            bench_text = paths["bench"].read_text(encoding="utf-8")
            new_manifest, new_index = edited_manifest(manifest_text, signal, pin)
            new_xdc, placeholder = edited_xdc(xdc_text, signal, pin)
            new_top = edited_top(top_text, signal, placeholder, new_index)
            new_bench = edited_bench(bench_text, signal, placeholder)
        except Refused as error:
            say("  REFUSED: %s" % error)
            return 1
        say(
            "  lane index %d (after the last digital one), replacing the spare "
            "output %s" % (new_index, placeholder)
        )
        # Nothing is written until every edit above has been derived.
        edits = [
            (paths["manifest"], new_manifest),
            (paths["xdc"], new_xdc),
            (paths["top"], new_top),
        ]
        if new_bench is not None:
            edits.append((paths["bench"], new_bench))
    try:
        for path, text in edits:
            backup = path.with_name(path.name + ".bak")
            if not backup.exists():
                backup.write_bytes(path.read_bytes())
            path.write_text(text, encoding="utf-8", newline="")
            say("  wrote %s (kept %s)" % (path.name, backup.name))

        if edits:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "zou_lab_control",
                    "fpga",
                    "--config",
                    str(paths["manifest"]),
                    "--emit-geometry-vh",
                    str(paths["header"]),
                ],
                cwd=str(root),
                capture_output=True,
                text=True,
            )
            if result.returncode not in (0, 1):
                say("  regenerating the geometry header failed:")
                say(result.stdout.strip() or result.stderr.strip())
                return 2
            say("  regenerated %s" % paths["header"].name)

        # The product's own three-way reconciliation.  A manifest, an XDC and a
        # top that disagree surface HERE, before any pulse file is touched.
        target = pulse_target_from_xdc()
        if signal not in target.by_key:
            say("  the rebuilt board target has no port %r" % signal)
            return 2
        driven = target.by_key[signal]
        if tuple(driven.lanes) != ("ch%02d" % new_index,):
            say(
                "  %r came back on %s, not on lane %d"
                % (signal, tuple(driven.lanes), new_index)
            )
            return 2
        say(
            "  board target rebuilt: %d lanes, %d ports, %s owns ch%02d"
            % (len(target.raw_lanes), len(target.ports), signal, new_index)
        )
    except Exception as error:
        say("  %s: %s" % (type(error).__name__, error))
        say("  the .bak files beside the edited files hold the originals.")
        return 2

    roots = [root] + extra
    documents = pulse_documents(roots, codec)
    if not documents:
        say("  no pulse files found under: " + ", ".join(str(item) for item in roots))
    planned = []
    problems = []
    already = 0
    for path, document in documents:
        try:
            tree = retargeted(document, target, codec)
        except Refused as error:
            problems.append((path, str(error)))
            continue
        except Exception as error:
            problems.append((path, "%s: %s" % (type(error).__name__, error)))
            continue
        if tree is None:
            already += 1
            continue
        planned.append((path, tree))
    # Every migration is computed before any of them is written, so one file
    # this version cannot read cannot leave the rest half moved.
    for path, tree in planned:
        backup = path.with_name(path.name + ".bak")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        write_readable_json(path, tree)
        say(
            "  migrated %s -> %d lanes (kept %s)"
            % (path.name, len(tree["target"]["raw_lanes"]), backup.name)
        )
    say(
        "  %d pulse file(s) migrated, %d already on this board, %d left alone"
        % (len(planned), already, len(problems))
    )
    for path, reason in problems:
        say("  LEFT ALONE %s" % path)
        say("             %s" % reason)
    if problems:
        say("")
        say("  Those files were NOT touched and the board will still refuse them.")
        say("  One this version cannot read at all predates the current pulse")
        say("  schema (an old 'repeat' field, say): re-copy it from")
        say("  packages/zlc_atom/src/zlc_atom/nodes/ if it is a stale copy of a")
        say("  shipped template, or open and re-save it in the pulse editor.")

    estimate = subprocess.run(
        [
            sys.executable,
            "-m",
            "zou_lab_control",
            "fpga",
            "--config",
            str(paths["manifest"]),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    say("")
    for line in (estimate.stdout or "").splitlines():
        stripped = line.strip()
        if stripped and (
            "%use" in stripped
            or "OK" in stripped
            or "OVER" in stripped
            or "RESULT" in stripped
            or stripped.startswith("geometry:")
        ):
            say("  " + stripped)
    if estimate.returncode == 1:
        say("  the part is OVER BUDGET for this geometry -- see the lines above.")
        return 1
    # A pulse file left behind is a failure of the job the operator asked for,
    # not a footnote: the board is changed and that file can no longer fire.
    return 3 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
