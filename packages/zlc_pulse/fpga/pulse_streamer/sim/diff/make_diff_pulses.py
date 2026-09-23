"""Author the differential-simulation pulses with the NEW model and write both file formats.

Each variant is short (hundreds to thousands of ticks) so a full-top xsim run
takes seconds, and together they cover: one-tick rows next to long rows, DAC
edges and Bresenham ramps on several buses, a whole-pulse bracket with Run
repeats, a partial bracket, TTL and DAC output delays across the run seam,
and a burst of one-tick rows.  Every variant stays within what the OLD engine
could express (at most one bracket), which is what makes the comparison fair.

usage: python make_diff_pulses.py <new checkout root> <out dir>
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import sys


def main() -> None:
    root, out_dir = sys.argv[1:3]
    sys.path.insert(0, root)
    import zou_lab_control  # noqa: F401
    from zlc_pulse.codec import sequence_from_tree, sequence_to_tree
    from zlc_pulse.model import AnalogStep, OutputDelay, PulseBracket, PulsePeriod

    fixture = pathlib.Path(root, "packages/zlc_atom/tests/pulses/imaging_template.json")
    base = sequence_from_tree(json.loads(fixture.read_text(encoding="utf-8")))
    states = {period.period_id: period.states for period in base.periods}
    S = states  # load, long_before, gap_0, short, gap_1, long_after

    def period(pid: str, ticks: int, pattern: str, *steps) -> PulsePeriod:
        return PulsePeriod(pid, ticks * 20, "ns", S[pattern], tuple(steps))

    plain = (
        period("a", 1, "load"),
        period("b", 3, "long_before"),
        period("c", 100, "gap_0"),
        period("d", 7, "short"),
        period("e", 1, "gap_1"),
        period("f", 250, "long_after"),
        period("g", 50, "load"),
    )
    dac = (
        period("a", 1, "load"),
        period("b", 3, "long_before"),
        period("c", 100, "gap_0", AnalogStep("da_bias_y", "edge", 300)),
        period("d", 7, "short", AnalogStep("da_bias_y", "ramp", -200)),
        period("e", 1, "gap_1"),
        period("f", 250, "long_after", AnalogStep("da_dipole", "edge", 100), AnalogStep("da_bias_x", "ramp", 400)),
        period("g", 50, "load", AnalogStep("da_bias_x", "edge", -100)),
    )
    burst = (
        period("h1", 1, "load"),
        period("h2", 1, "long_before"),
        period("h3", 1, "gap_0", AnalogStep("da_bias_y", "edge", 50)),
        period("h4", 2, "short", AnalogStep("da_bias_y", "ramp", -50)),
        period("h5", 1, "gap_1"),
        period("h6", 3, "long_after"),
    )
    delays = (
        OutputDelay("cooling", 60, "ns"),
        OutputDelay("probe", 20, "ns"),
        OutputDelay("da_bias_y", 40, "ns"),
    )

    def sequence(name, periods, *, brackets=(), delays=(), run_repeats=1):
        return dataclasses.replace(
            base, name=name, periods=periods, bindings=(), delays=delays,
            brackets=brackets, run_repeats=run_repeats,
        )

    variants = {
        "v1_plain": sequence("v1_plain", plain),
        "v2_dac": sequence("v2_dac", dac),
        "v3_bracket_whole": sequence("v3_bracket_whole", dac, brackets=(PulseBracket("b1", "a", "g", 3),), run_repeats=2),
        "v4_bracket_partial": sequence("v4_bracket_partial", dac, brackets=(PulseBracket("b1", "b", "d", 2),)),
        "v5_delays": sequence("v5_delays", dac, delays=delays, run_repeats=2),
        "v6_bracket_delays": sequence("v6_bracket_delays", dac, delays=delays, brackets=(PulseBracket("b1", "b", "d", 2),), run_repeats=2),
        "v7_one_tick_burst": sequence("v7_one_tick_burst", burst, run_repeats=3),
        "v8_burst_bracket": sequence("v8_burst_bracket", burst, brackets=(PulseBracket("b1", "h2", "h4", 3),), run_repeats=2),
        # the loop-start row itself carries a DAC edge (c) or a ramp (d)
        "v9_bracket_edge_start": sequence("v9_bracket_edge_start", dac, brackets=(PulseBracket("b1", "c", "d", 3),), run_repeats=2),
        "v10_bracket_ramp_start": sequence("v10_bracket_ramp_start", dac, brackets=(PulseBracket("b1", "d", "f", 2),), run_repeats=2),
        # a ramp ending at the rewind, with the loop-start row a one-tick row
        "v11_bracket_to_ramp_end": sequence("v11_bracket_to_ramp_end", dac, brackets=(PulseBracket("b1", "a", "d", 2),), run_repeats=1),
    }
    # scan slots: period c's duration and period f's da_dipole edge come from the table
    from zlc_pulse.model import PulseBinding, PulseFieldRef
    scan_bindings = (
        PulseBinding(PulseFieldRef("duration", "c"), "ns", scan=True),
        PulseBinding(PulseFieldRef("dac", "f", "da_dipole"), "value", scan=True),
    )
    scan_rows = [[2000, 100], [600, -300], [100, 250]]
    variants["v12_scan"] = dataclasses.replace(sequence("v12_scan", dac), bindings=scan_bindings)
    variants["v13_scan_bracket"] = dataclasses.replace(
        sequence("v13_scan_bracket", dac, brackets=(PulseBracket("b1", "b", "f", 2),), run_repeats=2), bindings=scan_bindings,
    )
    # the same scan with a bracket that does not end on a ramp row, and with delays
    variants["v14_scan_bracket_clean"] = dataclasses.replace(
        sequence("v14_scan_bracket_clean", dac, brackets=(PulseBracket("b1", "b", "e", 2),), run_repeats=2), bindings=scan_bindings,
    )
    variants["v15_scan_delays"] = dataclasses.replace(
        sequence("v15_scan_delays", dac, delays=delays, run_repeats=2), bindings=scan_bindings,
    )
    scan_json = json.dumps(scan_rows)

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, seq in variants.items():
        tree = sequence_to_tree(seq)
        sequence_from_tree(tree)  # the new codec accepts it
        (out / f"{name}.new.json").write_text(json.dumps(tree, indent=1), encoding="utf-8", newline="\n")
        periods = [{k: v for k, v in p.items() if k != "kind"} for p in tree["periods"]]
        brackets = tree["brackets"]
        bracket = None if not brackets else {k: v for k, v in brackets[0].items() if k != "bracket_id"}
        old = {
            "format": tree["format"], "name": tree["name"], "time_step_ns": tree["time_step_ns"],
            "target": tree["target"], "periods": periods, "bindings": tree["bindings"],
            "delays": tree["delays"], "bracket": bracket, "run_repeats": tree["run_repeats"],
        }
        (out / f"{name}.old.json").write_text(json.dumps(old, indent=1), encoding="utf-8", newline="\n")
        if name.startswith(("v12_", "v13_", "v14_", "v15_")):
            (out / f"{name}.scan.json").write_text(scan_json + "\n", encoding="utf-8", newline="\n")
        ticks = sum(int(round(p["duration"] / 20)) for p in tree["periods"])
        print(f"{name}: {len(tree['periods'])} periods, {ticks} ticks/pass, brackets={brackets}, delays={tree['delays']}, run={tree['run_repeats']}")


if __name__ == "__main__":
    main()
