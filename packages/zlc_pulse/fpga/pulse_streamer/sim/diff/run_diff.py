"""Differential full-top simulation of two streamer builds on the same authored pulses.

Each side is a checkout that has been BUILT (its generated BRAM simulation
models under ``fpga/build/ps/ps.srcs/sources_1/ip`` are what the real top
instantiates).  For every pulse variant, each side's own host code packs the
image, its own RTL plays it in xsim through the real ``zlc_pulse_streamer_top``,
and the per-clock pin dumps (25 TTL, 4x10 DAC data, 4 DAC clocks) are compared
tick for tick.  Both sides read the same pulse in their own file format.

usage:
  python run_diff.py --old <edge-table checkout> --new <period-table checkout> [--out DIR] [variant ...]

The variants come from make_diff_pulses.py, authored with the NEW checkout's
model; the old checkout must still be able to express them (one bracket).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
VIVADO = pathlib.Path(os.environ.get("ZLC_PS_VIVADO_BIN", r"C:/Xilinx/Vivado/2019.1/bin"))

ENGINES = {
    # engine source and the generated IP simulation models its top instantiates
    "zlc_edge_streamer.v": (
        "blk_mem_gen_edge_tick/simulation/blk_mem_gen_v8_4.v",
        "blk_mem_gen_edge_tick/sim/blk_mem_gen_edge_tick.v",
        "blk_mem_gen_edge_coeff/sim/blk_mem_gen_edge_coeff.v",
        "blk_mem_gen_edge_mask/sim/blk_mem_gen_edge_mask.v",
        "blk_mem_gen_busimg/sim/blk_mem_gen_busimg.v",
        "blk_mem_gen_scan/sim/blk_mem_gen_scan.v",
    ),
    "zlc_period_streamer.v": (
        "blk_mem_gen_rows/simulation/blk_mem_gen_v8_4.v",
        "blk_mem_gen_rows/sim/blk_mem_gen_rows.v",
        "blk_mem_gen_scan/sim/blk_mem_gen_scan.v",
    ),
}


def side_of(root: pathlib.Path) -> dict:
    rtl = root / "packages/zlc_pulse/fpga/pulse_streamer"
    engine = next((name for name in ENGINES if (rtl / name).is_file()), None)
    if engine is None:
        raise SystemExit(f"{root}: no known streamer engine under {rtl}")
    return {"root": root, "rtl": rtl, "engine": engine, "ips": ENGINES[engine],
            "ip": root / "packages/zlc_pulse/fpga/build/ps/ps.srcs/sources_1/ip"}


def run(args, cwd, log: pathlib.Path, env=None) -> subprocess.CompletedProcess:
    started = time.perf_counter()
    completed = subprocess.run([str(a) for a in args], cwd=str(cwd), capture_output=True, text=True, env=env)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8", errors="replace")
    print(f"    {pathlib.Path(str(args[0])).name}: rc={completed.returncode} {time.perf_counter() - started:.0f}s")
    return completed


def python_env(root: pathlib.Path) -> dict:
    env = dict(os.environ)
    parts = [str(p).replace("\\", "/") for p in sorted(root.glob("packages/*/src"))] + [str(root).replace("\\", "/")]
    env["PYTHONPATH"] = ";".join(parts)
    return env


def compile_side(name: str, side: dict, out: pathlib.Path) -> pathlib.Path:
    work = out / f"work_{name}"
    work.mkdir(parents=True, exist_ok=True)
    rtl = side["rtl"]
    sources = [rtl / "zlc_uart_bridge.v", rtl / side["engine"], rtl / "zlc_pulse_streamer_top.v"]
    sources += [side["ip"] / item for item in side["ips"]]
    sources.append(HERE / "tb_diff.v")
    for source in sources:
        if not source.is_file():
            raise SystemExit(f"[{name}] missing {source} (has this checkout been built?)")
    print(f"[{name}] compiling {side['engine']} + {len(side['ips'])} IP models from {side['root']}")
    if run([VIVADO / "xvlog.bat", "-sv", "-i", rtl, *sources], work, work / "xvlog.log").returncode != 0:
        raise SystemExit(f"[{name}] xvlog failed, see {work / 'xvlog.log'}")
    if run([VIVADO / "xelab.bat", "work.tb_diff", "-s", "sdiff"], work, work / "xelab.log").returncode != 0:
        raise SystemExit(f"[{name}] xelab failed, see {work / 'xelab.log'}")
    return work


def pack(name: str, side: dict, variant: str, out: pathlib.Path) -> tuple[pathlib.Path, dict]:
    pulse = out / "pulses" / f"{variant}.{name}.json"
    image = out / f"work_{name}" / f"{variant}.image.txt"
    meta = out / f"work_{name}" / f"{variant}.meta.json"
    completed = run([sys.executable, HERE / "pack_diff.py", side["root"], pulse, image, meta],
                    side["root"], out / f"work_{name}" / f"{variant}.pack.log", env=python_env(side["root"]))
    if completed.returncode != 0:
        raise SystemExit(f"[{name}] packing {variant} failed:\n{completed.stdout}{completed.stderr}")
    return image, json.loads(meta.read_text(encoding="ascii"))


def simulate(name: str, work: pathlib.Path, variant: str, image: pathlib.Path, runs: int, bank_ready: int) -> pathlib.Path:
    dump = work / f"{variant}.dump.txt"
    if dump.exists():
        dump.unlink()
    # xsim.bat splits arguments at '=', so the bench reads its parameters
    # from a file in the working directory instead of plusargs.
    (work / "diff_args.txt").write_text(
        f"{image.as_posix()}\n{runs}\n{dump.as_posix()}\n{bank_ready}\n", encoding="ascii", newline="\n")
    completed = run([VIVADO / "xsim.bat", "sdiff", "-runall"], work, work / f"{variant}.xsim.log")
    text = (work / f"{variant}.xsim.log").read_text(encoding="utf-8", errors="replace")
    if "DIFF-DUMP-DONE" not in text or completed.returncode != 0:
        raise SystemExit(f"[{name}] xsim {variant} did not finish cleanly:\n" + "\n".join(text.splitlines()[-25:]))
    return dump


def load_dump(path: pathlib.Path) -> list[tuple[int, int, int]]:
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        ttl, bus, clk = line.split()
        rows.append((int(ttl, 16), int(bus, 16), int(clk, 16)))
    return rows


def describe(row: tuple[int, int, int]) -> str:
    ttl, bus, clk = row
    buses = [(bus >> (10 * i)) & 0x3FF for i in range(4)]  # dipole, bias_y, bias_x, bias_z
    return f"ttl={ttl:07x} dac(dipole,bias_y,bias_x,bias_z)={buses} clk={clk:04b}"


def compare(variant: str, old: list, new: list) -> dict:
    def first_activity(rows):
        return next((index for index, (ttl, _bus, _clk) in enumerate(rows) if ttl != 0), None)

    a_old, a_new = first_activity(old), first_activity(new)
    result = {"variant": variant, "old_ticks": len(old), "new_ticks": len(new),
              "old_lead": a_old, "new_lead": a_new, "mismatches": 0, "first_mismatch": None,
              "ttl_and_clocks_identical": None, "dac_buses_differing": []}
    if a_old is None or a_new is None:
        result["first_mismatch"] = "no TTL activity on one side"
        return result
    o, n = old[a_old:], new[a_new:]
    common = min(len(o), len(n))
    result["compared_ticks"] = common
    result["length_after_activity"] = (len(o), len(n))
    ttl_same = True
    buses: set[str] = set()
    for index in range(common):
        if o[index] != n[index]:
            result["mismatches"] += 1
            if result["first_mismatch"] is None:
                result["first_mismatch"] = (index, describe(o[index]), describe(n[index]))
            if o[index][0] != n[index][0] or o[index][2] != n[index][2]:
                ttl_same = False
            changed = o[index][1] ^ n[index][1]
            for i, bus in enumerate(("dipole", "bias_y", "bias_x", "bias_z")):
                if (changed >> (10 * i)) & 0x3FF:
                    buses.add(bus)
    result["ttl_and_clocks_identical"] = ttl_same
    result["dac_buses_differing"] = sorted(buses)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", required=True, type=pathlib.Path)
    parser.add_argument("--new", required=True, type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "out")
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()
    sides = {"old": side_of(args.old.resolve()), "new": side_of(args.new.resolve())}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    authored = run([sys.executable, HERE / "make_diff_pulses.py", sides["new"]["root"], out / "pulses"],
                   sides["new"]["root"], out / "make_diff_pulses.log", env=python_env(sides["new"]["root"]))
    if authored.returncode != 0:
        raise SystemExit(f"authoring the pulses failed:\n{authored.stdout}{authored.stderr}")
    variants = sorted({p.name.split(".")[0] for p in (out / "pulses").glob("*.new.json")},
                      key=lambda v: int(v.split("_")[0][1:]))
    if args.variants:
        variants = [v for v in variants if v in args.variants]
    works = {name: compile_side(name, side, out) for name, side in sides.items()}
    report = []
    for variant in variants:
        print(f"== {variant}")
        dumps = {}
        for name, side in sides.items():
            image, meta = pack(name, side, variant, out)
            dumps[name] = load_dump(simulate(name, works[name], variant, image, meta["run_repeats"], meta.get("bank_ready", 3)))
            print(f"    [{name}] {meta['words']} words, {meta['duration_seconds'] * 1e9:.0f} ns/run, dump {len(dumps[name])} clocks"
                  + (f", scan wire rows {meta['wire_rows']}" if "wire_rows" in meta else ""))
        result = compare(variant, dumps["old"], dumps["new"])
        report.append(result)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\n==== SUMMARY ====")
    for result in report:
        same_length = result.get("length_after_activity", (0, 1))[0] == result.get("length_after_activity", (0, 1))[1]
        verdict = "IDENTICAL" if result["mismatches"] == 0 and same_length else "DIFFERS"
        print(f"{result['variant']:>24}: {verdict}  clocks old/new={result['old_ticks']}/{result['new_ticks']}"
              f" mismatches={result['mismatches']} ttl+clk identical={result['ttl_and_clocks_identical']}"
              f" buses={result['dac_buses_differing']} first={result['first_mismatch']}")


if __name__ == "__main__":
    main()
