"""Pack one pulse document with THIS checkout's host code into a bench image file.

usage: python pack_diff.py <checkout root> <pulse.json> <image.txt> <meta.json>
Run with the checkout's own PYTHONPATH.  The image is one "addr hexdata" line per
sparse word of pack_program's output; the bench overrides RUN_REPEAT_COUNT itself.
"""
from __future__ import annotations

import json
import pathlib
import sys


def main() -> None:
    root, pulse_json, image_txt, meta_json = sys.argv[1:5]
    sys.path.insert(0, root)
    import zou_lab_control  # noqa: F401  product bootstrap
    from zlc_pulse import resolve_api_parameters
    from zlc_pulse.codec import sequence_from_tree
    from zlc_pulse.compile import compile_sequence
    from zlc_pulse.wire import FROZEN_CLOCK_HZ, StreamerParams, pack_program

    from zlc_pulse.scan import prepare_scan_application
    from zlc_pulse.wire import CtrlWords, pack_scan_rows

    pulse_path = pathlib.Path(pulse_json)
    tree = json.loads(pulse_path.read_text(encoding="utf-8"))
    sequence = resolve_api_parameters(sequence_from_tree(tree))
    params = StreamerParams()
    # A sibling "<variant>.scan.json" holds the authored table (editor units,
    # one row per point, one column per scan slot); each side quantizes and
    # encodes it with its own rules and its own compile of the slots.
    stem = pulse_path.name.split(".")[0]
    scan_path = pulse_path.with_name(f"{stem}.scan.json")
    scan_meta = {}
    if scan_path.is_file():
        rows = json.loads(scan_path.read_text(encoding="utf-8"))
        prepared = prepare_scan_application(sequence, rows, params=params)
        if len(prepared) == 3:
            effective, scales, wire = prepared
            program = compile_sequence(sequence, params, FROZEN_CLOCK_HZ, slot_tick_scales=scales)
            scan_meta["slot_tick_scales"] = [int(s) for s in scales]
        else:
            effective, wire = prepared
            program = compile_sequence(sequence, params, FROZEN_CLOCK_HZ)
        words = pack_program(program, params, target=sequence.target)
        words.update(pack_scan_rows(wire, params, 0, 0))
        words[CtrlWords.SCAN_COUNT] = len(wire)
        words[CtrlWords.SCAN_ENABLE] = 1
        words[CtrlWords.SCAN_REPEAT_COUNT] = 1
        scan_meta.update({
            "scan_rows": len(wire),
            "effective_rows": [[float(v) for v in row] for row in effective],
            "wire_rows": [[int(v) for v in row] for row in wire],
            "bank_ready": (1 if len(wire) > 0 else 0) | (2 if len(wire) > params.bank_size else 0),
        })
    else:
        program = compile_sequence(sequence, params, FROZEN_CLOCK_HZ)
        words = pack_program(program, params, target=sequence.target)
    lines = [f"{address} {value & 0xFFFFFFFF:08x}" for address, value in sorted(words.items())]
    pathlib.Path(image_txt).write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    meta = {
        "words": len(words),
        "duration_seconds": float(program.duration_seconds),
        "digest": str(program.digest),
        "run_repeats": int(tree["run_repeats"]),
        "fingerprint": hex(int(program.geometry_fingerprint)),
        "bank_ready": 3,
        **scan_meta,
    }
    pathlib.Path(meta_json).write_text(json.dumps(meta, indent=1), encoding="ascii", newline="\n")
    print("packed", pulse_json, "->", len(words), "words,", meta["duration_seconds"] * 1e9, "ns per run")


if __name__ == "__main__":
    main()
