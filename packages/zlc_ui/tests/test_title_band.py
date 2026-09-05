"""The panel title strip: a shape in colour, and a repeat axis's landed count."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from zlc_ui.console.panel_card_view import data_structure_fragments


STRUCTURE = (
    (("repeat", 35), ("run", 100)),
    (("field.x", 10),),
    (("site", 35),),
)


def _line(fragments) -> str:
    return "".join(text for text, _colour, _elide in fragments)


def test_a_repeat_axis_reads_its_landed_count_where_the_others_read_their_size():
    sizes, names = data_structure_fragments(STRUCTURE, {"repeat": 34, "run": 37})
    assert _line(sizes) == "(34\u2713 \u00d7 37\u2713) \u00d7 (10) \u00d7 (35)"
    assert _line(names) == "(repeat \u00d7 run) \u00d7 (field.x) \u00d7 (site)"
    # The count wears the repeat domain's colour, like the size it replaces.
    plain_sizes, _names = data_structure_fragments(STRUCTURE)
    assert _line(plain_sizes) == "(35 \u00d7 100) \u00d7 (10) \u00d7 (35)"
    assert [colour for _t, colour, _e in sizes] == [colour for _t, colour, _e in plain_sizes]


def test_no_landed_count_means_the_size_is_shown():
    sizes, _names = data_structure_fragments(STRUCTURE, {})
    assert _line(sizes) == "(35 \u00d7 100) \u00d7 (10) \u00d7 (35)"
