"""Qt-chrome design tokens owned by :mod:`zlc_ui.fluent`.

This module defines the visual vocabulary used by reusable Qt controls and
custom host surfaces.
"""

from __future__ import annotations

#: The accent at full strength, kept for SMALL INDICATORS THAT CARRY NO
#: TEXT: the track of an on switch, the fill of a ticked box, the underline
#: of the current tab, the edge of a field taking input, a scroll handle
#: under the pointer.  Nothing with words on it is filled with it -- a chosen
#: item is SELECTED -- and it never outlines a selection.
ACCENT = "#77AADD"
#: ACCENT a step deeper: an on switch or a ticked box under the pointer.
ACCENT_HOVER = "#6DA2D5"
#: The hover wash, EVERYWHERE the pointer can act: a list or table row, a
#: popup item, a stepper cell, an empty grid card, a segmented choice, a
#: clickable row label.  Also the face of a grid card a rule reached rather
#: than a click, and of a Period card that is selected.
ACCENT_TINT = "#E4EFF9"
#: What is chosen, and what is being pressed: a chosen grid card, the
#: current segmented choice, the picked row of a popup list, a table, a tree
#: or the Add panel list, an active cycle combo -- and the face of a
#: standard command while it is held down, because a press IS choosing it.
#: The mid-point of ACCENT_TINT and ACCENT (TEXT ink 7.6:1), CIEDE2000 12.3
#: from the hover wash and 8.8 from a resting BUTTON_TINT command, so hover,
#: rest and chosen are three different things.  A selection is this fill
#: and nothing else: no outline, no second cue, rounded like everything it
#: sits among.
SELECTED = "#AACAEA"
#: A chosen item under the pointer: SELECTED a quarter of the way to ACCENT
#: (TEXT ink 7.0:1).
SELECTED_HOVER = "#9DC2E7"
#: The dashed edge of a grid card a rule reached rather than a click: the
#: accent halfway to the divider.  Dashed, because the edge says something
#: -- "derived, not marked" -- beside the ACCENT_TINT wash and the DERIVED_INK
#: name; no solid edge in this family outlines a selection.
SELECTION_EDGE = "#ACC4DD"
#: The face of a standard command: the accent a step past ACCENT_TINT, far
#: enough from a white field (CIEDE2000 10.3) that a button beside an edit
#: is a different object, not a second field.  ACCENT_TINT is the hover wash
#: and sat 1.17:1 on a white card -- a button there read as a selected cell,
#: not a thing to press.  Dark-blue (HOVER) ink on it reads 7.7:1; under the
#: pointer it steps a quarter of the way to ACCENT (#BDD6EF), and held down
#: it is SELECTED (5.8:1).
BUTTON_TINT = "#D4E5F5"
#: The face of a surface's ONE run or commit command (Start, Init devices,
#: Save Pulse): the accent's own hue, deep enough to carry white ink at 5.2:1.
#: The only dark blue face there is.
ACCENT_STRONG = "#2F6FB0"
#: The name on a grid card a rule reached rather than a click: 5.0:1 on the
#: ACCENT_TINT wash.  Ink only; it never edges anything.
DERIVED_INK = "#40688F"
#: Dark accent: the ink of a standard command (7.7:1 on BUTTON_TINT) and of a
#: command on a tinted card (9.9:1 on its white face).
HOVER = "#004578"
BG = "#F3F3F3"
TEXT = "#323130"
#: The edge of a field, and of an empty grid cell one can fill: 2.6:1 on
#: white.  As text it is a prompt only -- see MUTED_INK.
PLACEHOLDER = "#A19F9D"
#: The label of a command, and the text of a field, that cannot be used
#: now: still readable (3.2:1 on the BG face) where PLACEHOLDER read 2.4:1.
DISABLED_INK = "#8A8886"
#: Quiet text that still CARRIES information -- a status line, a kind
#: "(processor)", a summary, a hint, a table or grid header, a grid row's
#: sub-line, an unselected tab, every field placeholder that is not a pure
#: prompt: 5.6:1 on white, 5.0:1 on the BG page, 4.8:1 on ACCENT_TINT (a
#: hovered row), still lighter than TEXT.  GREY and GRAPHITE are strokes and
#: dots, never words.
MUTED_INK = "#6A6866"
DIVIDER = "#E1DFDD"
#: The solid neutral of a control part that has no colour of its own and no
#: edge either: an unticked box, a resting scroll handle.  Dark enough to
#: read on a white card AND on the BG face of a Spacer card, where the
#: DIVIDER grey was 1.1:1 and vanished.
NEUTRAL_FILL = "#C8C6C4"
GREEN = "#7FC2AD"
RED = "#CD7380"
#: Red at reading strength: the ink of a destructive command (6.6:1 on
#: white) and of an error line.  RED itself is a dot and a fill colour; as
#: text on white it was 3.3:1, too pale to read as the warning it is.
RED_DARK = "#9E3B4A"
#: RED as a wash: the face of every LABELLED destructive command -- Remove,
#: Delete, Clear, Unload, Close, Shutdown, Stop -- no edge, RED_DARK
#: ink 5.4:1.  It sits CIEDE2000 10.1 from white, the weight of BUTTON_TINT
#: (10.3), so a Remove among light-blue commands is a command too; a fainter
#: seat (#FBF0F1, 5.9 from white) sat nearer the disabled face (2.4) and
#: read as a hole in a grid of buttons.  Under the pointer it steps toward
#: RED (#EFCED3, 4.5:1), pressed further.  A destructive GLYPH (a row's ×)
#: wears it only under the pointer.
RED_TINT = "#F7E3E6"
ORANGE = "#D69A6E"
ORANGE_TINT = "#F6E3D4"
ORANGE_DARK = "#8A4B1F"
API_VIOLET = "#9B86C9"
API_VIOLET_DARK = "#5A4A8A"
#: A config binding is the board's own calibrated number -- a channel delay
#: or a DAC bias, loaded onto the sequencer rather than supplied per run.
#: Green, because it reads as "already settled" beside the orange a table
#: sweeps and the violet a caller writes.
CONFIG_GREEN = "#7FC2AD"
#: The config green at reading strength: a config title, a config-bound
#: value's ink, and the face of the "C" badge under its white letter (6.2:1;
#: white on CONFIG_GREEN itself was 2.0:1).
CONFIG_GREEN_DARK = "#2F6B58"
CONFIG_GREEN_TINT = "#D1E8E1"
GREY = "#A2A2A2"
GRAPHITE = "#808080"
SURFACE = "#FFFFFF"

MUTED_LABEL_STYLE = f"color: {MUTED_INK}; background: transparent; border: none;"

RADIUS = 4
CARD_TITLE_PX = 32
#: Clear space above, below and to the right of a panel card's title band --
#: one number, because the operator sees one inset all the way round the
#: band's contents.
CARD_TITLE_PAD = 4
#: Colours that pair a shape's numbers with the axis names under them: group
#: i of the structure is written in colour i on both lines, so the eye can
#: match "83x60" to "spatial-y x spatial-x" without counting brackets.
AXIS_GROUP_COLORS = ("#5A4A8A", "#8A4B1F", "#004578")
CARD_PAD = 10
FONT = "Segoe UI"
FONT_SIZE = 12
PADDING_V = 1
EDIT_PADDING_H = 4
#: Air either side of a button's label.  An edit's 4 px left a word almost
#: touching the edge of a button around it.
BUTTON_PADDING_H = 10
WINDOW_PAD = 14
TITLE_LEFT_INSET = WINDOW_PAD
#: The product mark in a window's own title bar, and the air after it.  The
#: MARK sits on the body's left column, so the title block still begins where
#: the content below it does; the words follow the mark.  Its size is a
#: FRACTION OF THE BAR rather than a scaled pixel count: the bar does not take
#: the Fluent scale, so a scaled mark shrank inside a bar that had not, and a
#: four-letter wordmark has nothing left to show at twelve pixels.
TITLE_ICON_RATIO = 0.625
TITLE_ICON_GAP = 8
COMBO_WIDTH = 16
COMBO_TRI_SIZE = 8

FLUENT_SCALE_MIN = 0.72
FLUENT_SCALE_MAX = 1.25
AUTO_SCALE_BASIS = (1280, 790)
AUTO_SCALE_MARGIN = (48, 88)
WINDOW_FALLBACK_PX = (1280, 760)
WINDOW_FALLBACK_MIN_PX = (960, 620)
WINDOW_MIN_PX = (980, 640)
WINDOW_MIN_FLOOR_PX = (820, 560)
WINDOW_MARGIN_PX = (40, 48)
WINDOW_MARGIN_FLOOR_PX = (28, 32)
WINDOW_TITLEBAR_PX = 36
WINDOW_TITLEBAR_FLOOR_PX = 28
WINDOW_MAX_FLOOR_PX = (360, 320)
WINDOW_SCREEN_FRACTION = 0.90

__all__ = [
    "ACCENT",
    "ACCENT_HOVER",
    "ACCENT_STRONG",
    "ACCENT_TINT",
    "BUTTON_TINT",
    "DERIVED_INK",
    "DISABLED_INK",
    "MUTED_INK",
    "NEUTRAL_FILL",
    "SELECTED",
    "SELECTED_HOVER",
    "SELECTION_EDGE",
    "API_VIOLET",
    "API_VIOLET_DARK",
    "BG",
    "BUTTON_PADDING_H",
    "AXIS_GROUP_COLORS",
    "CARD_PAD",
    "CARD_TITLE_PAD",
    "CARD_TITLE_PX",
    "COMBO_TRI_SIZE",
    "COMBO_WIDTH",
    "DIVIDER",
    "EDIT_PADDING_H",
    "FLUENT_SCALE_MAX",
    "FLUENT_SCALE_MIN",
    "FONT",
    "FONT_SIZE",
    "GREEN",
    "GRAPHITE",
    "GREY",
    "HOVER",
    "MUTED_LABEL_STYLE",
    "ORANGE",
    "ORANGE_DARK",
    "ORANGE_TINT",
    "PADDING_V",
    "PLACEHOLDER",
    "RADIUS",
    "RED",
    "RED_DARK",
    "RED_TINT",
    "SURFACE",
    "TEXT",
    "TITLE_LEFT_INSET",
    "WINDOW_PAD",
    "WINDOW_SCREEN_FRACTION",
]
