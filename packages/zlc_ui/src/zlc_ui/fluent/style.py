"""Qt-chrome design tokens owned by :mod:`zlc_ui.fluent`.

This module defines the visual vocabulary used by reusable Qt controls and
custom host surfaces.
"""

from __future__ import annotations

ACCENT = "#77AADD"
#: The accent at reading strength: a wash a dark number stays legible
#: on.  ACCENT itself is a BUTTON colour -- it carries white text --
#: and a table cell painted with it hides the very number the click
#: was asking to see.
ACCENT_TINT = "#E4EFF9"
#: The edge of a selected card: the accent brought halfway to the resting
#: divider.  The fill (ACCENT_TINT) is what says "selected"; the edge only
#: closes the shape.  A full-strength 2 px accent border read as an outline
#: drawn around a card, not as the card being picked.
SELECTION_EDGE = "#ACC4DD"
#: The face of a standard command: the accent a step past ACCENT_TINT.
#: ACCENT_TINT is the selection wash and sat 1.17:1 on a white card --
#: a button there read as a selected cell, not a thing to press.  Dark-blue
#: (HOVER) ink on it reads 7.8:1.
BUTTON_TINT = "#D6E6F5"
HOVER = "#004578"
BG = "#F3F3F3"
TEXT = "#323130"
PLACEHOLDER = "#A19F9D"
DIVIDER = "#E1DFDD"
GREEN = "#7FC2AD"
RED = "#CD7380"
#: Red at reading strength: the ink of a destructive command and of an error
#: line.  RED itself is a dot and a fill colour; as text on white it was
#: 3.3:1, too pale to read as the warning it is.
RED_DARK = "#9E3B4A"
#: The face of a destructive command: RED as a wash, carrying RED_DARK ink.
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
CONFIG_GREEN_DARK = "#2F6B58"
CONFIG_GREEN_TINT = "#D1E8E1"
GREY = "#A2A2A2"
GRAPHITE = "#808080"
SURFACE = "#FFFFFF"

MUTED_LABEL_STYLE = f"color: {GREY}; background: transparent; border: none;"

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
    "SURFACE",
    "TEXT",
    "TITLE_LEFT_INSET",
    "WINDOW_PAD",
    "WINDOW_SCREEN_FRACTION",
]
