"""Fixed preset geometry shared by GUI, notebook and export surfaces.

Only the nine declared panel presets are accepted.  A host may choose another
preset through ``PlotSession.set_size``; it may not inject an arbitrary
window size and thereby create backend-specific layout behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from ._validation import finite_real, integer
from .kinds import PlotKind
from .style import FontStyleConfig, PlotStyleConfig


_PANEL_PRESET_CELLS = (
    ("1x2", 1, 2),
    ("2x2", 2, 2),
    ("4x2", 4, 2),
    ("1x4", 1, 4),
    ("2x4", 2, 4),
    ("4x4", 4, 4),
    ("4x8", 4, 8),
    ("8x4", 8, 4),
    ("8x8", 8, 8),
)
PANEL_SIZE_NAMES = tuple(name for name, _, _ in _PANEL_PRESET_CELLS)


def _finite(value: object, field: str, *, positive: bool = False) -> float:
    result = finite_real(value, field)
    if positive and result <= 0.0:
        raise ValueError(f"{field} must be positive and finite")
    return result


def _round_pixel(value: float) -> int:
    return max(1, int(math.floor(float(value) + 0.5)))


@dataclass(frozen=True, slots=True)
class PixelSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", integer(self.width, "pixel width", minimum=1))
        object.__setattr__(self, "height", integer(self.height, "pixel height", minimum=1))

@dataclass(frozen=True, slots=True)
class Margins:
    left: int
    right: int
    bottom: int
    top: int

    def __post_init__(self) -> None:
        for field in ("left", "right", "bottom", "top"):
            value = integer(getattr(self, field), f"{field} margin")
            if value < 0:
                raise ValueError("margins must be non-negative")
            object.__setattr__(self, field, value)

@dataclass(frozen=True, slots=True)
class PanelPreset:
    name: str
    rows: int
    columns: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("panel preset name must be text")
        rows = integer(self.rows, "preset rows", minimum=1)
        columns = integer(self.columns, "preset columns", minimum=1)
        canonical = f"{rows}x{columns}"
        if self.name != canonical:
            raise ValueError(f"panel preset name must be {canonical!r}")


@dataclass(frozen=True, slots=True)
class ImageSplit:
    image: float
    distribution: float
    colorbar: float
    image_distribution_gap: float
    distribution_colorbar_gap: float

    def __post_init__(self) -> None:
        for field in (
            "image",
            "distribution",
            "colorbar",
            "image_distribution_gap",
            "distribution_colorbar_gap",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        total = (
            self.image
            + self.distribution
            + self.colorbar
            + self.image_distribution_gap
            + self.distribution_colorbar_gap
        )
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("image split widths and gaps must sum to 1")


@dataclass(frozen=True, slots=True)
class RollingSplit:
    history: float
    gap: float
    distribution: float

    def __post_init__(self) -> None:
        for field in ("history", "gap", "distribution"):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        if not math.isclose(
            self.history + self.gap + self.distribution,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("rolling split widths and gap must sum to 1")


@dataclass(frozen=True, slots=True)
class PlotLayoutConfig:
    """All logical geometry and fixed-size policy."""

    presets: tuple[PanelPreset, ...]
    default_preset: str
    design_dpi: float
    display_scale: float
    export_dpi: float
    panel_unit: PixelSize
    panel_margins: Margins
    pulse_left_margin_px: int
    image_split: ImageSplit
    rolling_split: RollingSplit
    facet_max_columns: int
    facet_double_rows_threshold: int
    facet_double_columns_threshold: int
    facet_column_gap_px: int
    facet_min_column_gap_px: int
    facet_row_gap_extra_px: int
    pulse_row_min_px: int
    pulse_period_min_px: int

    def __post_init__(self) -> None:
        presets = tuple(self.presets)
        if not presets or any(not isinstance(item, PanelPreset) for item in presets):
            raise ValueError("presets must contain PanelPreset values")
        names = tuple(item.name for item in presets)
        if len(names) != len(set(names)):
            raise ValueError("panel preset names must be unique")
        if names != PANEL_SIZE_NAMES:
            raise ValueError(
                "presets must be the nine fixed panel sizes in canonical order"
            )
        object.__setattr__(self, "presets", presets)
        if not isinstance(self.default_preset, str) or self.default_preset not in names:
            raise ValueError("default_preset must name one declared preset")
        for field in ("design_dpi", "display_scale", "export_dpi"):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        if self.export_dpi < self.live_dpi:
            raise ValueError("export_dpi must not be lower than live_dpi")
        if not isinstance(self.panel_unit, PixelSize):
            raise TypeError("panel_unit must be PixelSize")
        if not isinstance(self.panel_margins, Margins):
            raise TypeError("panel_margins must be Margins")
        if not isinstance(self.image_split, ImageSplit):
            raise TypeError("image_split must be ImageSplit")
        if not isinstance(self.rolling_split, RollingSplit):
            raise TypeError("rolling_split must be RollingSplit")
        for field in (
            "pulse_left_margin_px",
            "facet_max_columns",
            "facet_double_rows_threshold",
            "facet_double_columns_threshold",
            "facet_column_gap_px",
            "facet_min_column_gap_px",
            "facet_row_gap_extra_px",
            "pulse_row_min_px",
            "pulse_period_min_px",
        ):
            object.__setattr__(self, field, integer(getattr(self, field), field, minimum=1))
        if self.facet_min_column_gap_px > self.facet_column_gap_px:
            raise ValueError("minimum facet column gap cannot exceed its normal gap")

    @property
    def size_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.presets)

    @property
    def live_dpi(self) -> float:
        return self.design_dpi * self.display_scale

    @property
    def export_scale(self) -> float:
        return self.export_dpi / self.live_dpi

    @property
    def facet_max_cells(self) -> int:
        """Largest row-major facet capacity available in the fixed presets."""

        return max(item.rows * item.columns for item in self.presets)

    def validate_preset(self, preset: str) -> str:
        if not isinstance(preset, str):
            raise TypeError("panel preset must be text")
        canonical = preset.strip().lower().replace(" ", "")
        if canonical not in self.size_names:
            raise ValueError(
                f"unknown panel preset {preset!r}; choose from {', '.join(self.size_names)}"
            )
        return canonical

    def preset(self, preset: str) -> PanelPreset:
        canonical = self.validate_preset(preset)
        return next(item for item in self.presets if item.name == canonical)


DEFAULT_LAYOUT = PlotLayoutConfig(
    presets=tuple(
        PanelPreset(name, rows, columns)
        for name, rows, columns in _PANEL_PRESET_CELLS
    ),
    default_preset="2x2",
    design_dpi=300.0,
    display_scale=0.7,
    export_dpi=600.0,
    panel_unit=PixelSize(240, 180),
    # (left, right, bottom, top).  The horizontal pair is the instrument these
    # figures are modelled on: Confocal-GUIv2 live_plot/plot_strategy.py sets
    # fixed_data_px (480, 360) with margins_px (110, 110, 100, 40) under a
    # "canvas area (700, 500)" -- self-proving, since 110+480+110 = 700.  This
    # tree had carried 96 on the right, which is nobody's chosen number and
    # made the canvas 686 wide.  Symmetry restored.
    #
    # The VERTICAL pair is this project's, deliberately: 80 bottom and 70 top
    # rather than the reference's 100/40.  These panels carry a title where
    # the reference had none and sit in a grid where a tall bottom margin
    # doubles as the row gap, so the room is spent differently.  Kept on the
    # owner's call, and recorded here so the difference reads as a decision
    # rather than as the drift the 96 turned out to be.
    panel_margins=Margins(110, 110, 80, 70),
    pulse_left_margin_px=122,
    image_split=ImageSplit(0.75, 0.10, 0.10, 0.025, 0.025),
    rolling_split=RollingSplit(0.825, 0.025, 0.15),
    facet_max_columns=7,
    facet_double_rows_threshold=4,
    facet_double_columns_threshold=5,
    facet_column_gap_px=10,
    facet_min_column_gap_px=6,
    facet_row_gap_extra_px=4,
    pulse_row_min_px=26,
    pulse_period_min_px=46,
)


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    """Top-origin normalized rectangle used identically by every backend."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        for field in ("left", "top", "right", "bottom"):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        if not 0.0 <= self.left < self.right <= 1.0:
            raise ValueError("box requires 0 <= left < right <= 1")
        if not 0.0 <= self.top < self.bottom <= 1.0:
            raise ValueError("box requires 0 <= top < bottom <= 1")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def matplotlib_bounds(self) -> tuple[float, float, float, float]:
        return (self.left, 1.0 - self.bottom, self.width, self.height)


@dataclass(frozen=True, slots=True)
class AxesPlan:
    role: str
    box: NormalizedBox
    cell_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("axes role must be non-empty text")
        object.__setattr__(self, "role", self.role.strip())
        if not isinstance(self.box, NormalizedBox):
            raise TypeError("axes box must be NormalizedBox")
        if self.cell_index is not None:
            object.__setattr__(self, "cell_index", integer(self.cell_index, "cell_index"))
            if self.cell_index < 0:
                raise ValueError("cell_index must be non-negative")


@dataclass(frozen=True, slots=True)
class FacetTopology:
    """Render topology for homogeneous FacetGrid cells.

    Only the cell count (and optionally the drawn cell shape) is semantic; the
    row/column packing is always the layout's optimization.

    The shape is stated as HEIGHT OVER WIDTH, and the name says so.  It was
    ``cell_aspect``, and the two readers divided by it in opposite directions
    -- the packer treating it as width/height, the split as height/width.
    Both agreed while every image declared 1.0; a camera frame that is not
    square is where one of them would have been wrong.
    """

    cell_count: int
    cell_height_over_width: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_count", integer(self.cell_count, "cell_count", minimum=1))
        if self.cell_height_over_width is not None:
            object.__setattr__(
                self,
                "cell_height_over_width",
                _finite(
                    self.cell_height_over_width,
                    "cell_height_over_width",
                    positive=True,
                ),
            )

@dataclass(frozen=True, slots=True)
class FacetTypographyPlan:
    """Cell chrome uses the reference compact/normal typography tier.

    ``cell_title_max_width_pt`` is one cell's EXCLUSIVE title room: its own
    width plus the column gap, because two neighbours each annexing half a
    gap cannot collide.  ``cell_title_min_pt`` is the readable floor a title
    may shrink to before it is truncated instead.
    """

    tier: str
    scale: float
    cell_title_pt: float
    tick_pt: float
    outer_axis_label_pt: float
    cell_title_min_pt: float
    cell_title_max_width_pt: float

    def __post_init__(self) -> None:
        if self.tier not in {"compact", "normal"}:
            raise ValueError("facet typography tier must be 'compact' or 'normal'")
        for field in (
            "scale",
            "cell_title_pt",
            "tick_pt",
            "outer_axis_label_pt",
            "cell_title_min_pt",
            "cell_title_max_width_pt",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        if self.cell_title_min_pt > self.cell_title_pt:
            raise ValueError("the title floor cannot exceed the planned title size")


@dataclass(frozen=True, slots=True)
class SurfacePlan:
    """Fully resolved immutable geometry for one surface revision."""

    preset: str
    kind: str
    logical_size: tuple[int, int]
    raster_size: tuple[int, int]
    figure_size_inches: tuple[float, float]
    logical_dpi: float
    dpi: float
    device_pixel_ratio: float
    export_scale: float
    axes: tuple[AxesPlan, ...]
    facet_topology: FacetTopology | None
    facet_shape: tuple[int, int] | None
    recommended_preset: str | None
    facet_typography: FacetTypographyPlan | None
    #: FacetGrid only: the focused-cell geometry, resolved through the SAME
    #: image split the standalone Image kind gets -- (image, distribution,
    #: colorbar) AxesPlans over the overview's full data region.  The
    #: renderer uses it when the focused cell is an image so the focused
    #: view carries the standalone kind's complete chrome.
    facet_focus_axes: tuple[AxesPlan, ...] | None
    rolling_side_distribution: bool
    #: Image kinds only: the drawn box's height over its width, the ONE
    #: number that shaped the image slot.  The renderer reads the shape from
    #: here rather than interrogating the axes, because the axes only learns
    #: its aspect while its artists update -- which is after the box has been
    #: positioned, so on a first frame the axes had nothing to say and the
    #: box was left square for Matplotlib to shrink into half a pixel.
    image_height_over_width: float | None = None
    #: Image kinds only: the PICTURE's own height over its width, which is
    #: not the box's whenever the two differ -- the field is square by
    #: requirement and a frame that is not square is letterboxed in it.  The
    #: box size is chosen so that letterboxed picture lands on whole pixels.
    image_picture_height_over_width: float | None = None

    def __post_init__(self) -> None:
        for field in ("logical_size", "raster_size"):
            values = tuple(getattr(self, field))
            if len(values) != 2:
                raise ValueError(f"{field} must have width and height")
            values = tuple(integer(value, field, minimum=1) for value in values)
            object.__setattr__(self, field, values)
        inches = tuple(self.figure_size_inches)
        if len(inches) != 2:
            raise ValueError("figure_size_inches must have width and height")
        object.__setattr__(
            self,
            "figure_size_inches",
            tuple(_finite(value, "figure size", positive=True) for value in inches),
        )
        for field in ("logical_dpi", "dpi", "device_pixel_ratio", "export_scale"):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        axes = tuple(self.axes)
        if not axes or any(not isinstance(item, AxesPlan) for item in axes):
            raise ValueError("surface must contain AxesPlan values")
        object.__setattr__(self, "axes", axes)
        if self.facet_focus_axes is not None:
            focus_axes = tuple(self.facet_focus_axes)
            if not focus_axes or any(
                not isinstance(item, AxesPlan) for item in focus_axes
            ):
                raise ValueError("facet_focus_axes must contain AxesPlan values")
            object.__setattr__(self, "facet_focus_axes", focus_axes)
        if not isinstance(self.rolling_side_distribution, bool):
            raise TypeError("rolling_side_distribution must be bool")

def _kind_name(kind: object) -> str:
    value = getattr(kind, "value", kind)
    if not isinstance(value, str):
        raise TypeError("plot kind must be text or a string-valued enum")
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return PlotKind(key).value
    except ValueError as error:
        raise ValueError(f"unknown plot kind {value!r}") from error


def _panel_design_geometry(
    preset: PanelPreset,
    kind: str,
    layout: PlotLayoutConfig,
) -> tuple[int, int, Margins]:
    margins = layout.panel_margins
    if kind == "pulse_timeline":
        margins = Margins(
            layout.pulse_left_margin_px,
            margins.right,
            margins.bottom,
            margins.top,
        )
    data_width = preset.columns * layout.panel_unit.width
    data_height = preset.rows * layout.panel_unit.height
    return data_width, data_height, margins


def _data_box(
    data_width: int,
    data_height: int,
    margins: Margins,
) -> NormalizedBox:
    width = margins.left + data_width + margins.right
    height = margins.top + data_height + margins.bottom
    return NormalizedBox(
        margins.left / width,
        margins.top / height,
        (margins.left + data_width) / width,
        (margins.top + data_height) / height,
    )


def facet_shape(cell_count: int, *, max_columns: int) -> tuple[int, int]:
    count = integer(cell_count, "cell_count", minimum=1)
    cap = integer(max_columns, "max_columns", minimum=1)
    columns = min(cap, int(math.ceil(math.sqrt(count))))
    return (int(math.ceil(count / columns)), columns)


def facet_shape_for_cell_shape(
    cell_count: int,
    cell_height_over_width: float,
    region_px: tuple[int, int],
    *,
    max_columns: int,
) -> tuple[int, int]:
    count = integer(cell_count, "cell_count", minimum=1)
    aspect = _finite(
        cell_height_over_width, "cell_height_over_width", positive=True
    )
    if len(region_px) != 2:
        raise ValueError("region_px must contain width and height")
    region_width, region_height = (
        _finite(value, "region dimension", positive=True) for value in region_px
    )
    cap = min(integer(max_columns, "max_columns", minimum=1), count)
    best_key: tuple[float, int, int] | None = None
    best = (1, count)
    for columns in range(1, cap + 1):
        rows = int(math.ceil(count / columns))
        cell_width = region_width / columns
        cell_height = region_height / rows
        image_height = min(cell_height, cell_width * aspect)
        key = (image_height, -(rows * columns), -abs(rows - columns))
        if best_key is None or key > best_key:
            best_key = key
            best = (rows, columns)
    return best


def _facet_cells_union(axes: tuple[AxesPlan, ...]) -> NormalizedBox:
    """The full data region occupied by a FacetGrid overview's cells."""

    boxes = tuple(item.box for item in axes if item.role == "facet_cell")
    if not boxes:
        raise ValueError("FacetGrid surface has no cell geometry")
    return NormalizedBox(
        min(box.left for box in boxes),
        min(box.top for box in boxes),
        max(box.right for box in boxes),
        max(box.bottom for box in boxes),
    )


def facet_focus_box(plan: SurfacePlan) -> NormalizedBox:
    """Return the full data region occupied by a FacetGrid overview."""

    if not isinstance(plan, SurfacePlan):
        raise TypeError("plan must be SurfacePlan")
    if plan.kind != "facet_grid":
        raise TypeError("facet focus geometry requires a FacetGrid surface")
    return _facet_cells_union(plan.axes)


def recommended_facet_preset(
    topology: FacetTopology,
    layout: PlotLayoutConfig = DEFAULT_LAYOUT,
) -> str:
    if not isinstance(topology, FacetTopology):
        raise TypeError("topology must be FacetTopology")
    if not isinstance(layout, PlotLayoutConfig):
        raise TypeError("layout must be PlotLayoutConfig")
    rows, columns = facet_shape(
        topology.cell_count, max_columns=layout.facet_max_columns
    )
    row_units = 4 if rows > layout.facet_double_rows_threshold else 2
    column_units = 4 if columns > layout.facet_double_columns_threshold else 2
    return layout.validate_preset(f"{row_units}x{column_units}")


def _facet_scale(
    selected: PanelPreset,
    recommended: PanelPreset,
    style: PlotStyleConfig,
) -> tuple[str, float]:
    compact = selected.rows < recommended.rows or selected.columns < recommended.columns
    scale = style.fonts.facet_compact_scale if compact else style.fonts.facet_normal_scale
    return ("compact" if compact else "normal"), scale


def _facet_gaps(
    scale: float,
    style: PlotStyleConfig,
    layout: PlotLayoutConfig,
) -> tuple[int, int]:
    """(row gap, column gap) in design pixels for one typography scale."""

    row_gap = _round_pixel(
        style.fonts.tick_pt * scale * layout.design_dpi / 72.0
    ) + layout.facet_row_gap_extra_px
    column_gap = max(
        layout.facet_min_column_gap_px,
        _round_pixel(layout.facet_column_gap_px * scale),
    )
    return row_gap, column_gap


@lru_cache(maxsize=1024)
def _text_size_pt(
    text: str,
    families: tuple[str, ...],
    size_pt: float,
) -> tuple[float, float]:
    """Measure one line of text, in points, without a canvas.

    Cached: the answer is a property of the glyphs and cannot change while a
    session runs, and the tick policy asks it for every candidate label of
    every unit it considers -- laying out a font and walking its paths, a
    dozen times an axis, for every axis of every cell of a live grid.
    """

    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    width, height, _descent = TextToPath().get_text_width_height_descent(
        text,
        FontProperties(family=list(families), size=size_pt),
        False,
    )
    return float(width), float(height)


def _text_width_pt(text: str, families: tuple[str, ...], size_pt: float) -> float:
    return _text_size_pt(text, families, size_pt)[0]


def fitted_facet_cell_title(
    label: str,
    typography: FacetTypographyPlan,
    fonts: FontStyleConfig,
) -> tuple[str, float]:
    """The exact text and size one cell title may occupy without overlap.

    A title wider than the cell's exclusive room shrinks to fit; a title
    still too wide at the readable floor is truncated with an ellipsis,
    because two overlapping titles read as one wrong label while a
    shortened one reads as itself.
    """

    label = str(label)
    budget = typography.cell_title_max_width_pt
    width = _text_width_pt(label, fonts.sans_serif, typography.cell_title_pt)
    if width <= budget:
        return label, typography.cell_title_pt
    fitted = typography.cell_title_pt * budget / width
    if fitted >= typography.cell_title_min_pt:
        return label, fitted
    floor = typography.cell_title_min_pt
    for keep in range(len(label) - 1, 0, -1):
        shortened = label[:keep].rstrip() + "\N{HORIZONTAL ELLIPSIS}"
        if _text_width_pt(shortened, fonts.sans_serif, floor) <= budget:
            return shortened, floor
    return "\N{HORIZONTAL ELLIPSIS}", floor


def _facet_typography(
    selected: PanelPreset,
    recommended: PanelPreset,
    style: PlotStyleConfig,
    *,
    cell_title_max_width_pt: float,
) -> FacetTypographyPlan:
    tier, scale = _facet_scale(selected, recommended, style)
    cell_title_pt = style.fonts.tick_pt * scale
    return FacetTypographyPlan(
        tier=tier,
        scale=scale,
        cell_title_pt=cell_title_pt,
        tick_pt=style.fonts.tick_pt * scale,
        outer_axis_label_pt=style.fonts.axis_label_pt,
        cell_title_min_pt=min(style.fonts.facet_title_min_pt, cell_title_pt),
        cell_title_max_width_pt=cell_title_max_width_pt,
    )


def _split_image(
    data: NormalizedBox,
    split: ImageSplit,
    *,
    region_px: tuple[float, float] | None = None,
    image_height_over_width: float | None = None,
    scene: bool = False,
) -> tuple[AxesPlan, ...]:
    """The image and its two strips, measured in units of the IMAGE's width.

    An aspect-locked image does not fill the width it is given: its width is
    set by the region's HEIGHT, and the two coincide only where the preset is
    square.  Every width here used to be a fraction of the REGION, so on a
    wide preset the image drew at the left, the strips sat at the far right,
    and the surplus became a hole between them.

    ``image_height_over_width`` is the drawn box's height over its width (1.0
    for a square image, ``None`` when nothing locks it).  Where it does not bind,
    the unit IS the region's width and every box is exactly what it was --
    the split's five ratios sum to one, so the colorbar still ends at the
    region's right edge.  Where it binds, the leftover simply stays unused,
    on the side ``style.image_anchor`` leaves free.
    """

    span = data.width * split.image
    # Every other width is a ratio OF the image, not of the region -- and
    # where the aspect does not bind, the image IS the region's share, so the
    # unit stays the region's width and every box is arithmetically identical
    # to what it was.
    unit = data.width
    if image_height_over_width is not None and region_px is not None:
        width_px, height_px = (float(value) for value in region_px)
        shape = float(image_height_over_width)
        if width_px > 0.0 and height_px > 0.0 and shape > 0.0:
            drawn_px = min(width_px * split.image, height_px / shape)
            drawn = data.width * (drawn_px / width_px)
            if drawn < span:
                span = drawn
                unit = span / split.image
    cursor = data.left
    image = NormalizedBox(cursor, data.top, cursor + span, data.bottom)
    cursor = image.right + unit * split.image_distribution_gap
    distribution = NormalizedBox(
        cursor,
        data.top,
        cursor + unit * split.distribution,
        data.bottom,
    )
    cursor = distribution.right + unit * split.distribution_colorbar_gap
    colorbar = NormalizedBox(
        cursor, data.top, cursor + unit * split.colorbar, data.bottom
    )
    if scene:
        image = _scene_box(image, data, split, region_px, distribution)
    return (
        AxesPlan("image", image),
        AxesPlan("distribution", distribution),
        AxesPlan("colorbar", colorbar),
    )


def _scene_box(
    image: NormalizedBox,
    data: NormalizedBox,
    split: ImageSplit,
    region_px: tuple[float, float] | None,
    distribution: NormalizedBox,
) -> NormalizedBox:
    """The room a 3D scene of this image gets: the whole picture area.

    A heatmap can reserve margins because its chrome has fixed places --
    ticks under the bottom spine, labels left of the left one.  Turn a
    camera and a label that hung under the floor is beside the colorbar,
    so no margin can be reserved for a place that moves.  The scene and
    its labels therefore share ONE region and are cut at ONE edge, and
    that region is everything the picture side of the panel has: down to
    one padding from the figure's left and bottom edges, out to one
    padding from the rail beside it.  The top is the picture's own --
    the title's room is not the scene's to take.

    The padding is the gap the layout already leaves between the picture
    and that rail, and it is one VISUAL distance, so the vertical share
    is the horizontal one converted through the figure's pixel shape.
    """

    pad_x = split.image_distribution_gap * (image.width / split.image)
    pad_y = pad_x
    if region_px is not None:
        width_px, height_px = (float(value) for value in region_px)
        if width_px > 0.0 and height_px > 0.0:
            figure_w = width_px / data.width
            figure_h = height_px / data.height
            if figure_h > 0.0:
                pad_y = pad_x * figure_w / figure_h
    left = min(pad_x, image.left)
    right = max(distribution.left - pad_x, image.right)
    bottom = max(1.0 - pad_y, image.bottom)
    return NormalizedBox(left, image.top, min(right, 1.0), min(bottom, 1.0))


def _split_rolling(
    data: NormalizedBox,
    split: RollingSplit,
    show_distribution: bool,
) -> tuple[AxesPlan, ...]:
    if not show_distribution:
        return (AxesPlan("history", data),)
    history = NormalizedBox(
        data.left,
        data.top,
        data.left + data.width * split.history,
        data.bottom,
    )
    distribution = NormalizedBox(
        history.right + data.width * split.gap,
        data.top,
        data.right,
        data.bottom,
    )
    return (AxesPlan("history", history), AxesPlan("distribution", distribution))


def _facet_axes(
    count: int,
    shape: tuple[int, int],
    data_width: int,
    data_height: int,
    margins: Margins,
    row_gap: int,
    column_gap: int,
) -> tuple[AxesPlan, ...]:
    rows, columns = shape
    cell_width = max((data_width - (columns - 1) * column_gap) / columns, 1.0)
    cell_height = max((data_height - (rows - 1) * row_gap) / rows, 1.0)
    figure_width = margins.left + data_width + margins.right
    figure_height = margins.bottom + data_height + margins.top
    result = []
    for index in range(count):
        row, column = divmod(index, columns)
        left = margins.left + column * (cell_width + column_gap)
        bottom_from_origin = margins.bottom + (rows - 1 - row) * (cell_height + row_gap)
        top = figure_height - (bottom_from_origin + cell_height)
        result.append(
            AxesPlan(
                "facet_cell",
                NormalizedBox(
                    left / figure_width,
                    top / figure_height,
                    (left + cell_width) / figure_width,
                    (top + cell_height) / figure_height,
                ),
                index,
            )
        )
    return tuple(result)


def resolve_surface(
    preset: str | None = None,
    kind: object = "curve",
    facet_topology: FacetTopology | None = None,
    *,
    device_pixel_ratio: float = 1.0,
    export_scale: float | None = None,
    rolling_side_distribution: bool | None = None,
    image_height_over_width: float | None = None,
    image_picture_height_over_width: float | None = None,
    image_scene: bool = False,
    layout: PlotLayoutConfig,
    style: PlotStyleConfig,
) -> SurfacePlan:
    """Resolve one of the fixed presets into a backend-independent plan.

    ``device_pixel_ratio`` and ``export_scale`` affect only physical raster
    pixels and render DPI.  They never change logical size, normalized axes,
    facet packing or typography.
    """

    if not isinstance(layout, PlotLayoutConfig):
        raise TypeError("layout must be PlotLayoutConfig")
    if not isinstance(style, PlotStyleConfig):
        raise TypeError("style must be PlotStyleConfig")
    canonical_kind = _kind_name(kind)
    if canonical_kind == PlotKind.ROLLING.value:
        if not isinstance(rolling_side_distribution, bool):
            raise TypeError(
                "Rolling surfaces require an explicit rolling_side_distribution"
            )
    elif rolling_side_distribution is not None:
        raise ValueError(
            "rolling_side_distribution is accepted only for Rolling surfaces"
        )
    if image_height_over_width is not None:
        if canonical_kind != "image":
            raise ValueError(
                "image_height_over_width is accepted only for Image surfaces"
            )
        image_height_over_width = _finite(
            image_height_over_width, "image_height_over_width", positive=True
        )
    if not isinstance(image_scene, bool):
        raise TypeError("image_scene must be bool")
    if image_scene and canonical_kind not in ("image", "facet_grid"):
        raise ValueError(
            "image_scene is accepted only for Image and FacetGrid surfaces"
        )
    if canonical_kind == "facet_grid":
        if not isinstance(facet_topology, FacetTopology):
            raise TypeError("FacetGrid requires facet_topology")
        # One capacity for every facet layout: failing here beats collapsing
        # cells below one pixel downstream.
        if facet_topology.cell_count > layout.facet_max_cells:
            raise ValueError(
                f"FacetGrid needs {facet_topology.cell_count} cells, which "
                f"exceeds the fixed layout facet_max_cells capacity of "
                f"{layout.facet_max_cells}; pin an axis or change a fate to "
                "reduce the cells"
            )
        recommended_name = recommended_facet_preset(facet_topology, layout)
        selected_name = recommended_name if preset is None else layout.validate_preset(preset)
    else:
        if facet_topology is not None:
            raise ValueError("facet_topology is accepted only for FacetGrid")
        recommended_name = None
        selected_name = layout.default_preset if preset is None else layout.validate_preset(preset)
    selected = layout.preset(selected_name)
    data_width, data_height, margins = _panel_design_geometry(selected, canonical_kind, layout)
    figure_design_width = margins.left + data_width + margins.right
    figure_design_height = margins.top + data_height + margins.bottom
    logical_size = (
        _round_pixel(figure_design_width * layout.display_scale),
        _round_pixel(figure_design_height * layout.display_scale),
    )
    dpr = _finite(device_pixel_ratio, "device_pixel_ratio", positive=True)
    physical_export_scale = 1.0 if export_scale is None else _finite(
        export_scale, "export_scale", positive=True
    )
    physical_scale = dpr * physical_export_scale
    physical_dpi = layout.live_dpi * physical_scale
    raster_size = tuple(
        _round_pixel(value * physical_scale) for value in logical_size
    )
    # Derive inches from the already-rounded device raster.  Matplotlib
    # otherwise truncates ``figsize * dpi`` on fractional screen DPR and can
    # miss the fixed Qt/notebook surface by one physical pixel.
    figure_size_inches = tuple(
        pixels / physical_dpi for pixels in raster_size
    )
    data = _data_box(data_width, data_height, margins)
    typography = None
    shape = None
    show_distribution = False
    facet_focus_axes = None
    if canonical_kind == "image":
        axes = _split_image(
            data,
            layout.image_split,
            region_px=(data_width, data_height),
            image_height_over_width=image_height_over_width,
            scene=image_scene,
        )
    elif canonical_kind == "rolling":
        assert rolling_side_distribution is not None
        show_distribution = rolling_side_distribution
        axes = _split_rolling(data, layout.rolling_split, show_distribution)
    elif canonical_kind == "facet_grid":
        assert facet_topology is not None and recommended_name is not None
        recommended = layout.preset(recommended_name)
        _tier, scale = _facet_scale(selected, recommended, style)
        row_gap, column_gap = _facet_gaps(scale, style, layout)
        shape = (
            facet_shape(
                facet_topology.cell_count,
                max_columns=layout.facet_max_columns,
            )
            if facet_topology.cell_height_over_width is None
            else facet_shape_for_cell_shape(
                facet_topology.cell_count,
                facet_topology.cell_height_over_width,
                (data_width, data_height),
                max_columns=layout.facet_max_columns,
            )
        )
        columns = shape[1]
        cell_width = max((data_width - (columns - 1) * column_gap) / columns, 1.0)
        typography = _facet_typography(
            selected,
            recommended,
            style,
            # One cell's exclusive title room: its own width plus the column
            # gap, since two neighbours each annexing half a gap cannot
            # collide.  Design pixels become points at the design DPI.
            cell_title_max_width_pt=(
                (cell_width + column_gap) * 72.0 / layout.design_dpi
            ),
        )
        axes = _facet_axes(
            facet_topology.cell_count,
            shape,
            data_width,
            data_height,
            margins,
            row_gap,
            column_gap,
        )
        # A focused image cell is the standalone Image kind's surface: the
        # SAME _split_image split, applied over the overview's data region.
        union = _facet_cells_union(axes)
        facet_focus_axes = _split_image(
            union,
            layout.image_split,
            region_px=(
                union.width * figure_design_width,
                union.height * figure_design_height,
            ),
            image_height_over_width=facet_topology.cell_height_over_width,
            scene=image_scene,
        )
    else:
        axes = (AxesPlan("main", data),)
    return SurfacePlan(
        image_picture_height_over_width=image_picture_height_over_width,
        image_height_over_width=(
            image_height_over_width
            if facet_topology is None
            else facet_topology.cell_height_over_width
        ),
        preset=selected.name,
        kind=canonical_kind,
        logical_size=logical_size,
        raster_size=raster_size,
        figure_size_inches=figure_size_inches,
        logical_dpi=layout.live_dpi,
        dpi=physical_dpi,
        device_pixel_ratio=dpr,
        export_scale=physical_export_scale,
        axes=axes,
        facet_topology=facet_topology,
        facet_shape=shape,
        recommended_preset=recommended_name,
        facet_typography=typography,
        facet_focus_axes=facet_focus_axes,
        rolling_side_distribution=show_distribution,
    )


def recommended_pulse_preset(
    channel_count: int,
    period_count: int,
    *,
    layout: PlotLayoutConfig = DEFAULT_LAYOUT,
    style: PlotStyleConfig | None = None,
) -> str:
    """The smallest preset whose data region draws this pulse legibly.

    A pulse's size is decided by its content: a timeline with thirty channels
    and twenty periods needs a bigger surface than one with three of each, and
    picking per plot by hand is how two pulses end up drawn at densities that
    cannot be compared.  ``pulse_row_min_px`` and ``pulse_period_min_px`` are
    the floors that make a row and a period readable; both already lived here
    with nothing using them, which is a rule that was left behind.

    The largest preset is the answer when nothing fits -- an extreme pulse
    scrolls in its card rather than being drawn illegibly small.
    """

    from .config import DEFAULTS

    rows = max(1, integer(channel_count, "channel_count", minimum=0))
    periods = max(1, integer(period_count, "period_count", minimum=0))
    resolved_style = DEFAULTS.style if style is None else style
    by_area = sorted(layout.presets, key=lambda preset: preset.rows * preset.columns)
    for preset in by_area:
        plan = resolve_surface(
            preset.name,
            PlotKind.PULSE_TIMELINE,
            layout=layout,
            style=resolved_style,
        )
        data = next(item for item in plan.axes if item.role == "main").box
        width, height = plan.logical_size
        data_w = (data.right - data.left) * width
        data_h = (data.bottom - data.top) * height
        if (
            data_h >= rows * layout.pulse_row_min_px
            and data_w >= periods * layout.pulse_period_min_px
        ):
            return preset.name
    return by_area[-1].name


__all__ = [
    "AxesPlan",
    "DEFAULT_LAYOUT",
    "FacetTopology",
    "FacetTypographyPlan",
    "ImageSplit",
    "Margins",
    "NormalizedBox",
    "PANEL_SIZE_NAMES",
    "PanelPreset",
    "PixelSize",
    "PlotLayoutConfig",
    "RollingSplit",
    "SurfacePlan",
    "facet_shape",
    "recommended_pulse_preset",
    "facet_shape_for_cell_shape",
    "facet_focus_box",
    "recommended_facet_preset",
    "resolve_surface",
]
