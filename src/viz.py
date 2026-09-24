"""Chart helpers for static report figures.

Follows the project's data-viz rules: reference palette on a light surface, thin marks
(24px bars with a 4px rounded data end, 2px lines, >= 8px markers with a 2px surface ring),
solid hairline grid, text in ink tokens (never the series colour), selective direct labels.

Sizes are in design pixels: figures are laid out at 100 px per inch and saved at 2x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

# Reference palette, light mode (validated with the data-viz palette script)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"
BLUE_250 = "#86b6ef"   # ordinal pair, light end (2.06:1 on surface)
BLUE_550 = "#1c5cab"   # ordinal pair, dark end
DEEMPHASIS = AXIS

PX_PER_IN = 100
SAVE_DPI = 200
PT_PER_PX = 72 / PX_PER_IN
FONTS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def signed(x: float, decimals: int = 3) -> str:
    """Signed number with a true minus sign (U+2212), e.g. '−0.047' / '+0.008'."""
    return f"{x:+.{decimals}f}".replace("-", "\u2212")


def px(n: float) -> float:
    """Design pixels -> points (for line widths, marker sizes, font sizes)."""
    return n * PT_PER_PX


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONTS,
        "font.size": px(12),
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": px(1),
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": px(11),
        "ytick.labelsize": px(12),
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "grid.color": GRID,
        "grid.linewidth": px(1),
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def figure(width_px: int, height_px: int, plot_box: tuple[int, int, int, int]):
    """Figure with one axes placed at an exact pixel box: (left, top, right, bottom) margins in px."""
    fig = plt.figure(figsize=(width_px / PX_PER_IN, height_px / PX_PER_IN), dpi=PX_PER_IN)
    left, top, right, bottom = plot_box
    ax = fig.add_axes([
        left / width_px, bottom / height_px,
        (width_px - left - right) / width_px, (height_px - top - bottom) / height_px,
    ])
    return fig, ax


def header(fig, title: str, subtitle: str, top_px: int = 20, left_px: int = 24) -> None:
    w, h = fig.get_size_inches() * PX_PER_IN
    fig.text(left_px / w, 1 - top_px / h, title, ha="left", va="top",
             fontsize=px(16), fontweight="bold", color=INK)
    fig.text(left_px / w, 1 - (top_px + 26) / h, subtitle, ha="left", va="top",
             fontsize=px(12.5), color=INK_2, linespacing=1.45)


def footnote(fig, text: str, bottom_px: int = 14, left_px: int = 24) -> None:
    w, h = fig.get_size_inches() * PX_PER_IN
    fig.text(left_px / w, bottom_px / h, text, ha="left", va="bottom",
             fontsize=px(10.5), color=MUTED, linespacing=1.45)


def _data_per_px(ax) -> tuple[float, float]:
    fig = ax.figure
    bbox = ax.get_position()
    w_px = bbox.width * fig.get_size_inches()[0] * PX_PER_IN
    h_px = bbox.height * fig.get_size_inches()[1] * PX_PER_IN
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return abs(x1 - x0) / w_px, abs(y1 - y0) / h_px


def rounded_hbar(ax, y: float, value: float, thickness_px: float = 24, radius_px: float = 4,
                 color: str = SERIES_1) -> None:
    """Horizontal bar from x=0 with a rounded data end and a square baseline.

    Call after the axes limits are final: the rounding is converted from pixels to data units.
    """
    dx, dy = _data_per_px(ax)
    h = thickness_px * dy / 2
    rx, ry = min(radius_px * dx, value / 2), radius_px * dy
    verts = [
        (0, y - h), (value - rx, y - h),
        (value, y - h), (value, y - h + ry),          # quadratic corner
        (value, y + h - ry),
        (value, y + h), (value - rx, y + h),          # quadratic corner
        (0, y + h), (0, y - h),
    ]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def dot(ax, x: float, y: float, color: str, diameter_px: float = 10, ring_px: float = 2, **kw) -> None:
    """Filled marker with a surface-coloured ring (keeps it legible over lines)."""
    ax.plot([x], [y], "o", ms=px(diameter_px + 2 * ring_px), mfc=color, mec=SURFACE,
            mew=px(ring_px), zorder=4, **kw)


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=SAVE_DPI)
    return path


def show_saved(fig, path: str | Path):
    """Save, close the live figure, and return the PNG for display at its design width.

    Notebooks then show exactly the file used in the reports, with its fixed margins
    (the inline backend would otherwise re-crop the figure with a tight bounding box).
    """
    from IPython.display import Image

    path = save(fig, path)
    plt.close(fig)
    return Image(filename=str(path), width=int(fig.get_size_inches()[0] * PX_PER_IN))


def lab_top_ranked_chart(groups: list[str], employee_share: list[float], reference_index: int,
                         title: str, subtitle: str, note: str):
    """Share of Google employees among the lab model's top-ranked first visits (one series)."""
    apply_style()
    n = len(groups)
    fig, ax = figure(760, 90 + 56 * n + 156, plot_box=(200, 96, 72, 112))
    ax.set_xlim(0, 1)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n), groups)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", zorder=0)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    for i, share in enumerate(employee_share):
        color = DEEMPHASIS if i == reference_index else SERIES_1
        rounded_hbar(ax, i, share, color=color)
        ax.text(share + 0.012, i, f"{share:.1%}", va="center", ha="left",
                fontsize=px(12.5), color=INK, fontweight="bold" if i == 0 else "normal", zorder=5)
    header(fig, title, subtitle)
    footnote(fig, note)
    return fig


def auc_dumbbell_chart(rows: list[dict], title: str, subtitle: str, note: str,
                       xlim: tuple[float, float] = (0.84, 0.925)):
    """ROC-AUC on all first visits vs external visitors only, one row per model.

    Each row: {"label", "auc_all", "auc_external", "gap_text"}.
    """
    apply_style()
    n = len(rows)
    fig, ax = figure(760, 150 + 74 * n + 128, plot_box=(250, 150, 190, 96))
    ax.set_xlim(*xlim)
    ax.set_ylim(n - 0.5, -0.6)
    ax.set_yticks(range(n), [r["label"] for r in rows])
    ticks = np.round(np.arange(np.ceil(xlim[0] * 50) / 50, xlim[1] + 1e-9, 0.02), 2)
    ax.set_xticks(ticks, [f"{t:.2f}" for t in ticks])
    ax.grid(axis="x", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_xlabel("ROC-AUC", fontsize=px(11), color=MUTED, labelpad=6)
    for i, r in enumerate(rows):
        ax.plot([r["auc_external"], r["auc_all"]], [i, i], color=AXIS, lw=px(2),
                solid_capstyle="round", zorder=2)
        dot(ax, r["auc_all"], i, BLUE_250)
        dot(ax, r["auc_external"], i, BLUE_550)
        for x, align in [(r["auc_all"], "left" if r["auc_all"] >= r["auc_external"] else "right"),
                         (r["auc_external"], "right" if r["auc_all"] >= r["auc_external"] else "left")]:
            offset = 0.0035 if align == "left" else -0.0035
            ax.text(x + offset, i - 0.28, f"{x:.3f}", ha=align, va="center", fontsize=px(11), color=INK_2)
        ax.text(1.02, i, r["gap_text"], transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=px(11.5), color=INK, linespacing=1.4)
    header(fig, title, subtitle)
    # Legend (two series): dot swatches + ink text, placed under the subtitle
    w, h = fig.get_size_inches() * PX_PER_IN
    lx, ly = 24, 108
    for color, label in [(BLUE_250, "All first visits (how the lab evaluates)"),
                         (BLUE_550, "External visitors only")]:
        fig.add_artist(plt.Line2D([(lx + 5) / w], [1 - ly / h], marker="o", ms=px(10), mfc=color,
                                  mec=SURFACE, mew=0, transform=fig.transFigure))
        t = fig.text((lx + 16) / w, 1 - ly / h, label, ha="left", va="center", fontsize=px(12), color=INK_2)
        fig.canvas.draw()
        lx += 16 + t.get_window_extent().width / (fig.dpi / PX_PER_IN) + 28
    fig.text(1 - 180 / w, 1 - 132 / h, "Change on external visitors", ha="left", va="bottom",
             fontsize=px(11), color=MUTED)
    footnote(fig, note)
    return fig
