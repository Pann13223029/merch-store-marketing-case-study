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


def nice_ticks(vmax: float, max_ticks: int = 7) -> np.ndarray:
    """Ticks from 0 on a 1-2-2.5-5 step so labels are round numbers."""
    for step in [m * 10.0 ** e for e in range(-5, 3) for m in (1, 2, 2.5, 5)]:
        if vmax / step <= max_ticks - 1:
            return np.arange(0, vmax + 1e-12, step)
    return np.linspace(0, vmax, max_ticks)


def _pct_decimals(ticks: np.ndarray) -> int:
    """Fewest decimals that keep every percentage tick label exact."""
    for d in range(0, 4):
        if np.allclose(np.round(ticks * 100, d), ticks * 100):
            return d
    return 3


def rate_with_ci_chart(labels: list[str], rates: list[float], ci_low: list[float], ci_high: list[float],
                       title: str, subtitle: str, note: str, decimals: int = 2,
                       value_fmt=None, tick_fmt=None, highlight: set[int] | None = None):
    """Horizontal bars (one series) with 95% CI whiskers.

    Values are fractions shown as percentages unless value_fmt / tick_fmt are given (e.g. dollars).
    Bars whose index is not in `highlight` (when given) are drawn in the de-emphasis grey.
    """
    apply_style()
    n = len(labels)
    fig, ax = figure(760, 96 + 48 * n + 130, plot_box=(150, 100, 90, 100))
    xmax = max(ci_high) * 1.12
    ax.set_xlim(0, xmax)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n), labels)
    ticks = nice_ticks(xmax)
    if tick_fmt is None:
        ax.set_xticks(ticks, [f"{t:.{_pct_decimals(ticks)}%}" for t in ticks])
    else:
        ax.set_xticks(ticks, [tick_fmt(t) for t in ticks])
    ax.grid(axis="x", zorder=0)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    for i, (r, lo, hi) in enumerate(zip(rates, ci_low, ci_high)):
        color = SERIES_1 if highlight is None or i in highlight else DEEMPHASIS
        rounded_hbar(ax, i, r, thickness_px=20, color=color)
        ax.plot([lo, hi], [i, i], color=INK_2, lw=px(1.5), solid_capstyle="butt", zorder=4)
        if value_fmt is None:
            label_decimals = decimals + 1 if r < 0.001 else decimals   # 0.008% rather than 0.01%
            text = f"{r:.{label_decimals}%}"
        else:
            text = value_fmt(r)
        ax.text(hi + xmax * 0.012, i, text, va="center", ha="left", fontsize=px(12), color=INK, zorder=5)
    header(fig, title, subtitle)
    footnote(fig, note)
    return fig


def gains_chart(curves: dict[str, tuple[np.ndarray, np.ndarray]], colors: dict[str, str],
                title: str, subtitle: str, note: str, x_max: float = 0.5, mark: float = 0.10,
                key_at: tuple[float, float] = (0.14, 0.44)):
    """Cumulative gains: share of future buyers captured vs share of visitors targeted.

    curves: {label: (x_share_targeted, y_share_captured)}; a random-targeting diagonal is drawn in grey.
    The curves converge on the right, so instead of end labels the values at `mark` are listed in a
    small key placed in the empty area under the curves (`key_at`, data coordinates).
    """
    apply_style()
    fig, ax = figure(760, 520, plot_box=(72, 132, 40, 92))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)
    xt = np.linspace(0, x_max, 6)
    ax.set_xticks(xt, [f"{t:.0%}" for t in xt])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="both", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_xlabel("First-time visitors targeted, highest scores first", fontsize=px(11), color=MUTED, labelpad=6)
    ax.plot([0, x_max], [0, x_max], color=AXIS, lw=px(1.5), zorder=2)
    ax.text(x_max * 0.985, x_max * 0.985 - 0.035, "Random targeting", ha="right", va="top",
            fontsize=px(11), color=MUTED)
    ax.axvline(mark, color=AXIS, lw=px(1), zorder=1)
    values = {}
    for label, (x, y) in curves.items():
        keep = x <= x_max
        ax.plot(x[keep], y[keep], color=colors[label], lw=px(2), solid_joinstyle="round",
                solid_capstyle="round", zorder=3)
        values[label] = float(np.interp(mark, x, y))
        dot(ax, mark, values[label], colors[label], diameter_px=8)
    # key: values at the mark, text in ink with a coloured dot beside it
    kx, ky = key_at
    ax.text(kx, ky, f"Buyers reached by targeting the top {mark:.0%}", ha="left", va="center",
            fontsize=px(11.5), color=INK_2, fontweight="bold")
    _, dy = _data_per_px(ax)
    for i, (label, v) in enumerate(sorted(values.items(), key=lambda kv: -kv[1])):
        yy = ky - (i + 1) * 22 * dy
        dot(ax, kx + 0.004, yy, colors[label], diameter_px=8)
        ax.text(kx + 0.012, yy, f"{v:.0%}  {label}", ha="left", va="center", fontsize=px(11.5), color=INK)
    header(fig, title, subtitle)
    # legend row under the subtitle (>= 2 series)
    w, h = fig.get_size_inches() * PX_PER_IN
    lx, ly = 24, 100
    for label, color in colors.items():
        fig.add_artist(plt.Line2D([(lx) / w, (lx + 16) / w], [1 - ly / h, 1 - ly / h], color=color,
                                  lw=px(2), transform=fig.transFigure, solid_capstyle="round"))
        t = fig.text((lx + 22) / w, 1 - ly / h, label, ha="left", va="center", fontsize=px(12), color=INK_2)
        fig.canvas.draw()
        lx += 22 + t.get_window_extent().width / (fig.dpi / PX_PER_IN) + 26
    footnote(fig, note)
    return fig


def cumulative_gains(y: np.ndarray, s: np.ndarray, points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-s, kind="stable")
    captured = np.cumsum(y[order]) / y.sum()
    share = np.arange(1, len(y) + 1) / len(y)
    idx = np.unique(np.linspace(0, len(y) - 1, points).astype(int))
    return np.r_[0, share[idx]], np.r_[0, captured[idx]]
