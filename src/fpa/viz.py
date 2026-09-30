"""Chart styling for the notebooks (matplotlib). Colors follow a validated, colorblind-safe
palette: categorical hues in a fixed order, a blue<->red diverging pair with a gray midpoint,
and reserved status colors that always ship with a label."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ROLE = {"actual": SERIES[0], "forecast": SERIES[1], "budget": SERIES[2]}
STATUS = {"Red": "#d03b3b", "Amber": "#fab219", "Green": "#0ca30c"}
DIVERGING = LinearSegmentedColormap.from_list("fav_unfav", ["#d03b3b", "#e34948", "#f0efec", "#3987e5", "#1c5cab"])


def _system_sans() -> str:
    """First installed system sans (matplotlib can't resolve CSS names like system-ui)."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Helvetica Neue", "Helvetica", "Segoe UI", "Arial", "Liberation Sans", "DejaVu Sans"):
        if name in installed:
            return name
    return "sans-serif"


def use_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "figure.figsize": (10, 4), "figure.dpi": 110,
            "font.family": _system_sans(),
            "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
            "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
            "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-",
            "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
            "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
            "legend.frameon": False, "legend.labelcolor": INK_2,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
        }
    )


def usd(x: float, _pos=None) -> str:
    a = abs(x)
    s = f"${a / 1e6:.1f}M" if a >= 1e6 else (f"${a / 1e3:.0f}k" if a >= 1e3 else f"${a:.0f}")
    return f"-{s}" if x < 0 else s


def usd_axis(ax, axis: str = "y") -> None:
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(usd))


def pct_axis(ax, axis: str = "y") -> None:
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))


def month_axis(ax, months: tuple[int, ...] = (1, 7)) -> None:
    """Readable monthly date ticks, e.g. Jan-25, Jul-25."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))


def end_label(ax, x, y, text: str, dy: float = 0) -> None:
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", color=INK_2, fontsize=9)


__all__ = ["DIVERGING", "ROLE", "SERIES", "STATUS", "end_label", "month_axis", "pct_axis", "plt", "usd", "usd_axis", "use_style"]
