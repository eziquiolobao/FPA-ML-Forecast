"""Shared data access, formatting and chart styling for the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MARTS = ROOT / "data" / "marts"
REPORTS = ROOT / "reports"
REPO_URL = "https://github.com/eziquiolobao/FPA-ML-Forecast"

STATUS_ICON = {"Red": "🔴", "Amber": "🟠", "Green": "🟢"}
STATUS_COLOR = {"Red": "#d03b3b", "Amber": "#fab219", "Green": "#0ca30c"}
STATUS_LABEL = {"Red": "Investigate", "Amber": "Explain / monitor", "Green": "On track"}

# Validated colorblind-safe categorical slots, stepped separately for light and dark surfaces
_LIGHT = {
    "actual": "#2a78d6", "forecast": "#eb6834", "budget": "#1baf7a",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "neutral": "#eef0ea", "band_alpha": 0.12,
}
_DARK = {
    "actual": "#3987e5", "forecast": "#d95926", "budget": "#199e70",
    "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
    "neutral": "#2a2a28", "band_alpha": 0.28,
}


def palette() -> dict:
    try:
        return _DARK if st.context.theme.type == "dark" else _LIGHT
    except Exception:
        return _LIGHT


def load(name: str) -> pd.DataFrame:
    path = MARTS / f"{name}.parquet"
    return _load(path, path.stat().st_mtime)  # mtime in the cache key: refreshed marts are picked up


@st.cache_data
def _load(path: Path, _mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in ("ds", "origin"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def run_metadata() -> dict:
    return json.loads((MARTS / "run_metadata.json").read_text())


def as_of() -> pd.Period:
    return pd.Period(run_metadata()["as_of"], "M")


def money(x: float, decimals: int = 1) -> str:
    if x is None or pd.isna(x):
        return "-"
    a = abs(x)
    s = f"${a / 1e6:.{decimals}f}M" if a >= 1e6 else (f"${a / 1e3:.0f}k" if a >= 1e3 else f"${a:,.0f}")
    return f"-{s}" if x < 0 else s


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def md(text: str) -> str:
    """Escape dollar signs so Streamlit markdown doesn't render them as LaTeX math."""
    return (text or "").replace("$", "\\$")


def signed_money(x: float) -> str:
    return ("+" if x >= 0 else "") + money(x)


def style_fig(fig: go.Figure, height: int = 360, legend: bool = True, hover: str = "x unified") -> go.Figure:
    p = palette()
    fig.update_layout(
        height=height,
        margin=dict(l=64, r=16, t=40 if legend else 12, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=p["ink2"], size=13),
        hovermode=hover,
        hoverlabel=dict(font_size=12),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=p["ink2"])),
    )
    fig.update_xaxes(showgrid=False, linecolor=p["axis"], tickcolor=p["axis"], tickfont=dict(color=p["ink2"]), automargin=True)
    fig.update_yaxes(gridcolor=p["grid"], gridwidth=1, zeroline=False, linecolor=p["axis"], tickfont=dict(color=p["ink2"]), automargin=True)
    return fig


def sidebar() -> None:
    meta = run_metadata()
    with st.sidebar:
        st.markdown(f"**{meta['company']}**  \nBooks closed through **{as_of().strftime('%B %Y')}**")
        st.caption(f"Pipeline last run {meta.get('last_run_utc', '-')}. Refreshed automatically after each month-end close.")
        st.caption("⚠️ Fictional company. All data is synthetic and generated for this portfolio project.")


def status_badge(sev: str) -> str:
    return f"{STATUS_ICON[sev]} {sev}"
