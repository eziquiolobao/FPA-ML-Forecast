"""Variance rules: materiality thresholds, favourability and RAG severity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RED, AMBER, GREEN = "Red", "Amber", "Green"


@dataclass(frozen=True)
class Rules:
    default: dict
    by_fs_line: dict
    ytd_multiplier: float
    surprise_level: int
    drift_months: int

    @classmethod
    def load(cls, path: Path) -> Rules:
        cfg = yaml.safe_load(path.read_text())
        return cls(
            default=cfg["default"],
            by_fs_line=cfg.get("by_fs_line", {}) or {},
            ytd_multiplier=float(cfg.get("ytd_multiplier", 2.0)),
            surprise_level=int(cfg.get("surprise_level", 95)),
            drift_months=int(cfg.get("drift_months", 3)),
        )

    def thresholds(self, fs_line: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        abs_usd = fs_line.map(lambda f: self.by_fs_line.get(f, {}).get("abs_usd", self.default["abs_usd"]))
        pct = fs_line.map(lambda f: self.by_fs_line.get(f, {}).get("pct", self.default["pct"]))
        return abs_usd.to_numpy(dtype=float), pct.to_numpy(dtype=float)


def favourable_sign(account_type: pd.Series) -> np.ndarray:
    """+1 where actual > comparison is good (revenue), -1 where it is bad (costs)."""
    return np.where(account_type == "Revenue", 1.0, -1.0)


def is_material(variance: np.ndarray, base: np.ndarray, abs_usd: np.ndarray, pct: np.ndarray) -> np.ndarray:
    """Material when BOTH the dollar and the percentage thresholds are breached."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_var = np.abs(variance) / np.abs(base)
    return (np.abs(variance) >= abs_usd) & (np.nan_to_num(pct_var, nan=np.inf) >= pct) & ~np.isnan(variance)


def outside_range(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return ((actual < lo) | (actual > hi)) & ~np.isnan(lo) & ~np.isnan(hi)


def drift_run_length(signed_breach: pd.Series) -> pd.Series:
    """Consecutive months (ending at each row) with a same-direction breach. Input: +1/-1/0."""
    run, prev, out = 0, 0, []
    for v in signed_breach:
        run = run + 1 if (v != 0 and v == prev) else (1 if v != 0 else 0)
        prev = v
        out.append(run)
    return pd.Series(out, index=signed_breach.index)


def severity(material: np.ndarray, surprise: np.ndarray, unexpected_minor: np.ndarray, drift: np.ndarray) -> np.ndarray:
    """Red   = material vs budget AND outside the expected range -> investigate first.
    Amber = material but expected (plan was off), OR unexpected but below materiality,
            OR drifting the same way for several months -> explain / monitor.
    Green = on track."""
    red = material & surprise
    amber = ~red & (material | unexpected_minor | drift)
    return np.where(red, RED, np.where(amber, AMBER, GREEN))
