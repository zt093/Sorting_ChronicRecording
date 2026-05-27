from __future__ import annotations

"""
Master renderer:
  - Scans a detector run folder
    (e.g. I:/threshold_crossings_outputs/threshold_crossings_run_YYYYMMDD_HHMMSS)
  - For every sgch/threshold subfolder (sgch*_thr*uV), loads peak-to-peak + firing-rate
    vs time series (per-chunk).
  - If per-metric CSVs exist, can reuse them to avoid recomputation.
  - Computes 5-min rolling mean with outlier skipping (same rule as the per-pair script).
  - Produces ONE master figure: amplitude and firing-rate stacked vertically,
    sharing a time axis aligned across all subplots.
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import LDA as lda_helpers


DATA_PATH = None  # Leave as None to enter the detector run folder at runtime.
MIN_SESSIONS_PER_UNIT = 120
BIN_SIZE_SECONDS = 60.0
MIN_MINUTES_PER_HOUR = 30
ANALYZER_FOLDER_NAME = "sorting_analyzer_analysis.zarr"
MAX_ALIGNED_UNITS = None  # Set to None to plot every aligned unit.

CHRONIC_REC_RE = re.compile(r"Chronic_Rec_(?P<ymd>\d{8})_(?P<hms>\d{6})")
CHUNK_RE = re.compile(r"chunk_(?P<idx>\d+)")


@dataclass(frozen=True)
class PairId:
    sg_ch: int
    threshold_uv: float

    def folder_tag(self) -> str:
        # Matches how the detector names folders: sgch{sg_ch}_thr{thr}uV
        thr_str = f"{self.threshold_uv:.3f}".rstrip("0").rstrip(".")
        return f"sgch{self.sg_ch}_thr{thr_str}uV"

    def sort_key(self) -> tuple[int, float]:
        return (self.sg_ch, self.threshold_uv)

    def display_label(self) -> str:
        return f"sgch{self.sg_ch}  thr{self.threshold_uv:g}uV"


def parse_pair_id_from_folder_name(folder_name: str) -> PairId | None:
    # Examples:
    #   sgch337_thr200uV
    #   sgch279_thr500uV
    #   sgch279_thr500p1uV (unlikely, but handle p->.)
    m = re.match(r"^sgch(?P<sg>\d+)_thr(?P<thr>.+)uV$", folder_name)
    if not m:
        return None
    sg = int(m.group("sg"))
    thr_str = m.group("thr")
    thr_str = thr_str.replace("p", ".")
    try:
        thr = float(thr_str)
    except ValueError:
        return None
    return PairId(sg_ch=sg, threshold_uv=thr)


@dataclass(frozen=True)
class UnitSeriesId:
    final_group_key: str
    final_unit_id: int | None
    shank_id: int | None
    local_channel_on_shank: int | None

    def folder_tag(self) -> str:
        unit_tag = "unit_na" if self.final_unit_id is None else f"unit_{self.final_unit_id}"
        return f"{unit_tag}_{safe_slug(self.final_group_key)}"

    def sort_key(self) -> tuple[int, str]:
        unit_sort = self.final_unit_id if self.final_unit_id is not None else 10**12
        return (int(unit_sort), self.final_group_key)

    def display_label(self) -> str:
        unit_label = "Unit NA" if self.final_unit_id is None else f"Unit {self.final_unit_id}"
        location_parts = []
        if self.shank_id is not None:
            location_parts.append(f"shank {self.shank_id}")
        if self.local_channel_on_shank is not None:
            location_parts.append(f"ch {self.local_channel_on_shank}")
        location = f" ({', '.join(location_parts)})" if location_parts else ""
        return f"{unit_label}{location}"


def parse_recording_start_datetime_from_name(name: str) -> datetime | None:
    m = CHRONIC_REC_RE.search(name)
    if not m:
        return None
    ymd = m.group("ymd")
    hms = m.group("hms")
    return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")


def datetime_to_x_label_5p5a(dt: datetime) -> str:
    bucket = "5p" if dt.hour >= 17 else "5a"
    return f"{dt:%m}_{dt:%d}_{bucket}"


def prompt_for_data_path(default_path: Path | None) -> Path:
    raw_value = input(
        "\nEnter the threshold_crossings_run folder for Tuning_Weinan "
        "(press Enter to use configured DATA_PATH): "
    ).strip().strip('"').strip("'")
    if raw_value:
        return Path(raw_value)
    if default_path is not None:
        return Path(default_path)
    raise ValueError("A data path is required.")


def safe_slug(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "value"


def shorten_slug(value: object, max_length: int = 48) -> str:
    slug = safe_slug(value)
    if len(slug) <= max_length:
        return slug
    return slug[:max_length].rstrip("_.-") or slug[:max_length]


def log_status(message: str) -> None:
    print(f"[Tuning_Weinan] {message}", flush=True)


lda_helpers.log_status = log_status


def rolling_mean_skip_outlier(xs_min: np.ndarray, ys: np.ndarray, window_min: float = 5.0) -> np.ndarray:
    """
    For each point i, consider points within +/- window_min/2 around xs_min[i].
    Compute a mean after removing the single outlier farthest from the median.
    """
    xs_min = np.asarray(xs_min, dtype=float)
    ys = np.asarray(ys, dtype=float)
    out = np.full(xs_min.shape, np.nan, dtype=float)
    half = window_min / 2.0

    finite = np.isfinite(xs_min) & np.isfinite(ys)
    if not np.any(finite):
        return out

    for i in range(xs_min.size):
        if not np.isfinite(xs_min[i]) or not np.isfinite(ys[i]):
            continue
        mask = finite & (np.abs(xs_min - xs_min[i]) <= half)
        ywin = ys[mask]
        ywin = ywin[np.isfinite(ywin)]
        if ywin.size == 0:
            continue
        if ywin.size <= 2:
            out[i] = float(np.mean(ywin))
            continue
        med = float(np.median(ywin))
        dev = np.abs(ywin - med)
        out_idx = int(np.argmax(dev))
        ykeep = np.delete(ywin, out_idx)
        out[i] = float(np.mean(ykeep)) if ykeep.size else np.nan

    return out


def _epoch_min_to_datetime(xs_min_epoch: np.ndarray) -> list[datetime | None]:
    out: list[datetime | None] = []
    for x in xs_min_epoch:
        if not np.isfinite(x):
            out.append(None)
        else:
            out.append(datetime.fromtimestamp(float(x) * 60.0))
    return out


def plot_daily_cycles(
    ax: plt.Axes,
    xs_min_epoch: np.ndarray,
    ys: np.ndarray,
    *,
    ylabel: str,
    title: str,
    show_5min_avg: bool = True,
    normalize_each_day: bool = False,
    ylim_mode: str = "default",
) -> dict[str, np.ndarray]:
    """
    Plot one 24h cycle on x (0..24), overlaying consecutive days, color-coded by day index.
    Uses the per-point datetime derived from epoch-minutes.
    """
    dts = _epoch_min_to_datetime(xs_min_epoch)

    # Group indices by date
    day_to_idx: dict[datetime.date, list[int]] = {}
    for i, dt in enumerate(dts):
        if dt is None or (not np.isfinite(ys[i])):
            continue
        day_to_idx.setdefault(dt.date(), []).append(i)

    if not day_to_idx:
        ax.text(0.5, 0.5, "No valid datetime points to plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        grid = np.linspace(0.0, 24.0, 289)
        return {
            "x_hour": grid.astype(np.float32),
            "daily_values": np.zeros((0, grid.size), dtype=np.float32),
            "mean": np.full(grid.shape, np.nan, dtype=np.float32),
            "std": np.full(grid.shape, np.nan, dtype=np.float32),
            "day_labels": np.asarray([], dtype="U10"),
            "day_center": np.asarray([], dtype=np.float32),
            "day_scale": np.asarray([], dtype=np.float32),
        }

    days_sorted = sorted(day_to_idx.keys())
    n_days = len(days_sorted)
    x_grid = np.linspace(0.0, 24.0, 289)
    daily_grid = np.full((len(days_sorted), x_grid.size), np.nan, dtype=np.float64)
    daily_center = np.full((len(days_sorted),), np.nan, dtype=np.float64)
    daily_scale = np.full((len(days_sorted),), np.nan, dtype=np.float64)

    for di, day in enumerate(days_sorted):
        idx = np.array(day_to_idx[day], dtype=int)
        idx = idx[np.argsort(idx)]
        dt_day = [dts[i] for i in idx]  # type: ignore[list-item]
        # x in hours-of-day
        x_hour = np.array([dt.hour + dt.minute / 60.0 + dt.second / 3600.0 for dt in dt_day], dtype=float)
        y_day = ys[idx].astype(float)

        # sort by x_hour in case indices aren't strictly increasing in time-of-day
        o = np.argsort(x_hour)
        x_hour = x_hour[o]
        y_day = y_day[o]

        y_for_grid = y_day
        if show_5min_avg and y_day.size >= 2:
            # 5-minute avg within the day (x axis in minutes)
            x_min = x_hour * 60.0
            y_for_grid = rolling_mean_skip_outlier(x_min, y_day, window_min=5.0)

        if normalize_each_day:
            stat_vals = y_for_grid[np.isfinite(y_for_grid)]
            if stat_vals.size == 0:
                center = np.nan
                scale = np.nan
            else:
                center = float(np.nanmean(stat_vals))
                scale = float(np.nanstd(stat_vals))
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
            daily_center[di] = center
            daily_scale[di] = scale
            y_day_plot = (y_day - center) / scale
            y_for_grid = (y_for_grid - center) / scale
        else:
            y_day_plot = y_day

        ax.plot(x_hour, y_day_plot, color="0.45", linewidth=0.65, alpha=0.22, zorder=1)
        ax.scatter(x_hour, y_day_plot, color="0.45", s=5, alpha=0.08, zorder=1)
        if show_5min_avg and y_day.size >= 2:
            ax.plot(x_hour, y_for_grid, color="0.30", linewidth=0.8, alpha=0.18, zorder=2)

        finite = np.isfinite(x_hour) & np.isfinite(y_for_grid)
        if np.count_nonzero(finite) >= 2:
            daily_grid[di, :] = np.interp(
                x_grid,
                x_hour[finite],
                y_for_grid[finite],
                left=np.nan,
                right=np.nan,
            )

    finite_counts = np.sum(np.isfinite(daily_grid), axis=0)
    mean = np.full(x_grid.shape, np.nan, dtype=np.float64)
    std = np.full(x_grid.shape, np.nan, dtype=np.float64)
    valid = finite_counts > 0
    if np.any(valid):
        mean[valid] = np.nanmean(daily_grid[:, valid], axis=0)
        std[valid] = np.nanstd(daily_grid[:, valid], axis=0)
        band_mask = np.isfinite(mean) & np.isfinite(std)
        ax.fill_between(
            x_grid,
            mean - std,
            mean + std,
            where=band_mask,
            color="k",
            alpha=0.16,
            linewidth=0,
            zorder=8,
            label="mean +/- std",
        )
        ax.plot(x_grid, mean, color="k", linewidth=2.4, zorder=9, label="mean")

    ax.set_xlim(0.0, 24.0)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.grid(True, alpha=0.25)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)

    # Clamp y-limits using distribution of the 5-min average curve.
    # mean(y5) ± 6*std(y5), ignoring NaNs.
    if ylim_mode == "mean3std":
        finite_mean = np.isfinite(mean)
        if np.any(finite_mean):
            mu = float(np.nanmean(mean[finite_mean]))
            sigma = float(np.nanstd(mean[finite_mean]))
            if sigma > 0:
                ax.set_ylim(mu - 3.0 * sigma, mu + 3.0 * sigma)
            else:
                ax.set_ylim(mu - 1.0, mu + 1.0)
    elif ylim_mode != "default":
        raise ValueError(f"Unknown ylim_mode: {ylim_mode}")

    # Small note about date range
    if n_days >= 1:
        ax.text(
            0.01,
            0.98,
            f"{days_sorted[0]} → {days_sorted[-1]}  (days={n_days})",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            color="0.25",
        )

    return {
        "x_hour": x_grid.astype(np.float32),
        "daily_values": daily_grid.astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "day_labels": np.asarray([str(d) for d in days_sorted], dtype="U10"),
        "day_center": daily_center.astype(np.float32),
        "day_scale": daily_scale.astype(np.float32),
    }


def compute_binned_daily_cycles(
    xs_min_epoch: np.ndarray,
    ys: np.ndarray,
    *,
    bin_hours: float,
    normalize_each_day: bool = False,
) -> dict[str, np.ndarray]:
    """
    Bin points by hour-of-day within each date, then compute across-day mean/std.

    If normalize_each_day is true, values are standardized within each day before
    binning so each day contributes its own z-scored profile.
    """
    if bin_hours <= 0:
        raise ValueError("bin_hours must be > 0")

    dts = _epoch_min_to_datetime(xs_min_epoch)
    day_to_idx: dict[datetime.date, list[int]] = {}
    for i, dt in enumerate(dts):
        if dt is None or not np.isfinite(ys[i]):
            continue
        day_to_idx.setdefault(dt.date(), []).append(i)

    edges = np.arange(0.0, 24.0 + float(bin_hours), float(bin_hours), dtype=np.float64)
    if edges[-1] > 24.0:
        edges[-1] = 24.0
    centers = 0.5 * (edges[:-1] + edges[1:])

    days_sorted = sorted(day_to_idx.keys())
    daily_bin_mean = np.full((len(days_sorted), centers.size), np.nan, dtype=np.float64)
    daily_bin_std = np.full((len(days_sorted), centers.size), np.nan, dtype=np.float64)
    daily_bin_count = np.zeros((len(days_sorted), centers.size), dtype=np.int32)
    daily_center = np.full((len(days_sorted),), np.nan, dtype=np.float64)
    daily_scale = np.full((len(days_sorted),), np.nan, dtype=np.float64)

    for di, day in enumerate(days_sorted):
        idx = np.array(day_to_idx[day], dtype=int)
        dt_day = [dts[i] for i in idx]  # type: ignore[list-item]
        x_hour = np.array(
            [dt.hour + dt.minute / 60.0 + dt.second / 3600.0 for dt in dt_day],
            dtype=np.float64,
        )
        y_day = ys[idx].astype(np.float64)
        finite = np.isfinite(x_hour) & np.isfinite(y_day)
        x_hour = x_hour[finite]
        y_day = y_day[finite]
        if y_day.size == 0:
            continue

        if normalize_each_day:
            center = float(np.nanmean(y_day))
            scale = float(np.nanstd(y_day))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            daily_center[di] = center
            daily_scale[di] = scale
            y_day = (y_day - center) / scale

        bin_idx = np.searchsorted(edges, x_hour, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, centers.size - 1)
        for bi in range(centers.size):
            vals = y_day[bin_idx == bi]
            vals = vals[np.isfinite(vals)]
            daily_bin_count[di, bi] = int(vals.size)
            if vals.size:
                daily_bin_mean[di, bi] = float(np.nanmean(vals))
                daily_bin_std[di, bi] = float(np.nanstd(vals))

    finite_counts = np.sum(np.isfinite(daily_bin_mean), axis=0)
    across_day_mean = np.full(centers.shape, np.nan, dtype=np.float64)
    across_day_std = np.full(centers.shape, np.nan, dtype=np.float64)
    valid = finite_counts > 0
    if np.any(valid):
        across_day_mean[valid] = np.nanmean(daily_bin_mean[:, valid], axis=0)
        across_day_std[valid] = np.nanstd(daily_bin_mean[:, valid], axis=0)

    return {
        "bin_edges_hour": edges.astype(np.float32),
        "bin_center_hour": centers.astype(np.float32),
        "daily_bin_mean": daily_bin_mean.astype(np.float32),
        "daily_bin_std": daily_bin_std.astype(np.float32),
        "daily_bin_count": daily_bin_count.astype(np.int32),
        "mean": across_day_mean.astype(np.float32),
        "std": across_day_std.astype(np.float32),
        "n_days_per_bin": finite_counts.astype(np.int32),
        "day_labels": np.asarray([str(d) for d in days_sorted], dtype="U10"),
        "day_center": daily_center.astype(np.float32),
        "day_scale": daily_scale.astype(np.float32),
    }


def plot_binned_daily_cycles(
    ax: plt.Axes,
    xs_min_epoch: np.ndarray,
    ys: np.ndarray,
    *,
    bin_hours: float,
    ylabel: str,
    title: str,
    normalize_each_day: bool = False,
) -> dict[str, np.ndarray]:
    plot_data = compute_binned_daily_cycles(
        xs_min_epoch,
        ys,
        bin_hours=bin_hours,
        normalize_each_day=normalize_each_day,
    )
    centers = plot_data["bin_center_hour"].astype(float)
    daily = plot_data["daily_bin_mean"].astype(float)
    mean = plot_data["mean"].astype(float)
    std = plot_data["std"].astype(float)

    if daily.shape[0] == 0:
        ax.text(0.5, 0.5, "No valid datetime points to bin", ha="center", va="center", transform=ax.transAxes)
    else:
        for row in daily:
            finite = np.isfinite(row)
            if np.any(finite):
                ax.plot(centers[finite], row[finite], color="0.45", linewidth=0.7, alpha=0.22, zorder=1)
                ax.scatter(centers[finite], row[finite], color="0.45", s=8, alpha=0.10, zorder=1)

        band_mask = np.isfinite(mean) & np.isfinite(std)
        if np.any(band_mask):
            ax.fill_between(
                centers,
                mean - std,
                mean + std,
                where=band_mask,
                color="k",
                alpha=0.16,
                linewidth=0,
                zorder=8,
                label="mean +/- std",
            )
            ax.plot(centers[band_mask], mean[band_mask], color="k", linewidth=2.4, zorder=9, label="mean")

    ax.set_xlim(0.0, 24.0)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.grid(True, alpha=0.25)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    return plot_data


def _hour_of_day_from_epoch_min(xs_min_epoch: np.ndarray) -> np.ndarray:
    hours = np.full(np.asarray(xs_min_epoch).shape, np.nan, dtype=np.float64)
    for i, x in enumerate(xs_min_epoch):
        if not np.isfinite(x):
            continue
        dt = datetime.fromtimestamp(float(x) * 60.0)
        hours[i] = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    return hours


def circular_vector_strength(hours: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """
    Activity-weighted circular vector strength on a 24 h clock.

    Returns (R, preferred_hour). R is 0..1; preferred_hour is in [0, 24).
    """
    hours = np.asarray(hours, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(hours) & np.isfinite(values)
    hours = hours[finite]
    values = values[finite]
    if values.size == 0:
        return float("nan"), float("nan")

    min_v = float(np.nanmin(values))
    weights = values - min_v if min_v < 0 else values.copy()
    if not np.isfinite(np.sum(weights)) or np.sum(weights) <= 0:
        weights = np.ones_like(values, dtype=np.float64)

    theta = 2.0 * np.pi * hours / 24.0
    z = np.sum(weights * np.exp(1j * theta)) / np.sum(weights)
    preferred_hour = (np.angle(z) % (2.0 * np.pi)) * 24.0 / (2.0 * np.pi)
    return float(np.abs(z)), float(preferred_hour)


def modulation_depth(values: np.ndarray) -> float:
    """
    Circadian modulation depth: (max - min) / mean for finite positive radial values.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    denom = float(np.nanmean(finite))
    if not np.isfinite(denom) or abs(denom) <= 1e-12:
        return float("nan")
    return float((np.nanmax(finite) - np.nanmin(finite)) / denom)


def bin_by_time_of_day(
    xs_min_epoch: np.ndarray,
    values: np.ndarray,
    *,
    bin_minutes: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bin_minutes <= 0:
        raise ValueError("bin_minutes must be > 0")
    hours = _hour_of_day_from_epoch_min(xs_min_epoch)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(hours) & np.isfinite(values)
    hours = hours[finite]
    values = values[finite]

    bin_hours = float(bin_minutes) / 60.0
    edges = np.arange(0.0, 24.0 + bin_hours, bin_hours, dtype=np.float64)
    if edges[-1] > 24.0:
        edges[-1] = 24.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(centers.shape, np.nan, dtype=np.float64)
    var = np.full(centers.shape, np.nan, dtype=np.float64)
    std = np.full(centers.shape, np.nan, dtype=np.float64)
    count = np.zeros(centers.shape, dtype=np.int32)

    if values.size:
        bin_idx = np.searchsorted(edges, hours, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, centers.size - 1)
        for bi in range(centers.size):
            vals = values[bin_idx == bi]
            vals = vals[np.isfinite(vals)]
            count[bi] = int(vals.size)
            if vals.size:
                mean[bi] = float(np.nanmean(vals))
                var[bi] = float(np.nanvar(vals))
                std[bi] = float(np.nanstd(vals))
    return centers, mean, var, std, count


def plot_polar_time_of_day(
    ax: plt.Axes,
    xs_min_epoch: np.ndarray,
    values: np.ndarray,
    *,
    bin_minutes: float,
    title: str,
    radial_label: str,
    radial_stat: str = "mean",
) -> dict[str, np.ndarray]:
    centers, mean, var, std, count = bin_by_time_of_day(xs_min_epoch, values, bin_minutes=bin_minutes)
    if radial_stat == "mean":
        radial = mean
        band_low = mean - std
        band_high = mean + std
        band_label = "mean +/- std"
    elif radial_stat == "variance":
        radial = var
        band_low = None
        band_high = None
        band_label = None
    else:
        raise ValueError(f"Unknown radial_stat: {radial_stat}")

    theta = 2.0 * np.pi * centers / 24.0
    finite = np.isfinite(radial)
    r_strength, preferred_hour = circular_vector_strength(centers, radial)
    mod_depth = modulation_depth(radial)

    if np.any(finite):
        theta_closed = np.r_[theta[finite], theta[finite][0]]
        radial_closed = np.r_[radial[finite], radial[finite][0]]
        ax.plot(theta_closed, radial_closed, color="k", linewidth=2.2)
        ax.scatter(theta[finite], radial[finite], s=14, color="tab:blue", alpha=0.75)
        if band_low is not None and band_high is not None:
            band_low_finite = band_low[finite]
            band_high_finite = band_high[finite]
            band_low_closed = np.r_[band_low_finite, band_low_finite[0]]
            band_high_closed = np.r_[band_high_finite, band_high_finite[0]]
            ax.fill_between(
                theta_closed,
                band_low_closed,
                band_high_closed,
                color="tab:blue",
                alpha=0.14,
                label=band_label,
            )

        if np.isfinite(preferred_hour):
            pref_theta = 2.0 * np.pi * preferred_hour / 24.0
            pref_r = float(np.nanmax(radial[finite]))
            ax.plot([pref_theta, pref_theta], [0.0, pref_r], color="crimson", linewidth=1.5, alpha=0.8)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(2.0 * np.pi * np.array([0, 3, 6, 9, 12, 15, 18, 21]) / 24.0)
    ax.set_xticklabels(["00", "03", "06", "09", "12", "15", "18", "21"])
    ax.set_title(
        f"{title} | {radial_label}\n"
        f"R={r_strength:.3f}, modulation depth={mod_depth:.3f}, preferred={preferred_hour:.2f} h",
        fontsize=10,
    )
    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.35)

    return {
        "bin_center_hour": centers.astype(np.float32),
        "mean": mean.astype(np.float32),
        "variance": var.astype(np.float32),
        "std": std.astype(np.float32),
        "count": count.astype(np.int32),
        "radial_statistic": np.asarray([radial_stat]),
        "vector_strength_R": np.asarray([r_strength], dtype=np.float32),
        "modulation_depth": np.asarray([mod_depth], dtype=np.float32),
        "preferred_hour": np.asarray([preferred_hour], dtype=np.float32),
    }


def render_polar_example_pair(pair_dir: Path, out_dir: Path | None = None) -> None:
    xs, peak, peak5, fr, labels = load_series_from_pair_dir(pair_dir)
    if out_dir is None:
        out_dir = pair_dir / "polar_time_of_day"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("peakToPeak", peak, "Peak-to-peak [uV]"),
        ("firingRate", fr, "Firing rate [Hz]"),
    ]
    bins = [
        ("1min", 1.0),
        ("1hr", 60.0),
        ("2hr", 120.0),
    ]

    polar_views = [
        ("mean", "mean", lambda label: label),
        ("variance", "variance", lambda label: f"Variance of {label}"),
    ]

    for metric_tag, values, radial_label in metrics:
        for view_tag, radial_stat, radial_label_fn in polar_views:
            view_radial_label = radial_label_fn(radial_label)
            fig, axes = plt.subplots(
                1,
                len(bins),
                figsize=(16, 5.4),
                subplot_kw={"projection": "polar"},
            )
            plot_data: dict[str, np.ndarray] = {}
            for ax, (bin_tag, bin_minutes) in zip(axes, bins):
                data = plot_polar_time_of_day(
                    ax,
                    xs,
                    values,
                    bin_minutes=bin_minutes,
                    title=f"{pair_dir.name} {metric_tag} {view_tag} {bin_tag}",
                    radial_label=view_radial_label,
                    radial_stat=radial_stat,
                )
                for key, value in data.items():
                    plot_data[f"{bin_tag}_{key}"] = value

            fig.suptitle(f"{pair_dir.name}: time-of-day polar tuning ({metric_tag} {view_tag})", fontsize=14)
            fig.tight_layout(rect=[0, 0.02, 1, 0.94])
            out_png = out_dir / f"{pair_dir.name}_polarTimeOfDay_{metric_tag}_{view_tag}_1min_1hr_2hr.png"
            fig.savefig(out_png, dpi=200)
            plt.close(fig)

            out_npz = out_dir / f"{pair_dir.name}_polarTimeOfDay_{metric_tag}_{view_tag}_1min_1hr_2hr_plotData.npz"
            np.savez_compressed(
                str(out_npz),
                source_pair_dir=np.asarray([str(pair_dir.resolve())]),
                metric=np.asarray([metric_tag]),
                radial_label=np.asarray([view_radial_label]),
                radial_statistic=np.asarray([radial_stat]),
                **plot_data,
            )
            print(f"Saved polar plot -> {out_png}")
            print(f"Saved polar data -> {out_npz}")

def render_polar_all_pairs(pair_meta: list[tuple[PairId, Path]], run_root: Path) -> None:
    out_root = run_root / "polar_time_of_day_units"
    out_root.mkdir(parents=True, exist_ok=True)
    for i, (_, pair_dir) in enumerate(pair_meta, start=1):
        unit_out_dir = out_root / pair_dir.name
        print(f"Rendering polar plots [{i}/{len(pair_meta)}] -> {unit_out_dir}", flush=True)
        render_polar_example_pair(pair_dir, out_dir=unit_out_dir)


def render_polar_series(
    series_name: str,
    xs: np.ndarray,
    amplitude: np.ndarray,
    firing_rate: np.ndarray,
    out_dir: Path,
    include_series_name_in_filename: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("peakToPeak", amplitude, "Peak-to-peak [uV]"),
        ("firingRate", firing_rate, "Firing rate [Hz]"),
    ]
    bins = [
        ("1min", 1.0),
        ("1hr", 60.0),
        ("2hr", 120.0),
    ]
    polar_views = [
        ("mean", "mean", lambda label: label),
        ("variance", "variance", lambda label: f"Variance of {label}"),
    ]

    for metric_tag, values, radial_label in metrics:
        for view_tag, radial_stat, radial_label_fn in polar_views:
            view_radial_label = radial_label_fn(radial_label)
            fig, axes = plt.subplots(
                1,
                len(bins),
                figsize=(16, 5.4),
                subplot_kw={"projection": "polar"},
            )
            plot_data: dict[str, np.ndarray] = {}
            for ax, (bin_tag, bin_minutes) in zip(axes, bins):
                data = plot_polar_time_of_day(
                    ax,
                    xs,
                    values,
                    bin_minutes=bin_minutes,
                    title=f"{series_name} {metric_tag} {view_tag} {bin_tag}",
                    radial_label=view_radial_label,
                    radial_stat=radial_stat,
                )
                for key, value in data.items():
                    plot_data[f"{bin_tag}_{key}"] = value

            fig.suptitle(f"{series_name}: time-of-day polar tuning ({metric_tag} {view_tag})", fontsize=14)
            fig.tight_layout(rect=[0, 0.02, 1, 0.94])
            file_prefix = (
                f"{shorten_slug(series_name)}_"
                if include_series_name_in_filename
                else ""
            )
            out_png = out_dir / f"{file_prefix}polarTimeOfDay_{metric_tag}_{view_tag}_1min_1hr_2hr.png"
            fig.savefig(out_png, dpi=200)
            plt.close(fig)

            out_npz = out_dir / f"{file_prefix}polarTimeOfDay_{metric_tag}_{view_tag}_1min_1hr_2hr_plotData.npz"
            np.savez_compressed(
                str(out_npz),
                source_pair_dir=np.asarray([str(out_dir.resolve())]),
                metric=np.asarray([metric_tag]),
                radial_label=np.asarray([view_radial_label]),
                radial_statistic=np.asarray([radial_stat]),
                **plot_data,
            )
            print(f"Saved polar plot -> {out_png}")
            print(f"Saved polar data -> {out_npz}")


def render_aligned_polar_all(
    pair_meta: list[tuple[UnitSeriesId, None]],
    series_cache: dict[UnitSeriesId, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]],
    run_root: Path,
) -> None:
    out_root = run_root / "polar_time_of_day_units"
    out_root.mkdir(parents=True, exist_ok=True)
    for i, (pid, _) in enumerate(pair_meta, start=1):
        series_name = pid.folder_tag()
        print(f"Rendering polar plots [{i}/{len(pair_meta)}] -> {out_root / series_name}", flush=True)
        xs, peak, peak5, fr, labels = series_cache[pid]
        render_polar_series(
            series_name,
            xs,
            peak,
            fr,
            out_root / series_name,
            include_series_name_in_filename=False,
        )


def find_aligned_polar_unit(
    pair_meta: list[tuple[UnitSeriesId, None]],
    polar_pair: str | None,
) -> UnitSeriesId:
    if polar_pair is None:
        return pair_meta[0][0]
    matches = [
        pid
        for pid, _ in pair_meta
        if polar_pair in {pid.folder_tag(), pid.final_group_key, str(pid.final_unit_id)}
    ]
    if not matches:
        raise RuntimeError(f"No aligned unit matching {polar_pair!r}")
    return matches[0]


def read_series_csv(csv_path: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """
    Returns:
      xs_min_epoch (float),
      labels (list[str]),
      ys (float),
      y5 (float)
    Supports:
      peakToPeak csv columns or firingRate csv columns.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        xs = []
        labels = []
        ys = []
        y5 = []
        for row in reader:
            xs.append(float(row["time_min_epoch"]))
            labels.append(row["dt_label"])
            if "peak_to_peak_uV" in row:
                ys.append(float(row["peak_to_peak_uV"]))
                y5.append(float(row["peak_to_peak_5min_avg_uV"]))
            else:
                ys.append(float(row["firing_rate_Hz"]))
                y5.append(float(row["firing_rate_5min_avg_Hz"]))
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    y5_arr = np.asarray(y5, dtype=float)
    return xs_arr, labels, ys_arr, y5_arr


def compute_peak2peak_and_firing_from_npz(npz_path: Path) -> tuple[float, float]:
    with np.load(str(npz_path), allow_pickle=False) as z:
        waveforms = z["waveforms_uv"]  # (n_events, wf_len)
        crossing_samples = z["crossing_samples"]
        t0 = z["time_start_sec"]
        t1 = z["time_end_sec"]

    n_cross = int(crossing_samples.shape[0])
    t0_s = float(t0[0]) if np.asarray(t0).shape else float(t0)
    t1_s = float(t1[0]) if np.asarray(t1).shape else float(t1)
    dur = max(1e-12, t1_s - t0_s)
    firing_rate_hz = n_cross / dur

    if waveforms.shape[0] == 0:
        return float("nan"), float(firing_rate_hz)

    mean_wf = waveforms.mean(axis=0)
    peak2peak_uv = float(np.max(mean_wf) - np.min(mean_wf))
    return peak2peak_uv, float(firing_rate_hz)


def load_series_from_pair_dir(pair_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
      xs_min_epoch,
      peak2peak_uV,
      peak2peak_5min_avg,
      firing_rate_Hz,
      labels
    """
    npz_paths = [p for p in pair_dir.rglob("*.npz") if p.name.endswith("_threshold_crossings.npz") and "_chunk_" in p.name]
    if not npz_paths:
        raise RuntimeError(f"No chunk npz files found under: {pair_dir}")

    rows = []
    labels = []
    for npz_path in npz_paths:
        base_name = npz_path.name
        dt0 = parse_recording_start_datetime_from_name(base_name)

        with np.load(str(npz_path), allow_pickle=False) as z:
            t_start_sec = z["time_start_sec"]
            t_start = float(t_start_sec[0]) if np.asarray(t_start_sec).shape else float(t_start_sec)

        peak2peak, firing_rate = compute_peak2peak_and_firing_from_npz(npz_path)

        if dt0 is None:
            # Fallback: per-file minutes without datetime.
            xs_min = t_start / 60.0
            dt_label = f"unknown_{xs_min:.1f}min"
        else:
            dt_chunk = dt0 + timedelta(seconds=t_start)
            xs_min = dt_chunk.timestamp() / 60.0
            dt_label = datetime_to_x_label_5p5a(dt_chunk)

        rows.append((xs_min, peak2peak, firing_rate))
        labels.append(dt_label)

    # Sort by x
    order = np.argsort([r[0] for r in rows])
    rows_sorted = [rows[i] for i in order]
    labels_sorted = [labels[i] for i in order]

    xs = np.asarray([r[0] for r in rows_sorted], dtype=float)
    peak2peak = np.asarray([r[1] for r in rows_sorted], dtype=float)
    firing = np.asarray([r[2] for r in rows_sorted], dtype=float)

    peak2peak_5min = rolling_mean_skip_outlier(xs, peak2peak, window_min=5.0)
    return xs, peak2peak, peak2peak_5min, firing, labels_sorted


def build_lda_config(data_path: Path) -> lda_helpers.Config:
    config = lda_helpers.Config()
    config.data_path = data_path
    config.bin_size_seconds = float(BIN_SIZE_SECONDS)
    config.min_sessions_per_unit = int(MIN_SESSIONS_PER_UNIT)
    config.min_minutes_per_hour = int(MIN_MINUTES_PER_HOUR)
    config.analyzer_folder_name = str(ANALYZER_FOLDER_NAME)
    config.apply_smoothing = False
    config.apply_zscore = False
    return config


def normalize_aligned_input_path(data_path: Path) -> Path:
    """Accept helper CSV reports by using the surrounding alignment summary folder."""
    resolved = Path(data_path)
    if resolved.is_file() and resolved.suffix.lower() == ".csv":
        parent = resolved.parent
        if parent.name.lower().startswith("alignment_days_summary"):
            log_status(f"Using alignment summary folder for CSV input: {parent}")
            return parent
        raise ValueError(
            f"CSV input is not an alignment export JSON: {resolved}. "
            "Point to the alignment_days_summary folder, export_summary.json, "
            "or export_summary_sg_*.json."
        )
    return resolved


def load_found_units_filter_keys(data_path: Path) -> set[str] | None:
    resolved = Path(data_path)
    if not (
        resolved.is_file()
        and resolved.suffix.lower() == ".csv"
        and resolved.name.lower().startswith("found_units_by_sg_channel_threshold")
    ):
        return None

    table = pd.read_csv(resolved)
    if "final_group_key" not in table.columns:
        raise ValueError(f"Found-units CSV is missing final_group_key column: {resolved}")
    keys = {
        str(value).strip()
        for value in table["final_group_key"].dropna().tolist()
        if str(value).strip()
    }
    if not keys:
        raise ValueError(f"Found-units CSV did not contain any final_group_key values: {resolved}")
    log_status(f"Using {len(keys)} final_group_key value(s) from found-units CSV as an aligned-unit filter.")
    return keys


def build_firing_amplitude_feature_table(selected_units: pd.DataFrame) -> pd.DataFrame:
    unit_table = (
        selected_units[["final_group_key", "final_unit_id", "shank_id", "local_channel_on_shank"]]
        .drop_duplicates(subset=["final_group_key"])
        .sort_values(["final_unit_id", "final_group_key"], na_position="last")
        .reset_index(drop=True)
    )
    rows: list[dict] = []
    for unit_row in unit_table.itertuples(index=False):
        group_key = str(unit_row.final_group_key)
        for feature_type in ("firing_rate_hz", "average_amplitude_uv"):
            rows.append(
                {
                    "feature_key": f"{group_key}__{feature_type}",
                    "feature_column": f"{group_key}__{feature_type}",
                    "final_group_key": group_key,
                    "final_unit_id": lda_helpers.safe_int(unit_row.final_unit_id),
                    "shank_id": lda_helpers.safe_int(unit_row.shank_id),
                    "local_channel_on_shank": lda_helpers.safe_int(unit_row.local_channel_on_shank),
                    "feature_type": feature_type,
                }
            )
    return pd.DataFrame(rows)


def numeric_column_1d(table: pd.DataFrame, column_name: str) -> np.ndarray:
    values = table[column_name]
    if isinstance(values, pd.DataFrame):
        log_status(
            f"Column {column_name!r} appeared {values.shape[1]} times; using the first occurrence."
        )
        values = values.iloc[:, 0]
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def build_firing_amplitude_minute_vectors(
    selected_units: pd.DataFrame,
    session_table: pd.DataFrame,
    analyzers: dict[str, object],
    config: lda_helpers.Config,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    feature_table = build_firing_amplitude_feature_table(selected_units)
    feature_keys = feature_table["feature_key"].astype(str).tolist()
    feature_index = {key: index for index, key in enumerate(feature_keys)}
    members_by_session = {
        session_key: table.copy()
        for session_key, table in selected_units.groupby("session_key", sort=False)
    }

    samples: list[np.ndarray] = []
    metadata_rows: list[dict] = []
    total_sessions = len(session_table)
    for session_position, session_row in enumerate(session_table.itertuples(index=False), start=1):
        session_key = str(session_row.session_key)
        session_name = str(session_row.session_name)
        log_status(
            f"Binning firing/amplitude at {config.bin_size_seconds:.0f}s resolution for "
            f"session {session_position} / {total_sessions}: {session_name}"
        )
        analyzer = analyzers[session_key]
        spike_amplitudes_by_unit = lda_helpers.get_session_spike_amplitudes_by_unit(analyzer)
        valid_unit_ids = {int(unit_id) for unit_id in analyzer.sorting.get_unit_ids()}
        sampling_frequency = float(analyzer.sorting.get_sampling_frequency())
        try:
            session_duration_s = float(analyzer.recording.get_num_frames()) / float(
                analyzer.recording.get_sampling_frequency()
            )
        except Exception:
            all_spike_trains = []
            for unit_id in analyzer.sorting.get_unit_ids():
                spike_train = analyzer.sorting.get_unit_spike_train(unit_id=int(unit_id), segment_index=0)
                if len(spike_train) > 0:
                    all_spike_trains.append(float(spike_train[-1]) / sampling_frequency)
            session_duration_s = max(all_spike_trains) if all_spike_trains else 0.0

        n_complete_bins = int(session_duration_s // config.bin_size_seconds)
        if n_complete_bins < 1:
            continue

        bin_edges = np.arange(n_complete_bins + 1, dtype=float) * config.bin_size_seconds
        session_matrix = np.full((len(bin_edges) - 1, len(feature_keys)), np.nan, dtype=float)
        session_units = members_by_session.get(session_key, pd.DataFrame())

        for member_row in session_units.itertuples(index=False):
            unit_id = int(member_row.unit_id)
            if unit_id not in valid_unit_ids:
                continue
            group_key = str(member_row.final_group_key)
            spike_train_samples = analyzer.sorting.get_unit_spike_train(
                unit_id=unit_id,
                segment_index=0,
            )
            spike_times_s = np.asarray(spike_train_samples, dtype=float) / sampling_frequency
            counts, _ = np.histogram(spike_times_s, bins=bin_edges)
            n_bins = len(bin_edges) - 1
            bin_indices = np.searchsorted(bin_edges, spike_times_s, side="right") - 1
            bin_indices[spike_times_s == bin_edges[-1]] = n_bins - 1

            rate_feature_key = f"{group_key}__firing_rate_hz"
            amplitude_feature_key = f"{group_key}__average_amplitude_uv"
            session_matrix[:, feature_index[rate_feature_key]] = counts.astype(float) / config.bin_size_seconds
            session_matrix[:, feature_index[amplitude_feature_key]] = lda_helpers.binned_mean_abs_amplitude(
                spike_train_samples=np.asarray(spike_train_samples, dtype=float),
                spike_amplitudes=spike_amplitudes_by_unit.get(unit_id),
                bin_indices=bin_indices,
                n_bins=n_bins,
            )

        bin_centers = bin_edges[:-1] + config.bin_size_seconds / 2.0
        session_start_datetime = session_row.session_start_datetime
        for bin_index, bin_center_s in enumerate(bin_centers):
            bin_start_sec = float(bin_edges[bin_index])
            bin_end_sec = float(bin_edges[bin_index + 1])
            bin_start_datetime = session_start_datetime + timedelta(seconds=bin_start_sec)
            bin_end_datetime = session_start_datetime + timedelta(seconds=bin_end_sec)
            samples.append(session_matrix[bin_index])
            metadata_rows.append(
                {
                    "session_id": int(session_row.session_id),
                    "session_key": session_key,
                    "session_name": session_name,
                    "session_name_normalized": str(session_row.session_name_normalized),
                    "session_index": lda_helpers.safe_int(session_row.session_index),
                    "session_start_datetime": session_start_datetime.isoformat(sep=" "),
                    "minute_bin_index": int(bin_index),
                    "minute_start_sec": bin_start_sec,
                    "minute_end_sec": bin_end_sec,
                    "minute_center_s": float(bin_center_s),
                    "session_duration_s": float(session_duration_s),
                    "minute_start_datetime": bin_start_datetime.isoformat(sep=" "),
                    "minute_end_datetime": bin_end_datetime.isoformat(sep=" "),
                    "clock_hour_of_day": int(bin_start_datetime.hour),
                    "clock_minute_of_hour": int(bin_start_datetime.minute),
                    "calendar_day": bin_start_datetime.date().isoformat(),
                }
            )

    if not samples:
        raise RuntimeError("No firing/amplitude vectors were created. Check the sessions and bin size.")

    population_matrix = np.vstack(samples)
    metadata_table = pd.DataFrame(metadata_rows)
    log_status(
        f"Finished firing/amplitude binning: created {population_matrix.shape[0]} samples "
        f"x {population_matrix.shape[1]} columns across {len(session_table)} sessions"
    )
    return population_matrix, metadata_table, feature_table


def load_aligned_minute_data_like_tuning(
    data_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    found_units_filter_keys = load_found_units_filter_keys(data_path)
    aligned_input_path = normalize_aligned_input_path(data_path)
    config = build_lda_config(aligned_input_path)
    export_summary_path = lda_helpers.resolve_export_summary_path(aligned_input_path)
    log_status(f"Loading alignment export: {export_summary_path}")
    try:
        export_payload = lda_helpers.load_export_summary(export_summary_path)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Tuning_Weinan selected the Tuning.py-style alignment input path, "
            f"but the resolved JSON file could not be parsed: {export_summary_path}. "
            "If this is threshold-crossings data, point to a threshold_crossings_run folder "
            "or run with --input-mode threshold. If this is aligned data, use the same "
            "alignment export path that works in Tuning.py."
        ) from exc

    session_table = lda_helpers.build_session_table(export_payload=export_payload, config=config)
    log_status(f"Resolved {len(session_table)} sessions with real clock starts")
    analyzers, resolved_output_folders = lda_helpers.load_session_analyzers(
        session_table=session_table,
        config=config,
    )
    session_table = session_table.copy()
    session_table["resolved_output_folder"] = session_table["session_key"].map(resolved_output_folders)

    selected_units = lda_helpers.select_good_unit_groups(
        export_payload=export_payload,
        config=config,
        analyzers=analyzers,
    )
    if found_units_filter_keys is not None:
        before_count = int(selected_units["final_group_key"].nunique())
        selected_units = selected_units[
            selected_units["final_group_key"].astype(str).isin(found_units_filter_keys)
        ].copy()
        after_count = int(selected_units["final_group_key"].nunique())
        log_status(
            f"Applied found-units CSV filter: kept {after_count} / {before_count} "
            "selected aligned unit groups."
        )
        if selected_units.empty:
            raise RuntimeError(
                "No selected aligned units matched final_group_key values from the found-units CSV. "
                "Check MIN_SESSIONS_PER_UNIT or whether the CSV belongs to this alignment export."
            )
    log_status(
        f"Selected {selected_units['final_group_key'].nunique()} aligned unit groups "
        f"using MIN_SESSIONS_PER_UNIT={MIN_SESSIONS_PER_UNIT}"
    )

    minute_matrix, minute_metadata, feature_table = build_firing_amplitude_minute_vectors(
        selected_units=selected_units,
        session_table=session_table,
        analyzers=analyzers,
        config=config,
    )
    feature_columns = feature_table["feature_column"].astype(str).tolist()
    log_status(
        f"Materializing aligned matrix with {minute_matrix.shape[0]} bins x "
        f"{minute_matrix.shape[1]} firing-rate/amplitude columns."
    )
    minute_values = pd.DataFrame(minute_matrix, columns=feature_columns)
    minute_wide = pd.concat(
        [minute_metadata.reset_index(drop=True), minute_values.reset_index(drop=True)],
        axis=1,
    )
    return minute_wide, feature_table, selected_units, export_summary_path


def datetime_series_to_epoch_minutes(values: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(values, errors="coerce")
    xs = np.full(parsed.shape[0], np.nan, dtype=float)
    for index, value in enumerate(parsed):
        if pd.isna(value):
            continue
        xs[index] = value.to_pydatetime().timestamp() / 60.0
    return xs


def write_aligned_unit_usage_summary(
    selected_units: pd.DataFrame,
    output_root: Path,
    input_path: Path,
    export_summary_path: Path,
) -> None:
    rows: list[dict] = []
    for group_key, group_table in selected_units.groupby("final_group_key", sort=True):
        session_names = sorted(group_table["session_name"].astype(str).unique().tolist())
        session_keys = sorted(group_table["session_key"].astype(str).unique().tolist())
        unit_ids_by_session = " | ".join(
            f"{session}:{','.join(str(value) for value in sorted(table['unit_id'].dropna().astype(int).unique().tolist()))}"
            for session, table in group_table.groupby("session_name", sort=True)
        )
        first_row = group_table.iloc[0]
        rows.append(
            {
                "final_group_key": str(group_key),
                "final_unit_id": lda_helpers.safe_int(first_row.get("final_unit_id")),
                "shank_id": lda_helpers.safe_int(first_row.get("shank_id")),
                "local_channel_on_shank": lda_helpers.safe_int(first_row.get("local_channel_on_shank")),
                "n_session_names": int(len(session_names)),
                "n_session_keys": int(len(session_keys)),
                "n_member_rows": int(len(group_table)),
                "sessions": "; ".join(session_names),
                "session_keys": "; ".join(session_keys),
                "unit_ids_by_session": unit_ids_by_session,
            }
        )

    summary_table = pd.DataFrame(rows).sort_values(
        ["final_unit_id", "final_group_key"],
        na_position="last",
    )
    summary_csv = output_root / "tuning_weinan_units_used_summary.csv"
    summary_json = output_root / "tuning_weinan_units_used_summary.json"
    summary_table.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "export_summary_path": str(export_summary_path),
                "n_units": int(len(summary_table)),
                "n_unique_session_names": int(selected_units["session_name"].astype(str).nunique()),
                "session_names": sorted(selected_units["session_name"].astype(str).unique().tolist()),
                "units": summary_table.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log_status(f"Saved aligned unit usage summary -> {summary_csv}")


def build_aligned_series_cache(
    data_path: Path,
    max_units: int | None = MAX_ALIGNED_UNITS,
) -> tuple[Path, list[tuple[UnitSeriesId, None]], dict[UnitSeriesId, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]]]:
    minute_wide, feature_table, selected_units, export_summary_path = load_aligned_minute_data_like_tuning(data_path)
    output_parent = export_summary_path if export_summary_path.is_dir() else export_summary_path.parent
    output_root = output_parent / "tuning_weinan_aligned_units"
    output_root.mkdir(parents=True, exist_ok=True)
    write_aligned_unit_usage_summary(
        selected_units=selected_units,
        output_root=output_root,
        input_path=Path(data_path),
        export_summary_path=export_summary_path,
    )

    required_metadata = {"minute_start_datetime"}
    missing_metadata = sorted(required_metadata - set(minute_wide.columns))
    if missing_metadata:
        raise KeyError(f"Aligned data is missing required metadata columns: {missing_metadata}")

    feature_lookup: dict[tuple[str, str], pd.Series] = {}
    for row in feature_table.itertuples(index=False):
        feature_lookup[(str(row.final_group_key), str(row.feature_type))] = pd.Series(row._asdict())

    unit_rows = (
        feature_table[["final_group_key", "final_unit_id", "shank_id", "local_channel_on_shank"]]
        .drop_duplicates(subset=["final_group_key"])
        .sort_values(["final_unit_id", "final_group_key"], na_position="last")
    )
    if max_units is not None and max_units > 0 and len(unit_rows) > max_units:
        log_status(
            f"Limiting aligned plotting to first {max_units} / {len(unit_rows)} units. "
            "Use --max-aligned-units 0 to plot all units."
        )
        unit_rows = unit_rows.head(max_units)

    log_status(
        f"Converting aligned minute matrix to Weinan series for {len(unit_rows)} unit(s) "
        f"and {len(minute_wide)} minute bins."
    )
    xs_base = datetime_series_to_epoch_minutes(minute_wide["minute_start_datetime"])
    labels = []
    for value in pd.to_datetime(minute_wide["minute_start_datetime"], errors="coerce"):
        if pd.isna(value):
            labels.append("unknown")
        else:
            labels.append(datetime_to_x_label_5p5a(value.to_pydatetime()))

    pair_meta: list[tuple[UnitSeriesId, None]] = []
    series_cache: dict[UnitSeriesId, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}

    total_unit_rows = len(unit_rows)
    for unit_index, row in enumerate(unit_rows.itertuples(index=False), start=1):
        if unit_index == 1 or unit_index % 25 == 0 or unit_index == total_unit_rows:
            log_status(f"Preparing aligned unit series {unit_index} / {total_unit_rows}")
        group_key = str(row.final_group_key)
        amplitude_row = feature_lookup.get((group_key, "average_amplitude_uv"))
        firing_row = feature_lookup.get((group_key, "firing_rate_hz"))
        if firing_row is None:
            continue

        firing_column = str(firing_row["feature_column"])
        if firing_column not in minute_wide.columns:
            continue

        if amplitude_row is not None and str(amplitude_row["feature_column"]) in minute_wide.columns:
            amplitude = numeric_column_1d(minute_wide, str(amplitude_row["feature_column"]))
        else:
            amplitude = np.full(xs_base.shape, np.nan, dtype=float)

        firing = numeric_column_1d(minute_wide, firing_column)
        finite_mask = np.isfinite(xs_base) & (np.isfinite(amplitude) | np.isfinite(firing))
        if not np.any(finite_mask):
            continue

        series_id = UnitSeriesId(
            final_group_key=group_key,
            final_unit_id=lda_helpers.safe_int(row.final_unit_id),
            shank_id=lda_helpers.safe_int(row.shank_id),
            local_channel_on_shank=lda_helpers.safe_int(row.local_channel_on_shank),
        )
        xs = xs_base[finite_mask]
        peak = amplitude[finite_mask]
        fr = firing[finite_mask]
        selected_labels = [label for label, keep in zip(labels, finite_mask) if keep]
        order = np.argsort(xs)
        xs = xs[order]
        peak = peak[order]
        fr = fr[order]
        selected_labels = [selected_labels[int(index)] for index in order]
        peak5 = rolling_mean_skip_outlier(xs, peak, window_min=5.0)

        pair_meta.append((series_id, None))
        series_cache[series_id] = (xs, peak, peak5, fr, selected_labels)

    if not pair_meta:
        raise RuntimeError("No aligned units with firing_rate_hz data were found in the Tuning.py-style input.")

    pair_meta.sort(key=lambda item: item[0].sort_key())
    log_status(f"Prepared {len(pair_meta)} aligned-unit series for Weinan plots")
    return output_root, pair_meta, series_cache


def discover_threshold_pair_meta(run_root: Path, recursive: bool = True) -> list[tuple[PairId, Path]]:
    if not run_root.is_dir():
        return []
    candidate_dirs = [run_root, *run_root.rglob("*")] if recursive else list(run_root.iterdir())
    pair_dirs = sorted(
        p
        for p in candidate_dirs
        if p.is_dir() and p.name.startswith("sgch") and "_thr" in p.name
        and "polar_time_of_day_units" not in set(p.relative_to(run_root).parts)
    )
    pair_meta: list[tuple[PairId, Path]] = []
    for p in pair_dirs:
        pid = parse_pair_id_from_folder_name(p.name)
        if pid is not None:
            pair_meta.append((pid, p))
    pair_meta.sort(key=lambda t: t[0].sort_key())
    return pair_meta


def write_threshold_unit_usage_summary(
    pair_meta: list[tuple[PairId, Path]],
    series_cache: dict[PairId, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]],
    run_root: Path,
) -> None:
    rows: list[dict] = []
    for pid, pair_dir in pair_meta:
        xs, peak, peak5, fr, labels = series_cache[pid]
        datetimes = _epoch_min_to_datetime(xs)
        session_dates = sorted({dt.date().isoformat() for dt in datetimes if dt is not None})
        label_values = sorted({str(label) for label in labels if str(label)})
        rows.append(
            {
                "series_name": pair_dir.name,
                "sg_ch": int(pid.sg_ch),
                "threshold_uv": float(pid.threshold_uv),
                "pair_dir": str(pair_dir),
                "n_points": int(len(xs)),
                "n_session_dates": int(len(session_dates)),
                "session_dates": "; ".join(session_dates),
                "n_labels": int(len(label_values)),
                "labels": "; ".join(label_values),
            }
        )

    summary_table = pd.DataFrame(rows).sort_values(["sg_ch", "threshold_uv"])
    summary_csv = run_root / "tuning_weinan_units_used_summary.csv"
    summary_json = run_root / "tuning_weinan_units_used_summary.json"
    summary_table.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "input_path": str(run_root),
                "n_series": int(len(summary_table)),
                "n_unique_session_dates": int(
                    len(
                        {
                            date
                            for row in rows
                            for date in str(row["session_dates"]).split("; ")
                            if date
                        }
                    )
                ),
                "series": summary_table.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log_status(f"Saved threshold unit usage summary -> {summary_csv}")


def aligned_amplitude_ylabel(normalize_each_day: bool) -> str:
    return "Average amplitude day z-score" if normalize_each_day else "Average amplitude [uV]"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a master daily-cycle peak-to-peak/firing-rate plot from a threshold_crossings_run_* folder."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="threshold_crossings_run_* folder. If omitted, prompts interactively and can use DATA_PATH.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Folder for Tuning_Weinan outputs. Defaults to --run-root for standalone behavior.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("auto", "threshold", "aligned"),
        default="auto",
        help="Input format: threshold crossings folders, Tuning.py-style aligned export, or auto-detect.",
    )
    parser.add_argument(
        "--reuse-csv",
        action="store_true",
        help="Reuse existing peakToPeak/firingRate CSVs when present.",
    )
    parser.add_argument(
        "--no-reuse-csv",
        action="store_true",
        help="Recompute series from NPZ files even if CSVs exist.",
    )
    parser.add_argument(
        "--render-polar-example",
        action="store_true",
        help="Render polar time-of-day plots for one sgch*_thr*uV pair folder.",
    )
    parser.add_argument(
        "--render-polar-all",
        action="store_true",
        help="Render polar time-of-day plots for all series. This is now the default unless --render-polar-example is used.",
    )
    parser.add_argument(
        "--polar-pair",
        default=None,
        help="Pair folder name for --render-polar-example, e.g. sgch279_thr180uV. Defaults to the first pair.",
    )
    parser.add_argument(
        "--only-polar",
        action="store_true",
        help="When rendering a polar example, skip the master daily-cycle figures.",
    )
    parser.add_argument(
        "--max-aligned-units",
        type=int,
        default=MAX_ALIGNED_UNITS,
        help="Maximum aligned units to plot for Tuning.py-style input. Use 0 to plot all.",
    )
    args = parser.parse_args()

    input_path = prompt_for_data_path(DATA_PATH) if args.run_root is None else Path(args.run_root)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    run_root = input_path
    output_root = run_root if args.output_root is None else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        "Master plot x-axis mode: overlay 24h cycles (0–24 hours) and color-code consecutive days.",
        flush=True,
    )

    if args.reuse_csv and args.no_reuse_csv:
        raise ValueError("Choose only one of --reuse-csv or --no-reuse-csv.")
    reuse_amp = True
    reuse_fr = True

    # Discover pair folders. If none are present, fall through to the Tuning.py-style alignment input.
    pair_meta = [] if args.input_mode == "aligned" else discover_threshold_pair_meta(run_root)
    loaded_from_aligned = False
    series_cache = None
    if args.input_mode == "threshold" and not pair_meta:
        raise RuntimeError(f"No sgch*_thr*uV folders found under threshold input path: {run_root}")
    if args.input_mode == "aligned" or not pair_meta:
        max_aligned_units = None if args.max_aligned_units == 0 else args.max_aligned_units
        run_root, pair_meta, series_cache = build_aligned_series_cache(
            input_path,
            max_units=max_aligned_units,
        )
        loaded_from_aligned = True
    elif args.only_polar:
        reuse_amp = True
        reuse_fr = True
    elif args.reuse_csv:
        reuse_amp = True
        reuse_fr = True
    elif args.no_reuse_csv:
        reuse_amp = False
        reuse_fr = False
    else:
        reuse_amp = input("Reuse existing peakToPeak CSV if present? [Y/n] (default Y): ").strip().lower()
        reuse_fr = input("Reuse existing firingRate CSV if present? [Y/n] (default Y): ").strip().lower()
        reuse_amp = reuse_amp in ("", "y", "yes")
        reuse_fr = reuse_fr in ("", "y", "yes")

    print(
        f"Master plot input mode: {'aligned Tuning.py export' if loaded_from_aligned else 'threshold crossings run'}.",
        flush=True,
    )

    pair_meta.sort(key=lambda t: t[0].sort_key())

    should_render_polar_all = args.render_polar_all or not args.render_polar_example
    if loaded_from_aligned and should_render_polar_all:
        render_aligned_polar_all(pair_meta, series_cache, output_root)
        if args.only_polar:
            return 0
    elif should_render_polar_all:
        render_polar_all_pairs(pair_meta, output_root)
        if args.only_polar:
            return 0

    if loaded_from_aligned and args.render_polar_example:
        polar_pid = find_aligned_polar_unit(pair_meta, args.polar_pair)
        xs, peak, peak5, fr, labels = series_cache[polar_pid]
        render_polar_series(
            polar_pid.folder_tag(),
            xs,
            peak,
            fr,
            output_root / "polar_time_of_day_units" / polar_pid.folder_tag(),
            include_series_name_in_filename=False,
        )
        if args.only_polar:
            return 0
    elif args.render_polar_example:
        if args.polar_pair is None:
            polar_pair_dir = pair_meta[0][1]
        else:
            matches = [pdir for _, pdir in pair_meta if pdir.name == args.polar_pair]
            if not matches:
                raise RuntimeError(f"No pair folder named {args.polar_pair!r} under: {run_root}")
            polar_pair_dir = matches[0]
        render_polar_example_pair(
            polar_pair_dir,
            out_dir=output_root / "polar_time_of_day_units" / polar_pair_dir.name,
        )
        if args.only_polar:
            return 0

    all_x_min: list[float] = []
    if series_cache is None:
        series_cache = {}

    for pid, pdir in ([] if loaded_from_aligned else pair_meta):
        # Optional CSV reuse
        peak_csv = next(pdir.rglob("*peakToPeak_vs_time_*.csv"), None)
        firing_csv = next(pdir.rglob("*firingRate_vs_time_*.csv"), None)

        xs = peak = peak5 = fr = labels = None

        if reuse_amp and peak_csv is not None and reuse_fr and firing_csv is not None:
            xs, labels, peak, peak5 = read_series_csv(peak_csv)
            xs2, labels2, fr, fr5 = read_series_csv(firing_csv)
            # xs should match; if not, align by index (fast) or recompute (safe).
            if xs.shape == xs2.shape and np.allclose(xs, xs2, atol=1e-6, rtol=0):
                labels = labels
            else:
                # If mismatch, compute from NPZ for correctness.
                xs, peak, peak5, fr, labels = load_series_from_pair_dir(pdir)
        else:
            xs, peak, peak5, fr, labels = load_series_from_pair_dir(pdir)

        series_cache[pid] = (xs, peak, peak5, fr, labels)
        finite_mask = np.isfinite(xs)
        if np.any(finite_mask):
            all_x_min.extend(xs[finite_mask].tolist())

    if not loaded_from_aligned:
        write_threshold_unit_usage_summary(
            pair_meta=pair_meta,
            series_cache=series_cache,
            run_root=output_root,
        )

    label_tag = "dailyCycleOverlay"
    variants = [
        ("raw_defaultYlim", False, "default"),
        ("raw_meanTrace3stdYlim", False, "mean3std"),
        ("dayZ_defaultYlim", True, "default"),
        ("dayZ_meanTrace3stdYlim", True, "mean3std"),
    ]

    n_pairs = len(pair_meta)
    nrows = 2 * n_pairs
    fig_w = 14
    fig_h = max(6, 2.6 * nrows)
    log_status(f"Rendering {len(variants)} daily-cycle overlay figure variant(s) for {n_pairs} series.")

    for variant_tag, normalize_each_day, ylim_mode in variants:
        log_status(f"Rendering daily-cycle overlay variant: {variant_tag}")
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(fig_w, fig_h), sharex=True)
        if nrows == 1:
            axes = [axes]

        plot_data: dict[str, np.ndarray] = {}

        for row_i, (pid, _) in enumerate(pair_meta):
            xs, peak, peak5, fr, labels = series_cache[pid]

            ax_p2p = axes[2 * row_i]
            ax_fr = axes[2 * row_i + 1]

            tag = pid.display_label()
            amplitude_name = "average amplitude" if loaded_from_aligned else "peak-to-peak"
            p2p_ylabel = (
                "Average amplitude day z-score"
                if loaded_from_aligned and normalize_each_day
                else "Average amplitude [uV]"
                if loaded_from_aligned
                else "Peak-to-peak day z-score"
                if normalize_each_day
                else "Peak-to-peak [uV]"
            )
            fr_ylabel = "Firing-rate day z-score" if normalize_each_day else "Firing rate [Hz]"
            norm_note = "day-normalized" if normalize_each_day else "raw"
            ylim_note = "mean trace +/- 3 std y-limit" if ylim_mode == "mean3std" else "default y-limit"

            peak_plot_data = plot_daily_cycles(
                ax_p2p,
                xs,
                peak,
                ylabel=p2p_ylabel,
                title=f"{tag}  |  {amplitude_name}  |  {norm_note}, {ylim_note}",
                show_5min_avg=True,
                normalize_each_day=normalize_each_day,
                ylim_mode=ylim_mode,
            )

            fr_plot_data = plot_daily_cycles(
                ax_fr,
                xs,
                fr,
                ylabel=fr_ylabel,
                title=f"{tag}  |  firing rate  |  {norm_note}, {ylim_note}",
                show_5min_avg=True,
                normalize_each_day=normalize_each_day,
                ylim_mode=ylim_mode,
            )
            safe_tag = pid.folder_tag().replace(".", "p")
            for key, value in peak_plot_data.items():
                plot_data[f"{safe_tag}_peak_{key}"] = value
            for key, value in fr_plot_data.items():
                plot_data[f"{safe_tag}_firing_{key}"] = value

        axes[-1].set_xlabel("Hour of day (0-24), consecutive days overlaid")
        fig.suptitle(
            f"Master peak-to-peak & firing rate - 24h cycle overlay ({variant_tag})",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0.01, 1, 0.98])

        out_png = output_root / f"master_peak2peak_and_firingRate_{label_tag}_{variant_tag}.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        out_npz = output_root / f"master_peak2peak_and_firingRate_{label_tag}_{variant_tag}_plotData.npz"
        np.savez_compressed(
            str(out_npz),
            normalize_each_day=np.asarray([normalize_each_day], dtype=np.bool_),
            ylim_mode=np.asarray([ylim_mode]),
            smoothing_window_min=np.asarray([5.0], dtype=np.float32),
            outlier_rule=np.asarray(["drop_single_farthest_from_window_median_when_window_size_gt_2"]),
            **plot_data,
        )

        print(f"Saved master plot -> {out_png}")
        print(f"Saved lightweight plot data -> {out_npz}")

    bin_variants = [
        (1.0, "hourlyBin"),
        (2.0, "twoHourBin"),
    ]
    norm_variants = [
        ("raw", False),
        ("dayZ", True),
    ]

    for bin_hours, bin_tag in bin_variants:
        for norm_tag, normalize_each_day in norm_variants:
            log_status(f"Rendering binned daily-cycle variant: {bin_tag}_{norm_tag}")
            fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(fig_w, fig_h), sharex=True)
            if nrows == 1:
                axes = [axes]

            plot_data: dict[str, np.ndarray] = {}

            for row_i, (pid, _) in enumerate(pair_meta):
                xs, peak, peak5, fr, labels = series_cache[pid]

                ax_p2p = axes[2 * row_i]
                ax_fr = axes[2 * row_i + 1]

                tag = pid.display_label()
                amplitude_name = "average amplitude" if loaded_from_aligned else "peak-to-peak"
                p2p_ylabel = (
                    "Average amplitude day z-score"
                    if loaded_from_aligned and normalize_each_day
                    else "Average amplitude [uV]"
                    if loaded_from_aligned
                    else "Peak-to-peak day z-score"
                    if normalize_each_day
                    else "Peak-to-peak [uV]"
                )
                fr_ylabel = "Firing-rate day z-score" if normalize_each_day else "Firing rate [Hz]"
                norm_note = "day-standardized" if normalize_each_day else "raw"
                bin_note = f"{bin_hours:g} h bins"

                peak_plot_data = plot_binned_daily_cycles(
                    ax_p2p,
                    xs,
                    peak,
                    bin_hours=bin_hours,
                    ylabel=p2p_ylabel,
                    title=f"{tag}  |  {amplitude_name}  |  {bin_note}, {norm_note}",
                    normalize_each_day=normalize_each_day,
                )

                fr_plot_data = plot_binned_daily_cycles(
                    ax_fr,
                    xs,
                    fr,
                    bin_hours=bin_hours,
                    ylabel=fr_ylabel,
                    title=f"{tag}  |  firing rate  |  {bin_note}, {norm_note}",
                    normalize_each_day=normalize_each_day,
                )

                safe_tag = pid.folder_tag().replace(".", "p")
                for key, value in peak_plot_data.items():
                    plot_data[f"{safe_tag}_peak_{key}"] = value
                for key, value in fr_plot_data.items():
                    plot_data[f"{safe_tag}_firing_{key}"] = value

            axes[-1].set_xlabel("Hour of day (0-24), binned by local recording datetime")
            fig.suptitle(
                f"Master peak-to-peak & firing rate - {bin_hours:g} h binned 24h cycle ({norm_tag})",
                fontsize=14,
            )
            fig.tight_layout(rect=[0, 0.01, 1, 0.98])

            out_png = output_root / f"master_peak2peak_and_firingRate_{bin_tag}_{norm_tag}.png"
            fig.savefig(out_png, dpi=200)
            plt.close(fig)

            out_npz = output_root / f"master_peak2peak_and_firingRate_{bin_tag}_{norm_tag}_plotData.npz"
            np.savez_compressed(
                str(out_npz),
                bin_hours=np.asarray([bin_hours], dtype=np.float32),
                normalize_each_day=np.asarray([normalize_each_day], dtype=np.bool_),
                bin_statistic=np.asarray(["mean_within_day_then_mean_std_across_days"]),
                **plot_data,
            )

            print(f"Saved binned master plot -> {out_png}")
            print(f"Saved binned plot data   -> {out_npz}")

    return 0

    n_pairs = len(pair_meta)
    nrows = 2 * n_pairs
    fig_w = 14
    fig_h = max(6, 2.6 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(fig_w, fig_h), sharex=True)
    if nrows == 1:
        axes = [axes]

    label_tag = "dailyCycleOverlay"
    plot_data: dict[str, np.ndarray] = {}

    for row_i, (pid, _) in enumerate(pair_meta):
        xs, peak, peak5, fr, labels = series_cache[pid]

        ax_p2p = axes[2 * row_i]
        ax_fr = axes[2 * row_i + 1]

        tag = pid.display_label()

        peak_plot_data = plot_daily_cycles(
            ax_p2p,
            xs,
            peak,
            ylabel="Peak-to-peak [uV]",
            title=f"{tag}  |  peak-to-peak",
            show_5min_avg=True,
        )

        fr_plot_data = plot_daily_cycles(
            ax_fr,
            xs,
            fr,
            ylabel="Firing rate [Hz]",
            title=f"{tag}  |  firing rate",
            show_5min_avg=True,
        )
        safe_tag = pid.folder_tag().replace(".", "p")
        for key, value in peak_plot_data.items():
            plot_data[f"{safe_tag}_peak_{key}"] = value
        for key, value in fr_plot_data.items():
            plot_data[f"{safe_tag}_firing_{key}"] = value

    axes[-1].set_xlabel("Hour of day (0–24), consecutive days overlaid")

    fig.suptitle("Master peak-to-peak & firing rate — 24h cycle overlay (days color-coded)", fontsize=14)
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])

    out_png = run_root / f"master_peak2peak_and_firingRate_{label_tag}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    out_npz = run_root / f"master_peak2peak_and_firingRate_{label_tag}_plotData.npz"
    np.savez_compressed(str(out_npz), **plot_data)

    print(f"Saved master plot -> {out_png}")
    print(f"Saved lightweight plot data -> {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
