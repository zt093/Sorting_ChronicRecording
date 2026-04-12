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

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


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
) -> None:
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
        return

    days_sorted = sorted(day_to_idx.keys())
    n_days = len(days_sorted)
    cmap = cm.get_cmap("viridis", max(2, n_days))
    norm = mcolors.Normalize(vmin=0, vmax=max(1, n_days - 1))

    y5_all: list[np.ndarray] = []

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

        color = cmap(norm(di))
        ax.plot(x_hour, y_day, color=color, linewidth=0.8, alpha=0.75)
        ax.scatter(x_hour, y_day, color=color, s=6, alpha=0.25)

        if show_5min_avg and y_day.size >= 2:
            # 5-minute avg within the day (x axis in minutes)
            x_min = x_hour * 60.0
            y5 = rolling_mean_skip_outlier(x_min, y_day, window_min=5.0)
            ax.plot(x_hour, y5, color=color, linewidth=1.2, alpha=0.95)
            y5_all.append(y5)

    ax.set_xlim(0.0, 24.0)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.grid(True, alpha=0.25)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)

    # Clamp y-limits using distribution of the 5-min average curve.
    # mean(y5) ± 6*std(y5), ignoring NaNs.
    if show_5min_avg and y5_all:
        y5cat = np.concatenate([np.asarray(y) for y in y5_all], axis=0)
        finite = np.isfinite(y5cat)
        if np.any(finite):
            y5f = y5cat[finite].astype(float)
            mu = float(np.mean(y5f))
            sigma = float(np.std(y5f))
            if sigma > 0:
                ax.set_ylim(mu - 3.0 * sigma, mu + 3.0 * sigma)
            else:
                ax.set_ylim(mu - 1.0, mu + 1.0)

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


def main() -> int:
    run_root = Path(
        input(
            "Run folder (e.g. I:/threshold_crossings_outputs/threshold_crossings_run_YYYYMMDD_HHMMSS): "
        ).strip()
    )
    if not run_root.exists():
        raise FileNotFoundError(run_root)

    print(
        "Master plot x-axis mode: overlay 24h cycles (0–24 hours) and color-code consecutive days.",
        flush=True,
    )

    reuse_amp = input("Reuse existing peakToPeak CSV if present? [Y/n] (default Y): ").strip().lower()
    reuse_fr = input("Reuse existing firingRate CSV if present? [Y/n] (default Y): ").strip().lower()
    reuse_amp = (reuse_amp in ("", "y", "yes"))
    reuse_fr = (reuse_fr in ("", "y", "yes"))

    # Discover pair folders
    pair_dirs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("sgch") and "_thr" in p.name])
    pair_meta: list[tuple[PairId, Path]] = []
    for p in pair_dirs:
        pid = parse_pair_id_from_folder_name(p.name)
        if pid is not None:
            pair_meta.append((pid, p))
    if not pair_meta:
        raise RuntimeError(f"No sgch*_thr*uV subfolders found under: {run_root}")

    pair_meta.sort(key=lambda t: t[0].sort_key())

    all_x_min: list[float] = []
    series_cache = {}

    for pid, pdir in pair_meta:
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

    n_pairs = len(pair_meta)
    nrows = 2 * n_pairs
    fig_w = 14
    fig_h = max(6, 2.6 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(fig_w, fig_h), sharex=True)
    if nrows == 1:
        axes = [axes]

    label_tag = "dailyCycleOverlay"

    for row_i, (pid, _) in enumerate(pair_meta):
        xs, peak, peak5, fr, labels = series_cache[pid]

        ax_p2p = axes[2 * row_i]
        ax_fr = axes[2 * row_i + 1]

        tag = f"sgch{pid.sg_ch}  thr{pid.threshold_uv:g}uV"

        plot_daily_cycles(
            ax_p2p,
            xs,
            peak,
            ylabel="Peak-to-peak [uV]",
            title=f"{tag}  |  peak-to-peak",
            show_5min_avg=True,
        )

        plot_daily_cycles(
            ax_fr,
            xs,
            fr,
            ylabel="Firing rate [Hz]",
            title=f"{tag}  |  firing rate",
            show_5min_avg=True,
        )

    axes[-1].set_xlabel("Hour of day (0–24), consecutive days overlaid")

    fig.suptitle("Master peak-to-peak & firing rate — 24h cycle overlay (days color-coded)", fontsize=14)
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])

    out_png = run_root / f"master_peak2peak_and_firingRate_{label_tag}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved master plot -> {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

