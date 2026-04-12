from __future__ import annotations

import json
import re
import os
import shutil
import subprocess
import tempfile
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception as e:
    raise RuntimeError(
        "imageio is required for video rendering. "
        "Install with: pip install imageio"
    ) from e


WAVEFORM_PNG_SUFFIX = "_waveforms.png"
VIDEO_FPS = 60


CHRONIC_REC_RE = re.compile(r"Chronic_Rec_(?P<ymd>\d{8})_(?P<hms>\d{6})")
CHUNK_RE = re.compile(r"chunk_(?P<idx>\d+)")


def _rec_sort_key_from_filename(name: str) -> int:
    m = CHRONIC_REC_RE.search(name)
    if not m:
        return -1
    ymd = m.group("ymd")
    hms = m.group("hms")
    return int(ymd) * 1_000_000 + int(hms)


def _chunk_index_from_filename(name: str) -> int:
    m = CHUNK_RE.search(name)
    if not m:
        return -1
    return int(m.group("idx"))


def resolve_ffmpeg_exe() -> str:
    """
    Resolve ffmpeg executable path.
    Mirrors the approach used in `ManuscriptFigures/lfp_scatter_video.py`.
    """
    env_exe = os.environ.get("IMAGEIO_FFMPEG_EXE") or os.environ.get("FFMPEG_EXE")
    if env_exe and Path(env_exe).is_file():
        return env_exe

    exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if exe:
        return exe

    try:
        import imageio_ffmpeg  # type: ignore

        exe2 = imageio_ffmpeg.get_ffmpeg_exe()
        if exe2 and Path(exe2).is_file():
            return exe2
    except Exception:
        pass

    # Common conda layout fallback:
    # - base env: <conda_root>/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-*.exe
    # - current env: <conda_root>/envs/<env>/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-*.exe
    try:
        conda_root = Path(sys.prefix).parents[1]
        candidate_dirs = [
            conda_root / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries",
            Path(sys.prefix) / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries",
        ]
        for d in candidate_dirs:
            if not d.exists():
                continue
            cands = sorted(d.glob("ffmpeg-*.exe"), key=lambda p: p.stat().st_mtime, reverse=True)
            for c in cands[:3]:
                if c.is_file():
                    return str(c)
    except Exception:
        pass

    raise FileNotFoundError(
        "ffmpeg not found. Install `imageio-ffmpeg` or set IMAGEIO_FFMPEG_EXE/FFMPEG_EXE "
        "to a full ffmpeg executable path."
    )


def build_video_from_pngs(png_paths: list[Path], out_mp4: Path, fps: int = VIDEO_FPS) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg_exe = resolve_ffmpeg_exe()
    except FileNotFoundError as e:
        # Fallback: write a GIF so the pipeline can still complete.
        out_gif = out_mp4.with_suffix(".gif")
        print(
            f"[warn] {e}\n"
            f"Falling back to GIF rendering instead: {out_gif.name}",
            flush=True,
        )
        writer = imageio.get_writer(str(out_gif), mode="I", duration=1.0 / max(1, fps))
        try:
            for p in png_paths:
                img = imageio.imread(str(p))
                writer.append_data(img)
        finally:
            writer.close()
        return

    # ffmpeg expects an ordered pattern; make a temp numbered frame folder.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        for i, p in enumerate(png_paths):
            dst = tmpdir_p / f"frame_{i:05d}.png"
            shutil.copy2(str(p), str(dst))

        in_pattern = str(tmpdir_p / "frame_%05d.png")
        out_mp4_str = str(out_mp4)

        cmd = [
            ffmpeg_exe,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            in_pattern,
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            out_mp4_str,
        ]
        subprocess.run(cmd, check=True)


def compute_peak_to_peak_from_npz(npz_path: Path) -> float:
    with np.load(str(npz_path), allow_pickle=False) as z:
        waveforms = z["waveforms_uv"]  # shape (n_events, wf_len)
    if waveforms.shape[0] == 0:
        return float("nan")
    mean_wf = waveforms.mean(axis=0)
    # peak-to-peak of the chunk mean waveform
    return float(np.max(mean_wf) - np.min(mean_wf))


def compute_firing_rate_from_npz(npz_path: Path) -> float:
    """
    Compute firing rate for the chunk using:
      firing_rate_Hz = n_crossings / (time_end_sec - time_start_sec)
    """
    with np.load(str(npz_path), allow_pickle=False) as z:
        crossing_samples = z["crossing_samples"]
        n_cross = int(crossing_samples.shape[0])
        t0 = z["time_start_sec"]
        t1 = z["time_end_sec"]
        # Stored as arrays of shape (1,) by the detector; be tolerant here.
        t0_s = float(t0[0]) if np.asarray(t0).shape else float(t0)
        t1_s = float(t1[0]) if np.asarray(t1).shape else float(t1)
    dur = max(1e-12, t1_s - t0_s)
    return float(n_cross / dur)


def parse_recording_start_datetime_from_name(name: str) -> datetime | None:
    """
    Extract Chronic_Rec_YYYYMMDD_HHMMSS from a filename and convert to datetime.
    """
    m = CHRONIC_REC_RE.search(name)
    if not m:
        return None
    ymd = m.group("ymd")
    hms = m.group("hms")
    return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")


def datetime_to_x_label(dt: datetime) -> str:
    """
    Label style requested: MM_DD_5p / MM_DD_5a.
    Assumes 17:00+ => 5p bucket, otherwise 5a bucket.
    """
    bucket = "5p" if dt.hour >= 17 else "5a"
    return f"{dt:%m}_{dt:%d}_{bucket}"


def rolling_mean_skip_outlier(xs_min: np.ndarray, ys: np.ndarray, window_min: float = 5.0) -> np.ndarray:
    """
    For each point i, take points within +/- window_min/2 around xs_min[i],
    compute a 5-min mean after removing the single outlier (farthest from median).
    """
    xs_min = np.asarray(xs_min, dtype=float)
    ys = np.asarray(ys, dtype=float)
    out = np.full(xs_min.shape, np.nan, dtype=float)
    half = window_min / 2.0

    finite = np.isfinite(xs_min) & np.isfinite(ys)
    xs_f = xs_min[finite]
    ys_f = ys[finite]

    # If everything is non-finite, just return nan.
    if xs_f.size == 0:
        return out

    # Use original indexing for output, but compute window on finite arrays only.
    # We do the window selection using the original arrays for correctness with NaNs.
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


def load_cumulative_segment_start_sec(recording_summary_path: Path) -> float | None:
    if not recording_summary_path.exists():
        return None
    with open(recording_summary_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # Written by the detector main loop
    key = "cumulative_segment_start_sec"
    if key not in meta:
        return None
    return float(meta[key])


def main():
    run_root = Path(
        input(
            "Run folder (e.g. I:/threshold_crossings_outputs/threshold_crossings_run_YYYYMMDD_HHMMSS): "
        ).strip()
    )
    if not run_root.exists():
        raise FileNotFoundError(run_root)

    raw = input("Create video files (MP4/GIF) from chunk waveform PNGs? [n]: ").strip().lower()
    create_video = raw in ("y", "yes")

    raw_lbl = input(
        "X-axis labels: (1) MM_DD_5p/5a buckets, (2) actual minute ticks [1/2] (default 1): "
    ).strip()
    if raw_lbl == "":
        raw_lbl = "1"
    use_bucket_labels = raw_lbl in ("1", "bucket", "buckets")

    # Expect subfolders like: sgch337_thr500uV
    pair_dirs = sorted(
        [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("sgch") and "_thr" in p.name]
    )
    if not pair_dirs:
        raise RuntimeError(f"No sgch*_thr* folders found under: {run_root}")

    for pair_dir in pair_dirs:
        sgch_thr = pair_dir.name
        print(f"Processing pair folder: {sgch_thr}")

        # 1) Video: append all chunk waveform PNGs in chronic order
        if create_video:
            pngs = []
            for p in pair_dir.rglob("*.png"):
                if not p.name.endswith(WAVEFORM_PNG_SUFFIX):
                    continue
                if "_first10s_" in p.name:
                    continue
                pngs.append(p)

            def sort_key(p: Path):
                name = p.name
                return (
                    _rec_sort_key_from_filename(name),
                    _chunk_index_from_filename(name),
                    name,
                )

            pngs.sort(key=sort_key)

            out_mp4 = pair_dir / f"{sgch_thr}_threshold_crossings_{VIDEO_FPS}fps.mp4"
            if pngs:
                print(f"  Writing video with {len(pngs)} frames -> {out_mp4.name}")
                build_video_from_pngs(pngs, out_mp4, fps=VIDEO_FPS)
            else:
                print("  No *_waveforms.png files found; skipping video.")
        else:
            print("  Skipping video creation (mean-amplitude plot only).")

        # 2) Mean amplitude vs minute (per chunk)
        npz_paths = []
        for p in pair_dir.rglob("*.npz"):
            if p.name.endswith("_threshold_crossings.npz") and "_chunk_" in p.name:
                npz_paths.append(p)

        # Each chunk npz name is:
        #   <parent>__<stem>_chunk_NNNN_threshold_crossings.npz
        # So base recording id (for summary lookup) is:
        #   <parent>__<stem>
        def base_recording_id(npz_name: str) -> str:
            # Split on the literal token "_chunk_"
            return npz_name.split("_chunk_")[0]

        rows = []
        labels = []
        for npz_path in npz_paths:
            base_id = base_recording_id(npz_path.name)
            recording_summary_path = npz_path.parent / f"{base_id}_recording_summary.json"

            cum_start = load_cumulative_segment_start_sec(recording_summary_path)
            with np.load(str(npz_path), allow_pickle=False) as z:
                t_start_sec = float(z["time_start_sec"][0]) if z["time_start_sec"].shape else float(z["time_start_sec"])
                # (Optional) t_end_sec = float(z["time_end_sec"][0]) ...

            peak2peak = compute_peak_to_peak_from_npz(npz_path)
            firing_rate = compute_firing_rate_from_npz(npz_path)

            dt0 = parse_recording_start_datetime_from_name(npz_path.name)
            if dt0 is None:
                # Fallback: use chained minutes if recording datetime parsing fails.
                if cum_start is None:
                    x_min = t_start_sec / 60.0
                    dt_label = f"unknown_{x_min:.1f}min"
                else:
                    x_min = (cum_start + t_start_sec) / 60.0
                    dt_label = f"unknown_{x_min:.1f}min"
            else:
                dt_chunk = dt0 + timedelta(seconds=t_start_sec)
                x_min = dt_chunk.timestamp() / 60.0
                dt_label = datetime_to_x_label(dt_chunk)

            rows.append((x_min, peak2peak, firing_rate))
            labels.append(dt_label)

        if rows:
            # Sort by time and keep labels aligned
            order = np.argsort([r[0] for r in rows])
            rows_sorted = [rows[i] for i in order]
            labels_sorted = [labels[i] for i in order]

            xs = np.array([r[0] for r in rows_sorted], dtype=float)
            ys = np.array([r[1] for r in rows_sorted], dtype=float)
            frs = np.array([r[2] for r in rows_sorted], dtype=float)

            # 5-min rolling average with outlier skipping
            y5 = rolling_mean_skip_outlier(xs, ys, window_min=5.0)
            y5_fr = rolling_mean_skip_outlier(xs, frs, window_min=5.0)

            # Precompute x ticks once, so both plots match.
            if use_bucket_labels:
                x_start = float(xs.min())
                x_end = float(xs.max())
                dt_start = datetime.fromtimestamp(x_start * 60.0)
                dt_end = datetime.fromtimestamp(x_end * 60.0)

                tick_positions: list[float] = []
                tick_labels: list[str] = []

                day = dt_start.date()
                while day <= dt_end.date():
                    for dt_tick, lbl in (
                        (datetime(day.year, day.month, day.day, 5, 0, 0), "5a"),
                        (datetime(day.year, day.month, day.day, 17, 0, 0), "5p"),
                    ):
                        x_tick = dt_tick.timestamp() / 60.0
                        if (x_tick >= x_start - 1e-6) and (x_tick <= x_end + 1e-6):
                            tick_positions.append(float(x_tick))
                            tick_labels.append(f"{dt_tick:%m}_{dt_tick:%d}_{lbl}")
                    day = day + timedelta(days=1)

                if not tick_positions:
                    tick_positions = [x_start, x_end]
                    tick_labels = [f"{x_start:.1f}m", f"{x_end:.1f}m"]
            else:
                n = len(xs)
                max_ticks = 14
                if n <= max_ticks:
                    tick_idx = np.arange(n, dtype=int)
                else:
                    tick_idx = np.linspace(0, n - 1, num=max_ticks, dtype=int)
                tick_positions = xs[tick_idx].tolist()
                x0 = float(xs[0])
                tick_labels = [f"{(xs[i] - x0):.1f}m" for i in tick_idx]

            # Plot
            plt.figure(figsize=(11, 5))
            plt.plot(xs, ys, linewidth=0.9, alpha=0.65, label="chunk mean waveform peak-to-peak")
            plt.plot(xs, y5, linewidth=1.1, color="crimson", label="5-min avg (skip 1 outlier)")
            plt.scatter(xs, ys, s=8, alpha=0.35)
            if use_bucket_labels:
                plt.xlabel("Time (MM_DD_5p/5a)")
            else:
                plt.xlabel("Time (minutes)")
            plt.ylabel("Peak-to-peak of chunk mean waveform [uV]")
            plt.title(f"{sgch_thr}: peak-to-peak vs time")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best", framealpha=0.9)

            plt.xticks(tick_positions, tick_labels, rotation=90, ha="center")

            label_tag = "MMDD5p5aTicks" if use_bucket_labels else "actualMinuteTicks"
            out_plot = pair_dir / f"{sgch_thr}_peakToPeak_vs_time_{label_tag}.png"
            plt.tight_layout()
            plt.savefig(out_plot, dpi=200)
            plt.close()

            # Also save CSV for convenience
            out_csv = pair_dir / f"{sgch_thr}_peakToPeak_vs_time_{label_tag}.csv"
            with open(out_csv, "w", encoding="utf-8") as f:
                f.write("time_min_epoch,dt_label,peak_to_peak_uV,peak_to_peak_5min_avg_uV\n")
                for i in range(len(xs)):
                    f.write(
                        f"{xs[i]:.6f},{labels_sorted[i]},{ys[i]:.6f},{y5[i] if np.isfinite(y5[i]) else float('nan'):.6f}\n"
                    )

            print(f"  Saved plot -> {out_plot.name}")
            print(f"  Saved csv   -> {out_csv.name}")

            # --- Firing rate plot (same logic, just different y) ---
            plt.figure(figsize=(11, 5))
            plt.plot(xs, frs, linewidth=0.9, alpha=0.65, label="chunk firing rate [Hz]")
            plt.plot(xs, y5_fr, linewidth=1.1, color="crimson", label="5-min avg (skip 1 outlier)")
            plt.scatter(xs, frs, s=8, alpha=0.35)
            if use_bucket_labels:
                plt.xlabel("Time (MM_DD_5p/5a)")
            else:
                plt.xlabel("Time (minutes)")
            plt.ylabel("Firing rate [Hz]")
            plt.title(f"{sgch_thr}: firing rate vs time")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best", framealpha=0.9)
            plt.xticks(tick_positions, tick_labels, rotation=90, ha="center")

            out_plot_fr = pair_dir / f"{sgch_thr}_firingRate_vs_time_{label_tag}.png"
            plt.tight_layout()
            plt.savefig(out_plot_fr, dpi=200)
            plt.close()

            out_csv_fr = pair_dir / f"{sgch_thr}_firingRate_vs_time_{label_tag}.csv"
            with open(out_csv_fr, "w", encoding="utf-8") as f:
                f.write("time_min_epoch,dt_label,firing_rate_Hz,firing_rate_5min_avg_Hz\n")
                for i in range(len(xs)):
                    f.write(
                        f"{xs[i]:.6f},{labels_sorted[i]},{frs[i]:.6f},{y5_fr[i] if np.isfinite(y5_fr[i]) else float('nan'):.6f}\n"
                    )

            print(f"  Saved plot -> {out_plot_fr.name}")
            print(f"  Saved csv   -> {out_csv_fr.name}")
        else:
            print("  No chunk npz files found; skipping mean-amplitude plot.")


if __name__ == "__main__":
    main()