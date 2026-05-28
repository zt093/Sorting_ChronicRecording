from __future__ import annotations

import json
import re
import os
import shutil
import subprocess
import tempfile
import sys
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

try:
    import imageio.v2 as imageio
except Exception as e:
    raise RuntimeError(
        "imageio is required for video rendering. "
        "Install with: pip install imageio"
    ) from e


WAVEFORM_PNG_SUFFIX = "_waveforms.png"
VIDEO_FPS = 60
FRAME_DPI = 120
FRAME_FIGSIZE = (6.4, 4.8)


CHRONIC_REC_RE = re.compile(r"Chronic_Rec_(?P<ymd>\d{8})_(?P<hms>\d{6})")
CHUNK_RE = re.compile(r"chunk_(?P<idx>\d+)")


def _scalar_from_npz_array(value) -> float:
    arr = np.asarray(value)
    return float(arr[0]) if arr.shape else float(arr)


def _str_array(values) -> list[str]:
    return [str(v.decode("utf-8") if isinstance(v, bytes) else v) for v in values]


def _format_threshold_for_folder(value: float) -> str:
    s = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return s.replace("-", "neg").replace(".", "p")


def pair_folder_name(sg_ch: int, threshold_min_uv: float, threshold_max_uv: float | None = None) -> str:
    min_text = _format_threshold_for_folder(threshold_min_uv)
    if threshold_max_uv is None or not np.isfinite(threshold_max_uv):
        return f"sgch{int(sg_ch)}_thr{min_text}uV"
    max_text = _format_threshold_for_folder(threshold_max_uv)
    return f"sgch{int(sg_ch)}_thr{min_text}to{max_text}uV"


def pair_label(sg_ch: int, threshold_min_uv: float, threshold_max_uv: float | None = None) -> str:
    if threshold_max_uv is None or not np.isfinite(threshold_max_uv):
        return f"SG ch {int(sg_ch)} | threshold {float(threshold_min_uv):g} uV"
    return f"SG ch {int(sg_ch)} | threshold {float(threshold_min_uv):g}-{float(threshold_max_uv):g} uV"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)} min {sec:.1f} s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours} h {minutes} min {sec:.1f} s"


def parse_run_root_inputs(values: list[Path] | None) -> list[Path]:
    if values:
        return [Path(v) for v in values]
    raw = input(
        "Threshold_channel.py run folder(s), separated by semicolons "
        "(each contains run_config.json and minute_npz): "
    ).strip()
    roots = [Path(part.strip().strip('"')) for part in raw.split(";") if part.strip()]
    if not roots:
        raise ValueError("At least one run folder is required.")
    return roots


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


def collect_all_pairs_minute_npz(run_root: Path) -> list[Path]:
    minute_root = run_root / "minute_npz"
    if not minute_root.exists():
        raise RuntimeError(
            f"No minute_npz folder found under {run_root}. "
            "Run Threshold_channel.py first; this renderer expects its *_all_pairs.npz outputs."
        )
    npz_paths = sorted(minute_root.rglob("*_all_pairs.npz"))
    if not npz_paths:
        raise RuntimeError(f"No *_all_pairs.npz minute files found under: {minute_root}")

    def sort_key(p: Path) -> tuple[float, int, str]:
        try:
            with np.load(str(p), allow_pickle=False) as z:
                if "cumulative_start_sec" in z:
                    return (_scalar_from_npz_array(z["cumulative_start_sec"]), 0, p.name)
                if "cumulative_minute_index" in z:
                    return (_scalar_from_npz_array(z["cumulative_minute_index"]) * 60.0, 0, p.name)
                rec_dt = parse_recording_start_datetime_from_name(p.name)
                minute = int(_scalar_from_npz_array(z["minute_index"])) if "minute_index" in z else 0
                if rec_dt is not None:
                    return (rec_dt.timestamp() + minute * 60.0, 0, p.name)
        except Exception:
            pass
        return (float("inf"), _rec_sort_key_from_filename(p.name), p.name)

    return sorted(npz_paths, key=sort_key)


def load_run_prepost_samples(run_root: Path, wf_len: int, fs: float) -> tuple[int, int]:
    config_path = run_root / "run_config.json"
    if config_path.exists():
        try:
            meta = json.loads(config_path.read_text(encoding="utf-8"))
            pre = int(meta.get("pre_samples", 0))
            post = int(meta.get("post_samples", 0))
            if pre > 0 and post > 0 and pre + post == wf_len:
                return pre, post
        except Exception:
            pass
    # Threshold_channel.py defaults to -1 ms .. +2 ms. Fall back to that ratio.
    pre = int(round(float(fs) * 0.001))
    if pre <= 0 or pre >= wf_len:
        pre = max(1, wf_len // 3)
    return pre, wf_len - pre


def collect_minute_npz_from_run_roots(run_roots: list[Path]) -> list[Path]:
    all_paths: list[Path] = []
    for run_root in run_roots:
        all_paths.extend(collect_all_pairs_minute_npz(run_root))
    return all_paths


def default_combined_output_root(run_roots: list[Path]) -> Path:
    if len(run_roots) == 1:
        return run_roots[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_roots[0].parent / f"combined_threshold_videos_{stamp}"


def load_minute_waveform_cube(
    npz_paths: list[Path],
    *,
    contiguous_elapsed_time: bool = False,
) -> dict[str, np.ndarray | list[str]]:
    """
    Load Threshold_channel.py recording-level minute NPZs into time-major arrays.

    Expected per-minute schema:
      pair_ids, sg_ch, threshold_min_uv, threshold_max_uv, n_spikes,
      firing_rate_hz, amplitude_ptp_uv, mean_waveform_uv, cumulative_start_sec.
    """
    pair_ids: list[str] = []
    pair_index: dict[str, int] = {}
    sg_ch_ref: list[int] = []
    thr_min_ref: list[float] = []
    thr_max_ref: list[float] = []

    mean_frames: list[np.ndarray] = []
    n_spikes_frames: list[np.ndarray] = []
    firing_frames: list[np.ndarray] = []
    amp_frames: list[np.ndarray] = []
    cumulative_start_sec: list[float] = []
    recording_names: list[str] = []
    minute_indices: list[int] = []
    fs = None
    wf_len: int | None = None

    def expand_existing_frames(new_pair_count: int) -> None:
        old_pair_count = mean_frames[0].shape[0] if mean_frames else new_pair_count
        if new_pair_count <= old_pair_count or wf_len is None:
            return
        for i in range(len(mean_frames)):
            expanded_mean = np.full((new_pair_count, wf_len), np.nan, dtype=np.float32)
            expanded_mean[:old_pair_count, :] = mean_frames[i]
            mean_frames[i] = expanded_mean

            expanded_n = np.zeros(new_pair_count, dtype=np.int64)
            expanded_n[:old_pair_count] = n_spikes_frames[i]
            n_spikes_frames[i] = expanded_n

            expanded_firing = np.zeros(new_pair_count, dtype=np.float32)
            expanded_firing[:old_pair_count] = firing_frames[i]
            firing_frames[i] = expanded_firing

            expanded_amp = np.full(new_pair_count, np.nan, dtype=np.float32)
            expanded_amp[:old_pair_count] = amp_frames[i]
            amp_frames[i] = expanded_amp

    for npz_path in npz_paths:
        with np.load(str(npz_path), allow_pickle=False) as z:
            current_pair_ids = _str_array(z["pair_ids"])
            mean_wf = np.asarray(z["mean_waveform_uv"], dtype=np.float32)
            if mean_wf.ndim != 2:
                raise RuntimeError(f"{npz_path} has unexpected mean_waveform_uv shape: {mean_wf.shape}")
            if wf_len is None:
                wf_len = int(mean_wf.shape[1])
            elif mean_wf.shape[1] != wf_len:
                raise RuntimeError(
                    f"{npz_path} waveform length {mean_wf.shape[1]} does not match prior length {wf_len}"
                )

            current_sg_ch = np.asarray(z["sg_ch"], dtype=np.int32)
            current_thr_min = np.asarray(z["threshold_min_uv"], dtype=np.float32)
            current_thr_max = np.asarray(z["threshold_max_uv"], dtype=np.float32)
            for in_i, pid in enumerate(current_pair_ids):
                if pid in pair_index:
                    continue
                pair_index[pid] = len(pair_ids)
                pair_ids.append(pid)
                sg_ch_ref.append(int(current_sg_ch[in_i]))
                thr_min_ref.append(float(current_thr_min[in_i]))
                thr_max_ref.append(float(current_thr_max[in_i]))
                expand_existing_frames(len(pair_ids))

            n_pairs = len(pair_ids)
            remapped = np.full((n_pairs, wf_len), np.nan, dtype=np.float32)
            n_spikes_remap = np.zeros(n_pairs, dtype=np.int64)
            firing_remap = np.zeros(n_pairs, dtype=np.float32)
            amp_remap = np.full(n_pairs, np.nan, dtype=np.float32)
            current_n_spikes = np.asarray(z["n_spikes"], dtype=np.int64)
            current_firing = np.asarray(z["firing_rate_hz"], dtype=np.float32)
            current_amp = np.asarray(z["amplitude_ptp_uv"], dtype=np.float32)
            for in_i, pid in enumerate(current_pair_ids):
                out_i = pair_index[pid]
                remapped[out_i, :] = mean_wf[in_i, :]
                n_spikes_remap[out_i] = int(current_n_spikes[in_i])
                firing_remap[out_i] = float(current_firing[in_i])
                amp_remap[out_i] = float(current_amp[in_i])
            mean_frames.append(remapped)
            n_spikes_frames.append(n_spikes_remap)
            firing_frames.append(firing_remap)
            amp_frames.append(amp_remap)

            if contiguous_elapsed_time:
                cumulative_start_sec.append(float(len(cumulative_start_sec) * 60.0))
            elif "cumulative_start_sec" in z:
                cumulative_start_sec.append(_scalar_from_npz_array(z["cumulative_start_sec"]))
            elif "cumulative_minute_index" in z:
                cumulative_start_sec.append(_scalar_from_npz_array(z["cumulative_minute_index"]) * 60.0)
            else:
                cumulative_start_sec.append(float(len(cumulative_start_sec) * 60))

            if "recording_name" in z:
                recording_names.append(_str_array(z["recording_name"])[0])
            else:
                recording_names.append(npz_path.parent.name)

            if "minute_index" in z:
                minute_indices.append(int(_scalar_from_npz_array(z["minute_index"])))
            else:
                minute_indices.append(len(minute_indices))

            if fs is None and "sampling_rate_hz" in z:
                fs = _scalar_from_npz_array(z["sampling_rate_hz"])

    if not pair_ids:
        raise RuntimeError("No usable minute NPZ files were loaded.")

    return {
        "pair_ids": pair_ids,
        "sg_ch": np.asarray(sg_ch_ref, dtype=np.int32),
        "threshold_min_uv": np.asarray(thr_min_ref, dtype=np.float32),
        "threshold_max_uv": np.asarray(thr_max_ref, dtype=np.float32),
        "mean_waveform_uv": np.stack(mean_frames, axis=0),
        "n_spikes": np.stack(n_spikes_frames, axis=0),
        "firing_rate_hz": np.stack(firing_frames, axis=0),
        "amplitude_ptp_uv": np.stack(amp_frames, axis=0),
        "cumulative_start_sec": np.asarray(cumulative_start_sec, dtype=np.float64),
        "recording_names": recording_names,
        "minute_indices": np.asarray(minute_indices, dtype=np.int64),
        "sampling_rate_hz": np.asarray([fs if fs is not None else 30000.0], dtype=np.float64),
    }


def _figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvas(fig)
    canvas.draw()
    return _canvas_to_rgb(canvas)


def _canvas_to_rgb(canvas: FigureCanvas) -> np.ndarray:
    w, h = canvas.get_width_height()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    return np.ascontiguousarray(rgba[:, :, :3])


def render_waveform_frame(
    waveform_uv: np.ndarray,
    *,
    t_ms: np.ndarray,
    y_limits: tuple[float, float],
    unit_label: str,
    minute_number: int,
    cumulative_start_sec: float,
    n_spikes: int,
    firing_rate_hz: float,
    amplitude_ptp_uv: float,
    recording_name: str,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=FRAME_FIGSIZE, dpi=FRAME_DPI)
    wf = np.asarray(waveform_uv, dtype=np.float32)
    valid = np.isfinite(wf)
    if np.any(valid):
        ax.plot(t_ms, wf, color="black", linewidth=2.2)
    else:
        ax.text(0.5, 0.5, "No crossings in this minute", ha="center", va="center", transform=ax.transAxes)

    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Time around crossing [ms]")
    ax.set_ylabel("Mean waveform [uV]")
    ax.grid(True, alpha=0.22)

    elapsed_h = cumulative_start_sec / 3600.0
    day = int(cumulative_start_sec // 86400) + 1
    hour = int((cumulative_start_sec % 86400) // 3600)
    minute = int((cumulative_start_sec % 3600) // 60)
    amp_text = "nan" if not np.isfinite(amplitude_ptp_uv) else f"{amplitude_ptp_uv:.1f}"
    ax.set_title(
        f"{unit_label}\n"
        f"frame {minute_number} | elapsed {elapsed_h:.2f} h | day {day}, {hour:02d}:{minute:02d} | "
        f"N={int(n_spikes)} | FR={float(firing_rate_hz):.3f} Hz | p2p={amp_text} uV",
        fontsize=9,
    )
    ax.text(
        0.01,
        0.01,
        recording_name,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="0.35",
    )
    fig.tight_layout()
    frame = _figure_to_rgb(fig)
    plt.close(fig)
    return frame


def render_unit_video_from_minute_cube(
    cube: dict[str, np.ndarray | list[str]],
    pair_i: int,
    out_mp4: Path,
    *,
    run_root: Path,
    fps: int = VIDEO_FPS,
    overwrite: bool = True,
    progress_every_frames: int = 300,
) -> None:
    if out_mp4.exists() and not overwrite:
        print(f"  [skip] exists: {out_mp4}", flush=True)
        return

    unit_t0 = time.perf_counter()
    mean_waveforms = np.asarray(cube["mean_waveform_uv"])[:, pair_i, :]
    n_minutes, wf_len = mean_waveforms.shape
    fs = float(np.asarray(cube["sampling_rate_hz"])[0])
    pre_samples, _post_samples = load_run_prepost_samples(run_root, wf_len, fs)
    t_ms = (np.arange(wf_len, dtype=np.float64) - pre_samples) / fs * 1000.0

    finite_vals = mean_waveforms[np.isfinite(mean_waveforms)]
    if finite_vals.size:
        lo, hi = np.nanpercentile(finite_vals, [1.0, 99.0])
        pad = max(5.0, 0.15 * float(hi - lo))
        y_limits = (float(lo - pad), float(hi + pad))
        if not np.isfinite(y_limits[0]) or not np.isfinite(y_limits[1]) or y_limits[0] == y_limits[1]:
            center = float(np.nanmean(finite_vals))
            y_limits = (center - 50.0, center + 50.0)
    else:
        y_limits = (-100.0, 100.0)

    sg_ch = int(np.asarray(cube["sg_ch"])[pair_i])
    thr_min = float(np.asarray(cube["threshold_min_uv"])[pair_i])
    thr_max = float(np.asarray(cube["threshold_max_uv"])[pair_i])
    unit_label = pair_label(sg_ch, thr_min, thr_max)
    recording_names = cube["recording_names"]
    if not isinstance(recording_names, list):
        recording_names = [str(x) for x in recording_names]

    fig, ax = plt.subplots(figsize=FRAME_FIGSIZE, dpi=FRAME_DPI)
    canvas = FigureCanvas(fig)
    (line,) = ax.plot(t_ms, np.full(wf_len, np.nan, dtype=np.float32), color="black", linewidth=2.2)
    empty_text = ax.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    recording_text = ax.text(
        0.01,
        0.01,
        "",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="0.35",
    )
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Time around crossing [ms]")
    ax.set_ylabel("Mean waveform [uV]")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg_exe = resolve_ffmpeg_exe()
        output_path = out_mp4
        writer_kwargs = {
            "fps": fps,
            "codec": "libx264",
            "quality": 8,
            "macro_block_size": 2,
            "pixelformat": "yuv420p",
        }
    except FileNotFoundError as e:
        output_path = out_mp4.with_suffix(".gif")
        ffmpeg_exe = None
        writer_kwargs = {"mode": "I", "duration": 1.0 / max(1, fps)}
        print(
            f"  [warn] {e}\n"
            f"  Falling back to GIF rendering instead: {output_path.name}",
            flush=True,
        )

    old_env = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if ffmpeg_exe is not None:
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_exe
    try:
        writer = imageio.get_writer(str(output_path), **writer_kwargs)
        try:
            for frame_i in range(n_minutes):
                frame_t0 = time.perf_counter()
                wf = mean_waveforms[frame_i]
                if np.any(np.isfinite(wf)):
                    line.set_ydata(wf)
                    empty_text.set_text("")
                else:
                    line.set_ydata(np.full(wf_len, np.nan, dtype=np.float32))
                    empty_text.set_text("No crossings in this minute")

                cumulative_sec = float(np.asarray(cube["cumulative_start_sec"])[frame_i])
                elapsed_h = cumulative_sec / 3600.0
                day = int(cumulative_sec // 86400) + 1
                hour = int((cumulative_sec % 86400) // 3600)
                minute = int((cumulative_sec % 3600) // 60)
                n_spikes = int(np.asarray(cube["n_spikes"])[frame_i, pair_i])
                firing_rate_hz = float(np.asarray(cube["firing_rate_hz"])[frame_i, pair_i])
                amplitude_ptp_uv = float(np.asarray(cube["amplitude_ptp_uv"])[frame_i, pair_i])
                amp_text = "nan" if not np.isfinite(amplitude_ptp_uv) else f"{amplitude_ptp_uv:.1f}"
                ax.set_title(
                    f"{unit_label}\n"
                    f"frame {frame_i} | elapsed {elapsed_h:.2f} h | day {day}, {hour:02d}:{minute:02d} | "
                    f"N={n_spikes} | FR={firing_rate_hz:.3f} Hz | p2p={amp_text} uV",
                    fontsize=9,
                )
                recording_text.set_text(str(recording_names[frame_i]))
                canvas.draw()
                frame = _canvas_to_rgb(canvas)
                writer.append_data(frame)

                done = frame_i + 1
                should_report = (
                    done == 1
                    or done == n_minutes
                    or (progress_every_frames > 0 and done % progress_every_frames == 0)
                )
                if should_report:
                    elapsed = time.perf_counter() - unit_t0
                    frames_per_sec = done / max(elapsed, 1e-9)
                    remaining_frames = max(0, n_minutes - done)
                    eta_sec = remaining_frames / max(frames_per_sec, 1e-9)
                    print(
                        f"    frames {done}/{n_minutes} "
                        f"({100.0 * done / max(1, n_minutes):.1f}%) | "
                        f"elapsed {format_duration(elapsed)} | "
                        f"ETA {format_duration(eta_sec)} | "
                        f"last frame {format_duration(time.perf_counter() - frame_t0)}",
                        flush=True,
                    )
        finally:
            writer.close()
    finally:
        plt.close(fig)
        if old_env is None:
            os.environ.pop("IMAGEIO_FFMPEG_EXE", None)
        else:
            os.environ["IMAGEIO_FFMPEG_EXE"] = old_env
    print(f"  Done unit in {format_duration(time.perf_counter() - unit_t0)}", flush=True)


def render_videos_from_threshold_channel_results(
    run_roots: list[Path],
    *,
    output_root: Path,
    fps: int = VIDEO_FPS,
    overwrite: bool = True,
    limit_units: int | None = None,
    progress_every_frames: int = 300,
) -> None:
    run_t0 = time.perf_counter()
    print(f"\n=== Combined input folders: {len(run_roots)} ===", flush=True)
    for i, run_root in enumerate(run_roots, start=1):
        print(f"  input {i}: {run_root}", flush=True)
    print(f"Output folder: {output_root}", flush=True)
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    load_t0 = time.perf_counter()
    npz_paths = collect_minute_npz_from_run_roots(run_roots)
    print(f"Loading {len(npz_paths)} minute NPZ frame(s) from {len(run_roots)} folder(s)...", flush=True)
    cube = load_minute_waveform_cube(npz_paths, contiguous_elapsed_time=len(run_roots) > 1)
    print(f"Loaded minute NPZ data in {format_duration(time.perf_counter() - load_t0)}", flush=True)
    pair_ids = cube["pair_ids"]
    if not isinstance(pair_ids, list):
        pair_ids = [str(x) for x in pair_ids]
    n_units = len(pair_ids)
    n_minutes = int(np.asarray(cube["mean_waveform_uv"]).shape[0])
    duration_sec = n_minutes / float(fps)
    print(
        f"Found {n_units} unit(s) x {n_minutes} minute-frame(s). "
        f"Each output video will be {duration_sec:.1f} s at {fps} fps.",
        flush=True,
    )

    if limit_units is not None:
        n_units = min(n_units, int(limit_units))

    for pair_i in range(n_units):
        folder_elapsed = time.perf_counter() - run_t0
        avg_unit_sec = folder_elapsed / max(1, pair_i)
        eta_units = max(0, n_units - pair_i) * avg_unit_sec if pair_i > 0 else float("nan")
        sg_ch = int(np.asarray(cube["sg_ch"])[pair_i])
        thr_min = float(np.asarray(cube["threshold_min_uv"])[pair_i])
        thr_max = float(np.asarray(cube["threshold_max_uv"])[pair_i])
        folder = pair_folder_name(sg_ch, thr_min, thr_max)
        out_dir = output_root / folder
        out_mp4 = out_dir / f"{folder}_combined_minute_mean_waveforms_{fps}fps.mp4"
        eta_text = "unknown" if not np.isfinite(eta_units) else format_duration(eta_units)
        print(
            f"Rendering unit video [{pair_i + 1}/{n_units}] "
            f"(folder elapsed {format_duration(folder_elapsed)}, unit ETA {eta_text}) -> {out_mp4}",
            flush=True,
        )
        render_unit_video_from_minute_cube(
            cube,
            pair_i,
            out_mp4,
            run_root=run_roots[0],
            fps=fps,
            overwrite=overwrite,
            progress_every_frames=progress_every_frames,
        )

    print(
        f"Finished combined render in {format_duration(time.perf_counter() - run_t0)} "
        f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )


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


def legacy_chunk_png_main():
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render one video per channel/threshold unit from Threshold_channel.py "
            "recording-level minute NPZ outputs. Each recording minute becomes one "
            f"video frame; default FPS is {VIDEO_FPS}."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more Threshold_channel.py run folders. "
            "If omitted, prompts interactively; separate multiple prompted paths with semicolons."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=VIDEO_FPS,
        help="Output video frames per second. Default: 60.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip an output MP4 if it already exists.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Folder for combined videos. Default: the input run folder for one input, "
            "or combined_threshold_videos_YYYYMMDD_HHMMSS next to the first input for multiple inputs."
        ),
    )
    parser.add_argument(
        "--limit-units",
        type=int,
        default=None,
        help="Debug option: render only the first N units.",
    )
    parser.add_argument(
        "--progress-every-frames",
        type=int,
        default=300,
        help="Print per-unit frame progress every N frames. Use 0 to print first/last only.",
    )
    parser.add_argument(
        "--legacy-chunk-pngs",
        action="store_true",
        help="Use the older chunk PNG renderer/peak-to-peak plot workflow.",
    )
    args = parser.parse_args()

    if args.legacy_chunk_pngs:
        legacy_chunk_png_main()
        return 0

    if args.fps <= 0:
        raise ValueError("--fps must be positive.")

    run_roots = parse_run_root_inputs(args.run_root)
    output_root = args.output_root if args.output_root is not None else default_combined_output_root(run_roots)
    total_t0 = time.perf_counter()
    print(
        f"Combining {len(run_roots)} run folder(s). "
        f"Overall start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )
    for run_root in run_roots:
        if not run_root.exists():
            raise FileNotFoundError(run_root)
    output_root.mkdir(parents=True, exist_ok=True)
    render_videos_from_threshold_channel_results(
        run_roots,
        output_root=output_root,
        fps=int(args.fps),
        overwrite=not args.no_overwrite,
        limit_units=args.limit_units,
        progress_every_frames=int(args.progress_every_frames),
    )

    print(
        f"\nCombined videos finished in {format_duration(time.perf_counter() - total_t0)} "
        f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
