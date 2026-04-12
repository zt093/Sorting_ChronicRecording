"""
Scan SpikeGadgets .rec files under a root in time chunks, detect threshold crossings
on one SG channel (0-based electrode id from probe contact_ids), extract waveforms
(-1 ms .. +2 ms around the crossing sample), and save timestamps + waveforms.

Pipeline mirrors SortingLSNET_Feb2026.py for SG data:
  read_spikegadgets -> optional 30 kHz wrapper -> HW channel reorder (forward_conversion)
  -> optional reversed-cable group swap -> set_probe(LSNET_probe.json)
  -> optional spikeband bandpass (default 300–6000 Hz, same as sorting preprocessing)
  -> threshold detection and waveforms on the filtered trace.

Interactive use:
  python sg_channel_threshold_crossings.py

Recording selection expects Trodes-style chronic filenames:
  Chronic_Rec_YYYYMMDD_HHMMSS.rec  (e.g. Chronic_Rec_20260320_104430.rec)
You enter the first and last recording (filename or YYYYMMDD_HHMMSS); all chronic
files in that inclusive range (by timestamp in the name) are processed.

How outputs are saved (each run gets its own folder):
  <output_parent>/threshold_crossings_run_YYYYMMDD_HHMMSS/
    run_config.json
    Per recording (stem = sanitized parent + recording name):
      <stem>_recording_summary.json — total events, list of chunk artifacts
      <stem>_chunk_NNNN_threshold_crossings.npz — one per time chunk (then RAM released)
      <stem>_chunk_NNNN_...png — waveform overlay plot per chunk (see naming below)
  Each chunk .npz: crossing_samples/timestamps (file-local), *_cumulative (multi-session
    timeline), waveforms_uv, chunk_index, time_start_sec, time_end_sec.
  First successfully loaded recording: *_first10s_sgch*_filtered_trace_preview.png (10 s
    of the same channel/signal used for detection, after optional bandpass).
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import gc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
from probeinterface import read_probeinterface
from spikeinterface.core import BaseRecording


# ---------------------------------------------------------------------------
# Recording I/O (aligned with SortingLSNET_Feb2026.py)
# ---------------------------------------------------------------------------


class CustomSamplingFrequencyRecording(BaseRecording):
    """Force sampling frequency (same pattern as SortingLSNET_Feb2026)."""

    def __init__(self, recording, new_sampling_frequency):
        BaseRecording.__init__(
            self,
            sampling_frequency=new_sampling_frequency,
            channel_ids=recording.channel_ids,
            dtype=recording.get_dtype(),
        )
        for segment in recording._recording_segments:
            self.add_recording_segment(segment)
        self._kwargs = getattr(recording, "_kwargs", {})
        for key in recording.get_property_keys():
            self.set_property(key, recording.get_property(key))


def resolve_spikegadgets_rec_file(input_path: Path) -> Path:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Recording path does not exist: {input_path}")
    if input_path.suffix.lower() != ".rec":
        raise ValueError(
            f"Recording path must end with '.rec': {input_path}. "
            "Provide a .rec file or a .rec directory."
        )
    if input_path.is_file():
        return input_path
    if input_path.is_dir():
        nested_same_name_file = input_path / input_path.name
        if (
            nested_same_name_file.exists()
            and nested_same_name_file.is_file()
            and nested_same_name_file.suffix.lower() == ".rec"
        ):
            return nested_same_name_file
        rec_file_candidates = sorted([p for p in input_path.glob("*.rec") if p.is_file()])
        if len(rec_file_candidates) == 1:
            return rec_file_candidates[0]
        if len(rec_file_candidates) == 0:
            raise ValueError(
                f"No .rec file found inside .rec directory: {input_path}. "
                "Expected nested '<dir>/<dir>.rec' or exactly one *.rec file."
            )
        raise ValueError(
            f"Multiple .rec files inside directory: {input_path}. "
            "Point to an explicit .rec file. "
            f"Candidates: {[str(p) for p in rec_file_candidates]}"
        )
    raise ValueError(f"Unsupported recording path type: {input_path}")


def forward_conversion(hw_chan: int, totalchan: int) -> int:
    num_cards = totalchan // 32
    return ((hw_chan % 32) * num_cards) + (hw_chan // 32)


def reverse_conversion(new_hw_chan: int, totalchan: int) -> int:
    num_cards = totalchan // 32
    return (new_hw_chan % num_cards) * 32 + (new_hw_chan // num_cards)


def apply_hw_channel_map(rec_raw: BaseRecording) -> BaseRecording:
    totalchan = rec_raw.get_num_channels()
    channel_ids_rec = rec_raw.get_channel_ids()
    new_hw_chans = [forward_conversion(hw_chan, totalchan) for hw_chan in range(totalchan)]
    new_channel_order = [channel_ids_rec[c] for c in new_hw_chans]
    rec_hwmapped = rec_raw.select_channels(new_channel_order)
    return rec_hwmapped


def maybe_swap_reversed_cable_groups(rec_hwmapped: BaseRecording, rec_file_path: Path) -> BaseRecording:
    if not rec_file_path.name.endswith("reversed.rec"):
        return rec_hwmapped
    num_channels = rec_hwmapped.get_num_channels()
    if num_channels != 384:
        return rec_hwmapped
    group_size = num_channels // 3
    ids = list(rec_hwmapped.channel_ids)
    new_order = ids[2 * group_size :] + ids[group_size : 2 * group_size] + ids[:group_size]
    return rec_hwmapped.select_channels(new_order)


def load_recording_mapped(
    rec_path: Path,
    sampling_rate_hz: float,
    probe_path: Path,
) -> BaseRecording:
    rec_path = resolve_spikegadgets_rec_file(rec_path)
    rec_loaded = si.read_spikegadgets(file_path=str(rec_path))
    rec_loaded = CustomSamplingFrequencyRecording(rec_loaded, sampling_rate_hz)
    rec_hw = apply_hw_channel_map(rec_loaded)
    rec_hw = maybe_swap_reversed_cable_groups(rec_hw, rec_path)
    pi = read_probeinterface(str(probe_path))
    probe = pi.probes[0]
    n_rec = rec_hw.get_num_channels()
    n_prb = int(probe.get_contact_count())
    if n_rec != n_prb:
        raise ValueError(
            f"Recording has {n_rec} channels but probe has {n_prb} contacts; "
            "set_probe would misalign SG indices. Use a matching probe JSON or a different script path."
        )
    rec_hw.set_probe(probe, in_place=True)
    return rec_hw


def build_sg_to_recording_index(probe) -> dict[int, int]:
    """SG channel (0-based) -> integer index into recording.get_channel_ids()."""
    sg_to_idx: dict[int, int] = {}
    for i in range(len(probe.contact_ids)):
        sg_ch = int(probe.contact_ids[i]) - 1
        rec_idx = int(probe.device_channel_indices[i])
        if sg_ch in sg_to_idx and sg_to_idx[sg_ch] != rec_idx:
            raise ValueError(f"Duplicate SG channel map for sg_ch={sg_ch}")
        sg_to_idx[sg_ch] = rec_idx
    return sg_to_idx


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def find_threshold_crossings_down(
    x: np.ndarray,
    thresh: float,
    start_rel: int,
    end_rel: int,
) -> np.ndarray:
    """
    Downward crossing: x[k-1] > -thresh and x[k] <= -thresh (extracellular-style).
    Search k in [start_rel, end_rel) (end_rel exclusive), requires start_rel >= 1.
    """
    if end_rel <= start_rel or start_rel < 1:
        return np.array([], dtype=np.int64)
    prev = x[start_rel - 1 : end_rel - 1]
    cur = x[start_rel:end_rel]
    hits = np.nonzero((prev > -thresh) & (cur <= -thresh))[0] + start_rel
    return hits.astype(np.int64)


def find_threshold_crossings_up(
    x: np.ndarray,
    thresh: float,
    start_rel: int,
    end_rel: int,
) -> np.ndarray:
    if end_rel <= start_rel or start_rel < 1:
        return np.array([], dtype=np.int64)
    prev = x[start_rel - 1 : end_rel - 1]
    cur = x[start_rel:end_rel]
    hits = np.nonzero((prev < thresh) & (cur >= thresh))[0] + start_rel
    return hits.astype(np.int64)


def merge_refractory(sorted_indices: np.ndarray, refractory_samples: int) -> np.ndarray:
    if sorted_indices.size == 0:
        return sorted_indices
    out = [int(sorted_indices[0])]
    for v in sorted_indices[1:]:
        v = int(v)
        if v - out[-1] >= refractory_samples:
            out.append(v)
    return np.array(out, dtype=np.int64)


def _ensure_event_capacity(
    cross_buf: np.ndarray,
    wf_buf: np.ndarray,
    n_events: int,
    wf_len: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Double capacity when full; avoids millions of small list appends (GC + slowdown)."""
    cap = cross_buf.shape[0]
    if n_events < cap:
        return cross_buf, wf_buf, cap
    new_cap = max(cap * 2, cap + 65536)
    cb = np.empty(new_cap, dtype=np.int64)
    wb = np.empty((new_cap, wf_len), dtype=np.float32)
    if n_events > 0:
        cb[:n_events] = cross_buf[:n_events]
        wb[:n_events] = wf_buf[:n_events]
    return cb, wb, new_cap


def _minutes_tag(t0_min: float, t1_min: float) -> str:
    """Filename-safe minute range, e.g. min12p3-18p0 (decimal point -> p)."""

    def _fmt(m: float) -> str:
        return f"{m:.1f}".replace(".", "p")

    return f"min{_fmt(t0_min)}-{_fmt(t1_min)}"


def _save_chunk_waveform_plot(
    waveforms_uv: np.ndarray,
    *,
    fs: float,
    pre_samples: int,
    post_samples: int,
    sg_ch: int,
    t_start_min: float,
    t_end_min: float,
    out_png: Path,
    plot_max_traces: int = 12000,
    fixed_y_center: float = 0.0,
    fixed_y_half: float = 1.0,
    time_bar_ms: float = 1000.0,
    amp_bar_fraction: float = 0.25,
) -> None:
    """All waveforms (thin); bold mean overlay with fixed y scale + scale bars."""
    wf_len = pre_samples + post_samples
    t_ms = (np.arange(wf_len, dtype=np.float64) - pre_samples) / fs * 1000.0
    n = waveforms_uv.shape[0]
    y_span = 2.0 * float(fixed_y_half)
    y_min = float(fixed_y_center - fixed_y_half)
    y_max = float(fixed_y_center + fixed_y_half)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    if n == 0:
        ax.text(
            0.5,
            0.5,
            "No crossings in this chunk",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
    else:
        mean_wf = waveforms_uv.mean(axis=0)
        mean_abs = float(np.mean(np.abs(mean_wf)))
        plot_wf = waveforms_uv
        shown = n
        if n > plot_max_traces:
            rng = np.random.default_rng(0)
            sel = rng.choice(n, size=plot_max_traces, replace=False)
            plot_wf = waveforms_uv[sel]
            shown = plot_max_traces
        alpha = min(0.25, max(0.02, 8.0 / max(shown, 1)))
        n_show = plot_wf.shape[0]
        seg = np.empty((n_show, wf_len, 2), dtype=np.float32)
        seg[:, :, 0] = t_ms.astype(np.float32)
        seg[:, :, 1] = plot_wf.astype(np.float32, copy=False)
        lc = LineCollection(
            seg,
            colors="0.55",
            alpha=alpha,
            linewidths=0.35,
            rasterized=True,
        )
        ax.add_collection(lc)
        ax.plot(
            t_ms,
            mean_wf,
            color="k",
            linewidth=2.8,
            zorder=10,
        )

    # Fixed y limits across all chunks.
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))

    # Remove axes/ticks; use explicit scale bars instead.
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Compose title (keep minute info).
    if n == 0:
        title = (
            f"SG ch {sg_ch}  |  recording {t_start_min:.2f}–{t_end_min:.2f} min  "
            f"|  N=0"
        )
    else:
        title = (
            f"SG ch {sg_ch}  |  recording {t_start_min:.2f}–{t_end_min:.2f} min  "
            f"|  N={n}"
            + (f"  (plot shows {shown})" if shown < n else "")
            + f"  |  mean|µV|={mean_abs:.1f}"
        )
    ax.set_title(title, fontsize=10, pad=6)

    # Scale bars
    time_span_ms = float(t_ms[-1] - t_ms[0])
    if time_span_ms > 0 and y_span > 0:
        t_bar_ms = min(float(time_bar_ms), 0.3 * time_span_ms)
        x0 = float(t_ms[0]) + 0.06 * time_span_ms
        x1 = x0 + t_bar_ms
        y0 = y_min + 0.06 * y_span

        # Amplitude scale bar
        amp_bar = max(1e-6, float(amp_bar_fraction) * y_span)
        # keep bar inside axes
        if y0 + amp_bar > y_max:
            y0 = y_max - amp_bar - 0.02 * y_span
        x_bar = float(t_ms[0]) + 0.92 * time_span_ms
        ax.plot([x0, x1], [y0, y0], color="k", linewidth=3, solid_capstyle="butt", zorder=20)
        ax.plot([x_bar, x_bar], [y0, y0 + amp_bar], color="k", linewidth=3, zorder=20)

        # Labels
        if t_bar_ms >= 1000.0:
            t_lbl = f"{t_bar_ms / 1000.0:.1f} s"
        else:
            t_lbl = f"{t_bar_ms:.0f} ms"
        ax.text((x0 + x1) / 2.0, y0 - 0.03 * y_span, t_lbl, ha="center", va="top", fontsize=9)

        ax.text(
            x_bar + 0.01 * time_span_ms,
            y0 + 0.5 * amp_bar,
            f"{amp_bar:.0f} uV",
            ha="left",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_first_10s_trace_preview(
    rec: BaseRecording,
    channel_id,
    *,
    out_png: Path,
    fs: float,
    duration_sec: float,
    sg_ch: int,
    recording_stem: str,
    apply_spikeband: bool,
    bandpass_freq_min: float,
    bandpass_freq_max: float,
) -> None:
    """Plot first N seconds on the analysis channel (filtered trace if bandpass was applied)."""
    n_avail = rec.get_num_samples()
    n_plot = min(n_avail, int(round(duration_sec * fs)))
    if n_plot <= 0:
        return
    traces = rec.get_traces(
        start_frame=0,
        end_frame=n_plot,
        channel_ids=[channel_id],
        return_scaled=True,
    )
    y = traces[:, 0].astype(np.float64, copy=False)
    t = np.arange(n_plot, dtype=np.float64) / fs
    fig, ax = plt.subplots(figsize=(12, 4), dpi=120)
    ax.plot(t, y, color="0.2", linewidth=0.35, rasterized=True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("µV")
    if apply_spikeband:
        filt_lbl = f"spikeband {bandpass_freq_min:.0f}–{bandpass_freq_max:.0f} Hz"
    else:
        filt_lbl = "raw scaled (no bandpass)"
    ax.set_title(
        f"First {n_plot / fs:.3f} s  |  SG ch {sg_ch}  |  {recording_stem}  |  {filt_lbl}",
        fontsize=10,
    )
    ax.set_xlim(0.0, n_plot / fs)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def process_recording_save_per_chunk(
    rec: BaseRecording,
    channel_id,
    *,
    out_base: Path,
    sg_ch: int,
    session_cumulative_sample_offset: int,
    session_cumulative_time_offset_sec: float,
    fs: float,
    chunk_samples: int,
    threshold_uv: float,
    polarity: str,
    refractory_samples: int,
    pre_samples: int,
    post_samples: int,
    progress: bool = True,
    progress_prefix: str = "  ",
    resume: bool = False,
) -> tuple[int, list[dict]]:
    """
    Detect crossings in time chunks; after each chunk save .npz + waveform PNG, then free arrays.

    Returns (total_event_count_across_chunks, chunk_manifest_rows).
    """
    n = rec.get_num_samples()
    wf_len = pre_samples + post_samples
    last_kept = -10**18
    n_chunks = max(1, (n + chunk_samples - 1) // chunk_samples)
    chunk_i = 0
    c0 = 0
    t_detect0 = time.perf_counter()
    total_events = 0
    manifest: list[dict] = []
    fixed_y_center: float | None = None
    fixed_y_half: float | None = None

    while c0 < n:
        chunk_i += 1
        c1 = min(n, c0 + chunk_samples)
        t_chunk0 = time.perf_counter()

        # Chunk output paths/labels (used for resume to skip already computed chunks).
        chunk_tag = f"chunk_{chunk_i:04d}"
        npz_path = Path(f"{out_base}_{chunk_tag}_threshold_crossings.npz")
        t_start_min = (c0 / fs) / 60.0
        t_end_min = (min(c1, n) / fs) / 60.0
        min_tag = _minutes_tag(t_start_min, t_end_min)

        if resume and npz_path.exists():
            npz = None
            wfs_uv = None
            mean_abs_for_name = 0.0
            try:
                npz = np.load(str(npz_path), allow_pickle=False)
                crossings_samples = npz["crossing_samples"]
                n_chunk = int(crossings_samples.shape[0])

                # Match the original fixed-y logic (including its default-scale behavior).
                if fixed_y_center is None or fixed_y_half is None:
                    if n_chunk > 0:
                        wfs_uv = npz["waveforms_uv"]
                        mean_wf = wfs_uv.mean(axis=0)
                        mean_min = float(np.min(mean_wf))
                        mean_max = float(np.max(mean_wf))
                        mean_range = mean_max - mean_min
                        if mean_range <= 0:
                            mean_range = max(1e-6, float(np.max(np.abs(mean_wf))))
                        y_span = 1.2 * mean_range
                        fixed_y_half = y_span / 2.0
                        fixed_y_center = 0.5 * (mean_min + mean_max)
                        mean_abs_for_name = float(np.mean(np.abs(mean_wf)))
                    else:
                        fixed_y_center = 0.0
                        fixed_y_half = max(1.0, 1.2 * float(threshold_uv))

                fig_path: Path | None = None
                fig_pattern = (
                    f"{out_base.name}_{chunk_tag}_sgch{sg_ch}_{min_tag}_"
                    f"n{n_chunk}_meanAbs*uV_waveforms.png"
                )
                fig_candidates = list(out_base.parent.glob(fig_pattern))
                if fig_candidates:
                    fig_path = fig_candidates[0]

                if fig_path is None or not fig_path.exists():
                    # Need to (re)generate the waveform PNG for this chunk.
                    if n_chunk > 0:
                        if wfs_uv is None:
                            wfs_uv = npz["waveforms_uv"]
                        mean_wf = wfs_uv.mean(axis=0)
                        mean_abs_for_name = float(np.mean(np.abs(mean_wf)))
                    else:
                        wfs_uv = np.zeros((0, wf_len), dtype=np.float32)
                        mean_abs_for_name = 0.0

                    fig_name = (
                        f"{out_base.name}_{chunk_tag}_sgch{sg_ch}_{min_tag}_"
                        f"n{n_chunk}_meanAbs{mean_abs_for_name:.0f}uV_waveforms.png"
                    )
                    fig_path = out_base.parent / fig_name
                    _save_chunk_waveform_plot(
                        wfs_uv,
                        fs=fs,
                        pre_samples=pre_samples,
                        post_samples=post_samples,
                        sg_ch=sg_ch,
                        t_start_min=t_start_min,
                        t_end_min=t_end_min,
                        out_png=fig_path,
                        fixed_y_center=float(fixed_y_center),
                        fixed_y_half=float(fixed_y_half),
                    )

                manifest.append(
                    {
                        "chunk_index": chunk_i,
                        "n_crossings": int(n_chunk),
                        "time_start_sec": float(c0 / fs),
                        "time_end_sec": float(min(c1, n) / fs),
                        "npz": str(npz_path.resolve()),
                        "figure": str(fig_path.resolve()),
                    }
                )
                total_events += int(n_chunk)

                # Release any heavy arrays we loaded.
                del crossings_samples
                if wfs_uv is not None:
                    del wfs_uv
                gc.collect()

                c0 = c1
                continue
            except Exception:
                # Corrupt/unreadable NPZ -> recompute this chunk.
                pass
            finally:
                if npz is not None:
                    try:
                        npz.close()
                    except Exception:
                        pass

        buf_start = max(0, c0 - pre_samples - 1)
        buf_end = min(n, c1 + post_samples + 1)
        traces = rec.get_traces(
            start_frame=buf_start,
            end_frame=buf_end,
            channel_ids=[channel_id],
            return_scaled=True,
        )
        x = traces[:, 0].astype(np.float32, copy=False)

        det_start = max(c0, 1)
        det_end = c1
        start_rel = det_start - buf_start
        end_rel = det_end - buf_start

        if polarity == "negative":
            cand = find_threshold_crossings_down(x, threshold_uv, start_rel, end_rel)
        elif polarity == "positive":
            cand = find_threshold_crossings_up(x, threshold_uv, start_rel, end_rel)
        elif polarity == "both":
            c1a = find_threshold_crossings_down(x, threshold_uv, start_rel, end_rel)
            c1b = find_threshold_crossings_up(x, threshold_uv, start_rel, end_rel)
            cand = np.unique(np.concatenate([c1a, c1b]))
        else:
            raise ValueError(f"Unknown polarity: {polarity}")

        global_cross = cand.astype(np.int64) + buf_start
        global_cross.sort()

        cap = 1024
        cross_buf = np.empty(cap, dtype=np.int64)
        wf_buf = np.empty((cap, wf_len), dtype=np.float32)
        n_chunk = 0
        for g in merge_refractory(global_cross, refractory_samples):
            if g - last_kept < refractory_samples:
                continue
            loc = int(g - buf_start)
            if loc < pre_samples or loc + post_samples > x.shape[0]:
                continue
            cross_buf, wf_buf, cap = _ensure_event_capacity(cross_buf, wf_buf, n_chunk, wf_len)
            cross_buf[n_chunk] = int(g)
            wf_buf[n_chunk, :] = x[loc - pre_samples : loc + post_samples]
            n_chunk += 1
            last_kept = int(g)

        del traces, x

        t_start_min = (c0 / fs) / 60.0
        t_end_min = (min(c1, n) / fs) / 60.0
        mean_abs_for_name = 0.0
        if n_chunk > 0:
            mean_wf = wf_buf[:n_chunk].mean(axis=0)
            mean_abs_for_name = float(np.mean(np.abs(mean_wf)))
            if fixed_y_center is None or fixed_y_half is None:
                # Fixed y scale = 1.2 x (max(mean) - min(mean)) of the mean waveform
                # from the first chunk that contains at least one crossing.
                mean_min = float(np.min(mean_wf))
                mean_max = float(np.max(mean_wf))
                mean_range = mean_max - mean_min
                if mean_range <= 0:
                    mean_range = max(1e-6, float(np.max(np.abs(mean_wf))))
                y_span = 1.2 * mean_range
                fixed_y_half = y_span / 2.0
                fixed_y_center = 0.5 * (mean_min + mean_max)
        # If the whole recording had zero crossings, keep a sane default scale.
        if fixed_y_center is None or fixed_y_half is None:
            fixed_y_center = 0.0
            fixed_y_half = max(1.0, 1.2 * float(threshold_uv))

        chunk_tag = f"chunk_{chunk_i:04d}"
        npz_path = Path(f"{out_base}_{chunk_tag}_threshold_crossings.npz")
        crossings = cross_buf[:n_chunk].copy() if n_chunk else np.zeros(0, dtype=np.int64)
        wfs = wf_buf[:n_chunk].copy() if n_chunk else np.zeros((0, wf_len), dtype=np.float32)
        del cross_buf, wf_buf

        ts_sec = crossings.astype(np.float64) / fs
        crossing_samples_cumulative = crossings.astype(np.int64, copy=True) + int(
            session_cumulative_sample_offset
        )
        timestamps_sec_cumulative = ts_sec + float(session_cumulative_time_offset_sec)

        np.savez_compressed(
            str(npz_path),
            crossing_samples=crossings,
            crossing_samples_cumulative=crossing_samples_cumulative,
            timestamps_sec=ts_sec,
            timestamps_sec_cumulative=timestamps_sec_cumulative,
            waveforms_uv=wfs,
            sampling_rate_hz=np.array([fs]),
            chunk_index=np.array([chunk_i], dtype=np.int32),
            time_start_sec=np.array([c0 / fs], dtype=np.float64),
            time_end_sec=np.array([min(c1, n) / fs], dtype=np.float64),
        )

        min_tag = _minutes_tag(t_start_min, t_end_min)
        fig_name = (
            f"{out_base.name}_{chunk_tag}_sgch{sg_ch}_{min_tag}_"
            f"n{n_chunk}_meanAbs{mean_abs_for_name:.0f}uV_waveforms.png"
        )
        fig_path = out_base.parent / fig_name
        _save_chunk_waveform_plot(
            wfs,
            fs=fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
            sg_ch=sg_ch,
            t_start_min=t_start_min,
            t_end_min=t_end_min,
            out_png=fig_path,
            fixed_y_center=float(fixed_y_center),
            fixed_y_half=float(fixed_y_half),
        )

        manifest.append(
            {
                "chunk_index": chunk_i,
                "n_crossings": int(n_chunk),
                "time_start_sec": float(c0 / fs),
                "time_end_sec": float(min(c1, n) / fs),
                "npz": str(npz_path.resolve()),
                "figure": str(fig_path.resolve()),
            }
        )
        total_events += n_chunk

        del crossings, wfs, ts_sec, crossing_samples_cumulative, timestamps_sec_cumulative
        gc.collect()

        if progress:
            t_end = min(c1, n) / fs
            pct = 100.0 * t_end / (n / fs) if n else 100.0
            chunk_wall = time.perf_counter() - t_chunk0
            elapsed = time.perf_counter() - t_detect0
            print(
                f"{progress_prefix}chunk {chunk_i}/{n_chunks}  "
                f"rec time {c0/fs:.2f}–{t_end:.2f} s  ({pct:.1f}%)  "
                f"chunk events {n_chunk}  total {total_events}  "
                f"chunk {chunk_wall:.2f}s  elapsed {elapsed:.1f}s  (saved+released)",
                flush=True,
            )

        c0 = c1

    return total_events, manifest


CHRONIC_REC_NAME_RE = re.compile(
    r"^Chronic_Rec_(?P<ymd>\d{8})_(?P<hms>\d{6})\.rec$",
    re.IGNORECASE,
)


def chronic_rec_sort_key(path: Path) -> int | None:
    m = CHRONIC_REC_NAME_RE.match(path.name)
    if not m:
        return None
    return int(m.group("ymd")) * 1_000_000 + int(m.group("hms"))


def discover_chronic_rec_files(i_root: Path) -> list[Path]:
    """All Chronic_Rec_YYYYMMDD_HHMMSS.rec under root, sorted by embedded timestamp."""
    root = Path(i_root)
    if not root.exists():
        raise FileNotFoundError(f"Scan root not found: {root}")
    keyed: list[tuple[int, Path]] = []
    for p in root.rglob("*.rec"):
        if not p.is_file():
            continue
        k = chronic_rec_sort_key(p)
        if k is None:
            continue
        keyed.append((k, p.resolve()))
    keyed.sort(key=lambda t: t[0])
    # Stable unique paths (same key could theoretically collide; keep both)
    return [p for _, p in keyed]


def _format_threshold_for_folder(threshold_uv: float) -> str:
    """Filename/folder safe threshold string (e.g. 500.0 -> 500p0, 12.3 -> 12p3)."""
    s = f"{float(threshold_uv):.3f}".rstrip("0").rstrip(".")
    s = s.replace("-", "m").replace(".", "p")
    return s


def _recording_parent_stem_safe(rec_file: Path) -> tuple[str, str]:
    """Match the script's folder/file stem sanitization for outputs."""
    stem = rec_file.stem.replace(" ", "_")
    parent = rec_file.parent.name.replace(" ", "_")
    return parent, stem


def _pair_folder_name(sg_ch: int, threshold_uv: float) -> str:
    return f"sgch{int(sg_ch)}_thr{_format_threshold_for_folder(threshold_uv)}uV"


def _pair_out_base(run_output_dir: Path, rec_file: Path, sg_ch: int, threshold_uv: float) -> Path:
    parent, stem = _recording_parent_stem_safe(rec_file)
    pair_dir = run_output_dir / _pair_folder_name(sg_ch, threshold_uv)
    # Matches existing naming: <parent>__<stem>
    return pair_dir / f"{parent}__{stem}"


def _recording_summary_path_from_out_base(out_base: Path) -> Path:
    return out_base.parent / f"{out_base.name}_recording_summary.json"


def _is_recording_summary_complete(summary_path: Path) -> bool:
    """
    Decide whether a (recording, sg_ch, threshold_uv) pair is "done".

    We require the summary JSON to exist and every chunk NPZ referenced in it to exist too.
    """
    if not summary_path.exists():
        return False
    try:
        per_rec = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    chunks = per_rec.get("chunks", None)
    if not isinstance(chunks, list) or not chunks:
        return False
    for ch in chunks:
        if not isinstance(ch, dict):
            return False
        npz_s = ch.get("npz", None)
        if not npz_s:
            return False
        if not Path(npz_s).exists():
            return False
        fig_s = ch.get("figure", None)
        if not fig_s:
            return False
        if not Path(fig_s).exists():
            return False
    return True


def load_channel_threshold_pairs(config_path: Path) -> list[dict]:
    """
    Load (sg_ch, threshold_uv) pairs from JSON.

    Supported formats:
      1) JSON list:
         [{"sg_ch": 72, "threshold_uv": 500.0}, {"sg_ch": 73, "threshold_uv": 600.0}]
      2) JSON object with "pairs":
         {"pairs": [{"sg_ch": 72, "threshold_uv": 500.0}, ...]}
      3) JSON mapping sg_ch -> threshold:
         {"72": 500.0, "73": 600.0}

    Each entry must provide:
      - sg_ch (or "channel") as int
      - threshold_uv (or "threshold") as float (magnitude, must be > 0)
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    pairs: list[dict] = []

    def _normalize_one(obj: dict) -> dict:
        sg_ch = obj.get("sg_ch", obj.get("channel", None))
        thr = obj.get("threshold_uv", obj.get("threshold", None))
        if sg_ch is None or thr is None:
            raise ValueError(
                "Each pair must include sg_ch (or channel) and threshold_uv (or threshold). "
                f"Got keys: {list(obj.keys())}"
            )
        sg_ch = int(sg_ch)
        thr = float(thr)
        if thr <= 0:
            raise ValueError(f"threshold_uv must be > 0 magnitude. Got: {thr} for sg_ch={sg_ch}")
        return {"sg_ch": sg_ch, "threshold_uv": thr}

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("When config is a list, each element must be a dict/object.")
            pairs.append(_normalize_one(item))
    elif isinstance(raw, dict):
        if "pairs" in raw and isinstance(raw["pairs"], list):
            for item in raw["pairs"]:
                if not isinstance(item, dict):
                    raise ValueError("config['pairs'] must be a list of objects.")
                pairs.append(_normalize_one(item))
        else:
            for k, v in raw.items():
                sg_ch = int(k)
                pairs.append(_normalize_one({"sg_ch": sg_ch, "threshold_uv": v}))
    else:
        raise ValueError("Config JSON must be a list or an object.")

    # De-duplicate identical (sg_ch, threshold_uv) pairs while preserving order.
    seen: set[tuple[int, float]] = set()
    out: list[dict] = []
    for p in pairs:
        key = (int(p["sg_ch"]), float(p["threshold_uv"]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"sg_ch": int(p["sg_ch"]), "threshold_uv": float(p["threshold_uv"])})
    return out


DEFAULT_CHANNEL_THRESHOLDS_EXAMPLE = [
    {"sg_ch": 72, "threshold_uv": 500.0},
    {"sg_ch": 74, "threshold_uv": 500.0},
]


def parse_chronic_rec_boundary_key(user_text: str) -> int:
    """
    Accepts:
      Chronic_Rec_20260320_104430.rec
      Chronic_Rec_20260320_104430
      20260320_104430
    Returns sort key YYYYMMDD * 1e6 + HHMMSS for range filtering.
    """
    s = user_text.strip().strip('"').strip("'")
    name = Path(s).name
    if not name.lower().endswith(".rec"):
        name = name + ".rec"
    m = CHRONIC_REC_NAME_RE.match(name)
    if m:
        return int(m.group("ymd")) * 1_000_000 + int(m.group("hms"))
    stem = Path(s).stem
    m2 = re.match(r"^(\d{8})_(\d{6})$", stem)
    if m2:
        return int(m2.group(1)) * 1_000_000 + int(m2.group(2))
    raise ValueError(
        "Expected Chronic_Rec_YYYYMMDD_HHMMSS.rec or YYYYMMDD_HHMMSS, "
        f"got: {user_text!r}"
    )


def filter_chronic_recs_in_range(
    files: list[Path],
    first_key: int,
    last_key: int,
) -> list[Path]:
    if first_key > last_key:
        first_key, last_key = last_key, first_key
    out: list[Path] = []
    for p in files:
        k = chronic_rec_sort_key(p)
        if k is None:
            continue
        if first_key <= k <= last_key:
            out.append(p)
    return out


def prompt_line(message: str, default: str | None = None) -> str:
    if default is not None and str(default).strip() != "":
        raw = input(f"{message} [{default}]: ").strip()
        return raw if raw else str(default)
    raw = input(f"{message}: ").strip()
    return raw


def prompt_int(message: str, default: int) -> int:
    while True:
        raw = prompt_line(message, str(default))
        try:
            return int(raw, 10)
        except ValueError:
            print("Please enter an integer.")


def prompt_float(message: str, default: float) -> float:
    while True:
        raw = prompt_line(message, str(default))
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def prompt_choice(message: str, choices: tuple[str, ...], default: str) -> str:
    choices_lower = {c.lower(): c for c in choices}
    opt = "/".join(choices)
    while True:
        raw = prompt_line(f"{message} ({opt})", default)
        key = raw.lower()
        if key in choices_lower:
            return choices_lower[key]
        print(f"Choose one of: {', '.join(choices)}")


def prompt_yes_no(message: str, default_yes: bool = True) -> bool:
    default = "y" if default_yes else "n"
    while True:
        raw = prompt_line(f"{message} (y/n)", default).lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please enter y or n.")


def make_run_output_dir(output_parent: Path) -> Path:
    """Distinct folder per run: <parent>/threshold_crossings_run_YYYYMMDD_HHMMSS"""
    output_parent = Path(output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_parent / f"threshold_crossings_run_{stamp}"
    if run_dir.exists():
        run_dir = output_parent / f"threshold_crossings_run_{stamp}_{int(time.time())}"
    run_dir.mkdir(parents=False)
    return run_dir


def process_threshold_crossings_run(
    *,
    run_output_dir: Path,
    meta_run: dict,
    rec_files: list[Path],
    fs: float,
    probe_json: Path,
    channel_threshold_pairs: list[dict],
    polarity: str,
    chunk_samples: int,
    refractory_samples: int,
    pre_samples: int,
    post_samples: int,
    apply_spikeband: bool,
    bandpass_freq_min: float,
    bandpass_freq_max: float,
    resume: bool,
) -> int:
    t0_all = time.perf_counter()
    n_files = len(rec_files)
    print(f"\nProcessing {n_files} recording(s)…\n", flush=True)
    print(
        "Cumulative timeline: sessions are chained in chronic-time order. "
        "NPZ fields *_cumulative place events on one continuous clock (see run_config.json).",
        flush=True,
    )
    print(
        "Each time chunk is written to disk (NPZ + waveform figure) and freed from RAM before "
        "the next chunk.\n",
        flush=True,
    )

    cumulative_sample_offset = 0
    cumulative_time_offset_sec = 0.0
    session_ordinal = 0

    # If resuming, seed the preview cache so we don't overwrite existing "first 10s" images.
    trace_preview_saved_for_pairs: set[str] = set()
    if resume:
        for pair in channel_threshold_pairs:
            sg_ch = int(pair["sg_ch"])
            threshold_uv = float(pair["threshold_uv"])
            pair_key = f"{sg_ch}__{threshold_uv:.9g}"
            pair_dir = run_output_dir / _pair_folder_name(sg_ch, threshold_uv)
            pattern = f"*_first10s_sgch{sg_ch}_filtered_trace_preview.png"
            if pair_dir.exists() and any(pair_dir.glob(pattern)):
                trace_preview_saved_for_pairs.add(pair_key)

    previews_written_for_first_loaded_record = False

    for fi, rec_file in enumerate(rec_files):
        t0 = time.perf_counter()
        print(f"--- [{fi + 1}/{n_files}] {rec_file.name} ---", flush=True)

        if resume:
            all_pairs_complete = True
            for pair in channel_threshold_pairs:
                sg_ch = int(pair["sg_ch"])
                threshold_uv = float(pair["threshold_uv"])
                out_base = _pair_out_base(run_output_dir, rec_file, sg_ch, threshold_uv)
                summary_path = _recording_summary_path_from_out_base(out_base)
                if not _is_recording_summary_complete(summary_path):
                    all_pairs_complete = False
                    break
            if all_pairs_complete:
                first_pair = channel_threshold_pairs[0]
                first_sg_ch = int(first_pair["sg_ch"])
                first_thr = float(first_pair["threshold_uv"])
                first_out_base = _pair_out_base(run_output_dir, rec_file, first_sg_ch, first_thr)
                first_summary_path = _recording_summary_path_from_out_base(first_out_base)
                try:
                    per_rec = json.loads(first_summary_path.read_text(encoding="utf-8"))
                    dur_s = float(per_rec["seconds"])
                    n_samp = int(per_rec["cumulative_segment_end_sample"]) - int(
                        per_rec["cumulative_segment_start_sample"]
                    )
                    print(
                        f"  [skip] all pairs already complete for this recording. Advancing timeline by "
                        f"{dur_s:.2f}s ({n_samp} samples).",
                        flush=True,
                    )
                except Exception:
                    # Fall back to loading the recording to get duration.
                    all_pairs_complete = False

            if resume and all_pairs_complete:
                cumulative_sample_offset += int(n_samp)
                cumulative_time_offset_sec += float(dur_s)
                session_ordinal += 1
                continue

        print("  Loading recording (read + HW map + probe)…", flush=True)
        try:
            t_load = time.perf_counter()
            rec = load_recording_mapped(rec_file, fs, probe_json)
            print(f"  Loaded in {time.perf_counter() - t_load:.1f} s.", flush=True)
        except Exception as ex:
            print(f"  [skip] load failed: {ex}", flush=True)
            continue

        chan_ids = rec.get_channel_ids()
        n_samp = rec.get_num_samples()
        dur_s = n_samp / fs

        if apply_spikeband:
            print(
                f"  Applying bandpass {bandpass_freq_min:.0f}–{bandpass_freq_max:.0f} Hz (spikeband)…",
                flush=True,
            )
            t_bp = time.perf_counter()
            rec = spre.bandpass_filter(
                rec,
                freq_min=float(bandpass_freq_min),
                freq_max=float(bandpass_freq_max),
                dtype="float32",
            )
            print(f"  Bandpass done in {time.perf_counter() - t_bp:.1f} s.", flush=True)
        else:
            print("  Bandpass skipped (raw scaled traces for detection).", flush=True)

        parent, stem = _recording_parent_stem_safe(rec_file)

        print(
            f"  Duration {dur_s:.2f} s ({n_samp} samples @ {fs:.0f} Hz); detecting crossings…",
            flush=True,
        )

        # For the first successfully loaded recording in the run: save previews for all pairs
        # missing from trace_preview_saved_for_pairs.
        if not previews_written_for_first_loaded_record:
            for pair in channel_threshold_pairs:
                sg_ch = int(pair["sg_ch"])
                threshold_uv = float(pair["threshold_uv"])
                pair_key = f"{sg_ch}__{threshold_uv:.9g}"
                if pair_key in trace_preview_saved_for_pairs:
                    continue
                rec_idx_local = sg_ch
                if rec_idx_local >= len(chan_ids):
                    continue
                channel_id = chan_ids[rec_idx_local]
                preview_path = (
                    run_output_dir
                    / f"sgch{sg_ch}_thr{_format_threshold_for_folder(threshold_uv)}uV"
                    / f"{parent}__{stem}_first10s_sgch{sg_ch}_filtered_trace_preview.png"
                )
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                print(
                    f"  Saving first 10 s trace preview for sg_ch={sg_ch}, thr={threshold_uv} uV: {preview_path.name}…",
                    flush=True,
                )
                save_first_10s_trace_preview(
                    rec,
                    channel_id,
                    out_png=preview_path,
                    fs=fs,
                    duration_sec=10.0,
                    sg_ch=sg_ch,
                    recording_stem=f"{parent}__{stem}",
                    apply_spikeband=apply_spikeband,
                    bandpass_freq_min=float(bandpass_freq_min),
                    bandpass_freq_max=float(bandpass_freq_max),
                )
                trace_preview_saved_for_pairs.add(pair_key)

            previews_written_for_first_loaded_record = True

        pairs_processed = 0
        for pair in channel_threshold_pairs:
            sg_ch = int(pair["sg_ch"])
            threshold_uv = float(pair["threshold_uv"])
            rec_idx_local = sg_ch

            if rec_idx_local >= len(chan_ids):
                print(
                    f"  [skip] sg_ch={sg_ch} rec_idx={rec_idx_local} out of range for this file.",
                    flush=True,
                )
                continue

            channel_id = chan_ids[rec_idx_local]
            pair_folder = _pair_folder_name(sg_ch, threshold_uv)
            pair_dir = run_output_dir / pair_folder
            pair_dir.mkdir(parents=True, exist_ok=True)
            out_base = pair_dir / f"{parent}__{stem}"

            pair_key = f"{sg_ch}__{threshold_uv:.9g}"
            if pair_key not in trace_preview_saved_for_pairs:
                preview_path = (
                    pair_dir / f"{parent}__{stem}_first10s_sgch{sg_ch}_filtered_trace_preview.png"
                )
                print(
                    f"  Saving first 10 s trace preview for sg_ch={sg_ch}, thr={threshold_uv} uV: {preview_path.name}…",
                    flush=True,
                )
                save_first_10s_trace_preview(
                    rec,
                    channel_id,
                    out_png=preview_path,
                    fs=fs,
                    duration_sec=10.0,
                    sg_ch=sg_ch,
                    recording_stem=f"{parent}__{stem}",
                    apply_spikeband=apply_spikeband,
                    bandpass_freq_min=float(bandpass_freq_min),
                    bandpass_freq_max=float(bandpass_freq_max),
                )
                trace_preview_saved_for_pairs.add(pair_key)

            summary_path = _recording_summary_path_from_out_base(out_base)
            if resume and _is_recording_summary_complete(summary_path):
                print(
                    f"  [skip] already complete: sg_ch={sg_ch}, thr={threshold_uv:.3f} uV",
                    flush=True,
                )
                pairs_processed += 1
                continue

            print(
                f"  Running detection for sg_ch={sg_ch}, threshold_uv={threshold_uv:.3f} µV…",
                flush=True,
            )
            total_events, chunk_manifest = process_recording_save_per_chunk(
                rec,
                channel_id,
                out_base=out_base,
                sg_ch=sg_ch,
                session_cumulative_sample_offset=int(cumulative_sample_offset),
                session_cumulative_time_offset_sec=float(cumulative_time_offset_sec),
                fs=fs,
                chunk_samples=chunk_samples,
                threshold_uv=threshold_uv,
                polarity=polarity,
                refractory_samples=refractory_samples,
                pre_samples=pre_samples,
                post_samples=post_samples,
                progress=True,
                progress_prefix="  ",
                resume=resume,
            )

            per_rec = {
                "rec_file": str(rec_file.resolve()),
                "n_crossings": int(total_events),
                "seconds": float(rec.get_num_samples() / fs),
                "output_summary": str(summary_path.resolve()),
                "preprocessing": meta_run["preprocessing"],
                "session_ordinal": session_ordinal,
                "cumulative_segment_start_sample": int(cumulative_sample_offset),
                "cumulative_segment_start_sec": float(cumulative_time_offset_sec),
                "cumulative_segment_end_sample": int(cumulative_sample_offset + n_samp),
                "cumulative_segment_end_sec": float(cumulative_time_offset_sec + dur_s),
                "sg_ch": sg_ch,
                "threshold_uv": threshold_uv,
                "chunks": chunk_manifest,
            }
            summary_path.write_text(json.dumps(per_rec, indent=2), encoding="utf-8")

            pairs_processed += 1
            print(
                f"  Pair done: sg_ch={sg_ch}, thr={threshold_uv:.3f} uV -> {per_rec['n_crossings']} events.",
                flush=True,
            )

        dt = time.perf_counter() - t0
        print(
            f"  Done processing {pairs_processed}/{len(channel_threshold_pairs)} pair(s) for this recording in {dt:.1f} s wall.\n",
            flush=True,
        )
        cumulative_sample_offset += int(n_samp)
        cumulative_time_offset_sec += float(dur_s)
        session_ordinal += 1
        del rec

    print(
        f"All recordings finished in {time.perf_counter() - t0_all:.1f} s.\n"
        f"Outputs: {run_output_dir.resolve()}",
        flush=True,
    )
    return 0


def main() -> int:
    print("=== SG channel threshold crossings (interactive) ===\n")

    resume_prev = prompt_yes_no("Resume previous interrupted session?", default_yes=False)
    if resume_prev:
        run_dir = Path(
            prompt_line(
                "Directory of threshold_crossings_run_YYYYMMDD_HHMMSS folder",
                "",
            )
        )
        if not run_dir.exists():
            print(f"Run directory not found: {run_dir}", file=sys.stderr)
            return 1

        run_config_path = run_dir / "run_config.json"
        if not run_config_path.exists():
            print(f"Missing run_config.json in: {run_dir}", file=sys.stderr)
            return 1

        try:
            meta_run = json.loads(run_config_path.read_text(encoding="utf-8"))
        except Exception as ex:
            print(f"Failed to load run_config.json: {ex}", file=sys.stderr)
            return 1

        # Re-hydrate parameters from run_config.json (avoid asking again).
        required_keys = [
            "run_output_dir",
            "recording_files",
            "channel_threshold_pairs",
            "polarity",
            "chunk_samples",
            "refractory_samples",
            "pre_samples",
            "post_samples",
            "sampling_rate_hz",
            "probe_json",
            "preprocessing",
        ]
        for k in required_keys:
            if k not in meta_run:
                print(f"run_config.json is missing key: {k}", file=sys.stderr)
                return 1

        run_output_dir = Path(meta_run["run_output_dir"])
        rec_files = [Path(p) for p in meta_run["recording_files"]]

        channel_threshold_pairs: list[dict] = []
        for p in meta_run["channel_threshold_pairs"]:
            channel_threshold_pairs.append(
                {"sg_ch": int(p["sg_ch"]), "threshold_uv": float(p["threshold_uv"])}
            )

        polarity = meta_run["polarity"]
        chunk_samples = int(meta_run["chunk_samples"])
        refractory_samples = int(meta_run["refractory_samples"])
        pre_samples = int(meta_run["pre_samples"])
        post_samples = int(meta_run["post_samples"])
        fs = float(meta_run["sampling_rate_hz"])

        probe_json = Path(meta_run["probe_json"])
        if not probe_json.exists():
            print(f"Probe file not found from run_config: {probe_json}", file=sys.stderr)
            return 1

        preprocessing = meta_run.get("preprocessing", None)
        apply_spikeband = (
            preprocessing is not None
            and preprocessing.get("spikeband_bandpass_hz", None) is not None
        )
        if apply_spikeband:
            bandpass_freq_min = float(preprocessing["spikeband_bandpass_hz"][0])
            bandpass_freq_max = float(preprocessing["spikeband_bandpass_hz"][1])
        else:
            bandpass_freq_min = 300.0
            bandpass_freq_max = 6000.0

        return process_threshold_crossings_run(
            run_output_dir=run_output_dir,
            meta_run=meta_run,
            rec_files=rec_files,
            fs=fs,
            probe_json=probe_json,
            channel_threshold_pairs=channel_threshold_pairs,
            polarity=polarity,
            chunk_samples=chunk_samples,
            refractory_samples=refractory_samples,
            pre_samples=pre_samples,
            post_samples=post_samples,
            apply_spikeband=apply_spikeband,
            bandpass_freq_min=bandpass_freq_min,
            bandpass_freq_max=bandpass_freq_max,
            resume=True,
        )

    i_root = Path(
        prompt_line("Folder to scan for recordings (recursive)", "I:/")
    )
    try:
        chronic_all = discover_chronic_rec_files(i_root)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    print(f"Found {len(chronic_all)} Chronic_Rec_*.rec file(s) under {i_root.resolve()}")
    if chronic_all:
        print(f"  (earliest) {chronic_all[0].name}")
        print(f"  (latest)   {chronic_all[-1].name}")

    while True:
        first_s = prompt_line(
            "First recording (filename or YYYYMMDD_HHMMSS, inclusive)",
            chronic_all[0].name if chronic_all else "",
        )
        last_s = prompt_line(
            "Last recording (filename or YYYYMMDD_HHMMSS, inclusive)",
            chronic_all[-1].name if chronic_all else "",
        )
        try:
            k0 = parse_chronic_rec_boundary_key(first_s)
            k1 = parse_chronic_rec_boundary_key(last_s)
        except ValueError as e:
            print(f"{e}\n")
            continue
        rec_files = filter_chronic_recs_in_range(chronic_all, k0, k1)
        if not rec_files:
            print(
                "No recordings fall in that range. Check names (Chronic_Rec_YYYYMMDD_HHMMSS.rec) "
                "and try again.\n"
            )
            continue
        print(f"Will process {len(rec_files)} recording(s) in this inclusive range.")
        for i, p in enumerate(rec_files[:5]):
            print(f"  {i+1}. {p}")
        if len(rec_files) > 5:
            print(f"  ... and {len(rec_files) - 5} more")
        break

    output_parent = Path(
        prompt_line(
            "Parent folder for outputs (a new run subfolder will be created here)",
            "I:/threshold_crossings_outputs",
        )
    )
    run_output_dir = make_run_output_dir(output_parent)
    print(f"\nRun output folder: {run_output_dir.resolve()}")
    print(
        "Files written per run: run_config.json; per recording a *_recording_summary.json plus "
        "per time-chunk *_chunk_NNNN_threshold_crossings.npz and a waveform PNG (see filenames).\n"
    )

    probe_json = Path(
        prompt_line(
            "Probe JSON path",
            r"E:\Centimani\Recording\Scripts\LSNET_probe.json",
        )
    )
    if not probe_json.exists():
        print(f"Probe file not found: {probe_json}", file=sys.stderr)
        return 1

    config_path = Path(
        prompt_line(
            "Channel/threshold config JSON path",
            r"I:\channel_thresholds.json",
        )
    )
    chunk_sec = prompt_float("Chunk length (seconds)", 60.0)
    fs = prompt_float("Sampling rate (Hz)", 30000.0)
    polarity = prompt_choice(
        "Polarity",
        ("negative", "positive", "both"),
        "negative",
    )
    refractory_ms = prompt_float("Refractory period (ms) between kept events", 0.5)
    pre_ms = prompt_float("Waveform before crossing (ms)", 1.0)
    post_ms = prompt_float("Waveform after crossing (ms)", 2.0)

    apply_spikeband = prompt_yes_no(
        "Apply spikeband bandpass before detection (SortingLSNET-style 300–6000 Hz)",
        default_yes=True,
    )
    bandpass_freq_min = 300.0
    bandpass_freq_max = 6000.0
    if apply_spikeband:
        bandpass_freq_min = prompt_float("  Bandpass high-pass corner (Hz)", 300.0)
        bandpass_freq_max = prompt_float("  Bandpass low-pass corner (Hz)", 6000.0)
        if bandpass_freq_min >= bandpass_freq_max:
            print("bandpass_freq_min must be < bandpass_freq_max.", file=sys.stderr)
            return 1

    pi = read_probeinterface(str(probe_json))
    probe = pi.probes[0]
    sg_map = build_sg_to_recording_index(probe)
    fallback_used = False
    if not config_path.exists():
        print(
            f"[warn] Config file not found: {config_path}. Using built-in example pairs instead.",
            flush=True,
        )
        pairs = DEFAULT_CHANNEL_THRESHOLDS_EXAMPLE
        fallback_used = True
    else:
        pairs = load_channel_threshold_pairs(config_path)
    for p in pairs:
        if p["sg_ch"] not in sg_map:
            print(
                f"sg_ch={p['sg_ch']} not present in probe map (probe has {len(sg_map)} contacts).",
                file=sys.stderr,
            )
            return 1

    fs = float(fs)
    pre_samples = max(1, int(round(pre_ms * fs / 1000.0)))
    post_samples = max(1, int(round(post_ms * fs / 1000.0)))
    refractory_samples = max(1, int(round(refractory_ms * fs / 1000.0)))
    chunk_samples = max(1000, int(round(chunk_sec * fs)))

    channel_threshold_pairs: list[dict] = []
    for p in pairs:
        channel_threshold_pairs.append(
            {
                "sg_ch": int(p["sg_ch"]),
                "threshold_uv": float(p["threshold_uv"]),
            }
        )

    meta_run = {
        "run_output_dir": str(run_output_dir.resolve()),
        "output_parent": str(Path(output_parent).resolve()),
        "i_root": str(i_root.resolve()),
        "first_boundary_input": first_s,
        "last_boundary_input": last_s,
        "first_sort_key": int(min(k0, k1)),
        "last_sort_key": int(max(k0, k1)),
        "n_files": len(rec_files),
        "recording_files": [str(p.resolve()) for p in rec_files],
        "channel_threshold_config": str(config_path.resolve()),
        "channel_threshold_config_fallback_used": fallback_used,
        "channel_threshold_pairs": channel_threshold_pairs,
        "polarity": polarity,
        "chunk_sec": chunk_sec,
        "chunk_samples": chunk_samples,
        "sampling_rate_hz": fs,
        "pre_ms": pre_ms,
        "post_ms": post_ms,
        "pre_samples": pre_samples,
        "post_samples": post_samples,
        "refractory_ms": refractory_ms,
        "refractory_samples": refractory_samples,
        "probe_json": str(probe_json.resolve()),
        "preprocessing": (
            {
                "spikeband_bandpass_hz": [bandpass_freq_min, bandpass_freq_max],
                "dtype": "float32",
                "note": "Matches SortingLSNET_Feb2026 preproc.bandpass_filter; "
                "detection and saved waveforms use this filtered trace.",
            }
            if apply_spikeband
            else None
        ),
        "saved_files_note": (
            "Per recording: <parent>__<stem>_recording_summary.json lists chunk artifacts. "
            "Each time chunk: *_chunk_NNNN_threshold_crossings.npz (arrays + chunk_index, "
            "time_start_sec, time_end_sec) and a PNG of all waveforms with bold mean. "
            "Cumulative sample/time columns chain sessions (see cumulative_timeline)."
        ),
        "cumulative_timeline": {
            "ordering": "Chronic_Rec timestamp sort from discover_chronic_rec_files",
            "rule": "Offsets advance only after a file is fully processed. Skipped/failed files "
            "do not consume timeline space (next success abuts previous success).",
        },
    }
    (run_output_dir / "run_config.json").write_text(
        json.dumps(meta_run, indent=2), encoding="utf-8"
    )

    return process_threshold_crossings_run(
        run_output_dir=run_output_dir,
        meta_run=meta_run,
        rec_files=rec_files,
        fs=fs,
        probe_json=probe_json,
        channel_threshold_pairs=channel_threshold_pairs,
        polarity=polarity,
        chunk_samples=chunk_samples,
        refractory_samples=refractory_samples,
        pre_samples=pre_samples,
        post_samples=post_samples,
        apply_spikeband=apply_spikeband,
        bandpass_freq_min=bandpass_freq_min,
        bandpass_freq_max=bandpass_freq_max,
        resume=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
