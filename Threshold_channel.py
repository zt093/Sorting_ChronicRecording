"""
Scan SpikeGadgets .rec files under a root in time chunks, detect threshold crossings
on configured SG channel/threshold-range pairs, extract waveform windows
(-1 ms .. +2 ms around each crossing sample), and save minute-level mean waveforms
plus timestamps/crossing samples.

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

How outputs are saved (each recording date gets its own folder; existing folders
are not overwritten):
  <output_parent>/threshold_crossings_YYYYMMDD/
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
import os
import csv
import re
import sys
import time
from pathlib import Path

import gc


def _remove_pythonpath_entries_from_sys_path() -> None:
    """
    Keep external PYTHONPATH packages from shadowing the Conda environment.

    This script depends on SpikeInterface's matching probeinterface package; a
    stale probeinterface folder on PYTHONPATH can break read_spikegadgets().
    """
    pythonpath = os.environ.get("PYTHONPATH", "")
    if not pythonpath:
        return
    blocked: set[str] = set()
    for entry in pythonpath.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        try:
            blocked.add(str(Path(entry).resolve()).casefold())
        except Exception:
            blocked.add(entry.casefold())
    if not blocked:
        return

    kept: list[str] = []
    for entry in sys.path:
        if not entry:
            kept.append(entry)
            continue
        try:
            resolved = str(Path(entry).resolve()).casefold()
        except Exception:
            resolved = entry.casefold()
        if resolved not in blocked:
            kept.append(entry)
    sys.path[:] = kept


_remove_pythonpath_entries_from_sys_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
from probeinterface import read_probeinterface
from spikeinterface.core import BaseRecording

COMPUTE_HOURLY_CORRELOGRAMS = False
DEFAULT_PROBE_JSON = Path(r"E:\Curtis\spikeinterface\LSNET_probe.json")
DEFAULT_CHANNEL_THRESHOLD_CONFIG_JSON = Path(r"S:\channel_thresholds_from_sorted.json")
DEFAULT_OUTPUT_PARENT = Path(r"S:\Threshold_test")
DEFAULT_CHUNK_SEC = 300.0
DEFAULT_SAMPLING_RATE_HZ = 30000.0
DEFAULT_POLARITY = "negative"
DEFAULT_REFRACTORY_MS = 0.5
DEFAULT_PRE_MS = 1.0
DEFAULT_POST_MS = 2.0
DEFAULT_APPLY_SPIKEBAND = True
DEFAULT_BANDPASS_FREQ_MIN = 300.0
DEFAULT_BANDPASS_FREQ_MAX = 6000.0
DEFAULT_SAVE_TRACE_PREVIEWS = False


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


def _ensure_crossing_capacity(cross_buf: np.ndarray, n_events: int) -> np.ndarray:
    """Grow a crossing-only buffer without carrying individual waveforms."""
    cap = cross_buf.shape[0]
    if n_events < cap:
        return cross_buf
    new_cap = max(cap * 2, cap + 65536)
    cb = np.empty(new_cap, dtype=np.int64)
    if n_events > 0:
        cb[:n_events] = cross_buf[:n_events]
    return cb


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
    amp_bar_uv: float = 100.0,
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
        template_ptp_uv = float(np.ptp(mean_wf))
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
            + f"  |  template p2p={template_ptp_uv:.1f} uV"
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
        amp_bar = max(1e-6, float(amp_bar_uv))
        if amp_bar > 0.9 * y_span:
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


def _safe_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _compute_cv2_from_timestamps(timestamps_sec: np.ndarray) -> float:
    timestamps_sec = np.asarray(timestamps_sec, dtype=np.float64).ravel()
    if timestamps_sec.size < 3:
        return float("nan")
    isi = np.diff(timestamps_sec)
    if isi.size < 2:
        return float("nan")
    denom = isi[:-1] + isi[1:]
    valid = denom > 0
    if not np.any(valid):
        return float("nan")
    cv2 = 2.0 * np.abs(np.diff(isi)[valid]) / denom[valid]
    return float(np.mean(cv2)) if cv2.size else float("nan")


def _peak_to_trough_ms(waveform: np.ndarray, fs: float) -> float:
    waveform = np.asarray(waveform, dtype=np.float64).ravel()
    if waveform.size < 2 or fs <= 0:
        return float("nan")
    trough_i = int(np.argmin(waveform))
    peak_i = int(np.argmax(waveform))
    if abs(float(waveform[trough_i])) >= abs(float(waveform[peak_i])):
        if trough_i >= waveform.size - 1:
            return float("nan")
        post_peak_i = trough_i + int(np.argmax(waveform[trough_i:]))
        return float((post_peak_i - trough_i) / fs * 1000.0)
    if peak_i >= waveform.size - 1:
        return float("nan")
    post_trough_i = peak_i + int(np.argmin(waveform[peak_i:]))
    return float((post_trough_i - peak_i) / fs * 1000.0)


def _waveform_summary(waveforms_uv: np.ndarray, fs: float, wf_len: int) -> dict:
    waveforms_uv = np.asarray(waveforms_uv, dtype=np.float32)
    n = int(waveforms_uv.shape[0]) if waveforms_uv.ndim == 2 else 0
    if n == 0:
        mean_wf = np.full(wf_len, np.nan, dtype=np.float32)
        return {
            "n_spikes": 0,
            "mean_waveform_uv": mean_wf,
            "amplitude_ptp_uv": float("nan"),
            "mean_abs_waveform_uv": float("nan"),
            "peak_to_trough_ms": float("nan"),
        }
    mean_wf = np.mean(waveforms_uv, axis=0).astype(np.float32)
    return {
        "n_spikes": n,
        "mean_waveform_uv": mean_wf,
        "amplitude_ptp_uv": float(np.ptp(mean_wf)),
        "mean_abs_waveform_uv": float(np.mean(np.abs(mean_wf))),
        "peak_to_trough_ms": _peak_to_trough_ms(mean_wf, fs),
    }


def _mean_waveform_summary(mean_wf: np.ndarray, n_spikes: int, fs: float, wf_len: int) -> dict:
    n_spikes = int(n_spikes)
    if n_spikes <= 0:
        mean_wf = np.full(wf_len, np.nan, dtype=np.float32)
        return {
            "n_spikes": 0,
            "mean_waveform_uv": mean_wf,
            "amplitude_ptp_uv": float("nan"),
            "mean_abs_waveform_uv": float("nan"),
            "peak_to_trough_ms": float("nan"),
        }
    mean_wf = np.asarray(mean_wf, dtype=np.float32).reshape(wf_len)
    return {
        "n_spikes": n_spikes,
        "mean_waveform_uv": mean_wf,
        "amplitude_ptp_uv": float(np.ptp(mean_wf)),
        "mean_abs_waveform_uv": float(np.mean(np.abs(mean_wf))),
        "peak_to_trough_ms": _peak_to_trough_ms(mean_wf, fs),
    }


def _empty_minute_buffer(wf_len: int) -> dict:
    return {
        "crossings": [],
        "waveform_sum": np.zeros(wf_len, dtype=np.float64),
        "n_spikes": 0,
    }


def _add_event_to_minute_buffer(state: dict, crossing_sample: int, waveform_uv: np.ndarray, minute_samples: int, wf_len: int) -> None:
    minute_index = int(crossing_sample // minute_samples)
    buf = state["minute_buffers"].setdefault(minute_index, _empty_minute_buffer(wf_len))
    buf["crossings"].append(int(crossing_sample))
    buf["waveform_sum"] += np.asarray(waveform_uv, dtype=np.float64)
    buf["n_spikes"] = int(buf["n_spikes"]) + 1


def _save_mean_waveform_plot(
    mean_waveform_uv: np.ndarray,
    *,
    n_spikes: int,
    fs: float,
    pre_samples: int,
    post_samples: int,
    sg_ch: int,
    t_start_min: float,
    t_end_min: float,
    out_png: Path,
    fixed_y_center: float = 0.0,
    fixed_y_half: float = 1.0,
    time_bar_ms: float = 1000.0,
    amp_bar_fraction: float = 0.25,
    amp_bar_uv: float = 100.0,
) -> None:
    wf_len = pre_samples + post_samples
    t_ms = (np.arange(wf_len, dtype=np.float64) - pre_samples) / fs * 1000.0
    mean_wf = np.asarray(mean_waveform_uv, dtype=np.float32).reshape(wf_len)
    y_span = 2.0 * float(fixed_y_half)
    y_min = float(fixed_y_center - fixed_y_half)
    y_max = float(fixed_y_center + fixed_y_half)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    if int(n_spikes) <= 0 or not np.any(np.isfinite(mean_wf)):
        ax.text(
            0.5,
            0.5,
            "No crossings in this hour",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        template_ptp_uv = float("nan")
    else:
        template_ptp_uv = float(np.ptp(mean_wf))
        ax.plot(t_ms, mean_wf, color="k", linewidth=2.8, zorder=10)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    title = (
        f"SG ch {sg_ch}  |  recording {t_start_min:.2f}-{t_end_min:.2f} min  "
        f"|  N={int(n_spikes)}"
    )
    if np.isfinite(template_ptp_uv):
        title += f"  |  template p2p={template_ptp_uv:.1f} uV"
    ax.set_title(title, fontsize=10, pad=6)

    time_span_ms = float(t_ms[-1] - t_ms[0])
    if time_span_ms > 0 and y_span > 0:
        t_bar_ms = min(float(time_bar_ms), 0.3 * time_span_ms)
        x0 = float(t_ms[0]) + 0.06 * time_span_ms
        x1 = x0 + t_bar_ms
        y0 = y_min + 0.06 * y_span
        amp_bar = max(1e-6, float(amp_bar_uv))
        if amp_bar > 0.9 * y_span:
            amp_bar = max(1e-6, float(amp_bar_fraction) * y_span)
        if y0 + amp_bar > y_max:
            y0 = y_max - amp_bar - 0.02 * y_span
        x_bar = float(t_ms[0]) + 0.92 * time_span_ms
        ax.plot([x0, x1], [y0, y0], color="k", linewidth=3, solid_capstyle="butt", zorder=20)
        ax.plot([x_bar, x_bar], [y0, y0 + amp_bar], color="k", linewidth=3, zorder=20)
        t_lbl = f"{t_bar_ms / 1000.0:.1f} s" if t_bar_ms >= 1000.0 else f"{t_bar_ms:.0f} ms"
        ax.text((x0 + x1) / 2.0, y0 - 0.03 * y_span, t_lbl, ha="center", va="top", fontsize=9)
        ax.text(
            x_bar + 0.01 * time_span_ms,
            y0 + 0.5 * amp_bar,
            f"{amp_bar:.0f} uV",
            ha="left",
            va="center",
            fontsize=9,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _event_amplitude_uv(waveform_uv: np.ndarray, polarity: str) -> float:
    if waveform_uv.size == 0:
        return 0.0
    if polarity == "negative":
        return float(abs(np.min(waveform_uv)))
    if polarity == "positive":
        return float(np.max(waveform_uv))
    return float(np.max(np.abs(waveform_uv)))


def _isi_and_correlogram(
    timestamps_sec: np.ndarray,
    *,
    max_lag_ms: float = 100.0,
    bin_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps_sec = np.asarray(timestamps_sec, dtype=np.float64).ravel()
    timestamps_sec = np.sort(timestamps_sec[np.isfinite(timestamps_sec)])
    isi_sec = np.diff(timestamps_sec).astype(np.float32) if timestamps_sec.size >= 2 else np.zeros(0, dtype=np.float32)
    edges_ms = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms, dtype=np.float32)
    counts = np.zeros(edges_ms.size - 1, dtype=np.int64)
    if timestamps_sec.size >= 2:
        max_lag_s = max_lag_ms / 1000.0
        for i, t0 in enumerate(timestamps_sec):
            j0 = np.searchsorted(timestamps_sec, t0 - max_lag_s, side="left")
            j1 = np.searchsorted(timestamps_sec, t0 + max_lag_s, side="right")
            diffs_ms = (timestamps_sec[j0:j1] - t0) * 1000.0
            diffs_ms = diffs_ms[np.abs(diffs_ms) > 1e-12]
            if diffs_ms.size:
                counts += np.histogram(diffs_ms, bins=edges_ms)[0].astype(np.int64)
    centers_ms = 0.5 * (edges_ms[:-1] + edges_ms[1:])
    return isi_sec, centers_ms.astype(np.float32), counts


def _isi_only_and_empty_correlogram(
    timestamps_sec: np.ndarray,
    *,
    max_lag_ms: float = 100.0,
    bin_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps_sec = np.asarray(timestamps_sec, dtype=np.float64).ravel()
    timestamps_sec = np.sort(timestamps_sec[np.isfinite(timestamps_sec)])
    isi_sec = np.diff(timestamps_sec).astype(np.float32) if timestamps_sec.size >= 2 else np.zeros(0, dtype=np.float32)
    edges_ms = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms, dtype=np.float32)
    centers_ms = 0.5 * (edges_ms[:-1] + edges_ms[1:])
    return isi_sec, centers_ms.astype(np.float32), np.zeros(centers_ms.size, dtype=np.int64)


def _write_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, Path):
                item[key] = str(value)
            elif isinstance(value, (np.integer,)):
                item[key] = int(value)
            elif isinstance(value, (np.floating,)):
                item[key] = _safe_float(value)
            elif isinstance(value, float):
                item[key] = _safe_float(value)
            else:
                item[key] = value
        out.append(item)
    return out


def _save_minute_outputs(
    state: dict,
    *,
    crossings: np.ndarray,
    wfs: np.ndarray,
    fs: float,
    wf_len: int,
    total_samples: int,
    session_cumulative_sample_offset: int,
    session_cumulative_time_offset_sec: float,
) -> None:
    if crossings.size == 0:
        return
    minute_samples = int(round(60.0 * fs))
    minute_indices = np.unique(crossings // minute_samples)
    minute_dir = Path(state["out_base"].parent) / "minute_npz"
    minute_dir.mkdir(parents=True, exist_ok=True)

    for minute_index in minute_indices:
        minute_index = int(minute_index)
        m0 = minute_index * minute_samples
        m1 = min(m0 + minute_samples, int(total_samples))
        mask = (crossings >= m0) & (crossings < m1)
        if not np.any(mask):
            continue
        minute_crossings = crossings[mask].astype(np.int64, copy=True)
        minute_wfs = wfs[mask].astype(np.float32, copy=True)
        ts_sec = minute_crossings.astype(np.float64) / fs
        ts_cum = ts_sec + float(session_cumulative_time_offset_sec)
        summary = _waveform_summary(minute_wfs, fs, wf_len)
        cv2 = _compute_cv2_from_timestamps(ts_sec)
        duration_sec = max(0.0, float(m1 - m0) / fs)
        firing_rate_hz = float(minute_crossings.size / duration_sec) if duration_sec > 0 else 0.0
        minute_npz = minute_dir / f"{state['out_base'].name}_minute_{minute_index:06d}_spikes_waveforms.npz"
        np.savez_compressed(
            str(minute_npz),
            crossing_samples=minute_crossings,
            crossing_samples_cumulative=minute_crossings + int(session_cumulative_sample_offset),
            timestamps_sec=ts_sec,
            timestamps_sec_cumulative=ts_cum,
            waveforms_uv=minute_wfs,
            mean_waveform_uv=summary["mean_waveform_uv"],
            sampling_rate_hz=np.array([fs], dtype=np.float64),
            minute_index=np.array([minute_index], dtype=np.int32),
            minute_start_sec=np.array([m0 / fs], dtype=np.float64),
            minute_end_sec=np.array([m1 / fs], dtype=np.float64),
            minute_duration_sec=np.array([duration_sec], dtype=np.float64),
        )

        row = {
            "minute_index": minute_index,
            "time_start_sec": float(m0 / fs),
            "time_end_sec": float(m1 / fs),
            "duration_sec": duration_sec,
            "sg_ch": int(state["sg_ch"]),
            "threshold_min_uv": float(state["threshold_uv"]),
            "threshold_max_uv": _safe_float(state.get("threshold_max_uv")),
            "n_spikes": int(minute_crossings.size),
            "firing_rate_hz": firing_rate_hz,
            "amplitude_ptp_uv": _safe_float(summary["amplitude_ptp_uv"]),
            "mean_abs_waveform_uv": _safe_float(summary["mean_abs_waveform_uv"]),
            "cv2": _safe_float(cv2),
            "peak_to_trough_ms": _safe_float(summary["peak_to_trough_ms"]),
            "mean_waveform_uv": json.dumps([_safe_float(v) for v in summary["mean_waveform_uv"].tolist()]),
            "npz": str(minute_npz.resolve()),
        }
        state["minute_rows"].append(row)

        hour_index = minute_index // 60
        hour_buf = state["hour_buffers"].setdefault(
            hour_index,
            {"crossings": [], "timestamps": [], "waveforms": []},
        )
        hour_buf["crossings"].append(minute_crossings)
        hour_buf["timestamps"].append(ts_sec)
        hour_buf["waveforms"].append(minute_wfs)


def _flush_completed_hours(
    state: dict,
    *,
    fs: float,
    wf_len: int,
    pre_samples: int,
    post_samples: int,
    current_sample: int,
    total_samples: int,
    force: bool = False,
) -> None:
    hour_samples = int(round(3600.0 * fs))
    ready = []
    for hour_index in list(state["hour_buffers"].keys()):
        if force or ((hour_index + 1) * hour_samples <= current_sample):
            ready.append(int(hour_index))
    if not ready:
        return

    hourly_dir = Path(state["out_base"].parent) / "hourly"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    for hour_index in sorted(ready):
        buf = state["hour_buffers"].pop(hour_index)
        crossings = (
            np.concatenate(buf["crossings"]).astype(np.int64)
            if buf["crossings"]
            else np.zeros(0, dtype=np.int64)
        )
        timestamps = (
            np.concatenate(buf["timestamps"]).astype(np.float64)
            if buf["timestamps"]
            else np.zeros(0, dtype=np.float64)
        )
        waveforms = (
            np.concatenate(buf["waveforms"]).astype(np.float32)
            if buf["waveforms"]
            else np.zeros((0, wf_len), dtype=np.float32)
        )
        summary = _waveform_summary(waveforms, fs, wf_len)
        cv2 = _compute_cv2_from_timestamps(timestamps)
        if COMPUTE_HOURLY_CORRELOGRAMS:
            isi_sec, correlogram_lag_ms, correlogram_counts = _isi_and_correlogram(timestamps)
        else:
            isi_sec, correlogram_lag_ms, correlogram_counts = _isi_only_and_empty_correlogram(timestamps)
        hour_start_sec = float(hour_index * 3600.0)
        hour_end_sample = min((hour_index + 1) * hour_samples, int(total_samples))
        hour_end_sec = float(hour_end_sample / fs)
        duration_sec = max(0.0, hour_end_sec - hour_start_sec)
        firing_rate_hz = float(crossings.size / duration_sec) if duration_sec > 0 else 0.0
        npz_path = hourly_dir / f"{state['out_base'].name}_hour_{hour_index:04d}_isi_correlogram.npz"
        np.savez_compressed(
            str(npz_path),
            crossing_samples=crossings,
            timestamps_sec=timestamps,
            waveforms_uv=waveforms,
            mean_waveform_uv=summary["mean_waveform_uv"],
            isi_sec=isi_sec,
            correlogram_lag_ms=correlogram_lag_ms,
            correlogram_counts=correlogram_counts,
            hour_index=np.array([hour_index], dtype=np.int32),
            hour_start_sec=np.array([hour_start_sec], dtype=np.float64),
            hour_end_sec=np.array([hour_end_sec], dtype=np.float64),
            hour_duration_sec=np.array([duration_sec], dtype=np.float64),
        )
        fig_path = hourly_dir / f"{state['out_base'].name}_hour_{hour_index:04d}_sgch{state['sg_ch']}_n{crossings.size}_waveforms.png"
        _save_chunk_waveform_plot(
            waveforms,
            fs=fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
            sg_ch=int(state["sg_ch"]),
            t_start_min=hour_start_sec / 60.0,
            t_end_min=hour_end_sec / 60.0,
            out_png=fig_path,
            fixed_y_center=float(state["fixed_y_center"] if state["fixed_y_center"] is not None else 0.0),
            fixed_y_half=float(state["fixed_y_half"] if state["fixed_y_half"] is not None else max(1.0, 1.2 * float(state["threshold_uv"]))),
        )
        row = {
            "hour_index": hour_index,
            "time_start_sec": hour_start_sec,
            "time_end_sec": hour_end_sec,
            "duration_sec": duration_sec,
            "sg_ch": int(state["sg_ch"]),
            "threshold_min_uv": float(state["threshold_uv"]),
            "threshold_max_uv": _safe_float(state.get("threshold_max_uv")),
            "n_spikes": int(crossings.size),
            "firing_rate_hz": firing_rate_hz,
            "amplitude_ptp_uv": _safe_float(summary["amplitude_ptp_uv"]),
            "mean_abs_waveform_uv": _safe_float(summary["mean_abs_waveform_uv"]),
            "cv2": _safe_float(cv2),
            "peak_to_trough_ms": _safe_float(summary["peak_to_trough_ms"]),
            "isi_mean_ms": _safe_float(float(np.mean(isi_sec) * 1000.0) if isi_sec.size else float("nan")),
            "isi_median_ms": _safe_float(float(np.median(isi_sec) * 1000.0) if isi_sec.size else float("nan")),
            "npz": str(npz_path.resolve()),
            "figure": str(fig_path.resolve()),
        }
        state["hourly_rows"].append(row)
        del crossings, timestamps, waveforms, isi_sec, correlogram_lag_ms, correlogram_counts
        gc.collect()


def _ensure_all_minute_rows(
    state: dict,
    *,
    total_minutes: int,
    total_samples: int,
    fs: float,
    wf_len: int,
) -> None:
    existing = {int(row["minute_index"]) for row in state["minute_rows"]}
    minute_dir = Path(state["out_base"].parent) / "minute_npz"
    minute_dir.mkdir(parents=True, exist_ok=True)
    for minute_index in range(total_minutes):
        if minute_index in existing:
            continue
        m0_sec = float(minute_index * 60.0)
        m1_sec = float(min((minute_index + 1) * 60.0, float(total_samples) / fs))
        duration_sec = max(0.0, m1_sec - m0_sec)
        npz_path = minute_dir / f"{state['out_base'].name}_minute_{minute_index:06d}_spikes_waveforms.npz"
        mean_wf = np.full(wf_len, np.nan, dtype=np.float32)
        np.savez_compressed(
            str(npz_path),
            crossing_samples=np.zeros(0, dtype=np.int64),
            crossing_samples_cumulative=np.zeros(0, dtype=np.int64),
            timestamps_sec=np.zeros(0, dtype=np.float64),
            timestamps_sec_cumulative=np.zeros(0, dtype=np.float64),
            waveforms_uv=np.zeros((0, wf_len), dtype=np.float32),
            mean_waveform_uv=mean_wf,
            sampling_rate_hz=np.array([fs], dtype=np.float64),
            minute_index=np.array([minute_index], dtype=np.int32),
            minute_start_sec=np.array([m0_sec], dtype=np.float64),
            minute_end_sec=np.array([m1_sec], dtype=np.float64),
            minute_duration_sec=np.array([duration_sec], dtype=np.float64),
        )
        state["minute_rows"].append(
            {
                "minute_index": minute_index,
                "time_start_sec": m0_sec,
                "time_end_sec": m1_sec,
                "duration_sec": duration_sec,
                "sg_ch": int(state["sg_ch"]),
                "threshold_min_uv": float(state["threshold_uv"]),
                "threshold_max_uv": _safe_float(state.get("threshold_max_uv")),
                "n_spikes": 0,
                "firing_rate_hz": 0.0,
                "amplitude_ptp_uv": None,
                "mean_abs_waveform_uv": None,
                "cv2": None,
                "peak_to_trough_ms": None,
                "mean_waveform_uv": json.dumps([None] * wf_len),
                "npz": str(npz_path.resolve()),
            }
        )
    state["minute_rows"].sort(key=lambda row: int(row["minute_index"]))


def _ensure_all_hourly_rows(
    state: dict,
    *,
    total_hours: int,
    total_samples: int,
    fs: float,
    wf_len: int,
    pre_samples: int,
    post_samples: int,
) -> None:
    existing = {int(row["hour_index"]) for row in state["hourly_rows"]}
    hourly_dir = Path(state["out_base"].parent) / "hourly"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    for hour_index in range(total_hours):
        if hour_index in existing:
            continue
        h0_sec = float(hour_index * 3600.0)
        h1_sec = float(min((hour_index + 1) * 3600.0, float(total_samples) / fs))
        duration_sec = max(0.0, h1_sec - h0_sec)
        mean_wf = np.full(wf_len, np.nan, dtype=np.float32)
        npz_path = hourly_dir / f"{state['out_base'].name}_hour_{hour_index:04d}_isi_correlogram.npz"
        np.savez_compressed(
            str(npz_path),
            crossing_samples=np.zeros(0, dtype=np.int64),
            timestamps_sec=np.zeros(0, dtype=np.float64),
            waveforms_uv=np.zeros((0, wf_len), dtype=np.float32),
            mean_waveform_uv=mean_wf,
            isi_sec=np.zeros(0, dtype=np.float32),
            correlogram_lag_ms=np.arange(-99.5, 100.0, 1.0, dtype=np.float32),
            correlogram_counts=np.zeros(200, dtype=np.int64),
            hour_index=np.array([hour_index], dtype=np.int32),
            hour_start_sec=np.array([h0_sec], dtype=np.float64),
            hour_end_sec=np.array([h1_sec], dtype=np.float64),
            hour_duration_sec=np.array([duration_sec], dtype=np.float64),
        )
        fig_path = hourly_dir / f"{state['out_base'].name}_hour_{hour_index:04d}_sgch{state['sg_ch']}_n0_waveforms.png"
        _save_chunk_waveform_plot(
            np.zeros((0, wf_len), dtype=np.float32),
            fs=fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
            sg_ch=int(state["sg_ch"]),
            t_start_min=h0_sec / 60.0,
            t_end_min=h1_sec / 60.0,
            out_png=fig_path,
            fixed_y_center=float(state["fixed_y_center"] if state["fixed_y_center"] is not None else 0.0),
            fixed_y_half=float(state["fixed_y_half"] if state["fixed_y_half"] is not None else max(1.0, 1.2 * float(state["threshold_uv"]))),
        )
        state["hourly_rows"].append(
            {
                "hour_index": hour_index,
                "time_start_sec": h0_sec,
                "time_end_sec": h1_sec,
                "duration_sec": duration_sec,
                "sg_ch": int(state["sg_ch"]),
                "threshold_min_uv": float(state["threshold_uv"]),
                "threshold_max_uv": _safe_float(state.get("threshold_max_uv")),
                "n_spikes": 0,
                "firing_rate_hz": 0.0,
                "amplitude_ptp_uv": None,
                "mean_abs_waveform_uv": None,
                "cv2": None,
                "peak_to_trough_ms": None,
                "isi_mean_ms": None,
                "isi_median_ms": None,
                "npz": str(npz_path.resolve()),
                "figure": str(fig_path.resolve()),
            }
        )
    state["hourly_rows"].sort(key=lambda row: int(row["hour_index"]))


def _write_recording_minute_npz_outputs(
    states: list[dict],
    *,
    run_output_dir: Path,
    parent: str,
    stem: str,
    rec_file: Path,
    total_minutes: int,
    total_samples: int,
    fs: float,
    wf_len: int,
    session_ordinal: int,
    session_cumulative_sample_offset: int,
    session_cumulative_time_offset_sec: float,
    timing_totals: dict[str, float] | None = None,
) -> list[dict]:
    minute_dir = run_output_dir / "minute_npz" / f"{parent}__{stem}"
    minute_dir.mkdir(parents=True, exist_ok=True)
    pair_ids = np.asarray([str(state["pair_id"]) for state in states])
    sg_ch = np.asarray([int(state["sg_ch"]) for state in states], dtype=np.int32)
    threshold_min_uv = np.asarray([float(state["threshold_uv"]) for state in states], dtype=np.float32)
    threshold_max_uv = np.asarray(
        [
            float(state["threshold_max_uv"])
            if state.get("threshold_max_uv") is not None
            else np.nan
            for state in states
        ],
        dtype=np.float32,
    )
    recording_rows: list[dict] = []

    for minute_index in range(int(total_minutes)):
        m0 = int(minute_index * round(60.0 * fs))
        m1 = min(int((minute_index + 1) * round(60.0 * fs)), int(total_samples))
        minute_start_sec = float(m0 / fs)
        minute_end_sec = float(m1 / fs)
        duration_sec = max(0.0, minute_end_sec - minute_start_sec)
        cumulative_start_sec = float(session_cumulative_time_offset_sec + minute_start_sec)
        cumulative_end_sec = float(session_cumulative_time_offset_sec + minute_end_sec)
        cumulative_minute_index = int(np.floor(cumulative_start_sec / 60.0))

        n_pairs = len(states)
        n_spikes = np.zeros(n_pairs, dtype=np.int64)
        firing_rate_hz = np.zeros(n_pairs, dtype=np.float32)
        cv2 = np.full(n_pairs, np.nan, dtype=np.float32)
        amplitude_ptp_uv = np.full(n_pairs, np.nan, dtype=np.float32)
        mean_abs_waveform_uv = np.full(n_pairs, np.nan, dtype=np.float32)
        peak_to_trough_ms = np.full(n_pairs, np.nan, dtype=np.float32)
        mean_waveform_uv = np.full((n_pairs, wf_len), np.nan, dtype=np.float32)
        crossing_chunks: list[np.ndarray] = []
        timestamp_chunks: list[np.ndarray] = []
        event_pair_chunks: list[np.ndarray] = []
        timestamp_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
        minute_npz = minute_dir / f"{parent}__{stem}_minute_{minute_index:06d}_all_pairs.npz"

        for pair_i, state in enumerate(states):
            buf = state["minute_buffers"].get(minute_index)
            if buf is None or int(buf["n_spikes"]) <= 0:
                crossings = np.zeros(0, dtype=np.int64)
                summary = _mean_waveform_summary(
                    np.full(wf_len, np.nan, dtype=np.float32),
                    0,
                    fs,
                    wf_len,
                )
            else:
                crossings = np.asarray(buf["crossings"], dtype=np.int64)
                crossings.sort()
                count = int(buf["n_spikes"])
                mean_wf = (np.asarray(buf["waveform_sum"], dtype=np.float64) / float(count)).astype(np.float32)
                summary = _mean_waveform_summary(mean_wf, count, fs, wf_len)

            count = int(crossings.size)
            n_spikes[pair_i] = count
            if duration_sec > 0:
                firing_rate_hz[pair_i] = float(count / duration_sec)
            ts_sec = crossings.astype(np.float64) / fs
            ts_cum = ts_sec + float(session_cumulative_time_offset_sec)
            cv2[pair_i] = np.float32(_compute_cv2_from_timestamps(ts_sec))
            amplitude_ptp_uv[pair_i] = np.float32(summary["amplitude_ptp_uv"])
            mean_abs_waveform_uv[pair_i] = np.float32(summary["mean_abs_waveform_uv"])
            peak_to_trough_ms[pair_i] = np.float32(summary["peak_to_trough_ms"])
            mean_waveform_uv[pair_i, :] = summary["mean_waveform_uv"]
            timestamp_offsets[pair_i + 1] = timestamp_offsets[pair_i] + count

            crossing_chunks.append(crossings)
            timestamp_chunks.append(ts_sec)
            event_pair_chunks.append(np.full(count, pair_i, dtype=np.int32))

            row = {
                "minute_index": minute_index,
                "cumulative_minute_index": cumulative_minute_index,
                "time_start_sec": minute_start_sec,
                "time_end_sec": minute_end_sec,
                "cumulative_start_sec": cumulative_start_sec,
                "cumulative_end_sec": cumulative_end_sec,
                "duration_sec": duration_sec,
                "recording_name": rec_file.name,
                "session_ordinal": int(session_ordinal),
                "pair_id": str(state["pair_id"]),
                "pair_row_index": pair_i,
                "sg_ch": int(state["sg_ch"]),
                "threshold_min_uv": float(state["threshold_uv"]),
                "threshold_max_uv": _safe_float(state.get("threshold_max_uv")),
                "n_spikes": count,
                "firing_rate_hz": float(firing_rate_hz[pair_i]),
                "amplitude_ptp_uv": _safe_float(amplitude_ptp_uv[pair_i]),
                "mean_abs_waveform_uv": _safe_float(mean_abs_waveform_uv[pair_i]),
                "cv2": _safe_float(cv2[pair_i]),
                "peak_to_trough_ms": _safe_float(peak_to_trough_ms[pair_i]),
                "mean_waveform_uv": json.dumps([_safe_float(v) for v in mean_waveform_uv[pair_i].tolist()]),
                "npz": str(minute_npz.resolve()),
            }
            state["minute_rows"].append(row)

        all_crossing_samples = (
            np.concatenate(crossing_chunks).astype(np.int64)
            if crossing_chunks
            else np.zeros(0, dtype=np.int64)
        )
        all_timestamps_sec = (
            np.concatenate(timestamp_chunks).astype(np.float64)
            if timestamp_chunks
            else np.zeros(0, dtype=np.float64)
        )
        event_pair_index = (
            np.concatenate(event_pair_chunks).astype(np.int32)
            if event_pair_chunks
            else np.zeros(0, dtype=np.int32)
        )
        t_write0 = time.perf_counter()
        np.savez_compressed(
            str(minute_npz),
            pair_ids=pair_ids,
            sg_ch=sg_ch,
            threshold_min_uv=threshold_min_uv,
            threshold_max_uv=threshold_max_uv,
            n_spikes=n_spikes,
            firing_rate_hz=firing_rate_hz,
            cv2=cv2,
            amplitude_ptp_uv=amplitude_ptp_uv,
            mean_abs_waveform_uv=mean_abs_waveform_uv,
            peak_to_trough_ms=peak_to_trough_ms,
            mean_waveform_uv=mean_waveform_uv,
            all_timestamps_sec=all_timestamps_sec,
            all_timestamps_sec_cumulative=all_timestamps_sec + float(session_cumulative_time_offset_sec),
            all_crossing_samples=all_crossing_samples,
            all_crossing_samples_cumulative=all_crossing_samples + int(session_cumulative_sample_offset),
            timestamp_offsets=timestamp_offsets,
            event_pair_index=event_pair_index,
            minute_index=np.array([minute_index], dtype=np.int32),
            minute_start_sec=np.array([minute_start_sec], dtype=np.float64),
            minute_end_sec=np.array([minute_end_sec], dtype=np.float64),
            cumulative_minute_index=np.array([cumulative_minute_index], dtype=np.int64),
            cumulative_start_sec=np.array([cumulative_start_sec], dtype=np.float64),
            cumulative_end_sec=np.array([cumulative_end_sec], dtype=np.float64),
            recording_name=np.asarray([rec_file.name]),
            session_ordinal=np.array([session_ordinal], dtype=np.int32),
            sampling_rate_hz=np.array([fs], dtype=np.float64),
        )
        write_elapsed = time.perf_counter() - t_write0
        _add_timing(timing_totals, "detection_minute_npz_write", write_elapsed)
        _add_timing(timing_totals, "detection_npz_write", write_elapsed)
        recording_rows.append(
            {
                "minute_index": minute_index,
                "cumulative_minute_index": cumulative_minute_index,
                "time_start_sec": minute_start_sec,
                "time_end_sec": minute_end_sec,
                "cumulative_start_sec": cumulative_start_sec,
                "cumulative_end_sec": cumulative_end_sec,
                "duration_sec": duration_sec,
                "recording_name": rec_file.name,
                "session_ordinal": int(session_ordinal),
                "n_pairs": n_pairs,
                "n_events": int(all_crossing_samples.size),
                "npz": str(minute_npz.resolve()),
            }
        )
    return recording_rows


def _write_hourly_outputs_from_minute_means(
    state: dict,
    *,
    total_hours: int,
    total_samples: int,
    fs: float,
    wf_len: int,
    pre_samples: int,
    post_samples: int,
    timing_totals: dict[str, float] | None = None,
) -> None:
    hourly_dir = Path(state["out_base"].parent) / "hourly"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    minute_samples = int(round(60.0 * fs))
    hour_samples = int(round(3600.0 * fs))
    for hour_index in range(int(total_hours)):
        h0_sample = hour_index * hour_samples
        h1_sample = min((hour_index + 1) * hour_samples, int(total_samples))
        h0_sec = float(h0_sample / fs)
        h1_sec = float(h1_sample / fs)
        duration_sec = max(0.0, h1_sec - h0_sec)
        minute0 = int(h0_sample // minute_samples)
        minute1 = int(np.ceil(h1_sample / minute_samples))

        total_count = 0
        weighted_sum = np.zeros(wf_len, dtype=np.float64)
        crossing_parts: list[np.ndarray] = []
        for minute_index in range(minute0, minute1):
            buf = state["minute_buffers"].get(minute_index)
            if buf is None or int(buf["n_spikes"]) <= 0:
                continue
            count = int(buf["n_spikes"])
            total_count += count
            weighted_sum += np.asarray(buf["waveform_sum"], dtype=np.float64)
            crossing_parts.append(np.asarray(buf["crossings"], dtype=np.int64))

        crossings = (
            np.concatenate(crossing_parts).astype(np.int64)
            if crossing_parts
            else np.zeros(0, dtype=np.int64)
        )
        crossings.sort()
        timestamps = crossings.astype(np.float64) / fs
        if total_count > 0:
            mean_wf = (weighted_sum / float(total_count)).astype(np.float32)
        else:
            mean_wf = np.full(wf_len, np.nan, dtype=np.float32)
        summary = _mean_waveform_summary(mean_wf, total_count, fs, wf_len)
        cv2 = _compute_cv2_from_timestamps(timestamps)
        if COMPUTE_HOURLY_CORRELOGRAMS:
            isi_sec, correlogram_lag_ms, correlogram_counts = _isi_and_correlogram(timestamps)
        else:
            isi_sec, correlogram_lag_ms, correlogram_counts = _isi_only_and_empty_correlogram(timestamps)
        firing_rate_hz = float(total_count / duration_sec) if duration_sec > 0 else 0.0
        npz_path = hourly_dir / f"{state['out_base'].name}_hour_{hour_index:04d}_isi_correlogram.npz"
        t_hour_npz0 = time.perf_counter()
        np.savez_compressed(
            str(npz_path),
            crossing_samples=crossings,
            timestamps_sec=timestamps,
            mean_waveform_uv=summary["mean_waveform_uv"],
            isi_sec=isi_sec,
            correlogram_lag_ms=correlogram_lag_ms,
            correlogram_counts=correlogram_counts,
            hour_index=np.array([hour_index], dtype=np.int32),
            hour_start_sec=np.array([h0_sec], dtype=np.float64),
            hour_end_sec=np.array([h1_sec], dtype=np.float64),
            hour_duration_sec=np.array([duration_sec], dtype=np.float64),
        )
        hour_npz_elapsed = time.perf_counter() - t_hour_npz0
        _add_timing(timing_totals, "detection_hourly_npz_write", hour_npz_elapsed)
        _add_timing(timing_totals, "detection_npz_write", hour_npz_elapsed)
        fig_path = hourly_dir / f"{state['out_base'].name}_hour_{hour_index:04d}_sgch{state['sg_ch']}_n{total_count}_mean_waveform.png"
        t_plot0 = time.perf_counter()
        _save_mean_waveform_plot(
            summary["mean_waveform_uv"],
            n_spikes=total_count,
            fs=fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
            sg_ch=int(state["sg_ch"]),
            t_start_min=h0_sec / 60.0,
            t_end_min=h1_sec / 60.0,
            out_png=fig_path,
            fixed_y_center=float(state["fixed_y_center"] if state["fixed_y_center"] is not None else 0.0),
            fixed_y_half=float(state["fixed_y_half"] if state["fixed_y_half"] is not None else max(1.0, 1.2 * float(state["threshold_uv"]))),
        )
        _add_timing(timing_totals, "detection_plot_write", time.perf_counter() - t_plot0)
        state["hourly_rows"].append(
            {
                "hour_index": hour_index,
                "time_start_sec": h0_sec,
                "time_end_sec": h1_sec,
                "duration_sec": duration_sec,
                "sg_ch": int(state["sg_ch"]),
                "threshold_min_uv": float(state["threshold_uv"]),
                "threshold_max_uv": _safe_float(state.get("threshold_max_uv")),
                "n_spikes": int(total_count),
                "firing_rate_hz": firing_rate_hz,
                "amplitude_ptp_uv": _safe_float(summary["amplitude_ptp_uv"]),
                "mean_abs_waveform_uv": _safe_float(summary["mean_abs_waveform_uv"]),
                "cv2": _safe_float(cv2),
                "peak_to_trough_ms": _safe_float(summary["peak_to_trough_ms"]),
                "isi_mean_ms": _safe_float(float(np.mean(isi_sec) * 1000.0) if isi_sec.size else float("nan")),
                "isi_median_ms": _safe_float(float(np.median(isi_sec) * 1000.0) if isi_sec.size else float("nan")),
                "npz": str(npz_path.resolve()),
                "figure": str(fig_path.resolve()),
            }
        )
    state["hourly_rows"].sort(key=lambda row: int(row["hour_index"]))


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


def process_recording_save_per_chunk_multi_channel(
    rec: BaseRecording,
    rec_file: Path,
    chan_ids,
    *,
    run_output_dir: Path,
    meta_run: dict,
    channel_threshold_pairs: list[dict],
    session_ordinal: int,
    session_cumulative_sample_offset: int,
    session_cumulative_time_offset_sec: float,
    fs: float,
    chunk_samples: int,
    polarity: str,
    refractory_samples: int,
    pre_samples: int,
    post_samples: int,
    resume: bool,
    progress: bool = True,
    progress_prefix: str = "  ",
    timing_totals: dict[str, float] | None = None,
) -> int:
    """
    Chunk-first threshold detection for all unfinished configured channels.

    Output files and per-recording summaries intentionally match
    process_recording_save_per_chunk(), but traces are loaded once per chunk for
    all unfinished channels instead of once per channel/chunk.
    """
    n = rec.get_num_samples()
    dur_s = n / fs
    wf_len = pre_samples + post_samples
    n_chunks = max(1, (n + chunk_samples - 1) // chunk_samples)
    parent, stem = _recording_parent_stem_safe(rec_file)
    t_detect0 = time.perf_counter()
    t_func0 = t_detect0

    completed_pairs = 0
    states: list[dict] = []
    for pair in channel_threshold_pairs:
        sg_ch = int(pair["sg_ch"])
        rec_idx_local = sg_ch
        threshold_uv = float(pair["threshold_uv"])
        threshold_max_uv = (
            float(pair["threshold_max_uv"])
            if pair.get("threshold_max_uv", None) is not None
            else None
        )
        if rec_idx_local >= len(chan_ids):
            print(
                f"  [skip] sg_ch={sg_ch} rec_idx={rec_idx_local} out of range for this file.",
                flush=True,
            )
            continue

        channel_id = chan_ids[rec_idx_local]
        pair_folder = _pair_folder_name(sg_ch, threshold_uv, threshold_max_uv)
        pair_dir = run_output_dir / pair_folder
        pair_dir.mkdir(parents=True, exist_ok=True)
        out_base = pair_dir / f"{parent}__{stem}"
        summary_path = _recording_summary_path_from_out_base(out_base)

        if resume and _is_recording_summary_complete(summary_path):
            print(
                f"  [skip] already complete: sg_ch={sg_ch}, thr={threshold_uv:.3f} uV",
                flush=True,
            )
            completed_pairs += 1
            continue

        pair_id = _pair_folder_name(sg_ch, threshold_uv, threshold_max_uv)
        states.append(
            {
                "pair_index": len(states),
                "pair_id": pair_id,
                "sg_ch": sg_ch,
                "threshold_uv": threshold_uv,
                "threshold_max_uv": threshold_max_uv,
                "threshold_interval": str(pair.get("threshold_interval", "lower_inclusive_upper_exclusive")),
                "channel_id": channel_id,
                "out_base": out_base,
                "summary_path": summary_path,
                "last_kept": -10**18,
                "total_events": 0,
                "manifest": [],
                "fixed_y_center": None,
                "fixed_y_half": None,
                "minute_rows": [],
                "hourly_rows": [],
                "minute_buffers": {},
            }
        )

    if not states:
        _add_timing(timing_totals, "detection_total", time.perf_counter() - t_func0)
        return completed_pairs

    print(
        f"  Running chunk-first detection for {len(states)} unfinished channel/threshold pair(s)...",
        flush=True,
    )

    c0 = 0
    chunk_i = 0
    legacy_chunk_resume_enabled = False
    while c0 < n:
        chunk_i += 1
        c1 = min(n, c0 + chunk_samples)
        t_chunk0 = time.perf_counter()
        t_start_min = (c0 / fs) / 60.0
        t_end_min = (min(c1, n) / fs) / 60.0
        min_tag = _minutes_tag(t_start_min, t_end_min)
        chunk_tag = f"chunk_{chunk_i:04d}"
        states_to_compute: list[dict] = []
        reused_events = 0

        for state in states:
            sg_ch = int(state["sg_ch"])
            threshold_uv = float(state["threshold_uv"])
            out_base = state["out_base"]
            npz_path = Path(f"{out_base}_{chunk_tag}_threshold_crossings.npz")

            t_reuse0 = time.perf_counter()
            if legacy_chunk_resume_enabled and resume and npz_path.exists():
                npz = None
                wfs_uv = None
                mean_abs_for_name = 0.0
                try:
                    npz = np.load(str(npz_path), allow_pickle=False)
                    crossings_samples = npz["crossing_samples"]
                    n_chunk = int(crossings_samples.shape[0])
                    if n_chunk > 0:
                        state["last_kept"] = max(
                            int(state["last_kept"]),
                            int(crossings_samples[-1]),
                        )

                    if state["fixed_y_center"] is None or state["fixed_y_half"] is None:
                        if n_chunk > 0:
                            wfs_uv = npz["waveforms_uv"]
                            mean_wf = wfs_uv.mean(axis=0)
                            mean_min = float(np.min(mean_wf))
                            mean_max = float(np.max(mean_wf))
                            mean_range = mean_max - mean_min
                            if mean_range <= 0:
                                mean_range = max(1e-6, float(np.max(np.abs(mean_wf))))
                            y_span = 1.2 * mean_range
                            state["fixed_y_half"] = y_span / 2.0
                            state["fixed_y_center"] = 0.5 * (mean_min + mean_max)
                            mean_abs_for_name = float(np.mean(np.abs(mean_wf)))
                        else:
                            state["fixed_y_center"] = 0.0
                            state["fixed_y_half"] = max(1.0, 1.2 * threshold_uv)

                    fig_pattern = (
                        f"{out_base.name}_{chunk_tag}_sgch{sg_ch}_{min_tag}_"
                        f"n{n_chunk}_meanAbs*uV_waveforms.png"
                    )
                    fig_candidates = list(out_base.parent.glob(fig_pattern))
                    fig_path = fig_candidates[0] if fig_candidates else None

                    if fig_path is None or not fig_path.exists():
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
                            fixed_y_center=float(state["fixed_y_center"]),
                            fixed_y_half=float(state["fixed_y_half"]),
                        )

                    state["manifest"].append(
                        {
                            "chunk_index": chunk_i,
                            "n_crossings": int(n_chunk),
                            "time_start_sec": float(c0 / fs),
                            "time_end_sec": float(min(c1, n) / fs),
                            "npz": str(npz_path.resolve()),
                            "figure": str(fig_path.resolve()),
                        }
                    )
                    state["total_events"] += int(n_chunk)
                    reused_events += int(n_chunk)
                    del crossings_samples
                    if wfs_uv is not None:
                        del wfs_uv
                    continue
                except Exception:
                    pass
                finally:
                    if npz is not None:
                        try:
                            npz.close()
                        except Exception:
                            pass
                    _add_timing(
                        timing_totals,
                        "detection_resume_reuse",
                        time.perf_counter() - t_reuse0,
                    )

            states_to_compute.append(state)

        computed_events = 0
        if states_to_compute:
            buf_start = max(0, c0 - pre_samples - 1)
            buf_end = min(n, c1 + post_samples + 1)
            unique_channel_ids = []
            channel_id_to_col = {}
            for state in states_to_compute:
                channel_id = state["channel_id"]
                if channel_id in channel_id_to_col:
                    continue
                channel_id_to_col[channel_id] = len(unique_channel_ids)
                unique_channel_ids.append(channel_id)
            t_trace0 = time.perf_counter()
            traces = rec.get_traces(
                start_frame=buf_start,
                end_frame=buf_end,
                channel_ids=unique_channel_ids,
                return_scaled=True,
            )
            traces = traces.astype(np.float32, copy=False)
            _add_timing(
                timing_totals,
                "detection_trace_read",
                time.perf_counter() - t_trace0,
            )

            det_start = max(c0, 1)
            det_end = c1
            start_rel = det_start - buf_start
            end_rel = det_end - buf_start

            for state in states_to_compute:
                sg_ch = int(state["sg_ch"])
                threshold_uv = float(state["threshold_uv"])
                out_base = state["out_base"]
                col_i = int(channel_id_to_col[state["channel_id"]])
                x = traces[:, col_i]

                t_thresh0 = time.perf_counter()
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

                minute_samples = int(round(60.0 * fs))
                cap = 1024
                cross_buf = np.empty(cap, dtype=np.int64)
                chunk_waveform_sum = np.zeros(wf_len, dtype=np.float64)
                n_chunk = 0
                for g in merge_refractory(global_cross, refractory_samples):
                    if g - int(state["last_kept"]) < refractory_samples:
                        continue
                    loc = int(g - buf_start)
                    if loc < pre_samples or loc + post_samples > x.shape[0]:
                        continue
                    candidate_wf = x[loc - pre_samples : loc + post_samples]
                    threshold_max_uv = state.get("threshold_max_uv")
                    if threshold_max_uv is not None:
                        amp_uv = _event_amplitude_uv(candidate_wf, polarity)
                        if str(state.get("threshold_interval", "lower_inclusive_upper_exclusive")) == "lower_inclusive_upper_exclusive":
                            if amp_uv >= float(threshold_max_uv):
                                continue
                        elif amp_uv > float(threshold_max_uv):
                            continue
                    cross_buf = _ensure_crossing_capacity(cross_buf, n_chunk)
                    cross_buf[n_chunk] = int(g)
                    chunk_waveform_sum += np.asarray(candidate_wf, dtype=np.float64)
                    _add_event_to_minute_buffer(
                        state,
                        int(g),
                        candidate_wf,
                        minute_samples,
                        wf_len,
                    )
                    n_chunk += 1
                    state["last_kept"] = int(g)

                mean_abs_for_name = 0.0
                if n_chunk > 0:
                    mean_wf = (chunk_waveform_sum / float(n_chunk)).astype(np.float32)
                    mean_abs_for_name = float(np.mean(np.abs(mean_wf)))
                    if state["fixed_y_center"] is None or state["fixed_y_half"] is None:
                        mean_min = float(np.min(mean_wf))
                        mean_max = float(np.max(mean_wf))
                        mean_range = mean_max - mean_min
                        if mean_range <= 0:
                            mean_range = max(1e-6, float(np.max(np.abs(mean_wf))))
                        y_span = 1.2 * mean_range
                        state["fixed_y_half"] = y_span / 2.0
                        state["fixed_y_center"] = 0.5 * (mean_min + mean_max)
                if state["fixed_y_center"] is None or state["fixed_y_half"] is None:
                    state["fixed_y_center"] = 0.0
                    state["fixed_y_half"] = max(1.0, 1.2 * threshold_uv)

                crossings = (
                    cross_buf[:n_chunk].copy() if n_chunk else np.zeros(0, dtype=np.int64)
                )
                del cross_buf, chunk_waveform_sum
                _add_timing(
                    timing_totals,
                    "detection_thresholding",
                    time.perf_counter() - t_thresh0,
                )

                state["manifest"].append(
                    {
                        "chunk_index": chunk_i,
                        "n_crossings": int(n_chunk),
                        "time_start_sec": float(c0 / fs),
                        "time_end_sec": float(min(c1, n) / fs),
                        "npz": None,
                        "figure": None,
                    }
                )
                state["total_events"] += int(n_chunk)
                computed_events += int(n_chunk)

                del crossings

            del traces

        t_gc0 = time.perf_counter()
        gc.collect()
        _add_timing(timing_totals, "detection_gc", time.perf_counter() - t_gc0)

        if progress:
            t_end = min(c1, n) / fs
            pct = 100.0 * t_end / (n / fs) if n else 100.0
            chunk_wall = time.perf_counter() - t_chunk0
            elapsed = time.perf_counter() - t_detect0
            print(
                f"{progress_prefix}chunk {chunk_i}/{n_chunks}  "
                f"rec time {c0/fs:.2f}-{t_end:.2f} s  ({pct:.1f}%)  "
                f"channels computed {len(states_to_compute)} reused {len(states) - len(states_to_compute)}  "
                f"events computed {computed_events} reused {reused_events}  "
                f"chunk {chunk_wall:.2f}s  elapsed {elapsed:.1f}s  (saved+released)",
                flush=True,
            )

        c0 = c1

    total_minutes = max(1, int(np.ceil(float(n) / fs / 60.0)))
    total_hours = max(1, int(np.ceil(float(n) / fs / 3600.0)))
    t_finalize0 = time.perf_counter()
    recording_minute_rows = _write_recording_minute_npz_outputs(
        states,
        run_output_dir=run_output_dir,
        parent=parent,
        stem=stem,
        rec_file=rec_file,
        total_minutes=total_minutes,
        total_samples=int(n),
        fs=fs,
        wf_len=wf_len,
        session_ordinal=session_ordinal,
        session_cumulative_sample_offset=int(session_cumulative_sample_offset),
        session_cumulative_time_offset_sec=float(session_cumulative_time_offset_sec),
        timing_totals=timing_totals,
    )
    _add_timing(timing_totals, "detection_minute_finalize", time.perf_counter() - t_finalize0)
    recording_minute_summary_csv = run_output_dir / "minute_npz" / f"{parent}__{stem}_minute_summary.csv"
    recording_minute_summary_json = run_output_dir / "minute_npz" / f"{parent}__{stem}_minute_summary.json"
    t_recording_minute_summary0 = time.perf_counter()
    _write_csv_rows(recording_minute_summary_csv, recording_minute_rows)
    recording_minute_summary_json.write_text(
        json.dumps(_jsonable_rows(recording_minute_rows), indent=2),
        encoding="utf-8",
    )
    _add_timing(
        timing_totals,
        "recording_minute_summary_write",
        time.perf_counter() - t_recording_minute_summary0,
    )

    for state in states:
        t_hourly_finalize0 = time.perf_counter()
        _write_hourly_outputs_from_minute_means(
            state,
            total_hours=total_hours,
            total_samples=int(n),
            fs=fs,
            wf_len=wf_len,
            pre_samples=pre_samples,
            post_samples=post_samples,
            timing_totals=timing_totals,
        )
        _add_timing(timing_totals, "detection_hourly_finalize", time.perf_counter() - t_hourly_finalize0)
        minute_summary_csv = state["out_base"].parent / f"{state['out_base'].name}_minute_summary.csv"
        minute_summary_json = state["out_base"].parent / f"{state['out_base'].name}_minute_summary.json"
        hourly_summary_csv = state["out_base"].parent / f"{state['out_base'].name}_hourly_summary.csv"
        hourly_summary_json = state["out_base"].parent / f"{state['out_base'].name}_hourly_summary.json"
        t_pair_summary0 = time.perf_counter()
        _write_csv_rows(minute_summary_csv, state["minute_rows"])
        minute_summary_json.write_text(
            json.dumps(_jsonable_rows(state["minute_rows"]), indent=2),
            encoding="utf-8",
        )
        _write_csv_rows(hourly_summary_csv, state["hourly_rows"])
        hourly_summary_json.write_text(
            json.dumps(_jsonable_rows(state["hourly_rows"]), indent=2),
            encoding="utf-8",
        )
        _add_timing(timing_totals, "pair_minute_hourly_summary_write", time.perf_counter() - t_pair_summary0)
        per_rec = {
            "rec_file": str(rec_file.resolve()),
            "n_crossings": int(state["total_events"]),
            "seconds": float(n / fs),
            "output_summary": str(state["summary_path"].resolve()),
            "preprocessing": meta_run["preprocessing"],
            "session_ordinal": session_ordinal,
            "cumulative_segment_start_sample": int(session_cumulative_sample_offset),
            "cumulative_segment_start_sec": float(session_cumulative_time_offset_sec),
            "cumulative_segment_end_sample": int(session_cumulative_sample_offset + n),
            "cumulative_segment_end_sec": float(session_cumulative_time_offset_sec + dur_s),
            "sg_ch": int(state["sg_ch"]),
            "threshold_uv": float(state["threshold_uv"]),
            "threshold_min_uv": float(state["threshold_uv"]),
            "threshold_max_uv": _safe_float(state.get("threshold_max_uv")),
            "chunks": state["manifest"],
            "minute_summary_csv": str(minute_summary_csv.resolve()),
            "minute_summary_json": str(minute_summary_json.resolve()),
            "recording_minute_summary_csv": str(recording_minute_summary_csv.resolve()),
            "recording_minute_summary_json": str(recording_minute_summary_json.resolve()),
            "hourly_summary_csv": str(hourly_summary_csv.resolve()),
            "hourly_summary_json": str(hourly_summary_json.resolve()),
            "minute_npz_folder": str((run_output_dir / "minute_npz" / f"{parent}__{stem}").resolve()),
            "hourly_folder": str((state["out_base"].parent / "hourly").resolve()),
            "output_resolution_note": (
                "Processing chunks are internal. Stable outputs are one recording-level "
                "NPZ per minute containing every channel/range pair, per-pair minute "
                "CSV/JSON summaries, hourly ISI/correlogram NPZ files, and one hourly "
                "mean-waveform figure per hour."
            ),
        }
        t_summary0 = time.perf_counter()
        state["summary_path"].write_text(json.dumps(per_rec, indent=2), encoding="utf-8")
        _add_timing(
            timing_totals,
            "recording_summary_write",
            time.perf_counter() - t_summary0,
        )
        completed_pairs += 1
        print(
            f"  Pair done: sg_ch={int(state['sg_ch'])}, "
            f"thr={_threshold_label(float(state['threshold_uv']), state.get('threshold_max_uv'))} "
            f"-> {per_rec['n_crossings']} events.",
            flush=True,
        )

    _add_timing(timing_totals, "detection_total", time.perf_counter() - t_func0)
    return completed_pairs


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


def collect_recording_files_from_input(input_path: Path) -> list[Path]:
    """
    Accept one recording path or one folder.

    - A .rec file is processed by itself.
    - A .rec directory is resolved to its contained .rec file.
    - A normal folder contributes all Chronic_Rec_*.rec files under it, sorted
      by the timestamp in the filename.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Recording path not found: {input_path}")

    if input_path.suffix.lower() == ".rec":
        return [resolve_spikegadgets_rec_file(input_path).resolve()]

    if input_path.is_file():
        raise ValueError(f"Recording input file must end with .rec: {input_path}")

    rec_files = discover_chronic_rec_files(input_path)
    if not rec_files:
        raise FileNotFoundError(f"No Chronic_Rec_*.rec files found under: {input_path}")
    return rec_files


def collect_recording_files_from_inputs(input_paths: list[Path]) -> list[Path]:
    rec_files: list[Path] = []
    seen: set[Path] = set()
    for input_path in input_paths:
        for rec_file in collect_recording_files_from_input(input_path):
            resolved = rec_file.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rec_files.append(resolved)

    if not rec_files:
        raise FileNotFoundError("No recording files found in the provided input paths.")

    def _sort_key(path: Path) -> tuple[int, str]:
        key = chronic_rec_sort_key(path)
        return (key if key is not None else 10**18, str(path))

    return sorted(rec_files, key=_sort_key)


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


def _pair_folder_name(
    sg_ch: int,
    threshold_uv: float,
    threshold_max_uv: float | None = None,
) -> str:
    min_text = _format_threshold_for_folder(threshold_uv)
    if threshold_max_uv is None:
        return f"sgch{int(sg_ch)}_thr{min_text}uV"
    max_text = _format_threshold_for_folder(threshold_max_uv)
    return f"sgch{int(sg_ch)}_thr{min_text}to{max_text}uV"


def _threshold_label(threshold_uv: float, threshold_max_uv: float | None = None) -> str:
    if threshold_max_uv is None:
        return f"{float(threshold_uv):.3f} uV"
    return f"{float(threshold_uv):.3f}-{float(threshold_max_uv):.3f} uV"


def _pair_out_base(
    run_output_dir: Path,
    rec_file: Path,
    sg_ch: int,
    threshold_uv: float,
    threshold_max_uv: float | None = None,
) -> Path:
    parent, stem = _recording_parent_stem_safe(rec_file)
    pair_dir = run_output_dir / _pair_folder_name(sg_ch, threshold_uv, threshold_max_uv)
    # Matches existing naming: <parent>__<stem>
    return pair_dir / f"{parent}__{stem}"


def _recording_summary_path_from_out_base(out_base: Path) -> Path:
    return out_base.parent / f"{out_base.name}_recording_summary.json"


def _is_recording_summary_complete(summary_path: Path) -> bool:
    """
    Decide whether a (recording, sg_ch, threshold_uv) pair is "done".

    We require the summary JSON plus minute/hourly summary artifacts to exist.
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
        if npz_s and not Path(npz_s).exists():
            return False
        fig_s = ch.get("figure", None)
        if fig_s and not Path(fig_s).exists():
            return False
    for key in (
        "minute_summary_csv",
        "minute_summary_json",
        "recording_minute_summary_csv",
        "recording_minute_summary_json",
        "hourly_summary_csv",
        "hourly_summary_json",
    ):
        path_s = per_rec.get(key, None)
        if path_s and not Path(path_s).exists():
            return False
    minute_npz_folder = per_rec.get("minute_npz_folder", None)
    if minute_npz_folder and not Path(minute_npz_folder).exists():
        return False
    return True


def load_channel_threshold_pairs(config_path: Path) -> list[dict]:
    """
    Load channel threshold pairs/ranges from JSON.

    Supported formats:
      1) JSON list:
         [{"sg_ch": 72, "threshold_uv": 500.0}, {"sg_ch": 73, "threshold_uv": 600.0}]
      2) JSON object with "pairs":
         {"pairs": [{"sg_ch": 72, "threshold_uv": 500.0, "threshold_max_uv": 900.0}, ...]}
      3) JSON mapping sg_ch -> threshold:
         {"72": 500.0, "73": 600.0}

    Each entry must provide:
      - sg_ch (or "channel") as int
      - threshold_uv (or "threshold") as float (magnitude, must be > 0)
      - optional threshold_max_uv as upper amplitude bound
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8-sig"))

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
        threshold_max_uv = obj.get("threshold_max_uv", obj.get("threshold_max", None))
        if threshold_max_uv is not None:
            threshold_max_uv = float(threshold_max_uv)
            if threshold_max_uv <= thr:
                raise ValueError(
                    f"threshold_max_uv must be greater than threshold_uv for sg_ch={sg_ch}. "
                    f"Got threshold_uv={thr}, threshold_max_uv={threshold_max_uv}"
                )
        interval = str(obj.get("threshold_interval", "lower_inclusive_upper_exclusive"))
        return {
            "sg_ch": sg_ch,
            "threshold_uv": thr,
            "threshold_max_uv": threshold_max_uv,
            "threshold_interval": interval,
        }

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

    # De-duplicate identical ranges while preserving order.
    seen: set[tuple[int, float, float | None]] = set()
    out: list[dict] = []
    for p in pairs:
        threshold_max_uv = (
            float(p["threshold_max_uv"])
            if p.get("threshold_max_uv", None) is not None
            else None
        )
        key = (int(p["sg_ch"]), float(p["threshold_uv"]), threshold_max_uv)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "sg_ch": int(p["sg_ch"]),
                "threshold_uv": float(p["threshold_uv"]),
                "threshold_max_uv": threshold_max_uv,
                "threshold_interval": str(p.get("threshold_interval", "lower_inclusive_upper_exclusive")),
            }
        )
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


def prompt_path(message: str, default: str) -> Path:
    return Path(prompt_line(message, default).strip().strip('"').strip("'"))


def parse_input_paths(raw_value: str) -> list[Path]:
    parts = [
        part.strip().strip('"').strip("'")
        for part in re.split(r"[;\n]+", str(raw_value))
        if part.strip()
    ]
    if not parts:
        raise ValueError("No recording path provided.")
    paths = [Path(part) for part in parts]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Recording path(s) not found: {missing}")
    return paths


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


def _recording_date_folder_label(rec_file: Path) -> str:
    key = chronic_rec_sort_key(rec_file)
    if key is None:
        parent, stem = _recording_parent_stem_safe(rec_file)
        return f"unknown_rec_date_{parent}__{stem}"
    return str(key)[:8]


def group_recording_files_by_date(rec_files: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for rec_file in rec_files:
        date_label = _recording_date_folder_label(rec_file)
        groups.setdefault(date_label, []).append(rec_file)
    return groups


def make_recording_date_output_plan(
    output_parent: Path,
    rec_files: list[Path],
) -> dict[str, tuple[Path, list[Path]]]:
    """One output folder per recording date; fail instead of overwriting."""
    output_parent = Path(output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)
    groups = group_recording_files_by_date(rec_files)
    plan: dict[str, tuple[Path, list[Path]]] = {}
    existing: list[Path] = []
    for date_label, date_rec_files in groups.items():
        out_dir = output_parent / f"threshold_crossings_{date_label}"
        if out_dir.exists():
            existing.append(out_dir)
        plan[date_label] = (out_dir, date_rec_files)
    if existing:
        formatted = "\n".join(f"  {p}" for p in existing)
        raise FileExistsError(
            "Output folder(s) already exist; overwrite is disabled.\n"
            f"{formatted}\n"
            "Move or rename the existing folder(s), or use Resume previous interrupted session."
        )
    return plan


def _add_timing(timing: dict[str, float] | None, key: str, seconds: float) -> None:
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + float(seconds)


def _format_elapsed(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)} min {sec:.1f} s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours} h {minutes} min {sec:.1f} s"


TIMING_STEP_ROWS = (
    ("Resume complete-file skips", "resume_skip_complete_recording"),
    ("Load recording + HW map + probe", "load_recording"),
    ("Bandpass setup", "bandpass_setup"),
    ("First-10s preview plots", "preview_plots"),
    ("Detection total", "detection_total"),
    ("  trace read/filter realization", "detection_trace_read"),
    ("  threshold + waveform extraction", "detection_thresholding"),
    ("  minute finalization total", "detection_minute_finalize"),
    ("  hourly finalization total", "detection_hourly_finalize"),
    ("  NPZ writes total", "detection_npz_write"),
    ("    minute NPZ writes", "detection_minute_npz_write"),
    ("    hourly NPZ writes", "detection_hourly_npz_write"),
    ("  waveform PNG plots", "detection_plot_write"),
    ("  resume chunk reuse/checks", "detection_resume_reuse"),
    ("  garbage collection", "detection_gc"),
    ("Recording-level minute summary writes", "recording_minute_summary_write"),
    ("Pair minute/hourly summary writes", "pair_minute_hourly_summary_write"),
    ("Recording summary JSON writes", "recording_summary_write"),
)


def _print_timing_summary(timing: dict[str, float], total_wall: float) -> None:
    print("\nWall-time summary:", flush=True)
    major_accounted = (
        timing.get("resume_skip_complete_recording", 0.0)
        + timing.get("load_recording", 0.0)
        + timing.get("bandpass_setup", 0.0)
        + timing.get("preview_plots", 0.0)
        + timing.get("detection_total", 0.0)
    )
    rows = [
        ("Total wall time", total_wall),
        *[(label, timing.get(key, 0.0)) for label, key in TIMING_STEP_ROWS],
        ("Other loop overhead", max(0.0, total_wall - major_accounted)),
    ]
    for label, seconds in rows:
        print(f"  {label:<34} {_format_elapsed(seconds)}", flush=True)


def _print_per_recording_timing_summary(per_recording_timings: list[dict]) -> None:
    if not per_recording_timings:
        return
    print("\nPer-recording wall-time summary:", flush=True)
    for row in per_recording_timings:
        status = str(row.get("status", ""))
        pairs = row.get("pairs_processed", "")
        pairs_total = row.get("pairs_total", "")
        pair_text = ""
        if pairs != "" and pairs_total != "":
            pair_text = f", pairs {pairs}/{pairs_total}"
        print(
            f"  [{row.get('recording_index')}/{row.get('recording_count')}] "
            f"{row.get('recording_name')} - {status}, {row.get('wall_time')}{pair_text}",
            flush=True,
        )
        for label, key in TIMING_STEP_ROWS:
            seconds = float(row.get(f"{key}_seconds", 0.0) or 0.0)
            if seconds <= 0.0:
                continue
            print(f"      {label:<36} {_format_elapsed(seconds)}", flush=True)


def _timing_delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return float(after.get(key, 0.0) - before.get(key, 0.0))


def _write_run_timing_reports(
    run_output_dir: Path,
    timing_totals: dict[str, float],
    per_recording_timings: list[dict],
    total_wall: float,
) -> None:
    total_rows = [
        {
            "metric": key,
            "seconds": float(value),
            "formatted": _format_elapsed(float(value)),
        }
        for key, value in sorted(timing_totals.items())
    ]
    total_rows.insert(
        0,
        {
            "metric": "total_wall_time",
            "seconds": float(total_wall),
            "formatted": _format_elapsed(total_wall),
        },
    )
    _write_csv_rows(run_output_dir / "run_timing_summary.csv", total_rows)
    (run_output_dir / "run_timing_summary.json").write_text(
        json.dumps(_jsonable_rows(total_rows), indent=2),
        encoding="utf-8",
    )
    _write_csv_rows(run_output_dir / "per_recording_timing.csv", per_recording_timings)
    (run_output_dir / "per_recording_timing.json").write_text(
        json.dumps(_jsonable_rows(per_recording_timings), indent=2),
        encoding="utf-8",
    )
    step_rows: list[dict] = []
    for row in per_recording_timings:
        wall_seconds = float(row.get("wall_seconds", 0.0) or 0.0)
        step_rows.append(
            {
                "recording_index": int(row.get("recording_index", 0) or 0),
                "recording_count": int(row.get("recording_count", 0) or 0),
                "recording_file": row.get("recording_file", ""),
                "recording_name": row.get("recording_name", ""),
                "status": row.get("status", ""),
                "metric": "recording_wall_time",
                "label": "Recording wall time",
                "seconds": wall_seconds,
                "formatted": _format_elapsed(wall_seconds),
                "percent_of_recording_wall": 100.0 if wall_seconds > 0.0 else 0.0,
            }
        )
        for label, key in TIMING_STEP_ROWS:
            seconds = float(row.get(f"{key}_seconds", 0.0) or 0.0)
            step_rows.append(
                {
                    "recording_index": int(row.get("recording_index", 0) or 0),
                    "recording_count": int(row.get("recording_count", 0) or 0),
                    "recording_file": row.get("recording_file", ""),
                    "recording_name": row.get("recording_name", ""),
                    "status": row.get("status", ""),
                    "metric": key,
                    "label": label.strip(),
                    "seconds": seconds,
                    "formatted": _format_elapsed(seconds),
                    "percent_of_recording_wall": (
                        float(seconds / wall_seconds * 100.0)
                        if wall_seconds > 0.0
                        else 0.0
                    ),
                }
            )
    _write_csv_rows(run_output_dir / "per_recording_timing_steps.csv", step_rows)
    (run_output_dir / "per_recording_timing_steps.json").write_text(
        json.dumps(_jsonable_rows(step_rows), indent=2),
        encoding="utf-8",
    )


def _recording_timing_row(
    *,
    recording_index: int,
    recording_count: int,
    rec_file: Path,
    status: str,
    wall_seconds: float,
    timing_before: dict[str, float],
    timing_after: dict[str, float],
    n_samples: int | None = None,
    duration_seconds: float | None = None,
    pairs_processed: int | None = None,
    pairs_total: int | None = None,
    message: str = "",
) -> dict:
    row = {
        "recording_index": int(recording_index),
        "recording_count": int(recording_count),
        "recording_file": str(rec_file.resolve()),
        "recording_name": rec_file.name,
        "status": status,
        "message": message,
        "wall_seconds": float(wall_seconds),
        "wall_time": _format_elapsed(wall_seconds),
        "n_samples": "" if n_samples is None else int(n_samples),
        "duration_seconds": "" if duration_seconds is None else float(duration_seconds),
        "pairs_processed": "" if pairs_processed is None else int(pairs_processed),
        "pairs_total": "" if pairs_total is None else int(pairs_total),
    }
    for _, key in TIMING_STEP_ROWS:
        seconds = _timing_delta(timing_after, timing_before, key)
        row[f"{key}_seconds"] = seconds
        row[f"{key}_time"] = _format_elapsed(seconds)
    return row


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
    timing_totals: dict[str, float] = {}
    per_recording_timings: list[dict] = []
    n_files = len(rec_files)
    print(f"\nProcessing {n_files} recording(s)…\n", flush=True)
    print(
        "Cumulative timeline: sessions are chained in chronic-time order. "
        "NPZ fields *_cumulative place events on one continuous clock (see run_config.json).",
        flush=True,
    )
    print(
        "Processing chunks are internal. Stable outputs are one recording-level NPZ "
        "per minute, per-pair minute CSV/JSON summaries, hourly summaries, hourly "
        "mean-waveform figures, and timing reports.\n",
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
            threshold_max_uv = (
                float(pair["threshold_max_uv"])
                if pair.get("threshold_max_uv", None) is not None
                else None
            )
            pair_key = f"{sg_ch}__{threshold_uv:.9g}__{threshold_max_uv}"
            pair_dir = run_output_dir / _pair_folder_name(sg_ch, threshold_uv, threshold_max_uv)
            pattern = f"*_first10s_sgch{sg_ch}_filtered_trace_preview.png"
            if pair_dir.exists() and any(pair_dir.glob(pattern)):
                trace_preview_saved_for_pairs.add(pair_key)

    previews_written_for_first_loaded_record = not DEFAULT_SAVE_TRACE_PREVIEWS

    for fi, rec_file in enumerate(rec_files):
        t0 = time.perf_counter()
        timing_before = dict(timing_totals)
        print(f"--- [{fi + 1}/{n_files}] {rec_file.name} ---", flush=True)

        if resume:
            all_pairs_complete = True
            for pair in channel_threshold_pairs:
                sg_ch = int(pair["sg_ch"])
                threshold_uv = float(pair["threshold_uv"])
                threshold_max_uv = (
                    float(pair["threshold_max_uv"])
                    if pair.get("threshold_max_uv", None) is not None
                    else None
                )
                out_base = _pair_out_base(
                    run_output_dir, rec_file, sg_ch, threshold_uv, threshold_max_uv
                )
                summary_path = _recording_summary_path_from_out_base(out_base)
                if not _is_recording_summary_complete(summary_path):
                    all_pairs_complete = False
                    break
            if all_pairs_complete:
                first_pair = channel_threshold_pairs[0]
                first_sg_ch = int(first_pair["sg_ch"])
                first_thr = float(first_pair["threshold_uv"])
                first_thr_max = (
                    float(first_pair["threshold_max_uv"])
                    if first_pair.get("threshold_max_uv", None) is not None
                    else None
                )
                first_out_base = _pair_out_base(
                    run_output_dir, rec_file, first_sg_ch, first_thr, first_thr_max
                )
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
                _add_timing(
                    timing_totals,
                    "resume_skip_complete_recording",
                    time.perf_counter() - t0,
                )
                per_recording_timings.append(
                    _recording_timing_row(
                        recording_index=fi + 1,
                        recording_count=n_files,
                        rec_file=rec_file,
                        status="skipped_complete",
                        wall_seconds=time.perf_counter() - t0,
                        timing_before=timing_before,
                        timing_after=timing_totals,
                        n_samples=n_samp,
                        duration_seconds=dur_s,
                        pairs_processed=len(channel_threshold_pairs),
                        pairs_total=len(channel_threshold_pairs),
                        message="All channel/threshold pairs were already complete.",
                    )
                )
                cumulative_sample_offset += int(n_samp)
                cumulative_time_offset_sec += float(dur_s)
                session_ordinal += 1
                continue

        print("  Loading recording (read + HW map + probe)…", flush=True)
        try:
            t_load = time.perf_counter()
            rec = load_recording_mapped(rec_file, fs, probe_json)
            load_elapsed = time.perf_counter() - t_load
            _add_timing(timing_totals, "load_recording", load_elapsed)
            print(f"  Loaded in {load_elapsed:.1f} s.", flush=True)
        except Exception as ex:
            _add_timing(timing_totals, "load_recording", time.perf_counter() - t_load)
            print(f"  [skip] load failed: {ex}", flush=True)
            per_recording_timings.append(
                _recording_timing_row(
                    recording_index=fi + 1,
                    recording_count=n_files,
                    rec_file=rec_file,
                    status="load_failed",
                    wall_seconds=time.perf_counter() - t0,
                    timing_before=timing_before,
                    timing_after=timing_totals,
                    pairs_processed=0,
                    pairs_total=len(channel_threshold_pairs),
                    message=str(ex),
                )
            )
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
            bp_elapsed = time.perf_counter() - t_bp
            _add_timing(timing_totals, "bandpass_setup", bp_elapsed)
            print(f"  Bandpass done in {bp_elapsed:.1f} s.", flush=True)
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
            t_preview0 = time.perf_counter()
            for pair in channel_threshold_pairs:
                sg_ch = int(pair["sg_ch"])
                threshold_uv = float(pair["threshold_uv"])
                threshold_max_uv = (
                    float(pair["threshold_max_uv"])
                    if pair.get("threshold_max_uv", None) is not None
                    else None
                )
                pair_key = f"{sg_ch}__{threshold_uv:.9g}__{threshold_max_uv}"
                if pair_key in trace_preview_saved_for_pairs:
                    continue
                rec_idx_local = sg_ch
                if rec_idx_local >= len(chan_ids):
                    continue
                channel_id = chan_ids[rec_idx_local]
                preview_path = (
                    run_output_dir
                    / _pair_folder_name(sg_ch, threshold_uv, threshold_max_uv)
                    / f"{parent}__{stem}_first10s_sgch{sg_ch}_filtered_trace_preview.png"
                )
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                print(
                    f"  Saving first 10 s trace preview for sg_ch={sg_ch}, "
                    f"thr={_threshold_label(threshold_uv, threshold_max_uv)}: {preview_path.name}…",
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
            _add_timing(timing_totals, "preview_plots", time.perf_counter() - t_preview0)

        pairs_processed = process_recording_save_per_chunk_multi_channel(
            rec,
            rec_file,
            chan_ids,
            run_output_dir=run_output_dir,
            meta_run=meta_run,
            channel_threshold_pairs=channel_threshold_pairs,
            session_ordinal=session_ordinal,
            session_cumulative_sample_offset=int(cumulative_sample_offset),
            session_cumulative_time_offset_sec=float(cumulative_time_offset_sec),
            fs=fs,
            chunk_samples=chunk_samples,
            polarity=polarity,
            refractory_samples=refractory_samples,
            pre_samples=pre_samples,
            post_samples=post_samples,
            resume=resume,
            progress=True,
            progress_prefix="  ",
            timing_totals=timing_totals,
        )
        for pair in ():
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
        per_recording_timings.append(
            _recording_timing_row(
                recording_index=fi + 1,
                recording_count=n_files,
                rec_file=rec_file,
                status="processed",
                wall_seconds=dt,
                timing_before=timing_before,
                timing_after=timing_totals,
                n_samples=n_samp,
                duration_seconds=dur_s,
                pairs_processed=pairs_processed,
                pairs_total=len(channel_threshold_pairs),
            )
        )
        print(
            f"  Done processing {pairs_processed}/{len(channel_threshold_pairs)} pair(s) for this recording in {dt:.1f} s wall.\n",
            flush=True,
        )
        cumulative_sample_offset += int(n_samp)
        cumulative_time_offset_sec += float(dur_s)
        session_ordinal += 1
        del rec

    total_wall = time.perf_counter() - t0_all
    _write_run_timing_reports(run_output_dir, timing_totals, per_recording_timings, total_wall)
    print(
        f"All recordings finished in {total_wall:.1f} s.\n"
        f"Outputs: {run_output_dir.resolve()}",
        flush=True,
    )
    print(
        "Timing reports: run_timing_summary.csv/json and per_recording_timing.csv/json",
        flush=True,
    )
    _print_per_recording_timing_summary(per_recording_timings)
    _print_timing_summary(timing_totals, total_wall)
    return 0


def main() -> int:
    print("=== SG channel threshold crossings (interactive) ===\n")

    resume_prev = prompt_yes_no("Resume previous interrupted session?", default_yes=False)
    if resume_prev:
        run_dir = prompt_path(
            "Directory of threshold_crossings_* recording-date folder",
            "",
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
            pair = {"sg_ch": int(p["sg_ch"]), "threshold_uv": float(p["threshold_uv"])}
            if p.get("threshold_max_uv", None) is not None:
                pair["threshold_max_uv"] = float(p["threshold_max_uv"])
            pair["threshold_interval"] = str(p.get("threshold_interval", "lower_inclusive_upper_exclusive"))
            channel_threshold_pairs.append(pair)

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

    recording_input_raw = prompt_line(
        "Recording .rec file/folder path(s), separated by semicolons",
        r"W:\260220_rec\Chronic_Rec_20260220_201706.rec",
    )
    try:
        recording_inputs = parse_input_paths(recording_input_raw)
        rec_files = collect_recording_files_from_inputs(recording_inputs)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    print(f"Will process {len(rec_files)} recording(s).")
    for i, p in enumerate(rec_files[:5]):
        print(f"  {i+1}. {p}")
    if len(rec_files) > 5:
        print(f"  ... and {len(rec_files) - 5} more")

    output_parent = DEFAULT_OUTPUT_PARENT
    try:
        output_plan = make_recording_date_output_plan(output_parent, rec_files)
    except FileExistsError as ex:
        print(str(ex), file=sys.stderr)
        return 1
    print("\nOutput folder(s):")
    for date_label, (run_output_dir, date_rec_files) in output_plan.items():
        print(
            f"  {date_label}: {run_output_dir.resolve()} "
            f"({len(date_rec_files)} recording(s))"
        )
    print(
        "Files written per recording-date folder: run_config.json; per recording a *_recording_summary.json, "
        "minute CSV/JSON/NPZ outputs, hourly CSV/JSON/NPZ outputs, hourly waveform PNGs, "
        "and timing reports.\n"
    )

    probe_json = DEFAULT_PROBE_JSON
    if not probe_json.exists():
        print(f"Probe file not found: {probe_json}", file=sys.stderr)
        return 1

    chunk_sec = DEFAULT_CHUNK_SEC
    fs = DEFAULT_SAMPLING_RATE_HZ
    polarity = DEFAULT_POLARITY
    refractory_ms = DEFAULT_REFRACTORY_MS
    pre_ms = DEFAULT_PRE_MS
    post_ms = DEFAULT_POST_MS

    apply_spikeband = DEFAULT_APPLY_SPIKEBAND
    bandpass_freq_min = DEFAULT_BANDPASS_FREQ_MIN
    bandpass_freq_max = DEFAULT_BANDPASS_FREQ_MAX
    if apply_spikeband and bandpass_freq_min >= bandpass_freq_max:
        print("bandpass_freq_min must be < bandpass_freq_max.", file=sys.stderr)
        return 1

    pi = read_probeinterface(str(probe_json))
    probe = pi.probes[0]
    sg_map = build_sg_to_recording_index(probe)
    config_path = DEFAULT_CHANNEL_THRESHOLD_CONFIG_JSON
    try:
        pairs = load_channel_threshold_pairs(config_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as ex:
        print(f"Failed to load channel/threshold JSON: {ex}", file=sys.stderr)
        return 1
    print(
        f"Using JSON channel/threshold pairs: {len(pairs)} pair(s) from {config_path}.",
        flush=True,
    )

    for p in pairs:
        sg_ch = int(p["sg_ch"])
        if sg_ch not in sg_map:
            print(
                f"sg_ch={sg_ch} not present in probe map (probe has {len(sg_map)} contacts).",
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
                "threshold_max_uv": _safe_float(p.get("threshold_max_uv")),
                "threshold_interval": str(p.get("threshold_interval", "lower_inclusive_upper_exclusive")),
            }
        )

    status = 0
    for date_label, (run_output_dir, date_rec_files) in output_plan.items():
        run_output_dir.mkdir(parents=False)
        meta_run = {
            "run_output_dir": str(run_output_dir.resolve()),
            "output_parent": str(Path(output_parent).resolve()),
            "recording_date": date_label,
            "output_structure": "one_folder_per_recording_date_no_overwrite",
            "recording_input": recording_input_raw,
            "recording_inputs": [str(path.resolve()) for path in recording_inputs],
            "recording_input_types": [
                "file" if path.is_file() else "folder"
                for path in recording_inputs
            ],
            "i_root": (
                str(recording_inputs[0].resolve())
                if len(recording_inputs) == 1 and recording_inputs[0].is_dir()
                else str(recording_inputs[0].parent.resolve())
            ),
            "all_recording_files_requested": [str(p.resolve()) for p in rec_files],
            "first_boundary_input": date_rec_files[0].name,
            "last_boundary_input": date_rec_files[-1].name,
            "first_sort_key": chronic_rec_sort_key(date_rec_files[0]),
            "last_sort_key": chronic_rec_sort_key(date_rec_files[-1]),
            "n_files": len(date_rec_files),
            "recording_files": [str(p.resolve()) for p in date_rec_files],
            "channel_threshold_mode": "json_channel_threshold_pairs",
            "channel_threshold_ranges": None,
            "channel_threshold_config": str(config_path.resolve()),
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
            "save_trace_previews": bool(DEFAULT_SAVE_TRACE_PREVIEWS),
            "saved_files_note": (
                "Per recording: minute_npz/<parent>__<stem> contains one aggregate NPZ per "
                "real recording minute with every channel/range pair. Per channel/range: "
                "<parent>__<stem>_recording_summary.json lists minute and hourly artifacts. "
                "Minute outputs include per-pair summary CSV/JSON rows pointing to the "
                "recording-level minute NPZ files. Hourly outputs include summary CSV/JSON, "
                "ISI/correlogram NPZ files, and mean-waveform PNGs. "
                "Cumulative sample/time columns chain sessions within this recording-date folder "
                "(see cumulative_timeline)."
            ),
            "cumulative_timeline": {
                "ordering": "Recordings in this date folder sorted by Chronic_Rec timestamp",
                "rule": "Offsets advance only after a file is fully processed. Skipped/failed files "
                "do not consume timeline space (next success abuts previous success).",
            },
        }
        (run_output_dir / "run_config.json").write_text(
            json.dumps(meta_run, indent=2), encoding="utf-8"
        )

        run_status = process_threshold_crossings_run(
            run_output_dir=run_output_dir,
            meta_run=meta_run,
            rec_files=date_rec_files,
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
        if run_status != 0:
            status = run_status

    return status


if __name__ == "__main__":
    raise SystemExit(main())
