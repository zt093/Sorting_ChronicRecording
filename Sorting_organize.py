from __future__ import annotations

"""
Build per-unit/per-minute feature caches beside SpikeInterface analyzers.

Input
-----
Pass one or more paths to `sorting_analyzer_analysis.zarr`, parent output
folders, or root folders containing many analyzers. Multiple paths can be
provided as separate command-line arguments or as a comma-separated prompt
value. The script always builds caches for all analyzers found under each input
path.

Output
------
For each input folder, this script writes organized outputs to a new sibling
folder ending in `_org` by default:

    S:/260221_Sorting
    S:/260221_Sorting_org

Use `--cache-root <folder>` to redirect output to:

    <cache-root>/<analyzer parent name>/unit_feature_cache/

Files written per analyzer
--------------------------
cache_metadata.json
    Provenance and run metadata: analyzer path, cache path, sampling frequency,
    session duration, bin size, unit IDs, saved analyzer extensions, output file
    paths, notes about waveform limitations, and elapsed build time.

unit_static_stats.csv
    One row per unit. Includes total spikes, overall firing rate, overall CV2,
    mean/median ISI, 2 ms ISI-violation count, mean/median absolute spike
    amplitude when available, quality-metric amplitude/SNR/firing rate/ISI
    violation ratio/num spikes, template best channel, template peak-to-peak,
    template peak-to-trough, saved waveform count, and sampled-waveform
    peak-to-trough.

unit_minute_stats.csv
    One row per unit per time bin, default 60 seconds. Includes bin start/end,
    spike count, firing rate, mean/median absolute spike amplitude, mean/median
    ISI, 2 ms ISI-violation count, CV2, saved waveform snippet count in that
    bin, peak-to-trough, and whether peak-to-trough came from saved waveform
    snippets or the unit template fallback.

unit_summary.csv
    One row per sorted unit with scalar fields useful for downstream analysis:
    session/analyzer/output paths, unit ID, shank/channel/SG channel metadata,
    amplitude median, firing rate, ISI violation ratio, SNR, num spikes,
    trough-to-peak duration, waveform image path, vector keys, spike-time keys,
    and metadata source labels.

unit_summary.json
    JSON version of the sorted-unit summary. Includes the same scalar fields plus
    embedded waveform and autocorrelogram similarity vectors for each unit.

unit_waveforms.npz
    Compact waveform arrays: unit IDs, sampling frequency, bin edges, full
    template waveform per unit, sampled mean waveform per unit, and sample
    indices for saved waveform snippets. With `--save-minute-waveforms`, also
    stores sampled mean waveform arrays per unit per bin.

unit_similarity_vectors.npz
    Ready-to-load waveform and autocorrelogram similarity vectors per unit.
    Includes per-unit keyed arrays plus stacked matrices when vector lengths are
    consistent across units.

unit_spike_times.npz
    Full spike sample indices and spike times in seconds for every unit.

unit_correlograms.npz
    Analyzer correlogram and ISI-histogram arrays, when those extensions exist.

sorted_unit_feature_outputs/
    Per sorted unit, writes an output organization compatible with the
    per-minute/per-hour feature layout used by Threshold_channel.py:
    <sg/unit folder>/<session>_unit*_recording_summary.json,
    <session>_unit*_minute_summary.csv/json, per-minute NPZ files,
    <session>_unit*_hourly_summary.csv/json, hourly ISI/correlogram NPZ files,
    and hourly waveform PNGs. These are sorted-unit outputs only, not
    thresholding outputs, so unit_id is included alongside sg_ch.

Notes
-----
Firing rate, ISI, CV2, and spike-amplitude stats use all available spikes.
Waveform-derived values are limited by the saved `waveforms/random_spikes`
extensions. In current analyzers, saved waveforms may be capped at 500 random
snippets per unit, so minute peak-to-trough falls back to the unit template when
a bin has no saved snippets.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si


ANALYZER_FOLDER_NAME = "sorting_analyzer_analysis.zarr"
FEATURE_CACHE_FOLDER_NAME = "unit_feature_cache"


def now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed(elapsed_seconds: float) -> str:
    if elapsed_seconds < 60:
        return f"{elapsed_seconds:.2f}s"
    if elapsed_seconds < 3600:
        return f"{elapsed_seconds / 60.0:.2f}min"
    return f"{elapsed_seconds / 3600.0:.2f}h"


def log_progress(message: str, start_time: float | None = None) -> None:
    if start_time is None:
        print(f"[{now_label()}] {message}", flush=True)
    else:
        elapsed = perf_counter() - start_time
        print(f"[{now_label()} | +{format_elapsed(elapsed)}] {message}", flush=True)


def add_timing(timing: dict[str, float], key: str, start_time: float) -> None:
    timing[key] = timing.get(key, 0.0) + float(perf_counter() - start_time)


def write_timing_reports(
    *,
    cache_folder: Path,
    timing: dict[str, float],
    unit_timing_rows: list[dict],
    total_elapsed: float,
) -> dict[str, str]:
    summary_rows = [
        {
            "step": "total",
            "seconds": float(total_elapsed),
            "elapsed": format_elapsed(total_elapsed),
        }
    ]
    summary_rows.extend(
        {
            "step": key,
            "seconds": float(value),
            "elapsed": format_elapsed(value),
        }
        for key, value in sorted(timing.items())
    )
    summary_csv = cache_folder / "processing_timing_summary.csv"
    summary_json = cache_folder / "processing_timing_summary.json"
    per_unit_csv = cache_folder / "processing_timing_per_unit.csv"
    per_unit_json = cache_folder / "processing_timing_per_unit.json"
    write_csv_auto(summary_csv, summary_rows)
    write_json_rows(summary_json, summary_rows)
    write_csv_auto(per_unit_csv, unit_timing_rows)
    write_json_rows(per_unit_json, unit_timing_rows)
    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "per_unit_csv": str(per_unit_csv),
        "per_unit_json": str(per_unit_json),
    }


def discover_analyzer_folders(path: Path) -> list[Path]:
    """Accept an analyzer folder, an output folder, or a root containing analyzers."""
    if path.name == ANALYZER_FOLDER_NAME and path.exists():
        return [path]

    direct = path / ANALYZER_FOLDER_NAME
    if direct.exists():
        return [direct]

    if path.exists():
        return sorted(path.rglob(ANALYZER_FOLDER_NAME))
    return []


def select_analyzer_folders(path: Path) -> list[Path]:
    analyzer_folders = discover_analyzer_folders(path)
    if not analyzer_folders:
        raise FileNotFoundError(
            f"Could not find {ANALYZER_FOLDER_NAME} at or under: {path.resolve()}"
        )

    log_progress(f"Found {len(analyzer_folders)} analyzer folder(s); building all caches")
    return analyzer_folders


def parse_input_paths_text(raw_text: str) -> list[Path]:
    parts = [part.strip().strip('"').strip("'") for part in raw_text.split(",")]
    paths = [Path(part).expanduser() for part in parts if part]
    if not paths:
        raise ValueError("No input paths were provided.")
    return paths


def parse_input_paths(raw_values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw_value in raw_values:
        paths.extend(parse_input_paths_text(raw_value))
    if not paths:
        raise ValueError("No input paths were provided.")
    return paths


def load_analyzer(analyzer_folder: Path):
    start = perf_counter()
    log_progress(f"Loading SortingAnalyzer: {analyzer_folder}")
    analyzer = si.load_sorting_analyzer(
        folder=analyzer_folder,
        format="zarr",
        load_extensions=False,
    )
    log_progress("Loaded SortingAnalyzer", start)
    return analyzer


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_csv_auto(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    write_csv(path, rows, fieldnames)


def jsonable_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [jsonable_value(v) for v in value.tolist()]
    return value


def write_json_rows(path: Path, rows: list[dict]) -> None:
    save_json(path, [{key: jsonable_value(value) for key, value in row.items()} for row in rows])


def finite_or_blank(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except Exception:
        return value
    if not np.isfinite(value):
        return ""
    return value


def safe_int(value):
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def get_sampling_frequency(analyzer) -> float:
    for source in (analyzer, analyzer.sorting):
        try:
            return float(source.get_sampling_frequency())
        except Exception:
            pass
    try:
        return float(analyzer.sampling_frequency)
    except Exception:
        return 30000.0


def get_session_duration_s(analyzer, sampling_frequency: float) -> float:
    try:
        return float(analyzer.recording.get_num_frames()) / float(
            analyzer.recording.get_sampling_frequency()
        )
    except Exception:
        pass

    last_spike_s = 0.0
    for unit_id in analyzer.sorting.get_unit_ids():
        for segment_index in range(analyzer.sorting.get_num_segments()):
            spike_train = analyzer.sorting.get_unit_spike_train(
                unit_id=unit_id,
                segment_index=segment_index,
            )
            if len(spike_train) > 0:
                last_spike_s = max(last_spike_s, float(spike_train[-1]) / sampling_frequency)
    return last_spike_s


def compute_cv2(spike_train_samples: np.ndarray) -> float:
    spike_train_samples = np.asarray(spike_train_samples, dtype=float).ravel()
    if spike_train_samples.size < 3:
        return np.nan
    isi = np.diff(spike_train_samples)
    if isi.size < 2:
        return np.nan
    denominator = isi[:-1] + isi[1:]
    valid_mask = denominator > 0
    if not np.any(valid_mask):
        return np.nan
    cv2_values = 2.0 * np.abs(np.diff(isi)[valid_mask]) / denominator[valid_mask]
    if cv2_values.size == 0:
        return np.nan
    return float(np.mean(cv2_values))


def isi_and_correlogram(
    timestamps_s: np.ndarray,
    *,
    max_lag_ms: float = 100.0,
    bin_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps_s = np.asarray(timestamps_s, dtype=float).ravel()
    timestamps_s = np.sort(timestamps_s[np.isfinite(timestamps_s)])
    isi_s = np.diff(timestamps_s).astype(np.float32) if timestamps_s.size >= 2 else np.zeros(0, dtype=np.float32)
    edges_ms = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms, dtype=np.float32)
    counts = np.zeros(edges_ms.size - 1, dtype=np.int64)
    if timestamps_s.size >= 2:
        max_lag_s = max_lag_ms / 1000.0
        for i, t0 in enumerate(timestamps_s):
            j0 = np.searchsorted(timestamps_s, t0 - max_lag_s, side="left")
            j1 = np.searchsorted(timestamps_s, t0 + max_lag_s, side="right")
            diffs_ms = (timestamps_s[j0:j1] - t0) * 1000.0
            diffs_ms = diffs_ms[np.abs(diffs_ms) > 1e-12]
            if diffs_ms.size:
                counts += np.histogram(diffs_ms, bins=edges_ms)[0].astype(np.int64)
    centers_ms = 0.5 * (edges_ms[:-1] + edges_ms[1:])
    return isi_s, centers_ms.astype(np.float32), counts


def trough_to_peak_ms(waveform: np.ndarray, sampling_frequency: float) -> float:
    waveform = np.asarray(waveform, dtype=float).ravel()
    if waveform.size < 2 or sampling_frequency <= 0:
        return np.nan

    trough_index = int(np.argmin(waveform))
    peak_index = int(np.argmax(waveform))
    if abs(waveform[trough_index]) >= abs(waveform[peak_index]):
        if trough_index >= waveform.size - 1:
            return np.nan
        post_peak_index = trough_index + int(np.argmax(waveform[trough_index:]))
        return float((post_peak_index - trough_index) / sampling_frequency * 1000.0)

    if peak_index >= waveform.size - 1:
        return np.nan
    post_trough_index = peak_index + int(np.argmin(waveform[peak_index:]))
    return float((post_trough_index - peak_index) / sampling_frequency * 1000.0)


def waveform_summary(
    waveforms_2d: np.ndarray,
    fallback_waveform: np.ndarray | None,
    sampling_frequency: float,
) -> dict:
    waveforms_2d = np.asarray(waveforms_2d, dtype=float)
    if waveforms_2d.ndim == 2 and waveforms_2d.shape[0] > 0:
        mean_waveform = np.nanmean(waveforms_2d, axis=0).astype(np.float32)
        source = "sampled_waveforms"
    elif fallback_waveform is not None:
        mean_waveform = np.asarray(fallback_waveform, dtype=np.float32).ravel()
        source = "template"
    else:
        mean_waveform = np.zeros(0, dtype=np.float32)
        source = "none"
    if mean_waveform.size == 0:
        return {
            "mean_waveform_uv": mean_waveform,
            "amplitude_ptp_uv": np.nan,
            "mean_abs_waveform_uv": np.nan,
            "peak_to_trough_ms": np.nan,
            "waveform_source": source,
        }
    return {
        "mean_waveform_uv": mean_waveform,
        "amplitude_ptp_uv": float(np.ptp(mean_waveform)),
        "mean_abs_waveform_uv": float(np.mean(np.abs(mean_waveform))),
        "peak_to_trough_ms": trough_to_peak_ms(mean_waveform, sampling_frequency),
        "waveform_source": source,
    }


def save_waveform_plot(
    waveforms_2d: np.ndarray,
    mean_waveform: np.ndarray,
    *,
    sampling_frequency: float,
    out_png: Path,
    title: str,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    mean_waveform = np.asarray(mean_waveform, dtype=float).ravel()
    if mean_waveform.size == 0:
        mean_waveform = np.zeros(1, dtype=float)
    t_ms = np.arange(mean_waveform.size, dtype=float) / float(sampling_frequency) * 1000.0
    waveforms_2d = np.asarray(waveforms_2d, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
    if waveforms_2d.ndim == 2 and waveforms_2d.shape[0] > 0:
        max_lines = min(1000, waveforms_2d.shape[0])
        for row in waveforms_2d[:max_lines]:
            ax.plot(t_ms[: row.size], row, color="0.65", alpha=0.08, linewidth=0.35)
    ax.plot(t_ms, mean_waveform, color="k", linewidth=2.0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("uV")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def best_channel_waveforms(
    waveform_snippets: np.ndarray | None,
    mean_waveform_2d: np.ndarray | None,
    best_channel: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if best_channel is None:
        return np.zeros((0, 0), dtype=np.float32), None
    snippets_2d = np.zeros((0, 0), dtype=np.float32)
    if waveform_snippets is not None and waveform_snippets.ndim == 3 and waveform_snippets.shape[0] > 0:
        snippets_2d = waveform_snippets[:, :, int(best_channel)].astype(np.float32)
    fallback = None
    if mean_waveform_2d is not None and mean_waveform_2d.ndim == 2:
        fallback = mean_waveform_2d[:, int(best_channel)].astype(np.float32)
    return snippets_2d, fallback


def get_best_channel(waveform_2d: np.ndarray) -> int:
    waveform_2d = np.asarray(waveform_2d, dtype=float)
    if waveform_2d.ndim != 2 or waveform_2d.shape[1] == 0:
        return 0
    peak_to_peak_by_channel = np.nanmax(waveform_2d, axis=0) - np.nanmin(waveform_2d, axis=0)
    return int(np.nanargmax(peak_to_peak_by_channel))


def load_quality_metrics_by_unit(analyzer) -> dict[int, dict]:
    if not analyzer.has_extension("quality_metrics"):
        return {}
    try:
        metrics_df = analyzer.get_extension("quality_metrics").get_data()
    except Exception:
        return {}
    try:
        if "unit_id" not in metrics_df.columns:
            metrics_df = metrics_df.reset_index()
            if "unit_id" not in metrics_df.columns and "index" in metrics_df.columns:
                metrics_df = metrics_df.rename(columns={"index": "unit_id"})
        if "unit_id" not in metrics_df.columns:
            return {}
        rows = {}
        for row in metrics_df.to_dict(orient="records"):
            unit_id = safe_int(row.get("unit_id"))
            if unit_id is not None:
                rows[int(unit_id)] = row
        return rows
    except Exception:
        return {}


def load_spike_amplitudes_by_unit(analyzer) -> dict[int, np.ndarray]:
    if not analyzer.has_extension("spike_amplitudes"):
        return {}
    try:
        data = analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit")
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    segment_data = data.get(0, {})
    if not isinstance(segment_data, dict):
        return {}
    amplitudes = {}
    for unit_id, values in segment_data.items():
        parsed_unit_id = safe_int(unit_id)
        if parsed_unit_id is not None:
            amplitudes[int(parsed_unit_id)] = np.asarray(values, dtype=float).ravel()
    return amplitudes


def load_templates(analyzer) -> tuple[np.ndarray | None, dict[int, int]]:
    if not analyzer.has_extension("templates"):
        return None, {}
    try:
        templates = np.asarray(analyzer.get_extension("templates").get_data(), dtype=float)
    except Exception:
        return None, {}
    unit_ids = [int(unit_id) for unit_id in analyzer.sorting.get_unit_ids()]
    if templates.ndim != 3 or templates.shape[0] != len(unit_ids):
        return None, {}
    return templates, {unit_id: index for index, unit_id in enumerate(unit_ids)}


def load_waveform_snippets_by_unit(analyzer) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if not analyzer.has_extension("waveforms") or not analyzer.has_extension("random_spikes"):
        return {}
    try:
        waveforms_ext = analyzer.get_extension("waveforms")
        random_spikes = analyzer.get_extension("random_spikes").get_random_spikes()
    except Exception:
        return {}
    if random_spikes is None or len(random_spikes) == 0:
        return {}

    lookup = {}
    for unit_id in analyzer.sorting.get_unit_ids():
        parsed_unit_id = safe_int(unit_id)
        if parsed_unit_id is None:
            continue
        try:
            unit_index = analyzer.sorting.id_to_index(unit_id)
            waveforms = waveforms_ext.get_waveforms_one_unit(unit_id, force_dense=True)
        except Exception:
            continue
        waveforms = np.asarray(waveforms, dtype=float)
        if waveforms.ndim != 3 or waveforms.shape[0] == 0:
            continue
        unit_spike_mask = random_spikes["unit_index"] == unit_index
        unit_random_spikes = random_spikes[unit_spike_mask]
        if "segment_index" in unit_random_spikes.dtype.names:
            segment_mask = unit_random_spikes["segment_index"] == 0
            unit_random_spikes = unit_random_spikes[segment_mask]
            waveforms = waveforms[segment_mask]
        if len(unit_random_spikes) != waveforms.shape[0]:
            continue
        sample_indices = np.asarray(unit_random_spikes["sample_index"], dtype=float)
        lookup[int(parsed_unit_id)] = (sample_indices, waveforms)
    return lookup


def get_saved_extension_names(analyzer) -> list[str]:
    try:
        return sorted(str(name) for name in analyzer.get_saved_extension_names())
    except Exception:
        try:
            return sorted(str(name) for name in analyzer.get_loaded_extension_names())
        except Exception:
            return []


def normalize_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return np.zeros(1, dtype=float)
    norm = np.linalg.norm(values)
    if norm == 0 or not np.isfinite(norm):
        return values
    return values / norm


def waveform_similarity_vector_from_mean(mean_waveform: np.ndarray | None) -> np.ndarray:
    if mean_waveform is None:
        return np.zeros(1, dtype=float)
    mean_waveform = np.asarray(mean_waveform, dtype=float)
    if mean_waveform.ndim != 2 or mean_waveform.size == 0:
        return np.zeros(1, dtype=float)
    channel_index = int(np.nanargmax(np.nanmax(np.abs(mean_waveform), axis=0)))
    return normalize_vector(mean_waveform[:, channel_index])


def trough_to_next_peak_ms(waveform: np.ndarray | None, sampling_frequency: float) -> float:
    if waveform is None:
        return np.nan
    waveform = np.asarray(waveform, dtype=float).ravel()
    if waveform.size < 2 or sampling_frequency <= 0:
        return np.nan
    trough_index = int(np.argmin(waveform))
    if trough_index >= waveform.size - 1:
        return np.nan
    post_trough = waveform[trough_index + 1 :]
    if post_trough.size == 0:
        return np.nan
    peak_index = trough_index + 1 + int(np.argmax(post_trough))
    return float((peak_index - trough_index) / sampling_frequency * 1000.0)


def autocorrelogram_similarity_vector(
    correlograms: np.ndarray | None,
    unit_index: int | None,
) -> np.ndarray:
    if correlograms is None or unit_index is None:
        return np.zeros(1, dtype=float)
    try:
        if unit_index >= correlograms.shape[0]:
            return np.zeros(1, dtype=float)
        autocorr = np.asarray(correlograms[unit_index, unit_index], dtype=float).copy()
    except Exception:
        return np.zeros(1, dtype=float)
    if autocorr.size == 0:
        return np.zeros(1, dtype=float)
    center_index = autocorr.size // 2
    if 0 <= center_index < autocorr.size:
        autocorr[center_index] = 0.0
    return normalize_vector(autocorr)


def load_correlogram_data(analyzer) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not analyzer.has_extension("correlograms"):
        return None, None
    try:
        correlograms, bins = analyzer.get_extension("correlograms").get_data()
        return np.asarray(correlograms), np.asarray(bins, dtype=float)
    except Exception:
        return None, None


def load_unit_channel_mapping(output_folder: Path) -> dict[int, dict]:
    report_path = output_folder / "unit_channel_mapping_report.json"
    if not report_path.exists():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    mapping = {}
    for row in payload.get("units", []):
        unit_id = safe_int(row.get("unit_id"))
        if unit_id is not None:
            mapping[int(unit_id)] = row
    return mapping


def find_unit_summary_image(output_folder: Path, unit_id: int) -> Path | None:
    summary_folder = output_folder / "unit_summaries_analysis"
    if summary_folder.exists():
        matches = sorted(summary_folder.glob(f"unit_summary_*_{int(unit_id)}.png"))
        if matches:
            return matches[0]

    waveform_folder = output_folder / "unit_waveforms_analysis"
    if waveform_folder.exists():
        matches = sorted(waveform_folder.glob(f"unit_waveform_*_{int(unit_id)}.png"))
        if matches:
            return matches[0]
    return None


def get_sorting_unit_property(sorting, unit_id: int, property_name: str):
    try:
        return sorting.get_unit_property(unit_id, property_name)
    except Exception:
        pass
    try:
        values = sorting.get_property(property_name)
        unit_ids = list(sorting.get_unit_ids())
        if len(values) == len(unit_ids):
            return values[unit_ids.index(unit_id)]
    except Exception:
        pass
    return None


def infer_sorted_unit_channel_metadata(
    analyzer,
    unit_id: int,
    *,
    mapping_row: dict | None,
    template_best_channel: int | None,
) -> dict[str, int | str]:
    if mapping_row is not None:
        shank_id = safe_int(mapping_row.get("shank_id"))
        local_channel = safe_int(mapping_row.get("waveform_local_channel_index"))
        sg_channel = safe_int(mapping_row.get("device_channel_index_property"))
        if shank_id is not None and local_channel is not None:
            return {
                "shank_id": shank_id,
                "local_channel_on_shank": local_channel,
                "sg_channel": sg_channel if sg_channel is not None else local_channel,
                "metadata_source": "unit_channel_mapping_report",
            }

    local_channel = int(template_best_channel) if template_best_channel is not None else 0
    sg_channel = local_channel
    try:
        channel_ids = list(analyzer.channel_ids)
        if local_channel < len(channel_ids):
            sg_channel = int(channel_ids[local_channel])
    except Exception:
        pass

    shank_id = safe_int(get_sorting_unit_property(analyzer.sorting, unit_id, "shank_id"))
    if shank_id is None:
        shank_id = -1

    return {
        "shank_id": shank_id,
        "local_channel_on_shank": local_channel,
        "sg_channel": sg_channel,
        "metadata_source": "template_or_sorting_fallback",
    }


def build_cache_paths(
    analyzer_folder: Path,
    cache_dir_name: str,
    cache_root: Path | None = None,
    cache_root_is_cache_folder: bool = False,
) -> dict[str, Path]:
    if cache_root is None:
        cache_folder = analyzer_folder.parent / cache_dir_name
    elif cache_root_is_cache_folder:
        cache_folder = cache_root
    else:
        cache_folder = cache_root / analyzer_folder.parent.name / cache_dir_name
    return {
        "folder": cache_folder,
        "metadata": cache_folder / "cache_metadata.json",
        "static_csv": cache_folder / "unit_static_stats.csv",
        "minute_csv": cache_folder / "unit_minute_stats.csv",
        "unit_summary_csv": cache_folder / "unit_summary.csv",
        "unit_summary_json": cache_folder / "unit_summary.json",
        "waveforms_npz": cache_folder / "unit_waveforms.npz",
        "similarity_npz": cache_folder / "unit_similarity_vectors.npz",
        "spike_times_npz": cache_folder / "unit_spike_times.npz",
        "correlograms_npz": cache_folder / "unit_correlograms.npz",
        "sorted_unit_output_folder": cache_folder / "sorted_unit_feature_outputs",
    }


def save_correlogram_cache(analyzer, save_path: Path, unit_ids: list[int]) -> None:
    arrays = {"unit_ids": np.asarray(unit_ids, dtype=np.int64)}
    if analyzer.has_extension("correlograms"):
        try:
            correlograms, bins = analyzer.get_extension("correlograms").get_data()
            arrays["correlograms"] = np.asarray(correlograms)
            arrays["correlogram_bins_ms"] = np.asarray(bins, dtype=float)
        except Exception as exc:
            arrays["correlogram_error"] = np.asarray([str(exc)])
    if analyzer.has_extension("isi_histograms"):
        try:
            isi_histograms, isi_bins = analyzer.get_extension("isi_histograms").get_data()
            arrays["isi_histograms"] = np.asarray(isi_histograms)
            arrays["isi_histogram_bins_ms"] = np.asarray(isi_bins, dtype=float)
        except Exception as exc:
            arrays["isi_histogram_error"] = np.asarray([str(exc)])
    np.savez_compressed(save_path, **arrays)


def write_sorted_unit_feature_outputs(
    *,
    root_folder: Path,
    session_name: str,
    unit_id: int,
    channel_metadata: dict,
    spike_train: np.ndarray,
    spike_times_s: np.ndarray,
    amplitudes: np.ndarray | None,
    waveform_samples: np.ndarray | None,
    waveform_snippets_2d: np.ndarray,
    fallback_waveform: np.ndarray | None,
    sampling_frequency: float,
    session_duration_s: float,
) -> dict:
    sg_channel = channel_metadata.get("sg_channel")
    sg_label = "unknown" if sg_channel in ("", None) else str(sg_channel)
    unit_folder = root_folder / f"sgch{sg_label}_unit{int(unit_id)}"
    minute_dir = unit_folder / "minute_npz"
    hourly_dir = unit_folder / "hourly"
    minute_dir.mkdir(parents=True, exist_ok=True)
    hourly_dir.mkdir(parents=True, exist_ok=True)

    spike_train = np.asarray(spike_train, dtype=np.int64).ravel()
    spike_times_s = np.asarray(spike_times_s, dtype=np.float64).ravel()
    amplitudes = np.asarray(amplitudes, dtype=float).ravel() if amplitudes is not None else None
    waveform_samples = np.asarray(waveform_samples, dtype=float).ravel() if waveform_samples is not None else None
    waveform_snippets_2d = np.asarray(waveform_snippets_2d, dtype=np.float32)
    if waveform_snippets_2d.ndim != 2:
        waveform_snippets_2d = np.zeros((0, 0), dtype=np.float32)

    total_minutes = max(1, int(np.ceil(float(session_duration_s) / 60.0)))
    total_hours = max(1, int(np.ceil(float(session_duration_s) / 3600.0)))
    minute_rows: list[dict] = []
    hourly_rows: list[dict] = []

    for minute_index in range(total_minutes):
        start_s = float(minute_index * 60.0)
        end_s = float(min((minute_index + 1) * 60.0, session_duration_s))
        duration_s = max(0.0, end_s - start_s)
        mask = (spike_times_s >= start_s) & (spike_times_s < end_s)
        if minute_index == total_minutes - 1:
            mask = (spike_times_s >= start_s) & (spike_times_s <= end_s)
        minute_spikes = spike_train[mask]
        minute_times = spike_times_s[mask]
        minute_amps = amplitudes[mask] if amplitudes is not None and amplitudes.shape[0] == spike_train.shape[0] else None

        wf_mask = None
        minute_waveforms = np.zeros((0, waveform_snippets_2d.shape[1] if waveform_snippets_2d.ndim == 2 else 0), dtype=np.float32)
        if waveform_samples is not None and waveform_snippets_2d.shape[0] == waveform_samples.shape[0]:
            wf_times = waveform_samples / sampling_frequency if sampling_frequency > 0 else waveform_samples
            wf_mask = (wf_times >= start_s) & (wf_times < end_s)
            if minute_index == total_minutes - 1:
                wf_mask = (wf_times >= start_s) & (wf_times <= end_s)
            minute_waveforms = waveform_snippets_2d[wf_mask]

        summary = waveform_summary(minute_waveforms, fallback_waveform, sampling_frequency)
        cv2 = compute_cv2(minute_spikes)
        npz_path = minute_dir / f"{session_name}_unit{unit_id}_minute_{minute_index:06d}_spikes_waveforms.npz"
        np.savez_compressed(
            npz_path,
            spike_samples=minute_spikes.astype(np.int64),
            spike_samples_cumulative=minute_spikes.astype(np.int64),
            timestamps_sec=minute_times.astype(np.float64),
            timestamps_sec_cumulative=minute_times.astype(np.float64),
            waveforms_uv=minute_waveforms.astype(np.float32),
            mean_waveform_uv=summary["mean_waveform_uv"],
            sampling_rate_hz=np.array([sampling_frequency], dtype=np.float64),
            minute_index=np.array([minute_index], dtype=np.int32),
            minute_start_sec=np.array([start_s], dtype=np.float64),
            minute_end_sec=np.array([end_s], dtype=np.float64),
            minute_duration_sec=np.array([duration_s], dtype=np.float64),
            unit_id=np.array([unit_id], dtype=np.int64),
        )
        minute_rows.append(
            {
                "minute_index": minute_index,
                "time_start_sec": start_s,
                "time_end_sec": end_s,
                "duration_sec": duration_s,
                "unit_id": int(unit_id),
                "sg_ch": sg_channel,
                "n_spikes": int(minute_spikes.size),
                "firing_rate_hz": finite_or_blank(minute_spikes.size / duration_s if duration_s > 0 else np.nan),
                "amplitude_ptp_uv": finite_or_blank(summary["amplitude_ptp_uv"]),
                "mean_abs_waveform_uv": finite_or_blank(summary["mean_abs_waveform_uv"]),
                "amplitude_mean_abs": finite_or_blank(np.nanmean(np.abs(minute_amps)) if minute_amps is not None and minute_amps.size else np.nan),
                "amplitude_median_abs": finite_or_blank(np.nanmedian(np.abs(minute_amps)) if minute_amps is not None and minute_amps.size else np.nan),
                "cv2": finite_or_blank(cv2),
                "peak_to_trough_ms": finite_or_blank(summary["peak_to_trough_ms"]),
                "waveform_source": summary["waveform_source"],
                "mean_waveform_uv": json.dumps([jsonable_value(v) for v in summary["mean_waveform_uv"]]),
                "npz": str(npz_path.resolve()),
            }
        )

    for hour_index in range(total_hours):
        start_s = float(hour_index * 3600.0)
        end_s = float(min((hour_index + 1) * 3600.0, session_duration_s))
        duration_s = max(0.0, end_s - start_s)
        mask = (spike_times_s >= start_s) & (spike_times_s < end_s)
        if hour_index == total_hours - 1:
            mask = (spike_times_s >= start_s) & (spike_times_s <= end_s)
        hour_spikes = spike_train[mask]
        hour_times = spike_times_s[mask]
        hour_amps = amplitudes[mask] if amplitudes is not None and amplitudes.shape[0] == spike_train.shape[0] else None
        hour_waveforms = np.zeros((0, waveform_snippets_2d.shape[1] if waveform_snippets_2d.ndim == 2 else 0), dtype=np.float32)
        if waveform_samples is not None and waveform_snippets_2d.shape[0] == waveform_samples.shape[0]:
            wf_times = waveform_samples / sampling_frequency if sampling_frequency > 0 else waveform_samples
            wf_mask = (wf_times >= start_s) & (wf_times < end_s)
            if hour_index == total_hours - 1:
                wf_mask = (wf_times >= start_s) & (wf_times <= end_s)
            hour_waveforms = waveform_snippets_2d[wf_mask]
        summary = waveform_summary(hour_waveforms, fallback_waveform, sampling_frequency)
        isi_s, corr_lag_ms, corr_counts = isi_and_correlogram(hour_times)
        npz_path = hourly_dir / f"{session_name}_unit{unit_id}_hour_{hour_index:04d}_isi_correlogram.npz"
        np.savez_compressed(
            npz_path,
            spike_samples=hour_spikes.astype(np.int64),
            timestamps_sec=hour_times.astype(np.float64),
            waveforms_uv=hour_waveforms.astype(np.float32),
            mean_waveform_uv=summary["mean_waveform_uv"],
            isi_sec=isi_s,
            correlogram_lag_ms=corr_lag_ms,
            correlogram_counts=corr_counts,
            hour_index=np.array([hour_index], dtype=np.int32),
            hour_start_sec=np.array([start_s], dtype=np.float64),
            hour_end_sec=np.array([end_s], dtype=np.float64),
            hour_duration_sec=np.array([duration_s], dtype=np.float64),
            unit_id=np.array([unit_id], dtype=np.int64),
        )
        fig_path = hourly_dir / f"{session_name}_unit{unit_id}_hour_{hour_index:04d}_sgch{sg_label}_n{hour_spikes.size}_waveforms.png"
        save_waveform_plot(
            hour_waveforms,
            summary["mean_waveform_uv"],
            sampling_frequency=sampling_frequency,
            out_png=fig_path,
            title=f"Unit {unit_id} | SG ch {sg_label} | hour {hour_index} | N={hour_spikes.size}",
        )
        hourly_rows.append(
            {
                "hour_index": hour_index,
                "time_start_sec": start_s,
                "time_end_sec": end_s,
                "duration_sec": duration_s,
                "unit_id": int(unit_id),
                "sg_ch": sg_channel,
                "n_spikes": int(hour_spikes.size),
                "firing_rate_hz": finite_or_blank(hour_spikes.size / duration_s if duration_s > 0 else np.nan),
                "amplitude_ptp_uv": finite_or_blank(summary["amplitude_ptp_uv"]),
                "mean_abs_waveform_uv": finite_or_blank(summary["mean_abs_waveform_uv"]),
                "amplitude_mean_abs": finite_or_blank(np.nanmean(np.abs(hour_amps)) if hour_amps is not None and hour_amps.size else np.nan),
                "amplitude_median_abs": finite_or_blank(np.nanmedian(np.abs(hour_amps)) if hour_amps is not None and hour_amps.size else np.nan),
                "cv2": finite_or_blank(compute_cv2(hour_spikes)),
                "peak_to_trough_ms": finite_or_blank(summary["peak_to_trough_ms"]),
                "isi_mean_ms": finite_or_blank(float(np.mean(isi_s) * 1000.0) if isi_s.size else np.nan),
                "isi_median_ms": finite_or_blank(float(np.median(isi_s) * 1000.0) if isi_s.size else np.nan),
                "waveform_source": summary["waveform_source"],
                "npz": str(npz_path.resolve()),
                "figure": str(fig_path.resolve()),
            }
        )

    minute_summary_csv = unit_folder / f"{session_name}_unit{unit_id}_minute_summary.csv"
    minute_summary_json = unit_folder / f"{session_name}_unit{unit_id}_minute_summary.json"
    hourly_summary_csv = unit_folder / f"{session_name}_unit{unit_id}_hourly_summary.csv"
    hourly_summary_json = unit_folder / f"{session_name}_unit{unit_id}_hourly_summary.json"
    write_csv_auto(minute_summary_csv, minute_rows)
    write_json_rows(minute_summary_json, minute_rows)
    write_csv_auto(hourly_summary_csv, hourly_rows)
    write_json_rows(hourly_summary_json, hourly_rows)

    recording_summary = {
        "source": "Sorting_organize.py",
        "session_name": session_name,
        "unit_id": int(unit_id),
        "sg_ch": sg_channel,
        "seconds": float(session_duration_s),
        "n_spikes": int(spike_train.size),
        "sampling_rate_hz": float(sampling_frequency),
        "minute_summary_csv": str(minute_summary_csv.resolve()),
        "minute_summary_json": str(minute_summary_json.resolve()),
        "hourly_summary_csv": str(hourly_summary_csv.resolve()),
        "hourly_summary_json": str(hourly_summary_json.resolve()),
        "minute_npz_folder": str(minute_dir.resolve()),
        "hourly_folder": str(hourly_dir.resolve()),
        "output_resolution_note": (
            "Sorted-unit outputs arranged with the same feature layout used by Threshold_channel.py: "
            "minute NPZ files, minute CSV/JSON summaries, hourly ISI/correlogram NPZ files, "
            "and hourly waveform figures. These are sorted-unit spikes, not threshold crossings."
        ),
    }
    summary_path = unit_folder / f"{session_name}_unit{unit_id}_recording_summary.json"
    save_json(summary_path, recording_summary)
    return {
        "unit_id": int(unit_id),
        "sg_ch": sg_channel,
        "folder": str(unit_folder.resolve()),
        "recording_summary": str(summary_path.resolve()),
        "minute_summary_csv": str(minute_summary_csv.resolve()),
        "hourly_summary_csv": str(hourly_summary_csv.resolve()),
    }


def build_unit_feature_cache(
    analyzer,
    analyzer_folder: Path,
    *,
    cache_dir_name: str,
    cache_root: Path | None,
    cache_root_is_cache_folder: bool,
    bin_size_seconds: float,
    overwrite: bool,
    save_minute_waveforms: bool,
) -> Path:
    start = perf_counter()
    timing: dict[str, float] = {}
    unit_timing_rows: list[dict] = []
    log_progress("Starting per-minute unit-stat cache build")
    phase_start = perf_counter()
    paths = build_cache_paths(
        analyzer_folder,
        cache_dir_name,
        cache_root=cache_root,
        cache_root_is_cache_folder=cache_root_is_cache_folder,
    )
    output_folder = analyzer_folder.parent
    cache_folder = paths["folder"]
    if cache_folder.exists() and not overwrite:
        raise FileExistsError(
            f"Feature cache already exists: {cache_folder}. "
            "Use --overwrite-cache to rebuild it."
        )
    cache_folder.mkdir(parents=True, exist_ok=True)
    add_timing(timing, "prepare_output_folder", phase_start)

    phase_start = perf_counter()
    unit_ids = [int(unit_id) for unit_id in analyzer.sorting.get_unit_ids()]
    sampling_frequency = get_sampling_frequency(analyzer)
    log_progress(f"Resolved sampling frequency: {sampling_frequency:.3f} Hz", start)
    log_progress("Estimating session duration", start)
    session_duration_s = get_session_duration_s(analyzer, sampling_frequency)
    n_bins = int(np.ceil(session_duration_s / float(bin_size_seconds))) if session_duration_s > 0 else 0
    bin_edges_s = np.arange(n_bins + 1, dtype=float) * float(bin_size_seconds)
    bin_edges_samples = bin_edges_s * sampling_frequency
    add_timing(timing, "load_basic_metadata", phase_start)

    log_progress(f"Units: {len(unit_ids)} | duration: {session_duration_s:.2f}s | bins: {n_bins}", start)
    phase_start = perf_counter()
    log_progress("Loading quality_metrics extension", start)
    quality_metrics = load_quality_metrics_by_unit(analyzer)
    log_progress(f"Loaded quality metrics for {len(quality_metrics)} unit(s)", start)
    add_timing(timing, "load_quality_metrics", phase_start)

    phase_start = perf_counter()
    log_progress("Loading spike_amplitudes extension", start)
    spike_amplitudes_by_unit = load_spike_amplitudes_by_unit(analyzer)
    log_progress(f"Loaded spike amplitudes for {len(spike_amplitudes_by_unit)} unit(s)", start)
    add_timing(timing, "load_spike_amplitudes", phase_start)

    phase_start = perf_counter()
    log_progress("Loading templates extension", start)
    templates, unit_id_to_template_index = load_templates(analyzer)
    log_progress(f"Loaded templates for {len(unit_id_to_template_index)} unit(s)", start)
    add_timing(timing, "load_templates", phase_start)

    phase_start = perf_counter()
    log_progress("Loading waveforms/random_spikes extensions", start)
    waveform_snippets_by_unit = load_waveform_snippets_by_unit(analyzer)
    log_progress(f"Loaded sampled waveforms for {len(waveform_snippets_by_unit)} unit(s)", start)
    add_timing(timing, "load_waveforms_random_spikes", phase_start)

    phase_start = perf_counter()
    log_progress("Loading correlograms extension", start)
    correlograms, correlogram_bins = load_correlogram_data(analyzer)
    log_progress(
        "Loaded correlograms"
        if correlograms is not None
        else "No correlograms available",
        start,
    )
    add_timing(timing, "load_correlograms", phase_start)

    phase_start = perf_counter()
    unit_channel_mapping = load_unit_channel_mapping(output_folder)
    log_progress(f"Loaded channel mapping for {len(unit_channel_mapping)} unit(s)", start)
    add_timing(timing, "load_channel_mapping", phase_start)

    static_rows = []
    minute_rows = []
    unit_summary_rows = []
    unit_summary_records = []
    waveform_arrays = {
        "unit_ids": np.asarray(unit_ids, dtype=np.int64),
        "sampling_frequency": np.asarray([sampling_frequency], dtype=float),
        "bin_size_seconds": np.asarray([float(bin_size_seconds)], dtype=float),
        "bin_edges_s": bin_edges_s.astype(float),
    }
    similarity_arrays = {
        "unit_ids": np.asarray(unit_ids, dtype=np.int64),
        "sampling_frequency": np.asarray([sampling_frequency], dtype=float),
    }
    spike_time_arrays = {
        "unit_ids": np.asarray(unit_ids, dtype=np.int64),
        "sampling_frequency": np.asarray([sampling_frequency], dtype=float),
    }
    waveform_vector_rows = []
    autocorrelogram_vector_rows = []
    sorted_unit_output_records = []

    for unit_position, unit_id in enumerate(unit_ids, start=1):
        unit_start = perf_counter()
        log_progress(f"Processing unit {unit_position}/{len(unit_ids)}: unit_id={unit_id}", start)
        spike_train = np.asarray(
            analyzer.sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0),
            dtype=float,
        )
        spike_times_s = spike_train / sampling_frequency if sampling_frequency > 0 else spike_train
        spike_count = int(spike_train.size)
        spike_time_arrays[f"unit_{unit_id}_spike_samples"] = spike_train.astype(np.int64)
        spike_time_arrays[f"unit_{unit_id}_spike_times_s"] = spike_times_s.astype(np.float64)
        qmetrics = quality_metrics.get(unit_id, {})
        amplitudes = spike_amplitudes_by_unit.get(unit_id)
        if amplitudes is not None and amplitudes.shape[0] != spike_count:
            amplitudes = None

        template_best_channel = None
        template_peak_to_trough = np.nan
        template_peak_to_peak = np.nan
        template = None
        template_index = unit_id_to_template_index.get(unit_id)
        if templates is not None and template_index is not None:
            template = np.asarray(templates[template_index], dtype=float)
            template_best_channel = get_best_channel(template)
            template_waveform = template[:, template_best_channel]
            template_peak_to_trough = trough_to_peak_ms(template_waveform, sampling_frequency)
            template_peak_to_peak = float(np.ptp(template_waveform))
            waveform_arrays[f"unit_{unit_id}_template_waveform"] = template.astype(np.float32)

        waveform_samples = None
        waveform_snippets = None
        sampled_mean = None
        sampled_mean_peak_to_trough = np.nan
        sampled_waveform_count = 0
        waveform_payload = waveform_snippets_by_unit.get(unit_id)
        if waveform_payload is not None:
            waveform_samples, waveform_snippets = waveform_payload
            sampled_waveform_count = int(waveform_snippets.shape[0])
            sampled_mean = np.nanmean(waveform_snippets, axis=0)
            sampled_best_channel = get_best_channel(sampled_mean)
            sampled_mean_peak_to_trough = trough_to_peak_ms(
                sampled_mean[:, sampled_best_channel],
                sampling_frequency,
            )
            waveform_arrays[f"unit_{unit_id}_sampled_mean_waveform"] = sampled_mean.astype(np.float32)
            waveform_arrays[f"unit_{unit_id}_sampled_waveform_samples"] = waveform_samples.astype(np.float64)

        waveform_source = "saved_waveforms" if sampled_mean is not None else "template"
        waveform_vector_source = sampled_mean if sampled_mean is not None else template
        waveform_vector = waveform_similarity_vector_from_mean(waveform_vector_source)
        similarity_arrays[f"unit_{unit_id}_waveform_vector"] = waveform_vector.astype(np.float32)
        waveform_vector_rows.append(waveform_vector)

        try:
            unit_index = list(analyzer.sorting.get_unit_ids()).index(unit_id)
        except Exception:
            unit_index = template_index
        autocorrelogram_vector = autocorrelogram_similarity_vector(correlograms, unit_index)
        similarity_arrays[f"unit_{unit_id}_autocorrelogram_vector"] = autocorrelogram_vector.astype(np.float32)
        autocorrelogram_vector_rows.append(autocorrelogram_vector)

        if sampled_mean is not None:
            strongest_channel_index = int(np.nanargmax(np.nanmax(np.abs(sampled_mean), axis=0)))
            trough_to_peak_duration = trough_to_next_peak_ms(
                sampled_mean[:, strongest_channel_index],
                sampling_frequency,
            )
        elif template is not None and template_best_channel is not None:
            trough_to_peak_duration = trough_to_next_peak_ms(
                template[:, template_best_channel],
                sampling_frequency,
            )
        else:
            trough_to_peak_duration = np.nan

        channel_metadata = infer_sorted_unit_channel_metadata(
            analyzer,
            unit_id,
            mapping_row=unit_channel_mapping.get(unit_id),
            template_best_channel=template_best_channel,
        )
        sorted_unit_best_channel = None
        if sampled_mean is not None:
            sorted_unit_best_channel = int(np.nanargmax(np.nanmax(np.abs(sampled_mean), axis=0)))
        elif template_best_channel is not None:
            sorted_unit_best_channel = int(template_best_channel)
        sorted_unit_waveforms, sorted_unit_fallback = best_channel_waveforms(
            waveform_snippets,
            sampled_mean if sampled_mean is not None else template,
            sorted_unit_best_channel,
        )
        sorted_unit_output_records.append(
            write_sorted_unit_feature_outputs(
                root_folder=paths["sorted_unit_output_folder"],
                session_name=output_folder.name,
                unit_id=unit_id,
                channel_metadata=channel_metadata,
                spike_train=spike_train,
                spike_times_s=spike_times_s,
                amplitudes=amplitudes,
                waveform_samples=waveform_samples,
                waveform_snippets_2d=sorted_unit_waveforms,
                fallback_waveform=sorted_unit_fallback,
                sampling_frequency=sampling_frequency,
                session_duration_s=session_duration_s,
            )
        )
        image_path = find_unit_summary_image(output_folder, unit_id)
        unit_summary_row = {
            "session_name": output_folder.name,
            "analyzer_folder": str(analyzer_folder),
            "output_folder": str(output_folder),
            "unit_id": unit_id,
            "shank_id": channel_metadata["shank_id"],
            "local_channel_on_shank": channel_metadata["local_channel_on_shank"],
            "sg_channel": channel_metadata["sg_channel"],
            "channel_metadata_source": channel_metadata["metadata_source"],
            "amplitude_median": finite_or_blank(qmetrics.get("amplitude_median")),
            "firing_rate": finite_or_blank(qmetrics.get("firing_rate")),
            "isi_violations_ratio": finite_or_blank(qmetrics.get("isi_violations_ratio")),
            "snr": finite_or_blank(qmetrics.get("snr")),
            "num_spikes": finite_or_blank(qmetrics.get("num_spikes") if qmetrics.get("num_spikes") is not None else spike_count),
            "trough_to_peak_duration_ms": finite_or_blank(trough_to_peak_duration),
            "waveform_image_path": str(image_path) if image_path is not None else "",
            "waveform_vector_key": f"unit_{unit_id}_waveform_vector",
            "autocorrelogram_vector_key": f"unit_{unit_id}_autocorrelogram_vector",
            "waveform_vector_source": waveform_source,
            "sampled_waveform_count": sampled_waveform_count,
            "spike_samples_key": f"unit_{unit_id}_spike_samples",
            "spike_times_s_key": f"unit_{unit_id}_spike_times_s",
        }
        unit_summary_rows.append(unit_summary_row)
        unit_summary_records.append(
            {
                **unit_summary_row,
                "waveform_similarity_vector": waveform_vector.tolist(),
                "autocorrelogram_similarity_vector": autocorrelogram_vector.tolist(),
            }
        )

        isi_samples = np.diff(spike_train)
        isi_ms = isi_samples / sampling_frequency * 1000.0 if sampling_frequency > 0 else isi_samples
        static_rows.append(
            {
                "unit_id": unit_id,
                "total_spikes": spike_count,
                "duration_seconds": finite_or_blank(session_duration_s),
                "overall_firing_rate_hz": finite_or_blank(spike_count / session_duration_s if session_duration_s > 0 else np.nan),
                "overall_cv2": finite_or_blank(compute_cv2(spike_train)),
                "overall_mean_isi_ms": finite_or_blank(np.nanmean(isi_ms) if isi_ms.size else np.nan),
                "overall_median_isi_ms": finite_or_blank(np.nanmedian(isi_ms) if isi_ms.size else np.nan),
                "overall_isi_violations_2ms": int(np.count_nonzero(isi_ms < 2.0)) if isi_ms.size else 0,
                "amplitude_mean_abs": finite_or_blank(np.nanmean(np.abs(amplitudes)) if amplitudes is not None and amplitudes.size else np.nan),
                "amplitude_median_abs": finite_or_blank(np.nanmedian(np.abs(amplitudes)) if amplitudes is not None and amplitudes.size else np.nan),
                "quality_amplitude_median": finite_or_blank(qmetrics.get("amplitude_median")),
                "snr": finite_or_blank(qmetrics.get("snr")),
                "quality_firing_rate": finite_or_blank(qmetrics.get("firing_rate")),
                "quality_isi_violations_ratio": finite_or_blank(qmetrics.get("isi_violations_ratio")),
                "quality_num_spikes": finite_or_blank(qmetrics.get("num_spikes")),
                "template_best_channel_index": template_best_channel if template_best_channel is not None else "",
                "template_peak_to_peak": finite_or_blank(template_peak_to_peak),
                "template_peak_to_trough_ms": finite_or_blank(template_peak_to_trough),
                "sampled_waveform_count": sampled_waveform_count,
                "sampled_mean_peak_to_trough_ms": finite_or_blank(sampled_mean_peak_to_trough),
            }
        )

        counts, _ = np.histogram(spike_times_s, bins=bin_edges_s) if n_bins > 0 else ([], [])
        bin_indices = np.searchsorted(bin_edges_s, spike_times_s, side="right") - 1 if n_bins > 0 else np.array([])
        if n_bins > 0:
            bin_indices[spike_times_s == bin_edges_s[-1]] = n_bins - 1

        waveform_bin_indices = None
        if waveform_samples is not None and n_bins > 0:
            waveform_bin_indices = np.searchsorted(bin_edges_samples, waveform_samples, side="right") - 1
            waveform_bin_indices[waveform_samples == bin_edges_samples[-1]] = n_bins - 1

        for bin_index in range(n_bins):
            bin_mask = bin_indices == bin_index
            bin_spike_train = spike_train[bin_mask]
            bin_isi_samples = np.diff(bin_spike_train)
            bin_isi_ms = (
                bin_isi_samples / sampling_frequency * 1000.0
                if sampling_frequency > 0
                else bin_isi_samples
            )
            bin_amplitudes = amplitudes[bin_mask] if amplitudes is not None else None

            waveform_count = 0
            peak_to_trough = template_peak_to_trough
            peak_to_trough_source = "template"
            if waveform_snippets is not None and waveform_bin_indices is not None:
                bin_waveforms = waveform_snippets[waveform_bin_indices == bin_index]
                waveform_count = int(bin_waveforms.shape[0])
                if waveform_count > 0:
                    mean_waveform = np.nanmean(bin_waveforms, axis=0)
                    best_channel = get_best_channel(mean_waveform)
                    peak_to_trough = trough_to_peak_ms(mean_waveform[:, best_channel], sampling_frequency)
                    peak_to_trough_source = "saved_waveforms"
                    if save_minute_waveforms:
                        waveform_arrays[
                            f"unit_{unit_id}_minute_{bin_index:05d}_sampled_mean_waveform"
                        ] = mean_waveform.astype(np.float32)

            minute_rows.append(
                {
                    "unit_id": unit_id,
                    "minute_index": bin_index,
                    "start_sec": finite_or_blank(bin_edges_s[bin_index]),
                    "end_sec": finite_or_blank(bin_edges_s[bin_index + 1]),
                    "spike_count": int(counts[bin_index]),
                    "firing_rate_hz": finite_or_blank(counts[bin_index] / float(bin_size_seconds)),
                    "amplitude_mean_abs": finite_or_blank(np.nanmean(np.abs(bin_amplitudes)) if bin_amplitudes is not None and bin_amplitudes.size else np.nan),
                    "amplitude_median_abs": finite_or_blank(np.nanmedian(np.abs(bin_amplitudes)) if bin_amplitudes is not None and bin_amplitudes.size else np.nan),
                    "mean_isi_ms": finite_or_blank(np.nanmean(bin_isi_ms) if bin_isi_ms.size else np.nan),
                    "median_isi_ms": finite_or_blank(np.nanmedian(bin_isi_ms) if bin_isi_ms.size else np.nan),
                    "isi_violations_2ms": int(np.count_nonzero(bin_isi_ms < 2.0)) if bin_isi_ms.size else 0,
                    "cv2": finite_or_blank(compute_cv2(bin_spike_train)),
                    "waveform_snippet_count": waveform_count,
                    "peak_to_trough_ms": finite_or_blank(peak_to_trough),
                    "peak_to_trough_source": peak_to_trough_source,
                }
            )

        log_progress(
            f"Finished unit {unit_position}/{len(unit_ids)}: unit_id={unit_id}, "
            f"spikes={spike_count}, bins={n_bins}, unit_time={format_elapsed(perf_counter() - unit_start)}",
            start,
        )
        unit_elapsed = perf_counter() - unit_start
        unit_timing_rows.append(
            {
                "unit_position": unit_position,
                "num_units": len(unit_ids),
                "unit_id": int(unit_id),
                "sg_channel": channel_metadata["sg_channel"],
                "spike_count": spike_count,
                "num_bins": n_bins,
                "seconds": float(unit_elapsed),
                "elapsed": format_elapsed(unit_elapsed),
            }
        )
        timing["process_units_total"] = timing.get("process_units_total", 0.0) + float(unit_elapsed)

    static_fields = [
        "unit_id",
        "total_spikes",
        "duration_seconds",
        "overall_firing_rate_hz",
        "overall_cv2",
        "overall_mean_isi_ms",
        "overall_median_isi_ms",
        "overall_isi_violations_2ms",
        "amplitude_mean_abs",
        "amplitude_median_abs",
        "quality_amplitude_median",
        "snr",
        "quality_firing_rate",
        "quality_isi_violations_ratio",
        "quality_num_spikes",
        "template_best_channel_index",
        "template_peak_to_peak",
        "template_peak_to_trough_ms",
        "sampled_waveform_count",
        "sampled_mean_peak_to_trough_ms",
    ]
    minute_fields = [
        "unit_id",
        "minute_index",
        "start_sec",
        "end_sec",
        "spike_count",
        "firing_rate_hz",
        "amplitude_mean_abs",
        "amplitude_median_abs",
        "mean_isi_ms",
        "median_isi_ms",
        "isi_violations_2ms",
        "cv2",
        "waveform_snippet_count",
        "peak_to_trough_ms",
        "peak_to_trough_source",
    ]
    unit_summary_fields = [
        "session_name",
        "analyzer_folder",
        "output_folder",
        "unit_id",
        "shank_id",
        "local_channel_on_shank",
        "sg_channel",
        "channel_metadata_source",
        "amplitude_median",
        "firing_rate",
        "isi_violations_ratio",
        "snr",
        "num_spikes",
        "trough_to_peak_duration_ms",
        "waveform_image_path",
        "waveform_vector_key",
        "autocorrelogram_vector_key",
        "waveform_vector_source",
        "sampled_waveform_count",
        "spike_samples_key",
        "spike_times_s_key",
    ]

    if waveform_vector_rows:
        try:
            similarity_arrays["waveform_vectors"] = np.vstack(waveform_vector_rows).astype(np.float32)
        except Exception:
            pass
    if autocorrelogram_vector_rows:
        try:
            similarity_arrays["autocorrelogram_vectors"] = np.vstack(autocorrelogram_vector_rows).astype(np.float32)
        except Exception:
            pass
    if correlogram_bins is not None:
        similarity_arrays["correlogram_bins_ms"] = np.asarray(correlogram_bins, dtype=float)

    phase_start = perf_counter()
    log_progress(f"Writing static CSV: {paths['static_csv']}", start)
    write_csv(paths["static_csv"], static_rows, static_fields)
    log_progress(f"Writing minute CSV with {len(minute_rows)} row(s): {paths['minute_csv']}", start)
    write_csv(paths["minute_csv"], minute_rows, minute_fields)
    log_progress(f"Writing sorted-unit summary CSV: {paths['unit_summary_csv']}", start)
    write_csv(paths["unit_summary_csv"], unit_summary_rows, unit_summary_fields)
    log_progress(f"Writing sorted-unit summary JSON: {paths['unit_summary_json']}", start)
    save_json(
        paths["unit_summary_json"],
        {
            "analyzer_folder": str(analyzer_folder),
            "output_folder": str(output_folder),
            "units": unit_summary_records,
        },
    )
    log_progress(f"Writing waveform NPZ: {paths['waveforms_npz']}", start)
    np.savez_compressed(paths["waveforms_npz"], **waveform_arrays)
    log_progress(f"Writing similarity-vector NPZ: {paths['similarity_npz']}", start)
    np.savez_compressed(paths["similarity_npz"], **similarity_arrays)
    log_progress(f"Writing spike-time NPZ: {paths['spike_times_npz']}", start)
    np.savez_compressed(paths["spike_times_npz"], **spike_time_arrays)
    log_progress(f"Writing correlogram/ISI NPZ: {paths['correlograms_npz']}", start)
    save_correlogram_cache(analyzer, paths["correlograms_npz"], unit_ids)
    sorted_unit_output_index = paths["sorted_unit_output_folder"] / "sorted_unit_feature_output_index.json"
    log_progress(f"Writing sorted-unit feature output index: {sorted_unit_output_index}", start)
    save_json(
        sorted_unit_output_index,
        {
            "source": "Sorting_organize.py",
            "analyzer_folder": str(analyzer_folder),
            "output_folder": str(output_folder),
            "session_duration_seconds": float(session_duration_s),
            "sampling_frequency": float(sampling_frequency),
            "num_units": int(len(unit_ids)),
            "units": sorted_unit_output_records,
        },
    )
    add_timing(timing, "write_output_files", phase_start)

    elapsed = perf_counter() - start
    log_progress("Writing processing timing reports", start)
    timing_report_paths = write_timing_reports(
        cache_folder=cache_folder,
        timing=timing,
        unit_timing_rows=unit_timing_rows,
        total_elapsed=elapsed,
    )
    metadata = {
        "analyzer_folder": str(analyzer_folder),
        "cache_folder": str(cache_folder),
        "sampling_frequency": float(sampling_frequency),
        "session_duration_seconds": float(session_duration_s),
        "bin_size_seconds": float(bin_size_seconds),
        "num_bins": int(n_bins),
        "num_units": int(len(unit_ids)),
        "unit_ids": unit_ids,
        "saved_extensions": get_saved_extension_names(analyzer),
        "files": {key: str(path) for key, path in paths.items() if key != "folder"},
        "sorted_unit_feature_outputs": str(paths["sorted_unit_output_folder"]),
        "timing_reports": timing_report_paths,
        "notes": [
            "Spike times, firing rate, ISI, CV2, and spike amplitudes are computed from all available spikes when present.",
            "Waveform-derived minute peak-to-trough uses saved waveform snippets when a bin has any; otherwise it falls back to the unit template.",
            "Current analyzer waveforms may be capped by random_spikes max_spikes_per_unit.",
            "sorted_unit_feature_outputs uses the same per-minute/per-hour feature organization as Threshold_channel.py, but contains sorted-unit spikes only.",
        ],
        "elapsed_seconds": float(elapsed),
    }
    log_progress(f"Writing cache metadata: {paths['metadata']}", start)
    save_json(paths["metadata"], metadata)
    log_progress(f"Saved per-minute unit-stat cache to: {cache_folder}", start)
    log_progress(f"Cache build complete in {format_elapsed(elapsed)}")
    return cache_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-unit/per-minute stats caches from SpikeInterface "
            "sorting_analyzer_analysis.zarr folders."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "One or more paths to sorting_analyzer_analysis.zarr, parent output folders, "
            "or roots containing multiple analyzers. Comma-separated values are also accepted. "
            "If omitted, you will be prompted."
        ),
    )
    parser.add_argument(
        "--cache-dir-name",
        default=FEATURE_CACHE_FOLDER_NAME,
        help=f"Cache folder name. Default: {FEATURE_CACHE_FOLDER_NAME}.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help=(
            "Optional writable root for organized outputs. By default, a sibling "
            "folder ending in _org is created next to the input folder. If set, output goes to "
            "<cache-root>/<analyzer-output-folder>/<cache-dir-name>."
        ),
    )
    parser.add_argument(
        "--bin-size-seconds",
        type=float,
        default=60.0,
        help="Bin size for per-minute/per-bin stats.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Rebuild a cache if it already exists.",
    )
    parser.add_argument(
        "--save-minute-waveforms",
        action="store_true",
        help="Also save sampled mean waveform arrays per unit per bin; off by default to keep caches small.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.paths:
        input_paths = parse_input_paths(args.paths)
    else:
        raw_paths = input(
            "Enter one or more paths to sorting_analyzer_analysis.zarr, output folders, "
            "or analyzer roots, separated by commas: "
        ).strip()
        input_paths = parse_input_paths_text(raw_paths)

    analyzer_jobs: list[tuple[Path, Path, bool]] = []
    seen_analyzers: set[Path] = set()
    for input_path in input_paths:
        analyzer_folders = select_analyzer_folders(input_path)
        cache_root_is_cache_folder = len(analyzer_folders) == 1
        for analyzer_folder in analyzer_folders:
            resolved_analyzer = analyzer_folder.resolve()
            if resolved_analyzer in seen_analyzers:
                log_progress(f"Skipping duplicate analyzer folder: {analyzer_folder}")
                continue
            seen_analyzers.add(resolved_analyzer)
            analyzer_jobs.append((input_path, analyzer_folder, cache_root_is_cache_folder))

    if args.cache_root is not None:
        log_progress(f"Output root: {args.cache_root}")

    overall_start = perf_counter()
    for analyzer_number, (input_path, analyzer_folder, default_cache_root_is_cache_folder) in enumerate(
        analyzer_jobs,
        start=1,
    ):
        input_base = input_path.parent if input_path.name == ANALYZER_FOLDER_NAME else input_path
        if args.cache_root is None:
            cache_root = input_base.parent / f"{input_base.name}_org"
            cache_root_is_cache_folder = default_cache_root_is_cache_folder
            log_progress(f"Output root for {input_path}: {cache_root}")
        else:
            cache_root = args.cache_root
            cache_root_is_cache_folder = False
        log_progress(
            f"Analyzer {analyzer_number}/{len(analyzer_jobs)}: {analyzer_folder}",
            overall_start,
        )
        analyzer = load_analyzer(analyzer_folder)
        build_unit_feature_cache(
            analyzer=analyzer,
            analyzer_folder=analyzer_folder,
            cache_dir_name=args.cache_dir_name,
            cache_root=cache_root,
            cache_root_is_cache_folder=cache_root_is_cache_folder,
            bin_size_seconds=args.bin_size_seconds,
            overwrite=args.overwrite_cache,
            save_minute_waveforms=args.save_minute_waveforms,
        )
    log_progress(f"Finished {len(analyzer_jobs)} analyzer cache build(s)", overall_start)


if __name__ == "__main__":
    main()
