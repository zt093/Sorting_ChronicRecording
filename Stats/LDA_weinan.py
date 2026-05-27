from __future__ import annotations

"""
Population LDA analysis for aligned spike-sorting outputs.

This script reads the alignment export created by Alignment_html.py or
Alignment_days.py and builds population feature vectors across aligned good
unit groups. It supports two analysis modes:

single_day_5min
    Use 5-minute bins from one selected day as LDA samples.

multi_day_hourly
    Use 1-minute bins, then average them into day x clock-hour samples.

Each sample is a time bin or hourly aggregate. Each feature column is one
aligned unit-feature pair, such as one unit group's firing rate, amplitude, CV2,
or peak-to-trough width. Feature modes choose which columns enter LDA:
FR_ONLY, FR_AMP, FR_CV2, FR_PEAK_TO_TROUGH, or MULTI_FEATURE.

Firing-rate features are computed from binned spike counts divided by bin
duration. Average amplitude is binned from per-spike amplitudes when the
SpikeInterface `spike_amplitudes` extension is available. CV2 is computed from
spikes within each bin. Peak-to-trough width is computed from sampled individual
waveforms within each bin when the `waveforms` and `random_spikes` extensions
are available; otherwise it falls back to the session template for bins with
spikes. When APPLY_ZSCORE is True, every selected feature column is normalized
separately across samples before LDA, so each aligned unit-feature pair has its
own mean/std scaling. Missing feature values are filled by column mean before
analysis.

For clock-hour labels, the script evaluates both standard stratified
cross-validation and grouped-by-day cross-validation. Grouped-by-day CV trains
on some calendar days and predicts held-out calendar days. The output includes
LDA projections, confusion matrices, permutation-test summaries, metadata CSVs,
feature maps, and a JSON summary.

Multi-day analysis is supported as long as the sessions were aligned together
in the same export summary before running this script. For Alignment_days.py
exports, LDA prefers flattened `source_members` when available so it can use
the real underlying analyzer unit ids instead of synthetic day-level ids.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.decomposition import PCA


# =============================================================================
# Module 1: User Configuration
# =============================================================================

DATA_PATH = None  # Leave as None to always enter the path in the terminal.
LDA_MODE = "multi_day_hourly"  # "single_day_5min" or "multi_day_hourly"
SINGLE_DAY_DATE = None  # Optional "YYYY-MM-DD" date used by single_day_5min mode.
SINGLE_DAY_5MIN_BIN_SIZE_SECONDS = 300.0
MULTI_DAY_HOURLY_BIN_SIZE_SECONDS = 60.0
LABEL_TYPE = "clock_hour_of_day" 
MIN_FIRING_RATE_HZ = 0.05
APPLY_ZSCORE = True
APPLY_SMOOTHING = False
SMOOTHING_SIGMA_BINS = 1.0
HOURLY_WAVEFORM_WEIGHTING = "minute"  # "minute" or "crossing_count"

MIN_SESSIONS_PER_UNIT = 48
MIN_BINS_PER_LABEL = 2
CV_N_SPLITS = 5
RANDOM_SEED = 42
FEATURE_MODES = (
    "FR_ONLY",
    "FR_AMP",
    "FR_CV2",
    "FR_PEAK_TO_TROUGH",
    "FR_WAVEFORM",
    "WAVEFORM_ONLY",
    "MULTI_FEATURE",
)
MIN_MINUTES_PER_HOUR = 1
N_PERMUTATIONS = 200

ANALYZER_FOLDER_NAME = "sorting_analyzer_analysis.zarr"
OUTPUT_BASE_DIR = Path(r"S:\LDA")
ALIGNMENT_DAYS_SUMMARY_PREFIX = "alignment_days_summary"
UNIT_FEATURE_TYPES = (
    "firing_rate_hz",
    "average_amplitude_uv",
    "cv2",
    "peak_to_trough_ms",
)
CSV_METADATA_COLUMNS = {
    "final_sample_id",
    "final_sample_key",
    "session_id",
    "session_key",
    "session_name",
    "session_name_normalized",
    "session_index",
    "session_start_datetime",
    "minute_bin_index",
    "minute_start_sec",
    "minute_end_sec",
    "minute_center_s",
    "session_duration_s",
    "minute_start_datetime",
    "minute_end_datetime",
    "clock_hour_of_day",
    "clock_minute_of_hour",
    "calendar_day",
    "rec_file",
    "threshold_run_root",
    "threshold_run_name",
}


def safe_slug(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "value"


CIRCULAR_HOUR_CMAP = plt.get_cmap("twilight_shifted", 24)
CIRCULAR_HOUR_BOUNDARIES = np.arange(-0.5, 24.5, 1.0)
CIRCULAR_HOUR_NORM = BoundaryNorm(CIRCULAR_HOUR_BOUNDARIES, CIRCULAR_HOUR_CMAP.N)
CIRCULAR_HOUR_TICKS = list(range(24))
CIRCULAR_HOUR_LABEL = "Hour"


@dataclass
class Config:
    data_path: Path | None = DATA_PATH
    lda_mode: str = LDA_MODE
    single_day_date: str | None = SINGLE_DAY_DATE
    bin_size_seconds: float = MULTI_DAY_HOURLY_BIN_SIZE_SECONDS
    label_type: str = LABEL_TYPE
    min_firing_rate_hz: float = MIN_FIRING_RATE_HZ
    apply_zscore: bool = APPLY_ZSCORE
    apply_smoothing: bool = APPLY_SMOOTHING
    smoothing_sigma_bins: float = SMOOTHING_SIGMA_BINS
    hourly_waveform_weighting: str = HOURLY_WAVEFORM_WEIGHTING
    min_sessions_per_unit: int = MIN_SESSIONS_PER_UNIT
    min_bins_per_label: int = MIN_BINS_PER_LABEL
    cv_n_splits: int = CV_N_SPLITS
    random_seed: int = RANDOM_SEED
    feature_modes: tuple[str, ...] = FEATURE_MODES
    min_minutes_per_hour: int = MIN_MINUTES_PER_HOUR
    n_permutations: int = N_PERMUTATIONS
    analyzer_folder_name: str = ANALYZER_FOLDER_NAME
    output_base_dir: Path = OUTPUT_BASE_DIR


# =============================================================================
# Module 2: Shared Utilities and Unit Feature Helpers
# =============================================================================


# -----------------------------------------------------------------------------
# Basic Parsing and Logging
# -----------------------------------------------------------------------------


def safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not np.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(value)
    except Exception:
        return None


def log_status(message: str) -> None:
    print(f"[LDA] {message}", flush=True)


# -----------------------------------------------------------------------------
# LDA Mode Configuration
# -----------------------------------------------------------------------------


def normalize_lda_mode(lda_mode: str) -> str:
    normalized_mode = str(lda_mode or "").strip().lower()
    valid_modes = {"single_day_5min", "multi_day_hourly"}
    if normalized_mode not in valid_modes:
        raise ValueError(
            "Unsupported LDA_MODE. Use one of: single_day_5min, multi_day_hourly."
        )
    return normalized_mode


def apply_lda_mode_defaults(config: Config) -> Config:
    config.lda_mode = normalize_lda_mode(config.lda_mode)
    if config.lda_mode == "single_day_5min":
        config.bin_size_seconds = float(SINGLE_DAY_5MIN_BIN_SIZE_SECONDS)
    else:
        config.bin_size_seconds = float(MULTI_DAY_HOURLY_BIN_SIZE_SECONDS)
    return config


# -----------------------------------------------------------------------------
# Unit Feature Extraction
# -----------------------------------------------------------------------------


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


def get_best_channel(template: np.ndarray) -> int:
    template = np.asarray(template, dtype=float)
    if template.ndim != 2 or template.shape[1] == 0:
        return 0
    return int(np.argmax(np.ptp(template, axis=0)))


def build_metrics_lookup(analyzer) -> dict[int, dict[str, float | int | None]]:
    if not analyzer.has_extension("quality_metrics"):
        return {}

    metrics_df = analyzer.get_extension("quality_metrics").get_data()
    if "unit_id" not in metrics_df.columns:
        metrics_df = metrics_df.reset_index()
        if "unit_id" not in metrics_df.columns and "index" in metrics_df.columns:
            metrics_df = metrics_df.rename(columns={"index": "unit_id"})
    if "unit_id" not in metrics_df.columns:
        return {}

    desired_metrics = [
        "amplitude_median",
        "firing_rate",
        "isi_violations_ratio",
        "snr",
        "num_spikes",
    ]
    lookup: dict[int, dict[str, float | int | None]] = {}
    for row in metrics_df.itertuples(index=False):
        unit_id = safe_int(getattr(row, "unit_id", None))
        if unit_id is None:
            continue
        lookup[unit_id] = {
            metric_name: getattr(row, metric_name, None)
            for metric_name in desired_metrics
        }
    return lookup


def compute_session_unit_static_features(analyzer) -> dict[int, dict[str, float]]:
    unit_ids = [int(unit_id) for unit_id in analyzer.sorting.get_unit_ids()]
    if not unit_ids:
        return {}

    sampling_frequency = float(analyzer.sorting.get_sampling_frequency())
    metrics_lookup = build_metrics_lookup(analyzer)
    templates = None
    unit_id_to_template_index: dict[int, int] = {}
    if analyzer.has_extension("templates"):
        try:
            templates = np.asarray(analyzer.get_extension("templates").get_data())
            if templates.ndim == 3 and templates.shape[0] == len(unit_ids):
                unit_id_to_template_index = {
                    int(unit_id): index for index, unit_id in enumerate(unit_ids)
                }
            else:
                templates = None
        except Exception:
            templates = None

    feature_lookup: dict[int, dict[str, float]] = {}
    for unit_id in unit_ids:
        spike_train = analyzer.sorting.get_unit_spike_train(
            unit_id=unit_id,
            segment_index=0,
        )
        cv2_value = compute_cv2(spike_train)

        amplitude_value = np.nan
        peak_to_trough_value = np.nan
        template_index = unit_id_to_template_index.get(unit_id)
        if templates is not None and template_index is not None:
            template = np.asarray(templates[template_index], dtype=float)
            best_channel = get_best_channel(template)
            waveform = template[:, best_channel]
            amplitude_value = float(np.ptp(waveform))
            peak_to_trough_value = trough_to_peak_ms(waveform, sampling_frequency)

        amplitude_metric = safe_float(metrics_lookup.get(unit_id, {}).get("amplitude_median"))
        if amplitude_metric is not None:
            amplitude_value = abs(float(amplitude_metric))

        feature_lookup[unit_id] = {
            "average_amplitude_uv": float(amplitude_value),
            "cv2": float(cv2_value),
            "peak_to_trough_ms": float(peak_to_trough_value),
        }

    return feature_lookup


def get_session_spike_amplitudes_by_unit(analyzer) -> dict[int, np.ndarray]:
    if not analyzer.has_extension("spike_amplitudes"):
        return {}

    try:
        amplitudes_by_segment = analyzer.get_extension("spike_amplitudes").get_data(
            outputs="by_unit"
        )
    except Exception:
        return {}

    if not isinstance(amplitudes_by_segment, dict):
        return {}

    segment_amplitudes = amplitudes_by_segment.get(0, {})
    if not isinstance(segment_amplitudes, dict):
        return {}

    lookup: dict[int, np.ndarray] = {}
    for unit_id, amplitudes in segment_amplitudes.items():
        parsed_unit_id = safe_int(unit_id)
        if parsed_unit_id is None:
            continue
        lookup[int(parsed_unit_id)] = np.asarray(amplitudes, dtype=float).ravel()
    return lookup


def binned_mean_abs_amplitude(
    spike_train_samples: np.ndarray,
    spike_amplitudes: np.ndarray | None,
    bin_indices: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    values = np.full(n_bins, np.nan, dtype=float)
    if spike_amplitudes is None:
        return values

    spike_amplitudes = np.asarray(spike_amplitudes, dtype=float).ravel()
    if spike_amplitudes.shape[0] != spike_train_samples.shape[0]:
        return values

    valid_mask = (
        (bin_indices >= 0)
        & (bin_indices < n_bins)
        & np.isfinite(spike_amplitudes)
    )
    if not np.any(valid_mask):
        return values

    sums = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)
    np.add.at(sums, bin_indices[valid_mask], np.abs(spike_amplitudes[valid_mask]))
    np.add.at(counts, bin_indices[valid_mask], 1.0)
    nonzero_mask = counts > 0
    values[nonzero_mask] = sums[nonzero_mask] / counts[nonzero_mask]
    return values


def binned_cv2(
    spike_train_samples: np.ndarray,
    bin_indices: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    values = np.full(n_bins, np.nan, dtype=float)
    for bin_index in range(n_bins):
        bin_spike_train = spike_train_samples[bin_indices == bin_index]
        if bin_spike_train.size >= 3:
            values[bin_index] = compute_cv2(bin_spike_train)
    return values


def get_session_waveform_snippets_by_unit(
    analyzer,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if not analyzer.has_extension("waveforms") or not analyzer.has_extension("random_spikes"):
        return {}

    try:
        waveforms_ext = analyzer.get_extension("waveforms")
        random_spikes = analyzer.get_extension("random_spikes").get_random_spikes()
    except Exception:
        return {}

    if random_spikes is None or len(random_spikes) == 0:
        return {}

    lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for unit_id in analyzer.sorting.get_unit_ids():
        parsed_unit_id = safe_int(unit_id)
        if parsed_unit_id is None:
            continue
        try:
            unit_index = analyzer.sorting.id_to_index(unit_id)
            waveforms = waveforms_ext.get_waveforms_one_unit(unit_id, force_dense=True)
        except Exception:
            continue
        if waveforms is None or getattr(waveforms, "size", 0) == 0:
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
        lookup[int(parsed_unit_id)] = (sample_indices, np.asarray(waveforms, dtype=float))
    return lookup


def binned_waveform_peak_to_trough(
    waveform_spike_samples: np.ndarray | None,
    waveforms: np.ndarray | None,
    bin_edges_samples: np.ndarray,
    n_bins: int,
    sampling_frequency: float,
) -> np.ndarray:
    values = np.full(n_bins, np.nan, dtype=float)
    if waveform_spike_samples is None or waveforms is None:
        return values

    waveform_spike_samples = np.asarray(waveform_spike_samples, dtype=float).ravel()
    waveforms = np.asarray(waveforms, dtype=float)
    if waveforms.ndim != 3 or waveforms.shape[0] != waveform_spike_samples.shape[0]:
        return values

    waveform_bin_indices = np.searchsorted(
        bin_edges_samples,
        waveform_spike_samples,
        side="right",
    ) - 1
    waveform_bin_indices[waveform_spike_samples == bin_edges_samples[-1]] = n_bins - 1

    for bin_index in range(n_bins):
        bin_waveforms = waveforms[waveform_bin_indices == bin_index]
        if bin_waveforms.shape[0] == 0:
            continue
        mean_waveform = np.nanmean(bin_waveforms, axis=0)
        if mean_waveform.ndim != 2 or not np.any(np.isfinite(mean_waveform)):
            continue
        best_channel = get_best_channel(mean_waveform)
        values[bin_index] = trough_to_peak_ms(mean_waveform[:, best_channel], sampling_frequency)
    return values


def binned_template_peak_to_trough(
    peak_to_trough_ms: float | None,
    counts: np.ndarray,
) -> np.ndarray:
    values = np.full(counts.shape[0], np.nan, dtype=float)
    parsed_value = safe_float(peak_to_trough_ms)
    if parsed_value is None:
        return values
    values[counts > 0] = float(parsed_value)
    return values


def build_feature_table(selected_units: pd.DataFrame) -> pd.DataFrame:
    group_table = (
        selected_units[["final_group_key", "final_unit_id", "shank_id", "local_channel_on_shank"]]
        .drop_duplicates()
        .sort_values(["final_unit_id", "final_group_key"], na_position="last")
        .reset_index(drop=True)
    )

    rows: list[dict] = []
    for group_row in group_table.itertuples(index=False):
        group_key = str(group_row.final_group_key)
        for feature_type in UNIT_FEATURE_TYPES:
            rows.append(
                {
                    "feature_key": f"{group_key}__{feature_type}",
                    "final_group_key": group_key,
                    "final_unit_id": safe_int(group_row.final_unit_id),
                    "shank_id": safe_int(group_row.shank_id),
                    "local_channel_on_shank": safe_int(group_row.local_channel_on_shank),
                    "feature_type": feature_type,
                }
            )

    feature_table = pd.DataFrame(rows)
    if feature_table.empty:
        raise RuntimeError("No feature columns could be defined from the selected units.")
    feature_table["feature_column"] = [
        f"feature_{feature_index:04d}"
        for feature_index in range(1, len(feature_table) + 1)
    ]
    return feature_table


# =============================================================================
# Module 3: Input Paths and Alignment Export Loading
# =============================================================================


# -----------------------------------------------------------------------------
# Session and Path Resolution
# -----------------------------------------------------------------------------


def normalize_session_name(session_name: str) -> str:
    return re.sub(r"_sh\d+$", "", str(session_name).strip())


def extract_session_datetime_details(session_name: str, output_folder: str | None = None) -> dict | None:
    text_candidates = [
        ("session_name", str(session_name or "").strip()),
        ("output_folder", str(output_folder or "").strip()),
    ]
    patterns = [
        {
            "name": "yyyymmdd_hhmmss",
            "regex": r"(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})[_-]?(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})",
            "specificity_rank": 3,
            "granularity": "second",
        },
        {
            "name": "yymmdd_hh",
            "regex": r"(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})[_-]?(?P<hour>\d{2})",
            "specificity_rank": 2,
            "granularity": "hour",
        },
        {
            "name": "yyyymmdd",
            "regex": r"(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})",
            "specificity_rank": 1,
            "granularity": "day",
        },
        {
            "name": "yymmdd",
            "regex": r"(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})",
            "specificity_rank": 1,
            "granularity": "day",
        },
    ]

    best_match: dict | None = None
    best_score: tuple[int, int, int] | None = None
    for source_priority, (source_field, source_text) in enumerate(text_candidates):
        if not source_text:
            continue
        for pattern_index, pattern in enumerate(patterns):
            match = re.search(pattern["regex"], source_text)
            if match is None:
                continue
            group = match.groupdict()
            year = int(group["year"])
            if year < 100:
                year += 2000
            month = int(group["month"])
            day = int(group["day"])
            hour = int(group.get("hour") or 0)
            minute = int(group.get("minute") or 0)
            second = int(group.get("second") or 0)
            try:
                parsed_datetime = datetime(year, month, day, hour, minute, second)
            except ValueError:
                continue

            candidate = {
                "session_start_datetime": parsed_datetime,
                "source_field": source_field,
                "source_text": source_text,
                "matched_text": match.group(0),
                "pattern_name": pattern["name"],
                "granularity": pattern["granularity"],
            }
            candidate_score = (
                int(pattern["specificity_rank"]),
                -int(source_priority),
                -int(pattern_index),
            )
            if best_score is None or candidate_score > best_score:
                best_match = candidate
                best_score = candidate_score

    return best_match


def find_day_sorting_root_from_output_folder(output_folder: Path) -> Path | None:
    current = Path(output_folder)
    for candidate in [current, *current.parents]:
        if re.fullmatch(r"\d{6}_Sorting", candidate.name):
            return candidate
    return None


def find_sg_exports(root: Path) -> list[Path]:
    return sorted(root.rglob("export_summary_sg_*.json"))


def find_unique_sg_export(root: Path) -> Path | None:
    sg_matches = find_sg_exports(root)
    if len(sg_matches) == 1:
        return sg_matches[0]
    if len(sg_matches) > 1:
        raise RuntimeError(
            "Found multiple export_summary_sg_*.json files. "
            "Please point DATA_PATH to the exact file you want to analyze."
        )
    return None


def resolve_export_summary_path(data_path: Path) -> Path:
    resolved = Path(data_path)
    if resolved.is_file():
        return resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Data path does not exist: {resolved}")

    candidate = resolved / "units_alignment_summary" / "export_summary.json"
    if candidate.exists():
        return candidate

    if resolved.is_dir() and resolved.name.lower().startswith(ALIGNMENT_DAYS_SUMMARY_PREFIX):
        sg_matches = find_sg_exports(resolved)
        if sg_matches:
            return resolved

    day_candidate = resolved / "alignment_days_summary"
    if day_candidate.exists():
        sg_matches = find_sg_exports(day_candidate)
        if sg_matches:
            return day_candidate

    direct_day_summary_matches = sorted(
        child
        for child in resolved.glob(f"{ALIGNMENT_DAYS_SUMMARY_PREFIX}*")
        if child.is_dir()
    )
    if len(direct_day_summary_matches) == 1:
        sg_matches = find_sg_exports(direct_day_summary_matches[0])
        if sg_matches:
            return direct_day_summary_matches[0]
    if len(direct_day_summary_matches) > 1:
        raise RuntimeError(
            "Found multiple alignment_days_summary* folders. "
            "Please point DATA_PATH to the exact summary folder or export_summary_sg_*.json file."
        )

    direct_match = find_unique_sg_export(resolved)
    if direct_match is not None:
        return direct_match

    matches = sorted(resolved.rglob("export_summary.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            "Could not find export_summary.json or export_summary_sg_*.json. "
            "Point DATA_PATH to the export file or batch root."
        )
    raise RuntimeError(
        "Found multiple export_summary.json files. Please point DATA_PATH to the exact file."
    )


def prompt_for_data_path(default_path: Path | None) -> Path:
    prompt = (
        "\nEnter the alignment export path, Alignment_days summary folder, precomputed population CSV, "
        "or batch root for LDA "
        "(press Enter to use the configured DATA_PATH): "
    )
    raw_value = input(prompt).strip().strip('"').strip("'")
    if raw_value:
        return Path(raw_value)
    if default_path is not None:
        return Path(default_path)
    raise ValueError("A data path is required.")


def prompt_for_path_remap() -> tuple[Path, Path] | None:
    response = input(
        "\nA SortingAnalyzer path from the alignment export was not found.\n"
        "Do you want to remap an old root folder to a new root folder? [y/N]: "
    ).strip().lower()
    if response not in {"y", "yes"}:
        return None

    old_root_text = input("Enter the old root prefix exactly as stored in the export: ").strip().strip('"').strip("'")
    new_root_text = input("Enter the new root prefix on this machine: ").strip().strip('"').strip("'")
    if not old_root_text or not new_root_text:
        raise ValueError("Both old and new root prefixes are required for path remapping.")

    return Path(old_root_text), Path(new_root_text)


def remap_path_prefix(original_path: Path, old_root: Path, new_root: Path) -> Path:
    original_parts = original_path.parts
    old_parts = old_root.parts
    if len(original_parts) < len(old_parts):
        return original_path
    if tuple(part.lower() for part in original_parts[: len(old_parts)]) != tuple(
        part.lower() for part in old_parts
    ):
        return original_path
    return Path(new_root, *original_parts[len(old_parts) :])


def resolve_analyzer_folder_path(
    output_folder: Path,
    config: Config,
    path_remap: tuple[Path, Path] | None,
) -> tuple[Path, tuple[Path, Path] | None]:
    analyzer_folder = output_folder / config.analyzer_folder_name
    if analyzer_folder.exists():
        return analyzer_folder, path_remap

    updated_remap = path_remap
    remapped_output_folder = output_folder
    if updated_remap is not None:
        remapped_output_folder = remap_path_prefix(output_folder, updated_remap[0], updated_remap[1])
        remapped_analyzer_folder = remapped_output_folder / config.analyzer_folder_name
        if remapped_analyzer_folder.exists():
            log_status(
                f"Resolved moved analyzer folder: {output_folder} -> {remapped_output_folder}"
            )
            return remapped_analyzer_folder, updated_remap

    updated_remap = prompt_for_path_remap()
    if updated_remap is not None:
        remapped_output_folder = remap_path_prefix(output_folder, updated_remap[0], updated_remap[1])
        remapped_analyzer_folder = remapped_output_folder / config.analyzer_folder_name
        if remapped_analyzer_folder.exists():
            log_status(
                f"Resolved moved analyzer folder: {output_folder} -> {remapped_output_folder}"
            )
            return remapped_analyzer_folder, updated_remap

    raise FileNotFoundError(
        f"SortingAnalyzer folder not found: {analyzer_folder}"
    )


# -----------------------------------------------------------------------------
# Alignment Export Loading and Member Reconstruction
# -----------------------------------------------------------------------------


def load_export_summary(export_summary_path: Path) -> dict:
    if export_summary_path.is_dir() and export_summary_path.name.lower().startswith(ALIGNMENT_DAYS_SUMMARY_PREFIX):
        top_level_export_summary = export_summary_path / "export_summary.json"
        if top_level_export_summary.exists():
            top_level_payload = json.loads(top_level_export_summary.read_text(encoding="utf-8"))
            top_level_group_rows = top_level_payload.get("cross_session_alignment_groups", [])
            has_top_level_source_members = any(
                isinstance(group_row.get("source_members", []), list) and len(group_row.get("source_members", [])) > 0
                for group_row in top_level_group_rows
            )
            if has_top_level_source_members:
                if "member_mode" not in top_level_payload:
                    top_level_payload["member_mode"] = infer_export_member_mode(top_level_payload)
                log_status(
                    "Using top-level Alignment_days export_summary.json because it contains "
                    "flattened source_members with underlying analyzer unit_ids."
                )
                return top_level_payload

        sg_export_paths = find_sg_exports(export_summary_path)
        if not sg_export_paths:
            raise FileNotFoundError(
                f"Could not find export_summary_sg_*.json under Alignment_days summary folder: {export_summary_path}"
            )

        merged_group_rows: list[dict] = []
        reconstructed_group_count = 0
        original_source_group_count = 0
        for sg_export_path in sg_export_paths:
            payload = json.loads(sg_export_path.read_text(encoding="utf-8"))
            page_scope = payload.get("page_scope") or {}
            shank_id = safe_int(page_scope.get("shank_id"))
            sg_channel = safe_int(page_scope.get("sg_channel"))
            for group_row in payload.get("cross_session_alignment_groups", []):
                merged_row = dict(group_row)
                original_source_members = merged_row.get("source_members", [])
                if isinstance(original_source_members, list) and original_source_members:
                    original_source_group_count += 1
                else:
                    reconstructed_source_members = reconstruct_source_members_from_daily_exports(
                        group_row=merged_row,
                        shank_id=shank_id,
                        sg_channel=sg_channel,
                    )
                    if reconstructed_source_members:
                        reconstructed_group_count += 1
                    merged_row["source_members"] = reconstructed_source_members
                merged_row["final_unit_id"] = None
                merged_group_rows.append(merged_row)

        for merged_index, group_row in enumerate(merged_group_rows, start=1):
            if safe_int(group_row.get("final_unit_id")) is None:
                group_row["final_unit_id"] = merged_index

        if merged_group_rows and (original_source_group_count + reconstructed_group_count) == len(merged_group_rows):
            member_mode = "reconstructed_source_members" if reconstructed_group_count > 0 else "full_source_members"
        elif original_source_group_count > 0 or reconstructed_group_count > 0:
            member_mode = "mixed_members"
        else:
            member_mode = "day_only_members"

        return {
            "output_root": str(export_summary_path),
            "source_export_summary_paths": [str(path) for path in sg_export_paths],
            "member_mode": member_mode,
            "cross_session_alignment_groups": merged_group_rows,
        }

    payload = json.loads(export_summary_path.read_text(encoding="utf-8"))
    if "member_mode" not in payload:
        payload["member_mode"] = infer_export_member_mode(payload)
    return payload


def reconstruct_source_members_from_daily_exports(
    group_row: dict,
    shank_id: int | None,
    sg_channel: int | None,
) -> list[dict]:
    if shank_id is None or sg_channel is None:
        return []

    reconstructed_members: list[dict] = []
    daily_export_cache: dict[Path, dict] = {}

    for member in group_row.get("members", []):
        output_folder_text = str(member.get("output_folder", "") or "").strip()
        synthetic_unit_id = safe_int(member.get("unit_id"))
        if not output_folder_text or synthetic_unit_id is None:
            continue

        day_root = find_day_sorting_root_from_output_folder(Path(output_folder_text))
        if day_root is None:
            continue

        day_export_summary_path = (
            day_root
            / f"sh{int(shank_id)}"
            / "units_alignment_summary"
            / f"export_summary_sg_{int(sg_channel):03d}.json"
        )
        if not day_export_summary_path.exists():
            continue

        day_payload = daily_export_cache.get(day_export_summary_path)
        if day_payload is None:
            day_payload = json.loads(day_export_summary_path.read_text(encoding="utf-8"))
            daily_export_cache[day_export_summary_path] = day_payload

        matched_group = next(
            (
                row
                for row in day_payload.get("cross_session_alignment_groups", [])
                if safe_int(row.get("final_unit_id")) == synthetic_unit_id
            ),
            None,
        )
        if matched_group is None:
            continue

        for source_member in matched_group.get("members", []):
            payload = dict(source_member)
            payload["session_name"] = str(payload.get("session_name", "") or "")
            payload["session_index"] = safe_int(payload.get("session_index"))
            payload["unit_id"] = int(payload.get("unit_id"))
            payload["merge_group"] = str(payload.get("merge_group", "") or "")
            payload["align_group"] = str(payload.get("align_group", "") or "")
            payload["output_folder"] = str(payload.get("output_folder", "") or "")
            reconstructed_members.append(payload)

    return reconstructed_members


def iter_group_members(group_row: dict) -> list[dict]:
    source_members = group_row.get("source_members", [])
    if isinstance(source_members, list) and source_members:
        return source_members
    return group_row.get("members", [])


def infer_export_member_mode(export_payload: dict) -> str:
    group_rows = export_payload.get("cross_session_alignment_groups", [])
    if not group_rows:
        return "unknown"

    has_source_members = any(
        isinstance(group_row.get("source_members", []), list) and len(group_row.get("source_members", [])) > 0
        for group_row in group_rows
    )
    if has_source_members:
        return "full_source_members"

    has_day_members = any(
        isinstance(group_row.get("day_members", []), list) and len(group_row.get("day_members", [])) > 0
        for group_row in group_rows
    )
    if has_day_members:
        return "day_only_members"

    member_session_names = [
        str(member.get("session_name", "") or "").strip()
        for group_row in group_rows
        for member in group_row.get("members", [])
        if str(member.get("session_name", "") or "").strip()
    ]
    if not member_session_names:
        return "unknown"

    # Old Alignment_days exports usually use day codes like 260224 as session names.
    if all(re.fullmatch(r"\d{6}", session_name) for session_name in member_session_names):
        return "day_only_members"

    return "full_source_members"


def infer_group_selection_mode(
    export_payload: dict,
    group_rows: list[dict],
    config: Config,
) -> tuple[str, int]:
    member_mode = str(export_payload.get("member_mode", "unknown") or "unknown")
    exported_session_names = sorted(
        {
            str(member.get("session_name", "") or "").strip()
            for group_row in group_rows
            for member in group_row.get("members", [])
            if str(member.get("session_name", "") or "").strip()
        }
    )
    n_exported_sessions = len(exported_session_names)

    if member_mode in {"full_source_members", "reconstructed_source_members"}:
        return "unique_underlying_sessions", config.min_sessions_per_unit

    if member_mode in {"day_only_members", "mixed_members"} and n_exported_sessions > 0:
        effective_min_count = min(config.min_sessions_per_unit, n_exported_sessions)
        return "unique_export_sessions", effective_min_count

    return "member_rows", config.min_sessions_per_unit


# =============================================================================
# Module 4: Labels, Time Metadata, and Matrix Preprocessing
# =============================================================================


# -----------------------------------------------------------------------------
# Label and Session-Time Helpers
# -----------------------------------------------------------------------------


def extract_session_datetime(session_name: str, output_folder: str | None = None) -> datetime | None:
    details = extract_session_datetime_details(
        session_name=session_name,
        output_folder=output_folder,
    )
    if details is None:
        return None
    return details["session_start_datetime"]


def print_session_start_datetime_sources(session_table: pd.DataFrame) -> None:
    source_table = session_table[
        [
            "session_id",
            "session_name",
            "session_name_normalized",
            "session_start_datetime",
            "session_datetime_source_field",
            "session_datetime_source_text",
            "session_datetime_granularity",
            "session_datetime_pattern",
            "session_datetime_matched_text",
        ]
    ].copy()
    source_table["session_start_datetime"] = source_table["session_start_datetime"].map(
        lambda value: value.isoformat(sep=" ") if pd.notna(value) else ""
    )
    log_status("Session start datetime sources:")
    print(source_table.to_string(index=False), flush=True)


def resolve_label_series(metadata_table: pd.DataFrame, config: Config) -> tuple[pd.Series, str]:
    requested_label_type = str(config.label_type or "").strip().lower()
    label_aliases = {
        "clock_hour_of_day": "clock_hour_of_day",
        "hour_of_day": "clock_hour_of_day",
        "session_id": "session_id",
        "calendar_day": "calendar_day",
    }
    resolved_label_column = label_aliases.get(requested_label_type)
    if resolved_label_column is not None:
        if resolved_label_column not in metadata_table.columns:
            raise KeyError(f"Requested label column is missing from metadata: {resolved_label_column}")
        return metadata_table[resolved_label_column], resolved_label_column

    if requested_label_type == "day_number":
        if "calendar_day" not in metadata_table.columns:
            raise KeyError("Requested label type 'day_number' needs a calendar_day column.")
        ordered_days = pd.Index(pd.unique(metadata_table["calendar_day"].astype(str))).sort_values()
        day_number_lookup = {day: index for index, day in enumerate(ordered_days, start=1)}
        return metadata_table["calendar_day"].astype(str).map(day_number_lookup), "day_number"

    raise ValueError(
        "Unsupported label_type. Use one of: clock_hour_of_day, hour_of_day, session_id, "
        "calendar_day, day_number."
    )


def print_clock_hour_sample_pivot(metadata_table: pd.DataFrame, title: str) -> pd.DataFrame:
    if metadata_table.empty:
        log_status(f"{title}: no rows available.")
        return pd.DataFrame()

    pivot = (
        metadata_table.assign(sample_count=1)
        .pivot_table(
            index="calendar_day",
            columns="clock_hour_of_day",
            values="sample_count",
            aggfunc="sum",
            fill_value=0,
            dropna=False,
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    log_status(title)
    print(pivot.to_string(), flush=True)
    return pivot


def print_lda_label_vector(metadata_table: pd.DataFrame, labels: np.ndarray, label_column: str) -> None:
    label_table = metadata_table[
        [
            "final_sample_id",
            "final_sample_key",
            "calendar_day",
            "clock_hour_of_day",
        ]
    ].copy()
    label_table["lda_label"] = labels
    log_status(f"Exact LDA y vector uses label column: {label_column}")
    print(label_table.to_string(index=False), flush=True)
    log_status(f"Exact y vector: {labels.tolist()}")


def build_date_label_from_session_table(session_table: pd.DataFrame) -> str:
    parsed_dates: list[datetime] = []
    for row in session_table.itertuples(index=False):
        parsed_dt = extract_session_datetime(
            session_name=str(row.session_name),
            output_folder=str(row.output_folder),
        )
        if parsed_dt is not None:
            parsed_dates.append(parsed_dt)

    if not parsed_dates:
        return datetime.now().strftime("%y%m%d")

    unique_days = sorted({parsed_dt.strftime("%y%m%d") for parsed_dt in parsed_dates})
    if len(unique_days) == 1:
        return unique_days[0]
    return f"{unique_days[0]}_to_{unique_days[-1]}"


def filter_session_table_for_lda_mode(session_table: pd.DataFrame, config: Config) -> pd.DataFrame:
    if config.lda_mode != "single_day_5min":
        return session_table

    filtered_table = session_table.copy()
    session_dates = pd.to_datetime(filtered_table["session_start_datetime"]).dt.date.astype(str)
    unique_dates = sorted(pd.unique(session_dates).tolist())
    requested_date = (
        str(config.single_day_date).strip()
        if config.single_day_date is not None and str(config.single_day_date).strip()
        else None
    )

    if requested_date is None:
        if len(unique_dates) != 1:
            raise ValueError(
                "LDA_MODE='single_day_5min' analyzes one day at a time, but this input "
                f"contains multiple dates: {unique_dates}. Set SINGLE_DAY_DATE to one "
                "of these YYYY-MM-DD dates."
            )
        requested_date = unique_dates[0]
    elif requested_date not in unique_dates:
        raise ValueError(
            f"SINGLE_DAY_DATE={requested_date!r} was not found in the input sessions. "
            f"Available dates: {unique_dates}"
        )

    config.single_day_date = requested_date
    keep_mask = session_dates == requested_date
    filtered_table = filtered_table.loc[keep_mask].reset_index(drop=True)
    filtered_table["session_id"] = np.arange(1, len(filtered_table) + 1, dtype=int)
    log_status(
        f"single_day_5min mode: selected {requested_date} with "
        f"{len(filtered_table)} session(s); using {config.bin_size_seconds:.0f}s bins"
    )
    return filtered_table


# -----------------------------------------------------------------------------
# Matrix Smoothing, Missing Values, and Feature Scaling
# -----------------------------------------------------------------------------


def build_gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(np.ceil(3.0 * sigma_bins)))
    x_values = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x_values**2) / (2.0 * sigma_bins**2))
    kernel /= kernel.sum()
    return kernel


def smooth_population_matrix(
    population_matrix: np.ndarray,
    sigma_bins: float,
    feature_mask: np.ndarray | None = None,
) -> np.ndarray:
    kernel = build_gaussian_kernel(sigma_bins)
    smoothed = np.zeros_like(population_matrix, dtype=float)
    if feature_mask is None:
        feature_mask = np.ones(population_matrix.shape[1], dtype=bool)
    else:
        feature_mask = np.asarray(feature_mask, dtype=bool)
        if feature_mask.shape[0] != population_matrix.shape[1]:
            raise ValueError("feature_mask length must match the number of population features.")

    for feature_index in range(population_matrix.shape[1]):
        if feature_mask[feature_index]:
            smoothed[:, feature_index] = np.convolve(
                population_matrix[:, feature_index],
                kernel,
                mode="same",
            )
        else:
            smoothed[:, feature_index] = population_matrix[:, feature_index]
    return smoothed


def zscore_population_matrix(population_matrix: np.ndarray) -> np.ndarray:
    means = np.nanmean(population_matrix, axis=0)
    means[~np.isfinite(means)] = 0.0
    stds = np.nanstd(population_matrix, axis=0)
    stds[~np.isfinite(stds)] = 1.0
    stds[stds == 0] = 1.0
    zscored = (population_matrix - means) / stds
    zscored[~np.isfinite(zscored)] = 0.0
    return zscored


def fill_missing_feature_values(population_matrix: np.ndarray) -> np.ndarray:
    filled = np.asarray(population_matrix, dtype=float).copy()
    column_means = np.nanmean(filled, axis=0)
    column_means[~np.isfinite(column_means)] = 0.0
    missing_mask = ~np.isfinite(filled)
    if np.any(missing_mask):
        row_indices, column_indices = np.where(missing_mask)
        filled[row_indices, column_indices] = column_means[column_indices]
    return filled


# -----------------------------------------------------------------------------
# Feature Mode Selection
# -----------------------------------------------------------------------------


def normalize_feature_modes(feature_modes: tuple[str, ...] | list[str] | str) -> list[str]:
    if isinstance(feature_modes, str):
        raw_modes = [feature_modes]
    else:
        raw_modes = list(feature_modes)
    normalized_modes: list[str] = []
    valid_modes = {
        "FR_ONLY",
        "FR_AMP",
        "FR_CV2",
        "FR_PEAK_TO_TROUGH",
        "FR_WAVEFORM",
        "WAVEFORM_ONLY",
        "MULTI_FEATURE",
    }
    for mode in raw_modes:
        normalized_mode = str(mode).strip().upper()
        if normalized_mode not in valid_modes:
            raise ValueError(f"Unsupported feature mode: {mode}")
        if normalized_mode not in normalized_modes:
            normalized_modes.append(normalized_mode)
    if not normalized_modes:
        raise ValueError("At least one feature mode must be configured.")
    return normalized_modes


def is_waveform_feature_type(feature_type: str) -> bool:
    return str(feature_type).startswith("mean_waveform_uv_s")


def is_internal_weight_feature_type(feature_type: str) -> bool:
    return str(feature_type) == "n_crossings"


def subset_features_for_mode(
    population_matrix: np.ndarray,
    feature_table: pd.DataFrame,
    feature_mode: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    normalized_mode = str(feature_mode).strip().upper()
    allowed_feature_types_by_mode = {
        "FR_ONLY": {"firing_rate_hz"},
        "FR_AMP": {"firing_rate_hz", "average_amplitude_uv"},
        "FR_CV2": {"firing_rate_hz", "cv2"},
        "FR_PEAK_TO_TROUGH": {"firing_rate_hz", "peak_to_trough_ms"},
        "FR_WAVEFORM": {"firing_rate_hz"},
        "WAVEFORM_ONLY": set(),
        "MULTI_FEATURE": None,
    }
    if normalized_mode not in allowed_feature_types_by_mode:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    feature_types = feature_table["feature_type"].astype(str)
    if normalized_mode == "MULTI_FEATURE":
        keep_mask = (~feature_types.map(is_internal_weight_feature_type)).to_numpy()
    elif normalized_mode == "FR_WAVEFORM":
        keep_mask = (
            (feature_types == "firing_rate_hz")
            | feature_types.map(is_waveform_feature_type)
        ).to_numpy()
    elif normalized_mode == "WAVEFORM_ONLY":
        keep_mask = feature_types.map(is_waveform_feature_type).to_numpy()
    else:
        allowed_feature_types = allowed_feature_types_by_mode[normalized_mode]
        keep_mask = feature_types.isin(allowed_feature_types).to_numpy()

    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.shape[0] != population_matrix.shape[1]:
        raise ValueError("Feature subset mask length must match the population matrix width.")

    filtered_population = population_matrix[:, keep_mask]
    filtered_feature_table = feature_table.loc[keep_mask].reset_index(drop=True).copy()
    filtered_feature_table["feature_column"] = [
        f"feature_{feature_index:04d}"
        for feature_index in range(1, len(filtered_feature_table) + 1)
    ]
    return filtered_population, filtered_feature_table


def filter_hourly_samples_by_min_minutes(
    population_matrix: np.ndarray,
    metadata_table: pd.DataFrame,
    config: Config,
) -> tuple[np.ndarray, pd.DataFrame]:
    if "n_available_minutes_used" not in metadata_table.columns:
        raise KeyError("Hourly metadata is missing n_available_minutes_used.")
    keep_mask = (
        metadata_table["n_available_minutes_used"].fillna(0).astype(int)
        >= int(config.min_minutes_per_hour)
    ).to_numpy()
    filtered_population = population_matrix[keep_mask]
    filtered_metadata = metadata_table.loc[keep_mask].reset_index(drop=True)
    if filtered_metadata.empty:
        raise RuntimeError(
            "No hourly samples remained after applying MIN_MINUTES_PER_HOUR. "
            "Lower MIN_MINUTES_PER_HOUR or check the minute-bin coverage."
        )
    return filtered_population, filtered_metadata


# =============================================================================
# Module 5: Cross-Validation and Permutation Statistics
# =============================================================================


def predict_with_cv(
    population_matrix: np.ndarray,
    labels: np.ndarray,
    cv_splitter,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    labels = np.asarray(labels)
    predicted_labels = np.empty(labels.shape[0], dtype=labels.dtype)
    prediction_mask = np.zeros(labels.shape[0], dtype=bool)

    split_iter = cv_splitter.split(population_matrix, labels, groups)
    for train_indices, test_indices in split_iter:
        estimator = LinearDiscriminantAnalysis()
        estimator.fit(population_matrix[train_indices], labels[train_indices])
        predicted_labels[test_indices] = estimator.predict(population_matrix[test_indices])
        prediction_mask[test_indices] = True

    if not np.all(prediction_mask):
        raise RuntimeError("Cross-validation did not produce predictions for every sample.")
    return predicted_labels


def compute_empirical_p_value(observed_value: float, null_distribution: np.ndarray) -> float:
    null_distribution = np.asarray(null_distribution, dtype=float)
    if null_distribution.size == 0:
        return np.nan
    return float((1 + np.sum(null_distribution >= observed_value)) / (null_distribution.size + 1))


def circular_hour_error_hours(true_labels: np.ndarray, predicted_labels: np.ndarray) -> np.ndarray:
    true_values = np.asarray(true_labels, dtype=float).ravel()
    predicted_values = np.asarray(predicted_labels, dtype=float).ravel()
    raw_difference = np.abs(predicted_values - true_values)
    return np.minimum(raw_difference, 24.0 - raw_difference)


def mean_circular_hour_error_by_label(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    errors = circular_hour_error_hours(true_labels, predicted_labels)
    rows = []
    for label in sorted(pd.unique(true_labels).tolist()):
        mask = np.asarray(true_labels) == label
        label_errors = errors[mask]
        rows.append(
            {
                "label": to_jsonable_scalar(label),
                "n_samples": int(np.count_nonzero(mask)),
                "mean_circular_hour_error": float(np.nanmean(label_errors)),
                "median_circular_hour_error": float(np.nanmedian(label_errors)),
            }
        )
    per_label = pd.DataFrame(rows)
    if per_label.empty:
        return np.nan, per_label
    return float(per_label["mean_circular_hour_error"].mean()), per_label


def to_jsonable_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_permutation_test(
    population_matrix: np.ndarray,
    labels: np.ndarray,
    cv_splitter,
    config: Config,
    groups: np.ndarray | None = None,
) -> dict:
    rng = np.random.default_rng(config.random_seed)
    shuffled_scores = np.full(int(config.n_permutations), np.nan, dtype=float)
    for permutation_index in range(int(config.n_permutations)):
        shuffled_labels = rng.permutation(labels)
        shuffled_predictions = predict_with_cv(
            population_matrix=population_matrix,
            labels=shuffled_labels,
            cv_splitter=cv_splitter,
            groups=groups,
        )
        shuffled_scores[permutation_index] = balanced_accuracy_score(
            shuffled_labels,
            shuffled_predictions,
        )
    return {
        "n_permutations": int(config.n_permutations),
        "balanced_accuracy_distribution": shuffled_scores,
    }


def evaluate_cv_scheme(
    population_matrix: np.ndarray,
    labels: np.ndarray,
    config: Config,
    *,
    cv_name: str,
    cv_splitter,
    groups: np.ndarray | None = None,
) -> dict:
    predicted_labels = predict_with_cv(
        population_matrix=population_matrix,
        labels=labels,
        cv_splitter=cv_splitter,
        groups=groups,
    )
    unique_labels = sorted(pd.unique(labels).tolist())
    confusion = confusion_matrix(labels, predicted_labels, labels=unique_labels)
    accuracy = float(np.mean(predicted_labels == labels))
    balanced_accuracy = float(balanced_accuracy_score(labels, predicted_labels))
    sample_errors = circular_hour_error_hours(labels, predicted_labels)
    mean_sample_error = float(np.nanmean(sample_errors))
    mean_label_error, per_label_error = mean_circular_hour_error_by_label(
        true_labels=labels,
        predicted_labels=predicted_labels,
    )
    permutation_result = run_permutation_test(
        population_matrix=population_matrix,
        labels=labels,
        cv_splitter=cv_splitter,
        config=config,
        groups=groups,
    )
    shuffled_distribution = np.asarray(
        permutation_result["balanced_accuracy_distribution"],
        dtype=float,
    )
    return {
        "cv_name": cv_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": confusion,
        "confusion_labels": unique_labels,
        "predicted_labels": predicted_labels,
        "circular_hour_errors": sample_errors,
        "mean_circular_hour_error": mean_sample_error,
        "mean_label_circular_hour_error": mean_label_error,
        "per_label_circular_hour_error": per_label_error,
        "n_splits": int(cv_splitter.get_n_splits(population_matrix, labels, groups)),
        "groups_used": groups is not None,
        "n_permutations": int(permutation_result["n_permutations"]),
        "permutation_balanced_accuracy_distribution": shuffled_distribution,
        "permutation_balanced_accuracy_mean": float(np.mean(shuffled_distribution)),
        "permutation_balanced_accuracy_std": float(np.std(shuffled_distribution)),
        "permutation_p_value": compute_empirical_p_value(balanced_accuracy, shuffled_distribution),
    }


# =============================================================================
# Module 6: Session Tables, Unit Selection, and Population Matrix Building
# =============================================================================


# -----------------------------------------------------------------------------
# Session Table and Analyzer Loading
# -----------------------------------------------------------------------------


def build_session_table(export_payload: dict, config: Config) -> pd.DataFrame:
    group_rows = export_payload.get("cross_session_alignment_groups", [])
    if not group_rows:
        raise RuntimeError("No cross_session_alignment_groups were found in the export summary.")

    rows: list[dict] = []
    for group_row in group_rows:
        final_group_key = str(group_row.get("final_group_key", "")).strip()
        final_unit_id = safe_int(group_row.get("final_unit_id"))
        for member in iter_group_members(group_row):
            output_folder = str(member.get("output_folder", "") or "").strip()
            session_name = str(member.get("session_name", "") or "").strip()
            session_index = safe_int(member.get("session_index"))
            unit_id = safe_int(member.get("unit_id"))
            if not output_folder or not session_name or unit_id is None:
                continue
            rows.append(
                {
                    "session_key": output_folder,
                    "final_group_key": final_group_key,
                    "final_unit_id": final_unit_id,
                    "session_name": session_name,
                    "session_name_normalized": normalize_session_name(session_name),
                    "session_index": session_index,
                    "unit_id": unit_id,
                    "output_folder": output_folder,
                }
            )

    session_table = pd.DataFrame(rows)
    if session_table.empty:
        raise RuntimeError("The alignment export did not contain any valid member units.")

    grouped = (
        session_table[["session_key", "session_name", "session_name_normalized", "session_index", "output_folder"]]
        .drop_duplicates()
        .sort_values(["session_index", "session_name", "output_folder"], na_position="last")
        .reset_index(drop=True)
    )
    grouped["session_id"] = np.arange(1, len(grouped) + 1, dtype=int)
    session_datetime_details = [
        extract_session_datetime_details(
            session_name=str(row.session_name),
            output_folder=str(row.output_folder),
        )
        for row in grouped.itertuples(index=False)
    ]
    grouped["session_start_datetime"] = [
        details["session_start_datetime"] if details is not None else pd.NaT
        for details in session_datetime_details
    ]
    grouped["session_datetime_source_field"] = [
        str(details["source_field"]) if details is not None else ""
        for details in session_datetime_details
    ]
    grouped["session_datetime_source_text"] = [
        str(details["source_text"]) if details is not None else ""
        for details in session_datetime_details
    ]
    grouped["session_datetime_matched_text"] = [
        str(details["matched_text"]) if details is not None else ""
        for details in session_datetime_details
    ]
    grouped["session_datetime_pattern"] = [
        str(details["pattern_name"]) if details is not None else ""
        for details in session_datetime_details
    ]
    grouped["session_datetime_granularity"] = [
        str(details["granularity"]) if details is not None else ""
        for details in session_datetime_details
    ]
    missing_session_start = grouped["session_start_datetime"].isna()
    if missing_session_start.any():
        missing_names = grouped.loc[missing_session_start, "session_name"].tolist()
        raise ValueError(
            "Could not parse real session start datetime for these sessions: "
            f"{missing_names}. Clock-hour labels require real start timestamps."
        )
    print_session_start_datetime_sources(grouped)
    return grouped


def load_session_analyzers(
    session_table: pd.DataFrame,
    config: Config,
) -> tuple[
    dict[str, object],
    dict[str, str],
]:
    import spikeinterface.full as si

    analyzers: dict[str, object] = {}
    resolved_output_folders: dict[str, str] = {}
    path_remap: tuple[Path, Path] | None = None

    for row in session_table.itertuples(index=False):
        output_folder = Path(str(row.output_folder))
        log_status(f"Loading session analyzer: {row.session_name}")
        analyzer_folder, path_remap = resolve_analyzer_folder_path(
            output_folder=output_folder,
            config=config,
            path_remap=path_remap,
        )

        session_key = str(row.session_key)
        analyzers[session_key] = si.load_sorting_analyzer(
            folder=analyzer_folder,
            format="zarr",
            load_extensions=True,
        )
        resolved_output_folders[session_key] = str(analyzer_folder.parent)
        log_status(f"Loaded analyzer from: {analyzer_folder}")

    return analyzers, resolved_output_folders


# -----------------------------------------------------------------------------
# Aligned Unit Selection
# -----------------------------------------------------------------------------


def select_good_unit_groups(
    export_payload: dict,
    config: Config,
    analyzers: dict[str, object],
) -> pd.DataFrame:
    group_rows = export_payload.get("cross_session_alignment_groups", [])
    selected_rows: list[dict] = []
    analyzer_unit_ids_lookup = {
        session_key: {int(unit_id) for unit_id in analyzer.sorting.get_unit_ids()}
        for session_key, analyzer in analyzers.items()
    }
    dropped_missing_analyzer_units = 0
    log_status(f"Evaluating {len(group_rows)} aligned unit groups")
    selection_mode, effective_min_count = infer_group_selection_mode(
        export_payload=export_payload,
        group_rows=group_rows,
        config=config,
    )
    member_mode = str(export_payload.get("member_mode", "unknown") or "unknown")
    log_status(f"Detected export member mode: {member_mode}")
    if selection_mode == "unique_underlying_sessions":
        log_status(
            f"Applying MIN_SESSIONS_PER_UNIT against unique underlying aligned sessions "
            f"with threshold {effective_min_count}."
        )
    elif selection_mode == "unique_export_sessions":
        log_status(
            "Detected day-level Alignment_days export without flattened source_members. "
            f"Applying MIN_SESSIONS_PER_UNIT against unique exported day members "
            f"with effective threshold {effective_min_count}."
        )
    else:
        log_status(
            f"Applying MIN_SESSIONS_PER_UNIT against member rows with threshold {effective_min_count}."
        )

    for group_index, group_row in enumerate(group_rows, start=1):
        final_group_key = str(group_row.get("final_group_key", "")).strip()
        final_unit_id = safe_int(group_row.get("final_unit_id"))
        members = iter_group_members(group_row)
        valid_members: list[dict] = []
        if group_index == 1 or group_index % 100 == 0 or group_index == len(group_rows):
            log_status(f"Selecting good units: group {group_index} / {len(group_rows)}")

        for member in members:
            session_name = str(member.get("session_name", "") or "").strip()
            unit_id = safe_int(member.get("unit_id"))
            output_folder = str(member.get("output_folder", "") or "").strip()
            if not session_name or unit_id is None or not output_folder:
                continue

            session_key = output_folder
            valid_unit_ids = analyzer_unit_ids_lookup.get(session_key, set())
            if int(unit_id) not in valid_unit_ids:
                dropped_missing_analyzer_units += 1
                continue
            valid_members.append(
                {
                    "session_key": session_key,
                    "final_group_key": final_group_key,
                    "final_unit_id": final_unit_id,
                    "session_name": session_name,
                    "session_index": safe_int(member.get("session_index")),
                    "unit_id": unit_id,
                    "output_folder": output_folder,
                    "shank_id": safe_int(group_row.get("shank_id")),
                    "local_channel_on_shank": safe_int(group_row.get("local_channel_on_shank")),
                }
            )

        if selection_mode == "unique_underlying_sessions":
            group_presence_count = len(
                {
                    str(row["session_key"]).strip()
                    for row in valid_members
                    if str(row["session_key"]).strip()
                }
            )
        elif selection_mode == "unique_export_sessions":
            group_presence_count = len(
                {
                    str(row["session_name"]).strip()
                    for row in valid_members
                    if str(row["session_name"]).strip()
                }
            )
        else:
            group_presence_count = len(valid_members)

        if group_presence_count < effective_min_count:
            continue

        selected_rows.extend(valid_members)

    if dropped_missing_analyzer_units > 0:
        log_status(
            f"Skipped {dropped_missing_analyzer_units} member rows because their unit_id was not present "
            "in the corresponding loaded analyzer."
        )

    selected_table = pd.DataFrame(selected_rows)
    if selected_table.empty:
        raise RuntimeError(
            "No aligned unit groups passed the selection criteria. "
            "Try lowering MIN_SESSIONS_PER_UNIT or checking whether the Alignment_days export "
            "contains only day-level members instead of flattened source_members."
        )

    return selected_table.sort_values(
        ["final_unit_id", "session_index", "unit_id"],
        na_position="last",
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Population Feature Matrix Construction
# -----------------------------------------------------------------------------


def build_population_vectors(
    selected_units: pd.DataFrame,
    session_table: pd.DataFrame,
    analyzers: dict[str, object],
    config: Config,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    feature_table = build_feature_table(selected_units)
    feature_keys = feature_table["feature_key"].astype(str).tolist()
    feature_index = {key: index for index, key in enumerate(feature_keys)}
    smoothable_feature_mask = (
        feature_table["feature_type"].astype(str).to_numpy() == "firing_rate_hz"
    )

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
            f"Binning population at {config.bin_size_seconds:.0f}s resolution for "
            f"session {session_position} / {total_sessions}: {session_name}"
        )
        analyzer = analyzers[session_key]
        static_features_by_unit = compute_session_unit_static_features(analyzer)
        spike_amplitudes_by_unit = get_session_spike_amplitudes_by_unit(analyzer)
        waveform_snippets_by_unit = get_session_waveform_snippets_by_unit(analyzer)
        if not spike_amplitudes_by_unit:
            log_status(
                f"Session '{session_name}': no spike_amplitudes extension was available; "
                "binned amplitude features will be missing."
            )
        if not waveform_snippets_by_unit:
            log_status(
                f"Session '{session_name}': no waveforms/random_spikes extensions were available; "
                "binned peak-to-trough features will use template fallback for bins with spikes."
            )
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
            log_status(
                f"Skipping session '{session_name}' because duration {session_duration_s:.2f}s "
                f"is shorter than one full {config.bin_size_seconds:.0f}s bin."
            )
            continue

        bin_edges = np.arange(n_complete_bins + 1, dtype=float) * config.bin_size_seconds
        if len(bin_edges) < 2:
            continue
        bin_edges_samples = bin_edges * sampling_frequency

        session_matrix = np.zeros((len(bin_edges) - 1, len(feature_keys)), dtype=float)
        session_units = members_by_session.get(session_key, pd.DataFrame())
        skipped_invalid_units = 0
        log_status(
            f"Session '{session_name}': {len(session_units)} units, "
            f"{len(bin_edges) - 1} bins at {config.bin_size_seconds:.0f}s"
        )

        for member_row in session_units.itertuples(index=False):
            if int(member_row.unit_id) not in valid_unit_ids:
                skipped_invalid_units += 1
                continue
            feature_key = str(member_row.final_group_key)
            spike_train_samples = analyzer.sorting.get_unit_spike_train(
                unit_id=int(member_row.unit_id),
                segment_index=0,
            )
            spike_times_s = np.asarray(spike_train_samples, dtype=float) / sampling_frequency
            counts, _ = np.histogram(spike_times_s, bins=bin_edges)
            n_bins = len(bin_edges) - 1
            bin_indices = np.searchsorted(bin_edges, spike_times_s, side="right") - 1
            bin_indices[spike_times_s == bin_edges[-1]] = n_bins - 1
            rate_feature_key = f"{feature_key}__firing_rate_hz"
            session_matrix[:, feature_index[rate_feature_key]] = (
                counts.astype(float) / config.bin_size_seconds
            )

            static_features = static_features_by_unit.get(int(member_row.unit_id), {})
            amplitude_feature_key = f"{feature_key}__average_amplitude_uv"
            session_matrix[:, feature_index[amplitude_feature_key]] = binned_mean_abs_amplitude(
                spike_train_samples=np.asarray(spike_train_samples, dtype=float),
                spike_amplitudes=spike_amplitudes_by_unit.get(int(member_row.unit_id)),
                bin_indices=bin_indices,
                n_bins=n_bins,
            )

            cv2_feature_key = f"{feature_key}__cv2"
            session_matrix[:, feature_index[cv2_feature_key]] = binned_cv2(
                spike_train_samples=np.asarray(spike_train_samples, dtype=float),
                bin_indices=bin_indices,
                n_bins=n_bins,
            )

            peak_to_trough_feature_key = f"{feature_key}__peak_to_trough_ms"
            waveform_spike_samples = None
            waveforms = None
            waveform_payload = waveform_snippets_by_unit.get(int(member_row.unit_id))
            if waveform_payload is not None:
                waveform_spike_samples, waveforms = waveform_payload
            waveform_peak_to_trough = binned_waveform_peak_to_trough(
                waveform_spike_samples=waveform_spike_samples,
                waveforms=waveforms,
                bin_edges_samples=bin_edges_samples,
                n_bins=n_bins,
                sampling_frequency=sampling_frequency,
            )
            template_peak_to_trough = binned_template_peak_to_trough(
                peak_to_trough_ms=static_features.get("peak_to_trough_ms"),
                counts=counts,
            )
            missing_peak_to_trough = ~np.isfinite(waveform_peak_to_trough)
            waveform_peak_to_trough[missing_peak_to_trough] = template_peak_to_trough[
                missing_peak_to_trough
            ]
            session_matrix[:, feature_index[peak_to_trough_feature_key]] = waveform_peak_to_trough

        if skipped_invalid_units > 0:
            log_status(
                f"Session '{session_name}': skipped {skipped_invalid_units} units because their unit_id "
                "was not present in the loaded analyzer"
            )

        if config.apply_smoothing:
            log_status(f"Applying smoothing to session '{session_name}'")
            session_matrix = smooth_population_matrix(
                population_matrix=session_matrix,
                sigma_bins=config.smoothing_sigma_bins,
                feature_mask=smoothable_feature_mask,
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
                    "session_index": safe_int(session_row.session_index),
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
        raise RuntimeError("No population vectors were created. Check the sessions and bin size.")

    population_matrix = np.vstack(samples)
    metadata_table = pd.DataFrame(metadata_rows)
    log_status(
        f"Finished {config.bin_size_seconds:.0f}s binning: created "
        f"{population_matrix.shape[0]} samples across {len(session_table)} sessions"
    )
    missing_static_mask = feature_table["feature_type"].astype(str) != "firing_rate_hz"
    if missing_static_mask.any():
        n_missing_static = int(np.isnan(population_matrix[:, missing_static_mask.to_numpy()]).sum())
        if n_missing_static > 0:
            log_status(
                f"Computed non-firing-rate feature bins with {n_missing_static} missing values; "
                "these will remain NaN until downstream imputation/z-scoring."
            )
    return population_matrix, metadata_table, feature_table


# -----------------------------------------------------------------------------
# LDA Sample Aggregation and Sample Filtering
# -----------------------------------------------------------------------------


def aggregate_minutes_to_hourly_samples(
    minute_population_matrix: np.ndarray,
    minute_metadata_table: pd.DataFrame,
    feature_table: pd.DataFrame | None = None,
    waveform_weighting: str = "minute",
) -> tuple[np.ndarray, pd.DataFrame]:
    if minute_population_matrix.shape[0] != len(minute_metadata_table):
        raise ValueError("Minute population matrix row count does not match the minute metadata length.")

    if minute_population_matrix.size == 0:
        raise RuntimeError("No minute-level population vectors are available for hourly aggregation.")

    normalized_weighting = str(waveform_weighting or "minute").strip().lower()
    if normalized_weighting not in {"minute", "crossing_count"}:
        raise ValueError("waveform_weighting must be 'minute' or 'crossing_count'.")
    if feature_table is not None:
        feature_columns = feature_table["feature_column"].astype(str).tolist()
        if len(feature_columns) != minute_population_matrix.shape[1]:
            raise ValueError("Feature table length does not match the population matrix width.")
    else:
        feature_columns = [
            f"unit_{feature_index:04d}"
            for feature_index in range(1, minute_population_matrix.shape[1] + 1)
        ]

    waveform_feature_to_weight_column: dict[str, str] = {}
    if normalized_weighting == "crossing_count":
        if feature_table is None:
            raise ValueError("crossing_count waveform weighting requires a feature_table.")
        feature_lookup = {
            (str(row.final_group_key), str(row.feature_type)): str(row.feature_column)
            for row in feature_table.itertuples(index=False)
        }
        for row in feature_table.itertuples(index=False):
            feature_type = str(row.feature_type)
            if is_waveform_feature_type(feature_type):
                weight_column = feature_lookup.get((str(row.final_group_key), "n_crossings"))
                if weight_column is not None:
                    waveform_feature_to_weight_column[str(row.feature_column)] = weight_column
        log_status(
            f"Using crossing-count-weighted hourly averaging for "
            f"{len(waveform_feature_to_weight_column)} waveform feature columns."
        )
    minute_df = pd.concat(
        [
            minute_metadata_table.reset_index(drop=True),
            pd.DataFrame(minute_population_matrix, columns=feature_columns),
        ],
        axis=1,
    )

    hourly_vectors: list[np.ndarray] = []
    hourly_rows: list[dict] = []
    extra_grouping_keys = [
        column
        for column in ("threshold_run_name", "threshold_run_root")
        if column in minute_df.columns and minute_df[column].nunique(dropna=False) > 1
    ]
    grouping_keys = extra_grouping_keys + ["calendar_day", "clock_hour_of_day"]
    log_status(f"Hourly aggregation grouping keys: {grouping_keys}")
    grouped = minute_df.groupby(grouping_keys, sort=True, dropna=False)
    total_groups = len(grouped)

    for group_index, (group_values, group_table) in enumerate(grouped, start=1):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_lookup = dict(zip(grouping_keys, group_values))
        calendar_day = group_lookup["calendar_day"]
        clock_hour_of_day = group_lookup["clock_hour_of_day"]
        if group_index == 1 or group_index % 100 == 0 or group_index == total_groups:
            log_status(f"Aggregating hourly samples: group {group_index} / {total_groups}")

        if normalized_weighting == "minute" or not waveform_feature_to_weight_column:
            feature_values = group_table[feature_columns].to_numpy(dtype=float)
            hourly_vectors.append(np.nanmean(feature_values, axis=0))
        else:
            hourly_vector = np.full(len(feature_columns), np.nan, dtype=float)
            for feature_index, feature_column in enumerate(feature_columns):
                values = group_table[feature_column].to_numpy(dtype=float)
                weight_column = waveform_feature_to_weight_column.get(feature_column)
                if weight_column is None:
                    hourly_vector[feature_index] = np.nanmean(values)
                    continue
                weights = group_table[weight_column].to_numpy(dtype=float)
                valid_mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
                if np.any(valid_mask):
                    hourly_vector[feature_index] = float(
                        np.average(values[valid_mask], weights=weights[valid_mask])
                    )
                else:
                    hourly_vector[feature_index] = np.nanmean(values)
            hourly_vectors.append(hourly_vector)

        sorted_group = group_table.sort_values(
            ["minute_start_datetime", "session_id", "minute_bin_index"],
            na_position="last",
        )
        unique_session_ids = sorted(pd.unique(sorted_group["session_id"]).tolist())
        unique_session_names = [str(value) for value in pd.unique(sorted_group["session_name"]).tolist()]
        unique_clock_minutes_present = sorted(
            int(value)
            for value in pd.unique(sorted_group["clock_minute_of_hour"]).tolist()
            if safe_int(value) is not None
        )
        n_unique_clock_minutes_present = len(unique_clock_minutes_present)
        n_missing_clock_minutes = max(0, 60 - n_unique_clock_minutes_present)
        run_key_parts = [
            safe_slug(group_lookup[column])
            for column in extra_grouping_keys
            if str(group_lookup.get(column, "")).strip()
        ]
        sample_key_prefix = "__".join(run_key_parts + [str(calendar_day)])
        hourly_row = {
            column: group_lookup[column]
            for column in extra_grouping_keys
        }
        hourly_row.update(
            {
                "final_sample_id": int(group_index),
                "final_sample_key": f"{sample_key_prefix}__hour_{int(clock_hour_of_day):02d}",
                "calendar_day": str(calendar_day),
                "clock_hour_of_day": int(clock_hour_of_day),
                "hour_start_datetime": f"{calendar_day} {int(clock_hour_of_day):02d}:00:00",
                "n_available_minutes_used": int(len(sorted_group)),
                "n_unique_clock_minutes_present": int(n_unique_clock_minutes_present),
                "n_missing_clock_minutes": int(n_missing_clock_minutes),
                "any_minutes_missing": bool(n_missing_clock_minutes > 0),
                "session_ids": ",".join(str(value) for value in unique_session_ids),
                "session_names": " | ".join(unique_session_names),
                "n_sessions_contributing": int(len(unique_session_ids)),
                "first_session_id": int(unique_session_ids[0]) if unique_session_ids else np.nan,
                "first_session_name": unique_session_names[0] if unique_session_names else "",
                "waveform_hourly_weighting": normalized_weighting,
            }
        )

        hourly_rows.append(hourly_row)

    hourly_population_matrix = np.vstack(hourly_vectors)
    hourly_metadata_table = pd.DataFrame(hourly_rows).sort_values(
        grouping_keys,
        na_position="last",
    ).reset_index(drop=True)
    duplicate_sample_keys = hourly_metadata_table["final_sample_key"].duplicated()
    if duplicate_sample_keys.any():
        duplicate_keys = hourly_metadata_table.loc[duplicate_sample_keys, "final_sample_key"].tolist()
        raise RuntimeError(
            "Hourly aggregation produced duplicate day x hour sample keys, which indicates "
            f"unexpected collapsing logic: {duplicate_keys[:10]}"
        )
    log_status(
        f"Finished hourly aggregation: created {hourly_population_matrix.shape[0]} day x hour samples"
    )
    return hourly_population_matrix, hourly_metadata_table


def prepare_single_day_5min_samples(
    population_matrix: np.ndarray,
    metadata_table: pd.DataFrame,
    config: Config,
) -> tuple[np.ndarray, pd.DataFrame]:
    if population_matrix.shape[0] != len(metadata_table):
        raise ValueError("Population matrix row count does not match metadata length.")
    if population_matrix.size == 0:
        raise RuntimeError("No 5-minute population vectors are available for LDA.")

    selected_date = str(config.single_day_date or "").strip()
    sample_metadata = metadata_table.copy()
    if selected_date:
        keep_mask = sample_metadata["calendar_day"].astype(str) == selected_date
        sample_metadata = sample_metadata.loc[keep_mask].copy()
        if sample_metadata.empty:
            raise RuntimeError(
                f"No 5-minute bins remained after filtering to SINGLE_DAY_DATE={selected_date!r}."
            )

    sample_metadata = sample_metadata.sort_values(
        ["calendar_day", "minute_start_datetime", "session_id", "minute_bin_index"],
        na_position="last",
    )
    row_order = sample_metadata.index.to_numpy()
    sample_population = population_matrix[row_order]
    sample_metadata = sample_metadata.reset_index(drop=True)
    sample_metadata["final_sample_id"] = np.arange(1, len(sample_metadata) + 1, dtype=int)
    sample_metadata["final_sample_key"] = [
        f"{row.calendar_day}__session_{int(row.session_id):03d}__bin_{int(row.minute_bin_index):04d}"
        for row in sample_metadata.itertuples(index=False)
    ]
    sample_metadata["sample_start_datetime"] = sample_metadata["minute_start_datetime"]
    sample_metadata["sample_end_datetime"] = sample_metadata["minute_end_datetime"]
    sample_metadata["sample_duration_s"] = (
        sample_metadata["minute_end_sec"].astype(float)
        - sample_metadata["minute_start_sec"].astype(float)
    )
    log_status(
        f"single_day_5min mode: using {sample_population.shape[0]} "
        "5-minute bin(s) directly as LDA samples"
    )
    return sample_population, sample_metadata


def filter_unit_groups_by_binned_firing_rate(
    population_matrix: np.ndarray,
    selected_units: pd.DataFrame,
    feature_table: pd.DataFrame,
    config: Config,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if population_matrix.shape[1] != len(feature_table):
        raise ValueError("Feature count does not match the population matrix width.")

    ordered_feature_table = feature_table.reset_index(drop=True).copy()
    firing_rate_mask = ordered_feature_table["feature_type"].astype(str) == "firing_rate_hz"
    if not firing_rate_mask.any():
        raise RuntimeError("No firing-rate feature columns were defined for thresholding.")

    mean_binned_firing_rates = population_matrix[:, firing_rate_mask.to_numpy()].mean(axis=0)
    firing_rate_groups = ordered_feature_table.loc[
        firing_rate_mask,
        ["final_group_key", "final_unit_id", "shank_id", "local_channel_on_shank"],
    ].reset_index(drop=True)
    firing_rate_groups["mean_binned_firing_rate_hz"] = mean_binned_firing_rates
    firing_rate_groups["passed_min_firing_rate"] = (
        firing_rate_groups["mean_binned_firing_rate_hz"] >= config.min_firing_rate_hz
    )
    keep_group_keys = set(
        firing_rate_groups.loc[
            firing_rate_groups["passed_min_firing_rate"],
            "final_group_key",
        ].astype(str)
    )

    if not keep_group_keys:
        raise RuntimeError(
            "No aligned unit groups passed the MIN_FIRING_RATE_HZ threshold when computed from "
            "binned spike counts. Lower MIN_FIRING_RATE_HZ or increase BIN_SIZE_SECONDS."
        )

    keep_mask = ordered_feature_table["final_group_key"].astype(str).isin(keep_group_keys).to_numpy()
    filtered_population = population_matrix[:, keep_mask]
    filtered_selected_units = selected_units[
        selected_units["final_group_key"].astype(str).isin(keep_group_keys)
    ].copy()
    filtered_feature_table = ordered_feature_table.loc[keep_mask].reset_index(drop=True)

    log_status(
        f"Kept {len(keep_group_keys)} / {len(firing_rate_groups)} unit groups after applying "
        f"the binned firing-rate threshold; retained {len(filtered_feature_table)} feature columns"
    )
    return filtered_population, filtered_selected_units, filtered_feature_table, firing_rate_groups


def print_and_build_clock_hour_verification(metadata_table: pd.DataFrame) -> pd.DataFrame:
    verification_table = metadata_table[
        [
            "final_sample_key",
            "calendar_day",
            "clock_hour_of_day",
            "n_available_minutes_used",
            "n_missing_clock_minutes",
            "any_minutes_missing",
        ]
    ].copy().sort_values(["calendar_day", "clock_hour_of_day"], na_position="last").reset_index(drop=True)
    log_status("Hourly aggregation verification preview (first 20 rows):")
    if verification_table.empty:
        log_status("No verification rows available.")
    else:
        print(verification_table.head(20).to_string(index=False), flush=True)
    return verification_table


# =============================================================================
# Module 7: LDA Fitting and Decoding Evaluation
# =============================================================================


def filter_labels_for_lda(
    population_matrix: np.ndarray,
    metadata_table: pd.DataFrame,
    config: Config,
) -> tuple[np.ndarray, pd.DataFrame]:
    label_series, resolved_label_column = resolve_label_series(metadata_table, config)
    label_counts = label_series.value_counts()
    kept_labels = sorted(label_counts[label_counts >= config.min_bins_per_label].index.tolist())
    keep_mask = label_series.isin(kept_labels).to_numpy()
    filtered_population = population_matrix[keep_mask]
    filtered_metadata = metadata_table.loc[keep_mask].reset_index(drop=True)
    if filtered_metadata.empty or len(kept_labels) < 2:
        raise RuntimeError(
            f"LDA needs at least two '{resolved_label_column}' labels with enough bins. "
            "Check session timestamps or reduce MIN_BINS_PER_LABEL."
        )

    return filtered_population, filtered_metadata


def fit_lda(population_matrix: np.ndarray, labels: np.ndarray) -> tuple[LinearDiscriminantAnalysis, np.ndarray]:
    n_classes = len(np.unique(labels))
    n_features = population_matrix.shape[1]
    n_components = max(1, min(3, n_classes - 1, n_features))
    lda_model = LinearDiscriminantAnalysis(n_components=n_components)
    projection = lda_model.fit_transform(population_matrix, labels)
    return lda_model, projection


def evaluate_decoding(
    population_matrix: np.ndarray,
    labels: np.ndarray,
    metadata_table: pd.DataFrame,
    resolved_label_column: str,
    config: Config,
) -> dict[str, dict]:
    class_counts = pd.Series(labels).value_counts()
    smallest_class = int(class_counts.min())
    n_splits = max(2, min(config.cv_n_splits, smallest_class))
    if smallest_class < 2:
        raise RuntimeError(
            "Cross-validation needs at least two bins in each label after filtering."
        )

    results: dict[str, dict] = {}
    stratified_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )
    results["stratified"] = evaluate_cv_scheme(
        population_matrix=population_matrix,
        labels=labels,
        config=config,
        cv_name="stratified",
        cv_splitter=stratified_cv,
        groups=None,
    )

    grouped_allowed = str(resolved_label_column) == "clock_hour_of_day"
    if grouped_allowed:
        day_groups = metadata_table["calendar_day"].astype(str).to_numpy()
        n_unique_days = len(pd.unique(day_groups))
        if n_unique_days >= 2:
            grouped_splits = max(2, min(config.cv_n_splits, n_unique_days))
            grouped_cv = GroupKFold(n_splits=grouped_splits)
            results["grouped_by_day"] = evaluate_cv_scheme(
                population_matrix=population_matrix,
                labels=labels,
                config=config,
                cv_name="grouped_by_day",
                cv_splitter=grouped_cv,
                groups=day_groups,
            )
        else:
            log_status(
                "Skipping grouped-by-day CV because fewer than two calendar days remain "
                "after filtering."
            )

    return results


# =============================================================================
# Module 8: Plotting
# =============================================================================


# -----------------------------------------------------------------------------
# LDA Projection Plots
# -----------------------------------------------------------------------------


def plot_lda_2d(projection: np.ndarray, metadata_table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    hours = metadata_table["clock_hour_of_day"].to_numpy(dtype=float)
    calendar_days = metadata_table["calendar_day"].astype(str).to_numpy()
    unique_calendar_days = pd.unique(calendar_days)
    sort_columns = ["clock_hour_of_day"]
    if "sample_start_datetime" in metadata_table.columns:
        sort_columns = ["sample_start_datetime", "session_id", "minute_bin_index"]
    elif "hour_start_datetime" in metadata_table.columns:
        sort_columns = ["hour_start_datetime"]

    x_values = projection[:, 0]
    y_values = projection[:, 1] if projection.shape[1] >= 2 else np.zeros(len(projection))

    if len(unique_calendar_days) > 1:
        for calendar_day in unique_calendar_days:
            mask = calendar_days == calendar_day
            session_points = metadata_table.loc[mask].sort_values(sort_columns)
            if len(session_points) < 2:
                continue
            point_indices = session_points.index.to_numpy()
            ax.plot(
                x_values[point_indices],
                y_values[point_indices],
                color="#f4a3b5",
                linewidth=0.8,
                alpha=0.28,
                zorder=1,
            )

    scatter = ax.scatter(
        x_values,
        y_values,
        c=hours,
        cmap=CIRCULAR_HOUR_CMAP,
        norm=CIRCULAR_HOUR_NORM,
        s=44,
        alpha=0.92,
        edgecolors="black",
        linewidths=0.35,
        zorder=2,
    )

    ax.set_title("Population LDA Projection (2D)")
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        boundaries=CIRCULAR_HOUR_BOUNDARIES,
        ticks=CIRCULAR_HOUR_TICKS,
        spacing="proportional",
        drawedges=True,
    )
    colorbar.set_label(CIRCULAR_HOUR_LABEL)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_lda_3d(projection: np.ndarray, metadata_table: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    hours = metadata_table["clock_hour_of_day"].to_numpy(dtype=float)
    calendar_days = metadata_table["calendar_day"].astype(str).to_numpy()
    unique_calendar_days = pd.unique(calendar_days)
    sort_columns = ["clock_hour_of_day"]
    if "sample_start_datetime" in metadata_table.columns:
        sort_columns = ["sample_start_datetime", "session_id", "minute_bin_index"]
    elif "hour_start_datetime" in metadata_table.columns:
        sort_columns = ["hour_start_datetime"]

    x_values = projection[:, 0]
    y_values = projection[:, 1] if projection.shape[1] >= 2 else np.zeros(len(projection))
    z_values = projection[:, 2] if projection.shape[1] >= 3 else np.zeros(len(projection))

    if len(unique_calendar_days) > 1:
        for calendar_day in unique_calendar_days:
            mask = calendar_days == calendar_day
            session_points = metadata_table.loc[mask].sort_values(sort_columns)
            if len(session_points) < 2:
                continue
            point_indices = session_points.index.to_numpy()
            ax.plot(
                x_values[point_indices],
                y_values[point_indices],
                z_values[point_indices],
                color="#f4a3b5",
                linewidth=0.8,
                alpha=0.28,
                zorder=1,
            )

    scatter = ax.scatter(
        x_values,
        y_values,
        z_values,
        c=hours,
        cmap=CIRCULAR_HOUR_CMAP,
        norm=CIRCULAR_HOUR_NORM,
        s=36,
        alpha=0.92,
        edgecolors="black",
        linewidths=0.35,
    )

    ax.set_title("Population LDA Projection (3D)")
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_zlabel("LD3")
    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        boundaries=CIRCULAR_HOUR_BOUNDARIES,
        ticks=CIRCULAR_HOUR_TICKS,
        spacing="proportional",
        drawedges=True,
    )
    colorbar.set_label(CIRCULAR_HOUR_LABEL)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Decoding Confusion Matrix Plot
# -----------------------------------------------------------------------------


def plot_confusion_matrix(decoding_result: dict, output_path: Path) -> None:
    confusion = np.asarray(decoding_result["confusion_matrix"], dtype=float)
    labels = [str(label) for label in decoding_result["confusion_labels"]]
    cv_name = str(decoding_result.get("cv_name", "cross_validated"))

    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized = confusion / row_sums
    mean_sample_error = decoding_result.get("mean_circular_hour_error", np.nan)
    mean_label_error = decoding_result.get("mean_label_circular_hour_error", np.nan)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"LDA Confusion Matrix ({cv_name})\n"
        f"Mean circular error={mean_sample_error:.2f} h; "
        f"mean across labels={mean_label_error:.2f} h"
    )

    for row_index in range(normalized.shape[0]):
        for column_index in range(normalized.shape[1]):
            value = normalized[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def feature_label_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in [
            "feature_column",
            "feature_key",
            "source_column",
            "final_group_key",
            "final_unit_id",
            "shank_id",
            "local_channel_on_shank",
            "feature_type",
        ]
        if column in feature_table.columns
    ]
    return feature_table[columns].reset_index(drop=True).copy()


def save_feature_matrix_diagnostics(
    *,
    output_dir: Path,
    file_prefix: str,
    feature_table: pd.DataFrame,
    metadata_table: pd.DataFrame,
    raw_population_matrix: np.ndarray,
    processed_population_matrix: np.ndarray,
) -> None:
    population_columns = feature_table["feature_column"].astype(str).tolist()
    raw_df = pd.DataFrame(raw_population_matrix, columns=population_columns)
    processed_df = pd.DataFrame(processed_population_matrix, columns=population_columns)

    summary_rows = []
    labels = feature_label_table(feature_table)
    for feature_index, column in enumerate(population_columns):
        raw_values = raw_df[column].to_numpy(dtype=float)
        processed_values = processed_df[column].to_numpy(dtype=float)
        row = labels.iloc[feature_index].to_dict()
        row.update(
            {
                "raw_mean": float(np.nanmean(raw_values)),
                "raw_std": float(np.nanstd(raw_values)),
                "raw_min": float(np.nanmin(raw_values)),
                "raw_max": float(np.nanmax(raw_values)),
                "raw_missing_count": int(np.count_nonzero(~np.isfinite(raw_values))),
                "processed_mean": float(np.nanmean(processed_values)),
                "processed_std": float(np.nanstd(processed_values)),
                "processed_min": float(np.nanmin(processed_values)),
                "processed_max": float(np.nanmax(processed_values)),
            }
        )
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / f"{file_prefix}_feature_value_summary.csv", index=False)
    log_status(f"Saved {file_prefix}_feature_value_summary.csv")

    example_count = min(25, len(metadata_table))
    example_metadata = metadata_table.head(example_count).reset_index(drop=True)
    pd.concat([example_metadata, raw_df.head(example_count).reset_index(drop=True)], axis=1).to_csv(
        output_dir / f"{file_prefix}_example_raw_feature_vectors.csv",
        index=False,
    )
    log_status(f"Saved {file_prefix}_example_raw_feature_vectors.csv")
    pd.concat([example_metadata, processed_df.head(example_count).reset_index(drop=True)], axis=1).to_csv(
        output_dir / f"{file_prefix}_example_processed_feature_vectors.csv",
        index=False,
    )
    log_status(f"Saved {file_prefix}_example_processed_feature_vectors.csv")


def save_lda_model_diagnostics(
    *,
    output_dir: Path,
    file_prefix: str,
    lda_model: LinearDiscriminantAnalysis,
    feature_table: pd.DataFrame,
) -> None:
    labels = feature_label_table(feature_table)

    scalings = np.asarray(getattr(lda_model, "scalings_", np.empty((len(labels), 0))), dtype=float)
    if scalings.ndim == 2 and scalings.shape[0] == len(labels):
        scaling_df = labels.copy()
        for component_index in range(scalings.shape[1]):
            scaling_df[f"LD{component_index + 1}_scaling"] = scalings[:, component_index]
        component_columns = [column for column in scaling_df.columns if column.endswith("_scaling")]
        if component_columns:
            scaling_df["max_abs_ld_scaling"] = scaling_df[component_columns].abs().max(axis=1)
            scaling_df = scaling_df.sort_values("max_abs_ld_scaling", ascending=False)
        scaling_df.to_csv(output_dir / f"{file_prefix}_lda_feature_scalings.csv", index=False)
        log_status(f"Saved {file_prefix}_lda_feature_scalings.csv")

        top_df = scaling_df.head(80).copy()
        if not top_df.empty and component_columns:
            fig, ax = plt.subplots(figsize=(12, max(6, 0.18 * len(top_df))))
            plot_values = top_df["max_abs_ld_scaling"].to_numpy(dtype=float)
            plot_labels = top_df["feature_key"].astype(str).tolist()
            ax.barh(np.arange(len(top_df)), plot_values, color="#4c78a8")
            ax.set_yticks(np.arange(len(top_df)))
            ax.set_yticklabels(plot_labels, fontsize=6)
            ax.invert_yaxis()
            ax.set_xlabel("Max absolute LDA scaling")
            ax.set_title("Top LDA Feature Weights")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(output_dir / f"{file_prefix}_top_feature_weights.png", dpi=300)
            plt.close(fig)
            log_status(f"Saved {file_prefix}_top_feature_weights.png")

    coef = np.asarray(getattr(lda_model, "coef_", np.empty((0, len(labels)))), dtype=float)
    classes = np.asarray(getattr(lda_model, "classes_", []))
    if coef.ndim == 2 and coef.shape[1] == len(labels):
        coef_df = labels.copy()
        for class_index, class_label in enumerate(classes):
            coef_df[f"class_{class_label}_coef"] = coef[class_index]
        coef_df.to_csv(output_dir / f"{file_prefix}_lda_class_coefficients.csv", index=False)
        log_status(f"Saved {file_prefix}_lda_class_coefficients.csv")

    means = np.asarray(getattr(lda_model, "means_", np.empty((0, len(labels)))), dtype=float)
    if means.ndim == 2 and means.shape[1] == len(labels):
        mean_rows = []
        for class_index, class_label in enumerate(classes):
            for feature_index, feature_row in labels.iterrows():
                row = feature_row.to_dict()
                row["class_label"] = class_label
                row["class_mean"] = means[class_index, feature_index]
                mean_rows.append(row)
        pd.DataFrame(mean_rows).to_csv(output_dir / f"{file_prefix}_lda_class_feature_means.csv", index=False)
        log_status(f"Saved {file_prefix}_lda_class_feature_means.csv")

    explained = np.asarray(getattr(lda_model, "explained_variance_ratio_", []), dtype=float).ravel()
    if explained.size:
        pd.DataFrame(
            {
                "component": [f"LD{index + 1}" for index in range(explained.size)],
                "explained_variance_ratio": explained,
            }
        ).to_csv(output_dir / f"{file_prefix}_lda_explained_variance_ratio.csv", index=False)
        log_status(f"Saved {file_prefix}_lda_explained_variance_ratio.csv")


def save_feature_pca_visualization(
    *,
    output_dir: Path,
    file_prefix: str,
    metadata_table: pd.DataFrame,
    processed_population_matrix: np.ndarray,
) -> None:
    n_samples, n_features = processed_population_matrix.shape
    if n_samples < 2 or n_features < 2:
        return
    n_components = min(3, n_samples, n_features)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    projection = pca.fit_transform(processed_population_matrix)
    pca_df = pd.DataFrame(
        {
            f"PC{component_index + 1}": projection[:, component_index]
            for component_index in range(n_components)
        }
    )
    pd.concat([metadata_table.reset_index(drop=True), pca_df], axis=1).to_csv(
        output_dir / f"{file_prefix}_feature_pca_projection.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "component": [f"PC{index + 1}" for index in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    ).to_csv(output_dir / f"{file_prefix}_feature_pca_explained_variance.csv", index=False)
    log_status(f"Saved {file_prefix}_feature_pca_projection.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    hours = metadata_table["clock_hour_of_day"].to_numpy(dtype=float)
    y_values = projection[:, 1] if n_components >= 2 else np.zeros(n_samples)
    scatter = ax.scatter(
        projection[:, 0],
        y_values,
        c=hours,
        cmap=CIRCULAR_HOUR_CMAP,
        norm=CIRCULAR_HOUR_NORM,
        s=44,
        alpha=0.92,
        edgecolors="black",
        linewidths=0.35,
    )
    ax.set_title("Feature Matrix PCA Projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        boundaries=CIRCULAR_HOUR_BOUNDARIES,
        ticks=CIRCULAR_HOUR_TICKS,
        spacing="proportional",
        drawedges=True,
    )
    colorbar.set_label(CIRCULAR_HOUR_LABEL)
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_prefix}_feature_pca_2d.png", dpi=300)
    plt.close(fig)
    log_status(f"Saved {file_prefix}_feature_pca_2d.png")


# =============================================================================
# Module 9: Output Writing
# =============================================================================


def save_outputs(
    export_summary_path: Path,
    config: Config,
    date_label: str,
    feature_mode: str,
    resolved_label_column: str,
    session_table: pd.DataFrame,
    binned_population_matrix: np.ndarray,
    binned_metadata_table: pd.DataFrame,
    analysis_population_matrix: np.ndarray,
    analysis_metadata_table: pd.DataFrame,
    selected_units: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_stats: pd.DataFrame,
    verification_table: pd.DataFrame,
    lda_model: LinearDiscriminantAnalysis,
    raw_analysis_population_matrix: np.ndarray,
    projection: np.ndarray,
    decoding_results: dict[str, dict],
) -> Path:
    mode_slug = str(feature_mode).strip().lower()
    smoothing_slug = "_smooth" if config.apply_smoothing else ""
    waveform_weighting_slug = ""
    if config.lda_mode == "multi_day_hourly" and str(config.hourly_waveform_weighting).lower() != "minute":
        waveform_weighting_slug = f"_wfweight_{str(config.hourly_waveform_weighting).lower()}"
    file_stem = (
        f"lda_{date_label}_minsess_{config.min_sessions_per_unit}_{mode_slug}"
        f"{waveform_weighting_slug}{smoothing_slug}"
    )
    if config.lda_mode == "single_day_5min":
        file_stem = f"{file_stem}_single_day_5min"
    output_dir = config.output_base_dir / file_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    log_status(f"Saving output files to: {output_dir}")
    file_prefix = "lda"
    sample_output_name = "hour" if config.lda_mode == "multi_day_hourly" else "5min"
    binned_output_name = "minute" if config.lda_mode == "multi_day_hourly" else "5min_raw_bin"
    verification_output_name = "hourly" if config.lda_mode == "multi_day_hourly" else "5min"

    population_columns = feature_table["feature_column"].astype(str).tolist()
    binned_population_df = pd.DataFrame(binned_population_matrix, columns=population_columns)
    binned_population_with_metadata = pd.concat(
        [binned_metadata_table.reset_index(drop=True), binned_population_df],
        axis=1,
    )
    binned_population_with_metadata.to_csv(
        output_dir / f"{file_prefix}_{binned_output_name}_population_vectors.csv",
        index=False,
    )
    log_status(f"Saved {file_prefix}_{binned_output_name}_population_vectors.csv")
    binned_metadata_table.to_csv(output_dir / f"{file_prefix}_{binned_output_name}_metadata.csv", index=False)
    log_status(f"Saved {file_prefix}_{binned_output_name}_metadata.csv")

    sample_population_df = pd.DataFrame(analysis_population_matrix, columns=population_columns)
    sample_population_with_metadata = pd.concat(
        [analysis_metadata_table.reset_index(drop=True), sample_population_df],
        axis=1,
    )
    sample_population_with_metadata.to_csv(
        output_dir / f"{file_prefix}_{sample_output_name}_population_vectors.csv",
        index=False,
    )
    log_status(f"Saved {file_prefix}_{sample_output_name}_population_vectors.csv")
    analysis_metadata_table.to_csv(output_dir / f"{file_prefix}_{sample_output_name}_metadata.csv", index=False)
    log_status(f"Saved {file_prefix}_{sample_output_name}_metadata.csv")

    selected_units.to_csv(output_dir / f"{file_prefix}_selected_units.csv", index=False)
    log_status(f"Saved {file_prefix}_selected_units.csv")
    (
        feature_table.merge(
            feature_stats,
            on=["final_group_key", "final_unit_id", "shank_id", "local_channel_on_shank"],
            how="left",
        )
        .to_csv(output_dir / f"{file_prefix}_feature_map.csv", index=False)
    )
    log_status(f"Saved {file_prefix}_feature_map.csv")

    projection_df = pd.DataFrame(index=np.arange(len(analysis_metadata_table)))
    for dimension_index, column_name in enumerate(["LD1", "LD2", "LD3"]):
        if dimension_index < projection.shape[1]:
            projection_df[column_name] = projection[:, dimension_index]
        else:
            projection_df[column_name] = np.nan
    pd.concat([analysis_metadata_table.reset_index(drop=True), projection_df], axis=1).to_csv(
        output_dir / f"{file_prefix}_projection.csv",
        index=False,
    )
    log_status(f"Saved {file_prefix}_projection.csv")
    session_table.to_csv(output_dir / f"{file_prefix}_session_start_sources.csv", index=False)
    log_status(f"Saved {file_prefix}_session_start_sources.csv")
    verification_table.to_csv(output_dir / f"{file_prefix}_{verification_output_name}_verification.csv", index=False)
    log_status(f"Saved {file_prefix}_{verification_output_name}_verification.csv")

    save_feature_matrix_diagnostics(
        output_dir=output_dir,
        file_prefix=file_prefix,
        feature_table=feature_table,
        metadata_table=analysis_metadata_table,
        raw_population_matrix=raw_analysis_population_matrix,
        processed_population_matrix=analysis_population_matrix,
    )
    save_lda_model_diagnostics(
        output_dir=output_dir,
        file_prefix=file_prefix,
        lda_model=lda_model,
        feature_table=feature_table,
    )
    save_feature_pca_visualization(
        output_dir=output_dir,
        file_prefix=file_prefix,
        metadata_table=analysis_metadata_table,
        processed_population_matrix=analysis_population_matrix,
    )

    plot_lda_2d(projection, analysis_metadata_table, output_dir / f"{file_prefix}_2d.png")
    log_status(f"Saved {file_prefix}_2d.png")
    plot_lda_3d(projection, analysis_metadata_table, output_dir / f"{file_prefix}_3d.png")
    log_status(f"Saved {file_prefix}_3d.png")
    for cv_name, decoding_result in decoding_results.items():
        per_sample_error_df = analysis_metadata_table.reset_index(drop=True).copy()
        per_sample_error_df["true_label"] = resolve_label_series(analysis_metadata_table, config)[0].to_numpy()
        per_sample_error_df["predicted_label"] = decoding_result["predicted_labels"]
        per_sample_error_df["circular_hour_error"] = decoding_result["circular_hour_errors"]
        per_sample_error_df.to_csv(
            output_dir / f"{file_prefix}_{cv_name}_prediction_errors.csv",
            index=False,
        )
        log_status(f"Saved {file_prefix}_{cv_name}_prediction_errors.csv")
        decoding_result["per_label_circular_hour_error"].to_csv(
            output_dir / f"{file_prefix}_{cv_name}_per_label_prediction_error.csv",
            index=False,
        )
        log_status(f"Saved {file_prefix}_{cv_name}_per_label_prediction_error.csv")
        plot_confusion_matrix(
            decoding_result,
            output_dir / f"{file_prefix}_{cv_name}_confusion_matrix.png",
        )
        log_status(f"Saved {file_prefix}_{cv_name}_confusion_matrix.png")
        permutation_df = pd.DataFrame(
            {
                "permutation_index": np.arange(
                    1,
                    len(decoding_result["permutation_balanced_accuracy_distribution"]) + 1,
                    dtype=int,
                ),
                "balanced_accuracy": decoding_result["permutation_balanced_accuracy_distribution"],
            }
        )
        permutation_df.to_csv(
            output_dir / f"{file_prefix}_{cv_name}_permutation_balanced_accuracy.csv",
            index=False,
        )
        log_status(f"Saved {file_prefix}_{cv_name}_permutation_balanced_accuracy.csv")

    hourly_aggregation_grouping_keys = None
    if config.lda_mode == "multi_day_hourly":
        hourly_aggregation_grouping_keys = [
            column
            for column in ("threshold_run_name", "threshold_run_root")
            if column in analysis_metadata_table.columns
            and analysis_metadata_table[column].nunique(dropna=False) > 1
        ] + ["calendar_day", "clock_hour_of_day"]

    summary_payload = {
        "export_summary_path": str(export_summary_path),
        "lda_mode": str(config.lda_mode),
        "feature_mode": str(feature_mode),
        "label_type": resolved_label_column,
        "configured_label_type": str(config.label_type),
        "hourly_aggregation_grouping_keys": hourly_aggregation_grouping_keys,
        "bin_size_seconds": float(config.bin_size_seconds),
        "binned_bin_size_seconds": float(config.bin_size_seconds),
        "minute_bin_size_seconds": (
            float(config.bin_size_seconds)
            if config.lda_mode == "multi_day_hourly"
            else None
        ),
        "min_firing_rate_hz": float(config.min_firing_rate_hz),
        "min_minutes_per_hour": int(config.min_minutes_per_hour),
        "apply_zscore": bool(config.apply_zscore),
        "apply_smoothing": bool(config.apply_smoothing),
        "smoothing_sigma_bins": float(config.smoothing_sigma_bins),
        "hourly_waveform_weighting": str(config.hourly_waveform_weighting),
        "n_permutations": int(config.n_permutations),
        "n_binned_samples": int(binned_population_matrix.shape[0]),
        "binned_sample_output_name": binned_output_name,
        "n_minute_samples": int(binned_population_matrix.shape[0]) if config.lda_mode == "multi_day_hourly" else 0,
        "n_analysis_samples": int(analysis_population_matrix.shape[0]),
        "analysis_sample_output_name": sample_output_name,
        "n_hour_samples": int(analysis_population_matrix.shape[0]) if config.lda_mode == "multi_day_hourly" else 0,
        "n_features": int(analysis_population_matrix.shape[1]),
        "n_selected_unit_groups": int(selected_units["final_group_key"].nunique()),
        "n_sessions": int(binned_metadata_table["session_name"].nunique()),
        "n_calendar_days": int(analysis_metadata_table["calendar_day"].nunique()),
        "labels": [
            to_jsonable_scalar(value)
            for value in sorted(pd.unique(resolve_label_series(analysis_metadata_table, config)[0]).tolist())
        ],
        "decoding_results": {
            cv_name: {
                "n_splits": int(decoding_result["n_splits"]),
                "groups_used": bool(decoding_result["groups_used"]),
                "accuracy": float(decoding_result["accuracy"]),
                "balanced_accuracy": float(decoding_result["balanced_accuracy"]),
                "mean_circular_hour_error": float(decoding_result["mean_circular_hour_error"]),
                "mean_label_circular_hour_error": float(decoding_result["mean_label_circular_hour_error"]),
                "permutation_balanced_accuracy_mean": float(decoding_result["permutation_balanced_accuracy_mean"]),
                "permutation_balanced_accuracy_std": float(decoding_result["permutation_balanced_accuracy_std"]),
                "permutation_p_value": float(decoding_result["permutation_p_value"]),
            }
            for cv_name, decoding_result in decoding_results.items()
        },
        "date_label": date_label,
    }
    (output_dir / f"{file_prefix}_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    log_status(f"Saved {file_prefix}_summary.json")

    return output_dir


# =============================================================================
# Module 10: Precomputed CSV Input
# =============================================================================


def is_precomputed_population_csv(data_path: Path) -> bool:
    return Path(data_path).is_file() and Path(data_path).suffix.lower() == ".csv"


def parse_threshold_feature_column(column_name: str) -> tuple[str, str] | None:
    if "__" not in column_name:
        return None
    unit_key, feature_type = column_name.rsplit("__", 1)
    if not unit_key or not feature_type:
        return None
    return unit_key, feature_type


def build_feature_table_from_csv_columns(feature_columns: list[str]) -> pd.DataFrame:
    unit_keys = []
    for column in feature_columns:
        parsed = parse_threshold_feature_column(column)
        if parsed is None:
            continue
        unit_key, _ = parsed
        if unit_key not in unit_keys:
            unit_keys.append(unit_key)
    unit_id_lookup = {unit_key: index for index, unit_key in enumerate(unit_keys, start=1)}

    rows: list[dict] = []
    for feature_index, column in enumerate(feature_columns, start=1):
        parsed = parse_threshold_feature_column(column)
        if parsed is None:
            continue
        unit_key, feature_type = parsed
        sg_match = re.search(r"sgch(?P<sg>\d+)", unit_key)
        rows.append(
            {
                "feature_key": column,
                "final_group_key": unit_key,
                "final_unit_id": int(unit_id_lookup[unit_key]),
                "shank_id": np.nan,
                "local_channel_on_shank": safe_int(sg_match.group("sg")) if sg_match else np.nan,
                "feature_type": feature_type,
                "feature_column": f"feature_{feature_index:04d}",
                "source_column": column,
            }
        )

    feature_table = pd.DataFrame(rows)
    if feature_table.empty:
        raise RuntimeError(
            "No feature columns were found in the precomputed CSV. Expected columns like "
            "'sgch279_thr180uV__firing_rate_hz'."
        )
    return feature_table


def build_selected_units_from_csv_feature_table(
    feature_table: pd.DataFrame,
    metadata_table: pd.DataFrame,
) -> pd.DataFrame:
    session_names = metadata_table["session_name"].astype(str).drop_duplicates().tolist()
    session_name = "precomputed_csv" if not session_names else " | ".join(session_names[:3])
    if len(session_names) > 3:
        session_name = f"{session_name} | ..."

    group_table = (
        feature_table[
            ["final_group_key", "final_unit_id", "shank_id", "local_channel_on_shank"]
        ]
        .drop_duplicates()
        .sort_values(["final_unit_id", "final_group_key"], na_position="last")
        .reset_index(drop=True)
    )
    rows = []
    for group_row in group_table.itertuples(index=False):
        rows.append(
            {
                "session_key": "precomputed_csv",
                "final_group_key": str(group_row.final_group_key),
                "final_unit_id": safe_int(group_row.final_unit_id),
                "session_name": session_name,
                "session_index": 1,
                "unit_id": safe_int(group_row.final_unit_id),
                "output_folder": "",
                "shank_id": safe_int(group_row.shank_id),
                "local_channel_on_shank": safe_int(group_row.local_channel_on_shank),
            }
        )
    return pd.DataFrame(rows)


def build_session_table_from_csv_metadata(metadata_table: pd.DataFrame) -> pd.DataFrame:
    session_columns = [
        column
        for column in [
            "session_id",
            "session_key",
            "session_name",
            "session_name_normalized",
            "session_index",
            "session_start_datetime",
        ]
        if column in metadata_table.columns
    ]
    session_table = (
        metadata_table[session_columns]
        .drop_duplicates()
        .sort_values(["session_index", "session_name"], na_position="last")
        .reset_index(drop=True)
    )
    if "output_folder" not in session_table.columns:
        session_table["output_folder"] = ""
    if "resolved_output_folder" not in session_table.columns:
        session_table["resolved_output_folder"] = ""
    return session_table


def load_precomputed_population_csv(
    csv_path: Path,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log_status(f"Loading precomputed population CSV: {csv_path}")
    population_df = pd.read_csv(csv_path)
    required_columns = {
        "final_sample_key",
        "session_name",
        "minute_start_datetime",
        "clock_hour_of_day",
        "calendar_day",
    }
    missing_columns = sorted(required_columns - set(population_df.columns))
    if missing_columns:
        raise KeyError(
            "Precomputed population CSV is missing required metadata column(s): "
            f"{missing_columns}"
        )

    feature_columns = [
        column
        for column in population_df.columns
        if column not in CSV_METADATA_COLUMNS and parse_threshold_feature_column(column) is not None
    ]
    feature_table = build_feature_table_from_csv_columns(feature_columns)
    metadata_columns = [column for column in population_df.columns if column not in feature_columns]
    metadata_table = population_df[metadata_columns].copy()
    for column in ["clock_hour_of_day", "clock_minute_of_hour", "minute_bin_index", "session_id", "session_index"]:
        if column in metadata_table.columns:
            metadata_table[column] = pd.to_numeric(metadata_table[column], errors="coerce")

    population_matrix = population_df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    selected_units = build_selected_units_from_csv_feature_table(feature_table, metadata_table)
    session_table = build_session_table_from_csv_metadata(metadata_table)
    log_status(
        f"Loaded precomputed matrix with {population_matrix.shape[0]} samples x "
        f"{population_matrix.shape[1]} features from {len(selected_units)} unit groups"
    )
    return population_matrix, metadata_table, feature_table, selected_units, session_table


def run_pipeline_from_precomputed_csv(config: Config) -> list[Path]:
    if config.data_path is None:
        raise ValueError("Config.data_path cannot be None when running the pipeline.")

    config = apply_lda_mode_defaults(config)
    csv_path = Path(config.data_path)
    binned_population_matrix, binned_metadata_table, feature_table, selected_units, session_table = (
        load_precomputed_population_csv(csv_path)
    )
    date_label = build_date_label_from_session_table(session_table)
    if date_label == datetime.now().strftime("%y%m%d"):
        parsed_dates = pd.to_datetime(binned_metadata_table["calendar_day"], errors="coerce")
        valid_dates = parsed_dates.dropna()
        if not valid_dates.empty:
            unique_days = sorted(valid_dates.dt.strftime("%y%m%d").unique().tolist())
            date_label = unique_days[0] if len(unique_days) == 1 else f"{unique_days[0]}_to_{unique_days[-1]}"

    log_status(
        f"Using precomputed CSV input for LDA_MODE={config.lda_mode}; date label {date_label}"
    )
    log_status(
        "Applying MIN_FIRING_RATE_HZ using mean bin firing rate "
        "(spikes in bin divided by bin duration, averaged across bins)"
    )
    binned_population_matrix, selected_units, feature_table, feature_stats = filter_unit_groups_by_binned_firing_rate(
        population_matrix=binned_population_matrix,
        selected_units=selected_units,
        feature_table=feature_table,
        config=config,
    )

    if config.lda_mode == "multi_day_hourly":
        analysis_population_matrix, analysis_metadata_table = aggregate_minutes_to_hourly_samples(
            minute_population_matrix=binned_population_matrix,
            minute_metadata_table=binned_metadata_table,
            feature_table=feature_table,
            waveform_weighting=config.hourly_waveform_weighting,
        )
        print_and_build_clock_hour_verification(analysis_metadata_table)
        print_clock_hour_sample_pivot(
            analysis_metadata_table,
            title="Pivot table for hourly samples before label filtering (calendar_day x clock_hour_of_day)",
        )
        analysis_population_matrix, analysis_metadata_table = filter_hourly_samples_by_min_minutes(
            population_matrix=analysis_population_matrix,
            metadata_table=analysis_metadata_table,
            config=config,
        )
    else:
        analysis_population_matrix, analysis_metadata_table = prepare_single_day_5min_samples(
            population_matrix=binned_population_matrix,
            metadata_table=binned_metadata_table,
            config=config,
        )
        binned_population_matrix = analysis_population_matrix
        binned_metadata_table = analysis_metadata_table

    analysis_population_matrix, analysis_metadata_table = filter_labels_for_lda(
        population_matrix=analysis_population_matrix,
        metadata_table=analysis_metadata_table,
        config=config,
    )
    print_clock_hour_sample_pivot(
        analysis_metadata_table,
        title="Pivot table for final LDA samples (calendar_day x clock_hour_of_day)",
    )
    if config.lda_mode == "multi_day_hourly":
        verification_table = print_and_build_clock_hour_verification(analysis_metadata_table)
    else:
        verification_table = analysis_metadata_table[
            [
                "final_sample_id",
                "final_sample_key",
                "calendar_day",
                "clock_hour_of_day",
                "clock_minute_of_hour",
                "sample_start_datetime",
                "sample_end_datetime",
                "sample_duration_s",
            ]
        ].copy()

    labels, resolved_label_column = resolve_label_series(analysis_metadata_table, config)
    labels = labels.to_numpy()
    print_lda_label_vector(analysis_metadata_table, labels=labels, label_column=resolved_label_column)
    config.label_type = resolved_label_column

    output_dirs: list[Path] = []
    for feature_mode in normalize_feature_modes(config.feature_modes):
        log_status(f"Preparing feature mode: {feature_mode}")
        mode_binned_population_matrix, mode_feature_table = subset_features_for_mode(
            population_matrix=binned_population_matrix,
            feature_table=feature_table,
            feature_mode=feature_mode,
        )
        mode_analysis_population_matrix, _ = subset_features_for_mode(
            population_matrix=analysis_population_matrix,
            feature_table=feature_table,
            feature_mode=feature_mode,
        )
        raw_mode_analysis_population_matrix = mode_analysis_population_matrix.copy()
        mode_analysis_population_matrix = fill_missing_feature_values(mode_analysis_population_matrix)
        if config.apply_zscore:
            log_status(f"Applying z-scoring across features for mode {feature_mode}")
            mode_analysis_population_matrix = zscore_population_matrix(mode_analysis_population_matrix)

        log_status(f"Fitting LDA model for mode {feature_mode}")
        lda_model, projection = fit_lda(
            population_matrix=mode_analysis_population_matrix,
            labels=labels,
        )
        log_status(
            f"Fitted LDA for mode {feature_mode} with {projection.shape[1]} discriminant dimension(s)"
        )

        log_status(f"Running cross-validated decoding for mode {feature_mode}")
        decoding_results = evaluate_decoding(
            population_matrix=mode_analysis_population_matrix,
            labels=labels,
            metadata_table=analysis_metadata_table,
            resolved_label_column=resolved_label_column,
            config=config,
        )
        for cv_name, decoding_result in decoding_results.items():
            log_status(
                f"{feature_mode} {cv_name}: accuracy={decoding_result['accuracy']:.3f}, "
                f"balanced_accuracy={decoding_result['balanced_accuracy']:.3f}, "
                f"mean_hour_error={decoding_result['mean_label_circular_hour_error']:.2f}h, "
                f"permutation_p={decoding_result['permutation_p_value']:.4f}"
            )

        output_dir = save_outputs(
            export_summary_path=csv_path,
            config=config,
            date_label=date_label,
            feature_mode=feature_mode,
            resolved_label_column=resolved_label_column,
            session_table=session_table,
            binned_population_matrix=mode_binned_population_matrix,
            binned_metadata_table=binned_metadata_table,
            analysis_population_matrix=mode_analysis_population_matrix,
            analysis_metadata_table=analysis_metadata_table,
            selected_units=selected_units,
            feature_table=mode_feature_table,
            feature_stats=feature_stats,
            verification_table=verification_table,
            lda_model=lda_model,
            raw_analysis_population_matrix=raw_mode_analysis_population_matrix,
            projection=projection,
            decoding_results=decoding_results,
        )
        output_dirs.append(output_dir)

    return output_dirs


# =============================================================================
# Module 11: Pipeline Entry Point
# =============================================================================


def run_pipeline(config: Config) -> list[Path]:
    if config.data_path is None:
        raise ValueError("Config.data_path cannot be None when running the pipeline.")

    if is_precomputed_population_csv(Path(config.data_path)):
        return run_pipeline_from_precomputed_csv(config)

    config = apply_lda_mode_defaults(config)
    log_status(
        f"Running LDA_MODE={config.lda_mode} with BIN_SIZE_SECONDS={config.bin_size_seconds:.0f}"
    )

    export_summary_path = resolve_export_summary_path(config.data_path)
    log_status(f"Loading alignment export: {export_summary_path}")
    export_payload = load_export_summary(export_summary_path)

    session_table = build_session_table(export_payload=export_payload, config=config)
    log_status(f"Resolved {len(session_table)} sessions")
    session_table = filter_session_table_for_lda_mode(session_table, config)
    log_status(f"Kept {len(session_table)} sessions for LDA_MODE={config.lda_mode}")
    date_label = build_date_label_from_session_table(session_table)
    log_status(f"Using date label for outputs: {date_label}")

    analyzers, resolved_output_folders = load_session_analyzers(
        session_table=session_table,
        config=config,
    )
    session_table = session_table.copy()
    session_table["resolved_output_folder"] = session_table["session_key"].map(resolved_output_folders)
    selected_units = select_good_unit_groups(
        export_payload=export_payload,
        config=config,
        analyzers=analyzers,
    )
    log_status(
        f"Selected {selected_units['final_group_key'].nunique()} aligned unit groups "
        f"across {selected_units['session_name'].nunique()} sessions"
    )

    binned_population_matrix, binned_metadata_table, feature_table = build_population_vectors(
        selected_units=selected_units,
        session_table=session_table,
        analyzers=analyzers,
        config=config,
    )
    log_status(
        f"Built binned population matrix with shape {binned_population_matrix.shape[0]} bins x "
        f"{binned_population_matrix.shape[1]} features"
    )
    log_status(
        "Applying MIN_FIRING_RATE_HZ using mean bin firing rate "
        "(spikes in bin divided by bin duration, averaged across bins)"
    )
    binned_population_matrix, selected_units, feature_table, feature_stats = filter_unit_groups_by_binned_firing_rate(
        population_matrix=binned_population_matrix,
        selected_units=selected_units,
        feature_table=feature_table,
        config=config,
    )
    if config.lda_mode == "multi_day_hourly":
        analysis_population_matrix, analysis_metadata_table = aggregate_minutes_to_hourly_samples(
            minute_population_matrix=binned_population_matrix,
            minute_metadata_table=binned_metadata_table,
            feature_table=feature_table,
            waveform_weighting=config.hourly_waveform_weighting,
        )
        print_and_build_clock_hour_verification(analysis_metadata_table)
        print_clock_hour_sample_pivot(
            analysis_metadata_table,
            title="Pivot table for hourly samples before label filtering (calendar_day x clock_hour_of_day)",
        )
        analysis_population_matrix, analysis_metadata_table = filter_hourly_samples_by_min_minutes(
            population_matrix=analysis_population_matrix,
            metadata_table=analysis_metadata_table,
            config=config,
        )
        log_status(
            f"Kept {analysis_population_matrix.shape[0]} hourly samples after applying "
            f"MIN_MINUTES_PER_HOUR={config.min_minutes_per_hour}"
        )
    else:
        analysis_population_matrix, analysis_metadata_table = prepare_single_day_5min_samples(
            population_matrix=binned_population_matrix,
            metadata_table=binned_metadata_table,
            config=config,
        )
        binned_population_matrix = analysis_population_matrix
        binned_metadata_table = analysis_metadata_table
        print_clock_hour_sample_pivot(
            analysis_metadata_table,
            title="Pivot table for 5-minute samples before label filtering (calendar_day x clock_hour_of_day)",
        )

    analysis_population_matrix, analysis_metadata_table = filter_labels_for_lda(
        population_matrix=analysis_population_matrix,
        metadata_table=analysis_metadata_table,
        config=config,
    )
    print_clock_hour_sample_pivot(
        analysis_metadata_table,
        title="Pivot table for final LDA samples (calendar_day x clock_hour_of_day)",
    )
    if config.lda_mode == "multi_day_hourly":
        verification_table = print_and_build_clock_hour_verification(analysis_metadata_table)
    else:
        verification_table = analysis_metadata_table[
            [
                "final_sample_id",
                "final_sample_key",
                "calendar_day",
                "clock_hour_of_day",
                "clock_minute_of_hour",
                "sample_start_datetime",
                "sample_end_datetime",
                "sample_duration_s",
            ]
        ].copy()
    labels, resolved_label_column = resolve_label_series(analysis_metadata_table, config)
    log_status(
        f"After label filtering: {analysis_population_matrix.shape[0]} LDA samples across "
        f"{len(pd.unique(labels))} '{resolved_label_column}' labels"
    )

    labels = labels.to_numpy()
    print_lda_label_vector(analysis_metadata_table, labels=labels, label_column=resolved_label_column)
    config.label_type = resolved_label_column

    output_dirs: list[Path] = []
    for feature_mode in normalize_feature_modes(config.feature_modes):
        log_status(f"Preparing feature mode: {feature_mode}")
        mode_binned_population_matrix, mode_feature_table = subset_features_for_mode(
            population_matrix=binned_population_matrix,
            feature_table=feature_table,
            feature_mode=feature_mode,
        )
        mode_analysis_population_matrix, _ = subset_features_for_mode(
            population_matrix=analysis_population_matrix,
            feature_table=feature_table,
            feature_mode=feature_mode,
        )
        raw_mode_analysis_population_matrix = mode_analysis_population_matrix.copy()
        mode_analysis_population_matrix = fill_missing_feature_values(mode_analysis_population_matrix)
        if config.apply_zscore:
            log_status(f"Applying z-scoring across features for mode {feature_mode}")
            mode_analysis_population_matrix = zscore_population_matrix(mode_analysis_population_matrix)

        log_status(f"Fitting LDA model for mode {feature_mode}")
        lda_model, projection = fit_lda(
            population_matrix=mode_analysis_population_matrix,
            labels=labels,
        )
        log_status(
            f"Fitted LDA for mode {feature_mode} with {projection.shape[1]} discriminant dimension(s)"
        )

        log_status(f"Running cross-validated decoding for mode {feature_mode}")
        decoding_results = evaluate_decoding(
            population_matrix=mode_analysis_population_matrix,
            labels=labels,
            metadata_table=analysis_metadata_table,
            resolved_label_column=resolved_label_column,
            config=config,
        )
        for cv_name, decoding_result in decoding_results.items():
            log_status(
                f"{feature_mode} {cv_name}: accuracy={decoding_result['accuracy']:.3f}, "
                f"balanced_accuracy={decoding_result['balanced_accuracy']:.3f}, "
                f"mean_hour_error={decoding_result['mean_label_circular_hour_error']:.2f}h, "
                f"permutation_p={decoding_result['permutation_p_value']:.4f}"
            )

        output_dir = save_outputs(
            export_summary_path=export_summary_path,
            config=config,
            date_label=date_label,
            feature_mode=feature_mode,
            resolved_label_column=resolved_label_column,
            session_table=session_table,
            binned_population_matrix=mode_binned_population_matrix,
            binned_metadata_table=binned_metadata_table,
            analysis_population_matrix=mode_analysis_population_matrix,
            analysis_metadata_table=analysis_metadata_table,
            selected_units=selected_units,
            feature_table=mode_feature_table,
            feature_stats=feature_stats,
            verification_table=verification_table,
            lda_model=lda_model,
            raw_analysis_population_matrix=raw_mode_analysis_population_matrix,
            projection=projection,
            decoding_results=decoding_results,
        )
        output_dirs.append(output_dir)

    return output_dirs


def main() -> None:
    config = Config()
    config.data_path = prompt_for_data_path(config.data_path)
    log_status(
        "LDA supports LDA_MODE='multi_day_hourly' for day x hour samples and "
        "LDA_MODE='single_day_5min' for within-day 5-minute samples. Multi-day "
        "analysis is supported if all sessions were aligned together in the same "
        "export summary."
    )
    output_dirs = run_pipeline(config)
    for output_dir in output_dirs:
        log_status(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
