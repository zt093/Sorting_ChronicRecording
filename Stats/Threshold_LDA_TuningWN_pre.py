from __future__ import annotations

"""
Threshold-channel downstream analysis pipeline.

Input is a `threshold_crossings_*` run folder produced by
`Sorting_Check/Threshold_channel.py`. Each `(sg_ch, threshold)` detector pair is
treated as one unit, matching the threshold-unit interpretation used by
`LDA_weinan.py`.

The script materializes a minute-level population CSV, then runs:

1. LDA on that precomputed threshold population matrix.
2. Tuning_Weinan daily-cycle and polar plots directly from the threshold run.
3. Lightweight presentation summaries for the threshold units used.

This intentionally skips the alignment stages from Auto_align_LDA_pre_tuning.py:
there are no cross-session aligned sorted units here; the detector pair itself is
the stable unit identity.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import LDA_weinan as lda_threshold
import Tuning_Weinan as tuning_weinan


_ORIGINAL_TUNING_WEINAN_LOAD_SERIES_FROM_PAIR_DIR = tuning_weinan.load_series_from_pair_dir

DEFAULT_LDA_FEATURE_MODES = ("FR_ONLY", "FR_AMP", "FR_CV2", "FR_PEAK_TO_TROUGH", "MULTI_FEATURE")
DEFAULT_OUTPUT_SUBDIR = "threshold_LDA_TuningWN_pre"
ANALYSIS_MODES = ("baseline", "sham_drug_markers")
PHASE_MARKERS = {"baseline": "o", "sham": "s", "drug": "^"}
PHASE_LABELS = {"baseline": "baseline", "sham": "sham", "drug": "drug"}
CIRCULAR_HOUR_CMAP = plt.get_cmap("twilight_shifted", 24)
CIRCULAR_HOUR_BOUNDARIES = np.arange(-0.5, 24.5, 1.0)
CIRCULAR_HOUR_NORM = BoundaryNorm(CIRCULAR_HOUR_BOUNDARIES, CIRCULAR_HOUR_CMAP.N)


@dataclass
class PipelineConfig:
    run_roots: tuple[Path, ...]
    output_dir: Path | None = None
    analysis_mode: str = "baseline"
    lda_feature_modes: tuple[str, ...] = DEFAULT_LDA_FEATURE_MODES
    min_firing_rate_hz: float = 0.0
    min_minutes_per_hour: int = 1
    min_bins_per_label: int = 2
    cv_n_splits: int = 5
    n_permutations: int = 20
    apply_zscore: bool = True
    skip_lda: bool = False
    skip_tuning_weinan: bool = False
    skip_presentation: bool = False
    reuse_population_csv: bool = True
    tuning_weinan_only_polar: bool = True
    sham_sessions: tuple[str, ...] = ()
    drug_sessions: tuple[str, ...] = ()
    confirm_sham_drug: bool = False


def parse_yes_no(raw_text: str, *, default: bool) -> bool:
    cleaned = str(raw_text or "").strip().lower()
    if not cleaned:
        return bool(default)
    if cleaned in {"y", "yes", "true", "1"}:
        return True
    if cleaned in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"Expected yes/no, got: {raw_text!r}")


def prompt_yes_no(prompt_text: str, *, default: bool) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    return parse_yes_no(input(prompt_text + suffix), default=default)


@contextmanager
def patched_argv(argv: list[str]):
    original_argv = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = original_argv


_SCRIPT_START_TIME = time.perf_counter()


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {remaining_seconds:05.2f}s"
    if minutes > 0:
        return f"{minutes:d}m {remaining_seconds:05.2f}s"
    return f"{remaining_seconds:.2f}s"


def log_status(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = format_duration(time.perf_counter() - _SCRIPT_START_TIME)
    print(f"[Threshold_LDA_TuningWN_pre {timestamp} +{elapsed}] {message}", flush=True)


@contextmanager
def timed_stage(stage_name: str, timings: list[dict] | None = None):
    stage_start = time.perf_counter()
    started_at = datetime.now().isoformat(sep=" ", timespec="seconds")
    log_status(f"Starting {stage_name}")
    try:
        yield
    except Exception:
        elapsed_seconds = time.perf_counter() - stage_start
        if timings is not None:
            timings.append(
                {
                    "stage": stage_name,
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
                    "elapsed_seconds": float(elapsed_seconds),
                }
            )
        log_status(f"Failed {stage_name} after {format_duration(elapsed_seconds)}")
        raise
    else:
        elapsed_seconds = time.perf_counter() - stage_start
        if timings is not None:
            timings.append(
                {
                    "stage": stage_name,
                    "status": "completed",
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
                    "elapsed_seconds": float(elapsed_seconds),
                }
            )
        log_status(f"Finished {stage_name} in {format_duration(elapsed_seconds)}")


def print_runtime_summary(timings: list[dict], total_elapsed_seconds: float) -> None:
    log_status("Runtime summary:")
    for timing in timings:
        log_status(
            f"  {timing['stage']}: {format_duration(float(timing['elapsed_seconds']))} "
            f"({timing.get('status', 'completed')})"
        )
    log_status(f"  Total: {format_duration(total_elapsed_seconds)}")


def safe_slug(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "value"


def safe_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    except Exception:
        return None


def safe_int(value) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def parse_recording_start_datetime(recording_name: object) -> datetime | None:
    return tuning_weinan.parse_recording_start_datetime_from_name(str(recording_name or ""))


def threshold_unit_key(pair: tuning_weinan.PairId) -> str:
    return pair.folder_tag()


def parse_threshold_pair_folder_name(folder_name: str) -> tuning_weinan.PairId | None:
    match = re.match(r"^sgch(?P<sg>\d+)_thr(?P<thr>.+)uV$", folder_name)
    if match is None:
        return None
    threshold_text = match.group("thr").replace("p", ".")
    threshold_min_text = threshold_text.split("to", 1)[0]
    try:
        threshold_value = float(threshold_min_text)
        if "to" in threshold_text:
            threshold_max_text = threshold_text.split("to", 1)[1]
            threshold_value += float(threshold_max_text) / 1_000_000.0
        return tuning_weinan.PairId(
            sg_ch=int(match.group("sg")),
            threshold_uv=threshold_value,
        )
    except ValueError:
        return None


def discover_threshold_pair_meta(run_root: Path) -> list[tuple[tuning_weinan.PairId, Path]]:
    candidate_dirs = [Path(run_root), *Path(run_root).rglob("*")]
    pair_meta = []
    for path in candidate_dirs:
        relative_parts = set(path.relative_to(run_root).parts) if path != Path(run_root) else set()
        if "polar_time_of_day_units" in relative_parts:
            continue
        if not path.is_dir() or not path.name.startswith("sgch") or "_thr" not in path.name:
            continue
        pair = parse_threshold_pair_folder_name(path.name)
        if pair is not None:
            pair_meta.append((pair, path))
    pair_meta.sort(key=lambda item: item[0].sort_key())
    return pair_meta


def threshold_unit_key_from_dir(pair: tuning_weinan.PairId, pair_dir: Path) -> str:
    # Keep threshold ranges as distinct unit identities when Threshold_channel.py
    # wrote folders such as sgch12_thr200to300uV.
    return str(pair_dir.name) if pair_dir.name.startswith("sgch") else pair.folder_tag()


def threshold_min_from_unit_key(unit_key: str, fallback: float) -> float:
    match = re.search(r"_thr(?P<thr>.+)uV", str(unit_key))
    if match is None:
        return float(fallback)
    threshold_text = match.group("thr").replace("p", ".")
    threshold_min_text = threshold_text.split("to", 1)[0]
    try:
        return float(threshold_min_text)
    except ValueError:
        return float(fallback)


def threshold_feature_columns(unit_key: str) -> dict[str, str]:
    return {
        "firing_rate_hz": f"{unit_key}__firing_rate_hz",
        "average_amplitude_uv": f"{unit_key}__average_amplitude_uv",
        "cv2": f"{unit_key}__cv2",
        "peak_to_trough_ms": f"{unit_key}__peak_to_trough_ms",
    }


def default_output_dir(run_roots: tuple[Path, ...]) -> Path:
    if len(run_roots) == 1:
        return Path(run_roots[0]) / DEFAULT_OUTPUT_SUBDIR
    try:
        common_root = Path(os.path.commonpath([str(path.resolve()) for path in run_roots]))
    except Exception:
        common_root = Path.cwd()
    if common_root.is_file():
        common_root = common_root.parent
    if not common_root.exists() or not common_root.is_dir():
        common_root = Path(run_roots[0]).resolve().parent
    return common_root / DEFAULT_OUTPUT_SUBDIR


def resolve_output_dir(config: PipelineConfig) -> Path:
    output_dir = Path(config.output_dir) if config.output_dir is not None else default_output_dir(config.run_roots)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_run_config(run_root: Path) -> dict:
    config_path = Path(run_root) / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def minute_summary_paths_for_pair(pair_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(pair_dir.rglob("*_minute_summary.csv")):
        if path.name == "tuning_weinan_units_used_summary.csv":
            continue
        paths.append(path)
    return paths


def normalize_minute_summary_table(csv_path: Path) -> pd.DataFrame:
    table = pd.read_csv(csv_path)
    required = {"recording_name", "minute_index", "time_start_sec", "duration_sec"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"{csv_path} is missing minute summary columns: {missing}")
    return table


def build_threshold_population_csv(run_roots: tuple[Path, ...], output_dir: Path, *, force: bool = False) -> Path:
    population_csv = output_dir / "threshold_population_minute_features.csv"
    manifest_json = output_dir / "threshold_population_manifest.json"
    if population_csv.exists() and not force:
        reusable, reason = is_population_csv_reusable(population_csv, manifest_json, run_roots)
        if reusable:
            log_status(f"Reusing threshold population CSV: {population_csv}")
            return population_csv
        log_status(
            "Existing threshold population CSV does not match the requested input folders; "
            f"rebuilding it now ({reason})."
        )

    all_pair_meta: list[tuple[Path, tuning_weinan.PairId, Path]] = []
    for run_root in run_roots:
        pair_meta = discover_threshold_pair_meta(run_root)
        if not pair_meta:
            log_status(f"No sgch*_thr*uV threshold-pair folders found under: {run_root}")
            continue
        for pair, pair_dir in pair_meta:
            all_pair_meta.append((run_root, pair, pair_dir))
    if not all_pair_meta:
        raise RuntimeError(
            "No sgch*_thr*uV threshold-pair folders were found under any input folder: "
            f"{[str(path) for path in run_roots]}"
        )

    run_order_lookup = {Path(run_root).resolve(): index for index, run_root in enumerate(run_roots, start=1)}
    session_ordinal_lookup_by_run: dict[Path, dict[str, int]] = {}
    for run_root in run_roots:
        run_config = read_run_config(run_root)
        lookup: dict[str, int] = {}
        recording_files = run_config.get("recording_files") or []
        for index, recording_file in enumerate(recording_files, start=1):
            lookup[Path(str(recording_file)).name] = index
        session_ordinal_lookup_by_run[Path(run_root).resolve()] = lookup

    sample_rows: dict[str, dict] = {}
    unit_rows_by_key: dict[str, dict] = {}
    skipped_rows = 0

    unit_keys = []
    for _, pair, pair_dir in all_pair_meta:
        unit_key = threshold_unit_key_from_dir(pair, pair_dir)
        if unit_key not in unit_keys:
            unit_keys.append(unit_key)
    unit_id_lookup = {unit_key: index for index, unit_key in enumerate(unit_keys, start=1)}

    total_pair_dirs = len(all_pair_meta)
    for pair_position, (run_root, pair, pair_dir) in enumerate(all_pair_meta, start=1):
        run_root_resolved = Path(run_root).resolve()
        run_order = int(run_order_lookup[run_root_resolved])
        run_tag = f"run{run_order:03d}_{safe_slug(Path(run_root).name)}"
        session_ordinal_lookup = session_ordinal_lookup_by_run.get(run_root_resolved, {})
        unit_key = threshold_unit_key_from_dir(pair, pair_dir)
        feature_columns = threshold_feature_columns(unit_key)
        unit_rows_by_key.setdefault(
            unit_key,
            {
                "final_group_key": unit_key,
                "final_unit_id": unit_id_lookup[unit_key],
                "sg_ch": int(pair.sg_ch),
                "threshold_uv": threshold_min_from_unit_key(unit_key, float(pair.threshold_uv)),
                "input_run_count": 0,
                "input_runs": [],
                "pair_dirs": [],
            },
        )
        unit_rows_by_key[unit_key]["input_run_count"] += 1
        unit_rows_by_key[unit_key]["input_runs"].append(str(run_root_resolved))
        unit_rows_by_key[unit_key]["pair_dirs"].append(str(pair_dir))
        summary_paths = minute_summary_paths_for_pair(pair_dir)
        if not summary_paths:
            log_status(f"No minute summary CSVs for {unit_key}; falling back to chunk NPZ is not used for LDA CSV.")
            continue
        if pair_position == 1 or pair_position % 25 == 0 or pair_position == total_pair_dirs:
            log_status(
                f"Materializing threshold population: pair {pair_position} / {total_pair_dirs} "
                f"({unit_key}), samples so far={len(sample_rows)}"
            )

        for summary_path in summary_paths:
            table = normalize_minute_summary_table(summary_path)
            for row in table.itertuples(index=False):
                row_dict = row._asdict()
                recording_name = str(row_dict["recording_name"])
                start_dt = parse_recording_start_datetime(recording_name)
                if start_dt is None:
                    skipped_rows += 1
                    continue

                minute_index = int(row_dict["minute_index"])
                minute_start_sec = float(row_dict["time_start_sec"])
                duration_sec = float(row_dict["duration_sec"])
                minute_end_sec = minute_start_sec + duration_sec
                minute_start_dt = start_dt + timedelta(seconds=minute_start_sec)
                minute_end_dt = start_dt + timedelta(seconds=minute_end_sec)
                local_session_ordinal = int(row_dict.get("session_ordinal") or session_ordinal_lookup.get(recording_name, 1))
                session_ordinal = (run_order - 1) * 100000 + local_session_ordinal
                session_key = f"{run_tag}::{recording_name}"
                sample_key = f"{run_tag}__{safe_slug(recording_name)}__minute_{minute_index:06d}"

                sample = sample_rows.setdefault(
                    sample_key,
                    {
                        "final_sample_id": len(sample_rows) + 1,
                        "final_sample_key": sample_key,
                        "session_id": session_ordinal,
                        "session_key": session_key,
                        "session_name": recording_name,
                        "session_name_normalized": safe_slug(session_key),
                        "session_index": session_ordinal,
                        "session_start_datetime": start_dt.isoformat(sep=" "),
                        "minute_bin_index": minute_index,
                        "minute_start_sec": minute_start_sec,
                        "minute_end_sec": minute_end_sec,
                        "minute_center_s": minute_start_sec + duration_sec / 2.0,
                        "session_duration_s": np.nan,
                        "minute_start_datetime": minute_start_dt.isoformat(sep=" "),
                        "minute_end_datetime": minute_end_dt.isoformat(sep=" "),
                        "clock_hour_of_day": int(minute_start_dt.hour),
                        "clock_minute_of_hour": int(minute_start_dt.minute),
                        "calendar_day": minute_start_dt.date().isoformat(),
                        "rec_file": recording_name,
                        "threshold_run_root": str(run_root_resolved),
                        "threshold_run_name": Path(run_root).name,
                    },
                )
                firing_rate_hz = safe_float(row_dict.get("firing_rate_hz"))
                sample[feature_columns["firing_rate_hz"]] = 0.0 if firing_rate_hz is None else firing_rate_hz
                sample[feature_columns["average_amplitude_uv"]] = safe_float(row_dict.get("amplitude_ptp_uv"))
                sample[feature_columns["cv2"]] = safe_float(row_dict.get("cv2"))
                sample[feature_columns["peak_to_trough_ms"]] = safe_float(row_dict.get("peak_to_trough_ms"))

    if not sample_rows:
        raise RuntimeError(
            "No threshold minute samples were materialized. Check that Threshold_channel.py "
            "minute summary CSVs contain Chronic_Rec_YYYYMMDD_HHMMSS recording names."
        )

    feature_order: list[str] = []
    for unit_key in unit_keys:
        feature_order.extend(
            threshold_feature_columns(unit_key).values()
        )
    metadata_order = [
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
    ]
    population_df = pd.DataFrame(sample_rows.values())
    for column in feature_order:
        if column not in population_df.columns:
            population_df[column] = np.nan
    population_df = population_df.sort_values(["minute_start_datetime", "session_index", "minute_bin_index"])
    population_df["final_sample_id"] = np.arange(1, len(population_df) + 1, dtype=int)
    population_df = population_df[metadata_order + feature_order]
    population_df.to_csv(population_csv, index=False)

    unit_rows = []
    for row in unit_rows_by_key.values():
        normalized = dict(row)
        normalized["input_runs"] = " | ".join(normalized["input_runs"])
        normalized["pair_dirs"] = " | ".join(normalized["pair_dirs"])
        unit_rows.append(normalized)
    unit_table = pd.DataFrame(unit_rows).sort_values(["sg_ch", "threshold_uv"], na_position="last")
    unit_table.to_csv(output_dir / "threshold_units_used_summary.csv", index=False)
    manifest_json.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
                "input_run_roots": [str(Path(run_root).resolve()) for run_root in run_roots],
                "population_csv": str(population_csv.resolve()),
                "n_threshold_units": int(len(unit_table)),
                "n_minute_samples": int(len(population_df)),
                "n_recordings": int(population_df["session_key"].nunique()),
                "skipped_rows_without_recording_datetime": int(skipped_rows),
                "threshold_units": unit_table.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log_status(
        f"Saved threshold population CSV with {len(population_df)} minute samples "
        f"and {len(unit_table)} threshold units: {population_csv}"
    )
    return population_csv


def is_population_csv_reusable(
    population_csv: Path,
    manifest_json: Path,
    run_roots: tuple[Path, ...],
) -> tuple[bool, str]:
    if not manifest_json.exists():
        return False, "manifest is missing"
    try:
        manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"manifest could not be read: {exc}"

    requested_roots = [str(Path(path).resolve()) for path in run_roots]
    manifest_roots = [str(Path(path).resolve()) for path in manifest.get("input_run_roots", [])]
    if manifest_roots != requested_roots:
        return False, "manifest input_run_roots differ from this run"
    if str(Path(manifest.get("population_csv", "")).resolve()) != str(Path(population_csv).resolve()):
        return False, "manifest points to a different population CSV"
    return True, "manifest matches"


def run_lda(population_csv: Path, output_dir: Path, config: PipelineConfig) -> list[Path]:
    lda_config = lda_threshold.Config()
    lda_config.data_path = population_csv
    lda_config.output_base_dir = output_dir / "LDA_threshold"
    lda_config.lda_mode = "multi_day_hourly"
    lda_config.label_type = "clock_hour_of_day"
    lda_config.feature_modes = tuple(config.lda_feature_modes)
    lda_config.min_firing_rate_hz = float(config.min_firing_rate_hz)
    lda_config.min_sessions_per_unit = 1
    lda_config.min_minutes_per_hour = int(config.min_minutes_per_hour)
    lda_config.min_bins_per_label = int(config.min_bins_per_label)
    lda_config.cv_n_splits = int(config.cv_n_splits)
    lda_config.n_permutations = int(config.n_permutations)
    lda_config.apply_zscore = bool(config.apply_zscore)
    lda_config.apply_smoothing = False
    log_status(
        "Starting threshold LDA from precomputed 1-minute population CSV; "
        "samples are aggregated to real clock-hour bins by calendar_day x clock_hour_of_day."
    )
    return [Path(path) for path in lda_threshold.run_pipeline(lda_config)]


def threshold_tuning_weinan_output_files(output_root: Path) -> tuple[list[str], list[str]]:
    root = Path(output_root)
    output_files = sorted(
        str(path)
        for pattern in (
            "master_peak2peak_and_firingRate*.png",
            "master_peak2peak_and_firingRate*.npz",
            "tuning_weinan_units_used_summary.csv",
            "tuning_weinan_units_used_summary.json",
        )
        for path in root.glob(pattern)
        if path.is_file()
    )
    polar_root = root / "polar_time_of_day_units"
    polar_files = (
        sorted(str(path) for path in polar_root.rglob("*") if path.is_file())
        if polar_root.is_dir()
        else []
    )
    return output_files, polar_files


def load_weinan_series_from_minute_summaries(pair_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Tuning_Weinan's threshold path originally read chunk-level
    *_threshold_crossings.npz files. Newer Threshold_channel.py runs can instead
    contain per-pair *_minute_summary.csv files, so this mirrors the same return
    shape from those summaries.
    """
    summary_paths = minute_summary_paths_for_pair(pair_dir)
    if not summary_paths:
        raise RuntimeError(f"No minute summary CSV files found under: {pair_dir}")

    rows: list[tuple[float, float, float]] = []
    labels: list[str] = []
    for summary_path in summary_paths:
        table = normalize_minute_summary_table(summary_path)
        for row in table.itertuples(index=False):
            row_dict = row._asdict()
            recording_name = str(row_dict["recording_name"])
            start_dt = parse_recording_start_datetime(recording_name)
            if start_dt is None:
                continue
            minute_start_sec = float(row_dict["time_start_sec"])
            dt_minute = start_dt + timedelta(seconds=minute_start_sec)
            xs_min_epoch = dt_minute.timestamp() / 60.0
            amplitude = safe_float(row_dict.get("amplitude_ptp_uv"))
            firing = safe_float(row_dict.get("firing_rate_hz"))
            rows.append(
                (
                    xs_min_epoch,
                    float("nan") if amplitude is None else float(amplitude),
                    0.0 if firing is None else float(firing),
                )
            )
            labels.append(tuning_weinan.datetime_to_x_label_5p5a(dt_minute))

    if not rows:
        raise RuntimeError(
            f"No timestamped minute-summary rows could be read under: {pair_dir}. "
            "Expected recording_name values like Chronic_Rec_YYYYMMDD_HHMMSS."
        )

    order = np.argsort([row[0] for row in rows])
    rows_sorted = [rows[int(index)] for index in order]
    labels_sorted = [labels[int(index)] for index in order]
    xs = np.asarray([row[0] for row in rows_sorted], dtype=float)
    amplitude = np.asarray([row[1] for row in rows_sorted], dtype=float)
    firing = np.asarray([row[2] for row in rows_sorted], dtype=float)
    amplitude_5min = tuning_weinan.rolling_mean_skip_outlier(xs, amplitude, window_min=5.0)
    return xs, amplitude, amplitude_5min, firing, labels_sorted


def load_weinan_series_from_pair_dir(pair_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    try:
        return _ORIGINAL_TUNING_WEINAN_LOAD_SERIES_FROM_PAIR_DIR(pair_dir)
    except RuntimeError as exc:
        if "No chunk npz files found" not in str(exc):
            raise
        log_status(f"Using minute-summary fallback for Tuning_Weinan series: {pair_dir}")
        return load_weinan_series_from_minute_summaries(pair_dir)


def run_tuning_weinan(run_root: Path, output_root: Path, *, only_polar: bool = True) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_status(f"Starting Tuning_Weinan threshold plots for {run_root}")
    log_status(f"Tuning_Weinan output folder: {output_root}")
    original_parse_pair = tuning_weinan.parse_pair_id_from_folder_name
    original_load_series = tuning_weinan.load_series_from_pair_dir
    argv = [
        "Tuning_Weinan.py",
        "--run-root",
        str(run_root),
        "--output-root",
        str(output_root),
        "--input-mode",
        "threshold",
        "--render-polar-all",
        "--no-reuse-csv",
    ]
    if only_polar:
        argv.append("--only-polar")
    tuning_weinan.parse_pair_id_from_folder_name = parse_threshold_pair_folder_name
    tuning_weinan.load_series_from_pair_dir = load_weinan_series_from_pair_dir
    try:
        with patched_argv(argv):
            exit_code = int(tuning_weinan.main())
    finally:
        tuning_weinan.parse_pair_id_from_folder_name = original_parse_pair
        tuning_weinan.load_series_from_pair_dir = original_load_series

    if only_polar and exit_code == 0 and not (output_root / "tuning_weinan_units_used_summary.csv").is_file():
        log_status("Writing Tuning_Weinan usage summary for only-polar threshold run.")
        write_tuning_weinan_usage_summary_from_threshold_minutes(run_root, output_root)

    output_files, polar_files = threshold_tuning_weinan_output_files(output_root)
    return {
        "status": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "run_root": str(Path(run_root).resolve()),
        "output_dir": str(output_root.resolve()),
        "output_files": output_files,
        "polar_output_files": polar_files,
        "verification": {
            "output_dir": str(output_root.resolve()),
            "num_output_files": len(output_files),
            "num_polar_output_files": len(polar_files),
            "has_usage_summary": (output_root / "tuning_weinan_units_used_summary.csv").is_file(),
            "has_polar_output_dir": (output_root / "polar_time_of_day_units").is_dir(),
        },
    }


def write_tuning_weinan_usage_summary_from_threshold_minutes(run_root: Path, output_root: Path) -> None:
    pair_meta = discover_threshold_pair_meta(run_root)
    if not pair_meta:
        raise RuntimeError(f"No sgch*_thr*uV threshold-pair folders found under: {run_root}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    series_cache = {}
    total_pair_dirs = len(pair_meta)
    for pair_index, (pair, pair_dir) in enumerate(pair_meta, start=1):
        if pair_index == 1 or pair_index % 25 == 0 or pair_index == total_pair_dirs:
            log_status(
                f"Preparing Tuning_Weinan usage summary: pair {pair_index} / {total_pair_dirs} "
                f"({pair_dir.name})"
            )
        series_cache[pair] = load_weinan_series_from_pair_dir(pair_dir)

    tuning_weinan.write_threshold_unit_usage_summary(
        pair_meta=pair_meta,
        series_cache=series_cache,
        run_root=output_root,
    )


def run_tuning_weinan_combined(run_roots: tuple[Path, ...], output_root: Path, *, only_polar: bool = True) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_status(f"Starting combined Tuning_Weinan analysis for {len(run_roots)} threshold run(s)")
    log_status(f"Combined Tuning_Weinan output folder: {output_root}")

    grouped_series: dict[str, dict] = {}
    total_pair_dirs = 0
    for run_root in run_roots:
        pair_meta = discover_threshold_pair_meta(run_root)
        total_pair_dirs += len(pair_meta)
        log_status(f"Combined tuning input {Path(run_root).name}: {len(pair_meta)} threshold unit folder(s)")
        for pair, pair_dir in pair_meta:
            unit_key = threshold_unit_key_from_dir(pair, pair_dir)
            entry = grouped_series.setdefault(
                unit_key,
                {
                    "pair": pair,
                    "unit_key": unit_key,
                    "sg_ch": int(pair.sg_ch),
                    "threshold_uv": threshold_min_from_unit_key(unit_key, float(pair.threshold_uv)),
                    "pair_dirs": [],
                    "input_runs": [],
                    "xs": [],
                    "amplitude": [],
                    "firing": [],
                    "labels": [],
                },
            )
            xs, amplitude, _amplitude_5min, firing, labels = load_weinan_series_from_pair_dir(pair_dir)
            entry["pair_dirs"].append(str(pair_dir))
            entry["input_runs"].append(str(Path(run_root).resolve()))
            entry["xs"].append(xs)
            entry["amplitude"].append(amplitude)
            entry["firing"].append(firing)
            entry["labels"].extend(labels)

    if not grouped_series:
        raise RuntimeError(
            "No threshold unit series were found for combined Tuning_Weinan analysis: "
            f"{[str(path) for path in run_roots]}"
        )

    polar_root = output_root / "polar_time_of_day_units"
    polar_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    sorted_entries = sorted(
        grouped_series.values(),
        key=lambda entry: (int(entry["sg_ch"]), float(entry["threshold_uv"]), str(entry["unit_key"])),
    )
    for unit_index, entry in enumerate(sorted_entries, start=1):
        xs = np.concatenate(entry["xs"]).astype(float)
        amplitude = np.concatenate(entry["amplitude"]).astype(float)
        firing = np.concatenate(entry["firing"]).astype(float)
        labels = list(entry["labels"])
        order = np.argsort(xs)
        xs = xs[order]
        amplitude = amplitude[order]
        firing = firing[order]
        labels = [labels[int(index)] for index in order]

        unit_key = str(entry["unit_key"])
        log_status(
            f"Rendering combined tuning polar plots [{unit_index}/{len(sorted_entries)}] -> {unit_key}"
        )
        tuning_weinan.render_polar_series(
            unit_key,
            xs,
            amplitude,
            firing,
            polar_root / unit_key,
            include_series_name_in_filename=False,
        )

        datetimes = tuning_weinan._epoch_min_to_datetime(xs)
        session_dates = sorted({dt.date().isoformat() for dt in datetimes if dt is not None})
        summary_rows.append(
            {
                "series_name": unit_key,
                "sg_ch": int(entry["sg_ch"]),
                "threshold_uv": float(entry["threshold_uv"]),
                "n_points": int(len(xs)),
                "n_input_runs": int(len(set(entry["input_runs"]))),
                "input_runs": " | ".join(sorted(set(entry["input_runs"]))),
                "n_pair_dirs": int(len(entry["pair_dirs"])),
                "pair_dirs": " | ".join(entry["pair_dirs"]),
                "n_session_dates": int(len(session_dates)),
                "session_dates": "; ".join(session_dates),
                "n_labels": int(len({str(label) for label in labels if str(label)})),
            }
        )

    summary_table = pd.DataFrame(summary_rows)
    summary_csv = output_root / "tuning_weinan_units_used_summary.csv"
    summary_json = output_root / "tuning_weinan_units_used_summary.json"
    summary_table.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "input_run_roots": [str(Path(run_root).resolve()) for run_root in run_roots],
                "output_root": str(output_root.resolve()),
                "analysis_scope": "combined_threshold_runs",
                "n_input_runs": int(len(run_roots)),
                "n_pair_dirs_read": int(total_pair_dirs),
                "n_combined_series": int(len(summary_table)),
                "series": summary_table.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not only_polar:
        log_status(
            "Combined Tuning_Weinan currently renders polar summary plots only; "
            "giant stacked master plots are intentionally skipped for threshold runs."
        )

    output_files, polar_files = threshold_tuning_weinan_output_files(output_root)
    return {
        "status": "completed",
        "exit_code": 0,
        "run_roots": [str(Path(run_root).resolve()) for run_root in run_roots],
        "output_dir": str(output_root.resolve()),
        "analysis_scope": "combined_threshold_runs",
        "output_files": output_files,
        "polar_output_files": polar_files,
        "verification": {
            "output_dir": str(output_root.resolve()),
            "num_output_files": len(output_files),
            "num_polar_output_files": len(polar_files),
            "has_usage_summary": summary_csv.is_file(),
            "has_polar_output_dir": polar_root.is_dir(),
            "n_combined_series": int(len(summary_table)),
        },
    }


def failed_stage_payload(stage_name: str, exc: Exception, **extra) -> dict:
    payload = {
        "status": "failed",
        "stage": stage_name,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    payload.update(extra)
    return payload


def run_threshold_presentation(population_csv: Path, output_dir: Path) -> Path:
    presentation_dir = output_dir / "presentation_threshold"
    presentation_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(population_csv)

    feature_columns = [
        column
        for column in table.columns
        if column.endswith("__firing_rate_hz")
    ]
    rows = []
    for column in feature_columns:
        unit_key = column[: -len("__firing_rate_hz")]
        sg_match = re.search(r"sgch(?P<sg>\d+)", unit_key)
        thr_match = re.search(r"_thr(?P<thr>.+)uV", unit_key)
        threshold_text = thr_match.group("thr").replace("p", ".") if thr_match else ""
        threshold_min_text = threshold_text.split("to", 1)[0]
        values = pd.to_numeric(table[column], errors="coerce")
        present = values.fillna(0.0) > 0.0
        rows.append(
            {
                "final_group_key": unit_key,
                "sg_ch": int(sg_match.group("sg")) if sg_match else np.nan,
                "threshold_label": threshold_text,
                "threshold_uv": float(threshold_min_text) if threshold_min_text else np.nan,
                "n_minutes": int(values.notna().sum()),
                "n_active_minutes": int(present.sum()),
                "mean_firing_rate_hz": safe_float(values.mean()),
                "max_firing_rate_hz": safe_float(values.max()),
                "n_calendar_days": int(table.loc[present, "calendar_day"].nunique()) if "calendar_day" in table else 0,
            }
        )

    summary = pd.DataFrame(rows).sort_values(["sg_ch", "threshold_uv"], na_position="last")
    summary_csv = presentation_dir / "threshold_units_presentation_summary.csv"
    summary.to_csv(summary_csv, index=False)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(summary))))
        labels = [f"sg{int(row.sg_ch)} thr{row.threshold_uv:g}" for row in summary.itertuples(index=False)]
        ax.barh(labels, summary["mean_firing_rate_hz"].fillna(0.0), color="#4c78a8")
        ax.set_xlabel("Mean firing rate [Hz]")
        ax.set_ylabel("Threshold unit")
        ax.set_title("Threshold-unit mean firing rate")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(presentation_dir / "threshold_units_mean_firing_rate.png", dpi=200)
        plt.close(fig)

        pivot = summary.pivot_table(
            index="sg_ch",
            columns="threshold_uv",
            values="mean_firing_rate_hz",
            aggfunc="mean",
        ).sort_index()
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * pivot.shape[1]), max(4, 0.35 * pivot.shape[0])))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels([str(int(value)) for value in pivot.index])
        ax.set_xlabel("Threshold [uV]")
        ax.set_ylabel("SG channel")
        ax.set_title("Mean firing rate by channel and threshold")
        fig.colorbar(image, ax=ax, label="Hz")
        fig.tight_layout()
        fig.savefig(presentation_dir / "threshold_channel_threshold_heatmap.png", dpi=200)
        plt.close(fig)

    manifest = presentation_dir / "threshold_presentation_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "population_csv": str(population_csv),
                "summary_csv": str(summary_csv),
                "n_threshold_units": int(len(summary)),
                "outputs": [str(path) for path in sorted(presentation_dir.glob("*"))],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log_status(f"Saved threshold presentation summary: {manifest}")
    return manifest


def parse_token_list(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    return tuple(
        token.strip().strip('"').strip("'")
        for token in re.split(r"[,;\n]+", str(raw_value))
        if token.strip().strip('"').strip("'")
    )


def build_recording_table_from_population_csv(population_csv: Path) -> pd.DataFrame:
    table = pd.read_csv(population_csv)
    columns = [
        "session_id",
        "session_key",
        "session_name",
        "session_start_datetime",
        "threshold_run_name",
        "threshold_run_root",
    ]
    available = [column for column in columns if column in table.columns]
    recording_table = (
        table[available]
        .drop_duplicates()
        .sort_values(["session_start_datetime", "session_id", "session_name"], na_position="last")
        .reset_index(drop=True)
    )
    recording_table["recording_id"] = np.arange(1, len(recording_table) + 1, dtype=int)
    return recording_table


def select_recordings_from_tokens(
    recording_table: pd.DataFrame,
    tokens: tuple[str, ...],
) -> pd.DataFrame:
    if not tokens:
        return recording_table.iloc[0:0].copy()

    selected_indices: list[int] = []
    unmatched_tokens: list[str] = []
    lookup = recording_table.reset_index(drop=False).copy()
    for token in tokens:
        matches = pd.DataFrame()
        parsed_id = safe_int(token)
        if parsed_id is not None:
            matches = lookup[lookup["recording_id"].astype(int) == int(parsed_id)]
        for column in ("session_key", "session_name", "threshold_run_name"):
            if not matches.empty or column not in lookup.columns:
                continue
            matches = lookup[lookup[column].astype(str).str.lower() == token.lower()]
        if matches.empty:
            contains_mask = pd.Series(False, index=lookup.index)
            for column in ("session_key", "session_name", "threshold_run_name"):
                if column in lookup.columns:
                    contains_mask = contains_mask | lookup[column].astype(str).str.contains(
                        re.escape(token),
                        case=False,
                        regex=True,
                        na=False,
                    )
            matches = lookup[contains_mask]

        if matches.empty:
            unmatched_tokens.append(token)
        else:
            selected_indices.extend(int(index) for index in matches["index"].tolist())

    if unmatched_tokens:
        raise ValueError(f"Could not match sham/drug recording token(s): {unmatched_tokens}")

    return (
        recording_table.loc[sorted(set(selected_indices))]
        .sort_values("session_start_datetime")
        .reset_index(drop=True)
    )


def build_injection_phase_schedule(sham_recordings: pd.DataFrame, drug_recordings: pd.DataFrame) -> dict:
    if sham_recordings.empty or drug_recordings.empty:
        raise ValueError("Both sham and drug recording selections are required for sham/drug marker mode.")

    sham_table = sham_recordings.sort_values("session_start_datetime").drop_duplicates(
        ["session_key", "session_start_datetime"]
    )
    drug_table = drug_recordings.sort_values("session_start_datetime").drop_duplicates(
        ["session_key", "session_start_datetime"]
    )
    drug_starts = pd.to_datetime(drug_table["session_start_datetime"], errors="coerce")

    drug_intervals = []
    for drug_row in drug_table.itertuples(index=False):
        drug_start = pd.Timestamp(drug_row.session_start_datetime)
        drug_intervals.append(
            {
                "session_name": str(drug_row.session_name),
                "session_key": str(drug_row.session_key),
                "start": drug_start.isoformat(sep=" "),
                "end": (drug_start + timedelta(hours=24)).isoformat(sep=" "),
            }
        )

    sham_intervals = []
    for sham_row in sham_table.itertuples(index=False):
        sham_start = pd.Timestamp(sham_row.session_start_datetime)
        following_drugs = drug_table.loc[drug_starts > sham_start]
        if following_drugs.empty:
            raise ValueError(
                f"Sham recording {sham_row.session_name!r} at {sham_start} has no following drug recording."
            )
        drug_row = following_drugs.iloc[0]
        drug_start = pd.Timestamp(drug_row["session_start_datetime"])
        sham_intervals.append(
            {
                "session_name": str(sham_row.session_name),
                "session_key": str(sham_row.session_key),
                "paired_drug_session_name": str(drug_row["session_name"]),
                "paired_drug_session_key": str(drug_row["session_key"]),
                "start": sham_start.isoformat(sep=" "),
                "end": drug_start.isoformat(sep=" "),
            }
        )

    return {
        "label_type": "injection_phase_marker",
        "baseline_label": "baseline",
        "sham_label": "sham",
        "drug_label": "drug",
        "interpretation": (
            "LDA labels remain clock_hour_of_day. Marker shapes encode baseline/sham/drug: "
            "sham from each sham recording start until the next drug recording start; drug "
            "from each drug recording start until 24 hours later; drug has priority."
        ),
        "sham_intervals": sham_intervals,
        "drug_intervals": drug_intervals,
        "sham_recordings": sham_table.to_dict(orient="records"),
        "drug_recordings": drug_table.to_dict(orient="records"),
    }


def collect_injection_phase_schedule(population_csv: Path, config: PipelineConfig, output_dir: Path) -> dict | None:
    if config.analysis_mode != "sham_drug_markers":
        return None

    recording_table = build_recording_table_from_population_csv(population_csv)
    display_columns = [
        "recording_id",
        "session_name",
        "session_start_datetime",
        "threshold_run_name",
        "session_key",
    ]
    available_columns = [column for column in display_columns if column in recording_table.columns]
    print("\nAvailable threshold recordings for sham/drug marker selection:", flush=True)
    print(recording_table[available_columns].to_string(index=False), flush=True)

    sham_tokens = config.sham_sessions
    drug_tokens = config.drug_sessions
    if not sham_tokens:
        sham_tokens = parse_token_list(
            input("\nEnter sham recording_id(s), recording name(s), or tokens, separated by commas: ")
        )
    if not drug_tokens:
        drug_tokens = parse_token_list(
            input("Enter drug recording_id(s), recording name(s), or tokens, separated by commas: ")
        )

    sham_recordings = select_recordings_from_tokens(recording_table, sham_tokens)
    drug_recordings = select_recordings_from_tokens(recording_table, drug_tokens)
    schedule = build_injection_phase_schedule(sham_recordings, drug_recordings)

    print("\nSham/drug marker interpretation:", flush=True)
    print(f"  {schedule['interpretation']}", flush=True)
    print("  Sham intervals:", flush=True)
    for interval in schedule["sham_intervals"]:
        print(
            f"    {interval['session_name']} -> {interval['paired_drug_session_name']}: "
            f"[{interval['start']}, {interval['end']})",
            flush=True,
        )
    print("  Drug intervals:", flush=True)
    for interval in schedule["drug_intervals"]:
        print(
            f"    {interval['session_name']}: [{interval['start']}, {interval['end']})",
            flush=True,
        )
    if not config.confirm_sham_drug:
        confirm = input("Is this interpretation correct? Type YES to continue: ").strip()
        if confirm != "YES":
            raise RuntimeError("Sham/drug marker setup was not confirmed; stopping before LDA.")

    schedule_path = output_dir / "threshold_sham_drug_marker_schedule.json"
    schedule_path.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    log_status(f"Saved sham/drug marker schedule: {schedule_path}")
    return schedule


def assign_injection_phase_for_times(datetimes: pd.Series, schedule: dict) -> pd.Series:
    parsed = pd.to_datetime(datetimes, errors="coerce")
    phases = pd.Series("baseline", index=parsed.index, dtype=object)

    for interval in schedule.get("sham_intervals", []) or []:
        start = pd.Timestamp(interval["start"])
        end = pd.Timestamp(interval["end"])
        mask = parsed.notna() & (parsed >= start) & (parsed < end)
        phases.loc[mask] = "sham"

    for interval in schedule.get("drug_intervals", []) or []:
        start = pd.Timestamp(interval["start"])
        end = pd.Timestamp(interval["end"])
        mask = parsed.notna() & (parsed >= start) & (parsed < end)
        phases.loc[mask] = "drug"

    return phases


def add_phase_markers_to_lda_outputs(lda_dirs: list[Path], schedule: dict) -> list[Path]:
    output_paths: list[Path] = []
    for output_dir in lda_dirs:
        projection_csv = Path(output_dir) / "lda_projection.csv"
        if not projection_csv.exists():
            continue
        projection = pd.read_csv(projection_csv)
        time_column = "hour_start_datetime" if "hour_start_datetime" in projection.columns else "sample_start_datetime"
        if time_column not in projection.columns or "LD1" not in projection.columns:
            continue
        projection["injection_phase"] = assign_injection_phase_for_times(projection[time_column], schedule)
        projection.to_csv(projection_csv, index=False)

        y_values = (
            pd.to_numeric(projection["LD2"], errors="coerce").to_numpy(dtype=float)
            if "LD2" in projection.columns
            else np.zeros(len(projection), dtype=float)
        )
        x_values = pd.to_numeric(projection["LD1"], errors="coerce").to_numpy(dtype=float)
        hours = pd.to_numeric(projection["clock_hour_of_day"], errors="coerce").to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(10, 8))
        first_scatter = None
        for phase in ("baseline", "sham", "drug"):
            mask = projection["injection_phase"].astype(str).to_numpy() == phase
            if not np.any(mask):
                continue
            scatter = ax.scatter(
                x_values[mask],
                y_values[mask],
                c=hours[mask],
                cmap=CIRCULAR_HOUR_CMAP,
                norm=CIRCULAR_HOUR_NORM,
                marker=PHASE_MARKERS.get(phase, "o"),
                s=52,
                alpha=0.92,
                edgecolors="black",
                linewidths=0.45,
                label=PHASE_LABELS.get(phase, phase),
            )
            if first_scatter is None:
                first_scatter = scatter
        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        ax.set_title("LDA Projection - clock hour color, sham/drug marker shape")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if first_scatter is not None:
            colorbar = fig.colorbar(
                first_scatter,
                ax=ax,
                fraction=0.046,
                pad=0.04,
                boundaries=CIRCULAR_HOUR_BOUNDARIES,
                ticks=list(range(24)),
                spacing="proportional",
                drawedges=True,
            )
            colorbar.set_label("Hour")
        handles = [
            Line2D(
                [0],
                [0],
                marker=PHASE_MARKERS.get(phase, "o"),
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=8,
                label=PHASE_LABELS.get(phase, phase),
            )
            for phase in ("baseline", "sham", "drug")
            if phase in set(projection["injection_phase"].astype(str))
        ]
        if handles:
            ax.legend(handles=handles, title="marker", loc="best", frameon=True)
        fig.tight_layout()
        out_png = Path(output_dir) / "lda_2d_sham_drug_markers.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        output_paths.append(out_png)
        log_status(f"Saved sham/drug marker LDA plot: {out_png}")

        z_values = (
            pd.to_numeric(projection["LD3"], errors="coerce").to_numpy(dtype=float)
            if "LD3" in projection.columns
            else np.zeros(len(projection), dtype=float)
        )
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")
        first_scatter = None
        for phase in ("baseline", "sham", "drug"):
            mask = projection["injection_phase"].astype(str).to_numpy() == phase
            if not np.any(mask):
                continue
            scatter = ax.scatter(
                x_values[mask],
                y_values[mask],
                z_values[mask],
                c=hours[mask],
                cmap=CIRCULAR_HOUR_CMAP,
                norm=CIRCULAR_HOUR_NORM,
                marker=PHASE_MARKERS.get(phase, "o"),
                s=48,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.35,
                label=PHASE_LABELS.get(phase, phase),
            )
            if first_scatter is None:
                first_scatter = scatter
        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        ax.set_zlabel("LD3")
        ax.set_title("LDA Projection 3D - clock hour color, sham/drug marker shape")
        if first_scatter is not None:
            colorbar = fig.colorbar(
                first_scatter,
                ax=ax,
                fraction=0.046,
                pad=0.08,
                boundaries=CIRCULAR_HOUR_BOUNDARIES,
                ticks=list(range(24)),
                spacing="proportional",
                drawedges=True,
            )
            colorbar.set_label("Hour")
        if handles:
            ax.legend(handles=handles, title="marker", loc="best", frameon=True)
        fig.tight_layout()
        out_3d_png = Path(output_dir) / "lda_3d_sham_drug_markers.png"
        fig.savefig(out_3d_png, dpi=300)
        plt.close(fig)
        output_paths.append(out_3d_png)
        log_status(f"Saved sham/drug marker LDA 3D plot: {out_3d_png}")
    return output_paths


def parse_feature_modes(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return DEFAULT_LDA_FEATURE_MODES
    return tuple(part.strip().upper() for part in raw_value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LDA, Tuning_Weinan, and threshold presentation summaries on "
            "Threshold_channel.py threshold_crossings outputs."
        )
    )
    parser.add_argument("run_roots", nargs="*", help="One or more threshold_crossings_* run folders")
    parser.add_argument(
        "--run-root",
        dest="run_root_opts",
        action="append",
        help="threshold_crossings_* run folder. May be repeated; comma/semicolon-separated values are accepted.",
    )
    parser.add_argument("--output-dir", help=f"Output folder. Defaults to <run_root>/{DEFAULT_OUTPUT_SUBDIR}, or the common parent for multiple inputs")
    parser.add_argument("--force-rebuild-population-csv", action="store_true")
    parser.add_argument("--skip-lda", action="store_true")
    parser.add_argument("--skip-tuning-weinan", action="store_true")
    parser.add_argument("--skip-presentation", action="store_true")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not ask which stages to run; use CLI skip flags/defaults.",
    )
    parser.add_argument(
        "--tuning-weinan-master-plots",
        action="store_true",
        help=(
            "Also render Tuning_Weinan's giant stacked master plots. By default this wrapper "
            "renders per-threshold polar plots only to avoid oversized images when many thresholds exist."
        ),
    )
    parser.add_argument(
        "--analysis-mode",
        choices=ANALYSIS_MODES,
        default="baseline",
        help="baseline runs clock-hour LDA only; sham_drug_markers adds sham/drug marker-shape plots.",
    )
    parser.add_argument("--lda-feature-modes", help="Comma-separated LDA feature modes")
    parser.add_argument("--min-firing-rate-hz", type=float, default=0.0)
    parser.add_argument("--min-minutes-per-hour", type=int, default=1)
    parser.add_argument("--min-bins-per-label", type=int, default=2)
    parser.add_argument("--cv-n-splits", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--no-zscore", action="store_true")
    parser.add_argument("--sham-sessions", help="Comma/semicolon-separated sham recording IDs, names, or tokens.")
    parser.add_argument("--drug-sessions", help="Comma/semicolon-separated drug recording IDs, names, or tokens.")
    parser.add_argument("--confirm-sham-drug", action="store_true", help="Skip confirmation prompt for sham/drug marker intervals.")
    return parser.parse_args()


def jsonable_config(config: PipelineConfig) -> dict:
    payload = asdict(config)
    payload["run_roots"] = [str(path) for path in config.run_roots]
    if config.output_dir is not None:
        payload["output_dir"] = str(config.output_dir)
    return payload


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    raw_values: list[str] = []
    for value in args.run_root_opts or []:
        raw_values.extend(re.split(r"[;,]", value))
    raw_values.extend(args.run_roots or [])
    if not raw_values:
        raw_text = input("Enter one or more threshold_crossings_* run folders, separated by semicolons: ").strip()
        raw_values.extend(re.split(r"[;,]", raw_text))
    run_roots = tuple(
        Path(value.strip().strip('"').strip("'"))
        for value in raw_values
        if value.strip().strip('"').strip("'")
    )
    if not run_roots:
        raise ValueError("At least one threshold_crossings_* run folder is required.")
    for run_root in run_roots:
        if not run_root.exists() or not run_root.is_dir():
            raise NotADirectoryError(f"Threshold run folder not found: {run_root}")
    skip_lda = bool(args.skip_lda)
    skip_tuning_weinan = bool(args.skip_tuning_weinan)
    skip_presentation = bool(args.skip_presentation)
    if not bool(args.non_interactive):
        print("\nSelect analysis stages to run:", flush=True)
        skip_lda = not prompt_yes_no("Run LDA clock-hour analysis?", default=not skip_lda)
        skip_tuning_weinan = not prompt_yes_no("Run Tuning_Weinan?", default=not skip_tuning_weinan)
        skip_presentation = not prompt_yes_no("Run presentation summary?", default=not skip_presentation)
    return PipelineConfig(
        run_roots=run_roots,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        analysis_mode=str(args.analysis_mode),
        lda_feature_modes=parse_feature_modes(args.lda_feature_modes),
        min_firing_rate_hz=float(args.min_firing_rate_hz),
        min_minutes_per_hour=int(args.min_minutes_per_hour),
        min_bins_per_label=int(args.min_bins_per_label),
        cv_n_splits=int(args.cv_n_splits),
        n_permutations=int(args.n_permutations),
        apply_zscore=not bool(args.no_zscore),
        skip_lda=skip_lda,
        skip_tuning_weinan=skip_tuning_weinan,
        skip_presentation=skip_presentation,
        reuse_population_csv=not bool(args.force_rebuild_population_csv),
        tuning_weinan_only_polar=not bool(args.tuning_weinan_master_plots),
        sham_sessions=parse_token_list(args.sham_sessions),
        drug_sessions=parse_token_list(args.drug_sessions),
        confirm_sham_drug=bool(args.confirm_sham_drug),
    )


def run_pipeline(config: PipelineConfig) -> dict:
    start_time = time.perf_counter()
    timings: list[dict] = []
    output_dir = resolve_output_dir(config)
    log_status(f"Input threshold runs: {len(config.run_roots)}")
    for run_root in config.run_roots:
        log_status(f"  {run_root}")
    log_status(f"Output folder: {output_dir}")

    with timed_stage("threshold population CSV materialization", timings):
        population_csv = build_threshold_population_csv(
            config.run_roots,
            output_dir,
            force=not config.reuse_population_csv,
        )
    injection_schedule = None
    if not config.skip_lda:
        with timed_stage("LDA sham/drug marker setup", timings):
            injection_schedule = collect_injection_phase_schedule(population_csv, config, output_dir)

    result = {
        "input_run_roots": [str(path.resolve()) for path in config.run_roots],
        "output_dir": str(output_dir.resolve()),
        "config": jsonable_config(config),
        "population_csv": str(population_csv.resolve()),
        "lda_output_dirs": [],
        "sham_drug_marker_plots": [],
        "injection_phase_schedule": injection_schedule,
        "tuning_weinan_results": [],
        "presentation_manifest": None,
        "timings": timings,
    }

    if config.skip_lda:
        log_status("Skipping LDA by request.")
    else:
        with timed_stage("LDA threshold clock-hour analysis", timings):
            lda_dirs = run_lda(population_csv, output_dir, config)
        result["lda_output_dirs"] = [str(path.resolve()) for path in lda_dirs]
        if injection_schedule is not None:
            with timed_stage("LDA sham/drug marker plotting", timings):
                marker_plots = add_phase_markers_to_lda_outputs(lda_dirs, injection_schedule)
            result["sham_drug_marker_plots"] = [str(path.resolve()) for path in marker_plots]

    if config.skip_tuning_weinan:
        log_status("Skipping Tuning_Weinan by request.")
    else:
        tuning_output_base = output_dir / "Tuning_Weinan"
        tuning_output_base.mkdir(parents=True, exist_ok=True)
        try:
            with timed_stage("Tuning-Weinan combined threshold stats", timings):
                result["tuning_weinan_results"] = [
                    run_tuning_weinan_combined(
                        config.run_roots,
                        tuning_output_base,
                        only_polar=bool(config.tuning_weinan_only_polar),
                    )
                ]
        except Exception as exc:
            log_status(f"Combined Tuning_Weinan failed; continuing with later stages: {exc}")
            output_files, polar_files = threshold_tuning_weinan_output_files(tuning_output_base)
            result["tuning_weinan_results"] = [
                failed_stage_payload(
                    "Tuning-Weinan combined threshold stats",
                    exc,
                    run_roots=[str(path.resolve()) for path in config.run_roots],
                    output_dir=str(tuning_output_base.resolve()),
                    output_files=output_files,
                    polar_output_files=polar_files,
                    verification={
                        "output_dir": str(tuning_output_base.resolve()),
                        "num_output_files": len(output_files),
                        "num_polar_output_files": len(polar_files),
                        "has_usage_summary": (tuning_output_base / "tuning_weinan_units_used_summary.csv").is_file(),
                        "has_polar_output_dir": (tuning_output_base / "polar_time_of_day_units").is_dir(),
                    },
                )
            ]

    if config.skip_presentation:
        log_status("Skipping threshold presentation summary by request.")
    else:
        with timed_stage("threshold presentation summary", timings):
            result["presentation_manifest"] = str(run_threshold_presentation(population_csv, output_dir).resolve())

    result["elapsed_seconds"] = float(time.perf_counter() - start_time)
    summary_path = output_dir / "threshold_LDA_TuningWN_pre_run_summary.json"
    with timed_stage("write pipeline summary", timings):
        result["timings"] = timings
        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_runtime_summary(timings, total_elapsed_seconds=float(time.perf_counter() - start_time))
    log_status(f"Pipeline complete. Summary: {summary_path}")
    return result


def main() -> None:
    config = config_from_args(parse_args())
    run_pipeline(config)


if __name__ == "__main__":
    main()
