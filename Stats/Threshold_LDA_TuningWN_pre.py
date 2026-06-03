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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import LDA_weinan as lda_threshold
import Tuning_Weinan as tuning_weinan


_ORIGINAL_TUNING_WEINAN_LOAD_SERIES_FROM_PAIR_DIR = tuning_weinan.load_series_from_pair_dir

DEFAULT_LDA_FEATURE_MODES = ("FR_ONLY", "FR_AMP", "FR_CV2", "FR_PEAK_TO_TROUGH", "MULTI_FEATURE")
DEFAULT_OUTPUT_SUBDIR = "threshold_LDA_TuningWN_pre"
PORTABLE_CSV_OUTPUT_PREFIX = "threshold_csvs"
ANALYSIS_OUTPUT_DIR_NAMES = {
    "LDA_threshold",
    "Tuning_Weinan",
    DEFAULT_OUTPUT_SUBDIR,
    "threshold_presentation",
}
ANALYSIS_MODES = ("baseline", "sham_drug_markers", "treatment_markers")
PHASE_ORDER = (
    "baseline",
    "sham_saline",
    "drug_saline",
    "sham_caf",
    "drug_caf",
    "saline_sham",
    "saline_drug",
    "caffeine_sham",
    "caffeine_drug",
    "sham",
    "drug",
)
PHASE_MARKERS = {
    "baseline": "o",
    "sham": "s",
    "drug": "^",
    "sham_saline": "s",
    "drug_saline": "^",
    "sham_caf": "D",
    "drug_caf": "X",
    "saline_sham": "s",
    "saline_drug": "^",
    "caffeine_sham": "D",
    "caffeine_drug": "X",
}
PHASE_LABELS = {
    "baseline": "baseline",
    "sham": "sham",
    "drug": "drug",
    "sham_saline": "sham saline",
    "drug_saline": "drug saline",
    "sham_caf": "sham caffeine",
    "drug_caf": "drug caffeine",
    "saline_sham": "saline sham",
    "saline_drug": "saline drug",
    "caffeine_sham": "caffeine sham",
    "caffeine_drug": "caffeine drug",
}
CIRCULAR_HOUR_CMAP = plt.get_cmap("twilight_shifted", 24)
CIRCULAR_HOUR_BOUNDARIES = np.arange(-0.5, 24.5, 1.0)
CIRCULAR_HOUR_NORM = BoundaryNorm(CIRCULAR_HOUR_BOUNDARIES, CIRCULAR_HOUR_CMAP.N)
BASELINE_GEOMETRY_N_PERMUTATIONS = 500
BASELINE_GEOMETRY_PERMUTATION_SEED = 1729
FORBIDDEN_LDA_FEATURE_PATTERNS = (
    "clock",
    "hour",
    "time",
    "datetime",
    "date",
    "calendar",
    "sample",
    "order",
    "index",
    "label",
    "session",
    "recording",
)


@dataclass
class PipelineConfig:
    run_roots: tuple[Path, ...]
    output_dir: Path | None = None
    output_suffix: str | None = None
    selected_threshold_unit_keys: tuple[str, ...] = ()
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
    non_interactive: bool = False
    reuse_population_csv: bool = True
    tuning_weinan_only_polar: bool = True
    tuning_baseline_phase_overlays: bool = False
    sham_sessions: tuple[str, ...] = ()
    drug_sessions: tuple[str, ...] = ()
    saline_sham_sessions: tuple[str, ...] = ()
    saline_drug_sessions: tuple[str, ...] = ()
    caffeine_sham_sessions: tuple[str, ...] = ()
    caffeine_drug_sessions: tuple[str, ...] = ()
    confirm_sham_drug: bool = False
    baseline_only_marker_lda: bool = False


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


def output_suffix_slug(value: object) -> str:
    text = safe_slug(value)
    if text in {"value", ""}:
        return ""
    return text


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
        if (
            "polar_time_of_day_units" in relative_parts
            or any(part.startswith(PORTABLE_CSV_OUTPUT_PREFIX) for part in relative_parts)
            or any(part in ANALYSIS_OUTPUT_DIR_NAMES for part in relative_parts)
        ):
            continue
        if not path.is_dir() or not path.name.startswith("sgch") or "_thr" not in path.name:
            continue
        pair = parse_threshold_pair_folder_name(path.name)
        if pair is not None:
            pair_meta.append((pair, path))
    pair_meta.sort(key=lambda item: item[0].sort_key())
    return pair_meta


def discover_threshold_unit_options(run_roots: tuple[Path, ...]) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    for run_root in run_roots:
        for pair, pair_dir in discover_threshold_pair_meta(run_root):
            unit_key = threshold_unit_key_from_dir(pair, pair_dir)
            if unit_key in seen:
                continue
            seen.add(unit_key)
            rows.append(
                {
                    "unit_key": unit_key,
                    "sg_ch": int(pair.sg_ch),
                    "threshold_uv": threshold_min_from_unit_key(unit_key, float(pair.threshold_uv)),
                    "threshold_label": threshold_label_from_unit_key(unit_key, float(pair.threshold_uv)),
                }
            )
    rows.sort(key=lambda row: (int(row["sg_ch"]), float(row["threshold_uv"]), str(row["unit_key"])))
    return rows


def prompt_threshold_units_for_lda(run_roots: tuple[Path, ...]) -> tuple[str, ...]:
    options = discover_threshold_unit_options(run_roots)
    if not options:
        raise RuntimeError(
            "No sgch*_thr*uV threshold-pair folders were found before LDA setup. "
            f"Input folders: {[str(path) for path in run_roots]}"
        )

    by_channel: dict[int, list[dict]] = {}
    for row in options:
        by_channel.setdefault(int(row["sg_ch"]), []).append(row)

    print("\nAvailable threshold channels for LDA:", flush=True)
    for sg_ch in sorted(by_channel):
        labels = ", ".join(str(row["threshold_label"]) for row in by_channel[sg_ch])
        print(f"  SG channel {sg_ch}: {labels}", flush=True)

    raw = input(
        "\nEnter SG channel(s) to include in LDA, separated by commas "
        "(or press Enter / type all for all channels): "
    ).strip()
    if raw == "" or raw.lower() == "all":
        selected_channels = set(by_channel.keys())
    else:
        selected_channels = set()
        for token in re.split(r"[;,]", raw):
            token = token.strip()
            if not token:
                continue
            try:
                selected_channels.add(int(token))
            except ValueError as exc:
                raise ValueError(f"Channel selection must be SG channel numbers or 'all'; got {token!r}") from exc

    missing = sorted(ch for ch in selected_channels if ch not in by_channel)
    if missing:
        raise ValueError(f"Selected channel(s) not found in threshold units: {missing}")

    selected_rows = [row for row in options if int(row["sg_ch"]) in selected_channels]
    if not selected_rows:
        raise ValueError("No threshold units selected for LDA.")

    print("\nThreshold units selected for LDA:", flush=True)
    for idx, row in enumerate(selected_rows, start=1):
        print(
            f"  {idx:4d}. {row['unit_key']}  |  SG channel {row['sg_ch']}  |  threshold {row['threshold_label']}",
            flush=True,
        )
    print(f"Total selected threshold units: {len(selected_rows)}", flush=True)

    if not prompt_yes_no("Use these channels/thresholds for LDA?", default=True):
        raise RuntimeError("Threshold channel selection was not confirmed; stopping before LDA.")

    return tuple(str(row["unit_key"]) for row in selected_rows)


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


def threshold_label_from_unit_key(unit_key: str, fallback: float | None = None) -> str:
    match = re.search(r"_thr(?P<thr>.+)uV", str(unit_key))
    if match is None:
        return f"{float(fallback):g} uV" if fallback is not None else "unknown"
    threshold_text = match.group("thr").replace("p", ".")
    if "to" in threshold_text:
        lo, hi = threshold_text.split("to", 1)
        return f"{lo} to {hi} uV"
    return f"{threshold_text} uV"


def threshold_feature_columns(unit_key: str) -> dict[str, str]:
    return {
        "firing_rate_hz": f"{unit_key}__firing_rate_hz",
        "average_amplitude_uv": f"{unit_key}__average_amplitude_uv",
        "cv2": f"{unit_key}__cv2",
        "peak_to_trough_ms": f"{unit_key}__peak_to_trough_ms",
    }


def default_output_dir(run_roots: tuple[Path, ...]) -> Path:
    if len(run_roots) == 1:
        run_root = Path(run_roots[0])
        return run_root.with_name(f"{run_root.name}_{DEFAULT_OUTPUT_SUBDIR}")
    try:
        common_root = Path(os.path.commonpath([str(path.resolve()) for path in run_roots]))
    except Exception:
        common_root = Path.cwd()
    if common_root.is_file():
        common_root = common_root.parent
    if not common_root.exists() or not common_root.is_dir():
        common_root = Path(run_roots[0]).resolve().parent
    return common_root / DEFAULT_OUTPUT_SUBDIR


def output_dir_with_suffix(base_dir: Path, suffix: str) -> Path:
    suffix = output_suffix_slug(suffix)
    if not suffix:
        raise ValueError("Output folder suffix cannot be empty.")
    return Path(base_dir).with_name(f"{Path(base_dir).name}_{suffix}")


def unique_timestamped_output_dir(base_dir: Path) -> Path:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return base_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir.with_name(f"{base_dir.name}_{stamp}")
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}_{stamp}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def resolve_existing_output_dir_collision(
    requested_output_dir: Path,
    configured_suffix: str | None,
    *,
    allow_prompt: bool = True,
) -> Path:
    if not requested_output_dir.exists():
        return requested_output_dir

    if configured_suffix:
        candidate = output_dir_with_suffix(requested_output_dir, configured_suffix)
        if candidate.exists():
            raise FileExistsError(
                f"Output folder already exists after applying suffix {configured_suffix!r}: {candidate}"
            )
        return candidate

    if allow_prompt and sys.stdin is not None and sys.stdin.isatty():
        print(
            f"\nOutput folder already exists:\n  {requested_output_dir}\n",
            flush=True,
        )
        while True:
            raw_suffix = input(
                "Enter text to add after the folder name for this run "
                "(example: channels_12_45): "
            ).strip()
            suffix = output_suffix_slug(raw_suffix)
            if not suffix:
                print("Please enter a non-empty suffix.", flush=True)
                continue
            candidate = output_dir_with_suffix(requested_output_dir, suffix)
            print(f"Proposed output folder:\n  {candidate}", flush=True)
            if candidate.exists():
                print("That folder also exists. Please choose a different suffix.", flush=True)
                continue
            if prompt_yes_no("Use this output folder?", default=True):
                return candidate

    # Batch/non-interactive fallback: avoid overwriting and avoid blocking forever.
    return unique_timestamped_output_dir(requested_output_dir)


def output_dir_has_lda_results(output_dir: Path) -> bool:
    lda_root = Path(output_dir) / "LDA_threshold"
    if not lda_root.is_dir():
        return False
    return any(lda_root.iterdir())


def resolve_output_dir(config: PipelineConfig) -> Path:
    requested_output_dir = Path(config.output_dir) if config.output_dir is not None else default_output_dir(config.run_roots)
    if requested_output_dir.exists():
        if config.output_suffix:
            output_dir = resolve_existing_output_dir_collision(
                requested_output_dir,
                config.output_suffix,
                allow_prompt=not config.non_interactive,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            log_status(f"Output folder suffix requested; writing this run to: {output_dir}")
            return output_dir
        population_csv = requested_output_dir / "threshold_population_minute_features.csv"
        manifest_json = requested_output_dir / "threshold_population_manifest.json"
        reusable, reason = is_population_csv_reusable(
            population_csv,
            manifest_json,
            config.run_roots,
            selected_unit_keys=config.selected_threshold_unit_keys,
        )
        if reusable:
            if not config.skip_lda and output_dir_has_lda_results(requested_output_dir):
                log_status(
                    "Existing output folder already contains LDA results; resolving output-folder "
                    "collision to avoid overwriting them."
                )
                output_dir = resolve_existing_output_dir_collision(
                    requested_output_dir,
                    config.output_suffix,
                    allow_prompt=not config.non_interactive,
                )
                if output_dir != requested_output_dir:
                    log_status(f"Writing this LDA run to: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)
                return output_dir
            log_status(
                "Existing output folder contains a matching precomputed threshold population CSV; "
                f"reusing folder: {requested_output_dir}"
            )
            return requested_output_dir
        log_status(
            "Existing output folder did not contain a reusable threshold population CSV "
            f"({reason}); resolving output-folder collision."
        )
    output_dir = resolve_existing_output_dir_collision(
        requested_output_dir,
        config.output_suffix,
        allow_prompt=not config.non_interactive,
    )
    if output_dir != requested_output_dir:
        log_status(
            f"Output folder already exists; writing this run to: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_run_config(run_root: Path) -> dict:
    config_path = Path(run_root) / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8-sig"))


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


def build_threshold_population_csv(
    run_roots: tuple[Path, ...],
    output_dir: Path,
    *,
    force: bool = False,
    selected_unit_keys: tuple[str, ...] = (),
) -> Path:
    population_csv = output_dir / "threshold_population_minute_features.csv"
    manifest_json = output_dir / "threshold_population_manifest.json"
    if population_csv.exists() and not force:
        reusable, reason = is_population_csv_reusable(
            population_csv,
            manifest_json,
            run_roots,
            selected_unit_keys=selected_unit_keys,
        )
        if reusable:
            log_status(f"Reusing threshold population CSV: {population_csv}")
            return population_csv
        log_status(
            "Existing threshold population CSV does not match the requested input folders; "
            f"rebuilding it now ({reason})."
        )
    if not force:
        for run_root in run_roots:
            source_population_csv = Path(run_root) / "threshold_population_minute_features.csv"
            source_manifest_json = Path(run_root) / "threshold_population_manifest.json"
            reusable, reason = is_population_csv_reusable(
                source_population_csv,
                source_manifest_json,
                run_roots,
                selected_unit_keys=selected_unit_keys,
            )
            if reusable:
                log_status(
                    "Using precomputed threshold population CSV from input folder "
                    f"instead of rebuilding it in the output folder: {source_population_csv}"
                )
                return source_population_csv
            if source_population_csv.exists() or source_manifest_json.exists():
                log_status(
                    f"Input folder precomputed CSV is not reusable ({reason}): {Path(run_root)}"
                )

    all_pair_meta: list[tuple[Path, tuning_weinan.PairId, Path]] = []
    selected_set = set(selected_unit_keys)
    for run_root in run_roots:
        pair_meta = discover_threshold_pair_meta(run_root)
        if not pair_meta:
            log_status(f"No sgch*_thr*uV threshold-pair folders found under: {run_root}")
            continue
        for pair, pair_dir in pair_meta:
            unit_key = threshold_unit_key_from_dir(pair, pair_dir)
            if selected_set and unit_key not in selected_set:
                continue
            all_pair_meta.append((run_root, pair, pair_dir))
    if not all_pair_meta:
        raise RuntimeError(
            "No sgch*_thr*uV threshold-pair folders were found under any input folder: "
            f"{[str(path) for path in run_roots]} with selected units: {sorted(selected_set)}"
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
                parsed_session_ordinal = safe_int(row_dict.get("session_ordinal"))
                if parsed_session_ordinal is None:
                    parsed_session_ordinal = int(session_ordinal_lookup.get(recording_name, 1))
                local_session_ordinal = int(parsed_session_ordinal)
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
                "selected_threshold_unit_keys": list(selected_unit_keys),
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
    *,
    selected_unit_keys: tuple[str, ...] = (),
) -> tuple[bool, str]:
    if not population_csv.exists():
        return False, "population CSV is missing"
    if not manifest_json.exists():
        return False, "manifest is missing"
    try:
        manifest = json.loads(manifest_json.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return False, f"manifest could not be read: {exc}"

    requested_roots = [str(Path(path).resolve()) for path in run_roots]
    manifest_roots = [str(Path(path).resolve()) for path in manifest.get("input_run_roots", [])]
    portable_root = manifest.get("portable_threshold_run_root", None)
    portable_roots = [str(Path(portable_root).resolve())] if portable_root else []
    if manifest_roots != requested_roots and portable_roots != requested_roots:
        return False, "manifest input_run_roots differ from this run"
    manifest_selected = tuple(str(value) for value in manifest.get("selected_threshold_unit_keys", []))
    if manifest_selected != tuple(selected_unit_keys):
        return False, "manifest selected_threshold_unit_keys differ from this run"
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
    overlay_root = root / "baseline_phase_overlay_polar_time_of_day_units"
    if overlay_root.is_dir():
        polar_files.extend(str(path) for path in sorted(overlay_root.rglob("*")) if path.is_file())
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


def write_tuning_weinan_usage_summary_from_threshold_minutes(
    run_root: Path,
    output_root: Path,
    *,
    selected_unit_keys: tuple[str, ...] = (),
) -> None:
    pair_meta = discover_threshold_pair_meta(run_root)
    selected_set = set(selected_unit_keys)
    if selected_set:
        pair_meta = [
            (pair, pair_dir)
            for pair, pair_dir in pair_meta
            if threshold_unit_key_from_dir(pair, pair_dir) in selected_set
        ]
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


def run_tuning_weinan_combined(
    run_roots: tuple[Path, ...],
    output_root: Path,
    *,
    only_polar: bool = True,
    selected_unit_keys: tuple[str, ...] = (),
) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_status(f"Starting combined Tuning_Weinan analysis for {len(run_roots)} threshold run(s)")
    log_status(f"Combined Tuning_Weinan output folder: {output_root}")

    grouped_series: dict[str, dict] = {}
    selected_set = set(selected_unit_keys)
    total_pair_dirs = 0
    for run_root in run_roots:
        pair_meta = discover_threshold_pair_meta(run_root)
        total_pair_dirs += len(pair_meta)
        log_status(f"Combined tuning input {Path(run_root).name}: {len(pair_meta)} threshold unit folder(s)")
        for pair, pair_dir in pair_meta:
            unit_key = threshold_unit_key_from_dir(pair, pair_dir)
            if selected_set and unit_key not in selected_set:
                continue
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
                "selected_threshold_unit_keys": list(selected_unit_keys),
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
        "selected_threshold_unit_keys": list(selected_unit_keys),
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


TUNING_PHASE_STYLES = {
    "baseline": {"color": "black", "linestyle": "-", "linewidth": 2.4, "alpha": 0.95},
    "sham": {"color": "tab:orange", "linestyle": "--", "linewidth": 1.9, "alpha": 0.9},
    "drug": {"color": "tab:red", "linestyle": ":", "linewidth": 2.1, "alpha": 0.9},
    "sham_saline": {"color": "tab:blue", "linestyle": "--", "linewidth": 1.8, "alpha": 0.9},
    "drug_saline": {"color": "tab:blue", "linestyle": ":", "linewidth": 2.0, "alpha": 0.9},
    "sham_caf": {"color": "tab:purple", "linestyle": "--", "linewidth": 1.8, "alpha": 0.9},
    "drug_caf": {"color": "tab:purple", "linestyle": ":", "linewidth": 2.0, "alpha": 0.9},
}


def close_polar_curve(theta: np.ndarray, radial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(theta) & np.isfinite(radial)
    if not np.any(finite):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    theta_finite = theta[finite]
    radial_finite = radial[finite]
    return np.r_[theta_finite, theta_finite[0]], np.r_[radial_finite, radial_finite[0]]


def render_phase_overlay_polar_series(
    series_name: str,
    xs: np.ndarray,
    amplitude: np.ndarray,
    firing_rate: np.ndarray,
    phases: np.ndarray,
    phase_order: list[str],
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
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
    phases = np.asarray(phases).astype(str)
    available_phases = [phase for phase in phase_order if np.any(phases == phase)]
    if "baseline" not in available_phases:
        log_status(f"Skipping baseline-reference tuning overlay for {series_name}; no baseline points.")
        return output_paths

    for metric_tag, values, radial_label in metrics:
        values = np.asarray(values, dtype=float)
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
                for phase in available_phases:
                    phase_mask = phases == phase
                    centers, mean, var, std, count = tuning_weinan.bin_by_time_of_day(
                        np.asarray(xs, dtype=float)[phase_mask],
                        values[phase_mask],
                        bin_minutes=bin_minutes,
                    )
                    radial = mean if radial_stat == "mean" else var
                    theta = 2.0 * np.pi * centers / 24.0
                    theta_closed, radial_closed = close_polar_curve(theta, radial)
                    if theta_closed.size == 0:
                        continue
                    style = TUNING_PHASE_STYLES.get(
                        phase,
                        {"color": "0.3", "linestyle": "-.", "linewidth": 1.5, "alpha": 0.85},
                    )
                    label = PHASE_LABELS.get(phase, phase)
                    ax.plot(
                        theta_closed,
                        radial_closed,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        alpha=style["alpha"],
                        label=label,
                    )
                    if phase == "baseline" and radial_stat == "mean":
                        band_low = mean - std
                        band_high = mean + std
                        theta_band, low_closed = close_polar_curve(theta, band_low)
                        _, high_closed = close_polar_curve(theta, band_high)
                        if theta_band.size and low_closed.size == high_closed.size:
                            ax.fill_between(
                                theta_band,
                                low_closed,
                                high_closed,
                                color="black",
                                alpha=0.12,
                                label="baseline mean +/- std",
                            )
                    prefix = f"{bin_tag}_{phase}"
                    plot_data[f"{prefix}_bin_center_hour"] = centers.astype(np.float32)
                    plot_data[f"{prefix}_mean"] = mean.astype(np.float32)
                    plot_data[f"{prefix}_variance"] = var.astype(np.float32)
                    plot_data[f"{prefix}_std"] = std.astype(np.float32)
                    plot_data[f"{prefix}_count"] = count.astype(np.int32)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_xticks(2.0 * np.pi * np.array([0, 3, 6, 9, 12, 15, 18, 21]) / 24.0)
                ax.set_xticklabels(["00", "03", "06", "09", "12", "15", "18", "21"])
                ax.set_title(f"{metric_tag} {view_tag} {bin_tag}\nbaseline reference, phase overlays", fontsize=10)
                ax.set_rlabel_position(135)
                ax.grid(True, alpha=0.35)
                ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.14), fontsize=8, frameon=True)

            fig.suptitle(
                f"{series_name}: baseline-reference phase tuning ({metric_tag} {view_tag})",
                fontsize=14,
            )
            fig.tight_layout(rect=[0, 0.02, 1, 0.94])
            out_png = out_dir / f"baselinePhaseOverlay_polarTimeOfDay_{metric_tag}_{view_tag}_1min_1hr_2hr.png"
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            output_paths.append(out_png)

            out_npz = out_dir / f"baselinePhaseOverlay_polarTimeOfDay_{metric_tag}_{view_tag}_1min_1hr_2hr_plotData.npz"
            np.savez_compressed(
                str(out_npz),
                series_name=np.asarray([series_name]),
                metric=np.asarray([metric_tag]),
                radial_label=np.asarray([view_radial_label]),
                radial_statistic=np.asarray([radial_stat]),
                phase_order=np.asarray(available_phases),
                **plot_data,
            )
            output_paths.append(out_npz)
    return output_paths


def run_tuning_baseline_phase_overlay(
    run_roots: tuple[Path, ...],
    output_root: Path,
    schedule: dict,
    *,
    selected_unit_keys: tuple[str, ...] = (),
) -> dict:
    output_root = Path(output_root)
    polar_root = output_root / "baseline_phase_overlay_polar_time_of_day_units"
    polar_root.mkdir(parents=True, exist_ok=True)
    phase_order = [phase for phase in schedule.get("phase_order", PHASE_ORDER) if str(phase)]
    grouped_series: dict[str, dict] = {}
    selected_set = set(selected_unit_keys)
    total_pair_dirs = 0
    for run_root in run_roots:
        pair_meta = discover_threshold_pair_meta(run_root)
        total_pair_dirs += len(pair_meta)
        for pair, pair_dir in pair_meta:
            unit_key = threshold_unit_key_from_dir(pair, pair_dir)
            if selected_set and unit_key not in selected_set:
                continue
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
                },
            )
            xs, amplitude, _amplitude_5min, firing, _labels = load_weinan_series_from_pair_dir(pair_dir)
            entry["pair_dirs"].append(str(pair_dir))
            entry["input_runs"].append(str(Path(run_root).resolve()))
            entry["xs"].append(xs)
            entry["amplitude"].append(amplitude)
            entry["firing"].append(firing)

    output_files: list[str] = []
    summary_rows: list[dict] = []
    for unit_index, entry in enumerate(
        sorted(grouped_series.values(), key=lambda row: (int(row["sg_ch"]), float(row["threshold_uv"]), row["unit_key"])),
        start=1,
    ):
        xs = np.concatenate(entry["xs"]).astype(float)
        amplitude = np.concatenate(entry["amplitude"]).astype(float)
        firing = np.concatenate(entry["firing"]).astype(float)
        order = np.argsort(xs)
        xs = xs[order]
        amplitude = amplitude[order]
        firing = firing[order]
        datetimes = pd.Series(tuning_weinan._epoch_min_to_datetime(xs))
        phases = assign_injection_phase_for_times(datetimes, schedule, interval_duration=pd.Timedelta(0)).to_numpy()
        phase_counts = pd.Series(phases).astype(str).value_counts().to_dict()
        unit_key = str(entry["unit_key"])
        log_status(
            f"Rendering baseline-reference tuning overlays [{unit_index}/{len(grouped_series)}] -> {unit_key}"
        )
        unit_outputs = render_phase_overlay_polar_series(
            unit_key,
            xs,
            amplitude,
            firing,
            phases,
            phase_order,
            polar_root / unit_key,
        )
        output_files.extend(str(path) for path in unit_outputs)
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
                "phase_counts": json.dumps(phase_counts, sort_keys=True),
                "n_overlay_files": int(len(unit_outputs)),
            }
        )

    summary_table = pd.DataFrame(summary_rows)
    summary_csv = output_root / "tuning_baseline_phase_overlay_summary.csv"
    summary_json = output_root / "tuning_baseline_phase_overlay_summary.json"
    summary_table.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "input_run_roots": [str(Path(run_root).resolve()) for run_root in run_roots],
                "selected_threshold_unit_keys": list(selected_unit_keys),
                "output_root": str(output_root.resolve()),
                "analysis_scope": "baseline_reference_phase_overlay_tuning",
                "phase_order": phase_order,
                "n_input_runs": int(len(run_roots)),
                "n_pair_dirs_read": int(total_pair_dirs),
                "n_combined_series": int(len(summary_table)),
                "series": summary_table.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_files.extend([str(summary_csv), str(summary_json)])
    return {
        "status": "completed",
        "run_roots": [str(Path(run_root).resolve()) for run_root in run_roots],
        "selected_threshold_unit_keys": list(selected_unit_keys),
        "output_dir": str(output_root.resolve()),
        "analysis_scope": "baseline_reference_phase_overlay_tuning",
        "output_files": output_files,
        "verification": {
            "output_dir": str(output_root.resolve()),
            "num_output_files": len(output_files),
            "has_overlay_output_dir": polar_root.is_dir(),
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

    sham_candidates = []
    for sham_row in sham_table.itertuples(index=False):
        sham_start = pd.Timestamp(sham_row.session_start_datetime)
        following_drugs = drug_table.loc[drug_starts > sham_start]
        interval = {
            "session_name": str(sham_row.session_name),
            "session_key": str(sham_row.session_key),
            "start": sham_start.isoformat(sep=" "),
        }
        if following_drugs.empty:
            sham_end = sham_start + timedelta(hours=24)
            interval.update(
                {
                    "paired_drug_session_name": "",
                    "paired_drug_session_key": "",
                    "end": sham_end.isoformat(sep=" "),
                    "end_rule": "no_following_drug_use_24h_after_sham_start",
                }
            )
        else:
            drug_row = following_drugs.iloc[0]
            drug_start = pd.Timestamp(drug_row["session_start_datetime"])
            interval.update(
                {
                    "paired_drug_session_name": str(drug_row["session_name"]),
                    "paired_drug_session_key": str(drug_row["session_key"]),
                    "end": drug_start.isoformat(sep=" "),
                    "end_rule": "next_following_drug_start",
                }
            )
        sham_candidates.append((sham_start, interval))

    latest_sham_by_drug: dict[str, tuple[pd.Timestamp, dict]] = {}
    sham_intervals = []
    for sham_start, interval in sham_candidates:
        paired_key = str(interval.get("paired_drug_session_key", "")).strip()
        if not paired_key:
            sham_intervals.append(interval)
            continue
        current = latest_sham_by_drug.get(paired_key)
        if current is None or sham_start > current[0]:
            latest_sham_by_drug[paired_key] = (sham_start, interval)
    sham_intervals.extend(interval for _sham_start, interval in latest_sham_by_drug.values())
    sham_intervals.sort(key=lambda interval: str(interval.get("start", "")))

    return {
        "label_type": "injection_phase_marker",
        "baseline_label": "baseline",
        "sham_label": "sham",
        "drug_label": "drug",
        "phase_order": ["baseline", "sham", "drug"],
        "interpretation": (
            "LDA labels remain clock_hour_of_day. Marker shapes encode baseline/sham/drug: "
            "sham from each sham recording start until the next drug recording start; drug "
            "from each drug recording start until 24 hours later; sham without a following "
            "drug ends 24 hours after sham start; drug has priority."
        ),
        "sham_intervals": sham_intervals,
        "drug_intervals": drug_intervals,
        "sham_recordings": sham_table.to_dict(orient="records"),
        "drug_recordings": drug_table.to_dict(orient="records"),
    }


def _paired_phase_intervals(
    *,
    sham_recordings: pd.DataFrame,
    drug_recordings: pd.DataFrame,
    sham_phase: str,
    drug_phase: str,
) -> tuple[list[dict], list[dict]]:
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
                "phase": drug_phase,
                "session_name": str(drug_row.session_name),
                "session_key": str(drug_row.session_key),
                "start": drug_start.isoformat(sep=" "),
                "end": (drug_start + timedelta(hours=24)).isoformat(sep=" "),
            }
        )

    sham_candidates = []
    for sham_row in sham_table.itertuples(index=False):
        sham_start = pd.Timestamp(sham_row.session_start_datetime)
        following_drugs = drug_table.loc[drug_starts > sham_start]
        interval = {
            "phase": sham_phase,
            "session_name": str(sham_row.session_name),
            "session_key": str(sham_row.session_key),
            "start": sham_start.isoformat(sep=" "),
        }
        if following_drugs.empty:
            sham_end = sham_start + timedelta(hours=24)
            interval.update(
                {
                    "paired_drug_session_name": "",
                    "paired_drug_session_key": "",
                    "end": sham_end.isoformat(sep=" "),
                    "end_rule": "no_following_drug_use_24h_after_sham_start",
                }
            )
        else:
            drug_row = following_drugs.iloc[0]
            drug_start = pd.Timestamp(drug_row["session_start_datetime"])
            interval.update(
                {
                    "paired_drug_session_name": str(drug_row["session_name"]),
                    "paired_drug_session_key": str(drug_row["session_key"]),
                    "end": drug_start.isoformat(sep=" "),
                    "end_rule": "next_following_drug_start",
                }
            )
        sham_candidates.append((sham_start, interval))
    latest_sham_by_drug: dict[str, tuple[pd.Timestamp, dict]] = {}
    sham_intervals = []
    for sham_start, interval in sham_candidates:
        paired_key = str(interval.get("paired_drug_session_key", "")).strip()
        if not paired_key:
            sham_intervals.append(interval)
            continue
        current = latest_sham_by_drug.get(paired_key)
        if current is None or sham_start > current[0]:
            latest_sham_by_drug[paired_key] = (sham_start, interval)
    sham_intervals.extend(interval for _sham_start, interval in latest_sham_by_drug.values())
    sham_intervals.sort(key=lambda interval: str(interval.get("start", "")))
    return sham_intervals, drug_intervals


def build_treatment_phase_schedule(
    *,
    saline_sham_recordings: pd.DataFrame,
    saline_drug_recordings: pd.DataFrame,
    caffeine_sham_recordings: pd.DataFrame,
    caffeine_drug_recordings: pd.DataFrame,
) -> dict:
    if (
        saline_sham_recordings.empty
        or saline_drug_recordings.empty
        or caffeine_sham_recordings.empty
        or caffeine_drug_recordings.empty
    ):
        raise ValueError(
            "Treatment marker mode requires saline sham, saline drug, caffeine sham, "
            "and caffeine drug recording selections."
        )

    saline_sham, saline_drug = _paired_phase_intervals(
        sham_recordings=saline_sham_recordings,
        drug_recordings=saline_drug_recordings,
        sham_phase="sham_saline",
        drug_phase="drug_saline",
    )
    caffeine_sham, caffeine_drug = _paired_phase_intervals(
        sham_recordings=caffeine_sham_recordings,
        drug_recordings=caffeine_drug_recordings,
        sham_phase="sham_caf",
        drug_phase="drug_caf",
    )
    phase_intervals = saline_sham + caffeine_sham + saline_drug + caffeine_drug
    return {
        "label_type": "treatment_phase_marker",
        "baseline_label": "baseline",
        "phase_order": [
            "baseline",
            "sham_saline",
            "drug_saline",
            "sham_caf",
            "drug_caf",
        ],
        "interpretation": (
            "LDA labels remain clock_hour_of_day. Marker shapes encode baseline, sham saline, "
            "drug saline, sham caffeine, and drug caffeine. Sham intervals run from each sham "
            "recording start until the next matching drug recording start, or 24 hours after "
            "sham start if no later matching drug is selected. Drug intervals run from each "
            "drug recording start until 24 hours later. Drug intervals have priority if "
            "intervals overlap."
        ),
        "phase_intervals": phase_intervals,
        "sham_saline_intervals": saline_sham,
        "drug_saline_intervals": saline_drug,
        "sham_caf_intervals": caffeine_sham,
        "drug_caf_intervals": caffeine_drug,
        "saline_sham_intervals": saline_sham,
        "saline_drug_intervals": saline_drug,
        "caffeine_sham_intervals": caffeine_sham,
        "caffeine_drug_intervals": caffeine_drug,
        "saline_sham_recordings": saline_sham_recordings.to_dict(orient="records"),
        "saline_drug_recordings": saline_drug_recordings.to_dict(orient="records"),
        "caffeine_sham_recordings": caffeine_sham_recordings.to_dict(orient="records"),
        "caffeine_drug_recordings": caffeine_drug_recordings.to_dict(orient="records"),
    }


def collect_injection_phase_schedule(population_csv: Path, config: PipelineConfig, output_dir: Path) -> dict | None:
    if config.analysis_mode not in {"sham_drug_markers", "treatment_markers"}:
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
    marker_selection_label = "treatment phase" if config.analysis_mode == "treatment_markers" else "sham/drug"
    print(f"\nAvailable threshold recordings for {marker_selection_label} marker selection:", flush=True)
    print(recording_table[available_columns].to_string(index=False), flush=True)

    if config.analysis_mode == "treatment_markers":
        saline_sham_tokens = config.saline_sham_sessions
        saline_drug_tokens = config.saline_drug_sessions
        caffeine_sham_tokens = config.caffeine_sham_sessions
        caffeine_drug_tokens = config.caffeine_drug_sessions
        if not saline_sham_tokens:
            if config.non_interactive:
                raise ValueError("Non-interactive treatment marker mode requires --saline-sham-sessions.")
            saline_sham_tokens = parse_token_list(
                input("\nEnter saline sham recording_id(s), recording name(s), or tokens, separated by commas: ")
            )
        if not saline_drug_tokens:
            if config.non_interactive:
                raise ValueError("Non-interactive treatment marker mode requires --saline-drug-sessions.")
            saline_drug_tokens = parse_token_list(
                input("Enter saline drug recording_id(s), recording name(s), or tokens, separated by commas: ")
            )
        if not caffeine_sham_tokens:
            if config.non_interactive:
                raise ValueError("Non-interactive treatment marker mode requires --caffeine-sham-sessions.")
            caffeine_sham_tokens = parse_token_list(
                input("Enter caffeine sham recording_id(s), recording name(s), or tokens, separated by commas: ")
            )
        if not caffeine_drug_tokens:
            if config.non_interactive:
                raise ValueError("Non-interactive treatment marker mode requires --caffeine-drug-sessions.")
            caffeine_drug_tokens = parse_token_list(
                input("Enter caffeine drug recording_id(s), recording name(s), or tokens, separated by commas: ")
            )
        schedule = build_treatment_phase_schedule(
            saline_sham_recordings=select_recordings_from_tokens(recording_table, saline_sham_tokens),
            saline_drug_recordings=select_recordings_from_tokens(recording_table, saline_drug_tokens),
            caffeine_sham_recordings=select_recordings_from_tokens(recording_table, caffeine_sham_tokens),
            caffeine_drug_recordings=select_recordings_from_tokens(recording_table, caffeine_drug_tokens),
        )
        print("\nTreatment marker interpretation:", flush=True)
        print(f"  {schedule['interpretation']}", flush=True)
        for phase in schedule["phase_order"]:
            if phase == "baseline":
                continue
            print(f"  {PHASE_LABELS.get(phase, phase)} intervals:", flush=True)
            for interval in schedule.get(f"{phase}_intervals", []):
                paired = ""
                if str(interval.get("paired_drug_session_name", "")).strip():
                    paired = f" -> {interval['paired_drug_session_name']}"
                elif interval.get("end_rule") == "no_following_drug_use_24h_after_sham_start":
                    paired = " -> 24h after sham start"
                print(
                    f"    {interval['session_name']}{paired}: "
                    f"[{interval['start']}, {interval['end']})",
                    flush=True,
                )
        if not config.confirm_sham_drug and not config.non_interactive:
            confirm = input("Is this treatment marker setup correct? Type YES to continue: ").strip()
            if confirm.lower() not in {"yes", "y"}:
                raise RuntimeError("Treatment marker setup was not confirmed; stopping before LDA.")

        schedule_path = output_dir / "threshold_treatment_marker_schedule.json"
        schedule_path.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
        log_status(f"Saved treatment marker schedule: {schedule_path}")
        return schedule

    sham_tokens = config.sham_sessions
    drug_tokens = config.drug_sessions
    if not sham_tokens:
        if config.non_interactive:
            raise ValueError("Non-interactive sham/drug marker mode requires --sham-sessions.")
        sham_tokens = parse_token_list(
            input("\nEnter sham recording_id(s), recording name(s), or tokens, separated by commas: ")
        )
    if not drug_tokens:
        if config.non_interactive:
            raise ValueError("Non-interactive sham/drug marker mode requires --drug-sessions.")
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
        paired = (
            f" -> {interval['paired_drug_session_name']}"
            if str(interval.get("paired_drug_session_name", "")).strip()
            else " -> 24h after sham start"
        )
        print(
            f"    {interval['session_name']}{paired}: "
            f"[{interval['start']}, {interval['end']})",
            flush=True,
        )
    print("  Drug intervals:", flush=True)
    for interval in schedule["drug_intervals"]:
        print(
            f"    {interval['session_name']}: [{interval['start']}, {interval['end']})",
            flush=True,
        )
    if not config.confirm_sham_drug and not config.non_interactive:
        confirm = input("Is this interpretation correct? Type YES to continue: ").strip()
        if confirm.lower() not in {"yes", "y"}:
            raise RuntimeError("Sham/drug marker setup was not confirmed; stopping before LDA.")

    schedule_path = output_dir / "threshold_sham_drug_marker_schedule.json"
    schedule_path.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    log_status(f"Saved sham/drug marker schedule: {schedule_path}")
    return schedule


def assign_injection_phase_for_times(
    datetimes: pd.Series,
    schedule: dict,
    *,
    interval_duration: pd.Timedelta | None = None,
) -> pd.Series:
    parsed = pd.to_datetime(datetimes, errors="coerce")
    if interval_duration is None:
        interval_duration = pd.Timedelta(0)
    sample_end = parsed + interval_duration
    phases = pd.Series("baseline", index=parsed.index, dtype=object)

    def overlaps_interval(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        if interval_duration <= pd.Timedelta(0):
            return parsed.notna() & (parsed >= start) & (parsed < end)
        return parsed.notna() & (parsed < end) & (sample_end > start)

    phase_intervals = schedule.get("phase_intervals", None)
    if phase_intervals is not None:
        # Apply non-drug phases first, then drug phases so drug wins on overlap.
        ordered_intervals = sorted(
            phase_intervals,
            key=lambda interval: (
                1
                if str(interval.get("phase", "")).startswith("drug_")
                or str(interval.get("phase", "")).endswith("_drug")
                else 0
            ),
        )
        for interval in ordered_intervals:
            start = pd.Timestamp(interval["start"])
            end = pd.Timestamp(interval["end"])
            phase = str(interval.get("phase", "baseline"))
            mask = overlaps_interval(start, end)
            phases.loc[mask] = phase
        return phases

    for interval in schedule.get("sham_intervals", []) or []:
        start = pd.Timestamp(interval["start"])
        end = pd.Timestamp(interval["end"])
        mask = overlaps_interval(start, end)
        phases.loc[mask] = "sham"

    for interval in schedule.get("drug_intervals", []) or []:
        start = pd.Timestamp(interval["start"])
        end = pd.Timestamp(interval["end"])
        mask = overlaps_interval(start, end)
        phases.loc[mask] = "drug"

    return phases


def add_phase_markers_to_lda_outputs(lda_dirs: list[Path], schedule: dict) -> list[Path]:
    output_paths: list[Path] = []
    phase_order = [
        phase
        for phase in schedule.get("phase_order", PHASE_ORDER)
        if str(phase)
    ]
    plot_tag = "treatment_markers" if schedule.get("label_type") == "treatment_phase_marker" else "sham_drug_markers"
    plot_title = (
        "LDA Projection - clock hour color, injection marker shape"
        if plot_tag == "treatment_markers"
        else "LDA Projection - clock hour color, sham/drug marker shape"
    )
    for output_dir in lda_dirs:
        projection_csv = Path(output_dir) / "lda_projection.csv"
        if not projection_csv.exists():
            continue
        projection = pd.read_csv(projection_csv)
        time_column = "hour_start_datetime" if "hour_start_datetime" in projection.columns else "sample_start_datetime"
        if time_column not in projection.columns or "LD1" not in projection.columns:
            continue
        marker_interval_duration = pd.Timedelta(hours=1) if time_column == "hour_start_datetime" else pd.Timedelta(0)
        projection["injection_phase"] = assign_injection_phase_for_times(
            projection[time_column],
            schedule,
            interval_duration=marker_interval_duration,
        )
        phase_counts = projection["injection_phase"].astype(str).value_counts().to_dict()
        log_status(f"LDA marker phase counts for {output_dir}: {phase_counts}")
        projection.to_csv(projection_csv, index=False)

        y_values = (
            pd.to_numeric(projection["LD2"], errors="coerce").to_numpy(dtype=float)
            if "LD2" in projection.columns
            else np.zeros(len(projection), dtype=float)
        )
        if not np.isfinite(y_values).any():
            y_values = np.zeros(len(projection), dtype=float)
        x_values = pd.to_numeric(projection["LD1"], errors="coerce").to_numpy(dtype=float)
        hours = pd.to_numeric(projection["clock_hour_of_day"], errors="coerce").to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(10, 8))
        first_scatter = None
        for phase in phase_order:
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
        ax.set_title(plot_title)
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
            for phase in phase_order
            if phase in set(projection["injection_phase"].astype(str))
        ]
        if handles:
            ax.legend(
                handles=handles,
                title="phase",
                loc="upper center",
                bbox_to_anchor=(0.5, 1.16),
                ncol=min(len(handles), 5),
                frameon=True,
            )
        fig.tight_layout()
        out_png = Path(output_dir) / f"lda_2d_{plot_tag}.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_png)
        log_status(f"Saved phase marker LDA plot: {out_png}")

        z_values = (
            pd.to_numeric(projection["LD3"], errors="coerce").to_numpy(dtype=float)
            if "LD3" in projection.columns
            else np.zeros(len(projection), dtype=float)
        )
        if not np.isfinite(z_values).any():
            z_values = np.zeros(len(projection), dtype=float)
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")
        first_scatter = None
        for phase in phase_order:
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
        ax.set_title(plot_title.replace("LDA Projection", "LDA Projection 3D"))
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
            ax.legend(
                handles=handles,
                title="phase",
                loc="upper center",
                bbox_to_anchor=(0.5, 1.12),
                ncol=min(len(handles), 5),
                frameon=True,
            )
        fig.tight_layout()
        out_3d_png = Path(output_dir) / f"lda_3d_{plot_tag}.png"
        fig.savefig(out_3d_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_3d_png)
        log_status(f"Saved phase marker LDA 3D plot: {out_3d_png}")
    return output_paths


def feature_columns_from_population_table(table: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in table.columns
        if re.fullmatch(r"feature_\d+", str(column))
    ]


def fill_with_training_means(train_matrix: np.ndarray, all_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_matrix = np.asarray(train_matrix, dtype=float)
    all_matrix = np.asarray(all_matrix, dtype=float)
    column_means = np.nanmean(train_matrix, axis=0)
    column_means = np.where(np.isfinite(column_means), column_means, 0.0)
    train_filled = np.where(np.isfinite(train_matrix), train_matrix, column_means)
    all_filled = np.where(np.isfinite(all_matrix), all_matrix, column_means)
    return train_filled, all_filled, column_means


def baseline_only_feature_preprocessing(
    feature_columns: list[str],
    x_baseline: np.ndarray,
    x_all: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, pd.DataFrame]:
    x_baseline = np.asarray(x_baseline, dtype=float)
    x_all = np.asarray(x_all, dtype=float)
    if x_baseline.ndim != 2 or x_all.ndim != 2:
        raise ValueError("Baseline-only feature preprocessing expects 2D matrices.")
    if x_baseline.shape[1] != len(feature_columns) or x_all.shape[1] != len(feature_columns):
        raise ValueError("Feature column count does not match matrix shape.")

    baseline_finite = np.isfinite(x_baseline)
    finite_counts = baseline_finite.sum(axis=0)
    baseline_means = np.nanmean(x_baseline, axis=0)
    baseline_means = np.where(np.isfinite(baseline_means), baseline_means, 0.0)
    x_baseline_filled = np.where(baseline_finite, x_baseline, baseline_means)
    x_all_filled = np.where(np.isfinite(x_all), x_all, baseline_means)
    baseline_stds = np.nanstd(x_baseline_filled, axis=0)
    baseline_stds = np.where(np.isfinite(baseline_stds), baseline_stds, 0.0)
    positive_stds = baseline_stds[baseline_stds > 0]
    median_positive_std = float(np.nanmedian(positive_stds)) if positive_stds.size else 0.0
    min_allowed_std = max(1e-9, median_positive_std * 1e-6)
    keep_mask = (finite_counts >= 2) & (baseline_stds > min_allowed_std)

    stats = pd.DataFrame(
        {
            "feature_column": feature_columns,
            "baseline_finite_count": finite_counts.astype(int),
            "baseline_fill_mean": baseline_means,
            "baseline_std_after_fill": baseline_stds,
            "baseline_min_allowed_std": min_allowed_std,
            "kept_for_baseline_only_lda": keep_mask.astype(bool),
            "drop_reason": np.where(
                finite_counts < 2,
                "fewer_than_2_finite_baseline_values",
                np.where(
                    baseline_stds <= min_allowed_std,
                    "near_zero_baseline_variance",
                    "",
                ),
            ),
        }
    )
    kept_columns = [column for column, keep in zip(feature_columns, keep_mask) if bool(keep)]
    return kept_columns, x_baseline_filled[:, keep_mask], x_all_filled[:, keep_mask], stats


def fit_stable_baseline_lda(
    x_baseline_scaled: np.ndarray,
    baseline_labels: np.ndarray,
    *,
    n_components: int,
) -> LinearDiscriminantAnalysis:
    try:
        model = LinearDiscriminantAnalysis(
            n_components=n_components,
            solver="eigen",
            shrinkage="auto",
        )
        model.fit(x_baseline_scaled, baseline_labels)
        return model
    except Exception as exc:
        log_status(
            "Shrinkage baseline-only LDA failed; falling back to default SVD LDA. "
            f"Reason: {type(exc).__name__}: {exc}"
        )
        model = LinearDiscriminantAnalysis(n_components=n_components)
        model.fit(x_baseline_scaled, baseline_labels)
        return model


def plot_baseline_space_projection(
    projection: pd.DataFrame,
    output_path: Path,
    *,
    dimensions: int,
    phase_order: list[str],
    title: str,
) -> None:
    projection = projection.copy()
    projection["LD1"] = pd.to_numeric(projection["LD1"], errors="coerce")
    for column_name in ("LD2", "LD3"):
        if column_name in projection.columns:
            projection[column_name] = pd.to_numeric(projection[column_name], errors="coerce")
            if not np.isfinite(projection[column_name].to_numpy(dtype=float)).any():
                projection[column_name] = 0.0
        else:
            projection[column_name] = 0.0
    hours = pd.to_numeric(projection["clock_hour_of_day"], errors="coerce").to_numpy(dtype=float)
    handles = [
        Line2D(
            [0],
            [0],
            marker=PHASE_MARKERS.get(phase, "o"),
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=8,
            label=PHASE_LABELS.get(phase, phase),
        )
        for phase in phase_order
        if phase in set(projection["injection_phase"].astype(str))
    ]
    if dimensions == 3:
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Draw within-day/session trajectories first, behind the points.
    group_columns = [column for column in ("calendar_day", "session_ids", "session_names") if column in projection.columns]
    if "calendar_day" in group_columns:
        grouped = projection.sort_values(["calendar_day", "clock_hour_of_day"]).groupby(group_columns, dropna=False)
        for _group_key, group in grouped:
            group = group.sort_values("clock_hour_of_day")
            if len(group) < 2:
                continue
            if dimensions == 3:
                ax.plot(
                    group["LD1"],
                    group["LD2"],
                    group["LD3"],
                    color="0.25",
                    alpha=0.16,
                    linewidth=0.8,
                    zorder=1,
                )
            else:
                ax.plot(
                    group["LD1"],
                    group["LD2"],
                    color="0.25",
                    alpha=0.16,
                    linewidth=0.8,
                    zorder=1,
                )

    first_scatter = None
    for phase in phase_order:
        mask = projection["injection_phase"].astype(str).to_numpy() == phase
        if not np.any(mask):
            continue
        if dimensions == 3:
            scatter = ax.scatter(
                projection.loc[mask, "LD1"],
                projection.loc[mask, "LD2"],
                projection.loc[mask, "LD3"],
                c=hours[mask],
                cmap=CIRCULAR_HOUR_CMAP,
                norm=CIRCULAR_HOUR_NORM,
                marker=PHASE_MARKERS.get(phase, "o"),
                s=52,
                alpha=0.92,
                edgecolors="black",
                linewidths=0.45,
                zorder=3,
            )
        else:
            scatter = ax.scatter(
                projection.loc[mask, "LD1"],
                projection.loc[mask, "LD2"],
                c=hours[mask],
                cmap=CIRCULAR_HOUR_CMAP,
                norm=CIRCULAR_HOUR_NORM,
                marker=PHASE_MARKERS.get(phase, "o"),
                s=56,
                alpha=0.92,
                edgecolors="black",
                linewidths=0.45,
                zorder=3,
            )
        if first_scatter is None:
            first_scatter = scatter

    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    if dimensions == 3:
        ax.set_zlabel("LD3")
    ax.set_title(title)
    if dimensions == 2:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if first_scatter is not None:
        colorbar = fig.colorbar(
            first_scatter,
            ax=ax,
            fraction=0.046,
            pad=0.06 if dimensions == 3 else 0.04,
            boundaries=CIRCULAR_HOUR_BOUNDARIES,
            ticks=list(range(24)),
            spacing="proportional",
            drawedges=True,
        )
        colorbar.set_label("Hour")
    if handles:
        ax.legend(
            handles=handles,
            title="phase",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.14),
            ncol=min(len(handles), 5),
            frameon=True,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def sem_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return float("nan")
    return float(np.nanstd(values, ddof=1) / np.sqrt(values.size))


def safe_pearson_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if int(mask.sum()) < 2:
        return float("nan")
    x_values = x_values[mask]
    y_values = y_values[mask]
    if np.nanstd(x_values) == 0 or np.nanstd(y_values) == 0:
        return float("nan")
    return float(np.corrcoef(x_values, y_values)[0, 1])


def audit_lda_feature_list(output_dir: Path, output_prefix: str, projection_columns: list[str] | None = None) -> tuple[Path | None, dict]:
    output_dir = Path(output_dir)
    feature_map_path = output_dir / "lda_feature_map.csv"
    feature_columns = list(projection_columns or [])
    audit_rows: list[dict] = []
    if feature_map_path.exists():
        feature_map = pd.read_csv(feature_map_path)
        if "feature_column" in feature_map.columns:
            if feature_columns:
                feature_map = feature_map[feature_map["feature_column"].astype(str).isin(set(feature_columns))]
            for row in feature_map.itertuples(index=False):
                row_dict = row._asdict()
                searchable = " ".join(str(value) for value in row_dict.values())
                lower = searchable.lower()
                matches = [pattern for pattern in FORBIDDEN_LDA_FEATURE_PATTERNS if pattern in lower]
                audit_row = {
                    "feature_column": str(row_dict.get("feature_column", "")),
                    "feature_key": str(row_dict.get("feature_key", "")),
                    "feature_type": str(row_dict.get("feature_type", "")),
                    "forbidden_time_or_label_pattern_matches": ";".join(matches),
                    "passes_no_time_label_order_check": "__YES__" if not matches else "__NO__",
                }
                audit_rows.append(audit_row)
    else:
        for column in feature_columns:
            lower = str(column).lower()
            matches = [pattern for pattern in FORBIDDEN_LDA_FEATURE_PATTERNS if pattern in lower]
            audit_rows.append(
                {
                    "feature_column": str(column),
                    "feature_key": "",
                    "feature_type": "",
                    "forbidden_time_or_label_pattern_matches": ";".join(matches),
                    "passes_no_time_label_order_check": "__YES__" if not matches else "__NO__",
                }
            )

    if not audit_rows:
        audit_rows.append(
            {
                "feature_column": "",
                "feature_key": "",
                "feature_type": "",
                "forbidden_time_or_label_pattern_matches": "",
                "passes_no_time_label_order_check": "__NO__",
            }
        )
    audit = pd.DataFrame(audit_rows)
    suspicious = audit[audit["passes_no_time_label_order_check"] != "__YES__"]
    audit_path = output_dir / f"{output_prefix}_feature_time_label_audit.csv"
    audit.to_csv(audit_path, index=False)
    summary = {
        "feature_map_path": str(feature_map_path) if feature_map_path.exists() else "",
        "n_features_audited": int(len(audit)),
        "n_suspicious_features": int(len(suspicious)),
        "passes_no_clock_time_sample_order_feature_check": bool(suspicious.empty),
        "suspicious_feature_columns": suspicious["feature_column"].astype(str).tolist(),
        "audit_file": audit_path.name,
        "note": (
            "This audit checks LDA feature names/metadata for clock, time, date, sample, order, label, "
            "session, or recording terms. Projection metadata columns are not LDA features."
        ),
    }
    return audit_path, summary


def geometry_metrics_from_labeled_points(
    ld1: np.ndarray,
    ld2: np.ndarray,
    hour_labels: np.ndarray,
) -> dict:
    ld1 = np.asarray(ld1, dtype=float)
    ld2 = np.asarray(ld2, dtype=float)
    hour_labels = np.asarray(hour_labels, dtype=float)
    finite = np.isfinite(ld1) & np.isfinite(ld2) & np.isfinite(hour_labels)
    ld1 = ld1[finite]
    ld2 = ld2[finite]
    hour_labels = hour_labels[finite].astype(int)
    unique_hours = sorted(int(hour) for hour in np.unique(hour_labels) if 0 <= int(hour) <= 23)
    if len(unique_hours) < 2:
        return {
            "mean_within_hour_distance": float("nan"),
            "mean_between_hour_centroid_distance": float("nan"),
            "separation_ratio": float("nan"),
            "centroid_distance_vs_circular_time_difference_correlation": float("nan"),
        }
    centroids: dict[int, np.ndarray] = {}
    within_distances: list[float] = []
    for hour in unique_hours:
        mask = hour_labels == hour
        points = np.column_stack([ld1[mask], ld2[mask]])
        centroid = np.nanmean(points, axis=0)
        centroids[hour] = centroid
        within_distances.extend(np.linalg.norm(points - centroid, axis=1).astype(float).tolist())

    pair_distances: list[float] = []
    pair_deltas: list[int] = []
    for index, h1 in enumerate(unique_hours):
        for h2 in unique_hours[index + 1:]:
            pair_distances.append(float(np.linalg.norm(centroids[h1] - centroids[h2])))
            pair_deltas.append(int(min(abs(h1 - h2), 24 - abs(h1 - h2))))
    mean_within = float(np.nanmean(within_distances)) if within_distances else float("nan")
    mean_between = float(np.nanmean(pair_distances)) if pair_distances else float("nan")
    ratio = (
        float(mean_between / mean_within)
        if np.isfinite(mean_between) and np.isfinite(mean_within) and mean_within > 0
        else float("nan")
    )
    corr = safe_pearson_correlation(np.asarray(pair_deltas, dtype=float), np.asarray(pair_distances, dtype=float))
    return {
        "mean_within_hour_distance": mean_within,
        "mean_between_hour_centroid_distance": mean_between,
        "separation_ratio": ratio,
        "centroid_distance_vs_circular_time_difference_correlation": corr,
    }


def permutation_geometry_control(
    baseline: pd.DataFrame,
    *,
    n_permutations: int = BASELINE_GEOMETRY_N_PERMUTATIONS,
    random_seed: int = BASELINE_GEOMETRY_PERMUTATION_SEED,
) -> tuple[pd.DataFrame, dict]:
    ld1 = baseline["LD1"].to_numpy(dtype=float)
    ld2 = baseline["LD2"].to_numpy(dtype=float)
    labels = baseline["clock_hour_of_day"].to_numpy(dtype=int)
    observed = geometry_metrics_from_labeled_points(ld1, ld2, labels)
    rng = np.random.default_rng(int(random_seed))
    rows = []
    for permutation_index in range(1, int(n_permutations) + 1):
        shuffled = rng.permutation(labels)
        metrics = geometry_metrics_from_labeled_points(ld1, ld2, shuffled)
        metrics["permutation_index"] = permutation_index
        rows.append(metrics)
    permutation_table = pd.DataFrame(rows)

    def empirical_p_greater(metric_name: str) -> float:
        observed_value = float(observed.get(metric_name, float("nan")))
        values = pd.to_numeric(permutation_table[metric_name], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not np.isfinite(observed_value) or values.size == 0:
            return float("nan")
        return float((np.count_nonzero(values >= observed_value) + 1) / (values.size + 1))

    def empirical_p_abs(metric_name: str) -> float:
        observed_value = float(observed.get(metric_name, float("nan")))
        values = pd.to_numeric(permutation_table[metric_name], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not np.isfinite(observed_value) or values.size == 0:
            return float("nan")
        return float((np.count_nonzero(np.abs(values) >= abs(observed_value)) + 1) / (values.size + 1))

    summary = {
        "n_permutations": int(n_permutations),
        "random_seed": int(random_seed),
        "observed_separation_ratio": observed["separation_ratio"],
        "observed_distance_time_correlation": observed[
            "centroid_distance_vs_circular_time_difference_correlation"
        ],
        "null_separation_ratio_mean": float(np.nanmean(permutation_table["separation_ratio"])),
        "null_separation_ratio_sd": float(np.nanstd(permutation_table["separation_ratio"])),
        "null_distance_time_correlation_mean": float(
            np.nanmean(permutation_table["centroid_distance_vs_circular_time_difference_correlation"])
        ),
        "null_distance_time_correlation_sd": float(
            np.nanstd(permutation_table["centroid_distance_vs_circular_time_difference_correlation"])
        ),
        "p_separation_ratio_greater_equal_observed": empirical_p_greater("separation_ratio"),
        "p_abs_distance_time_correlation_greater_equal_observed_abs": empirical_p_abs(
            "centroid_distance_vs_circular_time_difference_correlation"
        ),
    }
    return permutation_table, summary


def parse_bool_series(series: pd.Series, *, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "nan", "none"}
    parsed = normalized.map(
        lambda value: True if value in true_values else False if value in false_values else bool(default)
    )
    return parsed.astype(bool)


def add_baseline_geometry_validation_outputs(
    projection: pd.DataFrame,
    output_dir: Path,
    *,
    prefix: str = "baseline_geometry",
    label_centroids: bool = True,
) -> list[Path]:
    output_paths: list[Path] = []
    required_columns = {"LD1", "clock_hour_of_day"}
    if not required_columns.issubset(set(projection.columns)):
        log_status(f"Skipping baseline geometry validation for {output_dir}; missing LD1/hour columns.")
        return output_paths

    table = projection.copy()
    table["clock_hour_of_day"] = pd.to_numeric(table["clock_hour_of_day"], errors="coerce")
    table["LD1"] = pd.to_numeric(table["LD1"], errors="coerce")
    if "LD2" in table.columns:
        table["LD2"] = pd.to_numeric(table["LD2"], errors="coerce")
        if not np.isfinite(table["LD2"].to_numpy(dtype=float)).any():
            table["LD2"] = 0.0
    else:
        table["LD2"] = 0.0
    if "baseline_only_fit_used_for_training" in table.columns:
        baseline_mask = parse_bool_series(table["baseline_only_fit_used_for_training"])
    elif "injection_phase" in table.columns:
        baseline_mask = table["injection_phase"].astype(str) == "baseline"
    else:
        baseline_mask = pd.Series(True, index=table.index)
    baseline = table.loc[
        baseline_mask
        & table["clock_hour_of_day"].notna()
        & table["clock_hour_of_day"].between(0, 23)
        & table["LD1"].notna()
        & table["LD2"].notna()
    ].copy()
    if baseline.empty:
        log_status(f"Skipping baseline geometry validation for {output_dir}; no baseline LDA samples found.")
        return output_paths
    baseline["clock_hour_of_day"] = baseline["clock_hour_of_day"].astype(int)

    centroid_rows: list[dict] = []
    for hour, group in baseline.groupby("clock_hour_of_day"):
        centroid_rows.append(
            {
                "clock_hour": int(hour),
                "LD1_centroid": float(group["LD1"].mean()),
                "LD2_centroid": float(group["LD2"].mean()),
                "n_samples": int(len(group)),
            }
        )
    centroids = pd.DataFrame(centroid_rows).sort_values("clock_hour").reset_index(drop=True)
    if len(centroids) < 2:
        log_status(f"Skipping baseline geometry validation for {output_dir}; fewer than two baseline hour centroids.")
        return output_paths

    centroid_lookup = {
        int(row.clock_hour): np.array([float(row.LD1_centroid), float(row.LD2_centroid)], dtype=float)
        for row in centroids.itertuples(index=False)
    }
    sample_distances: list[float] = []
    for row in baseline.itertuples(index=False):
        point = np.array([float(getattr(row, "LD1")), float(getattr(row, "LD2"))], dtype=float)
        centroid = centroid_lookup[int(getattr(row, "clock_hour_of_day"))]
        sample_distances.append(float(np.linalg.norm(point - centroid)))
    baseline["distance_to_own_hour_centroid"] = sample_distances
    compactness = (
        baseline.groupby("clock_hour_of_day")["distance_to_own_hour_centroid"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "clock_hour_of_day": "clock_hour",
                "mean": "mean_distance_to_own_centroid",
                "std": "sd_distance_to_own_centroid",
                "count": "n_samples",
            }
        )
    )
    compactness["sem_distance_to_own_centroid"] = compactness.apply(
        lambda row: float(row["sd_distance_to_own_centroid"] / np.sqrt(row["n_samples"]))
        if int(row["n_samples"]) > 1 and np.isfinite(row["sd_distance_to_own_centroid"])
        else float("nan"),
        axis=1,
    )

    pair_rows: list[dict] = []
    distance_matrix = np.full((24, 24), np.nan, dtype=float)
    available_hours = [int(hour) for hour in centroids["clock_hour"].tolist()]
    for h1 in available_hours:
        distance_matrix[h1, h1] = 0.0
    for index, h1 in enumerate(available_hours):
        for h2 in available_hours[index + 1:]:
            distance = float(np.linalg.norm(centroid_lookup[h1] - centroid_lookup[h2]))
            delta_h = int(min(abs(h1 - h2), 24 - abs(h1 - h2)))
            distance_matrix[h1, h2] = distance
            distance_matrix[h2, h1] = distance
            pair_rows.append(
                {
                    "hour_1": int(h1),
                    "hour_2": int(h2),
                    "circular_delta_h": int(delta_h),
                    "centroid_distance": distance,
                }
            )
    pair_distances = pd.DataFrame(pair_rows)
    if not pair_distances.empty:
        distance_by_delta = (
            pair_distances.groupby("circular_delta_h")["centroid_distance"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "mean_centroid_distance",
                    "std": "sd_centroid_distance",
                    "count": "n_pairs",
                }
            )
        )
        distance_by_delta["sem_centroid_distance"] = distance_by_delta.apply(
            lambda row: float(row["sd_centroid_distance"] / np.sqrt(row["n_pairs"]))
            if int(row["n_pairs"]) > 1 and np.isfinite(row["sd_centroid_distance"])
            else float("nan"),
            axis=1,
        )
    else:
        distance_by_delta = pd.DataFrame(
            columns=[
                "circular_delta_h",
                "mean_centroid_distance",
                "sd_centroid_distance",
                "n_pairs",
                "sem_centroid_distance",
            ]
        )

    mean_within = float(baseline["distance_to_own_hour_centroid"].mean())
    mean_between = float(pair_distances["centroid_distance"].mean()) if not pair_distances.empty else float("nan")
    separation_ratio = (
        float(mean_between / mean_within)
        if np.isfinite(mean_between) and np.isfinite(mean_within) and mean_within > 0
        else float("nan")
    )
    time_distance_corr = (
        safe_pearson_correlation(
            pair_distances["circular_delta_h"].to_numpy(dtype=float),
            pair_distances["centroid_distance"].to_numpy(dtype=float),
        )
        if not pair_distances.empty
        else float("nan")
    )

    output_dir = Path(output_dir)
    audit_path, feature_audit_summary = audit_lda_feature_list(output_dir, prefix)
    if audit_path is not None:
        output_paths.append(audit_path)
    permutation_table, permutation_summary = permutation_geometry_control(baseline)
    permutation_path = output_dir / f"{prefix}_permutation_random_label_control.csv"
    permutation_table.to_csv(permutation_path, index=False)
    output_paths.append(permutation_path)
    centroid_path = output_dir / f"{prefix}_centroids.csv"
    centroids.to_csv(centroid_path, index=False)
    output_paths.append(centroid_path)
    within_path = output_dir / f"{prefix}_within_hour_distances.csv"
    baseline.to_csv(within_path, index=False)
    output_paths.append(within_path)
    compactness_path = output_dir / f"{prefix}_compactness_by_hour.csv"
    compactness.to_csv(compactness_path, index=False)
    output_paths.append(compactness_path)
    pair_path = output_dir / f"{prefix}_centroid_pair_distances.csv"
    pair_distances.to_csv(pair_path, index=False)
    output_paths.append(pair_path)
    delta_path = output_dir / f"{prefix}_distance_by_circular_delta.csv"
    distance_by_delta.to_csv(delta_path, index=False)
    output_paths.append(delta_path)

    summary = {
        "mean_within_hour_distance": mean_within,
        "mean_between_hour_centroid_distance": mean_between,
        "separation_ratio": separation_ratio,
        "centroid_distance_vs_circular_time_difference_correlation": time_distance_corr,
        "n_baseline_samples": int(len(baseline)),
        "n_clock_hour_centroids": int(len(centroids)),
        "n_centroid_pairs": int(len(pair_distances)),
        "projection_dimensions": ["LD1", "LD2"],
        "feature_audit": feature_audit_summary,
        "random_label_permutation_control": permutation_summary,
    }
    summary_path = output_dir / f"{prefix}_summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_paths.append(summary_path)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax_projection = axes[0, 0]
    scatter = ax_projection.scatter(
        baseline["LD1"],
        baseline["LD2"],
        c=baseline["clock_hour_of_day"],
        cmap=CIRCULAR_HOUR_CMAP,
        norm=CIRCULAR_HOUR_NORM,
        s=22,
        alpha=0.28,
        edgecolors="none",
    )
    ordered_centroids = centroids.sort_values("clock_hour")
    ax_projection.scatter(
        ordered_centroids["LD1_centroid"],
        ordered_centroids["LD2_centroid"],
        c=ordered_centroids["clock_hour"],
        cmap=CIRCULAR_HOUR_CMAP,
        norm=CIRCULAR_HOUR_NORM,
        s=145,
        edgecolors="black",
        linewidths=1.25,
        zorder=4,
    )
    loop_centroids = ordered_centroids.copy()
    if len(loop_centroids) > 1:
        loop_centroids = pd.concat([loop_centroids, loop_centroids.iloc[[0]]], ignore_index=True)
        ax_projection.plot(
            loop_centroids["LD1_centroid"],
            loop_centroids["LD2_centroid"],
            color="black",
            linewidth=1.0,
            alpha=0.55,
            zorder=3,
        )
    if label_centroids:
        for row in ordered_centroids.itertuples(index=False):
            ax_projection.text(
                float(row.LD1_centroid),
                float(row.LD2_centroid),
                str(int(row.clock_hour)),
                fontsize=8,
                ha="center",
                va="center",
                color="black",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
                zorder=5,
            )
    ax_projection.set_xlabel("LDA1")
    ax_projection.set_ylabel("LDA2")
    ax_projection.set_title("Baseline samples in LDA space; centroids show clock-hour labels")
    ax_projection.spines["top"].set_visible(False)
    ax_projection.spines["right"].set_visible(False)
    colorbar = fig.colorbar(
        scatter,
        ax=ax_projection,
        boundaries=CIRCULAR_HOUR_BOUNDARIES,
        ticks=list(range(24)),
        spacing="proportional",
    )
    colorbar.set_label("Clock hour")

    ax_compactness = axes[0, 1]
    ax_compactness.errorbar(
        compactness["clock_hour"],
        compactness["mean_distance_to_own_centroid"],
        yerr=compactness["sem_distance_to_own_centroid"],
        fmt="o-",
        color="black",
        ecolor="0.35",
        elinewidth=1.0,
        capsize=3,
    )
    ax_compactness.set_xticks(range(24))
    ax_compactness.set_xlim(-0.5, 23.5)
    ax_compactness.set_xlabel("Clock hour")
    ax_compactness.set_ylabel("Distance to own hour centroid")
    ax_compactness.set_title("Within-hour compactness (mean +/- SEM)")
    ax_compactness.spines["top"].set_visible(False)
    ax_compactness.spines["right"].set_visible(False)

    ax_heatmap = axes[1, 0]
    image = ax_heatmap.imshow(distance_matrix, cmap="viridis", origin="lower")
    ax_heatmap.set_xticks(range(24))
    ax_heatmap.set_yticks(range(24))
    ax_heatmap.set_xlabel("Clock hour")
    ax_heatmap.set_ylabel("Clock hour")
    ax_heatmap.set_title("Pairwise centroid distances")
    heatbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.04)
    heatbar.set_label("LDA-space distance")

    ax_delta = axes[1, 1]
    if not pair_distances.empty:
        ax_delta.scatter(
            pair_distances["circular_delta_h"],
            pair_distances["centroid_distance"],
            s=22,
            color="0.35",
            alpha=0.22,
            edgecolors="none",
        )
    if not distance_by_delta.empty:
        ax_delta.errorbar(
            distance_by_delta["circular_delta_h"],
            distance_by_delta["mean_centroid_distance"],
            yerr=distance_by_delta["sem_centroid_distance"],
            fmt="o-",
            color="black",
            ecolor="0.35",
            elinewidth=1.0,
            capsize=3,
        )
    ax_delta.set_xticks(range(1, 13))
    ax_delta.set_xlim(0.5, 12.5)
    ax_delta.set_xlabel("Circular time difference (hours)")
    ax_delta.set_ylabel("Centroid distance")
    ax_delta.set_title("Centroid distance vs circular time difference")
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)

    fig.suptitle(
        (
            "Baseline LDA Geometry Validation: clock-hour structure vs random labels\n"
            f"mean within={mean_within:.3g}, mean between={mean_between:.3g}, "
            f"ratio={separation_ratio:.3g}, corr={time_distance_corr:.3g}; "
            f"perm p(ratio)={permutation_summary['p_separation_ratio_greater_equal_observed']:.3g}, "
            f"perm p(|corr|)={permutation_summary['p_abs_distance_time_correlation_greater_equal_observed_abs']:.3g}"
        ),
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    panel_path = output_dir / f"{prefix}_validation_panel.png"
    fig.savefig(panel_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(panel_path)
    log_status(f"Saved baseline LDA geometry validation panel: {panel_path}")
    return output_paths


def add_all_sample_lda_geometry_validation_outputs(lda_dirs: list[Path]) -> list[Path]:
    output_paths: list[Path] = []
    for output_dir in lda_dirs:
        output_dir = Path(output_dir)
        projection_path = output_dir / "lda_projection.csv"
        if not projection_path.exists():
            log_status(f"Skipping all-sample LDA geometry validation for {output_dir}; missing lda_projection.csv.")
            continue
        projection = pd.read_csv(projection_path)
        if "injection_phase" not in projection.columns:
            projection["injection_phase"] = "baseline"
        projection["baseline_only_fit_used_for_training"] = True
        output_paths.extend(
            add_baseline_geometry_validation_outputs(
                projection,
                output_dir,
                prefix="lda_geometry_all_samples",
            )
        )
    return output_paths


def add_baseline_only_marker_lda_outputs(lda_dirs: list[Path], schedule: dict) -> list[Path]:
    output_paths: list[Path] = []
    phase_order = [phase for phase in schedule.get("phase_order", PHASE_ORDER) if str(phase)]
    for output_dir in lda_dirs:
        output_dir = Path(output_dir)
        raw_population_csv = output_dir / "lda_hour_raw_population_vectors.csv"
        if not raw_population_csv.exists():
            log_status(
                f"Skipping baseline-only marker LDA for {output_dir}; missing {raw_population_csv.name}. "
                "Rerun LDA with the updated LDA_weinan.py so raw hourly vectors are saved."
            )
            continue
        table = pd.read_csv(raw_population_csv)
        feature_columns = feature_columns_from_population_table(table)
        if not feature_columns:
            log_status(f"Skipping baseline-only marker LDA for {output_dir}; no feature_* columns found.")
            continue
        time_column = "hour_start_datetime" if "hour_start_datetime" in table.columns else "sample_start_datetime"
        if time_column not in table.columns:
            log_status(f"Skipping baseline-only marker LDA for {output_dir}; no sample time column found.")
            continue
        table["injection_phase"] = assign_injection_phase_for_times(
            table[time_column],
            schedule,
            interval_duration=pd.Timedelta(hours=1) if time_column == "hour_start_datetime" else pd.Timedelta(0),
        )
        labels = pd.to_numeric(table["clock_hour_of_day"], errors="coerce")
        baseline_phase_mask = (table["injection_phase"].astype(str) == "baseline") & labels.notna()
        baseline_label_counts = labels.loc[baseline_phase_mask].astype(int).value_counts()
        eligible_baseline_labels = sorted(
            int(label) for label, count in baseline_label_counts.items() if int(count) >= 2
        )
        baseline_mask = baseline_phase_mask & labels.astype("Int64").isin(eligible_baseline_labels)
        baseline_labels = labels.loc[baseline_mask].astype(int).to_numpy()
        unique_labels = sorted(pd.unique(baseline_labels).tolist())
        if len(unique_labels) < 2:
            log_status(
                f"Skipping baseline-only marker LDA for {output_dir}; baseline has fewer than two clock-hour "
                "labels with at least two samples each."
            )
            continue
        if len(baseline_labels) <= len(unique_labels):
            log_status(
                f"Skipping baseline-only marker LDA for {output_dir}; baseline needs more samples "
                "than clock-hour labels for LDA fitting."
            )
            continue
        x_all = table[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        x_baseline = x_all[baseline_mask.to_numpy()]
        kept_feature_columns, x_baseline_filled, x_all_filled, preprocessing_stats = (
            baseline_only_feature_preprocessing(feature_columns, x_baseline, x_all)
        )
        if not kept_feature_columns:
            log_status(
                f"Skipping baseline-only marker LDA for {output_dir}; no features have enough baseline variance."
            )
            preprocessing_stats.to_csv(output_dir / "baseline_space_feature_preprocessing.csv", index=False)
            continue
        scaler = StandardScaler()
        x_baseline_scaled = scaler.fit_transform(x_baseline_filled)
        x_all_scaled = scaler.transform(x_all_filled)
        n_components = max(1, min(3, len(unique_labels) - 1, x_baseline_scaled.shape[1]))
        lda_model = fit_stable_baseline_lda(
            x_baseline_scaled,
            baseline_labels,
            n_components=n_components,
        )
        transformed = lda_model.transform(x_all_scaled)
        transformed_baseline = lda_model.transform(x_baseline_scaled)

        projection = table.copy()
        for dimension_index, column_name in enumerate(("LD1", "LD2", "LD3")):
            if dimension_index < transformed.shape[1]:
                projection[column_name] = transformed[:, dimension_index]
            else:
                projection[column_name] = np.nan
        projection["baseline_only_fit_used_for_training"] = baseline_mask.to_numpy()
        projection["baseline_only_training_role"] = np.where(
            baseline_mask.to_numpy(),
            "fit_baseline",
            "project_only_" + projection["injection_phase"].astype(str),
        )
        projection_path = output_dir / "baseline_space_projection.csv"
        projection.to_csv(projection_path, index=False)
        output_paths.append(projection_path)

        preprocessing_path = output_dir / "baseline_space_feature_preprocessing.csv"
        preprocessing_stats.to_csv(preprocessing_path, index=False)
        output_paths.append(preprocessing_path)

        z_abs = np.abs(x_all_scaled)
        projection_norm_columns = [column for column in ("LD1", "LD2", "LD3") if column in projection.columns]
        projection_values = projection[projection_norm_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        projection_norm = np.linalg.norm(np.nan_to_num(projection_values, nan=0.0), axis=1)
        projection_diagnostics = projection[
            [
                column
                for column in (
                    "hour_start_datetime",
                    "sample_start_datetime",
                    "calendar_day",
                    "clock_hour_of_day",
                    "session_names",
                    "injection_phase",
                    "baseline_only_training_role",
                )
                if column in projection.columns
            ]
        ].copy()
        projection_diagnostics["max_abs_baseline_scaled_feature_z"] = (
            np.nanmax(z_abs, axis=1) if z_abs.size else np.nan
        )
        projection_diagnostics["projection_norm"] = projection_norm
        projection_diagnostics = projection_diagnostics.sort_values("projection_norm", ascending=False)
        diagnostics_path = output_dir / "baseline_space_projection_diagnostics.csv"
        projection_diagnostics.to_csv(diagnostics_path, index=False)
        output_paths.append(diagnostics_path)

        scaler_payload = pd.DataFrame(
            {
                "feature_column": kept_feature_columns,
                "baseline_fill_mean": preprocessing_stats.loc[
                    preprocessing_stats["kept_for_baseline_only_lda"].astype(bool),
                    "baseline_fill_mean",
                ].to_numpy(dtype=float),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
            }
        )
        scaler_path = output_dir / "baseline_space_scaler.csv"
        scaler_payload.to_csv(scaler_path, index=False)
        output_paths.append(scaler_path)
        baseline_feature_audit_path, baseline_feature_audit_summary = audit_lda_feature_list(
            output_dir,
            "baseline_space_retained",
            projection_columns=kept_feature_columns,
        )
        if baseline_feature_audit_path is not None:
            output_paths.append(baseline_feature_audit_path)

        summary_path = output_dir / "baseline_space_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "mode": "baseline_only_clock_hour_lda_space",
                    "important_note": (
                        "Only baseline rows with baseline_only_fit_used_for_training=true were used to fit "
                        "the fill means, scaler, and LDA model. Sham/drug rows were transformed/projected only."
                    ),
                    "training_phase": "baseline",
                    "label_type": "clock_hour_of_day",
                    "n_training_samples": int(baseline_mask.sum()),
                    "n_projected_samples": int(len(projection)),
                    "n_input_features": int(len(feature_columns)),
                    "n_features_kept_after_baseline_variance_filter": int(len(kept_feature_columns)),
                    "n_features_dropped_before_fit": int(len(feature_columns) - len(kept_feature_columns)),
                    "lda_solver": str(getattr(lda_model, "solver", "unknown")),
                    "lda_shrinkage": str(getattr(lda_model, "shrinkage", "none")),
                    "baseline_projection_range": {
                        "LD1_min": float(np.nanmin(transformed_baseline[:, 0])) if transformed_baseline.size else float("nan"),
                        "LD1_max": float(np.nanmax(transformed_baseline[:, 0])) if transformed_baseline.size else float("nan"),
                        "LD2_min": float(np.nanmin(transformed_baseline[:, 1])) if transformed_baseline.shape[1] > 1 else float("nan"),
                        "LD2_max": float(np.nanmax(transformed_baseline[:, 1])) if transformed_baseline.shape[1] > 1 else float("nan"),
                    },
                    "all_projected_range": {
                        "LD1_min": float(np.nanmin(transformed[:, 0])) if transformed.size else float("nan"),
                        "LD1_max": float(np.nanmax(transformed[:, 0])) if transformed.size else float("nan"),
                        "LD2_min": float(np.nanmin(transformed[:, 1])) if transformed.shape[1] > 1 else float("nan"),
                        "LD2_max": float(np.nanmax(transformed[:, 1])) if transformed.shape[1] > 1 else float("nan"),
                    },
                    "n_baseline_clock_hour_labels": int(len(unique_labels)),
                    "baseline_clock_hour_labels": [int(value) for value in unique_labels],
                    "baseline_clock_hour_sample_counts_before_min_count_filter": {
                        str(int(label)): int(count) for label, count in baseline_label_counts.items()
                    },
                    "baseline_clock_hour_min_samples_for_fit": 2,
                    "baseline_clock_hours_excluded_from_fit_for_low_count": [
                        int(label)
                        for label, count in baseline_label_counts.items()
                        if int(count) < 2
                    ],
                    "phase_counts": projection["injection_phase"].astype(str).value_counts().to_dict(),
                    "training_role_counts": projection["baseline_only_training_role"].astype(str).value_counts().to_dict(),
                    "preprocessing_outputs": [
                        "baseline_space_feature_preprocessing.csv",
                        "baseline_space_projection_diagnostics.csv",
                        "baseline_space_retained_feature_time_label_audit.csv",
                    ],
                    "retained_feature_time_label_order_audit": baseline_feature_audit_summary,
                    "normal_lda_outputs_note": (
                        "Files named lda_2d.png, lda_3d.png, and lda_2d_sham_drug_markers.png are the normal "
                        "all-sample LDA outputs from LDA_weinan.py. Use baseline_space_*.png/csv for this "
                        "baseline-trained model."
                    ),
                    "requirements": [
                        "Near-zero-variance baseline features are dropped before fitting to avoid unstable projection leverage.",
                        "Scaler fit only on retained baseline features.",
                        "LDA fit only on retained baseline features and baseline clock-hour labels.",
                        "Baseline, sham, and drug transformed with the same fitted scaler and LDA model.",
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        output_paths.append(summary_path)

        plot_2d_path = output_dir / "baseline_space_2d.png"
        plot_baseline_space_projection(
            projection,
            plot_2d_path,
            dimensions=2,
            phase_order=phase_order,
            title="Baseline-only fit LDA space - sham/drug projected only",
        )
        output_paths.append(plot_2d_path)
        baseline_fit_projection = projection.loc[baseline_mask.to_numpy()].copy()
        baseline_fit_projection_path = output_dir / "baseline_space_fit_baseline_projection.csv"
        baseline_fit_projection.to_csv(baseline_fit_projection_path, index=False)
        output_paths.append(baseline_fit_projection_path)
        baseline_only_plot_2d_path = output_dir / "baseline_space_fit_baseline_only_2d.png"
        plot_baseline_space_projection(
            baseline_fit_projection,
            baseline_only_plot_2d_path,
            dimensions=2,
            phase_order=["baseline"],
            title="Baseline-only fit LDA space - baseline training samples only",
        )
        output_paths.append(baseline_only_plot_2d_path)
        plot_3d_path = output_dir / "baseline_space_3d.png"
        plot_baseline_space_projection(
            projection,
            plot_3d_path,
            dimensions=3,
            phase_order=phase_order,
            title="Baseline-only fit LDA space 3D - sham/drug projected only",
        )
        output_paths.append(plot_3d_path)
        baseline_only_plot_3d_path = output_dir / "baseline_space_fit_baseline_only_3d.png"
        plot_baseline_space_projection(
            baseline_fit_projection,
            baseline_only_plot_3d_path,
            dimensions=3,
            phase_order=["baseline"],
            title="Baseline-only fit LDA space 3D - baseline training samples only",
        )
        output_paths.append(baseline_only_plot_3d_path)
        output_paths.extend(add_baseline_geometry_validation_outputs(projection, output_dir))
        log_status(f"Saved baseline-only marker LDA outputs for {output_dir}")
    return output_paths


def parse_feature_modes(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return DEFAULT_LDA_FEATURE_MODES
    return tuple(part.strip().upper() for part in raw_value.split(",") if part.strip())


def select_threshold_units_by_channels(run_roots: tuple[Path, ...], channel_text: str) -> tuple[str, ...]:
    options = discover_threshold_unit_options(run_roots)
    by_channel: dict[int, list[dict]] = {}
    for row in options:
        by_channel.setdefault(int(row["sg_ch"]), []).append(row)
    if channel_text.strip().lower() == "all":
        selected_channels = set(by_channel.keys())
    else:
        selected_channels = {
            int(token.strip())
            for token in re.split(r"[;,]", channel_text)
            if token.strip()
        }
    missing = sorted(ch for ch in selected_channels if ch not in by_channel)
    if missing:
        raise ValueError(f"Selected channel(s) not found in threshold units: {missing}")
    selected = [row for row in options if int(row["sg_ch"]) in selected_channels]
    return tuple(str(row["unit_key"]) for row in selected)


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
    parser.add_argument(
        "--output-dir",
        help=(
            f"Output folder. Defaults to <run_root>/{DEFAULT_OUTPUT_SUBDIR}, or the common parent "
            "for multiple inputs. If the folder already exists, interactive runs ask for a suffix."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        help=(
            "Text to append to the output folder name if it already exists. "
            "Useful for non-interactive runs, e.g. --output-suffix channels_12_45."
        ),
    )
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
        help=(
            "Legacy threshold wrapper mode. baseline runs clock-hour LDA only; "
            "sham_drug_markers adds simple sham/drug marker-shape plots; "
            "treatment_markers adds saline/caffeine injection marker shapes."
        ),
    )
    parser.add_argument(
        "--lda-baseline-sham-drug",
        choices=["ask", "yes", "no"],
        default="ask",
        help=(
            "Auto_align_LDA_pre_tuning-style option: whether to add injection marker "
            "shapes to hour-label LDA figures. This does not change the LDA label."
        ),
    )
    parser.add_argument(
        "--baseline-only-marker-lda",
        choices=["ask", "yes", "no"],
        default="ask",
        help=(
            "When injection marker shapes are enabled, optionally fit a second LDA space "
            "using baseline samples only. The scaler and LDA are fit only on baseline "
            "clock-hour labels, then baseline/sham/drug samples are projected into that space."
        ),
    )
    parser.add_argument(
        "--tuning-baseline-phase-overlays",
        choices=["ask", "yes", "no"],
        default="ask",
        help=(
            "When injection marker shapes are enabled, optionally add tuning polar plots where "
            "baseline defines the reference mean/variation and sham/drug phases are overlaid "
            "as separate line styles."
        ),
    )
    parser.add_argument(
        "--lda-injection-label-mode",
        choices=["simple", "saline-caf"],
        default="simple",
        help=(
            "Marker-shape labels when --lda-baseline-sham-drug yes is used. "
            "simple uses baseline/sham/drug; saline-caf uses baseline/sham_saline/"
            "drug_saline/sham_caf/drug_caf."
        ),
    )
    parser.add_argument("--lda-feature-modes", help="Comma-separated LDA feature modes")
    parser.add_argument("--min-firing-rate-hz", type=float, default=0.0)
    parser.add_argument("--min-minutes-per-hour", type=int, default=1)
    parser.add_argument("--min-bins-per-label", type=int, default=2)
    parser.add_argument("--cv-n-splits", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--no-zscore", action="store_true")
    parser.add_argument("--lda-channels", help="SG channel selection for LDA, e.g. '12,45,337' or 'all'.")
    parser.add_argument("--sham-sessions", help="Comma/semicolon-separated sham recording IDs, names, or tokens.")
    parser.add_argument("--drug-sessions", help="Comma/semicolon-separated drug recording IDs, names, or tokens.")
    parser.add_argument(
        "--saline-sham-sessions",
        "--lda-sham-saline-sessions",
        dest="saline_sham_sessions",
        help="Comma/semicolon-separated saline sham recording IDs, names, or tokens.",
    )
    parser.add_argument(
        "--saline-drug-sessions",
        "--lda-drug-saline-sessions",
        dest="saline_drug_sessions",
        help="Comma/semicolon-separated saline drug recording IDs, names, or tokens.",
    )
    parser.add_argument(
        "--caffeine-sham-sessions",
        "--lda-sham-caf-sessions",
        dest="caffeine_sham_sessions",
        help="Comma/semicolon-separated caffeine sham recording IDs, names, or tokens.",
    )
    parser.add_argument(
        "--caffeine-drug-sessions",
        "--lda-drug-caf-sessions",
        dest="caffeine_drug_sessions",
        help="Comma/semicolon-separated caffeine drug recording IDs, names, or tokens.",
    )
    parser.add_argument(
        "--confirm-sham-drug",
        action="store_true",
        help="Skip confirmation prompt for sham/drug marker intervals.",
    )
    parser.add_argument(
        "--confirm-treatment-markers",
        "--lda-confirm-baseline-sham-drug",
        dest="confirm_sham_drug",
        action="store_true",
        help="Skip confirmation prompt for treatment marker intervals.",
    )
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
        if bool(args.non_interactive):
            raise ValueError("Non-interactive mode requires at least one run root argument or --run-root.")
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
    if skip_lda:
        selected_threshold_unit_keys = ()
        if args.lda_channels:
            log_status("--lda-channels was provided, but LDA is skipped; using all channels for non-LDA stages.")
    elif args.lda_channels:
        selected_threshold_unit_keys = select_threshold_units_by_channels(run_roots, str(args.lda_channels))
        print("\nThreshold units selected from --lda-channels:", flush=True)
        selected_set = set(selected_threshold_unit_keys)
        for row in discover_threshold_unit_options(run_roots):
            if str(row["unit_key"]) in selected_set:
                print(
                    f"  {row['unit_key']}  |  SG channel {row['sg_ch']}  |  threshold {row['threshold_label']}",
                    flush=True,
                )
    elif bool(args.non_interactive):
        selected_threshold_unit_keys = ()
        log_status("Non-interactive mode: using all discovered threshold channels for LDA.")
    else:
        selected_threshold_unit_keys = prompt_threshold_units_for_lda(run_roots)
    marker_choice = str(args.lda_baseline_sham_drug)
    injection_label_mode = str(args.lda_injection_label_mode)
    baseline_only_marker_lda = False
    tuning_baseline_phase_overlays = False
    if marker_choice == "ask" and str(args.analysis_mode) == "baseline" and not bool(args.non_interactive):
        marker_choice = "yes" if prompt_yes_no(
            "Add injection marker shapes to hour-label LDA figures?",
            default=False,
        ) else "no"
        if marker_choice == "yes":
            injection_label_mode = (
                "saline-caf"
                if prompt_yes_no(
                    "Use separate saline/caffeine sham/drug marker labels?",
                    default=False,
                )
                else "simple"
            )

    analysis_mode = str(args.analysis_mode)
    marker_mode_requested = marker_choice == "yes" or analysis_mode in {"sham_drug_markers", "treatment_markers"}
    if marker_mode_requested:
        if marker_choice == "yes":
            analysis_mode = "treatment_markers" if injection_label_mode == "saline-caf" else "sham_drug_markers"
        baseline_only_choice = str(args.baseline_only_marker_lda)
        if skip_lda and baseline_only_choice == "yes":
            log_status("--baseline-only-marker-lda yes was provided, but LDA is skipped; ignoring it.")
        elif baseline_only_choice == "yes":
            baseline_only_marker_lda = True
        elif baseline_only_choice == "ask" and not bool(args.non_interactive) and not skip_lda:
            baseline_only_marker_lda = prompt_yes_no(
                "Fit baseline-only clock-hour LDA space and project baseline/sham/drug into it?",
                default=False,
            )
        tuning_overlay_choice = str(args.tuning_baseline_phase_overlays)
        if skip_tuning_weinan and tuning_overlay_choice == "yes":
            log_status("--tuning-baseline-phase-overlays yes was provided, but Tuning_Weinan is skipped; ignoring it.")
        elif tuning_overlay_choice == "yes":
            tuning_baseline_phase_overlays = True
        elif tuning_overlay_choice == "ask" and not bool(args.non_interactive) and not skip_tuning_weinan:
            tuning_baseline_phase_overlays = prompt_yes_no(
                "Add baseline-reference tuning plots with sham/drug phase overlays?",
                default=False,
            )
    elif marker_choice == "no":
        analysis_mode = "baseline"
    return PipelineConfig(
        run_roots=run_roots,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        output_suffix=str(args.output_suffix) if args.output_suffix else None,
        selected_threshold_unit_keys=selected_threshold_unit_keys,
        analysis_mode=analysis_mode,
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
        non_interactive=bool(args.non_interactive),
        reuse_population_csv=not bool(args.force_rebuild_population_csv),
        tuning_weinan_only_polar=not bool(args.tuning_weinan_master_plots),
        tuning_baseline_phase_overlays=bool(tuning_baseline_phase_overlays),
        sham_sessions=parse_token_list(args.sham_sessions),
        drug_sessions=parse_token_list(args.drug_sessions),
        saline_sham_sessions=parse_token_list(args.saline_sham_sessions),
        saline_drug_sessions=parse_token_list(args.saline_drug_sessions),
        caffeine_sham_sessions=parse_token_list(args.caffeine_sham_sessions),
        caffeine_drug_sessions=parse_token_list(args.caffeine_drug_sessions),
        confirm_sham_drug=bool(args.confirm_sham_drug),
        baseline_only_marker_lda=bool(baseline_only_marker_lda),
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
            selected_unit_keys=config.selected_threshold_unit_keys,
        )
    injection_schedule = None
    marker_stage_label = "treatment marker" if config.analysis_mode == "treatment_markers" else "sham/drug marker"
    needs_injection_schedule = (
        config.analysis_mode in {"sham_drug_markers", "treatment_markers"}
        and (not config.skip_lda or bool(config.tuning_baseline_phase_overlays))
    )
    if needs_injection_schedule:
        with timed_stage(f"{marker_stage_label.capitalize()} setup", timings):
            injection_schedule = collect_injection_phase_schedule(population_csv, config, output_dir)

    result = {
        "input_run_roots": [str(path.resolve()) for path in config.run_roots],
        "output_dir": str(output_dir.resolve()),
        "config": jsonable_config(config),
        "population_csv": str(population_csv.resolve()),
        "lda_output_dirs": [],
        "phase_marker_plots": [],
        "sham_drug_marker_plots": [],
        "lda_geometry_validation_outputs": [],
        "baseline_only_marker_lda_outputs": [],
        "injection_phase_schedule": injection_schedule,
        "tuning_weinan_results": [],
        "tuning_baseline_phase_overlay_results": None,
        "presentation_manifest": None,
        "timings": timings,
    }

    if config.skip_lda:
        log_status("Skipping LDA by request.")
    else:
        with timed_stage("LDA threshold clock-hour analysis", timings):
            lda_dirs = run_lda(population_csv, output_dir, config)
        result["lda_output_dirs"] = [str(path.resolve()) for path in lda_dirs]
        with timed_stage("all-sample LDA geometry validation", timings):
            geometry_outputs = add_all_sample_lda_geometry_validation_outputs(lda_dirs)
        result["lda_geometry_validation_outputs"] = [
            str(path.resolve()) for path in geometry_outputs
        ]
        if injection_schedule is not None:
            with timed_stage(f"LDA {marker_stage_label} plotting", timings):
                marker_plots = add_phase_markers_to_lda_outputs(lda_dirs, injection_schedule)
            marker_plot_paths = [str(path.resolve()) for path in marker_plots]
            result["phase_marker_plots"] = marker_plot_paths
            result["sham_drug_marker_plots"] = marker_plot_paths
            if config.baseline_only_marker_lda:
                with timed_stage("baseline-only clock-hour LDA projection", timings):
                    baseline_outputs = add_baseline_only_marker_lda_outputs(lda_dirs, injection_schedule)
                result["baseline_only_marker_lda_outputs"] = [
                    str(path.resolve()) for path in baseline_outputs
                ]

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
                        selected_unit_keys=config.selected_threshold_unit_keys,
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
        if config.tuning_baseline_phase_overlays and injection_schedule is not None:
            try:
                with timed_stage("Tuning baseline-reference phase overlays", timings):
                    result["tuning_baseline_phase_overlay_results"] = run_tuning_baseline_phase_overlay(
                        config.run_roots,
                        tuning_output_base,
                        injection_schedule,
                        selected_unit_keys=config.selected_threshold_unit_keys,
                    )
            except Exception as exc:
                log_status(f"Baseline-reference tuning overlays failed; continuing with later stages: {exc}")
                result["tuning_baseline_phase_overlay_results"] = failed_stage_payload(
                    "Tuning baseline-reference phase overlays",
                    exc,
                    run_roots=[str(path.resolve()) for path in config.run_roots],
                    output_dir=str(tuning_output_base.resolve()),
                )
        elif config.tuning_baseline_phase_overlays:
            log_status("Skipping baseline-reference tuning overlays because no sham/drug marker schedule is available.")

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
