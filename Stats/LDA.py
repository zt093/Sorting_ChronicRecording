from __future__ import annotations

"""
Population LDA analysis for aligned spike-sorting outputs.

This script reads the alignment export created by Alignment_html.py or
Alignment_days.py and builds population firing-rate vectors across aligned good
units. Each sample in the LDA input matrix represents one time bin from one
session, and each feature represents one aligned unit group. If a unit group is
absent in a session, its feature value is filled with zero for that session.

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
import numpy as np
import pandas as pd
import spikeinterface.full as si
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

DATA_PATH = None  # Leave as None to always enter the path in the terminal.
BIN_SIZE_SECONDS = 60.0
LABEL_TYPE = "clock_hour_of_day"  # "clock_hour_of_day", "session_id", "calendar_day", or "day_number"
MIN_FIRING_RATE_HZ = 0.05
APPLY_ZSCORE = True
APPLY_SMOOTHING = False
SMOOTHING_SIGMA_BINS = 1.0

MIN_SESSIONS_PER_UNIT = 48
MIN_BINS_PER_LABEL = 2
CV_N_SPLITS = 5
RANDOM_SEED = 42

OUTPUT_FOLDER_NAME = "lda_population"
ANALYZER_FOLDER_NAME = "sorting_analyzer_analysis.zarr"
OUTPUT_BASE_DIR = Path(r"S:\LDA")
ALIGNMENT_DAYS_SUMMARY_PREFIX = "alignment_days_summary"


@dataclass
class Config:
    data_path: Path | None = DATA_PATH
    bin_size_seconds: float = BIN_SIZE_SECONDS
    label_type: str = LABEL_TYPE
    min_firing_rate_hz: float = MIN_FIRING_RATE_HZ
    apply_zscore: bool = APPLY_ZSCORE
    apply_smoothing: bool = APPLY_SMOOTHING
    smoothing_sigma_bins: float = SMOOTHING_SIGMA_BINS
    min_sessions_per_unit: int = MIN_SESSIONS_PER_UNIT
    min_bins_per_label: int = MIN_BINS_PER_LABEL
    cv_n_splits: int = CV_N_SPLITS
    random_seed: int = RANDOM_SEED
    output_folder_name: str = OUTPUT_FOLDER_NAME
    analyzer_folder_name: str = ANALYZER_FOLDER_NAME
    output_base_dir: Path = OUTPUT_BASE_DIR


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
        "\nEnter the alignment export path, Alignment_days summary folder, or batch root for LDA "
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


def print_hourly_sample_pivot(metadata_table: pd.DataFrame, title: str) -> pd.DataFrame:
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


def build_gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(np.ceil(3.0 * sigma_bins)))
    x_values = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x_values**2) / (2.0 * sigma_bins**2))
    kernel /= kernel.sum()
    return kernel


def smooth_population_matrix(population_matrix: np.ndarray, sigma_bins: float) -> np.ndarray:
    kernel = build_gaussian_kernel(sigma_bins)
    smoothed = np.zeros_like(population_matrix, dtype=float)
    for feature_index in range(population_matrix.shape[1]):
        smoothed[:, feature_index] = np.convolve(
            population_matrix[:, feature_index],
            kernel,
            mode="same",
        )
    return smoothed


def zscore_population_matrix(population_matrix: np.ndarray) -> np.ndarray:
    means = population_matrix.mean(axis=0)
    stds = population_matrix.std(axis=0)
    stds[stds == 0] = 1.0
    return (population_matrix - means) / stds


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


def build_population_vectors(
    selected_units: pd.DataFrame,
    session_table: pd.DataFrame,
    analyzers: dict[str, object],
    config: Config,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    unit_feature_keys = [
        str(value)
        for value in selected_units[["final_group_key", "final_unit_id"]]
        .drop_duplicates()
        .sort_values(["final_unit_id", "final_group_key"], na_position="last")
        ["final_group_key"]
        .tolist()
    ]
    feature_index = {key: index for index, key in enumerate(unit_feature_keys)}

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
        log_status(f"Binning minute-level population for session {session_position} / {total_sessions}: {session_name}")
        analyzer = analyzers[session_key]
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
                f"is shorter than one full minute bin."
            )
            continue

        bin_edges = np.arange(n_complete_bins + 1, dtype=float) * config.bin_size_seconds
        if len(bin_edges) < 2:
            continue

        session_matrix = np.zeros((len(bin_edges) - 1, len(unit_feature_keys)), dtype=float)
        session_units = members_by_session.get(session_key, pd.DataFrame())
        skipped_invalid_units = 0
        log_status(
            f"Session '{session_name}': {len(session_units)} units, {len(bin_edges) - 1} minute bins"
        )

        for member_row in session_units.itertuples(index=False):
            if int(member_row.unit_id) not in valid_unit_ids:
                skipped_invalid_units += 1
                continue
            feature_key = str(member_row.final_group_key)
            feature_column = feature_index[feature_key]
            spike_train_samples = analyzer.sorting.get_unit_spike_train(
                unit_id=int(member_row.unit_id),
                segment_index=0,
            )
            spike_times_s = np.asarray(spike_train_samples, dtype=float) / sampling_frequency
            counts, _ = np.histogram(spike_times_s, bins=bin_edges)
            session_matrix[:, feature_column] = counts.astype(float) / config.bin_size_seconds

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
        f"Finished minute-level binning: created {population_matrix.shape[0]} samples across {len(session_table)} sessions"
    )
    return population_matrix, metadata_table, unit_feature_keys


def aggregate_minutes_to_hourly_samples(
    minute_population_matrix: np.ndarray,
    minute_metadata_table: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    if minute_population_matrix.shape[0] != len(minute_metadata_table):
        raise ValueError("Minute population matrix row count does not match the minute metadata length.")

    if minute_population_matrix.size == 0:
        raise RuntimeError("No minute-level population vectors are available for hourly aggregation.")

    feature_columns = [
        f"unit_{feature_index:04d}"
        for feature_index in range(1, minute_population_matrix.shape[1] + 1)
    ]
    minute_df = pd.concat(
        [
            minute_metadata_table.reset_index(drop=True),
            pd.DataFrame(minute_population_matrix, columns=feature_columns),
        ],
        axis=1,
    )

    hourly_vectors: list[np.ndarray] = []
    hourly_rows: list[dict] = []
    grouping_keys = ["calendar_day", "clock_hour_of_day"]
    log_status(f"Hourly aggregation grouping keys: {grouping_keys}")
    grouped = minute_df.groupby(grouping_keys, sort=True, dropna=False)
    total_groups = len(grouped)

    for group_index, ((calendar_day, clock_hour_of_day), group_table) in enumerate(grouped, start=1):
        if group_index == 1 or group_index % 100 == 0 or group_index == total_groups:
            log_status(f"Aggregating hourly samples: group {group_index} / {total_groups}")

        feature_values = group_table[feature_columns].to_numpy(dtype=float)
        hourly_vectors.append(np.nanmean(feature_values, axis=0))

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

        hourly_rows.append(
            {
                "final_sample_id": int(group_index),
                "final_sample_key": f"{calendar_day}__hour_{int(clock_hour_of_day):02d}",
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
            }
        )

    hourly_population_matrix = np.vstack(hourly_vectors)
    hourly_metadata_table = pd.DataFrame(hourly_rows).sort_values(
        ["calendar_day", "clock_hour_of_day"],
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


def filter_unit_groups_by_binned_firing_rate(
    population_matrix: np.ndarray,
    selected_units: pd.DataFrame,
    feature_keys: list[str],
    config: Config,
) -> tuple[np.ndarray, pd.DataFrame, list[str], pd.DataFrame]:
    if population_matrix.shape[1] != len(feature_keys):
        raise ValueError("Feature count does not match the population matrix width.")

    mean_binned_firing_rates = population_matrix.mean(axis=0)
    keep_mask = mean_binned_firing_rates >= config.min_firing_rate_hz
    kept_feature_keys = [
        feature_key
        for feature_key, keep_feature in zip(feature_keys, keep_mask)
        if keep_feature
    ]

    feature_stats = (
        selected_units[["final_group_key", "final_unit_id", "shank_id", "local_channel_on_shank"]]
        .drop_duplicates()
        .set_index("final_group_key")
        .reindex(feature_keys)
        .reset_index()
        .rename(columns={"index": "final_group_key"})
    )
    feature_stats["mean_binned_firing_rate_hz"] = mean_binned_firing_rates
    feature_stats["passed_min_firing_rate"] = keep_mask

    if not np.any(keep_mask):
        raise RuntimeError(
            "No aligned unit groups passed the MIN_FIRING_RATE_HZ threshold when computed from "
            "binned spike counts. Lower MIN_FIRING_RATE_HZ or increase BIN_SIZE_SECONDS."
        )

    filtered_population = population_matrix[:, keep_mask]
    filtered_selected_units = selected_units[
        selected_units["final_group_key"].isin(set(kept_feature_keys))
    ].copy()

    log_status(
        f"Kept {len(kept_feature_keys)} / {len(feature_keys)} unit groups after applying "
        f"the binned firing-rate threshold"
    )
    return filtered_population, filtered_selected_units, kept_feature_keys, feature_stats


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
    config: Config,
) -> dict:
    class_counts = pd.Series(labels).value_counts()
    smallest_class = int(class_counts.min())
    n_splits = max(2, min(config.cv_n_splits, smallest_class))
    if smallest_class < 2:
        raise RuntimeError(
            "Cross-validation needs at least two bins in each label after filtering."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )
    estimator = LinearDiscriminantAnalysis()
    predicted_labels = cross_val_predict(estimator, population_matrix, labels, cv=cv)

    unique_labels = sorted(pd.unique(labels).tolist())
    confusion = confusion_matrix(labels, predicted_labels, labels=unique_labels)

    return {
        "n_splits": int(n_splits),
        "accuracy": float(np.mean(predicted_labels == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted_labels)),
        "confusion_matrix": confusion,
        "confusion_labels": unique_labels,
        "predicted_labels": predicted_labels,
    }


def plot_lda_2d(projection: np.ndarray, metadata_table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    hours = metadata_table["clock_hour_of_day"].to_numpy(dtype=float)
    calendar_days = metadata_table["calendar_day"].astype(str).to_numpy()

    x_values = projection[:, 0]
    y_values = projection[:, 1] if projection.shape[1] >= 2 else np.zeros(len(projection))

    for calendar_day in pd.unique(calendar_days):
        mask = calendar_days == calendar_day
        session_points = metadata_table.loc[mask].sort_values("clock_hour_of_day")
        if len(session_points) < 2:
            continue
        point_indices = session_points.index.to_numpy()
        ax.plot(
            x_values[point_indices],
            y_values[point_indices],
            color="#f4a3b5",
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )

    scatter = ax.scatter(
        x_values,
        y_values,
        c=hours,
        cmap="viridis",
        s=34,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.4,
        vmin=0,
        vmax=23,
        zorder=2,
    )

    ax.set_title("Population LDA Projection (2D)")
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Hour")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_lda_3d(projection: np.ndarray, metadata_table: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    hours = metadata_table["clock_hour_of_day"].to_numpy(dtype=float)
    calendar_days = metadata_table["calendar_day"].astype(str).to_numpy()

    x_values = projection[:, 0]
    y_values = projection[:, 1] if projection.shape[1] >= 2 else np.zeros(len(projection))
    z_values = projection[:, 2] if projection.shape[1] >= 3 else np.zeros(len(projection))

    for calendar_day in pd.unique(calendar_days):
        mask = calendar_days == calendar_day
        session_points = metadata_table.loc[mask].sort_values("clock_hour_of_day")
        if len(session_points) < 2:
            continue
        point_indices = session_points.index.to_numpy()
        ax.plot(
            x_values[point_indices],
            y_values[point_indices],
            z_values[point_indices],
            color="#f4a3b5",
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )

    scatter = ax.scatter(
        x_values,
        y_values,
        z_values,
        c=hours,
        cmap="viridis",
        s=24,
        alpha=0.85,
        vmin=0,
        vmax=23,
    )

    ax.set_title("Population LDA Projection (3D)")
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_zlabel("LD3")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Hour")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(decoding_result: dict, output_path: Path) -> None:
    confusion = np.asarray(decoding_result["confusion_matrix"], dtype=float)
    labels = [str(label) for label in decoding_result["confusion_labels"]]

    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized = confusion / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Cross-Validated LDA Confusion Matrix")

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


def save_outputs(
    export_summary_path: Path,
    config: Config,
    date_label: str,
    resolved_label_column: str,
    session_table: pd.DataFrame,
    minute_population_matrix: np.ndarray,
    minute_metadata_table: pd.DataFrame,
    hourly_population_matrix: np.ndarray,
    hourly_metadata_table: pd.DataFrame,
    selected_units: pd.DataFrame,
    feature_keys: list[str],
    feature_stats: pd.DataFrame,
    verification_table: pd.DataFrame,
    projection: np.ndarray,
    decoding_result: dict,
) -> Path:
    output_dir = config.output_base_dir / f"lda_{date_label}_minsess_{config.min_sessions_per_unit}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_status(f"Saving output files to: {output_dir}")
    file_prefix = f"lda_{date_label}_minsess_{config.min_sessions_per_unit}"

    population_columns = [
        f"unit_{feature_index:04d}"
        for feature_index in range(1, hourly_population_matrix.shape[1] + 1)
    ]
    minute_population_df = pd.DataFrame(minute_population_matrix, columns=population_columns)
    minute_population_with_metadata = pd.concat(
        [minute_metadata_table.reset_index(drop=True), minute_population_df],
        axis=1,
    )
    minute_population_with_metadata.to_csv(
        output_dir / f"{file_prefix}_minute_population_vectors.csv",
        index=False,
    )
    log_status(f"Saved {file_prefix}_minute_population_vectors.csv")
    minute_metadata_table.to_csv(output_dir / f"{file_prefix}_minute_metadata.csv", index=False)
    log_status(f"Saved {file_prefix}_minute_metadata.csv")

    hourly_population_df = pd.DataFrame(hourly_population_matrix, columns=population_columns)
    hourly_population_with_metadata = pd.concat(
        [hourly_metadata_table.reset_index(drop=True), hourly_population_df],
        axis=1,
    )
    hourly_population_with_metadata.to_csv(
        output_dir / f"{file_prefix}_hour_population_vectors.csv",
        index=False,
    )
    log_status(f"Saved {file_prefix}_hour_population_vectors.csv")
    hourly_metadata_table.to_csv(output_dir / f"{file_prefix}_hour_metadata.csv", index=False)
    log_status(f"Saved {file_prefix}_hour_metadata.csv")

    selected_units.to_csv(output_dir / f"{file_prefix}_selected_units.csv", index=False)
    log_status(f"Saved {file_prefix}_selected_units.csv")
    (
        feature_stats.set_index("final_group_key")
        .reindex(feature_keys)
        .reset_index()
        .rename(columns={"index": "final_group_key"})
        .assign(feature_column=population_columns)
        .to_csv(output_dir / f"{file_prefix}_feature_map.csv", index=False)
    )
    log_status(f"Saved {file_prefix}_feature_map.csv")

    projection_df = pd.DataFrame(index=np.arange(len(hourly_metadata_table)))
    for dimension_index, column_name in enumerate(["LD1", "LD2", "LD3"]):
        if dimension_index < projection.shape[1]:
            projection_df[column_name] = projection[:, dimension_index]
        else:
            projection_df[column_name] = np.nan
    pd.concat([hourly_metadata_table.reset_index(drop=True), projection_df], axis=1).to_csv(
        output_dir / f"{file_prefix}_projection.csv",
        index=False,
    )
    log_status(f"Saved {file_prefix}_projection.csv")
    session_table.to_csv(output_dir / f"{file_prefix}_session_start_sources.csv", index=False)
    log_status(f"Saved {file_prefix}_session_start_sources.csv")
    verification_table.to_csv(output_dir / f"{file_prefix}_hourly_verification.csv", index=False)
    log_status(f"Saved {file_prefix}_hourly_verification.csv")

    plot_lda_2d(projection, hourly_metadata_table, output_dir / f"{file_prefix}_2d.png")
    log_status(f"Saved {file_prefix}_2d.png")
    plot_lda_3d(projection, hourly_metadata_table, output_dir / f"{file_prefix}_3d.png")
    log_status(f"Saved {file_prefix}_3d.png")
    plot_confusion_matrix(decoding_result, output_dir / f"{file_prefix}_confusion_matrix.png")
    log_status(f"Saved {file_prefix}_confusion_matrix.png")

    summary_payload = {
        "export_summary_path": str(export_summary_path),
        "label_type": resolved_label_column,
        "configured_label_type": str(config.label_type),
        "hourly_aggregation_grouping_keys": ["calendar_day", "clock_hour_of_day"],
        "minute_bin_size_seconds": float(config.bin_size_seconds),
        "min_firing_rate_hz": float(config.min_firing_rate_hz),
        "apply_zscore": bool(config.apply_zscore),
        "apply_smoothing": bool(config.apply_smoothing),
        "smoothing_sigma_bins": float(config.smoothing_sigma_bins),
        "n_minute_samples": int(minute_population_matrix.shape[0]),
        "n_hour_samples": int(hourly_population_matrix.shape[0]),
        "n_features": int(hourly_population_matrix.shape[1]),
        "n_selected_unit_groups": int(selected_units["final_group_key"].nunique()),
        "n_sessions": int(minute_metadata_table["session_name"].nunique()),
        "n_calendar_days": int(hourly_metadata_table["calendar_day"].nunique()),
        "labels": sorted(pd.unique(resolve_label_series(hourly_metadata_table, config)[0]).tolist()),
        "cross_validation_splits": int(decoding_result["n_splits"]),
        "accuracy": float(decoding_result["accuracy"]),
        "balanced_accuracy": float(decoding_result["balanced_accuracy"]),
        "date_label": date_label,
    }
    (output_dir / f"{file_prefix}_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    log_status(f"Saved {file_prefix}_summary.json")

    return output_dir


def run_pipeline(config: Config) -> Path:
    if config.data_path is None:
        raise ValueError("Config.data_path cannot be None when running the pipeline.")

    export_summary_path = resolve_export_summary_path(config.data_path)
    log_status(f"Loading alignment export: {export_summary_path}")
    export_payload = load_export_summary(export_summary_path)

    session_table = build_session_table(export_payload=export_payload, config=config)
    log_status(f"Resolved {len(session_table)} sessions")
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

    minute_population_matrix, minute_metadata_table, feature_keys = build_population_vectors(
        selected_units=selected_units,
        session_table=session_table,
        analyzers=analyzers,
        config=config,
    )
    log_status(
        f"Built minute-level population matrix with shape {minute_population_matrix.shape[0]} bins x "
        f"{minute_population_matrix.shape[1]} features"
    )
    log_status(
        "Applying MIN_FIRING_RATE_HZ using mean bin firing rate "
        "(spikes in bin divided by bin duration, averaged across bins)"
    )
    minute_population_matrix, selected_units, feature_keys, feature_stats = filter_unit_groups_by_binned_firing_rate(
        population_matrix=minute_population_matrix,
        selected_units=selected_units,
        feature_keys=feature_keys,
        config=config,
    )
    hourly_population_matrix, hourly_metadata_table = aggregate_minutes_to_hourly_samples(
        minute_population_matrix=minute_population_matrix,
        minute_metadata_table=minute_metadata_table,
    )
    print_and_build_clock_hour_verification(hourly_metadata_table)
    print_hourly_sample_pivot(
        hourly_metadata_table,
        title="Pivot table for hourly samples before label filtering (calendar_day x clock_hour_of_day)",
    )

    hourly_population_matrix, hourly_metadata_table = filter_labels_for_lda(
        population_matrix=hourly_population_matrix,
        metadata_table=hourly_metadata_table,
        config=config,
    )
    print_hourly_sample_pivot(
        hourly_metadata_table,
        title="Pivot table for final LDA samples (calendar_day x clock_hour_of_day)",
    )
    verification_table = print_and_build_clock_hour_verification(hourly_metadata_table)
    labels, resolved_label_column = resolve_label_series(hourly_metadata_table, config)
    log_status(
        f"After label filtering: {hourly_population_matrix.shape[0]} day x hour samples across "
        f"{len(pd.unique(labels))} '{resolved_label_column}' labels"
    )

    if config.apply_zscore:
        log_status("Applying z-scoring across features")
        hourly_population_matrix = zscore_population_matrix(hourly_population_matrix)

    labels = labels.to_numpy()
    print_lda_label_vector(hourly_metadata_table, labels=labels, label_column=resolved_label_column)
    log_status("Fitting LDA model")
    lda_model, projection = fit_lda(population_matrix=hourly_population_matrix, labels=labels)
    log_status(f"Fitted LDA with {projection.shape[1]} discriminant dimension(s)")

    log_status("Running cross-validated decoding")
    decoding_result = evaluate_decoding(
        population_matrix=hourly_population_matrix,
        labels=labels,
        config=config,
    )
    log_status(
        f"Cross-validated accuracy={decoding_result['accuracy']:.3f}, "
        f"balanced_accuracy={decoding_result['balanced_accuracy']:.3f}"
    )

    output_dir = save_outputs(
        export_summary_path=export_summary_path,
        config=config,
        date_label=date_label,
        resolved_label_column=resolved_label_column,
        session_table=session_table,
        minute_population_matrix=minute_population_matrix,
        minute_metadata_table=minute_metadata_table,
        hourly_population_matrix=hourly_population_matrix,
        hourly_metadata_table=hourly_metadata_table,
        selected_units=selected_units,
        feature_keys=feature_keys,
        feature_stats=feature_stats,
        verification_table=verification_table,
        projection=projection,
        decoding_result=decoding_result,
    )
    return output_dir


def main() -> None:
    config = Config()
    config.data_path = prompt_for_data_path(config.data_path)
    log_status(
        "Multi-day analysis is supported if all sessions were aligned together in the same "
        "export summary. You can point LDA directly to an Alignment_days summary "
        "folder like 'S:\\alignment_days_summary_260224_260226'."
    )
    output_dir = run_pipeline(config)
    log_status(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
