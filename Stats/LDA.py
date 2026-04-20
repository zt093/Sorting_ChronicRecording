from __future__ import annotations

"""
Population LDA analysis for aligned spike-sorting outputs.

This script reads the alignment export created by Alignment_html.py and builds
population firing-rate vectors across aligned good units. Each sample in the
LDA input matrix represents one time bin from one session, and each feature
represents one aligned unit group. If a unit group is absent in a session, its
feature value is filled with zero for that session.

Multi-day analysis is supported as long as the sessions were aligned together
in the same export_summary.json or batch root before running this script.
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
BIN_SIZE_SECONDS = 300.0
LABEL_TYPE = "session_id"  # "hour_of_day", "session_id", or "day_number"
MIN_FIRING_RATE_HZ = 0.05
APPLY_ZSCORE = True
APPLY_SMOOTHING = False
SMOOTHING_SIGMA_BINS = 1.0

MIN_SESSIONS_PER_UNIT = 24
MIN_BINS_PER_LABEL = 2
CV_N_SPLITS = 5
RANDOM_SEED = 42

OUTPUT_FOLDER_NAME = "lda_population"
ANALYZER_FOLDER_NAME = "sorting_analyzer_analysis.zarr"
OUTPUT_BASE_DIR = Path(r"S:\LDA")


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


def resolve_export_summary_path(data_path: Path) -> Path:
    resolved = Path(data_path)
    if resolved.is_file():
        return resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Data path does not exist: {resolved}")

    candidate = resolved / "units_alignment_summary" / "export_summary.json"
    if candidate.exists():
        return candidate

    matches = sorted(resolved.rglob("export_summary.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            "Could not find export_summary.json. Point DATA_PATH to the file or batch root."
        )
    raise RuntimeError(
        "Found multiple export_summary.json files. Please point DATA_PATH to the exact file."
    )


def prompt_for_data_path(default_path: Path | None) -> Path:
    prompt = (
        "\nEnter the alignment export path or batch root for LDA "
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
    return json.loads(export_summary_path.read_text(encoding="utf-8"))


def extract_session_datetime(session_name: str, output_folder: str | None = None) -> datetime | None:
    text_candidates = [str(session_name), str(output_folder or "")]
    patterns = [
        r"(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})[_-]?(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})",
        r"(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})[_-]?(?P<hour>\d{2})",
        r"(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})",
        r"(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})",
    ]

    for text in text_candidates:
        for pattern in patterns:
            match = re.search(pattern, text)
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
                return datetime(year, month, day, hour, minute, second)
            except ValueError:
                continue
    return None


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
        raise RuntimeError("No cross_session_alignment_groups were found in export_summary.json.")

    rows: list[dict] = []
    for group_row in group_rows:
        final_group_key = str(group_row.get("final_group_key", "")).strip()
        final_unit_id = safe_int(group_row.get("final_unit_id"))
        for member in group_row.get("members", []):
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
    grouped["session_start_datetime"] = [
        extract_session_datetime(
            session_name=str(row.session_name),
            output_folder=str(row.output_folder),
        )
        for row in grouped.itertuples(index=False)
    ]
    missing_session_start = grouped["session_start_datetime"].isna()
    if missing_session_start.any():
        missing_names = grouped.loc[missing_session_start, "session_name"].tolist()
        raise ValueError(
            "Could not parse real session start datetime for these sessions: "
            f"{missing_names}. Clock-hour labels require real start timestamps."
        )
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
) -> pd.DataFrame:
    group_rows = export_payload.get("cross_session_alignment_groups", [])
    selected_rows: list[dict] = []
    log_status(f"Evaluating {len(group_rows)} aligned unit groups")

    for group_index, group_row in enumerate(group_rows, start=1):
        final_group_key = str(group_row.get("final_group_key", "")).strip()
        final_unit_id = safe_int(group_row.get("final_unit_id"))
        members = group_row.get("members", [])
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

        if len(valid_members) < config.min_sessions_per_unit:
            continue

        selected_rows.extend(valid_members)

    selected_table = pd.DataFrame(selected_rows)
    if selected_table.empty:
        raise RuntimeError(
            "No aligned unit groups passed the selection criteria. "
            "Try lowering MIN_SESSIONS_PER_UNIT or checking the alignment export."
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
        log_status(f"Binning session {session_position} / {total_sessions}: {session_name}")
        analyzer = analyzers[session_key]
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

        if session_duration_s <= config.bin_size_seconds:
            log_status(
                f"Skipping session '{session_name}' because duration {session_duration_s:.2f}s "
                f"is shorter than one bin."
            )
            continue

        bin_edges = np.arange(0.0, session_duration_s + config.bin_size_seconds, config.bin_size_seconds)
        if len(bin_edges) < 2:
            continue

        session_matrix = np.zeros((len(bin_edges) - 1, len(unit_feature_keys)), dtype=float)
        session_units = members_by_session.get(session_key, pd.DataFrame())
        log_status(
            f"Session '{session_name}': {len(session_units)} units, {len(bin_edges) - 1} bins"
        )

        for member_row in session_units.itertuples(index=False):
            feature_key = str(member_row.final_group_key)
            feature_column = feature_index[feature_key]
            spike_train_samples = analyzer.sorting.get_unit_spike_train(
                unit_id=int(member_row.unit_id),
                segment_index=0,
            )
            spike_times_s = np.asarray(spike_train_samples, dtype=float) / sampling_frequency
            counts, _ = np.histogram(spike_times_s, bins=bin_edges)
            session_matrix[:, feature_column] = counts.astype(float) / config.bin_size_seconds

        if config.apply_smoothing:
            log_status(f"Applying smoothing to session '{session_name}'")
            session_matrix = smooth_population_matrix(
                population_matrix=session_matrix,
                sigma_bins=config.smoothing_sigma_bins,
            )

        bin_centers = bin_edges[:-1] + config.bin_size_seconds / 2.0
        for bin_index, bin_center_s in enumerate(bin_centers):
            session_start_datetime = session_row.session_start_datetime
            bin_start_sec = float(bin_edges[bin_index])
            bin_end_sec = float(bin_edges[bin_index + 1])
            bin_start_datetime = session_start_datetime + timedelta(seconds=bin_start_sec)
            samples.append(session_matrix[bin_index])
            metadata_rows.append(
                {
                    "session_id": int(session_row.session_id),
                    "session_key": session_key,
                    "session_name": session_name,
                    "session_name_normalized": str(session_row.session_name_normalized),
                    "session_index": safe_int(session_row.session_index),
                    "session_start_datetime": session_start_datetime.isoformat(sep=" "),
                    "bin_index": int(bin_index),
                    "bin_start_sec": bin_start_sec,
                    "bin_end_sec": bin_end_sec,
                    "bin_center_s": float(bin_center_s),
                    "session_duration_s": float(session_duration_s),
                    "clock_hour_of_day": int(bin_start_datetime.hour),
                }
            )

    if not samples:
        raise RuntimeError("No population vectors were created. Check the sessions and bin size.")

    population_matrix = np.vstack(samples)
    metadata_table = pd.DataFrame(metadata_rows)
    log_status(
        f"Finished binning: created {population_matrix.shape[0]} samples across {len(session_table)} sessions"
    )
    return population_matrix, metadata_table, unit_feature_keys


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
        ["session_name", "session_start_datetime", "bin_start_sec", "clock_hour_of_day"]
    ].head(20).copy()
    log_status("Clock-hour verification preview (first 20 bins):")
    if verification_table.empty:
        log_status("No verification rows available.")
    else:
        print(verification_table.to_string(index=False), flush=True)
    return verification_table


def filter_labels_for_lda(
    population_matrix: np.ndarray,
    metadata_table: pd.DataFrame,
    config: Config,
) -> tuple[np.ndarray, pd.DataFrame]:
    label_counts = metadata_table["clock_hour_of_day"].value_counts()
    kept_labels = sorted(label_counts[label_counts >= config.min_bins_per_label].index.tolist())
    keep_mask = metadata_table["clock_hour_of_day"].isin(kept_labels).to_numpy()
    filtered_population = population_matrix[keep_mask]
    filtered_metadata = metadata_table.loc[keep_mask].reset_index(drop=True)
    if filtered_metadata.empty or len(kept_labels) < 2:
        raise RuntimeError(
            "LDA needs at least two clock-hour labels with enough bins. "
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
    session_ids = metadata_table["session_id"].to_numpy()

    x_values = projection[:, 0]
    y_values = projection[:, 1] if projection.shape[1] >= 2 else np.zeros(len(projection))

    for session_id in pd.unique(session_ids):
        mask = session_ids == session_id
        session_points = metadata_table.loc[mask].sort_values("bin_start_sec")
        if len(session_points) < 2:
            continue
        point_indices = session_points.index.to_numpy()
        ax.plot(
            x_values[point_indices],
            y_values[point_indices],
            color="#9a9a9a",
            linewidth=0.6,
            alpha=0.35,
            zorder=1,
        )

    scatter = ax.scatter(
        x_values,
        y_values,
        c=hours,
        cmap="viridis",
        s=28,
        alpha=0.85,
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

    x_values = projection[:, 0]
    y_values = projection[:, 1] if projection.shape[1] >= 2 else np.zeros(len(projection))
    z_values = projection[:, 2] if projection.shape[1] >= 3 else np.zeros(len(projection))

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
    population_matrix: np.ndarray,
    metadata_table: pd.DataFrame,
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
        for feature_index in range(1, population_matrix.shape[1] + 1)
    ]
    population_df = pd.DataFrame(population_matrix, columns=population_columns)
    population_with_metadata = pd.concat([metadata_table.reset_index(drop=True), population_df], axis=1)
    population_with_metadata.to_csv(output_dir / f"{file_prefix}_population_vectors.csv", index=False)
    log_status(f"Saved {file_prefix}_population_vectors.csv")

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

    projection_df = pd.DataFrame(index=np.arange(len(metadata_table)))
    for dimension_index, column_name in enumerate(["LD1", "LD2", "LD3"]):
        if dimension_index < projection.shape[1]:
            projection_df[column_name] = projection[:, dimension_index]
        else:
            projection_df[column_name] = np.nan
    pd.concat([metadata_table.reset_index(drop=True), projection_df], axis=1).to_csv(
        output_dir / f"{file_prefix}_projection.csv",
        index=False,
    )
    log_status(f"Saved {file_prefix}_projection.csv")
    verification_table.to_csv(output_dir / f"{file_prefix}_clock_hour_verification.csv", index=False)
    log_status(f"Saved {file_prefix}_clock_hour_verification.csv")

    plot_lda_2d(projection, metadata_table, output_dir / f"{file_prefix}_2d.png")
    log_status(f"Saved {file_prefix}_2d.png")
    plot_lda_3d(projection, metadata_table, output_dir / f"{file_prefix}_3d.png")
    log_status(f"Saved {file_prefix}_3d.png")
    plot_confusion_matrix(decoding_result, output_dir / f"{file_prefix}_confusion_matrix.png")
    log_status(f"Saved {file_prefix}_confusion_matrix.png")

    summary_payload = {
        "export_summary_path": str(export_summary_path),
        "label_type": config.label_type,
        "bin_size_seconds": float(config.bin_size_seconds),
        "min_firing_rate_hz": float(config.min_firing_rate_hz),
        "apply_zscore": bool(config.apply_zscore),
        "apply_smoothing": bool(config.apply_smoothing),
        "smoothing_sigma_bins": float(config.smoothing_sigma_bins),
        "n_samples": int(population_matrix.shape[0]),
        "n_features": int(population_matrix.shape[1]),
        "n_selected_unit_groups": int(selected_units["final_group_key"].nunique()),
        "n_sessions": int(metadata_table["session_name"].nunique()),
        "labels": sorted(metadata_table["clock_hour_of_day"].unique().tolist()),
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
    )
    log_status(
        f"Selected {selected_units['final_group_key'].nunique()} aligned unit groups "
        f"across {selected_units['session_name'].nunique()} sessions"
    )

    population_matrix, metadata_table, feature_keys = build_population_vectors(
        selected_units=selected_units,
        session_table=session_table,
        analyzers=analyzers,
        config=config,
    )
    log_status(
        f"Built population matrix with shape {population_matrix.shape[0]} bins x "
        f"{population_matrix.shape[1]} features"
    )
    log_status(
        "Applying MIN_FIRING_RATE_HZ using mean bin firing rate "
        "(spikes in bin divided by bin duration, averaged across bins)"
    )
    population_matrix, selected_units, feature_keys, feature_stats = filter_unit_groups_by_binned_firing_rate(
        population_matrix=population_matrix,
        selected_units=selected_units,
        feature_keys=feature_keys,
        config=config,
    )
    verification_table = print_and_build_clock_hour_verification(metadata_table)

    population_matrix, metadata_table = filter_labels_for_lda(
        population_matrix=population_matrix,
        metadata_table=metadata_table,
        config=config,
    )
    log_status(
        f"After label filtering: {population_matrix.shape[0]} samples across "
        f"{metadata_table['clock_hour_of_day'].nunique()} clock-hour labels"
    )

    if config.apply_zscore:
        log_status("Applying z-scoring across features")
        population_matrix = zscore_population_matrix(population_matrix)

    labels = metadata_table["clock_hour_of_day"].to_numpy()
    log_status("Fitting LDA model")
    lda_model, projection = fit_lda(population_matrix=population_matrix, labels=labels)
    log_status(f"Fitted LDA with {projection.shape[1]} discriminant dimension(s)")

    log_status("Running cross-validated decoding")
    decoding_result = evaluate_decoding(
        population_matrix=population_matrix,
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
        population_matrix=population_matrix,
        metadata_table=metadata_table,
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
        "export_summary.json or batch root."
    )
    output_dir = run_pipeline(config)
    log_status(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
