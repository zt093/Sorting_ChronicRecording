from __future__ import annotations

"""
Circadian-style tuning plots for aligned neural units.

Run this after Alignment_html.py or Alignment_days.py has exported aligned unit
summaries. The script uses the same alignment export/analyzer loading path as
LDA.py, bins each recording in real clock time at 1-minute resolution, folds
minutes into clock hours, and saves two plot families:

1. One aligned unit across days, with each calendar day kept as a separate line.
2. One calendar day across all aligned units, where each unit is normalized
   within that day before population aggregation.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import LDA as lda_helpers


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

DATA_PATH = None  # Leave as None to enter the alignment export path at runtime.
OUTPUT_BASE_DIR = Path(r"S:\Tuning")

MIN_SESSIONS_PER_UNIT = 66
BIN_SIZE_SECONDS = 60.0
MIN_MINUTES_PER_HOUR = 1

METRICS_TO_PLOT = (
    "firing_rate_hz",
    "average_amplitude_uv",
    "peak_to_trough_ms",
    "cv2",
)

# Choose which plot families to generate: ("type1", "type2"), ("type1",), or ("type2",).
PLOT_TYPES = ("type2",)

# Plot type 1: use "all" to make one plot per selected aligned unit, or provide
# final_unit_id values / final_group_key strings such as (1, 5, "align_A").
TYPE1_UNITS = "all"

# Plot type 2: "all" makes one output set per available calendar day.
# You can set "YYYY-MM-DD" to target one specific day. None uses the first day.
TYPE2_DAY = "all"
NORMALIZATION_METHODS = (
    "zscore",
    "minmax",
    "relative_mean",
    "relative_first",
)
VARIABILITY_MODE = "sem"  # "sem" or "iqr"

ANALYZER_FOLDER_NAME = "sorting_analyzer_analysis.zarr"

METRIC_LABELS = {
    "firing_rate_hz": "Firing rate (Hz)",
    "average_amplitude_uv": "Amplitude (uV)",
    "peak_to_trough_ms": "Peak width (ms)",
    "cv2": "CV2",
}


@dataclass
class Config:
    data_path: Path | None = DATA_PATH
    output_base_dir: Path = OUTPUT_BASE_DIR
    min_sessions_per_unit: int = MIN_SESSIONS_PER_UNIT
    bin_size_seconds: float = BIN_SIZE_SECONDS
    min_minutes_per_hour: int = MIN_MINUTES_PER_HOUR
    metrics_to_plot: tuple[str, ...] = METRICS_TO_PLOT
    plot_types: tuple[str, ...] = PLOT_TYPES
    type1_units: str | tuple[int | str, ...] = TYPE1_UNITS
    type2_day: str | None = TYPE2_DAY
    normalization_methods: tuple[str, ...] = NORMALIZATION_METHODS
    variability_mode: str = VARIABILITY_MODE
    analyzer_folder_name: str = ANALYZER_FOLDER_NAME
    metric_labels: dict[str, str] | None = None


def log_status(message: str) -> None:
    print(f"[Tuning] {message}", flush=True)


lda_helpers.log_status = log_status


def safe_slug(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "value"


def prompt_for_data_path(default_path: Path | None) -> Path:
    raw_value = input(
        "\nEnter the alignment export path, Alignment_days summary folder, or batch root for Tuning "
        "(press Enter to use configured DATA_PATH): "
    ).strip().strip('"').strip("'")
    if raw_value:
        return Path(raw_value)
    if default_path is not None:
        return Path(default_path)
    raise ValueError("A data path is required.")


def parse_time_of_day(value) -> float:
    """Return fractional clock hour in [0, 24)."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (datetime, pd.Timestamp)):
        dt = pd.Timestamp(value)
    else:
        dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return np.nan
    return float(dt.hour) + float(dt.minute) / 60.0 + float(dt.second) / 3600.0


def add_day_night_background(ax, day_start_hour: float = 7.0, day_end_hour: float = 19.0) -> None:
    """Shade night as gray and day as warm yellow on a 0-24 hour axis."""
    ax.axvspan(0, day_start_hour, color="#e8e8e8", alpha=0.65, zorder=0)
    ax.axvspan(day_start_hour, day_end_hour, color="#fff3c4", alpha=0.55, zorder=0)
    ax.axvspan(day_end_hour, 24, color="#e8e8e8", alpha=0.65, zorder=0)
    ax.axvline(day_start_hour, color="#9b8b34", linewidth=0.9, alpha=0.8)
    ax.axvline(day_end_hour, color="#555555", linewidth=0.9, alpha=0.8)


def setup_circadian_axis(ax, ylabel: str, title: str) -> None:
    add_day_night_background(ax)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 3, 6, 7, 9, 12, 15, 18, 19, 21, 24])
    ax.set_xticklabels(["00:00", "03:00", "06:00", "07:00", "09:00", "12:00", "15:00", "18:00", "19:00", "21:00", "24:00"])
    ax.set_xlabel("Time of day")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def normalize_unit_values(values: pd.Series, method: str) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").astype(float)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)

    method = str(method).strip().lower()
    if method == "zscore":
        center = float(finite.mean())
        scale = float(finite.std(ddof=0))
        if scale == 0 or not np.isfinite(scale):
            return pd.Series(np.nan, index=values.index, dtype=float)
        return (values - center) / scale
    if method == "minmax":
        min_value = float(finite.min())
        max_value = float(finite.max())
        scale = max_value - min_value
        if scale == 0 or not np.isfinite(scale):
            return pd.Series(np.nan, index=values.index, dtype=float)
        return (values - min_value) / scale
    if method == "relative_mean":
        baseline = float(finite.mean())
        if baseline == 0 or not np.isfinite(baseline):
            return pd.Series(np.nan, index=values.index, dtype=float)
        return (values - baseline) / baseline
    if method == "relative_first":
        baseline = float(finite.iloc[0])
        if baseline == 0 or not np.isfinite(baseline):
            return pd.Series(np.nan, index=values.index, dtype=float)
        return (values - baseline) / baseline
    raise ValueError(
        "Unsupported normalization method. Use zscore, minmax, relative_mean, or relative_first."
    )


def get_metric_label(metric: str, config: Config) -> str:
    labels = config.metric_labels or METRIC_LABELS
    return labels.get(metric, metric)


def build_lda_config(config: Config) -> lda_helpers.Config:
    lda_config = lda_helpers.Config()
    lda_config.data_path = config.data_path
    lda_config.bin_size_seconds = float(config.bin_size_seconds)
    lda_config.min_sessions_per_unit = int(config.min_sessions_per_unit)
    lda_config.min_minutes_per_hour = int(config.min_minutes_per_hour)
    lda_config.analyzer_folder_name = str(config.analyzer_folder_name)
    lda_config.apply_smoothing = False
    lda_config.apply_zscore = False
    return lda_config


def load_aligned_minute_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    lda_config = build_lda_config(config)
    export_summary_path = lda_helpers.resolve_export_summary_path(config.data_path)
    log_status(f"Loading alignment export: {export_summary_path}")
    export_payload = lda_helpers.load_export_summary(export_summary_path)

    session_table = lda_helpers.build_session_table(export_payload=export_payload, config=lda_config)
    log_status(f"Resolved {len(session_table)} sessions with real clock starts")
    analyzers, resolved_output_folders = lda_helpers.load_session_analyzers(
        session_table=session_table,
        config=lda_config,
    )
    session_table = session_table.copy()
    session_table["resolved_output_folder"] = session_table["session_key"].map(resolved_output_folders)

    selected_units = lda_helpers.select_good_unit_groups(
        export_payload=export_payload,
        config=lda_config,
        analyzers=analyzers,
    )
    log_status(
        f"Selected {selected_units['final_group_key'].nunique()} aligned unit groups "
        f"using MIN_SESSIONS_PER_UNIT={config.min_sessions_per_unit}"
    )

    minute_matrix, minute_metadata, feature_table = lda_helpers.build_population_vectors(
        selected_units=selected_units,
        session_table=session_table,
        analyzers=analyzers,
        config=lda_config,
    )
    feature_columns = feature_table["feature_column"].astype(str).tolist()
    minute_values = pd.DataFrame(minute_matrix, columns=feature_columns)
    minute_wide = pd.concat(
        [minute_metadata.reset_index(drop=True), minute_values.reset_index(drop=True)],
        axis=1,
    )
    minute_wide["time_of_day_hour"] = minute_wide["minute_start_datetime"].map(parse_time_of_day)
    return minute_wide, feature_table, selected_units, export_summary_path


def build_hourly_metric_table(
    minute_wide: pd.DataFrame,
    feature_table: pd.DataFrame,
    metrics: tuple[str, ...],
    min_minutes_per_hour: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    metadata_columns = [
        "calendar_day",
        "clock_hour_of_day",
        "session_id",
        "session_name",
        "minute_start_datetime",
        "clock_minute_of_hour",
    ]
    for feature_row in feature_table.itertuples(index=False):
        metric = str(feature_row.feature_type)
        if metric not in metrics:
            continue
        feature_column = str(feature_row.feature_column)
        if feature_column not in minute_wide.columns:
            continue
        unit_table = minute_wide[metadata_columns + [feature_column]].copy()
        unit_table = unit_table.rename(columns={feature_column: "metric_value"})
        unit_table["final_group_key"] = str(feature_row.final_group_key)
        unit_table["final_unit_id"] = lda_helpers.safe_int(feature_row.final_unit_id)
        unit_table["shank_id"] = lda_helpers.safe_int(feature_row.shank_id)
        unit_table["local_channel_on_shank"] = lda_helpers.safe_int(feature_row.local_channel_on_shank)
        unit_table["metric"] = metric
        rows.append(unit_table)

    if not rows:
        raise RuntimeError("No matching metric columns were found in the binned feature table.")

    long_minutes = pd.concat(rows, ignore_index=True)
    long_minutes["metric_value"] = pd.to_numeric(long_minutes["metric_value"], errors="coerce")
    long_minutes = long_minutes[np.isfinite(long_minutes["metric_value"])].copy()
    grouped = long_minutes.groupby(
        ["metric", "final_group_key", "final_unit_id", "shank_id", "local_channel_on_shank", "calendar_day", "clock_hour_of_day"],
        sort=True,
        dropna=False,
    )

    hourly_rows: list[dict] = []
    for group_key, group_table in grouped:
        unique_minutes = sorted(
            int(value)
            for value in pd.unique(group_table["clock_minute_of_hour"])
            if lda_helpers.safe_int(value) is not None
        )
        if len(unique_minutes) < int(min_minutes_per_hour):
            continue
        (
            metric,
            final_group_key,
            final_unit_id,
            shank_id,
            local_channel_on_shank,
            calendar_day,
            clock_hour_of_day,
        ) = group_key
        hourly_rows.append(
            {
                "metric": str(metric),
                "final_group_key": str(final_group_key),
                "final_unit_id": lda_helpers.safe_int(final_unit_id),
                "shank_id": lda_helpers.safe_int(shank_id),
                "local_channel_on_shank": lda_helpers.safe_int(local_channel_on_shank),
                "calendar_day": str(calendar_day),
                "clock_hour_of_day": int(clock_hour_of_day),
                "time_of_day_hour": float(clock_hour_of_day),
                "value": float(np.nanmean(group_table["metric_value"].to_numpy(dtype=float))),
                "n_minutes_used": int(len(group_table)),
                "n_unique_clock_minutes_present": int(len(unique_minutes)),
                "n_missing_clock_minutes": int(max(0, 60 - len(unique_minutes))),
                "session_names": " | ".join(str(value) for value in pd.unique(group_table["session_name"])),
            }
        )

    hourly_table = pd.DataFrame(hourly_rows)
    if hourly_table.empty:
        raise RuntimeError("No hourly metric rows passed the minimum-minute filter.")
    return hourly_table.sort_values(
        ["metric", "final_unit_id", "final_group_key", "calendar_day", "clock_hour_of_day"],
        na_position="last",
    ).reset_index(drop=True)


def choose_type1_units(hourly_table: pd.DataFrame, config: Config) -> pd.DataFrame:
    unit_table = hourly_table[["final_group_key", "final_unit_id", "shank_id", "local_channel_on_shank"]].drop_duplicates()
    if str(config.type1_units).lower() == "all":
        return unit_table.sort_values(["final_unit_id", "final_group_key"], na_position="last")

    if isinstance(config.type1_units, (str, int)):
        requested_values = (config.type1_units,)
    else:
        requested_values = tuple(config.type1_units)
    requested = set(requested_values)
    mask = unit_table["final_group_key"].isin({str(value) for value in requested}) | unit_table["final_unit_id"].isin(
        {int(value) for value in requested if str(value).isdigit()}
    )
    return unit_table.loc[mask].sort_values(["final_unit_id", "final_group_key"], na_position="last")


def plot_single_unit_across_days(
    hourly_table: pd.DataFrame,
    metric: str,
    unit_row: pd.Series,
    output_path: Path,
    config: Config,
) -> None:
    unit_mask = (
        (hourly_table["metric"] == metric)
        & (hourly_table["final_group_key"] == unit_row["final_group_key"])
    )
    plot_table = hourly_table.loc[unit_mask].copy()
    if plot_table.empty:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    colors = plt.get_cmap("tab20").colors
    for day_index, (calendar_day, day_table) in enumerate(plot_table.groupby("calendar_day", sort=True)):
        day_series = (
            day_table.groupby("clock_hour_of_day", sort=True)["value"]
            .mean()
            .reindex(range(24), fill_value=np.nan)
        )
        ax.plot(
            np.arange(24),
            day_series.to_numpy(dtype=float),
            marker=markers[day_index % len(markers)],
            color=colors[day_index % len(colors)],
            linewidth=1.8,
            markersize=4.5,
            label=str(calendar_day),
        )

    unit_label = f"Unit {unit_row['final_unit_id']} ({unit_row['final_group_key']})"
    setup_circadian_axis(
        ax,
        ylabel=get_metric_label(metric, config),
        title=f"{unit_label}: {get_metric_label(metric, config)} across days",
    )
    ax.legend(title="Day", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def summarize_normalized_day(
    hourly_table: pd.DataFrame,
    metric: str,
    calendar_day: str,
    normalization_method: str,
    variability_mode: str,
) -> pd.DataFrame:
    day_table = hourly_table[
        (hourly_table["metric"] == metric)
        & (hourly_table["calendar_day"].astype(str) == str(calendar_day))
    ].copy()
    if day_table.empty:
        return pd.DataFrame()

    normalized_tables: list[pd.DataFrame] = []
    for _, unit_table in day_table.groupby("final_group_key", sort=True):
        unit_table = unit_table.sort_values("clock_hour_of_day").copy()
        unit_table["normalized_value"] = normalize_unit_values(unit_table["value"], normalization_method)
        normalized_tables.append(unit_table)

    normalized = pd.concat(normalized_tables, ignore_index=True)
    normalized = normalized[np.isfinite(normalized["normalized_value"])].copy()
    if normalized.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for hour in range(24):
        values = normalized.loc[
            normalized["clock_hour_of_day"] == hour,
            "normalized_value",
        ].dropna().to_numpy(dtype=float)
        if values.size == 0:
            rows.append(
                {
                    "clock_hour_of_day": hour,
                    "time_of_day_hour": float(hour),
                    "center": np.nan,
                    "lower": np.nan,
                    "upper": np.nan,
                    "n_units": 0,
                }
            )
            continue
        if variability_mode == "iqr":
            center = float(np.nanmedian(values))
            lower = float(np.nanpercentile(values, 25))
            upper = float(np.nanpercentile(values, 75))
        else:
            center = float(np.nanmean(values))
            sem = float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
            lower = center - sem
            upper = center + sem
        rows.append(
            {
                "clock_hour_of_day": hour,
                "time_of_day_hour": float(hour),
                "center": center,
                "lower": lower,
                "upper": upper,
                "n_units": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def plot_day_population_trend(
    summary_table: pd.DataFrame,
    metric: str,
    calendar_day: str,
    normalization_method: str,
    variability_mode: str,
    output_path: Path,
    config: Config,
) -> None:
    if summary_table.empty:
        return
    x = summary_table["time_of_day_hour"].to_numpy(dtype=float)
    center = summary_table["center"].to_numpy(dtype=float)
    lower = summary_table["lower"].to_numpy(dtype=float)
    upper = summary_table["upper"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.fill_between(x, lower, upper, color="#5b8cc0", alpha=0.22, linewidth=0)
    ax.plot(x, center, color="#1f5f99", linewidth=2.4, marker="o", markersize=4.2)
    ylabel = f"Normalized {get_metric_label(metric, config)}"
    variability_label = "median + IQR" if variability_mode == "iqr" else "mean +/- SEM"
    setup_circadian_axis(
        ax,
        ylabel=ylabel,
        title=f"{calendar_day}: {metric} trend ({normalization_method}, {variability_label})",
    )
    secondary = ax.twinx()
    secondary.plot(
        x,
        summary_table["n_units"].to_numpy(dtype=float),
        color="#555555",
        linewidth=1.0,
        linestyle=":",
        alpha=0.65,
    )
    secondary.set_ylabel("Units contributing", color="#555555")
    secondary.tick_params(axis="y", labelcolor="#555555")
    secondary.spines["top"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_outputs(
    hourly_table: pd.DataFrame,
    feature_table: pd.DataFrame,
    selected_units: pd.DataFrame,
    export_summary_path: Path,
    config: Config,
) -> list[Path]:
    date_label = "_".join(sorted(hourly_table["calendar_day"].astype(str).unique()))
    if len(date_label) > 90:
        date_label = f"{hourly_table['calendar_day'].min()}_to_{hourly_table['calendar_day'].max()}"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = (
        config.output_base_dir
        / f"tuning_{safe_slug(date_label)}_minsess_{int(config.min_sessions_per_unit)}_{run_timestamp}"
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    unit_presence = (
        selected_units.groupby(["final_group_key", "final_unit_id"], dropna=False)
        .agg(
            member_rows=("session_key", "size"),
            unique_session_keys=("session_key", "nunique"),
            unique_session_names=("session_name", "nunique"),
        )
        .reset_index()
        .sort_values(["final_unit_id", "final_group_key"], na_position="last")
    )

    hourly_table.to_csv(root_dir / "tuning_hourly_metric_table.csv", index=False)
    feature_table.to_csv(root_dir / "tuning_feature_table.csv", index=False)
    selected_units.to_csv(root_dir / "tuning_selected_units.csv", index=False)
    unit_presence.to_csv(root_dir / "tuning_unit_presence_summary.csv", index=False)
    (root_dir / "tuning_summary.json").write_text(
        json.dumps(
            {
                "export_summary_path": str(export_summary_path),
                "run_timestamp": run_timestamp,
                "bin_size_seconds": float(config.bin_size_seconds),
                "min_sessions_per_unit": int(config.min_sessions_per_unit),
                "min_minutes_per_hour": int(config.min_minutes_per_hour),
                "metrics_to_plot": list(config.metrics_to_plot),
                "plot_types": list(config.plot_types),
                "type2_day": config.type2_day,
                "normalization_methods": list(config.normalization_methods),
                "variability_mode": str(config.variability_mode),
                "n_hourly_rows": int(len(hourly_table)),
                "n_aligned_units": int(hourly_table["final_group_key"].nunique()),
                "min_unique_sessions_per_aligned_unit": int(unit_presence["unique_session_keys"].min()),
                "max_unique_sessions_per_aligned_unit": int(unit_presence["unique_session_keys"].max()),
                "max_member_rows_per_aligned_unit": int(unit_presence["member_rows"].max()),
                "calendar_days": sorted(hourly_table["calendar_day"].astype(str).unique().tolist()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_dirs: list[Path] = []
    plot_types = {str(plot_type).strip().lower() for plot_type in config.plot_types}
    aliases = {
        "1": "type1",
        "plot1": "type1",
        "plot_type_1": "type1",
        "single_unit": "type1",
        "2": "type2",
        "plot2": "type2",
        "plot_type_2": "type2",
        "population": "type2",
    }
    plot_types = {aliases.get(plot_type, plot_type) for plot_type in plot_types}
    valid_plot_types = {"type1", "type2"}
    unknown_plot_types = sorted(plot_types - valid_plot_types)
    if unknown_plot_types:
        raise ValueError(
            f"Unsupported PLOT_TYPES values: {unknown_plot_types}. Use 'type1' and/or 'type2'."
        )

    if "type1" in plot_types:
        type1_units = choose_type1_units(hourly_table, config)
        type1_dir = root_dir / f"plot_type_1_single_unit_across_days_{safe_slug(date_label)}"
        for metric in config.metrics_to_plot:
            for unit_row in type1_units.itertuples(index=False):
                unit_series = pd.Series(unit_row._asdict())
                unit_slug = f"unit_{safe_slug(unit_series['final_unit_id'])}_{safe_slug(unit_series['final_group_key'])}"
                plot_single_unit_across_days(
                    hourly_table=hourly_table,
                    metric=metric,
                    unit_row=unit_series,
                    output_path=type1_dir / metric / f"{unit_slug}_{metric}.png",
                    config=config,
                )
        output_dirs.append(type1_dir)

    if "type2" in plot_types:
        available_days = sorted(hourly_table["calendar_day"].astype(str).unique().tolist())
        requested_type2_day = str(config.type2_day).strip() if config.type2_day is not None else ""
        if requested_type2_day.lower() in {"all", "*"}:
            selected_days = available_days
        elif requested_type2_day:
            if requested_type2_day not in set(available_days):
                raise ValueError(
                    f"TYPE2_DAY={requested_type2_day} is not available. Available days: {available_days}"
                )
            selected_days = [requested_type2_day]
        else:
            selected_days = [available_days[0]]

        variability_mode = str(config.variability_mode).strip().lower()
        if variability_mode not in {"sem", "iqr"}:
            raise ValueError("VARIABILITY_MODE must be 'sem' or 'iqr'.")

        for selected_day in selected_days:
            type2_dir = root_dir / f"plot_type_2_all_units_one_day_{safe_slug(selected_day)}"
            for metric in config.metrics_to_plot:
                for normalization_method in config.normalization_methods:
                    summary_table = summarize_normalized_day(
                        hourly_table=hourly_table,
                        metric=metric,
                        calendar_day=selected_day,
                        normalization_method=normalization_method,
                        variability_mode=variability_mode,
                    )
                    if summary_table.empty:
                        continue
                    method_dir = type2_dir / metric / normalization_method
                    method_dir.mkdir(parents=True, exist_ok=True)
                    summary_table.to_csv(
                        method_dir / f"{selected_day}_{metric}_{normalization_method}_{variability_mode}.csv",
                        index=False,
                    )
                    plot_day_population_trend(
                        summary_table=summary_table,
                        metric=metric,
                        calendar_day=selected_day,
                        normalization_method=normalization_method,
                        variability_mode=variability_mode,
                        output_path=method_dir / f"{selected_day}_{metric}_{normalization_method}_{variability_mode}.png",
                        config=config,
                    )
            output_dirs.append(type2_dir)
    return output_dirs


def run_pipeline(config: Config) -> list[Path]:
    if config.data_path is None:
        raise ValueError("Config.data_path cannot be None when running the pipeline.")
    config.metric_labels = config.metric_labels or METRIC_LABELS
    minute_wide, feature_table, selected_units, export_summary_path = load_aligned_minute_data(config)
    hourly_table = build_hourly_metric_table(
        minute_wide=minute_wide,
        feature_table=feature_table,
        metrics=tuple(config.metrics_to_plot),
        min_minutes_per_hour=int(config.min_minutes_per_hour),
    )
    log_status(f"Built hourly metric table with {len(hourly_table)} rows")
    output_dirs = save_outputs(
        hourly_table=hourly_table,
        feature_table=feature_table,
        selected_units=selected_units,
        export_summary_path=export_summary_path,
        config=config,
    )
    return output_dirs


def main() -> None:
    config = Config()
    config.data_path = prompt_for_data_path(config.data_path)
    output_dirs = run_pipeline(config)
    for output_dir in output_dirs:
        log_status(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
