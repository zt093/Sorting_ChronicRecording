from __future__ import annotations

"""
Presentation-ready summary plots for aligned-unit outputs.

This script is meant for quick figure generation after the sorting/alignment
pipeline has already run. It reads alignment exports produced by
`Alignment_html.py` and, when available, quality metrics stored in the saved
SortingAnalyzer folders produced by `Combined_NWB+Sorting+Analyze.py`.

High-level behavior
-------------------
1. Resolve the batch root, `unique_units_summary.csv`, or `export_summary.json`.
2. Load the cross-session unique-unit summary table exported by the alignment step.
3. Generate presentation-friendly plots such as:
   - number of unique units by exact session count
   - top channels ranked by persistent aligned units
   - quality metrics versus cross-session persistence
4. Save figures and a small JSON manifest into the batch `stats` folder by default.

Typical inputs
--------------
- A sorting batch root such as `260224_Sorting`
- Or `units_alignment_summary/export_summary.json`
- Or `units_alignment_summary/unique_units_summary.csv`

Typical outputs
---------------
- `units_by_exact_session_count.png`
- `top_channels_by_persistent_units.png`
- `quality_metrics_vs_persistence.png`
- `presentation_plots_manifest.json`
"""

import argparse
import json
from pathlib import Path
import re
import tkinter as tk
from tkinter import filedialog

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MAX_SESSIONS = 24
DEFAULT_ANALYZER_FOLDER_NAME = "sorting_analyzer_analysis.zarr"
DEFAULT_ALIGNMENT_EXPORT_FOLDER_NAME = "units_alignment_summary"
DEFAULT_ALIGNMENT_EXPORT_SUMMARY_NAME = "export_summary.json"
ACTIVE_WINDOW_LOWER_PERCENTILE = 5.0
ACTIVE_WINDOW_UPPER_PERCENTILE = 95.0
DISCARD_ABS_AMPLITUDE_MAX = 50.0
DISCARD_SNR_MAX = 3.0
DISCARD_ISI_VIOLATION_MIN = 2.0

BAR_COLOR = "#2f6b8a"
ACCENT_COLOR = "#c98b2d"
HEATMAP_CMAP = "YlGnBu"
GRID_COLOR = "#d8dde3"


def log_status(message: str) -> None:
    print(f"[presentations] {message}", flush=True)


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


def normalize_session_name(session_name: str) -> str:
    return re.sub(r"_sh\d+$", "", str(session_name).strip())


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def annotate_bars(ax, bars, values: list[int]) -> None:
    ymax = max(values) if values else 0
    offset = max(0.2, ymax * 0.015)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 300) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_status(f"Saving figure: {output_path}")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def resolve_output_dir(
    *,
    sorting_root: Path | None,
    unique_units_csv: Path,
    output_dir: str | None,
) -> Path:
    if output_dir:
        resolved = Path(output_dir)
        resolved.mkdir(parents=True, exist_ok=True)
        log_status(f"Using requested output folder: {resolved}")
        return resolved

    batch_root = sorting_root if sorting_root else unique_units_csv.parent.parent
    stats_candidates = [batch_root / "stats", batch_root / "Stats"]
    stats_dir = next((path for path in stats_candidates if path.exists()), stats_candidates[0])
    stats_dir.mkdir(parents=True, exist_ok=True)
    log_status(f"Using default stats output folder: {stats_dir}")
    return stats_dir


def resolve_unique_units_csv(
    sorting_root: Path | None,
    unique_units_csv: str | None,
    export_summary: str | None,
) -> Path:
    if sorting_root is not None:
        log_status(f"Using sorting root: {sorting_root}")
        resolved_export_summary = (
            sorting_root
            / DEFAULT_ALIGNMENT_EXPORT_FOLDER_NAME
            / DEFAULT_ALIGNMENT_EXPORT_SUMMARY_NAME
        )
        if not resolved_export_summary.exists():
            raise FileNotFoundError(
                "Could not find alignment export summary under "
                f"{sorting_root}. Expected: {resolved_export_summary}"
            )
        payload = json.loads(resolved_export_summary.read_text(encoding="utf-8"))
        csv_path = Path(payload["unique_units_summary_csv"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Referenced unique_units_summary.csv not found: {csv_path}")
        log_status(f"Resolved unique-units CSV from batch root: {csv_path}")
        return csv_path

    if unique_units_csv:
        path = Path(unique_units_csv)
        if not path.exists():
            raise FileNotFoundError(f"unique_units_summary.csv not found: {path}")
        log_status(f"Using provided unique-units CSV: {path}")
        return path

    if export_summary:
        export_summary_path = Path(export_summary)
        if not export_summary_path.exists():
            raise FileNotFoundError(f"export_summary.json not found: {export_summary_path}")
        log_status(f"Using provided export summary: {export_summary_path}")
        payload = json.loads(export_summary_path.read_text(encoding="utf-8"))
        csv_path = Path(payload["unique_units_summary_csv"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Referenced unique_units_summary.csv not found: {csv_path}")
        log_status(f"Resolved unique-units CSV from export summary: {csv_path}")
        return csv_path

    csv_path = choose_unique_units_csv_interactively()
    if csv_path is not None:
        return csv_path

    raise ValueError(
        "No input file selected. Provide --unique-units-csv or --export-summary, "
        "or choose a CSV when prompted."
    )


def choose_unique_units_csv_interactively() -> Path | None:
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_path = filedialog.askopenfilename(
            title="Select unique_units_summary.csv or export_summary.json",
            filetypes=[
                ("Supported files", "*.csv *.json"),
                ("CSV files", "*.csv"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
    except Exception:
        selected_path = ""

    if not selected_path:
        raw_value = input(
            "Enter path to unique_units_summary.csv or export_summary.json: "
        ).strip().strip('"').strip("'")
        if not raw_value:
            return None
        selected_path = raw_value

    selected = Path(selected_path)
    if not selected.exists():
        raise FileNotFoundError(f"Selected file not found: {selected}")

    if selected.suffix.lower() == ".json":
        payload = json.loads(selected.read_text(encoding="utf-8"))
        csv_value = payload.get("unique_units_summary_csv")
        if not csv_value:
            raise ValueError(
                f"{selected} does not contain 'unique_units_summary_csv'."
            )
        csv_path = Path(csv_value)
        if not csv_path.exists():
            raise FileNotFoundError(f"Referenced unique_units_summary.csv not found: {csv_path}")
        return csv_path

    return selected


def choose_sorting_root_interactively() -> Path | None:
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_folder = filedialog.askdirectory(
            title="Select Sorting batch root (for example: 260224_Sorting)",
            mustexist=True,
        )
        root.destroy()
    except Exception:
        selected_folder = ""

    if not selected_folder:
        raw_value = input(
            "Enter path to sorting batch root folder (for example: 260224_Sorting): "
        ).strip().strip('"').strip("'")
        if not raw_value:
            return None
        selected_folder = raw_value

    selected = Path(selected_folder)
    if not selected.exists():
        raise FileNotFoundError(f"Selected folder not found: {selected}")
    if not selected.is_dir():
        raise NotADirectoryError(f"Selected path is not a folder: {selected}")
    return selected


def resolve_sorting_root(sorting_root: str | None) -> Path | None:
    if sorting_root:
        path = Path(sorting_root)
        if not path.exists():
            raise FileNotFoundError(f"Sorting root folder not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Sorting root is not a folder: {path}")
        return path

    selected_root = choose_sorting_root_interactively()
    if selected_root is None:
        return None

    expected_export_summary = (
        selected_root
        / DEFAULT_ALIGNMENT_EXPORT_FOLDER_NAME
        / DEFAULT_ALIGNMENT_EXPORT_SUMMARY_NAME
    )
    if expected_export_summary.exists():
        return selected_root

    return None


def resolve_export_summary_path(
    *,
    sorting_root: Path | None,
    export_summary: str | None,
) -> Path | None:
    if export_summary:
        return Path(export_summary)

    if sorting_root is None:
        return None

    candidate_export_summary = (
        sorting_root
        / DEFAULT_ALIGNMENT_EXPORT_FOLDER_NAME
        / DEFAULT_ALIGNMENT_EXPORT_SUMMARY_NAME
    )
    if candidate_export_summary.exists():
        return candidate_export_summary
    return None


def resolve_batch_root(
    *,
    sorting_root: Path | None,
    unique_units_csv: Path,
    export_summary_path: Path | None,
) -> Path:
    if sorting_root is not None:
        return sorting_root
    if export_summary_path is not None:
        return export_summary_path.parent.parent
    return unique_units_csv.parent.parent


def load_unique_units_summary(csv_path: Path) -> pd.DataFrame:
    log_status(f"Loading unique-units summary table: {csv_path}")
    df = pd.read_csv(csv_path)
    required_columns = {
        "final_unit_id",
        "shank_id",
        "sg_channel",
        "num_sessions",
        "num_member_units",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {', '.join(missing)}"
        )

    for column in ["final_unit_id", "shank_id", "sg_channel", "num_sessions", "num_member_units"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["final_unit_id", "shank_id", "sg_channel", "num_sessions"]).copy()
    df["final_unit_id"] = df["final_unit_id"].astype(int)
    df["shank_id"] = df["shank_id"].astype(int)
    df["sg_channel"] = df["sg_channel"].astype(int)
    df["num_sessions"] = df["num_sessions"].astype(int)
    df["num_member_units"] = df["num_member_units"].fillna(0).astype(int)
    log_status(f"Loaded {len(df)} unique units from CSV")
    return df


def plot_units_by_exact_session_count(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    max_sessions: int,
    session_label: str = "Session",
    output_name: str = "units_by_exact_session_count.png",
) -> Path:
    log_status(f"Building plot: units by exact {session_label.lower()} count")
    session_counts = (
        df["num_sessions"]
        .value_counts()
        .reindex(range(1, max_sessions + 1), fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(session_counts.index, session_counts.values, color=BAR_COLOR, edgecolor="black", linewidth=0.8)
    annotate_bars(ax, bars, session_counts.astype(int).tolist())
    style_axis(ax)
    ax.set_title(f"Unique Units by Exact Number of {session_label}s", fontsize=14, pad=12)
    ax.set_xlabel(f"Number of {session_label}s Present")
    ax.set_ylabel("Number of Units")
    ax.set_xticks(range(1, max_sessions + 1))
    ax.set_xlim(0.25, max_sessions + 0.75)
    if session_counts.max() > 0:
        ax.set_ylim(0, session_counts.max() * 1.15 + 1)

    output_path = output_dir / output_name
    save_figure(fig, output_path)
    return output_path


def plot_top_channels_by_persistent_units(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    min_sessions_for_stable: int,
    top_n: int,
    session_label: str = "Session",
    output_name: str = "top_channels_by_persistent_units.png",
) -> Path | None:
    log_status(
        "Building plot: top channels by persistent units "
        f"(threshold >= {min_sessions_for_stable} {session_label.lower()}s)"
    )
    filtered = df[df["num_sessions"] >= min_sessions_for_stable].copy()
    if filtered.empty:
        log_status("Skipping top-channels plot because no units met the persistence threshold")
        return None

    grouped = (
        filtered.groupby(["shank_id", "sg_channel"])
        .agg(
            persistent_unit_count=("final_unit_id", "nunique"),
            average_sessions=("num_sessions", "mean"),
        )
        .reset_index()
        .sort_values(
            by=["persistent_unit_count", "average_sessions", "shank_id", "sg_channel"],
            ascending=[False, False, True, True],
        )
        .head(top_n)
    )

    labels = [
        f"sh{int(row.shank_id)} sg{int(row.sg_channel)}"
        for row in grouped.itertuples(index=False)
    ]
    values = grouped["persistent_unit_count"].astype(int).tolist()

    fig_height = max(5, 0.45 * len(grouped) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(labels, values, color="#4f7f4f", edgecolor="black", linewidth=0.8)
    style_axis(ax)
    ax.set_title(
        f"Top Channels by Units Present in >= {min_sessions_for_stable} {session_label}s",
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("Number of Persistent Units")
    ax.set_ylabel("Channel")
    ax.invert_yaxis()

    xmax = max(values) if values else 0
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(0.15, xmax * 0.01),
            bar.get_y() + bar.get_height() / 2.0,
            str(int(value)),
            va="center",
            fontsize=9,
        )

    output_path = output_dir / output_name
    save_figure(fig, output_path)
    return output_path


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


def discover_analyzer_folders_for_batch_root(batch_root: Path) -> list[Path]:
    analyzer_folders = sorted(
        {
            path
            for path in batch_root.rglob(DEFAULT_ANALYZER_FOLDER_NAME)
            if path.is_dir()
        },
        key=lambda path: (normalize_session_name(path.parent.name), path.parent.name, str(path)),
    )
    if not analyzer_folders:
        raise FileNotFoundError(
            f"No {DEFAULT_ANALYZER_FOLDER_NAME} folders found under {batch_root}"
        )
    return analyzer_folders


def is_auto_discarded_by_alignment_rule(
    *,
    amplitude_median: float | None,
    snr: float | None,
    isi_violations_ratio: float | None,
) -> bool:
    amplitude_value = safe_float(amplitude_median)
    snr_value = safe_float(snr)
    isi_value = safe_float(isi_violations_ratio)
    if amplitude_value is None or snr_value is None or isi_value is None:
        return False
    return (
        abs(amplitude_value) < DISCARD_ABS_AMPLITUDE_MAX
        or snr_value < DISCARD_SNR_MAX
        or isi_value > DISCARD_ISI_VIOLATION_MIN
    )


def estimate_active_window_from_spike_times(
    spike_times_s: np.ndarray,
    *,
    lower_percentile: float = ACTIVE_WINDOW_LOWER_PERCENTILE,
    upper_percentile: float = ACTIVE_WINDOW_UPPER_PERCENTILE,
) -> dict[str, float | None]:
    spike_times_s = np.asarray(spike_times_s, dtype=float).ravel()
    spike_times_s = spike_times_s[np.isfinite(spike_times_s)]
    if spike_times_s.size == 0:
        return {
            "active_window_start_s": None,
            "active_window_end_s": None,
            "active_duration_s": None,
        }

    spike_times_s = np.sort(spike_times_s)
    if spike_times_s.size >= 10:
        start_s = float(np.percentile(spike_times_s, lower_percentile))
        end_s = float(np.percentile(spike_times_s, upper_percentile))
    elif spike_times_s.size >= 2:
        start_s = float(spike_times_s[0])
        end_s = float(spike_times_s[-1])
    else:
        start_s = float(spike_times_s[0])
        end_s = float(spike_times_s[0])

    duration_s = max(0.0, end_s - start_s)
    if duration_s <= 0:
        duration_s = None

    return {
        "active_window_start_s": start_s,
        "active_window_end_s": end_s,
        "active_duration_s": duration_s,
    }


def load_quality_metrics_from_export_summary(export_summary_path: Path) -> pd.DataFrame:
    log_status(f"Loading quality metrics from export summary: {export_summary_path}")
    payload = json.loads(export_summary_path.read_text(encoding="utf-8"))
    manifest_rows = payload.get("cross_session_alignment_groups", [])
    if not manifest_rows:
        log_status("No cross-session alignment groups found; skipping quality-metric plots")
        return pd.DataFrame()

    import spikeinterface.full as si

    analyzer_cache: dict[Path, object] = {}
    metrics_cache: dict[Path, dict[int, dict[str, float | int | None]]] = {}
    rows: list[dict] = []

    for group_row in manifest_rows:
        final_group_key = str(group_row.get("final_group_key", ""))
        final_unit_id = safe_int(group_row.get("final_unit_id"))
        shank_id = safe_int(group_row.get("shank_id"))
        unique_session_count = len(
            {
                str(member.get("session_name", "")).strip()
                for member in group_row.get("members", [])
                if str(member.get("session_name", "")).strip()
            }
        )

        for member in group_row.get("members", []):
            output_folder_value = str(member.get("output_folder", "") or "").strip()
            if not output_folder_value:
                continue

            analyzer_folder = Path(output_folder_value) / DEFAULT_ANALYZER_FOLDER_NAME
            if not analyzer_folder.exists():
                continue

            if analyzer_folder not in analyzer_cache:
                analyzer_cache[analyzer_folder] = si.load_sorting_analyzer(
                    folder=analyzer_folder,
                    format="zarr",
                    load_extensions=True,
                )
                metrics_cache[analyzer_folder] = build_metrics_lookup(analyzer_cache[analyzer_folder])

            unit_id = safe_int(member.get("unit_id"))
            if unit_id is None:
                continue

            analyzer = analyzer_cache[analyzer_folder]
            metrics = metrics_cache[analyzer_folder].get(unit_id, {})
            spike_train_samples = analyzer.sorting.get_unit_spike_train(
                unit_id=int(unit_id),
                segment_index=0,
            )
            sampling_frequency = float(analyzer.sorting.get_sampling_frequency())
            spike_times_s = np.asarray(spike_train_samples, dtype=float) / sampling_frequency
            active_window = estimate_active_window_from_spike_times(spike_times_s)
            active_duration_s = safe_float(active_window["active_duration_s"])
            num_spikes = safe_float(metrics.get("num_spikes"))
            full_session_firing_rate = safe_float(metrics.get("firing_rate"))
            active_window_firing_rate = (
                float(num_spikes / active_duration_s)
                if num_spikes is not None and active_duration_s is not None and active_duration_s > 0
                else None
            )

            try:
                total_duration_s = float(analyzer.recording.get_num_frames()) / float(analyzer.recording.get_sampling_frequency())
            except Exception:
                total_duration_s = None

            rows.append(
                {
                    "final_group_key": final_group_key,
                    "final_unit_id": final_unit_id,
                    "shank_id": shank_id,
                    "session_name": str(member.get("session_name", "")),
                    "session_index": safe_int(member.get("session_index")),
                    "unit_id": unit_id,
                    "num_sessions": unique_session_count,
                    "amplitude_median": safe_float(metrics.get("amplitude_median")),
                    "amplitude_median_abs": (
                        abs(safe_float(metrics.get("amplitude_median")))
                        if safe_float(metrics.get("amplitude_median")) is not None
                        else None
                    ),
                    "firing_rate_full_session": full_session_firing_rate,
                    "firing_rate_active_window": active_window_firing_rate,
                    "isi_violations_ratio": safe_float(metrics.get("isi_violations_ratio")),
                    "snr": safe_float(metrics.get("snr")),
                    "num_spikes": num_spikes,
                    "session_duration_s": total_duration_s,
                    "active_window_start_s": safe_float(active_window["active_window_start_s"]),
                    "active_window_end_s": safe_float(active_window["active_window_end_s"]),
                    "active_duration_s": active_duration_s,
                    "active_fraction_of_session": (
                        float(active_duration_s / total_duration_s)
                        if active_duration_s is not None and total_duration_s is not None and total_duration_s > 0
                        else None
                    ),
                }
            )

    quality_df = pd.DataFrame(rows)
    log_status(f"Loaded quality-metric rows: {len(quality_df)}")
    return quality_df


def load_quality_metrics_all_units_from_batch_root(batch_root: Path) -> pd.DataFrame:
    log_status(f"Loading all session-unit quality metrics from batch root: {batch_root}")
    import spikeinterface.full as si

    analyzer_folders = discover_analyzer_folders_for_batch_root(batch_root)
    normalized_session_names = [
        normalize_session_name(analyzer_folder.parent.name)
        for analyzer_folder in analyzer_folders
    ]
    ordered_session_names = sorted(set(normalized_session_names))
    session_index_lookup = {
        session_name: index for index, session_name in enumerate(ordered_session_names)
    }

    rows: list[dict] = []
    total_units_seen = 0
    total_units_kept = 0
    total_units_discarded = 0
    for analyzer_index, analyzer_folder in enumerate(analyzer_folders, start=1):
        output_folder = analyzer_folder.parent
        session_name = normalize_session_name(output_folder.name)
        session_index = session_index_lookup[session_name]
        log_status(
            f"Loading analyzer {analyzer_index}/{len(analyzer_folders)} for session {session_index}: {analyzer_folder}"
        )
        analyzer = si.load_sorting_analyzer(
            folder=analyzer_folder,
            format="zarr",
            load_extensions=True,
        )
        metrics_lookup = build_metrics_lookup(analyzer)

        try:
            total_duration_s = float(analyzer.recording.get_num_frames()) / float(analyzer.recording.get_sampling_frequency())
        except Exception:
            total_duration_s = None

        for unit_id in analyzer.sorting.get_unit_ids():
            unit_id = int(unit_id)
            total_units_seen += 1
            metrics = metrics_lookup.get(unit_id, {})
            spike_train_samples = analyzer.sorting.get_unit_spike_train(
                unit_id=unit_id,
                segment_index=0,
            )
            sampling_frequency = float(analyzer.sorting.get_sampling_frequency())
            spike_times_s = np.asarray(spike_train_samples, dtype=float) / sampling_frequency
            active_window = estimate_active_window_from_spike_times(spike_times_s)
            active_duration_s = safe_float(active_window["active_duration_s"])
            num_spikes = safe_float(metrics.get("num_spikes"))
            amplitude_median = safe_float(metrics.get("amplitude_median"))
            snr = safe_float(metrics.get("snr"))
            isi_violations_ratio = safe_float(metrics.get("isi_violations_ratio"))

            if is_auto_discarded_by_alignment_rule(
                amplitude_median=amplitude_median,
                snr=snr,
                isi_violations_ratio=isi_violations_ratio,
            ):
                total_units_discarded += 1
                continue

            total_units_kept += 1

            rows.append(
                {
                    "source": "all_units_in_batch",
                    "session_name": session_name,
                    "session_index": session_index,
                    "unit_id": unit_id,
                    "amplitude_median": amplitude_median,
                    "amplitude_median_abs": (
                        abs(amplitude_median)
                        if amplitude_median is not None
                        else None
                    ),
                    "firing_rate_full_session": safe_float(metrics.get("firing_rate")),
                    "firing_rate_active_window": (
                        float(num_spikes / active_duration_s)
                        if num_spikes is not None and active_duration_s is not None and active_duration_s > 0
                        else None
                    ),
                    "isi_violations_ratio": isi_violations_ratio,
                    "snr": snr,
                    "num_spikes": num_spikes,
                    "session_duration_s": total_duration_s,
                    "active_window_start_s": safe_float(active_window["active_window_start_s"]),
                    "active_window_end_s": safe_float(active_window["active_window_end_s"]),
                    "active_duration_s": active_duration_s,
                    "active_fraction_of_session": (
                        float(active_duration_s / total_duration_s)
                        if active_duration_s is not None and total_duration_s is not None and total_duration_s > 0
                        else None
                    ),
                }
            )

    quality_df = pd.DataFrame(rows)
    log_status(
        "All-units session loader: "
        f"seen {total_units_seen} units, kept {total_units_kept}, "
        f"filtered by alignment discard rule {total_units_discarded}"
    )
    log_status(f"Loaded all-unit session quality rows: {len(quality_df)}")
    return quality_df


def plot_quality_metrics_vs_persistence(
    quality_df: pd.DataFrame,
    output_dir: Path,
    *,
    max_sessions: int,
    session_label: str = "Session",
    output_name: str = "quality_metrics_vs_persistence.png",
) -> Path | None:
    if quality_df.empty:
        log_status("Skipping quality-metrics figure because no quality data was available")
        return None

    log_status(f"Building plot: quality metrics vs cross-{session_label.lower()} persistence")
    group_metrics = (
        quality_df.groupby(["final_group_key", "num_sessions", "shank_id"], dropna=False)
        .agg(
            snr=("snr", "median"),
            amplitude_median_abs=("amplitude_median_abs", "median"),
            firing_rate_active_window=("firing_rate_active_window", "median"),
            isi_violations_ratio=("isi_violations_ratio", "median"),
        )
        .reset_index()
    )
    if group_metrics.empty:
        return None

    metric_specs = [
        ("snr", "Median SNR", "higher is better"),
        ("amplitude_median_abs", "Median Absolute Amplitude", "uV"),
        ("firing_rate_active_window", "Median Active-Window Firing Rate", "Hz"),
        ("isi_violations_ratio", "Median ISI Violations Ratio", "lower is better"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for ax, (metric_name, title, y_label) in zip(axes, metric_specs):
        metric_df = group_metrics.dropna(subset=[metric_name]).copy()
        if metric_df.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        x = metric_df["num_sessions"].to_numpy(dtype=float)
        y = metric_df[metric_name].to_numpy(dtype=float)
        jitter = np.linspace(-0.12, 0.12, num=len(metric_df)) if len(metric_df) > 1 else np.array([0.0])
        scatter = ax.scatter(
            x + jitter,
            y,
            alpha=0.65,
            s=28,
            color=BAR_COLOR,
            edgecolors="none",
            label="Blue dot: one aligned final unit",
        )

        summary = (
            metric_df.groupby("num_sessions")[metric_name]
            .median()
            .reindex(range(1, max_sessions + 1))
        )
        valid_summary = summary.dropna()
        summary_line = None
        if not valid_summary.empty:
            (summary_line,) = ax.plot(
                valid_summary.index.to_numpy(dtype=float),
                valid_summary.values.astype(float),
                color=ACCENT_COLOR,
                linewidth=2.1,
                marker="o",
                markersize=4,
                label="Orange line: median across aligned final units",
            )

        style_axis(ax)
        ax.set_title(title)
        ax.set_xlabel(f"Number of {session_label}s Present")
        ax.set_ylabel(y_label)
        ax.set_xticks(range(1, max_sessions + 1, max(1, max_sessions // 12)))
        ax.set_xlim(0.5, max_sessions + 0.5)
        if summary_line is not None:
            ax.legend(loc="best", fontsize=8, frameon=False)
        else:
            ax.legend(handles=[scatter], loc="best", fontsize=8, frameon=False)

    fig.suptitle(f"Quality Metrics vs Cross-{session_label} Persistence", fontsize=16, y=1.01)
    fig.tight_layout()

    output_path = output_dir / output_name
    save_figure(fig, output_path)
    return output_path


def plot_quality_metrics_by_session(
    quality_df: pd.DataFrame,
    output_dir: Path,
    *,
    max_sessions: int,
    session_label: str = "Session",
    output_name: str = "quality_metrics_by_session.png",
) -> Path | None:
    if quality_df.empty:
        log_status("Skipping session-index quality figure because no quality data was available")
        return None

    log_status(f"Building plot: quality metrics by {session_label.lower()} index")
    session_quality_df = quality_df.dropna(subset=["session_index"]).copy()
    if session_quality_df.empty:
        log_status("Skipping session-index quality figure because session indices were unavailable")
        return None

    metric_specs = [
        ("snr", "SNR by Session", "higher is better"),
        ("amplitude_median_abs", "Absolute Amplitude by Session", "uV"),
        ("firing_rate_active_window", "Active-Window Firing Rate by Session", "Hz"),
        ("isi_violations_ratio", "ISI Violations Ratio by Session", "lower is better"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for ax, (metric_name, title, y_label) in zip(axes, metric_specs):
        metric_df = session_quality_df.dropna(subset=[metric_name]).copy()
        if metric_df.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        metric_df = metric_df[np.isfinite(metric_df[metric_name].to_numpy(dtype=float))].copy()
        if metric_df.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        log_status(
            f"Session quality panel '{metric_name}': plotting {len(metric_df)} unit/session points"
        )

        metric_df = metric_df.sort_values(["session_index", "unit_id"]).reset_index(drop=True)
        x_base = metric_df["session_index"].to_numpy(dtype=float)
        if len(metric_df) > 1:
            repeats = metric_df.groupby("session_index").cumcount().to_numpy(dtype=float)
            counts = metric_df.groupby("session_index")["session_index"].transform("count").to_numpy(dtype=float)
            jitter = np.where(counts > 1, (repeats / (counts - 1 + 1e-12) - 0.5) * 0.36, 0.0)
        else:
            jitter = np.array([0.0])

        y_values = metric_df[metric_name].to_numpy(dtype=float)
        y_plot = y_values.copy()
        if metric_name == "isi_violations_ratio":
            zero_mask = np.isclose(y_values, 0.0, atol=1e-12)
            zero_count = int(np.sum(zero_mask))
            if zero_count > 0:
                log_status(
                    f"Session quality panel '{metric_name}': {zero_count} zero-valued points will overlap at y=0 without display jitter"
                )
                zero_session_offsets = metric_df.loc[zero_mask].groupby("session_index").cumcount().to_numpy(dtype=float)
                zero_session_counts = (
                    metric_df.loc[zero_mask]
                    .groupby("session_index")["session_index"]
                    .transform("count")
                    .to_numpy(dtype=float)
                )
                zero_y_jitter = np.where(
                    zero_session_counts > 1,
                    (zero_session_offsets / (zero_session_counts - 1 + 1e-12)) * 0.02,
                    0.0,
                )
                y_plot[zero_mask] = zero_y_jitter

        scatter = ax.scatter(
            x_base + jitter,
            y_plot,
            alpha=0.5,
            s=18,
            color=BAR_COLOR,
            edgecolors="none",
            label=(
                "Blue dot: one unit in that session"
                if metric_name != "isi_violations_ratio"
                else "Blue dot: one unit in that session (zero values vertically jittered for visibility)"
            ),
        )

        summary = (
            metric_df.groupby("session_index")[metric_name]
            .median()
            .reindex(range(0, max_sessions))
        )
        valid_summary = summary.dropna()
        summary_line = None
        if not valid_summary.empty:
            (summary_line,) = ax.plot(
                valid_summary.index.to_numpy(dtype=float),
                valid_summary.values.astype(float),
                color=ACCENT_COLOR,
                linewidth=2.1,
                marker="o",
                markersize=4,
                label="Orange line: median across units in that session",
            )

        style_axis(ax)
        ax.set_title(title)
        ax.set_xlabel(f"{session_label} Index")
        ax.set_ylabel(y_label)
        tick_step = max(1, max_sessions // 12)
        ax.set_xticks(range(0, max_sessions, tick_step))
        ax.set_xlim(-0.5, max_sessions - 0.5)
        if summary_line is not None:
            ax.legend(loc="best", fontsize=8, frameon=False)
        else:
            ax.legend(handles=[scatter], loc="best", fontsize=8, frameon=False)

    fig.suptitle(f"Quality Metrics by {session_label} Index", fontsize=16, y=1.01)
    fig.tight_layout()

    output_path = output_dir / output_name
    save_figure(fig, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create presentation-ready plots from Alignment_html and Combined_NWB+Sorting+Analyze outputs."
    )
    parser.add_argument(
        "--sorting-root",
        help=(
            "Sorting batch root folder that contains "
            f"{DEFAULT_ALIGNMENT_EXPORT_FOLDER_NAME}/{DEFAULT_ALIGNMENT_EXPORT_SUMMARY_NAME} "
            "(for example: 260224_Sorting)"
        ),
    )
    parser.add_argument(
        "--unique-units-csv",
        help="Path to unique_units_summary.csv exported by Alignment_html.py",
    )
    parser.add_argument(
        "--export-summary",
        help="Optional path to export_summary.json from Alignment_html.py for quality-metric plots",
    )
    parser.add_argument(
        "--output-dir",
        help="Folder where plots should be written. Defaults to stats/ or Stats/ under the batch root.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=DEFAULT_MAX_SESSIONS,
        help=f"Maximum session count shown on plots (default: {DEFAULT_MAX_SESSIONS})",
    )
    parser.add_argument(
        "--stable-threshold",
        type=int,
        default=2,
        help="Minimum number of sessions used to define a persistent unit for the channel-ranking plot",
    )
    parser.add_argument(
        "--top-n-channels",
        type=int,
        default=20,
        help="How many channels to show in the persistent-channel ranking plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_status("Starting presentations pipeline")

    should_try_sorting_root = bool(args.sorting_root) or (not args.unique_units_csv and not args.export_summary)
    resolved_sorting_root = resolve_sorting_root(args.sorting_root) if should_try_sorting_root else None

    unique_units_csv = resolve_unique_units_csv(
        sorting_root=resolved_sorting_root,
        unique_units_csv=args.unique_units_csv,
        export_summary=args.export_summary,
    )
    output_dir = resolve_output_dir(
        sorting_root=resolved_sorting_root,
        unique_units_csv=unique_units_csv,
        output_dir=args.output_dir,
    )

    df = load_unique_units_summary(unique_units_csv)
    plot_paths: list[Path] = []

    plot_paths.append(
        plot_units_by_exact_session_count(
            df,
            output_dir,
            max_sessions=args.max_sessions,
        )
    )

    top_channels_plot = plot_top_channels_by_persistent_units(
        df,
        output_dir,
        min_sessions_for_stable=args.stable_threshold,
        top_n=args.top_n_channels,
    )
    if top_channels_plot is not None:
        plot_paths.append(top_channels_plot)

    resolved_export_summary_for_quality = resolve_export_summary_path(
        sorting_root=resolved_sorting_root,
        export_summary=args.export_summary,
    )
    resolved_batch_root = resolve_batch_root(
        sorting_root=resolved_sorting_root,
        unique_units_csv=unique_units_csv,
        export_summary_path=resolved_export_summary_for_quality,
    )

    if resolved_export_summary_for_quality:
        log_status(f"Quality plots enabled from: {resolved_export_summary_for_quality}")
        quality_df = load_quality_metrics_from_export_summary(resolved_export_summary_for_quality)
        if not quality_df.empty:
            quality_csv_path = output_dir / "quality_metrics_source_table.csv"
            log_status(f"Saving quality-metric source table: {quality_csv_path}")
            quality_df.to_csv(quality_csv_path, index=False)
            plot_paths.append(quality_csv_path)

            quality_plot_path = plot_quality_metrics_vs_persistence(
                quality_df,
                output_dir,
                max_sessions=args.max_sessions,
            )
            if quality_plot_path is not None:
                plot_paths.append(quality_plot_path)

        else:
            log_status("No quality metrics were available to plot")
    else:
        log_status("No export summary available for quality-metric plots; making CSV-based plots only")

    try:
        all_units_quality_df = load_quality_metrics_all_units_from_batch_root(resolved_batch_root)
    except Exception as exc:
        log_status(f"Could not load all-unit session quality metrics: {exc}")
        all_units_quality_df = pd.DataFrame()

    if not all_units_quality_df.empty:
        quality_by_session_plot_path = plot_quality_metrics_by_session(
            all_units_quality_df,
            output_dir,
            max_sessions=args.max_sessions,
        )
        if quality_by_session_plot_path is not None:
            plot_paths.append(quality_by_session_plot_path)
    else:
        log_status("No all-unit session quality data were available for the session-index plot")

    manifest_path = output_dir / "presentation_plots_manifest.json"
    log_status(f"Writing plot manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(
            {
                "unique_units_csv": str(unique_units_csv),
                "export_summary": (
                    str(resolved_export_summary_for_quality)
                    if resolved_export_summary_for_quality is not None
                    else None
                ),
                "num_unique_units": int(len(df)),
                "max_sessions": int(args.max_sessions),
                "plots": [str(path) for path in plot_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log_status("Presentations pipeline complete")
    print(f"Saved {len(plot_paths)} output file(s) to: {output_dir}")
    for path in plot_paths:
        print(f" - {path}")
    print(f" - {manifest_path}")


if __name__ == "__main__":
    main()
