from __future__ import annotations

"""Run true-label LDA for 10-minute and 20-minute time-of-day bins.

The script aggregates the organized minute-long feature cache into compact
time-bin-wide matrices, then writes LDA plots, confusion matrices, logs, and
summary JSON for each requested bin size.
"""

import argparse
import json
import math
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_MINUTE_LONG = PROJECT_DIR / "derived" / "threshold_minute_features_long.csv"
DEFAULT_SCHEDULE = (
    PROJECT_DIR
    / "outputs"
    / "combined_population_summary"
    / "threshold_sham_drug_marker_schedule.json"
)
DEFAULT_OUT_BASE = PROJECT_DIR / "outputs" / "LDA_threshold" / "organized_timebin_feature_lda_true_labels"

METRICS = ("firing_rate_hz", "amplitude_ptp_uv", "cv2", "peak_to_trough_ms")
FEATURE_CONDITIONS = {
    "fr_only": ("firing_rate_hz",),
    "fr_amp": ("firing_rate_hz", "amplitude_ptp_uv"),
    "fr_cv2": ("firing_rate_hz", "cv2"),
    "fr_peak_to_trough": ("firing_rate_hz", "peak_to_trough_ms"),
    "multi_feature": METRICS,
}
PHASE_COLORS = {
    "baseline": "#6b7280",
    "sham": "#b45309",
    "drug": "#b91c1c",
}
PHASE_MARKERS = {
    "baseline": "o",
    "sham": "s",
    "drug": "^",
}


def log(message: str) -> None:
    print(f"[run_organized_timebin_lda] {message}", flush=True)


def load_schedule(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assign_phase(times: pd.Series, schedule: dict | None) -> pd.Series:
    phases = pd.Series("baseline", index=times.index, dtype=object)
    if not schedule:
        return phases
    parsed = pd.to_datetime(times)
    for interval in schedule.get("sham_intervals", []):
        start = pd.Timestamp(interval["start"])
        end = pd.Timestamp(interval["end"])
        phases.loc[(parsed >= start) & (parsed < end)] = "sham"
    for interval in schedule.get("drug_intervals", []):
        start = pd.Timestamp(interval["start"])
        end = pd.Timestamp(interval["end"])
        phases.loc[(parsed >= start) & (parsed < end)] = "drug"
    return phases


def feature_columns_for_condition(columns: list[str], metrics: tuple[str, ...]) -> list[str]:
    selected = []
    for column in columns:
        if "__" not in column:
            continue
        metric = column.rsplit("__", 1)[-1]
        if metric in metrics:
            selected.append(column)
    return selected


def fit_projection(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, LinearDiscriminantAnalysis]:
    pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        LinearDiscriminantAnalysis(n_components=3),
    )
    projection = pipeline.fit_transform(x, y)
    return projection, pipeline.named_steps["lineardiscriminantanalysis"]


def cross_validated_predictions(x: np.ndarray, y: np.ndarray, n_splits: int, random_seed: int) -> np.ndarray:
    min_count = int(pd.Series(y).value_counts().min())
    splits = max(2, min(int(n_splits), min_count))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_seed)
    pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        LinearDiscriminantAnalysis(),
    )
    return cross_val_predict(pipeline, x, y, cv=cv)


def aggregate_timebin_wide(
    minute_long_csv: Path,
    out_csv: Path,
    *,
    bin_minutes: int,
    chunksize: int,
) -> dict:
    group_columns = [
        "calendar_day",
        "time_bin_label",
        "bin_start_minute_of_day",
        "unit_id",
        "pair_id",
        "sg_ch",
        "threshold_min_uv",
        "threshold_label",
    ]
    accum: dict[tuple, dict[str, float]] = {}
    source_rows = 0
    for chunk_index, chunk in enumerate(pd.read_csv(minute_long_csv, chunksize=chunksize), start=1):
        source_rows += len(chunk)
        minute_of_day = chunk["clock_hour_of_day"].astype(int) * 60 + chunk["clock_minute_of_hour"].astype(int)
        chunk["time_bin_label"] = (minute_of_day // bin_minutes).astype(int)
        chunk["bin_start_minute_of_day"] = (chunk["time_bin_label"] * bin_minutes).astype(int)
        chunk["_minute_unit_rows"] = 1
        grouped = chunk.groupby(group_columns, dropna=False, sort=False)
        agg_kwargs = {"n_minute_unit_rows": ("_minute_unit_rows", "sum")}
        for metric in METRICS:
            agg_kwargs[f"{metric}_sum"] = (metric, "sum")
            agg_kwargs[f"{metric}_count"] = (metric, "count")
        stats = grouped.agg(**agg_kwargs).reset_index()
        for row in stats.itertuples(index=False):
            key = tuple(getattr(row, column) for column in group_columns)
            entry = accum.setdefault(key, {"n_minute_unit_rows": 0})
            entry["n_minute_unit_rows"] += int(row.n_minute_unit_rows)
            for metric in METRICS:
                entry[f"{metric}_sum"] = entry.get(f"{metric}_sum", 0.0) + float(getattr(row, f"{metric}_sum"))
                entry[f"{metric}_count"] = entry.get(f"{metric}_count", 0) + int(getattr(row, f"{metric}_count"))
        log(f"aggregated {bin_minutes}-min chunk {chunk_index}: source rows so far={source_rows:,}")

    rows = []
    for key, entry in accum.items():
        row = dict(zip(group_columns, key))
        start = pd.Timestamp(row["calendar_day"]) + pd.to_timedelta(int(row["bin_start_minute_of_day"]), unit="m")
        center_minute = int(row["bin_start_minute_of_day"]) + bin_minutes / 2.0
        row["bin_start_datetime"] = start
        row["bin_center_minute_of_day"] = float(center_minute)
        row["clock_phase_rad"] = float((center_minute / 1440.0) * 2.0 * math.pi)
        row["clock_phase_sin"] = math.sin(row["clock_phase_rad"])
        row["clock_phase_cos"] = math.cos(row["clock_phase_rad"])
        row["n_minute_unit_rows"] = int(entry["n_minute_unit_rows"])
        for metric in METRICS:
            count = entry.get(f"{metric}_count", 0)
            row[metric] = entry.get(f"{metric}_sum", 0.0) / count if count else np.nan
        rows.append(row)

    long = pd.DataFrame(rows)
    long = long.sort_values(["bin_start_datetime", "sg_ch", "threshold_min_uv", "pair_id"])
    id_columns = [
        "calendar_day",
        "time_bin_label",
        "bin_start_minute_of_day",
        "bin_center_minute_of_day",
        "bin_start_datetime",
        "clock_phase_rad",
        "clock_phase_sin",
        "clock_phase_cos",
    ]
    metric_frames = []
    for metric in METRICS:
        pivot = long.pivot_table(
            index=id_columns,
            columns="pair_id",
            values=metric,
            aggfunc="mean",
            sort=False,
        )
        pivot.columns = [f"{pair_id}__{metric}" for pair_id in pivot.columns]
        metric_frames.append(pivot)
    wide = pd.concat(metric_frames, axis=1).reset_index()
    wide = wide.sort_values("bin_start_datetime")
    wide.to_csv(out_csv, index=False)
    return {
        "source_minute_unit_rows": int(source_rows),
        "aggregated_long_rows": int(len(long)),
        "wide_rows": int(wide.shape[0]),
        "wide_columns": int(wide.shape[1]),
    }


def plot_2d(
    projection: np.ndarray,
    table: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    include_phase_markers: bool,
    bin_minutes: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.plot(projection[:, 0], projection[:, 1], color="#cbd5e1", linewidth=0.7, alpha=0.75, zorder=1)
    color_values = table["bin_start_minute_of_day"]
    if include_phase_markers:
        for phase in ["baseline", "sham", "drug"]:
            mask = table["injection_phase"].to_numpy() == phase
            if not mask.any():
                continue
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                c=color_values.loc[mask],
                cmap="twilight_shifted",
                vmin=0,
                vmax=1440 - bin_minutes,
                marker=PHASE_MARKERS[phase],
                edgecolors=PHASE_COLORS[phase],
                linewidths=0.8,
                s=34 if bin_minutes == 10 else 40,
                label=phase,
                zorder=2,
            )
        ax.legend(title="phase", loc="best", frameon=False)
    else:
        sc = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c=color_values,
            cmap="twilight_shifted",
            vmin=0,
            vmax=1440 - bin_minutes,
            s=30 if bin_minutes == 10 else 38,
            edgecolors="black",
            linewidths=0.2,
            zorder=2,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label(f"{bin_minutes}-min bin start, minute of day")
    ax.set_title(title)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_3d(
    projection: np.ndarray,
    table: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    include_phase_markers: bool,
    bin_minutes: int,
) -> None:
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    z = projection[:, 2] if projection.shape[1] >= 3 else np.zeros(projection.shape[0])
    ax.plot(projection[:, 0], projection[:, 1], z, color="#cbd5e1", linewidth=0.7, alpha=0.75)
    color_values = table["bin_start_minute_of_day"]
    if include_phase_markers:
        for phase in ["baseline", "sham", "drug"]:
            mask = table["injection_phase"].to_numpy() == phase
            if not mask.any():
                continue
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                z[mask],
                c=color_values.loc[mask],
                cmap="twilight_shifted",
                vmin=0,
                vmax=1440 - bin_minutes,
                marker=PHASE_MARKERS[phase],
                edgecolors=PHASE_COLORS[phase],
                linewidths=0.75,
                s=26 if bin_minutes == 10 else 32,
                label=phase,
            )
        ax.legend(title="phase", loc="best", frameon=False)
    else:
        sc = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            z,
            c=color_values,
            cmap="twilight_shifted",
            vmin=0,
            vmax=1440 - bin_minutes,
            s=24 if bin_minutes == 10 else 30,
            edgecolors="black",
            linewidths=0.15,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.8)
        cbar.set_label(f"{bin_minutes}-min bin start, minute of day")
    ax.set_title(title)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_zlabel("LD3" if projection.shape[1] >= 3 else "0")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    out_path: Path,
    *,
    title: str,
    bin_minutes: int,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    size = 14 if len(labels) > 100 else 12
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    include_values = len(labels) <= 72
    disp.plot(ax=ax, cmap="viridis", values_format=".2f", colorbar=True, include_values=include_values)
    tick_step = max(1, 60 // bin_minutes)
    tick_positions = list(range(0, len(labels), tick_step))
    tick_labels = [f"{(label * bin_minutes) // 60:02d}:00" for label in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)
    ax.set_title(title)
    ax.set_xlabel(f"predicted {bin_minutes}-min bin")
    ax.set_ylabel(f"true {bin_minutes}-min bin")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def cyclic_bin_distance(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int) -> np.ndarray:
    raw = np.abs(y_true.astype(int) - y_pred.astype(int))
    return np.minimum(raw, n_bins - raw)


def run_bin(args: argparse.Namespace, bin_minutes: int) -> dict:
    started = time.perf_counter()
    out_dir = args.out_base.resolve() / f"bin_{bin_minutes}min"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_csv = out_dir / f"organized_{bin_minutes}min_timebin_matrix_wide.csv"
    manifest_path = out_dir / f"organized_{bin_minutes}min_timebin_matrix_manifest.json"
    log(f"building/loading {bin_minutes}-min wide matrix")
    if matrix_csv.exists() and not args.rebuild_cache:
        table = pd.read_csv(matrix_csv)
        aggregation = {"cache_reused": True}
    else:
        aggregation = aggregate_timebin_wide(
            args.minute_long_csv,
            matrix_csv,
            bin_minutes=bin_minutes,
            chunksize=args.chunksize,
        )
        aggregation["cache_reused"] = False
        manifest_path.write_text(json.dumps(aggregation, indent=2), encoding="utf-8")
        table = pd.read_csv(matrix_csv)

    table["bin_start_datetime"] = pd.to_datetime(table["bin_start_datetime"])
    table = table.sort_values("bin_start_datetime").reset_index(drop=True)
    schedule = load_schedule(args.schedule_json)
    table["injection_phase"] = assign_phase(table["bin_start_datetime"], schedule)
    y = table["time_bin_label"].astype(int).to_numpy()
    n_bins = 1440 // bin_minutes
    labels = list(range(n_bins))
    all_columns = table.columns.tolist()
    results = {
        "bin_minutes": int(bin_minutes),
        "label_definition": f"time_bin_label=floor(minute_of_day/{bin_minutes})",
        "input_minute_long_csv": str(args.minute_long_csv.resolve()),
        "wide_matrix_csv": str(matrix_csv.resolve()),
        "wide_matrix_manifest": str(manifest_path.resolve()),
        "output_dir": str(out_dir),
        "n_samples": int(len(table)),
        "n_time_bin_labels": int(pd.Series(y).nunique()),
        "expected_time_bin_labels_per_day": int(n_bins),
        "phase_counts": table["injection_phase"].value_counts().to_dict(),
        "aggregation": aggregation,
        "conditions": {},
    }
    log(f"{bin_minutes}-min samples={len(table):,}, labels={pd.Series(y).nunique()}")

    for condition, metrics in FEATURE_CONDITIONS.items():
        condition_started = time.perf_counter()
        feature_columns = feature_columns_for_condition(all_columns, metrics)
        log(f"{bin_minutes}-min fitting {condition}: {len(feature_columns)} features")
        x = table[feature_columns].to_numpy(dtype=float)
        projection, lda = fit_projection(x, y)
        y_pred = cross_validated_predictions(x, y, args.cv_splits, args.random_seed)
        distances = cyclic_bin_distance(y, y_pred, n_bins)
        prefix = f"organized_{bin_minutes}min_{condition}"
        paths = {
            "plot_2d": out_dir / f"{prefix}_lda2d_timebin.png",
            "plot_3d": out_dir / f"{prefix}_lda3d_timebin.png",
            "plot_2d_sham_drug": out_dir / f"{prefix}_lda2d_timebin_sham_drug.png",
            "plot_3d_sham_drug": out_dir / f"{prefix}_lda3d_timebin_sham_drug.png",
            "confusion_matrix": out_dir / f"{prefix}_confusion_matrix_timebin_cv.png",
        }
        title_base = f"{condition} LDA, {bin_minutes}-min time-of-day labels"
        plot_2d(projection, table, paths["plot_2d"], title=title_base, include_phase_markers=False, bin_minutes=bin_minutes)
        plot_3d(projection, table, paths["plot_3d"], title=title_base, include_phase_markers=False, bin_minutes=bin_minutes)
        plot_2d(
            projection,
            table,
            paths["plot_2d_sham_drug"],
            title=f"{title_base}, sham/drug markers",
            include_phase_markers=True,
            bin_minutes=bin_minutes,
        )
        plot_3d(
            projection,
            table,
            paths["plot_3d_sham_drug"],
            title=f"{title_base}, sham/drug markers",
            include_phase_markers=True,
            bin_minutes=bin_minutes,
        )
        plot_confusion(
            y,
            y_pred,
            labels,
            paths["confusion_matrix"],
            title=f"{condition} cross-validated confusion matrix",
            bin_minutes=bin_minutes,
        )
        results["conditions"][condition] = {
            "metrics": list(metrics),
            "n_features": int(len(feature_columns)),
            "n_lda_dimensions": int(projection.shape[1]),
            "explained_variance_ratio": [float(v) for v in getattr(lda, "explained_variance_ratio_", [])],
            "cv_accuracy": float(accuracy_score(y, y_pred)),
            "cv_balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "within_1_bin_cyclic_accuracy": float(np.mean(distances <= 1)),
            "within_2_bins_cyclic_accuracy": float(np.mean(distances <= 2)),
            "mean_cyclic_bin_error": float(np.mean(distances)),
            "mean_cyclic_minute_error": float(np.mean(distances) * bin_minutes),
            "elapsed_seconds": float(time.perf_counter() - condition_started),
            "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        }

    results["elapsed_seconds"] = float(time.perf_counter() - started)
    summary_path = out_dir / f"organized_{bin_minutes}min_timebin_lda_summary.json"
    results["summary_json"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-long-csv", type=Path, default=DEFAULT_MINUTE_LONG)
    parser.add_argument("--schedule-json", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument("--bin-minutes", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--chunksize", type=int, default=750_000)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_base.mkdir(parents=True, exist_ok=True)
    run_log_path = args.out_base / "organized_timebin_lda_run_log.txt"
    combined_summary_path = args.out_base / "organized_timebin_lda_summary.json"
    with run_log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            started = time.perf_counter()
            summaries = {}
            for bin_minutes in args.bin_minutes:
                if 1440 % bin_minutes != 0:
                    raise ValueError(f"bin_minutes must divide 1440 exactly: {bin_minutes}")
                summaries[str(bin_minutes)] = run_bin(args, int(bin_minutes))
            combined = {
                "bin_minutes": [int(v) for v in args.bin_minutes],
                "output_base": str(args.out_base.resolve()),
                "run_log": str(run_log_path.resolve()),
                "summaries": summaries,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            combined_summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
            log(f"wrote combined summary: {combined_summary_path}")
            log(f"finished in {combined['elapsed_seconds']:.2f} seconds")
    print(f"Wrote time-bin LDA outputs to: {args.out_base.resolve()}")
    print(f"Summary: {combined_summary_path.resolve()}")
    print(f"Run log: {run_log_path.resolve()}")


if __name__ == "__main__":
    main()
