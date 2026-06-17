from __future__ import annotations

"""Run compact LDA plots from the organized hourly feature matrix.

This script intentionally writes only plots, confusion matrices, a run log, and
summary JSON. For shuffled-label controls, it also writes the shuffled label
assignment table needed to reproduce the null labels.
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
DEFAULT_MATRIX = PROJECT_DIR / "derived" / "threshold_hour_matrix_wide.csv"
DEFAULT_SCHEDULE = (
    PROJECT_DIR
    / "outputs"
    / "combined_population_summary"
    / "threshold_sham_drug_marker_schedule.json"
)
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs" / "LDA_threshold" / "organized_hourly_feature_lda"
DEFAULT_SHUFFLE_OUT_DIR = PROJECT_DIR / "outputs" / "LDA_threshold" / "organized_hourly_feature_lda_label_shuffle"
DEFAULT_CYCLE25_OUT_DIR = PROJECT_DIR / "outputs" / "LDA_threshold" / "organized_hourly_feature_lda_cycle25_true_labels"

FEATURE_CONDITIONS = {
    "fr_only": ("firing_rate_hz",),
    "fr_amp": ("firing_rate_hz", "amplitude_ptp_uv"),
    "fr_cv2": ("firing_rate_hz", "cv2"),
    "fr_peak_to_trough": ("firing_rate_hz", "peak_to_trough_ms"),
    "multi_feature": ("firing_rate_hz", "amplitude_ptp_uv", "cv2", "peak_to_trough_ms"),
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
    print(f"[run_organized_hourly_lda] {message}", flush=True)


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


def fit_projection(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, LinearDiscriminantAnalysis, object]:
    pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        LinearDiscriminantAnalysis(n_components=3),
    )
    projection = pipeline.fit_transform(x, y)
    lda = pipeline.named_steps["lineardiscriminantanalysis"]
    return projection, lda, pipeline


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


def plot_2d(
    projection: np.ndarray,
    table: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    include_phase_markers: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if projection.shape[1] < 2:
        raise ValueError("Need at least two LDA dimensions for 2D plot")
    ax.plot(
        projection[:, 0],
        projection[:, 1],
        color="#cbd5e1",
        linewidth=0.8,
        alpha=0.8,
        zorder=1,
    )
    # The table is chronologically sorted, so this line connects 23:00 to 00:00
    # across adjacent calendar days when both samples are present.
    if include_phase_markers:
        for phase in ["baseline", "sham", "drug"]:
            mask = table["injection_phase"].to_numpy() == phase
            if not mask.any():
                continue
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                c=table.loc[mask, "lda_plot_label"],
                cmap="twilight_shifted",
                vmin=0,
                vmax=int(table["lda_plot_label_max"].iloc[0]),
                marker=PHASE_MARKERS[phase],
                edgecolors=PHASE_COLORS[phase],
                linewidths=0.9,
                s=46,
                label=phase,
                zorder=2,
            )
        ax.legend(title="phase", loc="best", frameon=False)
    else:
        sc = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c=table["lda_plot_label"],
            cmap="twilight_shifted",
            vmin=0,
            vmax=int(table["lda_plot_label_max"].iloc[0]),
            s=42,
            edgecolors="black",
            linewidths=0.25,
            zorder=2,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label(str(table["lda_plot_label_name"].iloc[0]))
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
) -> None:
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    z = projection[:, 2] if projection.shape[1] >= 3 else np.zeros(projection.shape[0])
    ax.plot(
        projection[:, 0],
        projection[:, 1],
        z,
        color="#cbd5e1",
        linewidth=0.8,
        alpha=0.8,
        zorder=1,
    )
    if include_phase_markers:
        for phase in ["baseline", "sham", "drug"]:
            mask = table["injection_phase"].to_numpy() == phase
            if not mask.any():
                continue
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                z[mask],
                c=table.loc[mask, "lda_plot_label"],
                cmap="twilight_shifted",
                vmin=0,
                vmax=int(table["lda_plot_label_max"].iloc[0]),
                marker=PHASE_MARKERS[phase],
                edgecolors=PHASE_COLORS[phase],
                linewidths=0.8,
                s=38,
                label=phase,
            )
        ax.legend(title="phase", loc="best", frameon=False)
    else:
        sc = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            z,
            c=table["lda_plot_label"],
            cmap="twilight_shifted",
            vmin=0,
            vmax=int(table["lda_plot_label_max"].iloc[0]),
            s=34,
            edgecolors="black",
            linewidths=0.2,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.8)
        cbar.set_label(str(table["lda_plot_label_name"].iloc[0]))
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
    label_name: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="viridis", values_format=".2f", colorbar=True)
    ax.set_title(title)
    ax.set_xlabel(f"predicted {label_name}")
    ax.set_ylabel(f"true {label_name}")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def cycle25_elapsed_hour_labels(times: pd.Series) -> tuple[np.ndarray, pd.Timestamp]:
    anchor = pd.to_datetime(times).min()
    elapsed_hours = ((pd.to_datetime(times) - anchor).dt.total_seconds() / 3600.0).to_numpy()
    labels = np.floor(elapsed_hours + 1e-9).astype(int) % 25
    return labels, anchor


def save_shuffle_label_table(
    table: pd.DataFrame,
    original_hour: np.ndarray,
    shuffled_hour: np.ndarray,
    shuffle_indices: np.ndarray,
    out_path: Path,
    random_seed: int,
) -> None:
    label_table = pd.DataFrame(
        {
            "chronological_row": np.arange(len(table), dtype=int),
            "calendar_day": table["calendar_day"].astype(str).to_numpy(),
            "hour_start_datetime": table["hour_start_datetime"].astype(str).to_numpy(),
            "injection_phase": table["injection_phase"].astype(str).to_numpy(),
            "original_clock_hour_of_day": original_hour.astype(int),
            "shuffled_clock_hour_of_day": shuffled_hour.astype(int),
            "source_chronological_row_for_shuffled_label": shuffle_indices.astype(int),
            "source_calendar_day_for_shuffled_label": table.loc[shuffle_indices, "calendar_day"].astype(str).to_numpy(),
            "source_hour_start_datetime_for_shuffled_label": table.loc[
                shuffle_indices, "hour_start_datetime"
            ].astype(str).to_numpy(),
            "source_clock_hour_of_day_for_shuffled_label": original_hour[shuffle_indices].astype(int),
            "random_seed": int(random_seed),
            "shuffle_algorithm": "numpy.random.default_rng(seed).permutation(n_rows); shuffled_label=original_label[permutation_index]",
        }
    )
    label_table.to_csv(out_path, index=False)


def save_projection_table(
    table: pd.DataFrame,
    projection: np.ndarray,
    out_path: Path,
    *,
    y_fit: np.ndarray,
    original_label: np.ndarray,
    label_name: str,
) -> None:
    projection_table = table.copy()
    projection_table["original_lda_label"] = original_label
    projection_table["lda_fit_label"] = y_fit
    projection_table["lda_fit_label_name"] = label_name
    for dimension_index in range(3):
        column_name = f"LD{dimension_index + 1}"
        if dimension_index < projection.shape[1]:
            projection_table[column_name] = projection[:, dimension_index]
        else:
            projection_table[column_name] = np.nan
    projection_table.to_csv(out_path, index=False)


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.matrix_csv)
    table["hour_start_datetime"] = pd.to_datetime(table["hour_start_datetime"])
    table = table.sort_values("hour_start_datetime").reset_index(drop=True)
    schedule = load_schedule(args.schedule_json)
    table["injection_phase"] = assign_phase(table["hour_start_datetime"], schedule)
    clock_hour = table["clock_hour_of_day"].astype(int).to_numpy()
    cycle25_labels = None
    cycle25_anchor = None
    if args.label_mode == "cycle25_elapsed_hour":
        cycle25_labels, cycle25_anchor = cycle25_elapsed_hour_labels(table["hour_start_datetime"])
        original_label = cycle25_labels
        labels = list(range(25))
        label_name = "25-hour cycle elapsed-hour label"
        filename_label = "cycle25"
        table["cycle25_elapsed_hour_label"] = cycle25_labels
        table["lda_plot_label"] = cycle25_labels
        table["lda_plot_label_max"] = 24
        table["lda_plot_label_name"] = "25-hour cycle label"
    else:
        original_label = clock_hour
        labels = list(range(24))
        label_name = "clock hour"
        filename_label = "clockhour"
        table["lda_plot_label"] = clock_hour
        table["lda_plot_label_max"] = 23
        table["lda_plot_label_name"] = "clock hour"
    y = original_label.copy()
    shuffle_indices = None
    shuffle_label_table_path = None
    if args.shuffle_hour_labels:
        if args.label_mode != "clock_hour":
            raise ValueError("--shuffle-hour-labels is currently only supported with --label-mode clock_hour")
        rng = np.random.default_rng(int(args.random_seed))
        shuffle_indices = rng.permutation(len(original_label))
        y = original_label[shuffle_indices]
        table["lda_fit_clock_hour"] = y
        table["lda_plot_label"] = y
        table["lda_plot_label_name"] = "shuffled clock hour used for LDA"
        shuffle_label_table_path = out_dir / "organized_hourly_label_shuffle_assignments.csv"
        save_shuffle_label_table(
            table,
            original_label,
            y,
            shuffle_indices,
            shuffle_label_table_path,
            int(args.random_seed),
        )
    else:
        table["lda_fit_label"] = original_label
    all_columns = table.columns.tolist()
    results = {
        "input_matrix": str(args.matrix_csv.resolve()),
        "schedule_json": str(args.schedule_json.resolve()) if args.schedule_json and args.schedule_json.exists() else None,
        "output_dir": str(out_dir),
        "n_samples": int(len(table)),
        "n_labels": int(pd.Series(y).nunique()),
        "label_mode": (
            "shuffled_clock_hour"
            if args.shuffle_hour_labels
            else args.label_mode
        ),
        "label_name": label_name,
        "label_values": labels,
        "cycle25_anchor_datetime": str(cycle25_anchor) if cycle25_anchor is not None else None,
        "cycle25_label_definition": (
            "floor((hour_start_datetime - cycle25_anchor_datetime) / 1 hour) modulo 25"
            if args.label_mode == "cycle25_elapsed_hour"
            else None
        ),
        "label_random_seed": int(args.random_seed) if args.shuffle_hour_labels else None,
        "shuffle_algorithm": (
            "numpy.random.default_rng(seed).permutation(n_rows); shuffled_label=original_label[permutation_index]"
            if args.shuffle_hour_labels
            else None
        ),
        "shuffle_label_table": str(shuffle_label_table_path.resolve()) if shuffle_label_table_path else None,
        "same_label_after_shuffle_count": (
            int(np.sum(original_label == y)) if args.shuffle_hour_labels else None
        ),
        "same_label_after_shuffle_fraction": (
            float(np.mean(original_label == y)) if args.shuffle_hour_labels else None
        ),
        "phase_counts": table["injection_phase"].value_counts().to_dict(),
        "conditions": {},
    }
    log(f"loaded {len(table)} hourly samples from {args.matrix_csv}")
    log(f"label mode: {results['label_mode']}")
    if cycle25_anchor is not None:
        log(f"25-hour cycle anchor: {cycle25_anchor}")
    if shuffle_label_table_path:
        log(f"wrote shuffled label assignments: {shuffle_label_table_path}")
    log(f"injection phase counts: {results['phase_counts']}")

    for condition, metrics in FEATURE_CONDITIONS.items():
        condition_start = time.perf_counter()
        feature_columns = feature_columns_for_condition(all_columns, metrics)
        if not feature_columns:
            log(f"skipping {condition}: no matching columns")
            continue
        log(f"fitting {condition}: {len(feature_columns)} features")
        x = table[feature_columns].to_numpy(dtype=float)
        projection, lda, _pipeline = fit_projection(x, y)
        y_pred = cross_validated_predictions(
            x,
            y,
            n_splits=int(args.cv_splits),
            random_seed=int(args.random_seed),
        )
        prefix_suffix = "_label_shuffle" if args.shuffle_hour_labels else ("_cycle25" if args.label_mode == "cycle25_elapsed_hour" else "")
        prefix = f"organized_hourly_{condition}{prefix_suffix}"
        paths = {
            "plot_2d": out_dir / f"{prefix}_lda2d_{filename_label}.png",
            "plot_3d": out_dir / f"{prefix}_lda3d_{filename_label}.png",
            "plot_2d_sham_drug": out_dir / f"{prefix}_lda2d_{filename_label}_sham_drug.png",
            "plot_3d_sham_drug": out_dir / f"{prefix}_lda3d_{filename_label}_sham_drug.png",
            "confusion_matrix": out_dir / f"{prefix}_confusion_matrix_{filename_label}_cv.png",
            "projection_csv": out_dir / f"{prefix}_projection_{filename_label}.csv",
        }
        title_base = f"{condition} LDA, {'shuffled ' if args.shuffle_hour_labels else ''}{label_name} labels"
        save_projection_table(
            table,
            projection,
            paths["projection_csv"],
            y_fit=y,
            original_label=original_label,
            label_name=label_name,
        )
        plot_2d(projection, table, paths["plot_2d"], title=title_base, include_phase_markers=False)
        plot_3d(projection, table, paths["plot_3d"], title=title_base, include_phase_markers=False)
        plot_2d(
            projection,
            table,
            paths["plot_2d_sham_drug"],
            title=f"{title_base}, sham/drug markers",
            include_phase_markers=True,
        )
        plot_3d(
            projection,
            table,
            paths["plot_3d_sham_drug"],
            title=f"{title_base}, sham/drug markers",
            include_phase_markers=True,
        )
        plot_confusion(
            y,
            y_pred,
            labels,
            paths["confusion_matrix"],
            title=f"{condition} cross-validated confusion matrix",
            label_name=label_name,
        )
        results["conditions"][condition] = {
            "metrics": list(metrics),
            "n_features": int(len(feature_columns)),
            "n_lda_dimensions": int(projection.shape[1]),
            "explained_variance_ratio": [float(v) for v in getattr(lda, "explained_variance_ratio_", [])],
            "cv_accuracy": float(accuracy_score(y, y_pred)),
            "cv_balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "elapsed_seconds": float(time.perf_counter() - condition_start),
            "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        }
    results["elapsed_seconds"] = float(time.perf_counter() - start)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--schedule-json", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--label-mode",
        choices=["clock_hour", "cycle25_elapsed_hour"],
        default="clock_hour",
        help="Use normal 24-hour clock labels or elapsed-hour labels modulo a 25-hour cycle.",
    )
    parser.add_argument(
        "--shuffle-hour-labels",
        action="store_true",
        help="Randomly permute clock-hour labels before fitting and cross-validation.",
    )
    args = parser.parse_args()
    if args.shuffle_hour_labels and args.out_dir == DEFAULT_OUT_DIR:
        args.out_dir = DEFAULT_SHUFFLE_OUT_DIR
    if args.label_mode == "cycle25_elapsed_hour" and args.out_dir == DEFAULT_OUT_DIR:
        args.out_dir = DEFAULT_CYCLE25_OUT_DIR
    return args


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "organized_hourly_lda_run_log.txt"
    summary_path = out_dir / "organized_hourly_lda_summary.json"
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            summary = run(args)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log(f"wrote summary: {summary_path}")
            log(f"finished in {summary['elapsed_seconds']:.2f} seconds")
    print(f"Wrote LDA outputs to: {out_dir}")
    print(f"Summary: {summary_path}")
    print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()
