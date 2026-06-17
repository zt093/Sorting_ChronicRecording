from __future__ import annotations

"""Run PCA plots from the organized hourly feature matrix.

This script labels the 24 clock-hour clusters by annotating each hour's centroid
in 2D and 3D PCA space. PCA is unsupervised, so no confusion matrices are
written.
"""

import argparse
import json
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_MATRIX = PROJECT_DIR / "derived" / "threshold_hour_matrix_wide.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs" / "LDA_threshold" / "organized_hourly_feature_pca_clockhour"

FEATURE_CONDITIONS = {
    "fr_only": ("firing_rate_hz",),
    "fr_amp": ("firing_rate_hz", "amplitude_ptp_uv"),
    "fr_cv2": ("firing_rate_hz", "cv2"),
    "fr_peak_to_trough": ("firing_rate_hz", "peak_to_trough_ms"),
    "multi_feature": ("firing_rate_hz", "amplitude_ptp_uv", "cv2", "peak_to_trough_ms"),
}


def log(message: str) -> None:
    print(f"[run_organized_hourly_pca] {message}", flush=True)


def feature_columns_for_condition(columns: list[str], metrics: tuple[str, ...]) -> list[str]:
    selected = []
    for column in columns:
        if "__" not in column:
            continue
        metric = column.rsplit("__", 1)[-1]
        if metric in metrics:
            selected.append(column)
    return selected


def fit_pca_projection(x: np.ndarray) -> tuple[np.ndarray, PCA]:
    pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        PCA(n_components=3, random_state=0),
    )
    projection = pipeline.fit_transform(x)
    return projection, pipeline.named_steps["pca"]


def hour_centroids(projection: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    centroids = {}
    for hour in range(24):
        mask = labels == hour
        if mask.any():
            centroids[hour] = projection[mask].mean(axis=0)
    return centroids


def plot_pca_2d(projection: np.ndarray, table: pd.DataFrame, out_path: Path, *, title: str) -> None:
    labels = table["clock_hour_of_day"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.plot(projection[:, 0], projection[:, 1], color="#cbd5e1", linewidth=0.8, alpha=0.8, zorder=1)
    sc = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=labels,
        cmap="twilight_shifted",
        vmin=0,
        vmax=23,
        s=42,
        edgecolors="black",
        linewidths=0.25,
        zorder=2,
    )
    for hour, centroid in hour_centroids(projection, labels).items():
        ax.text(
            centroid[0],
            centroid[1],
            f"{hour:02d}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "black", "alpha": 0.78},
            zorder=3,
        )
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("clock hour")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pca_3d(projection: np.ndarray, table: pd.DataFrame, out_path: Path, *, title: str) -> None:
    labels = table["clock_hour_of_day"].astype(int).to_numpy()
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(projection[:, 0], projection[:, 1], projection[:, 2], color="#cbd5e1", linewidth=0.8, alpha=0.8)
    sc = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        projection[:, 2],
        c=labels,
        cmap="twilight_shifted",
        vmin=0,
        vmax=23,
        s=34,
        edgecolors="black",
        linewidths=0.2,
    )
    for hour, centroid in hour_centroids(projection, labels).items():
        ax.text(
            centroid[0],
            centroid[1],
            centroid[2],
            f"{hour:02d}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "black", "alpha": 0.72},
        )
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.8)
    cbar.set_label("clock hour")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.matrix_csv)
    table["hour_start_datetime"] = pd.to_datetime(table["hour_start_datetime"])
    table = table.sort_values("hour_start_datetime").reset_index(drop=True)
    labels = table["clock_hour_of_day"].astype(int).to_numpy()
    all_columns = table.columns.tolist()
    results = {
        "input_matrix": str(args.matrix_csv.resolve()),
        "output_dir": str(out_dir),
        "n_samples": int(len(table)),
        "n_clock_hour_labels": int(pd.Series(labels).nunique()),
        "label_definition": "clock_hour_of_day, 24 classes",
        "conditions": {},
    }
    log(f"loaded {len(table)} hourly samples from {args.matrix_csv}")

    for condition, metrics in FEATURE_CONDITIONS.items():
        condition_start = time.perf_counter()
        feature_columns = feature_columns_for_condition(all_columns, metrics)
        if not feature_columns:
            log(f"skipping {condition}: no matching columns")
            continue
        log(f"fitting PCA for {condition}: {len(feature_columns)} features")
        x = table[feature_columns].to_numpy(dtype=float)
        projection, pca = fit_pca_projection(x)
        prefix = f"organized_hourly_{condition}_pca_clockhour"
        paths = {
            "plot_2d": out_dir / f"{prefix}_2d_labeled_centroids.png",
            "plot_3d": out_dir / f"{prefix}_3d_labeled_centroids.png",
        }
        title = f"{condition} PCA, 24 clock-hour labels"
        plot_pca_2d(projection, table, paths["plot_2d"], title=title)
        plot_pca_3d(projection, table, paths["plot_3d"], title=title)
        try:
            silhouette = float(silhouette_score(projection, labels))
        except ValueError:
            silhouette = None
        results["conditions"][condition] = {
            "metrics": list(metrics),
            "n_features": int(len(feature_columns)),
            "n_pca_dimensions": int(projection.shape[1]),
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "explained_variance_ratio_sum_pc1_to_pc3": float(np.sum(pca.explained_variance_ratio_)),
            "clock_hour_silhouette_score_pc1_to_pc3": silhouette,
            "elapsed_seconds": float(time.perf_counter() - condition_start),
            "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        }
    results["elapsed_seconds"] = float(time.perf_counter() - start)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "organized_hourly_pca_run_log.txt"
    summary_path = out_dir / "organized_hourly_pca_summary.json"
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            summary = run(args)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log(f"wrote summary: {summary_path}")
            log(f"finished in {summary['elapsed_seconds']:.2f} seconds")
    print(f"Wrote PCA outputs to: {out_dir}")
    print(f"Summary: {summary_path}")
    print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()
