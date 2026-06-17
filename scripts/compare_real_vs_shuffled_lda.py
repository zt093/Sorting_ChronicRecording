from __future__ import annotations

"""Compare real-clock-hour LDA decoding against the shuffled-label null."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from run_organized_hourly_lda import (
    DEFAULT_MATRIX,
    DEFAULT_OUT_DIR,
    DEFAULT_SHUFFLE_OUT_DIR,
    FEATURE_CONDITIONS,
    cross_validated_predictions,
    feature_columns_for_condition,
)


DEFAULT_COMPARE_OUT_DIR = DEFAULT_OUT_DIR.parent / "organized_hourly_feature_lda_real_vs_shuffle_comparison"
DEFAULT_SHUFFLE_ASSIGNMENTS = DEFAULT_SHUFFLE_OUT_DIR / "organized_hourly_label_shuffle_assignments.csv"


def cyclic_hour_distance(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    raw = np.abs(y_true.astype(int) - y_pred.astype(int))
    return np.minimum(raw, 24 - raw)


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict:
    distances = cyclic_hour_distance(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return {
        "cv_accuracy": float(accuracy_score(y_true, y_pred)),
        "cv_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "within_1_cyclic_hour": float(np.mean(distances <= 1)),
        "within_2_cyclic_hours": float(np.mean(distances <= 2)),
        "mean_cyclic_hour_error": float(np.mean(distances)),
        "median_cyclic_hour_error": float(np.median(distances)),
        "normalized_confusion_diagonal_mean": float(np.mean(np.diag(cm))),
        "normalized_confusion_within_1_band_mean": float(
            np.mean([cm[i, [(i - 1) % 24, i, (i + 1) % 24]].sum() for i in range(24)])
        ),
        "normalized_confusion_within_2_band_mean": float(
            np.mean([cm[i, [((i + offset) % 24) for offset in range(-2, 3)]].sum() for i in range(24)])
        ),
    }


def load_labels(matrix_csv: Path, shuffle_assignments_csv: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    table = pd.read_csv(matrix_csv)
    table["hour_start_datetime"] = pd.to_datetime(table["hour_start_datetime"])
    table = table.sort_values("hour_start_datetime").reset_index(drop=True)
    original_hour = table["clock_hour_of_day"].astype(int).to_numpy()

    assignments = pd.read_csv(shuffle_assignments_csv)
    assignments = assignments.sort_values("chronological_row").reset_index(drop=True)
    shuffled_hour = assignments["shuffled_clock_hour_of_day"].astype(int).to_numpy()
    source_rows = assignments["source_chronological_row_for_shuffled_label"].astype(int).to_numpy()

    checks = {
        "n_samples": int(len(table)),
        "assignment_rows": int(len(assignments)),
        "row_count_matches": bool(len(table) == len(assignments)),
        "chronological_rows_are_0_to_n_minus_1": bool(
            np.array_equal(assignments["chronological_row"].to_numpy(), np.arange(len(assignments)))
        ),
        "source_rows_form_permutation": bool(np.array_equal(np.sort(source_rows), np.arange(len(assignments)))),
        "shuffled_labels_match_source_rows": bool(np.array_equal(shuffled_hour, original_hour[source_rows])),
        "original_label_counts": pd.Series(original_hour).value_counts().sort_index().astype(int).to_dict(),
        "shuffled_label_counts": pd.Series(shuffled_hour).value_counts().sort_index().astype(int).to_dict(),
        "label_counts_preserved": bool(
            pd.Series(original_hour).value_counts().sort_index().equals(
                pd.Series(shuffled_hour).value_counts().sort_index()
            )
        ),
        "same_clock_hour_after_shuffle_count": int(np.sum(original_hour == shuffled_hour)),
        "same_clock_hour_after_shuffle_fraction": float(np.mean(original_hour == shuffled_hour)),
    }
    return table, original_hour, shuffled_hour, checks


def plot_comparison(summary_table: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    x = np.arange(len(summary_table))
    width = 0.38
    labels = summary_table["condition"].tolist()

    axes[0].bar(x - width / 2, summary_table["real_cv_accuracy"], width, label="real")
    axes[0].bar(x + width / 2, summary_table["shuffled_cv_accuracy"], width, label="shuffled")
    axes[0].axhline(1 / 24, color="#6b7280", linewidth=1, linestyle="--", label="1/24")
    axes[0].set_title("Exact CV accuracy")
    axes[0].set_ylabel("fraction")

    axes[1].bar(x - width / 2, summary_table["real_within_1_cyclic_hour"], width, label="real")
    axes[1].bar(x + width / 2, summary_table["shuffled_within_1_cyclic_hour"], width, label="shuffled")
    axes[1].set_title("Within +/-1 cyclic hour")

    axes[2].bar(x - width / 2, summary_table["real_mean_cyclic_hour_error"], width, label="real")
    axes[2].bar(x + width / 2, summary_table["shuffled_mean_cyclic_hour_error"], width, label="shuffled")
    axes[2].set_title("Mean cyclic hour error")
    axes[2].set_ylabel("hours")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    table, original_hour, shuffled_hour, sanity_checks = load_labels(args.matrix_csv, args.shuffle_assignments_csv)
    labels = list(range(24))
    all_columns = table.columns.tolist()
    rows = []
    condition_results = {}

    for condition, metrics in FEATURE_CONDITIONS.items():
        feature_columns = feature_columns_for_condition(all_columns, metrics)
        x = table[feature_columns].to_numpy(dtype=float)
        real_pred = cross_validated_predictions(x, original_hour, args.cv_splits, args.random_seed)
        shuffled_pred = cross_validated_predictions(x, shuffled_hour, args.cv_splits, args.random_seed)
        real_stats = summarize_predictions(original_hour, real_pred, labels)
        shuffled_stats = summarize_predictions(shuffled_hour, shuffled_pred, labels)
        row = {
            "condition": condition,
            "n_features": int(len(feature_columns)),
            **{f"real_{key}": value for key, value in real_stats.items()},
            **{f"shuffled_{key}": value for key, value in shuffled_stats.items()},
        }
        row["delta_cv_accuracy"] = row["real_cv_accuracy"] - row["shuffled_cv_accuracy"]
        row["delta_within_1_cyclic_hour"] = row["real_within_1_cyclic_hour"] - row["shuffled_within_1_cyclic_hour"]
        row["delta_within_2_cyclic_hours"] = row["real_within_2_cyclic_hours"] - row["shuffled_within_2_cyclic_hours"]
        row["delta_mean_cyclic_hour_error"] = row["real_mean_cyclic_hour_error"] - row[
            "shuffled_mean_cyclic_hour_error"
        ]
        rows.append(row)
        condition_results[condition] = row

    comparison_table = pd.DataFrame(rows)
    comparison_csv = out_dir / "organized_hourly_real_vs_shuffle_lda_comparison.csv"
    comparison_png = out_dir / "organized_hourly_real_vs_shuffle_lda_comparison.png"
    comparison_json = out_dir / "organized_hourly_real_vs_shuffle_lda_comparison_summary.json"
    comparison_table.to_csv(comparison_csv, index=False)
    plot_comparison(comparison_table, comparison_png)

    summary = {
        "input_matrix": str(args.matrix_csv.resolve()),
        "shuffle_assignments_csv": str(args.shuffle_assignments_csv.resolve()),
        "output_dir": str(out_dir),
        "random_seed": int(args.random_seed),
        "cv_splits": int(args.cv_splits),
        "sanity_checks": sanity_checks,
        "chance_exact_accuracy_for_24_classes": 1 / 24,
        "conditions": condition_results,
        "outputs": {
            "comparison_csv": str(comparison_csv.resolve()),
            "comparison_png": str(comparison_png.resolve()),
            "comparison_json": str(comparison_json.resolve()),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    comparison_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--shuffle-assignments-csv", type=Path, default=DEFAULT_SHUFFLE_ASSIGNMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_COMPARE_OUT_DIR)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(f"Wrote comparison outputs to: {summary['output_dir']}")
    print(f"Comparison CSV: {summary['outputs']['comparison_csv']}")
    print(f"Comparison plot: {summary['outputs']['comparison_png']}")
    print(f"Summary: {summary['outputs']['comparison_json']}")


if __name__ == "__main__":
    main()
