from __future__ import annotations

"""Regress time-of-day phase as cos/sin theta from organized hourly features."""

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_MATRIX = PROJECT_DIR / "derived" / "threshold_hour_matrix_wide.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs" / "theta_regression" / "organized_hourly_theta_regression"

FEATURE_CONDITIONS = {
    "fr_only": ("firing_rate_hz",),
    "fr_amp": ("firing_rate_hz", "amplitude_ptp_uv"),
    "fr_cv2": ("firing_rate_hz", "cv2"),
    "fr_peak_to_trough": ("firing_rate_hz", "peak_to_trough_ms"),
    "multi_feature": ("firing_rate_hz", "amplitude_ptp_uv", "cv2", "peak_to_trough_ms"),
}


def log(message: str) -> None:
    print(f"[run_organized_hourly_theta_regression] {message}", flush=True)


def feature_columns_for_condition(columns: list[str], metrics: tuple[str, ...]) -> list[str]:
    selected = []
    for column in columns:
        if "__" not in column:
            continue
        metric = column.rsplit("__", 1)[-1]
        if metric in metrics:
            selected.append(column)
    return selected


def circular_difference_rad(pred_theta: np.ndarray, true_theta: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(pred_theta - true_theta), np.cos(pred_theta - true_theta))


def theta_to_hour(theta: np.ndarray) -> np.ndarray:
    return (np.mod(theta, 2.0 * np.pi) / (2.0 * np.pi)) * 24.0


def make_regressor(alphas: np.ndarray):
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        RidgeCV(alphas=alphas),
    )


def cv_predictions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cv_mode: str,
    groups: np.ndarray,
    random_seed: int,
    alphas: np.ndarray,
) -> tuple[np.ndarray, int]:
    if cv_mode == "sample_kfold":
        n_splits = min(5, len(y))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        return cross_val_predict(make_regressor(alphas), x, y, cv=cv), n_splits
    if cv_mode == "day_groupkfold":
        unique_groups = np.unique(groups)
        n_splits = min(5, len(unique_groups))
        cv = GroupKFold(n_splits=n_splits)
        return cross_val_predict(make_regressor(alphas), x, y, cv=cv, groups=groups), n_splits
    raise ValueError(f"Unsupported cv_mode: {cv_mode}")


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray, true_theta: np.ndarray) -> dict:
    pred_theta = np.arctan2(y_pred[:, 1], y_pred[:, 0])
    delta = circular_difference_rad(pred_theta, true_theta)
    abs_error_hours = np.abs(delta) * 24.0 / (2.0 * np.pi)
    pred_vector_length = np.sqrt(np.sum(y_pred**2, axis=1))
    return {
        "r2_cos": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "r2_sin": float(r2_score(y_true[:, 1], y_pred[:, 1])),
        "r2_mean_cos_sin": float((r2_score(y_true[:, 0], y_pred[:, 0]) + r2_score(y_true[:, 1], y_pred[:, 1])) / 2.0),
        "mean_abs_circular_error_hours": float(np.mean(abs_error_hours)),
        "median_abs_circular_error_hours": float(np.median(abs_error_hours)),
        "within_1h": float(np.mean(abs_error_hours <= 1.0)),
        "within_2h": float(np.mean(abs_error_hours <= 2.0)),
        "within_3h": float(np.mean(abs_error_hours <= 3.0)),
        "pred_vector_length_mean": float(np.mean(pred_vector_length)),
        "pred_vector_length_median": float(np.median(pred_vector_length)),
        "pred_vector_length_min": float(np.min(pred_vector_length)),
        "pred_vector_length_max": float(np.max(pred_vector_length)),
    }


def prediction_table(table: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, true_theta: np.ndarray) -> pd.DataFrame:
    pred_theta = np.arctan2(y_pred[:, 1], y_pred[:, 0])
    delta = circular_difference_rad(pred_theta, true_theta)
    return pd.DataFrame(
        {
            "calendar_day": table["calendar_day"].astype(str),
            "hour_start_datetime": table["hour_start_datetime"].astype(str),
            "clock_hour_of_day": table["clock_hour_of_day"].astype(int),
            "true_theta_rad": true_theta,
            "true_cos": y_true[:, 0],
            "true_sin": y_true[:, 1],
            "pred_cos": y_pred[:, 0],
            "pred_sin": y_pred[:, 1],
            "pred_theta_rad": pred_theta,
            "true_hour_continuous": theta_to_hour(true_theta),
            "pred_hour_continuous": theta_to_hour(pred_theta),
            "signed_circular_error_rad": delta,
            "abs_circular_error_hours": np.abs(delta) * 24.0 / (2.0 * np.pi),
            "pred_vector_length": np.sqrt(np.sum(y_pred**2, axis=1)),
        }
    )


def plot_unit_circle(predictions: pd.DataFrame, out_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    circle_theta = np.linspace(0, 2.0 * np.pi, 360)
    ax.plot(np.cos(circle_theta), np.sin(circle_theta), color="#94a3b8", linewidth=1.0)
    ax.scatter(
        predictions["pred_cos"],
        predictions["pred_sin"],
        c=predictions["clock_hour_of_day"],
        cmap="twilight_shifted",
        vmin=0,
        vmax=23,
        s=36,
        edgecolors="black",
        linewidths=0.2,
    )
    ax.scatter(predictions["true_cos"], predictions["true_sin"], s=8, color="#111827", alpha=0.25)
    ax.axhline(0, color="#cbd5e1", linewidth=0.8)
    ax.axvline(0, color="#cbd5e1", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("cos(theta)")
    ax.set_ylabel("sin(theta)")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_true_vs_pred_hour(predictions: pd.DataFrame, out_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    sc = ax.scatter(
        predictions["true_hour_continuous"],
        predictions["pred_hour_continuous"],
        c=predictions["abs_circular_error_hours"],
        cmap="viridis",
        s=38,
        edgecolors="black",
        linewidths=0.2,
    )
    ax.plot([0, 24], [0, 24], color="#111827", linewidth=1.0)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_yticks(range(0, 25, 3))
    ax.set_title(title)
    ax.set_xlabel("true hour, decoded from theta")
    ax.set_ylabel("predicted hour, decoded from atan2")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("absolute circular error, hours")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_error_histogram(predictions: pd.DataFrame, out_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(predictions["abs_circular_error_hours"], bins=np.linspace(0, 12, 25), color="#2563eb", edgecolor="white")
    ax.axvline(predictions["abs_circular_error_hours"].mean(), color="#b91c1c", linewidth=1.5, label="mean")
    ax.axvline(predictions["abs_circular_error_hours"].median(), color="#111827", linewidth=1.5, label="median")
    ax.set_title(title)
    ax.set_xlabel("absolute circular error, hours")
    ax.set_ylabel("sample count")
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_chronological(predictions: pd.DataFrame, out_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    times = pd.to_datetime(predictions["hour_start_datetime"])
    ax.plot(times, predictions["true_hour_continuous"], color="#111827", linewidth=1.0, label="true")
    ax.scatter(times, predictions["pred_hour_continuous"], c="#2563eb", s=16, alpha=0.75, label="predicted")
    ax.set_ylim(0, 24)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("hour decoded from phase")
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_condition_summary(summary_rows: list[dict], out_path: Path) -> None:
    summary = pd.DataFrame(summary_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for cv_mode, group in summary.groupby("cv_mode", sort=False):
        axes[0].plot(group["condition"], group["mean_abs_circular_error_hours"], marker="o", label=cv_mode)
        axes[1].plot(group["condition"], group["within_2h"], marker="o", label=cv_mode)
        axes[2].plot(group["condition"], group["r2_mean_cos_sin"], marker="o", label=cv_mode)
    axes[0].set_title("Mean circular error")
    axes[0].set_ylabel("hours")
    axes[1].set_title("Within 2 hours")
    axes[1].set_ylabel("fraction")
    axes[2].set_title("Mean R2 cos/sin")
    axes[2].set_ylabel("R2")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.matrix_csv)
    table["hour_start_datetime"] = pd.to_datetime(table["hour_start_datetime"])
    table = table.sort_values("hour_start_datetime").reset_index(drop=True)
    true_theta = table["clock_phase_rad"].astype(float).to_numpy()
    y = table[["clock_phase_cos", "clock_phase_sin"]].to_numpy(dtype=float)
    groups = table["calendar_day"].astype(str).to_numpy()
    alphas = np.logspace(args.alpha_min_exp, args.alpha_max_exp, args.n_alphas)
    all_columns = table.columns.tolist()
    summary_rows = []
    results = {
        "input_matrix": str(args.matrix_csv.resolve()),
        "output_dir": str(out_dir),
        "n_samples": int(len(table)),
        "n_days": int(pd.Series(groups).nunique()),
        "target_columns": ["clock_phase_cos", "clock_phase_sin"],
        "theta_definition": "clock_phase_rad = 2*pi*(clock_hour_of_day + 0.5)/24 for the hourly matrix",
        "target_math": {
            "theta": "2*pi*time_fraction_of_day",
            "y": "[cos(theta), sin(theta)]",
            "theta_pred": "atan2(pred_sin, pred_cos)",
            "circular_error": "atan2(sin(theta_pred-theta_true), cos(theta_pred-theta_true))",
            "error_hours": "abs(circular_error)*24/(2*pi)",
        },
        "python_modules": {
            "numpy": "circular math and alpha grid",
            "pandas": "CSV loading and prediction tables",
            "matplotlib": "PNG plots",
            "sklearn.impute.SimpleImputer": "mean imputation",
            "sklearn.preprocessing.StandardScaler": "feature standardization",
            "sklearn.linear_model.RidgeCV": "multi-output ridge regression with alpha selection",
            "sklearn.model_selection.KFold": "shuffled sample cross-validation",
            "sklearn.model_selection.GroupKFold": "held-out calendar-day cross-validation",
            "sklearn.model_selection.cross_val_predict": "out-of-fold predictions",
            "sklearn.metrics.r2_score": "cos/sin R2 metrics",
        },
        "ridge_alphas": [float(v) for v in alphas],
        "conditions": {},
    }
    log(f"loaded {len(table)} hourly samples from {args.matrix_csv}")

    for condition, metrics in FEATURE_CONDITIONS.items():
        feature_columns = feature_columns_for_condition(all_columns, metrics)
        x = table[feature_columns].to_numpy(dtype=float)
        condition_result = {
            "metrics": list(metrics),
            "n_features": int(len(feature_columns)),
            "cv_modes": {},
        }
        log(f"condition {condition}: {len(feature_columns)} features")

        final_model = make_regressor(alphas)
        final_model.fit(x, y)
        ridge = final_model.named_steps["ridgecv"]
        condition_result["full_fit_alpha"] = float(ridge.alpha_)

        for cv_mode in ["sample_kfold", "day_groupkfold"]:
            cv_started = time.perf_counter()
            y_pred, n_splits = cv_predictions(
                x,
                y,
                cv_mode=cv_mode,
                groups=groups,
                random_seed=args.random_seed,
                alphas=alphas,
            )
            stats = summarize_predictions(y, y_pred, true_theta)
            preds = prediction_table(table, y, y_pred, true_theta)
            prefix = f"organized_hourly_{condition}_theta_regression_{cv_mode}"
            paths = {
                "predictions_csv": out_dir / f"{prefix}_predictions.csv",
                "unit_circle": out_dir / f"{prefix}_unit_circle.png",
                "true_vs_pred_hour": out_dir / f"{prefix}_true_vs_pred_hour.png",
                "error_histogram": out_dir / f"{prefix}_error_histogram.png",
                "chronological": out_dir / f"{prefix}_chronological_phase.png",
            }
            preds.to_csv(paths["predictions_csv"], index=False)
            title_base = f"{condition} theta regression, {cv_mode}"
            plot_unit_circle(preds, paths["unit_circle"], title=f"{title_base}: predicted cos/sin")
            plot_true_vs_pred_hour(preds, paths["true_vs_pred_hour"], title=f"{title_base}: true vs predicted phase")
            plot_error_histogram(preds, paths["error_histogram"], title=f"{title_base}: circular error")
            plot_chronological(preds, paths["chronological"], title=f"{title_base}: chronological phase")

            mode_result = {
                "n_splits": int(n_splits),
                **stats,
                "elapsed_seconds": float(time.perf_counter() - cv_started),
                "outputs": {key: str(path.resolve()) for key, path in paths.items()},
            }
            condition_result["cv_modes"][cv_mode] = mode_result
            summary_rows.append(
                {
                    "condition": condition,
                    "cv_mode": cv_mode,
                    "n_features": int(len(feature_columns)),
                    **stats,
                }
            )
        results["conditions"][condition] = condition_result

    summary_csv = out_dir / "organized_hourly_theta_regression_metrics.csv"
    summary_plot = out_dir / "organized_hourly_theta_regression_condition_summary.png"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    plot_condition_summary(summary_rows, summary_plot)
    results["outputs"] = {
        "metrics_csv": str(summary_csv.resolve()),
        "condition_summary_plot": str(summary_plot.resolve()),
    }
    results["elapsed_seconds"] = float(time.perf_counter() - started)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--alpha-min-exp", type=float, default=-3.0)
    parser.add_argument("--alpha-max-exp", type=float, default=6.0)
    parser.add_argument("--n-alphas", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "organized_hourly_theta_regression_run_log.txt"
    summary_path = out_dir / "organized_hourly_theta_regression_summary.json"
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            summary = run(args)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log(f"wrote summary: {summary_path}")
            log(f"finished in {summary['elapsed_seconds']:.2f} seconds")
    print(f"Wrote theta regression outputs to: {out_dir}")
    print(f"Summary: {summary_path}")
    print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()
