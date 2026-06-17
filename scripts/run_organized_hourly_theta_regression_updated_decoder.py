from __future__ import annotations

"""Updated theta decoder with perturbation target comparisons and robust plots."""

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
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
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs" / "theta_regression" / "organized_hourly_theta_regression_updated_decoder"

FEATURE_CONDITIONS = {
    "fr_only": ("firing_rate_hz",),
    "fr_amp": ("firing_rate_hz", "amplitude_ptp_uv"),
    "fr_cv2": ("firing_rate_hz", "cv2"),
    "fr_peak_to_trough": ("firing_rate_hz", "peak_to_trough_ms"),
    "multi_feature": ("firing_rate_hz", "amplitude_ptp_uv", "cv2", "peak_to_trough_ms"),
}
CV_MODES = ("sample_kfold", "day_groupkfold")
BASE_TARGET_SPECS = (
    ("phase_only", ("cos", "sin"), None),
    ("phase_constant", ("cos", "sin", "constant"), None),
)
PERTURB_TARGET_SPECS = (
    ("phase_perturb", ("cos", "sin", "sham", "drug")),
    ("phase_constant_perturb", ("cos", "sin", "constant", "sham", "drug")),
)


def log(message: str) -> None:
    print(f"[run_organized_hourly_theta_regression_updated_decoder] {message}", flush=True)


def load_schedule(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def feature_columns_for_condition(columns: list[str], metrics: tuple[str, ...]) -> list[str]:
    return [
        column
        for column in columns
        if "__" in column and column.rsplit("__", 1)[-1] in metrics
    ]


def make_regressor(alphas: np.ndarray):
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        RidgeCV(alphas=alphas, fit_intercept=True),
    )


def circular_difference_rad(pred_theta: np.ndarray, true_theta: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(pred_theta - true_theta), np.cos(pred_theta - true_theta))


def theta_to_hour(theta: np.ndarray) -> np.ndarray:
    return (np.mod(theta, 2.0 * np.pi) / (2.0 * np.pi)) * 24.0


def assign_perturbations(
    times: pd.Series,
    schedule: dict | None,
    effect_hours: float,
) -> tuple[pd.DataFrame, list[dict]]:
    parsed = pd.to_datetime(times)
    sham = pd.Series(0, index=times.index, dtype=int)
    drug = pd.Series(0, index=times.index, dtype=int)
    intervals_used: list[dict] = []
    if not schedule:
        return pd.DataFrame({"perturbation_sham": sham, "perturbation_drug": drug}), intervals_used

    for phase, out in [("sham", sham), ("drug", drug)]:
        for interval in schedule.get(f"{phase}_intervals", []):
            start = pd.Timestamp(interval["start"])
            if effect_hours == 0:
                end = pd.Timestamp(interval["end"])
                end_rule = "schedule_interval"
            else:
                end = start + pd.Timedelta(hours=float(effect_hours))
                end_rule = f"onset_plus_{effect_hours:g}h"
            out.loc[(parsed >= start) & (parsed < end)] = 1
            intervals_used.append(
                {
                    "phase": phase,
                    "source_session_name": interval.get("session_name"),
                    "start": str(start),
                    "end": str(end),
                    "end_rule": end_rule,
                }
            )

    # Keep perturbation columns mutually exclusive. Drug has priority if windows overlap.
    sham.loc[drug == 1] = 0
    return pd.DataFrame({"perturbation_sham": sham, "perturbation_drug": drug}), intervals_used


def build_target(
    table: pd.DataFrame,
    target_columns: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    pieces = []
    names = []
    for column in target_columns:
        if column == "cos":
            pieces.append(table["clock_phase_cos"].to_numpy(dtype=float))
            names.append("clock_phase_cos")
        elif column == "sin":
            pieces.append(table["clock_phase_sin"].to_numpy(dtype=float))
            names.append("clock_phase_sin")
        elif column == "constant":
            pieces.append(np.ones(len(table), dtype=float))
            names.append("constant_1")
        elif column == "sham":
            pieces.append(table["perturbation_sham"].to_numpy(dtype=float))
            names.append("perturbation_sham")
        elif column == "drug":
            pieces.append(table["perturbation_drug"].to_numpy(dtype=float))
            names.append("perturbation_drug")
        else:
            raise ValueError(f"Unknown target column: {column}")
    return np.column_stack(pieces), names


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
    raise ValueError(f"Unsupported cv mode: {cv_mode}")


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if np.nanstd(y_true) == 0:
        return None
    return float(r2_score(y_true, y_pred))


def summarize_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    true_theta: np.ndarray,
) -> dict:
    cos_idx = target_names.index("clock_phase_cos")
    sin_idx = target_names.index("clock_phase_sin")
    pred_theta = np.arctan2(y_pred[:, sin_idx], y_pred[:, cos_idx])
    delta = circular_difference_rad(pred_theta, true_theta)
    abs_error_hours = np.abs(delta) * 24.0 / (2.0 * np.pi)
    pred_vector_length = np.sqrt(y_pred[:, cos_idx] ** 2 + y_pred[:, sin_idx] ** 2)
    summary = {
        "r2_cos": safe_r2(y_true[:, cos_idx], y_pred[:, cos_idx]),
        "r2_sin": safe_r2(y_true[:, sin_idx], y_pred[:, sin_idx]),
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
    if summary["r2_cos"] is not None and summary["r2_sin"] is not None:
        summary["r2_mean_cos_sin"] = float((summary["r2_cos"] + summary["r2_sin"]) / 2.0)
    else:
        summary["r2_mean_cos_sin"] = None

    for perturb_name in ["perturbation_sham", "perturbation_drug"]:
        if perturb_name not in target_names:
            continue
        idx = target_names.index(perturb_name)
        true_binary = y_true[:, idx].astype(int)
        pred_cont = y_pred[:, idx]
        pred_binary = (pred_cont >= 0.5).astype(int)
        summary[f"r2_{perturb_name}"] = safe_r2(true_binary, pred_cont)
        if len(np.unique(true_binary)) > 1:
            summary[f"balanced_accuracy_{perturb_name}"] = float(balanced_accuracy_score(true_binary, pred_binary))
            summary[f"f1_{perturb_name}"] = float(f1_score(true_binary, pred_binary, zero_division=0))
        else:
            summary[f"balanced_accuracy_{perturb_name}"] = None
            summary[f"f1_{perturb_name}"] = None
    return summary


def prediction_table(
    table: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    true_theta: np.ndarray,
) -> pd.DataFrame:
    cos_idx = target_names.index("clock_phase_cos")
    sin_idx = target_names.index("clock_phase_sin")
    pred_theta = np.arctan2(y_pred[:, sin_idx], y_pred[:, cos_idx])
    delta = circular_difference_rad(pred_theta, true_theta)
    out = pd.DataFrame(
        {
            "calendar_day": table["calendar_day"].astype(str),
            "hour_start_datetime": table["hour_start_datetime"].astype(str),
            "clock_hour_of_day": table["clock_hour_of_day"].astype(int),
            "perturbation_sham": table["perturbation_sham"].astype(int),
            "perturbation_drug": table["perturbation_drug"].astype(int),
            "true_theta_rad": true_theta,
            "true_cos": y_true[:, cos_idx],
            "true_sin": y_true[:, sin_idx],
            "pred_cos": y_pred[:, cos_idx],
            "pred_sin": y_pred[:, sin_idx],
            "pred_theta_rad": pred_theta,
            "true_hour_continuous": theta_to_hour(true_theta),
            "pred_hour_continuous": theta_to_hour(pred_theta),
            "signed_circular_error_rad": delta,
            "abs_circular_error_hours": np.abs(delta) * 24.0 / (2.0 * np.pi),
            "pred_vector_length": np.sqrt(y_pred[:, cos_idx] ** 2 + y_pred[:, sin_idx] ** 2),
        }
    )
    for name in ["constant_1", "perturbation_sham", "perturbation_drug"]:
        if name in target_names:
            idx = target_names.index(name)
            out[f"true_{name}"] = y_true[:, idx]
            out[f"pred_{name}"] = y_pred[:, idx]
    return out


def robust_limits(values_x: pd.Series, values_y: pd.Series, central_fraction: float) -> tuple[tuple[float, float], tuple[float, float]]:
    tail = (1.0 - central_fraction) / 2.0
    x_low, x_high = np.quantile(values_x, [tail, 1.0 - tail])
    y_low, y_high = np.quantile(values_y, [tail, 1.0 - tail])
    # Include the unit circle in the robust presentation view.
    x_low = min(float(x_low), -1.1)
    x_high = max(float(x_high), 1.1)
    y_low = min(float(y_low), -1.1)
    y_high = max(float(y_high), 1.1)
    return (x_low, x_high), (y_low, y_high)


def plot_unit_circle(predictions: pd.DataFrame, out_path: Path, *, title: str, robust_fraction: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    circle_theta = np.linspace(0, 2.0 * np.pi, 360)
    ax.plot(np.cos(circle_theta), np.sin(circle_theta), color="#94a3b8", linewidth=1.0)
    sc = ax.scatter(
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
    if robust_fraction is not None:
        xlim, ylim = robust_limits(predictions["pred_cos"], predictions["pred_sin"], robust_fraction)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    ax.axhline(0, color="#cbd5e1", linewidth=0.8)
    ax.axvline(0, color="#cbd5e1", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("cos(theta)")
    ax.set_ylabel("sin(theta)")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("clock hour")
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


def plot_polar_regression(predictions: pd.DataFrame, out_path: Path, *, title: str) -> None:
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    theta_true = predictions["true_theta_rad"].to_numpy()
    theta_pred = predictions["pred_theta_rad"].to_numpy()
    error_hours = predictions["abs_circular_error_hours"].to_numpy()
    sc = ax.scatter(theta_true, theta_to_hour(theta_pred), c=error_hours, cmap="viridis", s=34, alpha=0.85)
    theta_line = np.linspace(0, 2.0 * np.pi, 360)
    ax.plot(theta_line, theta_to_hour(theta_line), color="#111827", linewidth=1.0, label="identity")
    ax.set_ylim(0, 24)
    ax.set_title(title)
    ax.set_ylabel("predicted hour radius")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10), frameon=False)
    cbar = fig.colorbar(sc, ax=ax, pad=0.08)
    cbar.set_label("absolute circular error, hours")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_circular_heatmap(predictions: pd.DataFrame, out_path: Path, *, title: str) -> None:
    true_hour = predictions["clock_hour_of_day"].astype(int).to_numpy()
    pred_hour = np.floor(predictions["pred_hour_continuous"].to_numpy()).astype(int) % 24
    heat = np.zeros((24, 24), dtype=float)
    for true, pred in zip(true_hour, pred_hour):
        heat[true, pred] += 1
    row_sum = heat.sum(axis=1, keepdims=True)
    heat = np.divide(heat, row_sum, out=np.zeros_like(heat), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    im = ax.imshow(heat, cmap="viridis", origin="lower", vmin=0)
    ax.set_xticks(range(24))
    ax.set_yticks(range(24))
    ax.set_xlabel("predicted hour bin")
    ax.set_ylabel("true hour")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, pad=0.01, label="row-normalized fraction")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_error_by_hour(predictions: pd.DataFrame, out_path: Path, *, title: str) -> None:
    values = [
        predictions.loc[predictions["clock_hour_of_day"] == hour, "abs_circular_error_hours"].to_numpy()
        for hour in range(24)
    ]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.boxplot(values, positions=np.arange(24), showfliers=False)
    ax.set_xticks(range(24))
    ax.set_xlabel("true clock hour")
    ax.set_ylabel("absolute circular error, hours")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_chronological_by_day(predictions: pd.DataFrame, out_path: Path, *, title: str) -> None:
    days = sorted(predictions["calendar_day"].unique())
    n_days = len(days)
    fig, axes = plt.subplots(n_days, 1, figsize=(12, max(3, 1.7 * n_days)), sharey=True, constrained_layout=True)
    if n_days == 1:
        axes = [axes]
    for ax, day in zip(axes, days):
        day_table = predictions[predictions["calendar_day"] == day]
        times = pd.to_datetime(day_table["hour_start_datetime"])
        ax.plot(times, day_table["true_hour_continuous"], color="#111827", linewidth=1.0, label="true")
        ax.scatter(times, day_table["pred_hour_continuous"], c="#2563eb", s=15, alpha=0.8, label="predicted")
        ax.set_ylim(0, 24)
        ax.set_ylabel(str(day))
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("time")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_condition_summary(rows: list[dict], out_path: Path) -> None:
    summary = pd.DataFrame(rows)
    summary = summary[summary["target_design"] == "phase_only"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for cv_mode, group in summary.groupby("cv_mode", sort=False):
        axes[0].plot(group["condition"], group["mean_abs_circular_error_hours"], marker="o", label=cv_mode)
        axes[1].plot(group["condition"], group["within_2h"], marker="o", label=cv_mode)
        axes[2].plot(group["condition"], group["r2_mean_cos_sin"], marker="o", label=cv_mode)
    axes[0].set_title("Phase-only mean circular error")
    axes[0].set_ylabel("hours")
    axes[1].set_title("Phase-only within 2 hours")
    axes[1].set_ylabel("fraction")
    axes[2].set_title("Phase-only mean R2 cos/sin")
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
    table_base = pd.read_csv(args.matrix_csv)
    table_base["hour_start_datetime"] = pd.to_datetime(table_base["hour_start_datetime"])
    table_base = table_base.sort_values("hour_start_datetime").reset_index(drop=True)
    true_theta = table_base["clock_phase_rad"].astype(float).to_numpy()
    groups = table_base["calendar_day"].astype(str).to_numpy()
    schedule = load_schedule(args.schedule_json)
    alphas = np.logspace(args.alpha_min_exp, args.alpha_max_exp, args.n_alphas)
    all_columns = table_base.columns.tolist()
    summary_rows: list[dict] = []
    results = {
        "input_matrix": str(args.matrix_csv.resolve()),
        "schedule_json": str(args.schedule_json.resolve()) if args.schedule_json else None,
        "output_dir": str(out_dir),
        "n_samples": int(len(table_base)),
        "n_days": int(pd.Series(groups).nunique()),
        "ridge_alphas": [float(v) for v in alphas],
        "pipeline": [
            "SimpleImputer(strategy='mean')",
            "StandardScaler()",
            "RidgeCV(alphas=..., fit_intercept=True)",
        ],
        "robust_unit_circle_fraction": float(args.robust_unit_circle_fraction),
        "perturbation_effect_hours": [float(v) for v in args.perturbation_effect_hours],
        "conditions": {},
    }
    log(f"loaded {len(table_base)} hourly samples from {args.matrix_csv}")

    for condition, metrics in FEATURE_CONDITIONS.items():
        feature_columns = feature_columns_for_condition(all_columns, metrics)
        x = table_base[feature_columns].to_numpy(dtype=float)
        condition_result = {"metrics": list(metrics), "n_features": int(len(feature_columns)), "runs": {}}
        log(f"condition {condition}: {len(feature_columns)} features")

        run_specs = list(BASE_TARGET_SPECS)
        for effect_hours in args.perturbation_effect_hours:
            for name, columns in PERTURB_TARGET_SPECS:
                run_specs.append((name, columns, float(effect_hours)))

        for target_design, target_columns, effect_hours in run_specs:
            table = table_base.copy()
            intervals_used = []
            effect_label = "none"
            if effect_hours is not None:
                perturbations, intervals_used = assign_perturbations(table["hour_start_datetime"], schedule, effect_hours)
                table = pd.concat([table, perturbations], axis=1)
                effect_label = "schedule" if effect_hours == 0 else f"{effect_hours:g}h"
            else:
                table["perturbation_sham"] = 0
                table["perturbation_drug"] = 0
            y, target_names = build_target(table, target_columns)
            run_key = f"{target_design}_effect_{effect_label}"
            condition_result["runs"][run_key] = {
                "target_design": target_design,
                "target_names": target_names,
                "perturbation_effect_hours": effect_hours,
                "perturbation_counts": {
                    "sham": int(table["perturbation_sham"].sum()),
                    "drug": int(table["perturbation_drug"].sum()),
                },
                "intervals_used": intervals_used,
                "cv_modes": {},
            }

            final_model = make_regressor(alphas)
            final_model.fit(x, y)
            condition_result["runs"][run_key]["full_fit_alpha"] = float(final_model.named_steps["ridgecv"].alpha_)

            for cv_mode in CV_MODES:
                cv_started = time.perf_counter()
                y_pred, n_splits = cv_predictions(
                    x,
                    y,
                    cv_mode=cv_mode,
                    groups=groups,
                    random_seed=args.random_seed,
                    alphas=alphas,
                )
                stats = summarize_predictions(y, y_pred, target_names, true_theta)
                preds = prediction_table(table, y, y_pred, target_names, true_theta)
                prefix = f"organized_hourly_{condition}_{target_design}_effect_{effect_label}_{cv_mode}"
                paths = {
                    "predictions_csv": out_dir / f"{prefix}_predictions.csv",
                    "unit_circle_full": out_dir / f"{prefix}_unit_circle_full.png",
                    "unit_circle_robust99": out_dir / f"{prefix}_unit_circle_robust99.png",
                    "true_vs_pred_hour": out_dir / f"{prefix}_true_vs_pred_hour.png",
                    "polar_regression": out_dir / f"{prefix}_polar_regression.png",
                    "circular_heatmap": out_dir / f"{prefix}_circular_heatmap.png",
                    "error_by_hour": out_dir / f"{prefix}_error_by_hour.png",
                    "chronological_by_day": out_dir / f"{prefix}_chronological_by_day.png",
                }
                preds.to_csv(paths["predictions_csv"], index=False)
                title = f"{condition}, {target_design}, effect={effect_label}, {cv_mode}"
                plot_unit_circle(preds, paths["unit_circle_full"], title=f"{title}: full unit-circle QC")
                plot_unit_circle(
                    preds,
                    paths["unit_circle_robust99"],
                    title=f"{title}: central {args.robust_unit_circle_fraction:.0%} unit-circle view",
                    robust_fraction=args.robust_unit_circle_fraction,
                )
                plot_true_vs_pred_hour(preds, paths["true_vs_pred_hour"], title=f"{title}: true vs predicted phase")
                plot_polar_regression(preds, paths["polar_regression"], title=f"{title}: polar regression view")
                plot_circular_heatmap(preds, paths["circular_heatmap"], title=f"{title}: circular hour heatmap")
                plot_error_by_hour(preds, paths["error_by_hour"], title=f"{title}: error by true hour")
                plot_chronological_by_day(preds, paths["chronological_by_day"], title=f"{title}: by-day phase")

                mode_result = {
                    "n_splits": int(n_splits),
                    **stats,
                    "elapsed_seconds": float(time.perf_counter() - cv_started),
                    "outputs": {key: str(path.resolve()) for key, path in paths.items()},
                }
                condition_result["runs"][run_key]["cv_modes"][cv_mode] = mode_result
                summary_rows.append(
                    {
                        "condition": condition,
                        "target_design": target_design,
                        "effect_label": effect_label,
                        "perturbation_effect_hours": effect_hours,
                        "cv_mode": cv_mode,
                        "n_features": int(len(feature_columns)),
                        "n_target_columns": int(len(target_names)),
                        "sham_count": int(table["perturbation_sham"].sum()),
                        "drug_count": int(table["perturbation_drug"].sum()),
                        **stats,
                    }
                )
        results["conditions"][condition] = condition_result

    metrics_csv = out_dir / "organized_hourly_theta_regression_updated_decoder_metrics.csv"
    summary_plot = out_dir / "organized_hourly_theta_regression_updated_decoder_phase_only_summary.png"
    pd.DataFrame(summary_rows).to_csv(metrics_csv, index=False)
    plot_condition_summary(summary_rows, summary_plot)
    results["outputs"] = {
        "metrics_csv": str(metrics_csv.resolve()),
        "phase_only_summary_plot": str(summary_plot.resolve()),
    }
    results["elapsed_seconds"] = float(time.perf_counter() - started)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--schedule-json", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--perturbation-effect-hours", type=float, nargs="+", default=[0, 3, 6, 12, 24])
    parser.add_argument("--robust-unit-circle-fraction", type=float, default=0.99)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--alpha-min-exp", type=float, default=-3.0)
    parser.add_argument("--alpha-max-exp", type=float, default=6.0)
    parser.add_argument("--n-alphas", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "organized_hourly_theta_regression_updated_decoder_run_log.txt"
    summary_path = out_dir / "organized_hourly_theta_regression_updated_decoder_summary.json"
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            summary = run(args)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log(f"wrote summary: {summary_path}")
            log(f"finished in {summary['elapsed_seconds']:.2f} seconds")
    print(f"Wrote updated theta decoder outputs to: {out_dir}")
    print(f"Summary: {summary_path}")
    print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()
