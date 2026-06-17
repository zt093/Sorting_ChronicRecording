from __future__ import annotations

"""Curate the full updated theta decoder output into a compact presentation folder."""

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_DIR = PROJECT_DIR / "outputs" / "theta_regression" / "organized_hourly_theta_regression_updated_decoder"
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs" / "theta_regression" / "organized_hourly_theta_regression_decoder_curated"
DEFAULT_SCHEDULE = (
    PROJECT_DIR
    / "outputs"
    / "combined_population_summary"
    / "threshold_sham_drug_marker_schedule.json"
)

CONDITIONS = ["fr_only", "fr_amp", "fr_cv2", "fr_peak_to_trough", "multi_feature"]
CV_MODES = ["sample_kfold", "day_groupkfold"]


def log(message: str) -> None:
    print(f"[curate_theta_regression_updated_decoder] {message}", flush=True)


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "core": out_dir / "core_results",
        "diag": out_dir / "diagnostic_figures",
        "csv": out_dir / "csv",
        "logs": out_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def copy_file(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source": str(src.resolve()), "destination": str(dst.resolve()), "bytes": int(dst.stat().st_size)}


def source_name(condition: str, target: str, effect: str, cv: str, suffix: str) -> str:
    return f"organized_hourly_{condition}_{target}_effect_{effect}_{cv}_{suffix}"


def circular_difference_hours(pred_hour: np.ndarray, true_hour: np.ndarray) -> np.ndarray:
    raw = np.abs(pred_hour - true_hour)
    return np.minimum(raw, 24.0 - raw)


def load_schedule(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def perturbation_onsets(schedule: dict | None) -> list[dict]:
    if not schedule:
        return []
    rows = []
    for phase in ["sham", "drug"]:
        for interval in schedule.get(f"{phase}_intervals", []):
            rows.append(
                {
                    "phase": phase,
                    "start": pd.Timestamp(interval["start"]),
                    "session_name": interval.get("session_name", ""),
                }
            )
    return sorted(rows, key=lambda row: row["start"])


def add_onset_window_flags(predictions: pd.DataFrame, schedule: dict | None, hours_after: float = 3.0) -> pd.DataFrame:
    table = predictions.copy()
    times = pd.to_datetime(table["hour_start_datetime"])
    table["within_3h_after_sham_onset"] = 0
    table["within_3h_after_drug_onset"] = 0
    table["first_sample_after_sham_onset"] = 0
    table["first_sample_after_drug_onset"] = 0
    for onset in perturbation_onsets(schedule):
        phase = onset["phase"]
        start = onset["start"]
        end = start + pd.Timedelta(hours=float(hours_after))
        mask = (times >= start) & (times < end)
        table.loc[mask, f"within_3h_after_{phase}_onset"] = 1
        after = np.flatnonzero((times >= start).to_numpy())
        if len(after):
            table.loc[after[0], f"first_sample_after_{phase}_onset"] = 1
    # Drug has priority for display when windows overlap.
    table.loc[table["within_3h_after_drug_onset"] == 1, "within_3h_after_sham_onset"] = 0
    return table


def plot_feature_condition_summary(metrics: pd.DataFrame, out_path: Path) -> None:
    rows = metrics[(metrics["target_design"] == "phase_only") & (metrics["effect_label"] == "none")].copy()
    rows["mean_abs_circular_error_hours"] = rows["mean_abs_circular_error_hours"].astype(float)
    rows["within_2h"] = rows["within_2h"].astype(float)
    rows["r2_mean_cos_sin"] = rows["r2_mean_cos_sin"].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for cv_mode, group in rows.groupby("cv_mode", sort=False):
        group = group.set_index("condition").loc[CONDITIONS].reset_index()
        axes[0].plot(group["condition"], group["mean_abs_circular_error_hours"], marker="o", label=cv_mode)
        axes[1].plot(group["condition"], group["within_2h"], marker="o", label=cv_mode)
        axes[2].plot(group["condition"], group["r2_mean_cos_sin"], marker="o", label=cv_mode)
    axes[0].set_title("Phase decoding error")
    axes[0].set_ylabel("mean circular error, hours")
    axes[1].set_title("Near-phase accuracy")
    axes[1].set_ylabel("fraction within 2 h")
    axes[2].set_title("Cos/sin regression")
    axes[2].set_ylabel("mean R2")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_perturbation_stats(metrics: pd.DataFrame, out_path: Path) -> None:
    rows = metrics[
        (metrics["condition"] == "multi_feature")
        & (metrics["target_design"] == "phase_perturb")
        & (metrics["effect_label"] == "schedule")
    ].copy()
    for column in [
        "mean_abs_circular_error_hours",
        "within_2h",
        "r2_mean_cos_sin",
        "r2_perturbation_sham",
        "r2_perturbation_drug",
        "balanced_accuracy_perturbation_sham",
        "balanced_accuracy_perturbation_drug",
    ]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    x = np.arange(len(rows))
    labels = rows["cv_mode"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    width = 0.35
    axes[0].bar(x - width / 2, rows["balanced_accuracy_perturbation_sham"], width, label="sham")
    axes[0].bar(x + width / 2, rows["balanced_accuracy_perturbation_drug"], width, label="drug")
    axes[0].axhline(0.5, color="#6b7280", linestyle="--", linewidth=1)
    axes[0].set_title("Schedule perturbation decoding")
    axes[0].set_ylabel("balanced accuracy")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, rows["r2_perturbation_sham"], width, label="sham")
    axes[1].bar(x + width / 2, rows["r2_perturbation_drug"], width, label="drug")
    axes[1].axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    axes[1].set_title("Perturbation target R2")
    axes[1].set_ylabel("R2")
    axes[1].set_xticks(x, labels)

    axes[2].bar(x - width / 2, rows["mean_abs_circular_error_hours"], width, label="mean error")
    axes[2].bar(x + width / 2, rows["within_2h"], width, label="within 2 h")
    axes[2].set_title("Phase decoding in same model")
    axes[2].set_xticks(x, labels)
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def perturbation_improvement_table(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "mean_abs_circular_error_hours",
        "median_abs_circular_error_hours",
        "within_1h",
        "within_2h",
        "within_3h",
        "r2_mean_cos_sin",
        "r2_perturbation_sham",
        "r2_perturbation_drug",
        "balanced_accuracy_perturbation_sham",
        "balanced_accuracy_perturbation_drug",
        "f1_perturbation_sham",
        "f1_perturbation_drug",
    ]
    table = metrics.copy()
    for column in numeric_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    rows = []
    effects = ["schedule", "3h", "6h", "12h", "24h"]
    for condition in CONDITIONS:
        for cv_mode in CV_MODES:
            baseline_rows = table[
                (table["condition"] == condition)
                & (table["target_design"] == "phase_only")
                & (table["effect_label"] == "none")
                & (table["cv_mode"] == cv_mode)
            ]
            if baseline_rows.empty:
                continue
            baseline = baseline_rows.iloc[0]
            for effect in effects:
                perturb_rows = table[
                    (table["condition"] == condition)
                    & (table["target_design"] == "phase_perturb")
                    & (table["effect_label"] == effect)
                    & (table["cv_mode"] == cv_mode)
                ]
                if perturb_rows.empty:
                    continue
                perturb = perturb_rows.iloc[0]
                rows.append(
                    {
                        "condition": condition,
                        "cv_mode": cv_mode,
                        "effect_label": effect,
                        "baseline_target_design": "phase_only",
                        "perturb_target_design": "phase_perturb",
                        "baseline_mean_error_hours": baseline["mean_abs_circular_error_hours"],
                        "perturb_mean_error_hours": perturb["mean_abs_circular_error_hours"],
                        "delta_mean_error_hours": baseline["mean_abs_circular_error_hours"]
                        - perturb["mean_abs_circular_error_hours"],
                        "baseline_median_error_hours": baseline["median_abs_circular_error_hours"],
                        "perturb_median_error_hours": perturb["median_abs_circular_error_hours"],
                        "delta_median_error_hours": baseline["median_abs_circular_error_hours"]
                        - perturb["median_abs_circular_error_hours"],
                        "baseline_within_1h": baseline["within_1h"],
                        "perturb_within_1h": perturb["within_1h"],
                        "delta_within_1h": perturb["within_1h"] - baseline["within_1h"],
                        "baseline_within_2h": baseline["within_2h"],
                        "perturb_within_2h": perturb["within_2h"],
                        "delta_within_2h": perturb["within_2h"] - baseline["within_2h"],
                        "baseline_within_3h": baseline["within_3h"],
                        "perturb_within_3h": perturb["within_3h"],
                        "delta_within_3h": perturb["within_3h"] - baseline["within_3h"],
                        "baseline_r2_mean_cos_sin": baseline["r2_mean_cos_sin"],
                        "perturb_r2_mean_cos_sin": perturb["r2_mean_cos_sin"],
                        "delta_r2_mean_cos_sin": perturb["r2_mean_cos_sin"] - baseline["r2_mean_cos_sin"],
                        "r2_perturbation_sham": perturb["r2_perturbation_sham"],
                        "r2_perturbation_drug": perturb["r2_perturbation_drug"],
                        "balanced_accuracy_perturbation_sham": perturb["balanced_accuracy_perturbation_sham"],
                        "balanced_accuracy_perturbation_drug": perturb["balanced_accuracy_perturbation_drug"],
                        "f1_perturbation_sham": perturb["f1_perturbation_sham"],
                        "f1_perturbation_drug": perturb["f1_perturbation_drug"],
                    }
                )
    return pd.DataFrame(rows)


def plot_perturbation_improvement(stats: pd.DataFrame, out_path: Path) -> None:
    rows = stats[stats["condition"] == "multi_feature"].copy()
    effect_order = ["schedule", "3h", "6h", "12h", "24h"]
    rows["effect_label"] = pd.Categorical(rows["effect_label"], categories=effect_order, ordered=True)
    rows = rows.sort_values(["cv_mode", "effect_label"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for cv_mode, group in rows.groupby("cv_mode", sort=False):
        group = group.sort_values("effect_label")
        axes[0, 0].plot(group["effect_label"], group["delta_mean_error_hours"], marker="o", label=cv_mode)
        axes[0, 1].plot(group["effect_label"], group["delta_within_2h"], marker="o", label=cv_mode)
        axes[1, 0].plot(group["effect_label"], group["delta_r2_mean_cos_sin"], marker="o", label=cv_mode)
    axes[0, 0].axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Phase error change")
    axes[0, 0].set_ylabel("baseline error - perturb model error, h")
    axes[0, 1].axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Within-2h change")
    axes[0, 1].set_ylabel("perturb model - baseline")
    axes[1, 0].axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Cos/sin R2 change")
    axes[1, 0].set_ylabel("perturb model - baseline")

    width = 0.18
    x = np.arange(len(effect_order))
    sample = rows[rows["cv_mode"] == "sample_kfold"].set_index("effect_label").reindex(effect_order)
    day = rows[rows["cv_mode"] == "day_groupkfold"].set_index("effect_label").reindex(effect_order)
    axes[1, 1].bar(x - 1.5 * width, sample["balanced_accuracy_perturbation_sham"], width, label="sample sham")
    axes[1, 1].bar(x - 0.5 * width, sample["balanced_accuracy_perturbation_drug"], width, label="sample drug")
    axes[1, 1].bar(x + 0.5 * width, day["balanced_accuracy_perturbation_sham"], width, label="day sham")
    axes[1, 1].bar(x + 1.5 * width, day["balanced_accuracy_perturbation_drug"], width, label="day drug")
    axes[1, 1].axhline(0.5, color="#6b7280", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Perturbation-state decoding")
    axes[1, 1].set_ylabel("balanced accuracy")
    axes[1, 1].set_xticks(x, effect_order)
    axes[1, 1].legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def draw_onset_highlight_scatter(ax, table: pd.DataFrame, title: str, show_legend: bool = True):
    scatter = ax.scatter(
        table["true_hour_continuous"],
        table["pred_hour_continuous"],
        c=table["abs_circular_error_hours"],
        cmap="Greys",
        s=24,
        alpha=0.45,
        edgecolors="none",
        label="all samples",
    )
    sham = table["within_3h_after_sham_onset"].astype(bool)
    drug = table["within_3h_after_drug_onset"].astype(bool)
    first_sham = table["first_sample_after_sham_onset"].astype(bool)
    first_drug = table["first_sample_after_drug_onset"].astype(bool)
    ax.scatter(
        table.loc[sham, "true_hour_continuous"],
        table.loc[sham, "pred_hour_continuous"],
        marker="s",
        s=62,
        color="#f59e0b",
        edgecolors="black",
        linewidths=0.5,
        label="0-3 h after sham onset",
    )
    ax.scatter(
        table.loc[drug, "true_hour_continuous"],
        table.loc[drug, "pred_hour_continuous"],
        marker="^",
        s=70,
        color="#ef4444",
        edgecolors="black",
        linewidths=0.5,
        label="0-3 h after drug onset",
    )
    ax.scatter(
        table.loc[first_sham, "true_hour_continuous"],
        table.loc[first_sham, "pred_hour_continuous"],
        marker="*",
        s=210,
        color="#fbbf24",
        edgecolors="black",
        linewidths=0.7,
        label="first sample after sham onset",
    )
    ax.scatter(
        table.loc[first_drug, "true_hour_continuous"],
        table.loc[first_drug, "pred_hour_continuous"],
        marker="*",
        s=240,
        color="#dc2626",
        edgecolors="black",
        linewidths=0.7,
        label="first sample after drug onset",
    )
    ax.plot([0, 24], [0, 24], color="#2563eb", linewidth=1.1, label="perfect prediction")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 24)
    ax.set_xlabel("true clock hour")
    ax.set_ylabel("predicted clock hour")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    if show_legend:
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    return scatter


def plot_onset_highlight_scatter(predictions: pd.DataFrame, out_path: Path, title: str) -> None:
    table = predictions.copy()
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    scatter = draw_onset_highlight_scatter(ax, table, title, show_legend=True)
    fig.colorbar(scatter, ax=ax, label="absolute circular error, h")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_phase_vs_perturb_side_by_side(
    phase_only: pd.DataFrame,
    phase_perturb: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), sharex=True, sharey=True, constrained_layout=True)
    scatter = draw_onset_highlight_scatter(
        axes[0],
        phase_only,
        "(cos, sin) only",
        show_legend=False,
    )
    draw_onset_highlight_scatter(
        axes[1],
        phase_perturb,
        "(cos, sin, sham, drug)",
        show_legend=True,
    )
    phase_err = phase_only["abs_circular_error_hours"].mean()
    perturb_err = phase_perturb["abs_circular_error_hours"].mean()
    fig.suptitle(f"{title}\nmean error: phase-only {phase_err:.3f} h, phase+perturb {perturb_err:.3f} h")
    fig.colorbar(scatter, ax=axes, label="absolute circular error, h", shrink=0.88)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def add_perturbation_spans(ax, table: pd.DataFrame) -> None:
    times = pd.to_datetime(table["hour_start_datetime"])
    for column, color, label in [
        ("perturbation_sham", "#f59e0b", "sham"),
        ("perturbation_drug", "#ef4444", "drug"),
    ]:
        active = table[column].astype(int).to_numpy()
        started = None
        used_label = False
        for idx, value in enumerate(active):
            if value and started is None:
                started = times.iloc[idx]
            if started is not None and (not value or idx == len(active) - 1):
                end_idx = idx if not value else idx + 1
                if end_idx >= len(times):
                    end = times.iloc[-1] + pd.Timedelta(hours=1)
                else:
                    end = times.iloc[end_idx]
                ax.axvspan(started, end, color=color, alpha=0.18, label=label if not used_label else None)
                used_label = True
                started = None


def plot_perturbation_timeline(predictions: pd.DataFrame, out_path: Path) -> None:
    table = predictions.copy()
    table["time"] = pd.to_datetime(table["hour_start_datetime"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    add_perturbation_spans(axes[0], table)
    axes[0].plot(table["time"], table["true_hour_continuous"], color="#111827", linewidth=1.0, label="true phase")
    axes[0].scatter(table["time"], table["pred_hour_continuous"], color="#2563eb", s=16, alpha=0.75, label="predicted")
    axes[0].set_ylim(0, 24)
    axes[0].set_ylabel("hour")
    axes[0].set_title("Multi-feature schedule perturbation decoder: phase with perturbation windows")
    axes[0].legend(frameon=False, loc="upper right")

    add_perturbation_spans(axes[1], table)
    axes[1].plot(table["time"], table["pred_perturbation_sham"], color="#b45309", label="predicted sham")
    axes[1].plot(table["time"], table["pred_perturbation_drug"], color="#b91c1c", label="predicted drug")
    axes[1].axhline(0.5, color="#6b7280", linestyle="--", linewidth=1)
    axes[1].set_ylabel("score")
    axes[1].legend(frameon=False, loc="upper right")

    add_perturbation_spans(axes[2], table)
    axes[2].plot(table["time"], table["abs_circular_error_hours"], color="#4f46e5", linewidth=0.9)
    axes[2].set_ylabel("phase error h")
    axes[2].set_xlabel("time")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_3d_perturbation_axis(predictions: pd.DataFrame, out_path: Path) -> None:
    z = predictions["pred_perturbation_drug"] - predictions["pred_perturbation_sham"]
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), color="#94a3b8", linewidth=1.0)
    sc = ax.scatter(
        predictions["pred_cos"],
        predictions["pred_sin"],
        z,
        c=predictions["clock_hour_of_day"],
        cmap="twilight_shifted",
        vmin=0,
        vmax=23,
        s=34,
        edgecolors="black",
        linewidths=0.2,
    )
    if "within_3h_after_sham_onset" in predictions.columns:
        sham = predictions["within_3h_after_sham_onset"].astype(bool)
        drug = predictions["within_3h_after_drug_onset"].astype(bool)
        first_sham = predictions["first_sample_after_sham_onset"].astype(bool)
        first_drug = predictions["first_sample_after_drug_onset"].astype(bool)
        ax.scatter(
            predictions.loc[sham, "pred_cos"],
            predictions.loc[sham, "pred_sin"],
            z.loc[sham],
            marker="s",
            s=72,
            color="#f59e0b",
            edgecolors="black",
            linewidths=0.5,
            label="0-3 h after sham onset",
        )
        ax.scatter(
            predictions.loc[drug, "pred_cos"],
            predictions.loc[drug, "pred_sin"],
            z.loc[drug],
            marker="^",
            s=82,
            color="#ef4444",
            edgecolors="black",
            linewidths=0.5,
            label="0-3 h after drug onset",
        )
        ax.scatter(
            predictions.loc[first_sham, "pred_cos"],
            predictions.loc[first_sham, "pred_sin"],
            z.loc[first_sham],
            marker="*",
            s=230,
            color="#fbbf24",
            edgecolors="black",
            linewidths=0.7,
            label="first sample after sham onset",
        )
        ax.scatter(
            predictions.loc[first_drug, "pred_cos"],
            predictions.loc[first_drug, "pred_sin"],
            z.loc[first_drug],
            marker="*",
            s=260,
            color="#dc2626",
            edgecolors="black",
            linewidths=0.7,
            label="first sample after drug onset",
        )
    ax.set_title("Phase plane plus perturbation axis: z = pred_drug - pred_sham")
    ax.set_xlabel("predicted cos(theta)")
    ax.set_ylabel("predicted sin(theta)")
    ax.set_zlabel("predicted drug - sham")
    if "within_3h_after_sham_onset" in predictions.columns:
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.75, label="clock hour")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_readme(out_dir: Path, paths: dict[str, Path], copied: list[dict], summary: dict) -> None:
    readme = out_dir / "README.md"
    text = f"""# Curated Updated Theta Decoder Outputs

Source folder:

```text
{summary['source_dir']}
```

This folder is a compact presentation/QC subset of the full updated theta decoder run. The original full folder is left intact.

## Structure

```text
core_results/        key figures and top-level summary tables
diagnostic_figures/  selected QC figures for the main multi-feature model
csv/                 selected prediction CSVs and curated metrics
logs/                curation summary and run log
```

## Modeling Context

The curated outputs come from the updated decoder:

```text
features -> [cos(theta), sin(theta), perturbation_sham, perturbation_drug]
```

Perturbations are one-hot encoded:

```text
baseline = [0, 0]
sham     = [1, 0]
drug     = [0, 1]
```

One-hot sham/drug is preferred over a scalar `0/1/2` perturbation target because sham and drug are categorical states, not ordered magnitudes.

## Key Figures

- `core_results/Fig1_multi_feature_phase_decoding_sample_cv_true_vs_pred_hour.png`: main true-vs-predicted phase view for the best-performing `multi_feature` sample-CV model.
- `core_results/Fig1_multi_feature_phase_decoding_sample_cv_circular_heatmap.png`: circular binned true-vs-predicted hour heatmap for the same sample-CV model.
- `core_results/Fig2_multi_feature_phase_decoding_day_cv_true_vs_pred_hour.png`: held-out-day CV true-vs-predicted phase view.
- `core_results/Fig2_multi_feature_phase_decoding_day_cv_circular_heatmap.png`: held-out-day CV circular binned true-vs-predicted hour heatmap.
- `core_results/Fig3_feature_condition_summary.png`: basic phase-only decoder comparison across feature conditions.
- `core_results/Fig4_schedule_perturbation_decoding_stats.png`: schedule perturbation decoding accuracy/R2 and phase metrics.
- `core_results/Fig5_perturbation_timeline_marked.png`: chronological phase/predicted perturbation scores with sham/drug windows shaded.
- `core_results/Fig6_phase_plus_perturbation_3d_axis.png`: circular phase plane with z-axis `pred_drug - pred_sham`.
- `core_results/Fig7_perturbation_model_improvement_stats.png`: how much phase+perturbation improves over the matched phase-only decoder.
- `core_results/Fig8_multi_feature_schedule_sample_cv_onset_highlight_scatter.png`: sample-CV true-vs-predicted scatter with perturbation onsets and 0-3 h post-onset points highlighted.
- `core_results/Fig9_multi_feature_schedule_day_cv_onset_highlight_scatter.png`: held-out-day CV version of the onset-highlight scatter.
- `core_results/Fig10_multi_feature_sample_cv_phase_only_vs_perturb_scatter.png`: side-by-side sample-CV scatter comparing `(cos, sin)` only against `(cos, sin, sham, drug)`.
- `core_results/Fig11_multi_feature_day_cv_phase_only_vs_perturb_scatter.png`: side-by-side held-out-day CV version of the same comparison.

## Perturbation Improvement Statistics

- `core_results/perturbation_model_improvement_stats.csv` compares each `phase_perturb` model against the matched `phase_only` model for the same feature condition and CV mode.
- `delta_mean_error_hours = phase_only_error - phase_perturb_error`; positive values mean the perturbation target reduced phase decoding error.
- `delta_within_1h`, `delta_within_2h`, `delta_within_3h`, and `delta_r2_mean_cos_sin` are `phase_perturb - phase_only`; positive values mean the perturbation target improved the metric.
- Perturbation-state decoding is reported separately with sham/drug R2, balanced accuracy, and F1.

## Selected CSVs

- `core_results/decoder_summary_metrics.csv`: compact top-level table for phase-only feature comparisons plus selected multi-feature perturbation rows.
- `csv/all_curated_metrics.csv`: all 120 metrics rows from the full updated decoder run.
- `csv/organized_hourly_multi_feature_phase_only_effect_none_*_predictions.csv`: selected phase-only prediction tables.
- `csv/organized_hourly_multi_feature_phase_perturb_effect_schedule_*_predictions.csv`: selected schedule perturbation prediction tables.
- `csv/organized_hourly_multi_feature_phase_perturb_effect_schedule_*_predictions_with_onset3h_flags.csv`: selected schedule perturbation prediction tables with onset/highlight flags added.
- `csv/organized_hourly_multi_feature_phase_perturb_effect_24h_*_predictions.csv`: selected 24 h perturbation-window prediction tables.

## Notes

- The robust unit-circle view uses central 99% x/y limits while keeping the unit circle visible.
- Polar regression plots from the full run are intentionally omitted here.
- Full prediction CSVs are kept in `csv/`, separate from presentation figures.
- The full run generated {summary['source_png_count']} PNGs and {summary['source_csv_count']} CSVs; this curated folder keeps {summary['copied_file_count']} copied/generated core files.
"""
    readme.write_text(text, encoding="utf-8")


def curate(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    source = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    paths = ensure_dirs(out_dir)
    copied: list[dict] = []

    metrics_src = source / "organized_hourly_theta_regression_updated_decoder_metrics.csv"
    metrics = pd.read_csv(metrics_src)
    schedule = load_schedule(args.schedule_json)
    metrics_dst = paths["csv"] / "all_curated_metrics.csv"
    metrics.to_csv(metrics_dst, index=False)
    copied.append({"source": str(metrics_src.resolve()), "destination": str(metrics_dst.resolve()), "bytes": int(metrics_dst.stat().st_size)})

    top_metrics = metrics[
        (
            (metrics["target_design"] == "phase_only")
            & (metrics["effect_label"] == "none")
        )
        | (
            (metrics["condition"] == "multi_feature")
            & (metrics["target_design"] == "phase_perturb")
            & (metrics["effect_label"].isin(["schedule", "24h"]))
        )
    ].copy()
    top_metrics_dst = paths["core"] / "decoder_summary_metrics.csv"
    top_metrics.to_csv(top_metrics_dst, index=False)
    copied.append({"source": "generated_from_metrics", "destination": str(top_metrics_dst.resolve()), "bytes": int(top_metrics_dst.stat().st_size)})

    plot_feature_condition_summary(metrics, paths["core"] / "Fig3_feature_condition_summary.png")
    plot_perturbation_stats(metrics, paths["core"] / "Fig4_schedule_perturbation_decoding_stats.png")
    improvement_stats = perturbation_improvement_table(metrics)
    improvement_stats_dst = paths["core"] / "perturbation_model_improvement_stats.csv"
    improvement_stats.to_csv(improvement_stats_dst, index=False)
    copied.append(
        {
            "source": "generated_from_metrics",
            "destination": str(improvement_stats_dst.resolve()),
            "bytes": int(improvement_stats_dst.stat().st_size),
        }
    )
    plot_perturbation_improvement(improvement_stats, paths["core"] / "Fig7_perturbation_model_improvement_stats.png")

    selected_csv_specs = [
        ("multi_feature", "phase_only", "none", "sample_kfold"),
        ("multi_feature", "phase_only", "none", "day_groupkfold"),
        ("multi_feature", "phase_perturb", "schedule", "sample_kfold"),
        ("multi_feature", "phase_perturb", "schedule", "day_groupkfold"),
        ("multi_feature", "phase_perturb", "24h", "sample_kfold"),
        ("multi_feature", "phase_perturb", "24h", "day_groupkfold"),
    ]
    prediction_tables = {}
    for condition, target, effect, cv in selected_csv_specs:
        stem = source_name(condition, target, effect, cv, "predictions.csv")
        src = source / stem
        dst = paths["csv"] / stem
        copied.append(copy_file(src, dst))
        prediction_tables[(condition, target, effect, cv)] = pd.read_csv(dst)

    # Core phase-decoding composites are made by copying the clearest existing views.
    core_copy_specs = [
        (
            "Fig1_multi_feature_phase_decoding_sample_cv_true_vs_pred_hour.png",
            "multi_feature",
            "phase_only",
            "none",
            "sample_kfold",
            "true_vs_pred_hour.png",
        ),
        (
            "Fig1_multi_feature_phase_decoding_sample_cv_circular_heatmap.png",
            "multi_feature",
            "phase_only",
            "none",
            "sample_kfold",
            "circular_heatmap.png",
        ),
        (
            "Fig2_multi_feature_phase_decoding_day_cv_true_vs_pred_hour.png",
            "multi_feature",
            "phase_only",
            "none",
            "day_groupkfold",
            "true_vs_pred_hour.png",
        ),
        (
            "Fig2_multi_feature_phase_decoding_day_cv_circular_heatmap.png",
            "multi_feature",
            "phase_only",
            "none",
            "day_groupkfold",
            "circular_heatmap.png",
        ),
    ]
    for dst_name, condition, target, effect, cv, suffix in core_copy_specs:
        src = source / source_name(condition, target, effect, cv, suffix)
        copied.append(copy_file(src, paths["core"] / dst_name))

    diag_specs = [
        ("multi_feature_unit_circle_full_QC.png", "multi_feature", "phase_only", "none", "sample_kfold", "unit_circle_full.png"),
        ("multi_feature_unit_circle_robust99.png", "multi_feature", "phase_only", "none", "sample_kfold", "unit_circle_robust99.png"),
        ("multi_feature_error_by_hour.png", "multi_feature", "phase_only", "none", "sample_kfold", "error_by_hour.png"),
        ("multi_feature_schedule_perturb_circular_heatmap.png", "multi_feature", "phase_perturb", "schedule", "sample_kfold", "circular_heatmap.png"),
    ]
    for dst_name, condition, target, effect, cv, suffix in diag_specs:
        src = source / source_name(condition, target, effect, cv, suffix)
        copied.append(copy_file(src, paths["diag"] / dst_name))

    phase_only_sample = prediction_tables[("multi_feature", "phase_only", "none", "sample_kfold")]
    phase_only_day = prediction_tables[("multi_feature", "phase_only", "none", "day_groupkfold")]
    schedule_sample = prediction_tables[("multi_feature", "phase_perturb", "schedule", "sample_kfold")]
    schedule_day = prediction_tables[("multi_feature", "phase_perturb", "schedule", "day_groupkfold")]
    flagged_schedule_sample = add_onset_window_flags(schedule_sample, schedule, hours_after=3.0)
    flagged_schedule_day = add_onset_window_flags(schedule_day, schedule, hours_after=3.0)
    flagged_phase_only_sample = add_onset_window_flags(phase_only_sample, schedule, hours_after=3.0)
    flagged_phase_only_day = add_onset_window_flags(phase_only_day, schedule, hours_after=3.0)
    plot_perturbation_timeline(schedule_sample, paths["core"] / "Fig5_perturbation_timeline_marked.png")
    plot_3d_perturbation_axis(flagged_schedule_sample, paths["core"] / "Fig6_phase_plus_perturbation_3d_axis.png")
    plot_phase_vs_perturb_side_by_side(
        flagged_phase_only_sample,
        flagged_schedule_sample,
        paths["core"] / "Fig10_multi_feature_sample_cv_phase_only_vs_perturb_scatter.png",
        "Multi-feature sample-CV decoder comparison with perturbation onset highlights",
    )
    plot_phase_vs_perturb_side_by_side(
        flagged_phase_only_day,
        flagged_schedule_day,
        paths["core"] / "Fig11_multi_feature_day_cv_phase_only_vs_perturb_scatter.png",
        "Multi-feature held-out-day CV decoder comparison with perturbation onset highlights",
    )
    for cv_mode, table, figure_name in [
        (
            "sample_kfold",
            flagged_schedule_sample,
            "Fig8_multi_feature_schedule_sample_cv_onset_highlight_scatter.png",
        ),
        (
            "day_groupkfold",
            flagged_schedule_day,
            "Fig9_multi_feature_schedule_day_cv_onset_highlight_scatter.png",
        ),
    ]:
        flagged = table
        flagged_name = source_name(
            "multi_feature",
            "phase_perturb",
            "schedule",
            cv_mode,
            "predictions_with_onset3h_flags.csv",
        )
        flagged_dst = paths["csv"] / flagged_name
        flagged.to_csv(flagged_dst, index=False)
        copied.append(
            {
                "source": "generated_from_selected_predictions_and_schedule",
                "destination": str(flagged_dst.resolve()),
                "bytes": int(flagged_dst.stat().st_size),
            }
        )
        plot_onset_highlight_scatter(
            flagged,
            paths["core"] / figure_name,
            title=f"Multi-feature schedule perturbation decoder: {cv_mode} onset highlights",
        )

    source_png_count = len(list(source.glob("*.png")))
    source_csv_count = len(list(source.glob("*.csv")))
    summary = {
        "source_dir": str(source),
        "output_dir": str(out_dir),
        "source_png_count": int(source_png_count),
        "source_csv_count": int(source_csv_count),
        "copied_file_count": int(len(copied)),
        "copied_files": copied,
        "generated_figures": [
            str((paths["core"] / "Fig3_feature_condition_summary.png").resolve()),
            str((paths["core"] / "Fig4_schedule_perturbation_decoding_stats.png").resolve()),
            str((paths["core"] / "Fig5_perturbation_timeline_marked.png").resolve()),
            str((paths["core"] / "Fig6_phase_plus_perturbation_3d_axis.png").resolve()),
            str((paths["core"] / "Fig7_perturbation_model_improvement_stats.png").resolve()),
            str((paths["core"] / "Fig8_multi_feature_schedule_sample_cv_onset_highlight_scatter.png").resolve()),
            str((paths["core"] / "Fig9_multi_feature_schedule_day_cv_onset_highlight_scatter.png").resolve()),
            str((paths["core"] / "Fig10_multi_feature_sample_cv_phase_only_vs_perturb_scatter.png").resolve()),
            str((paths["core"] / "Fig11_multi_feature_day_cv_phase_only_vs_perturb_scatter.png").resolve()),
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_readme(out_dir, paths, copied, summary)
    summary_path = paths["logs"] / "curation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--schedule-json", type=Path, default=DEFAULT_SCHEDULE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / "curation_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        import contextlib

        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            summary = curate(args)
            log(f"finished in {time.perf_counter() - started:.2f} seconds")
    print(f"Wrote curated theta decoder outputs to: {out_dir}")
    print(f"Summary: {out_dir / 'logs' / 'curation_summary.json'}")
    print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()
