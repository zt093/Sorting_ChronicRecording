from __future__ import annotations

"""Fit per-feature theta/perturbation encoder models on hourly threshold features."""

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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_LONG_TABLE = PROJECT_DIR / "derived" / "threshold_hour_features_long.csv"
DEFAULT_SCHEDULE = (
    PROJECT_DIR
    / "outputs"
    / "combined_population_summary"
    / "threshold_sham_drug_marker_schedule.json"
)
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs" / "theta_regression" / "organized_hourly_theta_encoder"

FEATURE_METRICS = ("firing_rate_hz", "amplitude_ptp_uv", "cv2", "peak_to_trough_ms")
MODEL_SPECS = {
    "phase_only": ("clock_phase_cos", "clock_phase_sin"),
    "perturb_only": ("perturbation_sham", "perturbation_drug"),
    "phase_perturb": (
        "clock_phase_cos",
        "clock_phase_sin",
        "perturbation_sham",
        "perturbation_drug",
    ),
    "phase_perturb_interaction": (
        "clock_phase_cos",
        "clock_phase_sin",
        "perturbation_sham",
        "perturbation_drug",
        "sham_x_cos",
        "sham_x_sin",
        "drug_x_cos",
        "drug_x_sin",
    ),
}


def log(message: str) -> None:
    print(f"[run_organized_hourly_theta_encoder] {message}", flush=True)


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "core": out_dir / "core_results",
        "csv": out_dir / "csv",
        "logs": out_dir / "logs",
        "diagnostic": out_dir / "diagnostic_figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_schedule(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def effect_label(effect_hours: float) -> str:
    return "schedule" if float(effect_hours) == 0.0 else f"{effect_hours:g}h"


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
            if float(effect_hours) == 0.0:
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

    sham.loc[drug == 1] = 0
    return pd.DataFrame({"perturbation_sham": sham, "perturbation_drug": drug}), intervals_used


def add_design_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["sham_x_cos"] = out["perturbation_sham"] * out["clock_phase_cos"]
    out["sham_x_sin"] = out["perturbation_sham"] * out["clock_phase_sin"]
    out["drug_x_cos"] = out["perturbation_drug"] * out["clock_phase_cos"]
    out["drug_x_sin"] = out["perturbation_drug"] * out["clock_phase_sin"]
    return out


def adjusted_r2(r2: float, n_samples: int, n_predictors: int) -> float:
    if n_samples <= n_predictors + 1 or not np.isfinite(r2):
        return np.nan
    return 1.0 - (1.0 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1)


def fit_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: np.ndarray,
    random_seed: int,
) -> dict:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, fit_intercept=True))
    model.fit(X, y)
    pred = model.predict(X)
    r2_in = r2_score(y, pred)
    ridge = model.named_steps["ridgecv"]
    scaler = model.named_steps["standardscaler"]
    alpha = float(ridge.alpha_)

    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2 and len(y) >= 10:
        n_splits = min(5, len(unique_groups))
        cv = GroupKFold(n_splits=n_splits)
        cv_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True))
        cv_pred = cross_val_predict(cv_model, X, y, cv=cv, groups=groups)
        r2_day_cv = r2_score(y, cv_pred)
    else:
        cv_pred = np.full_like(y, np.nan, dtype=float)
        r2_day_cv = np.nan

    return {
        "model": model,
        "pred": pred,
        "cv_pred": cv_pred,
        "r2_in_sample": float(r2_in),
        "r2_adjusted": float(adjusted_r2(r2_in, len(y), X.shape[1])),
        "r2_day_groupkfold": float(r2_day_cv) if np.isfinite(r2_day_cv) else np.nan,
        "alpha": alpha,
        "coef_scaled": ridge.coef_.astype(float),
        "intercept": float(ridge.intercept_),
        "x_mean": scaler.mean_.astype(float),
        "x_scale": scaler.scale_.astype(float),
    }


def permutation_pvalue(observed: float, null_values: list[float]) -> float:
    if not null_values or not np.isfinite(observed):
        return np.nan
    null = np.asarray(null_values, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) == 0:
        return np.nan
    return float((np.sum(null >= observed) + 1.0) / (len(null) + 1.0))


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return pd.Series(adjusted, index=pvalues.index)
    valid_values = values[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    m = len(ranked)
    raw = ranked * m / np.arange(1, m + 1)
    corrected = np.minimum.accumulate(raw[::-1])[::-1]
    corrected = np.clip(corrected, 0, 1)
    out_valid = np.empty_like(corrected)
    out_valid[order] = corrected
    adjusted[valid] = out_valid
    return pd.Series(adjusted, index=pvalues.index)


def permutation_tests(
    design: pd.DataFrame,
    y: np.ndarray,
    observed_phase_delta: float,
    observed_perturb_delta: float,
    alpha: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if n_permutations <= 0:
        return np.nan, np.nan

    phase_null = []
    perturb_null = []
    base_columns = list(MODEL_SPECS["phase_perturb"])
    phase_only_columns = list(MODEL_SPECS["phase_only"])
    perturb_only_columns = list(MODEL_SPECS["perturb_only"])

    for _ in range(n_permutations):
        phase_perm = design.copy()
        phase_order = rng.permutation(len(phase_perm))
        phase_perm[["clock_phase_cos", "clock_phase_sin"]] = phase_perm[
            ["clock_phase_cos", "clock_phase_sin"]
        ].to_numpy()[phase_order]
        phase_perm = add_design_columns(phase_perm)

        X_full_phase = phase_perm[base_columns].to_numpy(dtype=float)
        X_perturb = phase_perm[perturb_only_columns].to_numpy(dtype=float)
        full_phase = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True))
        perturb = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True))
        full_phase.fit(X_full_phase, y)
        perturb.fit(X_perturb, y)
        phase_null.append(r2_score(y, full_phase.predict(X_full_phase)) - r2_score(y, perturb.predict(X_perturb)))

        perturb_perm = design.copy()
        perturb_order = rng.permutation(len(perturb_perm))
        perturb_perm[["perturbation_sham", "perturbation_drug"]] = perturb_perm[
            ["perturbation_sham", "perturbation_drug"]
        ].to_numpy()[perturb_order]
        perturb_perm = add_design_columns(perturb_perm)

        X_full_perturb = perturb_perm[base_columns].to_numpy(dtype=float)
        X_phase = perturb_perm[phase_only_columns].to_numpy(dtype=float)
        full_perturb = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True))
        phase = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True))
        full_perturb.fit(X_full_perturb, y)
        phase.fit(X_phase, y)
        perturb_null.append(r2_score(y, full_perturb.predict(X_full_perturb)) - r2_score(y, phase.predict(X_phase)))

    return (
        permutation_pvalue(observed_phase_delta, phase_null),
        permutation_pvalue(observed_perturb_delta, perturb_null),
    )


def summarize_feature(
    group: pd.DataFrame,
    metric: str,
    effect: str,
    alphas: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[dict, list[dict], pd.DataFrame]:
    valid = group.dropna(subset=[metric]).copy()
    valid = valid[np.isfinite(pd.to_numeric(valid[metric], errors="coerce"))].copy()
    y = valid[metric].to_numpy(dtype=float)
    design = valid[
        [
            "calendar_day",
            "hour_start_datetime",
            "clock_hour_of_day",
            "clock_phase_cos",
            "clock_phase_sin",
            "perturbation_sham",
            "perturbation_drug",
            "sham_x_cos",
            "sham_x_sin",
            "drug_x_cos",
            "drug_x_sin",
        ]
    ].copy()
    groups = valid["calendar_day"].astype(str).to_numpy()

    meta = {
        "effect_label": effect,
        "pair_id": str(valid["pair_id"].iloc[0]),
        "unit_id": valid["unit_id"].iloc[0],
        "sg_ch": valid["sg_ch"].iloc[0],
        "threshold_min_uv": valid["threshold_min_uv"].iloc[0],
        "threshold_label": valid["threshold_label"].iloc[0],
        "metric": metric,
        "n_samples": int(len(valid)),
        "n_days": int(valid["calendar_day"].nunique()),
        "n_sham": int(valid["perturbation_sham"].sum()),
        "n_drug": int(valid["perturbation_drug"].sum()),
    }

    model_results = {}
    coef_rows = []
    pred_table = pd.DataFrame(
        {
            "calendar_day": valid["calendar_day"],
            "hour_start_datetime": valid["hour_start_datetime"],
            "clock_hour_of_day": valid["clock_hour_of_day"],
            "pair_id": valid["pair_id"],
            "metric": metric,
            "observed": y,
        }
    )

    if len(valid) < 12 or np.nanstd(y) == 0:
        row = meta | {
            "skip_reason": "too_few_samples_or_constant_response",
            "phase_amplitude": np.nan,
            "preferred_theta": np.nan,
            "preferred_hour": np.nan,
            "delta_r2_phase_given_perturb": np.nan,
            "delta_r2_perturb_given_phase": np.nan,
            "p_phase_given_perturb": np.nan,
            "p_perturb_given_phase": np.nan,
        }
        for model_name in MODEL_SPECS:
            row[f"{model_name}_r2_in_sample"] = np.nan
            row[f"{model_name}_r2_adjusted"] = np.nan
            row[f"{model_name}_r2_day_groupkfold"] = np.nan
            row[f"{model_name}_alpha"] = np.nan
        return row, coef_rows, pred_table

    for model_name, columns in MODEL_SPECS.items():
        X = design[list(columns)].to_numpy(dtype=float)
        result = fit_model(X, y, groups, alphas, random_seed=0)
        model_results[model_name] = result
        pred_table[f"pred_{model_name}"] = result["pred"]
        pred_table[f"cv_pred_{model_name}"] = result["cv_pred"]
        for column, coef, x_mean, x_scale in zip(columns, result["coef_scaled"], result["x_mean"], result["x_scale"]):
            coef_rows.append(
                meta
                | {
                    "model_name": model_name,
                    "predictor": column,
                    "coef_scaled_x": float(coef),
                    "x_mean": float(x_mean),
                    "x_scale": float(x_scale),
                    "intercept_scaled_model": result["intercept"],
                    "alpha": result["alpha"],
                }
            )

    phase_model = model_results["phase_only"]
    phase_perturb_model = model_results["phase_perturb"]
    perturb_model = model_results["perturb_only"]
    beta_cos = phase_model["coef_scaled"][0]
    beta_sin = phase_model["coef_scaled"][1]
    preferred_theta = np.mod(np.arctan2(beta_sin, beta_cos), 2.0 * np.pi)
    preferred_hour = preferred_theta * 24.0 / (2.0 * np.pi)
    phase_amplitude = np.sqrt(beta_cos**2 + beta_sin**2)
    delta_phase = phase_perturb_model["r2_in_sample"] - perturb_model["r2_in_sample"]
    delta_perturb = phase_perturb_model["r2_in_sample"] - phase_model["r2_in_sample"]
    p_phase, p_perturb = permutation_tests(
        design,
        y,
        delta_phase,
        delta_perturb,
        phase_perturb_model["alpha"],
        n_permutations,
        rng,
    )

    row = meta | {
        "skip_reason": "",
        "phase_amplitude": float(phase_amplitude),
        "preferred_theta": float(preferred_theta),
        "preferred_hour": float(preferred_hour),
        "delta_r2_phase_given_perturb": float(delta_phase),
        "delta_r2_perturb_given_phase": float(delta_perturb),
        "p_phase_given_perturb": p_phase,
        "p_perturb_given_phase": p_perturb,
    }
    for model_name, result in model_results.items():
        row[f"{model_name}_r2_in_sample"] = result["r2_in_sample"]
        row[f"{model_name}_r2_adjusted"] = result["r2_adjusted"]
        row[f"{model_name}_r2_day_groupkfold"] = result["r2_day_groupkfold"]
        row[f"{model_name}_alpha"] = result["alpha"]
    return row, coef_rows, pred_table


def plot_phase_amplitude(metrics: pd.DataFrame, out_path: Path) -> None:
    rows = metrics[np.isfinite(metrics["phase_amplitude"])].copy()
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for metric, group in rows.groupby("metric"):
        ax.hist(group["phase_amplitude"], bins=30, alpha=0.55, label=metric)
    ax.set_xlabel("phase amplitude, response units per 1 SD phase predictor")
    ax.set_ylabel("feature count")
    ax.set_title("Encoder phase modulation amplitude")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_preferred_hour(metrics: pd.DataFrame, out_path: Path) -> None:
    rows = metrics[np.isfinite(metrics["preferred_hour"])].copy()
    theta = rows["preferred_hour"].to_numpy(dtype=float) / 24.0 * 2.0 * np.pi
    bins = np.linspace(0, 2 * np.pi, 25)
    counts, edges = np.histogram(theta, bins=bins)
    fig = plt.figure(figsize=(7, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    width = np.diff(edges)
    ax.bar(edges[:-1], counts, width=width, align="edge", color="#2563eb", alpha=0.75, edgecolor="white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Preferred phase hour across encoded features")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_delta_r2(metrics: pd.DataFrame, out_path: Path) -> None:
    rows = metrics[np.isfinite(metrics["delta_r2_phase_given_perturb"])].copy()
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for metric, group in rows.groupby("metric"):
        ax.scatter(
            group["delta_r2_phase_given_perturb"],
            group["delta_r2_perturb_given_phase"],
            s=26,
            alpha=0.7,
            label=metric,
        )
    ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_xlabel("delta R2 phase given perturbation")
    ax.set_ylabel("delta R2 perturbation given phase")
    ax.set_title("Incremental variance explained by phase and perturbation")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_top_feature_fits(predictions: pd.DataFrame, metrics: pd.DataFrame, out_path: Path, max_features: int = 6) -> None:
    top = (
        metrics[np.isfinite(metrics["delta_r2_phase_given_perturb"])]
        .sort_values("delta_r2_phase_given_perturb", ascending=False)
        .head(max_features)
    )
    if top.empty:
        return
    fig, axes = plt.subplots(len(top), 1, figsize=(12, 2.5 * len(top)), sharex=False, constrained_layout=True)
    if len(top) == 1:
        axes = [axes]
    for ax, row in zip(axes, top.itertuples(index=False)):
        subset = predictions[
            (predictions["pair_id"] == row.pair_id)
            & (predictions["metric"] == row.metric)
            & (predictions["effect_label"] == row.effect_label)
        ].copy()
        subset["time"] = pd.to_datetime(subset["hour_start_datetime"])
        ax.plot(subset["time"], subset["observed"], color="#111827", linewidth=1, label="observed")
        ax.plot(subset["time"], subset["pred_phase_perturb"], color="#2563eb", linewidth=1, label="phase+perturb fit")
        ax.set_title(f"{row.pair_id} {row.metric}: delta R2 phase={row.delta_r2_phase_given_perturb:.3f}")
        ax.set_ylabel(row.metric)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_readme(out_dir: Path, args: argparse.Namespace, summary: dict) -> None:
    text = f"""# Organized Hourly Theta Encoder

This folder contains per-feature encoder models:

```text
time-of-day phase and perturbation state -> individual threshold feature
```

Input:

```text
{args.long_table}
```

Schedule:

```text
{args.schedule_json}
```

Default output structure:

```text
core_results/        compact presentation tables and key figures
csv/                 full per-feature metrics, coefficients, and selected predictions
diagnostic_figures/  extra QC figures
logs/                run log and JSON summary
```

Models fit per `pair_id + metric`:

- `phase_only`: `y ~ cos(theta) + sin(theta)`
- `perturb_only`: `y ~ perturbation_sham + perturbation_drug`
- `phase_perturb`: `y ~ cos(theta) + sin(theta) + perturbation_sham + perturbation_drug`
- `phase_perturb_interaction`: adds sham/drug by cos/sin interaction terms

Fitting uses `StandardScaler()` followed by `RidgeCV(alphas=np.logspace({args.alpha_min_log10}, {args.alpha_max_log10}, {args.n_alphas}), fit_intercept=True)`.
Missing values are dropped per response feature before fitting.

Permutation p-values use {args.n_permutations} permutations per feature for the primary `phase_perturb` nested-model deltas.
Current effect windows: {', '.join(effect_label(x) for x in args.perturbation_effect_hours)}.

Run summary:

```text
features evaluated: {summary['n_feature_models']}
elapsed seconds: {summary['elapsed_seconds']:.2f}
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    paths = ensure_dirs(args.out_dir)
    rng = np.random.default_rng(args.random_seed)
    alphas = np.logspace(args.alpha_min_log10, args.alpha_max_log10, args.n_alphas)

    log(f"loading long table: {args.long_table}")
    table = pd.read_csv(args.long_table)
    table["hour_start_datetime"] = pd.to_datetime(table["hour_start_datetime"])
    for column in ["clock_phase_cos", "clock_phase_sin", *FEATURE_METRICS]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    schedule = load_schedule(args.schedule_json)

    all_metrics = []
    all_coefficients = []
    selected_predictions = []
    intervals_by_effect = {}
    grouped = list(table.groupby("pair_id", sort=False))
    log(f"fitting {len(grouped)} pair_id groups x {len(FEATURE_METRICS)} metrics")

    for effect_hours in args.perturbation_effect_hours:
        label = effect_label(effect_hours)
        perturb, intervals_used = assign_perturbations(table["hour_start_datetime"], schedule, effect_hours)
        intervals_by_effect[label] = intervals_used
        effect_table = pd.concat([table.reset_index(drop=True), perturb.reset_index(drop=True)], axis=1)
        effect_table = add_design_columns(effect_table)
        log(f"effect window {label}: sham samples={int(effect_table['perturbation_sham'].sum())}, drug samples={int(effect_table['perturbation_drug'].sum())}")

        for _, group in effect_table.groupby("pair_id", sort=False):
            for metric in FEATURE_METRICS:
                row, coef_rows, pred_table = summarize_feature(
                    group,
                    metric,
                    label,
                    alphas,
                    args.n_permutations,
                    rng,
                )
                all_metrics.append(row)
                all_coefficients.extend(coef_rows)
                if row.get("skip_reason", "") == "" and label == effect_label(args.perturbation_effect_hours[0]):
                    selected_predictions.append(pred_table.assign(effect_label=label))

    metrics = pd.DataFrame(all_metrics)
    coefficients = pd.DataFrame(all_coefficients)
    predictions = pd.concat(selected_predictions, ignore_index=True) if selected_predictions else pd.DataFrame()

    for p_col, q_col in [
        ("p_phase_given_perturb", "q_phase_given_perturb"),
        ("p_perturb_given_phase", "q_perturb_given_phase"),
    ]:
        metrics[q_col] = metrics.groupby(["effect_label", "metric"], group_keys=False)[p_col].apply(bh_fdr)

    metrics_csv = paths["csv"] / "encoder_all_feature_metrics.csv"
    coef_csv = paths["csv"] / "encoder_all_coefficients.csv"
    pred_csv = paths["csv"] / "encoder_selected_predictions.csv"
    metrics.to_csv(metrics_csv, index=False)
    coefficients.to_csv(coef_csv, index=False)
    predictions.to_csv(pred_csv, index=False)

    top = (
        metrics[metrics["effect_label"] == effect_label(args.perturbation_effect_hours[0])]
        .sort_values(["delta_r2_phase_given_perturb", "phase_perturb_r2_day_groupkfold"], ascending=False)
        .head(args.top_n)
    )
    top_csv = paths["core"] / "encoder_summary_top_features.csv"
    top.to_csv(top_csv, index=False)

    primary = metrics[metrics["effect_label"] == effect_label(args.perturbation_effect_hours[0])].copy()
    plot_phase_amplitude(primary, paths["core"] / "encoder_phase_amplitude_histogram.png")
    plot_preferred_hour(primary, paths["core"] / "encoder_preferred_hour_polar_histogram.png")
    plot_delta_r2(primary, paths["core"] / "encoder_delta_r2_phase_vs_perturb_scatter.png")
    plot_top_feature_fits(predictions, primary, paths["core"] / "encoder_top_feature_fits.png")

    summary = {
        "input_long_table": str(args.long_table.resolve()),
        "schedule_json": str(args.schedule_json.resolve()) if args.schedule_json else None,
        "output_dir": str(args.out_dir.resolve()),
        "effect_windows": [effect_label(x) for x in args.perturbation_effect_hours],
        "feature_metrics": list(FEATURE_METRICS),
        "model_specs": {key: list(value) for key, value in MODEL_SPECS.items()},
        "n_feature_models": int(len(metrics)),
        "n_coefficients": int(len(coefficients)),
        "n_permutations": int(args.n_permutations),
        "alpha_grid": [float(x) for x in alphas],
        "intervals_by_effect": intervals_by_effect,
        "outputs": {
            "metrics_csv": str(metrics_csv.resolve()),
            "coefficients_csv": str(coef_csv.resolve()),
            "selected_predictions_csv": str(pred_csv.resolve()),
            "top_features_csv": str(top_csv.resolve()),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    summary_json = paths["logs"] / "encoder_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(args.out_dir, args, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-table", type=Path, default=DEFAULT_LONG_TABLE)
    parser.add_argument("--schedule-json", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--perturbation-effect-hours", type=float, nargs="+", default=[0.0])
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--alpha-min-log10", type=float, default=-3)
    parser.add_argument("--alpha-max-log10", type=float, default=6)
    parser.add_argument("--n-alphas", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "logs" / "encoder_run_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            summary = run(args)
            log(f"finished in {time.perf_counter() - started:.2f} seconds")
    print(f"Wrote theta encoder outputs to: {args.out_dir.resolve()}")
    print(f"Summary: {args.out_dir.resolve() / 'logs' / 'encoder_summary.json'}")
    print(f"Run log: {log_path.resolve()}")


if __name__ == "__main__":
    main()
