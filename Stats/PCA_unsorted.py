from __future__ import annotations

"""
One-hour unsorted channel-level PCA pilot.

This script benchmarks dimensionality analysis on one 1-hour recording without
running spike sorting. It reuses the same recording loader, probe map,
bad-channel exclusion, per-shank channel selection, common reference,
spike-band filtering, and large-artifact blanking used by
Combined_NWB+Sorting+Analyze.py.

First-pass feature: spike-band RMS in 60-second bins.
"""

import csv
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import probeinterface as pi
import spikeinterface.preprocessing as preproc
from sklearn.decomposition import PCA


# =============================================================================
# User Configuration
# =============================================================================

DATA_PATH = None  # Leave as None to enter the recording file/folder in terminal.
OUTPUT_BASE_DIR = Path(r"S:\PCA_unsorted")
BIN_SIZE_SECONDS = 60.0
ONE_HOUR_SECONDS = 3600.0
FEATURE_TYPE = "spike_band_rms"
NORMALIZATION_METHOD = "per_channel_zscore_with_zero_std_channels_removed"


@dataclass
class Config:
    data_path: Path | None = DATA_PATH
    output_base_dir: Path = OUTPUT_BASE_DIR
    bin_size_seconds: float = BIN_SIZE_SECONDS
    one_hour_seconds: float = ONE_HOUR_SECONDS
    feature_type: str = FEATURE_TYPE
    normalization_method: str = NORMALIZATION_METHOD


# =============================================================================
# Pipeline Import and Logging
# =============================================================================


def load_sorting_pipeline_module():
    pipeline_path = Path(__file__).resolve().parents[1] / "Combined_NWB+Sorting+Analyze.py"
    sorting_check_dir = pipeline_path.parent
    repo_root = Path(__file__).resolve().parents[2]
    for import_path in (repo_root, sorting_check_dir):
        import_path_str = str(import_path)
        if import_path_str not in sys.path:
            sys.path.insert(0, import_path_str)
    spec = importlib.util.spec_from_file_location("combined_sorting_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import sorting pipeline from {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PIPELINE = load_sorting_pipeline_module()


def log_status(message: str) -> None:
    print(f"[PCA_unsorted] {message}", flush=True)


def format_elapsed_time(elapsed_sec: float) -> str:
    if hasattr(PIPELINE, "format_elapsed_time"):
        return PIPELINE.format_elapsed_time(float(elapsed_sec))
    if elapsed_sec < 60:
        return f"{elapsed_sec:.2f} seconds"
    if elapsed_sec < 3600:
        return f"{elapsed_sec / 60.0:.2f} minutes"
    return f"{elapsed_sec / 3600.0:.2f} hours"


class TimingLogger:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self.lines: list[str] = []
        self.total_start = perf_counter()

    def step(self, name: str, callback, *args, **kwargs):
        log_status(f"Starting {name}...")
        start_time = perf_counter()
        result = callback(*args, **kwargs)
        elapsed_sec = perf_counter() - start_time
        self.record(name, elapsed_sec)
        return result

    def record(self, name: str, elapsed_sec: float) -> None:
        self.timings[f"{name}_elapsed_sec"] = float(elapsed_sec)
        line = f"[timer] {name}: {format_elapsed_time(elapsed_sec)}"
        self.lines.append(line)
        print(line, flush=True)

    def finish_total(self) -> None:
        self.record("total_runtime", perf_counter() - self.total_start)

    def save(self, output_path: Path, extra_payload: dict | None = None) -> None:
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "timings": self.timings,
        }
        if extra_payload:
            payload.update(extra_payload)
        output_path.write_text(
            "\n".join(self.lines) + "\n\n" + json.dumps(payload, indent=2),
            encoding="utf-8",
        )


# =============================================================================
# Input and Metadata Helpers
# =============================================================================


def prompt_for_data_path(default_path: Path | None) -> Path:
    if default_path is not None:
        return Path(default_path)
    raw_value = input("\nEnter one recording file/folder path for the 1-hour PCA pilot: ").strip()
    input_paths = PIPELINE.parse_input_paths(raw_value)
    return input_paths[0]


def choose_one_data_file(data_files: list[Path]) -> Path:
    if not data_files:
        raise FileNotFoundError("No recording files were found.")
    if len(data_files) == 1:
        log_status(f"Using only recording file: {data_files[0]}")
        return data_files[0]

    log_status("Multiple recording files found. This pilot processes exactly one file.")
    for index, data_file in enumerate(data_files, start=1):
        print(f"  {index}: {data_file}")
    choice = input("Enter file number to process, or press Enter for the first file: ").strip()
    if not choice:
        return data_files[0]
    selected_index = int(choice)
    if selected_index < 1 or selected_index > len(data_files):
        raise ValueError(f"Invalid file number {selected_index}; expected 1-{len(data_files)}.")
    return data_files[selected_index - 1]


def build_one_shank_config(rec_path: Path, data_file: Path) -> dict:
    shank_configs, _ = PIPELINE.build_shank_configs(
        rec_path=rec_path,
        recording_method=PIPELINE.RECORDING_METHOD,
        probe_file=PIPELINE.PROBE_FILE,
        impedance_file=PIPELINE.IMPEDANCE_FILE,
    )
    if not shank_configs:
        raise RuntimeError("No shanks selected or available after bad-channel filtering.")
    if len(shank_configs) == 1:
        selected_config = shank_configs[0]
    else:
        shank_ids = [str(config["shank_id"]) for config in shank_configs]
        choice = input(
            f"This pilot processes one shank. Enter one shank from {shank_ids}, "
            f"or press Enter for {shank_ids[0]}: "
        ).strip()
        selected_shank = choice if choice else shank_ids[0]
        matches = [config for config in shank_configs if str(config["shank_id"]) == selected_shank]
        if not matches:
            raise ValueError(f"Selected shank {selected_shank!r} was not in {shank_ids}.")
        selected_config = matches[0]

    log_status(
        f"Selected shank {selected_config['shank_id']} with "
        f"{len(selected_config['electrode_df'])} channel(s) after bad-channel exclusion."
    )
    PIPELINE.preview_channel_resolution(
        data_files=[data_file],
        recording_method=PIPELINE.RECORDING_METHOD,
        shank_configs=[selected_config],
    )
    return selected_config


def parse_session_details(data_file: Path) -> dict:
    recording_label = PIPELINE.build_recording_label(data_file)
    match = re.search(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})", recording_label)
    if match:
        return {
            "session_id": recording_label,
            "date": match.group("date"),
            "clock_hour": int(match.group("hour")),
        }

    stem_match = re.search(
        r"(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})[_-]?(?P<hour>\d{2})",
        data_file.stem,
    )
    if stem_match:
        date_label = (
            f"{stem_match.group('year')}-{stem_match.group('month')}-"
            f"{stem_match.group('day')}"
        )
        return {
            "session_id": recording_label,
            "date": date_label,
            "clock_hour": int(stem_match.group("hour")),
        }

    return {
        "session_id": recording_label,
        "date": "",
        "clock_hour": "",
    }


def make_output_dir(config: Config, session_id: str, shank_id: str) -> Path:
    safe_session_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(session_id)).strip("_") or "session"
    output_dir = config.output_base_dir / f"pca_unsorted_{safe_session_id}_sh{shank_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# Preprocessing, Feature Extraction, and PCA
# =============================================================================


def apply_sorting_preprocessing(recording, timing: TimingLogger):
    filtering_total_start = perf_counter()

    log_status("Applying global median common reference.")
    reference_start = perf_counter()
    rec_cr = preproc.common_reference(recording, operator="median", reference="global")
    timing.record("referencing_setup", perf_counter() - reference_start)

    log_status("Applying 300-6000 Hz spike-band filter.")
    filter_setup_start = perf_counter()
    rec_filt = preproc.bandpass_filter(
        rec_cr,
        freq_min=300,
        freq_max=6000,
        dtype="float32",
    )
    timing.record("filtering_setup", perf_counter() - filter_setup_start)

    log_status("Applying existing large-artifact blanking step.")
    artifact_start = perf_counter()
    rec_denoised, artifact_metadata = PIPELINE.remove_big_noise_artifacts(rec_filt)
    timing.record("artifact_blanking", perf_counter() - artifact_start)
    timing.record("filtering", perf_counter() - filtering_total_start)
    return rec_denoised, artifact_metadata


def compute_spike_band_rms_features(recording, bin_size_seconds: float) -> np.ndarray:
    sampling_frequency = float(recording.get_sampling_frequency())
    n_frames = int(recording.get_num_frames())
    frames_per_bin = int(round(float(bin_size_seconds) * sampling_frequency))
    if frames_per_bin <= 0:
        raise ValueError("bin_size_seconds must produce at least one frame per bin.")
    n_bins = n_frames // frames_per_bin
    if n_bins <= 0:
        raise ValueError("Recording is shorter than one analysis bin.")

    n_channels = int(recording.get_num_channels())
    features = np.zeros((n_bins, n_channels), dtype=np.float32)
    for bin_index in range(n_bins):
        start_frame = bin_index * frames_per_bin
        end_frame = start_frame + frames_per_bin
        log_status(
            f"Computing RMS bin {bin_index + 1}/{n_bins} "
            f"({start_frame / sampling_frequency:.1f}-{end_frame / sampling_frequency:.1f} sec)"
        )
        traces = np.asarray(
            recording.get_traces(start_frame=start_frame, end_frame=end_frame),
            dtype=np.float32,
        )
        features[bin_index, :] = np.sqrt(np.mean(np.square(traces), axis=0))
    return features


def zscore_features_for_pca(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(features, axis=0)
    stds = np.nanstd(features, axis=0, ddof=0)
    valid_channels = np.isfinite(means) & np.isfinite(stds) & (stds > 0)
    if not np.any(valid_channels):
        raise RuntimeError("All channels had zero or invalid standard deviation; PCA cannot run.")
    zscored = (features[:, valid_channels] - means[valid_channels]) / stds[valid_channels]
    zscored = np.asarray(zscored, dtype=np.float32)
    return zscored, valid_channels, means, stds


def run_pca(features_z: np.ndarray) -> dict:
    n_components = min(features_z.shape[0], features_z.shape[1])
    if n_components < 1:
        raise RuntimeError(f"Feature matrix is too small for PCA: shape={features_z.shape}")

    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(features_z)
    explained_variance_ratio = np.asarray(model.explained_variance_ratio_, dtype=float)
    eigenvalues = np.asarray(model.explained_variance_, dtype=float)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    denominator = float(np.sum(np.square(eigenvalues)))
    participation_ratio = (
        float(np.square(np.sum(eigenvalues)) / denominator)
        if denominator > 0
        else np.nan
    )

    def pc_threshold(threshold: float) -> int:
        return int(np.searchsorted(cumulative_explained_variance, threshold, side="left") + 1)

    return {
        "model": model,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance": cumulative_explained_variance,
        "participation_ratio": participation_ratio,
        "pc80": pc_threshold(0.80),
        "pc90": pc_threshold(0.90),
    }


# =============================================================================
# Output Writing
# =============================================================================


def save_plots(pca_result: dict, output_dir: Path, file_prefix: str) -> tuple[Path, Path]:
    cumulative = pca_result["cumulative_explained_variance"]
    explained = pca_result["explained_variance_ratio"]

    cumulative_path = output_dir / f"{file_prefix}_cumulative_explained_variance.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, cumulative.size + 1), cumulative, marker="o", linewidth=1.5)
    ax.axhline(0.80, color="tab:orange", linestyle="--", linewidth=1.0, label="80%")
    ax.axhline(0.90, color="tab:green", linestyle="--", linewidth=1.0, label="90%")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.set_title("Cumulative Explained Variance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(cumulative_path, dpi=300)
    plt.close(fig)

    bar_path = output_dir / f"{file_prefix}_first10_explained_variance.png"
    n_plot = min(10, explained.size)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(1, n_plot + 1), explained[:n_plot])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance")
    ax.set_title("Explained Variance: First 10 PCs")
    ax.set_xticks(np.arange(1, n_plot + 1))
    fig.tight_layout()
    fig.savefig(bar_path, dpi=300)
    plt.close(fig)

    return cumulative_path, bar_path


def save_summary_csv(summary_path: Path, summary_row: dict) -> None:
    columns = [
        "session_id",
        "date",
        "clock_hour",
        "feature_type",
        "bin_size_sec",
        "normalization_method",
        "n_bins",
        "n_channels_used",
        "mean_activity",
        "median_activity",
        "pc80",
        "pc90",
        "participation_ratio",
        "ev_pc1",
        "ev_pc2",
        "ev_pc3",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: summary_row.get(column, "") for column in columns})


# =============================================================================
# Pipeline Entry Point
# =============================================================================


def run_pipeline(config: Config) -> Path:
    total_timing = TimingLogger()

    rec_path = prompt_for_data_path(config.data_path)
    input_paths = [rec_path]
    data_files = total_timing.step(
        "collect_data_files",
        PIPELINE.collect_data_files_from_inputs,
        input_paths,
        PIPELINE.RECORDING_METHOD,
    )
    data_file = choose_one_data_file(data_files)
    session_details = parse_session_details(data_file)
    shank_config = build_one_shank_config(rec_path=rec_path, data_file=data_file)
    shank_id = str(shank_config["shank_id"])
    output_dir = make_output_dir(config, session_details["session_id"], shank_id)
    file_prefix = f"{session_details['session_id']}_sh{shank_id}_{FEATURE_TYPE}"

    log_status(f"Output directory: {output_dir}")
    log_status("Loading exactly one hour from the selected recording.")
    recording, resolved_channel_ids, build_elapsed_sec, build_step_timings = total_timing.step(
        "loading",
        PIPELINE.create_shank_recording,
        data_files=[data_file],
        recording_method=PIPELINE.RECORDING_METHOD,
        electrode_df=shank_config["electrode_df"],
        first_file_max_duration_s=config.one_hour_seconds,
    )
    for step_name, elapsed_sec in build_step_timings.items():
        total_timing.timings[f"loading_detail_{step_name}"] = float(elapsed_sec)

    actual_duration_sec = float(recording.get_num_frames() / recording.get_sampling_frequency())
    log_status(
        f"Loaded recording shape: {recording.get_num_frames()} frames x "
        f"{recording.get_num_channels()} channels ({actual_duration_sec / 60.0:.2f} min)."
    )

    rec_preprocessed, artifact_metadata = apply_sorting_preprocessing(recording, total_timing)

    features = total_timing.step(
        "feature_extraction",
        compute_spike_band_rms_features,
        rec_preprocessed,
        float(config.bin_size_seconds),
    )
    mean_activity = float(np.nanmean(features))
    median_activity = float(np.nanmedian(features))
    features_z, valid_channels, channel_means, channel_stds = zscore_features_for_pca(features)
    n_removed_channels = int(np.count_nonzero(~valid_channels))
    if n_removed_channels:
        log_status(f"Removed {n_removed_channels} zero-std/invalid channel(s) before PCA.")

    pca_result = total_timing.step("PCA", run_pca, features_z)
    total_timing.finish_total()

    feature_path = output_dir / f"{file_prefix}_X_hour.npy"
    np.save(feature_path, features)
    log_status(f"Saved feature matrix: {feature_path} with shape {features.shape}")

    feature_csv_path = output_dir / f"{file_prefix}_X_hour.csv"
    channel_ids = [str(ch) for ch in rec_preprocessed.get_channel_ids()]
    pd.DataFrame(features, columns=[f"ch_{channel_id}" for channel_id in channel_ids]).to_csv(
        feature_csv_path,
        index=False,
    )
    log_status(f"Saved feature matrix CSV: {feature_csv_path}")

    cumulative_plot_path, bar_plot_path = save_plots(pca_result, output_dir, file_prefix)
    log_status(f"Saved cumulative explained variance plot: {cumulative_plot_path}")
    log_status(f"Saved first-10-PC explained variance plot: {bar_plot_path}")

    explained = pca_result["explained_variance_ratio"]
    summary_row = {
        "session_id": session_details["session_id"],
        "date": session_details["date"],
        "clock_hour": session_details["clock_hour"],
        "feature_type": config.feature_type,
        "bin_size_sec": float(config.bin_size_seconds),
        "normalization_method": config.normalization_method,
        "n_bins": int(features.shape[0]),
        "n_channels_used": int(features_z.shape[1]),
        "mean_activity": mean_activity,
        "median_activity": median_activity,
        "pc80": int(pca_result["pc80"]),
        "pc90": int(pca_result["pc90"]),
        "participation_ratio": float(pca_result["participation_ratio"]),
        "ev_pc1": float(explained[0]) if explained.size >= 1 else np.nan,
        "ev_pc2": float(explained[1]) if explained.size >= 2 else np.nan,
        "ev_pc3": float(explained[2]) if explained.size >= 3 else np.nan,
    }
    summary_path = output_dir / f"{file_prefix}_summary.csv"
    save_summary_csv(summary_path, summary_row)
    log_status(f"Saved one-row summary CSV: {summary_path}")

    metadata_path = output_dir / f"{file_prefix}_metadata.json"
    metadata = {
        "input_file": str(data_file),
        "output_dir": str(output_dir),
        "recording_method": str(PIPELINE.RECORDING_METHOD),
        "probe_file": str(PIPELINE.PROBE_FILE),
        "impedance_file": str(PIPELINE.IMPEDANCE_FILE),
        "shank_id": shank_id,
        "resolved_channel_ids": [str(ch) for ch in resolved_channel_ids],
        "channel_ids": channel_ids,
        "valid_channel_mask_for_pca": valid_channels.astype(bool).tolist(),
        "channel_means": channel_means.astype(float).tolist(),
        "channel_stds": channel_stds.astype(float).tolist(),
        "preprocessing": {
            "steps": ["common_reference", "bandpass_filter", "remove_big_noise_artifacts"],
            "common_reference": {"operator": "median", "reference": "global"},
            "bandpass_filter": {"freq_min": 300, "freq_max": 6000, "dtype": "float32"},
            "remove_big_noise_artifacts": artifact_metadata,
        },
        "summary": summary_row,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log_status(f"Saved metadata JSON: {metadata_path}")

    log_path = output_dir / f"{file_prefix}_timing.log"
    total_timing.save(
        log_path,
        extra_payload={
            "input_file": str(data_file),
            "output_dir": str(output_dir),
            "summary_csv": str(summary_path),
        },
    )
    log_status(f"Saved timing log: {log_path}")

    log_status(
        f"Done. PC80={summary_row['pc80']}, PC90={summary_row['pc90']}, "
        f"D_PR={summary_row['participation_ratio']:.3f}"
    )
    return output_dir


def main() -> None:
    config = Config()
    output_dir = run_pipeline(config)
    log_status(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
