from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path
import re
from statistics import mean

import numpy as np
from rec2nwb import EphysToNWBConverter


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_window_label(failure_summary_path: Path, failed_summary: dict) -> str:
    window_label = failed_summary.get("window_label")
    if window_label:
        return str(window_label)
    return failure_summary_path.parent.name


def infer_input_file(failed_summary: dict, input_file_override: str) -> str:
    if input_file_override:
        return str(Path(input_file_override))
    input_sources = failed_summary.get("input_sources", [])
    if input_sources:
        return str(input_sources[0])
    raise KeyError(
        "Could not determine input file. Please provide the .rec path when prompted."
    )


def infer_shank_id(failure_summary_path: Path, failed_summary: dict) -> str:
    shank_id = failed_summary.get("shank_id")
    if shank_id is not None:
        return str(shank_id)

    shank_folder_name = failure_summary_path.parent.parent.name
    match = re.search(r"_sh(\d+)$", shank_folder_name)
    if match:
        return match.group(1)

    raise KeyError(
        "Could not determine shank_id from the JSON or folder path."
    )


def find_session_root_from_failure(failure_summary_path: Path) -> Path:
    return failure_summary_path.parent.parent.parent


def find_successful_sorting_summaries(
    session_root: Path,
    window_label: str,
    failed_shank_id: str,
    input_file: str,
) -> list[Path]:
    matches = []
    for shank_dir in sorted(session_root.glob("*_sh*")):
        if shank_dir.name.endswith(f"_sh{failed_shank_id}"):
            continue
        candidate = shank_dir / window_label / "sorting_summary.json"
        if not candidate.exists():
            continue
        try:
            summary = load_json(candidate)
        except Exception:
            continue
        if summary.get("input_sources") == [input_file]:
            matches.append(candidate)
    return matches


def summarize_numeric_list(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def compute_chunked_signal_stats(
    data_file: Path,
    channel_ids: list[str],
    recording_method: str = "spikegadget",
    chunk_duration_sec: float = 30.0,
    sample_duration_sec: float = 300.0,
) -> dict:
    converter = EphysToNWBConverter(recording_method=recording_method)
    recording = converter._get_recording(data_file)
    channel_ids = converter._normalize_channel_ids(recording, channel_ids)
    fs = float(recording.get_sampling_frequency())
    total_frames = int(recording.get_num_frames())
    frames_to_analyze = min(total_frames, int(round(sample_duration_sec * fs)))
    chunk_frames = max(1, int(round(chunk_duration_sec * fs)))

    sum_x = None
    sum_x2 = None
    max_abs = None
    sampled_abs_values = []
    sampled_zero_counts = None
    total_count = 0
    analyzed_chunks = 0

    for start_frame in range(0, frames_to_analyze, chunk_frames):
        end_frame = min(frames_to_analyze, start_frame + chunk_frames)
        traces = recording.get_traces(
            channel_ids=channel_ids,
            start_frame=start_frame,
            end_frame=end_frame,
        ).astype(np.float64, copy=False)

        if sum_x is None:
            n_channels = traces.shape[1]
            sum_x = np.zeros(n_channels, dtype=np.float64)
            sum_x2 = np.zeros(n_channels, dtype=np.float64)
            max_abs = np.zeros(n_channels, dtype=np.float64)
            sampled_zero_counts = np.zeros(n_channels, dtype=np.int64)

        sum_x += traces.sum(axis=0)
        sum_x2 += np.square(traces).sum(axis=0)
        max_abs = np.maximum(max_abs, np.max(np.abs(traces), axis=0))
        sampled_zero_counts += np.sum(traces == 0, axis=0)
        total_count += traces.shape[0]
        analyzed_chunks += 1

        # Keep a lightweight sample of absolute values for percentile estimates.
        stride = max(1, traces.shape[0] // 5000)
        sampled_abs_values.append(np.abs(traces[::stride]))

    del recording
    gc.collect()

    if total_count == 0:
        return {"loadable": False}

    mean_x = sum_x / total_count
    variance = np.maximum((sum_x2 / total_count) - np.square(mean_x), 0.0)
    std_x = np.sqrt(variance)
    sampled_abs = np.concatenate(sampled_abs_values, axis=0) if sampled_abs_values else np.empty((0, len(channel_ids)))

    return {
        "loadable": True,
        "recording_method": recording_method,
        "sampling_frequency": fs,
        "total_frames_in_file": total_frames,
        "frames_analyzed": int(total_count),
        "seconds_analyzed": float(total_count / fs),
        "chunk_duration_sec": float(chunk_duration_sec),
        "analyzed_chunks": int(analyzed_chunks),
        "channel_ids": [str(ch) for ch in channel_ids],
        "per_channel_mean": mean_x.tolist(),
        "per_channel_std": std_x.tolist(),
        "per_channel_max_abs": max_abs.tolist(),
        "per_channel_zero_fraction": (sampled_zero_counts / total_count).tolist(),
        "per_channel_p99_abs_sampled": (
            np.percentile(sampled_abs, 99, axis=0).tolist() if sampled_abs.size else []
        ),
    }


def summarize_signal_stats(signal_stats: dict) -> dict:
    if not signal_stats.get("loadable"):
        return {"loadable": False}
    return {
        "frames_analyzed": signal_stats["frames_analyzed"],
        "seconds_analyzed": signal_stats["seconds_analyzed"],
        "std_summary": summarize_numeric_list(signal_stats["per_channel_std"]),
        "max_abs_summary": summarize_numeric_list(signal_stats["per_channel_max_abs"]),
        "p99_abs_summary": summarize_numeric_list(signal_stats["per_channel_p99_abs_sampled"]),
        "zero_fraction_summary": summarize_numeric_list(signal_stats["per_channel_zero_fraction"]),
    }


def load_property_array(preprocessed_folder: Path, property_name: str):
    candidates = sorted((preprocessed_folder / "properties").glob(f"{property_name}.*"))
    if not candidates:
        return None
    path = candidates[0]
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def load_preprocessed_property_summary(preprocessed_folder: Path) -> dict:
    property_summaries = {}
    properties_dir = preprocessed_folder / "properties"
    if not properties_dir.exists():
        return property_summaries

    for path in sorted(properties_dir.iterdir()):
        property_name = path.stem
        if property_name in property_summaries:
            continue
        arr = load_property_array(preprocessed_folder, property_name)
        if arr is None:
            property_summaries[property_name] = {"loadable": False}
            continue

        summary = {
            "loadable": True,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        if np.issubdtype(arr.dtype, np.number):
            summary["numeric_summary"] = summarize_numeric_list(np.ravel(arr).tolist())
        else:
            preview = [str(x) for x in np.ravel(arr)[:12].tolist()]
            summary["preview"] = preview
            summary["unique_count"] = int(len({str(x) for x in np.ravel(arr).tolist()}))
        property_summaries[property_name] = summary

    return property_summaries


def compare_lists(left: list, right: list) -> dict:
    return {
        "left_count": int(len(left)),
        "right_count": int(len(right)),
        "same": left == right,
        "left_only": [str(x) for x in left if x not in right],
        "right_only": [str(x) for x in right if x not in left],
    }


def build_comparison(
    failed_summary: dict,
    failed_preprocessing: dict,
    failed_property_summary: dict,
    success_summary: dict,
    success_preprocessing: dict,
    success_property_summary: dict,
    failed_signal_stats: dict,
    success_signal_stats: dict,
    success_path: Path,
) -> dict:
    failed_channels = failed_summary.get("channel_ids", [])
    success_channels = success_summary.get("channel_ids", [])
    failed_resolved = failed_preprocessing.get("resolved_channel_ids", [])
    success_resolved = success_preprocessing.get("resolved_channel_ids", [])

    comparison = {
        "success_sorting_summary_path": str(success_path),
        "success_shank_id": str(success_summary.get("shank_id")),
        "shared_input_file": failed_summary.get("input_sources", [None])[0],
        "window_label": failed_summary.get("window_label"),
        "num_frames_match": failed_summary.get("num_frames") == success_summary.get("num_frames"),
        "duration_seconds_match": failed_summary.get("duration_seconds") == success_summary.get("duration_seconds"),
        "num_channels_match": failed_summary.get("num_channels") == success_summary.get("num_channels"),
        "failed_num_units": failed_summary.get("num_units"),
        "success_num_units": success_summary.get("num_units"),
        "channel_ids_comparison": compare_lists(failed_channels, success_channels),
        "resolved_channel_ids_comparison": compare_lists(failed_resolved, success_resolved),
        "failed_property_keys": failed_summary.get("recording_property_keys", []),
        "success_property_keys": list(success_property_summary.keys()),
        "failed_preprocessed_properties": failed_property_summary,
        "success_preprocessed_properties": success_property_summary,
        "failed_raw_signal_summary": summarize_signal_stats(failed_signal_stats),
        "success_raw_signal_summary": summarize_signal_stats(success_signal_stats),
    }
    return comparison


def build_overview(failed_summary: dict, success_summaries: list[dict]) -> dict:
    success_unit_counts = [int(summary.get("num_units", 0)) for summary in success_summaries]
    return {
        "failed_shank_id": str(failed_summary.get("shank_id")),
        "input_file": failed_summary.get("input_sources", [None])[0],
        "window_label": failed_summary.get("window_label"),
        "failed_exception_type": failed_summary.get("exception_type"),
        "failed_exception_message": failed_summary.get("exception_message"),
        "failed_num_frames": failed_summary.get("num_frames"),
        "failed_num_channels": failed_summary.get("num_channels"),
        "failed_channel_ids": failed_summary.get("channel_ids"),
        "num_successful_comparators": int(len(success_summaries)),
        "successful_num_units": success_unit_counts,
        "successful_num_units_mean": float(mean(success_unit_counts)) if success_unit_counts else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare one failing shank/file run against successful shanks from the same recording."
    )
    parser.add_argument(
        "failure_summary",
        nargs="?",
        default=r"W:\20260224_001822_20260224_051826\20260224_001822_20260224_051826_sh22\0224_01\sorting_failure_summary.json",
        help="Path to sorting_failure_summary.json",
    )
    parser.add_argument(
        "--max-success",
        type=int,
        default=5,
        help="Maximum number of successful shanks to compare against",
    )
    parser.add_argument(
        "--sample-duration-sec",
        type=float,
        default=300.0,
        help="How many seconds of the .rec file to analyze for raw signal stats",
    )
    args = parser.parse_args()

    failure_summary_input = input(
        "Enter path to sorting_failure_summary.json "
        f"(press Enter to use default: {args.failure_summary}): "
    ).strip().strip('"').strip("'")
    failure_summary_path = Path(failure_summary_input) if failure_summary_input else Path(args.failure_summary)

    input_file_override = input(
        "Enter input .rec file path to match against successful shanks "
        "(press Enter to use the path recorded in the failure summary): "
    ).strip().strip('"').strip("'")

    failed_summary = load_json(failure_summary_path)
    if not isinstance(failed_summary, dict):
        raise ValueError(f"Expected a JSON object in {failure_summary_path}")

    window_label = infer_window_label(failure_summary_path, failed_summary)
    shank_id = infer_shank_id(failure_summary_path, failed_summary)
    input_file_path = Path(infer_input_file(failed_summary, input_file_override))
    failed_summary["window_label"] = window_label
    failed_summary["shank_id"] = shank_id
    failed_summary["input_sources"] = [str(input_file_path)]

    failed_preprocessing = load_json(failure_summary_path.parent / "preprocessing_metadata.json")
    failed_property_summary = load_preprocessed_property_summary(
        failure_summary_path.parent / "preprocessed_recording"
    )

    session_root = find_session_root_from_failure(failure_summary_path)
    success_paths = find_successful_sorting_summaries(
        session_root=session_root,
        window_label=window_label,
        failed_shank_id=shank_id,
        input_file=str(input_file_path),
    )

    selected_success_paths = success_paths[: args.max_success]
    success_summaries = [load_json(path) for path in selected_success_paths]
    failed_signal_stats = compute_chunked_signal_stats(
        data_file=input_file_path,
        channel_ids=failed_preprocessing.get("resolved_channel_ids", []),
        sample_duration_sec=args.sample_duration_sec,
    )

    comparisons = []
    for path, success_summary in zip(selected_success_paths, success_summaries):
        success_preprocessing = load_json(path.parent / "preprocessing_metadata.json")
        success_property_summary = load_preprocessed_property_summary(path.parent / "preprocessed_recording")
        success_signal_stats = compute_chunked_signal_stats(
            data_file=input_file_path,
            channel_ids=success_preprocessing.get("resolved_channel_ids", []),
            sample_duration_sec=args.sample_duration_sec,
        )
        comparisons.append(
            build_comparison(
                failed_summary=failed_summary,
                failed_preprocessing=failed_preprocessing,
                failed_property_summary=failed_property_summary,
                success_summary=success_summary,
                success_preprocessing=success_preprocessing,
                success_property_summary=success_property_summary,
                failed_signal_stats=failed_signal_stats,
                success_signal_stats=success_signal_stats,
                success_path=path,
            )
        )

    report = {
        "overview": build_overview(failed_summary, success_summaries),
        "failed_raw_signal_summary": summarize_signal_stats(failed_signal_stats),
        "comparisons": comparisons,
    }

    report_path = failure_summary_path.parent / "failed_shank_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved comparison report to: {report_path}")
    print(f"Compared failing shank {failed_summary['shank_id']} against {len(comparisons)} successful shank(s).")


if __name__ == "__main__":
    main()
