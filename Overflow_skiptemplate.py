# Input files should be from a same day

from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path
import re
import shutil
from time import perf_counter
import traceback

import matplotlib.pyplot as plt
import numpy as np
import probeinterface as pi
import spikeinterface.full as si
import spikeinterface.preprocessing as preproc
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw

from File_Organize import normalize_day_code
from rec2nwb import (
    EphysToNWBConverter,
    _build_electrode_df,
    _collect_data_files,
    _format_elapsed_time,
    _print_recording_channel_id_preview,
    _print_resolved_shank_channel_id_preview,
    _print_shank_namespace_preview,
    load_bad_ch,
    read_impedance_file,
)


RECORDING_METHOD = "spikegadget"
PROBE_FILE = Path(r"E:\Curtis\spikeinterface\LSNET_probe.json")
IMPEDANCE_FILE = Path(r"E:\Curtis\spikeinterface\imp_09222025_LSNET18.txt")
OUTPUT_ROOT = Path(r"S:\\")

SAVE_PREPROCESSED_RECORDING = True
OVERWRITE_SORTER_OUTPUT = True

MS_BEFORE = 1.0
MS_AFTER = 2.0
MAX_SPIKES_PER_UNIT = 500
MATERIALIZE_SHANK_RECORDING_AS_NUMPY = True

BIG_NOISE_NEGATIVE_THRESHOLD = -2500.0
BIG_NOISE_POSITIVE_THRESHOLD = 1000.0
BIG_NOISE_MARGIN_SAMPLES = 25
BIG_NOISE_BLANK_ALL_CHANNELS = True

MS5_SORTER_PARAMS = {
    "scheme": "2",
    "detect_sign": 0,
    "detect_threshold": 5.5,
    "npca_per_channel": 3,
    "npca_per_subdivision": 10,
    "snippet_mask_radius": 250,
    "scheme2_phase1_detect_channel_radius": 200,
    "scheme2_detect_channel_radius": 50,
    "scheme2_max_num_snippets_per_training_batch": 200,
    "scheme2_training_duration_sec": 300,
    "scheme2_training_recording_sampling_mode": "uniform",
    "whiten": True,
    "filter": False,
    "n_jobs": 1,
    "chunk_duration": "1s",
    "progress_bar": True,
}

MS5_OVERFLOW_FALLBACK_PARAMS = {
    "whiten": True,
    "skip_alignment": True,
}

def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def cleanup_preprocessed_recording_folder(output_folder: Path) -> None:
    preprocessed_folder = output_folder / "preprocessed_recording"
    if not preprocessed_folder.exists():
        return
    shutil.rmtree(preprocessed_folder, ignore_errors=True)
    print(f"Removed temporary preprocessed recording folder: {preprocessed_folder}")


def format_elapsed_time(elapsed_sec: float) -> str:
    if elapsed_sec < 60:
        return f"{elapsed_sec:.2f} seconds"
    if elapsed_sec < 3600:
        return f"{elapsed_sec / 60.0:.2f} minutes"
    return f"{elapsed_sec / 3600.0:.2f} hours"


def _timer_line(step_name: str, elapsed_sec: float) -> str:
    return f"[timer] {step_name}: {format_elapsed_time(elapsed_sec)}"


def print_timer(step_name: str, elapsed_sec: float) -> None:
    print(_timer_line(step_name, elapsed_sec))


def print_step_start(step_name: str) -> None:
    print(f"[timer] starting {step_name}...")


def time_step(step_name: str, callback, *args, **kwargs):
    print_step_start(step_name)
    start_time = perf_counter()
    result = callback(*args, **kwargs)
    elapsed_sec = perf_counter() - start_time
    print_timer(step_name, elapsed_sec)
    return result, elapsed_sec


def _safe_int_label(value) -> str:
    try:
        if value is None:
            return "unknown"
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            if np.isnan(value):
                return "unknown"
            return str(int(value))
        return str(int(value))
    except Exception:
        return str(value)


def is_overflow_sorting_error(exc: Exception) -> bool:
    exc_text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "python int too large to convert to c long" in exc_text
        or "overflowerror" in exc_text
        or "overflow" in exc_text
    )


def is_template_alignment_overflow_error(exc: Exception) -> bool:
    exc_text = f"{type(exc).__name__}: {exc}".lower()
    return is_overflow_sorting_error(exc) and "align_templates" in exc_text


def parse_input_paths(raw_value: str) -> list[Path]:
    parts = [part.strip().strip('"').strip("'") for part in re.split(r"[;\n]+", raw_value) if part.strip()]
    if not parts:
        raise ValueError("No recording path provided.")

    input_paths = [Path(part) for part in parts]
    missing_paths = [str(path) for path in input_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Recording path(s) not found: {missing_paths}")

    return input_paths


def find_overflow_report_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(folder.glob("overflow_error_report_*.json"))


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_overflow_report_folder(input_paths: list[Path]) -> bool:
    return (
        len(input_paths) == 1
        and input_paths[0].is_dir()
        and len(find_overflow_report_files(input_paths[0])) > 0
    )


def sanitize_session_description(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return sanitized or "session"


def confirm_terminal_action(title: str, lines: list[str]) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for line in lines:
        print(line)
    response = input("\nProceed? [y/N]: ").strip().lower()
    if response not in {"y", "yes"}:
        raise RuntimeError("Operation cancelled by user.")


def build_retry_tasks_from_overflow_report(report_path: Path) -> list[dict]:
    report = load_json(report_path)
    failures = report.get("failures", [])
    session_description = sanitize_session_description(
        str(report.get("session_description") or report_path.stem.replace("overflow_error_report_", "", 1))
    )

    tasks = []
    for failure in failures:
        input_file = Path(str(failure["input_file"]))
        if not input_file.exists():
            raise FileNotFoundError(f"Input file from overflow report does not exist: {input_file}")

        tasks.append(
            {
                "report_path": report_path,
                "session_description": session_description,
                "shank_id": str(failure["shank_id"]),
                "input_file": input_file,
                "recording_label": str(failure.get("recording_label", build_recording_label(input_file))),
            }
        )

    return tasks


def collect_data_files_from_inputs(input_paths: list[Path], recording_method: str) -> list[Path]:
    data_files = []
    seen_paths = set()

    for input_path in input_paths:
        for data_file in _collect_data_files(input_path, recording_method):
            data_file_resolved = data_file.resolve()
            if data_file_resolved in seen_paths:
                continue
            seen_paths.add(data_file_resolved)
            data_files.append(data_file)

    if not data_files:
        raise FileNotFoundError("No data files found in the provided input paths.")

    return sorted(data_files)


def build_batch_output_folder_for_inputs(input_paths: list[Path], data_files: list[Path]) -> Path:
    day_codes = sorted({extract_day_code_from_data_file(data_file) for data_file in data_files})
    if not day_codes:
        raise ValueError("Unable to determine day code from the selected data files.")
    if len(day_codes) != 1:
        raise ValueError(
            f"Expected data files from a single day, but found multiple day codes: {day_codes}"
        )

    output_root = OUTPUT_ROOT
    if not output_root.is_absolute():
        raise ValueError(f"OUTPUT_ROOT must be an absolute path, got: {OUTPUT_ROOT}")
    return output_root / f"{day_codes[0]}_Sorting"


def get_overflow_report_path(batch_output_folder: Path, session_description: str) -> Path:
    return batch_output_folder / f"overflow_error_report_{session_description}.json"


def get_overflow_skiptemplate_summary_path(batch_output_folder: Path) -> Path:
    return batch_output_folder / "overflow_skiptemplate_summary.json"


def get_sorting_unit_property(sorting, unit_id, property_name: str):
    try:
        return sorting.get_unit_property(unit_id, property_name)
    except Exception:
        pass
    try:
        values = sorting.get_property(property_name)
        unit_ids = list(sorting.get_unit_ids())
        if len(values) == len(unit_ids):
            return values[unit_ids.index(unit_id)]
    except Exception:
        pass
    return None


def build_unit_export_name(
    analyzer,
    unit_id,
    channel_idx: int | None = None,
    prefix: str = "unit",
) -> str:
    sorting = analyzer.sorting
    shank_id = get_sorting_unit_property(sorting, unit_id, "shank_id")

    channel_id = get_representative_channel_id(analyzer, unit_id, channel_idx=channel_idx)

    return f"{prefix}_shank{_safe_int_label(shank_id)}_ch{_safe_int_label(channel_id)}_{int(unit_id)}.png"


def get_representative_channel_index(analyzer, unit_id) -> int | None:
    try:
        waveforms_u = analyzer.get_extension("waveforms").get_waveforms_one_unit(unit_id)
    except Exception:
        return None

    if waveforms_u is None or getattr(waveforms_u, "size", 0) == 0:
        return None

    mean_waveform = waveforms_u.mean(axis=0)
    return int(np.argmax(np.max(np.abs(mean_waveform), axis=0)))


def get_representative_channel_id(analyzer, unit_id, channel_idx: int | None = None):
    try:
        device_channel_index_prop = None
        if "device_channel_index" in set(analyzer.recording.get_property_keys()):
            device_channel_index_prop = analyzer.recording.get_property("device_channel_index")
        if device_channel_index_prop is None:
            return "unknown"

        channel_ids_by_unit = si.get_template_extremum_channel(
            analyzer,
            peak_sign="neg",
            outputs="id",
        )
        if unit_id in channel_ids_by_unit:
            channel_id = channel_ids_by_unit[unit_id]
            channel_ids = list(analyzer.channel_ids)
            global_index = channel_ids.index(channel_id)
            if global_index < len(device_channel_index_prop):
                return device_channel_index_prop[global_index]
    except Exception:
        pass

    if channel_idx is not None:
        try:
            device_channel_index_prop = analyzer.recording.get_property("device_channel_index")
            if channel_idx < len(device_channel_index_prop):
                return device_channel_index_prop[channel_idx]
        except Exception:
            pass

    return "unknown"


def save_unit_channel_mapping_report(analyzer, output_folder: Path) -> Path:
    report_path = output_folder / "unit_channel_mapping_report.json"
    channel_ids = list(analyzer.channel_ids)
    channel_id_to_index = {str(channel_id): index for index, channel_id in enumerate(channel_ids)}

    try:
        extremum_channel_ids = si.get_template_extremum_channel(
            analyzer,
            peak_sign="neg",
            outputs="id",
        )
    except Exception:
        extremum_channel_ids = {}

    rows = []
    device_channel_index_prop = None
    if "device_channel_index" in set(analyzer.recording.get_property_keys()):
        try:
            device_channel_index_prop = analyzer.recording.get_property("device_channel_index")
        except Exception:
            device_channel_index_prop = None

    for unit_id in analyzer.sorting.get_unit_ids():
        unit_id_int = int(unit_id)
        waveform_local_channel_index = get_representative_channel_index(analyzer, unit_id)
        extremum_channel_id = extremum_channel_ids.get(
            unit_id,
            get_representative_channel_id(analyzer, unit_id, channel_idx=waveform_local_channel_index),
        )
        global_channel_index = channel_id_to_index.get(str(extremum_channel_id))
        device_channel_index = None
        if (
            device_channel_index_prop is not None
            and global_channel_index is not None
            and global_channel_index < len(device_channel_index_prop)
        ):
            try:
                device_channel_index = int(device_channel_index_prop[global_channel_index])
            except Exception:
                device_channel_index = device_channel_index_prop[global_channel_index]

        rows.append(
            {
                "unit_id": unit_id_int,
                "shank_id": get_sorting_unit_property(analyzer.sorting, unit_id, "shank_id"),
                "extremum_channel_id": str(extremum_channel_id),
                "global_channel_index": global_channel_index,
                "waveform_local_channel_index": waveform_local_channel_index,
                "device_channel_index_property": device_channel_index,
            }
        )

    save_json(
        report_path,
        {
            "output_folder": str(output_folder),
            "channel_ids": [str(channel_id) for channel_id in channel_ids],
            "units": rows,
        },
    )
    return report_path


def print_channel_information(recording) -> None:
    channel_ids = list(recording.get_channel_ids())
    property_keys = list(recording.get_property_keys())

    print("\n--- Loaded channel information ---")
    print(f"Number of channels: {len(channel_ids)}")
    print(f"Channel IDs: {channel_ids}")
    print(f"Channel property keys: {property_keys}")

    for prop_key in property_keys:
        try:
            prop_values = recording.get_property(prop_key)
        except Exception as exc:
            print(f"{prop_key}: <failed to load: {exc}>")
            continue
        print(f"{prop_key}: {prop_values}")


def concatenate_recordings_compat(recordings: list):
    if len(recordings) == 1:
        return recordings[0]

    if hasattr(si, "concatenate_recordings"):
        return si.concatenate_recordings(recordings)
    if hasattr(si, "append_recordings"):
        return si.append_recordings(recordings)
    if hasattr(si, "ConcatenateSegmentRecording"):
        return si.ConcatenateSegmentRecording(recordings)

    raise RuntimeError(
        "Unable to concatenate multiple recordings with this SpikeInterface install. "
        "Expected concatenate_recordings or append_recordings."
    )


def build_shank_probe(electrode_df):
    try:
        probe = pi.Probe(ndim=2, si_units="um")
        positions = electrode_df[["x", "y"]].to_numpy(dtype=float)
        probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 7.5})
        probe.set_device_channel_indices(
            electrode_df["device_channel_index"].astype(int).to_numpy()
        )
        probe.set_shank_ids(electrode_df["shank_id"].astype(str).to_numpy())
        return probe
    except Exception:
        return None


def attach_shank_metadata(recording, electrode_df):
    channel_labels = [str(ch) for ch in electrode_df["device_channel_index"].tolist()]

    try:
        recording = recording.rename_channels(channel_labels)
    except Exception:
        pass

    try:
        recording.set_property(
            "location",
            electrode_df[["x", "y"]].to_numpy(dtype=float),
        )
    except Exception:
        pass
    try:
        recording.set_property(
            "group",
            electrode_df["shank_id"].astype(str).to_numpy(),
        )
    except Exception:
        pass
    try:
        recording.set_property(
            "shank_id",
            electrode_df["shank_id"].astype(str).to_numpy(),
        )
    except Exception:
        pass
    try:
        recording.set_property(
            "device_channel_index",
            electrode_df["device_channel_index"].astype(int).to_numpy(),
        )
    except Exception:
        pass
    try:
        recording.set_property(
            "impedance_ohm",
            electrode_df["impedance_ohm"].to_numpy(dtype=float),
        )
    except Exception:
        pass

    shank_probe = build_shank_probe(electrode_df)
    if shank_probe is not None:
        try:
            recording = recording.set_probe(shank_probe, in_place=False)
        except Exception:
            pass

    return recording


def build_numpy_recording_from_traces(traces_list: list[np.ndarray], sampling_frequency: float, electrode_df, gains, offsets):
    if not traces_list:
        raise ValueError("No trace arrays were collected for this shank.")

    if len(traces_list) == 1:
        traces = traces_list[0]
    else:
        traces = np.concatenate(traces_list, axis=0)

    recording = si.NumpyRecording(
        traces_list=[traces],
        sampling_frequency=sampling_frequency,
        channel_ids=[str(ch) for ch in electrode_df["device_channel_index"].tolist()],
    )

    try:
        if gains is not None:
            recording.set_channel_gains(np.asarray(gains))
    except Exception:
        pass
    try:
        if offsets is not None:
            recording.set_channel_offsets(np.asarray(offsets))
    except Exception:
        pass

    return attach_shank_metadata(recording, electrode_df)


def materialize_recording_as_numpy(recording):
    traces = recording.get_traces()
    channel_ids = [str(ch) for ch in recording.get_channel_ids()]
    sampling_frequency = float(recording.get_sampling_frequency())

    numpy_recording = si.NumpyRecording(
        traces_list=[traces],
        sampling_frequency=sampling_frequency,
        channel_ids=channel_ids,
    )

    try:
        gains = recording.get_channel_gains()
        if gains is not None:
            numpy_recording.set_channel_gains(np.asarray(gains))
    except Exception:
        pass

    try:
        offsets = recording.get_channel_offsets()
        if offsets is not None:
            numpy_recording.set_channel_offsets(np.asarray(offsets))
    except Exception:
        pass

    for prop_key in recording.get_property_keys():
        try:
            numpy_recording.set_property(prop_key, recording.get_property(prop_key))
        except Exception:
            pass

    try:
        if hasattr(recording, "has_probe") and recording.has_probe():
            probe = recording.get_probe()
            numpy_recording = numpy_recording.set_probe(probe, in_place=False)
    except Exception:
        pass

    return numpy_recording


def remove_big_noise_artifacts(
    recording,
    negative_threshold: float = BIG_NOISE_NEGATIVE_THRESHOLD,
    positive_threshold: float = BIG_NOISE_POSITIVE_THRESHOLD,
    margin_samples: int = BIG_NOISE_MARGIN_SAMPLES,
    blank_all_channels: bool = BIG_NOISE_BLANK_ALL_CHANNELS,
):
    traces = np.asarray(recording.get_traces())
    if traces.ndim != 2:
        raise ValueError(f"Expected 2D traces array, got shape {traces.shape}")

    artifact_mask = (traces < negative_threshold) | (traces > positive_threshold)
    event_count = int(np.count_nonzero(artifact_mask))
    if event_count == 0:
        metadata = {
            "negative_threshold": float(negative_threshold),
            "positive_threshold": float(positive_threshold),
            "margin_samples": int(margin_samples),
            "blank_all_channels": bool(blank_all_channels),
            "num_threshold_crossings": 0,
            "num_masked_samples": 0,
            "num_masked_segments": 0,
        }
        return materialize_recording_as_numpy(recording), metadata

    if blank_all_channels:
        sample_hits = np.any(artifact_mask, axis=1)
        channel_hits = artifact_mask.any(axis=0)
        masked_channel_indices = np.flatnonzero(channel_hits)
    else:
        sample_hits = np.any(artifact_mask, axis=1)
        masked_channel_indices = None

    expanded_mask = np.zeros(sample_hits.shape[0], dtype=bool)
    hit_indices = np.flatnonzero(sample_hits)
    for hit_index in hit_indices:
        start = max(0, int(hit_index) - int(margin_samples))
        end = min(sample_hits.shape[0], int(hit_index) + int(margin_samples) + 1)
        expanded_mask[start:end] = True

    cleaned_traces = traces.copy()
    if blank_all_channels:
        cleaned_traces[expanded_mask, :] = 0
    else:
        cleaned_traces[artifact_mask] = 0
        for hit_index in hit_indices:
            start = max(0, int(hit_index) - int(margin_samples))
            end = min(sample_hits.shape[0], int(hit_index) + int(margin_samples) + 1)
            affected_channels = np.flatnonzero(artifact_mask[hit_index])
            if affected_channels.size > 0:
                cleaned_traces[start:end, affected_channels] = 0

    transitions = np.diff(np.r_[False, expanded_mask, False].astype(np.int8))
    segment_starts = np.flatnonzero(transitions == 1)
    segment_ends = np.flatnonzero(transitions == -1)
    artifact_segments = [
        {"start_frame": int(start), "end_frame": int(end)}
        for start, end in zip(segment_starts, segment_ends)
    ]

    cleaned_recording = si.NumpyRecording(
        traces_list=[cleaned_traces],
        sampling_frequency=float(recording.get_sampling_frequency()),
        channel_ids=[str(ch) for ch in recording.get_channel_ids()],
    )

    try:
        gains = recording.get_channel_gains()
        if gains is not None:
            cleaned_recording.set_channel_gains(np.asarray(gains))
    except Exception:
        pass

    try:
        offsets = recording.get_channel_offsets()
        if offsets is not None:
            cleaned_recording.set_channel_offsets(np.asarray(offsets))
    except Exception:
        pass

    for prop_key in recording.get_property_keys():
        try:
            cleaned_recording.set_property(prop_key, recording.get_property(prop_key))
        except Exception:
            pass

    try:
        if hasattr(recording, "has_probe") and recording.has_probe():
            cleaned_recording = cleaned_recording.set_probe(recording.get_probe(), in_place=False)
    except Exception:
        pass

    metadata = {
        "negative_threshold": float(negative_threshold),
        "positive_threshold": float(positive_threshold),
        "margin_samples": int(margin_samples),
        "blank_all_channels": bool(blank_all_channels),
        "num_threshold_crossings": int(event_count),
        "num_masked_samples": int(np.count_nonzero(expanded_mask)),
        "num_masked_segments": int(len(artifact_segments)),
        "masked_fraction": float(np.count_nonzero(expanded_mask) / traces.shape[0]),
        "artifact_segments_preview": artifact_segments[:50],
    }
    if blank_all_channels:
        metadata["masked_channel_indices"] = [int(idx) for idx in masked_channel_indices.tolist()]

    return cleaned_recording, metadata


def preprocess_recording(recording, window_metadata: dict):
    rec_cr = preproc.common_reference(recording, operator="median", reference="global")
    rec_filt = preproc.bandpass_filter(
        rec_cr,
        freq_min=300,
        freq_max=6000,
        dtype="float32",
    )
    rec_denoised, noise_metadata = remove_big_noise_artifacts(rec_filt)
    metadata = {
        "steps": ["common_reference", "bandpass_filter", "remove_big_noise_artifacts"],
        "params": {
            "common_reference": {"operator": "median", "reference": "global"},
            "bandpass_filter": {"freq_min": 300, "freq_max": 6000, "dtype": "float32"},
            "remove_big_noise_artifacts": noise_metadata,
            "window": window_metadata,
        },
    }
    return rec_denoised, metadata


def _build_ms5_scheme2_parameters(sorter_params: dict):
    import mountainsort5 as ms5

    return ms5.Scheme2SortingParameters(
        phase1_detect_channel_radius=sorter_params["scheme2_phase1_detect_channel_radius"],
        detect_channel_radius=sorter_params["scheme2_detect_channel_radius"],
        phase1_detect_threshold=sorter_params["detect_threshold"],
        phase1_detect_time_radius_msec=sorter_params.get("detect_time_radius_msec", 0.5),
        detect_time_radius_msec=sorter_params.get("detect_time_radius_msec", 0.5),
        phase1_npca_per_channel=sorter_params["npca_per_channel"],
        phase1_npca_per_subdivision=sorter_params["npca_per_subdivision"],
        detect_sign=sorter_params["detect_sign"],
        detect_threshold=sorter_params["detect_threshold"],
        snippet_T1=sorter_params.get("snippet_T1", 20),
        snippet_T2=sorter_params.get("snippet_T2", 20),
        snippet_mask_radius=sorter_params["snippet_mask_radius"],
        max_num_snippets_per_training_batch=sorter_params["scheme2_max_num_snippets_per_training_batch"],
        classifier_npca=None,
        training_duration_sec=sorter_params["scheme2_training_duration_sec"],
        training_recording_sampling_mode=sorter_params["scheme2_training_recording_sampling_mode"],
        classification_chunk_sec=sorter_params.get("classification_chunk_sec"),
    )


def _run_mountainsort5_skip_alignment_fallback(
    recording,
    sorter_params: dict,
):
    import mountainsort5 as ms5

    rec_ms5 = recording
    if sorter_params.get("whiten", True):
        rec_ms5 = preproc.whiten(rec_ms5, dtype="float32")

    scheme2_sorting_parameters = _build_ms5_scheme2_parameters(sorter_params)
    scheme2_module = importlib.import_module("mountainsort5.schemes.sorting_scheme2")
    original_sorting_scheme1 = scheme2_module.sorting_scheme1

    def sorting_scheme1_skip_alignment(*, recording, sorting_parameters):
        sorting_parameters.skip_alignment = True
        return original_sorting_scheme1(recording=recording, sorting_parameters=sorting_parameters)

    scheme2_module.sorting_scheme1 = sorting_scheme1_skip_alignment
    try:
        return ms5.sorting_scheme2(
            recording=rec_ms5,
            sorting_parameters=scheme2_sorting_parameters,
        )
    finally:
        scheme2_module.sorting_scheme1 = original_sorting_scheme1


def _build_sorting_failure_summary(
    *,
    output_folder: Path,
    sorter_run_folder: Path,
    input_sources: list[Path],
    shank_id: str,
    window_label: str,
    sorter_params: dict,
    recording,
    exc: Exception,
    traceback_text: str,
    sorter_name: str,
    fallback_attempted: bool,
    fallback_succeeded: bool,
):
    sampling_frequency = None
    num_frames = None
    duration_seconds = None
    num_channels = None
    channel_ids = []
    property_keys = []

    try:
        sampling_frequency = float(recording.get_sampling_frequency())
    except Exception:
        pass
    try:
        num_frames = int(recording.get_num_frames())
    except Exception:
        pass
    if sampling_frequency and num_frames is not None and sampling_frequency > 0:
        duration_seconds = float(num_frames / sampling_frequency)
    try:
        num_channels = int(recording.get_num_channels())
    except Exception:
        pass
    try:
        channel_ids = [str(ch) for ch in recording.get_channel_ids()]
    except Exception:
        pass
    try:
        property_keys = list(recording.get_property_keys())
    except Exception:
        pass

    return {
        "input_sources": [str(path) for path in input_sources],
        "output_folder": str(output_folder),
        "sorter_run_folder": str(sorter_run_folder),
        "shank_id": str(shank_id),
        "window_label": str(window_label),
        "sorter": sorter_name,
        "sorter_params": sorter_params,
        "sampling_frequency": sampling_frequency,
        "num_frames": num_frames,
        "duration_seconds": duration_seconds,
        "duration_hours": (duration_seconds / 3600.0) if duration_seconds is not None else None,
        "num_channels": num_channels,
        "channel_ids": channel_ids,
        "recording_property_keys": property_keys,
        "materialize_shank_recording_as_numpy": bool(MATERIALIZE_SHANK_RECORDING_AS_NUMPY),
        "save_preprocessed_recording": bool(SAVE_PREPROCESSED_RECORDING),
        "fallback_attempted": bool(fallback_attempted),
        "fallback_succeeded": bool(fallback_succeeded),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback_text,
    }


def build_shank_output_root(batch_output_folder: Path, session_description: str, shank_id: str) -> Path:
    return batch_output_folder / f"sh{shank_id}"


def build_hour_output_folder(batch_output_folder: Path, session_description: str, shank_id: str, hour_index: int) -> Path:
    return build_shank_output_root(batch_output_folder, session_description, shank_id) / str(hour_index)


def build_full_output_folder(batch_output_folder: Path, session_description: str, shank_id: str) -> Path:
    return build_shank_output_root(batch_output_folder, session_description, shank_id) / "full_recording"


def build_recording_output_folder(
    batch_output_folder: Path,
    session_description: str,
    shank_id: str,
    recording_label: str,
) -> Path:
    return build_shank_output_root(batch_output_folder, session_description, shank_id) / f"{recording_label}_sh{shank_id}"


def _remove_nwb_from_output_name(folder_name: str) -> str:
    cleaned = folder_name
    if cleaned.startswith("NWB_"):
        cleaned = cleaned[len("NWB_"):]
    cleaned = cleaned.replace("_NWB_", "_")
    if cleaned.endswith("_NWB"):
        cleaned = cleaned[:-len("_NWB")]
    if cleaned == "NWB":
        cleaned = "recording"
    return cleaned


def build_recording_label(data_file: Path) -> str:
    stem = data_file.stem
    match = re.search(r"(?P<date>\d{8})_(?P<time>\d{6})", stem)
    if match:
        date_value = normalize_day_code(match.group("date"))
        time_value = match.group("time")
        return f"{date_value}_{time_value[:2]}"

    match = re.search(r"(?P<date>\d{8}).*?(?P<hour>\d{2})", stem)
    if match:
        date_value = normalize_day_code(match.group("date"))
        hour_value = match.group("hour")
        return f"{date_value}_{hour_value}"

    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return sanitized or "recording"


def extract_day_code_from_data_file(data_file: Path) -> str:
    label = build_recording_label(data_file)
    day_code, _, _ = label.partition("_")
    if not day_code:
        raise ValueError(f"Unable to determine day code from data file: {data_file}")
    return day_code


def build_session_description(data_files: list[Path]) -> str:
    recording_labels = sorted(build_recording_label(data_file) for data_file in data_files)
    if not recording_labels:
        return "session"
    if len(recording_labels) == 1:
        return recording_labels[0]
    return f"{recording_labels[0]}_to_{recording_labels[-1]}"


def create_shank_recording(
    data_files: list[Path],
    recording_method: str,
    electrode_df,
    first_file_max_duration_s: float | None = None,
):
    build_start = perf_counter()
    step_timings: dict[str, float] = {}
    converter = EphysToNWBConverter(recording_method=recording_method)
    first_recording, step_timings["load_first_recording_elapsed_sec"] = time_step(
        "load_first_recording",
        converter._get_recording,
        data_files[0],
    )
    resolved_channel_ids = converter._resolve_shank_channel_ids(
        first_recording,
        electrode_df["device_channel_index"].tolist(),
    )
    resolved_channel_ids = converter._normalize_channel_ids(first_recording, resolved_channel_ids)
    sampling_frequency = float(first_recording.get_sampling_frequency())
    try:
        gains_all = first_recording.get_channel_gains()
    except Exception:
        gains_all = None
    try:
        offsets_all = first_recording.get_channel_offsets()
    except Exception:
        offsets_all = None
    channel_id_to_index = {ch: i for i, ch in enumerate(first_recording.get_channel_ids())}
    selected_indices = [channel_id_to_index[ch] for ch in resolved_channel_ids]
    gains = [gains_all[idx] for idx in selected_indices] if gains_all is not None else None
    offsets = [offsets_all[idx] for idx in selected_indices] if offsets_all is not None else None
    del first_recording
    gc.collect()

    if MATERIALIZE_SHANK_RECORDING_AS_NUMPY:
        traces_list = []
        for file_index, data_file in enumerate(data_files):
            end_frame = None
            if file_index == 0 and first_file_max_duration_s is not None and first_file_max_duration_s > 0:
                end_frame = int(round(first_file_max_duration_s * sampling_frequency))
                if end_frame <= 0:
                    raise ValueError("first_file_max_duration_s must produce at least one frame")

            step_name = f"read_shank_traces_file_{file_index + 1}"
            traces, step_timings[f"{step_name}_elapsed_sec"] = time_step(
                step_name,
                converter._read_recording_chunk_from_file,
                data_file=data_file,
                channel_ids=resolved_channel_ids,
                start_frame=0,
                end_frame=end_frame,
            )
            traces_list.append(traces)

        combined_recording, step_timings["build_numpy_recording_elapsed_sec"] = time_step(
            "build_numpy_recording",
            build_numpy_recording_from_traces,
            traces_list=traces_list,
            sampling_frequency=sampling_frequency,
            electrode_df=electrode_df,
            gains=gains,
            offsets=offsets,
        )
    else:
        selected_recordings = []
        for file_index, data_file in enumerate(data_files):
            recording, step_timings[f"load_recording_file_{file_index + 1}_elapsed_sec"] = time_step(
                f"load_recording_file_{file_index + 1}",
                converter._get_recording,
                data_file,
            )
            selected = recording.select_channels(resolved_channel_ids)

            if file_index == 0 and first_file_max_duration_s is not None and first_file_max_duration_s > 0:
                max_frames = min(
                    selected.get_num_frames(),
                    int(round(first_file_max_duration_s * selected.get_sampling_frequency())),
                )
                if max_frames <= 0:
                    raise ValueError("first_file_max_duration_s must produce at least one frame")
                selected = selected.frame_slice(start_frame=0, end_frame=max_frames)

            selected = attach_shank_metadata(selected, electrode_df)
            selected_recordings.append(selected)
            del recording

        combined_recording, step_timings["concatenate_recordings_elapsed_sec"] = time_step(
            "concatenate_shank_recordings",
            concatenate_recordings_compat,
            selected_recordings,
        )

    build_elapsed_sec = perf_counter() - build_start
    print_timer("build_shank_recording", build_elapsed_sec)
    return combined_recording, resolved_channel_ids, build_elapsed_sec, step_timings


def prepare_recording_for_pipeline(recording, window_label: str = "full_recording"):
    prep_start = perf_counter()
    step_timings: dict[str, float] = {}
    print_channel_information(recording)

    if "location" not in set(recording.get_property_keys()):
        raise RuntimeError("Recording has no channel locations after shank selection.")

    print(f"Channels: {recording.get_num_channels()}")
    print(f"Sampling frequency: {recording.get_sampling_frequency()}")
    print(f"Duration: {recording.get_num_frames() / recording.get_sampling_frequency():.2f} seconds")
    print(f"Channel IDs: {list(recording.get_channel_ids())}")

    window_metadata = {
        "window_label": str(window_label),
        "start_sec": 0.0,
        "end_sec": float(recording.get_num_frames() / recording.get_sampling_frequency()),
        "applied_duration_seconds": float(recording.get_num_frames() / recording.get_sampling_frequency()),
        "start_frame": 0,
        "end_frame": int(recording.get_num_frames()),
        "applied_num_frames": int(recording.get_num_frames()),
    }
    rec_for_sorting = recording
    print(
        "Sorting recording window: "
        f"{window_label} "
        f"({window_metadata['start_sec'] / 3600.0:.2f}h to {window_metadata['end_sec'] / 3600.0:.2f}h, "
        f"{window_metadata['applied_duration_seconds'] / 60.0:.2f} minutes)"
    )

    (rec_preprocessed, preprocessing_metadata), step_timings["preprocess_recording_elapsed_sec"] = time_step(
        f"{window_label}_preprocess_recording",
        preprocess_recording,
        rec_for_sorting,
        window_metadata,
    )
    preprocessing_metadata["window_label"] = str(window_label)
    preprocessing_metadata["window"] = window_metadata
    preprocessing_metadata["step_timings"] = {
        key: float(value) for key, value in step_timings.items()
    }

    prep_elapsed_sec = perf_counter() - prep_start
    print_timer(f"{window_label}_prepare_and_preprocess", prep_elapsed_sec)
    return rec_preprocessed, preprocessing_metadata, prep_elapsed_sec, step_timings


def run_sorter_pipeline(
    output_folder: Path,
    recording,
    input_sources: list[Path],
    shank_id: str,
    window_label: str = "full_recording",
):
    sorter_run_folder = output_folder / "sorter_output"
    sorted_sorting_folder = output_folder / "sorted_sorting"
    sorted_units_npz = output_folder / "sorted_units.npz"
    preprocessed_folder = output_folder / "preprocessed_recording"

    output_folder.mkdir(parents=True, exist_ok=True)
    step_timings: dict[str, float] = {}
    fallback_attempted = False
    fallback_succeeded = False

    rec_for_sorting = recording
    if SAVE_PREPROCESSED_RECORDING:
        print("Saving preprocessed recording to binary format for sorter compatibility...")
        rec_for_sorting, step_timings["save_preprocessed_recording_elapsed_sec"] = time_step(
            f"shank_{shank_id}_{window_label}_save_preprocessed_recording",
            rec_for_sorting.save,
            folder=preprocessed_folder,
            format="binary",
        )

    sorter_params = dict(MS5_SORTER_PARAMS)
    print(
        "Running MountainSort5 with conservative long-recording settings: "
        f"scheme={sorter_params['scheme']}, "
        f"training_duration_sec={sorter_params['scheme2_training_duration_sec']}, "
        f"chunk_duration={sorter_params['chunk_duration']}, "
        f"n_jobs={sorter_params['n_jobs']}"
    )

    try:
        sorting, sorting_elapsed_sec = time_step(
            f"shank_{shank_id}_{window_label}_run_sorter",
            ss.run_sorter,
            sorter_name="mountainsort5",
            recording=rec_for_sorting,
            folder=sorter_run_folder,
            remove_existing_folder=OVERWRITE_SORTER_OUTPUT,
            verbose=True,
            **sorter_params,
        )
    except Exception as exc:
        primary_traceback = traceback.format_exc()
        if is_template_alignment_overflow_error(exc):
            fallback_attempted = True
            fallback_params = dict(sorter_params)
            fallback_params.update(MS5_OVERFLOW_FALLBACK_PARAMS)
            print(
                "Primary MountainSort5 run overflowed during template alignment. "
                "Retrying with direct MountainSort5 fallback and skip_alignment=True "
                "while preserving the original detect_sign."
            )
            try:
                sorting, sorting_elapsed_sec = time_step(
                    f"shank_{shank_id}_{window_label}_run_sorter_fallback",
                    _run_mountainsort5_skip_alignment_fallback,
                    recording=rec_for_sorting,
                    sorter_params=fallback_params,
                )
                fallback_succeeded = True
                sorter_params = fallback_params
                step_timings["overflow_fallback_used"] = 1.0
                step_timings["overflow_fallback_stage"] = 1.0
                step_timings["run_sorter_fallback_elapsed_sec"] = float(sorting_elapsed_sec)
            except Exception as fallback_exc:
                failure_summary = _build_sorting_failure_summary(
                    output_folder=output_folder,
                    sorter_run_folder=sorter_run_folder,
                    input_sources=input_sources,
                    shank_id=shank_id,
                    window_label=window_label,
                    sorter_params=fallback_params,
                    recording=rec_for_sorting,
                    exc=fallback_exc,
                    traceback_text=traceback.format_exc(),
                    sorter_name="mountainsort5_skip_alignment_fallback",
                    fallback_attempted=True,
                    fallback_succeeded=False,
                )
                failure_summary["primary_exception_type"] = type(exc).__name__
                failure_summary["primary_exception_message"] = str(exc)
                failure_summary["primary_traceback"] = primary_traceback
                failure_summary_path = output_folder / "sorting_failure_summary.json"
                save_json(failure_summary_path, failure_summary)
                print(f"Saved sorter failure summary to: {failure_summary_path}")
                raise fallback_exc from exc
        else:
            failure_summary = _build_sorting_failure_summary(
                output_folder=output_folder,
                sorter_run_folder=sorter_run_folder,
                input_sources=input_sources,
                shank_id=shank_id,
                window_label=window_label,
                sorter_params=sorter_params,
                recording=rec_for_sorting,
                exc=exc,
                traceback_text=primary_traceback,
                sorter_name="mountainsort5",
                fallback_attempted=False,
                fallback_succeeded=False,
            )
            failure_summary_path = output_folder / "sorting_failure_summary.json"
            save_json(failure_summary_path, failure_summary)
            print(f"Saved sorter failure summary to: {failure_summary_path}")
            raise

    step_timings["run_sorter_elapsed_sec"] = float(sorting_elapsed_sec)

    if sorting.get_num_units() > 0:
        try:
            sorting.set_property(
                "shank_id",
                np.array([str(shank_id)] * sorting.get_num_units()),
            )
        except Exception:
            pass

    _, step_timings["save_sorting_folder_elapsed_sec"] = time_step(
        f"shank_{shank_id}_{window_label}_save_sorting_folder",
        sorting.save,
        folder=sorted_sorting_folder,
        overwrite=True,
    )
    _, step_timings["write_sorting_npz_elapsed_sec"] = time_step(
        f"shank_{shank_id}_{window_label}_write_sorting_npz",
        si.NpzSortingExtractor.write_sorting,
        sorting,
        sorted_units_npz,
    )

    summary = {
        "input_sources": [str(path) for path in input_sources],
        "output_folder": str(output_folder),
        "shank_id": str(shank_id),
        "window_label": str(window_label),
        "num_channels": int(rec_for_sorting.get_num_channels()),
        "sampling_frequency": float(rec_for_sorting.get_sampling_frequency()),
        "num_frames": int(rec_for_sorting.get_num_frames()),
        "duration_seconds": float(
            rec_for_sorting.get_num_frames() / rec_for_sorting.get_sampling_frequency()
        ),
        "num_units": int(sorting.get_num_units()),
        "sorting_elapsed_sec": float(sorting_elapsed_sec),
        "step_timings": step_timings,
        "channel_ids": [str(ch) for ch in rec_for_sorting.get_channel_ids()],
        "sorter": "mountainsort5_skip_alignment_fallback" if fallback_succeeded else "mountainsort5",
        "sorter_params": sorter_params,
        "overflow_fallback_attempted": bool(fallback_attempted),
        "overflow_fallback_succeeded": bool(fallback_succeeded),
    }
    save_json(output_folder / "sorting_summary.json", summary)

    print(f"Detected units: {sorting.get_num_units()}")
    print(f"Sorting time: {sorting_elapsed_sec:.2f} seconds")
    print(f"Saved sorting to: {sorted_sorting_folder}")
    print(f"Saved NPZ to: {sorted_units_npz}")
    print_timer(f"shank_{shank_id}_{window_label}_sorting", sorting_elapsed_sec)

    return sorting, sorting_elapsed_sec, step_timings


def create_or_load_analyzer(output_folder: Path, sorting, recording):
    analyzer_folder = output_folder / "sorting_analyzer_analysis.zarr"
    step_timings: dict[str, float] = {}
    if analyzer_folder.exists():
        analyzer, step_timings["load_sorting_analyzer_elapsed_sec"] = time_step(
            "load_sorting_analyzer",
            si.load_sorting_analyzer,
            folder=analyzer_folder,
            format="zarr",
            load_extensions=True,
        )
    else:
        analyzer, step_timings["create_sorting_analyzer_elapsed_sec"] = time_step(
            "create_sorting_analyzer",
            si.create_sorting_analyzer,
            sorting=sorting,
            recording=recording,
            format="memory",
        )
        initial_extensions = [
            ("random_spikes", {"method": "uniform", "max_spikes_per_unit": MAX_SPIKES_PER_UNIT}),
            ("waveforms", {"ms_before": MS_BEFORE, "ms_after": MS_AFTER}),
            ("templates", {"operators": ["average", "median", "std"]}),
            ("noise_levels", {}),
            ("spike_amplitudes", {"peak_sign": "neg"}),
            ("quality_metrics", {}),
            ("unit_locations", {"method": "monopolar_triangulation"}),
            ("correlograms", {"window_ms": 50.0, "bin_ms": 1.0, "method": "auto"}),
            ("isi_histograms", {"window_ms": 50.0, "bin_ms": 1.0, "method": "auto"}),
            ("template_similarity", {"method": "cosine_similarity"}),
        ]
        for ext_name, kwargs in initial_extensions:
            _, ext_elapsed_sec = time_step(
                f"compute_analyzer_extension_{ext_name}",
                analyzer.compute,
                ext_name,
                **kwargs,
            )
            step_timings[f"compute_{ext_name}_elapsed_sec"] = float(ext_elapsed_sec)
        _, step_timings["save_sorting_analyzer_elapsed_sec"] = time_step(
            "save_sorting_analyzer",
            analyzer.save_as,
            folder=analyzer_folder,
            format="zarr",
        )

    required_extensions = {
        "random_spikes": {"method": "uniform", "max_spikes_per_unit": MAX_SPIKES_PER_UNIT},
        "waveforms": {"ms_before": MS_BEFORE, "ms_after": MS_AFTER},
        "templates": {"operators": ["average", "median", "std"]},
        "noise_levels": {},
        "spike_amplitudes": {"peak_sign": "neg"},
        "quality_metrics": {},
        "unit_locations": {"method": "monopolar_triangulation"},
        "correlograms": {"window_ms": 50.0, "bin_ms": 1.0, "method": "auto"},
        "isi_histograms": {"window_ms": 50.0, "bin_ms": 1.0, "method": "auto"},
        "template_similarity": {"method": "cosine_similarity"},
    }
    for ext_name, kwargs in required_extensions.items():
        if not analyzer.has_extension(ext_name):
            _, ext_elapsed_sec = time_step(
                f"compute_missing_extension_{ext_name}",
                analyzer.compute,
                ext_name,
                **kwargs,
            )
            step_timings[f"compute_missing_{ext_name}_elapsed_sec"] = float(ext_elapsed_sec)

    return analyzer, step_timings


def examine_sorting(sorting):
    print("\n=== Sorting ===")
    print("Type:", type(sorting))
    print("Sorting class:", sorting.__class__.__name__)
    print("Number of segments:", sorting.get_num_segments())
    print("Number of units:", sorting.get_num_units())
    unit_ids = list(sorting.get_unit_ids())
    print("First unit IDs:", unit_ids[:10])
    print("Property keys:", list(sorting.get_property_keys()))

    if not unit_ids:
        return

    first_unit = unit_ids[0]
    print(f"Example spike train for unit {first_unit}:")
    for seg in range(sorting.get_num_segments()):
        st = sorting.get_unit_spike_train(unit_id=first_unit, segment_index=seg)
        print(f"  Segment {seg}: first 10 spikes -> {st[:10]}")


def examine_analyzer(analyzer):
    print("\n=== Sorting Analyzer ===")
    print("Type:", type(analyzer))
    print("Sorting units:", analyzer.sorting.get_num_units())
    print("Channel count:", len(analyzer.channel_ids))
    try:
        extension_names = sorted(analyzer.get_saved_extension_names())
    except ValueError:
        extension_names = sorted(analyzer.get_loaded_extension_names())
    print("Computed extensions:", extension_names)


def save_unit_summary_plots(analyzer, output_folder: Path):
    summary_folder = output_folder / "unit_summaries_analysis"
    summary_folder.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for unit_id in analyzer.sorting.get_unit_ids():
        sw.plot_unit_summary(analyzer, unit_id=unit_id)
        channel_idx = get_representative_channel_index(analyzer, unit_id)
        fig_name = build_unit_export_name(
            analyzer=analyzer,
            unit_id=unit_id,
            channel_idx=channel_idx,
            prefix="unit_summary",
        )
        plt.gcf().savefig(summary_folder / fig_name, dpi=200)
        plt.close("all")
        saved_count += 1

    return summary_folder, int(saved_count)


def save_unit_waveform_plots(analyzer, output_folder: Path):
    waveform_folder = output_folder / "unit_waveforms_analysis"
    waveform_folder.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    waveforms_ext = analyzer.get_extension("waveforms")
    for unit_id in analyzer.sorting.get_unit_ids():
        try:
            waveforms_u = waveforms_ext.get_waveforms_one_unit(unit_id)
        except Exception:
            waveforms_u = None

        if waveforms_u is None or getattr(waveforms_u, "size", 0) == 0:
            continue

        mean_waveform = waveforms_u.mean(axis=0)
        channel_idx = int(np.argmax(np.max(np.abs(mean_waveform), axis=0)))
        single_channel_waveforms = waveforms_u[:, :, channel_idx]
        average_waveform = np.mean(single_channel_waveforms, axis=0)
        time_axis = np.linspace(-MS_BEFORE, MS_AFTER, average_waveform.shape[0])

        plt.figure(figsize=(4, 6))
        for wf in single_channel_waveforms:
            plt.plot(time_axis, wf, color="gray", alpha=0.05)
        plt.plot(time_axis, average_waveform, color="red", linewidth=2)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.title(f"Unit {int(unit_id)} waveform")
        plt.tight_layout()
        fig_name = build_unit_export_name(
            analyzer=analyzer,
            unit_id=unit_id,
            channel_idx=channel_idx,
            prefix="unit_waveform",
        )
        plt.savefig(waveform_folder / fig_name, dpi=300)
        plt.close("all")
        saved_count += 1

    return waveform_folder, int(saved_count)


def run_analysis_pipeline(output_folder: Path, sorting, recording):
    analysis_start = perf_counter()
    if sorting.get_num_units() == 0:
        analysis_elapsed_sec = perf_counter() - analysis_start
        save_json(
            output_folder / "analysis_summary.json",
            {
                "output_folder": str(output_folder),
                "reason": "analysis_skipped_no_units",
                "num_units": 0,
                "analysis_elapsed_sec": float(analysis_elapsed_sec),
                "step_timings": {},
            },
        )
        print(f"Output folder: {output_folder}")
        print("\nAnalysis skipped: sorting contains 0 units, so no analyzer or plots were created.")
        print_timer("analysis_skipped_no_units", analysis_elapsed_sec)
        return analysis_elapsed_sec, {}

    analysis_step_timings: dict[str, float] = {}
    analyzer, analyzer_step_timings = create_or_load_analyzer(output_folder, sorting, recording)
    analysis_step_timings.update(analyzer_step_timings)

    print(f"Output folder: {output_folder}")
    examine_sorting(sorting)
    examine_analyzer(analyzer)

    channel_mapping_report_path = save_unit_channel_mapping_report(analyzer, output_folder)
    (summary_folder, unit_summary_plot_count), analysis_step_timings["save_unit_summary_plots_elapsed_sec"] = time_step(
        "save_unit_summary_plots",
        save_unit_summary_plots,
        analyzer,
        output_folder,
    )
    (waveform_folder, waveform_plot_count), analysis_step_timings["save_unit_waveform_plots_elapsed_sec"] = time_step(
        "save_unit_waveform_plots",
        save_unit_waveform_plots,
        analyzer,
        output_folder,
    )

    print("\nAnalysis complete.")
    print(f"Saved unit summaries to: {summary_folder}")
    print(f"Saved waveform plots to: {waveform_folder}")
    print(f"Saved unit-channel mapping report to: {channel_mapping_report_path}")
    analysis_elapsed_sec = perf_counter() - analysis_start
    print_timer("analysis_complete", analysis_elapsed_sec)
    save_json(
        output_folder / "analysis_summary.json",
        {
            "output_folder": str(output_folder),
            "reason": "analysis_completed",
            "num_units": int(sorting.get_num_units()),
            "unit_summary_folder": str(summary_folder),
            "waveform_folder": str(waveform_folder),
            "unit_channel_mapping_report": str(channel_mapping_report_path),
            "unit_summary_plot_count": int(unit_summary_plot_count),
            "waveform_plot_count": int(waveform_plot_count),
            "analysis_elapsed_sec": float(analysis_elapsed_sec),
            "step_timings": analysis_step_timings,
        },
    )
    return analysis_elapsed_sec, analysis_step_timings


def run_single_file_for_shank(
    data_file: Path,
    file_index: int,
    num_input_files: int,
    batch_output_folder: Path,
    session_description: str,
    recording_method: str,
    shank_config: dict,
    first_file_max_duration_s: float | None = None,
) -> dict:
    run_start = perf_counter()
    shank_id = str(shank_config["shank_id"])
    recording_label = build_recording_label(data_file)
    output_folder = build_recording_output_folder(
        batch_output_folder,
        session_description,
        shank_id,
        recording_label,
    )
    shank_output_root = build_shank_output_root(batch_output_folder, session_description, shank_id)
    shank_output_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print(f"Shank {shank_id} recording run {file_index} of {num_input_files}")
    print(f"Input file: {data_file}")
    print(f"Recording label: {recording_label}")
    print(f"Result folder: {output_folder}")
    print("#" * 60)

    recording = None
    rec_preprocessed = None
    sorting = None

    try:
        recording, resolved_channel_ids, recording_build_elapsed_sec, build_step_timings = create_shank_recording(
            data_files=[data_file],
            recording_method=recording_method,
            electrode_df=shank_config["electrode_df"],
            first_file_max_duration_s=first_file_max_duration_s,
        )
        duration_seconds = float(recording.get_num_frames() / recording.get_sampling_frequency())

        rec_preprocessed, preprocessing_metadata, preprocessing_elapsed_sec, preprocessing_step_timings = prepare_recording_for_pipeline(
            recording,
            window_label=recording_label,
        )
        preprocessing_metadata["shank_id"] = shank_id
        preprocessing_metadata["resolved_channel_ids"] = [str(ch) for ch in resolved_channel_ids]
        preprocessing_metadata["preprocessing_elapsed_sec"] = float(preprocessing_elapsed_sec)
        preprocessing_metadata["input_file"] = str(data_file)
        preprocessing_metadata["step_timings"] = {
            key: float(value) for key, value in preprocessing_step_timings.items()
        }
        save_json(output_folder / "preprocessing_metadata.json", preprocessing_metadata)

        sorting, sorting_elapsed_sec, sorting_step_timings = run_sorter_pipeline(
            output_folder=output_folder,
            recording=rec_preprocessed,
            input_sources=[data_file],
            shank_id=shank_id,
            window_label=recording_label,
        )
        if SAVE_PREPROCESSED_RECORDING:
            cleanup_preprocessed_recording_folder(output_folder)
        analysis_elapsed_sec, analysis_step_timings = run_analysis_pipeline(output_folder, sorting, rec_preprocessed)

        run_elapsed_sec = perf_counter() - run_start
        print_timer(f"shank_{shank_id}_{recording_label}_total", run_elapsed_sec)

        return {
            "shank_id": shank_id,
            "recording_label": recording_label,
            "input_file": str(data_file),
            "resolved_channel_ids": [str(ch) for ch in resolved_channel_ids],
            "duration_seconds": float(duration_seconds),
            "build_shank_recording_elapsed_sec": float(recording_build_elapsed_sec),
            "build_shank_recording_step_timings": build_step_timings,
            "preprocessing_elapsed_sec": float(preprocessing_elapsed_sec),
            "sorting_elapsed_sec": float(sorting_elapsed_sec),
            "analysis_elapsed_sec": float(analysis_elapsed_sec),
            "run_total_elapsed_sec": float(run_elapsed_sec),
            "step_timings": {
                "preprocessing": {
                    key: float(value) for key, value in preprocessing_step_timings.items()
                },
                "sorting": {
                    key: float(value) for key, value in sorting_step_timings.items()
                },
                "analysis": {
                    key: float(value) for key, value in analysis_step_timings.items()
                },
            },
            "output_folder": str(output_folder),
        }
    finally:
        del sorting
        del rec_preprocessed
        del recording
        gc.collect()


def process_shank_file(config: dict) -> dict:
    return run_single_file_for_shank(**config)


def run_retry_task(
    *,
    task: dict,
    batch_output_folder: Path,
    first_file_max_duration_s: float | None,
):
    data_file = Path(task["input_file"])
    session_description = str(task["session_description"])
    shank_id = str(task["shank_id"])

    shank_config, shank_config_elapsed_sec = build_single_shank_config(
        rec_path=data_file,
        recording_method=RECORDING_METHOD,
        probe_file=PROBE_FILE,
        impedance_file=IMPEDANCE_FILE,
        shank_id=shank_id,
    )
    preview_elapsed_sec = preview_channel_resolution([data_file], RECORDING_METHOD, [shank_config])

    batch_start_time = perf_counter()
    shank_output_root = build_shank_output_root(batch_output_folder, session_description, shank_id)
    shank_output_root.mkdir(parents=True, exist_ok=True)
    shank_batch = {
        "input_sources": [str(data_file)],
        "shank_id": shank_id,
        "num_input_files": 1,
        "runs": [],
        "num_runs_succeeded": 0,
        "status": "pending",
        "resort_scope": "overflow_report_retry",
        "source_overflow_report": str(task["report_path"]),
    }

    print("\n" + "=" * 60)
    print(f"Retrying failed overflow entry from report: {task['report_path']}")
    print(f"Session label: {session_description}")
    print(f"Shank id: {shank_id}")
    print(f"Input file: {data_file}")
    print("=" * 60)

    result = process_shank_file(
        {
            "data_file": data_file,
            "file_index": 1,
            "num_input_files": 1,
            "batch_output_folder": batch_output_folder,
            "session_description": session_description,
            "recording_method": RECORDING_METHOD,
            "shank_config": shank_config,
            "first_file_max_duration_s": first_file_max_duration_s,
        }
    )
    shank_batch["runs"].append(result)
    shank_batch["runs"].sort(key=lambda run: run["recording_label"])
    shank_batch["num_runs_succeeded"] = int(len(shank_batch["runs"]))

    shank_wall_clock_elapsed_sec = perf_counter() - batch_start_time
    shank_batch["wall_clock_elapsed_sec"] = float(shank_wall_clock_elapsed_sec)
    shank_batch["status"] = "success"

    shank_summary_path = (
        build_shank_output_root(batch_output_folder, session_description, shank_id)
        / f"batch_summary_{session_description}_sh{shank_id}.json"
    )
    save_json(shank_summary_path, shank_batch)

    return {
        "task": {
            "report_path": str(task["report_path"]),
            "session_description": str(task["session_description"]),
            "shank_id": str(task["shank_id"]),
            "input_file": str(task["input_file"]),
            "recording_label": str(task["recording_label"]),
        },
        "retry_status": "success",
        "retry_succeeded": True,
        "result": result,
        "shank_config_elapsed_sec": float(shank_config_elapsed_sec),
        "preview_elapsed_sec": float(preview_elapsed_sec),
        "wall_clock_elapsed_sec": float(shank_wall_clock_elapsed_sec),
        "summary_path": str(shank_summary_path),
    }


def build_single_shank_config(
    rec_path: Path,
    recording_method: str,
    probe_file: Path,
    impedance_file: Path | None,
    shank_id: str,
):
    config_start = perf_counter()
    source_folder = rec_path.parent if rec_path.is_file() else rec_path
    probe_group = pi.read_probeinterface(probe_file)
    probe = probe_group.probes[0]

    impedance_data = None
    if impedance_file is not None and impedance_file.exists():
        impedance_data = read_impedance_file(
            impedance_file,
            rec_probe=probe,
            ref_probe_path=probe_file,
        )
        print(f"Loaded impedance data for {len(impedance_data)} channels")
    elif impedance_file is not None:
        print(f"No impedance file found at {impedance_file}")

    probe_suffix = "SG" if recording_method == "spikegadget" else "Ripple"
    bad_folder = source_folder / f"bad_channel_screening_{probe_suffix}"
    bad_file = bad_folder / "bad_channels.txt"
    bad_ch_ids = load_bad_ch(bad_file)

    electrode_df_all = _build_electrode_df(
        probe,
        impedance_data=impedance_data,
        bad_ch_ids=bad_ch_ids,
    )
    unique_shanks = [str(shank) for shank in sorted(electrode_df_all["shank_id"].unique().tolist())]
    if str(shank_id) not in unique_shanks:
        raise ValueError(
            f"Requested shank {shank_id} was not found. Available shanks: {unique_shanks}"
        )

    electrode_df = electrode_df_all[
        electrode_df_all["shank_id"].astype(str) == str(shank_id)
    ].copy()
    if electrode_df.empty:
        raise RuntimeError(f"Shank {shank_id} has no channels after filtering.")
    electrode_df = electrode_df.sort_values("y", ascending=True).reset_index(drop=True)

    config_elapsed_sec = perf_counter() - config_start
    print_timer("build_single_shank_config", config_elapsed_sec)
    return {
        "shank_id": str(shank_id),
        "electrode_df": electrode_df,
    }, config_elapsed_sec


def preview_channel_resolution(
    data_files: list[Path],
    recording_method: str,
    shank_configs: list[dict],
) -> float:
    preview_start = perf_counter()
    if recording_method != "spikegadget" or not shank_configs:
        preview_elapsed_sec = perf_counter() - preview_start
        print_timer("preview_channel_resolution", preview_elapsed_sec)
        return preview_elapsed_sec

    converter = EphysToNWBConverter(recording_method=recording_method)
    first_recording = converter._get_recording(data_files[0])
    _print_recording_channel_id_preview(first_recording)
    total_channels = first_recording.get_num_channels()
    for cfg in shank_configs:
        sg_channel_indices = cfg["electrode_df"]["device_channel_index"].tolist()
        cfg["resolved_channel_ids"] = converter._resolve_shank_channel_ids(
            first_recording,
            sg_channel_indices,
        )
    _print_resolved_shank_channel_id_preview(shank_configs)
    _print_shank_namespace_preview(shank_configs, total_channels)
    del first_recording
    gc.collect()
    preview_elapsed_sec = perf_counter() - preview_start
    print_timer("preview_channel_resolution", preview_elapsed_sec)
    return preview_elapsed_sec


def main() -> None:
    overall_start = perf_counter()
    if RECORDING_METHOD not in {"spikegadget", "ripple"}:
        raise ValueError(f"Invalid RECORDING_METHOD: {RECORDING_METHOD}")
    if not PROBE_FILE.exists():
        raise FileNotFoundError(f"Probe file not found: {PROBE_FILE}")

    input_start = perf_counter()
    rec_path_str = input(
        "\nEnter recording file/folder path(s), or a sorting folder that contains overflow_error_report_*.json: "
    ).strip()
    input_paths = parse_input_paths(rec_path_str)

    if is_overflow_report_folder(input_paths):
        overflow_folder = input_paths[0]
        first_file_only_duration_input = input(
            "Enter seconds to use from each failed recording retry (press Enter for all data): "
        ).strip()
        first_file_only_duration = (
            float(first_file_only_duration_input)
            if first_file_only_duration_input else None
        )
        input_elapsed_sec = perf_counter() - input_start
        print_timer("collect_user_inputs", input_elapsed_sec)

        report_paths = find_overflow_report_files(overflow_folder)
        retry_tasks = []
        for report_path in report_paths:
            retry_tasks.extend(build_retry_tasks_from_overflow_report(report_path))
        if not retry_tasks:
            raise RuntimeError(f"No failures found in overflow report folder: {overflow_folder}")

        print(f"Overflow report folder: {overflow_folder}")
        print(f"Found {len(report_paths)} overflow report file(s)")
        print(f"Found {len(retry_tasks)} failed shank/session run(s) to retry")
        confirm_lines = [
            f"{index}. session={task['session_description']} | shank={task['shank_id']} | label={task['recording_label']} | input={task['input_file']}"
            for index, task in enumerate(retry_tasks, start=1)
        ]
        confirm_terminal_action(
            "Overflow Report Retry Plan",
            confirm_lines,
        )

        retry_results = []
        mode_start = perf_counter()
        for task_index, task in enumerate(retry_tasks, start=1):
            print("\n" + "=" * 60)
            print(f"Retry task {task_index} of {len(retry_tasks)}")
            print(f"Report: {task['report_path']}")
            print(f"Session: {task['session_description']}")
            print(f"Shank: {task['shank_id']}")
            print(f"Recording label: {task['recording_label']}")
            print(f"Input file: {task['input_file']}")
            print("=" * 60)

            retry_results.append(
                run_retry_task(
                    task=task,
                    batch_output_folder=overflow_folder,
                    first_file_max_duration_s=first_file_only_duration,
                )
            )

        overall_elapsed_sec = perf_counter() - overall_start
        mode_elapsed_sec = perf_counter() - mode_start
        num_retry_succeeded = sum(1 for item in retry_results if item.get("retry_succeeded"))
        num_retry_failed = len(retry_results) - num_retry_succeeded
        rerun_summary = {
            "mode": "overflow_skiptemplate_folder_retry",
            "report_folder": str(overflow_folder),
            "recording_method": RECORDING_METHOD,
            "num_reports": int(len(report_paths)),
            "num_retry_tasks": int(len(retry_tasks)),
            "num_retry_succeeded": int(num_retry_succeeded),
            "num_retry_failed": int(num_retry_failed),
            "all_retries_succeeded": bool(num_retry_failed == 0 and len(retry_results) == len(retry_tasks)),
            "status": (
                "all_retries_succeeded"
                if num_retry_failed == 0 and len(retry_results) == len(retry_tasks)
                else "retry_incomplete"
            ),
            "first_file_only_duration_sec": float(first_file_only_duration) if first_file_only_duration is not None else None,
            "collect_user_inputs_elapsed_sec": float(input_elapsed_sec),
            "retry_mode_wall_clock_elapsed_sec": float(mode_elapsed_sec),
            "overall_wall_clock_elapsed_sec": float(overall_elapsed_sec),
            "source_overflow_reports": [str(path) for path in report_paths],
            "session_labels": sorted({str(task["session_description"]) for task in retry_tasks}),
            "tasks": retry_results,
        }
        rerun_summary_path = get_overflow_skiptemplate_summary_path(overflow_folder)
        save_json(rerun_summary_path, rerun_summary)
        print(f"\nSaved overflow skiptemplate summary to: {rerun_summary_path}")
        print(f"Finished overflow-report retry processing in {_format_elapsed_time(mode_elapsed_sec)}.")
        return

    rec_path = input_paths[0]
    shank_id = input("Enter the shank id to resort: ").strip()
    if not shank_id:
        raise ValueError("A shank id is required.")

    session_description = input(
        "Enter the session label to use for output paths (press Enter to derive from recording file[s]): "
    ).strip()
    first_file_only_duration_input = input(
        "Enter seconds to use from only the first recording (press Enter for all data): "
    ).strip()
    first_file_only_duration = (
        float(first_file_only_duration_input)
        if first_file_only_duration_input else None
    )
    input_elapsed_sec = perf_counter() - input_start
    print_timer("collect_user_inputs", input_elapsed_sec)

    collect_files_start = perf_counter()
    data_files = collect_data_files_from_inputs(input_paths, RECORDING_METHOD)
    collect_files_elapsed_sec = perf_counter() - collect_files_start
    print(f"Input paths: {input_paths}")
    print(f"All data files: {data_files}")
    print_timer("collect_data_files", collect_files_elapsed_sec)

    output_setup_start = perf_counter()
    batch_output_folder = build_batch_output_folder_for_inputs(input_paths, data_files)
    batch_output_folder.mkdir(parents=True, exist_ok=True)
    if not session_description:
        session_description = build_session_description(data_files)
    print(f"Daily sorting root: {batch_output_folder}")
    print(f"Session label: {session_description}")
    print("NWB files will not be written. Sorting uses in-memory per-shank recordings.")
    output_setup_elapsed_sec = perf_counter() - output_setup_start
    print_timer("setup_output_folder", output_setup_elapsed_sec)

    confirm_terminal_action(
        "Manual Resort Plan",
        [
            f"session={session_description}",
            f"shank={shank_id}",
            f"input_paths={input_paths}",
            f"recording_files={data_files}",
            f"output_root={batch_output_folder}",
        ],
    )

    shank_config, shank_config_elapsed_sec = build_single_shank_config(
        rec_path=rec_path,
        recording_method=RECORDING_METHOD,
        probe_file=PROBE_FILE,
        impedance_file=IMPEDANCE_FILE,
        shank_id=shank_id,
    )
    preview_elapsed_sec = preview_channel_resolution(data_files, RECORDING_METHOD, [shank_config])

    batch_start_time = perf_counter()
    selected_shank_id = str(shank_config["shank_id"])
    shank_output_root = build_shank_output_root(batch_output_folder, session_description, selected_shank_id)
    shank_output_root.mkdir(parents=True, exist_ok=True)
    shank_batch = {
        "input_sources": [str(path) for path in data_files],
        "shank_id": selected_shank_id,
        "num_input_files": int(len(data_files)),
        "runs": [],
        "resort_scope": "single_shank_retry",
    }
    overflow_failures = []

    print(
        f"Processing shank {selected_shank_id} across {len(data_files)} recording file(s) for session {session_description}."
    )

    for file_index, data_file in enumerate(data_files, start=1):
        recording_label = build_recording_label(data_file)
        file_batch_start = perf_counter()
        print("\n" + "=" * 60)
        print(f"Starting recording file {file_index} of {len(data_files)}")
        print(f"Input file: {data_file}")
        print(f"Recording label: {recording_label}")
        print("=" * 60)

        result = process_shank_file(
            {
                "data_file": data_file,
                "file_index": file_index,
                "num_input_files": len(data_files),
                "batch_output_folder": batch_output_folder,
                "session_description": session_description,
                "recording_method": RECORDING_METHOD,
                "shank_config": shank_config,
                "first_file_max_duration_s": first_file_only_duration,
            }
        )
        shank_batch["runs"].append(result)

        file_elapsed_sec = perf_counter() - file_batch_start
        print_timer(f"recording_{recording_label}_resort_total", file_elapsed_sec)
        gc.collect()

    shank_batch["runs"].sort(key=lambda run: run["recording_label"])
    shank_wall_clock_elapsed_sec = perf_counter() - batch_start_time
    shank_batch["wall_clock_elapsed_sec"] = float(shank_wall_clock_elapsed_sec)

    shank_summary_path = (
        build_shank_output_root(batch_output_folder, session_description, selected_shank_id)
        / f"batch_summary_{session_description}_sh{selected_shank_id}.json"
    )
    save_json(shank_summary_path, shank_batch)
    print(f"\nSaved batch summary to: {shank_summary_path}")

    total_build_shank_recording_elapsed_sec = sum(
        run["build_shank_recording_elapsed_sec"] for run in shank_batch["runs"]
    )
    total_preprocessing_elapsed_sec = sum(
        run["preprocessing_elapsed_sec"] for run in shank_batch["runs"]
    )
    total_sorting_elapsed_sec = sum(run["sorting_elapsed_sec"] for run in shank_batch["runs"])
    total_analysis_elapsed_sec = sum(run["analysis_elapsed_sec"] for run in shank_batch["runs"])
    total_run_elapsed_sec = sum(run["run_total_elapsed_sec"] for run in shank_batch["runs"])

    batch_results = [
        {
            "shank_id": selected_shank_id,
            "num_input_files": int(len(shank_batch["runs"])),
            "build_shank_recording_elapsed_sec": float(total_build_shank_recording_elapsed_sec),
            "total_preprocessing_elapsed_sec": float(total_preprocessing_elapsed_sec),
            "total_sorting_elapsed_sec": float(total_sorting_elapsed_sec),
            "total_analysis_elapsed_sec": float(total_analysis_elapsed_sec),
            "total_run_elapsed_sec": float(total_run_elapsed_sec),
            "wall_clock_elapsed_sec": float(shank_wall_clock_elapsed_sec),
            "summary_path": str(shank_summary_path),
        }
    ]

    print(
        f"Completed shank {selected_shank_id}: "
        f"{len(shank_batch['runs'])} recording run(s), wall-clock elapsed time {_format_elapsed_time(shank_wall_clock_elapsed_sec)}"
    )
    print("Shank timing summary:")
    print(f"  build_shank_recording total: {format_elapsed_time(total_build_shank_recording_elapsed_sec)}")
    print(f"  preprocessing total: {format_elapsed_time(total_preprocessing_elapsed_sec)}")
    print(f"  sorting total: {format_elapsed_time(total_sorting_elapsed_sec)}")
    print(f"  analysis total: {format_elapsed_time(total_analysis_elapsed_sec)}")
    print(f"  per-recording run total: {format_elapsed_time(total_run_elapsed_sec)}")
    print(f"  wall-clock total: {format_elapsed_time(shank_wall_clock_elapsed_sec)}")

    total_wall_clock_elapsed_sec = perf_counter() - batch_start_time
    overall_elapsed_sec = perf_counter() - overall_start

    overflow_report_path = get_overflow_report_path(batch_output_folder, session_description)
    overflow_report = {
        "session_description": session_description,
        "recording_method": RECORDING_METHOD,
        "report_scope": "single_shank_retry",
        "report_folder": str(overflow_report_path.parent),
        "selected_shank_id": selected_shank_id,
        "status": "overflow_errors_detected" if overflow_failures else "no_overflow_errors",
        "message": (
            f"Detected {len(overflow_failures)} overflow error(s) during sorting."
            if overflow_failures else
            "No overflow errors were detected during sorting."
        ),
        "num_overflow_failures": int(len(overflow_failures)),
        "failures": overflow_failures,
    }
    save_json(overflow_report_path, overflow_report)
    print(f"Saved overflow error report to: {overflow_report_path}")

    total_build_shank_recording_elapsed_sec = sum(
        result["build_shank_recording_elapsed_sec"] for result in batch_results
    )
    total_preprocessing_elapsed_sec = sum(
        result["total_preprocessing_elapsed_sec"] for result in batch_results
    )
    total_sorting_elapsed_sec = sum(
        result["total_sorting_elapsed_sec"] for result in batch_results
    )
    total_analysis_elapsed_sec = sum(
        result["total_analysis_elapsed_sec"] for result in batch_results
    )
    total_run_elapsed_sec = sum(
        result["total_run_elapsed_sec"] for result in batch_results
    )

    print("\n" + "=" * 60)
    print("Batch sorting summary by shank")
    print("=" * 60)
    for result in batch_results:
        print(
            f"shank {result['shank_id']}: "
            f"{result['num_input_files']} input file(s), "
            f"wall-clock {_format_elapsed_time(result['wall_clock_elapsed_sec'])}, "
            f"sorter-only total {_format_elapsed_time(result['total_sorting_elapsed_sec'])}"
        )

    print("\n" + "=" * 60)
    print("Final timing summary")
    print("=" * 60)
    print(f"collect_user_inputs: {format_elapsed_time(input_elapsed_sec)}")
    print(f"collect_data_files: {format_elapsed_time(collect_files_elapsed_sec)}")
    print(f"setup_output_folder: {format_elapsed_time(output_setup_elapsed_sec)}")
    print(f"build_shank_config: {format_elapsed_time(shank_config_elapsed_sec)}")
    print(f"preview_channel_resolution: {format_elapsed_time(preview_elapsed_sec)}")
    print(f"build_shank_recordings total: {format_elapsed_time(total_build_shank_recording_elapsed_sec)}")
    print(f"preprocessing total: {format_elapsed_time(total_preprocessing_elapsed_sec)}")
    print(f"sorting total: {format_elapsed_time(total_sorting_elapsed_sec)}")
    print(f"analysis total: {format_elapsed_time(total_analysis_elapsed_sec)}")
    print(f"per-recording run totals: {format_elapsed_time(total_run_elapsed_sec)}")
    print(f"batch wall-clock: {format_elapsed_time(total_wall_clock_elapsed_sec)}")
    print(f"overall wall-clock: {format_elapsed_time(overall_elapsed_sec)}")

    batch_summary = {
        "session_description": session_description,
        "recording_method": RECORDING_METHOD,
        "input_sources": [str(path) for path in data_files],
        "selected_shanks": [selected_shank_id],
        "resort_scope": "single_shank_retry",
        "num_overflow_failures": int(len(overflow_failures)),
        "overflow_error_report": str(overflow_report_path),
        "timing": {
            "collect_user_inputs_elapsed_sec": float(input_elapsed_sec),
            "collect_data_files_elapsed_sec": float(collect_files_elapsed_sec),
            "setup_output_folder_elapsed_sec": float(output_setup_elapsed_sec),
            "build_shank_config_elapsed_sec": float(shank_config_elapsed_sec),
            "preview_channel_resolution_elapsed_sec": float(preview_elapsed_sec),
            "build_shank_recordings_total_elapsed_sec": float(total_build_shank_recording_elapsed_sec),
            "preprocessing_total_elapsed_sec": float(total_preprocessing_elapsed_sec),
            "sorting_total_elapsed_sec": float(total_sorting_elapsed_sec),
            "analysis_total_elapsed_sec": float(total_analysis_elapsed_sec),
            "per_recording_run_totals_elapsed_sec": float(total_run_elapsed_sec),
            "batch_wall_clock_elapsed_sec": float(total_wall_clock_elapsed_sec),
            "overall_wall_clock_elapsed_sec": float(overall_elapsed_sec),
        },
        "shanks": batch_results,
    }
    save_json(
        batch_output_folder / f"combined_batch_summary_{session_description}.json",
        batch_summary,
    )

    print(f"\nFinished combined processing in {_format_elapsed_time(total_wall_clock_elapsed_sec)}.")


if __name__ == "__main__":
    main()
