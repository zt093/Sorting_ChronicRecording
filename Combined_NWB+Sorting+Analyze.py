# Input files should be from a same day

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import os
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

HOUR_DURATION_SEC = 3600
SAVE_PREPROCESSED_RECORDING = True
OVERWRITE_SORTER_OUTPUT = True
MAX_PARALLEL_SHANKS = 4

MS_BEFORE = 1.0
MS_AFTER = 2.0
MAX_SPIKES_PER_UNIT = 500
MATERIALIZE_SHANK_RECORDING_AS_NUMPY = True

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


def parse_input_paths(raw_value: str) -> list[Path]:
    parts = [part.strip().strip('"').strip("'") for part in re.split(r"[;\n]+", raw_value) if part.strip()]
    if not parts:
        raise ValueError("No recording path provided.")

    input_paths = [Path(part) for part in parts]
    missing_paths = [str(path) for path in input_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Recording path(s) not found: {missing_paths}")

    return input_paths


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


def preprocess_recording(recording, window_metadata: dict):
    rec_cr = preproc.common_reference(recording, operator="median", reference="global")
    rec_filt = preproc.bandpass_filter(
        rec_cr,
        freq_min=300,
        freq_max=6000,
        dtype="float32",
    )
    metadata = {
        "steps": ["common_reference", "bandpass_filter"],
        "params": {
            "common_reference": {"operator": "median", "reference": "global"},
            "bandpass_filter": {"freq_min": 300, "freq_max": 6000, "dtype": "float32"},
            "window": window_metadata,
        },
    }
    return rec_filt, metadata


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
        sampling_frequency = None
        num_frames = None
        duration_seconds = None
        num_channels = None
        channel_ids = []
        property_keys = []

        try:
            sampling_frequency = float(rec_for_sorting.get_sampling_frequency())
        except Exception:
            pass
        try:
            num_frames = int(rec_for_sorting.get_num_frames())
        except Exception:
            pass
        if sampling_frequency and num_frames is not None and sampling_frequency > 0:
            duration_seconds = float(num_frames / sampling_frequency)
        try:
            num_channels = int(rec_for_sorting.get_num_channels())
        except Exception:
            pass
        try:
            channel_ids = [str(ch) for ch in rec_for_sorting.get_channel_ids()]
        except Exception:
            pass
        try:
            property_keys = list(rec_for_sorting.get_property_keys())
        except Exception:
            pass

        failure_summary = {
            "input_sources": [str(path) for path in input_sources],
            "output_folder": str(output_folder),
            "sorter_run_folder": str(sorter_run_folder),
            "shank_id": str(shank_id),
            "window_label": str(window_label),
            "sorter": "mountainsort5",
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
            "max_parallel_shanks": int(MAX_PARALLEL_SHANKS),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
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
        "sorter": "mountainsort5",
        "sorter_params": sorter_params,
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


def run_per_file_batch_for_shank(
    data_files: list[Path],
    batch_output_folder: Path,
    session_description: str,
    recording_method: str,
    shank_config: dict,
    first_file_max_duration_s: float | None = None,
) -> dict:
    batch_start_time = perf_counter()
    shank_id = str(shank_config["shank_id"])
    shank_output_root = build_shank_output_root(batch_output_folder, session_description, shank_id)
    shank_output_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Preparing shank {shank_id}")
    print(f"Output root: {shank_output_root}")
    print("=" * 60)

    batch_summary = {
        "input_sources": [str(path) for path in data_files],
        "shank_id": shank_id,
        "num_input_files": int(len(data_files)),
        "runs": [],
    }

    for file_index, data_file in enumerate(data_files, start=1):
        run_start = perf_counter()
        recording_label = build_recording_label(data_file)
        output_folder = build_recording_output_folder(
            batch_output_folder,
            session_description,
            shank_id,
            recording_label,
        )
        print("\n" + "#" * 60)
        print(f"Shank {shank_id} recording run {file_index} of {len(data_files)}")
        print(f"Input file: {data_file}")
        print(f"Recording label: {recording_label}")
        print(f"Result folder: {output_folder}")
        print("#" * 60)

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

        batch_summary["runs"].append(
            {
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
        )
        gc.collect()

    summary_path = shank_output_root / "batch_summary.json"
    wall_clock_elapsed_sec = perf_counter() - batch_start_time
    batch_summary["wall_clock_elapsed_sec"] = float(wall_clock_elapsed_sec)
    save_json(summary_path, batch_summary)
    print(f"\nSaved batch summary to: {summary_path}")

    total_build_shank_recording_elapsed_sec = sum(
        run["build_shank_recording_elapsed_sec"] for run in batch_summary["runs"]
    )
    total_preprocessing_elapsed_sec = sum(
        run["preprocessing_elapsed_sec"] for run in batch_summary["runs"]
    )
    total_sorting_elapsed_sec = sum(run["sorting_elapsed_sec"] for run in batch_summary["runs"])
    total_analysis_elapsed_sec = sum(run["analysis_elapsed_sec"] for run in batch_summary["runs"])
    total_run_elapsed_sec = sum(run["run_total_elapsed_sec"] for run in batch_summary["runs"])
    print(
        f"Completed shank {shank_id}: "
        f"{len(batch_summary['runs'])} recording run(s), wall-clock elapsed time {_format_elapsed_time(wall_clock_elapsed_sec)}"
    )
    print("Shank timing summary:")
    print(f"  build_shank_recording total: {format_elapsed_time(total_build_shank_recording_elapsed_sec)}")
    print(f"  preprocessing total: {format_elapsed_time(total_preprocessing_elapsed_sec)}")
    print(f"  sorting total: {format_elapsed_time(total_sorting_elapsed_sec)}")
    print(f"  analysis total: {format_elapsed_time(total_analysis_elapsed_sec)}")
    print(f"  per-recording run total: {format_elapsed_time(total_run_elapsed_sec)}")
    print(f"  wall-clock total: {format_elapsed_time(wall_clock_elapsed_sec)}")
    return {
        "shank_id": shank_id,
        "num_input_files": int(len(data_files)),
        "build_shank_recording_elapsed_sec": float(total_build_shank_recording_elapsed_sec),
        "total_preprocessing_elapsed_sec": float(total_preprocessing_elapsed_sec),
        "total_sorting_elapsed_sec": float(total_sorting_elapsed_sec),
        "total_analysis_elapsed_sec": float(total_analysis_elapsed_sec),
        "total_run_elapsed_sec": float(total_run_elapsed_sec),
        "wall_clock_elapsed_sec": float(wall_clock_elapsed_sec),
        "summary_path": str(summary_path),
    }


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
    except Exception as exc:
        if not is_overflow_sorting_error(exc):
            raise

        run_elapsed_sec = perf_counter() - run_start
        failure_summary_path = output_folder / "sorting_failure_summary.json"
        print(
            f"Overflow error while sorting shank {shank_id} for recording {recording_label}. "
            "Recording the failure and continuing with the remaining work."
        )
        print_timer(f"shank_{shank_id}_{recording_label}_total", run_elapsed_sec)

        return {
            "shank_id": shank_id,
            "recording_label": recording_label,
            "input_file": str(data_file),
            "resolved_channel_ids": [str(ch) for ch in resolved_channel_ids] if recording is not None else [],
            "duration_seconds": float(duration_seconds) if "duration_seconds" in locals() else None,
            "build_shank_recording_elapsed_sec": float(recording_build_elapsed_sec) if "recording_build_elapsed_sec" in locals() else 0.0,
            "build_shank_recording_step_timings": build_step_timings if "build_step_timings" in locals() else {},
            "preprocessing_elapsed_sec": float(preprocessing_elapsed_sec) if "preprocessing_elapsed_sec" in locals() else 0.0,
            "sorting_elapsed_sec": 0.0,
            "analysis_elapsed_sec": 0.0,
            "run_total_elapsed_sec": float(run_elapsed_sec),
            "status": "overflow_error",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "failure_summary_path": str(failure_summary_path),
            "step_timings": {
                "preprocessing": {
                    key: float(value) for key, value in (preprocessing_step_timings.items() if "preprocessing_step_timings" in locals() else [])
                },
                "sorting": {},
                "analysis": {},
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


def build_shank_configs(
    rec_path: Path,
    recording_method: str,
    probe_file: Path,
    impedance_file: Path | None,
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
    shank_str = input(
        f"Enter shank numbers to sort/analyze (e.g. 0,1,2) or press Enter for all {unique_shanks}: "
    ).strip()
    selected_shanks = [str(x) for x in re.findall(r"\d+", shank_str)] if shank_str else unique_shanks
    print(f"Selected shanks for this run: {selected_shanks}")

    shank_configs = []
    for shank_id in selected_shanks:
        electrode_df = electrode_df_all[
            electrode_df_all["shank_id"].astype(str) == str(shank_id)
        ].copy()
        if electrode_df.empty:
            print(f"Shank {shank_id} has no channels after filtering. Skipping.")
            continue
        electrode_df = electrode_df.sort_values("y", ascending=True).reset_index(drop=True)
        shank_configs.append(
            {
                "shank_id": shank_id,
                "electrode_df": electrode_df,
            }
        )

    config_elapsed_sec = perf_counter() - config_start
    print_timer("build_shank_configs", config_elapsed_sec)
    return shank_configs, config_elapsed_sec


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
        "\nEnter recording file/folder path(s) separated by semicolons: "
    ).strip()
    input_paths = parse_input_paths(rec_path_str)
    rec_path = input_paths[0]

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
    session_description = build_session_description(data_files)
    print(f"Daily sorting root: {batch_output_folder}")
    print(f"Session label: {session_description}")
    print("NWB files will not be written. Sorting uses in-memory per-shank recordings.")
    output_setup_elapsed_sec = perf_counter() - output_setup_start
    print_timer("setup_output_folder", output_setup_elapsed_sec)

    shank_configs, shank_config_elapsed_sec = build_shank_configs(
        rec_path=rec_path,
        recording_method=RECORDING_METHOD,
        probe_file=PROBE_FILE,
        impedance_file=IMPEDANCE_FILE,
    )
    if not shank_configs:
        raise RuntimeError("No shanks selected or available after filtering.")
    preview_elapsed_sec = preview_channel_resolution(data_files, RECORDING_METHOD, shank_configs)

    batch_start_time = perf_counter()
    shank_batches = {}
    overflow_failures = []
    for shank_config in shank_configs:
        shank_id = str(shank_config["shank_id"])
        shank_output_root = build_shank_output_root(batch_output_folder, session_description, shank_id)
        shank_output_root.mkdir(parents=True, exist_ok=True)
        shank_batches[shank_id] = {
            "input_sources": [str(path) for path in data_files],
            "shank_id": shank_id,
            "num_input_files": int(len(data_files)),
            "runs": [],
            "_batch_start_time": perf_counter(),
        }

    max_workers = min(MAX_PARALLEL_SHANKS, len(shank_configs))
    print(
        f"Processing recordings file-first with up to {max_workers} parallel shank worker(s) per file."
    )

    for file_index, data_file in enumerate(data_files, start=1):
        recording_label = build_recording_label(data_file)
        file_batch_start = perf_counter()
        print("\n" + "=" * 60)
        print(f"Starting recording file {file_index} of {len(data_files)}")
        print(f"Input file: {data_file}")
        print(f"Recording label: {recording_label}")
        print("=" * 60)

        run_configs = [
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
            for shank_config in shank_configs
        ]

        file_results = []
        if max_workers <= 1:
            for config in run_configs:
                file_results.append(process_shank_file(config))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_shank = {
                    executor.submit(process_shank_file, config): config["shank_config"]["shank_id"]
                    for config in run_configs
                }
                for future in as_completed(future_to_shank):
                    shank_id = str(future_to_shank[future])
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(f"Failed processing shank {shank_id} for file {data_file}: {exc}")
                        raise
                    file_results.append(result)

        for result in file_results:
            shank_batches[str(result["shank_id"])]["runs"].append(result)
            if result.get("status") == "overflow_error":
                overflow_failures.append(
                    {
                        "shank_id": str(result["shank_id"]),
                        "recording_label": str(result["recording_label"]),
                        "input_file": str(result["input_file"]),
                        "output_folder": str(result["output_folder"]),
                        "exception_type": str(result.get("exception_type", "")),
                        "exception_message": str(result.get("exception_message", "")),
                        "failure_summary_path": str(result.get("failure_summary_path", "")),
                    }
                )

        file_elapsed_sec = perf_counter() - file_batch_start
        print_timer(f"recording_{recording_label}_all_shanks_total", file_elapsed_sec)
        gc.collect()

    batch_results = []
    for shank_id, batch_summary in shank_batches.items():
        batch_summary["runs"].sort(key=lambda run: run["recording_label"])
        wall_clock_elapsed_sec = perf_counter() - batch_summary["_batch_start_time"]
        batch_summary["wall_clock_elapsed_sec"] = float(wall_clock_elapsed_sec)

        summary_path = (
            build_shank_output_root(batch_output_folder, session_description, shank_id)
            / f"batch_summary_{session_description}_sh{shank_id}.json"
        )
        save_json(
            summary_path,
            {key: value for key, value in batch_summary.items() if not key.startswith("_")}
        )
        print(f"\nSaved batch summary to: {summary_path}")

        total_build_shank_recording_elapsed_sec = sum(
            run["build_shank_recording_elapsed_sec"] for run in batch_summary["runs"]
        )
        total_preprocessing_elapsed_sec = sum(
            run["preprocessing_elapsed_sec"] for run in batch_summary["runs"]
        )
        total_sorting_elapsed_sec = sum(run["sorting_elapsed_sec"] for run in batch_summary["runs"])
        total_analysis_elapsed_sec = sum(run["analysis_elapsed_sec"] for run in batch_summary["runs"])
        total_run_elapsed_sec = sum(run["run_total_elapsed_sec"] for run in batch_summary["runs"])
        print(
            f"Completed shank {shank_id}: "
            f"{len(batch_summary['runs'])} recording run(s), wall-clock elapsed time {_format_elapsed_time(wall_clock_elapsed_sec)}"
        )
        print("Shank timing summary:")
        print(f"  build_shank_recording total: {format_elapsed_time(total_build_shank_recording_elapsed_sec)}")
        print(f"  preprocessing total: {format_elapsed_time(total_preprocessing_elapsed_sec)}")
        print(f"  sorting total: {format_elapsed_time(total_sorting_elapsed_sec)}")
        print(f"  analysis total: {format_elapsed_time(total_analysis_elapsed_sec)}")
        print(f"  per-recording run total: {format_elapsed_time(total_run_elapsed_sec)}")
        print(f"  wall-clock total: {format_elapsed_time(wall_clock_elapsed_sec)}")

        batch_results.append(
            {
                "shank_id": shank_id,
                "num_input_files": int(len(batch_summary["runs"])),
                "build_shank_recording_elapsed_sec": float(total_build_shank_recording_elapsed_sec),
                "total_preprocessing_elapsed_sec": float(total_preprocessing_elapsed_sec),
                "total_sorting_elapsed_sec": float(total_sorting_elapsed_sec),
                "total_analysis_elapsed_sec": float(total_analysis_elapsed_sec),
                "total_run_elapsed_sec": float(total_run_elapsed_sec),
                "wall_clock_elapsed_sec": float(wall_clock_elapsed_sec),
                "summary_path": str(summary_path),
            }
        )

    batch_results.sort(key=lambda result: int(result["shank_id"]))
    total_wall_clock_elapsed_sec = perf_counter() - batch_start_time
    overall_elapsed_sec = perf_counter() - overall_start

    overflow_report_path = get_overflow_report_path(batch_output_folder, session_description)
    overflow_report = {
        "session_description": session_description,
        "recording_method": RECORDING_METHOD,
        "report_scope": "batch_output_folder",
        "report_folder": str(overflow_report_path.parent),
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
    print(f"build_shank_configs: {format_elapsed_time(shank_config_elapsed_sec)}")
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
        "selected_shanks": [result["shank_id"] for result in batch_results],
        "num_overflow_failures": int(len(overflow_failures)),
        "overflow_error_report": str(overflow_report_path),
        "timing": {
            "collect_user_inputs_elapsed_sec": float(input_elapsed_sec),
            "collect_data_files_elapsed_sec": float(collect_files_elapsed_sec),
            "setup_output_folder_elapsed_sec": float(output_setup_elapsed_sec),
            "build_shank_configs_elapsed_sec": float(shank_config_elapsed_sec),
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
