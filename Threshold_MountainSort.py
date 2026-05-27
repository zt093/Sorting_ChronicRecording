"""
Hybrid threshold detector with MountainSort-style preprocessing, then stop.

This script keeps the threshold-crossing workflow from Threshold_channel.py, but
adds the preprocessing front end used by the MountainSort pipeline:

  read SpikeGadgets -> HW map/probe -> common median reference -> spikeband
  bandpass -> optional large-artifact blanking -> threshold detection outputs

It does not run MountainSort clustering/analyzer steps. The goal is to test
whether the MountainSort preprocessing improves channel-threshold detections
while preserving the threshold-based summaries and timing reports.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def _remove_pythonpath_entries_from_sys_path() -> None:
    """
    Keep external PYTHONPATH packages from shadowing the Conda environment.

    Threshold_channel.py does this before importing SpikeInterface/probeinterface;
    this script needs the same guard because read_spikegadgets depends on the
    probeinterface version installed in the active SpikeInterface environment.
    """
    pythonpath = os.environ.get("PYTHONPATH", "")
    if not pythonpath:
        return
    blocked: set[str] = set()
    for entry in pythonpath.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        try:
            blocked.add(str(Path(entry).resolve()).casefold())
        except Exception:
            blocked.add(entry.casefold())
    if not blocked:
        return

    kept: list[str] = []
    for entry in sys.path:
        if not entry:
            kept.append(entry)
            continue
        try:
            resolved = str(Path(entry).resolve()).casefold()
        except Exception:
            resolved = entry.casefold()
        if resolved not in blocked:
            kept.append(entry)
    sys.path[:] = kept


_remove_pythonpath_entries_from_sys_path()

import numpy as np
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
from probeinterface import read_probeinterface

from Threshold_channel import (
    DEFAULT_BANDPASS_FREQ_MAX,
    DEFAULT_BANDPASS_FREQ_MIN,
    DEFAULT_CHANNEL_THRESHOLD_CONFIG_JSON,
    DEFAULT_CHUNK_SEC,
    DEFAULT_OUTPUT_PARENT,
    DEFAULT_POLARITY,
    DEFAULT_POST_MS,
    DEFAULT_PRE_MS,
    DEFAULT_PROBE_JSON,
    DEFAULT_REFRACTORY_MS,
    DEFAULT_SAMPLING_RATE_HZ,
    _add_timing,
    _format_elapsed,
    _print_per_recording_timing_summary,
    _print_timing_summary,
    _recording_timing_row,
    _safe_float,
    _write_run_timing_reports,
    build_sg_to_recording_index,
    chronic_rec_sort_key,
    collect_recording_files_from_inputs,
    load_channel_threshold_pairs,
    load_recording_mapped,
    parse_input_paths,
    process_recording_save_per_chunk_multi_channel,
    prompt_path,
    prompt_yes_no,
)


DEFAULT_MS_COMMON_REFERENCE = True
DEFAULT_MS_BLANK_BIG_ARTIFACTS = True
DEFAULT_MS_OUTPUT_PARENT = Path(r"S:\Threshold_MS")
DEFAULT_BIG_NOISE_NEGATIVE_THRESHOLD_UV = -2500.0
DEFAULT_BIG_NOISE_POSITIVE_THRESHOLD_UV = 1000.0
DEFAULT_BIG_NOISE_MARGIN_SAMPLES = 25
DEFAULT_BIG_NOISE_BLANK_ALL_CHANNELS = True


def format_elapsed_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.2f} seconds"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.2f} minutes"
    return f"{seconds / 3600.0:.2f} hours"


def print_step_start(step_name: str) -> float:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] starting {step_name}...", flush=True)
    return time.perf_counter()


def print_step_done(step_name: str, start_time: float) -> float:
    elapsed = time.perf_counter() - start_time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] finished {step_name}: {format_elapsed_time(elapsed)}", flush=True)
    return elapsed


def make_ms_threshold_run_output_dir(output_parent: Path, rec_files: list[Path]) -> Path:
    output_parent = Path(output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)
    dates = []
    for rec_file in rec_files:
        key = chronic_rec_sort_key(rec_file)
        if key is not None:
            dates.append(str(key)[:8][2:])
    if not dates:
        rec_label = "unknown_rec_date"
    else:
        unique_dates = sorted(set(dates))
        rec_label = unique_dates[0] if len(unique_dates) == 1 else f"{unique_dates[0]}_{unique_dates[-1]}"
    run_label = datetime.now().strftime("%y%m%d")
    stem = f"threshold_mountainsort_{rec_label}_run_{run_label}"
    run_dir = output_parent / stem
    if run_dir.exists():
        suffix = 1
        while True:
            candidate = output_parent / f"{stem}_{suffix}"
            if not candidate.exists():
                run_dir = candidate
                break
            suffix += 1
    run_dir.mkdir(parents=False)
    return run_dir


def materialize_recording_as_numpy(recording):
    """Convert a lazy SpikeInterface recording to NumpyRecording, preserving metadata."""
    traces = np.asarray(recording.get_traces(return_scaled=True), dtype=np.float32)
    channel_ids = list(recording.get_channel_ids())
    numpy_recording = si.NumpyRecording(
        traces_list=[traces],
        sampling_frequency=float(recording.get_sampling_frequency()),
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
            numpy_recording = numpy_recording.set_probe(recording.get_probe(), in_place=False)
    except Exception:
        pass
    return numpy_recording


def remove_big_noise_artifacts(
    recording,
    *,
    negative_threshold_uv: float,
    positive_threshold_uv: float,
    margin_samples: int,
    blank_all_channels: bool,
):
    """
    Copy the MountainSort pipeline's large-artifact blanking idea.

    This materializes the recording, so it is optional and best used when the
    available RAM can hold one full preprocessed recording.
    """
    traces = np.asarray(recording.get_traces(return_scaled=True), dtype=np.float32)
    if traces.ndim != 2:
        raise ValueError(f"Expected 2D traces array, got shape {traces.shape}")

    artifact_mask = (traces < float(negative_threshold_uv)) | (traces > float(positive_threshold_uv))
    event_count = int(np.count_nonzero(artifact_mask))
    if event_count == 0:
        metadata = {
            "negative_threshold_uv": float(negative_threshold_uv),
            "positive_threshold_uv": float(positive_threshold_uv),
            "margin_samples": int(margin_samples),
            "blank_all_channels": bool(blank_all_channels),
            "num_threshold_crossings": 0,
            "num_masked_samples": 0,
            "num_masked_segments": 0,
            "masked_fraction": 0.0,
        }
        return materialize_recording_as_numpy(recording), metadata

    sample_hits = np.any(artifact_mask, axis=1)
    channel_hits = artifact_mask.any(axis=0)
    masked_channel_indices = np.flatnonzero(channel_hits)

    expanded_mask = np.zeros(sample_hits.shape[0], dtype=bool)
    for hit_index in np.flatnonzero(sample_hits):
        start = max(0, int(hit_index) - int(margin_samples))
        end = min(sample_hits.shape[0], int(hit_index) + int(margin_samples) + 1)
        expanded_mask[start:end] = True

    cleaned_traces = traces.copy()
    if blank_all_channels:
        cleaned_traces[expanded_mask, :] = 0.0
    else:
        cleaned_traces[artifact_mask] = 0.0
        for hit_index in np.flatnonzero(sample_hits):
            start = max(0, int(hit_index) - int(margin_samples))
            end = min(sample_hits.shape[0], int(hit_index) + int(margin_samples) + 1)
            affected_channels = np.flatnonzero(artifact_mask[hit_index])
            if affected_channels.size:
                cleaned_traces[start:end, affected_channels] = 0.0

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
        channel_ids=list(recording.get_channel_ids()),
    )
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
        "negative_threshold_uv": float(negative_threshold_uv),
        "positive_threshold_uv": float(positive_threshold_uv),
        "margin_samples": int(margin_samples),
        "blank_all_channels": bool(blank_all_channels),
        "num_threshold_crossings": int(event_count),
        "num_masked_samples": int(np.count_nonzero(expanded_mask)),
        "num_masked_segments": int(len(artifact_segments)),
        "masked_fraction": float(np.count_nonzero(expanded_mask) / traces.shape[0]),
        "masked_channel_indices": [int(idx) for idx in masked_channel_indices.tolist()],
        "artifact_segments_preview": artifact_segments[:50],
    }
    return cleaned_recording, metadata


def apply_mountainsort_frontend(
    recording,
    *,
    common_reference: bool,
    bandpass_freq_min: float,
    bandpass_freq_max: float,
    blank_big_artifacts: bool,
    negative_threshold_uv: float,
    positive_threshold_uv: float,
    margin_samples: int,
    blank_all_channels: bool,
    timing_totals: dict[str, float] | None = None,
):
    metadata = {
        "pipeline": "mountainsort_frontend_then_threshold_stop",
        "steps": [],
        "note": (
            "Uses MountainSort-style preprocessing before threshold detection. "
            "Whitening and clustering are intentionally skipped so threshold_uv remains in uV."
        ),
    }

    rec = recording
    if common_reference:
        t0 = print_step_start("MountainSort frontend: common median reference")
        rec = spre.common_reference(rec, operator="median", reference="global")
        elapsed = print_step_done("MountainSort frontend: common median reference", t0)
        _add_timing(timing_totals, "ms_common_reference", elapsed)
        metadata["steps"].append("common_reference")
        metadata["common_reference"] = {"operator": "median", "reference": "global"}

    t0 = print_step_start("MountainSort frontend: spikeband bandpass")
    rec = spre.bandpass_filter(
        rec,
        freq_min=float(bandpass_freq_min),
        freq_max=float(bandpass_freq_max),
        dtype="float32",
    )
    elapsed = print_step_done("MountainSort frontend: spikeband bandpass", t0)
    _add_timing(timing_totals, "ms_bandpass_filter", elapsed)
    metadata["steps"].append("bandpass_filter")
    metadata["bandpass_filter"] = {
        "freq_min": float(bandpass_freq_min),
        "freq_max": float(bandpass_freq_max),
        "dtype": "float32",
    }

    if blank_big_artifacts:
        t0 = print_step_start("MountainSort frontend: large-artifact blanking")
        rec, noise_metadata = remove_big_noise_artifacts(
            rec,
            negative_threshold_uv=float(negative_threshold_uv),
            positive_threshold_uv=float(positive_threshold_uv),
            margin_samples=int(margin_samples),
            blank_all_channels=bool(blank_all_channels),
        )
        elapsed = print_step_done("MountainSort frontend: large-artifact blanking", t0)
        _add_timing(timing_totals, "ms_big_artifact_blanking", elapsed)
        metadata["steps"].append("remove_big_noise_artifacts")
        metadata["remove_big_noise_artifacts"] = noise_metadata
    else:
        metadata["remove_big_noise_artifacts"] = None

    return rec, metadata


def process_threshold_mountainsort_run(
    *,
    run_output_dir: Path,
    meta_run: dict,
    rec_files: list[Path],
    fs: float,
    probe_json: Path,
    channel_threshold_pairs: list[dict],
    polarity: str,
    chunk_samples: int,
    refractory_samples: int,
    pre_samples: int,
    post_samples: int,
    bandpass_freq_min: float,
    bandpass_freq_max: float,
    resume: bool,
    common_reference: bool,
    blank_big_artifacts: bool,
    negative_threshold_uv: float,
    positive_threshold_uv: float,
    artifact_margin_samples: int,
    artifact_blank_all_channels: bool,
) -> int:
    t0_all = time.perf_counter()
    timing_totals: dict[str, float] = {}
    per_recording_timings: list[dict] = []
    n_files = len(rec_files)

    print(f"\nProcessing {n_files} recording(s) with MountainSort frontend, then threshold stop.\n", flush=True)
    print(
        "Progress lines include real wall-clock timestamps and elapsed seconds. "
        "Stable outputs follow Threshold_channel.py's minute/hourly summaries.\n",
        flush=True,
    )

    cumulative_sample_offset = 0
    cumulative_time_offset_sec = 0.0
    session_ordinal = 0

    for fi, rec_file in enumerate(rec_files):
        t_rec = time.perf_counter()
        timing_before = dict(timing_totals)
        print("\n" + "=" * 72, flush=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{fi + 1}/{n_files}] {rec_file}", flush=True)
        print("=" * 72, flush=True)

        if resume:
            all_pairs_complete = True
            for pair in channel_threshold_pairs:
                from Threshold_channel import _is_recording_summary_complete, _pair_out_base, _recording_summary_path_from_out_base

                out_base = _pair_out_base(
                    run_output_dir,
                    rec_file,
                    int(pair["sg_ch"]),
                    float(pair["threshold_uv"]),
                    _safe_float(pair.get("threshold_max_uv")),
                )
                if not _is_recording_summary_complete(_recording_summary_path_from_out_base(out_base)):
                    all_pairs_complete = False
                    break
            if all_pairs_complete:
                t_skip = time.perf_counter()
                rec_for_duration = load_recording_mapped(rec_file, fs, probe_json)
                n_samp = int(rec_for_duration.get_num_samples())
                dur_s = float(n_samp / fs)
                del rec_for_duration
                gc.collect()
                _add_timing(timing_totals, "resume_skip_complete_recording", time.perf_counter() - t_skip)
                per_recording_timings.append(
                    _recording_timing_row(
                        recording_index=fi + 1,
                        recording_count=n_files,
                        rec_file=rec_file,
                        status="skipped_complete",
                        wall_seconds=time.perf_counter() - t_rec,
                        timing_before=timing_before,
                        timing_after=timing_totals,
                        n_samples=n_samp,
                        duration_seconds=dur_s,
                        pairs_processed=len(channel_threshold_pairs),
                        pairs_total=len(channel_threshold_pairs),
                    )
                )
                cumulative_sample_offset += n_samp
                cumulative_time_offset_sec += dur_s
                session_ordinal += 1
                print(f"  [skip] all pairs complete; advanced timeline by {_format_elapsed(dur_s)}.", flush=True)
                continue

        t0 = print_step_start("load recording + hardware map + probe")
        rec_raw = load_recording_mapped(rec_file, fs, probe_json)
        elapsed = print_step_done("load recording + hardware map + probe", t0)
        _add_timing(timing_totals, "load_recording", elapsed)

        n_samp = int(rec_raw.get_num_samples())
        dur_s = float(n_samp / fs)
        chan_ids = list(rec_raw.get_channel_ids())
        print(
            f"  Recording duration {dur_s:.2f} s ({n_samp} samples @ {fs:.0f} Hz); "
            f"{len(chan_ids)} channels.",
            flush=True,
        )

        rec, preprocessing_metadata = apply_mountainsort_frontend(
            rec_raw,
            common_reference=common_reference,
            bandpass_freq_min=bandpass_freq_min,
            bandpass_freq_max=bandpass_freq_max,
            blank_big_artifacts=blank_big_artifacts,
            negative_threshold_uv=negative_threshold_uv,
            positive_threshold_uv=positive_threshold_uv,
            margin_samples=artifact_margin_samples,
            blank_all_channels=artifact_blank_all_channels,
            timing_totals=timing_totals,
        )
        meta_run["preprocessing"] = preprocessing_metadata
        (run_output_dir / "run_config.json").write_text(json.dumps(meta_run, indent=2), encoding="utf-8")

        t0 = print_step_start("threshold detection + minute/hourly outputs")
        pairs_processed = process_recording_save_per_chunk_multi_channel(
            rec,
            rec_file,
            chan_ids,
            run_output_dir=run_output_dir,
            meta_run=meta_run,
            channel_threshold_pairs=channel_threshold_pairs,
            session_ordinal=session_ordinal,
            session_cumulative_sample_offset=int(cumulative_sample_offset),
            session_cumulative_time_offset_sec=float(cumulative_time_offset_sec),
            fs=fs,
            chunk_samples=chunk_samples,
            polarity=polarity,
            refractory_samples=refractory_samples,
            pre_samples=pre_samples,
            post_samples=post_samples,
            resume=resume,
            progress=True,
            progress_prefix="  ",
            timing_totals=timing_totals,
        )
        print_step_done("threshold detection + minute/hourly outputs", t0)

        wall = time.perf_counter() - t_rec
        per_recording_timings.append(
            _recording_timing_row(
                recording_index=fi + 1,
                recording_count=n_files,
                rec_file=rec_file,
                status="processed",
                wall_seconds=wall,
                timing_before=timing_before,
                timing_after=timing_totals,
                n_samples=n_samp,
                duration_seconds=dur_s,
                pairs_processed=pairs_processed,
                pairs_total=len(channel_threshold_pairs),
            )
        )
        print(
            f"  Recording done: {pairs_processed}/{len(channel_threshold_pairs)} pair(s) "
            f"in {format_elapsed_time(wall)} wall-clock.",
            flush=True,
        )
        cumulative_sample_offset += n_samp
        cumulative_time_offset_sec += dur_s
        session_ordinal += 1
        del rec, rec_raw
        gc.collect()

    total_wall = time.perf_counter() - t0_all
    _write_run_timing_reports(run_output_dir, timing_totals, per_recording_timings, total_wall)
    print(f"\nAll recordings finished in {format_elapsed_time(total_wall)}.", flush=True)
    print(f"Outputs: {run_output_dir.resolve()}", flush=True)
    _print_per_recording_timing_summary(per_recording_timings)
    _print_timing_summary(timing_totals, total_wall)
    return 0


def main() -> int:
    print("=== Threshold crossings with MountainSort preprocessing, then stop ===\n", flush=True)

    resume_prev = prompt_yes_no("Resume previous interrupted Threshold_MountainSort session?", default_yes=False)
    if resume_prev:
        run_dir = prompt_path("Directory of threshold_mountainsort_* run folder", "")
        run_config_path = run_dir / "run_config.json"
        if not run_config_path.exists():
            print(f"Missing run_config.json in: {run_dir}", file=sys.stderr)
            return 1
        meta_run = json.loads(run_config_path.read_text(encoding="utf-8"))
        rec_files = [Path(p) for p in meta_run["recording_files"]]
        channel_threshold_pairs = meta_run["channel_threshold_pairs"]
        preprocessing = meta_run.get("preprocessing", {})
        artifact_cfg = preprocessing.get("remove_big_noise_artifacts") or {}
        return process_threshold_mountainsort_run(
            run_output_dir=Path(meta_run["run_output_dir"]),
            meta_run=meta_run,
            rec_files=rec_files,
            fs=float(meta_run["sampling_rate_hz"]),
            probe_json=Path(meta_run["probe_json"]),
            channel_threshold_pairs=channel_threshold_pairs,
            polarity=str(meta_run["polarity"]),
            chunk_samples=int(meta_run["chunk_samples"]),
            refractory_samples=int(meta_run["refractory_samples"]),
            pre_samples=int(meta_run["pre_samples"]),
            post_samples=int(meta_run["post_samples"]),
            bandpass_freq_min=float(preprocessing.get("bandpass_filter", {}).get("freq_min", DEFAULT_BANDPASS_FREQ_MIN)),
            bandpass_freq_max=float(preprocessing.get("bandpass_filter", {}).get("freq_max", DEFAULT_BANDPASS_FREQ_MAX)),
            resume=True,
            common_reference="common_reference" in preprocessing.get("steps", []),
            blank_big_artifacts="remove_big_noise_artifacts" in preprocessing.get("steps", []),
            negative_threshold_uv=float(artifact_cfg.get("negative_threshold_uv", DEFAULT_BIG_NOISE_NEGATIVE_THRESHOLD_UV)),
            positive_threshold_uv=float(artifact_cfg.get("positive_threshold_uv", DEFAULT_BIG_NOISE_POSITIVE_THRESHOLD_UV)),
            artifact_margin_samples=int(artifact_cfg.get("margin_samples", DEFAULT_BIG_NOISE_MARGIN_SAMPLES)),
            artifact_blank_all_channels=bool(artifact_cfg.get("blank_all_channels", DEFAULT_BIG_NOISE_BLANK_ALL_CHANNELS)),
        )

    rec_input = input(
        "Recording .rec file/folder path(s), separated by semicolons "
        r"[W:\260220_rec\Chronic_Rec_20260220_201706.rec]: "
    ).strip()
    if not rec_input:
        rec_input = r"W:\260220_rec\Chronic_Rec_20260220_201706.rec"
    try:
        recording_inputs = parse_input_paths(rec_input)
        rec_files = collect_recording_files_from_inputs(recording_inputs)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Will process {len(rec_files)} recording(s).", flush=True)
    for i, rec_file in enumerate(rec_files[:5], start=1):
        print(f"  {i}. {rec_file}", flush=True)
    if len(rec_files) > 5:
        print(f"  ... and {len(rec_files) - 5} more", flush=True)

    probe_json = DEFAULT_PROBE_JSON
    if not probe_json.exists():
        print(f"Probe file not found: {probe_json}", file=sys.stderr)
        return 1

    fs = float(DEFAULT_SAMPLING_RATE_HZ)
    chunk_sec = float(DEFAULT_CHUNK_SEC)
    chunk_samples = max(1000, int(round(chunk_sec * fs)))
    pre_samples = max(1, int(round(DEFAULT_PRE_MS * fs / 1000.0)))
    post_samples = max(1, int(round(DEFAULT_POST_MS * fs / 1000.0)))
    refractory_samples = max(1, int(round(DEFAULT_REFRACTORY_MS * fs / 1000.0)))

    config_path = DEFAULT_CHANNEL_THRESHOLD_CONFIG_JSON
    try:
        pairs = load_channel_threshold_pairs(config_path)
    except Exception as exc:
        print(f"Failed to load channel/threshold JSON: {exc}", file=sys.stderr)
        return 1

    probe_group = read_probeinterface(str(probe_json))
    sg_map = build_sg_to_recording_index(probe_group.probes[0])
    channel_threshold_pairs = []
    for pair in pairs:
        sg_ch = int(pair["sg_ch"])
        if sg_ch not in sg_map:
            print(f"sg_ch={sg_ch} not present in probe map.", file=sys.stderr)
            return 1
        channel_threshold_pairs.append(
            {
                "sg_ch": sg_ch,
                "threshold_uv": float(pair["threshold_uv"]),
                "threshold_max_uv": _safe_float(pair.get("threshold_max_uv")),
                "threshold_interval": str(pair.get("threshold_interval", "lower_inclusive_upper_exclusive")),
            }
        )

    print(
        f"Using {len(channel_threshold_pairs)} channel/threshold pair(s) from {config_path}.",
        flush=True,
    )
    print(
        "MountainSort frontend defaults: common median reference ON, "
        f"bandpass {DEFAULT_BANDPASS_FREQ_MIN:.0f}-{DEFAULT_BANDPASS_FREQ_MAX:.0f} Hz, "
        f"large-artifact blanking {'ON' if DEFAULT_MS_BLANK_BIG_ARTIFACTS else 'OFF'}.",
        flush=True,
    )
    blank_big_artifacts = prompt_yes_no(
        "Use MountainSort-style large-artifact blanking? This materializes each full recording in RAM",
        default_yes=DEFAULT_MS_BLANK_BIG_ARTIFACTS,
    )

    run_output_dir = make_ms_threshold_run_output_dir(DEFAULT_MS_OUTPUT_PARENT, rec_files)
    print(f"\nRun output folder: {run_output_dir.resolve()}", flush=True)

    preprocessing_steps = ["bandpass_filter"]
    if DEFAULT_MS_COMMON_REFERENCE:
        preprocessing_steps.insert(0, "common_reference")
    if blank_big_artifacts:
        preprocessing_steps.append("remove_big_noise_artifacts")

    initial_preprocessing = {
        "pipeline": "mountainsort_frontend_then_threshold_stop",
        "steps": preprocessing_steps,
        "common_reference": (
            {"operator": "median", "reference": "global"}
            if DEFAULT_MS_COMMON_REFERENCE
            else None
        ),
        "bandpass_filter": {
            "freq_min": float(DEFAULT_BANDPASS_FREQ_MIN),
            "freq_max": float(DEFAULT_BANDPASS_FREQ_MAX),
            "dtype": "float32",
        },
        "remove_big_noise_artifacts": (
            {
                "negative_threshold_uv": float(DEFAULT_BIG_NOISE_NEGATIVE_THRESHOLD_UV),
                "positive_threshold_uv": float(DEFAULT_BIG_NOISE_POSITIVE_THRESHOLD_UV),
                "margin_samples": int(DEFAULT_BIG_NOISE_MARGIN_SAMPLES),
                "blank_all_channels": bool(DEFAULT_BIG_NOISE_BLANK_ALL_CHANNELS),
            }
            if blank_big_artifacts
            else None
        ),
        "stop_before": "mountainsort5_run_sorter",
        "why_stop": "Threshold comparison only; no clustering/analyzer output is produced.",
    }

    meta_run = {
        "run_output_dir": str(run_output_dir.resolve()),
        "output_parent": str(DEFAULT_MS_OUTPUT_PARENT.resolve()),
        "recording_input": rec_input,
        "recording_inputs": [str(path.resolve()) for path in recording_inputs],
        "n_files": len(rec_files),
        "recording_files": [str(path.resolve()) for path in rec_files],
        "channel_threshold_mode": "json_channel_threshold_pairs",
        "channel_threshold_config": str(config_path.resolve()),
        "channel_threshold_pairs": channel_threshold_pairs,
        "polarity": DEFAULT_POLARITY,
        "chunk_sec": chunk_sec,
        "chunk_samples": chunk_samples,
        "sampling_rate_hz": fs,
        "pre_ms": DEFAULT_PRE_MS,
        "post_ms": DEFAULT_POST_MS,
        "pre_samples": pre_samples,
        "post_samples": post_samples,
        "refractory_ms": DEFAULT_REFRACTORY_MS,
        "refractory_samples": refractory_samples,
        "probe_json": str(probe_json.resolve()),
        "preprocessing": initial_preprocessing,
        "saved_files_note": (
            "Same threshold output family as Threshold_channel.py: recording-level "
            "minute NPZs, per-pair minute/hourly CSV/JSON, hourly waveforms, and timing reports."
        ),
    }
    (run_output_dir / "run_config.json").write_text(json.dumps(meta_run, indent=2), encoding="utf-8")

    return process_threshold_mountainsort_run(
        run_output_dir=run_output_dir,
        meta_run=meta_run,
        rec_files=rec_files,
        fs=fs,
        probe_json=probe_json,
        channel_threshold_pairs=channel_threshold_pairs,
        polarity=DEFAULT_POLARITY,
        chunk_samples=chunk_samples,
        refractory_samples=refractory_samples,
        pre_samples=pre_samples,
        post_samples=post_samples,
        bandpass_freq_min=DEFAULT_BANDPASS_FREQ_MIN,
        bandpass_freq_max=DEFAULT_BANDPASS_FREQ_MAX,
        resume=False,
        common_reference=DEFAULT_MS_COMMON_REFERENCE,
        blank_big_artifacts=blank_big_artifacts,
        negative_threshold_uv=DEFAULT_BIG_NOISE_NEGATIVE_THRESHOLD_UV,
        positive_threshold_uv=DEFAULT_BIG_NOISE_POSITIVE_THRESHOLD_UV,
        artifact_margin_samples=DEFAULT_BIG_NOISE_MARGIN_SAMPLES,
        artifact_blank_all_channels=DEFAULT_BIG_NOISE_BLANK_ALL_CHANNELS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
