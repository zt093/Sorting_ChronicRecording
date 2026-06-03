from __future__ import annotations

"""
Convert threshold-crossing chunk outputs into a wide CSV that LDA.py can read.

The converter treats each sgch*_thr*uV folder as one tracked unit. Each row is a
one-minute chunk, and each unit contributes firing rate, waveform summary
features, and the per-minute mean waveform samples.
"""

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


DEFAULT_INPUT_ROOT = Path(
    r"E:\Centimani\ChronicRecording4\threshold_crossings_outputs"
    r"\threshold_crossings_run_20260220_20260227_continuation"
)
DEFAULT_OUTPUT_NAME = "threshold_crossings_lda_population_vectors.csv"
CHRONIC_REC_RE = re.compile(r"Chronic_Rec_(?P<ymd>\d{8})_(?P<hms>\d{6})")
UNIT_FOLDER_RE = re.compile(r"^sgch(?P<sg>\d+)_thr(?P<thr>.+)uV$")


@dataclass(frozen=True)
class UnitFolder:
    tag: str
    path: Path
    sg_channel: int
    threshold_uv: float

    @property
    def sort_key(self) -> tuple[int, float, str]:
        return self.sg_channel, self.threshold_uv, self.tag


def log(message: str) -> None:
    print(f"[TC->LDA] {message}", flush=True)


def parse_unit_folder(path: Path) -> UnitFolder | None:
    match = UNIT_FOLDER_RE.match(path.name)
    if match is None:
        return None
    threshold_text = match.group("thr").replace("p", ".")
    try:
        threshold_uv = float(threshold_text)
    except ValueError:
        return None
    return UnitFolder(
        tag=path.name,
        path=path,
        sg_channel=int(match.group("sg")),
        threshold_uv=threshold_uv,
    )


def parse_recording_start_datetime(*texts: str) -> datetime | None:
    for text in texts:
        match = CHRONIC_REC_RE.search(str(text or ""))
        if match is None:
            continue
        return datetime.strptime(match.group("ymd") + match.group("hms"), "%Y%m%d%H%M%S")
    return None


def compute_cv2_from_seconds(timestamps_sec: np.ndarray) -> float:
    spike_times = np.asarray(timestamps_sec, dtype=float).ravel()
    spike_times = spike_times[np.isfinite(spike_times)]
    if spike_times.size < 3:
        return np.nan
    isi = np.diff(np.sort(spike_times))
    if isi.size < 2:
        return np.nan
    denominator = isi[:-1] + isi[1:]
    valid = denominator > 0
    if not np.any(valid):
        return np.nan
    cv2_values = 2.0 * np.abs(np.diff(isi)[valid]) / denominator[valid]
    return float(np.mean(cv2_values)) if cv2_values.size else np.nan


def trough_to_peak_ms(waveform: np.ndarray, sampling_rate_hz: float) -> float:
    waveform = np.asarray(waveform, dtype=float).ravel()
    if waveform.size < 2 or sampling_rate_hz <= 0:
        return np.nan
    trough_index = int(np.argmin(waveform))
    peak_index = int(np.argmax(waveform))
    if abs(waveform[trough_index]) >= abs(waveform[peak_index]):
        if trough_index >= waveform.size - 1:
            return np.nan
        post_peak_index = trough_index + int(np.argmax(waveform[trough_index:]))
        return float((post_peak_index - trough_index) / sampling_rate_hz * 1000.0)
    if peak_index >= waveform.size - 1:
        return np.nan
    post_trough_index = peak_index + int(np.argmin(waveform[peak_index:]))
    return float((post_trough_index - peak_index) / sampling_rate_hz * 1000.0)


def build_metadata_row(
    *,
    summary: dict,
    chunk: dict,
    session_name: str,
    session_start_datetime: datetime,
) -> dict[str, object]:
    chunk_index = int(chunk.get("chunk_index"))
    start_sec = float(chunk.get("time_start_sec", 0.0))
    end_sec = float(chunk.get("time_end_sec", start_sec))
    start_datetime = session_start_datetime + timedelta(seconds=start_sec)
    end_datetime = session_start_datetime + timedelta(seconds=end_sec)
    session_ordinal = int(summary.get("session_ordinal", 0))
    final_sample_key = f"{session_name}__chunk_{chunk_index:04d}"
    return {
        "final_sample_id": "",
        "final_sample_key": final_sample_key,
        "session_id": session_ordinal,
        "session_key": session_name,
        "session_name": session_name,
        "session_name_normalized": session_name,
        "session_index": session_ordinal,
        "session_start_datetime": session_start_datetime.isoformat(sep=" "),
        "minute_bin_index": chunk_index - 1,
        "minute_start_sec": start_sec,
        "minute_end_sec": end_sec,
        "minute_center_s": start_sec + (end_sec - start_sec) / 2.0,
        "session_duration_s": float(summary.get("seconds", np.nan)),
        "minute_start_datetime": start_datetime.isoformat(sep=" "),
        "minute_end_datetime": end_datetime.isoformat(sep=" "),
        "clock_hour_of_day": int(start_datetime.hour),
        "clock_minute_of_hour": int(start_datetime.minute),
        "calendar_day": start_datetime.date().isoformat(),
        "rec_file": str(summary.get("rec_file", "")),
    }


def summarize_chunk(npz_path: Path) -> tuple[dict[str, float], np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        timestamps_sec = np.asarray(data["timestamps_sec"], dtype=float).ravel()
        waveforms_uv = np.asarray(data["waveforms_uv"], dtype=float)
        sampling_rate = float(np.asarray(data["sampling_rate_hz"]).ravel()[0])
        time_start_sec = float(np.asarray(data["time_start_sec"]).ravel()[0])
        time_end_sec = float(np.asarray(data["time_end_sec"]).ravel()[0])

    duration_s = max(time_end_sec - time_start_sec, np.nan)
    n_crossings = int(timestamps_sec.size)
    if waveforms_uv.ndim == 2 and waveforms_uv.shape[0] > 0:
        mean_waveform = np.nanmean(waveforms_uv, axis=0)
        peak_to_peak = np.ptp(waveforms_uv, axis=1)
        average_amplitude = float(np.nanmean(np.abs(peak_to_peak)))
    else:
        mean_waveform = np.full(0, np.nan, dtype=float)
        average_amplitude = np.nan

    features = {
        "firing_rate_hz": float(n_crossings / duration_s) if duration_s > 0 else np.nan,
        "average_amplitude_uv": average_amplitude,
        "cv2": compute_cv2_from_seconds(timestamps_sec),
        "peak_to_trough_ms": trough_to_peak_ms(mean_waveform, sampling_rate),
        "n_crossings": float(n_crossings),
    }
    return features, np.asarray(mean_waveform, dtype=float).ravel()


def iter_summary_payloads(unit_folder: UnitFolder) -> list[tuple[Path, dict]]:
    rows: list[tuple[Path, dict]] = []
    for summary_path in sorted(unit_folder.path.glob("*_recording_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append((summary_path, payload))
    return rows


def convert_threshold_crossings(
    input_root: Path,
    output_csv: Path,
    *,
    max_summaries_per_unit: int | None = None,
    max_chunks_per_summary: int | None = None,
) -> Path:
    input_root = input_root.resolve()
    unit_folders = [
        parsed
        for parsed in (parse_unit_folder(path) for path in input_root.iterdir() if path.is_dir())
        if parsed is not None
    ]
    unit_folders = sorted(unit_folders, key=lambda unit: unit.sort_key)
    if not unit_folders:
        raise FileNotFoundError(f"No sgch*_thr*uV folders found under {input_root}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    log(f"Found {len(unit_folders)} threshold-crossing unit folders")

    metadata_by_key: dict[str, dict[str, object]] = {}
    feature_rows_by_key: dict[str, dict[str, float]] = {}
    feature_columns: list[str] = []
    feature_column_seen: set[str] = set()
    expected_waveform_len: int | None = None
    processed_chunks = 0
    start_time = time.perf_counter()

    for unit_index, unit_folder in enumerate(unit_folders, start=1):
        summary_rows = iter_summary_payloads(unit_folder)
        if max_summaries_per_unit is not None:
            summary_rows = summary_rows[: max_summaries_per_unit]
        log(
            f"Reading {unit_folder.tag} ({unit_index}/{len(unit_folders)}): "
            f"{len(summary_rows)} recording summaries"
        )

        for summary_path, summary in summary_rows:
            session_start = parse_recording_start_datetime(
                str(summary.get("rec_file", "")),
                summary_path.name,
            )
            if session_start is None:
                raise ValueError(f"Could not parse recording datetime from {summary_path}")
            session_name = f"{summary_path.stem.removesuffix('_recording_summary')}"
            chunks = list(summary.get("chunks", []))
            if max_chunks_per_summary is not None:
                chunks = chunks[: max_chunks_per_summary]

            for chunk in chunks:
                npz_path = Path(str(chunk.get("npz", "")))
                if not npz_path.exists():
                    npz_path = unit_folder.path / npz_path.name
                if not npz_path.exists():
                    raise FileNotFoundError(f"Chunk npz not found: {chunk.get('npz')}")

                metadata = build_metadata_row(
                    summary=summary,
                    chunk=chunk,
                    session_name=session_name,
                    session_start_datetime=session_start,
                )
                row_key = str(metadata["final_sample_key"])
                if row_key not in metadata_by_key:
                    metadata_by_key[row_key] = metadata
                    feature_rows_by_key[row_key] = {}

                chunk_features, mean_waveform = summarize_chunk(npz_path)
                if expected_waveform_len is None and mean_waveform.size > 0:
                    expected_waveform_len = int(mean_waveform.size)
                waveform_len = expected_waveform_len or int(mean_waveform.size)
                if mean_waveform.size < waveform_len:
                    padded = np.full(waveform_len, np.nan, dtype=float)
                    padded[: mean_waveform.size] = mean_waveform
                    mean_waveform = padded

                row = feature_rows_by_key[row_key]
                for feature_name, value in chunk_features.items():
                    column = f"{unit_folder.tag}__{feature_name}"
                    row[column] = value
                    if column not in feature_column_seen:
                        feature_column_seen.add(column)
                        feature_columns.append(column)
                for sample_index, value in enumerate(mean_waveform[:waveform_len]):
                    column = f"{unit_folder.tag}__mean_waveform_uv_s{sample_index:03d}"
                    row[column] = float(value)
                    if column not in feature_column_seen:
                        feature_column_seen.add(column)
                        feature_columns.append(column)

                processed_chunks += 1

    metadata_columns = [
        "final_sample_id",
        "final_sample_key",
        "session_id",
        "session_key",
        "session_name",
        "session_name_normalized",
        "session_index",
        "session_start_datetime",
        "minute_bin_index",
        "minute_start_sec",
        "minute_end_sec",
        "minute_center_s",
        "session_duration_s",
        "minute_start_datetime",
        "minute_end_datetime",
        "clock_hour_of_day",
        "clock_minute_of_hour",
        "calendar_day",
        "rec_file",
    ]
    sorted_keys = sorted(
        metadata_by_key,
        key=lambda key: (
            str(metadata_by_key[key]["minute_start_datetime"]),
            int(metadata_by_key[key]["session_id"]),
            int(metadata_by_key[key]["minute_bin_index"]),
        ),
    )
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_columns + feature_columns)
        writer.writeheader()
        for sample_index, row_key in enumerate(sorted_keys, start=1):
            output_row = dict(metadata_by_key[row_key])
            output_row["final_sample_id"] = sample_index
            output_row.update(feature_rows_by_key[row_key])
            writer.writerow(output_row)

    elapsed_s = time.perf_counter() - start_time
    log(
        f"Wrote {len(sorted_keys)} samples x {len(feature_columns)} feature columns to {output_csv}"
    )
    log(
        f"Processed {processed_chunks} unit-chunks in {elapsed_s:.1f}s "
        f"({processed_chunks / max(elapsed_s, 1e-9):.1f} unit-chunks/s)"
    )
    return output_csv


def estimate_runtime(input_root: Path, sample_summaries_per_unit: int, sample_chunks_per_summary: int) -> None:
    unit_folders = [
        parsed
        for parsed in (parse_unit_folder(path) for path in input_root.iterdir() if path.is_dir())
        if parsed is not None
    ]
    unit_folders = sorted(unit_folders, key=lambda unit: unit.sort_key)
    total_chunks = 0
    for unit_folder in unit_folders:
        for _, summary in iter_summary_payloads(unit_folder):
            total_chunks += len(summary.get("chunks", []))

    temporary_output = input_root / "_lda_converter_timing_sample.csv"
    start = time.perf_counter()
    convert_threshold_crossings(
        input_root=input_root,
        output_csv=temporary_output,
        max_summaries_per_unit=sample_summaries_per_unit,
        max_chunks_per_summary=sample_chunks_per_summary,
    )
    elapsed_s = time.perf_counter() - start
    temporary_output.unlink(missing_ok=True)
    sampled_chunks = min(len(unit_folders), len(unit_folders)) * sample_summaries_per_unit * sample_chunks_per_summary
    rate = sampled_chunks / max(elapsed_s, 1e-9)
    estimated_s = total_chunks / max(rate, 1e-9)
    log(
        f"Timing sample: {sampled_chunks} unit-chunks in {elapsed_s:.1f}s. "
        f"Full dataset has {total_chunks} unit-chunks; estimated full conversion "
        f"time is {estimated_s / 60.0:.1f} minutes."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-summaries-per-unit", type=int, default=None)
    parser.add_argument("--max-chunks-per-summary", type=int, default=None)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--estimate-summaries-per-unit", type=int, default=2)
    parser.add_argument("--estimate-chunks-per-summary", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv or (args.input_root / DEFAULT_OUTPUT_NAME)
    if args.estimate_only:
        estimate_runtime(
            input_root=args.input_root,
            sample_summaries_per_unit=args.estimate_summaries_per_unit,
            sample_chunks_per_summary=args.estimate_chunks_per_summary,
        )
        return
    convert_threshold_crossings(
        input_root=args.input_root,
        output_csv=output_csv,
        max_summaries_per_unit=args.max_summaries_per_unit,
        max_chunks_per_summary=args.max_chunks_per_summary,
    )


if __name__ == "__main__":
    main()
