from __future__ import annotations

"""
Materialize Threshold_channel.py outputs into a precomputed LDA population CSV.

This converter is meant to run after:

    Sorting_Check/Threshold_channel.py

and before either:

    Sorting_Check/Stats/Threshold_LDA_TuningWN_pre.py
    Sorting_Check/Stats/LDA_weinan.py

It reads the per-pair *_minute_summary.csv files written by Threshold_channel.py
and writes the same minute-level wide CSV and manifest that
Threshold_LDA_TuningWN_pre.py normally builds internally. If the CSV is written
to the Threshold_LDA_TuningWN_pre output directory, that wrapper can reuse it
instead of rebuilding it.
"""

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT_SUBDIR = "threshold_csvs"
DEFAULT_OUTPUT_NAME = "threshold_population_minute_features.csv"
MANIFEST_NAME = "threshold_population_manifest.json"
UNITS_SUMMARY_NAME = "threshold_units_used_summary.csv"
PORTABLE_RUN_CONFIG_NAME = "run_config.json"
CHRONIC_REC_RE = re.compile(r"Chronic_Rec_(?P<ymd>\d{8})_(?P<hms>\d{6})", re.IGNORECASE)
PAIR_FOLDER_RE = re.compile(r"^sgch(?P<sg>\d+)_thr(?P<thr>.+)uV$", re.IGNORECASE)
FEATURE_TYPES = (
    "firing_rate_hz",
    "average_amplitude_uv",
    "cv2",
    "peak_to_trough_ms",
)
METADATA_COLUMNS = [
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
    "threshold_run_root",
    "threshold_run_name",
]


@dataclass(frozen=True)
class PairFolder:
    unit_key: str
    sg_ch: int
    threshold_uv: float
    threshold_label: str
    path: Path

    @property
    def sort_key(self) -> tuple[int, float, str]:
        return self.sg_ch, self.threshold_uv, self.unit_key


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = format_duration(time.perf_counter() - SCRIPT_START_TIME)
    print(f"[Threshold CSV {now} +{elapsed}] {message}", flush=True)


SCRIPT_START_TIME = time.perf_counter()


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {remaining_seconds:05.2f}s"
    if minutes > 0:
        return f"{minutes:d}m {remaining_seconds:05.2f}s"
    return f"{remaining_seconds:.2f}s"


def safe_slug(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "value"


def safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    except Exception:
        return None


def parse_recording_start_datetime(recording_name: object) -> datetime | None:
    match = CHRONIC_REC_RE.search(str(recording_name or ""))
    if match is None:
        return None
    return datetime.strptime(match.group("ymd") + match.group("hms"), "%Y%m%d%H%M%S")


def threshold_min_from_unit_key(unit_key: str, fallback: float = np.nan) -> float:
    match = re.search(r"_thr(?P<thr>.+)uV", str(unit_key))
    if match is None:
        return float(fallback)
    threshold_text = match.group("thr").replace("p", ".")
    threshold_min_text = threshold_text.split("to", 1)[0]
    try:
        return float(threshold_min_text)
    except ValueError:
        return float(fallback)


def threshold_label_from_unit_key(unit_key: str) -> str:
    match = re.search(r"_thr(?P<thr>.+)uV", str(unit_key))
    if match is None:
        return "unknown"
    threshold_text = match.group("thr").replace("p", ".")
    if "to" in threshold_text:
        lo, hi = threshold_text.split("to", 1)
        return f"{lo} to {hi} uV"
    return f"{threshold_text} uV"


def parse_pair_folder(path: Path) -> PairFolder | None:
    match = PAIR_FOLDER_RE.match(path.name)
    if match is None:
        return None
    unit_key = path.name
    threshold_uv = threshold_min_from_unit_key(unit_key)
    if not np.isfinite(threshold_uv):
        return None
    return PairFolder(
        unit_key=unit_key,
        sg_ch=int(match.group("sg")),
        threshold_uv=float(threshold_uv),
        threshold_label=threshold_label_from_unit_key(unit_key),
        path=path,
    )


def discover_pair_folders(run_root: Path) -> list[PairFolder]:
    run_root = Path(run_root)
    candidates = [run_root, *run_root.rglob("*")]
    pairs: list[PairFolder] = []
    for path in candidates:
        if not path.is_dir() or not path.name.startswith("sgch") or "_thr" not in path.name:
            continue
        relative_parts = set(path.relative_to(run_root).parts) if path != run_root else set()
        if "polar_time_of_day_units" in relative_parts:
            continue
        pair = parse_pair_folder(path)
        if pair is not None:
            pairs.append(pair)
    pairs.sort(key=lambda pair: pair.sort_key)
    return pairs


def read_run_config(run_root: Path) -> dict:
    config_path = Path(run_root) / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def minute_summary_paths_for_pair(pair_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(pair_dir.rglob("*_minute_summary.csv")):
        if path.name == "tuning_weinan_units_used_summary.csv":
            continue
        paths.append(path)
    return paths


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"recording_name", "minute_index", "time_start_sec", "duration_sec"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise KeyError(f"{csv_path} is missing minute summary columns: {missing}")
        return [dict(row) for row in reader]


def threshold_feature_columns(unit_key: str) -> dict[str, str]:
    return {
        "firing_rate_hz": f"{unit_key}__firing_rate_hz",
        "average_amplitude_uv": f"{unit_key}__average_amplitude_uv",
        "cv2": f"{unit_key}__cv2",
        "peak_to_trough_ms": f"{unit_key}__peak_to_trough_ms",
    }


def default_output_dir(run_roots: tuple[Path, ...]) -> Path:
    if len(run_roots) == 1:
        return Path(run_roots[0]) / DEFAULT_OUTPUT_SUBDIR
    try:
        common_root = Path(os.path.commonpath([str(path.resolve()) for path in run_roots]))
    except Exception:
        common_root = Path(run_roots[0]).resolve().parent
    if common_root.is_file():
        common_root = common_root.parent
    if not common_root.exists() or not common_root.is_dir():
        common_root = Path(run_roots[0]).resolve().parent
    return common_root / DEFAULT_OUTPUT_SUBDIR


def parse_date_token(token: object) -> date | None:
    text = str(token or "").strip()
    if re.fullmatch(r"20\d{6}", text):
        try:
            return datetime.strptime(text, "%Y%m%d").date()
        except ValueError:
            return None
    if re.fullmatch(r"\d{6}", text):
        try:
            return datetime.strptime("20" + text, "%Y%m%d").date()
        except ValueError:
            return None
    return None


def dates_from_text(value: object) -> list[date]:
    text = str(value or "")
    dates: list[date] = []
    for match in re.finditer(r"(?<!\d)(20\d{6}|\d{6})(?!\d)", text):
        parsed = parse_date_token(match.group(1))
        if parsed is not None:
            dates.append(parsed)
    return dates


def infer_date_span_from_run_root_name(run_root: Path) -> tuple[date, date] | None:
    """
    Prefer recording dates encoded after threshold_crossings_.

    Examples:
      threshold_crossings_260220_260221_run_260519 -> 2026-02-20 .. 2026-02-21
      threshold_crossings_260222_run_260520        -> 2026-02-22

    The trailing run_YYMMDD token is intentionally ignored because it is the
    processing date, not the recording date.
    """
    name = Path(run_root).name
    match = re.search(
        r"threshold_crossings_(?P<start>\d{6}|20\d{6})(?:_(?P<end>\d{6}|20\d{6}))?(?:_|$)",
        name,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    start = parse_date_token(match.group("start"))
    end_text = match.group("end")
    end = parse_date_token(end_text) if end_text else start
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return start, end


def infer_run_root_date_span(run_root: Path) -> tuple[date, date] | None:
    parsed_from_name = infer_date_span_from_run_root_name(run_root)
    if parsed_from_name is not None:
        return parsed_from_name

    run_config = read_run_config(run_root)
    config_dates: list[date] = []
    for key in ("recording_date", "first_boundary_input", "last_boundary_input", "first_sort_key", "last_sort_key"):
        config_dates.extend(dates_from_text(run_config.get(key)))
    for recording_file in run_config.get("recording_files") or []:
        config_dates.extend(dates_from_text(Path(str(recording_file)).name))
    if config_dates:
        return min(config_dates), max(config_dates)

    name_dates = dates_from_text(Path(run_root).name)
    if name_dates:
        return min(name_dates), max(name_dates)
    return None


def date_range_label(dates: list[date]) -> str:
    if not dates:
        return "undated"
    first = dates[0].strftime("%y%m%d")
    last = dates[-1].strftime("%y%m%d")
    if first == last:
        return first
    return f"{first}_{last}"


def group_run_roots_by_contiguous_dates(run_roots: tuple[Path, ...]) -> list[tuple[str, tuple[Path, ...]]]:
    dated: list[tuple[date, date, Path]] = []
    undated: list[Path] = []
    for run_root in run_roots:
        parsed_span = infer_run_root_date_span(run_root)
        if parsed_span is None:
            undated.append(run_root)
        else:
            dated.append((parsed_span[0], parsed_span[1], run_root))

    groups: list[tuple[str, tuple[Path, ...]]] = []
    if dated:
        dated.sort(key=lambda item: (item[0], item[1], str(item[2].resolve())))
        current_dates: list[date] = []
        current_roots: list[Path] = []
        previous_end: date | None = None
        for start_date, end_date, run_root in dated:
            if previous_end is not None and (start_date - previous_end).days > 1:
                groups.append((date_range_label(current_dates), tuple(current_roots)))
                current_dates = []
                current_roots = []
            current_dates.extend([start_date, end_date])
            current_roots.append(run_root)
            previous_end = max(previous_end, end_date) if previous_end is not None else end_date
        if current_roots:
            groups.append((date_range_label(current_dates), tuple(current_roots)))

    if undated:
        groups.append(("undated", tuple(undated)))

    return groups or [("undated", run_roots)]


def common_parent_for_run_roots(run_roots: tuple[Path, ...]) -> Path:
    try:
        common_root = Path(os.path.commonpath([str(path.resolve()) for path in run_roots]))
    except Exception:
        return Path(run_roots[0]).resolve().parent
    if common_root.is_file():
        return common_root.parent
    if not common_root.exists() or not common_root.is_dir():
        return Path(run_roots[0]).resolve().parent
    return common_root


def default_output_dir_for_group(
    *,
    all_run_roots: tuple[Path, ...],
    group_run_roots: tuple[Path, ...],
    date_label: str,
) -> Path:
    folder_name = f"{DEFAULT_OUTPUT_SUBDIR}_{date_label}"
    if len(all_run_roots) == 1:
        return Path(group_run_roots[0]) / folder_name
    return common_parent_for_run_roots(all_run_roots) / folder_name


def output_dir_for_group(
    *,
    requested_output_dir: Path | None,
    total_groups: int,
    all_run_roots: tuple[Path, ...],
    group_run_roots: tuple[Path, ...],
    date_label: str,
) -> Path:
    if requested_output_dir is None:
        return default_output_dir_for_group(
            all_run_roots=all_run_roots,
            group_run_roots=group_run_roots,
            date_label=date_label,
        )
    requested_output_dir = Path(requested_output_dir)
    if total_groups <= 1:
        return requested_output_dir
    return requested_output_dir / f"{DEFAULT_OUTPUT_SUBDIR}_{date_label}"


def parse_path_list(values: list[str]) -> tuple[Path, ...]:
    raw_parts: list[str] = []
    for value in values:
        raw_parts.extend(re.split(r"[;,]", str(value)))
    paths = tuple(
        Path(part.strip().strip('"').strip("'"))
        for part in raw_parts
        if part.strip().strip('"').strip("'")
    )
    if not paths:
        raise ValueError("At least one threshold_crossings_* run folder is required.")
    for path in paths:
        if not path.exists() or not path.is_dir():
            raise NotADirectoryError(f"Threshold run folder not found: {path}")
    return paths


def parse_token_list(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return ()
    return tuple(
        token.strip()
        for token in re.split(r"[;,]", raw_value)
        if token.strip()
    )


def select_unit_keys_by_channels(run_roots: tuple[Path, ...], channel_text: str) -> tuple[str, ...]:
    selected_channels = {
        int(token.strip())
        for token in re.split(r"[;,]", channel_text)
        if token.strip()
    }
    if not selected_channels:
        return ()

    selected_keys: list[str] = []
    seen: set[str] = set()
    available_channels: set[int] = set()
    for run_root in run_roots:
        for pair in discover_pair_folders(run_root):
            available_channels.add(pair.sg_ch)
            if pair.sg_ch in selected_channels and pair.unit_key not in seen:
                seen.add(pair.unit_key)
                selected_keys.append(pair.unit_key)

    missing = sorted(selected_channels - available_channels)
    if missing:
        raise ValueError(f"Selected channel(s) not found in threshold units: {missing}")
    return tuple(selected_keys)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def is_population_csv_reusable(
    population_csv: Path,
    manifest_json: Path,
    run_roots: tuple[Path, ...],
    *,
    selected_unit_keys: tuple[str, ...] = (),
) -> tuple[bool, str]:
    if not population_csv.exists():
        return False, "population CSV is missing"
    if not manifest_json.exists():
        return False, "manifest is missing"
    try:
        manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"manifest could not be read: {exc}"
    requested_roots = [str(Path(path).resolve()) for path in run_roots]
    manifest_roots = [str(Path(path).resolve()) for path in manifest.get("input_run_roots", [])]
    portable_root = manifest.get("portable_threshold_run_root", None)
    portable_roots = [str(Path(portable_root).resolve())] if portable_root else []
    if manifest_roots != requested_roots and portable_roots != requested_roots:
        return False, "manifest input_run_roots differ from this run"
    manifest_selected = tuple(str(value) for value in manifest.get("selected_threshold_unit_keys", []))
    if manifest_selected != tuple(selected_unit_keys):
        return False, "manifest selected_threshold_unit_keys differ from this run"
    if str(Path(manifest.get("population_csv", "")).resolve()) != str(population_csv.resolve()):
        return False, "manifest points to a different population CSV"
    return True, "manifest matches"


def convert_threshold_channel_outputs(
    run_roots: tuple[Path, ...],
    output_dir: Path,
    *,
    output_name: str = DEFAULT_OUTPUT_NAME,
    selected_unit_keys: tuple[str, ...] = (),
    force: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    population_csv = output_dir / output_name
    manifest_json = output_dir / MANIFEST_NAME
    log("Starting Threshold_channel.py -> LDA population CSV conversion")
    log(f"Input run roots: {len(run_roots)}")
    for index, run_root in enumerate(run_roots, start=1):
        log(f"  run {index:03d}: {Path(run_root).resolve()}")
    log(f"Output directory: {output_dir.resolve()}")
    log(f"Output CSV: {population_csv.resolve()}")
    if selected_unit_keys:
        log(f"Selected threshold units: {len(selected_unit_keys)}")
    else:
        log("Selected threshold units: all discovered units")

    if not force:
        reusable, reason = is_population_csv_reusable(
            population_csv,
            manifest_json,
            run_roots,
            selected_unit_keys=selected_unit_keys,
        )
        if reusable:
            log(f"Reusing existing population CSV: {population_csv}")
            return population_csv
        log(f"Building population CSV ({reason}).")

    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    selected_set = set(selected_unit_keys)
    run_order_lookup = {Path(run_root).resolve(): index for index, run_root in enumerate(run_roots, start=1)}
    session_ordinal_lookup_by_run: dict[Path, dict[str, int]] = {}
    all_pair_meta: list[tuple[Path, PairFolder]] = []

    for run_root in run_roots:
        run_root_resolved = Path(run_root).resolve()
        t_run0 = time.perf_counter()
        run_config = read_run_config(run_root)
        lookup: dict[str, int] = {}
        for index, recording_file in enumerate(run_config.get("recording_files") or [], start=1):
            lookup[Path(str(recording_file)).name] = index
        session_ordinal_lookup_by_run[run_root_resolved] = lookup

        pairs = discover_pair_folders(run_root)
        if not pairs:
            log(f"No sgch*_thr*uV folders found under: {run_root}")
            continue
        log(
            f"Discovered {len(pairs)} threshold pair folder(s) in {run_root_resolved} "
            f"in {format_duration(time.perf_counter() - t_run0)}"
        )
        for pair in pairs:
            if selected_set and pair.unit_key not in selected_set:
                continue
            all_pair_meta.append((Path(run_root), pair))

    if not all_pair_meta:
        raise RuntimeError(
            "No threshold pair folders were found under the requested input folder(s) "
            f"with selected units: {sorted(selected_set)}"
        )

    unit_keys: list[str] = []
    for _, pair in all_pair_meta:
        if pair.unit_key not in unit_keys:
            unit_keys.append(pair.unit_key)
    unit_id_lookup = {unit_key: index for index, unit_key in enumerate(unit_keys, start=1)}

    sample_rows: dict[str, dict] = {}
    unit_rows_by_key: dict[str, dict] = {}
    portable_minute_rows_by_unit: dict[str, list[dict]] = {}
    skipped_rows_without_datetime = 0
    processed_minute_rows = 0
    total_pair_dirs = len(all_pair_meta)

    for pair_position, (run_root, pair) in enumerate(all_pair_meta, start=1):
        t_pair0 = time.perf_counter()
        run_root_resolved = Path(run_root).resolve()
        run_order = int(run_order_lookup[run_root_resolved])
        run_tag = f"run{run_order:03d}_{safe_slug(Path(run_root).name)}"
        session_ordinal_lookup = session_ordinal_lookup_by_run.get(run_root_resolved, {})
        feature_columns = threshold_feature_columns(pair.unit_key)
        unit_rows_by_key.setdefault(
            pair.unit_key,
            {
                "final_group_key": pair.unit_key,
                "final_unit_id": unit_id_lookup[pair.unit_key],
                "sg_ch": int(pair.sg_ch),
                "threshold_uv": float(pair.threshold_uv),
                "threshold_label": pair.threshold_label,
                "input_run_count": 0,
                "input_runs": [],
                "pair_dirs": [],
            },
        )
        unit_rows_by_key[pair.unit_key]["input_run_count"] += 1
        unit_rows_by_key[pair.unit_key]["input_runs"].append(str(run_root_resolved))
        unit_rows_by_key[pair.unit_key]["pair_dirs"].append(str(pair.path.resolve()))

        summary_paths = minute_summary_paths_for_pair(pair.path)
        if not summary_paths:
            log(f"No minute summary CSVs for {pair.unit_key}; skipping.")
            continue
        log(
            f"Pair {pair_position}/{total_pair_dirs}: {pair.unit_key} "
            f"({len(summary_paths)} minute summary file(s)); samples so far={len(sample_rows)}"
        )

        pair_minute_rows = 0
        for summary_index, summary_path in enumerate(summary_paths, start=1):
            t_summary0 = time.perf_counter()
            summary_rows = read_csv_rows(summary_path)
            log(
                f"  summary {summary_index}/{len(summary_paths)}: {summary_path.name} "
                f"({len(summary_rows)} row(s))"
            )
            for row in summary_rows:
                recording_name = str(row.get("recording_name", ""))
                start_dt = parse_recording_start_datetime(recording_name)
                if start_dt is None:
                    skipped_rows_without_datetime += 1
                    continue

                minute_index = int(float(row["minute_index"]))
                minute_start_sec = float(row["time_start_sec"])
                duration_sec = float(row["duration_sec"])
                minute_end_sec = minute_start_sec + duration_sec
                minute_start_dt = start_dt + timedelta(seconds=minute_start_sec)
                minute_end_dt = start_dt + timedelta(seconds=minute_end_sec)
                row_session_ordinal = safe_float(row.get("session_ordinal"))
                local_session_ordinal = (
                    int(row_session_ordinal)
                    if row_session_ordinal is not None
                    else int(session_ordinal_lookup.get(recording_name, 1))
                )
                session_ordinal = (run_order - 1) * 100000 + local_session_ordinal
                session_key = f"{run_tag}::{recording_name}"
                sample_key = f"{run_tag}__{safe_slug(recording_name)}__minute_{minute_index:06d}"

                sample = sample_rows.setdefault(
                    sample_key,
                    {
                        "final_sample_id": len(sample_rows) + 1,
                        "final_sample_key": sample_key,
                        "session_id": session_ordinal,
                        "session_key": session_key,
                        "session_name": recording_name,
                        "session_name_normalized": safe_slug(session_key),
                        "session_index": session_ordinal,
                        "session_start_datetime": start_dt.isoformat(sep=" "),
                        "minute_bin_index": minute_index,
                        "minute_start_sec": minute_start_sec,
                        "minute_end_sec": minute_end_sec,
                        "minute_center_s": minute_start_sec + duration_sec / 2.0,
                        "session_duration_s": np.nan,
                        "minute_start_datetime": minute_start_dt.isoformat(sep=" "),
                        "minute_end_datetime": minute_end_dt.isoformat(sep=" "),
                        "clock_hour_of_day": int(minute_start_dt.hour),
                        "clock_minute_of_hour": int(minute_start_dt.minute),
                        "calendar_day": minute_start_dt.date().isoformat(),
                        "rec_file": recording_name,
                        "threshold_run_root": str(run_root_resolved),
                        "threshold_run_name": Path(run_root).name,
                    },
                )
                firing_rate_hz = safe_float(row.get("firing_rate_hz"))
                amplitude_ptp_uv = safe_float(row.get("amplitude_ptp_uv"))
                cv2 = safe_float(row.get("cv2"))
                peak_to_trough_ms = safe_float(row.get("peak_to_trough_ms"))
                sample[feature_columns["firing_rate_hz"]] = 0.0 if firing_rate_hz is None else firing_rate_hz
                sample[feature_columns["average_amplitude_uv"]] = amplitude_ptp_uv
                sample[feature_columns["cv2"]] = cv2
                sample[feature_columns["peak_to_trough_ms"]] = peak_to_trough_ms
                portable_minute_rows_by_unit.setdefault(pair.unit_key, []).append(
                    {
                        "recording_name": recording_name,
                        "minute_index": minute_index,
                        "time_start_sec": minute_start_sec,
                        "time_end_sec": minute_end_sec,
                        "duration_sec": duration_sec,
                        "session_ordinal": session_ordinal,
                        "pair_id": pair.unit_key,
                        "sg_ch": int(pair.sg_ch),
                        "threshold_min_uv": float(pair.threshold_uv),
                        "threshold_label": pair.threshold_label,
                        "n_spikes": row.get("n_spikes", ""),
                        "firing_rate_hz": 0.0 if firing_rate_hz is None else firing_rate_hz,
                        "amplitude_ptp_uv": amplitude_ptp_uv,
                        "mean_abs_waveform_uv": safe_float(row.get("mean_abs_waveform_uv")),
                        "cv2": cv2,
                        "peak_to_trough_ms": peak_to_trough_ms,
                        "minute_start_datetime": minute_start_dt.isoformat(sep=" "),
                        "minute_end_datetime": minute_end_dt.isoformat(sep=" "),
                        "calendar_day": minute_start_dt.date().isoformat(),
                        "clock_hour_of_day": int(minute_start_dt.hour),
                        "clock_minute_of_hour": int(minute_start_dt.minute),
                        "source_run_root": str(run_root_resolved),
                        "source_pair_dir": str(pair.path.resolve()),
                    }
                )
                processed_minute_rows += 1
                pair_minute_rows += 1
            log(
                f"  finished summary {summary_index}/{len(summary_paths)} in "
                f"{format_duration(time.perf_counter() - t_summary0)}; "
                f"total samples={len(sample_rows)}"
            )
        log(
            f"Finished pair {pair_position}/{total_pair_dirs}: {pair.unit_key}; "
            f"processed {pair_minute_rows} pair-minute row(s) in "
            f"{format_duration(time.perf_counter() - t_pair0)}"
        )

    if not sample_rows:
        raise RuntimeError(
            "No threshold minute samples were materialized. Check that Threshold_channel.py "
            "wrote per-pair *_minute_summary.csv files with Chronic_Rec_YYYYMMDD_HHMMSS names."
        )

    feature_order: list[str] = []
    for unit_key in unit_keys:
        feature_order.extend(threshold_feature_columns(unit_key).values())

    sorted_sample_rows = sorted(
        sample_rows.values(),
        key=lambda row: (
            str(row["minute_start_datetime"]),
            int(row["session_index"]),
            int(row["minute_bin_index"]),
        ),
    )
    for sample_index, row in enumerate(sorted_sample_rows, start=1):
        row["final_sample_id"] = sample_index
        for column in feature_order:
            row.setdefault(column, "")

    t_write0 = time.perf_counter()
    write_csv(population_csv, sorted_sample_rows, METADATA_COLUMNS + feature_order)
    log(f"Wrote population CSV in {format_duration(time.perf_counter() - t_write0)}")

    unit_rows = []
    for row in unit_rows_by_key.values():
        normalized = dict(row)
        normalized["input_runs"] = " | ".join(normalized["input_runs"])
        normalized["pair_dirs"] = " | ".join(normalized["pair_dirs"])
        unit_rows.append(normalized)
    unit_rows.sort(key=lambda row: (int(row["sg_ch"]), float(row["threshold_uv"]), str(row["final_group_key"])))
    t_units0 = time.perf_counter()
    write_csv(
        output_dir / UNITS_SUMMARY_NAME,
        unit_rows,
        [
            "final_group_key",
            "final_unit_id",
            "sg_ch",
            "threshold_uv",
            "threshold_label",
            "input_run_count",
            "input_runs",
            "pair_dirs",
        ],
    )
    log(f"Wrote units summary in {format_duration(time.perf_counter() - t_units0)}")

    t_portable0 = time.perf_counter()
    portable_summary_paths: list[str] = []
    portable_fieldnames = [
        "recording_name",
        "minute_index",
        "time_start_sec",
        "time_end_sec",
        "duration_sec",
        "session_ordinal",
        "pair_id",
        "sg_ch",
        "threshold_min_uv",
        "threshold_label",
        "n_spikes",
        "firing_rate_hz",
        "amplitude_ptp_uv",
        "mean_abs_waveform_uv",
        "cv2",
        "peak_to_trough_ms",
        "minute_start_datetime",
        "minute_end_datetime",
        "calendar_day",
        "clock_hour_of_day",
        "clock_minute_of_hour",
        "source_run_root",
        "source_pair_dir",
    ]
    for unit_key, rows in portable_minute_rows_by_unit.items():
        rows.sort(
            key=lambda row: (
                str(row["minute_start_datetime"]),
                int(row["session_ordinal"]),
                int(row["minute_index"]),
            )
        )
        pair_dir = output_dir / unit_key
        summary_path = pair_dir / f"{unit_key}_minute_summary.csv"
        write_csv(summary_path, rows, portable_fieldnames)
        portable_summary_paths.append(str(summary_path.resolve()))
    log(
        f"Wrote portable pair minute summaries for {len(portable_minute_rows_by_unit)} unit(s) "
        f"in {format_duration(time.perf_counter() - t_portable0)}"
    )

    t_manifest0 = time.perf_counter()
    manifest_json.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
                "input_run_roots": [str(Path(run_root).resolve()) for run_root in run_roots],
                "selected_threshold_unit_keys": list(selected_unit_keys),
                "population_csv": str(population_csv.resolve()),
                "n_threshold_units": int(len(unit_rows)),
                "n_minute_samples": int(len(sorted_sample_rows)),
                "n_recordings": int(len({row["session_key"] for row in sorted_sample_rows})),
                "processed_pair_minute_rows": int(processed_minute_rows),
                "skipped_rows_without_recording_datetime": int(skipped_rows_without_datetime),
                "portable_threshold_run_root": str(output_dir.resolve()),
                "portable_pair_minute_summary_files": portable_summary_paths,
                "threshold_units": unit_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"Wrote manifest in {format_duration(time.perf_counter() - t_manifest0)}")

    portable_run_config = {
        "created_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "output_structure": "portable_threshold_csv_run_root",
        "source_run_roots": [str(Path(run_root).resolve()) for run_root in run_roots],
        "run_output_dir": str(output_dir.resolve()),
        "recording_files": sorted({str(row["rec_file"]) for row in sorted_sample_rows}),
        "selected_threshold_unit_keys": list(selected_unit_keys),
        "population_csv": str(population_csv.resolve()),
        "n_threshold_units": int(len(unit_rows)),
        "n_minute_samples": int(len(sorted_sample_rows)),
        "note": (
            "This folder was generated by Threshold_convert_csv.py. It contains a precomputed "
            "LDA population CSV plus lightweight sgch*_thr*uV/*_minute_summary.csv folders "
            "that can be used by Threshold_LDA_TuningWN_pre.py tuning and presentation stages."
        ),
    }
    (output_dir / PORTABLE_RUN_CONFIG_NAME).write_text(
        json.dumps(portable_run_config, indent=2),
        encoding="utf-8",
    )
    log(f"Wrote portable run config: {output_dir / PORTABLE_RUN_CONFIG_NAME}")

    elapsed_s = time.perf_counter() - start_time
    log(
        f"Wrote {len(sorted_sample_rows)} minute samples x {len(feature_order)} feature columns "
        f"for {len(unit_rows)} threshold units to {population_csv}"
    )
    log(f"Finished in {elapsed_s:.1f}s")
    return population_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_roots",
        nargs="*",
        help="One or more Threshold_channel.py threshold_crossings_* output folders.",
    )
    parser.add_argument(
        "--run-root",
        dest="run_root_opts",
        action="append",
        help="Threshold_channel.py output folder. May be repeated; comma/semicolon-separated values are accepted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            f"Output folder. By default, contiguous dates are written to dated folders like "
            f"{DEFAULT_OUTPUT_SUBDIR}_260220_260226. Non-contiguous date blocks are written "
            "to separate folders. If --output-dir is used with multiple date blocks, dated "
            "subfolders are created under that directory."
        ),
    )
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--selected-unit",
        action="append",
        default=[],
        help="Specific threshold unit key/folder name to include, e.g. sgch12_thr200uV. May be repeated.",
    )
    parser.add_argument(
        "--lda-channels",
        help="SG channel selection to include, e.g. '12,45,337'. Omit for all discovered units.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when an existing CSV and manifest match the requested inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_roots = []
    raw_roots.extend(args.run_root_opts or [])
    raw_roots.extend(args.run_roots or [])
    if not raw_roots:
        raw_text = input("Enter one or more threshold_crossings_* run folders, separated by semicolons: ").strip()
        raw_roots.append(raw_text)
    run_roots = parse_path_list(raw_roots)

    selected_unit_keys = parse_token_list(";".join(args.selected_unit or []))
    if args.lda_channels:
        channel_selected = select_unit_keys_by_channels(run_roots, str(args.lda_channels))
        selected_unit_keys = tuple(dict.fromkeys([*selected_unit_keys, *channel_selected]))
        log(f"Selected {len(selected_unit_keys)} unit(s) from --lda-channels/--selected-unit.")

    groups = group_run_roots_by_contiguous_dates(run_roots)
    log(f"Date grouping produced {len(groups)} output group(s).")
    for group_index, (date_label, group_roots) in enumerate(groups, start=1):
        log(
            f"Output group {group_index}/{len(groups)}: {date_label} "
            f"({len(group_roots)} run root(s))"
        )
        output_dir = output_dir_for_group(
            requested_output_dir=Path(args.output_dir) if args.output_dir is not None else None,
            total_groups=len(groups),
            all_run_roots=run_roots,
            group_run_roots=group_roots,
            date_label=date_label,
        )
        convert_threshold_channel_outputs(
            group_roots,
            output_dir,
            output_name=str(args.output_name),
            selected_unit_keys=selected_unit_keys,
            force=bool(args.force),
        )


if __name__ == "__main__":
    main()
