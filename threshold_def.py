from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter


DEFAULT_ALIGNMENT_DIR = r"S:\alignment_days_summary_260221_260226"
DEFAULT_MIN_SESSIONS = 1
DEFAULT_SINGLE_LOWER_FACTOR = 0.60
DEFAULT_SINGLE_UPPER_FACTOR = 1.60
DEFAULT_MULTI_LOWER_FACTOR = 0.60
DEFAULT_MULTI_UPPER_FACTOR = 1.60
DEFAULT_MIN_LOWER_UV = 25.0
DEFAULT_ROUND_TO_UV = 5.0


def format_elapsed(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.2f}min"
    return f"{seconds / 3600.0:.2f}h"


def log_status(message: str, start_time: float | None = None) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if start_time is None:
        print(f"[{stamp}] {message}", flush=True)
    else:
        print(f"[{stamp} | +{format_elapsed(perf_counter() - start_time)}] {message}", flush=True)


def add_timing(timing: dict[str, float], key: str, start_time: float) -> None:
    timing[key] = timing.get(key, 0.0) + float(perf_counter() - start_time)


@dataclass
class UnitRow:
    sg_channel: int
    final_unit_id: int
    amplitude_uv: float | None
    query_threshold_uv: float | None
    num_sessions: int | None
    total_member_sessions: int | None
    label: str
    summary_path: str


def prompt_line(message: str, default: str | None = None) -> str:
    if default is None or str(default).strip() == "":
        return input(f"{message}: ").strip()
    raw = input(f"{message} [{default}]: ").strip()
    return raw if raw else str(default)


def prompt_float(message: str, default: float) -> float:
    while True:
        raw = prompt_line(message, str(default))
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def prompt_int(message: str, default: int) -> int:
    while True:
        raw = prompt_line(message, str(default))
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")


def _parse_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def _parse_int(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _amplitude_from_row(row: dict) -> float | None:
    for key in ("max_abs_amplitude_uv", "amplitude_uv", "template_amplitude_uv"):
        value = _parse_float(row.get(key))
        if value is not None:
            return abs(value)
    amps = str(row.get("amplitudes_found", "")).strip()
    if amps:
        values = []
        for token in re.split(r"[;,]\s*", amps):
            value = _parse_float(token)
            if value is not None:
                values.append(abs(value))
        if values:
            return max(values)
    return None


def _amplitude_from_summary_txt(summary_path: str | Path) -> float | None:
    path = Path(str(summary_path))
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    values = []
    for match in re.finditer(r"amplitude_median\s*=\s*([-+]?\d+(?:\.\d+)?)", text):
        value = _parse_float(match.group(1))
        if value is not None:
            values.append(abs(value))
    if not values:
        return None
    return float(median(values))


def _found_amplitude_lookup(alignment_dir: Path) -> dict[int, float]:
    found_csv = alignment_dir / "found_units_by_sg_channel_threshold.csv"
    if not found_csv.exists():
        return {}
    by_unit: dict[int, list[float]] = {}
    with found_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            final_unit_id = _parse_int(raw.get("final_unit_id"))
            amp = _amplitude_from_row(raw)
            if final_unit_id is None or amp is None:
                continue
            by_unit.setdefault(int(final_unit_id), []).append(float(amp))
    return {unit_id: float(median(values)) for unit_id, values in by_unit.items() if values}


def load_unit_rows(alignment_dir: Path) -> tuple[list[UnitRow], Path]:
    unique_csv = alignment_dir / "unique_units_summary.csv"
    if not unique_csv.exists():
        raise FileNotFoundError(
            f"Could not find unique_units_summary.csv in {alignment_dir}"
        )

    found_lookup = _found_amplitude_lookup(alignment_dir)
    rows: list[UnitRow] = []
    with unique_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            sg_channel = _parse_int(raw.get("sg_channel"))
            final_unit_id = _parse_int(raw.get("final_unit_id"))
            if sg_channel is None or final_unit_id is None:
                continue
            summary_path = str(raw.get("summary_path", "")).strip()
            amplitude_uv = _amplitude_from_summary_txt(summary_path)
            if amplitude_uv is None:
                amplitude_uv = found_lookup.get(int(final_unit_id))
            rows.append(
                UnitRow(
                    sg_channel=int(sg_channel),
                    final_unit_id=int(final_unit_id),
                    amplitude_uv=amplitude_uv,
                    query_threshold_uv=_parse_float(raw.get("query_threshold_uv")),
                    num_sessions=_parse_int(raw.get("num_sessions")),
                    total_member_sessions=_parse_int(raw.get("num_member_units")),
                    label=str(raw.get("final_unit_label", f"unit_{final_unit_id:04d}")),
                    summary_path=summary_path,
                )
            )
    return rows, unique_csv


def dedupe_units(rows: list[UnitRow], min_sessions: int) -> list[UnitRow]:
    best: dict[tuple[int, int], UnitRow] = {}
    for row in rows:
        session_count = row.num_sessions or row.total_member_sessions or 0
        if session_count < min_sessions:
            continue
        key = (row.sg_channel, row.final_unit_id)
        old = best.get(key)
        if old is None:
            best[key] = row
            continue
        old_amp = old.amplitude_uv if old.amplitude_uv is not None else -1.0
        new_amp = row.amplitude_uv if row.amplitude_uv is not None else -1.0
        if new_amp > old_amp:
            best[key] = row
    return list(best.values())


def round_to(value: float, step: float, *, mode: str = "nearest") -> float:
    if step <= 0:
        return float(value)
    scaled = value / step
    if mode == "down":
        out = math.floor(scaled) * step
    elif mode == "up":
        out = math.ceil(scaled) * step
    else:
        out = round(scaled) * step
    return float(round(out, 6))


def unit_amplitude(row: UnitRow) -> float:
    if row.amplitude_uv is not None:
        return float(row.amplitude_uv)
    if row.query_threshold_uv is not None:
        return float(row.query_threshold_uv)
    return 100.0


def build_channel_ranges(
    units: list[UnitRow],
    *,
    min_lower_uv: float,
    single_lower_factor: float,
    single_upper_factor: float,
    multi_lower_factor: float,
    multi_upper_factor: float,
    round_to_uv: float,
) -> tuple[list[dict], list[dict]]:
    by_channel: dict[int, list[UnitRow]] = {}
    for row in units:
        by_channel.setdefault(row.sg_channel, []).append(row)

    pairs: list[dict] = []
    report_rows: list[dict] = []
    for sg_channel in sorted(by_channel):
        ch_units = sorted(by_channel[sg_channel], key=unit_amplitude)
        amps = [unit_amplitude(row) for row in ch_units]
        unit_ids = [int(row.final_unit_id) for row in ch_units]
        labels = [row.label for row in ch_units]

        if len(ch_units) == 1:
            amp = amps[0]
            lower = max(float(min_lower_uv), amp * float(single_lower_factor))
            upper = max(lower + round_to_uv, amp * float(single_upper_factor))
            lower = round_to(lower, round_to_uv, mode="down")
            upper = round_to(upper, round_to_uv, mode="up")
            range_rows = [(lower, upper, unit_ids, labels)]
            strategy = "single_unit_scaled_range"
        else:
            gaps = [amps[i + 1] - amps[i] for i in range(len(amps) - 1)]
            split_i = max(range(len(gaps)), key=lambda i: gaps[i]) if gaps else 0
            boundary = 0.5 * (amps[split_i] + amps[split_i + 1])
            lower1 = max(float(min_lower_uv), min(amps[: split_i + 1]) * float(multi_lower_factor))
            upper1 = boundary
            lower2 = boundary
            upper2 = max(lower2 + round_to_uv, max(amps[split_i + 1 :]) * float(multi_upper_factor))
            lower1 = round_to(lower1, round_to_uv, mode="down")
            upper1 = round_to(upper1, round_to_uv, mode="up")
            lower2 = round_to(lower2, round_to_uv, mode="down")
            upper2 = round_to(upper2, round_to_uv, mode="up")
            range_rows = [
                (lower1, upper1, unit_ids[: split_i + 1], labels[: split_i + 1]),
                (lower2, upper2, unit_ids[split_i + 1 :], labels[split_i + 1 :]),
            ]
            strategy = "multi_unit_largest_gap_two_ranges"

        for range_index, (lower, upper, ids, range_labels) in enumerate(range_rows, start=1):
            if upper <= lower:
                upper = lower + max(round_to_uv, 1.0)
            pair = {
                "sg_ch": int(sg_channel),
                "threshold_uv": float(lower),
                "threshold_max_uv": float(upper),
                "range_index": int(range_index),
                "source_final_unit_ids": [int(v) for v in ids],
                "source_final_unit_labels": range_labels,
            }
            pairs.append(pair)
            report_rows.append(
                {
                    "sg_ch": int(sg_channel),
                    "range_index": int(range_index),
                    "threshold_uv": float(lower),
                    "threshold_max_uv": float(upper),
                    "n_units_on_channel": int(len(ch_units)),
                    "unit_ids_on_channel": ";".join(str(v) for v in unit_ids),
                    "unit_amplitudes_uv": ";".join(f"{v:.3f}" for v in amps),
                    "range_unit_ids": ";".join(str(v) for v in ids),
                    "strategy": strategy,
                }
            )
    return pairs, report_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_timing_reports(output_json: Path, timing: dict[str, float], total_elapsed: float) -> tuple[Path, Path]:
    rows = [
        {
            "step": "total",
            "seconds": float(total_elapsed),
            "elapsed": format_elapsed(total_elapsed),
        }
    ]
    rows.extend(
        {
            "step": key,
            "seconds": float(value),
            "elapsed": format_elapsed(value),
        }
        for key, value in sorted(timing.items())
    )
    csv_path = output_json.with_name(output_json.stem + "_timing.csv")
    json_path = output_json.with_name(output_json.stem + "_timing.json")
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Define per-channel threshold ranges from an aligned sorting summary."
    )
    parser.add_argument("alignment_dir", nargs="?", help="Folder like S:\\alignment_days_summary_260221_260226")
    parser.add_argument("--output-json", help="Output JSON path.")
    parser.add_argument("--min-sessions", type=int, default=None)
    parser.add_argument("--min-lower-uv", type=float, default=None)
    parser.add_argument("--single-lower-factor", type=float, default=None)
    parser.add_argument("--single-upper-factor", type=float, default=None)
    parser.add_argument("--multi-lower-factor", type=float, default=None)
    parser.add_argument("--multi-upper-factor", type=float, default=None)
    parser.add_argument("--round-to-uv", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    total_start = perf_counter()
    timing: dict[str, float] = {}
    log_status("Starting threshold definition")
    args = parse_args()
    alignment_dir = Path(args.alignment_dir or prompt_line("Aligned summary folder", DEFAULT_ALIGNMENT_DIR)).expanduser()
    if not alignment_dir.exists():
        print(f"Aligned summary folder not found: {alignment_dir}")
        return 1

    default_output = alignment_dir / "channel_thresholds_from_sorted.json"
    output_json = Path(args.output_json or prompt_line("Output threshold JSON path", str(default_output)))
    min_sessions = args.min_sessions if args.min_sessions is not None else prompt_int("Minimum sessions/member sessions per unit", DEFAULT_MIN_SESSIONS)
    min_lower_uv = args.min_lower_uv if args.min_lower_uv is not None else prompt_float("Minimum lower threshold (uV)", DEFAULT_MIN_LOWER_UV)
    single_lower_factor = args.single_lower_factor if args.single_lower_factor is not None else prompt_float("Single-unit lower factor", DEFAULT_SINGLE_LOWER_FACTOR)
    single_upper_factor = args.single_upper_factor if args.single_upper_factor is not None else prompt_float("Single-unit upper factor", DEFAULT_SINGLE_UPPER_FACTOR)
    multi_lower_factor = args.multi_lower_factor if args.multi_lower_factor is not None else prompt_float("Multi-unit lower factor", DEFAULT_MULTI_LOWER_FACTOR)
    multi_upper_factor = args.multi_upper_factor if args.multi_upper_factor is not None else prompt_float("Multi-unit upper factor", DEFAULT_MULTI_UPPER_FACTOR)
    round_to_uv = args.round_to_uv if args.round_to_uv is not None else prompt_float("Round thresholds to nearest uV step", DEFAULT_ROUND_TO_UV)

    step_start = perf_counter()
    log_status(f"Loading aligned units from {alignment_dir}", total_start)
    rows, source_csv = load_unit_rows(alignment_dir)
    add_timing(timing, "load_units_and_amplitudes", step_start)
    log_status(f"Loaded {len(rows)} row(s) in {format_elapsed(perf_counter() - step_start)}", total_start)

    step_start = perf_counter()
    log_status(f"Filtering/deduplicating {len(rows)} unit row(s)", total_start)
    units = dedupe_units(rows, int(min_sessions))
    add_timing(timing, "filter_dedupe_units", step_start)
    log_status(f"Kept {len(units)} unit(s) in {format_elapsed(perf_counter() - step_start)}", total_start)
    if not units:
        print("No units found after filtering.")
        return 1

    step_start = perf_counter()
    log_status(
        f"Building channel ranges for {len(units)} unit(s) on {len({row.sg_channel for row in units})} channel(s)",
        total_start,
    )
    pairs, report_rows = build_channel_ranges(
        units,
        min_lower_uv=float(min_lower_uv),
        single_lower_factor=float(single_lower_factor),
        single_upper_factor=float(single_upper_factor),
        multi_lower_factor=float(multi_lower_factor),
        multi_upper_factor=float(multi_upper_factor),
        round_to_uv=float(round_to_uv),
    )
    add_timing(timing, "build_threshold_ranges", step_start)
    log_status(f"Built {len(pairs)} threshold range(s) in {format_elapsed(perf_counter() - step_start)}", total_start)

    step_start = perf_counter()
    log_status(f"Writing outputs to {output_json}", total_start)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "threshold_channel_pairs_v2",
        "note": (
            "Use Threshold_channel.py JSON mode. threshold_uv is the lower threshold; "
            "threshold_max_uv is the upper amplitude limit for range-style thresholding. "
            "Channels/units are discovered from unique_units_summary.csv; amplitudes are read "
            "from exported unit summary.txt amplitude_median values, with found_units_by_sg_channel_threshold.csv "
            "used only as an optional amplitude fallback."
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "alignment_dir": str(alignment_dir.resolve()),
        "source_csv": str(source_csv.resolve()),
        "min_sessions": int(min_sessions),
        "parameters": {
            "min_lower_uv": float(min_lower_uv),
            "single_lower_factor": float(single_lower_factor),
            "single_upper_factor": float(single_upper_factor),
            "multi_lower_factor": float(multi_lower_factor),
            "multi_upper_factor": float(multi_upper_factor),
            "round_to_uv": float(round_to_uv),
        },
        "n_units": int(len(units)),
        "n_channels": int(len({row.sg_channel for row in units})),
        "n_pairs": int(len(pairs)),
        "pairs": pairs,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_csv = output_json.with_suffix(".csv")
    write_csv(report_csv, report_rows)
    add_timing(timing, "write_outputs", step_start)
    log_status(f"Wrote threshold outputs in {format_elapsed(perf_counter() - step_start)}", total_start)
    total_elapsed = perf_counter() - total_start
    timing_csv, timing_json = write_timing_reports(output_json, timing, total_elapsed)

    print(f"Loaded units: {len(units)} from {source_csv}")
    print(f"Channels with units: {payload['n_channels']}")
    print(f"Threshold pairs/ranges: {len(pairs)}")
    print(f"Wrote JSON: {output_json}")
    print(f"Wrote CSV report: {report_csv}")
    print(f"Wrote timing CSV: {timing_csv}")
    print(f"Wrote timing JSON: {timing_json}")
    print(f"Finished in {format_elapsed(total_elapsed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
