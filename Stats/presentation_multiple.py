from __future__ import annotations

"""
Presentation-ready summary plots for multi-day alignment outputs.

This is a wrapper around `presentations.py` for the export bundle produced by
`Alignment_days.py`. It can render two views of the same multi-day alignment:

- day-based: persistence means the number of aligned days present
- hour-based: persistence means the number of underlying source hour/session
  exports present across those days
"""

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import pandas as pd

import presentations as base_presentations


DEFAULT_DAYS_SUMMARY_FOLDER_PREFIX = "alignment_days_summary"
DEFAULT_OUTPUT_SUBFOLDER_NAME = "stats"
DEFAULT_BASIS = "both"


BASIS_LABELS = {
    "day": "Day",
    "hour": "Hour",
}


def log_status(message: str) -> None:
    print(f"[presentation_multiple] {message}", flush=True)


def choose_days_root_interactively() -> Path | None:
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_folder = filedialog.askdirectory(
            title="Select a folder that contains alignment_days_summary/export_summary.json",
            mustexist=True,
        )
        root.destroy()
    except Exception:
        selected_folder = ""

    if not selected_folder:
        raw_value = input(
            "Enter path to a folder that contains alignment_days_summary/export_summary.json: "
        ).strip().strip('"').strip("'")
        if not raw_value:
            return None
        selected_folder = raw_value

    selected = Path(selected_folder)
    if not selected.exists():
        raise FileNotFoundError(f"Selected folder not found: {selected}")
    if not selected.is_dir():
        raise NotADirectoryError(f"Selected path is not a folder: {selected}")
    return selected


def find_days_export_summary(root: Path) -> Path:
    root = Path(root)
    if root.is_file() and root.name == "export_summary.json":
        return root

    direct_candidate = root / "export_summary.json"
    if direct_candidate.exists():
        return direct_candidate

    summary_candidates = sorted(
        [
            path / "export_summary.json"
            for path in root.glob(f"{DEFAULT_DAYS_SUMMARY_FOLDER_PREFIX}*")
            if path.is_dir() and (path / "export_summary.json").exists()
        ]
    )
    if summary_candidates:
        return summary_candidates[0]

    nested_candidates = sorted(
        root.rglob(f"{DEFAULT_DAYS_SUMMARY_FOLDER_PREFIX}*/export_summary.json")
    )
    if nested_candidates:
        return nested_candidates[0]

    raise FileNotFoundError(
        "Could not find Alignment_days export_summary.json under "
        f"{root}. Expected something like "
        f"{DEFAULT_DAYS_SUMMARY_FOLDER_PREFIX}*/export_summary.json"
    )


def resolve_export_summary_path(
    days_root: str | None,
    export_summary: str | None,
) -> Path:
    if export_summary:
        export_summary_path = Path(export_summary)
        if not export_summary_path.exists():
            raise FileNotFoundError(f"export_summary.json not found: {export_summary_path}")
        return export_summary_path

    selected_root = Path(days_root) if days_root else choose_days_root_interactively()
    if selected_root is None:
        raise ValueError(
            "No input folder selected. Provide --days-root or --export-summary, "
            "or choose a folder when prompted."
        )
    return find_days_export_summary(selected_root)


def resolve_output_dir(export_summary_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        resolved = Path(output_dir)
        resolved.mkdir(parents=True, exist_ok=True)
        log_status(f"Using requested output folder: {resolved}")
        return resolved

    summary_root = export_summary_path.parent
    resolved = summary_root / DEFAULT_OUTPUT_SUBFOLDER_NAME
    resolved.mkdir(parents=True, exist_ok=True)
    log_status(f"Using default output folder under day summary root: {resolved}")
    return resolved


def basis_output_dir(base_output_dir: Path, basis: str, selected_basis: str) -> Path:
    if selected_basis != "both":
        base_output_dir.mkdir(parents=True, exist_ok=True)
        return base_output_dir

    resolved = base_output_dir / f"{basis}_based"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _safe_int(value) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _first_int_from_members(group_row: dict, key: str) -> int | None:
    for member_list_key in ("day_members", "members", "source_members"):
        for member in group_row.get(member_list_key, []) or []:
            value = _safe_int(member.get(key))
            if value is not None:
                return value
    return None


def total_source_hours_for_group(group_row: dict) -> int:
    day_members = group_row.get("day_members") or []
    if day_members:
        total_hours = 0
        for day_member in day_members:
            day_hour_count = _safe_int(day_member.get("num_source_sessions"))
            if day_hour_count is None:
                day_hour_count = len(day_member.get("source_sessions_present") or [])
            total_hours += int(day_hour_count)
        if total_hours > 0:
            return total_hours

    source_members = group_row.get("source_members") or group_row.get("members") or []
    num_source_sessions = _safe_int(group_row.get("num_source_sessions"))
    if num_source_sessions is not None:
        return int(num_source_sessions)
    return len(source_members)


def load_unique_units_for_basis(
    payload: dict,
    unique_units_csv: Path,
    *,
    basis: str,
) -> pd.DataFrame:
    if basis == "day":
        return base_presentations.load_unique_units_summary(unique_units_csv)

    rows: list[dict] = []
    for group_row in payload.get("cross_session_alignment_groups", []) or []:
        source_members = group_row.get("source_members") or group_row.get("members") or []
        total_source_hours = total_source_hours_for_group(group_row)

        final_unit_id = _safe_int(group_row.get("final_unit_id"))
        shank_id = _safe_int(group_row.get("shank_id")) or _first_int_from_members(group_row, "shank_id")
        sg_channel = _safe_int(group_row.get("sg_channel")) or _first_int_from_members(group_row, "sg_channel")
        if final_unit_id is None or shank_id is None or sg_channel is None:
            continue

        rows.append(
            {
                "final_unit_id": final_unit_id,
                "shank_id": shank_id,
                "sg_channel": sg_channel,
                "num_sessions": int(total_source_hours),
                "num_member_units": int(len(source_members)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "Could not build hour-based unique-unit table from export summary. "
            "Expected cross_session_alignment_groups with source members."
        )
    log_status(f"Loaded {len(df)} hour-based unique units from export summary")
    return df


def remap_quality_persistence_to_basis(quality_df: pd.DataFrame, payload: dict, *, basis: str) -> pd.DataFrame:
    if quality_df.empty:
        return quality_df

    persistence_counts: dict[str, int] = {}
    for group_row in payload.get("cross_session_alignment_groups", []) or []:
        final_group_key = str(group_row.get("final_group_key", ""))
        if not final_group_key:
            continue
        if basis == "hour":
            persistence_count = total_source_hours_for_group(group_row)
        else:
            persistence_count = _safe_int(group_row.get("num_day_members"))
            if persistence_count is None:
                persistence_count = len(group_row.get("day_members") or [])
            if persistence_count <= 0:
                persistence_count = _safe_int(group_row.get("num_sessions")) or 0
        if persistence_count > 0:
            persistence_counts[final_group_key] = int(persistence_count)

    if not persistence_counts:
        return quality_df

    remapped = quality_df.copy()
    remapped["num_sessions"] = remapped["final_group_key"].map(persistence_counts).fillna(
        remapped["num_sessions"]
    )
    remapped["num_sessions"] = remapped["num_sessions"].astype(int)
    return remapped


def run_presentations_for_basis(
    *,
    basis: str,
    payload: dict,
    unique_units_csv: Path,
    quality_df: pd.DataFrame,
    output_dir: Path,
    max_sessions: int | None,
    stable_threshold: int,
    top_n_channels: int,
) -> tuple[Path, list[Path], int]:
    label = BASIS_LABELS[basis]
    log_status(f"Starting {basis}-based presentation plots")

    df = load_unique_units_for_basis(payload, unique_units_csv, basis=basis)
    effective_max_sessions = max_sessions
    if effective_max_sessions is None:
        effective_max_sessions = max(1, int(df["num_sessions"].max()))
    log_status(
        f"{basis}-based x-axis maximum is {effective_max_sessions} {label.lower()}s"
    )
    plot_paths: list[Path] = []

    plot_paths.append(
        base_presentations.plot_units_by_exact_session_count(
            df,
            output_dir,
            max_sessions=effective_max_sessions,
            session_label=label,
        )
    )

    top_channels_plot = base_presentations.plot_top_channels_by_persistent_units(
        df,
        output_dir,
        min_sessions_for_stable=stable_threshold,
        top_n=top_n_channels,
        session_label=label,
    )
    if top_channels_plot is not None:
        plot_paths.append(top_channels_plot)

    basis_quality_df = remap_quality_persistence_to_basis(quality_df, payload, basis=basis)
    if not basis_quality_df.empty:
        quality_csv_path = output_dir / "quality_metrics_source_table.csv"
        log_status(f"Saving {basis}-based quality-metric source table: {quality_csv_path}")
        basis_quality_df.to_csv(quality_csv_path, index=False)
        plot_paths.append(quality_csv_path)

        quality_plot_path = base_presentations.plot_quality_metrics_vs_persistence(
            basis_quality_df,
            output_dir,
            max_sessions=effective_max_sessions,
            session_label=label,
        )
        if quality_plot_path is not None:
            plot_paths.append(quality_plot_path)
    else:
        log_status(f"No quality metrics were available to plot from the {basis}-based export summary")

    return output_dir, plot_paths, len(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create presentation-ready plots from Alignment_days.py exports."
    )
    parser.add_argument(
        "--days-root",
        help=(
            "Folder that contains an alignment_days_summary*/export_summary.json file, "
            "or the summary folder itself."
        ),
    )
    parser.add_argument(
        "--export-summary",
        help="Optional path to export_summary.json produced by Alignment_days.py",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Folder where plots should be written. Defaults to <alignment_days_summary>/stats. "
            "When --basis both is used, day_based and hour_based subfolders are created."
        ),
    )
    parser.add_argument(
        "--basis",
        choices=["day", "hour", "both"],
        default=DEFAULT_BASIS,
        help="Which persistence view to render. Defaults to both day-based and hour-based outputs.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help=(
            "Optional maximum count shown on plots for the selected basis. By default, "
            "each basis uses its observed maximum, so hour-based plots can span total "
            "hours across multiple days."
        ),
    )
    parser.add_argument(
        "--stable-threshold",
        type=int,
        default=2,
        help=(
            "Minimum number of days or hours used to define a persistent unit for "
            "the channel-ranking plot."
        ),
    )
    parser.add_argument(
        "--top-n-channels",
        type=int,
        default=20,
        help="How many channels to show in the persistent-channel ranking plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_status("Starting multi-day presentations pipeline")

    export_summary_path = resolve_export_summary_path(
        days_root=args.days_root,
        export_summary=args.export_summary,
    )
    log_status(f"Using Alignment_days export summary: {export_summary_path}")

    payload = json.loads(export_summary_path.read_text(encoding="utf-8"))
    unique_units_csv = Path(payload["unique_units_summary_csv"])
    if not unique_units_csv.exists():
        raise FileNotFoundError(f"Referenced unique_units_summary.csv not found: {unique_units_csv}")

    base_output_dir = resolve_output_dir(export_summary_path, args.output_dir)
    bases = ["day", "hour"] if args.basis == "both" else [args.basis]
    all_plot_paths: list[Path] = []
    manifest_runs: list[dict] = []
    quality_df = base_presentations.load_quality_metrics_from_export_summary(export_summary_path)

    for basis in bases:
        output_dir = basis_output_dir(base_output_dir, basis, args.basis)
        basis_output, plot_paths, num_unique_units = run_presentations_for_basis(
            basis=basis,
            payload=payload,
            unique_units_csv=unique_units_csv,
            quality_df=quality_df,
            output_dir=output_dir,
            max_sessions=args.max_sessions,
            stable_threshold=args.stable_threshold,
            top_n_channels=args.top_n_channels,
        )
        all_plot_paths.extend(plot_paths)
        manifest_runs.append(
            {
                "basis": basis,
                "output_dir": str(basis_output),
                "num_unique_units": int(num_unique_units),
                "plots": [str(path) for path in plot_paths],
            }
        )

    manifest_path = base_output_dir / "presentation_multiple_manifest.json"
    log_status(f"Writing plot manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(
            {
                "unique_units_csv": str(unique_units_csv),
                "export_summary": str(export_summary_path),
                "basis": args.basis,
                "max_sessions": (
                    int(args.max_sessions) if args.max_sessions is not None else None
                ),
                "runs": manifest_runs,
                "plots": [str(path) for path in all_plot_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log_status("Multi-day presentations pipeline complete")
    print(f"Saved {len(all_plot_paths)} output file(s) under: {base_output_dir}")
    for path in all_plot_paths:
        print(f" - {path}")
    print(f" - {manifest_path}")


if __name__ == "__main__":
    main()
