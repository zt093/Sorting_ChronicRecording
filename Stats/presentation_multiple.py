from __future__ import annotations

"""
Presentation-ready summary plots for multi-day alignment outputs.

This is a thin wrapper around `presentations.py` that targets the export bundle
produced by `Alignment_days.py`. It reuses the same plotting logic, but changes
the default input discovery so the script can start from a folder that contains
an `alignment_days_summary*/export_summary.json`.
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
        help="Folder where plots should be written. Defaults to <alignment_days_summary>/stats.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=base_presentations.DEFAULT_MAX_SESSIONS,
        help=(
            "Maximum day count shown on plots. The plotting code still uses the shared "
            "session-oriented axis naming."
        ),
    )
    parser.add_argument(
        "--stable-threshold",
        type=int,
        default=2,
        help="Minimum number of days used to define a persistent unit for the channel-ranking plot",
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

    output_dir = resolve_output_dir(export_summary_path, args.output_dir)
    df = base_presentations.load_unique_units_summary(unique_units_csv)
    plot_paths: list[Path] = []

    plot_paths.append(
        base_presentations.plot_units_by_exact_session_count(
            df,
            output_dir,
            max_sessions=args.max_sessions,
        )
    )

    top_channels_plot = base_presentations.plot_top_channels_by_persistent_units(
        df,
        output_dir,
        min_sessions_for_stable=args.stable_threshold,
        top_n=args.top_n_channels,
    )
    if top_channels_plot is not None:
        plot_paths.append(top_channels_plot)

    quality_df = base_presentations.load_quality_metrics_from_export_summary(export_summary_path)
    if not quality_df.empty:
        quality_csv_path = output_dir / "quality_metrics_source_table.csv"
        log_status(f"Saving quality-metric source table: {quality_csv_path}")
        quality_df.to_csv(quality_csv_path, index=False)
        plot_paths.append(quality_csv_path)

        quality_plot_path = base_presentations.plot_quality_metrics_vs_persistence(
            quality_df,
            output_dir,
            max_sessions=args.max_sessions,
        )
        if quality_plot_path is not None:
            plot_paths.append(quality_plot_path)
    else:
        log_status("No quality metrics were available to plot from the day-level export summary")

    try:
        all_units_quality_df = base_presentations.load_quality_metrics_all_units_from_batch_root(
            export_summary_path.parent
        )
    except Exception as exc:
        log_status(f"Could not load all-unit day quality metrics: {exc}")
        all_units_quality_df = pd.DataFrame()

    if not all_units_quality_df.empty:
        quality_by_session_plot_path = base_presentations.plot_quality_metrics_by_session(
            all_units_quality_df,
            output_dir,
            max_sessions=args.max_sessions,
        )
        if quality_by_session_plot_path is not None:
            plot_paths.append(quality_by_session_plot_path)
    else:
        log_status(
            "No all-unit quality data were available for the day-index plot. "
            "This is expected when analyzer folders are not discoverable under the summary root."
        )

    manifest_path = output_dir / "presentation_multiple_manifest.json"
    log_status(f"Writing plot manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(
            {
                "unique_units_csv": str(unique_units_csv),
                "export_summary": str(export_summary_path),
                "num_unique_units": int(len(df)),
                "max_sessions": int(args.max_sessions),
                "plots": [str(path) for path in plot_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log_status("Multi-day presentations pipeline complete")
    print(f"Saved {len(plot_paths)} output file(s) to: {output_dir}")
    for path in plot_paths:
        print(f" - {path}")
    print(f" - {manifest_path}")


if __name__ == "__main__":
    main()
