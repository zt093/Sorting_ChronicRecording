from __future__ import annotations

"""
Automatic unit alignment/export pipeline.

This script combines the non-interactive parts of Alignment_html.py and
Alignment_days.py:

1. For each input daily `*_Sorting` folder, load the raw analyzer outputs from
   Combined_NWB+Sorting+Analyze.py, apply the same automatic discard/merge/align
   thresholds used by Alignment_html.py, and export the same per-page and
   all-shank summary bundle into `units_alignment_summary_auto`.
2. If more than one day is present, load those per-day auto exports, apply the
   same cross-day alignment logic used by Alignment_days.py, and export the
   same cross-day summary bundle into `alignment_days_summary_<first>_<last>_auto`.

There is no manual HTML review server, no browser launch, and existing manual
alignment manifests are ignored. SpikeInterface/Zarr compatibility handling from
Alignment_html.py is still used during analyzer loading.
"""

import argparse
import builtins
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import os
import re
import sys
import time
import traceback


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
STATS_DIR = SCRIPT_DIR / "Stats"
if str(STATS_DIR) not in sys.path:
    sys.path.insert(0, str(STATS_DIR))

import Alignment_html as html_review
import Alignment_days as day_review
import LDA as lda_review
import Tuning as tuning_review
import presentation_multiple as presentation_review


AUTO_DAY_EXPORT_FOLDER_NAME = "units_alignment_summary_auto"
AUTO_DAY_SUMMARY_SUFFIX = "_auto"
AUTO_STATS_SUFFIX = "_auto"

# User setting: one shared persistence threshold for both LDA.py and Tuning.py.
# Change this value here when you want both downstream modules to use a new
# minimum number of sessions per aligned unit.
MIN_SESSIONS_PER_UNIT_AUTO = 138


@dataclass
class LDASettings:
    lda_output_base_dir: Path = lda_review.OUTPUT_BASE_DIR
    lda_mode: str = lda_review.LDA_MODE
    lda_single_day_date: str | None = lda_review.SINGLE_DAY_DATE
    lda_min_firing_rate_hz: float = lda_review.MIN_FIRING_RATE_HZ
    lda_min_sessions_per_unit: int = MIN_SESSIONS_PER_UNIT_AUTO
    lda_min_bins_per_label: int = lda_review.MIN_BINS_PER_LABEL
    lda_cv_n_splits: int = lda_review.CV_N_SPLITS
    lda_n_permutations: int = lda_review.N_PERMUTATIONS
    lda_feature_modes: tuple[str, ...] = lda_review.FEATURE_MODES
    lda_extra_label_types: tuple[str, ...] = ()
    lda_use_baseline_sham_drug: bool = False
    lda_sham_session_tokens: str = ""
    lda_drug_session_tokens: str = ""
    lda_confirm_baseline_sham_drug: bool = False
    lda_injection_phase_schedule: dict | None = None
    lda_path_remap_old: Path | None = None
    lda_path_remap_new: Path | None = None


@dataclass
class TuningSettings:
    tuning_output_base_dir: Path = tuning_review.OUTPUT_BASE_DIR
    tuning_min_sessions_per_unit: int = MIN_SESSIONS_PER_UNIT_AUTO
    tuning_min_minutes_per_hour: int = tuning_review.MIN_MINUTES_PER_HOUR
    tuning_bin_size_seconds: float = tuning_review.BIN_SIZE_SECONDS
    tuning_metrics_to_plot: tuple[str, ...] = tuning_review.METRICS_TO_PLOT
    tuning_plot_types: tuple[str, ...] = tuning_review.PLOT_TYPES
    tuning_type1_units: str | tuple[int | str, ...] = tuning_review.TYPE1_UNITS
    tuning_type2_day: str | None = tuning_review.TYPE2_DAY
    tuning_normalization_methods: tuple[str, ...] = tuning_review.NORMALIZATION_METHODS
    tuning_variability_mode: str = tuning_review.VARIABILITY_MODE


@dataclass
class PresentationSettings:
    presentation_basis: str = presentation_review.DEFAULT_BASIS
    presentation_stable_threshold: int = 2
    presentation_top_n_channels: int = 20
    presentation_max_sessions: int | None = None


@dataclass
class PipelineOptions:
    input_roots: list[Path]
    skip_cross_day: bool = False
    skip_presentation: bool = False
    skip_lda: bool = False
    skip_tuning: bool = False
    stop_on_error: bool = False
    lda: LDASettings | None = None
    tuning: TuningSettings | None = None
    presentation: PresentationSettings | None = None


class AutoAlignmentState(html_review.AlignmentState):
    """Alignment_html state with manual manifest loading disabled."""

    def apply_manifest_if_available(self) -> None:
        for unit in self._iter_all_units():
            unit.merge_group = ""
            unit.align_group = ""
            unit.exclude_from_auto_align = False
            unit.is_noise = False
            unit.is_discarded = html_review.is_unit_auto_discarded(unit)


class AutoAlignmentDaysState(day_review.AlignmentDaysState):
    """Alignment_days state with manual day manifest loading disabled."""

    def apply_manifest_if_available(self) -> None:
        for unit in self._iter_all_units():
            unit.merge_group = ""
            unit.align_group = ""
            unit.exclude_from_auto_align = False
            unit.is_noise = False
            unit.is_discarded = html_review.is_unit_auto_discarded(unit)
            setattr(unit, "_source_merge_overrides", {})


_SCRIPT_START_TIME = time.perf_counter()


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


def show_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = format_duration(time.perf_counter() - _SCRIPT_START_TIME)
    print(f"[auto {timestamp} +{elapsed}] {message}", flush=True)


@contextmanager
def timed_stage(stage_name: str, timings: list[dict] | None = None):
    start_time = time.perf_counter()
    show_progress(f"Starting {stage_name}")
    try:
        yield
    except Exception:
        elapsed_seconds = time.perf_counter() - start_time
        if timings is not None:
            timings.append(
                {
                    "stage": stage_name,
                    "elapsed_seconds": float(elapsed_seconds),
                    "status": "failed",
                }
            )
        show_progress(f"Failed {stage_name} after {format_duration(elapsed_seconds)}")
        raise
    else:
        elapsed_seconds = time.perf_counter() - start_time
        if timings is not None:
            timings.append(
                {
                    "stage": stage_name,
                    "elapsed_seconds": float(elapsed_seconds),
                    "status": "completed",
                }
            )
        show_progress(f"Finished {stage_name} in {format_duration(elapsed_seconds)}")


def print_runtime_summary(timings: list[dict]) -> None:
    show_progress("Runtime summary:")
    for timing in timings:
        show_progress(
            f"  {timing['stage']}: {format_duration(timing['elapsed_seconds'])} "
            f"({timing.get('status', 'completed')})"
        )
    show_progress(f"  Total: {format_duration(time.perf_counter() - _SCRIPT_START_TIME)}")


def clean_input_root_token(raw_token: str) -> str:
    token = str(raw_token or "").strip().strip('"').strip("'").strip()
    if not token:
        return ""

    # Be tolerant of pasted/escaped Windows paths such as cc"S:\260223_Sorting.
    # The old parser treated that as a relative path under cwd. If a drive-rooted
    # path appears inside the token, use it as the intended path.
    drive_match = re.search(r"[A-Za-z]:[\\/].*", token)
    if drive_match is not None:
        token = token[drive_match.start() :]
        token = token.strip().strip('"').strip("'").strip()
    return token


def parse_input_roots_text(raw_text: str) -> list[Path]:
    parts = [clean_input_root_token(part) for part in raw_text.split(",")]
    roots = [Path(part) for part in parts if part]
    if not roots:
        raise ValueError("No input folders were provided.")
    return roots


def parse_csv_text(raw_text: str | None) -> tuple[str, ...]:
    if raw_text is None:
        return ()
    return tuple(part.strip() for part in raw_text.split(",") if part.strip())


def parse_yes_no(raw_text: str, *, default: bool = False) -> bool:
    cleaned = str(raw_text or "").strip().lower()
    if not cleaned:
        return bool(default)
    if cleaned in {"y", "yes", "true", "1"}:
        return True
    if cleaned in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"Expected yes/no, got: {raw_text!r}")


def prompt_yes_no(prompt_text: str, *, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    return parse_yes_no(input(prompt_text + suffix), default=default)


def parse_type1_units_text(raw_text: str | None):
    if raw_text is None:
        return tuning_review.TYPE1_UNITS
    cleaned = raw_text.strip()
    if not cleaned or cleaned.lower() == "all":
        return "all"
    values: list[int | str] = []
    for part in cleaned.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token) if token.isdigit() else token)
    return tuple(values)


def ensure_auto_folder(path: Path) -> Path:
    path = Path(path)
    if path.name.endswith(AUTO_STATS_SUFFIX):
        return path
    return path.with_name(f"{path.name}{AUTO_STATS_SUFFIX}")


def move_output_dir_to_auto_suffix(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    target_dir = ensure_auto_folder(output_dir)
    if output_dir == target_dir:
        return output_dir
    if target_dir.exists():
        show_progress(f"Auto output folder already exists, keeping new files in: {output_dir}")
        return output_dir
    output_dir.rename(target_dir)
    return target_dir


def immediate_child_under(path: Path, parent: Path) -> Path:
    path = Path(path)
    parent = Path(parent)
    relative = path.relative_to(parent)
    first_part = relative.parts[0]
    return parent / first_part


def prompt_for_input_roots() -> list[Path]:
    raw_text = input(
        "Enter one or more daily *_Sorting folders or parent folders, separated by commas: "
    ).strip()
    if not raw_text:
        raise ValueError("No input folders were provided.")
    return parse_input_roots_text(raw_text)


def prompt_for_pipeline_options(args: argparse.Namespace) -> PipelineOptions:
    if args.input_roots:
        selected_roots = parse_input_roots_text(args.input_roots)
    else:
        selected_roots = prompt_for_input_roots()

    lda_feature_modes = parse_csv_text(args.lda_feature_modes) or lda_review.FEATURE_MODES
    lda_extra_label_types = parse_csv_text(args.lda_extra_label_types)
    tuning_plot_types = parse_csv_text(args.tuning_plot_types) or tuning_review.PLOT_TYPES
    tuning_metrics_to_plot = parse_csv_text(args.tuning_metrics_to_plot) or tuning_review.METRICS_TO_PLOT
    tuning_normalization_methods = (
        parse_csv_text(args.tuning_normalization_methods)
        or tuning_review.NORMALIZATION_METHODS
    )
    min_sessions_per_unit = int(args.min_sessions_per_unit)

    if args.skip_lda:
        use_baseline_sham_drug = False
    elif args.lda_baseline_sham_drug == "yes":
        use_baseline_sham_drug = True
    elif args.lda_baseline_sham_drug == "no":
        use_baseline_sham_drug = False
    else:
        use_baseline_sham_drug = prompt_yes_no(
            "Add optional LDA labels baseline / sham / drug?",
            default=False,
        )
    sham_session_tokens = str(args.lda_sham_sessions or "").strip()
    drug_session_tokens = str(args.lda_drug_sessions or "").strip()
    if use_baseline_sham_drug and not sham_session_tokens:
        sham_session_tokens = input(
            "Enter sham injection session_id(s) or session name(s), separated by commas: "
        ).strip()
    if use_baseline_sham_drug and not drug_session_tokens:
        drug_session_tokens = input(
            "Enter drug injection session_id(s) or session name(s), separated by commas: "
        ).strip()
    confirm_baseline_sham_drug = bool(args.lda_confirm_baseline_sham_drug)
    if use_baseline_sham_drug and not confirm_baseline_sham_drug:
        confirm_baseline_sham_drug = prompt_yes_no(
            "Auto-confirm the derived baseline/sham/drug intervals after session matching?",
            default=False,
        )

    return PipelineOptions(
        input_roots=selected_roots,
        skip_cross_day=bool(args.skip_cross_day),
        skip_presentation=bool(args.skip_presentation),
        skip_lda=bool(args.skip_lda),
        skip_tuning=bool(args.skip_tuning),
        stop_on_error=bool(args.stop_on_error),
        lda=LDASettings(
            lda_output_base_dir=Path(args.lda_output_base_dir) if args.lda_output_base_dir else lda_review.OUTPUT_BASE_DIR,
            lda_mode=str(args.lda_mode),
            lda_single_day_date=args.lda_single_day_date,
            lda_min_firing_rate_hz=float(args.lda_min_firing_rate_hz),
            lda_min_sessions_per_unit=min_sessions_per_unit,
            lda_min_bins_per_label=int(args.lda_min_bins_per_label),
            lda_cv_n_splits=int(args.lda_cv_n_splits),
            lda_n_permutations=int(args.lda_n_permutations),
            lda_feature_modes=tuple(lda_feature_modes),
            lda_extra_label_types=tuple(lda_extra_label_types),
            lda_use_baseline_sham_drug=bool(use_baseline_sham_drug),
            lda_sham_session_tokens=sham_session_tokens,
            lda_drug_session_tokens=drug_session_tokens,
            lda_confirm_baseline_sham_drug=bool(confirm_baseline_sham_drug),
            lda_path_remap_old=Path(args.lda_path_remap_old) if args.lda_path_remap_old else None,
            lda_path_remap_new=Path(args.lda_path_remap_new) if args.lda_path_remap_new else None,
        ),
        tuning=TuningSettings(
            tuning_output_base_dir=Path(args.tuning_output_base_dir) if args.tuning_output_base_dir else tuning_review.OUTPUT_BASE_DIR,
            tuning_min_sessions_per_unit=min_sessions_per_unit,
            tuning_min_minutes_per_hour=int(args.tuning_min_minutes_per_hour),
            tuning_bin_size_seconds=float(args.tuning_bin_size_seconds),
            tuning_metrics_to_plot=tuple(tuning_metrics_to_plot),
            tuning_plot_types=tuple(tuning_plot_types),
            tuning_type1_units=parse_type1_units_text(args.tuning_type1_units),
            tuning_type2_day=args.tuning_type2_day,
            tuning_normalization_methods=tuple(tuning_normalization_methods),
            tuning_variability_mode=str(args.tuning_variability_mode),
        ),
        presentation=PresentationSettings(
            presentation_basis=str(args.presentation_basis),
            presentation_stable_threshold=int(args.presentation_stable_threshold),
            presentation_top_n_channels=int(args.presentation_top_n_channels),
            presentation_max_sessions=args.presentation_max_sessions,
        ),
    )


def auto_day_summary_folder_name(day_roots: list[Path]) -> str:
    base_name = day_review.build_day_summary_folder_name(day_roots)
    if base_name.endswith(AUTO_DAY_SUMMARY_SUFFIX):
        return base_name
    return f"{base_name}{AUTO_DAY_SUMMARY_SUFFIX}"


def configure_auto_output_names(day_summary_folder_name: str | None = None) -> None:
    html_review.DEFAULT_EXPORT_FOLDER_NAME = AUTO_DAY_EXPORT_FOLDER_NAME
    day_review.DEFAULT_EXPORT_FOLDER_NAME = AUTO_DAY_EXPORT_FOLDER_NAME
    if day_summary_folder_name is not None:
        day_review.DAY_SUMMARY_FOLDER_NAME = day_summary_folder_name
        day_review.build_day_summary_folder_name = lambda _day_roots: day_summary_folder_name


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_review.json.dumps(payload, indent=2), encoding="utf-8")


def require_existing_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Expected {label} was not written: {path}")
    return path


def require_existing_dir(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Expected {label} folder was not written: {path}")
    return path


def verify_export_result_files(export_result: dict, *, label: str) -> dict:
    file_keys = [
        "export_manifest_path",
        "unique_units_json_path",
        "unique_units_csv_path",
        "discarded_units_json_path",
        "discarded_units_csv_path",
        "noise_units_json_path",
        "noise_units_csv_path",
    ]
    verified_files = []
    for key in file_keys:
        raw_path = export_result.get(key)
        if not raw_path:
            continue
        verified_files.append(str(require_existing_file(Path(raw_path), f"{label} {key}")))
    return {"verified_files": verified_files}


def verify_page_export_files(page_export: dict, *, label: str) -> dict:
    page_exports = list(page_export.get("page_exports") or [])
    if not page_exports:
        raise RuntimeError(f"{label} did not produce any per-page export summaries.")
    verified_pages = []
    for page_result in page_exports:
        verify_export_result_files(page_result, label=f"{label} page")
        verified_pages.append(str(page_result.get("export_manifest_path", "")))
    return {
        "num_verified_page_exports": len(verified_pages),
        "verified_page_manifests": verified_pages,
    }


def load_json_file(path: Path) -> dict:
    return html_review.json.loads(Path(path).read_text(encoding="utf-8"))


def export_result_from_manifest(manifest_path: Path) -> dict:
    manifest_path = require_existing_file(manifest_path, "export manifest")
    payload = load_json_file(manifest_path)
    return {
        "export_manifest_path": str(manifest_path),
        "unique_units_json_path": str(payload.get("unique_units_summary_json", "")),
        "unique_units_csv_path": str(payload.get("unique_units_summary_csv", "")),
        "discarded_units_json_path": str(payload.get("discarded_units_summary_json", "")),
        "discarded_units_csv_path": str(payload.get("discarded_units_summary_csv", "")),
        "noise_units_json_path": str(payload.get("noise_units_summary_json", "")),
        "noise_units_csv_path": str(payload.get("noise_units_summary_csv", "")),
        "num_unique_units": len(payload.get("cross_session_alignment_groups") or []),
        "num_alignment_groups": len(payload.get("cross_session_alignment_groups") or []),
        "num_discarded_groups": None,
        "num_noise_groups": None,
    }


def page_export_result_from_manifest(manifest_path: Path) -> dict:
    export_result = export_result_from_manifest(manifest_path)
    payload = load_json_file(manifest_path)
    page_scope = payload.get("page_scope")
    if page_scope is not None:
        export_result["page_scope"] = page_scope
    return export_result


def build_existing_page_export(page_manifest_paths: list[Path]) -> dict:
    return {
        "num_pages_exported": len(page_manifest_paths),
        "page_exports": [
            page_export_result_from_manifest(manifest_path)
            for manifest_path in page_manifest_paths
        ],
    }


def existing_day_auto_export_paths(day_root: Path) -> tuple[Path, Path, list[Path]]:
    summary_root = Path(day_root) / AUTO_DAY_EXPORT_FOLDER_NAME
    summary_manifest = summary_root / "export_summary.json"
    page_manifest_paths = sorted(
        Path(day_root).glob(f"sh*/{AUTO_DAY_EXPORT_FOLDER_NAME}/export_summary_sg_*.json")
    )
    return summary_root, summary_manifest, page_manifest_paths


def existing_cross_day_auto_export_paths(common_root: Path, summary_folder_name: str) -> tuple[Path, Path, list[Path]]:
    summary_root = Path(common_root) / summary_folder_name
    summary_manifest = summary_root / "export_summary.json"
    page_manifest_paths = sorted(summary_root.glob("sh*/export_summary_sg_*.json"))
    return summary_root, summary_manifest, page_manifest_paths


def selected_day_folders_match(selection_file: Path, day_roots: list[Path]) -> bool:
    if not selection_file.exists():
        return False
    saved_roots = [
        str(Path(line.strip()).resolve())
        for line in selection_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    current_roots = [str(Path(day_root).resolve()) for day_root in day_roots]
    return saved_roots == current_roots


def build_reused_export_payload(
    *,
    summary_root: Path,
    summary_manifest: Path,
    page_manifest_paths: list[Path],
    label: str,
) -> tuple[dict, dict]:
    if not page_manifest_paths:
        raise FileNotFoundError(f"No existing {label} per-page auto export manifests found.")
    summary_export = export_result_from_manifest(summary_manifest)
    page_export = build_existing_page_export(page_manifest_paths)
    verification = {
        "summary_root": str(require_existing_dir(summary_root, f"{label} summary")),
        "page_exports": verify_page_export_files(page_export, label=label),
        "summary_export": verify_export_result_files(summary_export, label=f"{label} summary"),
    }
    return page_export, {"summary_export": summary_export, "verification": verification}


def path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def require_lda_settings(options: PipelineOptions) -> LDASettings:
    if options.lda is None:
        options.lda = LDASettings()
    return options.lda


def require_tuning_settings(options: PipelineOptions) -> TuningSettings:
    if options.tuning is None:
        options.tuning = TuningSettings()
    return options.tuning


def require_presentation_settings(options: PipelineOptions) -> PresentationSettings:
    if options.presentation is None:
        options.presentation = PresentationSettings()
    return options.presentation


def settings_payload(settings) -> dict:
    payload = asdict(settings)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


@contextmanager
def patched_input_once(response: str):
    original_input = builtins.input

    def input_once(_prompt: str = "") -> str:
        return response

    builtins.input = input_once
    try:
        yield
    finally:
        builtins.input = original_input


def select_lda_sessions_from_tokens(session_table, tokens: str, prompt_text: str):
    with patched_input_once(tokens):
        return lda_review.prompt_for_session_selection(session_table, prompt_text)


def pipeline_options_payload(options: PipelineOptions) -> dict:
    lda_settings = require_lda_settings(options)
    tuning_settings = require_tuning_settings(options)
    presentation_settings = require_presentation_settings(options)
    lda_payload = settings_payload(lda_settings)
    lda_payload["lda_path_remap_old"] = path_or_none(lda_settings.lda_path_remap_old)
    lda_payload["lda_path_remap_new"] = path_or_none(lda_settings.lda_path_remap_new)
    return {
        "input_roots": [str(path) for path in options.input_roots],
        "skip_cross_day": bool(options.skip_cross_day),
        "skip_presentation": bool(options.skip_presentation),
        "skip_lda": bool(options.skip_lda),
        "skip_tuning": bool(options.skip_tuning),
        "stop_on_error": bool(options.stop_on_error),
        "lda_settings": lda_payload,
        "tuning_settings": settings_payload(tuning_settings),
        "presentation_settings": settings_payload(presentation_settings),
    }


def run_single_day_auto_export(day_root: Path) -> dict:
    summary_root, summary_manifest, page_manifest_paths = existing_day_auto_export_paths(day_root)
    try:
        page_export, reused_payload = build_reused_export_payload(
            summary_root=summary_root,
            summary_manifest=summary_manifest,
            page_manifest_paths=page_manifest_paths,
            label=f"within-day {day_root.name}",
        )
    except Exception as exc:
        show_progress(f"No reusable within-day auto export for {day_root.name}: {exc}")
    else:
        payload = {
            "status": "reused",
            "day_root": str(day_root),
            "summary_root": str(summary_root),
            "num_sessions": None,
            "num_pages_exported": page_export.get("num_pages_exported", 0),
            "page_export": page_export,
            **reused_payload,
        }
        show_progress(
            f"Reusing existing within-day auto export: {day_root.name} -> {summary_root}"
        )
        return payload

    show_progress(f"Preparing within-day auto alignment for: {day_root}")
    state = AutoAlignmentState(day_root, progress_callback=show_progress)
    page_export = state.export_all_pages_decisions()
    summary_export = state.export_summary_bundle()
    payload = {
        "status": "computed",
        "day_root": str(day_root),
        "summary_root": str(state.summary_root),
        "num_sessions": len(state.sessions),
        "num_pages_exported": page_export.get("num_pages_exported", 0),
        "page_export": page_export,
        "summary_export": summary_export,
        "verification": {
            "summary_root": str(require_existing_dir(state.summary_root, "within-day auto summary")),
            "page_exports": verify_page_export_files(page_export, label=f"within-day {day_root.name}"),
            "summary_export": verify_export_result_files(summary_export, label=f"within-day {day_root.name} summary"),
        },
    }
    write_json(state.summary_root / "auto_alignment_run_summary.json", payload)
    show_progress(
        f"Finished within-day auto export: {day_root.name} -> {state.summary_root}"
    )
    return payload


def write_selected_day_folders(common_root: Path, summary_folder_name: str, day_roots: list[Path]) -> Path:
    selection_file = common_root / summary_folder_name / "selected_day_folders.txt"
    selection_file.parent.mkdir(parents=True, exist_ok=True)
    selection_file.write_text(
        "\n".join(str(day_root) for day_root in day_roots) + "\n",
        encoding="utf-8",
    )
    return selection_file


def run_cross_day_auto_export(
    common_root: Path,
    summary_folder_name: str,
    *,
    allow_existing_reuse: bool = True,
) -> dict:
    summary_root, summary_manifest, page_manifest_paths = existing_cross_day_auto_export_paths(
        common_root,
        summary_folder_name,
    )
    if allow_existing_reuse:
        try:
            page_export, reused_payload = build_reused_export_payload(
                summary_root=summary_root,
                summary_manifest=summary_manifest,
                page_manifest_paths=page_manifest_paths,
                label="cross-day",
            )
        except Exception as exc:
            show_progress(f"No reusable cross-day auto export under {summary_root}: {exc}")
        else:
            payload = {
                "status": "reused",
                "common_root": str(common_root),
                "summary_root": str(summary_root),
                "num_days": None,
                "num_pages_exported": page_export.get("num_pages_exported", 0),
                "page_export": page_export,
                **reused_payload,
            }
            show_progress(f"Reusing existing cross-day auto export -> {summary_root}")
            return payload
    else:
        show_progress(
            "Cross-day auto export will be recomputed because at least one requested day "
            "was newly computed or the saved day selection did not match."
        )

    show_progress(f"Preparing cross-day auto alignment under: {common_root}")
    state = AutoAlignmentDaysState(common_root, progress_callback=show_progress)
    page_export = state.export_all_pages_decisions()
    summary_export = state.export_summary_bundle()
    payload = {
        "status": "computed",
        "common_root": str(common_root),
        "summary_root": str(state.summary_root),
        "num_days": len(state.sessions),
        "num_pages_exported": page_export.get("num_pages_exported", 0),
        "page_export": page_export,
        "summary_export": summary_export,
        "verification": {
            "summary_root": str(require_existing_dir(state.summary_root, "cross-day auto summary")),
            "page_exports": verify_page_export_files(page_export, label="cross-day"),
            "summary_export": verify_export_result_files(summary_export, label="cross-day summary"),
        },
    }
    write_json(state.summary_root / "auto_alignment_days_run_summary.json", payload)
    show_progress(f"Finished cross-day auto export -> {state.summary_root}")
    return payload


def run_presentation_auto_export(
    export_summary_path: Path,
    *,
    options: PipelineOptions,
) -> dict:
    presentation_settings = require_presentation_settings(options)
    presentation_review.DEFAULT_DAYS_SUMMARY_FOLDER_PREFIX = day_review.DAY_SUMMARY_FOLDER_NAME
    payload = html_review.json.loads(export_summary_path.read_text(encoding="utf-8"))
    unique_units_csv = Path(payload["unique_units_summary_csv"])
    if not unique_units_csv.exists():
        raise FileNotFoundError(f"Referenced unique_units_summary.csv not found: {unique_units_csv}")

    base_output_dir = export_summary_path.parent / "stats_auto"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    session_summary_csv, session_summary_json, session_summary_payload = (
        presentation_review.save_source_session_summary(payload, base_output_dir)
    )
    bases = (
        ["day", "hour"]
        if presentation_settings.presentation_basis == "both"
        else [presentation_settings.presentation_basis]
    )
    all_plot_paths: list[Path] = [session_summary_csv, session_summary_json]
    manifest_runs: list[dict] = []
    quality_df = presentation_review.base_presentations.load_quality_metrics_from_export_summary(
        export_summary_path
    )

    for basis in bases:
        output_dir = presentation_review.basis_output_dir(
            base_output_dir,
            basis,
            presentation_settings.presentation_basis,
        )
        basis_output, plot_paths, num_unique_units = presentation_review.run_presentations_for_basis(
            basis=basis,
            payload=payload,
            unique_units_csv=unique_units_csv,
            quality_df=quality_df,
            output_dir=output_dir,
            max_sessions=presentation_settings.presentation_max_sessions,
            stable_threshold=presentation_settings.presentation_stable_threshold,
            top_n_channels=presentation_settings.presentation_top_n_channels,
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
    manifest_path.write_text(
        html_review.json.dumps(
            {
                "unique_units_csv": str(unique_units_csv),
                "export_summary": str(export_summary_path),
                "basis": presentation_settings.presentation_basis,
                "max_sessions": (
                    int(presentation_settings.presentation_max_sessions)
                    if presentation_settings.presentation_max_sessions is not None
                    else None
                ),
                "input_session_summary": session_summary_payload,
                "input_session_counts_by_day_csv": str(session_summary_csv),
                "input_session_counts_summary_json": str(session_summary_json),
                "runs": manifest_runs,
                "plots": [str(path) for path in all_plot_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": str(require_existing_dir(base_output_dir, "presentation auto stats")),
        "manifest_path": str(require_existing_file(manifest_path, "presentation manifest")),
        "plots": [str(path) for path in all_plot_paths],
        "verification": {
            "output_dir": str(base_output_dir),
            "manifest_path": str(manifest_path),
            "num_verified_outputs": len(
                [require_existing_file(Path(path), "presentation output") for path in all_plot_paths]
            )
            + 1,
        },
    }


def run_lda_auto_export(export_summary_path: Path, *, options: PipelineOptions) -> dict:
    lda_settings = require_lda_settings(options)
    config = lda_review.Config()
    config.data_path = export_summary_path
    config.output_base_dir = ensure_auto_folder(Path(lda_settings.lda_output_base_dir))
    config.lda_mode = lda_settings.lda_mode
    config.single_day_date = lda_settings.lda_single_day_date
    config.min_firing_rate_hz = float(lda_settings.lda_min_firing_rate_hz)
    config.min_sessions_per_unit = int(lda_settings.lda_min_sessions_per_unit)
    config.min_bins_per_label = int(lda_settings.lda_min_bins_per_label)
    config.cv_n_splits = int(lda_settings.lda_cv_n_splits)
    config.n_permutations = int(lda_settings.lda_n_permutations)
    config.feature_modes = tuple(lda_settings.lda_feature_modes)
    config.extra_label_types = tuple(lda_settings.lda_extra_label_types)
    config.injection_phase_schedule = lda_settings.lda_injection_phase_schedule

    original_prompt_for_optional_injection_phase_analysis = (
        lda_review.prompt_for_optional_injection_phase_analysis
    )
    original_prompt_for_path_remap = lda_review.prompt_for_path_remap
    configured_path_remap = (
        (lda_settings.lda_path_remap_old, lda_settings.lda_path_remap_new)
        if lda_settings.lda_path_remap_old is not None and lda_settings.lda_path_remap_new is not None
        else None
    )

    def noninteractive_injection_phase_analysis(config_arg, session_table):
        return config_arg

    def configured_prompt_for_path_remap():
        return configured_path_remap

    lda_review.prompt_for_optional_injection_phase_analysis = noninteractive_injection_phase_analysis
    lda_review.prompt_for_path_remap = configured_prompt_for_path_remap
    try:
        output_dirs = lda_review.run_pipeline(config)
    finally:
        lda_review.prompt_for_optional_injection_phase_analysis = (
            original_prompt_for_optional_injection_phase_analysis
        )
        lda_review.prompt_for_path_remap = original_prompt_for_path_remap

    moved_output_dirs = [move_output_dir_to_auto_suffix(path) for path in output_dirs]
    for output_dir in moved_output_dirs:
        require_existing_dir(output_dir, "LDA auto output")
    return {
        "output_base_dir": str(config.output_base_dir),
        "output_dirs": [str(path) for path in moved_output_dirs],
        "verification": {
            "num_output_dirs": len(moved_output_dirs),
            "output_dirs": [str(path) for path in moved_output_dirs],
        },
    }


def collect_lda_baseline_sham_drug_schedule(
    export_summary_path: Path,
    *,
    options: PipelineOptions,
) -> None:
    lda_settings = require_lda_settings(options)
    if not lda_settings.lda_use_baseline_sham_drug:
        return
    if lda_settings.lda_injection_phase_schedule is not None:
        return

    show_progress("Preparing baseline / sham / drug LDA label setup before stats stages.")
    config = lda_review.Config()
    config.data_path = export_summary_path
    config.lda_mode = lda_settings.lda_mode
    config.single_day_date = lda_settings.lda_single_day_date
    config.min_sessions_per_unit = int(lda_settings.lda_min_sessions_per_unit)
    config.extra_label_types = tuple(
        label_type
        for label_type in lda_settings.lda_extra_label_types
        if label_type != "injection_phase"
    )
    config = lda_review.apply_lda_mode_defaults(config)
    export_payload = lda_review.load_export_summary(export_summary_path)
    session_table = lda_review.build_session_table(export_payload=export_payload, config=config)
    session_table = lda_review.filter_session_table_for_lda_mode(session_table, config)

    display_columns = [
        "session_id",
        "session_name",
        "session_start_datetime",
        "session_datetime_source_field",
        "session_datetime_matched_text",
    ]
    available_columns = [column for column in display_columns if column in session_table.columns]
    print("\nAvailable sessions for baseline / sham / drug LDA labels:", flush=True)
    print(session_table[available_columns].to_string(index=False), flush=True)
    print(
        "\nBaseline is assigned automatically to samples outside sham/drug intervals.",
        flush=True,
    )
    sham_sessions = select_lda_sessions_from_tokens(
        session_table,
        lda_settings.lda_sham_session_tokens,
        "Enter sham injection session_id(s) or session name(s), separated by commas: ",
    )
    drug_sessions = select_lda_sessions_from_tokens(
        session_table,
        lda_settings.lda_drug_session_tokens,
        "Enter drug injection session_id(s) or session name(s), separated by commas: ",
    )
    schedule = lda_review.build_injection_phase_schedule(sham_sessions, drug_sessions)

    print("\nBaseline / sham / drug interpretation to confirm:", flush=True)
    print(f"  {schedule['interpretation']}", flush=True)
    print("  Sham intervals:", flush=True)
    for interval in schedule["sham_intervals"]:
        print(
            f"    {interval['session_name']} -> {interval['paired_drug_session_name']}: "
            f"[{interval['start']}, {interval['end']})",
            flush=True,
        )
    print("  Drug intervals:", flush=True)
    for interval in schedule["drug_intervals"]:
        print(
            f"    {interval['session_name']}: [{interval['start']}, {interval['end']})",
            flush=True,
        )
    if lda_settings.lda_confirm_baseline_sham_drug:
        confirm = "YES"
        print("  Auto-confirmed from startup setting.", flush=True)
    else:
        confirm = input("Is this interpretation correct? Type YES to continue: ").strip()
    if confirm != "YES":
        raise RuntimeError("Baseline / sham / drug setup was not confirmed; stopping before stats.")

    lda_settings.lda_injection_phase_schedule = schedule
    if "injection_phase" not in lda_settings.lda_extra_label_types:
        lda_settings.lda_extra_label_types = tuple(
            [*tuple(lda_settings.lda_extra_label_types or ()), "injection_phase"]
        )


def run_tuning_auto_export(export_summary_path: Path, *, options: PipelineOptions) -> dict:
    tuning_settings = require_tuning_settings(options)
    config = tuning_review.Config()
    config.data_path = export_summary_path
    config.output_base_dir = ensure_auto_folder(Path(tuning_settings.tuning_output_base_dir))
    config.min_sessions_per_unit = int(tuning_settings.tuning_min_sessions_per_unit)
    config.min_minutes_per_hour = int(tuning_settings.tuning_min_minutes_per_hour)
    config.bin_size_seconds = float(tuning_settings.tuning_bin_size_seconds)
    config.metrics_to_plot = tuple(tuning_settings.tuning_metrics_to_plot)
    config.plot_types = tuple(tuning_settings.tuning_plot_types)
    config.type1_units = tuning_settings.tuning_type1_units
    config.type2_day = tuning_settings.tuning_type2_day
    config.normalization_methods = tuple(tuning_settings.tuning_normalization_methods)
    config.variability_mode = str(tuning_settings.tuning_variability_mode)

    output_dirs = [Path(path) for path in tuning_review.run_pipeline(config)]
    if output_dirs:
        root_dir = immediate_child_under(output_dirs[0], config.output_base_dir)
        moved_root_dir = move_output_dir_to_auto_suffix(root_dir)
        moved_output_dirs = [moved_root_dir / path.relative_to(root_dir) for path in output_dirs]
    else:
        moved_root_dir = None
        moved_output_dirs = []
    return {
        "output_base_dir": str(config.output_base_dir),
        "root_output_dir": str(moved_root_dir) if moved_root_dir is not None else None,
        "output_dirs": [str(path) for path in moved_output_dirs],
        "verification": {
            "root_output_dir": str(require_existing_dir(moved_root_dir, "Tuning auto output root"))
            if moved_root_dir is not None
            else None,
            "num_output_dirs": len(moved_output_dirs),
            "existing_output_dirs": [str(path) for path in moved_output_dirs if Path(path).is_dir()],
            "missing_output_dirs": [str(path) for path in moved_output_dirs if not Path(path).is_dir()],
        },
    }


def failed_stage_payload(stage_name: str, exc: Exception) -> dict:
    return {
        "status": "failed",
        "stage": stage_name,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def run_auto_pipeline(options: PipelineOptions) -> dict:
    timings: list[dict] = []
    selected_roots = [path.resolve() for path in options.input_roots]
    day_roots = day_review.discover_day_sorting_roots(selected_roots)
    common_root = Path(os.path.commonpath([str(day_root) for day_root in day_roots]))
    day_summary_folder_name = auto_day_summary_folder_name(day_roots)

    configure_auto_output_names(day_summary_folder_name=day_summary_folder_name)

    show_progress("Pipeline inputs and configuration are locked in at startup.")
    show_progress(html_review.json.dumps(pipeline_options_payload(options), indent=2))
    show_progress(f"Discovered {len(day_roots)} day folder(s)")
    for day_root in day_roots:
        show_progress(f"  day: {day_root}")

    day_results = []
    with timed_stage("within-day auto alignment exports", timings):
        for day_root in day_roots:
            day_results.append(run_single_day_auto_export(day_root))

    selected_days_match_existing_cross_export = selected_day_folders_match(
        common_root / day_summary_folder_name / "selected_day_folders.txt",
        day_roots,
    )
    all_days_reused = all(result.get("status") == "reused" for result in day_results)
    allow_cross_day_reuse = bool(all_days_reused and selected_days_match_existing_cross_export)
    selection_file = write_selected_day_folders(common_root, day_summary_folder_name, day_roots)
    cross_day_result = None
    stats_export_summary_path: Path | None = None
    if len(day_roots) > 1 and not options.skip_cross_day:
        with timed_stage("cross-day auto alignment export", timings):
            cross_day_result = run_cross_day_auto_export(
                common_root,
                day_summary_folder_name,
                allow_existing_reuse=allow_cross_day_reuse,
            )
        stats_export_summary_path = Path(cross_day_result["summary_export"]["export_manifest_path"])
    elif len(day_roots) <= 1:
        show_progress("Only one day was provided; skipping cross-day alignment export.")
        stats_export_summary_path = Path(day_results[0]["summary_export"]["export_manifest_path"])
    else:
        show_progress("Skipping cross-day alignment export by request.")
        stats_export_summary_path = Path(day_results[-1]["summary_export"]["export_manifest_path"])

    if not options.skip_lda:
        with timed_stage("LDA baseline/sham/drug label setup", timings):
            collect_lda_baseline_sham_drug_schedule(
                stats_export_summary_path,
                options=options,
            )

    presentation_result = None
    if not options.skip_presentation and cross_day_result is not None:
        try:
            with timed_stage("presentation_multiple auto stats", timings):
                presentation_result = run_presentation_auto_export(
                    stats_export_summary_path,
                    options=options,
                )
        except Exception as exc:
            presentation_result = failed_stage_payload("presentation_multiple auto stats", exc)
            show_progress(f"presentation_multiple failed; continuing with later stages: {exc}")
            if options.stop_on_error:
                raise
    elif options.skip_presentation:
        show_progress("Skipping presentation_multiple stats by request.")
    else:
        show_progress("Skipping presentation_multiple stats because no cross-day export was produced.")

    lda_result = None
    if not options.skip_lda:
        try:
            with timed_stage("LDA auto stats", timings):
                lda_result = run_lda_auto_export(stats_export_summary_path, options=options)
        except Exception as exc:
            lda_result = failed_stage_payload("LDA auto stats", exc)
            show_progress(f"LDA failed; continuing with later stages: {exc}")
            if options.stop_on_error:
                raise
    else:
        show_progress("Skipping LDA by request.")

    tuning_result = None
    if not options.skip_tuning:
        try:
            with timed_stage("Tuning auto stats", timings):
                tuning_result = run_tuning_auto_export(stats_export_summary_path, options=options)
        except Exception as exc:
            tuning_result = failed_stage_payload("Tuning auto stats", exc)
            show_progress(f"Tuning failed; the alignment outputs remain saved: {exc}")
            if options.stop_on_error:
                raise
    else:
        show_progress("Skipping Tuning by request.")

    run_payload = {
        "input_roots": [str(path) for path in selected_roots],
        "options": pipeline_options_payload(options),
        "day_roots": [str(path) for path in day_roots],
        "common_root": str(common_root),
        "selected_day_folders": str(selection_file),
        "day_export_folder_name": AUTO_DAY_EXPORT_FOLDER_NAME,
        "day_summary_folder_name": day_summary_folder_name,
        "stats_export_summary_path": str(stats_export_summary_path),
        "day_results": day_results,
        "cross_day_result": cross_day_result,
        "presentation_result": presentation_result,
        "lda_result": lda_result,
        "tuning_result": tuning_result,
        "timings": timings,
    }
    write_json(common_root / day_summary_folder_name / "alignment_auto_run_summary.json", run_payload)
    print_runtime_summary(timings)
    return run_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run automatic within-day and cross-day unit alignment exports without "
            "serving the manual HTML review app."
        )
    )
    parser.add_argument(
        "input_roots",
        nargs="?",
        help=(
            "One or more daily *_Sorting folders or parent folders, separated by commas. "
            "If omitted, you will be prompted in the terminal."
        ),
    )
    parser.add_argument(
        "--skip-cross-day",
        action="store_true",
        help="Only write per-day units_alignment_summary_auto exports.",
    )
    parser.add_argument("--skip-presentation", action="store_true", help="Skip presentation_multiple.py outputs.")
    parser.add_argument("--skip-lda", action="store_true", help="Skip LDA.py outputs.")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip Tuning.py outputs.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop when a downstream stats stage fails.")
    parser.add_argument("--lda-output-base-dir", help="Base folder for LDA auto outputs. Defaults to OUTPUT_BASE_DIR_auto.")
    parser.add_argument("--tuning-output-base-dir", help="Base folder for Tuning auto outputs. Defaults to OUTPUT_BASE_DIR_auto.")
    parser.add_argument(
        "--presentation-basis",
        choices=["day", "hour", "both"],
        default=presentation_review.DEFAULT_BASIS,
        help="Presentation persistence basis.",
    )
    parser.add_argument("--presentation-stable-threshold", type=int, default=2)
    parser.add_argument("--presentation-top-n-channels", type=int, default=20)
    parser.add_argument("--presentation-max-sessions", type=int, default=None)
    parser.add_argument(
        "--lda-mode",
        choices=["single_day_5min", "multi_day_hourly"],
        default=lda_review.LDA_MODE,
    )
    parser.add_argument("--lda-single-day-date", default=lda_review.SINGLE_DAY_DATE)
    parser.add_argument("--lda-min-firing-rate-hz", type=float, default=lda_review.MIN_FIRING_RATE_HZ)
    parser.add_argument(
        "--min-sessions-per-unit",
        type=int,
        default=MIN_SESSIONS_PER_UNIT_AUTO,
        help=(
            "Shared minimum sessions per aligned unit for both LDA and Tuning. "
            "Defaults to MIN_SESSIONS_PER_UNIT_AUTO at the top of this script."
        ),
    )
    parser.add_argument("--lda-min-bins-per-label", type=int, default=lda_review.MIN_BINS_PER_LABEL)
    parser.add_argument("--lda-cv-n-splits", type=int, default=lda_review.CV_N_SPLITS)
    parser.add_argument("--lda-n-permutations", type=int, default=lda_review.N_PERMUTATIONS)
    parser.add_argument(
        "--lda-feature-modes",
        help="Comma-separated LDA feature modes. Defaults to LDA.py FEATURE_MODES.",
    )
    parser.add_argument(
        "--lda-extra-label-types",
        default="",
        help=(
            "Comma-separated extra LDA label types. Use --lda-baseline-sham-drug for "
            "baseline/sham/drug so that setup happens before stats stages."
        ),
    )
    parser.add_argument(
        "--lda-baseline-sham-drug",
        choices=["ask", "yes", "no"],
        default="ask",
        help="Whether to add the optional baseline/sham/drug LDA label analysis.",
    )
    parser.add_argument(
        "--lda-sham-sessions",
        help="Comma-separated sham injection session_id(s) or session name(s).",
    )
    parser.add_argument(
        "--lda-drug-sessions",
        help="Comma-separated drug injection session_id(s) or session name(s).",
    )
    parser.add_argument(
        "--lda-confirm-baseline-sham-drug",
        action="store_true",
        help="Auto-confirm the derived baseline/sham/drug intervals after matching sessions.",
    )
    parser.add_argument("--lda-path-remap-old", help="Old analyzer root prefix stored in exports.")
    parser.add_argument("--lda-path-remap-new", help="New analyzer root prefix on this machine.")
    parser.add_argument("--tuning-min-minutes-per-hour", type=int, default=tuning_review.MIN_MINUTES_PER_HOUR)
    parser.add_argument("--tuning-bin-size-seconds", type=float, default=tuning_review.BIN_SIZE_SECONDS)
    parser.add_argument(
        "--tuning-metrics-to-plot",
        help="Comma-separated tuning metrics. Defaults to Tuning.py METRICS_TO_PLOT.",
    )
    parser.add_argument(
        "--tuning-plot-types",
        help="Comma-separated tuning plot types. Defaults to Tuning.py PLOT_TYPES.",
    )
    parser.add_argument(
        "--tuning-type1-units",
        help="Use all or comma-separated final_unit_id/group keys. Defaults to Tuning.py TYPE1_UNITS.",
    )
    parser.add_argument(
        "--tuning-type2-day",
        default=tuning_review.TYPE2_DAY,
        help="Use all, a YYYY-MM-DD day, or empty string for first day.",
    )
    parser.add_argument(
        "--tuning-normalization-methods",
        help="Comma-separated normalization methods. Defaults to Tuning.py NORMALIZATION_METHODS.",
    )
    parser.add_argument(
        "--tuning-variability-mode",
        choices=["sem", "iqr"],
        default=tuning_review.VARIABILITY_MODE,
    )
    args = parser.parse_args()

    options = prompt_for_pipeline_options(args)

    try:
        result = run_auto_pipeline(options)
    except Exception as exc:
        print(f"[error] {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise

    show_progress("Automatic alignment pipeline complete.")
    show_progress(f"Run summary: {Path(result['common_root']) / result['day_summary_folder_name'] / 'alignment_auto_run_summary.json'}")


if __name__ == "__main__":
    main()
