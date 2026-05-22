from __future__ import annotations

"""
Automatic unit alignment/export pipeline.

This script combines the non-interactive parts of Alignment_html.py and
Alignment_days.py. It supports two input modes:

1. Raw analyzer mode: daily `*_Sorting` folders are loaded through the original
   SpikeInterface analyzer path.
2. Sorting_organize cache mode: `*_Sorting_org` / `unit_feature_cache` outputs
   from Sorting_organize.py are used for alignment, presentation, LDA, and
   tuning without reopening analyzers.

There is no manual HTML review server, no browser launch, and existing manual
alignment manifests are ignored.
"""

import argparse
import builtins
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
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
import Tuning_Weinan as tuning_weinan_review
import presentation_multiple as presentation_review


_ORIGINAL_DAY_LOAD_MEMBER_SNAPSHOT = day_review.load_member_snapshot

AUTO_DAY_EXPORT_FOLDER_NAME = "units_alignment_summary_auto"
AUTO_DAY_SUMMARY_SUFFIX = "_auto"
AUTO_STATS_SUFFIX = "_auto"
ORGANIZED_CACHE_FOLDER_NAME = "unit_feature_cache"
ORGANIZED_DAY_ROOT_PATTERN = re.compile(r"(?P<day_code>\d{6})_Sorting(?:_org)?$")
ORGANIZED_ENRICHED_MEMBER_KEYS = {
    "shank_id",
    "local_channel_on_shank",
    "sg_channel",
    "waveform_similarity_vector",
    "autocorrelogram_similarity_vector",
}

# User setting: one shared persistence threshold for both LDA.py and Tuning.py.
# Change this value here when you want both downstream modules to use a new
# minimum number of sessions per aligned unit.
MIN_SESSIONS_PER_UNIT_AUTO = 138
LDA_N_PERMUTATIONS_AUTO = 20
LDA_WAVEFORM_FEATURE_MODES = {"FR_WAVEFORM", "WAVEFORM_ONLY", "MULTI_FEATURE"}


@dataclass
class LDASettings:
    lda_output_base_dir: Path = lda_review.OUTPUT_BASE_DIR
    lda_label_type: str = lda_review.LABEL_TYPE
    lda_mode: str = lda_review.LDA_MODE
    lda_single_day_date: str | None = lda_review.SINGLE_DAY_DATE
    lda_min_firing_rate_hz: float = lda_review.MIN_FIRING_RATE_HZ
    lda_min_sessions_per_unit: int = MIN_SESSIONS_PER_UNIT_AUTO
    lda_min_bins_per_label: int = lda_review.MIN_BINS_PER_LABEL
    lda_cv_n_splits: int = lda_review.CV_N_SPLITS
    lda_n_permutations: int = LDA_N_PERMUTATIONS_AUTO
    lda_feature_modes: tuple[str, ...] = lda_review.FEATURE_MODES
    lda_use_waveform_features: bool = False
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
class TuningWeinanSettings:
    tuning_weinan_max_aligned_units: int | None = tuning_weinan_review.MAX_ALIGNED_UNITS


@dataclass
class PresentationSettings:
    presentation_basis: str = presentation_review.DEFAULT_BASIS
    presentation_stable_threshold: int = 2
    presentation_top_n_channels: int = 20
    presentation_max_sessions: int | None = None


@dataclass
class PipelineOptions:
    input_roots: list[Path]
    resume_export_summary_path: Path | None = None
    skip_cross_day: bool = False
    skip_presentation: bool = False
    skip_lda: bool = False
    skip_tuning: bool = False
    skip_tuning_weinan: bool = False
    run_lda_hour_label: bool = True
    run_lda_drug_label: bool = False
    stop_on_error: bool = False
    overwrite_auto_exports: bool = False
    lda: LDASettings | None = None
    tuning: TuningSettings | None = None
    tuning_weinan: TuningWeinanSettings | None = None
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


class OrganizedAutoAlignmentState(AutoAlignmentState):
    """Alignment_html state loaded from Sorting_organize.py unit_feature_cache outputs."""

    def __init__(self, root_folder: Path, progress_callback=None):
        self.root_folder = Path(root_folder).resolve()
        self.summary_root = self.root_folder / AUTO_DAY_EXPORT_FOLDER_NAME
        self.summary_root.mkdir(parents=True, exist_ok=True)
        (
            self.sessions,
            self.pages_by_shank,
            self.cache_folder,
            self.load_reports,
        ) = load_all_sessions_from_organized_caches(
            self.root_folder,
            progress_callback=progress_callback,
        )
        self._lock = html_review.threading.RLock()
        self._undo_stack: list[dict[str, dict]] = []
        self._stable_unit_aliases: dict[str, dict[str, str]] = {}
        self._stable_next_alias_index: dict[str, int] = {}
        self.discovered_shank_folder_ids = sorted(
            str(shank_id) for shank_id in self.pages_by_shank.keys()
        )
        self.loaded_shank_ids = list(self.discovered_shank_folder_ids)
        self.empty_shank_folder_ids: list[str] = []
        if progress_callback is not None:
            progress_callback("Applying organized-cache auto alignment defaults...")
        self.apply_manifest_if_available()
        if progress_callback is not None:
            progress_callback("Building organized-cache auto-merge suggestions...")
        self.sync_auto_merge_groups()
        if progress_callback is not None:
            progress_callback("Organized-cache startup state ready.")


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


def parse_day_code_from_auto_root(folder: Path) -> str | None:
    match = ORGANIZED_DAY_ROOT_PATTERN.fullmatch(Path(folder).name)
    if not match:
        return None
    return match.group("day_code")


def is_organized_cache_folder(path: Path) -> bool:
    path = Path(path)
    return (
        path.is_dir()
        and path.name == ORGANIZED_CACHE_FOLDER_NAME
        and (path / "unit_summary.json").is_file()
    )


def find_organized_day_root_for_cache(cache_folder: Path) -> Path:
    cache_folder = Path(cache_folder).resolve()
    for parent in cache_folder.parents:
        if parse_day_code_from_auto_root(parent) is not None:
            return parent
    return cache_folder.parent


def discover_organized_cache_folders(path: Path) -> list[Path]:
    path = Path(path).resolve()
    if is_organized_cache_folder(path):
        return [path]
    if not path.exists() or not path.is_dir():
        return []
    if path.name.startswith(day_review.ALIGNMENT_DAYS_SUMMARY_PREFIX) and (path / "export_summary.json").is_file():
        return []
    return sorted(
        cache_folder
        for cache_folder in path.rglob(ORGANIZED_CACHE_FOLDER_NAME)
        if is_organized_cache_folder(cache_folder)
    )


def discover_organized_day_roots(input_roots: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()

    for raw_root in input_roots:
        root = Path(raw_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Input path does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {root}")

        if parse_day_code_from_auto_root(root) is not None and discover_organized_cache_folders(root):
            candidate_roots = [root]
        else:
            child_day_roots = [
                child.resolve()
                for child in sorted(root.iterdir())
                if child.is_dir()
                and parse_day_code_from_auto_root(child) is not None
                and discover_organized_cache_folders(child)
            ]
            if child_day_roots:
                candidate_roots = child_day_roots
            else:
                cache_folders = discover_organized_cache_folders(root)
                candidate_roots = (
                    [find_organized_day_root_for_cache(cache_folder) for cache_folder in cache_folders]
                    if cache_folders
                    else []
                )

        for candidate_root in candidate_roots:
            candidate_root = Path(candidate_root).resolve()
            if candidate_root in seen:
                continue
            seen.add(candidate_root)
            discovered.append(candidate_root)

    return sorted(discovered)


def looks_like_organized_input(input_roots: list[Path]) -> bool:
    return bool(discover_organized_day_roots(input_roots))


def discover_day_roots_for_mode(input_roots: list[Path], *, organized_input: bool) -> list[Path]:
    if organized_input:
        day_roots = discover_organized_day_roots(input_roots)
        if not day_roots:
            joined_roots = ", ".join(str(Path(path).resolve()) for path in input_roots)
            raise FileNotFoundError(
                "No Sorting_organize.py unit_feature_cache folders were found in: "
                f"{joined_roots}"
            )
        return day_roots
    return day_review.discover_day_sorting_roots(input_roots)


def patch_day_review_for_auto_roots() -> None:
    day_review.parse_day_code_from_sorting_root = parse_day_code_from_auto_root
    day_review.discover_day_sorting_roots = (
        lambda input_roots: discover_day_roots_for_mode(input_roots, organized_input=True)
    )

    def load_member_snapshot_from_organized_payload(member_payload: dict, cache: dict[str, dict]) -> dict:
        if not ORGANIZED_ENRICHED_MEMBER_KEYS.issubset(member_payload.keys()):
            missing_keys = sorted(ORGANIZED_ENRICHED_MEMBER_KEYS.difference(member_payload.keys()))
            raise RuntimeError(
                "Organized-cache cross-day alignment requires enriched per-day export members, "
                f"but this member is missing: {missing_keys}. Rerun with --overwrite-auto-exports "
                "so within-day cache exports are regenerated."
            )
        return {
            "output_folder": str(member_payload.get("output_folder", "") or ""),
            "analyzer_folder": str(member_payload.get("analyzer_folder", "") or ""),
            "unit_id": int(member_payload["unit_id"]),
            "shank_id": int(member_payload["shank_id"]),
            "local_channel_on_shank": int(member_payload["local_channel_on_shank"]),
            "sg_channel": int(member_payload["sg_channel"]),
            "amplitude_median": html_review.safe_float(member_payload.get("amplitude_median")),
            "firing_rate": html_review.safe_float(member_payload.get("firing_rate")),
            "isi_violations_ratio": html_review.safe_float(member_payload.get("isi_violations_ratio")),
            "snr": html_review.safe_float(member_payload.get("snr")),
            "num_spikes": html_review.safe_int(member_payload.get("num_spikes")),
            "waveform_similarity_vector": list(
                member_payload.get("waveform_similarity_vector") or [0.0]
            ),
            "autocorrelogram_similarity_vector": list(
                member_payload.get("autocorrelogram_similarity_vector") or [0.0]
            ),
            "trough_to_peak_duration_ms": html_review.safe_float(
                member_payload.get("trough_to_peak_duration_ms")
            ),
            "waveform_image_path": str(member_payload.get("waveform_image_path", "") or ""),
        }

    day_review.load_member_snapshot = load_member_snapshot_from_organized_payload


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


def prompt_int(prompt_text: str, *, default: int) -> int:
    raw_text = input(f"{prompt_text} [{default}]: ").strip()
    if not raw_text:
        return int(default)
    return int(raw_text)


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


def filter_lda_feature_modes_for_waveform(feature_modes, *, use_waveform_features: bool) -> tuple[str, ...]:
    normalized_modes = [str(mode).strip().upper() for mode in feature_modes if str(mode).strip()]
    if use_waveform_features:
        return tuple(normalized_modes)
    filtered_modes = [
        mode
        for mode in normalized_modes
        if mode not in LDA_WAVEFORM_FEATURE_MODES
    ]
    if filtered_modes:
        return tuple(filtered_modes)
    show_progress(
        "Waveform LDA features are disabled, but all requested LDA feature modes used waveform; "
        "falling back to FR_ONLY."
    )
    return ("FR_ONLY",)


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
        "Enter one or more daily *_Sorting folders, *_Sorting_org folders, unit_feature_cache folders, or parent folders, separated by commas: "
    ).strip()
    if not raw_text:
        raise ValueError("No input folders were provided.")
    return parse_input_roots_text(raw_text)


def clean_path_text(path_text: str | Path) -> str:
    cleaned = str(path_text).strip()
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def find_export_summary_in_folder(folder: Path) -> Path:
    direct_path = folder / "export_summary.json"
    if direct_path.is_file():
        return direct_path
    candidates = sorted(folder.glob("**/export_summary.json"))
    cross_day_candidates = [
        path
        for path in candidates
        if path.parent.name.startswith("alignment_days_summary_")
    ]
    candidates_to_check = cross_day_candidates or candidates
    for candidate in candidates_to_check:
        try:
            payload = load_json_file(candidate)
        except Exception:
            continue
        if payload.get("cross_session_alignment_groups"):
            return candidate
    raise FileNotFoundError(
        "Could not find an auto cross-day export_summary.json under folder: "
        f"{folder}"
    )


def resolve_export_summary_input(path_text: str | Path) -> Path:
    path = Path(clean_path_text(path_text)).expanduser()
    if path.is_dir():
        path = find_export_summary_in_folder(path)
    return require_existing_file(path, "auto cross-day alignment export_summary.json")


def prompt_for_resume_export_summary() -> Path:
    raw_text = input(
        "Enter previous auto cross-day alignment export_summary.json path, or its folder: "
    ).strip()
    if not raw_text:
        raise ValueError("No previous alignment export was provided.")
    return resolve_export_summary_input(raw_text)


def infer_resume_input_roots(export_summary_path: Path) -> list[Path]:
    export_summary_path = Path(export_summary_path).resolve()
    if export_summary_path.name == "export_summary.json" and export_summary_path.parent.parent.exists():
        return [export_summary_path.parent.parent]
    return [export_summary_path.parent]


def prompt_for_pipeline_options(args: argparse.Namespace) -> PipelineOptions:
    resume_export_summary_path: Path | None = None
    resume_mode = bool(args.resume_export_summary)
    if not args.input_roots and not resume_mode:
        resume_mode = prompt_yes_no(
            "Resume analysis from a previous auto cross-day alignment result?",
            default=False,
        )
    if resume_mode:
        resume_export_summary_path = resolve_export_summary_input(args.resume_export_summary) if args.resume_export_summary else prompt_for_resume_export_summary()
        selected_roots = parse_input_roots_text(args.input_roots) if args.input_roots else infer_resume_input_roots(resume_export_summary_path)
    elif args.input_roots:
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
    use_waveform_features = False

    if resume_mode:
        run_lda_hour_label = prompt_yes_no("Run LDA with hour labels?", default=not bool(args.skip_lda))
        run_lda_drug_label = prompt_yes_no("Add sham/drug marker shapes to hour-label LDA figures?", default=False)
        if run_lda_drug_label:
            run_lda_hour_label = True
        run_tuning = prompt_yes_no("Run Tuning?", default=not bool(args.skip_tuning))
        run_tuning_weinan = prompt_yes_no("Run Tuning-Weinan?", default=not bool(args.skip_tuning_weinan))
        run_presentation = prompt_yes_no("Run presentation_multiple analysis?", default=not bool(args.skip_presentation))
        skip_lda = not (run_lda_hour_label or run_lda_drug_label)
        skip_tuning = not run_tuning
        skip_tuning_weinan = not run_tuning_weinan
        skip_presentation = not run_presentation
        use_baseline_sham_drug = bool(run_lda_drug_label)
    elif args.skip_lda:
        run_lda_hour_label = False
        run_lda_drug_label = False
        skip_lda = True
        skip_tuning = bool(args.skip_tuning)
        skip_tuning_weinan = bool(args.skip_tuning_weinan)
        skip_presentation = bool(args.skip_presentation)
        use_baseline_sham_drug = False
    elif args.lda_baseline_sham_drug == "yes":
        run_lda_hour_label = True
        run_lda_drug_label = True
        skip_lda = False
        skip_tuning = bool(args.skip_tuning)
        skip_tuning_weinan = bool(args.skip_tuning_weinan)
        skip_presentation = bool(args.skip_presentation)
        use_baseline_sham_drug = True
    elif args.lda_baseline_sham_drug == "no":
        run_lda_hour_label = True
        run_lda_drug_label = False
        skip_lda = False
        skip_tuning = bool(args.skip_tuning)
        skip_tuning_weinan = bool(args.skip_tuning_weinan)
        skip_presentation = bool(args.skip_presentation)
        use_baseline_sham_drug = False
    else:
        use_baseline_sham_drug = prompt_yes_no(
            "Add sham/drug marker shapes to hour-label LDA figures?",
            default=False,
        )
        run_lda_hour_label = True
        run_lda_drug_label = bool(use_baseline_sham_drug)
        skip_lda = False
        skip_tuning = bool(args.skip_tuning)
        skip_tuning_weinan = bool(args.skip_tuning_weinan)
        skip_presentation = bool(args.skip_presentation)
    if skip_lda and skip_tuning and skip_tuning_weinan:
        min_sessions_per_unit = MIN_SESSIONS_PER_UNIT_AUTO
    elif args.min_sessions_per_unit is None:
        min_sessions_per_unit = prompt_int(
            "Minimum sessions an aligned unit must appear in for LDA and Tuning",
            default=MIN_SESSIONS_PER_UNIT_AUTO,
        )
    else:
        min_sessions_per_unit = int(args.min_sessions_per_unit)

    if skip_lda:
        use_waveform_features = False
    elif args.lda_use_waveform_features == "yes":
        use_waveform_features = True
    elif args.lda_use_waveform_features == "no":
        use_waveform_features = False
    else:
        use_waveform_features = prompt_yes_no(
            "Use cached waveform as LDA feature?",
            default=False,
        )
    lda_feature_modes = filter_lda_feature_modes_for_waveform(
        lda_feature_modes,
        use_waveform_features=use_waveform_features,
    )

    sham_session_tokens = str(args.lda_sham_sessions or "").strip()
    drug_session_tokens = str(args.lda_drug_sessions or "").strip()
    confirm_baseline_sham_drug = bool(args.lda_confirm_baseline_sham_drug)

    return PipelineOptions(
        input_roots=selected_roots,
        resume_export_summary_path=resume_export_summary_path,
        skip_cross_day=bool(args.skip_cross_day),
        skip_presentation=bool(skip_presentation),
        skip_lda=bool(skip_lda),
        skip_tuning=bool(skip_tuning),
        skip_tuning_weinan=bool(skip_tuning_weinan),
        run_lda_hour_label=bool(run_lda_hour_label),
        run_lda_drug_label=bool(run_lda_drug_label),
        stop_on_error=bool(args.stop_on_error),
        overwrite_auto_exports=bool(args.overwrite_auto_exports),
        lda=LDASettings(
            lda_output_base_dir=Path(args.lda_output_base_dir) if args.lda_output_base_dir else lda_review.OUTPUT_BASE_DIR,
            lda_label_type=str(args.lda_label_type),
            lda_mode=str(args.lda_mode),
            lda_single_day_date=args.lda_single_day_date,
            lda_min_firing_rate_hz=float(args.lda_min_firing_rate_hz),
            lda_min_sessions_per_unit=min_sessions_per_unit,
            lda_min_bins_per_label=int(args.lda_min_bins_per_label),
            lda_cv_n_splits=int(args.lda_cv_n_splits),
            lda_n_permutations=int(args.lda_n_permutations),
            lda_feature_modes=tuple(lda_feature_modes),
            lda_use_waveform_features=bool(use_waveform_features),
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
    tuning_weinan=TuningWeinanSettings(
            tuning_weinan_max_aligned_units=(
                None
                if args.tuning_weinan_max_aligned_units is None
                else int(args.tuning_weinan_max_aligned_units)
            ),
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


def load_organized_unit_summary(cache_folder: Path) -> dict:
    summary_path = Path(cache_folder) / "unit_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing organized unit summary: {summary_path}")
    payload = load_json_file(summary_path)
    if isinstance(payload, list):
        return {"units": payload}
    if not isinstance(payload, dict):
        raise TypeError(f"{summary_path} is not a JSON object or unit list.")
    return payload


def load_unit_vector(cache_folder: Path, key: str) -> list[float]:
    if not key:
        return [0.0]
    similarity_path = Path(cache_folder) / "unit_similarity_vectors.npz"
    if not similarity_path.is_file():
        return [0.0]
    try:
        import numpy as np

        with np.load(similarity_path) as arrays:
            if key not in arrays:
                return [0.0]
            vector = np.asarray(arrays[key], dtype=float).ravel()
            if vector.size == 0:
                return [0.0]
            return vector.astype(float).tolist()
    except Exception:
        return [0.0]


def build_session_from_organized_cache(
    *,
    cache_folder: Path,
    session_index: int,
) -> tuple[html_review.SessionSummary, dict]:
    payload = load_organized_unit_summary(cache_folder)
    units_payload = list(payload.get("units") or [])
    if not units_payload:
        raise ValueError(f"No units found in organized cache: {cache_folder}")

    first_unit = units_payload[0]
    output_folder = Path(str(first_unit.get("output_folder") or payload.get("output_folder") or "")).resolve()
    analyzer_folder = Path(str(first_unit.get("analyzer_folder") or payload.get("analyzer_folder") or "")).resolve()
    session_name = str(first_unit.get("session_name") or output_folder.name or f"session_{session_index:03d}")

    session = html_review.SessionSummary(
        session_name=session_name,
        session_index=session_index,
        output_folder=str(output_folder),
        analyzer_folder=str(analyzer_folder),
    )

    skipped_units = 0
    for unit_payload in units_payload:
        unit_id = html_review.safe_int(unit_payload.get("unit_id"))
        shank_id = html_review.safe_int(unit_payload.get("shank_id"))
        local_channel = html_review.safe_int(unit_payload.get("local_channel_on_shank"))
        sg_channel = html_review.safe_int(unit_payload.get("sg_channel"))
        if unit_id is None or shank_id is None or local_channel is None or sg_channel is None:
            skipped_units += 1
            continue

        waveform_vector = unit_payload.get("waveform_similarity_vector")
        if not waveform_vector:
            waveform_vector = load_unit_vector(
                cache_folder,
                str(unit_payload.get("waveform_vector_key") or ""),
            )
        autocorrelogram_vector = unit_payload.get("autocorrelogram_similarity_vector")
        if not autocorrelogram_vector:
            autocorrelogram_vector = load_unit_vector(
                cache_folder,
                str(unit_payload.get("autocorrelogram_vector_key") or ""),
            )

        unit = html_review.UnitSummary(
            session_name=session_name,
            session_index=session_index,
            analyzer_folder=str(analyzer_folder),
            output_folder=str(output_folder),
            unit_id=int(unit_id),
            shank_id=int(shank_id),
            local_channel_on_shank=int(local_channel),
            sg_channel=int(sg_channel),
            amplitude_median=html_review.safe_float(unit_payload.get("amplitude_median")),
            firing_rate=html_review.safe_float(unit_payload.get("firing_rate")),
            isi_violations_ratio=html_review.safe_float(unit_payload.get("isi_violations_ratio")),
            snr=html_review.safe_float(unit_payload.get("snr")),
            num_spikes=html_review.safe_int(unit_payload.get("num_spikes")),
            waveform_similarity_vector=list(waveform_vector or [0.0]),
            autocorrelogram_similarity_vector=list(autocorrelogram_vector or [0.0]),
            trough_to_peak_duration_ms=html_review.safe_float(
                unit_payload.get("trough_to_peak_duration_ms")
            ),
            waveform_image_path=str(unit_payload.get("waveform_image_path") or ""),
        )
        session.units.append(unit)

    report = {
        "status": "loaded",
        "source": "Sorting_organize.py",
        "session_name": session.session_name,
        "session_index": int(session.session_index),
        "output_folder": session.output_folder,
        "analyzer_folder": session.analyzer_folder,
        "cache_folder": str(cache_folder),
        "unit_count": len(session.units),
        "skipped_units": int(skipped_units),
        "shank_ids": sorted({int(unit.shank_id) for unit in session.units}),
        "sg_channels": sorted({int(unit.sg_channel) for unit in session.units}),
    }
    return session, report


def build_pages_from_sessions(
    sessions: list[html_review.SessionSummary],
) -> dict[int, dict[str, html_review.PageSummary]]:
    pages_by_shank: dict[int, dict[str, html_review.PageSummary]] = defaultdict(dict)
    shank_to_channels: dict[int, set[int]] = defaultdict(set)
    for session in sessions:
        for unit in session.units:
            shank_to_channels[int(unit.shank_id)].add(int(unit.sg_channel))

    for shank_id, sg_channels in sorted(shank_to_channels.items()):
        for sg_channel in sorted(sg_channels):
            page_sessions: list[html_review.SessionSummary] = []
            for session in sessions:
                filtered_units = [
                    unit
                    for unit in session.units
                    if int(unit.shank_id) == int(shank_id)
                    and int(unit.sg_channel) == int(sg_channel)
                ]
                page_sessions.append(
                    html_review.SessionSummary(
                        session_name=session.session_name,
                        session_index=session.session_index,
                        output_folder=session.output_folder,
                        analyzer_folder=session.analyzer_folder,
                        units=filtered_units,
                    )
                )
            page = html_review.PageSummary(
                shank_id=int(shank_id),
                sg_channel=int(sg_channel),
                sessions=page_sessions,
            )
            pages_by_shank[int(shank_id)][page.page_id] = page
    return dict(pages_by_shank)


def load_all_sessions_from_organized_caches(
    root_folder: Path,
    progress_callback=None,
) -> tuple[list[html_review.SessionSummary], dict[int, dict[str, html_review.PageSummary]], Path, list[dict]]:
    cache_folders = discover_organized_cache_folders(root_folder)
    if not cache_folders:
        raise FileNotFoundError(
            f"No {ORGANIZED_CACHE_FOLDER_NAME} folders with unit_summary.json were found under: {root_folder}"
        )

    if progress_callback is not None:
        progress_callback(
            f"Found {len(cache_folders)} organized feature cache folder(s)."
        )

    sessions: list[html_review.SessionSummary] = []
    load_reports: list[dict] = []
    for session_index, cache_folder in enumerate(cache_folders):
        if progress_callback is not None:
            progress_callback(
                f"Loading organized cache {session_index + 1}/{len(cache_folders)}: {cache_folder}"
            )
        session, report = build_session_from_organized_cache(
            cache_folder=cache_folder,
            session_index=session_index,
        )
        sessions.append(session)
        load_reports.append(report)
        if progress_callback is not None:
            progress_callback(
                f"Loaded {session.session_name}: {len(session.units)} unit(s) from organized cache"
            )

    cache_folder = Path(root_folder) / AUTO_DAY_EXPORT_FOLDER_NAME / "_cache"
    cache_folder.mkdir(parents=True, exist_ok=True)
    pages_by_shank = build_pages_from_sessions(sessions)
    return sessions, pages_by_shank, cache_folder, load_reports


def unit_lookup_keys(unit) -> list[str]:
    return [
        f"output::{unit.output_folder}::{int(unit.unit_id)}",
        f"session::{unit.session_name}::{int(unit.session_index)}::{int(unit.unit_id)}",
        f"session::{unit.session_name}::{int(unit.unit_id)}",
    ]


def organized_member_payload_from_unit(unit) -> dict:
    return {
        "session_name": unit.session_name,
        "session_index": int(unit.session_index),
        "unit_id": int(unit.unit_id),
        "merge_group": unit.merge_group,
        "align_group": unit.align_group,
        "output_folder": unit.output_folder,
        "analyzer_folder": unit.analyzer_folder,
        "shank_id": int(unit.shank_id),
        "local_channel_on_shank": int(unit.local_channel_on_shank),
        "sg_channel": int(unit.sg_channel),
        "amplitude_median": unit.amplitude_median,
        "firing_rate": unit.firing_rate,
        "isi_violations_ratio": unit.isi_violations_ratio,
        "snr": unit.snr,
        "num_spikes": unit.num_spikes,
        "waveform_similarity_vector": list(unit.waveform_similarity_vector or [0.0]),
        "autocorrelogram_similarity_vector": list(
            unit.autocorrelogram_similarity_vector or [0.0]
        ),
        "trough_to_peak_duration_ms": unit.trough_to_peak_duration_ms,
        "waveform_image_path": unit.waveform_image_path,
        "member_source": "Sorting_organize.py",
    }


def enrich_organized_export_members(state: OrganizedAutoAlignmentState) -> None:
    unit_lookup: dict[str, dict] = {}
    for unit in state._iter_all_units():
        payload = organized_member_payload_from_unit(unit)
        for key in unit_lookup_keys(unit):
            unit_lookup[key] = payload

    manifest_paths = [state.summary_root / "export_summary.json"]
    manifest_paths.extend(
        sorted(Path(state.root_folder).glob(f"sh*/{AUTO_DAY_EXPORT_FOLDER_NAME}/export_summary_sg_*.json"))
    )
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            continue
        payload = load_json_file(manifest_path)
        changed = False
        for group in payload.get("cross_session_alignment_groups", []):
            enriched_members = []
            for member in group.get("members", []):
                member_payload = dict(member)
                lookup_candidates = [
                    f"output::{member_payload.get('output_folder', '')}::{html_review.safe_int(member_payload.get('unit_id'))}",
                    (
                        f"session::{member_payload.get('session_name', '')}::"
                        f"{html_review.safe_int(member_payload.get('session_index'))}::"
                        f"{html_review.safe_int(member_payload.get('unit_id'))}"
                    ),
                    f"session::{member_payload.get('session_name', '')}::{html_review.safe_int(member_payload.get('unit_id'))}",
                ]
                for key in lookup_candidates:
                    enriched = unit_lookup.get(key)
                    if enriched is not None:
                        member_payload.update(enriched)
                        changed = True
                        break
                enriched_members.append(member_payload)
            group["members"] = enriched_members
        if changed:
            write_json(manifest_path, payload)


def build_organized_cache_index(input_roots: list[Path]) -> dict[str, dict]:
    cache_index: dict[str, dict] = {}
    cache_folders: list[Path] = []
    for input_root in input_roots:
        cache_folders.extend(discover_organized_cache_folders(input_root))
    for cache_folder in sorted({path.resolve() for path in cache_folders}):
        payload = load_organized_unit_summary(cache_folder)
        units_payload = list(payload.get("units") or [])
        unit_ids: set[int] = set()
        output_folder = ""
        analyzer_folder = ""
        for unit_payload in units_payload:
            unit_id = html_review.safe_int(unit_payload.get("unit_id"))
            if unit_id is not None:
                unit_ids.add(int(unit_id))
            if not output_folder:
                output_folder = str(unit_payload.get("output_folder") or payload.get("output_folder") or "")
            if not analyzer_folder:
                analyzer_folder = str(unit_payload.get("analyzer_folder") or payload.get("analyzer_folder") or "")
        if output_folder:
            cache_info = {
                "cache_folder": cache_folder,
                "unit_ids": unit_ids,
                "unit_summary": payload,
                "minute_stats_path": cache_folder / "unit_minute_stats.csv",
                "metadata_path": cache_folder / "cache_metadata.json",
                "analyzer_folder": analyzer_folder,
            }
            cache_index[str(output_folder)] = cache_info
            try:
                cache_index[str(Path(output_folder).resolve())] = cache_info
            except Exception:
                pass
    return cache_index


def resolve_cache_for_output_folder(cache_index: dict[str, dict], output_folder: str) -> dict | None:
    if not output_folder:
        return None
    candidates = [str(output_folder)]
    try:
        candidates.append(str(Path(output_folder).resolve()))
    except Exception:
        pass
    for candidate in candidates:
        cache_info = cache_index.get(candidate)
        if cache_info is not None:
            return cache_info
    lowered = {str(key).lower(): value for key, value in cache_index.items()}
    for candidate in candidates:
        cache_info = lowered.get(str(candidate).lower())
        if cache_info is not None:
            return cache_info
    return None


def select_good_unit_groups_from_organized_cache(
    *,
    export_payload: dict,
    config,
    cache_index: dict[str, dict],
):
    pd = lda_review.pd
    selected_rows: list[dict] = []
    group_rows = export_payload.get("cross_session_alignment_groups", [])
    for group_row in group_rows:
        final_group_key = str(group_row.get("final_group_key", "")).strip()
        final_unit_id = lda_review.safe_int(group_row.get("final_unit_id"))
        valid_members: list[dict] = []
        for member in lda_review.iter_group_members(group_row):
            output_folder = str(member.get("output_folder", "") or "").strip()
            session_name = str(member.get("session_name", "") or "").strip()
            unit_id = lda_review.safe_int(member.get("unit_id"))
            if not output_folder or not session_name or unit_id is None:
                continue
            cache_info = resolve_cache_for_output_folder(cache_index, output_folder)
            if cache_info is None or int(unit_id) not in cache_info["unit_ids"]:
                continue
            valid_members.append(
                {
                    "session_key": output_folder,
                    "final_group_key": final_group_key,
                    "final_unit_id": final_unit_id,
                    "session_name": session_name,
                    "session_index": lda_review.safe_int(member.get("session_index")),
                    "unit_id": int(unit_id),
                    "output_folder": output_folder,
                    "shank_id": lda_review.safe_int(group_row.get("shank_id")),
                    "local_channel_on_shank": lda_review.safe_int(
                        group_row.get("local_channel_on_shank")
                    ),
                }
            )

        deduped_members: list[dict] = []
        seen_session_keys: set[str] = set()
        for row in valid_members:
            session_key = str(row["session_key"])
            if session_key in seen_session_keys:
                continue
            seen_session_keys.add(session_key)
            deduped_members.append(row)
        if len(deduped_members) < int(config.min_sessions_per_unit):
            continue
        for row in deduped_members:
            row["group_presence_count"] = int(len(deduped_members))
            row["min_sessions_per_unit"] = int(config.min_sessions_per_unit)
            row["selection_mode"] = "organized_cache_unique_sessions"
        selected_rows.extend(deduped_members)

    selected_table = pd.DataFrame(selected_rows)
    if selected_table.empty:
        raise RuntimeError(
            "No aligned unit groups passed the organized-cache selection criteria. "
            "Try lowering MIN_SESSIONS_PER_UNIT or verify the cache folders match the alignment export."
        )
    return selected_table.sort_values(
        ["final_unit_id", "session_index", "unit_id"],
        na_position="last",
    ).reset_index(drop=True)


def parse_cached_waveform(value) -> list[float]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        values = html_review.json.loads(text)
    except Exception:
        return []
    if not isinstance(values, list):
        return []
    parsed = []
    for item in values:
        value_float = html_review.safe_float(item)
        if value_float is not None:
            parsed.append(float(value_float))
    return parsed


def load_cached_unit_minute_summary(cache_info: dict, unit_id: int):
    pd = lda_review.pd
    cache_folder = Path(cache_info["cache_folder"])
    index_path = cache_folder / "sorted_unit_feature_outputs" / "sorted_unit_feature_output_index.json"
    if index_path.is_file():
        try:
            payload = load_json_file(index_path)
            for unit_record in payload.get("units", []) or []:
                if html_review.safe_int(unit_record.get("unit_id")) == int(unit_id):
                    minute_csv = Path(str(unit_record.get("minute_summary_csv") or ""))
                    if minute_csv.is_file():
                        return pd.read_csv(minute_csv)
        except Exception:
            pass

    matches = sorted(
        (cache_folder / "sorted_unit_feature_outputs").glob(
            f"sgch*_unit{int(unit_id)}/*_unit{int(unit_id)}_minute_summary.csv"
        )
    )
    if matches:
        return pd.read_csv(matches[0])
    return None


def infer_cached_waveform_sample_count(cache_index: dict[str, dict], configured_sample_count: int | None) -> int:
    if configured_sample_count is not None:
        sample_count = html_review.safe_int(configured_sample_count)
        if sample_count is None or sample_count < 0:
            raise ValueError("waveform_feature_sample_count must be None or a non-negative integer.")
        if sample_count == 0:
            return 0
        return int(sample_count)

    for cache_info in cache_index.values():
        for unit_id in sorted(cache_info.get("unit_ids") or []):
            unit_minutes = load_cached_unit_minute_summary(cache_info, int(unit_id))
            if unit_minutes is None or "mean_waveform_uv" not in unit_minutes.columns:
                continue
            for raw_value in unit_minutes["mean_waveform_uv"].dropna().tolist():
                waveform = parse_cached_waveform(raw_value)
                if waveform:
                    return int(len(waveform))
    return 0


def first_row_if_dataframe(row_or_table, pd_module):
    if isinstance(row_or_table, pd_module.DataFrame):
        if row_or_table.empty:
            return None
        return row_or_table.iloc[0]
    return row_or_table


def build_population_vectors_from_organized_cache(
    *,
    selected_units,
    session_table,
    cache_index: dict[str, dict],
    config,
):
    np = lda_review.np
    pd = lda_review.pd
    waveform_sample_count = infer_cached_waveform_sample_count(
        cache_index,
        getattr(config, "waveform_feature_sample_count", None),
    )
    if waveform_sample_count > 0:
        lda_review.log_status(
            f"Adding {waveform_sample_count} cached mean-waveform sample feature(s) "
            "per aligned unit group."
        )
    feature_table = lda_review.build_feature_table(
        selected_units,
        waveform_sample_count=waveform_sample_count,
    )
    feature_keys = feature_table["feature_key"].astype(str).tolist()
    feature_index = {key: index for index, key in enumerate(feature_keys)}
    members_by_session = {
        session_key: table.copy()
        for session_key, table in selected_units.groupby("session_key", sort=False)
    }

    samples: list[np.ndarray] = []
    metadata_rows: list[dict] = []
    total_sessions = len(session_table)
    metric_column_by_feature_type = {
        "firing_rate_hz": "firing_rate_hz",
        "average_amplitude_uv": "amplitude_mean_abs",
        "cv2": "cv2",
        "peak_to_trough_ms": "peak_to_trough_ms",
    }

    for session_position, session_row in enumerate(session_table.itertuples(index=False), start=1):
        session_key = str(session_row.session_key)
        session_name = str(session_row.session_name)
        lda_review.log_status(
            f"Loading organized minute stats for session {session_position} / "
            f"{total_sessions}: {session_name}"
        )
        cache_info = resolve_cache_for_output_folder(cache_index, session_key)
        if cache_info is None:
            lda_review.log_status(f"Skipping {session_name}: no organized cache found.")
            continue
        minute_stats_path = Path(cache_info["minute_stats_path"])
        if not minute_stats_path.is_file():
            lda_review.log_status(f"Skipping {session_name}: missing {minute_stats_path}")
            continue
        minute_stats = pd.read_csv(minute_stats_path)
        if minute_stats.empty:
            continue
        minute_stats["unit_id"] = pd.to_numeric(minute_stats["unit_id"], errors="coerce")
        minute_stats["minute_index"] = pd.to_numeric(minute_stats["minute_index"], errors="coerce")
        minute_stats["start_sec"] = pd.to_numeric(minute_stats["start_sec"], errors="coerce")
        minute_stats["end_sec"] = pd.to_numeric(minute_stats["end_sec"], errors="coerce")
        minute_stats = minute_stats.dropna(subset=["minute_index", "start_sec", "end_sec"])
        if minute_stats.empty:
            continue

        minute_rows = (
            minute_stats[["minute_index", "start_sec", "end_sec"]]
            .drop_duplicates()
            .sort_values("minute_index")
            .reset_index(drop=True)
        )
        session_matrix = np.full((len(minute_rows), len(feature_keys)), np.nan, dtype=float)
        unit_rows_by_id = {
            int(unit_id): table.copy()
            for unit_id, table in minute_stats.groupby("unit_id", sort=False)
            if lda_review.safe_int(unit_id) is not None
        }
        unit_waveform_rows_by_id = {}
        session_units = members_by_session.get(session_key, pd.DataFrame())
        for member_row in session_units.itertuples(index=False):
            unit_id = int(member_row.unit_id)
            unit_table = unit_rows_by_id.get(unit_id)
            if unit_table is None or unit_table.empty:
                continue
            unit_table = unit_table.set_index("minute_index", drop=False)
            waveform_table = None
            if waveform_sample_count > 0:
                if unit_id not in unit_waveform_rows_by_id:
                    cached_waveforms = load_cached_unit_minute_summary(cache_info, unit_id)
                    if cached_waveforms is not None and not cached_waveforms.empty:
                        cached_waveforms["minute_index"] = pd.to_numeric(
                            cached_waveforms["minute_index"],
                            errors="coerce",
                        )
                        cached_waveforms = cached_waveforms.dropna(subset=["minute_index"])
                        cached_waveforms["minute_index"] = cached_waveforms["minute_index"].astype(int)
                        unit_waveform_rows_by_id[unit_id] = cached_waveforms.set_index(
                            "minute_index",
                            drop=False,
                        )
                    else:
                        unit_waveform_rows_by_id[unit_id] = None
                waveform_table = unit_waveform_rows_by_id.get(unit_id)
            feature_key_prefix = str(member_row.final_group_key)
            for minute_position, minute_row in enumerate(minute_rows.itertuples(index=False)):
                source_row = unit_table.loc[minute_row.minute_index] if minute_row.minute_index in unit_table.index else None
                if source_row is None:
                    continue
                source_row = first_row_if_dataframe(source_row, pd)
                if source_row is None:
                    continue
                for feature_type, metric_column in metric_column_by_feature_type.items():
                    full_feature_key = f"{feature_key_prefix}__{feature_type}"
                    if full_feature_key not in feature_index or metric_column not in unit_table.columns:
                        continue
                    session_matrix[minute_position, feature_index[full_feature_key]] = lda_review.safe_float(
                        source_row.get(metric_column)
                    )
                if waveform_table is not None and int(minute_row.minute_index) in waveform_table.index:
                    waveform_row = waveform_table.loc[int(minute_row.minute_index)]
                    waveform_row = first_row_if_dataframe(waveform_row, pd)
                    if waveform_row is None:
                        continue
                    waveform = parse_cached_waveform(waveform_row.get("mean_waveform_uv"))
                    if waveform:
                        waveform_features = lda_review.resample_waveform_vector(
                            lda_review.np.asarray(waveform, dtype=float),
                            waveform_sample_count,
                        )
                        for waveform_sample_index, waveform_value in enumerate(waveform_features):
                            waveform_feature_key = (
                                f"{feature_key_prefix}__mean_waveform_uv_s{waveform_sample_index:03d}"
                            )
                            if waveform_feature_key in feature_index:
                                session_matrix[
                                    minute_position,
                                    feature_index[waveform_feature_key],
                                ] = waveform_value

        session_start_datetime = session_row.session_start_datetime
        for bin_index, minute_row in enumerate(minute_rows.itertuples(index=False)):
            bin_start_sec = float(minute_row.start_sec)
            bin_end_sec = float(minute_row.end_sec)
            bin_center_s = bin_start_sec + (bin_end_sec - bin_start_sec) / 2.0
            bin_start_datetime = session_start_datetime + lda_review.timedelta(seconds=bin_start_sec)
            bin_end_datetime = session_start_datetime + lda_review.timedelta(seconds=bin_end_sec)
            samples.append(session_matrix[bin_index])
            metadata_rows.append(
                {
                    "session_id": int(session_row.session_id),
                    "session_key": session_key,
                    "session_name": session_name,
                    "session_name_normalized": str(session_row.session_name_normalized),
                    "session_index": lda_review.safe_int(session_row.session_index),
                    "session_start_datetime": session_start_datetime.isoformat(sep=" "),
                    "minute_bin_index": int(minute_row.minute_index),
                    "minute_start_sec": bin_start_sec,
                    "minute_end_sec": bin_end_sec,
                    "minute_center_s": float(bin_center_s),
                    "session_duration_s": float(max(minute_rows["end_sec"])),
                    "minute_start_datetime": bin_start_datetime.isoformat(sep=" "),
                    "minute_end_datetime": bin_end_datetime.isoformat(sep=" "),
                    "clock_hour_of_day": int(bin_start_datetime.hour),
                    "clock_minute_of_hour": int(bin_start_datetime.minute),
                    "calendar_day": bin_start_datetime.date().isoformat(),
                }
            )

    if not samples:
        raise RuntimeError("No organized-cache population vectors were created.")
    population_matrix = np.vstack(samples)
    metadata_table = pd.DataFrame(metadata_rows)
    if getattr(config, "apply_smoothing", False):
        smoothable_feature_mask = (
            feature_table["feature_type"].astype(str).to_numpy() == "firing_rate_hz"
        )
        lda_review.log_status("Applying smoothing to organized-cache firing-rate features")
        population_matrix = lda_review.smooth_population_matrix(
            population_matrix=population_matrix,
            sigma_bins=float(config.smoothing_sigma_bins),
            feature_mask=smoothable_feature_mask,
        )
    lda_review.log_status(
        f"Finished organized-cache binning: created {population_matrix.shape[0]} samples "
        f"x {population_matrix.shape[1]} features"
    )
    return population_matrix, metadata_table, feature_table


def load_cache_metadata(cache_info: dict) -> dict:
    metadata_path = Path(cache_info.get("metadata_path", ""))
    if not metadata_path.is_file():
        return {}
    try:
        return load_json_file(metadata_path)
    except Exception:
        return {}


def build_quality_metrics_from_organized_export(
    export_summary_path: Path,
    *,
    cache_index: dict[str, dict],
):
    pd = presentation_review.pd
    payload = load_json_file(export_summary_path)
    rows: list[dict] = []
    for group_row in payload.get("cross_session_alignment_groups", []) or []:
        final_group_key = str(group_row.get("final_group_key", "") or "")
        final_unit_id = html_review.safe_int(group_row.get("final_unit_id"))
        shank_id = html_review.safe_int(group_row.get("shank_id"))
        members = group_row.get("source_members") or group_row.get("members") or []
        unique_session_count = len(
            {
                str(member.get("session_name", "") or "").strip()
                for member in members
                if str(member.get("session_name", "") or "").strip()
            }
        )
        for member in members:
            output_folder = str(member.get("output_folder", "") or "").strip()
            cache_info = resolve_cache_for_output_folder(cache_index, output_folder)
            metadata = load_cache_metadata(cache_info) if cache_info is not None else {}
            session_duration_s = html_review.safe_float(metadata.get("session_duration_seconds"))
            num_spikes = html_review.safe_float(member.get("num_spikes"))
            firing_rate = html_review.safe_float(member.get("firing_rate"))
            amplitude_median = html_review.safe_float(member.get("amplitude_median"))
            rows.append(
                {
                    "final_group_key": final_group_key,
                    "final_unit_id": final_unit_id,
                    "shank_id": shank_id or html_review.safe_int(member.get("shank_id")),
                    "session_name": str(member.get("session_name", "") or ""),
                    "session_index": html_review.safe_int(member.get("session_index")),
                    "unit_id": html_review.safe_int(member.get("unit_id")),
                    "num_sessions": int(unique_session_count),
                    "amplitude_median": amplitude_median,
                    "amplitude_median_abs": abs(amplitude_median) if amplitude_median is not None else None,
                    "firing_rate_full_session": firing_rate,
                    "firing_rate_active_window": firing_rate,
                    "isi_violations_ratio": html_review.safe_float(member.get("isi_violations_ratio")),
                    "snr": html_review.safe_float(member.get("snr")),
                    "num_spikes": num_spikes,
                    "session_duration_s": session_duration_s,
                    "active_window_start_s": 0.0 if session_duration_s is not None else None,
                    "active_window_end_s": session_duration_s,
                    "active_duration_s": session_duration_s,
                    "active_fraction_of_session": 1.0 if session_duration_s is not None else None,
                    "quality_source": "Sorting_organize.py",
                }
            )
    quality_df = pd.DataFrame(rows)
    presentation_review.log_status(
        f"Loaded organized-cache quality-metric rows: {len(quality_df)}"
    )
    return quality_df


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


def remove_dir_if_exists(path: Path, *, label: str) -> None:
    path = Path(path)
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(f"Cannot overwrite {label}; path is not a directory: {path}")
    show_progress(f"Removing existing {label}: {path}")
    html_review.shutil.rmtree(path)


def remove_within_day_auto_exports(day_root: Path) -> None:
    summary_root, _summary_manifest, _page_manifest_paths = existing_day_auto_export_paths(day_root)
    remove_dir_if_exists(summary_root, label="within-day auto summary")
    for shank_folder in sorted(Path(day_root).glob("sh*")):
        if not shank_folder.is_dir():
            continue
        page_summary_root = shank_folder / AUTO_DAY_EXPORT_FOLDER_NAME
        remove_dir_if_exists(page_summary_root, label="within-day per-page auto summary")


def remove_cross_day_auto_exports(common_root: Path, summary_folder_name: str) -> None:
    summary_root, _summary_manifest, _page_manifest_paths = existing_cross_day_auto_export_paths(
        common_root,
        summary_folder_name,
    )
    remove_dir_if_exists(summary_root, label="cross-day auto summary")


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


def export_members_are_organized_enriched(manifest_paths: list[Path]) -> bool:
    saw_member = False
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            continue
        payload = load_json_file(manifest_path)
        for group in payload.get("cross_session_alignment_groups", []) or []:
            for member in group.get("members", []) or []:
                saw_member = True
                if not ORGANIZED_ENRICHED_MEMBER_KEYS.issubset(member.keys()):
                    return False
    return saw_member


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


def require_tuning_weinan_settings(options: PipelineOptions) -> TuningWeinanSettings:
    if options.tuning_weinan is None:
        options.tuning_weinan = TuningWeinanSettings()
    return options.tuning_weinan


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


def build_injection_recording_table(session_table):
    table = session_table.reset_index(drop=False).copy()
    table["_recording_name"] = table["session_name"].map(lda_review.normalize_session_name)
    table["_recording_start"] = lda_review.pd.to_datetime(
        table["session_start_datetime"],
        errors="coerce",
    )
    rows: list[dict] = []
    grouped = table.groupby(["_recording_name", "_recording_start"], dropna=False, sort=True)
    for recording_id, ((_recording_name, _recording_start), group) in enumerate(grouped, start=1):
        sorted_group = group.sort_values("session_id")
        rows.append(
            {
                "recording_id": int(recording_id),
                "recording_name": str(_recording_name),
                "session_start_datetime": (
                    _recording_start.isoformat(sep=" ")
                    if hasattr(_recording_start, "isoformat") and not lda_review.pd.isna(_recording_start)
                    else ""
                ),
                "num_session_rows": int(len(sorted_group)),
                "session_ids": ",".join(str(value) for value in sorted_group["session_id"].astype(int).tolist()),
                "session_names": ",".join(sorted_group["session_name"].astype(str).tolist()),
                "source_indices": sorted_group["index"].astype(int).tolist(),
            }
        )
    return lda_review.pd.DataFrame(rows)


def split_selection_tokens(tokens: str) -> list[str]:
    return [
        token
        for token in (
            part.strip().strip('"').strip("'")
            for part in re.split(r"[,;\n]+", str(tokens or ""))
        )
        if token
    ]


def select_lda_recordings_from_tokens(session_table, tokens: str, prompt_text: str):
    raw_tokens = split_selection_tokens(tokens)
    if not raw_tokens:
        raw_tokens = split_selection_tokens(input(prompt_text).strip())
    if not raw_tokens:
        return session_table.iloc[0:0].copy()

    recording_table = build_injection_recording_table(session_table)
    selected_source_indices: list[int] = []
    unmatched_tokens: list[str] = []
    for token in raw_tokens:
        matches = lda_review.pd.DataFrame()
        parsed_id = html_review.safe_int(token)
        if parsed_id is not None and "recording_id" in recording_table.columns:
            matches = recording_table[recording_table["recording_id"].astype(int) == int(parsed_id)]
        if matches.empty:
            name_values = recording_table["recording_name"].astype(str)
            matches = recording_table[name_values.str.lower() == token.lower()]
        if matches.empty:
            matches = recording_table[
                recording_table["recording_name"].astype(str).str.contains(
                    re.escape(token),
                    case=False,
                    regex=True,
                    na=False,
                )
            ]
        if matches.empty:
            with patched_input_once(token):
                matched_sessions = lda_review.prompt_for_session_selection(
                    session_table,
                    prompt_text,
                )
            if matched_sessions.empty:
                unmatched_tokens.append(token)
            else:
                selected_session_ids = set(matched_sessions["session_id"].astype(int).tolist())
                source_matches = session_table[
                    session_table["session_id"].astype(int).isin(selected_session_ids)
                ]
                selected_source_indices.extend(int(index) for index in source_matches.index.tolist())
            continue
        for source_indices in matches["source_indices"].tolist():
            selected_source_indices.extend(int(index) for index in source_indices)

    if unmatched_tokens:
        raise ValueError(f"Could not match recording token(s): {unmatched_tokens}")
    selected_source_indices = sorted(set(selected_source_indices))
    return session_table.loc[selected_source_indices].sort_values("session_start_datetime").reset_index(drop=True)


def pipeline_options_payload(options: PipelineOptions) -> dict:
    lda_settings = require_lda_settings(options)
    tuning_settings = require_tuning_settings(options)
    presentation_settings = require_presentation_settings(options)
    lda_payload = settings_payload(lda_settings)
    lda_payload["lda_path_remap_old"] = path_or_none(lda_settings.lda_path_remap_old)
    lda_payload["lda_path_remap_new"] = path_or_none(lda_settings.lda_path_remap_new)
    return {
        "input_roots": [str(path) for path in options.input_roots],
        "resume_export_summary_path": path_or_none(options.resume_export_summary_path),
        "skip_cross_day": bool(options.skip_cross_day),
        "skip_presentation": bool(options.skip_presentation),
        "skip_lda": bool(options.skip_lda),
        "skip_tuning": bool(options.skip_tuning),
        "skip_tuning_weinan": bool(options.skip_tuning_weinan),
        "run_lda_hour_label": bool(options.run_lda_hour_label),
        "run_lda_drug_label": bool(options.run_lda_drug_label),
        "stop_on_error": bool(options.stop_on_error),
        "overwrite_auto_exports": bool(options.overwrite_auto_exports),
        "lda_settings": lda_payload,
        "tuning_settings": settings_payload(tuning_settings),
        "tuning_weinan_settings": settings_payload(require_tuning_weinan_settings(options)),
        "presentation_settings": settings_payload(presentation_settings),
    }


def run_single_day_auto_export(
    day_root: Path,
    *,
    organized_input: bool = False,
    overwrite: bool = False,
) -> dict:
    if overwrite:
        remove_within_day_auto_exports(day_root)
    summary_root, summary_manifest, page_manifest_paths = existing_day_auto_export_paths(day_root)
    if not overwrite:
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
            if organized_input:
                manifest_paths = [summary_manifest, *page_manifest_paths]
                if export_members_are_organized_enriched(manifest_paths):
                    show_progress(
                        f"Reusable within-day auto export is already cache-enriched: {day_root.name}"
                    )
                else:
                    show_progress(
                        f"Reusable within-day auto export needs cache enrichment: {day_root.name}"
                    )
                    enrich_organized_export_members(
                        OrganizedAutoAlignmentState(day_root, progress_callback=show_progress)
                    )
                    page_manifest_paths = existing_day_auto_export_paths(day_root)[2]
                    page_export = build_existing_page_export(page_manifest_paths)
                    reused_payload["summary_export"] = export_result_from_manifest(summary_manifest)
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
    else:
        show_progress(f"Overwrite requested; recomputing within-day auto export for {day_root.name}.")

    if organized_input:
        show_progress(f"Preparing within-day auto alignment from organized cache: {day_root}")
        state = OrganizedAutoAlignmentState(day_root, progress_callback=show_progress)
    else:
        show_progress(f"Preparing within-day auto alignment for: {day_root}")
        state = AutoAlignmentState(day_root, progress_callback=show_progress)
    page_export = state.export_all_pages_decisions()
    summary_export = state.export_summary_bundle()
    if organized_input:
        enrich_organized_export_members(state)
        summary_export = export_result_from_manifest(Path(summary_export["export_manifest_path"]))
        page_export = build_existing_page_export(existing_day_auto_export_paths(day_root)[2])
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
    cache_index = (
        build_organized_cache_index(options.input_roots)
        if looks_like_organized_input(options.input_roots)
        else {}
    )
    if cache_index:
        quality_df = build_quality_metrics_from_organized_export(
            export_summary_path,
            cache_index=cache_index,
        )
    else:
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


def default_stats_output_base_dir(export_summary_path: Path, name: str) -> Path:
    return Path(export_summary_path).parent / name


def build_tagged_lda_output_run_dir(config, date_label: str, label_types: list[str], waveform_tag: str) -> Path:
    run_id = config.output_run_id or lda_review.build_run_id()
    config.output_run_id = run_id
    folder_parts = [
        f"lda_{lda_review.output_slug(lda_review.concise_date_label(date_label), 48)}",
        f"minsess{int(config.min_sessions_per_unit)}",
        waveform_tag,
    ]
    threshold_tag = lda_review.build_threshold_tag(config.data_path)
    if threshold_tag:
        folder_parts.append(lda_review.output_slug(threshold_tag, 180))
    if lda_review.input_path_has_auto(config.data_path):
        folder_parts.append("auto")
    folder_stem = "__".join(folder_parts)
    output_dir = lda_review.unique_concise_output_dir(
        base_dir=config.output_base_dir,
        folder_stem=folder_stem,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def run_lda_auto_export(export_summary_path: Path, *, options: PipelineOptions) -> dict:
    lda_settings = require_lda_settings(options)
    config = lda_review.Config()
    config.data_path = export_summary_path
    if lda_settings.lda_output_base_dir == lda_review.OUTPUT_BASE_DIR:
        config.output_base_dir = default_stats_output_base_dir(export_summary_path, "LDA_auto")
    else:
        config.output_base_dir = ensure_auto_folder(Path(lda_settings.lda_output_base_dir))
    config.label_type = str(lda_settings.lda_label_type)
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
    if not lda_settings.lda_use_waveform_features:
        config.waveform_feature_sample_count = 0

    original_prompt_for_optional_injection_phase_analysis = (
        lda_review.prompt_for_optional_injection_phase_analysis
    )
    original_prompt_for_path_remap = lda_review.prompt_for_path_remap
    original_build_output_run_dir = lda_review.build_output_run_dir
    configured_path_remap = (
        (lda_settings.lda_path_remap_old, lda_settings.lda_path_remap_new)
        if lda_settings.lda_path_remap_old is not None and lda_settings.lda_path_remap_new is not None
        else None
    )

    def noninteractive_injection_phase_analysis(config_arg, session_table):
        return config_arg

    def configured_prompt_for_path_remap():
        return configured_path_remap

    waveform_tag = "waveform" if lda_settings.lda_use_waveform_features else "no_waveform"

    def tagged_build_output_run_dir(*, config, date_label, label_types):
        return build_tagged_lda_output_run_dir(
            config,
            date_label=date_label,
            label_types=label_types,
            waveform_tag=waveform_tag,
        )

    cache_index = (
        build_organized_cache_index(options.input_roots)
        if looks_like_organized_input(options.input_roots)
        else {}
    )
    original_load_session_analyzers = lda_review.load_session_analyzers
    original_select_good_unit_groups = lda_review.select_good_unit_groups
    original_build_population_vectors = lda_review.build_population_vectors

    def cache_load_session_analyzers(session_table, config):
        resolved_output_folders = {
            str(row.session_key): str(row.output_folder)
            for row in session_table.itertuples(index=False)
        }
        lda_review.log_status(
            "Using Sorting_organize.py cache input; skipping analyzer loading for LDA."
        )
        return {}, resolved_output_folders

    def cache_select_good_unit_groups(export_payload, config, analyzers):
        return select_good_unit_groups_from_organized_cache(
            export_payload=export_payload,
            config=config,
            cache_index=cache_index,
        )

    def cache_build_population_vectors(selected_units, session_table, analyzers, config):
        return build_population_vectors_from_organized_cache(
            selected_units=selected_units,
            session_table=session_table,
            cache_index=cache_index,
            config=config,
        )

    lda_review.prompt_for_optional_injection_phase_analysis = noninteractive_injection_phase_analysis
    lda_review.prompt_for_path_remap = configured_prompt_for_path_remap
    lda_review.build_output_run_dir = tagged_build_output_run_dir
    if cache_index:
        lda_review.load_session_analyzers = cache_load_session_analyzers
        lda_review.select_good_unit_groups = cache_select_good_unit_groups
        lda_review.build_population_vectors = cache_build_population_vectors
    try:
        output_dirs = lda_review.run_pipeline(config)
    finally:
        lda_review.prompt_for_optional_injection_phase_analysis = (
            original_prompt_for_optional_injection_phase_analysis
        )
        lda_review.prompt_for_path_remap = original_prompt_for_path_remap
        lda_review.build_output_run_dir = original_build_output_run_dir
        lda_review.load_session_analyzers = original_load_session_analyzers
        lda_review.select_good_unit_groups = original_select_good_unit_groups
        lda_review.build_population_vectors = original_build_population_vectors

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
    config.label_type = str(lda_settings.lda_label_type)
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
    print("\nAvailable recordings for baseline / sham / drug LDA labels:", flush=True)
    recording_table = build_injection_recording_table(session_table)
    recording_display_columns = [
        "recording_id",
        "recording_name",
        "session_start_datetime",
        "num_session_rows",
        "session_ids",
    ]
    print(recording_table[recording_display_columns].to_string(index=False), flush=True)
    print(
        "\nYou can enter recording_id values or recording_name tokens such as 260223_00; "
        "all matching shank/session rows will be selected together.",
        flush=True,
    )
    print("\nUnderlying session rows:", flush=True)
    print(session_table[available_columns].to_string(index=False), flush=True)
    print(
        "\nBaseline is assigned automatically to samples outside sham/drug intervals.",
        flush=True,
    )
    sham_sessions = select_lda_recordings_from_tokens(
        session_table,
        lda_settings.lda_sham_session_tokens,
        "Enter sham injection recording_id(s), recording token(s), session_id(s), or session name(s), separated by commas: ",
    )
    drug_sessions = select_lda_recordings_from_tokens(
        session_table,
        lda_settings.lda_drug_session_tokens,
        "Enter drug injection recording_id(s), recording token(s), session_id(s), or session name(s), separated by commas: ",
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


def run_tuning_auto_export(export_summary_path: Path, *, options: PipelineOptions) -> dict:
    tuning_settings = require_tuning_settings(options)
    config = tuning_review.Config()
    config.data_path = export_summary_path
    if tuning_settings.tuning_output_base_dir == tuning_review.OUTPUT_BASE_DIR:
        config.output_base_dir = default_stats_output_base_dir(export_summary_path, "Tuning_auto")
    else:
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

    cache_index = (
        build_organized_cache_index(options.input_roots)
        if looks_like_organized_input(options.input_roots)
        else {}
    )
    original_load_aligned_minute_data = tuning_review.load_aligned_minute_data

    def cache_load_aligned_minute_data(tuning_config):
        lda_config = tuning_review.build_lda_config(tuning_config)
        lda_config.waveform_feature_sample_count = 0
        export_path = lda_review.resolve_export_summary_path(tuning_config.data_path)
        tuning_review.log_status(f"Loading alignment export: {export_path}")
        export_payload = lda_review.load_export_summary(export_path)
        session_table = lda_review.build_session_table(export_payload=export_payload, config=lda_config)
        session_table = lda_review.filter_session_table_for_lda_mode(session_table, lda_config)
        selected_units = select_good_unit_groups_from_organized_cache(
            export_payload=export_payload,
            config=lda_config,
            cache_index=cache_index,
        )
        minute_matrix, minute_metadata, feature_table = build_population_vectors_from_organized_cache(
            selected_units=selected_units,
            session_table=session_table,
            cache_index=cache_index,
            config=lda_config,
        )
        feature_columns = feature_table["feature_column"].astype(str).tolist()
        minute_values = lda_review.pd.DataFrame(minute_matrix, columns=feature_columns)
        minute_wide = lda_review.pd.concat(
            [minute_metadata.reset_index(drop=True), minute_values.reset_index(drop=True)],
            axis=1,
        )
        minute_wide["time_of_day_hour"] = minute_wide["minute_start_datetime"].map(
            tuning_review.parse_time_of_day
        )
        tuning_review.log_status(
            "Using Sorting_organize.py cache input; skipping analyzer loading for Tuning."
        )
        return minute_wide, feature_table, selected_units, export_path

    if cache_index:
        tuning_review.load_aligned_minute_data = cache_load_aligned_minute_data
    try:
        output_dirs = [Path(path) for path in tuning_review.run_pipeline(config)]
    finally:
        tuning_review.load_aligned_minute_data = original_load_aligned_minute_data
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


@contextmanager
def patched_argv(argv: list[str]):
    original_argv = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = original_argv


def run_tuning_weinan_auto_export(export_summary_path: Path, *, options: PipelineOptions) -> dict:
    tuning_settings = require_tuning_settings(options)
    weinan_settings = require_tuning_weinan_settings(options)
    cache_index = (
        build_organized_cache_index(options.input_roots)
        if looks_like_organized_input(options.input_roots)
        else {}
    )

    original_min_sessions = tuning_weinan_review.MIN_SESSIONS_PER_UNIT
    original_bin_size = tuning_weinan_review.BIN_SIZE_SECONDS
    original_load_aligned_minute_data = tuning_weinan_review.load_aligned_minute_data_like_tuning

    def cache_load_aligned_minute_data(data_path: Path):
        aligned_input_path = tuning_weinan_review.normalize_aligned_input_path(data_path)
        config = tuning_weinan_review.build_lda_config(aligned_input_path)
        config.waveform_feature_sample_count = 0
        export_path = lda_review.resolve_export_summary_path(aligned_input_path)
        tuning_weinan_review.log_status(f"Loading alignment export: {export_path}")
        export_payload = lda_review.load_export_summary(export_path)
        session_table = lda_review.build_session_table(export_payload=export_payload, config=config)
        session_table = lda_review.filter_session_table_for_lda_mode(session_table, config)
        selected_units = select_good_unit_groups_from_organized_cache(
            export_payload=export_payload,
            config=config,
            cache_index=cache_index,
        )
        minute_matrix, minute_metadata, feature_table = build_population_vectors_from_organized_cache(
            selected_units=selected_units,
            session_table=session_table,
            cache_index=cache_index,
            config=config,
        )
        feature_columns = feature_table["feature_column"].astype(str).tolist()
        minute_values = lda_review.pd.DataFrame(minute_matrix, columns=feature_columns)
        minute_wide = lda_review.pd.concat(
            [minute_metadata.reset_index(drop=True), minute_values.reset_index(drop=True)],
            axis=1,
        )
        tuning_weinan_review.log_status(
            "Using Sorting_organize.py cache input; skipping analyzer loading for Tuning-Weinan."
        )
        return minute_wide, feature_table, selected_units, export_path

    output_root = Path(export_summary_path).parent / "tuning_weinan_aligned_units"
    argv = [
        "Tuning_Weinan.py",
        "--run-root",
        str(export_summary_path),
        "--input-mode",
        "aligned",
    ]
    if weinan_settings.tuning_weinan_max_aligned_units is not None:
        argv.extend(["--max-aligned-units", str(int(weinan_settings.tuning_weinan_max_aligned_units))])

    tuning_weinan_review.MIN_SESSIONS_PER_UNIT = int(tuning_settings.tuning_min_sessions_per_unit)
    tuning_weinan_review.BIN_SIZE_SECONDS = float(tuning_settings.tuning_bin_size_seconds)
    if cache_index:
        tuning_weinan_review.load_aligned_minute_data_like_tuning = cache_load_aligned_minute_data
    try:
        with patched_argv(argv):
            exit_code = tuning_weinan_review.main()
    finally:
        tuning_weinan_review.MIN_SESSIONS_PER_UNIT = original_min_sessions
        tuning_weinan_review.BIN_SIZE_SECONDS = original_bin_size
        tuning_weinan_review.load_aligned_minute_data_like_tuning = original_load_aligned_minute_data

    output_files = sorted(
        str(path)
        for pattern in ("*.png", "*.npz", "*.csv", "*.json")
        for path in output_root.glob(pattern)
    )
    polar_files = sorted(str(path) for path in (output_root / "polar_time_of_day_units").rglob("*") if path.is_file()) if (output_root / "polar_time_of_day_units").is_dir() else []
    return {
        "exit_code": int(exit_code),
        "output_dir": str(require_existing_dir(output_root, "Tuning-Weinan auto output")),
        "output_files": output_files,
        "polar_output_files": polar_files,
        "verification": {
            "output_dir": str(output_root),
            "num_output_files": len(output_files),
            "num_polar_output_files": len(polar_files),
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


def options_for_lda_label(options: PipelineOptions, *, label_type: str, use_drug_labels: bool) -> PipelineOptions:
    current_lda_settings = require_lda_settings(options)
    extra_label_types = tuple(
        label_type_text
        for label_type_text in tuple(current_lda_settings.lda_extra_label_types or ())
        if str(label_type_text) != "injection_phase"
    )
    lda_settings = replace(
        current_lda_settings,
        lda_label_type=label_type,
        lda_extra_label_types=tuple() if use_drug_labels else extra_label_types,
        lda_use_baseline_sham_drug=bool(use_drug_labels),
        lda_injection_phase_schedule=current_lda_settings.lda_injection_phase_schedule,
    )
    return replace(options, lda=lda_settings)


def run_selected_lda_auto_exports(
    stats_export_summary_path: Path,
    *,
    options: PipelineOptions,
    timings: list[dict],
) -> dict | None:
    if options.skip_lda:
        show_progress("Skipping LDA by request.")
        return None

    lda_results: dict[str, dict] = {}
    if options.run_lda_drug_label:
        try:
            with timed_stage("LDA sham/drug marker setup", timings):
                collect_lda_baseline_sham_drug_schedule(
                    stats_export_summary_path,
                    options=options,
                )
            lda_results["sham_drug_marker_setup"] = {
                "status": "completed",
                "used_for": "hour_label_marker_shapes",
            }
        except Exception as exc:
            lda_results["sham_drug_marker_setup"] = failed_stage_payload(
                "LDA sham/drug marker setup",
                exc,
            )
            show_progress(f"LDA sham/drug marker setup failed; continuing with later stages: {exc}")
            if options.stop_on_error:
                raise

    if options.run_lda_hour_label:
        hour_options = options_for_lda_label(
            options,
            label_type="clock_hour_of_day",
            use_drug_labels=False,
        )
        try:
            with timed_stage("LDA hour-label auto stats", timings):
                lda_results["hour_label"] = run_lda_auto_export(
                    stats_export_summary_path,
                    options=hour_options,
                )
        except Exception as exc:
            lda_results["hour_label"] = failed_stage_payload("LDA hour-label auto stats", exc)
            show_progress(f"LDA hour-label failed; continuing with later stages: {exc}")
            if options.stop_on_error:
                raise

    if options.run_lda_drug_label:
        show_progress(
            "Sham/drug selection was used as marker shapes on hour-label LDA plots; "
            "no separate baseline/sham/drug LDA analysis was run."
        )

    if not lda_results:
        show_progress("Skipping LDA because no LDA label analysis was selected.")
        return None
    return lda_results


def run_presentation_stage(
    stats_export_summary_path: Path,
    *,
    options: PipelineOptions,
    timings: list[dict],
    allow_presentation: bool = True,
    skip_reason: str = "Skipping presentation_multiple stats because no cross-day export was produced.",
) -> dict | None:
    if options.skip_presentation:
        show_progress("Skipping presentation_multiple stats by request.")
        return None
    if not allow_presentation:
        show_progress(skip_reason)
        return None
    try:
        with timed_stage("presentation_multiple auto stats", timings):
            return run_presentation_auto_export(
                stats_export_summary_path,
                options=options,
            )
    except Exception as exc:
        presentation_result = failed_stage_payload("presentation_multiple auto stats", exc)
        show_progress(f"presentation_multiple failed; continuing with later stages: {exc}")
        if options.stop_on_error:
            raise
        return presentation_result


def run_auto_pipeline(options: PipelineOptions) -> dict:
    timings: list[dict] = []
    selected_roots = [path.resolve() for path in options.input_roots]
    show_progress("Checking whether input roots are Sorting_organize.py cache folders...")
    organized_input = looks_like_organized_input(selected_roots)
    if organized_input:
        patch_day_review_for_auto_roots()
        show_progress("Detected Sorting_organize.py cache input mode.")
    else:
        show_progress("Sorting_organize.py cache input mode was not detected.")
    day_roots = discover_day_roots_for_mode(
        selected_roots,
        organized_input=organized_input,
    )
    common_root = Path(os.path.commonpath([str(day_root) for day_root in day_roots]))
    day_summary_folder_name = auto_day_summary_folder_name(day_roots)

    configure_auto_output_names(day_summary_folder_name=day_summary_folder_name)

    show_progress("Pipeline inputs and configuration are locked in at startup.")
    show_progress(html_review.json.dumps(pipeline_options_payload(options), indent=2))
    if options.overwrite_auto_exports:
        show_progress(
            "Existing per-day and cross-day auto alignment exports will be deleted and recomputed."
        )
    else:
        show_progress(
            "Existing per-day auto alignment exports will be reused when complete; "
            "cross-day alignment is recomputed when the selected day range changes."
        )
    show_progress(f"Discovered {len(day_roots)} day folder(s)")
    for day_root in day_roots:
        show_progress(f"  day: {day_root}")

    day_results = []
    with timed_stage("within-day auto alignment exports", timings):
        for day_root in day_roots:
            day_results.append(
                run_single_day_auto_export(
                    day_root,
                    organized_input=organized_input,
                    overwrite=options.overwrite_auto_exports,
                )
            )

    selected_days_match_existing_cross_export = selected_day_folders_match(
        common_root / day_summary_folder_name / "selected_day_folders.txt",
        day_roots,
    )
    all_days_reused = all(result.get("status") == "reused" for result in day_results)
    allow_cross_day_reuse = bool(
        all_days_reused
        and selected_days_match_existing_cross_export
        and not options.overwrite_auto_exports
        and not organized_input
    )
    if (
        (options.overwrite_auto_exports or organized_input)
        and len(day_roots) > 1
        and not options.skip_cross_day
    ):
        remove_cross_day_auto_exports(common_root, day_summary_folder_name)
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

    lda_result = run_selected_lda_auto_exports(
        stats_export_summary_path,
        options=options,
        timings=timings,
    )

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

    tuning_weinan_result = None
    if not options.skip_tuning_weinan:
        try:
            with timed_stage("Tuning-Weinan auto stats", timings):
                tuning_weinan_result = run_tuning_weinan_auto_export(
                    stats_export_summary_path,
                    options=options,
                )
        except Exception as exc:
            tuning_weinan_result = failed_stage_payload("Tuning-Weinan auto stats", exc)
            show_progress(f"Tuning-Weinan failed; continuing with later stages: {exc}")
            if options.stop_on_error:
                raise
    else:
        show_progress("Skipping Tuning-Weinan by request.")

    presentation_result = run_presentation_stage(
        stats_export_summary_path,
        options=options,
        timings=timings,
        allow_presentation=cross_day_result is not None,
    )

    run_payload = {
        "input_roots": [str(path) for path in selected_roots],
        "input_mode": "sorting_organize_cache" if organized_input else "sorting_analyzer",
        "reuse_policy": {
            "reuse_existing_within_day_auto_exports": not bool(options.overwrite_auto_exports),
            "reuse_existing_cross_day_auto_export": bool(allow_cross_day_reuse),
            "cross_day_recomputed_when_day_selection_changes": True,
        },
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
        "tuning_weinan_result": tuning_weinan_result,
        "timings": timings,
    }
    write_json(common_root / day_summary_folder_name / "alignment_auto_run_summary.json", run_payload)
    print_runtime_summary(timings)
    return run_payload


def run_resume_analysis_pipeline(options: PipelineOptions) -> dict:
    timings: list[dict] = []
    if options.resume_export_summary_path is None:
        raise ValueError("Resume mode requires a previous auto cross-day export_summary.json.")

    stats_export_summary_path = resolve_export_summary_input(options.resume_export_summary_path)
    selected_roots = [path.resolve() for path in options.input_roots]
    show_progress("Checking whether resumed analysis has Sorting_organize.py cache inputs...")
    organized_input = looks_like_organized_input(selected_roots)
    if organized_input:
        patch_day_review_for_auto_roots()
        show_progress("Detected Sorting_organize.py cache input mode for resumed analysis.")
    else:
        show_progress("No Sorting_organize.py cache input was detected for resumed analysis.")

    show_progress("Resuming downstream analysis from previous alignment export.")
    show_progress(f"Previous alignment export: {stats_export_summary_path}")
    show_progress(html_review.json.dumps(pipeline_options_payload(options), indent=2))

    lda_result = run_selected_lda_auto_exports(
        stats_export_summary_path,
        options=options,
        timings=timings,
    )

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

    tuning_weinan_result = None
    if not options.skip_tuning_weinan:
        try:
            with timed_stage("Tuning-Weinan auto stats", timings):
                tuning_weinan_result = run_tuning_weinan_auto_export(
                    stats_export_summary_path,
                    options=options,
                )
        except Exception as exc:
            tuning_weinan_result = failed_stage_payload("Tuning-Weinan auto stats", exc)
            show_progress(f"Tuning-Weinan failed; continuing with later stages: {exc}")
            if options.stop_on_error:
                raise
    else:
        show_progress("Skipping Tuning-Weinan by request.")

    presentation_result = run_presentation_stage(
        stats_export_summary_path,
        options=options,
        timings=timings,
        allow_presentation=True,
    )

    run_payload = {
        "input_roots": [str(path) for path in selected_roots],
        "input_mode": "sorting_organize_cache" if organized_input else "sorting_analyzer",
        "resume_mode": True,
        "options": pipeline_options_payload(options),
        "stats_export_summary_path": str(stats_export_summary_path),
        "presentation_result": presentation_result,
        "lda_result": lda_result,
        "tuning_result": tuning_result,
        "tuning_weinan_result": tuning_weinan_result,
        "timings": timings,
    }
    summary_path = stats_export_summary_path.parent / "resume_analysis_run_summary.json"
    write_json(summary_path, run_payload)
    print_runtime_summary(timings)
    show_progress(f"Resume analysis summary: {summary_path}")
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
            "One or more daily *_Sorting folders, Sorting_organize.py *_Sorting_org folders, "
            "unit_feature_cache folders, or parent folders, separated by commas. If omitted, "
            "you will be prompted in the terminal."
        ),
    )
    parser.add_argument(
        "--resume-export-summary",
        help=(
            "Skip alignment and run downstream analysis from a previous auto cross-day "
            "export_summary.json, or from the folder containing it."
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
    parser.add_argument("--skip-tuning-weinan", action="store_true", help="Skip Tuning_Weinan.py outputs.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop when a downstream stats stage fails.")
    parser.add_argument(
        "--overwrite-auto-exports",
        action="store_true",
        help=(
            "Delete and recompute existing within-day and cross-day auto alignment exports "
            "instead of reusing complete manifests."
        ),
    )
    parser.add_argument(
        "--lda-output-base-dir",
        help="Base folder for LDA auto outputs. Defaults to LDA_auto beside the alignment export.",
    )
    parser.add_argument(
        "--tuning-output-base-dir",
        help="Base folder for Tuning auto outputs. Defaults to Tuning_auto beside the alignment export.",
    )
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
    parser.add_argument(
        "--lda-label-type",
        default=lda_review.LABEL_TYPE,
        help="Primary LDA label type. Resume prompts can run hour and drug labels separately.",
    )
    parser.add_argument("--lda-single-day-date", default=lda_review.SINGLE_DAY_DATE)
    parser.add_argument("--lda-min-firing-rate-hz", type=float, default=lda_review.MIN_FIRING_RATE_HZ)
    parser.add_argument(
        "--min-sessions-per-unit",
        type=int,
        default=None,
        help=(
            "Shared minimum sessions per aligned unit for both LDA and Tuning. "
            f"If omitted, prompts at startup. Default prompt value: {MIN_SESSIONS_PER_UNIT_AUTO}."
        ),
    )
    parser.add_argument("--lda-min-bins-per-label", type=int, default=lda_review.MIN_BINS_PER_LABEL)
    parser.add_argument("--lda-cv-n-splits", type=int, default=lda_review.CV_N_SPLITS)
    parser.add_argument(
        "--lda-n-permutations",
        type=int,
        default=LDA_N_PERMUTATIONS_AUTO,
        help=(
            "Number of label-shuffle permutation tests for LDA p-values. "
            f"Default for this auto pipeline: {LDA_N_PERMUTATIONS_AUTO}."
        ),
    )
    parser.add_argument(
        "--lda-feature-modes",
        help="Comma-separated LDA feature modes. Defaults to LDA.py FEATURE_MODES.",
    )
    parser.add_argument(
        "--lda-use-waveform-features",
        choices=["ask", "yes", "no"],
        default="ask",
        help=(
            "Whether to include cached waveform features in LDA feature modes. "
            "If no, FR_WAVEFORM, WAVEFORM_ONLY, and MULTI_FEATURE are removed."
        ),
    )
    parser.add_argument(
        "--lda-extra-label-types",
        default="",
        help=(
            "Comma-separated extra LDA label types. Sham/drug markers for hour-label "
            "plots are controlled by --lda-baseline-sham-drug."
        ),
    )
    parser.add_argument(
        "--lda-baseline-sham-drug",
        choices=["ask", "yes", "no"],
        default="ask",
        help="Whether to add sham/drug marker shapes to hour-label LDA figures.",
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
    parser.add_argument(
        "--tuning-weinan-max-aligned-units",
        type=int,
        default=tuning_weinan_review.MAX_ALIGNED_UNITS,
        help="Maximum aligned units to plot for Tuning_Weinan.py. Use 0 to plot all units.",
    )
    args = parser.parse_args()

    options = prompt_for_pipeline_options(args)
    show_progress("Startup prompts are complete; beginning pipeline execution.")

    try:
        if options.resume_export_summary_path is not None:
            result = run_resume_analysis_pipeline(options)
        else:
            result = run_auto_pipeline(options)
    except Exception as exc:
        print(f"[error] {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise

    show_progress("Automatic alignment pipeline complete.")
    if "common_root" in result:
        show_progress(f"Run summary: {Path(result['common_root']) / result['day_summary_folder_name'] / 'alignment_auto_run_summary.json'}")


if __name__ == "__main__":
    main()
