from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import spikeinterface.full as si


DEFAULT_ANALYZER_FOLDER_NAME = "sorting_analyzer_analysis.zarr"
DEFAULT_ALIGNMENT_EXPORT_FOLDER_NAME = "units_alignment_summary"
DEFAULT_ALIGNMENT_EXPORT_SUMMARY_NAME = "export_summary.json"
DEFAULT_MS_BEFORE = 1.0
DEFAULT_MS_AFTER = 2.0
ACTIVE_WINDOW_LOWER_PERCENTILE = 5.0
ACTIVE_WINDOW_UPPER_PERCENTILE = 95.0
DEFAULT_MIN_SESSIONS_PRESENT = 15
DEFAULT_MIN_SPIKES_AFTER_FILTER = 20
DEFAULT_WAVEFORM_ABS_THRESHOLD_UV = 2500.0

# True: make one compact figure where each aligned unit is shown once as a summary cell.
ALL_UNITS_ONLY = True
# True: make the session-by-session aligned-unit grid with blank cells for missing sessions.
ALL_UNITS_WITH_SESSIONS = True


def log_status(message: str) -> None:
    print(f"[all_units_grid] {message}", flush=True)


def normalize_session_name(session_name: str) -> str:
    return re.sub(r"_sh\d+$", "", str(session_name).strip())


@dataclass(frozen=True)
class AlignedUnitMember:
    session_name: str
    session_index: int
    unit_id: int
    original_session_name: str | None = None
    output_folder: str | None = None
    analyzer_folder: str | None = None
    align_group: str | None = None
    merge_group: str | None = None


@dataclass(frozen=True)
class AlignedUnitGroup:
    group_id: str
    members: tuple[AlignedUnitMember, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionUnitData:
    session_name: str
    session_index: int
    unit_id: int
    output_folder: Path
    analyzer_folder: Path
    waveform_time_axis_ms: np.ndarray
    spike_waveforms: np.ndarray
    spike_times_s: np.ndarray
    firing_rate_hz: float | None
    representative_channel_index: int


@dataclass
class ProcessedCell:
    session_name: str
    session_index: int
    unit_id: int
    n_spikes_kept: int
    average_waveform: np.ndarray
    waveform_time_axis_ms: np.ndarray
    isi_counts: np.ndarray
    isi_edges_ms: np.ndarray
    firing_rate_hz: float | None
    representative_channel_index: int


@dataclass
class ProcessedAlignedUnitRow:
    group_id: str
    members_present: int
    cells_by_session: dict[str, ProcessedCell]
    average_firing_rate_hz: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UniqueUnitSummary:
    group_id: str
    average_waveform: np.ndarray
    waveform_time_axis_ms: np.ndarray
    isi_counts: np.ndarray
    isi_edges_ms: np.ndarray
    average_firing_rate_hz: float | None
    members_present: int
    total_spikes_kept: int


@dataclass(frozen=True)
class AlignedUnitsGridConfig:
    min_sessions_present: int = DEFAULT_MIN_SESSIONS_PRESENT
    min_spikes_after_filter: int = DEFAULT_MIN_SPIKES_AFTER_FILTER
    waveform_abs_threshold_uv: float = DEFAULT_WAVEFORM_ABS_THRESHOLD_UV
    isi_range_ms: tuple[float, float] = (0.0, 50.0)
    isi_bin_ms: float = 1.0
    waveform_panel_width_ratio: float = 1.0
    isi_panel_width_ratio: float = 1.35
    figure_facecolor: str = "white"
    waveform_color: str = "black"
    isi_color: str = "black"
    row_height_in: float = 0.72
    session_width_in: float = 0.95
    max_figure_width_in: float | None = None
    dpi: int = 500
    png_facecolor: str = "white"
    waveform_linewidth: float = 0.55
    isi_linewidth: float = 0.0
    hide_axes: bool = True
    left_margin: float = 0.03
    right_margin: float = 0.005
    top_margin: float = 0.005
    bottom_margin: float = 0.005
    wspace: float = 0.01
    hspace: float = 0.12
    analyzer_folder_name: str = DEFAULT_ANALYZER_FOLDER_NAME
    ms_before_fallback: float = DEFAULT_MS_BEFORE
    ms_after_fallback: float = DEFAULT_MS_AFTER
    max_pixels_per_side: int = 60000
    max_rows_per_figure: int | None = 24
    show_session_headers: bool = True
    session_header_fontsize: int = 6
    session_header_height_in: float = 0.22
    left_annotation_width_in: float = 0.55
    average_fr_fontsize: int = 5
    average_fr_text_color: str = "black"
    row_gap_equivalent_uv: float = 2000.0
    amplitude_scalebar_uv: float | None = None
    scalebar_linewidth: float = 1.2
    scalebar_fontsize: int = 6
    scalebar_color: str = "black"


def make_member_lookup_key(member: AlignedUnitMember) -> tuple[str, int]:
    if member.output_folder is None:
        raise ValueError(
            f"Aligned member {member.session_name} unit {member.unit_id} has no output_folder."
        )
    return (str(Path(member.output_folder)), int(member.unit_id))


def session_name_sort_key(session_name: str) -> tuple[Any, ...]:
    text = str(session_name).strip()
    numbers = [int(item) for item in re.findall(r"\d+", text)]
    if len(numbers) >= 2:
        return tuple(numbers)
    if len(numbers) == 1:
        return (numbers[0],)
    return (10**9, text)


def load_alignment_groups_from_export_summary(
    export_summary_path: str | Path,
    *,
    analyzer_folder_name: str = DEFAULT_ANALYZER_FOLDER_NAME,
) -> tuple[list[str], list[AlignedUnitGroup]]:
    """
    Load Alignment_html.py export_summary.json and convert it to aligned-unit groups.

    The export summary stores each member's `output_folder`. This function resolves
    the paired analyzer as:
        output_folder / analyzer_folder_name
    """
    export_summary_path = Path(export_summary_path)
    payload = json.loads(export_summary_path.read_text(encoding="utf-8"))
    manifest_rows = payload.get("cross_session_alignment_groups", [])

    groups: list[AlignedUnitGroup] = []
    session_order_lookup: dict[str, int] = {}

    for row_index, row in enumerate(manifest_rows):
        members: list[AlignedUnitMember] = []
        for member in row.get("members", []):
            original_session_name = str(member["session_name"])
            session_name = normalize_session_name(original_session_name)
            session_index = int(member.get("session_index", 10**9))
            unit_id = int(member["unit_id"])
            output_folder = Path(str(member.get("output_folder", "") or ""))
            analyzer_folder = (
                output_folder / analyzer_folder_name
                if str(output_folder)
                else None
            )
            session_order_lookup[session_name] = min(
                session_order_lookup.get(session_name, session_index),
                session_index,
            )
            members.append(
                AlignedUnitMember(
                    session_name=session_name,
                    session_index=session_index,
                    unit_id=unit_id,
                    original_session_name=original_session_name,
                    output_folder=str(output_folder) if str(output_folder) else None,
                    analyzer_folder=str(analyzer_folder) if analyzer_folder is not None else None,
                    align_group=member.get("align_group"),
                    merge_group=member.get("merge_group"),
                )
            )

        groups.append(
            AlignedUnitGroup(
                group_id=str(row.get("final_group_key", f"group_{row_index:04d}")),
                members=tuple(members),
                metadata={
                    "final_unit_id": row.get("final_unit_id"),
                    "representative_session": row.get("representative_session"),
                    "representative_unit_id": row.get("representative_unit_id"),
                    "shank_id": row.get("shank_id"),
                    "local_channel_on_shank": row.get("local_channel_on_shank"),
                    "export_folder": row.get("export_folder"),
                },
            )
        )

    ordered_sessions = sorted(session_order_lookup.keys(), key=session_name_sort_key)
    return ordered_sessions, groups


def filter_aligned_units_by_session_presence(
    aligned_groups: list[AlignedUnitGroup],
    *,
    min_sessions_present: int = 20,
) -> list[AlignedUnitGroup]:
    filtered = [
        group
        for group in aligned_groups
        if len({member.session_name for member in group.members}) >= min_sessions_present
    ]
    filtered.sort(
        key=lambda group: (
            -len({member.session_name for member in group.members}),
            group.group_id,
        )
    )
    return filtered


def resolve_output_png_path(
    export_summary_path: str | Path,
    output_png_path: str | Path | None = None,
    *,
    stem: str = "aligned_units_grid",
) -> Path:
    export_summary_path = Path(export_summary_path)
    if output_png_path is not None:
        return Path(output_png_path)
    batch_root = export_summary_path.parent.parent
    stats_candidates = [batch_root / "stats", batch_root / "Stats"]
    stats_folder = next((path for path in stats_candidates if path.exists()), stats_candidates[0])
    stats_folder.mkdir(parents=True, exist_ok=True)
    return stats_folder / f"{stem}.png"


def resolve_export_summary_path_from_batch_root(
    batch_root: str | Path,
    *,
    export_folder_name: str = DEFAULT_ALIGNMENT_EXPORT_FOLDER_NAME,
    export_summary_name: str = DEFAULT_ALIGNMENT_EXPORT_SUMMARY_NAME,
) -> Path:
    batch_root = Path(batch_root)
    export_summary_path = batch_root / export_folder_name / export_summary_name
    log_status(f"Looking for alignment export summary in: {batch_root}")
    if export_summary_path.exists():
        log_status(f"Found alignment export summary: {export_summary_path}")
        return export_summary_path
    raise FileNotFoundError(
        f"Could not find alignment export summary at: {export_summary_path}"
    )


def _load_analyzer_cached(
    analyzer_folder: Path,
    analyzer_cache: dict[Path, Any],
):
    analyzer_folder = analyzer_folder.resolve()
    if analyzer_folder not in analyzer_cache:
        log_status(f"Loading analyzer: {analyzer_folder}")
        analyzer_cache[analyzer_folder] = si.load_sorting_analyzer(
            folder=analyzer_folder,
            format="zarr",
            load_extensions=True,
        )
    return analyzer_cache[analyzer_folder]


def _infer_waveform_time_axis_ms(
    analyzer,
    n_samples: int,
    *,
    ms_before_fallback: float,
    ms_after_fallback: float,
) -> np.ndarray:
    waveform_ext = analyzer.get_extension("waveforms")

    for attr_name in ("params", "_params"):
        params = getattr(waveform_ext, attr_name, None)
        if isinstance(params, dict):
            ms_before = params.get("ms_before")
            ms_after = params.get("ms_after")
            if ms_before is not None and ms_after is not None:
                return np.linspace(float(-ms_before), float(ms_after), int(n_samples))

    try:
        ext_info = waveform_ext.to_dict()
    except Exception:
        ext_info = {}
    if isinstance(ext_info, dict):
        params = ext_info.get("params", {})
        ms_before = params.get("ms_before")
        ms_after = params.get("ms_after")
        if ms_before is not None and ms_after is not None:
            return np.linspace(float(-ms_before), float(ms_after), int(n_samples))

    return np.linspace(
        float(-ms_before_fallback),
        float(ms_after_fallback),
        int(n_samples),
    )


def _extract_representative_channel_waveforms(
    analyzer,
    unit_id: int,
    *,
    ms_before_fallback: float,
    ms_after_fallback: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    waveform_ext = analyzer.get_extension("waveforms")
    waveforms = waveform_ext.get_waveforms_one_unit(unit_id=unit_id)
    if waveforms is None or getattr(waveforms, "size", 0) == 0:
        raise ValueError(f"No waveforms available for unit {unit_id}.")

    waveforms = np.asarray(waveforms)
    if waveforms.ndim == 2:
        n_samples = int(waveforms.shape[1])
        time_axis_ms = _infer_waveform_time_axis_ms(
            analyzer,
            n_samples,
            ms_before_fallback=ms_before_fallback,
            ms_after_fallback=ms_after_fallback,
        )
        return waveforms.astype(float), time_axis_ms, 0

    if waveforms.ndim != 3:
        raise ValueError(
            f"Expected waveforms with 2 or 3 dimensions; got shape {waveforms.shape}."
        )

    mean_waveform = waveforms.mean(axis=0)
    representative_channel_index = int(np.argmax(np.max(np.abs(mean_waveform), axis=0)))
    channel_waveforms = waveforms[:, :, representative_channel_index].astype(float)
    time_axis_ms = _infer_waveform_time_axis_ms(
        analyzer,
        int(channel_waveforms.shape[1]),
        ms_before_fallback=ms_before_fallback,
        ms_after_fallback=ms_after_fallback,
    )
    return channel_waveforms, time_axis_ms, representative_channel_index


def _estimate_active_window_from_spike_times(
    spike_times_s: np.ndarray,
    *,
    lower_percentile: float = ACTIVE_WINDOW_LOWER_PERCENTILE,
    upper_percentile: float = ACTIVE_WINDOW_UPPER_PERCENTILE,
) -> tuple[float | None, float | None, float | None]:
    spike_times_s = np.asarray(spike_times_s, dtype=float).ravel()
    spike_times_s = spike_times_s[np.isfinite(spike_times_s)]
    if spike_times_s.size == 0:
        return None, None, None

    spike_times_s = np.sort(spike_times_s)
    if spike_times_s.size >= 10:
        start_s = float(np.percentile(spike_times_s, lower_percentile))
        end_s = float(np.percentile(spike_times_s, upper_percentile))
    elif spike_times_s.size >= 2:
        start_s = float(spike_times_s[0])
        end_s = float(spike_times_s[-1])
    else:
        start_s = float(spike_times_s[0])
        end_s = float(spike_times_s[0])

    duration_s = max(0.0, end_s - start_s)
    if duration_s <= 0:
        duration_s = None
    return start_s, end_s, duration_s


def _get_unit_firing_rate_hz(_analyzer, _unit_id: int, spike_train_samples: np.ndarray) -> float | None:
    spike_train_samples = np.asarray(spike_train_samples, dtype=float).ravel()
    if spike_train_samples.size == 0:
        return None

    sampling_frequency = float(_analyzer.sorting.get_sampling_frequency())
    spike_times_s = spike_train_samples / sampling_frequency
    _, _, active_duration_s = _estimate_active_window_from_spike_times(spike_times_s)
    if active_duration_s is None or active_duration_s <= 0:
        return None
    return float(spike_train_samples.size / active_duration_s)


def load_session_unit_data_from_member(
    member: AlignedUnitMember,
    *,
    analyzer_cache: dict[Path, Any] | None = None,
    config: AlignedUnitsGridConfig | None = None,
) -> SessionUnitData:
    config = config or AlignedUnitsGridConfig()
    analyzer_cache = analyzer_cache if analyzer_cache is not None else {}

    if member.output_folder is None:
        raise ValueError(
            f"Aligned member {member.session_name} unit {member.unit_id} has no output_folder."
        )

    output_folder = Path(member.output_folder)
    analyzer_folder = (
        Path(member.analyzer_folder)
        if member.analyzer_folder is not None
        else output_folder / config.analyzer_folder_name
    )
    if not analyzer_folder.exists():
        raise FileNotFoundError(
            f"Analyzer folder not found for session {member.session_name}: {analyzer_folder}"
        )

    analyzer = _load_analyzer_cached(analyzer_folder, analyzer_cache)
    spike_waveforms, waveform_time_axis_ms, representative_channel_index = (
        _extract_representative_channel_waveforms(
            analyzer,
            int(member.unit_id),
            ms_before_fallback=config.ms_before_fallback,
            ms_after_fallback=config.ms_after_fallback,
        )
    )
    spike_train_samples = analyzer.sorting.get_unit_spike_train(
        unit_id=int(member.unit_id),
        segment_index=0,
    )
    sampling_frequency = float(analyzer.sorting.get_sampling_frequency())
    spike_times_s = np.asarray(spike_train_samples, dtype=float) / sampling_frequency
    firing_rate_hz = _get_unit_firing_rate_hz(analyzer, int(member.unit_id), spike_train_samples)

    return SessionUnitData(
        session_name=member.session_name,
        session_index=member.session_index,
        unit_id=int(member.unit_id),
        output_folder=output_folder,
        analyzer_folder=analyzer_folder,
        waveform_time_axis_ms=waveform_time_axis_ms,
        spike_waveforms=spike_waveforms,
        spike_times_s=spike_times_s,
        firing_rate_hz=firing_rate_hz,
        representative_channel_index=representative_channel_index,
    )


def load_session_unit_data_from_alignment_groups(
    aligned_groups: list[AlignedUnitGroup],
    *,
    config: AlignedUnitsGridConfig | None = None,
) -> dict[tuple[str, int], SessionUnitData]:
    config = config or AlignedUnitsGridConfig()
    analyzer_cache: dict[Path, Any] = {}
    loaded: dict[tuple[str, int], SessionUnitData] = {}
    total_members = sum(len(group.members) for group in aligned_groups)
    processed_members = 0

    for group in aligned_groups:
        for member in group.members:
            processed_members += 1
            key = make_member_lookup_key(member)
            if key in loaded:
                if processed_members == 1 or processed_members % 25 == 0 or processed_members == total_members:
                    log_status(
                        f"Scanning aligned members: {processed_members}/{total_members} "
                        f"(unique loaded: {len(loaded)})"
                    )
                continue
            try:
                loaded[key] = load_session_unit_data_from_member(
                    member,
                    analyzer_cache=analyzer_cache,
                    config=config,
                )
            except Exception as exc:
                print(
                    f"[all_units_grid] Skipping {member.session_name} unit {member.unit_id}: {exc}"
                )
                continue
            if processed_members == 1 or processed_members % 25 == 0 or processed_members == total_members:
                log_status(
                    f"Scanning aligned members: {processed_members}/{total_members} "
                    f"(unique loaded: {len(loaded)})"
                )
    log_status(f"Finished loading session-unit data for {len(loaded)} unique unit/session pairs.")
    return loaded


def filter_noisy_waveforms(
    spike_waveforms: np.ndarray,
    *,
    abs_threshold_uv: float = 1500.0,
) -> np.ndarray:
    waveforms = np.asarray(spike_waveforms, dtype=float)
    if waveforms.ndim != 2:
        raise ValueError(
            f"Expected waveforms with shape (n_spikes, n_samples); got {waveforms.shape}."
        )
    keep_mask = np.all(np.abs(waveforms) <= float(abs_threshold_uv), axis=1)
    return waveforms[keep_mask]


def compute_average_waveform(spike_waveforms: np.ndarray) -> np.ndarray:
    waveforms = np.asarray(spike_waveforms, dtype=float)
    if waveforms.ndim != 2 or waveforms.shape[0] == 0:
        raise ValueError("Average waveform requires an array with shape (n_spikes, n_samples).")
    return waveforms.mean(axis=0)


def compute_isi_histogram(
    spike_times_s: np.ndarray,
    *,
    isi_range_ms: tuple[float, float] = (0.0, 50.0),
    isi_bin_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    spike_times_s = np.asarray(spike_times_s, dtype=float).ravel()
    edges_ms = np.arange(isi_range_ms[0], isi_range_ms[1] + isi_bin_ms, isi_bin_ms)
    if spike_times_s.size < 2:
        return np.zeros(edges_ms.size - 1, dtype=int), edges_ms

    isi_ms = np.diff(np.sort(spike_times_s)) * 1000.0
    counts, edges_ms = np.histogram(isi_ms, bins=edges_ms)
    return counts.astype(int), edges_ms.astype(float)


def prepare_aligned_units_grid_data(
    ordered_sessions: list[str],
    aligned_groups: list[AlignedUnitGroup],
    session_unit_data_lookup: dict[tuple[str, int], SessionUnitData],
    *,
    config: AlignedUnitsGridConfig | None = None,
) -> list[ProcessedAlignedUnitRow]:
    config = config or AlignedUnitsGridConfig()
    session_rank = {session_name: rank for rank, session_name in enumerate(ordered_sessions)}
    rows: list[ProcessedAlignedUnitRow] = []
    candidate_groups = filter_aligned_units_by_session_presence(
        aligned_groups,
        min_sessions_present=config.min_sessions_present,
    )
    log_status(
        f"Preparing processed rows from {len(candidate_groups)} aligned groups "
        f"that meet the minimum session threshold."
    )

    for group_index, group in enumerate(candidate_groups, start=1):
        cells_by_session: dict[str, ProcessedCell] = {}
        unique_sessions = {member.session_name for member in group.members}
        duplicate_session_count = len(group.members) - len(unique_sessions)
        if duplicate_session_count > 0:
            log_status(
                f"Group {group.group_id} has {duplicate_session_count} duplicate member(s) after "
                "session-name normalization; using the first valid unit per session column."
            )

        for member in sorted(
            group.members,
            key=lambda item: (
                session_rank.get(item.session_name, 10**9),
                item.session_index,
                item.unit_id,
            ),
        ):
            key = make_member_lookup_key(member)
            session_unit_data = session_unit_data_lookup.get(key)
            if session_unit_data is None:
                continue
            if member.session_name in cells_by_session:
                continue

            filtered_waveforms = filter_noisy_waveforms(
                session_unit_data.spike_waveforms,
                abs_threshold_uv=config.waveform_abs_threshold_uv,
            )
            if filtered_waveforms.shape[0] < config.min_spikes_after_filter:
                continue

            average_waveform = compute_average_waveform(filtered_waveforms)
            isi_counts, isi_edges_ms = compute_isi_histogram(
                session_unit_data.spike_times_s,
                isi_range_ms=config.isi_range_ms,
                isi_bin_ms=config.isi_bin_ms,
            )
            cells_by_session[member.session_name] = ProcessedCell(
                session_name=member.session_name,
                session_index=member.session_index,
                unit_id=int(member.unit_id),
                n_spikes_kept=int(filtered_waveforms.shape[0]),
                average_waveform=average_waveform,
                waveform_time_axis_ms=session_unit_data.waveform_time_axis_ms,
                isi_counts=isi_counts,
                isi_edges_ms=isi_edges_ms,
                firing_rate_hz=session_unit_data.firing_rate_hz,
                representative_channel_index=session_unit_data.representative_channel_index,
            )

        if len(unique_sessions) < config.min_sessions_present:
            continue
        if not cells_by_session:
            continue

        firing_rates = [
            float(cell.firing_rate_hz)
            for cell in cells_by_session.values()
            if cell.firing_rate_hz is not None and np.isfinite(cell.firing_rate_hz)
        ]
        average_firing_rate_hz = float(np.mean(firing_rates)) if firing_rates else None

        rows.append(
            ProcessedAlignedUnitRow(
                group_id=group.group_id,
                members_present=len(unique_sessions),
                cells_by_session=cells_by_session,
                average_firing_rate_hz=average_firing_rate_hz,
                metadata=dict(group.metadata),
            )
        )
        if group_index == 1 or group_index % 25 == 0 or group_index == len(candidate_groups):
            log_status(
                f"Prepared row {group_index}/{len(candidate_groups)} "
                f"(rows kept so far: {len(rows)})"
            )

    rows.sort(key=lambda row: (-row_peak_amplitude_uv(row), -row.members_present, row.group_id))
    log_status(f"Finished preparing {len(rows)} plotted aligned-unit rows.")
    return rows


def determine_shared_waveform_limits(
    processed_rows: list[ProcessedAlignedUnitRow],
) -> tuple[np.ndarray, tuple[float, float]]:
    cells = [
        cell
        for row in processed_rows
        for cell in row.cells_by_session.values()
    ]
    if not cells:
        raise ValueError("No processed waveform data available to determine shared limits.")

    waveform_lengths = {cell.average_waveform.shape[0] for cell in cells}
    if len(waveform_lengths) != 1:
        raise ValueError(
            f"Waveform sample counts differ across sessions: {sorted(waveform_lengths)}"
        )

    reference_time_axis = cells[0].waveform_time_axis_ms
    for cell in cells[1:]:
        if cell.waveform_time_axis_ms.shape != reference_time_axis.shape:
            raise ValueError("Waveform time-axis shapes differ across sessions.")

    max_abs = float(max(np.max(np.abs(cell.average_waveform)) for cell in cells))
    if max_abs <= 0:
        max_abs = 1.0
    return reference_time_axis, (-1.05 * max_abs, 1.05 * max_abs)


def determine_rowwise_isi_y_limits(
    processed_rows: list[ProcessedAlignedUnitRow],
) -> dict[str, tuple[float, float]]:
    row_limits: dict[str, tuple[float, float]] = {}
    for row in processed_rows:
        max_count = max(
            (int(np.max(cell.isi_counts)) for cell in row.cells_by_session.values()),
            default=0,
        )
        row_limits[row.group_id] = (0.0, float(max(1, int(np.ceil(max_count * 1.05)))))
    return row_limits


def row_peak_amplitude_uv(row: ProcessedAlignedUnitRow) -> float:
    return max(
        (float(np.max(np.abs(cell.average_waveform))) for cell in row.cells_by_session.values()),
        default=0.0,
    )


def build_unique_unit_summaries(
    processed_rows: list[ProcessedAlignedUnitRow],
) -> list[UniqueUnitSummary]:
    summaries: list[UniqueUnitSummary] = []
    for row in processed_rows:
        cells = list(row.cells_by_session.values())
        if not cells:
            continue

        total_spikes_kept = int(sum(cell.n_spikes_kept for cell in cells))
        if total_spikes_kept <= 0:
            continue

        waveform_time_axis_ms = cells[0].waveform_time_axis_ms
        weighted_waveform = sum(
            cell.average_waveform * float(cell.n_spikes_kept)
            for cell in cells
        ) / float(total_spikes_kept)
        combined_isi_counts = np.sum(
            [cell.isi_counts.astype(float) for cell in cells],
            axis=0,
        )
        summaries.append(
            UniqueUnitSummary(
                group_id=row.group_id,
                average_waveform=np.asarray(weighted_waveform, dtype=float),
                waveform_time_axis_ms=waveform_time_axis_ms,
                isi_counts=np.asarray(combined_isi_counts, dtype=float),
                isi_edges_ms=cells[0].isi_edges_ms,
                average_firing_rate_hz=row.average_firing_rate_hz,
                members_present=row.members_present,
                total_spikes_kept=total_spikes_kept,
            )
        )
    return summaries


def determine_shared_waveform_limits_for_unique_units(
    unique_units: list[UniqueUnitSummary],
) -> tuple[np.ndarray, tuple[float, float]]:
    if not unique_units:
        raise ValueError("No unique-unit summaries available.")
    reference_time_axis = unique_units[0].waveform_time_axis_ms
    max_abs = float(max(np.max(np.abs(unit.average_waveform)) for unit in unique_units))
    if max_abs <= 0:
        max_abs = 1.0
    return reference_time_axis, (-1.05 * max_abs, 1.05 * max_abs)


def _compute_safe_save_dpi(
    fig_width_in: float,
    fig_height_in: float,
    requested_dpi: int,
    *,
    max_pixels_per_side: int,
) -> int:
    width_px = fig_width_in * requested_dpi
    height_px = fig_height_in * requested_dpi
    longest_side_px = max(width_px, height_px)
    if longest_side_px <= max_pixels_per_side:
        return int(requested_dpi)

    scale = float(max_pixels_per_side) / float(longest_side_px)
    safe_dpi = max(72, int(np.floor(requested_dpi * scale)))
    return safe_dpi


def _format_voltage_label(uv_value: float) -> str:
    if abs(float(uv_value)) >= 1000.0:
        return f"{float(uv_value) / 1000.0:g} mV"
    return f"{float(uv_value):g} uV"


def _nice_scalebar_uv(y_span_uv: float) -> float:
    target = max(float(y_span_uv) * 0.25, 1.0)
    nice_values = np.array(
        [
            10.0,
            20.0,
            50.0,
            100.0,
            200.0,
            500.0,
            1000.0,
            2000.0,
            5000.0,
            10000.0,
        ],
        dtype=float,
    )
    valid = nice_values[nice_values <= target]
    if valid.size == 0:
        return float(nice_values[0])
    return float(valid[-1])


def _draw_figure_amplitude_scalebar(
    fig: plt.Figure,
    *,
    waveform_y_limits: tuple[float, float],
    config: AlignedUnitsGridConfig,
    x0: float = 0.012,
    waveform_axes_height_frac: float | None = None,
) -> None:
    y_span_uv = float(waveform_y_limits[1] - waveform_y_limits[0])
    if y_span_uv <= 0:
        return

    if waveform_axes_height_frac is None or waveform_axes_height_frac <= 0:
        waveform_axes_height_frac = 0.10

    fig_width_in, fig_height_in = fig.get_size_inches()
    bar_height_frac = waveform_axes_height_frac
    if bar_height_frac <= 0:
        return

    requested_bar_uv = (
        _nice_scalebar_uv(y_span_uv)
        if config.amplitude_scalebar_uv is None
        else float(config.amplitude_scalebar_uv)
    )
    bar_height_frac = bar_height_frac * (requested_bar_uv / y_span_uv)
    if bar_height_frac <= 0:
        return

    ax_bar = fig.add_axes([0, 0, 1, 1], zorder=10)
    ax_bar.set_axis_off()

    y0 = 0.03
    y1 = y0 + bar_height_frac
    ax_bar.plot(
        [x0, x0],
        [y0, y1],
        transform=ax_bar.transAxes,
        color=config.scalebar_color,
        linewidth=config.scalebar_linewidth,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax_bar.text(
        x0 + 0.006,
        (y0 + y1) / 2.0,
        _format_voltage_label(requested_bar_uv),
        transform=ax_bar.transAxes,
        ha="left",
        va="center",
        fontsize=config.scalebar_fontsize,
        color=config.scalebar_color,
        clip_on=False,
    )


def _split_rows_into_pages(
    processed_rows: list[ProcessedAlignedUnitRow],
    ordered_sessions: list[str],
    *,
    config: AlignedUnitsGridConfig,
) -> list[list[ProcessedAlignedUnitRow]]:
    if not processed_rows:
        return []

    max_rows_by_pixels = max(
        1,
        int(
            np.floor(
                config.max_pixels_per_side / max(config.row_height_in * config.dpi, 1e-9)
            )
        ),
    )
    max_rows = max_rows_by_pixels
    if config.max_rows_per_figure is not None:
        max_rows = max(1, min(max_rows, int(config.max_rows_per_figure)))

    pages: list[list[ProcessedAlignedUnitRow]] = []
    for start in range(0, len(processed_rows), max_rows):
        pages.append(processed_rows[start : start + max_rows])
    return pages


def _build_page_output_path(output_png_path: str | Path, page_index: int, num_pages: int) -> Path:
    output_png_path = Path(output_png_path)
    if num_pages <= 1:
        return output_png_path
    return output_png_path.with_name(
        f"{output_png_path.stem}_page{page_index + 1:02d}{output_png_path.suffix}"
    )


def plot_all_units_only_grid(
    unique_units: list[UniqueUnitSummary],
    *,
    output_png_path: str | Path,
    config: AlignedUnitsGridConfig | None = None,
    units_per_row: int = 12,
) -> list[Path]:
    config = config or AlignedUnitsGridConfig()
    if not unique_units:
        raise ValueError("No unique-unit summaries available to plot.")

    waveform_x_axis_ms, waveform_y_limits = determine_shared_waveform_limits_for_unique_units(
        unique_units
    )
    max_isi_count = max((float(np.max(unit.isi_counts)) for unit in unique_units), default=1.0)
    isi_ylim = (0.0, max(1.0, 1.05 * max_isi_count))

    units_per_row = max(1, int(units_per_row))
    grid_rows = int(np.ceil(len(unique_units) / float(units_per_row)))
    rows_per_page = config.max_rows_per_figure or 24
    num_pages = int(np.ceil(grid_rows / float(rows_per_page)))
    saved_paths: list[Path] = []

    for page_index in range(num_pages):
        start_row = page_index * rows_per_page
        end_row = min(grid_rows, (page_index + 1) * rows_per_page)
        page_row_count = end_row - start_row
        page_units = unique_units[start_row * units_per_row : end_row * units_per_row]
        log_status(
            f"Rendering all-units-only page {page_index + 1}/{num_pages} "
            f"with {len(page_units)} unique units."
        )

        fig_width = units_per_row * config.session_width_in
        fig_height = max(1.0, page_row_count * config.row_height_in)
        fig = plt.figure(
            figsize=(fig_width, fig_height),
            dpi=config.dpi,
            facecolor=config.figure_facecolor,
        )
        width_ratios: list[float] = []
        for _ in range(units_per_row):
            width_ratios.extend([config.waveform_panel_width_ratio, config.isi_panel_width_ratio])
        outer_grid = gridspec.GridSpec(
            page_row_count,
            units_per_row * 2,
            figure=fig,
            width_ratios=width_ratios,
            left=config.left_margin,
            right=1.0 - config.right_margin,
            top=1.0 - config.top_margin,
            bottom=config.bottom_margin,
            wspace=config.wspace,
            hspace=config.hspace,
        )

        first_waveform_ax = None
        for unit_offset, unit in enumerate(page_units):
            row_index = unit_offset // units_per_row
            col_index = unit_offset % units_per_row
            ax_wf = fig.add_subplot(outer_grid[row_index, col_index * 2])
            ax_isi = fig.add_subplot(outer_grid[row_index, col_index * 2 + 1])
            if first_waveform_ax is None:
                first_waveform_ax = ax_wf
            for ax in (ax_wf, ax_isi):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            ax_wf.plot(
                waveform_x_axis_ms,
                unit.average_waveform,
                color=config.waveform_color,
                linewidth=config.waveform_linewidth,
            )
            ax_wf.set_xlim(float(waveform_x_axis_ms[0]), float(waveform_x_axis_ms[-1]))
            ax_wf.set_ylim(*waveform_y_limits)
            if unit.average_firing_rate_hz is not None:
                ax_wf.text(
                    0.02,
                    0.98,
                    f"{unit.average_firing_rate_hz:.2f} Hz",
                    transform=ax_wf.transAxes,
                    ha="left",
                    va="top",
                    fontsize=config.average_fr_fontsize,
                    color=config.average_fr_text_color,
                    clip_on=False,
                )

            ax_isi.bar(
                unit.isi_edges_ms[:-1],
                unit.isi_counts,
                width=float(np.diff(unit.isi_edges_ms)[0]),
                align="edge",
                color=config.isi_color,
                linewidth=config.isi_linewidth,
            )
            ax_isi.set_xlim(*config.isi_range_ms)
            ax_isi.set_ylim(*isi_ylim)

        _draw_figure_amplitude_scalebar(
            fig,
            waveform_y_limits=waveform_y_limits,
            config=config,
            waveform_axes_height_frac=(
                first_waveform_ax.get_position().height if first_waveform_ax is not None else None
            ),
        )

        page_output_path = _build_page_output_path(output_png_path, page_index, num_pages)
        save_dpi = _compute_safe_save_dpi(
            fig_width,
            fig_height,
            config.dpi,
            max_pixels_per_side=config.max_pixels_per_side,
        )
        log_status(f"Saving all-units-only figure to: {page_output_path}")
        fig.savefig(
            page_output_path,
            dpi=save_dpi,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor=config.png_facecolor,
        )
        plt.close(fig)
        saved_paths.append(Path(page_output_path))

    return saved_paths


def plot_aligned_units_summary_grid(
    processed_rows: list[ProcessedAlignedUnitRow],
    ordered_sessions: list[str],
    *,
    config: AlignedUnitsGridConfig | None = None,
    output_png_path: str | Path | None = None,
    per_page_waveform_scale: bool = True,
) -> tuple[plt.Figure, Path | None]:
    config = config or AlignedUnitsGridConfig()
    if not processed_rows:
        raise ValueError("No aligned units passed filtering; nothing to plot.")
    if not ordered_sessions:
        raise ValueError("ordered_sessions is empty.")

    log_status(
        f"Plotting grid with {len(processed_rows)} rows across {len(ordered_sessions)} sessions."
    )
    waveform_x_axis_ms, waveform_y_limits = determine_shared_waveform_limits(processed_rows)
    rowwise_isi_limits = determine_rowwise_isi_y_limits(processed_rows)
    n_rows = len(processed_rows)
    n_sessions = len(ordered_sessions)
    header_rows = 1 if config.show_session_headers else 0
    width_ratios: list[float] = []
    for _session_name in ordered_sessions:
        width_ratios.extend(
            [config.waveform_panel_width_ratio, config.isi_panel_width_ratio]
        )

    fig_width = (n_sessions * config.session_width_in) + config.left_annotation_width_in
    if config.max_figure_width_in is not None:
        fig_width = min(fig_width, config.max_figure_width_in)
    fig_height = max(
        1.0,
        (n_rows * config.row_height_in) + (header_rows * config.session_header_height_in),
    )

    fig = plt.figure(
        figsize=(fig_width, fig_height),
        dpi=config.dpi,
        facecolor=config.figure_facecolor,
    )
    height_ratios = []
    if config.show_session_headers:
        height_ratios.append(config.session_header_height_in)
    height_ratios.extend([config.row_height_in] * n_rows)
    outer_grid = gridspec.GridSpec(
        n_rows + header_rows,
        (n_sessions * 2) + 1,
        figure=fig,
        width_ratios=[config.left_annotation_width_in / max(config.session_width_in, 1e-9)] + width_ratios,
        height_ratios=height_ratios,
        left=config.left_margin,
        right=1.0 - config.right_margin,
        top=1.0 - config.top_margin,
        bottom=config.bottom_margin,
        wspace=config.wspace,
        hspace=config.hspace,
    )

    if config.show_session_headers:
        ax_header_left = fig.add_subplot(outer_grid[0, 0])
        ax_header_left.axis("off")
        for session_index, session_name in enumerate(ordered_sessions):
            base_col = 1 + session_index * 2
            ax_header = fig.add_subplot(outer_grid[0, base_col : base_col + 2])
            ax_header.axis("off")
            ax_header.text(
                0.5,
                0.5,
                session_name,
                ha="center",
                va="center",
                fontsize=config.session_header_fontsize,
                color="black",
                transform=ax_header.transAxes,
            )

    first_waveform_ax = None
    for row_index, row in enumerate(processed_rows):
        isi_ylim = rowwise_isi_limits[row.group_id]
        grid_row_index = row_index + header_rows
        ax_row_label = fig.add_subplot(outer_grid[grid_row_index, 0])
        ax_row_label.axis("off")
        if row.average_firing_rate_hz is not None:
            ax_row_label.text(
                0.98,
                0.98,
                f"{row.average_firing_rate_hz:.2f} Hz",
                transform=ax_row_label.transAxes,
                ha="right",
                va="top",
                fontsize=config.average_fr_fontsize,
                color=config.average_fr_text_color,
                clip_on=False,
            )
        for session_index, session_name in enumerate(ordered_sessions):
            cell = row.cells_by_session.get(session_name)
            base_col = 1 + session_index * 2
            ax_wf = fig.add_subplot(outer_grid[grid_row_index, base_col])
            ax_isi = fig.add_subplot(outer_grid[grid_row_index, base_col + 1])
            if first_waveform_ax is None:
                first_waveform_ax = ax_wf

            if config.hide_axes:
                for ax in (ax_wf, ax_isi):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)

            if cell is None:
                ax_wf.axis("off")
                ax_isi.axis("off")
                continue

            ax_wf.plot(
                waveform_x_axis_ms,
                cell.average_waveform,
                color=config.waveform_color,
                linewidth=config.waveform_linewidth,
            )
            ax_wf.set_xlim(float(waveform_x_axis_ms[0]), float(waveform_x_axis_ms[-1]))
            ax_wf.set_ylim(*waveform_y_limits)

            ax_isi.bar(
                cell.isi_edges_ms[:-1],
                cell.isi_counts,
                width=float(np.diff(cell.isi_edges_ms)[0]),
                align="edge",
                color=config.isi_color,
                linewidth=config.isi_linewidth,
            )
            ax_isi.set_xlim(*config.isi_range_ms)
            ax_isi.set_ylim(*isi_ylim)

    _draw_figure_amplitude_scalebar(
        fig,
        waveform_y_limits=waveform_y_limits,
        config=config,
        x0=max(0.012, config.left_margin * 0.5),
        waveform_axes_height_frac=(
            first_waveform_ax.get_position().height if first_waveform_ax is not None else None
        ),
    )

    saved_path: Path | None = None
    if output_png_path is not None:
        saved_path = Path(output_png_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        save_dpi = _compute_safe_save_dpi(
            fig_width,
            fig_height,
            config.dpi,
            max_pixels_per_side=config.max_pixels_per_side,
        )
        if save_dpi != config.dpi:
            log_status(
                "Figure is very large; reducing save dpi from "
                f"{config.dpi} to {save_dpi} to stay within PNG backend limits."
            )
        log_status(f"Saving figure to: {saved_path}")
        fig.savefig(
            saved_path,
            dpi=save_dpi,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor=config.png_facecolor,
        )
        log_status("Figure saved successfully.")
    return fig, saved_path


def save_aligned_units_summary_grid_pages(
    processed_rows: list[ProcessedAlignedUnitRow],
    ordered_sessions: list[str],
    *,
    output_png_path: str | Path,
    config: AlignedUnitsGridConfig | None = None,
) -> list[Path]:
    config = config or AlignedUnitsGridConfig()
    page_config = AlignedUnitsGridConfig(**{**config.__dict__, "row_height_in": max(config.row_height_in, 0.95)})
    pages = _split_rows_into_pages(
        processed_rows,
        ordered_sessions,
        config=page_config,
    )
    if not pages:
        raise ValueError("No processed rows available to save.")

    output_paths: list[Path] = []
    if len(pages) > 1:
        log_status(
            f"Figure is split across {len(pages)} pages to stay within backend size limits."
        )

    for page_index, page_rows in enumerate(pages):
        page_output_path = _build_page_output_path(output_png_path, page_index, len(pages))
        log_status(
            f"Rendering page {page_index + 1}/{len(pages)} "
            f"with {len(page_rows)} aligned-unit rows."
        )
        fig, saved_path = plot_aligned_units_summary_grid(
            processed_rows=page_rows,
            ordered_sessions=ordered_sessions,
            config=page_config,
            output_png_path=page_output_path,
        )
        plt.close(fig)
        if saved_path is None:
            raise RuntimeError(f"Expected saved path for page {page_index + 1}.")
        output_paths.append(saved_path)

    return output_paths


def save_aligned_units_summary_grid_single_figure(
    processed_rows: list[ProcessedAlignedUnitRow],
    ordered_sessions: list[str],
    *,
    output_png_path: str | Path,
    config: AlignedUnitsGridConfig | None = None,
) -> Path:
    config = config or AlignedUnitsGridConfig()
    fig, saved_path = plot_aligned_units_summary_grid(
        processed_rows=processed_rows,
        ordered_sessions=ordered_sessions,
        config=config,
        output_png_path=output_png_path,
    )
    plt.close(fig)
    if saved_path is None:
        raise RuntimeError("Expected saved path for combined single-figure output.")
    return saved_path


def save_aligned_units_summary_grid_from_export_summary(
    export_summary_path: str | Path,
    *,
    output_png_path: str | Path | None = None,
    config: AlignedUnitsGridConfig | None = None,
) -> dict[str, list[Path]]:
    config = config or AlignedUnitsGridConfig()
    log_status(f"Starting aligned-units grid build from export summary: {export_summary_path}")
    ordered_sessions, aligned_groups = load_alignment_groups_from_export_summary(
        export_summary_path,
        analyzer_folder_name=config.analyzer_folder_name,
    )
    log_status(
        f"Loaded alignment summary with {len(ordered_sessions)} sessions and "
        f"{len(aligned_groups)} aligned groups."
    )
    session_unit_data_lookup = load_session_unit_data_from_alignment_groups(
        aligned_groups,
        config=config,
    )
    processed_rows = prepare_aligned_units_grid_data(
        ordered_sessions=ordered_sessions,
        aligned_groups=aligned_groups,
        session_unit_data_lookup=session_unit_data_lookup,
        config=config,
    )
    outputs: dict[str, list[Path]] = {}

    if ALL_UNITS_WITH_SESSIONS:
        resolved_output_png_path_split = resolve_output_png_path(
            export_summary_path,
            output_png_path=output_png_path,
            stem="aligned_units_grid_split",
        )
        saved_paths = save_aligned_units_summary_grid_pages(
            processed_rows=processed_rows,
            ordered_sessions=ordered_sessions,
            output_png_path=resolved_output_png_path_split,
            config=config,
        )
        resolved_output_png_path_full = resolve_output_png_path(
            export_summary_path,
            output_png_path=None,
            stem="aligned_units_grid_full",
        )
        full_path = save_aligned_units_summary_grid_single_figure(
            processed_rows=processed_rows,
            ordered_sessions=ordered_sessions,
            output_png_path=resolved_output_png_path_full,
            config=config,
        )
        outputs["all_units_with_sessions_split"] = saved_paths
        outputs["all_units_with_sessions_full"] = [full_path]

    if ALL_UNITS_ONLY:
        unique_units = build_unique_unit_summaries(processed_rows)
        all_units_only_output_png_path = resolve_output_png_path(
            export_summary_path,
            output_png_path=None,
            stem="all_unique_units_grid",
        )
        outputs["all_units_only"] = plot_all_units_only_grid(
            unique_units,
            output_png_path=all_units_only_output_png_path,
            config=config,
            units_per_row=12,
        )

    total_files = sum(len(paths) for paths in outputs.values())
    log_status(f"Completed aligned-units grid build across {total_files} output file(s).")
    return outputs


def save_aligned_units_summary_grid_from_batch_root(
    batch_root: str | Path,
    *,
    output_png_path: str | Path | None = None,
    config: AlignedUnitsGridConfig | None = None,
    export_folder_name: str = DEFAULT_ALIGNMENT_EXPORT_FOLDER_NAME,
    export_summary_name: str = DEFAULT_ALIGNMENT_EXPORT_SUMMARY_NAME,
) -> dict[str, list[Path]]:
    export_summary_path = resolve_export_summary_path_from_batch_root(
        batch_root,
        export_folder_name=export_folder_name,
        export_summary_name=export_summary_name,
    )
    return save_aligned_units_summary_grid_from_export_summary(
        export_summary_path,
        output_png_path=output_png_path,
        config=config,
    )


def example_usage() -> None:
    batch_root = Path(r"PATH\TO\260224_Sorting")
    output_png_paths_by_mode = save_aligned_units_summary_grid_from_batch_root(
        batch_root,
        config=AlignedUnitsGridConfig(
            min_sessions_present=20,
            min_spikes_after_filter=20,
            waveform_abs_threshold_uv=1500.0,
            isi_range_ms=(0.0, 50.0),
            isi_bin_ms=1.0,
            dpi=500,
        ),
    )
    for mode_name, paths in output_png_paths_by_mode.items():
        print(f"{mode_name}: {len(paths)} file(s)")
        for path in paths:
            print(f"  {path}")


def choose_batch_root() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(
            "tkinter is not available; please pass the batch root folder on the command line."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected_folder = filedialog.askdirectory(
            title="Select a Sorting batch root (for example 260224_Sorting)",
            mustexist=True,
            parent=root,
        )
    finally:
        root.destroy()

    if not selected_folder:
        raise SystemExit("No batch root folder selected.")
    return Path(selected_folder)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact aligned-units waveform/ISI grid from a Sorting batch root "
            "(for example 260224_Sorting)."
        )
    )
    parser.add_argument(
        "batch_root",
        nargs="?",
        help="Sorting batch root folder that contains units_alignment_summary/export_summary.json",
    )
    parser.add_argument(
        "--output",
        dest="output_png_path",
        help="Optional output PNG path. Defaults to stats/aligned_units_grid.png under the batch root.",
    )
    args = parser.parse_args()

    batch_root = Path(args.batch_root) if args.batch_root else choose_batch_root()
    output_png_paths_by_mode = save_aligned_units_summary_grid_from_batch_root(
        batch_root,
        output_png_path=args.output_png_path,
    )
    for mode_name, paths in output_png_paths_by_mode.items():
        if len(paths) == 1:
            print(f"{mode_name}: {paths[0]}")
        else:
            print(f"{mode_name}: {len(paths)} files")
            for path in paths:
                print(f"  {path}")


__all__ = [
    "AlignedUnitGroup",
    "AlignedUnitMember",
    "AlignedUnitsGridConfig",
    "ProcessedAlignedUnitRow",
    "ProcessedCell",
    "SessionUnitData",
    "UniqueUnitSummary",
    "build_unique_unit_summaries",
    "compute_average_waveform",
    "compute_isi_histogram",
    "determine_rowwise_isi_y_limits",
    "determine_shared_waveform_limits",
    "example_usage",
    "filter_aligned_units_by_session_presence",
    "filter_noisy_waveforms",
    "load_alignment_groups_from_export_summary",
    "load_session_unit_data_from_alignment_groups",
    "load_session_unit_data_from_member",
    "plot_all_units_only_grid",
    "plot_aligned_units_summary_grid",
    "prepare_aligned_units_grid_data",
    "resolve_export_summary_path_from_batch_root",
    "resolve_output_png_path",
    "save_aligned_units_summary_grid_pages",
    "save_aligned_units_summary_grid_from_batch_root",
    "save_aligned_units_summary_grid_from_export_summary",
    "save_aligned_units_summary_grid_single_figure",
]


if __name__ == "__main__":
    main()
