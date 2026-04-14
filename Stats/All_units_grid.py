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
    representative_channel_index: int


@dataclass
class ProcessedAlignedUnitRow:
    group_id: str
    members_present: int
    cells_by_session: dict[str, ProcessedCell]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignedUnitsGridConfig:
    min_sessions_present: int = 20
    min_spikes_after_filter: int = 20
    waveform_abs_threshold_uv: float = 1500.0
    isi_range_ms: tuple[float, float] = (0.0, 50.0)
    isi_bin_ms: float = 1.0
    waveform_panel_width_ratio: float = 1.0
    isi_panel_width_ratio: float = 1.35
    figure_facecolor: str = "white"
    waveform_color: str = "black"
    isi_color: str = "black"
    row_height_in: float = 0.42
    session_width_in: float = 0.95
    max_figure_width_in: float | None = None
    dpi: int = 500
    png_facecolor: str = "white"
    waveform_linewidth: float = 0.55
    isi_linewidth: float = 0.0
    hide_axes: bool = True
    left_margin: float = 0.005
    right_margin: float = 0.005
    top_margin: float = 0.005
    bottom_margin: float = 0.005
    wspace: float = 0.01
    hspace: float = 0.01
    analyzer_folder_name: str = DEFAULT_ANALYZER_FOLDER_NAME
    ms_before_fallback: float = DEFAULT_MS_BEFORE
    ms_after_fallback: float = DEFAULT_MS_AFTER
    max_pixels_per_side: int = 60000
    max_rows_per_figure: int | None = None
    show_session_headers: bool = True
    session_header_fontsize: int = 6
    session_header_height_in: float = 0.22


def make_member_lookup_key(member: AlignedUnitMember) -> tuple[str, int]:
    if member.output_folder is None:
        raise ValueError(
            f"Aligned member {member.session_name} unit {member.unit_id} has no output_folder."
        )
    return (str(Path(member.output_folder)), int(member.unit_id))


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

    ordered_sessions = [
        session_name
        for session_name, _session_index in sorted(
            session_order_lookup.items(),
            key=lambda item: (item[1], item[0]),
        )
    ]
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
) -> Path:
    export_summary_path = Path(export_summary_path)
    if output_png_path is not None:
        return Path(output_png_path)
    batch_root = export_summary_path.parent.parent
    stats_candidates = [batch_root / "stats", batch_root / "Stats"]
    stats_folder = next((path for path in stats_candidates if path.exists()), stats_candidates[0])
    stats_folder.mkdir(parents=True, exist_ok=True)
    return stats_folder / "aligned_units_grid.png"


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

    return SessionUnitData(
        session_name=member.session_name,
        session_index=member.session_index,
        unit_id=int(member.unit_id),
        output_folder=output_folder,
        analyzer_folder=analyzer_folder,
        waveform_time_axis_ms=waveform_time_axis_ms,
        spike_waveforms=spike_waveforms,
        spike_times_s=spike_times_s,
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
                representative_channel_index=session_unit_data.representative_channel_index,
            )

        if len(unique_sessions) < config.min_sessions_present:
            continue
        if not cells_by_session:
            continue

        rows.append(
            ProcessedAlignedUnitRow(
                group_id=group.group_id,
                members_present=len(unique_sessions),
                cells_by_session=cells_by_session,
                metadata=dict(group.metadata),
            )
        )
        if group_index == 1 or group_index % 25 == 0 or group_index == len(candidate_groups):
            log_status(
                f"Prepared row {group_index}/{len(candidate_groups)} "
                f"(rows kept so far: {len(rows)})"
            )

    rows.sort(key=lambda row: (-row.members_present, row.group_id))
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


def plot_aligned_units_summary_grid(
    processed_rows: list[ProcessedAlignedUnitRow],
    ordered_sessions: list[str],
    *,
    config: AlignedUnitsGridConfig | None = None,
    output_png_path: str | Path | None = None,
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

    fig_width = n_sessions * config.session_width_in
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
        n_sessions * 2,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        left=config.left_margin,
        right=1.0 - config.right_margin,
        top=1.0 - config.top_margin,
        bottom=config.bottom_margin,
        wspace=config.wspace,
        hspace=config.hspace,
    )

    if config.show_session_headers:
        for session_index, session_name in enumerate(ordered_sessions):
            ax_header = fig.add_subplot(outer_grid[0, session_index * 2 : session_index * 2 + 2])
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

    for row_index, row in enumerate(processed_rows):
        isi_ylim = rowwise_isi_limits[row.group_id]
        for session_index, session_name in enumerate(ordered_sessions):
            cell = row.cells_by_session.get(session_name)
            grid_row_index = row_index + header_rows
            ax_wf = fig.add_subplot(outer_grid[grid_row_index, session_index * 2])
            ax_isi = fig.add_subplot(outer_grid[grid_row_index, session_index * 2 + 1])

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
    pages = _split_rows_into_pages(
        processed_rows,
        ordered_sessions,
        config=config,
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
            config=config,
            output_png_path=page_output_path,
        )
        plt.close(fig)
        if saved_path is None:
            raise RuntimeError(f"Expected saved path for page {page_index + 1}.")
        output_paths.append(saved_path)

    return output_paths


def save_aligned_units_summary_grid_from_export_summary(
    export_summary_path: str | Path,
    *,
    output_png_path: str | Path | None = None,
    config: AlignedUnitsGridConfig | None = None,
) -> list[Path]:
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
    resolved_output_png_path = resolve_output_png_path(
        export_summary_path,
        output_png_path=output_png_path,
    )
    saved_paths = save_aligned_units_summary_grid_pages(
        processed_rows=processed_rows,
        ordered_sessions=ordered_sessions,
        output_png_path=resolved_output_png_path,
        config=config,
    )
    if len(saved_paths) == 1:
        log_status(f"Completed aligned-units grid build: {saved_paths[0]}")
    else:
        log_status(
            f"Completed aligned-units grid build across {len(saved_paths)} files. "
            f"First page: {saved_paths[0]}"
        )
    return saved_paths


def save_aligned_units_summary_grid_from_batch_root(
    batch_root: str | Path,
    *,
    output_png_path: str | Path | None = None,
    config: AlignedUnitsGridConfig | None = None,
    export_folder_name: str = DEFAULT_ALIGNMENT_EXPORT_FOLDER_NAME,
    export_summary_name: str = DEFAULT_ALIGNMENT_EXPORT_SUMMARY_NAME,
) -> list[Path]:
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
    output_png_paths = save_aligned_units_summary_grid_from_batch_root(
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
    print(f"Saved aligned units grid to {len(output_png_paths)} file(s):")
    for path in output_png_paths:
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
    output_png_paths = save_aligned_units_summary_grid_from_batch_root(
        batch_root,
        output_png_path=args.output_png_path,
    )
    if len(output_png_paths) == 1:
        print(f"Saved aligned units grid to: {output_png_paths[0]}")
    else:
        print(f"Saved aligned units grid to {len(output_png_paths)} files:")
        for path in output_png_paths:
            print(f"  {path}")


__all__ = [
    "AlignedUnitGroup",
    "AlignedUnitMember",
    "AlignedUnitsGridConfig",
    "ProcessedAlignedUnitRow",
    "ProcessedCell",
    "SessionUnitData",
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
    "plot_aligned_units_summary_grid",
    "prepare_aligned_units_grid_data",
    "resolve_export_summary_path_from_batch_root",
    "resolve_output_png_path",
    "save_aligned_units_summary_grid_pages",
    "save_aligned_units_summary_grid_from_batch_root",
    "save_aligned_units_summary_grid_from_export_summary",
]


if __name__ == "__main__":
    main()
