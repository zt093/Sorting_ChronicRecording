"""
All-units grid: mean waveform ± std + ISI histogram per unit (Fig4 manuscript style).

Adapted from ManuscriptFigures/fig4_unit_figures_module.plot_all_units_grid — uses
SortingAnalyzer waveforms extension instead of a WaveformExtractor.

Post-curation: shank / local channel come from ``merged_unit_id_mapping.csv``
(same as ``unit_summary_curated`` filenames: ``unit_summary_shank{S}_ch{L}_...``).
Outputs are written under ``<output_folder>/ALL_units_grid_waveformISI/``: one
PNG per shank plus, when multiple shanks are present, an additional
``ALL_units_grid_waveformISI_allShanks_*.png`` with every unit in shank/local
sort order (same per-cell labels). Waveform + ISI panels include labeled scale
bars; unit
titles use ``local_channel_on_shank``, firing rate, and % ISI < 2 ms (Fig4).
Probe-based fallback applies only if no mapping dict is passed.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_mapping_by_unit_from_merged_csv(
    output_folder: Path,
    accepted_unit_ids: np.ndarray,
    best_channel_id_per_unit: np.ndarray | None,
) -> dict[int, dict]:
    """
    Same logic as SortingLSNET_Feb2026_curation unit_summary_curated naming:
    merged_unit_id_mapping.csv plus optional hw-based inference for merged units.
    Keys: hw_channel, sg_channel, shank_id, local_channel_on_shank.
    """
    mapping_csv = Path(output_folder) / "merged_unit_id_mapping.csv"
    if not mapping_csv.is_file():
        raise FileNotFoundError(
            "merged_unit_id_mapping.csv not found (required for shank/ch labels matching "
            f"unit_summary_curated filenames): {mapping_csv}"
        )
    mapping_df = pd.read_csv(mapping_csv)
    required_cols = {
        "merged_unit_id",
        "hw_channel",
        "sg_channel",
        "shank_id",
        "local_channel_on_shank",
    }
    if not required_cols.issubset(set(mapping_df.columns)):
        raise ValueError(
            f"Missing required columns in {mapping_csv}. "
            f"Required: {sorted(required_cols)}; found: {list(mapping_df.columns)}"
        )
    mapping_by_unit: dict[int, dict] = {
        int(r.merged_unit_id): {
            "hw_channel": int(r.hw_channel),
            "sg_channel": int(r.sg_channel),
            "shank_id": int(r.shank_id),
            "local_channel_on_shank": int(r.local_channel_on_shank),
        }
        for r in mapping_df.itertuples(index=False)
    }
    accepted_unit_ids = np.asarray(accepted_unit_ids, dtype=np.int64)
    mapped_unit_ids = set(mapping_by_unit.keys())
    missing_units = sorted(set(accepted_unit_ids.tolist()) - mapped_unit_ids)
    if (
        missing_units
        and best_channel_id_per_unit is not None
        and len(best_channel_id_per_unit) == len(accepted_unit_ids)
    ):
        analyzer_hw_by_unit = {
            int(uid): int(ch)
            for uid, ch in zip(
                accepted_unit_ids.tolist(),
                best_channel_id_per_unit.astype(np.int64).tolist(),
            )
        }
        hw_meta_cols = ["sg_channel", "shank_id", "local_channel_on_shank"]
        hw_meta_lookup: dict[int, dict] = {}
        for hw, grp in mapping_df.groupby("hw_channel"):
            uniq = grp[hw_meta_cols].drop_duplicates()
            if len(uniq) == 1:
                row = uniq.iloc[0]
                hw_meta_lookup[int(hw)] = {
                    "sg_channel": int(row["sg_channel"]),
                    "shank_id": int(row["shank_id"]),
                    "local_channel_on_shank": int(row["local_channel_on_shank"]),
                }
        for uid in missing_units:
            hw = analyzer_hw_by_unit.get(int(uid), None)
            if hw is None:
                continue
            meta = hw_meta_lookup.get(int(hw), None)
            if meta is None:
                continue
            mapping_by_unit[int(uid)] = {
                "hw_channel": int(hw),
                "sg_channel": int(meta["sg_channel"]),
                "shank_id": int(meta["shank_id"]),
                "local_channel_on_shank": int(meta["local_channel_on_shank"]),
            }
        n_inferred = sum(1 for u in missing_units if int(u) in mapping_by_unit)
        if n_inferred > 0:
            print(
                f"[all_units_grid] Inferred {n_inferred} missing merged-unit mapping(s) from hw channel "
                "(same as unit_summary_curated)."
            )
    return mapping_by_unit


def _get_single_probe_from_recording(recording):
    """Return a probeinterface.Probe (unwrap ProbeGroup if needed)."""
    try:
        p = recording.get_probe()
    except Exception:
        return None
    if p is None:
        return None
    if hasattr(p, "probes"):
        plist = getattr(p, "probes", None)
        if plist is not None and len(plist) > 0:
            return plist[0]
    return p


def _normalize_shank_id(val) -> int:
    try:
        if isinstance(val, str) and val.strip().isdigit():
            return int(val.strip())
        return int(val)
    except Exception:
        return -1


def _build_hw_channel_to_shank_local(
    recording,
    channel_ids: np.ndarray,
) -> dict[int, tuple[int, int]]:
    """
    Map hardware channel id -> (shank_id, local_index_on_shank 0..n-1).

    Local index orders contacts on the same shank by depth (y coordinate, shallow first).
    For standard LSNET 12-contact shanks, local_index is 0–11.
    """
    probe = _get_single_probe_from_recording(recording)
    if probe is None:
        return {}
    try:
        pos = np.asarray(probe.contact_positions, dtype=float)
        shank_raw = np.asarray(probe.shank_ids)
    except Exception:
        return {}
    n = len(channel_ids)
    if pos.shape[0] != n or len(shank_raw) != n:
        return {}

    shank_ids = np.array([_normalize_shank_id(s) for s in shank_raw], dtype=int)
    local_on_shank = np.zeros(n, dtype=int)
    for s in np.unique(shank_ids):
        mask = shank_ids == s
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue
        ycol = pos[idxs, 1] if pos.shape[1] > 1 else pos[idxs, 0]
        order = idxs[np.argsort(-ycol)]
        for li, ii in enumerate(order):
            local_on_shank[ii] = int(li)

    out: dict[int, tuple[int, int]] = {}
    ch_list = [int(c) for c in channel_ids]
    for i, hw in enumerate(ch_list):
        out[hw] = (int(shank_ids[i]), int(local_on_shank[i]))
    return out


def _isi_lt_ms_violation_pct(
    spike_train_samples: np.ndarray | None,
    fs: float,
    thresh_ms: float,
) -> float:
    """Percent of ISIs strictly below thresh_ms (same convention as Fig4 population plots)."""
    if spike_train_samples is None or len(spike_train_samples) < 3 or fs <= 0:
        return 0.0
    isi_ms = np.diff(spike_train_samples.astype(float)) / fs * 1000.0
    if isi_ms.size == 0:
        return 0.0
    return float(100.0 * float(np.sum(isi_ms < thresh_ms)) / float(len(isi_ms)))


def _qm_fr_snr(qm, unit_id: int) -> tuple[Optional[float], Optional[float]]:
    if qm is None:
        return None, None
    uid = int(unit_id)
    fr_hz = None
    snr = None
    try:
        if uid in qm.index:
            row = qm.loc[uid]
            if "firing_rate" in qm.columns:
                fr_hz = float(row["firing_rate"])
            if "snr" in qm.columns:
                snr = float(row["snr"])
    except Exception:
        pass
    if fr_hz is None and "unit_id" in qm.columns:
        try:
            sub = qm[qm["unit_id"].astype(int) == uid]
            if len(sub) == 1:
                if "firing_rate" in sub.columns:
                    fr_hz = float(sub.iloc[0]["firing_rate"])
                if "snr" in sub.columns:
                    snr = float(sub.iloc[0]["snr"])
        except Exception:
            pass
    return fr_hz, snr


@dataclass(frozen=True)
class AllUnitsGridConfig:
    rec_path: Path
    show_unit_text: bool
    n_cols: int
    unit_wf_span_uv_min: float
    unit_gap_uv: float
    x_margin_ms: float
    mean_lw: float
    std_alpha: float
    std_band_mult: float
    wf_color: str
    std_color: str
    use_physical_scaling: bool
    scalebar_amp_uv: float
    scalebar_amp_in: float
    uv_per_in_mult: float
    scalebar_time_ms: float
    scalebar_time_in: float
    isi_max_ms: float
    isi_bin_ms: float
    isi_bar_face: str
    isi_bar_edge: str
    isi_bar_lw: float
    use_shared_isi_ymax: bool
    normalize_isi_by_unit_max: bool
    isi_norm_ymax: float
    show_fr_text: bool
    show_snr_text: bool
    show_isi_viol_pct_text: bool
    isi_viol_thresh_ms: float
    group_by_shank: bool
    plot_all_shanks_combined_page: bool
    unit_title_use_shank_local: bool
    unit_text_fontsize: int
    unit_text_color: str
    show_global_isi_scalebar: bool
    show_global_isi_scalebar_labels: bool
    isi_scalebar_x_ms: float
    isi_scalebar_y_count: float
    isi_scalebar_x_in: float
    isi_scalebar_y_in: float
    isi_scalebar_lw: float
    isi_scalebar_color: str
    isi_scalebar_label_fontsize: int
    isi_scalebar_x_ax: float
    isi_scalebar_y_ax: float
    show_global_scalebar: bool
    show_global_scalebar_labels: bool
    scalebar_lw: float
    scalebar_color: str
    scalebar_label_fontsize: int
    scalebar_x_ax: float
    scalebar_y_ax: float
    show_fig_title: bool
    fig_title_fontsize: int
    fig_title_color: str
    fig_title_y_ax: float
    savefig_pad_inches: float
    dpi: int


def default_all_units_grid_config(rec_path: Path) -> AllUnitsGridConfig:
    """Defaults aligned with Fig4_4S_Recording_units.py GRID_CONFIG."""
    return AllUnitsGridConfig(
        rec_path=Path(rec_path),
        show_unit_text=True,
        n_cols=12,
        unit_wf_span_uv_min=500.0,
        unit_gap_uv=10.0,
        x_margin_ms=0.05,
        mean_lw=0.6,
        std_alpha=0.25,
        std_band_mult=1.0,
        wf_color="k",
        std_color="k",
        use_physical_scaling=True,
        scalebar_amp_uv=100.0,
        scalebar_amp_in=0.25 / 3.0,
        uv_per_in_mult=2.0,
        scalebar_time_ms=1.0,
        scalebar_time_in=0.12,
        isi_max_ms=20.0,
        isi_bin_ms=1.0,
        isi_bar_face="k",
        isi_bar_edge="w",
        isi_bar_lw=0.0,
        use_shared_isi_ymax=True,
        normalize_isi_by_unit_max=True,
        isi_norm_ymax=1.05,
        show_fr_text=True,
        show_snr_text=False,
        show_isi_viol_pct_text=True,
        isi_viol_thresh_ms=2.0,
        group_by_shank=True,
        plot_all_shanks_combined_page=True,
        unit_title_use_shank_local=True,
        unit_text_fontsize=5,
        unit_text_color="k",
        show_global_isi_scalebar=True,
        show_global_isi_scalebar_labels=True,
        isi_scalebar_x_ms=10.0,
        isi_scalebar_y_count=10.0,
        isi_scalebar_x_in=0.25,
        isi_scalebar_y_in=0.25,
        isi_scalebar_lw=2.0,
        isi_scalebar_color="k",
        isi_scalebar_label_fontsize=9,
        isi_scalebar_x_ax=0.06,
        isi_scalebar_y_ax=0.002,
        show_global_scalebar=True,
        show_global_scalebar_labels=True,
        scalebar_lw=2.0,
        scalebar_color="k",
        scalebar_label_fontsize=9,
        scalebar_x_ax=0.015,
        scalebar_y_ax=0.002,
        show_fig_title=True,
        fig_title_fontsize=12,
        fig_title_color="k",
        fig_title_y_ax=1.085,
        savefig_pad_inches=0.12,
        dpi=600,
    )


def plot_all_units_grid_waveform_isi(
    sorting,
    recording,
    analyzer,
    out_fig_dir: Path,
    config: AllUnitsGridConfig,
    ms_before: float,
    ms_after: float,
    ordered_channel_ids: Optional[Iterable] = None,
    mapping_by_unit: Optional[dict[int, dict]] = None,
) -> Path:
    """
    Waveforms from analyzer.get_extension('waveforms').get_waveforms_one_unit.
    Skips units with fewer than 20 spikes in the extracted waveforms.

    If ``mapping_by_unit`` is set (keys = unit_id, values include ``shank_id``,
    ``local_channel_on_shank`` from merged_unit_id_mapping.csv), shank layout and
    ``ch`` labels match unit_summary_curated exports. Otherwise shank/local are
    inferred from the recording probe (legacy).
    """
    out_fig_dir = Path(out_fig_dir)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    wf_ext = analyzer.get_extension("waveforms")
    channel_ids = np.array(analyzer.channel_ids)

    x_min = float(-ms_before - config.x_margin_ms)
    x_max = float(ms_after + config.x_margin_ms)
    x_span_ms = x_max - x_min

    uv_per_in = (config.scalebar_amp_uv / config.scalebar_amp_in) * config.uv_per_in_mult
    ms_per_in = config.scalebar_time_ms / config.scalebar_time_in

    wf_w_in = (x_span_ms / ms_per_in) if config.use_physical_scaling else 1.35
    isi_w_in = wf_w_in
    wf_isi_gap_in = 0.08

    left_margin_in = 0.10
    right_margin_in = 0.05
    # Extra headroom when a figure title is drawn (avoids overlap with top row of cells).
    top_margin_in = 0.18 if config.show_fig_title else 0.05
    bottom_margin_in = 0.20
    col_gap_in = 0.12
    row_gap_in = (config.unit_gap_uv / uv_per_in) if config.use_physical_scaling else 0.20

    fs = float(recording.get_sampling_frequency())
    rec_dur_s = float(recording.get_num_frames()) / fs

    unit_ids_sorted = sorted([int(u) for u in sorting.get_unit_ids()], key=lambda u: int(u))

    qm = None
    try:
        qm = analyzer.get_extension("quality_metrics").get_data()
    except Exception:
        qm = None

    n_samples = None
    for uid in unit_ids_sorted:
        try:
            wfs = wf_ext.get_waveforms_one_unit(unit_id=uid)
        except Exception:
            continue
        if wfs is None or wfs.size == 0 or wfs.shape[0] < 20:
            continue
        n_samples = int(wfs.shape[1])
        break
    if n_samples is None:
        raise RuntimeError(
            "No unit with ≥20 waveforms; cannot build grid. Check waveforms extension."
        )

    t_ms = np.linspace(-ms_before, ms_after, n_samples)

    hw_to_shank_local = (
        {}
        if mapping_by_unit is not None
        else _build_hw_channel_to_shank_local(recording, channel_ids)
    )

    unit_data = []
    max_required_span_uv = 0.0
    skipped_no_mapping = 0

    for unit_id in unit_ids_sorted:
        try:
            wfs = wf_ext.get_waveforms_one_unit(unit_id=unit_id)
        except Exception:
            continue
        if wfs is None or wfs.shape[0] < 20:
            continue

        mean_allch = wfs.mean(axis=0)
        peak_by_ch = np.max(np.abs(mean_allch), axis=0)
        rep_ch_local_idx = int(np.argmax(peak_by_ch))
        max_amp_channel_id = channel_ids[rep_ch_local_idx]

        wfs_1ch = wfs[:, :, rep_ch_local_idx]
        mean_1ch = wfs_1ch.mean(axis=0)
        std_1ch = wfs_1ch.std(axis=0)
        y_lo = mean_1ch - config.std_band_mult * std_1ch
        y_hi = mean_1ch + config.std_band_mult * std_1ch

        required_span_uv = float(np.max(y_hi) - np.min(y_lo))
        max_required_span_uv = max(max_required_span_uv, required_span_uv)

        st = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0)
        if st is None or len(st) < 3:
            isi_ms = np.array([], dtype=float)
        else:
            isi_ms = np.diff(st).astype(float) / fs * 1000.0
            isi_ms = isi_ms[(isi_ms > 0) & (isi_ms < config.isi_max_ms)]

        fr_hz, snr = _qm_fr_snr(qm, unit_id)
        if fr_hz is None:
            fr_hz = float(len(st)) / rec_dur_s if (st is not None and rec_dur_s > 0) else 0.0

        if mapping_by_unit is not None:
            meta = mapping_by_unit.get(int(unit_id))
            if meta is None:
                skipped_no_mapping += 1
                continue
            shank_id = int(meta["shank_id"])
            local_on_shank = int(meta["local_channel_on_shank"])
        else:
            int_hw = int(max_amp_channel_id)
            sl = hw_to_shank_local.get(int_hw)
            shank_id = int(sl[0]) if sl is not None else None
            local_on_shank = int(sl[1]) if sl is not None else None
        isi_viol_pct = _isi_lt_ms_violation_pct(st, fs, config.isi_viol_thresh_ms)

        unit_data.append(
            dict(
                unit_id=int(unit_id),
                max_amp_channel_id=max_amp_channel_id,
                shank_id=shank_id,
                local_on_shank=local_on_shank,
                isi_viol_pct=isi_viol_pct,
                mean=mean_1ch,
                std=std_1ch,
                y_lo=y_lo,
                y_hi=y_hi,
                isi_ms=isi_ms,
                fr_hz=fr_hz,
                snr=snr,
            )
        )

    if skipped_no_mapping:
        print(
            f"[all_units_grid] Skipped {skipped_no_mapping} unit(s) with no entry in merged "
            "unit mapping (no unit_summary_curated-style shank/ch)."
        )

    if ordered_channel_ids is not None:
        channel_id_to_index = {str(c): i for i, c in enumerate(ordered_channel_ids)}
        n_ordered = len(ordered_channel_ids)

        def _sort_key(d):
            idx = channel_id_to_index.get(str(d["max_amp_channel_id"]), n_ordered)
            return (idx, d["unit_id"])

        unit_data.sort(key=_sort_key)
    elif config.group_by_shank:
        unit_data.sort(
            key=lambda d: (
                d["shank_id"] if d["shank_id"] is not None else 10**9,
                d["local_on_shank"] if d["local_on_shank"] is not None else 10**9,
                d["unit_id"],
            )
        )
    else:
        unit_data.sort(key=lambda d: d["unit_id"])

    isi_bins = np.arange(0.0, config.isi_max_ms + config.isi_bin_ms, config.isi_bin_ms)

    unit_wf_span_uv = max(float(config.unit_wf_span_uv_min), float(max_required_span_uv)) * 1.10
    wf_h_in = (unit_wf_span_uv / uv_per_in) if config.use_physical_scaling else 1.15

    n_units = len(unit_data)
    if n_units == 0:
        raise RuntimeError("No units passed waveform/ISI inclusion; empty grid.")

    if (
        config.group_by_shank
        and ordered_channel_ids is None
        and any(d["shank_id"] is not None for d in unit_data)
    ):
        shank_vals = sorted(
            {d["shank_id"] for d in unit_data if d["shank_id"] is not None}
        )
        page_groups: list[tuple[Optional[int], list]] = [
            (s, [d for d in unit_data if d["shank_id"] == s]) for s in shank_vals
        ]
        orphan = [d for d in unit_data if d["shank_id"] is None]
        if orphan:
            page_groups.append((None, orphan))
    else:
        page_groups = [(None, unit_data)]

    last_out: Optional[Path] = None
    for page_shank, page_units in page_groups:
        if not page_units:
            continue
        last_out = _plot_all_units_grid_one_page(
            page_units=page_units,
            page_shank=page_shank,
            combined_all_shanks=False,
            t_ms=t_ms,
            isi_bins=isi_bins,
            out_fig_dir=out_fig_dir,
            config=config,
            ms_before=ms_before,
            ms_after=ms_after,
            x_min=x_min,
            x_max=x_max,
            x_span_ms=x_span_ms,
            uv_per_in=uv_per_in,
            ms_per_in=ms_per_in,
            wf_w_in=wf_w_in,
            isi_w_in=isi_w_in,
            wf_isi_gap_in=wf_isi_gap_in,
            left_margin_in=left_margin_in,
            right_margin_in=right_margin_in,
            top_margin_in=top_margin_in,
            bottom_margin_in=bottom_margin_in,
            col_gap_in=col_gap_in,
            row_gap_in=row_gap_in,
            unit_wf_span_uv=unit_wf_span_uv,
            wf_h_in=wf_h_in,
        )

    if (
        config.plot_all_shanks_combined_page
        and ordered_channel_ids is None
        and len(page_groups) > 1
    ):
        last_out = _plot_all_units_grid_one_page(
            page_units=unit_data,
            page_shank=None,
            combined_all_shanks=True,
            t_ms=t_ms,
            isi_bins=isi_bins,
            out_fig_dir=out_fig_dir,
            config=config,
            ms_before=ms_before,
            ms_after=ms_after,
            x_min=x_min,
            x_max=x_max,
            x_span_ms=x_span_ms,
            uv_per_in=uv_per_in,
            ms_per_in=ms_per_in,
            wf_w_in=wf_w_in,
            isi_w_in=isi_w_in,
            wf_isi_gap_in=wf_isi_gap_in,
            left_margin_in=left_margin_in,
            right_margin_in=right_margin_in,
            top_margin_in=top_margin_in,
            bottom_margin_in=bottom_margin_in,
            col_gap_in=col_gap_in,
            row_gap_in=row_gap_in,
            unit_wf_span_uv=unit_wf_span_uv,
            wf_h_in=wf_h_in,
        )

    assert last_out is not None
    return last_out


def _plot_all_units_grid_one_page(
    *,
    page_units: list,
    page_shank: Optional[int],
    combined_all_shanks: bool,
    t_ms: np.ndarray,
    isi_bins: np.ndarray,
    out_fig_dir: Path,
    config: AllUnitsGridConfig,
    ms_before: float,
    ms_after: float,
    x_min: float,
    x_max: float,
    x_span_ms: float,
    uv_per_in: float,
    ms_per_in: float,
    wf_w_in: float,
    isi_w_in: float,
    wf_isi_gap_in: float,
    left_margin_in: float,
    right_margin_in: float,
    top_margin_in: float,
    bottom_margin_in: float,
    col_gap_in: float,
    row_gap_in: float,
    unit_wf_span_uv: float,
    wf_h_in: float,
) -> Path:
    """Render one grid page (one shank or all units if shank grouping disabled)."""
    unit_data = page_units
    global_isi_ymax = 1.0
    if (not config.normalize_isi_by_unit_max) and config.use_shared_isi_ymax:
        max_count = 0
        for d in unit_data:
            if d["isi_ms"].size > 0:
                counts, _ = np.histogram(d["isi_ms"], bins=isi_bins)
                if counts.size > 0:
                    max_count = max(max_count, int(counts.max()))
        global_isi_ymax = float(max(1, int(np.ceil(max_count * 1.10))))

    n_units = len(unit_data)
    n_rows = int(math.ceil(n_units / float(config.n_cols)))
    cell_w_in = wf_w_in + wf_isi_gap_in + isi_w_in

    fig_w_in = left_margin_in + right_margin_in + (config.n_cols * cell_w_in) + (
        (config.n_cols - 1) * col_gap_in
    )
    fig_h_in = bottom_margin_in + top_margin_in + (n_rows * wf_h_in) + (
        (n_rows - 1) * row_gap_in
    )

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=config.dpi, facecolor="w")
    ax_global = fig.add_axes([0, 0, 1, 1])
    ax_global.set_axis_off()

    if config.show_fig_title:
        rec_path = Path(config.rec_path)
        rec_stem = rec_path.stem

        animal_match = re.search(r"(LSNET\d+)", rec_path.as_posix())
        animal_id = animal_match.group(1) if animal_match else None

        date_token = None
        stem_parts = rec_stem.split("_")
        if stem_parts and len(stem_parts[0]) == 8 and stem_parts[0].isdigit():
            date_token = stem_parts[0]

        if animal_id and date_token:
            animal_session = f"{animal_id} {date_token}"
        elif animal_id:
            animal_session = animal_id
        elif date_token:
            animal_session = date_token
        else:
            animal_session = rec_stem

        base_title = (
            animal_session + ": All Units - Mean +- STD Waveforms with ISI Histograms"
        )
        if combined_all_shanks:
            fig_title_text = f"{base_title} (all shanks, {n_units} units)"
        elif page_shank is not None:
            fig_title_text = f"{base_title} (shank {page_shank})"
        else:
            fig_title_text = base_title

        ax_global.text(
            0.5,
            config.fig_title_y_ax,
            fig_title_text,
            transform=ax_global.transAxes,
            ha="center",
            va="top",
            fontsize=config.fig_title_fontsize,
            color=config.fig_title_color,
            clip_on=False,
        )

    def inch_to_figx(x_in: float) -> float:
        return x_in / fig_w_in

    def inch_to_figy(y_in: float) -> float:
        return y_in / fig_h_in

    for idx, d in enumerate(unit_data):
        r = idx // config.n_cols
        c = idx % config.n_cols

        cell_x_in = left_margin_in + c * (cell_w_in + col_gap_in)
        cell_y_in = bottom_margin_in + (n_rows - 1 - r) * (wf_h_in + row_gap_in)

        wf_left = inch_to_figx(cell_x_in)
        wf_bottom = inch_to_figy(cell_y_in)
        wf_w = inch_to_figx(wf_w_in)
        wf_h = inch_to_figy(wf_h_in)

        ax_wf = fig.add_axes([wf_left, wf_bottom, wf_w, wf_h])
        ax_wf.set_axis_off()

        local_min = float(np.min(d["y_lo"]))
        local_max = float(np.max(d["y_hi"]))
        y_shift = (0.5 * unit_wf_span_uv) - (0.5 * (local_min + local_max))
        y_lo_s = d["y_lo"] + y_shift
        y_hi_s = d["y_hi"] + y_shift
        y_mean_s = d["mean"] + y_shift

        ax_wf.fill_between(
            t_ms,
            y_lo_s,
            y_hi_s,
            color=config.std_color,
            alpha=config.std_alpha,
            linewidth=0,
        )
        ax_wf.plot(
            t_ms,
            y_mean_s,
            color=config.wf_color,
            linewidth=config.mean_lw,
        )
        ax_wf.set_xlim(x_min, x_max)
        ax_wf.set_ylim(0.0, unit_wf_span_uv)

        if config.show_unit_text:
            parts = []
            if config.unit_title_use_shank_local and d.get("local_on_shank") is not None:
                parts.append(f"ch{d['local_on_shank']}")
            else:
                parts.append(f"hw{int(d['max_amp_channel_id'])}")
            if config.show_fr_text:
                parts.append(f"{d['fr_hz']:.1f}Hz")
            if config.show_isi_viol_pct_text:
                parts.append(f"{d['isi_viol_pct']:.0f}%<{config.isi_viol_thresh_ms:g}ms")
            if config.show_snr_text and d.get("snr") is not None and np.isfinite(d["snr"]):
                parts.append(f"SNR{d['snr']:.1f}")
            txt = " ".join(parts)
            ax_wf.text(
                0.02,
                0.98,
                txt,
                transform=ax_wf.transAxes,
                ha="left",
                va="top",
                fontsize=config.unit_text_fontsize,
                color=config.unit_text_color,
                clip_on=False,
            )

        isi_left = inch_to_figx(cell_x_in + wf_w_in + wf_isi_gap_in)
        isi_bottom = wf_bottom
        isi_w = inch_to_figx(isi_w_in)
        isi_h = wf_h

        ax_isi = fig.add_axes([isi_left, isi_bottom, isi_w, isi_h])
        ax_isi.set_axis_off()

        if d["isi_ms"].size > 0:
            counts, edges = np.histogram(d["isi_ms"], bins=isi_bins)
        else:
            counts = np.zeros(len(isi_bins) - 1, dtype=int)
            edges = isi_bins

        bin_width = float(edges[1] - edges[0])
        left_edges = edges[:-1]

        if config.normalize_isi_by_unit_max:
            cmax = float(np.max(counts)) if counts.size > 0 else 0.0
            counts_plot = counts.astype(float) / cmax if cmax > 0 else counts.astype(float)
        else:
            counts_plot = counts.astype(float)

        ax_isi.bar(
            left_edges,
            counts_plot,
            width=bin_width,
            align="edge",
            color=config.isi_bar_face,
            edgecolor=config.isi_bar_edge,
            linewidth=config.isi_bar_lw,
        )
        ax_isi.set_xlim(0.0, config.isi_max_ms)

        if config.normalize_isi_by_unit_max:
            ax_isi.set_ylim(0.0, config.isi_norm_ymax)
        else:
            if config.use_shared_isi_ymax:
                ax_isi.set_ylim(0.0, global_isi_ymax)
            else:
                local_ymax = float(max(1, int(np.ceil(np.max(counts_plot) * 1.10))))
                ax_isi.set_ylim(0.0, local_ymax)

    if config.show_global_scalebar:
        time_ax_len = config.scalebar_time_in / fig_w_in
        amp_ax_len = config.scalebar_amp_in / fig_h_in
        x0 = config.scalebar_x_ax
        y0 = config.scalebar_y_ax

        ax_global.plot(
            [x0, x0 + time_ax_len],
            [y0, y0],
            transform=ax_global.transAxes,
            color=config.scalebar_color,
            linewidth=config.scalebar_lw,
            solid_capstyle="butt",
            clip_on=False,
        )
        ax_global.plot(
            [x0, x0],
            [y0, y0 + amp_ax_len],
            transform=ax_global.transAxes,
            color=config.scalebar_color,
            linewidth=config.scalebar_lw,
            solid_capstyle="butt",
            clip_on=False,
        )

        if config.show_global_scalebar_labels:
            ax_global.text(
                x0 + time_ax_len / 2,
                y0 - 0.008,
                f"{config.scalebar_time_ms:g} ms",
                transform=ax_global.transAxes,
                ha="center",
                va="top",
                fontsize=config.scalebar_label_fontsize,
                color=config.scalebar_color,
                clip_on=False,
            )
            ax_global.text(
                x0 - 0.008,
                y0 + amp_ax_len / 2,
                f"{config.scalebar_amp_uv:g} uV",
                transform=ax_global.transAxes,
                ha="right",
                va="center",
                fontsize=config.scalebar_label_fontsize,
                color=config.scalebar_color,
                clip_on=False,
            )

    if config.show_global_isi_scalebar:
        isi_scalebar_x_in = isi_w_in * (config.isi_scalebar_x_ms / config.isi_max_ms)
        x_ax_len = isi_scalebar_x_in / fig_w_in
        y_ax_len = config.isi_scalebar_y_in / fig_h_in
        x0 = config.isi_scalebar_x_ax
        y0 = config.isi_scalebar_y_ax

        ax_global.plot(
            [x0, x0 + x_ax_len],
            [y0, y0],
            transform=ax_global.transAxes,
            color=config.isi_scalebar_color,
            linewidth=config.isi_scalebar_lw,
            solid_capstyle="butt",
            clip_on=False,
        )

        if not config.normalize_isi_by_unit_max:
            ax_global.plot(
                [x0, x0],
                [y0, y0 + y_ax_len],
                transform=ax_global.transAxes,
                color=config.isi_scalebar_color,
                linewidth=config.isi_scalebar_lw,
                solid_capstyle="butt",
                clip_on=False,
            )

        if config.show_global_isi_scalebar_labels:
            ax_global.text(
                x0 + x_ax_len / 2,
                y0 - 0.008,
                f"{config.isi_scalebar_x_ms:g} ms",
                transform=ax_global.transAxes,
                ha="center",
                va="top",
                fontsize=config.isi_scalebar_label_fontsize,
                color=config.isi_scalebar_color,
                clip_on=False,
            )
            if not config.normalize_isi_by_unit_max:
                ax_global.text(
                    x0 - 0.008,
                    y0 + y_ax_len / 2,
                    f"{config.isi_scalebar_y_count:g}",
                    transform=ax_global.transAxes,
                    ha="right",
                    va="center",
                    fontsize=config.isi_scalebar_label_fontsize,
                    color=config.isi_scalebar_color,
                    clip_on=False,
                )

    if combined_all_shanks:
        shank_part = "_allShanks"
    elif page_shank is not None:
        shank_part = f"_shank{int(page_shank)}"
    else:
        shank_part = ""
    out_base = out_fig_dir / (
        f"ALL_units_grid_waveformISI"
        f"{shank_part}"
        f"_cols{config.n_cols}"
        f"_ms{ms_before:g}-{ms_after:g}"
        f"_span{int(config.unit_wf_span_uv_min)}uV"
        f"_gap{int(config.unit_gap_uv)}uV"
        f"_isi{int(config.isi_max_ms)}ms"
        f"_txt{int(config.show_unit_text)}"
    )
    pad = float(config.savefig_pad_inches)
    fig.savefig(str(out_base) + ".png", dpi=config.dpi, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)

    print(f"[all_units_grid] Saved: {out_base}.png")
    return out_base


def save_postcuration_all_units_grid(
    output_folder: Path,
    sorting,
    recording,
    analyzer,
    *,
    rec_path: Path | None = None,
    ms_before: float = 1.0,
    ms_after: float = 2.0,
    ordered_channel_ids: Optional[Iterable] = None,
    best_channel_id_per_unit: np.ndarray | None = None,
    mapping_by_unit: Optional[dict[int, dict]] = None,
) -> Path:
    """
    Loads ``merged_unit_id_mapping.csv`` under ``output_folder`` so shank/ch match
    ``unit_summary_curated`` filenames. Pass ``best_channel_id_per_unit`` (template
    best channel per unit, same as curation export) to infer merged units missing
    from the CSV.

    Figures are saved under ``output_folder / "ALL_units_grid_waveformISI"``.
    """
    output_folder = Path(output_folder)
    grid_subdir = output_folder / "ALL_units_grid_waveformISI"
    grid_subdir.mkdir(parents=True, exist_ok=True)
    rp = Path(rec_path) if rec_path is not None else output_folder.parent
    cfg = default_all_units_grid_config(rp)
    accepted_unit_ids = np.asarray(sorting.get_unit_ids(), dtype=np.int64)
    if mapping_by_unit is None:
        mapping_by_unit = build_mapping_by_unit_from_merged_csv(
            output_folder,
            accepted_unit_ids,
            best_channel_id_per_unit,
        )
    return plot_all_units_grid_waveform_isi(
        sorting,
        recording,
        analyzer,
        grid_subdir,
        cfg,
        ms_before=ms_before,
        ms_after=ms_after,
        ordered_channel_ids=ordered_channel_ids,
        mapping_by_unit=mapping_by_unit,
    )
