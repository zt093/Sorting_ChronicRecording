"""
Day-level HTML alignment review.

This reuses the browser review workflow from Alignment_html.py, but changes the
input source:

- Alignment_html.py aligns raw per-session units within one day.
- Alignment_days.py aligns the exported per-day unique units across days.

Expected input layout
---------------------
Point this script at either:
- one organized daily sorting folder such as ``260224_Sorting``, or
- a parent folder that contains multiple ``*_Sorting`` folders.

For each day, this loader reads per-page exports produced by Alignment_html.py:
``sh*/units_alignment_summary/export_summary_sg_*.json``.

Each exported cross-session group is collapsed into one synthetic "unit" for
that day/page by averaging the member-unit metrics and similarity vectors.
Those synthetic units are then aligned across days, still scoped to the same
shank and SG channel.

The final export keeps both views:
- `day_members`: one synthetic member per day-level aligned unit
- `members` / `source_members`: flattened original source-session members

This makes downstream analysis able to work from the full aligned-session
membership while still preserving the day-level alignment structure.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import traceback
import webbrowser

from http.server import ThreadingHTTPServer

import numpy as np
import spikeinterface.full as si

import Alignment_html as html_review
from Units_alignment_UI import (
    DEFAULT_EXPORT_FOLDER_NAME,
    PageSummary,
    SessionSummary,
    UnitSummary,
    build_metrics_lookup,
    choose_output_root,
    ensure_required_extensions,
    find_unit_summary_image,
    get_autocorrelogram_vector,
    get_trough_to_peak_duration_ms,
    get_waveform_vector,
    infer_unit_channel_metadata,
    load_unit_channel_mapping,
    safe_float,
    safe_int,
)


DAY_SORTING_ROOT_PATTERN = re.compile(r"(?P<day_code>\d{6})_Sorting$")
DAY_MANIFEST_NAME = "alignment_days_manifest.json"
DAY_SUMMARY_FOLDER_NAME = "alignment_days_summary"
HTML_TITLE = "Alignment Review Across Days"
SOURCE_ALIGNMENT_MANIFEST_NAME = "alignment_manifest.json"


def day_unit_record_key(unit: UnitSummary) -> str:
    return f"{unit.session_index}:{unit.shank_id}:{unit.sg_channel}:{unit.unit_id}"


html_review.MANIFEST_NAME = DAY_MANIFEST_NAME
html_review.HTML_TITLE = HTML_TITLE
html_review.unit_record_key = day_unit_record_key
_BASE_BUILD_HTML_SHELL = html_review.build_html_shell


def parse_day_code_from_sorting_root(folder: Path) -> str | None:
    match = DAY_SORTING_ROOT_PATTERN.fullmatch(folder.name)
    if not match:
        return None
    return match.group("day_code")


def build_day_summary_folder_name(day_roots: list[Path]) -> str:
    day_codes = [
        parse_day_code_from_sorting_root(day_root)
        for day_root in day_roots
    ]
    clean_day_codes = [day_code for day_code in day_codes if day_code]
    if not clean_day_codes:
        return DAY_SUMMARY_FOLDER_NAME
    return f"{DAY_SUMMARY_FOLDER_NAME}_{clean_day_codes[0]}_{clean_day_codes[-1]}"


def parse_input_roots_text(raw_text: str) -> list[Path]:
    parts = [part.strip().strip('"').strip("'") for part in raw_text.split(",")]
    roots = [Path(part) for part in parts if part]
    if not roots:
        raise ValueError("No input folders were provided.")
    return roots


def prompt_for_input_roots() -> list[Path]:
    raw_text = input(
        "Enter one or more daily *_Sorting folders or parent folders, separated by commas: "
    ).strip()
    if not raw_text:
        raise ValueError("No input folders were provided.")
    return parse_input_roots_text(raw_text)


def discover_day_sorting_roots(input_roots: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()

    for raw_root in input_roots:
        root_folder = raw_root.resolve()
        if not root_folder.exists():
            raise FileNotFoundError(f"Input path does not exist: {root_folder}")
        if not root_folder.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {root_folder}")

        if parse_day_code_from_sorting_root(root_folder) is not None:
            if root_folder not in seen:
                seen.add(root_folder)
                discovered.append(root_folder)
            continue

        day_roots = sorted(
            [
                child.resolve()
                for child in root_folder.glob("*_Sorting")
                if child.is_dir() and parse_day_code_from_sorting_root(child) is not None
            ]
        )
        for day_root in day_roots:
            if day_root in seen:
                continue
            seen.add(day_root)
            discovered.append(day_root)

    if not discovered:
        joined_roots = ", ".join(str(path.resolve()) for path in input_roots)
        raise FileNotFoundError(
            "No organized daily sorting folders like '260224_Sorting' were found in: "
            f"{joined_roots}"
        )
    return sorted(discovered)


def average_scalar(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def average_int(values: list[int | None]) -> int | None:
    clean = [int(value) for value in values if value is not None]
    if not clean:
        return None
    return int(round(sum(clean) / len(clean)))


def average_similarity_vectors(vectors: list[list[float]]) -> list[float]:
    arrays = [
        np.asarray(vector, dtype=float).ravel()
        for vector in vectors
        if vector is not None and len(vector) > 0
    ]
    if not arrays:
        return [0.0]

    shared_length = min(array.size for array in arrays)
    if shared_length <= 0:
        return [0.0]

    stacked = np.stack([array[:shared_length] for array in arrays], axis=0)
    mean_vector = stacked.mean(axis=0)
    norm = np.linalg.norm(mean_vector)
    if norm > 0:
        mean_vector = mean_vector / norm
    return mean_vector.astype(float).tolist()


def first_existing_path(paths: list[str]) -> str:
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.exists():
            return str(path)
    return ""


def copy2_if_needed(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    try:
        if src.resolve() == dst.resolve():
            return False
    except Exception:
        pass
    html_review.shutil.copy2(src, dst)
    return True


def resolve_analyzer_folder(output_folder: Path) -> Path:
    direct = output_folder / "sorting_analyzer_analysis.zarr"
    if direct.is_dir():
        return direct

    matches = sorted(
        path
        for path in output_folder.glob("sorting_analyzer_analysis.zarr")
        if path.is_dir()
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find sorting_analyzer_analysis.zarr under {output_folder}"
    )


def load_output_context(output_folder: Path, cache: dict[str, dict]) -> dict:
    cache_key = str(output_folder.resolve())
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    analyzer_folder = resolve_analyzer_folder(output_folder)
    analyzer = si.load_sorting_analyzer(
        folder=analyzer_folder,
        format="zarr",
        load_extensions=True,
    )
    ensure_required_extensions(analyzer)
    context = {
        "analyzer_folder": str(analyzer_folder),
        "analyzer": analyzer,
        "metrics_lookup": build_metrics_lookup(analyzer),
        "unit_channel_mapping": load_unit_channel_mapping(output_folder),
    }
    cache[cache_key] = context
    return context


def load_member_snapshot(member_payload: dict, cache: dict[str, dict]) -> dict:
    output_folder = Path(str(member_payload.get("output_folder") or "")).resolve()
    if not output_folder.exists():
        raise FileNotFoundError(f"Missing output folder for exported member: {output_folder}")

    unit_id = int(member_payload["unit_id"])
    context = load_output_context(output_folder, cache)
    analyzer = context["analyzer"]
    metadata = infer_unit_channel_metadata(
        analyzer,
        unit_id,
        mapping_row=context["unit_channel_mapping"].get(unit_id),
    )
    metrics = context["metrics_lookup"].get(unit_id, {})
    waveform_vector = get_waveform_vector(analyzer, unit_id).tolist()
    autocorrelogram_vector = get_autocorrelogram_vector(analyzer, unit_id).tolist()
    trough_to_peak_duration_ms = get_trough_to_peak_duration_ms(analyzer, unit_id)
    waveform_image_path = find_unit_summary_image(output_folder, unit_id)

    return {
        "output_folder": str(output_folder),
        "analyzer_folder": context["analyzer_folder"],
        "unit_id": unit_id,
        "shank_id": int(metadata["shank_id"]),
        "local_channel_on_shank": int(metadata["local_channel_on_shank"]),
        "sg_channel": int(metadata["sg_channel"]),
        "amplitude_median": safe_float(metrics.get("amplitude_median")),
        "firing_rate": safe_float(metrics.get("firing_rate")),
        "isi_violations_ratio": safe_float(metrics.get("isi_violations_ratio")),
        "snr": safe_float(metrics.get("snr")),
        "num_spikes": safe_int(metrics.get("num_spikes")),
        "waveform_similarity_vector": waveform_vector,
        "autocorrelogram_similarity_vector": autocorrelogram_vector,
        "trough_to_peak_duration_ms": trough_to_peak_duration_ms,
        "waveform_image_path": str(waveform_image_path) if waveform_image_path else "",
    }


def build_day_group_unit(
    *,
    session_name: str,
    session_index: int,
    synthetic_unit_id: int,
    group_payload: dict,
    page_scope: dict,
    export_manifest_path: Path,
    cache: dict[str, dict],
) -> UnitSummary:
    member_snapshots: list[dict] = []
    member_errors: list[str] = []
    for member_payload in group_payload.get("members", []):
        try:
            member_snapshots.append(load_member_snapshot(member_payload, cache))
        except Exception as exc:
            member_errors.append(str(exc))

    shank_id = safe_int(page_scope.get("shank_id"))
    sg_channel = safe_int(page_scope.get("sg_channel"))

    if member_snapshots:
        representative = member_snapshots[0]
        shank_id = representative["shank_id"] if shank_id is None else shank_id
        sg_channel = representative["sg_channel"] if sg_channel is None else sg_channel
        local_channel_on_shank = int(representative["local_channel_on_shank"])
        output_folder = str(representative["output_folder"])
        analyzer_folder = str(representative["analyzer_folder"])
        amplitude_median = average_scalar([item["amplitude_median"] for item in member_snapshots])
        firing_rate = average_scalar([item["firing_rate"] for item in member_snapshots])
        isi_violations_ratio = average_scalar([item["isi_violations_ratio"] for item in member_snapshots])
        snr = average_scalar([item["snr"] for item in member_snapshots])
        num_spikes = average_int([item["num_spikes"] for item in member_snapshots])
        waveform_similarity_vector = average_similarity_vectors(
            [item["waveform_similarity_vector"] for item in member_snapshots]
        )
        autocorrelogram_similarity_vector = average_similarity_vectors(
            [item["autocorrelogram_similarity_vector"] for item in member_snapshots]
        )
        trough_to_peak_duration_ms = average_scalar(
            [item["trough_to_peak_duration_ms"] for item in member_snapshots]
        )
        waveform_image_path = first_existing_path(
            [*group_payload.get("images", []), representative["waveform_image_path"]]
        )
    else:
        if shank_id is None or sg_channel is None:
            raise ValueError(
                "Exported group is missing page scope and no member units could be loaded "
                f"for session {session_name}, synthetic unit {synthetic_unit_id}. "
                f"Member load errors: {member_errors}"
            )
        local_channel_on_shank = safe_int(group_payload.get("local_channel_on_shank"))
        if local_channel_on_shank is None:
            local_channel_on_shank = int(sg_channel)
        output_folder = ""
        analyzer_folder = ""
        amplitude_median = None
        firing_rate = None
        isi_violations_ratio = None
        snr = None
        num_spikes = None
        waveform_similarity_vector = [0.0]
        autocorrelogram_similarity_vector = [0.0]
        trough_to_peak_duration_ms = None
        waveform_image_path = first_existing_path(group_payload.get("images", []))

    unit = UnitSummary(
        session_name=session_name,
        session_index=session_index,
        analyzer_folder=analyzer_folder,
        output_folder=output_folder,
        unit_id=int(synthetic_unit_id),
        shank_id=int(shank_id),
        local_channel_on_shank=int(local_channel_on_shank),
        sg_channel=int(sg_channel),
        amplitude_median=amplitude_median,
        firing_rate=firing_rate,
        isi_violations_ratio=isi_violations_ratio,
        snr=snr,
        num_spikes=num_spikes,
        waveform_similarity_vector=waveform_similarity_vector,
        autocorrelogram_similarity_vector=autocorrelogram_similarity_vector,
        trough_to_peak_duration_ms=trough_to_peak_duration_ms,
        waveform_image_path=waveform_image_path,
    )
    setattr(unit, "_source_members", [dict(member) for member in group_payload.get("members", [])])
    setattr(unit, "_source_images", [str(item) for item in group_payload.get("images", []) if item])
    setattr(unit, "_source_page_scope", dict(page_scope))
    setattr(unit, "_source_export_manifest_path", str(export_manifest_path))
    setattr(unit, "_source_group_key", str(group_payload.get("final_group_key", "") or ""))
    return unit


def load_all_days_from_exports(
    input_root: Path,
    progress_callback=None,
) -> tuple[list[SessionSummary], dict[int, dict[str, PageSummary]], Path]:
    selection_glob = sorted(input_root.glob(f"{DAY_SUMMARY_FOLDER_NAME}*/selected_day_folders.txt"))
    selection_file = selection_glob[0] if selection_glob else input_root / DAY_SUMMARY_FOLDER_NAME / "selected_day_folders.txt"
    legacy_selection_file = input_root / "selected_day_folders.txt"
    if input_root.is_dir() and selection_file.exists():
        input_roots = [
            Path(line.strip())
            for line in selection_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif input_root.is_dir() and legacy_selection_file.exists():
        input_roots = [
            Path(line.strip())
            for line in legacy_selection_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        input_roots = [input_root]

    day_roots = discover_day_sorting_roots(input_roots)
    common_root = Path(os.path.commonpath([str(day_root) for day_root in day_roots]))
    summary_root = common_root / build_day_summary_folder_name(day_roots)
    summary_root.mkdir(parents=True, exist_ok=True)

    if progress_callback is not None:
        progress_callback(
            f"Found {len(day_roots)} day folder(s). Loading exported per-day alignment groups..."
        )

    sessions: list[SessionSummary] = []
    cache: dict[str, dict] = {}
    synthetic_unit_ids: dict[int, int] = defaultdict(int)

    for session_index, day_root in enumerate(day_roots):
        session_name = parse_day_code_from_sorting_root(day_root) or day_root.name
        session_summary = SessionSummary(
            session_name=session_name,
            session_index=session_index,
            output_folder=str(day_root),
            analyzer_folder="",
        )
        export_manifest_paths = sorted(
            day_root.glob(f"sh*/{DEFAULT_EXPORT_FOLDER_NAME}/export_summary_sg_*.json")
        )
        if progress_callback is not None:
            progress_callback(
                f"Loading day {session_index + 1}/{len(day_roots)}: {session_name} "
                f"({len(export_manifest_paths)} exported page(s))"
            )

        for export_manifest_path in export_manifest_paths:
            payload = json.loads(export_manifest_path.read_text(encoding="utf-8"))
            page_scope = payload.get("page_scope") or {}
            for group_payload in payload.get("cross_session_alignment_groups", []):
                synthetic_unit_ids[session_index] += 1
                session_summary.units.append(
                    build_day_group_unit(
                        session_name=session_name,
                        session_index=session_index,
                        synthetic_unit_id=synthetic_unit_ids[session_index],
                        group_payload=group_payload,
                        page_scope=page_scope,
                        export_manifest_path=export_manifest_path,
                        cache=cache,
                    )
                )

        sessions.append(session_summary)

    pages_by_shank: dict[int, dict[str, PageSummary]] = defaultdict(dict)
    shank_to_channels: dict[int, set[int]] = defaultdict(set)
    for session in sessions:
        for unit in session.units:
            shank_to_channels[int(unit.shank_id)].add(int(unit.sg_channel))

    if not shank_to_channels:
        raise FileNotFoundError(
            "No exported day-level inputs were found. Run Alignment_html.py page exports first so "
            "each day has files like 'sh*/units_alignment_summary/export_summary_sg_*.json'."
        )

    for shank_id, sg_channels in sorted(shank_to_channels.items()):
        for sg_channel in sorted(sg_channels):
            aligned_sessions: list[SessionSummary] = []
            for session in sessions:
                filtered_units = [
                    unit
                    for unit in session.units
                    if int(unit.shank_id) == int(shank_id)
                    and int(unit.sg_channel) == int(sg_channel)
                ]
                aligned_sessions.append(
                    SessionSummary(
                        session_name=session.session_name,
                        session_index=session.session_index,
                        output_folder=session.output_folder,
                        analyzer_folder=session.analyzer_folder,
                        units=filtered_units,
                    )
                )
            page = PageSummary(
                shank_id=int(shank_id),
                sg_channel=int(sg_channel),
                sessions=aligned_sessions,
            )
            pages_by_shank[int(shank_id)][page.page_id] = page

    return sessions, dict(pages_by_shank), summary_root


def build_days_html_shell() -> str:
    shell = _BASE_BUILD_HTML_SHELL()
    shell = shell.replace(
        "</style>",
        "    .source-preview-stack { display:grid; gap:8px; margin-top:8px; }\n"
        "    .source-preview-stack img { display:block; }\n"
        "    .mini-note { color:var(--muted); font-size:0.82rem; margin-top:6px; }\n"
        "</style>",
    )
    shell = shell.replace(
        '<button data-command="align">Align</button><button data-command="unalign">Unalign</button><button data-command="merge">Merge</button><button data-command="unmerge">Unmerge</button><button data-command="noise">Noise</button><button data-command="clear_noise">Clear Noise</button><button data-command="similarity">Similarity</button>',
        '<button data-command="align">Align</button><button data-command="merge_overlap">Merge Overlap</button><button data-command="unalign">Unalign</button><button data-command="noise">Noise</button><button data-command="clear_noise">Clear Noise</button><button data-command="similarity">Similarity</button>',
    )
    shell = shell.replace(
        'placeholder="align r2 u11&#10;merge r4&#10;similarity u1 u2"',
        'placeholder="align r2 u11&#10;merge_overlap u1 u2&#10;similarity u1 u2"',
    )
    shell = shell.replace(
        'placeholder="session, unit, align, merge"',
        'placeholder="day, unit, align"',
    )
    shell = shell.replace(
        "    loadState().catch((err) => setLog(err.message));",
        "    function renderUnitCard(unit) {\n"
        "      const tags = [];\n"
        "      if (unit.align_group) tags.push(`<span class=\"tag align\">align=${unit.align_group}</span>`);\n"
        "      if (unit.is_discarded) tags.push('<span class=\"tag discarded\">discarded</span>');\n"
        "      if (unit.is_noise && !unit.is_discarded) tags.push('<span class=\"tag noise\">noise</span>');\n"
        "      if (unit.source_session_count != null) tags.push(`<span class=\"tag\">${unit.source_session_count} src session(s)</span>`);\n"
        "      const zoomImage = unit.source_preview_images && unit.source_preview_images.length ? unit.source_preview_images[0] : unit.waveform_image_path;\n"
        "      const zoomLabel = `${unit.alias} | ${unit.session_name} | u${unit.unit_id} | sh${unit.shank_id} | sg${unit.sg_channel}`;\n"
        "      const previewImages = (unit.source_preview_images && unit.source_preview_images.length ? unit.source_preview_images : [unit.waveform_image_path]).slice(0, 3);\n"
        "      const previewHtml = `<div class=\"source-preview-stack\">${previewImages.map((imageSrc, index) => `<img loading=\"lazy\" src=\"${imageSrc}\" alt=\"${unit.session_name} source preview ${index + 1}\">`).join(\"\")}</div>`;\n"
        "      const extraNote = unit.source_session_count >= 3 ? \"Showing 3 within-day source summaries.\" : `Showing ${previewImages.length} within-day source summar${previewImages.length === 1 ? \"y\" : \"ies\"}.`;\n"
        "      return `<article class=\"unit-card\" data-image-src=\"${zoomImage}\" data-image-label=\"${zoomLabel}\"><div class=\"unit-head\"><div><div class=\"unit-name\">${unit.alias} | ${unit.session_name} | u${unit.unit_id}</div><div class=\"muted\">sh${unit.shank_id} | sg${unit.sg_channel}</div></div></div><div class=\"unit-tags\">${tags.join(\"\")}</div><div class=\"metrics\"><div>FR: ${metric(unit.firing_rate)} Hz</div><div>SNR: ${metric(unit.snr)}</div><div><strong>Amp: ${metric(unit.amplitude_median)}</strong></div><div>ISI: ${metric(unit.isi_violations_ratio)}</div><div>Spikes: ${unit.num_spikes ?? \"nan\"}</div><div>Within-day sessions: ${unit.source_session_count ?? 0}</div></div>${previewHtml}<div class=\"mini-note\">${extraNote}</div></article>`;\n"
        "    }\n"
        "    loadState().catch((err) => setLog(err.message));",
    )
    return shell


class AlignmentDaysState(html_review.AlignmentState):
    def __init__(self, root_folder: Path, progress_callback=None):
        self.root_folder = root_folder.resolve()
        self.sessions, self.pages_by_shank, _cache_folder = load_all_days_from_exports(
            self.root_folder,
            progress_callback=progress_callback,
        )
        self.summary_root = _cache_folder
        self.summary_root.mkdir(parents=True, exist_ok=True)
        self.cache_folder = self.summary_root / "_cache"
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self._lock = html_review.threading.RLock()
        self._undo_stack: list[dict[str, dict]] = []
        self._stable_unit_aliases: dict[str, dict[str, str]] = {}
        self._stable_next_alias_index: dict[str, int] = {}
        self.discovered_shank_folder_ids = sorted(str(shank_id) for shank_id in self.pages_by_shank.keys())
        self.loaded_shank_ids = list(self.discovered_shank_folder_ids)
        self.empty_shank_folder_ids: list[str] = []
        if progress_callback is not None:
            progress_callback("Applying saved day-alignment manifest if available...")
        self.apply_manifest_if_available()
        if progress_callback is not None:
            progress_callback("Startup state ready.")

    def snapshot(self) -> dict[str, dict]:
        snapshot = super().snapshot()
        for unit in self._iter_all_units():
            snapshot[day_unit_record_key(unit)]["source_merge_overrides"] = dict(
                getattr(unit, "_source_merge_overrides", {})
            )
        return snapshot

    def restore_snapshot(self, snapshot: dict[str, dict]) -> None:
        super().restore_snapshot(snapshot)
        for unit in self._iter_all_units():
            state = snapshot.get(day_unit_record_key(unit), {})
            setattr(unit, "_source_merge_overrides", dict(state.get("source_merge_overrides", {})))

    def apply_manifest_if_available(self) -> None:
        manifest_path = self.summary_root / DAY_MANIFEST_NAME
        if not manifest_path.exists():
            return

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return

        manifest_units: dict[str, dict] = {}
        for session_payload in payload.get("sessions", []):
            for unit_payload in session_payload.get("units", []):
                for key in self._manifest_lookup_keys_from_payload(unit_payload):
                    manifest_units[key] = unit_payload

        for unit in self._iter_all_units():
            saved = {}
            for key in self._manifest_lookup_keys_for_unit(unit):
                if key in manifest_units:
                    saved = manifest_units[key]
                    break
            unit.merge_group = ""
            unit.align_group = str(saved.get("align_group", "") or "")
            unit.exclude_from_auto_align = bool(saved.get("exclude_from_auto_align", False))
            unit.is_noise = bool(saved.get("is_noise", False))
            unit.is_discarded = bool(saved.get("is_discarded", False))

    def sync_auto_merge_groups(self) -> None:
        for unit in self._iter_all_units():
            unit.merge_group = ""

    def final_group_key_for_unit(
        self,
        unit: UnitSummary,
        auto_align_lookup: dict[str, str] | None = None,
    ) -> str:
        if unit.align_group:
            return f"sh{unit.shank_id}_sg{unit.sg_channel}__align__{html_review.sanitize_token(unit.align_group)}"
        if auto_align_lookup is not None:
            auto_group_name = auto_align_lookup.get(day_unit_record_key(unit))
            if auto_group_name:
                return auto_group_name
        return f"s{unit.session_index:03d}_u{unit.unit_id}"

    def discard_group_key_for_unit(self, unit: UnitSummary) -> str:
        if unit.align_group:
            return f"discarded__sh{unit.shank_id}_sg{unit.sg_channel}__align__{html_review.sanitize_token(unit.align_group)}"
        return f"discarded__s{unit.session_index:03d}_u{unit.unit_id}"

    def run_page_command(
        self,
        page: PageSummary,
        command_name: str,
        alias_tokens: list[str],
        *,
        unit_alias_map: dict[str, UnitSummary] | None = None,
        row_alias_map: dict[str, list[UnitSummary]] | None = None,
    ) -> tuple[str, bool]:
        normalized = command_name.strip().lower()
        if normalized in {"merge", "unmerge"}:
            raise ValueError("Alignment_days.py does not support merge or unmerge. Use align/unalign only.")

        if normalized in {"similarity", "similarities", "compare"}:
            return super().run_page_command(
                page,
                command_name,
                alias_tokens,
                unit_alias_map=unit_alias_map,
                row_alias_map=row_alias_map,
            )

        units = self.resolve_command_units(
            page,
            alias_tokens,
            unit_alias_map=unit_alias_map,
            row_alias_map=row_alias_map,
        )
        if normalized == "align":
            if len(units) < 2:
                raise ValueError("align needs at least two units.")
            self._validate_no_overlapping_source_sessions(units)
            group_name, unit_count = self.assign_units_to_group(
                attr_name="align_group",
                base_name=f"align_sh{page.shank_id}_sg{page.sg_channel}",
                scope_tag=f"sh{page.shank_id}_sg{page.sg_channel}",
                selected_units=units,
                expand_existing_members=False,
            )
            for unit in units:
                unit.merge_group = ""
            return f"align {group_name} on {unit_count} unit(s)", True
        if normalized in {"merge_overlap", "mergeoverlap", "resolve_overlap", "resolveoverlap"}:
            if len(units) < 2:
                raise ValueError("merge_overlap needs at least two units.")
            if not self._collect_overlapping_source_sessions(units):
                raise ValueError(
                    "merge_overlap needs selected day-level units that share at least one original source session. "
                    "Use align for non-overlapping units."
                )
            group_name, unit_count = self.assign_units_to_group(
                attr_name="align_group",
                base_name=f"align_sh{page.shank_id}_sg{page.sg_channel}",
                scope_tag=f"sh{page.shank_id}_sg{page.sg_channel}",
                selected_units=units,
                expand_existing_members=False,
            )
            merged_session_count, merged_member_count = self._apply_overlap_merge_overrides(
                page=page,
                units=units,
                align_group_name=group_name,
            )
            for unit in units:
                unit.merge_group = ""
            return (
                f"merge_overlap {group_name} on {unit_count} unit(s); "
                f"resolved {merged_session_count} overlapping source session(s) across {merged_member_count} source member(s)",
                True,
            )
        if normalized == "unalign":
            cleared_count = 0
            for unit in units:
                if unit.align_group.strip():
                    cleared_count += 1
                unit.align_group = ""
                unit.merge_group = ""
                setattr(unit, "_source_merge_overrides", {})
                unit.exclude_from_auto_align = True
            return f"cleared alignment on {cleared_count} unit(s)", True
        if normalized == "noise":
            for unit in units:
                unit.is_noise = True
                unit.merge_group = ""
            return f"marked {len(units)} unit(s) as noise", True
        if normalized in {"clear_noise", "cleannoise", "denoise"}:
            for unit in units:
                unit.is_noise = False
                unit.merge_group = ""
            return f"cleared noise on {len(units)} unit(s)", True
        raise ValueError(f"Unknown command: {command_name}")

    def save_manifest_state(self) -> Path:
        manifest_path = self.summary_root / DAY_MANIFEST_NAME
        payload = {
            "output_root": str(self.root_folder),
            "sessions": [
                {
                    "session_name": session.session_name,
                    "session_index": session.session_index,
                    "output_folder": session.output_folder,
                    "analyzer_folder": session.analyzer_folder,
                    "units": [asdict(unit) for unit in session.units],
                }
                for session in self.sessions
            ],
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._sync_all_source_exports()
        self._sync_all_source_manifests()
        return manifest_path

    def save_manifest_state_for_all_pages(self) -> dict:
        manifest_path = self.save_manifest_state()
        return {
            "root_manifest_path": str(manifest_path),
            "shank_manifest_paths": [],
        }

    def save_manifest_state_for_page(self, shank_id: int, page_id: str) -> tuple[Path, Path]:
        _page = self.get_page(shank_id, page_id)
        manifest_path = self.save_manifest_state()
        return manifest_path, manifest_path

    def _source_members_for_unit(self, unit: UnitSummary) -> list[dict]:
        merge_overrides = dict(getattr(unit, "_source_merge_overrides", {}))
        members: list[dict] = []
        for item in getattr(unit, "_source_members", []):
            payload = dict(item)
            session_key = self._source_member_session_key(payload, unit)
            override_merge_group = merge_overrides.get(session_key, "")
            if override_merge_group:
                payload["merge_group"] = override_merge_group
            members.append(payload)
        return members

    def _source_images_for_unit(self, unit: UnitSummary) -> list[str]:
        images = [str(item) for item in getattr(unit, "_source_images", []) if item]
        if not images and unit.waveform_image_path:
            images.append(str(unit.waveform_image_path))
        return images

    def _source_member_session_key(self, member: dict, unit: UnitSummary | None = None) -> str:
        output_folder = str(member.get("output_folder", "") or "").strip()
        session_name = str(member.get("session_name", "") or "").strip()
        session_index = safe_int(member.get("session_index"))
        if output_folder:
            return f"output::{output_folder}"
        if session_name:
            if session_index is not None:
                return f"session::{session_name}::{int(session_index)}"
            return f"session::{session_name}"
        if unit is not None:
            return f"day::{unit.session_name}::{int(unit.session_index)}"
        return "day::unknown"

    def _source_session_names_for_unit(self, unit: UnitSummary) -> list[str]:
        source_members = self._source_members_for_unit(unit)
        session_names: list[str] = []
        seen: set[str] = set()
        for member in source_members:
            session_name = str(member.get("session_name", "") or "")
            if not session_name or session_name in seen:
                continue
            seen.add(session_name)
            session_names.append(session_name)
        if not session_names and unit.session_name:
            session_names.append(str(unit.session_name))
        return session_names

    def _source_session_keys_for_unit(self, unit: UnitSummary) -> set[str]:
        source_members = self._source_members_for_unit(unit)
        session_keys: set[str] = set()
        for member in source_members:
            output_folder = str(member.get("output_folder", "") or "").strip()
            session_name = str(member.get("session_name", "") or "").strip()
            session_index = safe_int(member.get("session_index"))
            if output_folder:
                session_keys.add(f"output::{output_folder}")
            elif session_name:
                if session_index is not None:
                    session_keys.add(f"session::{session_name}::{int(session_index)}")
                else:
                    session_keys.add(f"session::{session_name}")
        if not session_keys:
            session_keys.add(f"day::{unit.session_name}::{int(unit.session_index)}")
        return session_keys

    def _source_members_by_session_key(self, unit: UnitSummary) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        source_members = self._source_members_for_unit(unit)
        for member in source_members:
            session_key = self._source_member_session_key(member, unit)
            grouped[session_key].append(dict(member))
        if not grouped:
            grouped[f"day::{unit.session_name}::{int(unit.session_index)}"].append(
                {
                    "session_name": unit.session_name,
                    "session_index": int(unit.session_index),
                    "unit_id": int(unit.unit_id),
                    "output_folder": str(unit.output_folder or ""),
                }
            )
        return grouped

    def _collect_overlapping_source_sessions(
        self,
        units: list[UnitSummary],
    ) -> dict[str, list[tuple[UnitSummary, list[dict]]]]:
        grouped: dict[str, list[tuple[UnitSummary, list[dict]]]] = defaultdict(list)
        for unit in units:
            for session_key, members in self._source_members_by_session_key(unit).items():
                grouped[session_key].append((unit, members))
        return {
            session_key: entries
            for session_key, entries in grouped.items()
            if len(entries) >= 2
        }

    def _set_source_merge_override(self, unit: UnitSummary, session_key: str, merge_group: str) -> None:
        overrides = dict(getattr(unit, "_source_merge_overrides", {}))
        overrides[session_key] = merge_group
        setattr(unit, "_source_merge_overrides", overrides)

    def _apply_overlap_merge_overrides(
        self,
        *,
        page: PageSummary,
        units: list[UnitSummary],
        align_group_name: str,
    ) -> tuple[int, int]:
        overlapping_sessions = self._collect_overlapping_source_sessions(units)
        if not overlapping_sessions:
            raise ValueError(
                "merge_overlap needs selected day-level units that share at least one original source session."
            )

        merged_session_count = 0
        merged_member_count = 0
        for session_index, (session_key, entries) in enumerate(sorted(overlapping_sessions.items()), start=1):
            merge_group_name = (
                f"merge_overlap_sh{page.shank_id}_sg{page.sg_channel}_"
                f"{html_review.sanitize_token(align_group_name)}_{session_index:02d}"
            )
            merged_session_count += 1
            for unit, members in entries:
                self._set_source_merge_override(unit, session_key, merge_group_name)
                merged_member_count += len(members)

        return merged_session_count, merged_member_count

    def _format_overlap_members(self, members: list[dict]) -> str:
        labels: list[str] = []
        for member in members:
            session_name = str(member.get("session_name", "") or "")
            unit_id = safe_int(member.get("unit_id"))
            if session_name and unit_id is not None:
                labels.append(f"{session_name}:u{int(unit_id)}")
            elif unit_id is not None:
                labels.append(f"u{int(unit_id)}")
        return ", ".join(labels) if labels else "<unknown units>"

    def _validate_no_overlapping_source_sessions(self, units: list[UnitSummary]) -> None:
        overlaps: list[tuple[UnitSummary, UnitSummary, str, str, str]] = []
        for session_key, entries in self._collect_overlapping_source_sessions(units).items():
            left, left_members = entries[0]
            for right, right_members in entries[1:]:
                overlaps.append(
                    (
                        left,
                        right,
                        session_key,
                        self._format_overlap_members(left_members),
                        self._format_overlap_members(right_members),
                    )
                )

        if not overlaps:
            return

        details = [
            (
                f"{left.session_name}:u{left.unit_id} overlaps {right.session_name}:u{right.unit_id} "
                f"via {session_key} | left source units: {left_members} | right source units: {right_members}"
            )
            for left, right, session_key, left_members, right_members in overlaps[:5]
        ]
        raise ValueError(
            "Cannot align day-level units that already share original source sessions. "
            "This would create overlapping same-session alignments in the synced source results. "
            f"Examples: {'; '.join(details)}"
        )

    def _image_url(self, image_path: str) -> str:
        if not image_path:
            return ""
        try:
            return f"/image?path={html_review.quote(str(Path(image_path).resolve()))}"
        except Exception:
            return ""

    def _build_day_unit_payload(self, unit: UnitSummary, alias: str, *, is_discarded: bool) -> dict:
        source_session_names = self._source_session_names_for_unit(unit)
        source_preview_images: list[str] = []
        for path in self._source_images_for_unit(unit)[:3]:
            image_url = self._image_url(path)
            if image_url:
                source_preview_images.append(image_url)
        waveform_image_url = self._image_url(str(unit.waveform_image_path))
        if not source_preview_images and waveform_image_url:
            source_preview_images = [waveform_image_url]
        return {
            "alias": alias,
            "session_name": unit.session_name,
            "session_index": int(unit.session_index),
            "unit_id": int(unit.unit_id),
            "shank_id": int(unit.shank_id),
            "sg_channel": int(unit.sg_channel),
            "local_channel_on_shank": int(unit.local_channel_on_shank),
            "amplitude_median": unit.amplitude_median,
            "firing_rate": unit.firing_rate,
            "isi_violations_ratio": unit.isi_violations_ratio,
            "snr": unit.snr,
            "num_spikes": unit.num_spikes,
            "merge_group": "",
            "align_group": unit.align_group,
            "exclude_from_auto_align": bool(unit.exclude_from_auto_align),
            "is_discarded": bool(is_discarded),
            "is_noise": bool(unit.is_noise),
            "waveform_image_path": waveform_image_url,
            "source_session_count": len(source_session_names),
            "source_sessions_present": source_session_names,
            "source_preview_images": source_preview_images,
        }

    def build_page_payload(self, page: PageSummary) -> dict:
        display_rows = self.build_page_display_rows_local(page)
        page_summary = self.summarize_page(page)
        unit_alias_map, row_alias_map, alias_by_unit_key = self._build_alias_maps(page, display_rows)
        rows_payload: list[dict] = []

        for row_index, row_units in enumerate(display_rows, start=1):
            flat_units = [
                unit
                for session in page.sessions
                for unit in row_units.get(session.session_index, [])
            ]
            sessions_present = [
                session.session_name
                for session in page.sessions
                if row_units.get(session.session_index, [])
            ]
            units_payload = [
                self._build_day_unit_payload(
                    unit,
                    alias_by_unit_key.get(day_unit_record_key(unit), ""),
                    is_discarded=bool(unit.is_discarded),
                )
                for unit in flat_units
            ]
            rows_payload.append(
                {
                    "row_index": row_index,
                    "row_alias": f"r{row_index}",
                    "row_kind": self.row_kind_for_units(flat_units),
                    "sessions_present": sessions_present,
                    "num_units": len(flat_units),
                    "units": units_payload,
                }
            )

        return {
            "page_id": page.page_id,
            "page_type": "channel",
            "title": page.title,
            "shank_id": int(page.shank_id),
            "sg_channel": int(page.sg_channel),
            "summary": page_summary,
            "rows": rows_payload,
            "available_unit_aliases": self._sorted_aliases(list(unit_alias_map)),
            "available_row_aliases": self._sorted_aliases(list(row_alias_map)),
        }

    def build_discarded_page_payload(self) -> dict:
        discarded_units = [unit for unit in self._iter_all_units() if unit.is_discarded]
        groups: dict[str, list[UnitSummary]] = {}
        for unit in sorted(
            discarded_units,
            key=lambda item: (item.shank_id, item.session_index, item.sg_channel, item.unit_id),
        ):
            groups.setdefault(self.discard_group_key_for_unit(unit), []).append(unit)
        alias_by_unit_key = self._get_stable_aliases_for_units(
            cache_key="discarded_all",
            ordered_units=[unit for units in groups.values() for unit in units],
        )

        rows_payload: list[dict] = []
        for row_index, units in enumerate(groups.values(), start=1):
            sessions_present: list[str] = []
            seen_sessions: set[str] = set()
            units_payload = []
            for unit in units:
                if unit.session_name not in seen_sessions:
                    sessions_present.append(unit.session_name)
                    seen_sessions.add(unit.session_name)
                units_payload.append(
                    self._build_day_unit_payload(
                        unit,
                        alias_by_unit_key[day_unit_record_key(unit)],
                        is_discarded=True,
                    )
                )
            rows_payload.append(
                {
                    "row_index": row_index,
                    "row_alias": f"r{row_index}",
                    "row_kind": "discarded",
                    "sessions_present": sessions_present,
                    "num_units": len(units_payload),
                    "units": units_payload,
                }
            )

        return {
            "page_id": "__discarded_all__",
            "page_type": "discarded",
            "title": "Discarded Units Across All Shanks",
            "shank_id": -1,
            "sg_channel": None,
            "summary": {
                "session_counts": [],
                "total_units": len(discarded_units),
                "total_discarded_units": len(discarded_units),
            },
            "rows": rows_payload,
            "available_unit_aliases": self._sorted_aliases(list(alias_by_unit_key.values())),
            "available_row_aliases": [f"r{i}" for i in range(1, len(rows_payload) + 1)],
        }

    def _collect_session_page_units(
        self,
        session: SessionSummary,
        shank_id: int,
        sg_channel: int,
    ) -> list[UnitSummary]:
        return [
            unit
            for unit in session.units
            if int(unit.shank_id) == int(shank_id) and int(unit.sg_channel) == int(sg_channel)
        ]

    def _flatten_source_members(
        self,
        units: list[UnitSummary],
        *,
        propagated_align_group: str | None = None,
    ) -> list[dict]:
        flattened: list[dict] = []
        for unit in units:
            source_members = self._source_members_for_unit(unit)
            if source_members:
                for member in source_members:
                    payload = dict(member)
                    payload["session_name"] = str(payload.get("session_name", "") or "")
                    payload["session_index"] = safe_int(payload.get("session_index"))
                    payload["unit_id"] = int(payload.get("unit_id"))
                    payload["merge_group"] = str(payload.get("merge_group", "") or "")
                    payload["align_group"] = str(
                        propagated_align_group
                        if propagated_align_group is not None
                        else payload.get("align_group", "") or ""
                    )
                    payload["output_folder"] = str(payload.get("output_folder", "") or "")
                    flattened.append(payload)
                continue
            flattened.append(
                {
                    "session_name": str(unit.session_name),
                    "session_index": int(unit.session_index),
                    "unit_id": int(unit.unit_id),
                    "merge_group": str(unit.merge_group or ""),
                    "align_group": str(propagated_align_group if propagated_align_group is not None else unit.align_group or ""),
                    "output_folder": str(unit.output_folder or ""),
                }
            )
        return flattened

    def _build_original_page_groups(
        self,
        session: SessionSummary,
        shank_id: int,
        sg_channel: int,
    ) -> tuple[dict[str, list[UnitSummary]], dict[str, list[UnitSummary]], dict[str, list[UnitSummary]]]:
        page_units = self._collect_session_page_units(session, shank_id, sg_channel)
        same_day_align_groups: dict[str, list[UnitSummary]] = defaultdict(list)
        for unit in page_units:
            if unit.is_noise or unit.is_discarded:
                continue
            align_name = unit.align_group.strip()
            if align_name:
                same_day_align_groups[align_name].append(unit)

        final_groups: dict[str, list[UnitSummary]] = {}
        discarded_groups: dict[str, list[UnitSummary]] = {}
        noise_groups: dict[str, list[UnitSummary]] = {}
        for unit in sorted(page_units, key=lambda item: (item.session_index, item.unit_id)):
            if unit.is_discarded:
                discarded_groups.setdefault(f"discarded::{session.session_index}:{unit.unit_id}", []).append(unit)
                continue
            if unit.is_noise:
                noise_groups.setdefault(f"noise::{session.session_index}:{unit.unit_id}", []).append(unit)
                continue
            align_name = unit.align_group.strip()
            if align_name and len(same_day_align_groups.get(align_name, [])) >= 2:
                final_groups.setdefault(f"align::{html_review.sanitize_token(align_name)}", []).append(unit)
            else:
                final_groups[f"unit::{session.session_index}:{unit.unit_id}"] = [unit]
        return final_groups, discarded_groups, noise_groups

    def _build_source_summary_text(
        self,
        *,
        group_index: int,
        group_key: str,
        shank_id: int,
        sg_channel: int,
        local_channel_on_shank: int,
        member_units: list[dict],
    ) -> str:
        sessions_present: list[str] = []
        seen_sessions: set[str] = set()
        for member in member_units:
            session_name = str(member.get("session_name", "") or "")
            if session_name and session_name not in seen_sessions:
                seen_sessions.add(session_name)
                sessions_present.append(session_name)
        lines = [
            f"Final unit #{group_index}",
            f"Group key: {group_key}",
            f"Shank: {shank_id}",
            f"Channel: {local_channel_on_shank}",
            f"SG channel: {sg_channel}",
            f"Appears in {len(sessions_present)} session(s): {', '.join(sessions_present)}",
            f"Total member units: {len(member_units)}",
            "",
            "Members:",
        ]
        for member in member_units:
            lines.extend(
                [
                    f"- {member.get('session_name', '')} | unit {member.get('unit_id', '')}",
                    f"  merge_group={member.get('merge_group', '') or '<none>'}",
                    f"  align_group={member.get('align_group', '') or '<none>'}",
                    f"  output_folder={member.get('output_folder', '')}",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def _build_original_unique_unit_summary_row(
        self,
        *,
        group_index: int,
        group_key: str,
        group_folder: Path,
        summary_path: Path,
        copied_images: list[str],
        shank_id: int,
        sg_channel: int,
        local_channel_on_shank: int,
        member_units: list[dict],
    ) -> dict:
        session_names: list[str] = []
        seen_sessions: set[str] = set()
        members_by_session: dict[str, list[int]] = defaultdict(list)
        for member in member_units:
            session_name = str(member.get("session_name", "") or "")
            if session_name not in seen_sessions:
                seen_sessions.add(session_name)
                session_names.append(session_name)
            members_by_session[session_name].append(int(member.get("unit_id")))
        representative = member_units[0] if member_units else {}
        return {
            "final_unit_id": group_index,
            "final_unit_label": f"unit_{group_index:04d}",
            "final_group_key": group_key,
            "export_folder": str(group_folder),
            "summary_path": str(summary_path),
            "shank_id": int(shank_id),
            "channel": int(local_channel_on_shank),
            "sg_channel": int(sg_channel),
            "representative_session": str(representative.get("session_name", "") or ""),
            "representative_unit_id": int(representative.get("unit_id", 0) or 0),
            "num_sessions": len(session_names),
            "sessions_present": session_names,
            "session_members": [
                {"session_name": session_name, "unit_ids": sorted(member_unit_ids)}
                for session_name, member_unit_ids in members_by_session.items()
            ],
            "num_member_units": len(member_units),
            "member_units": [
                {
                    "session_name": str(member.get("session_name", "") or ""),
                    "session_index": safe_int(member.get("session_index")),
                    "unit_id": int(member.get("unit_id")),
                    "merge_group": str(member.get("merge_group", "") or ""),
                    "align_group": str(member.get("align_group", "") or ""),
                    "waveform_image_path": "",
                }
                for member in member_units
            ],
            "representative_waveform_image": copied_images[0] if copied_images else "",
            "waveform_images": copied_images,
        }

    def _build_original_discard_or_noise_row(
        self,
        *,
        status: str,
        group_key: str,
        shank_id: int,
        sg_channel: int,
        local_channel_on_shank: int,
        member_units: list[dict],
    ) -> dict:
        session_names: list[str] = []
        seen_sessions: set[str] = set()
        for member in member_units:
            session_name = str(member.get("session_name", "") or "")
            if session_name not in seen_sessions:
                seen_sessions.add(session_name)
                session_names.append(session_name)
        row = {
            "status": status,
            "shank_id": int(shank_id),
            "channel": int(local_channel_on_shank),
            "sg_channel": int(sg_channel),
            "num_sessions": len(session_names),
            "sessions_present": session_names,
            "num_member_units": len(member_units),
            "member_units": [
                {
                    "session_name": str(member.get("session_name", "") or ""),
                    "session_index": safe_int(member.get("session_index")),
                    "unit_id": int(member.get("unit_id")),
                    "merge_group": str(member.get("merge_group", "") or ""),
                    "align_group": str(member.get("align_group", "") or ""),
                    "waveform_image_path": "",
                }
                for member in member_units
            ],
        }
        if status == "discarded":
            row["discard_group_key"] = group_key
            row["discard_reason"] = "discarded in Alignment_days"
        else:
            row["noise_group_key"] = group_key
        return row

    def _write_original_export_for_session_page(
        self,
        session: SessionSummary,
        shank_id: int,
        sg_channel: int,
    ) -> dict | None:
        page_units = self._collect_session_page_units(session, shank_id, sg_channel)
        if not page_units:
            return None

        page_summary_root = Path(session.output_folder) / f"sh{int(shank_id)}" / DEFAULT_EXPORT_FOLDER_NAME
        page_summary_root.mkdir(parents=True, exist_ok=True)
        export_folder = page_summary_root / f"exported_units_sg_{int(sg_channel):03d}"
        export_folder.mkdir(parents=True, exist_ok=True)

        final_groups, discarded_groups, noise_groups = self._build_original_page_groups(
            session,
            shank_id,
            sg_channel,
        )
        unique_unit_rows: list[dict] = []
        discarded_unit_rows: list[dict] = []
        noise_unit_rows: list[dict] = []
        manifest_rows: list[dict] = []

        for group_index, (_group_sort_key, units) in enumerate(sorted(final_groups.items()), start=1):
            representative = units[0]
            propagated_align_group = representative.align_group.strip() if len(units) >= 2 else None
            member_units = self._flatten_source_members(
                units,
                propagated_align_group=propagated_align_group,
            )
            group_key = (
                f"sh{int(shank_id)}_sg{int(sg_channel)}__align__{html_review.sanitize_token(propagated_align_group)}"
                if propagated_align_group
                else str(getattr(representative, "_source_group_key", "") or f"s{session.session_index:03d}_u{representative.unit_id}")
            )
            group_folder = export_folder / f"unit_{group_index:04d}"
            group_folder.mkdir(parents=True, exist_ok=True)
            copied_images: list[str] = []
            seen_image_sources: set[str] = set()
            image_index = 1
            for unit in units:
                for image_path in self._source_images_for_unit(unit):
                    if not image_path or image_path in seen_image_sources:
                        continue
                    seen_image_sources.add(image_path)
                    src = Path(image_path)
                    dst = group_folder / f"waveform_{image_index:02d}_{unit.session_name.replace(' ', '_')}_u{unit.unit_id}{src.suffix or '.png'}"
                    if copy2_if_needed(src, dst):
                        copied_images.append(str(dst))
                        image_index += 1

            summary_path = group_folder / "summary.txt"
            summary_path.write_text(
                self._build_source_summary_text(
                    group_index=group_index,
                    group_key=group_key,
                    shank_id=int(shank_id),
                    sg_channel=int(sg_channel),
                    local_channel_on_shank=int(representative.local_channel_on_shank),
                    member_units=member_units,
                ),
                encoding="utf-8",
            )
            unique_unit_rows.append(
                self._build_original_unique_unit_summary_row(
                    group_index=group_index,
                    group_key=group_key,
                    group_folder=group_folder,
                    summary_path=summary_path,
                    copied_images=copied_images,
                    shank_id=int(shank_id),
                    sg_channel=int(sg_channel),
                    local_channel_on_shank=int(representative.local_channel_on_shank),
                    member_units=member_units,
                )
            )
            manifest_rows.append(
                {
                    "final_unit_id": group_index,
                    "final_group_key": group_key,
                    "export_folder": str(group_folder),
                    "representative_session": str(member_units[0].get("session_name", "") or session.session_name),
                    "representative_unit_id": int(member_units[0].get("unit_id", representative.unit_id)),
                    "shank_id": int(shank_id),
                    "local_channel_on_shank": int(representative.local_channel_on_shank),
                    "members": member_units,
                    "images": copied_images,
                }
            )

        for group_index, (_group_sort_key, units) in enumerate(sorted(discarded_groups.items()), start=1):
            representative = units[0]
            member_units = self._flatten_source_members(units)
            discarded_unit_rows.append(
                self._build_original_discard_or_noise_row(
                    status="discarded",
                    group_key=f"discarded__s{session.session_index:03d}_{group_index:04d}",
                    shank_id=int(shank_id),
                    sg_channel=int(sg_channel),
                    local_channel_on_shank=int(representative.local_channel_on_shank),
                    member_units=member_units,
                )
            )

        for group_index, (_group_sort_key, units) in enumerate(sorted(noise_groups.items()), start=1):
            representative = units[0]
            member_units = self._flatten_source_members(units)
            noise_unit_rows.append(
                self._build_original_discard_or_noise_row(
                    status="noise",
                    group_key=f"noise__s{session.session_index:03d}_{group_index:04d}",
                    shank_id=int(shank_id),
                    sg_channel=int(sg_channel),
                    local_channel_on_shank=int(representative.local_channel_on_shank),
                    member_units=member_units,
                )
            )

        unique_units_json_path = page_summary_root / f"unique_units_summary_sg_{int(sg_channel):03d}.json"
        unique_units_json_path.write_text(json.dumps(unique_unit_rows, indent=2), encoding="utf-8")
        unique_units_csv_path = page_summary_root / f"unique_units_summary_sg_{int(sg_channel):03d}.csv"
        self.write_unique_units_summary_csv(unique_units_csv_path, unique_unit_rows)

        discarded_units_json_path = page_summary_root / f"discarded_units_summary_sg_{int(sg_channel):03d}.json"
        discarded_units_json_path.write_text(json.dumps(discarded_unit_rows, indent=2), encoding="utf-8")
        discarded_units_csv_path = page_summary_root / f"discarded_units_summary_sg_{int(sg_channel):03d}.csv"
        self.write_discarded_units_summary_csv(discarded_units_csv_path, discarded_unit_rows)

        noise_units_json_path = page_summary_root / f"noise_units_summary_sg_{int(sg_channel):03d}.json"
        noise_units_json_path.write_text(json.dumps(noise_unit_rows, indent=2), encoding="utf-8")
        noise_units_csv_path = page_summary_root / f"noise_units_summary_sg_{int(sg_channel):03d}.csv"
        self.write_noise_units_summary_csv(noise_units_csv_path, noise_unit_rows)

        export_manifest_path = page_summary_root / f"export_summary_sg_{int(sg_channel):03d}.json"
        export_manifest_path.write_text(
            json.dumps(
                {
                    "output_root": str(session.output_folder),
                    "page_scope": {
                        "shank_id": int(shank_id),
                        "sg_channel": int(sg_channel),
                        "page_id": f"sg{int(sg_channel)}",
                    },
                    "unique_units_summary_json": str(unique_units_json_path),
                    "unique_units_summary_csv": str(unique_units_csv_path),
                    "discarded_units_summary_json": str(discarded_units_json_path),
                    "discarded_units_summary_csv": str(discarded_units_csv_path),
                    "noise_units_summary_json": str(noise_units_json_path),
                    "noise_units_summary_csv": str(noise_units_csv_path),
                    "cross_session_alignment_groups": manifest_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "session_name": session.session_name,
            "session_root": str(session.output_folder),
            "shank_id": int(shank_id),
            "sg_channel": int(sg_channel),
            "export_manifest_path": str(export_manifest_path),
            "num_unique_units": len(unique_unit_rows),
            "num_discarded_groups": len(discarded_unit_rows),
            "num_noise_groups": len(noise_unit_rows),
        }

    def _sync_source_exports_for_page(self, page: PageSummary) -> list[dict]:
        sync_results: list[dict] = []
        for session in page.sessions:
            result = self._write_original_export_for_session_page(
                session,
                shank_id=int(page.shank_id),
                sg_channel=int(page.sg_channel),
            )
            if result is not None:
                sync_results.append(result)
        return sync_results

    def _sync_all_source_exports(self) -> list[dict]:
        sync_results: list[dict] = []
        for shank_pages in self.pages_by_shank.values():
            for page in shank_pages.values():
                sync_results.extend(self._sync_source_exports_for_page(page))
        return sync_results

    def _source_unit_manifest_key(self, unit_payload: dict) -> str:
        output_folder = str(unit_payload.get("output_folder", "") or "")
        unit_id = safe_int(unit_payload.get("unit_id"))
        if output_folder and unit_id is not None:
            return f"output::{output_folder}::{int(unit_id)}"
        session_name = str(unit_payload.get("session_name", "") or "")
        shank_id = safe_int(unit_payload.get("shank_id"))
        sg_channel = safe_int(unit_payload.get("sg_channel"))
        if session_name and shank_id is not None and sg_channel is not None and unit_id is not None:
            return f"session::{session_name}::{int(shank_id)}::{int(sg_channel)}::{int(unit_id)}"
        if session_name and unit_id is not None:
            return f"session::{session_name}::{int(unit_id)}"
        return json.dumps(unit_payload, sort_keys=True)

    def _build_source_manifest_updates_for_session(self, session: SessionSummary) -> dict[str, dict]:
        updates: dict[str, dict] = {}
        for shank_id, shank_pages in self.pages_by_shank.items():
            for page in shank_pages.values():
                page_units = self._collect_session_page_units(session, int(shank_id), int(page.sg_channel))
                if not page_units:
                    continue
                same_day_align_groups: dict[str, list[UnitSummary]] = defaultdict(list)
                for unit in page_units:
                    if unit.is_noise or unit.is_discarded:
                        continue
                    align_name = unit.align_group.strip()
                    if align_name:
                        same_day_align_groups[align_name].append(unit)

                for unit in page_units:
                    propagated_align_group = None
                    align_name = unit.align_group.strip()
                    if align_name and len(same_day_align_groups.get(align_name, [])) >= 2:
                        propagated_align_group = align_name
                    for member in self._flatten_source_members(
                        [unit],
                        propagated_align_group=propagated_align_group,
                    ):
                        payload = dict(member)
                        payload["session_name"] = str(payload.get("session_name", "") or "")
                        payload["session_index"] = safe_int(payload.get("session_index"))
                        payload["output_folder"] = str(payload.get("output_folder", "") or "")
                        payload["unit_id"] = int(payload.get("unit_id"))
                        payload["merge_group"] = str(payload.get("merge_group", "") or "")
                        payload["align_group"] = str(payload.get("align_group", "") or "")
                        payload["exclude_from_auto_align"] = bool(unit.exclude_from_auto_align)
                        payload["is_noise"] = bool(unit.is_noise)
                        payload["shank_id"] = int(safe_int(payload.get("shank_id")) or unit.shank_id)
                        payload["sg_channel"] = int(safe_int(payload.get("sg_channel")) or unit.sg_channel)
                        payload["local_channel_on_shank"] = int(
                            safe_int(payload.get("local_channel_on_shank")) or unit.local_channel_on_shank
                        )
                        payload.setdefault("analyzer_folder", "")
                        updates[self._source_unit_manifest_key(payload)] = payload
        return updates

    def _sync_source_manifests_for_session(self, session: SessionSummary) -> dict | None:
        updates = self._build_source_manifest_updates_for_session(session)
        if not updates:
            return None

        day_root = Path(session.output_folder)
        summary_root = day_root / DEFAULT_EXPORT_FOLDER_NAME
        summary_root.mkdir(parents=True, exist_ok=True)
        root_manifest_path = summary_root / SOURCE_ALIGNMENT_MANIFEST_NAME

        if root_manifest_path.exists():
            try:
                payload = json.loads(root_manifest_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"output_root": str(day_root), "sessions": []}
        else:
            payload = {"output_root": str(day_root), "sessions": []}

        session_payloads: dict[str, dict] = {}
        ordered_session_keys: list[str] = []
        for session_payload in payload.get("sessions", []):
            session_key = str(session_payload.get("output_folder", "") or session_payload.get("session_name", "") or "")
            if not session_key:
                continue
            session_payloads[session_key] = session_payload
            ordered_session_keys.append(session_key)

        for update in updates.values():
            session_key = str(update.get("output_folder", "") or update.get("session_name", "") or "")
            if not session_key:
                continue
            session_payload = session_payloads.setdefault(
                session_key,
                {
                    "session_name": str(update.get("session_name", "") or ""),
                    "session_index": safe_int(update.get("session_index")),
                    "output_folder": str(update.get("output_folder", "") or ""),
                    "analyzer_folder": str(update.get("analyzer_folder", "") or ""),
                    "units": [],
                },
            )
            if session_key not in ordered_session_keys:
                ordered_session_keys.append(session_key)
            if not session_payload.get("session_name"):
                session_payload["session_name"] = str(update.get("session_name", "") or "")
            if session_payload.get("session_index") is None and update.get("session_index") is not None:
                session_payload["session_index"] = safe_int(update.get("session_index"))
            if not session_payload.get("output_folder"):
                session_payload["output_folder"] = str(update.get("output_folder", "") or "")
            if not session_payload.get("analyzer_folder") and update.get("analyzer_folder"):
                session_payload["analyzer_folder"] = str(update.get("analyzer_folder", "") or "")

            existing_units = session_payload.get("units", [])
            existing_by_key = {
                self._source_unit_manifest_key(unit_payload): dict(unit_payload)
                for unit_payload in existing_units
            }
            update_key = self._source_unit_manifest_key(update)
            merged_payload = dict(existing_by_key.get(update_key, {}))
            merged_payload.update(update)
            existing_by_key[update_key] = merged_payload

            existing_order = [self._source_unit_manifest_key(unit_payload) for unit_payload in existing_units]
            session_payload["units"] = [existing_by_key[key] for key in existing_order if key in existing_by_key]
            for key, unit_payload in existing_by_key.items():
                if key not in existing_order:
                    session_payload["units"].append(unit_payload)

        payload["output_root"] = str(day_root)
        payload["sessions"] = [session_payloads[key] for key in ordered_session_keys if key in session_payloads]
        root_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        affected_shank_ids = sorted(
            {
                int(safe_int(update.get("shank_id")) or -1)
                for update in updates.values()
                if safe_int(update.get("shank_id")) is not None
            }
        )
        shank_manifest_paths: list[str] = []
        for shank_id in affected_shank_ids:
            shank_root = day_root / f"sh{int(shank_id)}" / DEFAULT_EXPORT_FOLDER_NAME
            shank_root.mkdir(parents=True, exist_ok=True)
            shank_manifest_path = shank_root / SOURCE_ALIGNMENT_MANIFEST_NAME
            shank_payload = {
                "output_root": str(day_root / f"sh{int(shank_id)}"),
                "sessions": [],
            }
            for session_payload in payload["sessions"]:
                shank_units = [
                    unit_payload
                    for unit_payload in session_payload.get("units", [])
                    if int(safe_int(unit_payload.get("shank_id")) or -1) == int(shank_id)
                ]
                if not shank_units:
                    continue
                shank_payload["sessions"].append(
                    {
                        "session_name": session_payload.get("session_name", ""),
                        "session_index": session_payload.get("session_index"),
                        "output_folder": session_payload.get("output_folder", ""),
                        "analyzer_folder": session_payload.get("analyzer_folder", ""),
                        "units": shank_units,
                    }
                )
            shank_manifest_path.write_text(json.dumps(shank_payload, indent=2), encoding="utf-8")
            shank_manifest_paths.append(str(shank_manifest_path))

        return {
            "day_root": str(day_root),
            "root_manifest_path": str(root_manifest_path),
            "shank_manifest_paths": shank_manifest_paths,
            "num_updated_units": len(updates),
        }

    def _sync_all_source_manifests(self) -> list[dict]:
        sync_results: list[dict] = []
        for session in self.sessions:
            result = self._sync_source_manifests_for_session(session)
            if result is not None:
                sync_results.append(result)
        return sync_results

    def _build_day_member_payload(self, unit: UnitSummary) -> dict:
        source_members = self._source_members_for_unit(unit)
        source_sessions_present: list[str] = []
        seen_source_sessions: set[str] = set()
        source_session_members: dict[str, list[int]] = defaultdict(list)
        for member in source_members:
            session_name = str(member.get("session_name", "") or "")
            unit_id = safe_int(member.get("unit_id"))
            if not session_name or unit_id is None:
                continue
            if session_name not in seen_source_sessions:
                seen_source_sessions.add(session_name)
                source_sessions_present.append(session_name)
            source_session_members[session_name].append(int(unit_id))

        return {
            "session_name": unit.session_name,
            "session_index": unit.session_index,
            "unit_id": unit.unit_id,
            "merge_group": unit.merge_group,
            "align_group": unit.align_group,
            "output_folder": unit.output_folder,
            "shank_id": int(unit.shank_id),
            "sg_channel": int(unit.sg_channel),
            "local_channel_on_shank": int(unit.local_channel_on_shank),
            "source_group_key": str(getattr(unit, "_source_group_key", "") or ""),
            "source_export_manifest_path": str(
                getattr(unit, "_source_export_manifest_path", "") or ""
            ),
            "source_page_scope": dict(getattr(unit, "_source_page_scope", {}) or {}),
            "num_source_members": len(source_members),
            "source_sessions_present": source_sessions_present,
            "num_source_sessions": len(source_sessions_present),
            "source_session_members": [
                {"session_name": session_name, "unit_ids": sorted(member_unit_ids)}
                for session_name, member_unit_ids in source_session_members.items()
            ],
        }

    def export_summary_bundle_for_page(self, shank_id: int, page_id: str) -> dict:
        page = self.get_page(shank_id, page_id)
        page_summary_root = self.summary_root / f"sh{page.shank_id}"
        page_summary_root.mkdir(parents=True, exist_ok=True)
        export_folder = page_summary_root / f"exported_units_sg_{page.sg_channel:03d}"
        export_folder.mkdir(parents=True, exist_ok=True)

        page_units = [unit for session in page.sessions for unit in session.units]
        eligible_units = [
            unit
            for unit in page_units
            if not unit.is_discarded
            and not unit.is_noise
            and not unit.align_group.strip()
            and not unit.exclude_from_auto_align
        ]
        page_auto_lookup: dict[str, str] = {}
        auto_rows, _grouped_keys = self._build_html_auto_align_rows(eligible_units)
        for row in auto_rows:
            component_units = [
                unit
                for session in page.sessions
                for unit in row.get(session.session_index, [])
            ]
            if not component_units:
                continue
            group_name = (
                f"sh{page.shank_id}_sg{page.sg_channel}__auto__"
                f"s{component_units[0].session_index:03d}_u{component_units[0].unit_id:04d}"
            )
            for unit in component_units:
                page_auto_lookup[day_unit_record_key(unit)] = group_name

        final_groups: dict[str, list[UnitSummary]] = {}
        discarded_groups: dict[str, list[UnitSummary]] = {}
        noise_groups: dict[str, list[UnitSummary]] = {}
        for unit in page_units:
            if unit.is_discarded:
                discarded_groups.setdefault(self.discard_group_key_for_unit(unit), []).append(unit)
                continue
            if unit.is_noise:
                noise_groups.setdefault(
                    self.discard_group_key_for_unit(unit).replace("discarded__", "noise__"),
                    [],
                ).append(unit)
                continue
            final_groups.setdefault(
                self.final_group_key_for_unit(unit, auto_align_lookup=page_auto_lookup),
                [],
            ).append(unit)

        manifest_rows = []
        unique_unit_rows = []
        discarded_unit_rows = []
        noise_unit_rows = []

        for group_index, (group_key, units) in enumerate(sorted(final_groups.items()), start=1):
            group_folder = export_folder / f"unit_{group_index:04d}"
            group_folder.mkdir(parents=True, exist_ok=True)
            representative = units[0]
            source_members = self._flatten_source_members(units)
            day_members = [self._build_day_member_payload(unit) for unit in units]
            source_sessions_present: list[str] = []
            seen_source_sessions: set[str] = set()
            source_session_members: dict[str, list[int]] = defaultdict(list)
            for member in source_members:
                session_name = str(member.get("session_name", "") or "")
                unit_id = safe_int(member.get("unit_id"))
                if not session_name or unit_id is None:
                    continue
                if session_name not in seen_source_sessions:
                    seen_source_sessions.add(session_name)
                    source_sessions_present.append(session_name)
                source_session_members[session_name].append(int(unit_id))
            copied_images: list[str] = []
            for item_index, unit in enumerate(units, start=1):
                src = Path(unit.waveform_image_path)
                dst = group_folder / (
                    f"waveform_{item_index:02d}_{unit.session_name.replace(' ', '_')}_u{unit.unit_id}.png"
                )
                if copy2_if_needed(src, dst):
                    copied_images.append(str(dst))

            summary_path = group_folder / "summary.txt"
            summary_path.write_text(
                self.build_group_summary_text(group_index, group_key, units),
                encoding="utf-8",
            )
            unique_unit_row = self.build_unique_unit_summary_row(
                group_index=group_index,
                group_key=group_key,
                units=units,
                group_folder=group_folder,
                copied_images=copied_images,
                summary_path=summary_path,
            )
            unique_unit_row["day_members"] = day_members
            unique_unit_row["num_day_members"] = len(day_members)
            unique_unit_row["source_members"] = source_members
            unique_unit_row["num_source_members"] = len(source_members)
            unique_unit_row["source_sessions_present"] = source_sessions_present
            unique_unit_row["num_source_sessions"] = len(source_sessions_present)
            unique_unit_row["source_session_members"] = [
                {"session_name": session_name, "unit_ids": sorted(member_unit_ids)}
                for session_name, member_unit_ids in source_session_members.items()
            ]
            unique_unit_rows.append(unique_unit_row)
            manifest_rows.append(
                {
                    "final_unit_id": group_index,
                    "final_group_key": group_key,
                    "export_folder": str(group_folder),
                    "representative_session": representative.session_name,
                    "representative_unit_id": representative.unit_id,
                    "shank_id": representative.shank_id,
                    "local_channel_on_shank": representative.local_channel_on_shank,
                    "members": source_members,
                    "day_members": day_members,
                    "num_day_members": len(day_members),
                    "num_source_members": len(source_members),
                    "source_sessions_present": source_sessions_present,
                    "num_source_sessions": len(source_sessions_present),
                    "source_session_members": [
                        {"session_name": session_name, "unit_ids": sorted(member_unit_ids)}
                        for session_name, member_unit_ids in source_session_members.items()
                    ],
                    "source_members": source_members,
                    "images": copied_images,
                }
            )

        for group_key, units in sorted(discarded_groups.items()):
            discarded_unit_rows.append(self.build_discarded_unit_summary_row(group_key=group_key, units=units))
        for group_key, units in sorted(noise_groups.items()):
            noise_unit_rows.append(self.build_noise_unit_summary_row(group_key=group_key, units=units))

        unique_units_json_path = page_summary_root / f"unique_units_summary_sg_{page.sg_channel:03d}.json"
        unique_units_json_path.write_text(json.dumps(unique_unit_rows, indent=2), encoding="utf-8")
        unique_units_csv_path = page_summary_root / f"unique_units_summary_sg_{page.sg_channel:03d}.csv"
        self.write_unique_units_summary_csv(unique_units_csv_path, unique_unit_rows)

        discarded_units_json_path = page_summary_root / f"discarded_units_summary_sg_{page.sg_channel:03d}.json"
        discarded_units_json_path.write_text(json.dumps(discarded_unit_rows, indent=2), encoding="utf-8")
        discarded_units_csv_path = page_summary_root / f"discarded_units_summary_sg_{page.sg_channel:03d}.csv"
        self.write_discarded_units_summary_csv(discarded_units_csv_path, discarded_unit_rows)

        noise_units_json_path = page_summary_root / f"noise_units_summary_sg_{page.sg_channel:03d}.json"
        noise_units_json_path.write_text(json.dumps(noise_unit_rows, indent=2), encoding="utf-8")
        noise_units_csv_path = page_summary_root / f"noise_units_summary_sg_{page.sg_channel:03d}.csv"
        self.write_noise_units_summary_csv(noise_units_csv_path, noise_unit_rows)

        export_manifest_path = page_summary_root / f"export_summary_sg_{page.sg_channel:03d}.json"
        export_manifest_path.write_text(
            json.dumps(
                {
                    "output_root": str(self.root_folder),
                    "page_scope": {
                        "shank_id": int(page.shank_id),
                        "sg_channel": int(page.sg_channel),
                        "page_id": page.page_id,
                    },
                    "unique_units_summary_json": str(unique_units_json_path),
                    "unique_units_summary_csv": str(unique_units_csv_path),
                    "discarded_units_summary_json": str(discarded_units_json_path),
                    "discarded_units_summary_csv": str(discarded_units_csv_path),
                    "noise_units_summary_json": str(noise_units_json_path),
                    "noise_units_summary_csv": str(noise_units_csv_path),
                    "cross_session_alignment_groups": manifest_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "export_manifest_path": str(export_manifest_path),
            "unique_units_json_path": str(unique_units_json_path),
            "unique_units_csv_path": str(unique_units_csv_path),
            "discarded_units_json_path": str(discarded_units_json_path),
            "discarded_units_csv_path": str(discarded_units_csv_path),
            "noise_units_json_path": str(noise_units_json_path),
            "noise_units_csv_path": str(noise_units_csv_path),
            "num_unique_units": len(unique_unit_rows),
            "num_discarded_groups": len(discarded_unit_rows),
            "num_noise_groups": len(noise_unit_rows),
            "num_alignment_groups": len(manifest_rows),
            "page_scope": f"shank {page.shank_id}, SG {page.sg_channel}",
        }

    def export_summary_bundle(self) -> dict:
        export_folder = self.summary_root / "exported_units"
        export_folder.mkdir(parents=True, exist_ok=True)

        auto_align_lookup = self.build_auto_align_lookup_multi_shank()
        final_groups: dict[str, list[UnitSummary]] = {}
        discarded_groups: dict[str, list[UnitSummary]] = {}
        noise_groups: dict[str, list[UnitSummary]] = {}

        for unit in self._iter_all_units():
            if unit.is_discarded:
                discarded_groups.setdefault(self.discard_group_key_for_unit(unit), []).append(unit)
                continue
            if unit.is_noise:
                noise_groups.setdefault(
                    self.discard_group_key_for_unit(unit).replace("discarded__", "noise__"),
                    [],
                ).append(unit)
                continue
            final_groups.setdefault(
                self.final_group_key_for_unit(unit, auto_align_lookup=auto_align_lookup),
                [],
            ).append(unit)

        manifest_rows = []
        unique_unit_rows = []
        discarded_unit_rows = []
        noise_unit_rows = []

        for group_index, (group_key, units) in enumerate(sorted(final_groups.items()), start=1):
            group_folder = export_folder / f"unit_{group_index:04d}"
            group_folder.mkdir(parents=True, exist_ok=True)
            representative = units[0]
            source_members = self._flatten_source_members(units)
            day_members = [self._build_day_member_payload(unit) for unit in units]
            source_sessions_present: list[str] = []
            seen_source_sessions: set[str] = set()
            source_session_members: dict[str, list[int]] = defaultdict(list)
            for member in source_members:
                session_name = str(member.get("session_name", "") or "")
                unit_id = safe_int(member.get("unit_id"))
                if not session_name or unit_id is None:
                    continue
                if session_name not in seen_source_sessions:
                    seen_source_sessions.add(session_name)
                    source_sessions_present.append(session_name)
                source_session_members[session_name].append(int(unit_id))

            copied_images: list[str] = []
            for item_index, unit in enumerate(units, start=1):
                src = Path(unit.waveform_image_path)
                dst = group_folder / (
                    f"waveform_{item_index:02d}_{unit.session_name.replace(' ', '_')}_u{unit.unit_id}.png"
                )
                if copy2_if_needed(src, dst):
                    copied_images.append(str(dst))

            summary_path = group_folder / "summary.txt"
            summary_path.write_text(
                self.build_group_summary_text(group_index, group_key, units),
                encoding="utf-8",
            )
            unique_unit_row = self.build_unique_unit_summary_row(
                group_index=group_index,
                group_key=group_key,
                units=units,
                group_folder=group_folder,
                copied_images=copied_images,
                summary_path=summary_path,
            )
            unique_unit_row["day_members"] = day_members
            unique_unit_row["num_day_members"] = len(day_members)
            unique_unit_row["source_members"] = source_members
            unique_unit_row["num_source_members"] = len(source_members)
            unique_unit_row["source_sessions_present"] = source_sessions_present
            unique_unit_row["num_source_sessions"] = len(source_sessions_present)
            unique_unit_row["source_session_members"] = [
                {"session_name": session_name, "unit_ids": sorted(member_unit_ids)}
                for session_name, member_unit_ids in source_session_members.items()
            ]
            unique_unit_rows.append(unique_unit_row)
            manifest_rows.append(
                {
                    "final_unit_id": group_index,
                    "final_group_key": group_key,
                    "export_folder": str(group_folder),
                    "representative_session": representative.session_name,
                    "representative_unit_id": representative.unit_id,
                    "shank_id": representative.shank_id,
                    "local_channel_on_shank": representative.local_channel_on_shank,
                    "members": source_members,
                    "day_members": day_members,
                    "num_day_members": len(day_members),
                    "num_source_members": len(source_members),
                    "source_sessions_present": source_sessions_present,
                    "num_source_sessions": len(source_sessions_present),
                    "source_session_members": [
                        {"session_name": session_name, "unit_ids": sorted(member_unit_ids)}
                        for session_name, member_unit_ids in source_session_members.items()
                    ],
                    "source_members": source_members,
                    "images": copied_images,
                }
            )

        for group_key, units in sorted(discarded_groups.items()):
            discarded_unit_rows.append(
                self.build_discarded_unit_summary_row(group_key=group_key, units=units)
            )
        for group_key, units in sorted(noise_groups.items()):
            noise_unit_rows.append(self.build_noise_unit_summary_row(group_key=group_key, units=units))

        unique_units_json_path = self.summary_root / "unique_units_summary.json"
        unique_units_json_path.write_text(json.dumps(unique_unit_rows, indent=2), encoding="utf-8")
        unique_units_csv_path = self.summary_root / "unique_units_summary.csv"
        self.write_unique_units_summary_csv(unique_units_csv_path, unique_unit_rows)

        discarded_units_json_path = self.summary_root / "discarded_units_summary.json"
        discarded_units_json_path.write_text(json.dumps(discarded_unit_rows, indent=2), encoding="utf-8")
        discarded_units_csv_path = self.summary_root / "discarded_units_summary.csv"
        self.write_discarded_units_summary_csv(discarded_units_csv_path, discarded_unit_rows)

        noise_units_json_path = self.summary_root / "noise_units_summary.json"
        noise_units_json_path.write_text(json.dumps(noise_unit_rows, indent=2), encoding="utf-8")
        noise_units_csv_path = self.summary_root / "noise_units_summary.csv"
        self.write_noise_units_summary_csv(noise_units_csv_path, noise_unit_rows)

        export_manifest_path = self.summary_root / "export_summary.json"
        export_manifest_path.write_text(
            json.dumps(
                {
                    "output_root": str(self.root_folder),
                    "member_mode": "full_source_members",
                    "unique_units_summary_json": str(unique_units_json_path),
                    "unique_units_summary_csv": str(unique_units_csv_path),
                    "discarded_units_summary_json": str(discarded_units_json_path),
                    "discarded_units_summary_csv": str(discarded_units_csv_path),
                    "noise_units_summary_json": str(noise_units_json_path),
                    "noise_units_summary_csv": str(noise_units_csv_path),
                    "cross_session_alignment_groups": manifest_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "export_manifest_path": str(export_manifest_path),
            "unique_units_json_path": str(unique_units_json_path),
            "unique_units_csv_path": str(unique_units_csv_path),
            "discarded_units_json_path": str(discarded_units_json_path),
            "discarded_units_csv_path": str(discarded_units_csv_path),
            "noise_units_json_path": str(noise_units_json_path),
            "noise_units_csv_path": str(noise_units_csv_path),
            "num_unique_units": len(unique_unit_rows),
            "num_discarded_groups": len(discarded_unit_rows),
            "num_noise_groups": len(noise_unit_rows),
            "num_alignment_groups": len(manifest_rows),
        }


def serve_alignment_days_app(
    root_folder: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    def show_progress(message: str) -> None:
        print(f"[loading] {message}", flush=True)

    show_progress(f"Preparing day-alignment app for: {root_folder}")
    state = AlignmentDaysState(root_folder, progress_callback=show_progress)
    class AlignmentDaysRequestHandler(html_review.AlignmentRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._html_response(build_days_html_shell())
                return
            super().do_GET()

    AlignmentDaysRequestHandler.state = state

    handler_class = type(
        "BoundAlignmentDaysRequestHandler",
        (AlignmentDaysRequestHandler,),
        {},
    )
    server = ThreadingHTTPServer((host, port), handler_class)
    url = f"http://{host}:{port}/"
    print(f"Serving day-level alignment review at: {url}")
    print(f"Input root: {root_folder}")
    print(f"Summary root: {state.summary_root}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Serve day-level HTML alignment review using exports produced by Alignment_html.py."
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
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", default=8765, type=int, help="Port to bind")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()

    if args.input_roots:
        selected_roots = parse_input_roots_text(args.input_roots)
    else:
        selected_roots = prompt_for_input_roots()

    selected_roots = [path.resolve() for path in selected_roots]
    day_roots = discover_day_sorting_roots(selected_roots)
    common_root = Path(os.path.commonpath([str(path) for path in selected_roots]))
    summary_folder_name = build_day_summary_folder_name(day_roots)
    selection_file = common_root / summary_folder_name / "selected_day_folders.txt"
    selection_file.parent.mkdir(parents=True, exist_ok=True)
    selection_file.write_text(
        "\n".join(str(path) for path in selected_roots) + "\n",
        encoding="utf-8",
    )

    try:
        serve_alignment_days_app(
            common_root,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
    except Exception as exc:
        print(f"[error] {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
