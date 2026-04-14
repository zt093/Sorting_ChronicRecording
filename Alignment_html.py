"""
HTML alignment review app.

This script reuses the core alignment logic from Units_alignment_UI.py, but
serves a browser-based review and editing app that can cover multiple shanks
from one batch output root.

High-level structure
--------------------
1. `load_all_sessions_multi_shank(...)` loads all analyzer folders into
   `SessionSummary` / `UnitSummary` objects.
2. `AlignmentState` owns all mutable decision state (align, merge, noise,
   discard-derived visibility, undo snapshots, manifest save/load).
3. `build_app_payload()` converts the current Python state into the JSON payload
   consumed by the browser UI.
4. `build_html_shell()` is the browser app shell (layout + client-side logic).
5. `AlignmentRequestHandler` exposes the local HTTP API used by the browser UI.

Save/export tiers
-----------------
- Page actions: current SG page only.
- All-pages actions: every loaded SG page across the batch, including pages
  hidden from the selectors.
- Summary actions: one combined summary across all loaded shanks.

Visibility rules
----------------
- Discarded units are hidden from normal SG page review and shown on the global
  discarded page instead.
- SG pages with no reviewable non-discarded units are hidden from the page
  selectors, but still reported in the summary/notice payload.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
import traceback
from urllib.parse import parse_qs, quote, unquote, urlparse
import webbrowser

import spikeinterface.full as si

from Units_alignment_UI import (
    AUTO_MERGE_MIN_SIMILARITY,
    DEFAULT_EXPORT_FOLDER_NAME,
    PageSummary,
    SessionSummary,
    UnitSummary,
    build_discard_reason,
    build_metrics_lookup,
    choose_output_root,
    compute_amplitude_similarity,
    compute_autocorrelogram_similarity,
    compute_similarity,
    compute_waveform_similarity,
    discover_analyzer_folders,
    ensure_required_extensions,
    find_unit_summary_image,
    format_metric,
    get_autocorrelogram_vector,
    get_trough_to_peak_duration_ms,
    get_waveform_vector,
    infer_unit_channel_metadata,
    is_unit_auto_discarded,
    load_unit_channel_mapping,
    safe_float,
    safe_int,
    sanitize_token,
    save_waveform_card_image,
    session_name_from_output_folder,
    unit_record_key,
)


HTML_TITLE = "Alignment Review"
MANIFEST_NAME = "alignment_manifest.json"
AUTO_ALIGN_MIN_SIMILARITY = 0.75
FORCE_WAVEFORM_LINK_THRESHOLD = 0.99


def iter_all_units(sessions: list[SessionSummary]) -> list[UnitSummary]:
    units: list[UnitSummary] = []
    for session in sessions:
        units.extend(session.units)
    return units


def load_all_sessions_multi_shank(
    root_folder: Path,
    progress_callback=None,
) -> tuple[list[SessionSummary], dict[int, dict[str, PageSummary]], Path]:
    analyzer_folders = discover_analyzer_folders(root_folder)
    cache_folder = root_folder / DEFAULT_EXPORT_FOLDER_NAME / "_cache"
    sessions: list[SessionSummary] = []
    total_sessions = len(analyzer_folders)

    if progress_callback is not None:
        progress_callback(
            f"Found {total_sessions} analyzer folder(s).\nLoading sessions for HTML review..."
        )

    for session_index, analyzer_folder in enumerate(analyzer_folders):
        output_folder = analyzer_folder.parent
        session_name = session_name_from_output_folder(output_folder, session_index)
        if progress_callback is not None:
            progress_callback(
                f"Loading session {session_index + 1}/{total_sessions}: {session_name}"
            )

        unit_channel_mapping = load_unit_channel_mapping(output_folder)
        analyzer = si.load_sorting_analyzer(
            folder=analyzer_folder,
            format="zarr",
            load_extensions=True,
        )
        ensure_required_extensions(analyzer)
        metrics_lookup = build_metrics_lookup(analyzer)

        session_summary = SessionSummary(
            session_name=session_name,
            session_index=session_index,
            output_folder=str(output_folder),
            analyzer_folder=str(analyzer_folder),
        )

        for unit_id in analyzer.sorting.get_unit_ids():
            unit_id_int = int(unit_id)
            metadata = infer_unit_channel_metadata(
                analyzer,
                unit_id,
                mapping_row=unit_channel_mapping.get(unit_id_int),
            )
            metrics = metrics_lookup.get(unit_id_int, {})
            waveform_vector = get_waveform_vector(analyzer, unit_id)
            autocorrelogram_vector = get_autocorrelogram_vector(analyzer, unit_id)
            trough_to_peak_duration_ms = get_trough_to_peak_duration_ms(analyzer, unit_id)
            preferred_image_path = find_unit_summary_image(output_folder, unit_id_int)
            image_path = (
                cache_folder
                / "waveforms"
                / f"session_{session_index:03d}"
                / f"shank{metadata['shank_id']}_ch{metadata['local_channel_on_shank']}_unit{unit_id_int}.png"
            )
            if preferred_image_path is None and not image_path.exists():
                save_waveform_card_image(
                    analyzer=analyzer,
                    unit_id=unit_id_int,
                    save_path=image_path,
                    session_name=session_name,
                    shank_id=metadata["shank_id"],
                    channel_id=metadata["local_channel_on_shank"],
                )

            session_summary.units.append(
                UnitSummary(
                    session_name=session_name,
                    session_index=session_index,
                    analyzer_folder=str(analyzer_folder),
                    output_folder=str(output_folder),
                    unit_id=unit_id_int,
                    shank_id=metadata["shank_id"],
                    local_channel_on_shank=metadata["local_channel_on_shank"],
                    sg_channel=metadata["sg_channel"],
                    amplitude_median=safe_float(metrics.get("amplitude_median")),
                    firing_rate=safe_float(metrics.get("firing_rate")),
                    isi_violations_ratio=safe_float(metrics.get("isi_violations_ratio")),
                    snr=safe_float(metrics.get("snr")),
                    num_spikes=safe_int(metrics.get("num_spikes")),
                    waveform_similarity_vector=waveform_vector.tolist(),
                    autocorrelogram_similarity_vector=autocorrelogram_vector.tolist(),
                    trough_to_peak_duration_ms=trough_to_peak_duration_ms,
                    waveform_image_path=str(preferred_image_path or image_path),
                )
            )

        sessions.append(session_summary)

    pages_by_shank: dict[int, dict[str, PageSummary]] = defaultdict(dict)
    shank_to_channels: dict[int, set[int]] = defaultdict(set)
    for session in sessions:
        for unit in session.units:
            shank_to_channels[int(unit.shank_id)].add(int(unit.sg_channel))

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

    return sessions, dict(pages_by_shank), cache_folder


def discover_shank_folder_ids(root_folder: Path) -> list[str]:
    shank_ids: list[str] = []
    for child in sorted(root_folder.iterdir()) if root_folder.exists() else []:
        if not child.is_dir():
            continue
        name = child.name.strip()
        if len(name) >= 3 and name.lower().startswith("sh") and name[2:].isdigit():
            shank_ids.append(name[2:])
    return shank_ids


class AlignmentState:
    def __init__(self, root_folder: Path, progress_callback=None):
        self.root_folder = root_folder
        self.summary_root = root_folder / DEFAULT_EXPORT_FOLDER_NAME
        self.summary_root.mkdir(parents=True, exist_ok=True)
        self.discovered_shank_folder_ids = discover_shank_folder_ids(root_folder)
        self.sessions, self.pages_by_shank, self.cache_folder = load_all_sessions_multi_shank(
            root_folder,
            progress_callback=progress_callback,
        )
        self._lock = threading.RLock()
        self._undo_stack: list[dict[str, dict]] = []
        self._stable_unit_aliases: dict[str, dict[str, str]] = {}
        self._stable_next_alias_index: dict[str, int] = {}
        self.loaded_shank_ids = sorted(str(shank_id) for shank_id in self.pages_by_shank.keys())
        self.empty_shank_folder_ids = sorted(
            shank_id for shank_id in self.discovered_shank_folder_ids if shank_id not in set(self.loaded_shank_ids)
        )
        if progress_callback is not None:
            progress_callback("Applying saved alignment manifest if available...")
        self.apply_manifest_if_available()
        if progress_callback is not None:
            progress_callback("Building auto-merge suggestions...")
        self.sync_auto_merge_groups()
        if progress_callback is not None:
            progress_callback("Startup state ready.")

    def _iter_all_units(self) -> list[UnitSummary]:
        return iter_all_units(self.sessions)

    def snapshot(self) -> dict[str, dict]:
        snapshot: dict[str, dict] = {}
        for unit in self._iter_all_units():
            snapshot[unit_record_key(unit)] = {
                "merge_group": unit.merge_group,
                "align_group": unit.align_group,
                "exclude_from_auto_align": unit.exclude_from_auto_align,
                "is_noise": unit.is_noise,
                "is_discarded": unit.is_discarded,
            }
        return snapshot

    def restore_snapshot(self, snapshot: dict[str, dict]) -> None:
        for unit in self._iter_all_units():
            state = snapshot.get(unit_record_key(unit))
            if state is None:
                continue
            unit.merge_group = str(state.get("merge_group", "") or "")
            unit.align_group = str(state.get("align_group", "") or "")
            unit.exclude_from_auto_align = bool(state.get("exclude_from_auto_align", False))
            unit.is_noise = bool(state.get("is_noise", False))
            unit.is_discarded = bool(state.get("is_discarded", False))
        self.sync_auto_merge_groups()

    def push_undo_snapshot(self, snapshot: dict[str, dict]) -> None:
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 100:
            self._undo_stack = self._undo_stack[-100:]

    def undo(self) -> dict:
        with self._lock:
            if not self._undo_stack:
                raise ValueError("No earlier change is available to undo.")
            snapshot = self._undo_stack.pop()
            self.restore_snapshot(snapshot)
            return {"message": "Undid the last change batch.", "app": self.build_app_payload()}

    def apply_manifest_if_available(self) -> None:
        # Manifest loading is intentionally layered:
        # 1. read per-shank manifests left by older single-shank workflows
        # 2. read the batch-level manifest at the selected root
        # 3. for each unit, keep the newest matching saved record
        #
        # Matching does not rely only on session_index because batch loading and
        # older per-shank loading can assign different session indices.
        manifest_paths: list[Path] = []
        for shank_id in self.discovered_shank_folder_ids:
            shank_manifest = self.root_folder / f"sh{shank_id}" / DEFAULT_EXPORT_FOLDER_NAME / MANIFEST_NAME
            if shank_manifest.exists():
                manifest_paths.append(shank_manifest)
        root_manifest = self.summary_root / MANIFEST_NAME
        if root_manifest.exists():
            manifest_paths.append(root_manifest)

        manifest_units: dict[str, tuple[float, dict]] = {}
        for manifest_path in manifest_paths:
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            try:
                manifest_mtime = manifest_path.stat().st_mtime
            except Exception:
                manifest_mtime = 0.0
            for session_payload in payload.get("sessions", []):
                for unit_payload in session_payload.get("units", []):
                    for key in self._manifest_lookup_keys_from_payload(unit_payload):
                        previous = manifest_units.get(key)
                        if previous is None or manifest_mtime >= previous[0]:
                            manifest_units[key] = (manifest_mtime, unit_payload)

        for unit in self._iter_all_units():
            saved = {}
            for key in self._manifest_lookup_keys_for_unit(unit):
                if key in manifest_units:
                    saved = manifest_units[key][1]
                    break
            unit.merge_group = str(saved.get("merge_group", "") or "")
            unit.align_group = str(saved.get("align_group", "") or "")
            unit.exclude_from_auto_align = bool(saved.get("exclude_from_auto_align", False))
            unit.is_noise = bool(saved.get("is_noise", False))
            unit.is_discarded = is_unit_auto_discarded(unit)

    def _manifest_lookup_keys_for_unit(self, unit: UnitSummary) -> list[str]:
        analyzer_folder = str(unit.analyzer_folder or "")
        output_folder = str(unit.output_folder or "")
        session_name = str(unit.session_name or "")
        return [
            f"analyzer::{analyzer_folder}::{int(unit.unit_id)}",
            f"output::{output_folder}::{int(unit.unit_id)}",
            f"session::{session_name}::{int(unit.shank_id)}::{int(unit.sg_channel)}::{int(unit.unit_id)}",
            f"session::{session_name}::{int(unit.shank_id)}::{int(unit.unit_id)}",
            f"{int(unit.session_index)}:{int(unit.shank_id)}:{int(unit.sg_channel)}:{int(unit.unit_id)}",
            f"{int(unit.session_index)}:{int(unit.shank_id)}:{int(unit.unit_id)}",
            unit_record_key(unit),
        ]

    def _manifest_lookup_keys_from_payload(self, unit_payload: dict) -> list[str]:
        keys: list[str] = []
        session_index = unit_payload.get("session_index")
        unit_id = unit_payload.get("unit_id")
        shank_id = unit_payload.get("shank_id")
        sg_channel = unit_payload.get("sg_channel")
        analyzer_folder = unit_payload.get("analyzer_folder")
        output_folder = unit_payload.get("output_folder")
        session_name = unit_payload.get("session_name")
        if analyzer_folder is not None and unit_id is not None:
            keys.append(f"analyzer::{str(analyzer_folder)}::{int(unit_id)}")
        if output_folder is not None and unit_id is not None:
            keys.append(f"output::{str(output_folder)}::{int(unit_id)}")
        if session_name is not None and shank_id is not None and sg_channel is not None and unit_id is not None:
            keys.append(f"session::{str(session_name)}::{int(shank_id)}::{int(sg_channel)}::{int(unit_id)}")
        if session_name is not None and shank_id is not None and unit_id is not None:
            keys.append(f"session::{str(session_name)}::{int(shank_id)}::{int(unit_id)}")
        if session_index is not None and shank_id is not None and sg_channel is not None and unit_id is not None:
            keys.append(f"{int(session_index)}:{int(shank_id)}:{int(sg_channel)}:{int(unit_id)}")
        if session_index is not None and shank_id is not None and unit_id is not None:
            keys.append(f"{int(session_index)}:{int(shank_id)}:{int(unit_id)}")
        if session_index is not None and unit_id is not None:
            keys.append(f"{int(session_index)}:{int(unit_id)}")
        return keys

    def sync_auto_merge_groups(self) -> None:
        # Auto-merge is recomputed from the current decision state after every
        # mutating command batch. Manual merge names are preserved; only the
        # generated "__alignmerge__" / "__automerge__" groups are rebuilt.
        units = self._iter_all_units()
        for unit in units:
            if unit.merge_group.startswith("__alignmerge__") or unit.merge_group.startswith("__automerge__"):
                unit.merge_group = ""

        grouped_units: dict[tuple[int, str], list[UnitSummary]] = {}
        for unit in units:
            if unit.is_discarded or unit.is_noise:
                continue
            align_name = unit.align_group.strip()
            if not align_name:
                continue
            grouped_units.setdefault((int(unit.session_index), align_name), []).append(unit)

        for (_session_index, align_name), members in grouped_units.items():
            if len(members) < 2:
                continue
            auto_merge_name = f"__alignmerge__{sanitize_token(align_name)}"
            for unit in members:
                if not unit.merge_group or unit.merge_group.startswith("__alignmerge__") or unit.merge_group.startswith("__automerge__"):
                    unit.merge_group = auto_merge_name

        merge_candidate_groups: dict[tuple[int, int], list[UnitSummary]] = {}
        for unit in units:
            if unit.is_discarded or unit.is_noise:
                continue
            merge_candidate_groups.setdefault((int(unit.session_index), int(unit.sg_channel)), []).append(unit)

        for (session_index, sg_channel), members in merge_candidate_groups.items():
            if len(members) < 2:
                continue
            adjacency: dict[str, set[str]] = {}
            sorted_members = sorted(members, key=lambda item: item.unit_id)
            for i, left in enumerate(sorted_members):
                if left.merge_group and not (left.merge_group.startswith("__alignmerge__") or left.merge_group.startswith("__automerge__")):
                    continue
                for right in sorted_members[i + 1 :]:
                    if right.merge_group and not (right.merge_group.startswith("__alignmerge__") or right.merge_group.startswith("__automerge__")):
                        continue
                    waveform_score = compute_waveform_similarity(left, right)
                    amplitude_score = compute_amplitude_similarity(left, right)
                    autocorrelogram_score = compute_autocorrelogram_similarity(left, right)
                    if (
                        waveform_score > FORCE_WAVEFORM_LINK_THRESHOLD
                        and amplitude_score > AUTO_ALIGN_MIN_SIMILARITY
                        and autocorrelogram_score > AUTO_ALIGN_MIN_SIMILARITY
                    ):
                        left_key = unit_record_key(left)
                        right_key = unit_record_key(right)
                        adjacency.setdefault(left_key, set()).add(right_key)
                        adjacency.setdefault(right_key, set()).add(left_key)

            visited: set[str] = set()
            for start_key in sorted(adjacency):
                if start_key in visited:
                    continue
                stack = [start_key]
                component_keys: set[str] = set()
                while stack:
                    current_key = stack.pop()
                    if current_key in visited:
                        continue
                    visited.add(current_key)
                    component_keys.add(current_key)
                    stack.extend(adjacency.get(current_key, set()) - visited)
                if len(component_keys) < 2:
                    continue
                component_units = sorted(
                    [unit for unit in members if unit_record_key(unit) in component_keys],
                    key=lambda item: item.unit_id,
                )
                auto_merge_name = (
                    f"__automerge__s{session_index:03d}_sg{sg_channel:03d}_u{component_units[0].unit_id:04d}"
                )
                for unit in component_units:
                    if not unit.merge_group or unit.merge_group.startswith("__alignmerge__") or unit.merge_group.startswith("__automerge__"):
                        unit.merge_group = auto_merge_name

    def summarize_page(self, page: PageSummary) -> dict:
        session_counts = [
            {
                "session_name": session.session_name,
                "unit_count": len(session.units),
                "discarded_unit_count": sum(
                    1 for unit in session.units if unit.is_discarded and not unit.is_noise
                ),
            }
            for session in page.sessions
        ]
        return {
            "session_counts": session_counts,
            "total_units": sum(item["unit_count"] for item in session_counts),
            "total_discarded_units": sum(item["discarded_unit_count"] for item in session_counts),
        }

    def row_kind_for_units(self, row_units: list[UnitSummary]) -> str:
        if not row_units:
            return "empty"
        if all(unit.is_noise for unit in row_units):
            return "noise"
        if any(unit.align_group.strip() for unit in row_units):
            return "manual_align"
        if len(row_units) >= 2:
            return "auto_align"
        return "singleton"

    def _build_row_from_units(self, units: list[UnitSummary]) -> dict[int, list[UnitSummary]]:
        row: dict[int, list[UnitSummary]] = {}
        for unit in sorted(units, key=lambda item: (item.session_index, item.unit_id)):
            row.setdefault(unit.session_index, []).append(unit)
        return row

    def _row_sort_key(self, row: dict[int, list[UnitSummary]]) -> tuple[int, int]:
        first_session = min(row.keys()) if row else 10**9
        first_unit = (
            min(unit.unit_id for units in row.values() for unit in units)
            if row
            else 10**9
        )
        return (first_session, first_unit)

    def _passes_auto_align_pair(self, left: UnitSummary, right: UnitSummary) -> bool:
        waveform_score = compute_waveform_similarity(left, right)
        amplitude_score = compute_amplitude_similarity(left, right)
        autocorrelogram_score = compute_autocorrelogram_similarity(left, right)
        return (
            waveform_score >= AUTO_ALIGN_MIN_SIMILARITY
            and amplitude_score >= AUTO_ALIGN_MIN_SIMILARITY
            and autocorrelogram_score >= AUTO_ALIGN_MIN_SIMILARITY
        )

    def _is_forced_auto_align_pair(self, left: UnitSummary, right: UnitSummary) -> bool:
        waveform_score = compute_waveform_similarity(left, right)
        amplitude_score = compute_amplitude_similarity(left, right)
        autocorrelogram_score = compute_autocorrelogram_similarity(left, right)
        return (
            waveform_score > FORCE_WAVEFORM_LINK_THRESHOLD
            and amplitude_score >= AUTO_ALIGN_MIN_SIMILARITY
            and autocorrelogram_score >= AUTO_ALIGN_MIN_SIMILARITY
        )

    def _build_html_auto_align_rows(
        self,
        units: list[UnitSummary],
    ) -> tuple[list[dict[int, list[UnitSummary]]], set[str]]:
        # HTML auto-row logic differs slightly from the original Tk version:
        # - same-session auto-merge groups are used as seed components first
        # - forced cross-session links are processed before regular links
        # - the align step still will not merge two components that already
        #   contain the same session index
        #
        # Result: a final row may contain multiple units from one session, but
        # only when those units were merged before the cross-session align step.
        eligible_units = sorted(units, key=lambda item: (item.session_index, item.unit_id))
        if not eligible_units:
            return [], set()

        units_lookup = {unit_record_key(unit): unit for unit in eligible_units}
        components: dict[str, set[str]] = {}
        component_sessions: dict[str, set[int]] = {}
        component_for_key: dict[str, str] = {}

        merge_seed_groups: dict[str, list[UnitSummary]] = {}
        for unit in eligible_units:
            merge_name = unit.merge_group.strip()
            if merge_name:
                seed_key = f"s{unit.session_index}::merge::{merge_name}"
            else:
                seed_key = f"unit::{unit_record_key(unit)}"
            merge_seed_groups.setdefault(seed_key, []).append(unit)

        for seed_key, members in merge_seed_groups.items():
            member_keys = {unit_record_key(unit) for unit in members}
            components[seed_key] = member_keys
            component_sessions[seed_key] = {int(unit.session_index) for unit in members}
            for member_key in member_keys:
                component_for_key[member_key] = seed_key

        session_to_units: dict[int, list[UnitSummary]] = {}
        for unit in eligible_units:
            session_to_units.setdefault(int(unit.session_index), []).append(unit)

        sorted_session_indices = sorted(session_to_units)
        forced_pairs: list[tuple[float, float, float, float, str, str]] = []
        regular_pairs: list[tuple[float, float, float, float, str, str]] = []
        for left_index, left_session_index in enumerate(sorted_session_indices):
            for right_session_index in sorted_session_indices[left_index + 1 :]:
                for left in session_to_units[left_session_index]:
                    for right in session_to_units[right_session_index]:
                        waveform_score = compute_waveform_similarity(left, right)
                        amplitude_score = compute_amplitude_similarity(left, right)
                        autocorrelogram_score = compute_autocorrelogram_similarity(left, right)
                        if not self._passes_auto_align_pair(left, right):
                            continue
                        pair_info = (
                            compute_similarity(left, right),
                            waveform_score,
                            amplitude_score,
                            autocorrelogram_score,
                            unit_record_key(left),
                            unit_record_key(right),
                        )
                        if self._is_forced_auto_align_pair(left, right):
                            forced_pairs.append(pair_info)
                        else:
                            regular_pairs.append(pair_info)

        forced_pairs.sort(
            key=lambda item: (-item[1], -item[2], -item[3], -item[0], item[4], item[5])
        )
        regular_pairs.sort(
            key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4], item[5])
        )

        for pair_group in (forced_pairs, regular_pairs):
            for _score, _waveform_score, _amplitude_score, _autocorr_score, left_key, right_key in pair_group:
                left_component_key = component_for_key[left_key]
                right_component_key = component_for_key[right_key]
                if left_component_key == right_component_key:
                    continue
                left_sessions = component_sessions[left_component_key]
                right_sessions = component_sessions[right_component_key]
                if left_sessions & right_sessions:
                    continue

                merged_keys = components[left_component_key] | components[right_component_key]
                merged_sessions = left_sessions | right_sessions
                components[left_component_key] = merged_keys
                component_sessions[left_component_key] = merged_sessions
                for member_key in merged_keys:
                    component_for_key[member_key] = left_component_key
                del components[right_component_key]
                del component_sessions[right_component_key]

        rows: list[dict[int, list[UnitSummary]]] = []
        grouped_keys: set[str] = set()
        for component_keys in components.values():
            row_units = sorted(
                [units_lookup[key] for key in component_keys if key in units_lookup],
                key=lambda item: (item.session_index, item.unit_id),
            )
            if len({int(unit.session_index) for unit in row_units}) < 2:
                continue
            rows.append(self._build_row_from_units(row_units))
            grouped_keys.update(unit_record_key(unit) for unit in row_units)

        return sorted(rows, key=self._row_sort_key), grouped_keys

    def build_page_display_rows_local(self, page: PageSummary) -> list[dict[int, list[UnitSummary]]]:
        # Normal SG page row order:
        # 1. manual align rows
        # 2. auto rows
        # 3. leftover kept singletons
        # 4. noise rows
        #
        # Discarded units are excluded here and shown only on the global
        # discarded page.
        all_units = [unit for session in page.sessions for unit in session.units]
        kept_units = [unit for unit in all_units if not unit.is_discarded and not unit.is_noise]
        noise_units = [unit for unit in all_units if unit.is_noise and not unit.is_discarded]

        rows: list[dict[int, list[UnitSummary]]] = []
        manual_align_rows: dict[str, list[UnitSummary]] = {}
        for unit in sorted(kept_units, key=lambda item: (item.session_index, item.unit_id)):
            align_name = unit.align_group.strip()
            if not align_name:
                continue
            scoped_align = f"sh{unit.shank_id}_sg{unit.sg_channel}::{align_name}"
            manual_align_rows.setdefault(scoped_align, []).append(unit)

        rows.extend(
            self._build_row_from_units(units)
            for _group_name, units in sorted(
                manual_align_rows.items(),
                key=lambda item: (
                    min(unit.session_index for unit in item[1]),
                    min(unit.unit_id for unit in item[1]),
                ),
            )
        )

        auto_align_units = [
            unit
            for unit in kept_units
            if not unit.align_group.strip() and not unit.exclude_from_auto_align
        ]
        auto_align_rows, auto_aligned_keys = self._build_html_auto_align_rows(auto_align_units)
        rows.extend(auto_align_rows)

        remaining_kept = [
            unit
            for unit in kept_units
            if not unit.align_group.strip() and unit_record_key(unit) not in auto_aligned_keys
        ]
        for unit in sorted(remaining_kept, key=lambda item: (item.session_index, item.unit_id)):
            rows.append(self._build_row_from_units([unit]))

        noise_rows: list[dict[int, list[UnitSummary]]] = []
        noise_align_rows: dict[str, list[UnitSummary]] = {}
        assigned_noise: set[str] = set()
        for unit in sorted(noise_units, key=lambda item: (item.session_index, item.unit_id)):
            align_name = unit.align_group.strip()
            if not align_name:
                continue
            scoped_align = f"sh{unit.shank_id}_sg{unit.sg_channel}::{align_name}"
            noise_align_rows.setdefault(scoped_align, []).append(unit)
            assigned_noise.add(unit_record_key(unit))

        noise_rows.extend(
            self._build_row_from_units(units)
            for _group_name, units in sorted(
                noise_align_rows.items(),
                key=lambda item: (
                    min(unit.session_index for unit in item[1]),
                    min(unit.unit_id for unit in item[1]),
                ),
            )
        )
        for unit in sorted(noise_units, key=lambda item: (item.session_index, item.unit_id)):
            if unit_record_key(unit) in assigned_noise:
                continue
            noise_rows.append(self._build_row_from_units([unit]))

        rows.sort(key=self._row_sort_key)
        noise_rows.sort(key=self._row_sort_key)
        return rows + noise_rows

    def _page_cache_key(self, shank_id: int, page_id: str) -> str:
        return f"sh{int(shank_id)}::{page_id}"

    def _sorted_aliases(self, aliases: list[str]) -> list[str]:
        return sorted(
            aliases,
            key=lambda alias: int(alias[1:]) if len(alias) > 1 and alias[1:].isdigit() else alias,
        )

    def _get_stable_aliases_for_units(
        self,
        *,
        cache_key: str,
        ordered_units: list[UnitSummary],
    ) -> dict[str, str]:
        alias_by_unit_key = self._stable_unit_aliases.setdefault(cache_key, {})
        next_index = self._stable_next_alias_index.get(cache_key, 1)
        for unit in ordered_units:
            unit_key = unit_record_key(unit)
            if unit_key in alias_by_unit_key:
                continue
            alias_by_unit_key[unit_key] = f"u{next_index}"
            next_index += 1
        self._stable_next_alias_index[cache_key] = next_index
        return dict(alias_by_unit_key)

    def _build_alias_maps(
        self,
        page: PageSummary,
        display_rows: list[dict[int, list[UnitSummary]]],
    ) -> tuple[dict[str, UnitSummary], dict[str, list[UnitSummary]], dict[str, str]]:
        ordered_units = [
            unit
            for row_units in display_rows
            for session in page.sessions
            for unit in row_units.get(session.session_index, [])
        ]
        alias_by_unit_key = self._get_stable_aliases_for_units(
            cache_key=self._page_cache_key(page.shank_id, page.page_id),
            ordered_units=ordered_units,
        )
        unit_alias_map: dict[str, UnitSummary] = {}
        row_alias_map: dict[str, list[UnitSummary]] = {}

        for row_index, row_units in enumerate(display_rows, start=1):
            row_alias = f"r{row_index}"
            row_members: list[UnitSummary] = []
            visible_sessions = [
                session for session in page.sessions if row_units.get(session.session_index, [])
            ]
            for session in visible_sessions:
                for unit in row_units.get(session.session_index, []):
                    alias = alias_by_unit_key[unit_record_key(unit)]
                    unit_alias_map[alias] = unit
                    row_members.append(unit)
            row_alias_map[row_alias] = row_members

        return unit_alias_map, row_alias_map, alias_by_unit_key

    def build_page_payload(self, page: PageSummary) -> dict:
        # The browser payload is derived from the current display rows, so alias
        # maps, row kinds, and the visible card order all stay aligned.
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
            units_payload = []
            for unit in flat_units:
                units_payload.append(
                    {
                        "alias": alias_by_unit_key.get(unit_record_key(unit), ""),
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
                        "merge_group": unit.merge_group,
                        "align_group": unit.align_group,
                        "exclude_from_auto_align": bool(unit.exclude_from_auto_align),
                        "is_discarded": bool(unit.is_discarded),
                        "is_noise": bool(unit.is_noise),
                        "waveform_image_path": f"/image?path={quote(str(Path(unit.waveform_image_path).resolve()))}",
                    }
                )

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

    def _page_has_visible_units(self, page: PageSummary) -> bool:
        # This is the selector-visibility gate for SG pages. We intentionally
        # reuse the current row builder so the hidden/visible decision matches
        # what the user would actually see on that page.
        display_rows = self.build_page_display_rows_local(page)
        return any(
            row_units.get(session.session_index, [])
            for row_units in display_rows
            for session in page.sessions
        )

    def build_discarded_page_payload(self) -> dict:
        # The discarded view is global across all shanks. It still uses the
        # shared "rows" payload shape so the client can reuse most rendering
        # code, even though the page is conceptually a flat discarded gallery.
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
            sessions_present = []
            seen_sessions: set[str] = set()
            units_payload = []
            for unit in units:
                if unit.session_name not in seen_sessions:
                    sessions_present.append(unit.session_name)
                    seen_sessions.add(unit.session_name)
                units_payload.append(
                    {
                        "alias": alias_by_unit_key[unit_record_key(unit)],
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
                        "merge_group": unit.merge_group,
                        "align_group": unit.align_group,
                        "exclude_from_auto_align": bool(unit.exclude_from_auto_align),
                        "is_discarded": True,
                        "is_noise": bool(unit.is_noise),
                        "waveform_image_path": f"/image?path={quote(str(Path(unit.waveform_image_path).resolve()))}",
                    }
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

    def build_app_payload(self) -> dict:
        # Build the browser-facing state payload. Pages that would render with
        # no reviewable units are omitted from the selectors and tracked in the
        # hidden-page / hidden-shank summary fields instead.
        shank_entries = []
        selectable_pages = 0
        selectable_rows = 0
        hidden_pages: list[str] = []
        hidden_shank_ids: list[int] = []
        loaded_pages = sum(len(shank_pages) for shank_pages in self.pages_by_shank.values())

        for shank_id in sorted(self.pages_by_shank):
            page_entries = []
            for page in sorted(self.pages_by_shank[shank_id].values(), key=lambda item: item.sg_channel):
                if not self._page_has_visible_units(page):
                    hidden_pages.append(f"sh{page.shank_id} sg{page.sg_channel}")
                    continue
                page_payload = self.build_page_payload(page)
                selectable_pages += 1
                selectable_rows += len(page_payload["rows"])
                page_entries.append(page_payload)
            if page_entries:
                shank_entries.append({"shank_id": int(shank_id), "pages": page_entries})
            else:
                hidden_shank_ids.append(int(shank_id))

        discarded_page = self.build_discarded_page_payload()
        shank_entries.append({"shank_id": -1, "pages": [discarded_page]})

        return {
            "output_root": str(self.root_folder),
            "summary_root": str(self.summary_root),
            "summary": {
                "num_loaded_shanks": len(self.pages_by_shank),
                "num_selectable_shanks": len(self.pages_by_shank) - len(hidden_shank_ids),
                "num_loaded_pages": loaded_pages,
                "num_selectable_pages": selectable_pages,
                "num_selectable_rows": selectable_rows,
                "num_hidden_pages": len(hidden_pages),
                "num_hidden_shanks": len(hidden_shank_ids),
            },
            "empty_shank_folder_ids": self.empty_shank_folder_ids,
            "hidden_page_labels": hidden_pages,
            "hidden_shank_ids": hidden_shank_ids,
            "shanks": shank_entries,
        }

    def save_manifest_state(self) -> Path:
        manifest_path = self.summary_root / MANIFEST_NAME
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
        return manifest_path

    def save_manifest_state_for_all_pages(self) -> dict:
        # "All pages" keeps both:
        # 1. the batch-level manifest at the selected root, and
        # 2. per-shank manifests under each shank folder for compatibility with
        #    the older per-shank workflow.
        #
        # This save scope includes every loaded page/shank, even if a page is
        # currently hidden from the selectors because it has no reviewable
        # non-discarded units.
        root_manifest_path = self.save_manifest_state()
        shank_manifest_paths: list[str] = []
        for shank_id in sorted(self.pages_by_shank):
            shank_root = self.root_folder / f"sh{int(shank_id)}" / DEFAULT_EXPORT_FOLDER_NAME
            shank_root.mkdir(parents=True, exist_ok=True)
            shank_manifest_path = shank_root / MANIFEST_NAME
            shank_sessions = []
            for session in self.sessions:
                shank_units = [unit for unit in session.units if int(unit.shank_id) == int(shank_id)]
                shank_sessions.append(
                    {
                        "session_name": session.session_name,
                        "session_index": session.session_index,
                        "output_folder": session.output_folder,
                        "analyzer_folder": session.analyzer_folder,
                        "units": [asdict(unit) for unit in shank_units],
                    }
                )
            shank_payload = {
                "output_root": str(self.root_folder / f"sh{int(shank_id)}"),
                "sessions": shank_sessions,
            }
            shank_manifest_path.write_text(json.dumps(shank_payload, indent=2), encoding="utf-8")
            shank_manifest_paths.append(str(shank_manifest_path))
        return {
            "root_manifest_path": str(root_manifest_path),
            "shank_manifest_paths": shank_manifest_paths,
        }

    def _write_shank_manifest(self, shank_id: int) -> Path:
        shank_root = self.root_folder / f"sh{int(shank_id)}" / DEFAULT_EXPORT_FOLDER_NAME
        shank_root.mkdir(parents=True, exist_ok=True)
        shank_manifest_path = shank_root / MANIFEST_NAME
        shank_sessions = []
        for session in self.sessions:
            shank_units = [unit for unit in session.units if int(unit.shank_id) == int(shank_id)]
            shank_sessions.append(
                {
                    "session_name": session.session_name,
                    "session_index": session.session_index,
                    "output_folder": session.output_folder,
                    "analyzer_folder": session.analyzer_folder,
                    "units": [asdict(unit) for unit in shank_units],
                }
            )
        shank_payload = {
            "output_root": str(self.root_folder / f"sh{int(shank_id)}"),
            "sessions": shank_sessions,
        }
        shank_manifest_path.write_text(json.dumps(shank_payload, indent=2), encoding="utf-8")
        return shank_manifest_path

    def save_manifest_state_for_page(self, shank_id: int, page_id: str) -> tuple[Path, Path]:
        # Page save is intentionally narrow:
        # - only the current SG page is refreshed in the shared root manifest
        # - other pages/shanks already stored in that manifest are preserved
        # - the current shank's compatibility manifest is also refreshed so a
        #   later single-shank load sees the newest page decisions immediately
        #
        # In other words:
        # - page save writes root manifest + current shank manifest
        # - all-pages save writes root manifest + every shank manifest
        page = self.get_page(shank_id, page_id)
        manifest_path = self.summary_root / MANIFEST_NAME
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"output_root": str(self.root_folder), "sessions": []}
        else:
            payload = {"output_root": str(self.root_folder), "sessions": []}

        saved_sessions = {
            int(session_payload.get("session_index", -1)): session_payload
            for session_payload in payload.get("sessions", [])
            if session_payload.get("session_index") is not None
        }

        for session in self.sessions:
            session_payload = saved_sessions.setdefault(
                int(session.session_index),
                {
                    "session_name": session.session_name,
                    "session_index": int(session.session_index),
                    "output_folder": session.output_folder,
                    "analyzer_folder": session.analyzer_folder,
                    "units": [],
                },
            )
            session_payload["session_name"] = session.session_name
            session_payload["output_folder"] = session.output_folder
            session_payload["analyzer_folder"] = session.analyzer_folder

        for session in page.sessions:
            session_payload = saved_sessions.setdefault(
                int(session.session_index),
                {
                    "session_name": session.session_name,
                    "session_index": int(session.session_index),
                    "output_folder": session.output_folder,
                    "analyzer_folder": session.analyzer_folder,
                    "units": [],
                },
            )
            existing_units = [
                unit_payload
                for unit_payload in session_payload.get("units", [])
                if not (
                    int(unit_payload.get("shank_id", -1)) == int(page.shank_id)
                    and int(unit_payload.get("sg_channel", -1)) == int(page.sg_channel)
                )
            ]
            existing_units.extend(asdict(unit) for unit in session.units)
            session_payload["units"] = existing_units

        payload["output_root"] = str(self.root_folder)
        payload["sessions"] = [saved_sessions[index] for index in sorted(saved_sessions)]
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        shank_manifest_path = self._write_shank_manifest(int(page.shank_id))
        return manifest_path, shank_manifest_path

    def build_auto_align_lookup_multi_shank(self, min_similarity: float = 0.75) -> dict[str, str]:
        auto_align_lookup: dict[str, str] = {}
        for shank_pages in self.pages_by_shank.values():
            for page in shank_pages.values():
                eligible_units = [
                    unit
                    for session in page.sessions
                    for unit in session.units
                    if not unit.is_discarded
                    and not unit.is_noise
                    and not unit.align_group.strip()
                    and not unit.exclude_from_auto_align
                ]
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
                        auto_align_lookup[unit_record_key(unit)] = group_name
        return auto_align_lookup

    def final_group_key_for_unit(self, unit: UnitSummary, auto_align_lookup: dict[str, str] | None = None) -> str:
        session_tag = f"s{unit.session_index:03d}"
        base_id = f"{session_tag}_u{unit.unit_id}"
        merge_channel_tag = f"sh{unit.shank_id}_ch{unit.local_channel_on_shank}"
        merge_key = (
            f"{merge_channel_tag}__{session_tag}__merge__{sanitize_token(unit.merge_group)}"
            if unit.merge_group
            else base_id
        )
        if unit.align_group:
            align_channel_tag = f"sh{unit.shank_id}_sg{unit.sg_channel}"
            return f"{align_channel_tag}__align__{sanitize_token(unit.align_group)}"
        if auto_align_lookup is not None:
            auto_group_name = auto_align_lookup.get(unit_record_key(unit))
            if auto_group_name:
                return auto_group_name
        return merge_key

    def discard_group_key_for_unit(self, unit: UnitSummary) -> str:
        if unit.align_group:
            return f"discarded__sh{unit.shank_id}_sg{unit.sg_channel}__align__{sanitize_token(unit.align_group)}"
        if unit.merge_group:
            return (
                f"discarded__s{unit.session_index:03d}_sh{unit.shank_id}_ch{unit.local_channel_on_shank}"
                f"__merge__{sanitize_token(unit.merge_group)}"
            )
        return f"discarded__s{unit.session_index:03d}_u{unit.unit_id}"

    def build_unique_unit_summary_row(
        self,
        *,
        group_index: int,
        group_key: str,
        units: list[UnitSummary],
        group_folder: Path,
        copied_images: list[str],
        summary_path: Path,
    ) -> dict:
        sorted_units = sorted(units, key=lambda unit: (unit.session_index, unit.unit_id))
        representative = sorted_units[0]
        session_names = []
        seen_session_names: set[str] = set()
        members_by_session: dict[str, list[int]] = {}
        for unit in sorted_units:
            if unit.session_name not in seen_session_names:
                session_names.append(unit.session_name)
                seen_session_names.add(unit.session_name)
            members_by_session.setdefault(unit.session_name, []).append(int(unit.unit_id))

        return {
            "final_unit_id": group_index,
            "final_unit_label": f"unit_{group_index:04d}",
            "final_group_key": group_key,
            "export_folder": str(group_folder),
            "summary_path": str(summary_path),
            "shank_id": int(representative.shank_id),
            "channel": int(representative.local_channel_on_shank),
            "sg_channel": int(representative.sg_channel),
            "representative_session": representative.session_name,
            "representative_unit_id": int(representative.unit_id),
            "num_sessions": len(session_names),
            "sessions_present": session_names,
            "session_members": [
                {"session_name": session_name, "unit_ids": sorted(member_unit_ids)}
                for session_name, member_unit_ids in members_by_session.items()
            ],
            "num_member_units": len(sorted_units),
            "member_units": [
                {
                    "session_name": unit.session_name,
                    "session_index": int(unit.session_index),
                    "unit_id": int(unit.unit_id),
                    "merge_group": unit.merge_group,
                    "align_group": unit.align_group,
                    "waveform_image_path": unit.waveform_image_path,
                }
                for unit in sorted_units
            ],
            "representative_waveform_image": copied_images[0] if copied_images else "",
            "waveform_images": copied_images,
        }

    def build_discarded_unit_summary_row(self, *, group_key: str, units: list[UnitSummary]) -> dict:
        sorted_units = sorted(units, key=lambda unit: (unit.session_index, unit.unit_id))
        representative = sorted_units[0]
        session_names = []
        seen_session_names: set[str] = set()
        for unit in sorted_units:
            if unit.session_name not in seen_session_names:
                session_names.append(unit.session_name)
                seen_session_names.add(unit.session_name)

        return {
            "discard_group_key": group_key,
            "status": "discarded",
            "discard_reason": build_discard_reason(representative),
            "shank_id": int(representative.shank_id),
            "channel": int(representative.local_channel_on_shank),
            "sg_channel": int(representative.sg_channel),
            "num_sessions": len(session_names),
            "sessions_present": session_names,
            "num_member_units": len(sorted_units),
            "member_units": [
                {
                    "session_name": unit.session_name,
                    "session_index": int(unit.session_index),
                    "unit_id": int(unit.unit_id),
                    "amplitude_median": unit.amplitude_median,
                    "snr": unit.snr,
                    "isi_violations_ratio": unit.isi_violations_ratio,
                    "merge_group": unit.merge_group,
                    "align_group": unit.align_group,
                    "waveform_image_path": unit.waveform_image_path,
                }
                for unit in sorted_units
            ],
        }

    def build_noise_unit_summary_row(self, *, group_key: str, units: list[UnitSummary]) -> dict:
        sorted_units = sorted(units, key=lambda unit: (unit.session_index, unit.unit_id))
        representative = sorted_units[0]
        session_names = []
        seen_session_names: set[str] = set()
        for unit in sorted_units:
            if unit.session_name not in seen_session_names:
                session_names.append(unit.session_name)
                seen_session_names.add(unit.session_name)

        return {
            "noise_group_key": group_key,
            "status": "noise",
            "shank_id": int(representative.shank_id),
            "channel": int(representative.local_channel_on_shank),
            "sg_channel": int(representative.sg_channel),
            "num_sessions": len(session_names),
            "sessions_present": session_names,
            "num_member_units": len(sorted_units),
            "member_units": [
                {
                    "session_name": unit.session_name,
                    "session_index": int(unit.session_index),
                    "unit_id": int(unit.unit_id),
                    "amplitude_median": unit.amplitude_median,
                    "snr": unit.snr,
                    "isi_violations_ratio": unit.isi_violations_ratio,
                    "merge_group": unit.merge_group,
                    "align_group": unit.align_group,
                    "waveform_image_path": unit.waveform_image_path,
                }
                for unit in sorted_units
            ],
        }

    def write_unique_units_summary_csv(self, csv_path: Path, rows: list[dict]) -> None:
        fieldnames = [
            "final_unit_id", "final_unit_label", "shank_id", "channel", "sg_channel",
            "num_sessions", "sessions_present", "num_member_units", "member_units",
            "representative_session", "representative_unit_id", "representative_waveform_image",
            "export_folder", "summary_path", "final_group_key",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "final_unit_id": row["final_unit_id"],
                        "final_unit_label": row["final_unit_label"],
                        "shank_id": row["shank_id"],
                        "channel": row["channel"],
                        "sg_channel": row["sg_channel"],
                        "num_sessions": row["num_sessions"],
                        "sessions_present": "; ".join(row["sessions_present"]),
                        "num_member_units": row["num_member_units"],
                        "member_units": "; ".join(f"{item['session_name']}:u{item['unit_id']}" for item in row["member_units"]),
                        "representative_session": row["representative_session"],
                        "representative_unit_id": row["representative_unit_id"],
                        "representative_waveform_image": row["representative_waveform_image"],
                        "export_folder": row["export_folder"],
                        "summary_path": row["summary_path"],
                        "final_group_key": row["final_group_key"],
                    }
                )

    def write_discarded_units_summary_csv(self, csv_path: Path, rows: list[dict]) -> None:
        fieldnames = [
            "status", "discard_group_key", "discard_reason", "shank_id", "channel",
            "sg_channel", "num_sessions", "sessions_present", "num_member_units", "member_units",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "status": row["status"],
                        "discard_group_key": row["discard_group_key"],
                        "discard_reason": row["discard_reason"],
                        "shank_id": row["shank_id"],
                        "channel": row["channel"],
                        "sg_channel": row["sg_channel"],
                        "num_sessions": row["num_sessions"],
                        "sessions_present": "; ".join(row["sessions_present"]),
                        "num_member_units": row["num_member_units"],
                        "member_units": "; ".join(f"{item['session_name']}:u{item['unit_id']}" for item in row["member_units"]),
                    }
                )

    def write_noise_units_summary_csv(self, csv_path: Path, rows: list[dict]) -> None:
        fieldnames = [
            "status", "noise_group_key", "shank_id", "channel",
            "sg_channel", "num_sessions", "sessions_present", "num_member_units", "member_units",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "status": row["status"],
                        "noise_group_key": row["noise_group_key"],
                        "shank_id": row["shank_id"],
                        "channel": row["channel"],
                        "sg_channel": row["sg_channel"],
                        "num_sessions": row["num_sessions"],
                        "sessions_present": "; ".join(row["sessions_present"]),
                        "num_member_units": row["num_member_units"],
                        "member_units": "; ".join(f"{item['session_name']}:u{item['unit_id']}" for item in row["member_units"]),
                    }
                )

    def build_group_summary_text(self, group_index: int, group_key: str, units: list[UnitSummary]) -> str:
        session_names = []
        seen_session_names: set[str] = set()
        for unit in sorted(units, key=lambda item: (item.session_index, item.unit_id)):
            if unit.session_name not in seen_session_names:
                session_names.append(unit.session_name)
                seen_session_names.add(unit.session_name)
        lines = [
            f"Final unit #{group_index}",
            f"Group key: {group_key}",
            f"Shank: {units[0].shank_id}",
            f"Channel: {units[0].local_channel_on_shank}",
            f"SG channel: {units[0].sg_channel}",
            f"Appears in {len(session_names)} session(s): {', '.join(session_names)}",
            f"Total member units: {len(units)}",
            "",
            "Members:",
        ]
        for unit in units:
            lines.extend(
                [
                    f"- {unit.session_name} | unit {unit.unit_id}",
                    f"  amplitude_median={format_metric(unit.amplitude_median)}",
                    f"  firing_rate={format_metric(unit.firing_rate)}",
                    f"  isi_violations_ratio={format_metric(unit.isi_violations_ratio)}",
                    f"  snr={format_metric(unit.snr)}",
                    f"  num_spikes={format_metric(unit.num_spikes)}",
                    f"  is_discarded={unit.is_discarded}",
                    f"  merge_group={unit.merge_group or '<none>'}",
                    f"  align_group={unit.align_group or '<none>'}",
                    f"  analyzer_folder={unit.analyzer_folder}",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def export_summary_bundle(self) -> dict:
        # Combined summary export across every loaded shank. This is the
        # all-shank reporting bundle, not the per-page export format.
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
                noise_groups.setdefault(self.discard_group_key_for_unit(unit).replace("discarded__", "noise__"), []).append(unit)
                continue
            final_groups.setdefault(self.final_group_key_for_unit(unit, auto_align_lookup=auto_align_lookup), []).append(unit)

        manifest_rows = []
        unique_unit_rows = []
        discarded_unit_rows = []
        noise_unit_rows = []

        for group_index, (group_key, units) in enumerate(sorted(final_groups.items()), start=1):
            group_folder = export_folder / f"unit_{group_index:04d}"
            group_folder.mkdir(parents=True, exist_ok=True)
            representative = units[0]
            copied_images: list[str] = []
            for item_index, unit in enumerate(units, start=1):
                src = Path(unit.waveform_image_path)
                dst = group_folder / f"waveform_{item_index:02d}_{unit.session_name.replace(' ', '_')}_u{unit.unit_id}.png"
                if src.exists():
                    shutil.copy2(src, dst)
                    copied_images.append(str(dst))

            summary_path = group_folder / "summary.txt"
            summary_path.write_text(self.build_group_summary_text(group_index, group_key, units), encoding="utf-8")
            unique_unit_rows.append(
                self.build_unique_unit_summary_row(
                    group_index=group_index,
                    group_key=group_key,
                    units=units,
                    group_folder=group_folder,
                    copied_images=copied_images,
                    summary_path=summary_path,
                )
            )
            manifest_rows.append(
                {
                    "final_unit_id": group_index,
                    "final_group_key": group_key,
                    "export_folder": str(group_folder),
                    "representative_session": representative.session_name,
                    "representative_unit_id": representative.unit_id,
                    "shank_id": representative.shank_id,
                    "local_channel_on_shank": representative.local_channel_on_shank,
                    "members": [
                        {
                            "session_name": unit.session_name,
                            "session_index": unit.session_index,
                            "unit_id": unit.unit_id,
                            "merge_group": unit.merge_group,
                            "align_group": unit.align_group,
                            "output_folder": unit.output_folder,
                        }
                        for unit in units
                    ],
                    "images": copied_images,
                }
            )

        for group_key, units in sorted(discarded_groups.items()):
            discarded_unit_rows.append(self.build_discarded_unit_summary_row(group_key=group_key, units=units))
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

    def export_summary_bundle_for_page(self, shank_id: int, page_id: str) -> dict:
        # Page export writes only one SG page into the corresponding shank's
        # units_alignment_summary folder, using SG-specific filenames so
        # multiple page exports from the same shank do not overwrite each other.
        page = self.get_page(shank_id, page_id)
        page_summary_root = self.root_folder / f"sh{page.shank_id}" / DEFAULT_EXPORT_FOLDER_NAME
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
                page_auto_lookup[unit_record_key(unit)] = group_name

        final_groups: dict[str, list[UnitSummary]] = {}
        discarded_groups: dict[str, list[UnitSummary]] = {}
        noise_groups: dict[str, list[UnitSummary]] = {}
        for unit in page_units:
            if unit.is_discarded:
                discarded_groups.setdefault(self.discard_group_key_for_unit(unit), []).append(unit)
                continue
            if unit.is_noise:
                noise_groups.setdefault(self.discard_group_key_for_unit(unit).replace("discarded__", "noise__"), []).append(unit)
                continue
            final_groups.setdefault(self.final_group_key_for_unit(unit, auto_align_lookup=page_auto_lookup), []).append(unit)

        manifest_rows = []
        unique_unit_rows = []
        discarded_unit_rows = []
        noise_unit_rows = []

        for group_index, (group_key, units) in enumerate(sorted(final_groups.items()), start=1):
            group_folder = export_folder / f"unit_{group_index:04d}"
            group_folder.mkdir(parents=True, exist_ok=True)
            representative = units[0]
            copied_images: list[str] = []
            for item_index, unit in enumerate(units, start=1):
                src = Path(unit.waveform_image_path)
                dst = group_folder / f"waveform_{item_index:02d}_{unit.session_name.replace(' ', '_')}_u{unit.unit_id}.png"
                if src.exists():
                    shutil.copy2(src, dst)
                    copied_images.append(str(dst))

            summary_path = group_folder / "summary.txt"
            summary_path.write_text(self.build_group_summary_text(group_index, group_key, units), encoding="utf-8")
            unique_unit_rows.append(
                self.build_unique_unit_summary_row(
                    group_index=group_index,
                    group_key=group_key,
                    units=units,
                    group_folder=group_folder,
                    copied_images=copied_images,
                    summary_path=summary_path,
                )
            )
            manifest_rows.append(
                {
                    "final_unit_id": group_index,
                    "final_group_key": group_key,
                    "export_folder": str(group_folder),
                    "representative_session": representative.session_name,
                    "representative_unit_id": representative.unit_id,
                    "shank_id": representative.shank_id,
                    "local_channel_on_shank": representative.local_channel_on_shank,
                    "members": [
                        {
                            "session_name": unit.session_name,
                            "session_index": unit.session_index,
                            "unit_id": unit.unit_id,
                            "merge_group": unit.merge_group,
                            "align_group": unit.align_group,
                            "output_folder": unit.output_folder,
                        }
                        for unit in units
                    ],
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

    def export_all_pages_decisions(self) -> dict:
        # This exports page-style outputs for every loaded SG page, including
        # pages that are hidden in the UI because they have no reviewable
        # non-discarded units. That keeps the per-shank page export set
        # complete, while `export_summary_bundle()` remains the single combined
        # all-shank summary export.
        page_export_results: list[dict] = []
        for shank_id in sorted(self.pages_by_shank):
            for page in sorted(self.pages_by_shank[shank_id].values(), key=lambda item: item.sg_channel):
                page_export_results.append(
                    self.export_summary_bundle_for_page(
                        shank_id=int(shank_id),
                        page_id=page.page_id,
                    )
                )
        return {
            "num_pages_exported": len(page_export_results),
            "page_exports": page_export_results,
        }

    def get_page(self, shank_id: int, page_id: str) -> PageSummary:
        if page_id == "__discarded_all__":
            raise ValueError("Commands are only available on SG channel pages, not the discarded-units page.")
        shank_pages = self.pages_by_shank.get(int(shank_id))
        if not shank_pages or page_id not in shank_pages:
            raise ValueError(f"Unknown page: shank={shank_id}, page={page_id}")
        return shank_pages[page_id]

    def _resolve_aliases_for_page(self, page: PageSummary) -> tuple[dict[str, UnitSummary], dict[str, list[UnitSummary]]]:
        display_rows = self.build_page_display_rows_local(page)
        unit_alias_map, row_alias_map, _alias_by_unit_key = self._build_alias_maps(page, display_rows)
        return unit_alias_map, row_alias_map

    def resolve_command_units(self, page: PageSummary, alias_tokens: list[str]) -> list[UnitSummary]:
        if not alias_tokens:
            raise ValueError("Add at least one alias such as u1 or r1.")
        unit_alias_map, row_alias_map = self._resolve_aliases_for_page(page)
        resolved: list[UnitSummary] = []
        seen: set[str] = set()
        for alias in alias_tokens:
            key = alias.strip().lower()
            if key in row_alias_map:
                for unit in row_alias_map[key]:
                    unit_key = unit_record_key(unit)
                    if unit_key in seen:
                        continue
                    seen.add(unit_key)
                    resolved.append(unit)
                continue
            unit = unit_alias_map.get(key)
            if unit is None:
                raise ValueError(f"Unknown alias: {alias}")
            unit_key = unit_record_key(unit)
            if unit_key in seen:
                continue
            seen.add(unit_key)
            resolved.append(unit)
        return resolved

    def resolve_single_command_unit(self, page: PageSummary, alias_token: str) -> UnitSummary:
        units = self.resolve_command_units(page, [alias_token])
        if len(units) != 1:
            raise ValueError(f"{alias_token} resolved to {len(units)} units. Use a single unit alias like u1 for similarity.")
        return units[0]

    def build_existing_group_members(self, attr_name: str) -> dict[str, set[str]]:
        groups: dict[str, set[str]] = {}
        for unit in self._iter_all_units():
            group_name = getattr(unit, attr_name, "").strip()
            if not group_name:
                continue
            scope = (
                f"sh{unit.shank_id}_sg{unit.sg_channel}"
                if attr_name == "align_group"
                else f"s{unit.session_index}_sh{unit.shank_id}_sg{unit.sg_channel}"
            )
            groups.setdefault(f"{scope}::{group_name}", set()).add(unit_record_key(unit))
        return groups

    def assign_units_to_group(self, *, attr_name: str, base_name: str, scope_tag: str, selected_units: list[UnitSummary], expand_existing_members: bool = True) -> tuple[str, int]:
        if len(selected_units) < 2:
            raise ValueError("At least two units are required.")
        units_lookup = {unit_record_key(unit): unit for unit in self._iter_all_units()}
        existing_members = self.build_existing_group_members(attr_name)
        if attr_name == "align_group":
            scope_prefix = f"{scope_tag}::"
        else:
            scope_prefix = f"s{selected_units[0].session_index}_sh{selected_units[0].shank_id}_sg{selected_units[0].sg_channel}::"
        existing_names = {
            scoped_key.split("::", maxsplit=1)[1]
            for scoped_key in existing_members
            if scoped_key.startswith(scope_prefix)
        }
        selected_keys = {unit_record_key(unit) for unit in selected_units}
        existing_names_in_selection = sorted({getattr(unit, attr_name).strip() for unit in selected_units if getattr(unit, attr_name).strip()})
        expanded_keys = set(selected_keys)
        if expand_existing_members:
            for name in existing_names_in_selection:
                expanded_keys.update(existing_members.get(f"{scope_prefix}{name}", set()))
        if existing_names_in_selection:
            group_name = existing_names_in_selection[0]
        else:
            next_index = 1
            group_name = f"{base_name}_{next_index:02d}"
            while group_name in existing_names:
                next_index += 1
                group_name = f"{base_name}_{next_index:02d}"
        for unit_key in expanded_keys:
            unit = units_lookup.get(unit_key)
            if unit is None:
                continue
            setattr(unit, attr_name, group_name)
            if attr_name == "align_group":
                unit.exclude_from_auto_align = False
        return group_name, len(expanded_keys)

    def run_page_command(self, page: PageSummary, command_name: str, alias_tokens: list[str]) -> tuple[str, bool]:
        normalized = command_name.strip().lower()
        if normalized in {"similarity", "similarities", "compare"}:
            if len(alias_tokens) != 2:
                raise ValueError("similarity needs exactly two unit aliases, for example: similarity u1 u2")
            left = self.resolve_single_command_unit(page, alias_tokens[0])
            right = self.resolve_single_command_unit(page, alias_tokens[1])
            return (
                f"similarity {alias_tokens[0]} vs {alias_tokens[1]} | "
                f"waveform={compute_waveform_similarity(left, right):.3f}, "
                f"amplitude={compute_amplitude_similarity(left, right):.3f}, "
                f"autocorrelogram={compute_autocorrelogram_similarity(left, right):.3f}, "
                f"total={compute_similarity(left, right):.3f}",
                False,
            )

        units = self.resolve_command_units(page, alias_tokens)
        if normalized == "align":
            session_indices = {unit.session_index for unit in units}
            if len(units) < 2:
                raise ValueError("align needs at least two units.")
            if len(session_indices) == 1:
                group_name, unit_count = self.assign_units_to_group(
                    attr_name="merge_group",
                    base_name=f"merge_s{units[0].session_index:03d}_sh{page.shank_id}_sg{page.sg_channel}",
                    scope_tag=f"s{units[0].session_index}_sh{page.shank_id}_sg{page.sg_channel}",
                    selected_units=units,
                    expand_existing_members=True,
                )
                return (
                    f"same-session align converted to merge {group_name} on {unit_count} unit(s)",
                    True,
                )
            group_name, unit_count = self.assign_units_to_group(
                attr_name="align_group",
                base_name=f"align_sh{page.shank_id}_sg{page.sg_channel}",
                scope_tag=f"sh{page.shank_id}_sg{page.sg_channel}",
                selected_units=units,
                expand_existing_members=False,
            )
            return f"align {group_name} on {unit_count} unit(s)", True
        if normalized == "unalign":
            cleared_count = 0
            cleared_merge_count = 0
            for unit in units:
                if unit.align_group.strip():
                    cleared_count += 1
                unit.align_group = ""
                if unit.merge_group.strip():
                    cleared_merge_count += 1
                unit.merge_group = ""
                unit.exclude_from_auto_align = True
            return (
                f"cleared alignment on {cleared_count} unit(s) and merge on {cleared_merge_count} unit(s)",
                True,
            )
        if normalized == "merge":
            session_indices = {unit.session_index for unit in units}
            if len(units) < 2:
                raise ValueError("merge needs at least two units.")
            if len(session_indices) != 1:
                raise ValueError("merge needs units from one session only.")
            group_name, unit_count = self.assign_units_to_group(
                attr_name="merge_group",
                base_name=f"merge_s{units[0].session_index:03d}_sh{page.shank_id}_sg{page.sg_channel}",
                scope_tag=f"s{units[0].session_index}_sh{page.shank_id}_sg{page.sg_channel}",
                selected_units=units,
                expand_existing_members=True,
            )
            return f"merge {group_name} on {unit_count} unit(s)", True
        if normalized == "unmerge":
            cleared_count = 0
            for unit in units:
                if unit.merge_group.strip():
                    cleared_count += 1
                unit.merge_group = ""
            return f"cleared merge on {cleared_count} unit(s)", True
        if normalized == "noise":
            for unit in units:
                unit.is_noise = True
            return f"marked {len(units)} unit(s) as noise", True
        if normalized in {"clear_noise", "cleannoise", "denoise"}:
            for unit in units:
                unit.is_noise = False
            return f"cleared noise on {len(units)} unit(s)", True
        raise ValueError(f"Unknown command: {command_name}")

    def apply_commands(self, shank_id: int, page_id: str, raw_text: str) -> dict:
        with self._lock:
            page = self.get_page(shank_id, page_id)
            before_snapshot = self.snapshot()
            messages: list[str] = []
            changed_state = False
            # Commands are applied top-to-bottom as one batch. If any command
            # changes state, we recompute discard status and generated
            # auto-merge groups once at the end, then store one undo snapshot.
            for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = line.split()
                if len(tokens) < 2:
                    raise ValueError(f"Line {line_number}: expected a command followed by one or more aliases.")
                result_text, command_changed_state = self.run_page_command(page, tokens[0], tokens[1:])
                messages.append(f"Line {line_number}: {result_text}")
                changed_state = changed_state or command_changed_state
            if not messages:
                raise ValueError("Only blank lines or comments were provided.")
            if changed_state:
                for unit in self._iter_all_units():
                    unit.is_discarded = is_unit_auto_discarded(unit)
                self.sync_auto_merge_groups()
                self.push_undo_snapshot(before_snapshot)
            return {"messages": messages, "changed_state": changed_state, "app": self.build_app_payload()}


def build_html_shell() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{HTML_TITLE}</title>
  <style>
    :root {{
      --bg: #efe8de; --panel: #fffaf3; --line: #d9cebf; --ink: #1d1915; --muted: #675f56;
      --accent: #0f6a5a; --accent-soft: #d9efe8; --warn: #8b5b17; --warn-soft: #f6e8d3;
      --danger: #9b4336; --danger-soft: #f6dfda; --shadow: 0 14px 34px rgba(46,31,18,0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family:"Segoe UI",sans-serif; color:var(--ink);
      background: radial-gradient(circle at top left, #f8ead2 0, transparent 26%),
                  radial-gradient(circle at top right, #e3efe6 0, transparent 24%),
                  linear-gradient(180deg, #f5eee6 0%, var(--bg) 100%); }}
    .shell {{ max-width:1800px; margin:0 auto; padding:22px; }}
    .hero,.toolbar,.panel {{ background:rgba(255,250,243,0.9); border:1px solid var(--line); border-radius:22px; box-shadow:var(--shadow); }}
    .hero {{ padding:22px; margin-bottom:16px; }}
    .hero h1 {{ margin:0 0 6px; font-size:2rem; }}
    .muted {{ color:var(--muted); }}
    .notice {{ margin-top:12px; padding:12px 14px; border-radius:14px; background:var(--warn-soft); border:1px solid var(--warn); color:var(--warn); }}
    .stats,.actions,.command-buttons,.pills,.badges,.unit-tags {{ display:flex; gap:8px; flex-wrap:wrap; }}
    .stats {{ margin-top:14px; }}
    .stat,.pill,.badge,.tag {{ border:1px solid var(--line); background:white; }}
    .stat {{ border-radius:14px; padding:10px 14px; min-width:130px; }}
    .toolbar {{ padding:16px; margin-bottom:16px; display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:12px; }}
    .toolbar label {{ display:block; font-size:0.84rem; color:var(--muted); margin-bottom:6px; }}
    .toolbar select,.toolbar input,textarea {{ width:100%; border:1px solid var(--line); border-radius:12px; padding:10px 12px; font:inherit; background:white; }}
    .layout {{ display:grid; grid-template-columns:320px minmax(0,1fr); gap:16px; align-items:start; }}
    .panel {{ padding:16px; }}
    .command-panel {{ position: sticky; top: 18px; align-self: start; max-height: calc(100vh - 36px); overflow: auto; }}
    .panel h2 {{ margin:0 0 12px; font-size:1.05rem; }}
    button {{ border:1px solid var(--line); border-radius:12px; background:white; color:var(--ink); padding:10px 12px; font:inherit; cursor:pointer; }}
    button.primary {{ background:var(--accent); color:white; border-color:var(--accent); }}
    button.warn {{ background:var(--warn-soft); color:var(--warn); }}
    button:disabled {{ opacity:0.65; cursor:wait; }}
    .log {{ margin-top:12px; border-top:1px solid var(--line); padding-top:12px; white-space:pre-wrap; color:var(--muted); min-height:120px; }}
    .page-header {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:12px; }}
    .page-header h2 {{ margin:0; font-size:1.35rem; }}
    .pill,.badge,.tag {{ border-radius:999px; padding:5px 10px; font-size:0.8rem; }}
    .rows {{ display:grid; gap:14px; }}
    .bottom-nav {{ display:flex; justify-content:flex-end; gap:8px; margin-top:16px; }}
    .row-card {{ border:1px solid var(--line); border-radius:20px; overflow:hidden; background:rgba(255,255,255,0.75); }}
    .row-card.manual_align {{ border-color:#2d8a70; }} .row-card.auto_align {{ border-color:#2b73a2; }}
    .row-card.noise {{ border-color:var(--warn); }} .row-card.singleton {{ border-color:#b8aea2; }} .row-card.discarded {{ border-color:var(--danger); }}
    .row-summary {{ display:flex; justify-content:space-between; gap:12px; padding:14px 16px; background:rgba(255,250,243,0.95); border-bottom:1px solid var(--line); }}
    .unit-grid {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; padding:16px; }}
    .unit-card {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:12px; min-height:100%; }}
    .unit-card.merged {{ border:2px solid #2b73a2; box-shadow: inset 0 0 0 1px rgba(43,115,162,0.15); }}
    .unit-head {{ display:flex; justify-content:space-between; gap:10px; margin-bottom:8px; }}
    .unit-name {{ font-weight:600; }}
    .tag.align {{ background:var(--accent-soft); color:var(--accent); }} .tag.noise {{ background:var(--warn-soft); color:var(--warn); }} .tag.discarded {{ background:var(--danger-soft); color:var(--danger); }}
    .metrics {{ color:var(--muted); font-size:0.9rem; line-height:1.35; margin-bottom:8px; display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:4px 12px; }}
    img {{ width:100%; border-radius:12px; border:1px solid var(--line); background:white; }}
    .lightbox {{ position:fixed; inset:0; display:none; align-items:center; justify-content:center; padding:24px; background:rgba(20,16,12,0.72); z-index:9999; }}
    .lightbox.open {{ display:flex; }}
    .lightbox-card {{ max-width:min(1100px, 94vw); max-height:94vh; background:var(--panel); border:1px solid var(--line); border-radius:20px; box-shadow:var(--shadow); overflow:hidden; }}
    .lightbox-card img {{ display:block; width:auto; max-width:min(1100px, 94vw); max-height:calc(94vh - 56px); border:none; border-radius:0; }}
    .lightbox-caption {{ padding:14px 16px; color:var(--muted); border-top:1px solid var(--line); background:white; }}
    .empty {{ border:1px dashed var(--line); border-radius:18px; padding:30px; text-align:center; color:var(--muted); }}
    @media (max-width:1600px) {{ .unit-grid {{ grid-template-columns:repeat(3, minmax(0, 1fr)); }} }}
    @media (max-width:1200px) {{ .layout {{ grid-template-columns:1fr; }} .toolbar {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .unit-grid {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width:700px) {{ .toolbar {{ grid-template-columns:1fr; }} .shell {{ padding:12px; }} .unit-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>{HTML_TITLE}</h1>
      <div class="muted" id="root-path"></div>
      <div id="empty-shanks-notice"></div>
      <div class="stats">
        <div class="stat"><strong id="stat-shanks"></strong><div class="muted">Shanks</div></div>
        <div class="stat"><strong id="stat-pages"></strong><div class="muted">Pages</div></div>
        <div class="stat"><strong id="stat-rows"></strong><div class="muted">Display Rows</div></div>
      </div>
    </section>
    <section class="toolbar">
      <div><label for="shank-select">Shank</label><select id="shank-select"></select><div class="actions" style="margin-top:8px;"><button id="prev-shank-btn">Previous Shank</button><button id="next-shank-btn">Next Shank</button><span class="muted" id="shank-nav-message"></span></div></div>
      <div><label for="page-select">Page</label><select id="page-select"></select><div class="actions" style="margin-top:8px;"><button id="prev-page-btn">Previous Page</button><button id="next-page-btn">Next Page</button><span class="muted" id="page-nav-message"></span></div></div>
      <div><label for="kind-select">Row Kind</label><select id="kind-select"><option value="all">All rows</option><option value="manual_align">Manual align</option><option value="auto_align">Auto align</option><option value="singleton">Singleton</option><option value="noise">Noise</option><option value="discarded">Discarded</option></select></div>
      <div><label for="search-input">Search</label><input id="search-input" type="text" placeholder="session, unit, align, merge"></div>
      <div><label>Summary Folder</label><div class="pill" id="summary-root"></div></div>
    </section>
    <div class="layout">
      <section class="panel command-panel">
        <h2>Command Panel</h2>
        <div class="command-buttons">
          <button data-command="align">Align</button><button data-command="unalign">Unalign</button><button data-command="merge">Merge</button><button data-command="unmerge">Unmerge</button><button data-command="noise">Noise</button><button data-command="clear_noise">Clear Noise</button><button data-command="similarity">Similarity</button>
        </div>
        <textarea id="command-text" rows="9" placeholder="align r2 u11&#10;merge r4&#10;similarity u1 u2"></textarea>
        <div class="actions">
          <button class="primary" id="apply-btn">Apply Commands</button><button id="clear-btn">Clear Commands</button><button id="undo-btn">Undo</button>
        </div>
        <div class="actions" style="margin-top:8px;">
          <button id="save-page-btn">Save Page Decisions</button><button class="warn" id="export-page-btn">Export Page</button>
        </div>
        <div class="actions" style="margin-top:8px;">
          <button id="save-all-pages-btn">Save All Pages Decisions</button><button class="warn" id="export-all-pages-btn">Export All Pages Decisions</button>
        </div>
        <div class="actions" style="margin-top:8px;">
          <button id="save-summary-btn">Save Summary</button><button class="warn" id="export-summary-btn">Export Summary</button>
        </div>
        <div class="log" id="command-log">One line per command. Commands run top to bottom once.
Use aliases like u1 and r1 from the current page.</div>
      </section>
      <section class="panel"><div id="app"></div></section>
    </div>
  </div>
  <div class="lightbox" id="lightbox">
    <div class="lightbox-card">
      <img id="lightbox-image" alt="Expanded unit waveform">
      <div class="lightbox-caption" id="lightbox-caption"></div>
    </div>
  </div>
  <script>
    let DATA = null;
    const shankSelect = document.getElementById("shank-select");
    const pageSelect = document.getElementById("page-select");
    const kindSelect = document.getElementById("kind-select");
    const searchInput = document.getElementById("search-input");
    const commandText = document.getElementById("command-text");
    const commandLog = document.getElementById("command-log");
    const app = document.getElementById("app");
    const lightbox = document.getElementById("lightbox");
    const lightboxImage = document.getElementById("lightbox-image");
    const lightboxCaption = document.getElementById("lightbox-caption");
    const pageNavMessage = document.getElementById("page-nav-message");
    const shankNavMessage = document.getElementById("shank-nav-message");
    const busyButtonLabels = new Map();
    function metric(value, digits = 3) {{ return value === null || value === undefined ? "nan" : Number(value).toFixed(digits); }}
    function currentShank() {{ return DATA.shanks.find((entry) => String(entry.shank_id) === shankSelect.value) || DATA.shanks[0]; }}
    function currentPage() {{ const shank = currentShank(); return shank.pages.find((page) => page.page_id === pageSelect.value) || shank.pages[0]; }}
    function setShankNavMessage(message) {{ shankNavMessage.textContent = message || ""; const bottomMessage = document.getElementById("bottom-shank-nav-message"); if (bottomMessage) bottomMessage.textContent = message || ""; }}
    function goToShank(delta) {{ const currentIndex = DATA.shanks.findIndex((entry) => String(entry.shank_id) === shankSelect.value); const nextIndex = Math.max(0, Math.min(DATA.shanks.length - 1, currentIndex + delta)); if (nextIndex === currentIndex) {{ if (delta > 0) setShankNavMessage("This is the last shank."); else if (delta < 0) setShankNavMessage("This is the first shank."); return; }} setShankNavMessage(""); setPageNavMessage(""); shankSelect.value = String(DATA.shanks[nextIndex].shank_id); populatePages(); clearCommandsForNavigation(); render(); }}
    function setPageNavMessage(message) {{ pageNavMessage.textContent = message || ""; const bottomMessage = document.getElementById("bottom-page-nav-message"); if (bottomMessage) bottomMessage.textContent = message || ""; }}
    function goToPage(delta) {{ const pages = currentShank().pages; const currentIndex = pages.findIndex((page) => page.page_id === pageSelect.value); const nextIndex = Math.max(0, Math.min(pages.length - 1, currentIndex + delta)); if (nextIndex === currentIndex) {{ if (delta > 0) setPageNavMessage("This is the last page of this shank."); else if (delta < 0) setPageNavMessage("This is the first page of this shank."); return; }} setPageNavMessage(""); pageSelect.value = pages[nextIndex].page_id; clearCommandsForNavigation(); render(); }}
    function setLog(lines) {{ commandLog.textContent = Array.isArray(lines) ? lines.join("\\n") : String(lines || ""); }}
    function clearCommandsForNavigation() {{ commandText.value = ""; setLog("Cleared command input for the newly selected page."); }}
    function setButtonsBusy(buttonIds, busyText) {{ buttonIds.forEach((buttonId) => {{ const button = document.getElementById(buttonId); if (!button) return; if (!busyButtonLabels.has(buttonId)) busyButtonLabels.set(buttonId, button.textContent); button.disabled = true; if (busyText) button.textContent = busyText; }}); }}
    function restoreButtons(buttonIds) {{ buttonIds.forEach((buttonId) => {{ const button = document.getElementById(buttonId); if (!button) return; button.disabled = false; if (busyButtonLabels.has(buttonId)) button.textContent = busyButtonLabels.get(buttonId); }}); }}
    async function fetchJson(url, options = undefined) {{ const response = await fetch(url, options); const payload = await response.json(); if (!response.ok) throw new Error(payload.error || "Request failed"); return payload; }}
    function populateShanks() {{ shankSelect.innerHTML = ""; DATA.shanks.forEach((entry) => {{ const option = document.createElement("option"); option.value = String(entry.shank_id); option.textContent = Number(entry.shank_id) === -1 ? "Discarded Units" : `Shank ${{entry.shank_id}}`; shankSelect.appendChild(option); }}); }}
    function populatePages() {{ pageSelect.innerHTML = ""; currentShank().pages.forEach((page) => {{ const option = document.createElement("option"); option.value = page.page_id; option.textContent = page.page_type === "discarded" ? "Discarded Units" : `SG ${{page.sg_channel}}`; pageSelect.appendChild(option); }}); }}
    function renderUnitCard(unit) {{ const tags = []; const hasAutoMerge = unit.merge_group && (unit.merge_group.startsWith("__alignmerge__") || unit.merge_group.startsWith("__automerge__")); const hasManualMerge = unit.merge_group && !hasAutoMerge; if (unit.align_group) tags.push(`<span class="tag align">align=${{unit.align_group}}</span>`); if (hasManualMerge) tags.push(`<span class="tag">merge=${{unit.merge_group}}</span>`); else if (hasAutoMerge) tags.push('<span class="tag auto">auto-merge</span>'); if (unit.is_discarded) tags.push('<span class="tag discarded">discarded</span>'); if (unit.is_noise && !unit.is_discarded) tags.push('<span class="tag noise">noise</span>'); const cardClass = hasManualMerge ? "unit-card merged" : "unit-card"; const zoomLabel = `${{unit.alias}} | ${{unit.session_name}} | u${{unit.unit_id}} | sh${{unit.shank_id}} | sg${{unit.sg_channel}}`; return `<article class="${{cardClass}}" data-image-src="${{unit.waveform_image_path}}" data-image-label="${{zoomLabel}}"><div class="unit-head"><div><div class="unit-name">${{unit.alias}} | ${{unit.session_name}} | u${{unit.unit_id}}</div><div class="muted">sh${{unit.shank_id}} | sg${{unit.sg_channel}}</div></div></div><div class="unit-tags">${{tags.join("")}}</div><div class="metrics"><div>FR: ${{metric(unit.firing_rate)}} Hz</div><div>SNR: ${{metric(unit.snr)}}</div><div><strong>Amp: ${{metric(unit.amplitude_median)}}</strong></div><div>ISI: ${{metric(unit.isi_violations_ratio)}}</div><div>Spikes: ${{unit.num_spikes ?? "nan"}}</div></div><img loading="lazy" src="${{unit.waveform_image_path}}" alt="${{unit.session_name}} unit ${{unit.unit_id}}"></article>`; }}
    function openLightbox(imageSrc, label) {{ lightboxImage.src = imageSrc; lightboxCaption.textContent = label || ""; lightbox.classList.add("open"); }}
    function closeLightbox() {{ lightbox.classList.remove("open"); lightboxImage.removeAttribute("src"); lightboxCaption.textContent = ""; }}
    function render() {{ document.getElementById("root-path").textContent = DATA.output_root; document.getElementById("summary-root").textContent = DATA.summary_root; document.getElementById("stat-shanks").textContent = `${{DATA.summary.num_selectable_shanks}} selectable / ${{DATA.summary.num_loaded_shanks}} loaded`; document.getElementById("stat-pages").textContent = `${{DATA.summary.num_selectable_pages}} selectable / ${{DATA.summary.num_loaded_pages}} loaded`; document.getElementById("stat-rows").textContent = DATA.summary.num_selectable_rows; const emptyNotice = document.getElementById("empty-shanks-notice"); const noticeParts = []; if (DATA.empty_shank_folder_ids && DATA.empty_shank_folder_ids.length) noticeParts.push(`No sorted units were loaded for shank folder(s): ${{DATA.empty_shank_folder_ids.map((item) => `sh${{item}}`).join(", ")}}`); if (DATA.hidden_shank_ids && DATA.hidden_shank_ids.length) noticeParts.push(`No reviewable non-discarded pages for shank(s): ${{DATA.hidden_shank_ids.map((item) => `sh${{item}}`).join(", ")}}`); if (DATA.hidden_page_labels && DATA.hidden_page_labels.length) noticeParts.push(`Hidden empty pages: ${{DATA.hidden_page_labels.join(", ")}}`); if (noticeParts.length) {{ emptyNotice.innerHTML = noticeParts.map((text) => `<div class="notice">${{text}}</div>`).join(""); }} else {{ emptyNotice.innerHTML = ""; }} const page = currentPage(); const isDiscardedPage = page.page_type === "discarded"; const currentShankId = Number(currentShank().shank_id); const pageHeading = currentShankId === -1 ? page.title : `Shank ${{page.shank_id}} | ${{page.title}}`; const kindFilter = kindSelect.value; const search = searchInput.value.trim().toLowerCase(); const rows = page.rows.filter((row) => {{ if (kindFilter !== "all" && row.row_kind !== kindFilter) return false; if (!search) return true; const haystack = [row.row_alias, row.row_kind, ...row.sessions_present, ...row.units.map((unit) => `${{unit.alias}} ${{unit.session_name}} u${{unit.unit_id}} sh${{unit.shank_id}} sg${{unit.sg_channel}} ${{unit.align_group}} ${{unit.merge_group}}`)].join(" ").toLowerCase(); return haystack.includes(search); }}); const visibleUnits = rows.flatMap((row) => row.units); const summaryText = isDiscardedPage ? `${{page.summary.total_discarded_units}} discarded unit(s) across all shanks` : `${{page.summary.total_units}} unit(s) on this channel, ${{page.summary.total_discarded_units}} auto-discarded`; if (!(isDiscardedPage ? visibleUnits.length : rows.length)) {{ app.innerHTML = `<div class="empty">${{isDiscardedPage ? "No discarded units match the current filters." : "No rows match the current filters."}}</div>`; document.getElementById("apply-btn").disabled = isDiscardedPage; document.getElementById("save-page-btn").disabled = isDiscardedPage; document.getElementById("export-page-btn").disabled = isDiscardedPage; return; }} const contentHtml = isDiscardedPage ? `<div class="unit-grid">${{visibleUnits.map(renderUnitCard).join("")}}</div>` : `<div class="rows">${{rows.map((row) => `<section class="row-card ${{row.row_kind}}"><div class="row-summary"><div><strong>Row ${{row.row_index}} | ${{row.row_alias}}</strong><div class="muted">Shown in: ${{row.sessions_present.join(", ") || "none"}}</div></div><div class="badges"><span class="badge">${{row.row_kind}}</span><span class="badge">${{row.num_units}} unit(s)</span></div></div><div class="unit-grid">${{row.units.map(renderUnitCard).join("")}}</div></section>`).join("")}}</div>`; app.innerHTML = `<div class="page-header"><div><h2>${{pageHeading}}</h2><div class="muted">${{summaryText}}</div></div><div class="pills"><span class="pill">${{isDiscardedPage ? visibleUnits.length : rows.length}} visible ${{isDiscardedPage ? "unit(s)" : "row(s)"}}</span><span class="pill">${{page.available_unit_aliases.length}} unit alias(es)</span>${{isDiscardedPage ? "" : `<span class="pill">${{page.available_row_aliases.length}} row alias(es)</span>`}}${{isDiscardedPage ? '<span class="pill">Commands disabled on this page</span>' : ''}}</div></div>${{contentHtml}}<div class="bottom-nav"><button id="bottom-prev-shank-btn">Previous Shank</button><button id="bottom-next-shank-btn">Next Shank</button><span class="muted" id="bottom-shank-nav-message"></span><button id="bottom-prev-page-btn">Previous Page</button><button id="bottom-next-page-btn">Next Page</button><span class="muted" id="bottom-page-nav-message"></span></div>`; document.getElementById("bottom-prev-shank-btn").addEventListener("click", () => goToShank(-1)); document.getElementById("bottom-next-shank-btn").addEventListener("click", () => goToShank(1)); document.getElementById("bottom-prev-page-btn").addEventListener("click", () => goToPage(-1)); document.getElementById("bottom-next-page-btn").addEventListener("click", () => goToPage(1)); document.getElementById("apply-btn").disabled = isDiscardedPage; document.getElementById("save-page-btn").disabled = isDiscardedPage; document.getElementById("export-page-btn").disabled = isDiscardedPage; const bottomPageMessage = document.getElementById("bottom-page-nav-message"); if (bottomPageMessage) bottomPageMessage.textContent = pageNavMessage.textContent; const bottomShankMessage = document.getElementById("bottom-shank-nav-message"); if (bottomShankMessage) bottomShankMessage.textContent = shankNavMessage.textContent; }}
    function insertCommandTemplate(name) {{ const existing = commandText.value; const prefix = existing && !existing.endsWith("\\n") ? "\\n" : ""; commandText.value += `${{prefix}}${{name}} `; commandText.focus(); commandText.selectionStart = commandText.selectionEnd = commandText.value.length; }}
    async function loadState() {{ const payload = await fetchJson("/api/state"); DATA = payload.app; populateShanks(); populatePages(); render(); }}
    async function applyCommands() {{ const payload = {{ shank_id: Number(shankSelect.value), page_id: pageSelect.value, commands: commandText.value }}; const result = await fetchJson("/api/commands", {{ method: "POST", headers: {{ "Content-Type": "application/json" }}, body: JSON.stringify(payload) }}); DATA = result.app; populateShanks(); shankSelect.value = String(payload.shank_id); populatePages(); pageSelect.value = payload.page_id; render(); setLog(result.messages); }}
    async function postSimple(url) {{ const result = await fetchJson(url, {{ method: "POST", headers: {{ "Content-Type": "application/json" }}, body: "{{}}" }}); DATA = result.app || DATA; render(); if (result.message) setLog(result.message); if (result.messages) setLog(result.messages); if (result.export_result) {{ if (result.export_result.page_scope) setLog([`Scope: ${{result.export_result.page_scope}}`, `Export manifest: ${{result.export_result.export_manifest_path}}`, `Unique units: ${{result.export_result.num_unique_units}}`, `Discarded groups: ${{result.export_result.num_discarded_groups}}`, `Noise groups: ${{result.export_result.num_noise_groups ?? 0}}`, `Alignment groups: ${{result.export_result.num_alignment_groups}}`]); else if (result.export_result.num_pages_exported !== undefined) setLog([`Scope: all loaded pages`, `Pages exported: ${{result.export_result.num_pages_exported}}`]); else setLog([`Scope: full summary`, `Export manifest: ${{result.export_result.export_manifest_path}}`, `Unique units: ${{result.export_result.num_unique_units}}`, `Discarded groups: ${{result.export_result.num_discarded_groups}}`, `Noise groups: ${{result.export_result.num_noise_groups ?? 0}}`, `Alignment groups: ${{result.export_result.num_alignment_groups}}`]); }} }}
    async function postWithPage(url) {{ const payload = {{ shank_id: Number(shankSelect.value), page_id: pageSelect.value }}; const result = await fetchJson(url, {{ method: "POST", headers: {{ "Content-Type": "application/json" }}, body: JSON.stringify(payload) }}); DATA = result.app || DATA; render(); if (result.message) setLog(result.message); if (result.export_result) setLog([`${{result.export_result.page_scope ? `Scope: ${{result.export_result.page_scope}}` : "Scope: current page"}}`, `Export manifest: ${{result.export_result.export_manifest_path}}`, `Unique units: ${{result.export_result.num_unique_units}}`, `Discarded groups: ${{result.export_result.num_discarded_groups}}`, `Noise groups: ${{result.export_result.num_noise_groups ?? 0}}`, `Alignment groups: ${{result.export_result.num_alignment_groups}}`]); }}
    shankSelect.addEventListener("change", () => {{ setPageNavMessage(""); populatePages(); clearCommandsForNavigation(); render(); }});
    pageSelect.addEventListener("change", () => {{ setPageNavMessage(""); clearCommandsForNavigation(); render(); }}); kindSelect.addEventListener("change", render); searchInput.addEventListener("input", render);
    commandText.addEventListener("keydown", (event) => {{ if (event.ctrlKey && event.key === "Enter") {{ event.preventDefault(); applyCommands().catch((err) => setLog(err.message)); }} }});
    document.querySelectorAll("[data-command]").forEach((button) => {{ button.addEventListener("click", () => insertCommandTemplate(button.dataset.command)); }});
    document.getElementById("apply-btn").addEventListener("click", async () => {{ setLog("Applying commands..."); setButtonsBusy(["apply-btn"], "Applying..."); try {{ await applyCommands(); }} catch (err) {{ setLog(err.message); }} finally {{ restoreButtons(["apply-btn"]); }} }});
    document.getElementById("undo-btn").addEventListener("click", async () => {{ setLog("Undo in progress..."); setButtonsBusy(["undo-btn"], "Undoing..."); try {{ await postSimple("/api/undo"); }} catch (err) {{ setLog(err.message); }} finally {{ restoreButtons(["undo-btn"]); }} }});
    document.getElementById("save-page-btn").addEventListener("click", async () => {{ setLog("Saving page decisions..."); setButtonsBusy(["save-page-btn"], "Saving..."); try {{ await postWithPage("/api/save_page"); }} catch (err) {{ setLog(err.message); }} finally {{ restoreButtons(["save-page-btn"]); }} }});
    document.getElementById("export-page-btn").addEventListener("click", async () => {{ setLog("Exporting current page..."); setButtonsBusy(["export-page-btn"], "Exporting..."); try {{ await postWithPage("/api/export_page"); }} catch (err) {{ setLog(err.message); }} finally {{ restoreButtons(["export-page-btn"]); }} }});
    document.getElementById("save-all-pages-btn").addEventListener("click", async () => {{ setLog("Saving all loaded pages decisions..."); setButtonsBusy(["save-all-pages-btn"], "Saving..."); try {{ await postSimple("/api/save_all_pages"); }} catch (err) {{ setLog(err.message); }} finally {{ restoreButtons(["save-all-pages-btn"]); }} }});
    document.getElementById("export-all-pages-btn").addEventListener("click", async () => {{ setLog("Exporting all loaded pages decisions..."); setButtonsBusy(["export-all-pages-btn"], "Exporting..."); try {{ await postSimple("/api/export_all_pages"); }} catch (err) {{ setLog(err.message); }} finally {{ restoreButtons(["export-all-pages-btn"]); }} }});
    document.getElementById("save-summary-btn").addEventListener("click", async () => {{ setLog("Saving summary..."); setButtonsBusy(["save-summary-btn"], "Saving..."); try {{ await postSimple("/api/save_summary"); }} catch (err) {{ setLog(err.message); }} finally {{ restoreButtons(["save-summary-btn"]); }} }});
    document.getElementById("export-summary-btn").addEventListener("click", async () => {{ setLog("Exporting summary..."); setButtonsBusy(["export-summary-btn"], "Exporting..."); try {{ await postSimple("/api/export_summary"); }} catch (err) {{ setLog(err.message); }} finally {{ restoreButtons(["export-summary-btn"]); }} }});
    document.getElementById("clear-btn").addEventListener("click", () => {{ commandText.value = ""; setLog("Cleared command input."); }});
    document.getElementById("prev-shank-btn").addEventListener("click", () => goToShank(-1));
    document.getElementById("next-shank-btn").addEventListener("click", () => goToShank(1));
    document.getElementById("prev-page-btn").addEventListener("click", () => goToPage(-1));
    document.getElementById("next-page-btn").addEventListener("click", () => goToPage(1));
    app.addEventListener("dblclick", (event) => {{ const card = event.target.closest(".unit-card"); if (!card) return; openLightbox(card.dataset.imageSrc, card.dataset.imageLabel); }});
    lightbox.addEventListener("click", (event) => {{ if (event.target === lightbox) closeLightbox(); }});
    document.addEventListener("keydown", (event) => {{ if (event.key === "Escape" && lightbox.classList.contains("open")) closeLightbox(); }});
    loadState().catch((err) => setLog(err.message));
  </script>
</body>
</html>"""


class AlignmentRequestHandler(BaseHTTPRequestHandler):
    state: AlignmentState | None = None

    def _json_response(self, payload: dict, status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _html_response(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _handle_error(self, exc: Exception) -> None:
        self._json_response({"error": str(exc), "traceback": traceback.format_exc()}, status=500)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html_response(build_html_shell())
            return
        if parsed.path == "/api/state":
            assert self.state is not None
            self._json_response({"app": self.state.build_app_payload()})
            return
        if parsed.path == "/image":
            query = parse_qs(parsed.query)
            target = query.get("path", [""])[0]
            image_path = Path(unquote(target))
            if not image_path.exists() or not image_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                return
            try:
                payload = image_path.read_bytes()
            except Exception:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not read image")
                return
            lower_name = image_path.name.lower()
            if lower_name.endswith(".png"):
                mime_type = "image/png"
            elif lower_name.endswith(".jpg") or lower_name.endswith(".jpeg"):
                mime_type = "image/jpeg"
            elif lower_name.endswith(".gif"):
                mime_type = "image/gif"
            else:
                mime_type = "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        try:
            assert self.state is not None
            # API tiers mirror the UI buttons:
            # - page: current SG page only
            # - all pages: every loaded SG page, including hidden ones
            # - summary: one combined all-shank bundle
            if self.path == "/api/commands":
                payload = self._read_json()
                result = self.state.apply_commands(
                    shank_id=int(payload.get("shank_id")),
                    page_id=str(payload.get("page_id")),
                    raw_text=str(payload.get("commands", "")),
                )
                self._json_response(result)
                return
            if self.path == "/api/undo":
                self._json_response(self.state.undo())
                return
            if self.path == "/api/save_page":
                payload = self._read_json()
                root_manifest_path, shank_manifest_path = self.state.save_manifest_state_for_page(
                    shank_id=int(payload.get("shank_id")),
                    page_id=str(payload.get("page_id")),
                )
                self._json_response(
                    {
                        "message": (
                            f"Saved page decisions to root manifest: {root_manifest_path}\n"
                            f"Saved current shank manifest to: {shank_manifest_path}"
                        ),
                        "app": self.state.build_app_payload(),
                    }
                )
                return
            if self.path == "/api/export_page":
                payload = self._read_json()
                self.state.save_manifest_state_for_page(
                    shank_id=int(payload.get("shank_id")),
                    page_id=str(payload.get("page_id")),
                )
                export_result = self.state.export_summary_bundle_for_page(
                    shank_id=int(payload.get("shank_id")),
                    page_id=str(payload.get("page_id")),
                )
                self._json_response({"message": "Page export complete.", "export_result": export_result, "app": self.state.build_app_payload()})
                return
            if self.path == "/api/save_all_pages":
                save_result = self.state.save_manifest_state_for_all_pages()
                self._json_response(
                    {
                        "message": (
                            f"Saved all page decisions to root manifest: {save_result['root_manifest_path']}"
                        ),
                        "app": self.state.build_app_payload(),
                    }
                )
                return
            if self.path == "/api/export_all_pages":
                self.state.save_manifest_state_for_all_pages()
                export_result = self.state.export_all_pages_decisions()
                self._json_response(
                    {
                        "message": "All page exports complete, including hidden pages.",
                        "export_result": export_result,
                        "app": self.state.build_app_payload(),
                    }
                )
                return
            if self.path == "/api/save_summary":
                manifest_path = self.state.save_manifest_state()
                self._json_response({"message": f"Saved decisions to: {manifest_path}", "app": self.state.build_app_payload()})
                return
            if self.path == "/api/export_summary":
                self.state.save_manifest_state()
                export_result = self.state.export_summary_bundle()
                self._json_response({"message": "Export complete.", "export_result": export_result, "app": self.state.build_app_payload()})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self._handle_error(exc)

    def log_message(self, format: str, *args) -> None:
        return


def serve_alignment_app(root_folder: Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    def show_progress(message: str) -> None:
        print(f"[loading] {message}", flush=True)

    show_progress(f"Preparing alignment app for: {root_folder}")
    state = AlignmentState(root_folder, progress_callback=show_progress)
    handler_class = type("BoundAlignmentRequestHandler", (AlignmentRequestHandler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler_class)
    url = f"http://{host}:{port}/"
    print(f"Serving alignment review at: {url}")
    print(f"Batch root: {root_folder}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve HTML alignment review using Units_alignment_UI logic.")
    parser.add_argument("output_root", nargs="?", help="Sorting+Analyze batch output root")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", default=8765, type=int, help="Port to bind")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else choose_output_root()
    serve_alignment_app(output_root, host=args.host, port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
