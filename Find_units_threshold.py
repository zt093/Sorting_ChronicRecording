"""
Find sorted/aligned units by SpikeGadget channel and optional minimum absolute amplitude.

Interactive use:
    python Sorting_Check/Find_units_threshold.py

What it searches
----------------
1. Alignment outputs:
   - sh*/units_alignment_summary/*units_summary*.json
   - alignment_days_summary*/**/*units_summary*.json
   - all_shanks_alignment_summary/*units_summary*.json

   These rows are matched by `sg_channel`. If amplitude values are present in the
   row, member rows, or the row's summary.txt, the script can also keep only units
   whose absolute amplitude is at least the threshold you entered.

2. Raw analyzer mapping reports:
   - **/unit_channel_mapping_report.json

   These rows map raw sorter `unit_id` to SG channel through
   `device_channel_index_property`.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUMMARY_NAME_PATTERNS = (
    "*units_summary*.json",
    "all_shanks_*units_summary*.json",
)
MAPPING_REPORT_NAME = "unit_channel_mapping_report.json"
FILTERED_EXPORT_SUFFIX = "_filtered_alignment_days_export"


@dataclass(frozen=True)
class QueryPair:
    sg_channel: int
    threshold_uv: float | None = None


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except Exception:
        return default


def parse_paths(raw_text: str) -> list[Path]:
    parts = [
        part.strip().strip('"').strip("'")
        for part in re.split(r"[;,\n]+", raw_text)
        if part.strip()
    ]
    paths = [Path(part).expanduser() for part in parts]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Path(s) do not exist: {missing}")
    return paths


def parse_query_pairs(raw_text: str) -> list[QueryPair]:
    """
    Accepted examples:
      72:500, 74:450
      72, 74, 101
      [{"sg_ch": 72, "threshold_uv": 500}, {"sg_channel": 74, "threshold": -450}]
      {"72": 500, "74": -450}
    """
    text = raw_text.strip()
    if not text:
        raise ValueError("No SG channels were provided.")

    def from_obj(obj: Any) -> list[QueryPair]:
        pairs: list[QueryPair] = []
        if isinstance(obj, dict):
            if any(key in obj for key in ("sg_ch", "sg_channel", "channel")):
                sg_channel = safe_int(obj.get("sg_ch", obj.get("sg_channel", obj.get("channel"))))
                threshold = safe_float(obj.get("threshold_uv", obj.get("threshold")))
                if sg_channel is None:
                    raise ValueError(f"Could not parse SG channel from: {obj}")
                return [QueryPair(sg_channel=sg_channel, threshold_uv=threshold)]
            for key, value in obj.items():
                sg_channel = safe_int(key)
                threshold = safe_float(value)
                if sg_channel is None:
                    raise ValueError(f"Could not parse SG channel from key: {key!r}")
                pairs.append(QueryPair(sg_channel=sg_channel, threshold_uv=threshold))
            return pairs
        if isinstance(obj, list):
            for item in obj:
                pairs.extend(from_obj(item))
            return pairs
        raise ValueError(f"Unsupported JSON pair format: {type(obj).__name__}")

    if text[0] in "[{":
        return dedupe_query_pairs(from_obj(json.loads(text)))

    pairs: list[QueryPair] = []
    for token in re.split(r"[;,\n]+", text):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            left, right = token.split(":", 1)
        elif "=" in token:
            left, right = token.split("=", 1)
        else:
            left, right = token, ""
        sg_channel = safe_int(left.strip())
        threshold = safe_float(right.strip())
        if sg_channel is None:
            raise ValueError(f"Could not parse SG channel from token: {token!r}")
        pairs.append(QueryPair(sg_channel=sg_channel, threshold_uv=threshold))

    return dedupe_query_pairs(pairs)


def dedupe_query_pairs(pairs: list[QueryPair]) -> list[QueryPair]:
    seen: set[tuple[int, float | None]] = set()
    out: list[QueryPair] = []
    for pair in pairs:
        key = (int(pair.sg_channel), pair.threshold_uv)
        if key in seen:
            continue
        seen.add(key)
        out.append(pair)
    return out


def discover_summary_jsons(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in SUMMARY_NAME_PATTERNS:
        for path in root.rglob(pattern):
            if path.is_file() and not is_generated_filtered_export_path(path):
                paths.add(path)
    return sorted(paths, key=lambda path: (summary_path_priority(root, path), str(path)))


def is_generated_filtered_export_path(path: Path) -> bool:
    return any(part.endswith(FILTERED_EXPORT_SUFFIX) for part in path.parts)


def path_depth_from_root(root: Path, path: Path) -> int:
    try:
        return len(path.relative_to(root).parts)
    except ValueError:
        return len(path.parts)


def summary_path_priority(root: Path, path: Path) -> tuple[int, int]:
    # Root-level summaries contain the global final_unit_id values. Shank-level
    # summaries are useful fallback inputs, but their IDs are local to that file.
    root_level = 0 if path.parent.resolve() == root.resolve() else 1
    return root_level, path_depth_from_root(root, path)


def discover_mapping_reports(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob(MAPPING_REPORT_NAME) if path.is_file())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_summary_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in (
            "unique_units",
            "discarded_units",
            "noise_units",
            "cross_session_alignment_groups",
            "rows",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def infer_summary_kind(path: Path, row: dict[str, Any]) -> str:
    status = str(row.get("status", "") or "").strip()
    if status:
        return status
    name = path.name.lower()
    if "discarded" in name:
        return "discarded"
    if "noise" in name:
        return "noise"
    if "unique" in name:
        return "unique"
    if "export_summary" in name:
        return "export"
    return "summary"


def collect_member_payloads(row: dict[str, Any]) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for key in ("member_units", "members", "source_members", "day_members"):
        value = row.get(key)
        if isinstance(value, list):
            members.extend(item for item in value if isinstance(item, dict))
    return members


def parse_amplitudes_from_summary_text(path: Path) -> list[float]:
    if not path.exists() or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    amplitudes: list[float] = []
    for match in re.finditer(r"amplitude_median\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text):
        value = safe_float(match.group(1))
        if value is not None:
            amplitudes.append(value)
    return amplitudes


def collect_amplitudes(row: dict[str, Any]) -> list[float]:
    amplitudes: list[float] = []
    for key in ("amplitude_median", "median_amplitude", "amplitude_median_abs"):
        value = safe_float(row.get(key))
        if value is not None:
            amplitudes.append(value)

    for member in collect_member_payloads(row):
        for key in ("amplitude_median", "median_amplitude", "amplitude_median_abs"):
            value = safe_float(member.get(key))
            if value is not None:
                amplitudes.append(value)

    summary_path = row.get("summary_path")
    if summary_path:
        amplitudes.extend(parse_amplitudes_from_summary_text(Path(str(summary_path))))

    return amplitudes


def max_abs_amplitude(amplitudes: list[float]) -> float | None:
    if not amplitudes:
        return None
    return max(abs(float(amplitude)) for amplitude in amplitudes)


def best_amplitude_distance(amplitudes: list[float], threshold_uv: float | None) -> float | None:
    if threshold_uv is None or not amplitudes:
        return None
    threshold = float(threshold_uv)
    abs_threshold = abs(threshold)
    distances = [abs(abs(float(amplitude)) - abs_threshold) for amplitude in amplitudes]
    return min(distances) if distances else None


def amplitude_match_status(
    amplitudes: list[float],
    threshold_uv: float | None,
) -> str:
    if threshold_uv is None:
        return "not_requested"
    if not amplitudes:
        return "unknown_no_amplitude_in_summary"
    threshold = abs(float(threshold_uv))
    strongest = max_abs_amplitude(amplitudes)
    if strongest is None:
        return "unknown_no_amplitude_in_summary"
    return "match" if strongest >= threshold else "below_threshold"


def row_identity(row: dict[str, Any]) -> tuple[str, str]:
    final_id = row.get(
        "final_unit_id",
        row.get("global_final_unit_id", row.get("global_discarded_group_id", "")),
    )
    label = row.get(
        "final_unit_label",
        row.get("global_final_unit_label", row.get("global_discarded_group_label", "")),
    )
    return str(final_id or ""), str(label or "")


def stringify_members(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for member in collect_member_payloads(row):
        session_name = str(member.get("session_name", "") or "")
        unit_id = member.get("unit_id", "")
        amp = member.get("amplitude_median", "")
        if amp not in ("", None):
            parts.append(f"{session_name}:u{unit_id}:amp={amp}")
        else:
            parts.append(f"{session_name}:u{unit_id}")
    return "; ".join(parts)


def count_unique_member_output_folders(row: dict[str, Any]) -> int:
    output_folders: set[str] = set()
    for member in collect_member_payloads(row):
        output_folder = str(member.get("output_folder") or member.get("session_key") or "").strip()
        if output_folder:
            output_folders.add(output_folder)
    return len(output_folders)


def count_total_member_sessions(row: dict[str, Any]) -> int:
    session_names: set[str] = set()
    for member in collect_member_payloads(row):
        session_name = str(member.get("session_name") or "").strip()
        if session_name:
            session_names.add(session_name)
    if session_names:
        return len(session_names)
    return count_unique_member_output_folders(row)


def build_summary_match(
    *,
    query: QueryPair,
    path: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    amplitudes = collect_amplitudes(row)
    final_id, final_label = row_identity(row)
    best_distance = best_amplitude_distance(amplitudes, query.threshold_uv)
    strongest = max_abs_amplitude(amplitudes)
    return {
        "match_source": "alignment_summary",
        "summary_kind": infer_summary_kind(path, row),
        "query_sg_channel": query.sg_channel,
        "query_threshold_uv": query.threshold_uv,
        "amplitude_match_status": amplitude_match_status(amplitudes, query.threshold_uv),
        "best_amplitude_distance_uv": best_distance,
        "max_abs_amplitude_uv": strongest,
        "amplitudes_found": "; ".join(f"{value:.6g}" for value in amplitudes),
        "summary_json": str(path),
        "summary_path": str(row.get("summary_path", "") or ""),
        "export_folder": str(row.get("export_folder", "") or ""),
        "shank_id": row.get("shank_id", ""),
        "local_channel_on_shank": row.get("channel", row.get("local_channel_on_shank", "")),
        "sg_channel": row.get("sg_channel", ""),
        "final_unit_id": final_id,
        "final_unit_label": final_label,
        "final_group_key": str(row.get("final_group_key", row.get("discard_group_key", row.get("noise_group_key", ""))) or ""),
        "representative_session": str(row.get("representative_session", "") or ""),
        "representative_unit_id": row.get("representative_unit_id", ""),
        "num_sessions": row.get("num_sessions", ""),
        "total_member_sessions": count_total_member_sessions(row),
        "unique_member_output_folders": count_unique_member_output_folders(row),
        "sessions_present": "; ".join(str(item) for item in row.get("sessions_present", []) or []),
        "num_member_units": row.get("num_member_units", ""),
        "member_units": stringify_members(row),
        "waveform_image": str(row.get("representative_waveform_image", "") or ""),
    }


def find_summary_matches(
    roots: list[Path],
    queries_by_sg: dict[int, list[QueryPair]],
    include_below_threshold: bool,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for root in roots:
        for path in discover_summary_jsons(root):
            try:
                rows = iter_summary_rows(load_json(path))
            except Exception as exc:
                print(f"[skip] Could not read {path}: {exc}")
                continue
            for row in rows:
                sg_channel = safe_int(row.get("sg_channel"))
                if sg_channel is None or sg_channel not in queries_by_sg:
                    continue
                for query in queries_by_sg[sg_channel]:
                    match = build_summary_match(
                        query=query,
                        path=path,
                        row=row,
                    )
                    if (
                        not include_below_threshold
                        and match["amplitude_match_status"] == "below_threshold"
                    ):
                        continue
                    matches.append(match)
    return dedupe_summary_matches(matches)


def summary_match_key(row: dict[str, Any]) -> tuple[Any, ...]:
    group_key = str(row.get("final_group_key") or "").strip()
    if group_key:
        identity = ("group", group_key)
    else:
        identity = (
            "representative",
            str(row.get("representative_session") or ""),
            str(row.get("representative_unit_id") or ""),
            str(row.get("member_units") or ""),
        )
    return (
        row.get("query_sg_channel"),
        row.get("query_threshold_uv"),
        row.get("summary_kind"),
        identity,
    )


def summary_match_priority(row: dict[str, Any]) -> tuple[int, int, int]:
    path = Path(str(row.get("summary_json") or ""))
    parent_name = path.parent.name.lower()
    is_shank_summary = re.fullmatch(r"sh\d+", parent_name) is not None
    has_final_id = safe_int(row.get("final_unit_id")) is not None
    return (
        1 if is_shank_summary else 0,
        0 if has_final_id else 1,
        len(path.parts),
    )


def dedupe_summary_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = summary_match_key(row)
        existing = best_by_key.get(key)
        if existing is None or summary_match_priority(row) < summary_match_priority(existing):
            best_by_key[key] = row
    return list(best_by_key.values())


def build_mapping_match(query: QueryPair, path: Path, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "match_source": "raw_unit_channel_mapping",
        "summary_kind": "raw_mapping",
        "query_sg_channel": query.sg_channel,
        "query_threshold_uv": query.threshold_uv,
        "amplitude_match_status": "not_available_in_mapping_report",
        "best_amplitude_distance_uv": "",
        "max_abs_amplitude_uv": "",
        "amplitudes_found": "",
        "summary_json": str(path),
        "summary_path": "",
        "export_folder": str(path.parent),
        "shank_id": row.get("shank_id", ""),
        "local_channel_on_shank": row.get("waveform_local_channel_index", ""),
        "sg_channel": row.get("device_channel_index_property", ""),
        "final_unit_id": "",
        "final_unit_label": "",
        "final_group_key": "",
        "representative_session": path.parent.name,
        "representative_unit_id": row.get("unit_id", ""),
        "num_sessions": "",
        "total_member_sessions": "",
        "unique_member_output_folders": "",
        "sessions_present": "",
        "num_member_units": "",
        "member_units": f"{path.parent.name}:u{row.get('unit_id', '')}",
        "waveform_image": "",
    }


def find_mapping_matches(
    roots: list[Path],
    queries_by_sg: dict[int, list[QueryPair]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for root in roots:
        for path in discover_mapping_reports(root):
            try:
                payload = load_json(path)
            except Exception as exc:
                print(f"[skip] Could not read {path}: {exc}")
                continue
            rows = payload.get("units", []) if isinstance(payload, dict) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sg_channel = safe_int(row.get("device_channel_index_property"))
                if sg_channel is None or sg_channel not in queries_by_sg:
                    continue
                for query in queries_by_sg[sg_channel]:
                    matches.append(build_mapping_match(query, path, row))
    return matches


def write_outputs(rows: list[dict[str, Any]], output_base: Path) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_base.with_suffix(".csv")
    json_path = output_base.with_suffix(".json")
    fieldnames = [
        "match_source",
        "summary_kind",
        "query_sg_channel",
        "query_threshold_uv",
        "amplitude_match_status",
        "best_amplitude_distance_uv",
        "max_abs_amplitude_uv",
        "amplitudes_found",
        "shank_id",
        "local_channel_on_shank",
        "sg_channel",
        "final_unit_id",
        "final_unit_label",
        "representative_session",
        "representative_unit_id",
        "num_sessions",
        "total_member_sessions",
        "unique_member_output_folders",
        "sessions_present",
        "num_member_units",
        "member_units",
        "final_group_key",
        "summary_json",
        "summary_path",
        "export_folder",
        "waveform_image",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def infer_group_sg_channel(group_row: dict[str, Any], fallback_page_scope: dict[str, Any] | None = None) -> int | None:
    for key in ("sg_channel", "query_sg_channel"):
        value = safe_int(group_row.get(key))
        if value is not None:
            return value

    for member_key in ("day_members", "members", "source_members", "member_units"):
        members = group_row.get(member_key)
        if not isinstance(members, list):
            continue
        for member in members:
            if not isinstance(member, dict):
                continue
            for key in ("sg_channel", "query_sg_channel"):
                value = safe_int(member.get(key))
                if value is not None:
                    return value
            page_scope = member.get("source_page_scope")
            if isinstance(page_scope, dict):
                value = safe_int(page_scope.get("sg_channel"))
                if value is not None:
                    return value

    if fallback_page_scope:
        value = safe_int(fallback_page_scope.get("sg_channel"))
        if value is not None:
            return value

    for text_key in ("final_group_key", "source_group_key", "discard_group_key", "noise_group_key"):
        text = str(group_row.get(text_key, "") or "")
        match = re.search(r"(?:^|_)sg(?P<sg>\d+)(?:_|:|$)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group("sg"))
    return None


def discover_alignment_export_paths(root: Path) -> list[Path]:
    if root.is_file() and root.name.startswith("export_summary"):
        return [] if is_generated_filtered_export_path(root) else [root]

    candidates: set[Path] = set()
    direct = root / "export_summary.json"
    if direct.exists():
        candidates.add(direct)

    for path in root.rglob("export_summary.json"):
        if not is_generated_filtered_export_path(path):
            candidates.add(path)
    for path in root.rglob("export_summary_sg_*.json"):
        if not is_generated_filtered_export_path(path):
            candidates.add(path)
    return sorted(candidates, key=lambda path: (export_path_priority(root, path), str(path)))


def export_path_priority(root: Path, path: Path) -> tuple[int, int]:
    root_level = 0 if path.parent.resolve() == root.resolve() else 1
    return root_level, path_depth_from_root(root, path)


def load_alignment_export(path: Path) -> dict[str, Any] | None:
    try:
        payload = load_json(path)
    except Exception as exc:
        print(f"[skip] Could not read export {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("cross_session_alignment_groups"), list):
        return None
    return payload


def build_filtered_alignment_export(
    *,
    roots: list[Path],
    queries_by_sg: dict[int, list[QueryPair]],
    include_below_threshold: bool,
    output_base: Path,
) -> Path | None:
    filtered_groups: list[dict[str, Any]] = []
    source_export_paths: list[str] = []
    query_rows: list[dict[str, Any]] = []
    seen_group_keys: set[tuple[Any, ...]] = set()

    for root in roots:
        for export_path in discover_alignment_export_paths(root):
            payload = load_alignment_export(export_path)
            if payload is None:
                continue
            source_export_paths.append(str(export_path))
            page_scope = payload.get("page_scope") if isinstance(payload.get("page_scope"), dict) else {}
            for group_row in payload.get("cross_session_alignment_groups", []):
                if not isinstance(group_row, dict):
                    continue
                sg_channel = infer_group_sg_channel(group_row, fallback_page_scope=page_scope)
                if sg_channel is None or sg_channel not in queries_by_sg:
                    continue

                for query in queries_by_sg[sg_channel]:
                    amplitudes = collect_amplitudes(group_row)
                    status = amplitude_match_status(amplitudes, query.threshold_uv)
                    if not include_below_threshold and status == "below_threshold":
                        continue

                    group_copy = dict(group_row)
                    if "sg_channel" not in group_copy:
                        group_copy["sg_channel"] = int(sg_channel)
                    if "page_scope" not in group_copy and page_scope:
                        group_copy["page_scope"] = dict(page_scope)
                    if "source_members" not in group_copy and isinstance(group_copy.get("members"), list):
                        # LDA/Tuning prefer source_members. Per-page Alignment_days exports may
                        # only have members, but those members are already source analyzer units.
                        group_copy["source_members"] = list(group_copy.get("members", []))

                    dedupe_key = group_identity_key(group_copy, sg_channel)
                    if dedupe_key not in seen_group_keys:
                        seen_group_keys.add(dedupe_key)
                        filtered_groups.append(group_copy)

                    query_rows.append(
                        {
                            "source_export_summary": str(export_path),
                            "sg_channel": int(sg_channel),
                            "query_threshold_uv": query.threshold_uv,
                            "amplitude_match_status": status,
                            "best_amplitude_distance_uv": best_amplitude_distance(amplitudes, query.threshold_uv),
                            "max_abs_amplitude_uv": max_abs_amplitude(amplitudes),
                            "amplitudes_found": amplitudes,
                            "total_member_sessions": count_total_member_sessions(group_copy),
                            "unique_member_output_folders": count_unique_member_output_folders(group_copy),
                            "final_unit_id": group_copy.get("final_unit_id"),
                            "final_group_key": group_copy.get("final_group_key"),
                        }
                    )

    if not filtered_groups:
        return None

    for index, group_row in enumerate(filtered_groups, start=1):
        if safe_int(group_row.get("final_unit_id")) is None:
            group_row["final_unit_id"] = index

    export_dir = output_base.with_name(output_base.name + "_filtered_alignment_days_export")
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "export_summary.json"
    payload = {
        "output_root": str(export_dir),
        "member_mode": "full_source_members",
        "filter_note": (
            "Filtered export generated by Sorting_Check/Find_units_threshold.py "
            "for direct use with Sorting_Check/Stats/LDA.py and Sorting_Check/Stats/Tuning.py."
        ),
        "source_export_summary_paths": sorted(set(source_export_paths)),
        "queries": [
            {"sg_channel": pair.sg_channel, "threshold_uv": pair.threshold_uv}
            for pairs in queries_by_sg.values()
            for pair in pairs
        ],
        "filter_matches": query_rows,
        "cross_session_alignment_groups": filtered_groups,
    }
    export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return export_path


def group_identity_key(group_row: dict[str, Any], sg_channel: int) -> tuple[Any, ...]:
    group_key = str(
        group_row.get("final_group_key")
        or group_row.get("source_group_key")
        or group_row.get("discard_group_key")
        or group_row.get("noise_group_key")
        or ""
    ).strip()
    if group_key:
        return sg_channel, "group", group_key

    member_parts: list[tuple[str, str, str]] = []
    for member in collect_member_payloads(group_row):
        member_parts.append(
            (
                str(member.get("session_name") or member.get("session_key") or ""),
                str(member.get("unit_id") or ""),
                str(member.get("output_folder") or ""),
            )
        )
    if member_parts:
        return sg_channel, "members", tuple(sorted(member_parts))

    return sg_channel, "row", json.dumps(group_row, sort_keys=True)[:500]


def prompt_yes_no(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    value = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "true", "1"}


def build_default_output_base(roots: list[Path]) -> Path:
    if len(roots) == 1 and roots[0].is_dir():
        return roots[0] / "found_units_by_sg_channel_threshold"
    return Path.cwd() / "found_units_by_sg_channel_threshold"


def main() -> None:
    print("=== Find units by SG channel and threshold ===\n")
    roots = parse_paths(
        input("Search folder(s), separated by comma/semicolon: ").strip()
    )
    query_pairs = parse_query_pairs(
        input(
            "SG channel/min absolute unit amplitude pairs "
            "(examples: 279:180, 74:450 OR 72,74 OR JSON): "
        ).strip()
    )

    include_below_threshold = False
    create_filtered_export = True
    include_raw_mapping = False

    default_output = build_default_output_base(roots)
    output_text = input(
        f"Output base path without extension [{default_output}]: "
    ).strip()
    output_base = Path(output_text).expanduser() if output_text else default_output

    queries_by_sg: dict[int, list[QueryPair]] = {}
    for pair in query_pairs:
        queries_by_sg.setdefault(pair.sg_channel, []).append(pair)

    print("\nSearching alignment summaries...")
    rows = find_summary_matches(
        roots=roots,
        queries_by_sg=queries_by_sg,
        include_below_threshold=include_below_threshold,
    )

    if include_raw_mapping:
        print("Searching raw unit-channel mapping reports...")
        rows.extend(find_mapping_matches(roots=roots, queries_by_sg=queries_by_sg))

    rows.sort(
        key=lambda row: (
            safe_int(row.get("query_sg_channel"), 10**9) or 10**9,
            str(row.get("match_source", "")),
            str(row.get("summary_json", "")),
            safe_int(row.get("representative_unit_id"), 10**9) or 10**9,
        )
    )

    csv_path, json_path = write_outputs(rows, output_base)
    filtered_export_path = None
    if create_filtered_export:
        print("Writing filtered Alignment_days export for LDA/Tuning...")
        filtered_export_path = build_filtered_alignment_export(
            roots=roots,
            queries_by_sg=queries_by_sg,
            include_below_threshold=include_below_threshold,
            output_base=output_base,
        )

    print("\nDone.")
    print(f"Queries: {len(query_pairs)}")
    print(f"Matches written: {len(rows)}")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    if filtered_export_path is not None:
        print(f"Filtered Alignment_days export for LDA/Tuning: {filtered_export_path}")
        print("Use this path when LDA.py or Tuning.py asks for the alignment export path.")
    elif create_filtered_export:
        print("No filtered Alignment_days export was written because no aligned export groups matched.")
    if not rows:
        print("No matches found. Check that the input folder contains alignment summaries or mapping reports.")


if __name__ == "__main__":
    main()
