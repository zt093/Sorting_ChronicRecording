"Summarize the results to include multiiple shanks alignment results from Units_alignment_UI.py"
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import tkinter as tk
from tkinter import filedialog, messagebox


ALIGNMENT_FOLDER_NAME = "units_alignment_summary"
UNIQUE_SUMMARY_NAME = "unique_units_summary.json"
DISCARDED_SUMMARY_NAME = "discarded_units_summary.json"
EXPORT_SUMMARY_NAME = "export_summary.json"
OUTPUT_FOLDER_NAME = "all_shanks_alignment_summary"


@dataclass
class ShankSummarySource:
    shank_id: int
    shank_folder: Path
    summary_root: Path
    unique_units_json_path: Path
    discarded_units_json_path: Path | None
    export_summary_json_path: Path | None


def make_hidden_root() -> tk.Tk:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def choose_batch_root() -> Path:
    root = make_hidden_root()
    selected_folder = filedialog.askdirectory(
        title="Select a daily sorting root containing shank folders",
        mustexist=True,
        parent=root,
    )
    root.destroy()
    if not selected_folder:
        raise SystemExit("No folder selected.")
    return Path(selected_folder)


def safe_int(value, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def extract_shank_id(path: Path) -> int | None:
    match = re.search(r"sh(\d+)", path.name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"sh(\d+)", str(path), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def natural_shank_key(path: Path) -> tuple[int, str]:
    shank_id = extract_shank_id(path)
    if shank_id is not None:
        return (shank_id, path.name.lower())
    return (10**9, path.name.lower())


def discover_shank_summary_sources(batch_root: Path) -> list[ShankSummarySource]:
    summary_sources: list[ShankSummarySource] = []
    for child in sorted(batch_root.iterdir(), key=natural_shank_key):
        if not child.is_dir():
            continue
        shank_id = extract_shank_id(child)
        if shank_id is None:
            continue

        summary_root = child / ALIGNMENT_FOLDER_NAME
        unique_units_json_path = summary_root / UNIQUE_SUMMARY_NAME
        if not unique_units_json_path.exists():
            continue

        export_summary_json_path = summary_root / EXPORT_SUMMARY_NAME
        discarded_units_json_path = summary_root / DISCARDED_SUMMARY_NAME
        summary_sources.append(
            ShankSummarySource(
                shank_id=shank_id,
                shank_folder=child,
                summary_root=summary_root,
                unique_units_json_path=unique_units_json_path,
                discarded_units_json_path=discarded_units_json_path
                if discarded_units_json_path.exists()
                else None,
                export_summary_json_path=export_summary_json_path
                if export_summary_json_path.exists()
                else None,
            )
        )

    if not summary_sources:
        raise FileNotFoundError(
            f"No {UNIQUE_SUMMARY_NAME} files were found under shank folders in {batch_root}"
        )
    return summary_sources


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_combined_unit_rows(
    summary_sources: list[ShankSummarySource],
) -> tuple[list[dict], list[dict]]:
    combined_rows: list[dict] = []
    source_rows: list[dict] = []
    next_global_id = 1

    for source in summary_sources:
        unique_units_payload = load_json(source.unique_units_json_path)
        if not isinstance(unique_units_payload, list):
            raise ValueError(
                f"Expected a list in {source.unique_units_json_path}, got {type(unique_units_payload).__name__}"
            )

        for shank_local_index, row in enumerate(unique_units_payload, start=1):
            shank_id = safe_int(row.get("shank_id"), source.shank_id)
            sessions_present = list(row.get("sessions_present", []))
            member_units = list(row.get("member_units", []))
            waveform_images = list(row.get("waveform_images", []))

            combined_row = {
                "global_final_unit_id": next_global_id,
                "global_final_unit_label": f"unit_{next_global_id:04d}",
                "source_shank_final_unit_id": safe_int(row.get("final_unit_id"), shank_local_index),
                "source_shank_final_unit_label": str(
                    row.get("final_unit_label") or f"unit_{shank_local_index:04d}"
                ),
                "source_summary_root": str(source.summary_root),
                "source_unique_units_summary_json": str(source.unique_units_json_path),
                "source_export_summary_json": str(source.export_summary_json_path)
                if source.export_summary_json_path is not None
                else "",
                "shank_id": shank_id,
                "channel": safe_int(row.get("channel"), -1),
                "sg_channel": safe_int(row.get("sg_channel"), -1),
                "final_group_key": str(row.get("final_group_key", "")),
                "representative_session": str(row.get("representative_session", "")),
                "representative_unit_id": safe_int(row.get("representative_unit_id"), -1),
                "num_sessions": safe_int(row.get("num_sessions"), len(sessions_present)),
                "sessions_present": sessions_present,
                "num_member_units": safe_int(row.get("num_member_units"), len(member_units)),
                "member_units": member_units,
                "export_folder": str(row.get("export_folder", "")),
                "summary_path": str(row.get("summary_path", "")),
                "representative_waveform_image": str(row.get("representative_waveform_image", "")),
                "waveform_images": waveform_images,
            }
            combined_rows.append(combined_row)

            source_rows.append(
                {
                    "global_final_unit_id": next_global_id,
                    "shank_id": shank_id,
                    "source_shank_final_unit_id": combined_row["source_shank_final_unit_id"],
                    "source_shank_final_unit_label": combined_row["source_shank_final_unit_label"],
                    "num_sessions": combined_row["num_sessions"],
                    "num_member_units": combined_row["num_member_units"],
                    "representative_session": combined_row["representative_session"],
                    "representative_unit_id": combined_row["representative_unit_id"],
                }
            )
            next_global_id += 1

    return combined_rows, source_rows


def build_combined_discarded_rows(summary_sources: list[ShankSummarySource]) -> list[dict]:
    combined_rows: list[dict] = []
    next_global_id = 1

    for source in summary_sources:
        if source.discarded_units_json_path is None:
            continue
        discarded_payload = load_json(source.discarded_units_json_path)
        if not isinstance(discarded_payload, list):
            raise ValueError(
                f"Expected a list in {source.discarded_units_json_path}, got {type(discarded_payload).__name__}"
            )

        for row in discarded_payload:
            member_units = list(row.get("member_units", []))
            combined_rows.append(
                {
                    "global_discarded_group_id": next_global_id,
                    "global_discarded_group_label": f"discarded_{next_global_id:04d}",
                    "source_summary_root": str(source.summary_root),
                    "source_discarded_units_summary_json": str(source.discarded_units_json_path),
                    "shank_id": safe_int(row.get("shank_id"), source.shank_id),
                    "channel": safe_int(row.get("channel"), -1),
                    "sg_channel": safe_int(row.get("sg_channel"), -1),
                    "discard_group_key": str(row.get("discard_group_key", "")),
                    "discard_reason": str(row.get("discard_reason", "")),
                    "num_sessions": safe_int(row.get("num_sessions"), 0),
                    "sessions_present": list(row.get("sessions_present", [])),
                    "num_member_units": safe_int(row.get("num_member_units"), len(member_units)),
                    "member_units": member_units,
                }
            )
            next_global_id += 1

    return combined_rows


def build_per_shank_summary(combined_rows: list[dict]) -> list[dict]:
    by_shank: dict[int, list[dict]] = {}
    for row in combined_rows:
        by_shank.setdefault(int(row["shank_id"]), []).append(row)

    shank_rows: list[dict] = []
    for shank_id in sorted(by_shank):
        rows = by_shank[shank_id]
        session_counter: Counter[str] = Counter()
        sg_counter: Counter[int] = Counter()
        for row in rows:
            for session_name in row["sessions_present"]:
                session_counter[str(session_name)] += 1
            sg_counter[int(row["sg_channel"])] += 1

        shank_rows.append(
            {
                "shank_id": shank_id,
                "num_unique_units": len(rows),
                "num_multi_session_units": sum(1 for row in rows if int(row["num_sessions"]) > 1),
                "num_single_session_units": sum(1 for row in rows if int(row["num_sessions"]) == 1),
                "total_member_units": sum(int(row["num_member_units"]) for row in rows),
                "sessions_seen": sorted(session_counter),
                "num_sessions_seen": len(session_counter),
                "top_sessions_by_unique_units": [
                    {"session_name": session_name, "num_units": count}
                    for session_name, count in session_counter.most_common(10)
                ],
                "sg_channels": sorted(sg_counter),
                "num_sg_channels": len(sg_counter),
            }
        )
    return shank_rows


def build_per_session_summary(combined_rows: list[dict]) -> list[dict]:
    session_counter: Counter[str] = Counter()
    shank_counter_by_session: dict[str, Counter[int]] = {}

    for row in combined_rows:
        shank_id = int(row["shank_id"])
        for session_name in row["sessions_present"]:
            session_name = str(session_name)
            session_counter[session_name] += 1
            shank_counter_by_session.setdefault(session_name, Counter())[shank_id] += 1

    per_session_rows: list[dict] = []
    for session_name in sorted(session_counter):
        shank_counter = shank_counter_by_session[session_name]
        per_session_rows.append(
            {
                "session_name": session_name,
                "num_unique_units_across_shanks": session_counter[session_name],
                "num_shanks_present": len(shank_counter),
                "units_by_shank": [
                    {"shank_id": shank_id, "num_units": count}
                    for shank_id, count in sorted(shank_counter.items())
                ],
            }
        )
    return per_session_rows


def build_overview(
    batch_root: Path,
    summary_sources: list[ShankSummarySource],
    combined_rows: list[dict],
    discarded_rows: list[dict],
    per_shank_rows: list[dict],
    per_session_rows: list[dict],
) -> dict:
    all_sessions = sorted(
        {
            str(session_name)
            for row in combined_rows
            for session_name in row["sessions_present"]
        }
    )
    return {
        "batch_root": str(batch_root),
        "num_shanks_with_alignment_summary": len(summary_sources),
        "shanks_with_alignment_summary": [source.shank_id for source in summary_sources],
        "num_unique_units_across_shanks": len(combined_rows),
        "num_discarded_groups_across_shanks": len(discarded_rows),
        "num_discarded_member_units_across_shanks": sum(
            int(row["num_member_units"]) for row in discarded_rows
        ),
        "num_multi_session_units_across_shanks": sum(
            1 for row in combined_rows if int(row["num_sessions"]) > 1
        ),
        "num_single_session_units_across_shanks": sum(
            1 for row in combined_rows if int(row["num_sessions"]) == 1
        ),
        "total_member_units_across_shanks": sum(
            int(row["num_member_units"]) for row in combined_rows
        ),
        "sessions_seen": all_sessions,
        "num_sessions_seen": len(all_sessions),
        "per_shank": per_shank_rows,
        "per_session": per_session_rows,
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_combined_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "global_final_unit_id",
        "global_final_unit_label",
        "shank_id",
        "source_shank_final_unit_id",
        "source_shank_final_unit_label",
        "channel",
        "sg_channel",
        "num_sessions",
        "sessions_present",
        "num_member_units",
        "member_units",
        "representative_session",
        "representative_unit_id",
        "representative_waveform_image",
        "export_folder",
        "summary_path",
        "final_group_key",
        "source_summary_root",
        "source_unique_units_summary_json",
        "source_export_summary_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "global_final_unit_id": row["global_final_unit_id"],
                    "global_final_unit_label": row["global_final_unit_label"],
                    "shank_id": row["shank_id"],
                    "source_shank_final_unit_id": row["source_shank_final_unit_id"],
                    "source_shank_final_unit_label": row["source_shank_final_unit_label"],
                    "channel": row["channel"],
                    "sg_channel": row["sg_channel"],
                    "num_sessions": row["num_sessions"],
                    "sessions_present": "; ".join(row["sessions_present"]),
                    "num_member_units": row["num_member_units"],
                    "member_units": "; ".join(
                        f"{item.get('session_name', '')}:u{item.get('unit_id', '')}"
                        for item in row["member_units"]
                    ),
                    "representative_session": row["representative_session"],
                    "representative_unit_id": row["representative_unit_id"],
                    "representative_waveform_image": row["representative_waveform_image"],
                    "export_folder": row["export_folder"],
                    "summary_path": row["summary_path"],
                    "final_group_key": row["final_group_key"],
                    "source_summary_root": row["source_summary_root"],
                    "source_unique_units_summary_json": row["source_unique_units_summary_json"],
                    "source_export_summary_json": row["source_export_summary_json"],
                }
            )


def write_discarded_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "global_discarded_group_id",
        "global_discarded_group_label",
        "shank_id",
        "channel",
        "sg_channel",
        "num_sessions",
        "sessions_present",
        "num_member_units",
        "member_units",
        "discard_group_key",
        "discard_reason",
        "source_summary_root",
        "source_discarded_units_summary_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "global_discarded_group_id": row["global_discarded_group_id"],
                    "global_discarded_group_label": row["global_discarded_group_label"],
                    "shank_id": row["shank_id"],
                    "channel": row["channel"],
                    "sg_channel": row["sg_channel"],
                    "num_sessions": row["num_sessions"],
                    "sessions_present": "; ".join(row["sessions_present"]),
                    "num_member_units": row["num_member_units"],
                    "member_units": "; ".join(
                        f"{item.get('session_name', '')}:u{item.get('unit_id', '')}"
                        for item in row["member_units"]
                    ),
                    "discard_group_key": row["discard_group_key"],
                    "discard_reason": row["discard_reason"],
                    "source_summary_root": row["source_summary_root"],
                    "source_discarded_units_summary_json": row["source_discarded_units_summary_json"],
                }
            )


def write_shank_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "shank_id",
        "num_unique_units",
        "num_multi_session_units",
        "num_single_session_units",
        "total_member_units",
        "num_sessions_seen",
        "sessions_seen",
        "num_sg_channels",
        "sg_channels",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "shank_id": row["shank_id"],
                    "num_unique_units": row["num_unique_units"],
                    "num_multi_session_units": row["num_multi_session_units"],
                    "num_single_session_units": row["num_single_session_units"],
                    "total_member_units": row["total_member_units"],
                    "num_sessions_seen": row["num_sessions_seen"],
                    "sessions_seen": "; ".join(row["sessions_seen"]),
                    "num_sg_channels": row["num_sg_channels"],
                    "sg_channels": "; ".join(str(item) for item in row["sg_channels"]),
                }
            )


def write_session_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "session_name",
        "num_unique_units_across_shanks",
        "num_shanks_present",
        "units_by_shank",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "session_name": row["session_name"],
                    "num_unique_units_across_shanks": row["num_unique_units_across_shanks"],
                    "num_shanks_present": row["num_shanks_present"],
                    "units_by_shank": "; ".join(
                        f"sh{item['shank_id']}:{item['num_units']}"
                        for item in row["units_by_shank"]
                    ),
                }
            )


def build_summary_text(overview: dict) -> str:
    lines = [
        "All-Shanks Alignment Summary",
        f"Batch root: {overview['batch_root']}",
        f"Shanks with alignment summary: {overview['num_shanks_with_alignment_summary']}",
        f"Shank ids: {', '.join(str(item) for item in overview['shanks_with_alignment_summary'])}",
        f"Unique final units across shanks: {overview['num_unique_units_across_shanks']}",
        f"Discarded groups across shanks: {overview['num_discarded_groups_across_shanks']}",
        f"Discarded member units across shanks: {overview['num_discarded_member_units_across_shanks']}",
        f"Multi-session final units: {overview['num_multi_session_units_across_shanks']}",
        f"Single-session final units: {overview['num_single_session_units_across_shanks']}",
        f"Total member units across shanks: {overview['total_member_units_across_shanks']}",
        f"Sessions seen: {overview['num_sessions_seen']}",
        "",
        "Per-shank summary:",
    ]
    for row in overview["per_shank"]:
        lines.append(
            f"- sh{row['shank_id']}: "
            f"{row['num_unique_units']} final units, "
            f"{row['num_multi_session_units']} multi-session, "
            f"{row['num_single_session_units']} single-session, "
            f"{row['num_sessions_seen']} sessions"
        )
    lines.append("")
    lines.append("Per-session summary:")
    for row in overview["per_session"]:
        lines.append(
            f"- {row['session_name']}: "
            f"{row['num_unique_units_across_shanks']} final units across "
            f"{row['num_shanks_present']} shank(s)"
        )
    return "\n".join(lines).strip() + "\n"


def export_all_shanks_summary(batch_root: Path) -> Path:
    summary_sources = discover_shank_summary_sources(batch_root)
    combined_rows, _source_rows = build_combined_unit_rows(summary_sources)
    discarded_rows = build_combined_discarded_rows(summary_sources)
    per_shank_rows = build_per_shank_summary(combined_rows)
    per_session_rows = build_per_session_summary(combined_rows)
    overview = build_overview(
        batch_root=batch_root,
        summary_sources=summary_sources,
        combined_rows=combined_rows,
        discarded_rows=discarded_rows,
        per_shank_rows=per_shank_rows,
        per_session_rows=per_session_rows,
    )

    output_root = batch_root / OUTPUT_FOLDER_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    write_json(output_root / "all_shanks_unique_units_summary.json", combined_rows)
    write_combined_csv(output_root / "all_shanks_unique_units_summary.csv", combined_rows)
    write_json(output_root / "all_shanks_discarded_units_summary.json", discarded_rows)
    write_discarded_csv(output_root / "all_shanks_discarded_units_summary.csv", discarded_rows)
    write_json(output_root / "all_shanks_overview.json", overview)
    write_json(output_root / "all_shanks_per_shank_summary.json", per_shank_rows)
    write_json(output_root / "all_shanks_per_session_summary.json", per_session_rows)
    write_shank_csv(output_root / "all_shanks_per_shank_summary.csv", per_shank_rows)
    write_session_csv(output_root / "all_shanks_per_session_summary.csv", per_session_rows)
    (output_root / "all_shanks_summary.txt").write_text(
        build_summary_text(overview),
        encoding="utf-8",
    )

    return output_root


def main() -> None:
    batch_root = choose_batch_root()
    try:
        output_root = export_all_shanks_summary(batch_root)
    except Exception as exc:
        root = make_hidden_root()
        messagebox.showerror(
            "Alignment Summary",
            f"Failed to build all-shanks summary.\n\nReason:\n{exc}",
            parent=root,
        )
        root.destroy()
        raise SystemExit(1)

    root = make_hidden_root()
    messagebox.showinfo(
        "Alignment Summary",
        "All-shanks summary export complete.\n\n"
        f"Saved to:\n{output_root}",
        parent=root,
    )
    root.destroy()
    print(f"Saved all-shanks alignment summary to: {output_root}")


if __name__ == "__main__":
    main()
