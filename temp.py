from __future__ import annotations

"""
Organize legacy LDA output folders into the current LDA layout.

Legacy layout:
    S:/LDA/lda_260221_to_260226_minsess_120_fr_only/
        lda_260221_to_260226_minsess_120_fr_only_2d.png
        ...

Current layout:
    S:/LDA/organized_previous__260221_to_260226__multi_day_hourly__labels-clock_hour_of_day__minsess120__legacy/
        lda_run_config.json
        clock_hour_of_day/
            fr_only/
                lda_2d.png
                ...

Run first as a dry run:
    python Sorting_Check/temp.py

Actually move files:
    python Sorting_Check/temp.py --apply
"""

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


LDA_ROOT = Path(r"S:\LDA")
DEFAULT_LABEL = "clock_hour_of_day"

FEATURE_MODES = (
    "fr_peak_to_trough",
    "multi_feature",
    "waveform_only",
    "fr_waveform",
    "fr_only",
    "fr_amp",
    "fr_cv2",
)


@dataclass(frozen=True)
class LegacyLdaFolder:
    path: Path
    date_label: str
    min_sessions: int
    feature_mode: str
    lda_mode: str
    label: str
    is_threshold: bool
    is_smoothed: bool

    @property
    def group_key(self) -> tuple[str, int, str, str, bool, bool]:
        return (
            self.date_label,
            self.min_sessions,
            self.lda_mode,
            self.label,
            self.is_threshold,
            self.is_smoothed,
        )


def slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-_.") or "value"


def parse_legacy_folder(path: Path) -> LegacyLdaFolder | None:
    name = path.name
    if "__" in name:
        return None
    if not name.startswith("lda_"):
        return None
    if "_minsess_" not in name:
        return None

    stem = name[len("lda_") :]
    date_label, rest = stem.split("_minsess_", 1)
    match = re.match(r"(?P<min_sessions>\d+)_(?P<tail>.+)$", rest)
    if not match:
        return None

    min_sessions = int(match.group("min_sessions"))
    tail = match.group("tail")

    lda_mode = "single_day_5min" if tail.endswith("_single_day_5min") else "multi_day_hourly"
    if lda_mode == "single_day_5min":
        tail = tail[: -len("_single_day_5min")]

    is_threshold = tail.endswith("_threshold")
    if is_threshold:
        tail = tail[: -len("_threshold")]

    is_smoothed = tail.endswith("_smooth")
    if is_smoothed:
        tail = tail[: -len("_smooth")]

    label = DEFAULT_LABEL
    feature_mode = None
    for mode in FEATURE_MODES:
        if tail == mode:
            feature_mode = mode
            break
        suffix = f"_{mode}"
        if tail.endswith(suffix):
            feature_mode = mode
            label = tail[: -len(suffix)]
            break
    if feature_mode is None:
        return None
    if not label:
        label = DEFAULT_LABEL

    return LegacyLdaFolder(
        path=path,
        date_label=date_label,
        min_sessions=min_sessions,
        feature_mode=feature_mode,
        lda_mode=lda_mode,
        label=label,
        is_threshold=is_threshold,
        is_smoothed=is_smoothed,
    )


def build_group_folder(root: Path, item: LegacyLdaFolder) -> Path:
    parts = [
        "organized_previous",
        item.date_label,
        item.lda_mode,
        f"labels-{slug(item.label)}",
        f"minsess{item.min_sessions}",
    ]
    if item.is_smoothed:
        parts.append("smooth")
    if item.is_threshold:
        parts.append("threshold")
    parts.append("legacy")
    return root / "__".join(parts)


def simplified_filename(file_path: Path, legacy_folder_name: str) -> str:
    name = file_path.name
    if name.startswith(legacy_folder_name):
        suffix = name[len(legacy_folder_name) :]
        if suffix.startswith("_"):
            suffix = suffix[1:]
        return f"lda_{suffix}" if suffix else name
    return name


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def write_manifest(group_dir: Path, items: list[LegacyLdaFolder], apply: bool) -> None:
    payload = {
        "created_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "source": "Sorting_Check/temp.py legacy LDA organizer",
        "applied_moves": bool(apply),
        "date_label": items[0].date_label,
        "lda_mode": items[0].lda_mode,
        "label": items[0].label,
        "min_sessions_per_unit": int(items[0].min_sessions),
        "is_threshold": bool(items[0].is_threshold),
        "is_smoothed": bool(items[0].is_smoothed),
        "feature_modes": sorted({item.feature_mode for item in items}),
        "legacy_folders": [str(item.path) for item in items],
    }
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "lda_run_config.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def organize(root: Path, apply: bool) -> None:
    legacy_items = [
        parsed
        for path in sorted(root.iterdir(), key=lambda p: p.name)
        if path.is_dir()
        for parsed in [parse_legacy_folder(path)]
        if parsed is not None
    ]

    if not legacy_items:
        print(f"No legacy LDA folders found under: {root}")
        return

    grouped: dict[tuple[str, int, str, str, bool, bool], list[LegacyLdaFolder]] = {}
    for item in legacy_items:
        grouped.setdefault(item.group_key, []).append(item)

    print(f"Found {len(legacy_items)} legacy feature-mode folder(s) in {len(grouped)} group(s).")
    print("Mode:", "APPLY - moving files" if apply else "DRY RUN - no files moved")

    for group_index, items in enumerate(grouped.values(), start=1):
        items = sorted(items, key=lambda item: item.feature_mode)
        group_dir = build_group_folder(root, items[0])
        print(f"\n[{group_index}/{len(grouped)}] {group_dir}")
        if apply:
            write_manifest(group_dir, items, apply=apply)
        else:
            print("  would write lda_run_config.json")

        for item in items:
            target_dir = group_dir / slug(item.label) / slug(item.feature_mode)
            print(f"  {item.path.name} -> {target_dir.relative_to(root)}")
            if apply:
                target_dir.mkdir(parents=True, exist_ok=True)

            for source_file in sorted(p for p in item.path.iterdir() if p.is_file()):
                target_name = simplified_filename(source_file, item.path.name)
                target_path = unique_path(target_dir / target_name)
                if apply:
                    shutil.move(str(source_file), str(target_path))
                else:
                    print(f"    {source_file.name} -> {target_name}")

            if apply:
                try:
                    item.path.rmdir()
                except OSError:
                    print(f"    [kept] source folder not empty: {item.path}")

    if apply:
        print("\nDone organizing legacy LDA outputs.")
    else:
        print("\nDry run complete. Re-run with --apply to move files.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Organize legacy LDA output folders.")
    parser.add_argument("--root", type=Path, default=LDA_ROOT, help="LDA output root folder.")
    parser.add_argument("--apply", action="store_true", help="Move files instead of printing a dry run.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    organize(root=root, apply=bool(args.apply))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
