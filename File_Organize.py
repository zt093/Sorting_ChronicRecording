from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path(r"H:\\")  # Fallback root when you want one path for both source and target.
DEFAULT_TARGET_ROOT = DEFAULT_ROOT  # Final write location for organized folders.
DEFAULT_SHANK_COUNT = 32
DEFAULT_REC_ORGANIZATION = True  # True: move .rec files into daily split *_rec folders.
DEFAULT_SORTING_ORGANIZATION = False  # True: organize sorting results into *_Sorting folders.
DEFAULT_CLEANUP_EMPTY = False  # True: remove empty folders left behind after moves.
DEFAULT_DELETE_PREPROCESS = False  # True: delete preprocessed_recording folders after organization.
DEFAULT_DELETE_FAILED_SORTING_RESULTS = False  # True: ask for a failed session label to delete.
DEFAULT_SOURCE_ROOTS = [
    Path(r"W:\260224_Sorting"),
    Path(r"W:\260224_rec\0224_12_23\260224_Sorting"),
]  # Default source folders to merge when sorting organization is enabled.

REC_FILE_PATTERN = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})")
SORTING_SESSION_PATTERN = re.compile(
    r"(?P<start>\d{6,8})_(?P<start_time>\d{6})_(?P<end>\d{6,8})_(?P<end_time>\d{6})(?P<rec_suffix>_rec)?$"
)
SHANK_FOLDER_PATTERN = re.compile(r".+_sh(?P<shank>\d+)$")
RUN_FOLDER_PATTERN = re.compile(r"(?P<mmdd>\d{4})_(?P<hour>\d{2})$")
ORGANIZED_SORTING_ROOT_PATTERN = re.compile(r"(?P<day_code>\d{6})_Sorting$")


@dataclass(frozen=True)
class MovePlan:
    source: Path
    destination: Path
    kind: str


def normalize_day_code(raw_date: str) -> str:
    if len(raw_date) == 8:
        return raw_date[2:]
    if len(raw_date) == 6:
        return raw_date
    raise ValueError(f"Unsupported date format: {raw_date}")


def parse_day_code_from_rec_file(rec_file: Path) -> str | None:
    match = REC_FILE_PATTERN.search(rec_file.name)
    if not match:
        return None
    return normalize_day_code(match.group("date"))


def parse_rec_session_hour(rec_file: Path) -> int | None:
    match = REC_FILE_PATTERN.search(rec_file.name)
    if not match:
        return None
    return int(match.group("time")[:2])


def build_rec_bucket_name(day_code: str, rec_file: Path) -> str:
    session_hour = parse_rec_session_hour(rec_file)
    if session_hour is None:
        raise ValueError(f"Unsupported .rec file name: {rec_file}")
    if session_hour < 12:
        return f"{day_code}_00_to_11_rec"
    return f"{day_code}_12_to_23_rec"


def parse_day_code_from_session_folder(folder: Path) -> str | None:
    match = SORTING_SESSION_PATTERN.fullmatch(folder.name)
    if not match:
        return None
    if match.group("rec_suffix"):
        return normalize_day_code(match.group("start"))
    return normalize_day_code(match.group("start"))


def parse_day_code_from_sorting_root(folder: Path) -> str | None:
    match = ORGANIZED_SORTING_ROOT_PATTERN.fullmatch(folder.name)
    if not match:
        return None
    return match.group("day_code")


def build_rec_target(root: Path, day_code: str, rec_file: Path) -> Path:
    return root / build_rec_bucket_name(day_code, rec_file) / rec_file.name


def build_sorting_target(root: Path, day_code: str, shank_id: int, run_folder: Path) -> Path:
    run_match = RUN_FOLDER_PATTERN.fullmatch(run_folder.name)
    if not run_match:
        raise ValueError(f"Unsupported sorting run folder name: {run_folder}")
    return root / f"{day_code}_Sorting" / f"sh{shank_id}" / f"{day_code}_{run_match.group('hour')}_sh{shank_id}"


def list_session_run_dirs(session_dir: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for shank_dir in session_dir.iterdir():
        if not shank_dir.is_dir():
            continue
        if not SHANK_FOLDER_PATTERN.fullmatch(shank_dir.name):
            continue
        for run_dir in shank_dir.iterdir():
            if run_dir.is_dir() and RUN_FOLDER_PATTERN.fullmatch(run_dir.name):
                run_dirs.append(run_dir)
    return run_dirs


def build_session_recording_span(day_code: str, run_dirs: list[Path]) -> str:
    hours = sorted(
        {
            RUN_FOLDER_PATTERN.fullmatch(run_dir.name).group("hour")
            for run_dir in run_dirs
            if RUN_FOLDER_PATTERN.fullmatch(run_dir.name)
        }
    )
    if not hours:
        return day_code
    if len(hours) == 1:
        return f"{day_code}_{hours[0]}"
    return f"{day_code}_{hours[0]}_to_{day_code}_{hours[-1]}"


def build_shank_summary_target(root: Path, day_code: str, shank_id: int, session_span: str) -> Path:
    return root / f"{day_code}_Sorting" / f"sh{shank_id}" / f"batch_summary_{session_span}_sh{shank_id}.json"


def build_session_summary_target(root: Path, day_code: str, session_span: str, filename: str) -> Path:
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".json"
    return root / f"{day_code}_Sorting" / f"{stem}_{session_span}{suffix}"


def ensure_unique_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination

    counter = 1
    while True:
        candidate = destination.with_name(f"{destination.name}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


def prompt_yes_no(prompt: str, default: bool) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt} [{default_label}]: ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please answer y or n.")


def prompt_path(prompt: str, default: Path) -> Path:
    response = input(f"{prompt} [{default}]: ").strip()
    return Path(response) if response else default


def prompt_source_roots(default_roots: list[Path]) -> list[Path]:
    default_text = ", ".join(str(path) for path in default_roots)
    response = input(
        "Enter source roots separated by commas, or press Enter to use defaults "
        f"[{default_text}]: "
    ).strip()
    if not response:
        return default_roots
    return [Path(part.strip()) for part in response.split(",") if part.strip()]


def apply_interactive_settings(args: argparse.Namespace) -> argparse.Namespace:
    print("Interactive setup: press Enter to keep the default shown in brackets.")

    if not args.dry_run:
        args.dry_run = prompt_yes_no("Preview only with dry run?", True)

    if args.rec_organization is None:
        args.rec_organization = prompt_yes_no(
            "Organize .rec files into 00_to_11 and 12_to_23 *_rec folders?",
            DEFAULT_REC_ORGANIZATION,
        )

    if args.sorting_organization is None:
        args.sorting_organization = prompt_yes_no(
            "Organize sorting results into *_Sorting folders?",
            DEFAULT_SORTING_ORGANIZATION,
        )

    if not args.source_root and args.sorting_organization:
        args.source_root = prompt_source_roots(DEFAULT_SOURCE_ROOTS)

    if args.target_root is None:
        args.target_root = prompt_path("Enter target root", DEFAULT_TARGET_ROOT)

    if args.cleanup_empty is None:
        args.cleanup_empty = prompt_yes_no(
            "Remove empty directories after moves finish?",
            DEFAULT_CLEANUP_EMPTY,
        )

    if args.delete_preprocess is None:
        args.delete_preprocess = prompt_yes_no(
            "Delete preprocessed_recording folders after organization?",
            DEFAULT_DELETE_PREPROCESS,
        )

    if args.delete_failed_sorting_results is None:
        args.delete_failed_sorting_results = prompt_yes_no(
            "Prompt to delete failed sorting result folders?",
            DEFAULT_DELETE_FAILED_SORTING_RESULTS,
        )

    args.rec_dry_run = args.dry_run
    args.sorting_dry_run = args.dry_run
    args.cleanup_empty_dry_run = args.dry_run
    args.delete_preprocess_dry_run = args.dry_run
    args.delete_failed_sorting_results_dry_run = args.dry_run

    return args


def apply_default_settings(args: argparse.Namespace) -> argparse.Namespace:
    if args.rec_organization is None:
        args.rec_organization = DEFAULT_REC_ORGANIZATION
    if args.sorting_organization is None:
        args.sorting_organization = DEFAULT_SORTING_ORGANIZATION
    if args.cleanup_empty is None:
        args.cleanup_empty = DEFAULT_CLEANUP_EMPTY
    if args.delete_preprocess is None:
        args.delete_preprocess = DEFAULT_DELETE_PREPROCESS
    if args.delete_failed_sorting_results is None:
        args.delete_failed_sorting_results = DEFAULT_DELETE_FAILED_SORTING_RESULTS
    if args.target_root is None:
        args.target_root = DEFAULT_TARGET_ROOT
    args.rec_dry_run = args.dry_run
    args.sorting_dry_run = args.dry_run
    args.cleanup_empty_dry_run = args.dry_run
    args.delete_preprocess_dry_run = args.dry_run
    args.delete_failed_sorting_results_dry_run = args.dry_run
    return args


def confirm_enabled_settings(args: argparse.Namespace) -> argparse.Namespace:
    if args.rec_organization:
        args.rec_dry_run = prompt_yes_no(
            "Use dry run for .rec organization?",
            True,
        )
    if args.sorting_organization:
        args.sorting_dry_run = prompt_yes_no(
            "Use dry run for sorting organization?",
            True,
        )
    if args.cleanup_empty:
        args.cleanup_empty_dry_run = prompt_yes_no(
            "Use dry run for removing empty directories?",
            True,
        )
    if args.delete_preprocess:
        args.delete_preprocess_dry_run = prompt_yes_no(
            "Use dry run for deleting preprocessed_recording folders?",
            True,
        )
    if args.delete_failed_sorting_results:
        args.delete_failed_sorting_results_dry_run = prompt_yes_no(
            "Use dry run for deleting failed sorting result folders?",
            True,
        )
    return args


def iter_rec_files(root: Path):
    for path in root.rglob("*.rec"):
        if path.is_file():
            yield path


def is_rec_already_organized(path: Path, target_root: Path, day_code: str) -> bool:
    try:
        return path.parent == target_root / build_rec_bucket_name(day_code, path)
    except Exception:
        return False


def collect_rec_moves(source_root: Path, target_root: Path) -> list[MovePlan]:
    moves: list[MovePlan] = []
    for rec_file in iter_rec_files(source_root):
        day_code = parse_day_code_from_rec_file(rec_file)
        if day_code is None:
            print(f"Skipping .rec without recognizable date: {rec_file}")
            continue
        if is_rec_already_organized(rec_file, target_root, day_code):
            continue
        destination = build_rec_target(target_root, day_code, rec_file)
        if rec_file.resolve() == destination.resolve():
            continue
        if destination.exists():
            destination = ensure_unique_destination(destination)
        moves.append(MovePlan(source=rec_file, destination=destination, kind="rec"))
    return moves


def iter_sorting_session_dirs(root: Path):
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.endswith("_Sorting"):
            continue
        if parse_day_code_from_session_folder(child) is None:
            continue
        yield child


def collect_sorting_moves(source_root: Path, target_root: Path) -> list[MovePlan]:
    moves: list[MovePlan] = []
    if parse_day_code_from_sorting_root(source_root) is not None:
        return collect_organized_sorting_root_moves(source_root, target_root)

    for session_dir in iter_sorting_session_dirs(source_root):
        day_code = parse_day_code_from_session_folder(session_dir)
        if day_code is None or session_dir.name.endswith("_rec"):
            continue
        session_run_dirs = list_session_run_dirs(session_dir)
        session_span = build_session_recording_span(day_code, session_run_dirs)

        for shank_dir in session_dir.iterdir():
            if not shank_dir.is_dir():
                continue
            shank_match = SHANK_FOLDER_PATTERN.fullmatch(shank_dir.name)
            if not shank_match:
                continue
            shank_id = int(shank_match.group("shank"))

            for run_dir in shank_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                if not RUN_FOLDER_PATTERN.fullmatch(run_dir.name):
                    continue
                destination = build_sorting_target(target_root, day_code, shank_id, run_dir)
                if run_dir.resolve() == destination.resolve():
                    continue
                if destination.exists():
                    destination = ensure_unique_destination(destination)
                moves.append(MovePlan(source=run_dir, destination=destination, kind="sorting"))

            batch_summary_path = shank_dir / "batch_summary.json"
            if batch_summary_path.is_file():
                destination = build_shank_summary_target(target_root, day_code, shank_id, session_span)
                if batch_summary_path.resolve() != destination.resolve():
                    if destination.exists():
                        destination = ensure_unique_destination(destination)
                    moves.append(
                        MovePlan(
                            source=batch_summary_path,
                            destination=destination,
                            kind="sorting_summary",
                        )
                    )

        for session_filename in ("combined_batch_summary.json", "overflow_error_report.json"):
            session_file = session_dir / session_filename
            if not session_file.is_file():
                continue
            destination = build_session_summary_target(target_root, day_code, session_span, session_filename)
            if session_file.resolve() == destination.resolve():
                continue
            if destination.exists():
                destination = ensure_unique_destination(destination)
            moves.append(
                MovePlan(
                    source=session_file,
                    destination=destination,
                    kind="session_summary",
                )
            )

    return moves


def collect_organized_sorting_root_moves(source_root: Path, target_root: Path) -> list[MovePlan]:
    moves: list[MovePlan] = []
    day_code = parse_day_code_from_sorting_root(source_root)
    if day_code is None:
        return moves

    target_sorting_root = target_root / f"{day_code}_Sorting"
    for child in source_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("sh"):
            continue

        for entry in child.iterdir():
            destination = target_sorting_root / child.name / entry.name
            if entry.resolve() == destination.resolve():
                continue
            if destination.exists():
                destination = ensure_unique_destination(destination)
            move_kind = "sorting" if entry.is_dir() else "sorting_summary"
            moves.append(MovePlan(source=entry, destination=destination, kind=move_kind))

    for filename in ("combined_batch_summary.json", "overflow_error_report.json"):
        session_file = source_root / filename
        if not session_file.is_file():
            continue
        destination = target_sorting_root / session_file.name
        if session_file.resolve() == destination.resolve():
            continue
        if destination.exists():
            destination = ensure_unique_destination(destination)
        moves.append(
            MovePlan(
                source=session_file,
                destination=destination,
                kind="session_summary",
            )
        )

    return moves


def create_shank_folders(root: Path, day_codes: set[str], shank_count: int, dry_run: bool) -> None:
    for day_code in sorted(day_codes):
        sorting_root = root / f"{day_code}_Sorting"
        for shank_id in range(shank_count):
            target = sorting_root / f"sh{shank_id}"
            if dry_run:
                continue
            target.mkdir(parents=True, exist_ok=True)


def print_plan(moves: list[MovePlan], limit: int) -> None:
    if not moves:
        print("No moves are needed.")
        return

    print(f"Planned moves: {len(moves)}")
    for move in moves[:limit]:
        print(f"[{move.kind}] {move.source} -> {move.destination}")
    if len(moves) > limit:
        print(f"... {len(moves) - limit} more move(s) not shown")


def execute_moves(moves: list[MovePlan], dry_run: bool) -> None:
    for move in moves:
        if dry_run:
            continue
        move.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(move.source), str(move.destination))


def remove_empty_directories(root: Path, dry_run: bool) -> None:
    directories = sorted(
        [path for path in root.rglob("*") if path.is_dir()],
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        if directory == root:
            continue
        try:
            next(directory.iterdir())
        except StopIteration:
            if dry_run:
                print(f"[cleanup] would remove empty directory: {directory}")
            else:
                directory.rmdir()
        except OSError:
            continue


def delete_preprocessed_recording_folders(root: Path, dry_run: bool) -> int:
    deleted_count = 0
    for folder in root.rglob("preprocessed_recording"):
        if not folder.is_dir():
            continue
        deleted_count += 1
        if dry_run:
            print(f"[delete_preprocess] would remove: {folder}")
            continue
        shutil.rmtree(folder, ignore_errors=True)
        print(f"[delete_preprocess] removed: {folder}")
    return deleted_count


def prompt_failed_session_label() -> str:
    return input(
        "Enter sorting session label to delete (for example 260224_12), "
        "or press Enter to skip: "
    ).strip()


def delete_sorting_session_results(root: Path, session_label: str, dry_run: bool) -> int:
    deleted_count = 0
    if not session_label:
        return deleted_count

    pattern = re.compile(rf"^{re.escape(session_label)}_sh\d+$")
    for sorting_root in root.glob("*_Sorting"):
        if not sorting_root.is_dir():
            continue
        for shank_dir in sorting_root.iterdir():
            if not shank_dir.is_dir() or not shank_dir.name.startswith("sh"):
                continue
            for result_dir in shank_dir.iterdir():
                if not result_dir.is_dir():
                    continue
                if not pattern.fullmatch(result_dir.name):
                    continue
                deleted_count += 1
                if dry_run:
                    print(f"[delete_sorting_session] would remove: {result_dir}")
                    continue
                shutil.rmtree(result_dir, ignore_errors=True)
                print(f"[delete_sorting_session] removed: {result_dir}")
    return deleted_count


def resolve_organization_roots(args: argparse.Namespace) -> tuple[list[Path], Path]:
    if args.source_root:
        raw_source_roots = args.source_root
        raw_target_root = args.target_root if args.target_root is not None else DEFAULT_TARGET_ROOT
    else:
        raw_source_roots = DEFAULT_SOURCE_ROOTS if args.sorting_organization else [args.root]
        raw_target_root = args.target_root if args.target_root is not None else DEFAULT_TARGET_ROOT

    source_roots: list[Path] = []
    seen_roots: set[Path] = set()
    for raw_root in raw_source_roots:
        resolved_root = raw_root.resolve()
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)
        source_roots.append(resolved_root)

    target_root = raw_target_root.resolve()
    return source_roots, target_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Organize W: recordings into daily split *_rec folders and sorting results into "
            "daily *_Sorting/shX/day_hour_shX folders."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Single root to scan and write to when no merge source roots are configured.",
    )
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        help=(
            "Source root to scan for recordings or sorting results. Repeat this flag "
            "to merge multiple source roots. Can point to raw session folders or an existing *_Sorting folder."
        ),
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        help=(
            "Destination root to write organized results into. Defaults to W:\\ when "
            "not provided."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the planned moves without changing any files.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for settings at runtime instead of only using the DEFAULT_* values.",
    )
    parser.add_argument(
        "--confirm-enabled",
        action="store_true",
        help="Only ask about actions that are currently enabled by DEFAULT_* values or CLI flags.",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Run enabled actions without confirmation prompts.",
    )
    parser.add_argument(
        "--rec-organization",
        action="store_true",
        default=None,
        help="Organize .rec files into daily 00_to_11 and 12_to_23 *_rec folders.",
    )
    parser.add_argument(
        "--sorting-organization",
        action="store_true",
        default=None,
        help="Organize sorting outputs into daily *_Sorting folders.",
    )
    parser.add_argument(
        "--shank-count",
        type=int,
        default=DEFAULT_SHANK_COUNT,
        help="Number of shank folders to create under each daily sorting folder.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=50,
        help="Maximum number of planned moves to print.",
    )
    parser.add_argument(
        "--cleanup-empty",
        action="store_true",
        default=None,
        help="Remove empty directories under each source root after moves finish.",
    )
    parser.add_argument(
        "--delete-preprocess",
        action="store_true",
        default=None,
        help="Delete all preprocessed_recording folders under each source root.",
    )
    parser.add_argument(
        "--delete-failed-sorting-results",
        action="store_true",
        default=None,
        help="Interactively delete sorting result folders for one session label such as 260224_12.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interactive:
        args = apply_interactive_settings(args)
    else:
        args = apply_default_settings(args)
        if not args.no_confirm:
            args = confirm_enabled_settings(args)
    source_roots, target_root = resolve_organization_roots(args)
    for source_root in source_roots:
        if not source_root.exists():
            raise FileNotFoundError(f"Source root path does not exist: {source_root}")
        if not source_root.is_dir():
            raise NotADirectoryError(f"Source root path is not a directory: {source_root}")
    if not target_root.exists():
        raise FileNotFoundError(f"Target root path does not exist: {target_root}")
    if not target_root.is_dir():
        raise NotADirectoryError(f"Target root path is not a directory: {target_root}")

    rec_moves: list[MovePlan] = []
    sorting_moves: list[MovePlan] = []
    if args.rec_organization:
        for source_root in source_roots:
            rec_moves.extend(collect_rec_moves(source_root, target_root))
    if args.sorting_organization:
        for source_root in source_roots:
            sorting_moves.extend(collect_sorting_moves(source_root, target_root))
    all_moves = rec_moves + sorting_moves

    sorting_days = {
        move.destination.parent.parent.name.replace("_Sorting", "")
        for move in sorting_moves
        if move.destination.parent.parent.name.endswith("_Sorting")
    }
    if args.sorting_organization:
        create_shank_folders(
            target_root,
            sorting_days,
            args.shank_count,
            dry_run=args.sorting_dry_run,
        )

    print(f"Source roots: {source_roots}")
    print(f"Target root: {target_root}")
    print(f"rec_organization: {args.rec_organization}")
    print(f"sorting_organization: {args.sorting_organization}")
    print(f"cleanup_empty: {args.cleanup_empty}")
    print(f"delete_preprocess: {args.delete_preprocess}")
    print(f"delete_failed_sorting_results: {args.delete_failed_sorting_results}")
    if args.rec_organization:
        print(f"rec dry run: {args.rec_dry_run}")
    if args.sorting_organization:
        print(f"sorting dry run: {args.sorting_dry_run}")
    if args.cleanup_empty:
        print(f"cleanup_empty dry run: {args.cleanup_empty_dry_run}")
    if args.delete_preprocess:
        print(f"delete_preprocess dry run: {args.delete_preprocess_dry_run}")
    if args.delete_failed_sorting_results:
        print(
            "delete_failed_sorting_results dry run: "
            f"{args.delete_failed_sorting_results_dry_run}"
        )
    print(f".rec moves: {len(rec_moves)}")
    print(f"Sorting moves: {len(sorting_moves)}")
    print_plan(all_moves, limit=args.preview)

    execute_moves(rec_moves, dry_run=args.rec_dry_run)
    execute_moves(sorting_moves, dry_run=args.sorting_dry_run)

    if args.delete_failed_sorting_results:
        session_label = prompt_failed_session_label()
        deleted_session_count = delete_sorting_session_results(
            root=target_root,
            session_label=session_label,
            dry_run=args.delete_failed_sorting_results_dry_run,
        )
        print(f"sorting result folders matched for deletion: {deleted_session_count}")

    if args.delete_preprocess:
        deleted_count = 0
        for source_root in source_roots:
            deleted_count += delete_preprocessed_recording_folders(
                source_root,
                dry_run=args.delete_preprocess_dry_run,
            )
        print(f"preprocessed_recording folders matched: {deleted_count}")

    if args.cleanup_empty:
        for source_root in source_roots:
            remove_empty_directories(source_root, dry_run=args.cleanup_empty_dry_run)

    print("Organization step(s) complete.")


if __name__ == "__main__":
    main()
