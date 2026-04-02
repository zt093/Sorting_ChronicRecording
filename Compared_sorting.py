from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import spikeinterface.full as si
from scipy.optimize import linear_sum_assignment


SHANK_PATTERNS = (
    re.compile(r"shank[_-]?(\d+)", flags=re.IGNORECASE),
    re.compile(r"_sh(\d+)", flags=re.IGNORECASE),
)


@dataclass(frozen=True)
class SortingRun:
    pipeline_name: str
    output_folder: Path
    sorting_path: Path
    shank_id: str
    window_label: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PreparedSorting:
    sorting: Any
    normalized_sorting: Any
    normalized_to_original_unit_id: dict[int, Any]


@dataclass(frozen=True)
class LocalComparison:
    unit1_ids: list[int]
    unit2_ids: list[int]
    event_counts1: dict[int, int]
    event_counts2: dict[int, int]
    agreement_scores: dict[tuple[int, int], float]
    match_event_count: dict[tuple[int, int], int]
    hungarian_match_12: dict[int, int | None]
    hungarian_match_21: dict[int, int | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two sorting pipelines per shank/hour and test whether one result "
            "looks like an inclusion (subset) of the other."
        )
    )
    parser.add_argument("--pipeline-a", type=Path, help="Root folder for pipeline A outputs.")
    parser.add_argument("--pipeline-b", type=Path, help="Root folder for pipeline B outputs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Folder where CSV/JSON reports will be written.",
    )
    parser.add_argument(
        "--pipeline-a-name",
        default=None,
        help="Display name used in reports for pipeline A.",
    )
    parser.add_argument(
        "--pipeline-b-name",
        default=None,
        help="Display name used in reports for pipeline B.",
    )
    parser.add_argument(
        "--delta-time-ms",
        type=float,
        default=None,
        help="Coincidence window passed to spikeinterface.compare_two_sorters.",
    )
    parser.add_argument(
        "--match-score",
        type=float,
        default=None,
        help="Agreement-score threshold used by SpikeInterface matching.",
    )
    parser.add_argument(
        "--subset-coverage",
        type=float,
        default=None,
        help=(
            "Coverage threshold for saying source-unit spikes are contained in a matched unit "
            "from the other pipeline."
        ),
    )
    return parser.parse_args()


def prompt_path(prompt_text: str, default: Path | None = None, must_exist: bool = True) -> Path:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw_value = input(f"{prompt_text}{suffix}: ").strip().strip('"').strip("'")
        if not raw_value and default is not None:
            return default
        if not raw_value:
            print("A path is required.")
            continue
        path = Path(raw_value).expanduser()
        if not must_exist or path.exists():
            return path
        print(f"Path does not exist: {path}")


def prompt_text(prompt_text: str, default: str) -> str:
    raw_value = input(f"{prompt_text} [{default}]: ").strip()
    return raw_value or default


def prompt_float(prompt_text: str, default: float) -> float:
    while True:
        raw_value = input(f"{prompt_text} [{default}]: ").strip()
        if not raw_value:
            return default
        try:
            return float(raw_value)
        except ValueError:
            print("Please enter a valid number.")


def resolve_inputs(args: argparse.Namespace) -> argparse.Namespace:
    if args.pipeline_a is None:
        args.pipeline_a = prompt_path("Enter pipeline A root folder")
    else:
        args.pipeline_a = args.pipeline_a.expanduser()

    if args.pipeline_b is None:
        args.pipeline_b = prompt_path("Enter pipeline B root folder")
    else:
        args.pipeline_b = args.pipeline_b.expanduser()

    if args.output is None:
        args.output = prompt_path(
            "Enter output report folder",
            default=Path("compared_sorting_output"),
            must_exist=False,
        )
    else:
        args.output = args.output.expanduser()

    if args.pipeline_a_name is None:
        args.pipeline_a_name = "pipeline_a"

    if args.pipeline_b_name is None:
        args.pipeline_b_name = "pipeline_b"

    if args.delta_time_ms is None:
        args.delta_time_ms = 0.4

    if args.match_score is None:
        args.match_score = 0.5

    if args.subset_coverage is None:
        args.subset_coverage = 0.8

    return args


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_shank_id_from_text(text: str | None) -> str | None:
    if not text:
        return None
    for pattern in SHANK_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def infer_shank_id(output_folder: Path, metadata: dict[str, Any]) -> str:
    candidates: list[str] = []

    for key in ("shank_id",):
        value = metadata.get(key)
        if value is not None:
            candidates.append(str(value))

    for key in ("input_nwb", "output_folder"):
        value = metadata.get(key)
        if value:
            candidates.append(str(value))

    input_sources = metadata.get("input_sources")
    if isinstance(input_sources, list):
        candidates.extend(str(value) for value in input_sources)

    for candidate in candidates:
        shank_id = infer_shank_id_from_text(candidate)
        if shank_id is not None:
            return shank_id

    for part in (output_folder, *output_folder.parents):
        shank_id = infer_shank_id_from_text(part.name)
        if shank_id is not None:
            return shank_id

    return "unknown"


def infer_window_label(output_folder: Path, metadata: dict[str, Any]) -> str:
    for key in ("window_label", "hour_index"):
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return output_folder.name


def load_run_metadata(output_folder: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for filename in ("sorting_summary.json", "preprocessing_metadata.json", "analysis_summary.json"):
        metadata.update(safe_read_json(output_folder / filename))
    return metadata


def find_candidate_output_folders(root: Path) -> list[Path]:
    candidates: set[Path] = set()

    if root.is_file():
        if root.name == "sorted_units.npz":
            candidates.add(root.parent)
        elif root.name == "si_folder.json":
            candidates.add(root.parent)
        else:
            raise FileNotFoundError(f"Unsupported file input: {root}")
    else:
        if (root / "sorted_sorting").exists() or (root / "sorted_units.npz").exists():
            candidates.add(root)
        for path in root.rglob("sorted_sorting"):
            candidates.add(path.parent)
        for path in root.rglob("sorted_units.npz"):
            candidates.add(path.parent)

    return sorted(candidates)


def pick_sorting_path(output_folder: Path) -> Path | None:
    sorted_sorting = output_folder / "sorted_sorting"
    sorted_units = output_folder / "sorted_units.npz"
    if sorted_sorting.exists():
        return sorted_sorting
    if sorted_units.exists():
        return sorted_units
    return None


def discover_runs(root: Path, pipeline_name: str) -> dict[tuple[str, str], SortingRun]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    discovered: dict[tuple[str, str], SortingRun] = {}
    for output_folder in find_candidate_output_folders(root):
        sorting_path = pick_sorting_path(output_folder)
        if sorting_path is None:
            continue

        metadata = load_run_metadata(output_folder)
        shank_id = infer_shank_id(output_folder, metadata)
        if shank_id == "unknown":
            continue
        window_label = infer_window_label(output_folder, metadata)
        key = (shank_id, window_label)
        discovered[key] = SortingRun(
            pipeline_name=pipeline_name,
            output_folder=output_folder,
            sorting_path=sorting_path,
            shank_id=shank_id,
            window_label=window_label,
            metadata=metadata,
        )

    return discovered


def load_sorting(run: SortingRun):
    return si.load_extractor(run.sorting_path)


def prepare_sorting_for_comparison(sorting) -> PreparedSorting:
    original_unit_ids = list(sorting.get_unit_ids())
    normalized_unit_ids = list(range(len(original_unit_ids)))
    normalized_to_original = {
        normalized_id: original_id
        for normalized_id, original_id in zip(normalized_unit_ids, original_unit_ids)
    }

    unit_dict: dict[int, Any] = {}
    for normalized_id, original_id in normalized_to_original.items():
        unit_dict[normalized_id] = sorting.get_unit_spike_train(original_id)

    normalized_sorting = si.NumpySorting.from_unit_dict(
        unit_dict,
        sampling_frequency=float(sorting.get_sampling_frequency()),
    )
    return PreparedSorting(
        sorting=sorting,
        normalized_sorting=normalized_sorting,
        normalized_to_original_unit_id=normalized_to_original,
    )


def is_missing_match(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, int)):
        if isinstance(value, float) and math.isnan(value):
            return True
        if int(value) < 0:
            return True
    return False


def normalize_match_id(value: Any, valid_ids) -> int | None:
    if is_missing_match(value):
        return None
    try:
        normalized = int(value)
    except Exception:
        return None
    if normalized < 0:
        return None
    if normalized not in valid_ids:
        return None
    return normalized


def count_matching_events(spike_train_a, spike_train_b, delta_frames: int) -> int:
    a = np.asarray(spike_train_a, dtype=np.int64)
    b = np.asarray(spike_train_b, dtype=np.int64)
    i = 0
    j = 0
    matches = 0

    while i < len(a) and j < len(b):
        diff = int(a[i] - b[j])
        if abs(diff) <= delta_frames:
            matches += 1
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1

    return matches


def compare_sortings_locally(
    sorting_a: PreparedSorting,
    sorting_b: PreparedSorting,
    delta_time_ms: float,
    match_score: float,
) -> LocalComparison:
    sf_a = float(sorting_a.normalized_sorting.get_sampling_frequency())
    sf_b = float(sorting_b.normalized_sorting.get_sampling_frequency())
    sampling_frequency = sf_a if sf_a > 0 else sf_b
    delta_frames = max(1, int(round(delta_time_ms * sampling_frequency / 1000.0)))

    unit1_ids = [int(unit_id) for unit_id in sorting_a.normalized_sorting.get_unit_ids()]
    unit2_ids = [int(unit_id) for unit_id in sorting_b.normalized_sorting.get_unit_ids()]

    event_counts1: dict[int, int] = {}
    event_counts2: dict[int, int] = {}
    spike_trains1: dict[int, np.ndarray] = {}
    spike_trains2: dict[int, np.ndarray] = {}

    for unit_id in unit1_ids:
        train = np.asarray(
            sorting_a.normalized_sorting.get_unit_spike_train(unit_id),
            dtype=np.int64,
        )
        spike_trains1[unit_id] = train
        event_counts1[unit_id] = int(train.size)

    for unit_id in unit2_ids:
        train = np.asarray(
            sorting_b.normalized_sorting.get_unit_spike_train(unit_id),
            dtype=np.int64,
        )
        spike_trains2[unit_id] = train
        event_counts2[unit_id] = int(train.size)

    agreement_scores: dict[tuple[int, int], float] = {}
    match_event_count: dict[tuple[int, int], int] = {}
    score_matrix = np.zeros((len(unit1_ids), len(unit2_ids)), dtype=np.float64)

    for row_index, unit_a in enumerate(unit1_ids):
        for col_index, unit_b in enumerate(unit2_ids):
            matched_spikes = count_matching_events(
                spike_trains1[unit_a],
                spike_trains2[unit_b],
                delta_frames,
            )
            denom = event_counts1[unit_a] + event_counts2[unit_b] - matched_spikes
            agreement = float(matched_spikes / denom) if denom > 0 else 0.0
            agreement_scores[(unit_a, unit_b)] = agreement
            match_event_count[(unit_a, unit_b)] = matched_spikes
            score_matrix[row_index, col_index] = agreement

    hungarian_match_12 = {unit_id: None for unit_id in unit1_ids}
    hungarian_match_21 = {unit_id: None for unit_id in unit2_ids}

    if len(unit1_ids) > 0 and len(unit2_ids) > 0:
        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        for row_index, col_index in zip(row_ind.tolist(), col_ind.tolist()):
            score = float(score_matrix[row_index, col_index])
            if score >= match_score:
                unit_a = unit1_ids[row_index]
                unit_b = unit2_ids[col_index]
                hungarian_match_12[unit_a] = unit_b
                hungarian_match_21[unit_b] = unit_a

    return LocalComparison(
        unit1_ids=unit1_ids,
        unit2_ids=unit2_ids,
        event_counts1=event_counts1,
        event_counts2=event_counts2,
        agreement_scores=agreement_scores,
        match_event_count=match_event_count,
        hungarian_match_12=hungarian_match_12,
        hungarian_match_21=hungarian_match_21,
    )


def build_unit_rows(
    comparison: LocalComparison,
    run_a: SortingRun,
    run_b: SortingRun,
    prepared_a: PreparedSorting,
    prepared_b: PreparedSorting,
    subset_coverage: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    reverse_rows: list[dict[str, Any]] = []

    event_counts_a = comparison.event_counts1
    event_counts_b = comparison.event_counts2
    agreement_scores = comparison.agreement_scores
    match_event_count = comparison.match_event_count
    hungarian_a_to_b = comparison.hungarian_match_12
    hungarian_b_to_a = comparison.hungarian_match_21
    valid_unit_a_ids = set(event_counts_a.keys())
    valid_unit_b_ids = set(event_counts_b.keys())

    for unit_a in comparison.unit1_ids:
        original_unit_a = prepared_a.normalized_to_original_unit_id.get(int(unit_a), unit_a)
        matched_b = normalize_match_id(hungarian_a_to_b.get(unit_a), valid_unit_b_ids)
        unit_a_spikes = int(event_counts_a[unit_a])
        row: dict[str, Any] = {
            "shank_id": run_a.shank_id,
            "window_label": run_a.window_label,
            f"{run_a.pipeline_name}_unit_id": original_unit_a,
            f"{run_b.pipeline_name}_unit_id": None,
            f"{run_a.pipeline_name}_spike_count": unit_a_spikes,
            f"{run_b.pipeline_name}_spike_count": None,
            "matched_spike_count": 0,
            "agreement_score": 0.0,
            f"{run_a.pipeline_name}_coverage": 0.0,
            f"{run_b.pipeline_name}_coverage": 0.0,
            f"{run_a.pipeline_name}_contained_in_{run_b.pipeline_name}": False,
            "match_kind": "unmatched",
        }
        if matched_b is not None:
            original_unit_b = prepared_b.normalized_to_original_unit_id.get(int(matched_b), matched_b)
            matched_spikes = int(match_event_count[(unit_a, matched_b)])
            unit_b_spikes = int(event_counts_b[matched_b])
            coverage_a = matched_spikes / unit_a_spikes if unit_a_spikes else 0.0
            coverage_b = matched_spikes / unit_b_spikes if unit_b_spikes else 0.0
            agreement = float(agreement_scores[(unit_a, matched_b)])
            row.update(
                {
                    f"{run_b.pipeline_name}_unit_id": original_unit_b,
                    f"{run_b.pipeline_name}_spike_count": unit_b_spikes,
                    "matched_spike_count": matched_spikes,
                    "agreement_score": agreement,
                    f"{run_a.pipeline_name}_coverage": coverage_a,
                    f"{run_b.pipeline_name}_coverage": coverage_b,
                    f"{run_a.pipeline_name}_contained_in_{run_b.pipeline_name}": coverage_a >= subset_coverage,
                    "match_kind": "matched",
                }
            )
        rows.append(row)

    for unit_b in comparison.unit2_ids:
        original_unit_b = prepared_b.normalized_to_original_unit_id.get(int(unit_b), unit_b)
        matched_a = normalize_match_id(hungarian_b_to_a.get(unit_b), valid_unit_a_ids)
        unit_b_spikes = int(event_counts_b[unit_b])
        row = {
            "shank_id": run_a.shank_id,
            "window_label": run_a.window_label,
            f"{run_a.pipeline_name}_unit_id": None,
            f"{run_b.pipeline_name}_unit_id": original_unit_b,
            f"{run_a.pipeline_name}_spike_count": None,
            f"{run_b.pipeline_name}_spike_count": unit_b_spikes,
            "matched_spike_count": 0,
            "agreement_score": 0.0,
            f"{run_a.pipeline_name}_coverage": 0.0,
            f"{run_b.pipeline_name}_coverage": 0.0,
            f"{run_b.pipeline_name}_contained_in_{run_a.pipeline_name}": False,
            "match_kind": "unmatched",
        }
        if matched_a is not None:
            original_unit_a = prepared_a.normalized_to_original_unit_id.get(int(matched_a), matched_a)
            matched_spikes = int(match_event_count[(matched_a, unit_b)])
            unit_a_spikes = int(event_counts_a[matched_a])
            coverage_a = matched_spikes / unit_a_spikes if unit_a_spikes else 0.0
            coverage_b = matched_spikes / unit_b_spikes if unit_b_spikes else 0.0
            agreement = float(agreement_scores[(matched_a, unit_b)])
            row.update(
                {
                    f"{run_a.pipeline_name}_unit_id": original_unit_a,
                    f"{run_a.pipeline_name}_spike_count": unit_a_spikes,
                    "matched_spike_count": matched_spikes,
                    "agreement_score": agreement,
                    f"{run_a.pipeline_name}_coverage": coverage_a,
                    f"{run_b.pipeline_name}_coverage": coverage_b,
                    f"{run_b.pipeline_name}_contained_in_{run_a.pipeline_name}": coverage_b >= subset_coverage,
                    "match_kind": "matched",
                }
            )
        reverse_rows.append(row)

    return rows, reverse_rows


def summarize_pair(
    comparison: LocalComparison,
    run_a: SortingRun,
    run_b: SortingRun,
    prepared_a: PreparedSorting,
    prepared_b: PreparedSorting,
    subset_coverage: float,
) -> dict[str, Any]:
    unit_rows_a, unit_rows_b = build_unit_rows(
        comparison,
        run_a,
        run_b,
        prepared_a,
        prepared_b,
        subset_coverage,
    )

    contained_a = [row for row in unit_rows_a if row["match_kind"] == "matched" and row[f"{run_a.pipeline_name}_contained_in_{run_b.pipeline_name}"]]
    contained_b = [row for row in unit_rows_b if row["match_kind"] == "matched" and row[f"{run_b.pipeline_name}_contained_in_{run_a.pipeline_name}"]]

    subset_a_in_b = len(contained_a) == len(unit_rows_a)
    subset_b_in_a = len(contained_b) == len(unit_rows_b)

    unmatched_a = [row for row in unit_rows_a if row["match_kind"] == "unmatched"]
    unmatched_b = [row for row in unit_rows_b if row["match_kind"] == "unmatched"]

    if subset_a_in_b and subset_b_in_a:
        relation = "approximately_equal"
    elif subset_a_in_b:
        relation = f"{run_a.pipeline_name}_subset_of_{run_b.pipeline_name}"
    elif subset_b_in_a:
        relation = f"{run_b.pipeline_name}_subset_of_{run_a.pipeline_name}"
    else:
        relation = "different"

    return {
        "shank_id": run_a.shank_id,
        "window_label": run_a.window_label,
        "pipeline_a_output_folder": str(run_a.output_folder),
        "pipeline_b_output_folder": str(run_b.output_folder),
        "pipeline_a_units": len(unit_rows_a),
        "pipeline_b_units": len(unit_rows_b),
        "pipeline_a_contained_units": len(contained_a),
        "pipeline_b_contained_units": len(contained_b),
        "pipeline_a_unmatched_units": len(unmatched_a),
        "pipeline_b_unmatched_units": len(unmatched_b),
        "subset_coverage_threshold": subset_coverage,
        f"{run_a.pipeline_name}_subset_of_{run_b.pipeline_name}": subset_a_in_b,
        f"{run_b.pipeline_name}_subset_of_{run_a.pipeline_name}": subset_b_in_a,
        "relation": relation,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = resolve_inputs(parse_args())

    runs_a = discover_runs(args.pipeline_a, args.pipeline_a_name)
    runs_b = discover_runs(args.pipeline_b, args.pipeline_b_name)

    common_keys = sorted(set(runs_a) & set(runs_b))
    only_a = sorted(set(runs_a) - set(runs_b))
    only_b = sorted(set(runs_b) - set(runs_a))

    if not common_keys:
        raise RuntimeError(
            "No common (shank_id, window_label) outputs were found between the two pipeline roots."
        )

    args.output.mkdir(parents=True, exist_ok=True)

    pair_summaries: list[dict[str, Any]] = []
    unit_rows_a: list[dict[str, Any]] = []
    unit_rows_b: list[dict[str, Any]] = []

    for key in common_keys:
        run_a = runs_a[key]
        run_b = runs_b[key]

        sorting_a = prepare_sorting_for_comparison(load_sorting(run_a))
        sorting_b = prepare_sorting_for_comparison(load_sorting(run_b))
        comparison = compare_sortings_locally(
            sorting_a,
            sorting_b,
            delta_time_ms=args.delta_time_ms,
            match_score=args.match_score,
        )
        rows_a, rows_b = build_unit_rows(
            comparison=comparison,
            run_a=run_a,
            run_b=run_b,
            prepared_a=sorting_a,
            prepared_b=sorting_b,
            subset_coverage=args.subset_coverage,
        )

        pair_summaries.append(
            summarize_pair(
                comparison=comparison,
                run_a=run_a,
                run_b=run_b,
                prepared_a=sorting_a,
                prepared_b=sorting_b,
                subset_coverage=args.subset_coverage,
            )
        )
        unit_rows_a.extend(rows_a)
        unit_rows_b.extend(rows_b)

    overview = {
        "pipeline_a_root": str(args.pipeline_a.expanduser().resolve()),
        "pipeline_b_root": str(args.pipeline_b.expanduser().resolve()),
        "pipeline_a_name": args.pipeline_a_name,
        "pipeline_b_name": args.pipeline_b_name,
        "delta_time_ms": args.delta_time_ms,
        "match_score": args.match_score,
        "subset_coverage": args.subset_coverage,
        "common_pairs_compared": len(common_keys),
        "only_in_pipeline_a": [{"shank_id": shank_id, "window_label": window_label} for shank_id, window_label in only_a],
        "only_in_pipeline_b": [{"shank_id": shank_id, "window_label": window_label} for shank_id, window_label in only_b],
        "pair_summaries": pair_summaries,
    }

    write_csv(args.output / "pair_summary.csv", pair_summaries)
    write_csv(args.output / f"{args.pipeline_a_name}_units_vs_{args.pipeline_b_name}.csv", unit_rows_a)
    write_csv(args.output / f"{args.pipeline_b_name}_units_vs_{args.pipeline_a_name}.csv", unit_rows_b)
    (args.output / "comparison_overview.json").write_text(
        json.dumps(overview, indent=2),
        encoding="utf-8",
    )

    print(f"Compared {len(common_keys)} common shank/window pairs.")
    print(f"Reports written to: {args.output.resolve()}")
    for summary in pair_summaries:
        print(
            f"shank {summary['shank_id']} window {summary['window_label']}: "
            f"{summary['relation']} "
            f"({args.pipeline_a_name}: {summary['pipeline_a_units']} units, "
            f"{args.pipeline_b_name}: {summary['pipeline_b_units']} units)"
        )
    if only_a:
        print(f"Pairs only found in {args.pipeline_a_name}: {len(only_a)}")
    if only_b:
        print(f"Pairs only found in {args.pipeline_b_name}: {len(only_b)}")


if __name__ == "__main__":
    main()
