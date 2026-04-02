from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import probeinterface
from probeinterface import read_probeinterface
import spikeinterface.extractors as se
import spikeinterface.full as si
from pynwb import NWBHDF5IO

if not hasattr(probeinterface, "read_spikegadgets"):
    def _probeinterface_read_spikegadgets_stub(*args, **kwargs):
        return None

    probeinterface.read_spikegadgets = _probeinterface_read_spikegadgets_stub


def load_recording(file_path: Path):
    """Load a recording with the appropriate SpikeInterface reader."""
    suffix = file_path.suffix.lower()

    if suffix == ".rec":
        return si.read_spikegadgets(file_path=str(file_path))

    if suffix == ".nwb":
        if hasattr(se, "read_nwb_recording"):
            return se.read_nwb_recording(file_path=str(file_path))
        if hasattr(si, "read_nwb_recording"):
            return si.read_nwb_recording(file_path=str(file_path))
        if hasattr(se, "NwbRecordingExtractor"):
            return se.NwbRecordingExtractor(file_path=str(file_path))
        if hasattr(si, "NwbRecordingExtractor"):
            return si.NwbRecordingExtractor(file_path=str(file_path))
        raise RuntimeError(
            "NWB reader not available in current SpikeInterface install. "
            "Expected read_nwb_recording or NwbRecordingExtractor."
        )

    raise ValueError(
        f"Unsupported file type for {file_path}. "
        "Only .rec and .nwb files are supported."
    )


def collect_input_files(paths: Iterable[str]) -> list[Path]:
    """Expand input paths into a flat list of .rec/.nwb files."""
    files: list[Path] = []

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            if path.suffix.lower() not in {".rec", ".nwb"}:
                raise ValueError(f"Unsupported file: {path}")
            files.append(path)
            continue

        rec_files = sorted(path.glob("*.rec"))
        nwb_files = sorted(path.glob("*.nwb"))
        files.extend(rec_files)
        files.extend(nwb_files)

    unique_files = sorted(dict.fromkeys(p.resolve() for p in files))
    if not unique_files:
        raise FileNotFoundError("No .rec or .nwb files were found in the provided paths.")
    return unique_files


def collect_nwb_files(paths: Iterable[str]) -> list[Path]:
    """Expand input paths into a flat list of .nwb files only."""
    files: list[Path] = []

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            if path.suffix.lower() != ".nwb":
                raise ValueError(f"Expected .nwb file, got: {path}")
            files.append(path)
            continue

        files.extend(sorted(path.rglob("*.nwb")))

    unique_files = sorted(dict.fromkeys(p.resolve() for p in files))
    if not unique_files:
        raise FileNotFoundError("No .nwb files were found in the provided paths.")
    return unique_files


def summarize_array(values, max_items: int = 5) -> str:
    """Short summary for gains/offsets arrays."""
    if values is None:
        return "None"

    values = np.asarray(values)
    if values.size == 0:
        return "[]"

    preview = ", ".join(str(x) for x in values[:max_items])
    if values.size > max_items:
        preview += ", ..."

    unique_count = np.unique(values).size
    return f"[{preview}] (n={values.size}, unique={unique_count})"


def summarize_list(values, max_items: int = 12) -> str:
    """Short summary for channel IDs and similar lists."""
    if values is None:
        return "None"

    values = list(values)
    if not values:
        return "[]"

    preview = ", ".join(str(x) for x in values[:max_items])
    if len(values) > max_items:
        preview += ", ..."
    return f"[{preview}] (n={len(values)})"


def format_duration(duration_seconds: float) -> str:
    """Format seconds as HH:MM:SS.ss."""
    total_seconds = float(duration_seconds)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def normalize_channel_token(channel_id) -> str:
    """Normalize channel identifiers across int/str/NumPy scalar types."""
    value = channel_id.item() if isinstance(channel_id, np.generic) else channel_id

    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def infer_probe_sidecar(nwb_path: Path) -> Path:
    """Infer the probe sidecar path created by the sorting pipeline."""
    return nwb_path.parent / f"{nwb_path.stem}_probe.json"


def get_channel_tokens(recording, file_path: Path) -> tuple[list[str], list]:
    """
    Return comparable channel tokens and the original channel IDs used to read traces.

    For pipeline NWBs, prefer the probe sidecar's device_channel_indices when present.
    """
    original_channel_ids = list(recording.get_channel_ids())
    comparable_tokens = [normalize_channel_token(ch) for ch in original_channel_ids]

    if file_path.suffix.lower() != ".nwb":
        return comparable_tokens, original_channel_ids

    sidecar_path = infer_probe_sidecar(file_path)
    if not sidecar_path.exists():
        return comparable_tokens, original_channel_ids

    try:
        probe_group = read_probeinterface(sidecar_path)
        if not probe_group.probes:
            return comparable_tokens, original_channel_ids

        device_channel_indices = list(probe_group.probes[0].device_channel_indices)
        if len(device_channel_indices) != len(original_channel_ids):
            return comparable_tokens, original_channel_ids

        comparable_tokens = [normalize_channel_token(ch) for ch in device_channel_indices]
    except Exception:
        return comparable_tokens, original_channel_ids

    return comparable_tokens, original_channel_ids


def inspect_recording(file_path: Path, sample_frames: int = 1000) -> dict:
    """Collect datatype and recording metadata."""
    recording = load_recording(file_path)

    dtype_declared = recording.get_dtype()
    num_channels = recording.get_num_channels()
    num_segments = recording.get_num_segments()
    num_frames = recording.get_num_frames(segment_index=0)
    sampling_frequency = recording.get_sampling_frequency()
    channel_ids = recording.get_channel_ids()
    gains = recording.get_channel_gains()
    offsets = recording.get_channel_offsets()
    duration_seconds = num_frames / sampling_frequency if sampling_frequency else 0.0

    end_frame = min(sample_frames, num_frames)
    trace_sample = recording.get_traces(start_frame=0, end_frame=end_frame)

    return {
        "path": file_path,
        "extractor": type(recording).__name__,
        "dtype_declared": str(dtype_declared),
        "dtype_sample": str(trace_sample.dtype),
        "sample_shape": tuple(trace_sample.shape),
        "sampling_frequency_hz": sampling_frequency,
        "num_channels": num_channels,
        "num_frames_segment0": num_frames,
        "duration_seconds": duration_seconds,
        "duration_hms": format_duration(duration_seconds),
        "num_segments": num_segments,
        "channel_ids": list(channel_ids),
        "channel_id_type": type(channel_ids[0]).__name__ if len(channel_ids) else "None",
        "gains": summarize_array(gains),
        "offsets": summarize_array(offsets),
    }


def inspect_nwb_spikegadget_correspondence(nwb_path: Path) -> dict:
    """
    Inspect how NWB channel order corresponds to stored electrode metadata.

    For NWBs written by this pipeline, ``device_channel_index`` is the
    SpikeGadgets channel index that was written into the electrode table.
    """
    recording = load_recording(nwb_path)
    extractor_channel_ids = list(recording.get_channel_ids())
    extractor_channel_tokens = [normalize_channel_token(ch) for ch in extractor_channel_ids]

    rows: list[dict] = []
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        electrodes_df = nwbfile.electrodes.to_dataframe()
        electrical_series = nwbfile.acquisition.get("ElectricalSeries")
        if electrical_series is None:
            raise RuntimeError("No acquisition['ElectricalSeries'] found in NWB file.")

        electrode_region = list(electrical_series.electrodes.data[:])
        if len(electrode_region) != len(extractor_channel_ids):
            raise RuntimeError(
                "ElectricalSeries electrode count does not match extracted channel count "
                f"({len(electrode_region)} vs {len(extractor_channel_ids)})."
            )

        for order_index, electrode_table_index in enumerate(electrode_region):
            electrode_row = electrodes_df.iloc[int(electrode_table_index)]
            group_name = getattr(electrode_row.get("group", None), "name", "None")
            rows.append(
                {
                    "order_index": order_index,
                    "extractor_channel_id": extractor_channel_ids[order_index],
                    "extractor_channel_token": extractor_channel_tokens[order_index],
                    "electrode_table_index": int(electrode_table_index),
                    "device_channel_index": electrode_row.get("device_channel_index", None),
                    "label": electrode_row.get("label", None),
                    "group_name": group_name,
                    "x": electrode_row.get("rel_x", None),
                    "y": electrode_row.get("rel_y", None),
                }
            )

    return {
        "path": nwb_path,
        "channel_count": len(rows),
        "rows": rows,
    }


def print_nwb_spikegadget_correspondence(summary: dict, max_rows: int | None = None) -> None:
    print("=" * 80)
    print("NWB channel to SpikeGadgets correspondence")
    print(f"NWB file: {summary['path']}")
    print(f"Mapped channels: {summary['channel_count']}")
    print(
        "Columns: order_index | extractor_channel_id | electrode_table_index | "
        "device_channel_index | label | group"
    )

    rows = summary["rows"]
    if max_rows is not None:
        rows = rows[:max_rows]

    for row in rows:
        print(
            f"{row['order_index']:>4} | "
            f"{str(row['extractor_channel_id']):>20} | "
            f"{row['electrode_table_index']:>5} | "
            f"{str(row['device_channel_index']):>20} | "
            f"{str(row['label']):<20} | "
            f"{row['group_name']}"
        )

    if max_rows is not None and summary["channel_count"] > max_rows:
        print(f"... showing first {max_rows} of {summary['channel_count']} mapped channels")


def compare_recordings(
    rec_path: Path,
    nwb_path: Path,
    sample_frames: int = 1000,
    max_channels: int | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict:
    """Compare traces between a .rec file and an .nwb file."""
    rec_recording = load_recording(rec_path)
    nwb_recording = load_recording(nwb_path)

    rec_tokens, rec_channel_ids = get_channel_tokens(rec_recording, rec_path)
    nwb_tokens, nwb_channel_ids = get_channel_tokens(nwb_recording, nwb_path)

    rec_token_to_channel = {
        token: channel_id for token, channel_id in zip(rec_tokens, rec_channel_ids)
    }
    nwb_token_to_channel = {
        token: channel_id for token, channel_id in zip(nwb_tokens, nwb_channel_ids)
    }

    shared_tokens = sorted(
        set(rec_token_to_channel).intersection(nwb_token_to_channel),
        key=lambda token: (int(token) if token.lstrip("-").isdigit() else token),
    )
    if not shared_tokens:
        raise RuntimeError(
            "No shared channels found between the .rec and .nwb files. "
            "If the NWB is a shank export, make sure its *_probe.json sidecar is present."
        )

    if max_channels is not None:
        shared_tokens = shared_tokens[:max_channels]

    rec_selected_channels = [rec_token_to_channel[token] for token in shared_tokens]
    nwb_selected_channels = [nwb_token_to_channel[token] for token in shared_tokens]

    rec_num_frames = rec_recording.get_num_frames(segment_index=0)
    nwb_num_frames = nwb_recording.get_num_frames(segment_index=0)
    rec_sampling_frequency = rec_recording.get_sampling_frequency()
    nwb_sampling_frequency = nwb_recording.get_sampling_frequency()
    compared_frames = min(sample_frames, rec_num_frames, nwb_num_frames)
    if compared_frames <= 0:
        raise RuntimeError("No frames available to compare.")

    rec_traces = rec_recording.get_traces(
        channel_ids=rec_selected_channels,
        start_frame=0,
        end_frame=compared_frames,
    )
    nwb_traces = nwb_recording.get_traces(
        channel_ids=nwb_selected_channels,
        start_frame=0,
        end_frame=compared_frames,
    )

    diff = rec_traces.astype(np.float64) - nwb_traces.astype(np.float64)
    abs_diff = np.abs(diff)

    return {
        "rec_path": rec_path,
        "nwb_path": nwb_path,
        "rec_extractor": type(rec_recording).__name__,
        "nwb_extractor": type(nwb_recording).__name__,
        "rec_dtype": str(rec_recording.get_dtype()),
        "nwb_dtype": str(nwb_recording.get_dtype()),
        "rec_sampling_frequency_hz": rec_sampling_frequency,
        "nwb_sampling_frequency_hz": nwb_sampling_frequency,
        "rec_num_frames_segment0": rec_num_frames,
        "nwb_num_frames_segment0": nwb_num_frames,
        "rec_duration_seconds": rec_num_frames / rec_sampling_frequency if rec_sampling_frequency else 0.0,
        "nwb_duration_seconds": nwb_num_frames / nwb_sampling_frequency if nwb_sampling_frequency else 0.0,
        "rec_duration_hms": format_duration(rec_num_frames / rec_sampling_frequency if rec_sampling_frequency else 0.0),
        "nwb_duration_hms": format_duration(nwb_num_frames / nwb_sampling_frequency if nwb_sampling_frequency else 0.0),
        "shared_channel_count": len(shared_tokens),
        "compared_channels": shared_tokens,
        "compared_frames": compared_frames,
        "shape_equal": rec_traces.shape == nwb_traces.shape,
        "exact_equal": np.array_equal(rec_traces, nwb_traces),
        "allclose": np.allclose(rec_traces, nwb_traces, atol=atol, rtol=rtol),
        "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
    }


def print_report(summary: dict) -> None:
    print("=" * 80)
    print(f"File: {summary['path']}")
    print(f"Extractor: {summary['extractor']}")
    print(f"Declared dtype: {summary['dtype_declared']}")
    print(f"Trace sample dtype: {summary['dtype_sample']}")
    print(f"Trace sample shape: {summary['sample_shape']}")
    print(f"Sampling frequency (Hz): {summary['sampling_frequency_hz']}")
    print(f"Channels: {summary['num_channels']}")
    print(f"Frames in segment 0: {summary['num_frames_segment0']}")
    print(f"Duration (s): {summary['duration_seconds']}")
    print(f"Duration (HH:MM:SS.ss): {summary['duration_hms']}")
    print(f"Segments: {summary['num_segments']}")
    print(f"Channel IDs: {summarize_list(summary['channel_ids'])}")
    print(f"Channel ID type: {summary['channel_id_type']}")
    print(f"Channel gains: {summary['gains']}")
    print(f"Channel offsets: {summary['offsets']}")

    if summary["dtype_declared"] != summary["dtype_sample"]:
        print("Note: declared dtype and sampled trace dtype differ.")


def print_comparison_report(summary: dict) -> None:
    print("=" * 80)
    print("Comparison report")
    print(f".rec file: {summary['rec_path']}")
    print(f".nwb file: {summary['nwb_path']}")
    print(f".rec extractor: {summary['rec_extractor']}")
    print(f".nwb extractor: {summary['nwb_extractor']}")
    print(f".rec dtype: {summary['rec_dtype']}")
    print(f".nwb dtype: {summary['nwb_dtype']}")
    print(f".rec sampling frequency (Hz): {summary['rec_sampling_frequency_hz']}")
    print(f".nwb sampling frequency (Hz): {summary['nwb_sampling_frequency_hz']}")
    print(f".rec frames in segment 0: {summary['rec_num_frames_segment0']}")
    print(f".nwb frames in segment 0: {summary['nwb_num_frames_segment0']}")
    print(f".rec duration (s): {summary['rec_duration_seconds']}")
    print(f".nwb duration (s): {summary['nwb_duration_seconds']}")
    print(f".rec duration (HH:MM:SS.ss): {summary['rec_duration_hms']}")
    print(f".nwb duration (HH:MM:SS.ss): {summary['nwb_duration_hms']}")
    print(f"Shared channels found: {summary['shared_channel_count']}")
    print(f"Frames compared: {summary['compared_frames']}")
    print(f"Shape equal: {summary['shape_equal']}")
    print(f"Exact equal: {summary['exact_equal']}")
    print(f"All close: {summary['allclose']}")
    print(f"Max abs diff: {summary['max_abs_diff']}")
    print(f"Mean abs diff: {summary['mean_abs_diff']}")

    preview = ", ".join(summary["compared_channels"][:10])
    if summary["shared_channel_count"] > 10:
        preview += ", ..."
    print(f"Compared channel tokens: {preview}")

    if summary["rec_sampling_frequency_hz"] != summary["nwb_sampling_frequency_hz"]:
        print("Note: sampling frequencies differ, so frame-by-frame equality is not expected.")
    if summary["rec_dtype"] != summary["nwb_dtype"]:
        print("Note: dtypes differ, which can also explain numerical differences.")


def compare_rec_to_nwb_group(
    rec_path: Path,
    nwb_paths: list[Path],
    sample_frames: int = 1000,
    max_channels: int | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> list[dict]:
    """Compare one .rec file against multiple .nwb files."""
    results: list[dict] = []

    for nwb_path in nwb_paths:
        try:
            results.append(
                compare_recordings(
                    rec_path=rec_path,
                    nwb_path=nwb_path,
                    sample_frames=sample_frames,
                    max_channels=max_channels,
                    atol=atol,
                    rtol=rtol,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "rec_path": rec_path,
                    "nwb_path": nwb_path,
                    "error": str(exc),
                }
            )

    return results


def print_group_comparison_report(results: list[dict]) -> None:
    print("=" * 80)
    print("Batch comparison summary")
    print(f"NWB files compared: {len(results)}")

    success_count = sum(1 for result in results if "error" not in result)
    error_count = len(results) - success_count
    exact_count = sum(1 for result in results if result.get("exact_equal"))
    allclose_count = sum(1 for result in results if result.get("allclose"))

    print(f"Successful comparisons: {success_count}")
    print(f"Comparisons with errors: {error_count}")
    print(f"Exact equal count: {exact_count}")
    print(f"All close count: {allclose_count}")

    for result in results:
        print("-" * 80)
        print(f"NWB: {result['nwb_path']}")
        if "error" in result:
            print(f"Error: {result['error']}")
            continue

        print(f"Shared channels: {result['shared_channel_count']}")
        print(f"Frames compared: {result['compared_frames']}")
        print(f"Exact equal: {result['exact_equal']}")
        print(f"All close: {result['allclose']}")
        print(f"Max abs diff: {result['max_abs_diff']}")
        print(f"Mean abs diff: {result['mean_abs_diff']}")
        print(
            "Durations (rec/nwb): "
            f"{result['rec_duration_hms']} / {result['nwb_duration_hms']}"
        )
        print(
            "Sampling frequencies (rec/nwb): "
            f"{result['rec_sampling_frequency_hz']} / {result['nwb_sampling_frequency_hz']}"
        )
        print(f"Dtypes (rec/nwb): {result['rec_dtype']} / {result['nwb_dtype']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check SpikeInterface datatype and metadata for .rec and .nwb files."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="One or more .rec/.nwb files or folders containing them.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=1000,
        help="Number of frames to read for the sample dtype check.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("REC_PATH", "NWB_PATH"),
        help="Compare one .rec file against one .nwb file while keeping the datatype report.",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=None,
        help="Limit the number of shared channels used during --compare.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for the numeric allclose comparison.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for the numeric allclose comparison.",
    )
    parser.add_argument(
        "--compare-many",
        nargs="+",
        metavar=("REC_PATH", "NWB_PATH_OR_DIR"),
        help=(
            "Compare one .rec file against multiple .nwb files. "
            "Pass a .rec path first, then one or more .nwb files or folders."
        ),
    )
    parser.add_argument(
        "--show-spikegadget-map",
        nargs="+",
        metavar="NWB_PATH_OR_DIR",
        help=(
            "Show the NWB channel order and its stored device_channel_index "
            "(SpikeGadgets channel index) for one or more NWB files or folders."
        ),
    )
    parser.add_argument(
        "--map-max-rows",
        type=int,
        default=None,
        help="Limit the number of channel-mapping rows printed by --show-spikegadget-map.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_paths = list(args.paths)
    if args.compare:
        input_paths.extend(args.compare)
    if args.compare_many:
        input_paths.extend(args.compare_many)
    if args.show_spikegadget_map:
        input_paths.extend(args.show_spikegadget_map)

    if not input_paths:
        manual_path = input(
            "Enter one or more .rec/.nwb files or folders (comma-separated): "
        ).strip()
        input_paths = [p.strip().strip('"').strip("'") for p in manual_path.split(",") if p.strip()]

    if not input_paths:
        raise ValueError("No input paths were provided.")

    files = collect_input_files(input_paths)
    for file_path in files:
        try:
            summary = inspect_recording(file_path, sample_frames=args.sample_frames)
            print_report(summary)
            if file_path.suffix.lower() == ".nwb":
                mapping_summary = inspect_nwb_spikegadget_correspondence(file_path)
                print_nwb_spikegadget_correspondence(
                    mapping_summary,
                    max_rows=args.map_max_rows,
                )
        except Exception as exc:
            print("=" * 80)
            print(f"File: {file_path}")
            print(f"Error: {exc}")

    if args.compare:
        rec_path = Path(args.compare[0]).expanduser().resolve()
        nwb_path = Path(args.compare[1]).expanduser().resolve()
        try:
            comparison = compare_recordings(
                rec_path=rec_path,
                nwb_path=nwb_path,
                sample_frames=args.sample_frames,
                max_channels=args.max_channels,
                atol=args.atol,
                rtol=args.rtol,
            )
            print_comparison_report(comparison)
        except Exception as exc:
            print("=" * 80)
            print("Comparison report")
            print(f".rec file: {rec_path}")
            print(f".nwb file: {nwb_path}")
            print(f"Error: {exc}")

    if args.compare_many:
        if len(args.compare_many) < 2:
            raise ValueError("--compare-many requires a .rec path and at least one .nwb path or folder.")

        rec_path = Path(args.compare_many[0]).expanduser().resolve()
        nwb_paths = collect_nwb_files(args.compare_many[1:])

        print("=" * 80)
        print(f"Running batch comparison for .rec file: {rec_path}")
        print(f"Found {len(nwb_paths)} NWB file(s) to compare.")

        results = compare_rec_to_nwb_group(
            rec_path=rec_path,
            nwb_paths=nwb_paths,
            sample_frames=args.sample_frames,
            max_channels=args.max_channels,
            atol=args.atol,
            rtol=args.rtol,
        )
        print_group_comparison_report(results)

    if args.show_spikegadget_map:
        nwb_paths = collect_nwb_files(args.show_spikegadget_map)
        for nwb_path in nwb_paths:
            try:
                summary = inspect_nwb_spikegadget_correspondence(nwb_path)
                print_nwb_spikegadget_correspondence(summary, max_rows=args.map_max_rows)
            except Exception as exc:
                print("=" * 80)
                print("NWB channel to SpikeGadgets correspondence")
                print(f"NWB file: {nwb_path}")
                print(f"Error: {exc}")


if __name__ == "__main__":
    main()
