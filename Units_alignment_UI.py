"""
Units Alignment UI

Usage notes for this script:
- Input should be one shank folder only.
- The UI builds one page per SG channel and lets the user review that single shank
  across all sessions.
- Merge groups are for duplicate units within the same session.
- Align groups are for the same neuron across different sessions on the same
  SG channel page.
- Units that fail the discard thresholds are auto-marked as discarded and
  shown in a separate review section.
- Noise units are excluded from the exported summary.
- Export writes summary files only. This script no longer creates curated session
  analyzers.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


MAX_SPIKES_PER_UNIT = 500
MS_BEFORE = 1.0
MS_AFTER = 2.0
CARD_IMAGE_SIZE = (320, 205)
SESSION_COLUMN_MIN_WIDTH = 360
DEFAULT_EXPORT_FOLDER_NAME = "units_alignment_summary"
DESIRED_METRICS = [
    "amplitude_median",
    "firing_rate",
    "isi_violations_ratio",
    "snr",
    "num_spikes",
]
WAVEFORM_SIMILARITY_WEIGHT = 0.8
AMPLITUDE_SIMILARITY_WEIGHT = 0.2
DISCARD_ABS_AMPLITUDE_MAX = 15.0
DISCARD_SNR_MAX = 3.0
DISCARD_ISI_VIOLATION_MIN = 2.0

if Image is not None:
    try:
        PIL_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    except AttributeError:
        PIL_RESAMPLE_LANCZOS = Image.LANCZOS
else:
    PIL_RESAMPLE_LANCZOS = None


@dataclass
class UnitSummary:
    session_name: str
    session_index: int
    analyzer_folder: str
    output_folder: str
    unit_id: int
    shank_id: int
    local_channel_on_shank: int
    sg_channel: int
    amplitude_median: float | None
    firing_rate: float | None
    isi_violations_ratio: float | None
    snr: float | None
    num_spikes: int | None
    waveform_similarity_vector: list[float]
    waveform_image_path: str
    merge_group: str = ""
    align_group: str = ""
    exclude_from_auto_align: bool = False
    is_discarded: bool = False
    is_noise: bool = False


@dataclass
class SessionSummary:
    session_name: str
    session_index: int
    output_folder: str
    analyzer_folder: str
    units: list[UnitSummary] = field(default_factory=list)

    @property
    def safe_name(self) -> str:
        return sanitize_token(self.session_name)


@dataclass
class PageSummary:
    shank_id: int
    sg_channel: int
    sessions: list[SessionSummary]

    @property
    def page_id(self) -> str:
        return f"sg{self.sg_channel}"

    @property
    def title(self) -> str:
        return f"SG Channel {self.sg_channel}"


REVIEW_PAGE_ID = "__review__"
DISCARDED_PAGE_ID = "__discarded__"


@dataclass
class SimilarityCandidate:
    left_key: str
    right_key: str
    left_label: str
    right_label: str
    score: float


def make_hidden_root() -> tk.Tk:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def choose_output_root() -> Path:
    root = make_hidden_root()
    selected_folder = filedialog.askdirectory(
        title="Select a Sorting+Analyze output root",
        mustexist=True,
        parent=root,
    )
    root.destroy()
    if not selected_folder:
        raise SystemExit("No output folder selected.")
    return Path(selected_folder)


def create_loading_window(root: tk.Tk) -> tuple[tk.Toplevel, tk.StringVar]:
    window = tk.Toplevel()
    window.title("Loading Units Alignment UI")
    window.geometry("520x140")
    window.resizable(False, False)
    window.attributes("-topmost", True)

    message_var = tk.StringVar(value="Preparing startup...")
    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Loading data",
        font=("Segoe UI", 12, "bold"),
    ).pack(anchor="w")
    ttk.Label(
        frame,
        textvariable=message_var,
        justify="left",
        wraplength=480,
    ).pack(anchor="w", pady=(10, 0))
    ttk.Label(
        frame,
        text="Large folders can take a while before the main window is ready.",
        justify="left",
        wraplength=480,
    ).pack(anchor="w", pady=(10, 0))

    window.update_idletasks()
    return window, message_var


def safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if np.isnan(value):
            return None
        return value
    except Exception:
        return None


def safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(value)
    except Exception:
        return None


def load_unit_channel_mapping(output_folder: Path) -> dict[int, dict]:
    report_path = output_folder / "unit_channel_mapping_report.json"
    if not report_path.exists():
        return {}

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    mapping: dict[int, dict] = {}
    for row in payload.get("units", []):
        unit_id = safe_int(row.get("unit_id"))
        if unit_id is None:
            continue
        mapping[unit_id] = row
    return mapping


def find_unit_summary_image(output_folder: Path, unit_id: int) -> Path | None:
    summary_folder = output_folder / "unit_summaries_analysis"
    if summary_folder.exists():
        matches = sorted(summary_folder.glob(f"unit_summary_*_{int(unit_id)}.png"))
        if matches:
            return matches[0]

    waveform_folder = output_folder / "unit_waveforms_analysis"
    if waveform_folder.exists():
        matches = sorted(waveform_folder.glob(f"unit_waveform_*_{int(unit_id)}.png"))
        if matches:
            return matches[0]

    return None


def format_metric(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "nan"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def sanitize_token(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "item"


def discover_analyzer_folders(root_folder: Path) -> list[Path]:
    analyzer_folders = sorted(
        {
            p
            for p in root_folder.rglob("sorting_analyzer_analysis.zarr")
            if p.is_dir()
        },
        key=session_sort_key,
    )
    if not analyzer_folders:
        raise FileNotFoundError(
            f"No sorting_analyzer_analysis.zarr folders found under {root_folder}"
        )
    return analyzer_folders


def session_sort_key(path: Path):
    name = path.parent.name
    digits = re.findall(r"\d+", name)
    numeric_key = int(digits[-1]) if digits else 10**9
    return (numeric_key, name.lower(), str(path))


def session_name_from_output_folder(output_folder: Path, index: int) -> str:
    name = output_folder.name.strip()
    if name:
        return name
    return f"Session {index + 1}"


def ensure_required_extensions(analyzer) -> None:
    required_extensions = {
        "random_spikes": {"method": "uniform", "max_spikes_per_unit": MAX_SPIKES_PER_UNIT},
        "waveforms": {"ms_before": MS_BEFORE, "ms_after": MS_AFTER},
        "templates": {"operators": ["average", "median", "std"]},
        "noise_levels": {},
        "spike_amplitudes": {"peak_sign": "neg"},
        "quality_metrics": {},
        "unit_locations": {"method": "monopolar_triangulation"},
        "correlograms": {"window_ms": 50.0, "bin_ms": 1.0, "method": "auto"},
        "isi_histograms": {"window_ms": 50.0, "bin_ms": 1.0, "method": "auto"},
        "template_similarity": {"method": "cosine_similarity"},
    }
    for extension_name, kwargs in required_extensions.items():
        if not analyzer.has_extension(extension_name):
            analyzer.compute(extension_name, **kwargs)


def build_metrics_lookup(analyzer) -> dict[int, dict[str, float | int]]:
    if not analyzer.has_extension("quality_metrics"):
        return {}

    metrics_df = analyzer.get_extension("quality_metrics").get_data()
    if "unit_id" not in metrics_df.columns:
        metrics_df = metrics_df.reset_index()
        if "unit_id" not in metrics_df.columns and "index" in metrics_df.columns:
            metrics_df = metrics_df.rename(columns={"index": "unit_id"})
    if "unit_id" not in metrics_df.columns:
        return {}

    lookup: dict[int, dict[str, float | int]] = {}
    for row in metrics_df.itertuples(index=False):
        unit_id = safe_int(getattr(row, "unit_id", None))
        if unit_id is None:
            continue
        lookup[unit_id] = {metric_name: getattr(row, metric_name, None) for metric_name in DESIRED_METRICS}
    return lookup


def infer_unit_channel_metadata(analyzer, unit_id, mapping_row: dict | None = None) -> dict[str, int]:
    if mapping_row is not None:
        shank_id = safe_int(mapping_row.get("shank_id"))
        local_channel = safe_int(mapping_row.get("waveform_local_channel_index"))
        sg_channel = safe_int(mapping_row.get("device_channel_index_property"))
        if shank_id is not None and local_channel is not None:
            if sg_channel is None:
                sg_channel = local_channel
            return {
                "shank_id": shank_id,
                "local_channel_on_shank": local_channel,
                "sg_channel": sg_channel,
            }

    templates_ext = analyzer.get_extension("templates")
    average_templates = templates_ext.get_data(operator="average", outputs="numpy")
    unit_ids = list(analyzer.sorting.get_unit_ids())
    unit_index = unit_ids.index(unit_id)
    template = average_templates[unit_index]
    channel_index = int(np.argmax(np.max(np.abs(template), axis=0)))

    channel_ids = list(analyzer.channel_ids)
    sg_channel = channel_ids[channel_index] if channel_index < len(channel_ids) else channel_index
    try:
        sg_channel = int(sg_channel)
    except Exception:
        sg_channel = channel_index

    shank_id: int | None = None
    try:
        group_values = analyzer.get_recording_property("group")
    except Exception:
        group_values = None
    if group_values is not None and len(group_values) > channel_index:
        match = re.search(r"(\d+)", str(group_values[channel_index]))
        if match:
            shank_id = int(match.group(1))
    elif analyzer.rec_attributes.get("probegroup") is not None:
        try:
            probegroup = analyzer.rec_attributes.get("probegroup")
            probes = getattr(probegroup, "probes", None)
            if probes:
                probe = probes[0]
                if hasattr(probe, "shank_ids") and channel_index < len(probe.shank_ids):
                    shank_id = int(probe.shank_ids[channel_index])
        except Exception:
            pass

    if shank_id is None:
        raise ValueError(
            "Could not determine shank_id from analyzer recording metadata. "
            "This UI now requires explicit single-shank metadata."
        )

    return {
        "shank_id": shank_id,
        "local_channel_on_shank": channel_index,
        "sg_channel": sg_channel,
    }


def get_waveform_vector(analyzer, unit_id) -> np.ndarray:
    waveforms_ext = analyzer.get_extension("waveforms")
    waveforms = waveforms_ext.get_waveforms_one_unit(unit_id)
    if waveforms is None or getattr(waveforms, "size", 0) == 0:
        return np.zeros(1, dtype=float)

    mean_waveform = waveforms.mean(axis=0)
    channel_index = int(np.argmax(np.max(np.abs(mean_waveform), axis=0)))
    single_channel_mean = mean_waveform[:, channel_index].astype(float)
    norm = np.linalg.norm(single_channel_mean)
    if norm == 0:
        return single_channel_mean
    return single_channel_mean / norm


def save_waveform_card_image(
    analyzer,
    unit_id: int,
    save_path: Path,
    session_name: str,
    shank_id: int,
    channel_id: int,
) -> None:
    waveforms_ext = analyzer.get_extension("waveforms")
    waveforms = waveforms_ext.get_waveforms_one_unit(unit_id)
    if waveforms is None or getattr(waveforms, "size", 0) == 0:
        fig, ax = plt.subplots(figsize=(6.8, 2.8))
        ax.text(0.5, 0.5, "No waveform", ha="center", va="center")
        ax.axis("off")
    else:
        mean_waveform = waveforms.mean(axis=0)
        channel_index = int(np.argmax(np.max(np.abs(mean_waveform), axis=0)))
        single_channel_waveforms = waveforms[:, :, channel_index]
        average_waveform = single_channel_waveforms.mean(axis=0)
        std_waveform = single_channel_waveforms.std(axis=0)
        time_axis = np.linspace(-MS_BEFORE, MS_AFTER, average_waveform.shape[0])

        fig, ax = plt.subplots(figsize=(6.8, 2.8))
        ax.fill_between(
            time_axis,
            average_waveform - std_waveform,
            average_waveform + std_waveform,
            color="#9ecae1",
            alpha=0.6,
        )
        ax.plot(time_axis, average_waveform, color="#08519c", linewidth=2.0)
        ax.axhline(0, color="#bdbdbd", linewidth=0.8)
        ax.set_title(f"{session_name} | Unit {unit_id}", fontsize=10)
        ax.set_xlabel("ms")
        ax.set_ylabel("uV")
        ax.text(
            0.98,
            0.05,
            f"sh {shank_id} ch {channel_id}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def load_all_sessions(
    root_folder: Path,
    progress_callback=None,
) -> tuple[list[SessionSummary], dict[str, PageSummary], Path]:
    analyzer_folders = discover_analyzer_folders(root_folder)
    cache_folder = root_folder / DEFAULT_EXPORT_FOLDER_NAME / "_cache"
    sessions: list[SessionSummary] = []
    discovered_shank_ids: set[int] = set()
    total_sessions = len(analyzer_folders)

    if progress_callback is not None:
        progress_callback(
            f"Found {total_sessions} analyzer folder(s).\nLoading sessions and preparing waveform pages..."
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

            unit_summary = UnitSummary(
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
                waveform_image_path=str(preferred_image_path or image_path),
            )
            session_summary.units.append(unit_summary)
            discovered_shank_ids.add(int(metadata["shank_id"]))

        sessions.append(session_summary)

    if len(discovered_shank_ids) > 1:
        raise ValueError(
            "The selected input contains multiple shanks, but this UI assumes one shank only. "
            f"Found shank ids: {sorted(discovered_shank_ids)}"
        )

    page_summaries: dict[str, PageSummary] = {}
    if progress_callback is not None:
        progress_callback("Building channel pages...")
    page_keys: set[int] = set()
    for session in sessions:
        for unit in session.units:
            page_keys.add(unit.sg_channel)

    shank_id = next(iter(discovered_shank_ids), 0)
    for sg_channel in sorted(page_keys):
        aligned_sessions: list[SessionSummary] = []
        for session in sessions:
            filtered_units = [
                unit
                for unit in session.units
                if unit.sg_channel == sg_channel
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
            shank_id=shank_id,
            sg_channel=sg_channel,
            sessions=aligned_sessions,
        )
        page_summaries[page.page_id] = page

    return sessions, page_summaries, cache_folder


def unit_record_key(unit: UnitSummary) -> str:
    return f"{unit.session_index}:{unit.unit_id}"


def compute_amplitude_similarity(a: UnitSummary, b: UnitSummary) -> float:
    amplitude_a = safe_float(a.amplitude_median)
    amplitude_b = safe_float(b.amplitude_median)
    if amplitude_a is None or amplitude_b is None:
        return 0.5

    scale = max(abs(amplitude_a), abs(amplitude_b), 1e-6)
    relative_difference = abs(amplitude_a - amplitude_b) / scale
    return max(0.0, min(1.0, 1.0 - relative_difference))


def compute_waveform_similarity(a: UnitSummary, b: UnitSummary) -> float:
    va = np.asarray(a.waveform_similarity_vector, dtype=float)
    vb = np.asarray(b.waveform_similarity_vector, dtype=float)
    if va.size == 0 or vb.size == 0:
        return 0.0
    length = min(va.size, vb.size)
    va = va[:length]
    vb = vb[:length]
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    waveform_score = float(np.dot(va, vb) / (na * nb))
    return max(0.0, min(1.0, waveform_score))


def compute_similarity(a: UnitSummary, b: UnitSummary) -> float:
    waveform_score = compute_waveform_similarity(a, b)
    amplitude_score = compute_amplitude_similarity(a, b)
    score = (
        WAVEFORM_SIMILARITY_WEIGHT * waveform_score
        + AMPLITUDE_SIMILARITY_WEIGHT * amplitude_score
    )
    return max(0.0, min(1.0, score))


def is_unit_auto_discarded(unit: UnitSummary) -> bool:
    checks = evaluate_discard_criteria(unit)
    return checks["all_pass"]


def evaluate_discard_criteria(unit: UnitSummary) -> dict[str, bool | float | None]:
    amplitude = safe_float(unit.amplitude_median)
    snr = safe_float(unit.snr)
    isi_ratio = safe_float(unit.isi_violations_ratio)
    if amplitude is None or snr is None or isi_ratio is None:
        return {
            "all_pass": False,
            "amplitude_pass": False,
            "snr_pass": False,
            "isi_pass": False,
            "amplitude_abs": abs(amplitude) if amplitude is not None else None,
            "snr": snr,
            "isi_value": None,
        }
    amplitude_pass = abs(amplitude) < DISCARD_ABS_AMPLITUDE_MAX
    snr_pass = snr < DISCARD_SNR_MAX
    # Use the exact same ISI value that is shown on the unit card so the
    # displayed metric and discard logic cannot disagree.
    isi_pass = isi_ratio > DISCARD_ISI_VIOLATION_MIN
    return {
        "all_pass": amplitude_pass or snr_pass or isi_pass,
        "amplitude_pass": amplitude_pass,
        "snr_pass": snr_pass,
        "isi_pass": isi_pass,
        "amplitude_abs": abs(amplitude),
        "snr": snr,
        "isi_value": isi_ratio,
    }


def build_discard_reason(unit: UnitSummary) -> str:
    if not unit.is_discarded:
        return ""
    return (
        f"|amp|<{DISCARD_ABS_AMPLITUDE_MAX:g}, "
        f"SNR<{DISCARD_SNR_MAX:g}, "
        f"ISI>{DISCARD_ISI_VIOLATION_MIN:g}"
    )


def build_discard_check_text(unit: UnitSummary) -> str:
    checks = evaluate_discard_criteria(unit)
    amplitude_abs = checks["amplitude_abs"]
    snr = checks["snr"]
    isi_value = checks["isi_value"]
    return (
        "Discard check: "
        f"|amp|={format_metric(amplitude_abs)}"
        f" (<{DISCARD_ABS_AMPLITUDE_MAX:g}: {'yes' if checks['amplitude_pass'] else 'no'}), "
        f"SNR={format_metric(snr)}"
        f" (<{DISCARD_SNR_MAX:g}: {'yes' if checks['snr_pass'] else 'no'}), "
        f"ISI={format_metric(isi_value)}"
        f" (>{DISCARD_ISI_VIOLATION_MIN:g}: {'yes' if checks['isi_pass'] else 'no'})"
    )


def passes_auto_align_thresholds(
    a: UnitSummary,
    b: UnitSummary,
    min_waveform_similarity: float = 0.75,
    min_amplitude_similarity: float = 0.75,
) -> bool:
    waveform_score = compute_waveform_similarity(a, b)
    amplitude_score = compute_amplitude_similarity(a, b)
    return (
        waveform_score >= min_waveform_similarity
        and amplitude_score >= min_amplitude_similarity
    )


def build_strict_auto_align_rows(
    units: list[UnitSummary],
    min_similarity: float = 0.75,
) -> tuple[list[list[UnitSummary]], set[str]]:
    eligible_units = sorted(units, key=lambda item: (item.session_index, item.unit_id))
    candidate_pairs: list[tuple[float, float, float, str, str]] = []
    for i, left in enumerate(eligible_units):
        for right in eligible_units[i + 1:]:
            if left.session_index == right.session_index:
                continue
            if passes_auto_align_thresholds(
                left,
                right,
                min_waveform_similarity=min_similarity,
                min_amplitude_similarity=min_similarity,
            ):
                candidate_pairs.append(
                    (
                        compute_similarity(left, right),
                        compute_waveform_similarity(left, right),
                        compute_amplitude_similarity(left, right),
                        unit_record_key(left),
                        unit_record_key(right),
                    )
                )

    units_lookup = {unit_record_key(unit): unit for unit in eligible_units}
    strict_rows: list[list[UnitSummary]] = []
    grouped_keys: set[str] = set()
    unit_to_row_index: dict[str, int] = {}

    candidate_pairs.sort(
        key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4])
    )

    def can_add_to_row(unit: UnitSummary, row_units: list[UnitSummary]) -> bool:
        if any(existing.session_index == unit.session_index for existing in row_units):
            return False
        return all(
            passes_auto_align_thresholds(
                unit,
                existing,
                min_waveform_similarity=min_similarity,
                min_amplitude_similarity=min_similarity,
            )
            for existing in row_units
        )

    for _score, _waveform_score, _amplitude_score, left_key, right_key in candidate_pairs:
        left = units_lookup.get(left_key)
        right = units_lookup.get(right_key)
        if left is None or right is None:
            continue

        left_row_index = unit_to_row_index.get(left_key)
        right_row_index = unit_to_row_index.get(right_key)

        if left_row_index is None and right_row_index is None:
            row_index = len(strict_rows)
            strict_rows.append([left, right])
            unit_to_row_index[left_key] = row_index
            unit_to_row_index[right_key] = row_index
            continue

        if left_row_index is not None and right_row_index is None:
            row_units = strict_rows[left_row_index]
            if can_add_to_row(right, row_units):
                row_units.append(right)
                unit_to_row_index[right_key] = left_row_index
            continue

        if left_row_index is None and right_row_index is not None:
            row_units = strict_rows[right_row_index]
            if can_add_to_row(left, row_units):
                row_units.append(left)
                unit_to_row_index[left_key] = right_row_index
            continue

        if left_row_index == right_row_index:
            continue

        left_row_units = strict_rows[left_row_index]
        right_row_units = strict_rows[right_row_index]
        same_session_overlap = {
            unit.session_index for unit in left_row_units
        } & {
            unit.session_index for unit in right_row_units
        }
        if same_session_overlap:
            continue
        if not all(
            passes_auto_align_thresholds(
                left_unit,
                right_unit,
                min_waveform_similarity=min_similarity,
                min_amplitude_similarity=min_similarity,
            )
            for left_unit in left_row_units
            for right_unit in right_row_units
        ):
            continue

        merged_row = sorted(
            left_row_units + right_row_units,
            key=lambda item: (item.session_index, item.unit_id),
        )
        strict_rows[left_row_index] = merged_row
        strict_rows[right_row_index] = []
        for unit in merged_row:
            unit_to_row_index[unit_record_key(unit)] = left_row_index

    final_rows: list[list[UnitSummary]] = []
    for row_units in strict_rows:
        if len(row_units) < 2:
            continue
        sorted_row = sorted(row_units, key=lambda item: (item.session_index, item.unit_id))
        final_rows.append(sorted_row)
        grouped_keys.update(unit_record_key(unit) for unit in sorted_row)

    return final_rows, grouped_keys


def compute_page_similarity_rows(page: PageSummary) -> list[str]:
    rows: list[str] = []
    all_units = [unit for session in page.sessions for unit in session.units]
    for i, left in enumerate(all_units):
        for right in all_units[i + 1:]:
            if left.session_index == right.session_index:
                continue
            score = compute_similarity(left, right)
            if score >= 0.70:
                rows.append(
                    f"{left.session_name} u{left.unit_id} <-> "
                    f"{right.session_name} u{right.unit_id}: {score:.3f}"
                )
    if not rows:
        rows.append("No strong cross-session waveform matches on this page yet.")
    return sorted(rows, reverse=True)


def compute_page_similarity_candidates(
    page: PageSummary,
    min_score: float = 0.45,
    max_candidates: int = 18,
) -> list[SimilarityCandidate]:
    candidates: list[SimilarityCandidate] = []
    all_units = [unit for session in page.sessions for unit in session.units if not unit.is_noise]
    for i, left in enumerate(all_units):
        for right in all_units[i + 1:]:
            if left.session_index == right.session_index:
                continue
            score = compute_similarity(left, right)
            if score < min_score:
                continue
            candidates.append(
                SimilarityCandidate(
                    left_key=unit_record_key(left),
                    right_key=unit_record_key(right),
                    left_label=f"{left.session_name} u{left.unit_id}",
                    right_label=f"{right.session_name} u{right.unit_id}",
                    score=score,
                )
            )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:max_candidates]


def summarize_page_unit_counts(page: PageSummary) -> dict:
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
    total_units = sum(item["unit_count"] for item in session_counts)
    total_discarded_units = sum(item["discarded_unit_count"] for item in session_counts)
    return {
        "session_counts": session_counts,
        "total_units": total_units,
        "total_discarded_units": total_discarded_units,
    }


def build_page_display_rows(page: PageSummary, min_similarity: float = 0.75) -> list[dict[int, list[UnitSummary]]]:
    all_units = [unit for session in page.sessions for unit in session.units]

    def row_sort_key(row: dict[int, list[UnitSummary]]):
        first_session = min(row.keys()) if row else 10**9
        first_unit = (
            min(unit.unit_id for units in row.values() for unit in units)
            if row
            else 10**9
        )
        return (first_session, first_unit)

    def build_row_from_units(units: list[UnitSummary]) -> dict[int, list[UnitSummary]]:
        row: dict[int, list[UnitSummary]] = {}
        for unit in sorted(units, key=lambda item: (item.session_index, item.unit_id)):
            row.setdefault(unit.session_index, []).append(unit)
        return row

    def build_auto_align_rows(units: list[UnitSummary]) -> tuple[list[dict[int, list[UnitSummary]]], set[str]]:
        eligible_units = [
            unit
            for unit in units
            if not unit.align_group.strip() and not unit.exclude_from_auto_align
        ]
        strict_rows, grouped_keys = build_strict_auto_align_rows(
            eligible_units,
            min_similarity=min_similarity,
        )
        return [build_row_from_units(row_units) for row_units in strict_rows], grouped_keys

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
        build_row_from_units(units)
        for _group_name, units in sorted(
            manual_align_rows.items(),
            key=lambda item: (
                min(unit.session_index for unit in item[1]),
                min(unit.unit_id for unit in item[1]),
            ),
        )
    )

    auto_align_rows, auto_aligned_keys = build_auto_align_rows(kept_units)
    rows.extend(sorted(auto_align_rows, key=row_sort_key))

    remaining_kept = [
        unit
        for unit in kept_units
        if not unit.align_group.strip() and unit_record_key(unit) not in auto_aligned_keys
    ]
    for unit in sorted(remaining_kept, key=lambda item: (item.session_index, item.unit_id)):
        rows.append(build_row_from_units([unit]))

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
        build_row_from_units(units)
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
        noise_rows.append(build_row_from_units([unit]))

    rows.sort(key=row_sort_key)
    noise_rows.sort(key=row_sort_key)
    return rows + noise_rows


def build_pair_components(pair_ids: list[str]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {}
    for pair_id in pair_ids:
        left_key, right_key = pair_id.split("|", maxsplit=1)
        adjacency.setdefault(left_key, set()).add(right_key)
        adjacency.setdefault(right_key, set()).add(left_key)

    components: list[set[str]] = []
    visited: set[str] = set()
    for node in adjacency:
        if node in visited:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(adjacency.get(current, set()) - visited)
        if component:
            components.append(component)
    return components


def summarize_decisions(
    sessions: list[SessionSummary],
    auto_align_lookup: dict[str, str] | None = None,
) -> dict:
    totals = {
        "total_units": 0,
        "kept_units": 0,
        "discarded_units": 0,
        "noise_units": 0,
        "merge_groups": 0,
        "merged_units": 0,
        "alignment_groups": 0,
        "aligned_units": 0,
    }
    merge_groups_by_session: dict[tuple[int, int, str], set[int]] = {}
    align_groups: dict[str, set[str]] = {}
    detected_shank_ids: set[int] = set()

    for session in sessions:
        for unit in session.units:
            totals["total_units"] += 1
            detected_shank_ids.add(int(unit.shank_id))

            if unit.is_discarded:
                totals["discarded_units"] += 1
            elif unit.is_noise:
                totals["noise_units"] += 1
            else:
                totals["kept_units"] += 1

            if unit.merge_group and not unit.is_discarded and not unit.is_noise:
                merge_groups_by_session.setdefault(
                    (
                        int(unit.session_index),
                        int(unit.local_channel_on_shank),
                        unit.merge_group,
                    ),
                    set(),
                ).add(int(unit.unit_id))

            if not unit.is_discarded and not unit.is_noise:
                scoped_align_key = ""
                if unit.align_group:
                    scoped_align_key = (
                        f"sh{unit.shank_id}_sg{unit.sg_channel}::"
                        f"{sanitize_token(unit.align_group)}"
                    )
                elif auto_align_lookup is not None:
                    scoped_align_key = auto_align_lookup.get(unit_record_key(unit), "")
                if scoped_align_key:
                    align_groups.setdefault(scoped_align_key, set()).add(unit_record_key(unit))

    for (_session_index, _channel_index, _merge_name), unit_ids in merge_groups_by_session.items():
        if len(unit_ids) < 2:
            continue
        totals["merge_groups"] += 1
        totals["merged_units"] += len(unit_ids)

    for _align_group, members in align_groups.items():
        if len(members) < 2:
            continue
        totals["alignment_groups"] += 1
        totals["aligned_units"] += len(members)

    return {
        "totals": totals,
        "shank_id": next(iter(sorted(detected_shank_ids)), None),
    }


class AlignmentApp:
    def __init__(self, root: tk.Tk, output_root: Path, progress_callback=None):
        self.root = root
        self.output_root = output_root
        self.sessions, self.pages, self.cache_folder = load_all_sessions(
            output_root,
            progress_callback=progress_callback,
        )
        self.page_ids = [
            page.page_id
            for page in sorted(
                self.pages.values(),
                key=lambda page: (page.shank_id, page.sg_channel),
            )
        ] + [DISCARDED_PAGE_ID, REVIEW_PAGE_ID]
        self.image_cache: dict[str, ImageTk.PhotoImage] = {}
        self.unit_control_vars: dict[str, dict[str, tk.Variable]] = {}
        self._decision_state_version = 0
        self._page_display_rows_cache: dict[tuple[str, float, int], list[dict[int, list[UnitSummary]]]] = {}
        self._auto_align_lookup_cache: dict[tuple[float, int], dict[str, str]] = {}
        self._render_generation = 0
        self._image_load_job: str | None = None
        self._pending_image_loads: list[tuple[int, ttk.Frame, str]] = []

        self.manifest_path = output_root / DEFAULT_EXPORT_FOLDER_NAME / "alignment_manifest.json"
        self.summary_root = output_root / DEFAULT_EXPORT_FOLDER_NAME
        self.summary_root.mkdir(parents=True, exist_ok=True)

        self.root.title("Units Alignment UI")
        self.root.geometry("1600x950")
        self._build_layout()
        self._load_manifest_if_available()
        if self.page_ids:
            self.page_listbox.selection_set(0)
            self._render_selected_page()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        sidebar = ttk.Frame(outer)
        sidebar.pack(side="left", fill="y")
        sidebar.configure(width=220)
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="Pages").pack(anchor="w")
        self.page_listbox = tk.Listbox(sidebar, width=20, height=40, exportselection=False)
        self.page_listbox.pack(fill="y", expand=False)
        self.page_listbox.bind("<<ListboxSelect>>", lambda event: self._render_selected_page())
        for page_id in self.page_ids:
            if page_id == REVIEW_PAGE_ID:
                label = "Final Review"
            elif page_id == DISCARDED_PAGE_ID:
                label = "Discarded Units"
            else:
                label = self.pages[page_id].title
            self.page_listbox.insert("end", label)

        controls = ttk.Frame(sidebar)
        controls.pack(fill="x", pady=(10, 0))
        nav_controls = ttk.Frame(controls)
        nav_controls.pack(anchor="w", pady=2)
        ttk.Button(nav_controls, text="Previous", command=self.go_to_previous_page, width=12).pack(side="left", padx=(0, 4))
        ttk.Button(nav_controls, text="Next", command=self.go_to_next_page, width=12).pack(side="left")
        ttk.Button(controls, text="Save Decisions", command=self.save_manifest, width=24).pack(anchor="w", pady=2)
        ttk.Button(controls, text="Export Summary", command=self.export_summary, width=24).pack(anchor="w", pady=2)
        ttk.Button(controls, text="Reload Page", command=self.reload_current_page, width=24).pack(anchor="w", pady=2)

        info_text = (
            "Workflow\n"
            "- Merge Selected: combine units from the same session\n"
            "- Align Selected: same neuron across sessions\n"
            "- Discarded Units: auto-flagged by amplitude/SNR/ISI thresholds\n"
            "- Mark Selected as Noise: exclude from final summary export"
        )
        ttk.Label(sidebar, text=info_text, justify="left", wraplength=190).pack(anchor="w", pady=(12, 0))

        main = ttk.Frame(outer)
        main.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self.page_title_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.page_title_var, font=("Segoe UI", 14, "bold")).pack(anchor="w")

        self.status_var = tk.StringVar(value="")
        status_label = ttk.Label(
            main,
            textvariable=self.status_var,
            justify="left",
            wraplength=1150,
        )
        status_label.pack(anchor="w", fill="x", pady=(6, 8))

        self.selection_panel = ttk.LabelFrame(main, text="Selection Actions", padding=8)
        self.selection_panel.pack(fill="x", expand=False, pady=(0, 8))
        selection_buttons = ttk.Frame(self.selection_panel)
        selection_buttons.pack(fill="x")
        ttk.Button(selection_buttons, text="Merge Selected", command=self.merge_selected_units).pack(side="left")
        ttk.Button(selection_buttons, text="Unmerge Selected", command=self.unmerge_selected_units).pack(side="left", padx=(8, 0))
        ttk.Button(selection_buttons, text="Align Selected", command=self.align_selected_units).pack(side="left", padx=(8, 0))
        ttk.Button(selection_buttons, text="Unalign Selected", command=self.unalign_selected_units).pack(side="left", padx=(8, 0))
        ttk.Button(selection_buttons, text="Mark Selected as Noise", command=self.mark_selected_as_noise).pack(side="left", padx=(8, 0))
        ttk.Button(selection_buttons, text="Clear Noise on Selected", command=self.clear_noise_on_selected).pack(side="left", padx=(8, 0))
        ttk.Button(selection_buttons, text="Debug Selected QC", command=self.debug_selected_qc).pack(side="left", padx=(8, 0))
        ttk.Button(selection_buttons, text="Clear Selection", command=self.clear_selected_units).pack(side="left", padx=(8, 0))
        self.selection_summary_var = tk.StringVar(value="Select one or more units on the current channel page, then choose an action.")
        ttk.Label(
            self.selection_panel,
            textvariable=self.selection_summary_var,
            justify="left",
            wraplength=1150,
        ).pack(anchor="w", pady=(6, 0))

        self.results_panel = ttk.LabelFrame(main, text="Selected Unit Similarity", padding=8)

        canvas_container = ttk.Frame(main)
        canvas_container.pack(fill="both", expand=True)
        canvas_container.rowconfigure(0, weight=1)
        canvas_container.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_container, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(
            yscrollcommand=y_scrollbar.set,
        )

        self.page_content = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.page_content, anchor="nw")
        self.page_content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        self.page_content.bind("<Enter>", self._bind_mousewheel)
        self.page_content.bind("<Leave>", self._unbind_mousewheel)

    def _on_content_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfig(self.canvas_window, width=self.canvas.winfo_width())

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _bind_mousewheel(self, _event=None) -> None:
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None) -> None:
        self.root.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event) -> None:
        if getattr(event, "delta", 0) == 0:
            return
        steps = int(-event.delta / 120)
        if steps == 0:
            steps = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(steps, "units")

    def _selected_page(self) -> PageSummary | None:
        selection = self.page_listbox.curselection()
        if not selection:
            return None
        page_id = self.page_ids[selection[0]]
        if page_id in {REVIEW_PAGE_ID, DISCARDED_PAGE_ID}:
            return None
        return self.pages[page_id]

    def _selected_page_id(self) -> str | None:
        selection = self.page_listbox.curselection()
        if not selection:
            return None
        page_id = self.page_ids[selection[0]]
        if page_id in {REVIEW_PAGE_ID, DISCARDED_PAGE_ID}:
            return None
        return page_id

    def _selected_page_index(self) -> int | None:
        selection = self.page_listbox.curselection()
        if not selection:
            return None
        return int(selection[0])

    def _select_page_index(self, index: int) -> None:
        if not self.page_ids:
            return
        index = max(0, min(index, len(self.page_ids) - 1))
        self.page_listbox.selection_clear(0, "end")
        self.page_listbox.selection_set(index)
        self.page_listbox.activate(index)
        self.page_listbox.see(index)
        self._render_selected_page()

    def go_to_previous_page(self) -> None:
        current_index = self._selected_page_index()
        if current_index is None:
            self._select_page_index(0)
            return
        self._select_page_index(current_index - 1)

    def go_to_next_page(self) -> None:
        current_index = self._selected_page_index()
        if current_index is None:
            self._select_page_index(0)
            return
        self._select_page_index(current_index + 1)

    def _render_selected_page(self) -> None:
        self._apply_control_state_to_units()
        self._start_new_render_cycle()
        selected_index = self._selected_page_index()
        if selected_index is not None:
            selected_page_id = self.page_ids[selected_index]
            if selected_page_id == REVIEW_PAGE_ID:
                self._render_review_page()
                return
            if selected_page_id == DISCARDED_PAGE_ID:
                self._render_discarded_page()
                return

        page = self._selected_page()
        for child in self.page_content.winfo_children():
            child.destroy()
        if page is None:
            self.page_title_var.set("No page selected")
            self.status_var.set("")
            self._hide_similarity_results()
            return

        self.page_title_var.set(page.title)
        page_summary = summarize_page_unit_counts(page)
        summary_parts = [
            (
                f"{item['session_name']}: {item['unit_count']} unit(s)"
                + (
                    f" ({item['discarded_unit_count']} discarded)"
                    if item["discarded_unit_count"] > 0
                    else ""
                )
            )
            for item in page_summary["session_counts"]
        ]
        self.status_var.set(
            f"Total on this channel: {page_summary['total_units']} unit(s)"
            f" | Discarded on this channel: {page_summary['total_discarded_units']}\n"
            + " | ".join(summary_parts)
        )
        self._update_selection_summary(page)
        self._hide_similarity_results()

        display_rows = self._get_page_display_rows(page)

        grid_wrapper = ttk.Frame(self.page_content)
        grid_wrapper.pack(fill="both", expand=True)

        canvases: list[tk.Canvas] = []
        horizontal_scrollbars: list[ttk.Scrollbar] = []

        def sync_horizontal_scroll(*args):
            for row_canvas in canvases:
                row_canvas.xview(*args)

        def update_horizontal_scrollbars(first, last):
            for scrollbar in horizontal_scrollbars:
                scrollbar.set(first, last)

        def register_scrollable_canvas(parent: ttk.Frame) -> tuple[tk.Canvas, ttk.Frame]:
            canvas = tk.Canvas(parent, highlightthickness=0, height=1)
            canvas.pack(side="left", fill="x", expand=True)
            inner = ttk.Frame(canvas)
            canvas.create_window((0, 0), window=inner, anchor="nw")

            def on_inner_configure(_event=None, target_canvas=canvas):
                target_canvas.configure(scrollregion=target_canvas.bbox("all"))
                target_canvas.configure(height=inner.winfo_reqheight())

            inner.bind("<Configure>", on_inner_configure)
            canvas.configure(xscrollcommand=update_horizontal_scrollbars)
            canvases.append(canvas)
            return canvas, inner

        if not display_rows:
            empty_frame = ttk.Frame(grid_wrapper, padding=8)
            empty_frame.pack(fill="x")
            empty_label = ttk.Label(
                empty_frame,
                text="No units were found on this page across the selected sessions.",
            )
            empty_label.pack(anchor="w")
            self.canvas.yview_moveto(0.0)
            return

        for row_index, row_units in enumerate(display_rows, start=1):
            row_frame = ttk.Frame(grid_wrapper)
            row_frame.pack(fill="x")
            row_all_units = [
                unit
                for session in page.sessions
                for unit in row_units.get(session.session_index, [])
            ]
            present_sessions = [
                session.session_name
                for session in page.sessions
                if row_units.get(session.session_index, [])
            ]
            row_label_frame = ttk.Frame(row_frame, width=220)
            row_label_frame.pack(side="left", padx=(6, 2), pady=(4, 0), anchor="n", fill="y")
            row_label_frame.pack_propagate(False)
            ttk.Label(
                row_label_frame,
                text=f"Row {row_index}",
                anchor="w",
                justify="left",
            ).pack(fill="x")
            row_button_frame = ttk.Frame(row_label_frame)
            row_button_frame.pack(fill="x", pady=(6, 6))
            ttk.Button(
                row_button_frame,
                text="Select Row",
                command=lambda units=row_all_units, current_page=page: self._select_units_for_row(units, current_page),
                width=12,
            ).pack(side="left")
            ttk.Button(
                row_button_frame,
                text="Clear Row",
                command=lambda units=row_all_units, current_page=page: self._clear_row_selection(units, current_page),
                width=10,
            ).pack(side="left", padx=(4, 0))
            ttk.Label(
                row_label_frame,
                text="Shown:\n" + ("\n".join(present_sessions) if present_sessions else "none"),
                anchor="w",
                justify="left",
                wraplength=210,
            ).pack(fill="x")

            row_scroll_area = ttk.Frame(row_frame)
            row_scroll_area.pack(side="left", fill="x", expand=True)
            _row_canvas, row_cells = register_scrollable_canvas(row_scroll_area)

            row_scrollbar = ttk.Scrollbar(row_scroll_area, orient="horizontal", command=sync_horizontal_scroll)
            row_scrollbar.pack(fill="x", padx=0, pady=(0, 4))
            horizontal_scrollbars.append(row_scrollbar)

            visible_sessions = [
                session for session in page.sessions if row_units.get(session.session_index, [])
            ]
            for col_index, session in enumerate(visible_sessions):
                row_cells.columnconfigure(col_index, minsize=SESSION_COLUMN_MIN_WIDTH)
                session_card = ttk.LabelFrame(row_cells, text=session.session_name, padding=6)
                session_card.grid(row=0, column=col_index, sticky="nsew", padx=6, pady=4)
                units = row_units.get(session.session_index, [])
                for unit in units:
                    self._render_unit_card(session_card, unit, defer_image=True)

        self.canvas.yview_moveto(0.0)

    def reload_current_page(self) -> None:
        self._invalidate_render_caches()
        self._render_selected_page()

    def _render_review_page(self) -> None:
        self._start_new_render_cycle()
        for child in self.page_content.winfo_children():
            child.destroy()
        self._hide_similarity_results()

        summary = self._summarize_decisions()
        totals = summary["totals"]
        shank_id = summary["shank_id"]
        self.page_title_var.set("Final Review")
        self.status_var.set(
            "Review current decisions before export."
        )
        self.selection_summary_var.set("Selection actions are available on channel pages.")

        review_frame = ttk.Frame(self.page_content, padding=8)
        review_frame.pack(fill="both", expand=True)

        totals_frame = ttk.LabelFrame(review_frame, text="Global Totals", padding=10)
        totals_frame.pack(fill="x", expand=False, pady=(0, 10))
        totals_text = (
            f"Detected shank: {shank_id if shank_id is not None else 'unknown'}\n"
            f"Total units: {totals['total_units']}\n"
            f"Kept units: {totals['kept_units']}\n"
            f"Discarded units: {totals['discarded_units']}\n"
            f"Noise units: {totals['noise_units']}\n"
            f"Merge groups: {totals['merge_groups']}\n"
            f"Merged units involved: {totals['merged_units']}\n"
            f"Alignment groups: {totals['alignment_groups']}\n"
            f"Aligned units involved: {totals['aligned_units']}"
        )
        ttk.Label(totals_frame, text=totals_text, justify="left").pack(anchor="w")

        kept_groups, discarded_groups, noise_groups = self._build_review_groups()
        self._render_review_group_section(
            parent=review_frame,
            title=f"Kept Final Units ({len(kept_groups)})",
            groups=kept_groups,
            empty_text="No kept units are currently selected.",
        )
        self._render_review_group_section(
            parent=review_frame,
            title=f"Noise Units ({len(noise_groups)})",
            groups=noise_groups,
            empty_text="No units are currently marked as noise.",
        )

        footer = ttk.Frame(review_frame)
        footer.pack(fill="x", pady=(10, 0))
        ttk.Button(footer, text="Save Decisions", command=self.save_manifest).pack(side="left")
        ttk.Button(footer, text="Export Summary", command=self.export_summary).pack(side="left", padx=(8, 0))
        self.canvas.yview_moveto(0.0)

    def _render_discarded_page(self) -> None:
        self._start_new_render_cycle()
        for child in self.page_content.winfo_children():
            child.destroy()
        self._hide_similarity_results()

        summary = self._summarize_decisions()
        totals = summary["totals"]
        self.page_title_var.set("Discarded Units")
        self.status_var.set(
            f"Auto-discarded units across all channel pages: {totals['discarded_units']}"
        )
        self.selection_summary_var.set("Selection actions are available on channel pages.")

        discarded_frame = ttk.Frame(self.page_content, padding=8)
        discarded_frame.pack(fill="both", expand=True)

        _kept_groups, discarded_groups, _noise_groups = self._build_review_groups()
        self._render_review_group_section(
            parent=discarded_frame,
            title=f"Discarded Units ({len(discarded_groups)})",
            groups=discarded_groups,
            empty_text="No units currently meet the discard thresholds.",
        )
        self.canvas.yview_moveto(0.0)

    def _summarize_decisions(self) -> dict:
        return summarize_decisions(
            self.sessions,
            auto_align_lookup=self._build_auto_align_lookup(),
        )

    def _start_new_render_cycle(self) -> None:
        self._render_generation += 1
        self._pending_image_loads.clear()
        if self._image_load_job is not None:
            try:
                self.root.after_cancel(self._image_load_job)
            except Exception:
                pass
            self._image_load_job = None

    def _queue_image_load(self, host_frame: ttk.Frame, image_path: str) -> None:
        self._pending_image_loads.append((self._render_generation, host_frame, image_path))
        if self._image_load_job is None:
            self._image_load_job = self.root.after(1, self._drain_pending_image_loads)

    def _drain_pending_image_loads(self) -> None:
        self._image_load_job = None
        batch_size = 6
        processed = 0
        while self._pending_image_loads and processed < batch_size:
            generation, host_frame, image_path = self._pending_image_loads.pop(0)
            if generation != self._render_generation:
                continue
            try:
                if not host_frame.winfo_exists():
                    continue
            except Exception:
                continue
            try:
                image = self._get_image(image_path)
            except Exception:
                continue
            image_label = ttk.Label(host_frame, image=image)
            image_label.image = image
            image_label.pack(anchor="w", pady=(4, 4))
            processed += 1
        if self._pending_image_loads:
            self._image_load_job = self.root.after(1, self._drain_pending_image_loads)

    def _build_review_groups(
        self,
    ) -> tuple[
        list[tuple[str, list[UnitSummary]]],
        list[tuple[str, list[UnitSummary]]],
        list[tuple[str, list[UnitSummary]]],
    ]:
        auto_align_lookup = self._build_auto_align_lookup()
        kept_groups: dict[str, list[UnitSummary]] = {}
        discarded_groups: dict[str, list[UnitSummary]] = {}
        noise_groups: dict[str, list[UnitSummary]] = {}

        for unit in self._iter_all_units():
            if unit.is_discarded:
                if unit.align_group:
                    group_key = (
                        f"discarded__sh{unit.shank_id}_sg{unit.sg_channel}"
                        f"__align__{sanitize_token(unit.align_group)}"
                    )
                elif unit.merge_group:
                    group_key = (
                        f"discarded__s{unit.session_index:03d}_sh{unit.shank_id}_ch{unit.local_channel_on_shank}"
                        f"__merge__{sanitize_token(unit.merge_group)}"
                    )
                else:
                    group_key = f"discarded__s{unit.session_index:03d}_u{unit.unit_id}"
                discarded_groups.setdefault(group_key, []).append(unit)
            elif unit.is_noise:
                if unit.align_group:
                    group_key = f"noise__sh{unit.shank_id}_sg{unit.sg_channel}__align__{sanitize_token(unit.align_group)}"
                elif unit.merge_group:
                    group_key = (
                        f"noise__s{unit.session_index:03d}_sh{unit.shank_id}_ch{unit.local_channel_on_shank}"
                        f"__merge__{sanitize_token(unit.merge_group)}"
                    )
                else:
                    group_key = f"noise__s{unit.session_index:03d}_u{unit.unit_id}"
                noise_groups.setdefault(group_key, []).append(unit)
            else:
                group_key = self._final_group_key_for_unit(unit, auto_align_lookup=auto_align_lookup)
                kept_groups.setdefault(group_key, []).append(unit)

        def sort_group_items(groups: dict[str, list[UnitSummary]]) -> list[tuple[str, list[UnitSummary]]]:
            return sorted(
                groups.items(),
                key=lambda item: (
                    min(unit.session_index for unit in item[1]),
                    min(unit.sg_channel for unit in item[1]),
                    min(unit.unit_id for unit in item[1]),
                ),
            )

        return (
            sort_group_items(kept_groups),
            sort_group_items(discarded_groups),
            sort_group_items(noise_groups),
        )

    def _render_review_group_section(
        self,
        *,
        parent: ttk.Frame,
        title: str,
        groups: list[tuple[str, list[UnitSummary]]],
        empty_text: str,
    ) -> None:
        section = ttk.LabelFrame(parent, text=title, padding=8)
        section.pack(fill="x", expand=False, pady=(0, 10))
        if not groups:
            ttk.Label(section, text=empty_text, justify="left").pack(anchor="w")
            return

        for group_index, (_group_key, units) in enumerate(groups, start=1):
            sorted_units = sorted(units, key=lambda unit: (unit.session_index, unit.unit_id))
            row = ttk.Frame(section)
            row.pack(fill="x", pady=(0, 8))

            shown_sessions = []
            seen_sessions: set[str] = set()
            for unit in sorted_units:
                if unit.session_name not in seen_sessions:
                    shown_sessions.append(unit.session_name)
                    seen_sessions.add(unit.session_name)

            label = ttk.Frame(row, width=220)
            label.pack(side="left", padx=(0, 8), anchor="n", fill="y")
            label.pack_propagate(False)
            ttk.Label(
                label,
                text=f"Group {group_index}",
                justify="left",
                anchor="w",
            ).pack(fill="x")
            ttk.Label(
                label,
                text=(
                    f"Shank {sorted_units[0].shank_id} | SG {sorted_units[0].sg_channel} | ch {sorted_units[0].local_channel_on_shank}\n"
                    f"Shown:\n" + "\n".join(shown_sessions)
                ),
                justify="left",
                anchor="w",
                wraplength=210,
            ).pack(fill="x")

            cells = ttk.Frame(row)
            cells.pack(side="left", fill="x", expand=True)
            units_by_session: dict[tuple[int, str], list[UnitSummary]] = {}
            for unit in sorted_units:
                units_by_session.setdefault((unit.session_index, unit.session_name), []).append(unit)

            for col_index, (_session_index, session_name) in enumerate(sorted(units_by_session.keys())):
                cells.columnconfigure(col_index, minsize=SESSION_COLUMN_MIN_WIDTH)
                session_card = ttk.LabelFrame(cells, text=session_name, padding=6)
                session_card.grid(row=0, column=col_index, sticky="nsew", padx=6, pady=4)
                for unit in units_by_session[(_session_index, session_name)]:
                    self._render_unit_card(session_card, unit, defer_image=False)

    def _get_image(self, image_path: str):
        if image_path not in self.image_cache:
            if Image is not None and ImageTk is not None:
                pil_image = Image.open(image_path)
                pil_image.thumbnail(CARD_IMAGE_SIZE, PIL_RESAMPLE_LANCZOS)
                self.image_cache[image_path] = ImageTk.PhotoImage(pil_image)
            else:
                self.image_cache[image_path] = tk.PhotoImage(file=image_path)
        return self.image_cache[image_path]

    def _similarity_color(self, score: float) -> str:
        if score >= 0.85:
            return "#0b7a28"
        if score >= 0.70:
            return "#b26a00"
        return "#b22222"

    def _units_by_key(self) -> dict[str, UnitSummary]:
        return {unit_record_key(unit): unit for unit in self._iter_all_units()}

    def _selected_units_for_current_page(self) -> tuple[PageSummary | None, list[UnitSummary]]:
        page = self._selected_page()
        if page is None:
            return None, []
        units_lookup = self._units_by_key()
        selected_units: list[UnitSummary] = []
        for session in page.sessions:
            for unit in session.units:
                vars_for_unit = self._ensure_unit_vars(unit)
                if bool(vars_for_unit["selected"].get()):
                    selected_units.append(units_lookup[unit_record_key(unit)])
        return page, selected_units

    def _update_selection_summary(self, page: PageSummary | None = None) -> None:
        current_page = page or self._selected_page()
        if current_page is None:
            self.selection_summary_var.set("Selection actions are available on channel pages.")
            return
        _, selected_units = self._selected_units_for_current_page()
        if not selected_units:
            self.selection_summary_var.set(
                "Select one or more units on the current channel page, then choose an action."
            )
            return
        labels = ", ".join(f"{unit.session_name} u{unit.unit_id}" for unit in selected_units[:6])
        if len(selected_units) > 6:
            labels += ", ..."
        self.selection_summary_var.set(
            f"Selected {len(selected_units)} unit(s): {labels}"
        )

    def _show_similarity_results(self, text: str) -> None:
        for child in self.results_panel.winfo_children():
            child.destroy()
        ttk.Label(
            self.results_panel,
            text=text,
            justify="left",
            wraplength=1150,
        ).pack(anchor="w")
        if not self.results_panel.winfo_manager():
            self.results_panel.pack(fill="x", expand=False, pady=(0, 8), before=self.canvas.master)

    def _hide_similarity_results(self) -> None:
        for child in self.results_panel.winfo_children():
            child.destroy()
        if self.results_panel.winfo_manager():
            self.results_panel.pack_forget()

    def _invalidate_render_caches(self) -> None:
        self._page_display_rows_cache.clear()
        self._auto_align_lookup_cache.clear()

    def _mark_decisions_changed(self) -> None:
        self._decision_state_version += 1
        self._invalidate_render_caches()

    def _get_page_display_rows(
        self,
        page: PageSummary,
        min_similarity: float = 0.75,
    ) -> list[dict[int, list[UnitSummary]]]:
        cache_key = (page.page_id, float(min_similarity), self._decision_state_version)
        cached_rows = self._page_display_rows_cache.get(cache_key)
        if cached_rows is not None:
            return cached_rows
        display_rows = build_page_display_rows(page, min_similarity=min_similarity)
        self._page_display_rows_cache[cache_key] = display_rows
        return display_rows

    def show_selected_similarity(self) -> None:
        page, selected_units = self._selected_units_for_current_page()
        if page is None:
            messagebox.showinfo("No page selected", "Open a channel page first.")
            return
        if len(selected_units) < 2:
            messagebox.showinfo("Not enough units", "Select at least two units first.")
            return

        lines = [
            "Pairwise waveform similarity for selected units:",
        ]
        for i, left in enumerate(selected_units):
            for right in selected_units[i + 1:]:
                score = compute_similarity(left, right)
                lines.append(
                    f"- {left.session_name} u{left.unit_id} <-> {right.session_name} u{right.unit_id}: {score:.3f}"
                )
        self._show_similarity_results("\n".join(lines))

    def debug_selected_qc(self) -> None:
        page, selected_units = self._selected_units_for_current_page()
        if page is None:
            messagebox.showinfo("No page selected", "Open a channel page first.")
            return
        if not selected_units:
            messagebox.showinfo("No units selected", "Select at least one unit first.")
            return

        selected_units = sorted(selected_units, key=lambda unit: (unit.session_index, unit.unit_id))
        lines = [
            "Selected unit QC debug",
            "",
            (
                "Discard rule requires ANY of these:\n"
                f"- |amplitude_median| < {DISCARD_ABS_AMPLITUDE_MAX:g}\n"
                f"- snr < {DISCARD_SNR_MAX:g}\n"
                f"- isi_violations_ratio > {DISCARD_ISI_VIOLATION_MIN:g}"
            ),
            "",
        ]
        for unit in selected_units:
            checks = evaluate_discard_criteria(unit)
            raw_isi = safe_float(unit.isi_violations_ratio)
            lines.extend(
                [
                    f"{unit.session_name} | unit {unit.unit_id}",
                    f"raw amplitude_median = {format_metric(unit.amplitude_median)}",
                    f"raw snr = {format_metric(unit.snr)}",
                    f"raw isi_violations_ratio = {format_metric(raw_isi)}",
                    f"|amp| check: {'PASS' if checks['amplitude_pass'] else 'FAIL'}",
                    f"snr check: {'PASS' if checks['snr_pass'] else 'FAIL'}",
                    f"isi check: {'PASS' if checks['isi_pass'] else 'FAIL'}",
                    f"final is_discarded = {'YES' if checks['all_pass'] else 'NO'}",
                    "",
                ]
            )
        messagebox.showinfo("Selected QC Debug", "\n".join(lines).strip())

    def clear_selected_units(self) -> None:
        page, selected_units = self._selected_units_for_current_page()
        if page is None:
            return
        self._clear_selection_for_units(selected_units)
        self._update_selection_summary(page)
        self._hide_similarity_results()
        self._render_selected_page()

    def _clear_selection_for_units(self, units: list[UnitSummary]) -> None:
        for unit in units:
            self._ensure_unit_vars(unit)["selected"].set(False)

    def _set_selection_for_units(self, units: list[UnitSummary], selected: bool) -> None:
        for unit in units:
            self._ensure_unit_vars(unit)["selected"].set(selected)

    def _select_units_for_row(self, units: list[UnitSummary], page: PageSummary) -> None:
        self._set_selection_for_units(units, True)
        self._update_selection_summary(page)
        self._hide_similarity_results()

    def _clear_row_selection(self, units: list[UnitSummary], page: PageSummary) -> None:
        self._set_selection_for_units(units, False)
        self._update_selection_summary(page)
        self._hide_similarity_results()

    def mark_selected_as_noise(self) -> None:
        page, selected_units = self._selected_units_for_current_page()
        if page is None:
            messagebox.showinfo("No page selected", "Open a channel page first.")
            return
        if not selected_units:
            messagebox.showinfo("No units selected", "Select at least one unit first.")
            return
        for unit in selected_units:
            vars_for_unit = self._ensure_unit_vars(unit)
            vars_for_unit["is_noise"].set(True)
            unit.is_noise = True
        self._mark_decisions_changed()
        self._clear_selection_for_units(selected_units)
        self._update_selection_summary(page)
        self._hide_similarity_results()
        self._render_selected_page()
        messagebox.showinfo("Noise updated", f"Marked {len(selected_units)} unit(s) as noise.")

    def clear_noise_on_selected(self) -> None:
        page, selected_units = self._selected_units_for_current_page()
        if page is None:
            messagebox.showinfo("No page selected", "Open a channel page first.")
            return
        if not selected_units:
            messagebox.showinfo("No units selected", "Select at least one unit first.")
            return
        for unit in selected_units:
            vars_for_unit = self._ensure_unit_vars(unit)
            vars_for_unit["is_noise"].set(False)
            unit.is_noise = False
        self._mark_decisions_changed()
        self._clear_selection_for_units(selected_units)
        self._update_selection_summary(page)
        self._hide_similarity_results()
        self._render_selected_page()
        messagebox.showinfo("Noise updated", f"Cleared noise flag on {len(selected_units)} unit(s).")

    def _build_existing_group_members(
        self,
        attr_name: str,
        scope_fn,
    ) -> dict[str, set[str]]:
        groups: dict[str, set[str]] = {}
        for unit in self._iter_all_units():
            group_name = getattr(unit, attr_name, "").strip()
            if not group_name:
                continue
            scoped_key = f"{scope_fn(unit)}::{group_name}"
            groups.setdefault(scoped_key, set()).add(unit_record_key(unit))
        return groups

    def _assign_selected_units_to_group(
        self,
        *,
        attr_name: str,
        base_name: str,
        scope_tag: str,
        selected_units: list[UnitSummary],
        validation_message: str,
    ) -> tuple[str, int] | None:
        units_lookup = self._units_by_key()
        existing_members = self._build_existing_group_members(
            attr_name,
            scope_fn=lambda unit: (
                f"sh{unit.shank_id}_sg{unit.sg_channel}"
                if attr_name == "align_group"
                else f"s{unit.session_index}_sh{unit.shank_id}_sg{unit.sg_channel}"
            ),
        )
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
        existing_names_in_selection = sorted(
            {
                getattr(unit, attr_name).strip()
                for unit in selected_units
                if getattr(unit, attr_name).strip()
            }
        )
        expanded_keys = set(selected_keys)
        for name in existing_names_in_selection:
            expanded_keys.update(existing_members.get(f"{scope_prefix}{name}", set()))

        if len(expanded_keys) < 2:
            messagebox.showinfo("Not enough units", validation_message)
            return None

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
            vars_for_unit = self._ensure_unit_vars(unit)
            vars_for_unit[attr_name].set(group_name)
            setattr(unit, attr_name, group_name)
            if attr_name == "align_group":
                unit.exclude_from_auto_align = False

        return group_name, len(expanded_keys)

    def merge_selected_units(self) -> None:
        page, selected_units = self._selected_units_for_current_page()
        if page is None:
            messagebox.showinfo("No page selected", "Open a channel page first.")
            return
        if len(selected_units) < 2:
            messagebox.showinfo("Not enough units", "Select at least two units to merge.")
            return
        session_indices = {unit.session_index for unit in selected_units}
        if len(session_indices) != 1:
            messagebox.showinfo(
                "Merge requires one session",
                "Merge is only for units from the same session. Select units from one session column.",
            )
            return

        result = self._assign_selected_units_to_group(
            attr_name="merge_group",
            base_name=f"merge_s{selected_units[0].session_index:03d}_sh{page.shank_id}_sg{page.sg_channel}",
            scope_tag=f"s{selected_units[0].session_index}_sh{page.shank_id}_sg{page.sg_channel}",
            selected_units=selected_units,
            validation_message="Select at least two units from the same session to create a merge group.",
        )
        if result is None:
            return
        group_name, unit_count = result
        self._mark_decisions_changed()
        self._clear_selection_for_units(selected_units)
        self._update_selection_summary(page)
        self._hide_similarity_results()
        self._render_selected_page()
        messagebox.showinfo("Merge assigned", f"Assigned merge group {group_name} to {unit_count} unit(s).")

    def align_selected_units(self) -> None:
        page, selected_units = self._selected_units_for_current_page()
        if page is None:
            messagebox.showinfo("No page selected", "Open a channel page first.")
            return
        if len(selected_units) < 2:
            messagebox.showinfo("Not enough units", "Select at least two units to align.")
            return
        session_indices = {unit.session_index for unit in selected_units}
        if len(session_indices) < 2:
            messagebox.showinfo(
                "Align requires multiple sessions",
                "Align is for matching the same neuron across sessions. Select units from at least two session columns.",
            )
            return

        result = self._assign_selected_units_to_group(
            attr_name="align_group",
            base_name=f"align_sh{page.shank_id}_sg{page.sg_channel}",
            scope_tag=f"sh{page.shank_id}_sg{page.sg_channel}",
            selected_units=selected_units,
            validation_message="Select at least two units to create an alignment group.",
        )
        if result is None:
            return
        group_name, unit_count = result
        self._mark_decisions_changed()
        self._clear_selection_for_units(selected_units)
        self._update_selection_summary(page)
        self._hide_similarity_results()
        self._render_selected_page()
        messagebox.showinfo("Alignment assigned", f"Assigned alignment group {group_name} to {unit_count} unit(s).")

    def unmerge_selected_units(self) -> None:
        page, selected_units = self._selected_units_for_current_page()
        if page is None:
            messagebox.showinfo("No page selected", "Open a channel page first.")
            return
        if not selected_units:
            messagebox.showinfo("No units selected", "Select at least one unit first.")
            return
        cleared_count = 0
        for unit in selected_units:
            vars_for_unit = self._ensure_unit_vars(unit)
            if vars_for_unit["merge_group"].get().strip():
                cleared_count += 1
            vars_for_unit["merge_group"].set("")
            unit.merge_group = ""
        self._mark_decisions_changed()
        self._clear_selection_for_units(selected_units)
        self._update_selection_summary(page)
        self._hide_similarity_results()
        self._render_selected_page()
        messagebox.showinfo("Merge cleared", f"Cleared merge group on {cleared_count} unit(s).")

    def unalign_selected_units(self) -> None:
        page, selected_units = self._selected_units_for_current_page()
        if page is None:
            messagebox.showinfo("No page selected", "Open a channel page first.")
            return
        if not selected_units:
            messagebox.showinfo("No units selected", "Select at least one unit first.")
            return
        cleared_count = 0
        for unit in selected_units:
            vars_for_unit = self._ensure_unit_vars(unit)
            if vars_for_unit["align_group"].get().strip():
                cleared_count += 1
            vars_for_unit["align_group"].set("")
            unit.align_group = ""
            unit.exclude_from_auto_align = True
        self._mark_decisions_changed()
        self._clear_selection_for_units(selected_units)
        self._update_selection_summary(page)
        self._hide_similarity_results()
        self._render_selected_page()
        messagebox.showinfo("Alignment cleared", f"Cleared alignment group on {cleared_count} unit(s).")

    def _ensure_unit_vars(self, unit: UnitSummary) -> dict[str, tk.Variable]:
        key = unit_record_key(unit)
        if key not in self.unit_control_vars:
            self.unit_control_vars[key] = {
                "merge_group": tk.StringVar(value=unit.merge_group),
                "align_group": tk.StringVar(value=unit.align_group),
                "is_noise": tk.BooleanVar(value=unit.is_noise),
                "selected": tk.BooleanVar(value=False),
            }
        return self.unit_control_vars[key]

    def _render_unit_card(self, parent: ttk.Frame, unit: UnitSummary, *, defer_image: bool) -> None:
        vars_for_unit = self._ensure_unit_vars(unit)
        card = ttk.Frame(parent, relief="solid", padding=6)
        card.pack(fill="x", expand=True, pady=6)

        select_row = ttk.Frame(card)
        select_row.pack(fill="x")
        ttk.Checkbutton(
            select_row,
            text="Select",
            variable=vars_for_unit["selected"],
            command=self._update_selection_summary,
        ).pack(side="left")
        if unit.merge_group or unit.align_group or unit.is_discarded or unit.is_noise:
            tags = []
            if unit.merge_group:
                tags.append(f"merge={unit.merge_group}")
            if unit.align_group:
                tags.append(f"align={unit.align_group}")
            if unit.is_discarded:
                tags.append(f"discarded ({build_discard_reason(unit)})")
            if unit.is_noise and not unit.is_discarded:
                tags.append("noise")
            ttk.Label(select_row, text=" | ".join(tags), justify="right").pack(side="right")

        header = ttk.Label(
            card,
            text=(
                f"Unit {unit.unit_id} | sg {unit.sg_channel}\n"
                f"FR {format_metric(unit.firing_rate)} Hz | "
                f"SNR {format_metric(unit.snr)}"
            ),
            justify="left",
        )
        header.pack(anchor="w")

        image_host = ttk.Frame(card)
        image_host.pack(anchor="w", fill="x")
        if defer_image:
            self._queue_image_load(image_host, unit.waveform_image_path)
        else:
            image = self._get_image(unit.waveform_image_path)
            image_label = ttk.Label(image_host, image=image)
            image_label.image = image
            image_label.pack(anchor="w", pady=(4, 4))

        metrics_text = (
            f"Amplitude median: {format_metric(unit.amplitude_median)}\n"
            f"ISI violation ratio: {format_metric(unit.isi_violations_ratio)}\n"
            f"Num spikes: {format_metric(unit.num_spikes)}\n"
            f"{build_discard_check_text(unit)}"
        )
        ttk.Label(card, text=metrics_text, justify="left").pack(anchor="w")

    def _iter_all_units(self) -> list[UnitSummary]:
        units: list[UnitSummary] = []
        for session in self.sessions:
            units.extend(session.units)
        return units

    def _is_auto_merge_from_align(self, merge_group: str) -> bool:
        return bool(merge_group) and merge_group.startswith("__alignmerge__")

    def _sync_merge_groups_from_align_groups(self) -> None:
        for unit in self._iter_all_units():
            if self._is_auto_merge_from_align(unit.merge_group):
                unit.merge_group = ""

        grouped_units: dict[tuple[int, str], list[UnitSummary]] = {}
        for unit in self._iter_all_units():
            if unit.is_discarded or unit.is_noise:
                continue
            align_name = unit.align_group.strip()
            if not align_name:
                continue
            grouped_units.setdefault(
                (int(unit.session_index), align_name),
                [],
            ).append(unit)

        for (_session_index, align_name), units in grouped_units.items():
            if len(units) < 2:
                continue
            auto_merge_name = f"__alignmerge__{sanitize_token(align_name)}"
            for unit in units:
                if not unit.merge_group or self._is_auto_merge_from_align(unit.merge_group):
                    unit.merge_group = auto_merge_name

        for unit in self._iter_all_units():
            vars_for_unit = self._ensure_unit_vars(unit)
            vars_for_unit["merge_group"].set(unit.merge_group)

    def _apply_control_state_to_units(self) -> None:
        for unit in self._iter_all_units():
            vars_for_unit = self._ensure_unit_vars(unit)
            unit.merge_group = vars_for_unit["merge_group"].get().strip()
            unit.align_group = vars_for_unit["align_group"].get().strip()
            unit.is_noise = bool(vars_for_unit["is_noise"].get())
            unit.is_discarded = is_unit_auto_discarded(unit)
        self._sync_merge_groups_from_align_groups()

    def save_manifest(self, show_message: bool = True) -> None:
        self._apply_control_state_to_units()
        payload = {
            "output_root": str(self.output_root),
            "sessions": [],
        }
        for session in self.sessions:
            payload["sessions"].append(
                {
                    "session_name": session.session_name,
                    "session_index": session.session_index,
                    "output_folder": session.output_folder,
                    "analyzer_folder": session.analyzer_folder,
                    "units": [asdict(unit) for unit in session.units],
                }
            )
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if show_message:
            messagebox.showinfo("Saved", f"Saved decisions to:\n{self.manifest_path}")

    def _load_manifest_if_available(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return

        manifest_units: dict[str, dict] = {}
        for session_payload in payload.get("sessions", []):
            for unit_payload in session_payload.get("units", []):
                key = f"{unit_payload.get('session_index')}:{unit_payload.get('unit_id')}"
                manifest_units[key] = unit_payload

        for unit in self._iter_all_units():
            key = unit_record_key(unit)
            if key not in manifest_units:
                continue
            saved = manifest_units[key]
            unit.merge_group = str(saved.get("merge_group", "") or "")
            unit.align_group = str(saved.get("align_group", "") or "")
            unit.exclude_from_auto_align = bool(saved.get("exclude_from_auto_align", False))
            unit.is_noise = bool(saved.get("is_noise", False))
            unit.is_discarded = is_unit_auto_discarded(unit)
            vars_for_unit = self._ensure_unit_vars(unit)
            vars_for_unit["merge_group"].set(unit.merge_group)
            vars_for_unit["align_group"].set(unit.align_group)
            vars_for_unit["is_noise"].set(unit.is_noise)
        self._sync_merge_groups_from_align_groups()
        self._mark_decisions_changed()

    def export_summary(self) -> None:
        self.status_var.set(
            f"Exporting summary...\nSaving to: {self.summary_root}"
        )
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            self.save_manifest(show_message=False)
            export_folder = self.summary_root / "exported_units"
            export_folder.mkdir(parents=True, exist_ok=True)

            self._apply_control_state_to_units()
            auto_align_lookup = self._build_auto_align_lookup()
            final_groups: dict[str, list[UnitSummary]] = {}
            discarded_groups: dict[str, list[UnitSummary]] = {}

            for unit in self._iter_all_units():
                if unit.is_discarded:
                    discard_key = self._discard_group_key_for_unit(unit)
                    discarded_groups.setdefault(discard_key, []).append(unit)
                    continue
                if unit.is_noise:
                    continue

                final_key = self._final_group_key_for_unit(unit, auto_align_lookup=auto_align_lookup)
                final_groups.setdefault(final_key, []).append(unit)

            manifest_rows = []
            unique_unit_rows = []
            discarded_unit_rows = []
            total_groups = len(final_groups)
            for group_index, (group_key, units) in enumerate(sorted(final_groups.items()), start=1):
                self.status_var.set(
                    "Exporting summary...\n"
                    f"Saving to: {self.summary_root}\n"
                    f"Writing final unit {group_index}/{total_groups}"
                )
                self.root.update_idletasks()
                group_folder = export_folder / f"unit_{group_index:04d}"
                group_folder.mkdir(parents=True, exist_ok=True)

                representative = units[0]
                copied_images = []
                for item_index, unit in enumerate(units, start=1):
                    src = Path(unit.waveform_image_path)
                    dst = group_folder / f"waveform_{item_index:02d}_{unit.session_name.replace(' ', '_')}_u{unit.unit_id}.png"
                    if src.exists():
                        dst.write_bytes(src.read_bytes())
                        copied_images.append(str(dst))

                summary_text = self._build_group_summary_text(group_index, group_key, units)
                summary_path = group_folder / "summary.txt"
                summary_path.write_text(summary_text, encoding="utf-8")
                unique_unit_rows.append(
                    self._build_unique_unit_summary_row(
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
                discarded_unit_rows.append(
                    self._build_discarded_unit_summary_row(
                        group_key=group_key,
                        units=units,
                    )
                )

            unique_units_json_path = self.summary_root / "unique_units_summary.json"
            unique_units_json_path.write_text(
                json.dumps(unique_unit_rows, indent=2),
                encoding="utf-8",
            )
            unique_units_csv_path = self.summary_root / "unique_units_summary.csv"
            self._write_unique_units_summary_csv(unique_units_csv_path, unique_unit_rows)
            discarded_units_json_path = self.summary_root / "discarded_units_summary.json"
            discarded_units_json_path.write_text(
                json.dumps(discarded_unit_rows, indent=2),
                encoding="utf-8",
            )
            discarded_units_csv_path = self.summary_root / "discarded_units_summary.csv"
            self._write_discarded_units_summary_csv(discarded_units_csv_path, discarded_unit_rows)

            export_manifest_path = self.summary_root / "export_summary.json"
            export_manifest_path.write_text(
                json.dumps(
                    {
                        "output_root": str(self.output_root),
                        "unique_units_summary_json": str(unique_units_json_path),
                        "unique_units_summary_csv": str(unique_units_csv_path),
                        "discarded_units_summary_json": str(discarded_units_json_path),
                        "discarded_units_summary_csv": str(discarded_units_csv_path),
                        "cross_session_alignment_groups": manifest_rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.status_var.set(
                f"Export complete. Saved to: {self.summary_root}"
            )
            messagebox.showinfo(
                "Export complete",
                f"Saved summary folder to:\n{self.summary_root}\n\n"
                f"Files:\n"
                f"- {export_manifest_path.name}\n"
                f"- {unique_units_json_path.name}\n"
                f"- {unique_units_csv_path.name}\n"
                f"- {discarded_units_json_path.name}\n"
                f"- {discarded_units_csv_path.name}\n"
                f"- exported_units\\\n\n"
                f"Unique final units: {len(unique_unit_rows)}\n"
                f"Discarded groups: {len(discarded_unit_rows)}\n"
                f"Cross-session groups kept: {len(manifest_rows)}",
            )
        except Exception as exc:
            self.status_var.set(f"Export failed: {exc}")
            messagebox.showerror(
                "Export failed",
                "Export did not complete.\n\n"
                f"Reason:\n{exc}\n\n"
                f"Traceback:\n{traceback.format_exc()}",
            )
        finally:
            self.root.configure(cursor="")
            self.root.update_idletasks()

    def _build_unique_unit_summary_row(
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
                {
                    "session_name": session_name,
                    "unit_ids": sorted(member_unit_ids),
                }
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

    def _build_auto_align_lookup(self, min_similarity: float = 0.75) -> dict[str, str]:
        cache_key = (float(min_similarity), self._decision_state_version)
        cached_lookup = self._auto_align_lookup_cache.get(cache_key)
        if cached_lookup is not None:
            return cached_lookup

        auto_align_lookup: dict[str, str] = {}
        for page in self.pages.values():
            if page.page_id == REVIEW_PAGE_ID:
                continue
            all_units = [unit for session in page.sessions for unit in session.units]
            eligible_units = [
                unit
                for unit in all_units
                if not unit.is_discarded
                and not unit.is_noise
                and not unit.align_group.strip()
                and not unit.exclude_from_auto_align
            ]
            strict_rows, _grouped_keys = build_strict_auto_align_rows(
                eligible_units,
                min_similarity=min_similarity,
            )
            for component_units in strict_rows:
                group_name = (
                    f"sh{page.shank_id}_sg{page.sg_channel}__auto__"
                    f"s{component_units[0].session_index:03d}_u{component_units[0].unit_id:04d}"
                )
                for unit in component_units:
                    auto_align_lookup[unit_record_key(unit)] = group_name
        self._auto_align_lookup_cache[cache_key] = auto_align_lookup
        return auto_align_lookup

    def _final_group_key_for_unit(self, unit: UnitSummary, auto_align_lookup: dict[str, str] | None = None) -> str:
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

    def _discard_group_key_for_unit(self, unit: UnitSummary) -> str:
        if unit.align_group:
            return f"discarded__sh{unit.shank_id}_sg{unit.sg_channel}__align__{sanitize_token(unit.align_group)}"
        if unit.merge_group:
            return (
                f"discarded__s{unit.session_index:03d}_sh{unit.shank_id}_ch{unit.local_channel_on_shank}"
                f"__merge__{sanitize_token(unit.merge_group)}"
            )
        return f"discarded__s{unit.session_index:03d}_u{unit.unit_id}"

    def _write_unique_units_summary_csv(self, csv_path: Path, rows: list[dict]) -> None:
        fieldnames = [
            "final_unit_id",
            "final_unit_label",
            "shank_id",
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
                        "member_units": "; ".join(
                            f"{item['session_name']}:u{item['unit_id']}"
                            for item in row["member_units"]
                        ),
                        "representative_session": row["representative_session"],
                        "representative_unit_id": row["representative_unit_id"],
                        "representative_waveform_image": row["representative_waveform_image"],
                        "export_folder": row["export_folder"],
                        "summary_path": row["summary_path"],
                        "final_group_key": row["final_group_key"],
                    }
                )

    def _build_discarded_unit_summary_row(self, *, group_key: str, units: list[UnitSummary]) -> dict:
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

    def _write_discarded_units_summary_csv(self, csv_path: Path, rows: list[dict]) -> None:
        fieldnames = [
            "status",
            "discard_group_key",
            "discard_reason",
            "shank_id",
            "channel",
            "sg_channel",
            "num_sessions",
            "sessions_present",
            "num_member_units",
            "member_units",
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
                        "member_units": "; ".join(
                            f"{item['session_name']}:u{item['unit_id']}"
                            for item in row["member_units"]
                        ),
                    }
                )

    def _build_group_summary_text(self, group_index: int, group_key: str, units: list[UnitSummary]) -> str:
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


def main() -> None:
    output_root = choose_output_root()
    root = tk.Tk()
    root.withdraw()
    loading_window, loading_message_var = create_loading_window(root)

    def update_loading(message: str) -> None:
        loading_message_var.set(message)
        loading_window.update_idletasks()
        root.update()

    try:
        app = AlignmentApp(root, output_root, progress_callback=update_loading)
    except Exception as exc:
        try:
            loading_window.destroy()
        except Exception:
            pass
        root.withdraw()
        messagebox.showerror("Units Alignment UI", str(exc), parent=root)
        root.destroy()
        raise SystemExit(1)
    loading_window.destroy()
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
