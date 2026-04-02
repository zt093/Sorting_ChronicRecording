#NOTE: This Code input should be one shank only!
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.curation as scur
import spikeinterface.full as si
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


MAX_SPIKES_PER_UNIT = 500
MS_BEFORE = 1.0
MS_AFTER = 2.0
CARD_IMAGE_SIZE = (280, 180)
DEFAULT_EXPORT_FOLDER_NAME = "units_alignment_summary"
DESIRED_METRICS = [
    "amplitude_median",
    "firing_rate",
    "isi_violations_ratio",
    "snr",
    "num_spikes",
]


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
    local_channel_on_shank: int
    sessions: list[SessionSummary]

    @property
    def page_id(self) -> str:
        return f"ch{self.local_channel_on_shank}"

    @property
    def title(self) -> str:
        return f"Channel {self.local_channel_on_shank}"


REVIEW_PAGE_ID = "__review__"


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
    digits = re.findall(r"\d+", name)
    if digits:
        hour_number = int(digits[-1])
        suffix = "st" if hour_number % 10 == 1 and hour_number % 100 != 11 else (
            "nd" if hour_number % 10 == 2 and hour_number % 100 != 12 else (
                "rd" if hour_number % 10 == 3 and hour_number % 100 != 13 else "th"
            )
        )
        return f"{hour_number}{suffix} hour"
    return f"Session {index + 1}: {name}"


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


def infer_unit_channel_metadata(analyzer, unit_id) -> dict[str, int]:
    templates_ext = analyzer.get_extension("templates")
    average_templates = templates_ext.get_data(outputs="average")
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
    property_keys = set(analyzer.recording.get_property_keys())
    if "group" in property_keys:
        group_values = analyzer.recording.get_property("group")
        if channel_index < len(group_values):
            match = re.search(r"(\d+)", str(group_values[channel_index]))
            if match:
                shank_id = int(match.group(1))
    elif analyzer.recording.has_probe():
        try:
            probe = analyzer.recording.get_probe()
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
        fig, ax = plt.subplots(figsize=(4.2, 2.5))
        ax.text(0.5, 0.5, "No waveform", ha="center", va="center")
        ax.axis("off")
    else:
        mean_waveform = waveforms.mean(axis=0)
        channel_index = int(np.argmax(np.max(np.abs(mean_waveform), axis=0)))
        single_channel_waveforms = waveforms[:, :, channel_index]
        average_waveform = single_channel_waveforms.mean(axis=0)
        std_waveform = single_channel_waveforms.std(axis=0)
        time_axis = np.linspace(-MS_BEFORE, MS_AFTER, average_waveform.shape[0])

        fig, ax = plt.subplots(figsize=(4.2, 2.5))
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


def load_all_sessions(root_folder: Path) -> tuple[list[SessionSummary], dict[str, PageSummary], Path]:
    analyzer_folders = discover_analyzer_folders(root_folder)
    cache_folder = root_folder / DEFAULT_EXPORT_FOLDER_NAME / "_cache"
    sessions: list[SessionSummary] = []
    discovered_shank_ids: set[int] = set()

    for session_index, analyzer_folder in enumerate(analyzer_folders):
        output_folder = analyzer_folder.parent
        session_name = session_name_from_output_folder(output_folder, session_index)
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
            metadata = infer_unit_channel_metadata(analyzer, unit_id)
            metrics = metrics_lookup.get(unit_id_int, {})
            waveform_vector = get_waveform_vector(analyzer, unit_id)
            image_path = (
                cache_folder
                / "waveforms"
                / f"session_{session_index:03d}"
                / f"shank{metadata['shank_id']}_ch{metadata['local_channel_on_shank']}_unit{unit_id_int}.png"
            )
            if not image_path.exists():
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
                waveform_image_path=str(image_path),
            )
            session_summary.units.append(unit_summary)
            discovered_shank_ids.add(int(metadata["shank_id"]))

        sessions.append(session_summary)

    if len(discovered_shank_ids) > 1:
        raise ValueError(
            "The selected input contains multiple shanks, but this UI now assumes one shank only. "
            f"Found shank ids: {sorted(discovered_shank_ids)}"
        )

    page_summaries: dict[str, PageSummary] = {}
    page_keys: set[int] = set()
    for session in sessions:
        for unit in session.units:
            page_keys.add(unit.local_channel_on_shank)

    shank_id = next(iter(discovered_shank_ids), 0)
    for channel_id in sorted(page_keys):
        aligned_sessions: list[SessionSummary] = []
        for session in sessions:
            filtered_units = [
                unit
                for unit in session.units
                if unit.local_channel_on_shank == channel_id
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
            local_channel_on_shank=channel_id,
            sessions=aligned_sessions,
        )
        page_summaries[page.page_id] = page

    return sessions, page_summaries, cache_folder


def unit_record_key(unit: UnitSummary) -> str:
    return f"{unit.session_index}:{unit.unit_id}"


def compute_similarity(a: UnitSummary, b: UnitSummary) -> float:
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
    score = float(np.dot(va, vb) / (na * nb))
    return max(-1.0, min(1.0, score))


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


def build_session_curation_payload(session: SessionSummary) -> dict:
    labels_by_unit: dict[str, list[str]] = {}
    merge_groups_by_name: dict[tuple[int, str], set[int]] = {}

    for unit in session.units:
        labels_by_unit[str(unit.unit_id)] = ["reject"] if unit.is_noise else ["accept"]
        if unit.is_noise:
            continue
        if unit.merge_group:
            merge_groups_by_name.setdefault(
                (int(unit.local_channel_on_shank), unit.merge_group),
                set(),
            ).add(int(unit.unit_id))

    merge_groups = [
        sorted(unit_ids)
        for _, unit_ids in sorted(merge_groups_by_name.items())
        if len(unit_ids) > 1
    ]

    return {
        "labelsByUnit": labels_by_unit,
        "mergeGroups": merge_groups,
    }


def create_curated_analyzer_from_session(session: SessionSummary, export_root: Path) -> dict:
    analyzer_folder = Path(session.analyzer_folder)
    sorting_analyzer = si.load_sorting_analyzer(
        folder=analyzer_folder,
        format="zarr",
        load_extensions=True,
    )
    ensure_required_extensions(sorting_analyzer)

    curation_payload = build_session_curation_payload(session)
    curated_sorting = scur.apply_sortingview_curation(
        sorting_analyzer.sorting,
        uri_or_json=curation_payload,
        include_labels=["accept"],
        skip_merge=False,
    )

    session_export_root = export_root / f"session_{session.session_index:03d}_{session.safe_name}"
    session_export_root.mkdir(parents=True, exist_ok=True)
    curation_json_path = session_export_root / "curation_applied.json"
    curation_json_path.write_text(json.dumps(curation_payload, indent=2), encoding="utf-8")

    curated_analyzer_folder = session_export_root / "sorting_analyzer_curated.zarr"
    if curated_analyzer_folder.exists():
        shutil.rmtree(curated_analyzer_folder)

    num_units_after = int(curated_sorting.get_num_units())
    if num_units_after > 0:
        curated_analyzer = si.create_sorting_analyzer(
            sorting=curated_sorting,
            recording=sorting_analyzer.recording,
            format="memory",
        )
        ensure_required_extensions(curated_analyzer)
        curated_analyzer.save_as(folder=curated_analyzer_folder, format="zarr")
        curated_analyzer_folder_str = str(curated_analyzer_folder)
    else:
        curated_analyzer_folder_str = None

    return {
        "session_name": session.session_name,
        "session_index": session.session_index,
        "source_analyzer_folder": str(analyzer_folder),
        "source_output_folder": session.output_folder,
        "curation_json_path": str(curation_json_path),
        "curated_analyzer_folder": curated_analyzer_folder_str,
        "num_units_before": len(session.units),
        "num_units_after": num_units_after,
        "merge_groups": curation_payload["mergeGroups"],
        "rejected_unit_ids": sorted(
            int(unit.unit_id)
            for unit in session.units
            if unit.is_noise
        ),
    }


def summarize_decisions(sessions: list[SessionSummary]) -> dict:
    totals = {
        "total_units": 0,
        "kept_units": 0,
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

            if unit.is_noise:
                totals["noise_units"] += 1
            else:
                totals["kept_units"] += 1

            if unit.merge_group and not unit.is_noise:
                merge_groups_by_session.setdefault(
                    (
                        int(unit.session_index),
                        int(unit.local_channel_on_shank),
                        unit.merge_group,
                    ),
                    set(),
                ).add(int(unit.unit_id))

            if unit.align_group and not unit.is_noise:
                scoped_align_key = (
                    f"sh{unit.shank_id}_ch{unit.local_channel_on_shank}::"
                    f"{sanitize_token(unit.align_group)}"
                )
                align_groups.setdefault(scoped_align_key, set()).add(unit_record_key(unit))

    for (session_index, _merge_name), unit_ids in merge_groups_by_session.items():
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
    def __init__(self, root: tk.Tk, output_root: Path):
        self.root = root
        self.output_root = output_root
        self.sessions, self.pages, self.cache_folder = load_all_sessions(output_root)
        self.page_ids = sorted(self.pages.keys()) + [REVIEW_PAGE_ID]
        self.image_cache: dict[str, ImageTk.PhotoImage] = {}
        self.unit_control_vars: dict[str, dict[str, tk.Variable]] = {}
        self.page_pair_vars: dict[str, dict[str, tk.BooleanVar]] = {}

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

        ttk.Label(sidebar, text="Pages").pack(anchor="w")
        self.page_listbox = tk.Listbox(sidebar, width=28, height=40, exportselection=False)
        self.page_listbox.pack(fill="y", expand=False)
        self.page_listbox.bind("<<ListboxSelect>>", lambda event: self._render_selected_page())
        for page_id in self.page_ids:
            label = "Final Review" if page_id == REVIEW_PAGE_ID else self.pages[page_id].title
            self.page_listbox.insert("end", label)

        controls = ttk.Frame(sidebar)
        controls.pack(fill="x", pady=(10, 0))
        nav_controls = ttk.Frame(controls)
        nav_controls.pack(fill="x", pady=2)
        ttk.Button(nav_controls, text="Previous", command=self.go_to_previous_page).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(nav_controls, text="Next", command=self.go_to_next_page).pack(side="left", fill="x", expand=True, padx=(2, 0))
        ttk.Button(controls, text="Save Decisions", command=self.save_manifest).pack(fill="x", pady=2)
        ttk.Button(controls, text="Export Summary + Curated", command=self.export_summary).pack(fill="x", pady=2)
        ttk.Button(controls, text="Reload Page", command=self._render_selected_page).pack(fill="x", pady=2)

        info_text = (
            "Workflow\n"
            "- merge_group: combine units from the same session\n"
            "- align_group: same neuron across sessions\n"
            "- noise: exclude from final summary and curated session export"
        )
        ttk.Label(sidebar, text=info_text, justify="left").pack(anchor="w", pady=(12, 0))

        main = ttk.Frame(outer)
        main.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self.page_title_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.page_title_var, font=("Segoe UI", 14, "bold")).pack(anchor="w")

        self.similarity_var = tk.StringVar(value="")
        similarity_label = ttk.Label(
            main,
            textvariable=self.similarity_var,
            justify="left",
            wraplength=1150,
        )
        similarity_label.pack(anchor="w", fill="x", pady=(6, 8))

        self.similarity_panel = ttk.LabelFrame(main, text="Similarity Panel", padding=8)
        self.similarity_panel.pack(fill="x", expand=False, pady=(0, 8))

        self.canvas = tk.Canvas(main, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(main, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.page_content = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.page_content, anchor="nw")
        self.page_content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_content_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _selected_page(self) -> PageSummary | None:
        selection = self.page_listbox.curselection()
        if not selection:
            return None
        page_id = self.page_ids[selection[0]]
        if page_id == REVIEW_PAGE_ID:
            return None
        return self.pages[page_id]

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
        selected_index = self._selected_page_index()
        if selected_index is not None and self.page_ids[selected_index] == REVIEW_PAGE_ID:
            self._render_review_page()
            return

        page = self._selected_page()
        for child in self.page_content.winfo_children():
            child.destroy()
        if page is None:
            self.page_title_var.set("No page selected")
            self.similarity_var.set("")
            self._render_similarity_panel(None)
            return

        self.page_title_var.set(page.title)
        similarity_rows = compute_page_similarity_rows(page)
        self.similarity_var.set("Similarity hints: " + " | ".join(similarity_rows[:8]))
        self._render_similarity_panel(page)

        columns_frame = ttk.Frame(self.page_content)
        columns_frame.pack(fill="both", expand=True)

        for col_index, session in enumerate(page.sessions):
            session_frame = ttk.LabelFrame(columns_frame, text=session.session_name, padding=8)
            session_frame.grid(row=0, column=col_index, sticky="nsew", padx=6, pady=6)
            columns_frame.columnconfigure(col_index, weight=1)

            if not session.units:
                ttk.Label(session_frame, text="No units on this page for this session.").pack(anchor="w")
                continue

            for unit in session.units:
                self._render_unit_card(session_frame, unit)

        self.canvas.yview_moveto(0.0)

    def _render_review_page(self) -> None:
        for child in self.page_content.winfo_children():
            child.destroy()
        self._render_similarity_panel(None)

        summary = summarize_decisions(self.sessions)
        totals = summary["totals"]
        shank_id = summary["shank_id"]
        self.page_title_var.set("Final Review")
        self.similarity_var.set(
            "Review current decisions before export. This version assumes the selected input contains one shank only."
        )

        review_frame = ttk.Frame(self.page_content, padding=8)
        review_frame.pack(fill="both", expand=True)

        totals_frame = ttk.LabelFrame(review_frame, text="Global Totals", padding=10)
        totals_frame.pack(fill="x", expand=False, pady=(0, 10))
        totals_text = (
            f"Detected shank: {shank_id if shank_id is not None else 'unknown'}\n"
            f"Total units: {totals['total_units']}\n"
            f"Kept units: {totals['kept_units']}\n"
            f"Noise units: {totals['noise_units']}\n"
            f"Merge groups: {totals['merge_groups']}\n"
            f"Merged units involved: {totals['merged_units']}\n"
            f"Alignment groups: {totals['alignment_groups']}\n"
            f"Aligned units involved: {totals['aligned_units']}"
        )
        ttk.Label(totals_frame, text=totals_text, justify="left").pack(anchor="w")

        footer = ttk.Frame(review_frame)
        footer.pack(fill="x", pady=(10, 0))
        ttk.Button(footer, text="Save Decisions", command=self.save_manifest).pack(side="left")
        ttk.Button(footer, text="Export Summary + Curated", command=self.export_summary).pack(side="left", padx=(8, 0))
        self.canvas.yview_moveto(0.0)

    def _get_image(self, image_path: str):
        if image_path not in self.image_cache:
            if Image is not None and ImageTk is not None:
                pil_image = Image.open(image_path)
                pil_image.thumbnail(CARD_IMAGE_SIZE)
                self.image_cache[image_path] = ImageTk.PhotoImage(pil_image)
            else:
                self.image_cache[image_path] = tk.PhotoImage(file=image_path)
        return self.image_cache[image_path]

    def _render_similarity_panel(self, page: PageSummary | None) -> None:
        for child in self.similarity_panel.winfo_children():
            child.destroy()

        if page is None:
            ttk.Label(self.similarity_panel, text="No page selected.").pack(anchor="w")
            return

        page_id = page.page_id
        candidates = compute_page_similarity_candidates(page)
        if not candidates:
            ttk.Label(
                self.similarity_panel,
                text="No cross-session candidate pairs above the current similarity threshold.",
            ).pack(anchor="w")
            return

        self.page_pair_vars.setdefault(page_id, {})
        header = ttk.Frame(self.similarity_panel)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(
            header,
            text="Check likely pairs, then click Apply Selected Alignment to assign one align group.",
        ).pack(side="left", anchor="w")
        ttk.Button(
            header,
            text="Apply Selected Alignment",
            command=lambda current_page=page: self.apply_selected_pair_alignment(current_page),
        ).pack(side="right")

        rows_frame = ttk.Frame(self.similarity_panel)
        rows_frame.pack(fill="x", expand=False)
        for candidate in candidates:
            pair_id = f"{candidate.left_key}|{candidate.right_key}"
            var = self.page_pair_vars[page_id].setdefault(pair_id, tk.BooleanVar(value=False))
            row = ttk.Frame(rows_frame)
            row.pack(fill="x", pady=1)
            check = ttk.Checkbutton(row, variable=var)
            check.pack(side="left")
            score_label = tk.Label(
                row,
                text=f"{candidate.score:.3f}",
                width=7,
                fg=self._similarity_color(candidate.score),
            )
            score_label.pack(side="left")
            ttk.Label(
                row,
                text=f"{candidate.left_label}  <->  {candidate.right_label}",
            ).pack(side="left", anchor="w")

    def _similarity_color(self, score: float) -> str:
        if score >= 0.85:
            return "#0b7a28"
        if score >= 0.70:
            return "#b26a00"
        return "#b22222"

    def _units_by_key(self) -> dict[str, UnitSummary]:
        return {unit_record_key(unit): unit for unit in self._iter_all_units()}

    def apply_selected_pair_alignment(self, page: PageSummary) -> None:
        page_vars = self.page_pair_vars.get(page.page_id, {})
        selected_pair_ids = [pair_id for pair_id, var in page_vars.items() if var.get()]
        if not selected_pair_ids:
            messagebox.showinfo("No pairs selected", "Select one or more candidate pairs first.")
            return

        units_lookup = self._units_by_key()
        selected_components = build_pair_components(selected_pair_ids)
        if not selected_components:
            messagebox.showinfo("Not enough units", "At least two units are required for an alignment group.")
            return

        all_units = self._iter_all_units()
        page_channel_tag = f"sh{page.shank_id}_ch{page.local_channel_on_shank}"
        existing_align_members: dict[str, set[str]] = {}
        for unit in all_units:
            align_name = unit.align_group.strip()
            if align_name:
                scoped_key = f"sh{unit.shank_id}_ch{unit.local_channel_on_shank}::{align_name}"
                existing_align_members.setdefault(scoped_key, set()).add(unit_record_key(unit))

        existing_align_names = {
            scoped_key.split("::", maxsplit=1)[1]
            for scoped_key in existing_align_members.keys()
            if scoped_key.startswith(f"{page_channel_tag}::")
        }
        assigned_groups: list[tuple[str, int]] = []
        base_name = f"align_sh{page.shank_id}_ch{page.local_channel_on_shank}"
        next_group_index = 1

        for component in selected_components:
            component = {unit_key for unit_key in component if unit_key in units_lookup}
            if len(component) < 2:
                continue

            existing_names_in_component = sorted(
                {
                    units_lookup[unit_key].align_group.strip()
                    for unit_key in component
                    if units_lookup[unit_key].align_group.strip()
                }
            )
            expanded_component = set(component)
            for align_name in existing_names_in_component:
                expanded_component.update(
                    existing_align_members.get(f"{page_channel_tag}::{align_name}", set())
                )

            if existing_names_in_component:
                proposed_name = existing_names_in_component[0]
            else:
                proposed_name = f"{base_name}_{next_group_index:02d}"
                while proposed_name in existing_align_names:
                    next_group_index += 1
                    proposed_name = f"{base_name}_{next_group_index:02d}"
                existing_align_names.add(proposed_name)
                next_group_index += 1

            for unit_key in expanded_component:
                unit = units_lookup.get(unit_key)
                if unit is None:
                    continue
                vars_for_unit = self._ensure_unit_vars(unit)
                vars_for_unit["align_group"].set(proposed_name)
                unit.align_group = proposed_name

            existing_align_members[f"{page_channel_tag}::{proposed_name}"] = set(expanded_component)
            for old_name in existing_names_in_component:
                if old_name != proposed_name:
                    existing_align_members.pop(f"{page_channel_tag}::{old_name}", None)

            assigned_groups.append((proposed_name, len(expanded_component)))

        for pair_id in selected_pair_ids:
            page_vars[pair_id].set(False)

        self._render_selected_page()
        if not assigned_groups:
            messagebox.showinfo("Not enough units", "No valid pair groups were selected.")
            return
        message_lines = [
            f"{group_name}: {unit_count} unit(s)"
            for group_name, unit_count in assigned_groups
        ]
        messagebox.showinfo(
            "Alignment assigned",
            "Assigned alignment groups:\n" + "\n".join(message_lines),
        )

    def _ensure_unit_vars(self, unit: UnitSummary) -> dict[str, tk.Variable]:
        key = unit_record_key(unit)
        if key not in self.unit_control_vars:
            self.unit_control_vars[key] = {
                "merge_group": tk.StringVar(value=unit.merge_group),
                "align_group": tk.StringVar(value=unit.align_group),
                "is_noise": tk.BooleanVar(value=unit.is_noise),
            }
        return self.unit_control_vars[key]

    def _render_unit_card(self, parent: ttk.Frame, unit: UnitSummary) -> None:
        vars_for_unit = self._ensure_unit_vars(unit)
        card = ttk.Frame(parent, relief="solid", padding=6)
        card.pack(fill="x", expand=True, pady=6)

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

        image = self._get_image(unit.waveform_image_path)
        image_label = ttk.Label(card, image=image)
        image_label.image = image
        image_label.pack(anchor="w", pady=(4, 4))

        metrics_text = (
            f"Amplitude median: {format_metric(unit.amplitude_median)}\n"
            f"ISI violation ratio: {format_metric(unit.isi_violations_ratio)}\n"
            f"Num spikes: {format_metric(unit.num_spikes)}"
        )
        ttk.Label(card, text=metrics_text, justify="left").pack(anchor="w")

        form = ttk.Frame(card)
        form.pack(fill="x", pady=(5, 0))
        ttk.Label(form, text="Merge group").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=vars_for_unit["merge_group"], width=16).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Label(form, text="Align group").grid(row=1, column=0, sticky="w")
        ttk.Entry(form, textvariable=vars_for_unit["align_group"], width=16).grid(row=1, column=1, sticky="ew", padx=(6, 0))
        ttk.Checkbutton(form, text="Mark as noise", variable=vars_for_unit["is_noise"]).grid(row=2, column=0, columnspan=2, sticky="w")
        form.columnconfigure(1, weight=1)

    def _iter_all_units(self) -> list[UnitSummary]:
        units: list[UnitSummary] = []
        for session in self.sessions:
            units.extend(session.units)
        return units

    def _apply_control_state_to_units(self) -> None:
        for unit in self._iter_all_units():
            vars_for_unit = self._ensure_unit_vars(unit)
            unit.merge_group = vars_for_unit["merge_group"].get().strip()
            unit.align_group = vars_for_unit["align_group"].get().strip()
            unit.is_noise = bool(vars_for_unit["is_noise"].get())

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
            unit.is_noise = bool(saved.get("is_noise", False))
            vars_for_unit = self._ensure_unit_vars(unit)
            vars_for_unit["merge_group"].set(unit.merge_group)
            vars_for_unit["align_group"].set(unit.align_group)
            vars_for_unit["is_noise"].set(unit.is_noise)

    def export_summary(self) -> None:
        self.save_manifest(show_message=False)
        export_folder = self.summary_root / "exported_units"
        export_folder.mkdir(parents=True, exist_ok=True)
        curated_sessions_root = self.summary_root / "curated_sessions"
        curated_sessions_root.mkdir(parents=True, exist_ok=True)

        self._apply_control_state_to_units()
        curated_session_exports = []
        for session in self.sessions:
            curated_session_exports.append(
                create_curated_analyzer_from_session(
                    session=session,
                    export_root=curated_sessions_root,
                )
            )

        final_groups: dict[str, list[UnitSummary]] = {}

        for unit in self._iter_all_units():
            if unit.is_noise:
                continue

            session_tag = f"s{unit.session_index:03d}"
            base_id = f"{session_tag}_u{unit.unit_id}"
            channel_tag = f"sh{unit.shank_id}_ch{unit.local_channel_on_shank}"
            merge_key = (
                f"{channel_tag}__{session_tag}__merge__{sanitize_token(unit.merge_group)}"
                if unit.merge_group
                else base_id
            )
            align_key = (
                f"{channel_tag}__align__{sanitize_token(unit.align_group)}"
                if unit.align_group
                else merge_key
            )
            final_key = align_key
            final_groups.setdefault(final_key, []).append(unit)

        manifest_rows = []
        for group_index, (group_key, units) in enumerate(sorted(final_groups.items()), start=1):
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

            manifest_rows.append(
                {
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

        export_manifest_path = self.summary_root / "export_summary.json"
        export_manifest_path.write_text(
            json.dumps(
                {
                    "output_root": str(self.output_root),
                    "curated_sessions": curated_session_exports,
                    "cross_session_alignment_groups": manifest_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        messagebox.showinfo(
            "Export complete",
            f"Saved summary folder to:\n{self.summary_root}\n\n"
            f"Cross-session groups kept: {len(manifest_rows)}\n"
            f"Curated session analyzers: {len(curated_session_exports)}",
        )

    def _build_group_summary_text(self, group_index: int, group_key: str, units: list[UnitSummary]) -> str:
        lines = [
            f"Final unit #{group_index}",
            f"Group key: {group_key}",
            f"Shank: {units[0].shank_id}",
            f"Channel: {units[0].local_channel_on_shank}",
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
    try:
        app = AlignmentApp(root, output_root)
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("Units Alignment UI", str(exc), parent=root)
        root.destroy()
        raise SystemExit(1)
    root.mainloop()


if __name__ == "__main__":
    main()
