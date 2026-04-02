from pathlib import Path
import json
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import re

import matplotlib.pyplot as plt

import spikeinterface.curation as scur
import spikeinterface.full as si
import spikeinterface.widgets as sw


DEFAULT_ANALYZER_FOLDER = Path(
    r"W:\6h_Testing\NWB_20260220_201706_NWB_20260221_011710\NWB_20260220_201706_NWB_20260221_011710_sh4_first30min\sorting_analyzer_analysis.zarr"
)
MAX_SPIKES_PER_UNIT = 500
MS_BEFORE = 1.0
MS_AFTER = 2.0
DESIRED_UNIT_TABLE_PROPERTIES = [
    "amplitude_median",
    "firing_rate",
    "isi_violations_ratio",
    "num_spikes",
    "snr",
]


def make_hidden_root() -> tk.Tk:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def choose_analyzer_folder(default_folder: Path) -> Path:
    root = make_hidden_root()
    initial_dir = default_folder if default_folder.is_dir() else default_folder.parent
    selected_folder = filedialog.askdirectory(
        title="Select a SortingAnalyzer zarr folder",
        initialdir=str(initial_dir),
        mustexist=True,
        parent=root,
    )
    root.destroy()

    if not selected_folder:
        raise SystemExit("No analyzer folder selected.")

    analyzer_folder = Path(selected_folder)
    if analyzer_folder.suffix.lower() != ".zarr":
        raise ValueError(f"Selected folder is not a .zarr store: {analyzer_folder}")

    return analyzer_folder


def choose_curation_source(default_json_path: Path):
    root = make_hidden_root()

    curation_url = simpledialog.askstring(
        "SortingView curation",
        "Paste the SortingView curation URL.\n\n"
        "Leave this blank and press OK to choose a JSON file instead.",
        parent=root,
    )

    if curation_url is not None and curation_url.strip():
        root.destroy()
        return curation_url.strip()

    selected_json = filedialog.askopenfilename(
        title="Select a SortingView curation JSON file",
        initialdir=str(default_json_path.parent),
        initialfile=default_json_path.name,
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        parent=root,
    )
    root.destroy()

    if not selected_json:
        raise SystemExit("No curation URL or JSON file provided.")

    return Path(selected_json)


def choose_curation_mode() -> str:
    root = make_hidden_root()
    use_folder_mode = messagebox.askyesnocancel(
        "Curation mode",
        "Choose curation workflow.\n\n"
        "Yes: folder-based good / bad / merge workflow (Recommended)\n"
        "No: SortingView URL/JSON workflow (Backup)\n"
        "Cancel: exit",
        parent=root,
    )
    root.destroy()

    if use_folder_mode is None:
        raise SystemExit("No curation mode selected.")
    return "folder" if use_folder_mode else "sortingview"


def save_text(text: str, path: Path):
    path.write_text(text, encoding="utf-8")
    print(f"Saved: {path}")


def save_json(payload: dict, path: Path):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


def extract_figurl(plot_result) -> str | None:
    candidates = []
    for attr_name in ("url", "figurl", "uri"):
        value = getattr(plot_result, attr_name, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
            except Exception:
                value = None
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    if hasattr(plot_result, "view"):
        view = getattr(plot_result, "view")
        for attr_name in ("url", "figurl", "uri"):
            value = getattr(view, attr_name, None)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
                except Exception:
                    value = None
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())

    for candidate in candidates:
        if candidate.startswith("http"):
            return candidate
    return candidates[0] if candidates else None


def prompt_for_link(title: str, prompt: str) -> str | None:
    root = make_hidden_root()
    value = simpledialog.askstring(title, prompt, parent=root)
    root.destroy()
    if value is None or not value.strip():
        return None
    return value.strip()


def get_or_prompt_figurl(plot_result, title: str, prompt: str) -> str | None:
    figurl = extract_figurl(plot_result)
    if figurl:
        return figurl
    return prompt_for_link(title, prompt)


def create_unit_figure_folder(analysis_folder: Path) -> Path:
    unit_figure_folder = analysis_folder / "unit_summary_before_curation"
    unit_figure_folder.mkdir(parents=True, exist_ok=True)
    (unit_figure_folder / "good_units").mkdir(exist_ok=True)
    (unit_figure_folder / "bad_units").mkdir(exist_ok=True)
    for merge_index in range(5):
        (unit_figure_folder / f"merge{merge_index}").mkdir(exist_ok=True)
    return unit_figure_folder


def collect_root_unit_pngs(unit_figure_folder: Path) -> list[tuple[int, Path]]:
    unit_pngs = []
    for png_path in sorted(unit_figure_folder.glob("unit_summary_*.png")):
        unit_id = parse_unit_id_from_png_name(png_path.name)
        if unit_id is not None:
            unit_pngs.append((unit_id, png_path))
    return unit_pngs


def parse_unit_id_from_png_name(name: str) -> int | None:
    match = re.search(r"unit_summary_shank(\d+)_ch(\d+)_sg(\d+)_(\d+)$", Path(name).stem)
    if match:
        return int(match.group(4))

    match = re.search(r"unit_summary_(\d+)", name)
    if match:
        return int(match.group(1))

    return None


def parse_shank_channel_from_png_name(name: str) -> tuple[int | None, int | None]:
    match = re.search(r"unit_summary_shank(\d+)_ch(\d+)_sg(\d+)_(\d+)$", Path(name).stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def get_quality_metrics_lookup(analyzer) -> dict[int, dict[str, float]]:
    if not analyzer.has_extension("quality_metrics"):
        return {}

    metrics_df = analyzer.get_extension("quality_metrics").get_data()
    if "unit_id" not in metrics_df.columns:
        metrics_df = metrics_df.reset_index()
        if "unit_id" not in metrics_df.columns and "index" in metrics_df.columns:
            metrics_df = metrics_df.rename(columns={"index": "unit_id"})

    if "unit_id" not in metrics_df.columns:
        return {}

    metrics_lookup = {}
    for row in metrics_df.itertuples(index=False):
        unit_id = int(getattr(row, "unit_id"))
        metrics_lookup[unit_id] = {
            "firing_rate": getattr(row, "firing_rate", float("nan")),
            "snr": getattr(row, "snr", float("nan")),
        }
    return metrics_lookup


def infer_unit_channel_metadata(analyzer, unit_id) -> dict[str, int]:
    waveforms = analyzer.get_extension("waveforms").get_waveforms_one_unit(unit_id)
    if waveforms.size == 0:
        channel_index = 0
    else:
        channel_index = int((abs(waveforms.mean(axis=0))).max(axis=0).argmax())

    channel_ids = list(analyzer.channel_ids)
    sg_channel = channel_ids[channel_index] if channel_index < len(channel_ids) else channel_index
    try:
        sg_channel = int(sg_channel)
    except Exception:
        sg_channel = channel_index

    shank_id = 0
    if analyzer.recording.get_property_keys() and "group" in analyzer.recording.get_property_keys():
        group_values = analyzer.recording.get_property("group")
        if channel_index < len(group_values):
            match = re.search(r"(\d+)", str(group_values[channel_index]))
            if match:
                shank_id = int(match.group(1))

    return {
        "shank_id": shank_id,
        "local_channel_on_shank": channel_index,
        "sg_channel": sg_channel,
    }


def save_prepare_stage_unit_summaries(analyzer, analysis_folder: Path) -> Path:
    unit_figure_folder = create_unit_figure_folder(analysis_folder)
    metrics_lookup = get_quality_metrics_lookup(analyzer)

    for png_path in unit_figure_folder.glob("unit_summary_*.png"):
        png_path.unlink()

    for unit_id in analyzer.sorting.get_unit_ids():
        unit_id_int = int(unit_id)
        channel_metadata = infer_unit_channel_metadata(analyzer, unit_id)
        quality_metrics = metrics_lookup.get(unit_id_int, {})
        firing_rate = quality_metrics.get("firing_rate", float("nan"))
        snr = quality_metrics.get("snr", float("nan"))
        firing_rate_text = f"{float(firing_rate):.3f}" if firing_rate == firing_rate else "nan"
        snr_text = f"{float(snr):.3f}" if snr == snr else "nan"

        sw.plot_unit_summary(analyzer, unit_id=unit_id)
        figure = plt.gcf()
        figure.suptitle(
            f"Unit {unit_id_int} | shank {channel_metadata['shank_id']}, "
            f"ch {channel_metadata['local_channel_on_shank']}, "
            f"sg {channel_metadata['sg_channel']} | "
            f"FR={firing_rate_text} Hz, SNR={snr_text}",
            fontsize=11,
        )
        figure_name = (
            f"unit_summary_shank{channel_metadata['shank_id']}_"
            f"ch{channel_metadata['local_channel_on_shank']}_"
            f"sg{channel_metadata['sg_channel']}_"
            f"{unit_id_int}.png"
        )
        plt.savefig(unit_figure_folder / figure_name, dpi=250)
        plt.close()

    return unit_figure_folder


def folder_curation_has_assignments(unit_figure_folder: Path) -> bool:
    for folder_name in ["good_units", "bad_units", "merge0", "merge1", "merge2", "merge3", "merge4"]:
        if any((unit_figure_folder / folder_name).glob("*.png")):
            return True
    return False


def sync_labels_json_from_folders(unit_figure_folder: Path, labels_json_path: Path):
    labels_by_unit = {}
    for png_path in sorted((unit_figure_folder / "good_units").glob("*.png")):
        unit_id = parse_unit_id_from_png_name(png_path.name)
        if unit_id is not None:
            labels_by_unit[str(unit_id)] = "good"

    for png_path in sorted((unit_figure_folder / "bad_units").glob("*.png")):
        unit_id = parse_unit_id_from_png_name(png_path.name)
        if unit_id is not None:
            labels_by_unit[str(unit_id)] = "noise"

    save_json({"labelsByUnit": labels_by_unit}, labels_json_path)
    return labels_by_unit


def build_curation_json_from_folders(unit_figure_folder: Path, labels_json_path: Path, curation_json_path: Path):
    if labels_json_path.exists():
        labels_payload = json.loads(labels_json_path.read_text(encoding="utf-8"))
    else:
        labels_payload = {"labelsByUnit": {}}

    labels_by_unit_raw = labels_payload.get("labelsByUnit", {})
    all_units = [unit_id for unit_id, _ in collect_root_unit_pngs(unit_figure_folder)]
    for folder_name in ("good_units", "bad_units"):
        for png_path in sorted((unit_figure_folder / folder_name).glob("*.png")):
            unit_id = parse_unit_id_from_png_name(png_path.name)
            if unit_id is not None:
                all_units.append(unit_id)

    labels_by_unit = {}
    for unit_id in sorted(set(all_units)):
        label_value = str(labels_by_unit_raw.get(str(unit_id), "")).lower()
        labels_by_unit[str(unit_id)] = ["accept"] if label_value == "good" else ["reject"]

    merge_groups = []
    for merge_index in range(5):
        merge_folder = unit_figure_folder / f"merge{merge_index}"
        merge_units_by_channel = {}
        for png_path in sorted(merge_folder.glob("*.png")):
            unit_id = parse_unit_id_from_png_name(png_path.name)
            shank_id, channel_id = parse_shank_channel_from_png_name(png_path.name)
            if unit_id is None or shank_id is None or channel_id is None:
                continue
            merge_units_by_channel.setdefault((shank_id, channel_id), set()).add(int(unit_id))

        for unit_ids in merge_units_by_channel.values():
            if len(unit_ids) > 1:
                merge_groups.append(sorted(unit_ids))

    curation_payload = {
        "labelsByUnit": labels_by_unit,
        "mergeGroups": merge_groups,
    }
    save_json(curation_payload, curation_json_path)
    return curation_payload


def write_folder_curation_instructions(analysis_folder: Path, unit_figure_folder: Path):
    instructions = (
        "Folder-based curation is prepared.\n"
        f"Move unit summary PNGs under: {unit_figure_folder}\n"
        "  - good_units/: accepted units\n"
        "  - bad_units/: rejected units\n"
        "  - merge0..merge4/: units to merge\n"
        "Merge rule: only units on the same (shank, channel) are merged together.\n"
        "After organizing the folders, rerun this script and choose the folder-based option again.\n"
    )
    save_text(instructions, analysis_folder / "CURATION_NEXT_STEPS.txt")


def get_unit_table_properties(analyzer) -> list[str]:
    if not analyzer.has_extension("quality_metrics"):
        return []

    metrics_df = analyzer.get_extension("quality_metrics").get_data()
    available_metrics = {str(column) for column in metrics_df.columns}
    metrics_list = [
        metric_name
        for metric_name in DESIRED_UNIT_TABLE_PROPERTIES
        if metric_name in available_metrics
    ]
    print(f"Adding metrics to table: {metrics_list}")
    return metrics_list


def ensure_summary_extensions(analyzer):
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


def create_curated_analyzer(sorting_analyzer, accepted_sorting):
    if not sorting_analyzer.has_recording():
        raise ValueError(
            "The loaded SortingAnalyzer does not have its recording attached, "
            "so a curated analyzer cannot be created from it."
        )

    curated_analyzer = si.create_sorting_analyzer(
        sorting=accepted_sorting,
        recording=sorting_analyzer.recording,
        format="memory",
    )
    ensure_summary_extensions(curated_analyzer)
    return curated_analyzer


def apply_curation_source(
    sorting_analyzer,
    analyzer_folder: Path,
    analysis_folder: Path,
    curation_source,
    curation_source_info: dict,
    source_summary_link: str | None,
):
    if isinstance(curation_source, str):
        save_text(curation_source + "\n", analysis_folder / "sortingview_curated_link.txt")
    else:
        save_text(str(curation_source) + "\n", analysis_folder / "sortingview_curation_json_path.txt")

    print(f"Applying curation from: {curation_source}")
    accepted_sorting = scur.apply_sortingview_curation(
        sorting_analyzer.sorting,
        uri_or_json=curation_source,
        include_labels=["accept"],
        skip_merge=False,
    )

    curated_analyzer = create_curated_analyzer(sorting_analyzer, accepted_sorting)
    curated_analyzer_folder = analyzer_folder.with_name(f"{analyzer_folder.stem}_curated.zarr")
    curated_analyzer.save_as(folder=curated_analyzer_folder, format="zarr")
    print(f"Saved curated analyzer: {curated_analyzer_folder}")

    print("Plotting curated sorting summary...")
    curated_unit_table_properties = get_unit_table_properties(curated_analyzer)
    curated_plot = sw.plot_sorting_summary(
        curated_analyzer,
        min_similarity_for_correlograms=0.2,
        backend="sortingview",
        unit_table_properties=curated_unit_table_properties,
    )
    curated_summary_link = get_or_prompt_figurl(
        curated_plot,
        "Curated summary link",
        "Paste the figurl link that was generated for the curated sorting summary.",
    )
    if curated_summary_link:
        save_text(curated_summary_link + "\n", analysis_folder / "sortingview_curated_summary_link.txt")

    save_json(
        {
            "analyzer_folder": str(analyzer_folder),
            "curated_analyzer_folder": str(curated_analyzer_folder),
            "curation_source": curation_source_info,
            "sortingview_curation_summary_link": source_summary_link,
            "sortingview_curated_summary_link": curated_summary_link,
        },
        analysis_folder / "curation_outputs.json",
    )

    root = make_hidden_root()
    show_unit_summaries = messagebox.askyesno(
        "Curated unit summaries",
        "Plot a matplotlib unit summary for each curated unit?",
        parent=root,
    )
    root.destroy()

    if show_unit_summaries:
        for unit_id in curated_analyzer.sorting.get_unit_ids():
            sw.plot_unit_summary(curated_analyzer, unit_id=unit_id)


def run_sortingview_mode(sorting_analyzer, analyzer_folder: Path, analysis_folder: Path):
    unit_table_properties = get_unit_table_properties(sorting_analyzer)
    print("Launching SortingView summary for manual curation...")
    curation_plot = sw.plot_sorting_summary(
        sorting_analyzer,
        min_similarity_for_correlograms=0.2,
        curation=True,
        backend="sortingview",
        unit_table_properties=unit_table_properties,
    )

    curation_summary_link = get_or_prompt_figurl(
        curation_plot,
        "SortingView summary link",
        "Paste the figurl link that was generated for the curation summary.",
    )
    if curation_summary_link:
        save_text(curation_summary_link + "\n", analysis_folder / "sortingview_curation_summary_link.txt")

    default_json_path = analyzer_folder.with_name("curation.json")
    curation_source = choose_curation_source(default_json_path)
    curation_source_info = {
        "type": "sortingview_url" if isinstance(curation_source, str) else "json_file",
        "value": curation_source if isinstance(curation_source, str) else str(curation_source),
        "mode": "sortingview_backup",
    }
    if not isinstance(curation_source, str) and curation_source.resolve() != default_json_path.resolve():
        save_text(curation_source.read_text(encoding="utf-8"), default_json_path)

    apply_curation_source(
        sorting_analyzer=sorting_analyzer,
        analyzer_folder=analyzer_folder,
        analysis_folder=analysis_folder,
        curation_source=curation_source,
        curation_source_info=curation_source_info,
        source_summary_link=curation_summary_link,
    )


def run_folder_mode(sorting_analyzer, analyzer_folder: Path, analysis_folder: Path):
    unit_table_properties = get_unit_table_properties(sorting_analyzer)
    unit_figure_folder = create_unit_figure_folder(analysis_folder)
    labels_json_path = analysis_folder / "curation_labels.json"
    curation_json_path = analysis_folder / "curation.json"

    if folder_curation_has_assignments(unit_figure_folder):
        print("Detected folder-based curation assignments. Building curation JSON and applying curation...")
        sync_labels_json_from_folders(unit_figure_folder, labels_json_path)
        build_curation_json_from_folders(unit_figure_folder, labels_json_path, curation_json_path)
        curation_source_info = {
            "type": "json_file",
            "value": str(curation_json_path),
            "mode": "folder_based_default",
        }
        apply_curation_source(
            sorting_analyzer=sorting_analyzer,
            analyzer_folder=analyzer_folder,
            analysis_folder=analysis_folder,
            curation_source=curation_json_path,
            curation_source_info=curation_source_info,
            source_summary_link=(analysis_folder / "sortingview_reference_link.txt").read_text(encoding="utf-8").strip()
            if (analysis_folder / "sortingview_reference_link.txt").exists()
            else None,
        )
        return

    print("Preparing folder-based curation summaries...")
    save_prepare_stage_unit_summaries(sorting_analyzer, analysis_folder)
    reference_plot = sw.plot_sorting_summary(
        sorting_analyzer,
        min_similarity_for_correlograms=0.2,
        curation=True,
        backend="sortingview",
        unit_table_properties=unit_table_properties,
    )
    reference_link = get_or_prompt_figurl(
        reference_plot,
        "SortingView reference link",
        "Paste the figurl link that was generated for the reference SortingView summary.",
    )
    if reference_link:
        save_text(reference_link + "\n", analysis_folder / "sortingview_reference_link.txt")

    save_json({"labelsByUnit": {}}, labels_json_path)
    write_folder_curation_instructions(analysis_folder, unit_figure_folder)
    root = make_hidden_root()
    messagebox.showinfo(
        "Folder curation prepared",
        f"Unit summaries are ready in:\n{unit_figure_folder}\n\n"
        "Move PNGs into good_units / bad_units / merge* folders, then rerun this script and choose folder-based mode again.",
        parent=root,
    )
    root.destroy()


def main():
    analyzer_folder = choose_analyzer_folder(DEFAULT_ANALYZER_FOLDER)
    sorting_analyzer = si.load_sorting_analyzer(
        folder=analyzer_folder,
        format="zarr",
        load_extensions=True,
    )
    ensure_summary_extensions(sorting_analyzer)
    analysis_folder = analyzer_folder.parent

    print(f"Loaded analyzer: {analyzer_folder}")
    curation_mode = choose_curation_mode()
    if curation_mode == "folder":
        run_folder_mode(sorting_analyzer, analyzer_folder, analysis_folder)
    else:
        run_sortingview_mode(sorting_analyzer, analyzer_folder, analysis_folder)


if __name__ == "__main__":
    main()
