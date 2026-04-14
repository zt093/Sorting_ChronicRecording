from pathlib import Path
import json
import re
import shutil
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, CheckButtons
import spikeinterface.full as si
import spikeinterface.widgets as sw
import spikeinterface.curation as scur

from lsnet_pipeline.recording_io import load_saved_recording_with_probe

from StimulationLSNET_curation_utils import (
    run_waveform_curation_and_unit_figures,
)

# --------------------- CONFIGURATION (defaults; set by env or interactive prompts) ---------------------
# Backward compatibility: set LSNET_PIPELINE_OUTPUT_FOLDER (+ optional LSNET_PIPELINE_RECORDING_DIR,
# LSNET_PIPELINE_CURATION_STAGE) to skip prompts and run non-interactively.
# Optional env:
#   LSNET_PIPELINE_PRESET=1|2 — apply preset (1=EBL512/512ch/top-level; 2=LSNET18 Chronic + LSNET_probe)
#   LSNET_PIPELINE_CURATION_STAGE=1|2|prepare_figures|apply_curation
#   LSNET_PIPELINE_RESOLVE_TOP_LEVEL_ONLY=0|1 — override preset shank resolution (when mixed top/shanks)


@dataclass(frozen=True)
class CurationInteractivePreset:
    """Bundled defaults for interactive prompts (browse initial dirs, SG map size, shank resolution)."""

    key: str
    title: str
    default_output_folder: Path
    default_probe_file: Path
    user_fallback_sg_totalchan: int
    # If True and both top-level sorting and shank_* folders exist, use top-level only (no prompt).
    top_level_only_when_mixed: bool


CURATION_PRESETS: dict[str, CurationInteractivePreset] = {
    "1": CurationInteractivePreset(
        key="1",
        title="EBL512-style: 512-channel map, prefer top-level sorting when shank folders exist",
        default_output_folder=Path(
            r"F:\EBL512_4sh\20260402_114212_B3_1.rec\ephys512_inspect_20260403_163904\shank_1"
        ),
        default_probe_file=Path(r"H:\FromYuhang\128\128_script\util\NET-EBL-4-512.json"),
        user_fallback_sg_totalchan=512,
        top_level_only_when_mixed=True,
    ),
    "2": CurationInteractivePreset(
        key="2",
        title="LSNET18 Chronic example: ms5 output + LSNET_probe.json",
        default_output_folder=Path(
            r"E:\Centimani\Recording\LSNET18_06162025\Chronic_Rec_20260227_144213.rec\ms5_d50min_0412_2043"
        ),
        default_probe_file=Path(r"E:\Centimani\Recording\Scripts\LSNET_probe.json"),
        user_fallback_sg_totalchan=384,
        top_level_only_when_mixed=True,
    ),
}


def apply_preset_by_id(preset_key: str) -> CurationInteractivePreset:
    """Set browse defaults, SG channel count, and shank-resolution behavior from a preset."""
    global DEFAULT_BROWSE_SORTING_OUTPUT_FOLDER, DEFAULT_PROBE_FILE, USER_FALLBACK_SG_TOTALCHAN
    global RESOLVE_TOP_LEVEL_ONLY_DEFAULT, USER_PROBE_FILE

    pk = (preset_key or "1").strip()
    if pk not in CURATION_PRESETS:
        pk = "1"
    p = CURATION_PRESETS[pk]
    DEFAULT_BROWSE_SORTING_OUTPUT_FOLDER = Path(p.default_output_folder)
    DEFAULT_PROBE_FILE = Path(p.default_probe_file)
    USER_FALLBACK_SG_TOTALCHAN = int(p.user_fallback_sg_totalchan)
    RESOLVE_TOP_LEVEL_ONLY_DEFAULT = bool(p.top_level_only_when_mixed)
    USER_PROBE_FILE = DEFAULT_PROBE_FILE
    return p


# Populated by apply_preset_by_id (default = preset 1).
DEFAULT_BROWSE_SORTING_OUTPUT_FOLDER = Path(
    r"F:\EBL512_4sh\20260402_114212_B3_1.rec\ephys512_inspect_20260403_163904\shank_1"
)
DEFAULT_PROBE_FILE = Path(r"H:\FromYuhang\128\128_script\util\NET-EBL-4-512.json")
RESOLVE_TOP_LEVEL_ONLY_DEFAULT = True
USER_FALLBACK_SG_TOTALCHAN = 512

apply_preset_by_id("1")

USER_OUTPUT_FOLDER = DEFAULT_BROWSE_SORTING_OUTPUT_FOLDER
USER_RECORDING_DIR = None
DURATION_MIN_SORTING = 50

# Internal: "prepare_figures" | "apply_curation" (interactive uses 1 / 2)
CURATION_STAGE = "prepare_figures"
RUN_LABEL_UI_IN_PREPARE = True
REGENERATE_PREPARE_STAGE_PLOTS = False
SHOW_PREPARE_UI_STATS = True

RUN_WAVEFORM_LEVEL_CURATION = False
USER_STIM_TS_PATH = None
USER_STIM_TS_MAT_VAR = "stim_times_arr"
PLOT_POSTCURATION_SUMMARY_PLOTS = True
PLOT_POSTCURATION_WAVEFORMS = True
# Fig4-style population metrics (FR, SNR, ISI viol, spike amp); see lsnet_curation_unit_summary_fig4.py
PLOT_POSTCURATION_FIG4_POPULATION_METRICS = True
# Full grid: mean waveform ± std + ISI per unit; see lsnet_curation_all_units_grid.py
PLOT_POSTCURATION_ALL_UNITS_GRID_WAVEFORM_ISI = True
PRINT_CURATED_PLOT_LOGS = False
DEBUG_TRACE_POSTCURATION_MAPPING = True

USER_PROBE_FILE = DEFAULT_PROBE_FILE
# ---------------------------------------------------------------------------------------------------


def normalize_curation_stage_token(token: str) -> str:
    """
    Map user/env input to prepare_figures or apply_curation.
    Accepts 1 / 2, short names, or full stage names.
    """
    if token is None:
        raise ValueError("Curation stage is empty.")
    s = str(token).strip().lower()
    if not s:
        raise ValueError("Curation stage is empty.")
    if s in ("1", "prepare_figures", "prepare", "prep", "p", "label", "figures"):
        return "prepare_figures"
    if s in ("2", "apply_curation", "apply", "a", "export", "curated"):
        return "apply_curation"
    raise ValueError(
        f"Unknown curation stage {token!r}. Use 1 or 2 (or prepare_figures / apply_curation)."
    )


def _tk_initialdir_for_folder(path: Path) -> str:
    path = Path(path)
    if path.is_dir():
        return str(path)
    if path.parent.is_dir():
        return str(path.parent)
    return str(Path.home())


def _tk_initialdir_and_file(path: Path) -> tuple[str, str]:
    path = Path(path)
    name = path.name
    if path.is_file():
        return str(path.parent), name
    if path.parent.is_dir():
        return str(path.parent), name
    return str(Path.home()), name


def _has_sorting_artifacts(folder: Path) -> bool:
    return (
        (folder / "sorted_sorting").exists()
        or (folder / "sorted_units.npz").exists()
        or (folder / "sorter_output").exists()
    )


def normalize_pipeline_output_folder(folder: Path) -> Path:
    """
    If the user picked a SpikeInterface sorting-analyzer directory (e.g.
    sorting_analyzer_before_curation.zarr), use its parent as the pipeline output
    folder (where saved_rec / saved_recording / sorted_sorting live).
    """
    p = folder.resolve()
    if not p.is_dir():
        return p
    name_l = p.name.lower()
    if name_l.endswith(".zarr") and name_l.startswith("sorting_analyzer"):
        return p.parent
    if p.name in ("sorting_analyzer_before_curation", "sorting_analyzer"):
        return p.parent
    return p


def _resolve_saved_rec_binary_folder(output_folder: Path) -> Path:
    """Prefer saved_rec (LSNET); fall back to saved_recording (SortingEBL128)."""
    saved_rec = output_folder / "saved_rec"
    saved_recording = output_folder / "saved_recording"
    run_meta = output_folder / "preprocessing_metadata_for_run.json"
    if run_meta.exists():
        try:
            with open(run_meta, "r", encoding="utf-8") as f:
                md = json.load(f)
            src = md.get("params", {}).get("source_saved_rec_path", None)
            if src:
                return Path(src)
        except Exception:
            pass
    if (saved_rec / "binary.json").exists():
        return saved_rec
    if (saved_recording / "binary.json").exists():
        return saved_recording
    return saved_rec


def infer_recording_dir_from_output(output_folder: Path) -> Path:
    """Parent .rec for .../file.rec/ms5_* or .../file.rec/ephys*_inspect_*/shank_k."""
    p = output_folder.resolve()
    if p.name.startswith("shank_") and p.parent.name.startswith("ephys"):
        return p.parent.parent
    return p.parent


def apply_env_config():
    """Apply LSNET_PIPELINE_* environment overrides (non-interactive / batch)."""
    global USER_OUTPUT_FOLDER, USER_RECORDING_DIR, CURATION_STAGE, RESOLVE_TOP_LEVEL_ONLY_DEFAULT

    preset = os.environ.get("LSNET_PIPELINE_PRESET", "").strip()
    if preset in CURATION_PRESETS:
        apply_preset_by_id(preset)

    out = os.environ.get("LSNET_PIPELINE_OUTPUT_FOLDER", "").strip()
    rec = os.environ.get("LSNET_PIPELINE_RECORDING_DIR", "").strip()
    stage = os.environ.get("LSNET_PIPELINE_CURATION_STAGE", "").strip()
    if out:
        USER_OUTPUT_FOLDER = normalize_pipeline_output_folder(Path(out))
    if rec:
        USER_RECORDING_DIR = Path(rec)
    elif USER_OUTPUT_FOLDER is not None:
        USER_RECORDING_DIR = infer_recording_dir_from_output(USER_OUTPUT_FOLDER)
    if stage:
        CURATION_STAGE = normalize_curation_stage_token(stage)

    tl = os.environ.get("LSNET_PIPELINE_RESOLVE_TOP_LEVEL_ONLY", "").strip().lower()
    if tl in ("1", "true", "yes", "y"):
        RESOLVE_TOP_LEVEL_ONLY_DEFAULT = True
    elif tl in ("0", "false", "no", "n"):
        RESOLVE_TOP_LEVEL_ONLY_DEFAULT = False


def prompt_path_or_browse(
    prompt: str,
    *,
    must_exist: bool = True,
    is_file: bool = False,
    default_folder: Path | None = None,
    default_file: Path | None = None,
) -> Path | None:
    """Prompt for a path; empty line opens a tkinter file/folder dialog."""
    import tkinter as tk
    from tkinter import filedialog

    line = input(f"{prompt}\n  Path (leave empty to browse): ").strip()
    if line:
        p = Path(line)
        if must_exist and not p.exists():
            raise FileNotFoundError(p)
        return p
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if is_file:
            probe = Path(default_file or DEFAULT_PROBE_FILE)
            init_dir, init_file = _tk_initialdir_and_file(probe)
            sel = filedialog.askopenfilename(
                title="Select file",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")],
                initialdir=init_dir,
                initialfile=init_file,
            )
        else:
            init_dir = _tk_initialdir_for_folder(Path(default_folder or DEFAULT_BROWSE_SORTING_OUTPUT_FOLDER))
            sel = filedialog.askdirectory(title="Select folder", initialdir=init_dir)
    finally:
        root.destroy()
    if not sel:
        return None
    return Path(sel)


def prompt_interactive_config():
    """Terminal prompts + optional browse dialogs. No argparse / CLI flags."""
    global USER_OUTPUT_FOLDER, USER_RECORDING_DIR, USER_PROBE_FILE, USER_FALLBACK_SG_TOTALCHAN, CURATION_STAGE
    global RESOLVE_TOP_LEVEL_ONLY_DEFAULT

    print("\n=== LSNET / EBL curation — interactive setup ===\n")
    print("Choose a configuration preset (sets default browse folders, SG map size, shank resolution):\n")
    for k in sorted(CURATION_PRESETS.keys(), key=lambda x: int(x) if x.isdigit() else x):
        p = CURATION_PRESETS[k]
        print(f"  {k} — {p.title}")
        print(f"      default output browse: {p.default_output_folder}")
        print(f"      default probe browse:    {p.default_probe_file}")
    print()
    preset_key = input("Choice [1]: ").strip() or "1"
    if preset_key not in CURATION_PRESETS:
        print(f"  Unknown preset {preset_key!r}; using 1.")
        preset_key = "1"
    preset = apply_preset_by_id(preset_key)
    print(f"Using preset {preset.key}: {preset.title}\n")

    print("Sorting output folder (contains sorted_sorting/ or shank_*/ subfolders for EBL512).")
    out = prompt_path_or_browse(
        "Output folder",
        must_exist=True,
        is_file=False,
        default_folder=DEFAULT_BROWSE_SORTING_OUTPUT_FOLDER,
    )
    if out is None:
        raise SystemExit("No output folder selected.")
    USER_OUTPUT_FOLDER = normalize_pipeline_output_folder(out.resolve())
    USER_RECORDING_DIR = infer_recording_dir_from_output(USER_OUTPUT_FOLDER)
    print(f"  Using recording dir (inferred): {USER_RECORDING_DIR}")

    print("\nProbe JSON (ProbeInterface).")
    pr = prompt_path_or_browse(
        "Probe file",
        must_exist=True,
        is_file=True,
        default_file=DEFAULT_PROBE_FILE,
    )
    if pr is None:
        raise SystemExit("No probe file selected.")
    USER_PROBE_FILE = pr.resolve()

    fb = input(
        f"\nUSER_FALLBACK_SG_TOTALCHAN (HW→SG map size) [{USER_FALLBACK_SG_TOTALCHAN}]: "
    ).strip()
    if fb:
        USER_FALLBACK_SG_TOTALCHAN = int(fb)

    stage_disp = "1" if CURATION_STAGE == "prepare_figures" else "2"
    st = input(
        "\nCuration stage:\n"
        "  1 = prepare_figures (label units)\n"
        "  2 = apply_curation (export curated)\n"
        f"  Enter 1 or 2 [{stage_disp}]: "
    ).strip()
    if st:
        CURATION_STAGE = normalize_curation_stage_token(st)

    print("\nConfiguration saved for this run.\n")


def resolve_output_folders_for_run() -> list[Path]:
    """
    If USER_OUTPUT_FOLDER contains EBL-style shank_* subfolders with sortings, optionally run each.
    """
    base_orig = Path(USER_OUTPUT_FOLDER).resolve()
    base = normalize_pipeline_output_folder(base_orig)
    if base != base_orig:
        print(
            f"[note] Using pipeline folder {base} (not analyzer subfolder {base_orig.name})"
        )
    if not base.is_dir():
        raise FileNotFoundError(f"Output folder not found: {base}")
    shanks = sorted(
        [p for p in base.glob("shank_*") if p.is_dir() and _has_sorting_artifacts(p)],
        key=lambda x: x.name,
    )
    top_has = _has_sorting_artifacts(base)
    if not shanks:
        return [base]
    if not top_has:
        print(
            f"\nDetected {len(shanks)} shank folder(s) with sorting artifacts "
            f"({', '.join(s.name for s in shanks)}); top-level folder has no sorting."
        )
        ans = input("Run curation on ALL shanks sequentially? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            return shanks
        pick = input(
            "Enter one shank id (0–3) to run a single shank, or 'q' to quit: "
        ).strip()
        if pick.lower() == "q":
            raise SystemExit(0)
        only = base / f"shank_{int(pick)}"
        if not only.is_dir() or not _has_sorting_artifacts(only):
            raise FileNotFoundError(f"No sorting at {only}")
        return [only]
    if RESOLVE_TOP_LEVEL_ONLY_DEFAULT:
        print(
            f"\nNote: both top-level sorting and {len(shanks)} shank_* folder(s) exist. "
            "Preset: using top-level only."
        )
        return [base]

    print(f"\nNote: both top-level sorting and {len(shanks)} shank_* folder(s) exist.")
    ans = input("[1] top-level only  [2] all shanks only  [3] top then all shanks: ").strip()
    if ans == "2":
        return shanks
    if ans == "3":
        return [base] + shanks
    return [base]
# -------------------------------------------------------


QUALITY_METRICS_PARAMS_BOTH = {
    "snr": {"peak_sign": "both"},
    "amplitude_median": {"peak_sign": "both"},
}


def find_latest_output_folder(base_data_folder: Path, duration_min: int) -> Path:
    pattern = f"ms5_d{int(duration_min)}min_*"
    candidates = sorted(
        [p for p in base_data_folder.glob(pattern) if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No output folder matching {pattern} under {base_data_folder}")
    return candidates[0]


def load_stim_ts(stim_path: Path, mat_var_name: str) -> np.ndarray:
    suffix = stim_path.suffix.lower()
    if suffix == ".npy":
        ts = np.load(stim_path)
    elif suffix == ".npz":
        d = np.load(stim_path)
        if len(d.files) == 0:
            raise ValueError(f"No arrays found in npz file: {stim_path}")
        ts = d[d.files[0]]
    elif suffix == ".mat":
        import scipy.io as sio

        mat = sio.loadmat(stim_path, squeeze_me=True)
        if mat_var_name not in mat:
            raise KeyError(f"MAT variable '{mat_var_name}' not found in {stim_path}")
        ts = np.asarray(mat[mat_var_name])
    else:
        raise ValueError(f"Unsupported stim timestamp file type: {stim_path}")
    return np.sort(np.asarray(ts).ravel().astype(np.int64))


def resolve_output_sorting_recording():
    if USER_OUTPUT_FOLDER is None:
        if USER_RECORDING_DIR is None:
            raise ValueError(
                "USER_OUTPUT_FOLDER is not set. Set LSNET_PIPELINE_OUTPUT_FOLDER or run interactively."
            )
        output_folder = find_latest_output_folder(USER_RECORDING_DIR, DURATION_MIN_SORTING)
    else:
        output_folder = normalize_pipeline_output_folder(Path(USER_OUTPUT_FOLDER))
    if not output_folder.exists():
        raise FileNotFoundError(f"Output folder not found: {output_folder}")

    saved_rec_folder = _resolve_saved_rec_binary_folder(output_folder)

    if not (saved_rec_folder / "binary.json").exists():
        raise FileNotFoundError(
            f"Saved recording not found (tried saved_rec/ and saved_recording/): {saved_rec_folder}"
        )

    sorted_sorting_folder = output_folder / "sorted_sorting"
    sorted_units_npz = output_folder / "sorted_units.npz"
    sorter_output_folder = output_folder / "sorter_output"
    sorting = None
    if sorted_sorting_folder.exists():
        sorting = si.load_extractor(sorted_sorting_folder)
    elif sorted_units_npz.exists():
        sorting = si.NpzSortingExtractor(sorted_units_npz)
    elif sorter_output_folder.exists():
        try:
            sorting = si.load_extractor(sorter_output_folder)
        except Exception:
            sorter_subfolder = sorter_output_folder / "sorting"
            if sorter_subfolder.exists():
                sorting = si.load_extractor(sorter_subfolder)
    if sorting is None:
        raise FileNotFoundError(
            "No sorting found. Expected one of: sorted_sorting/, sorted_units.npz, sorter_output/"
        )

    rec_good = load_saved_recording_with_probe(
        saved_rec_folder,
        USER_PROBE_FILE,
        fallback_sg_totalchan=USER_FALLBACK_SG_TOTALCHAN,
    )
    fs = float(rec_good.get_sampling_frequency())
    if abs(fs - 30000.0) > 1e-6:
        raise ValueError(f"Recording sampling frequency is {fs} Hz, expected 30000 Hz.")

    return output_folder, sorting, rec_good


def create_or_load_analyzer(output_folder: Path, sorting, recording):
    analyzer_folder = output_folder / "sorting_analyzer_before_curation.zarr"
    if analyzer_folder.exists():
        analyzer = si.load_sorting_analyzer(folder=analyzer_folder, format="zarr", load_extensions=True)
    else:
        analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording, format="memory")
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
        analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
        analyzer.compute("templates", operators=["average", "median", "std"])
        analyzer.compute("noise_levels")
        analyzer.compute("spike_amplitudes", peak_sign="both")
        analyzer.compute("quality_metrics", qm_params=QUALITY_METRICS_PARAMS_BOTH)
        analyzer.save_as(folder=analyzer_folder, format="zarr")

    if not analyzer.has_extension("random_spikes"):
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    if not analyzer.has_extension("waveforms"):
        analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
    if not analyzer.has_extension("templates"):
        analyzer.compute("templates", operators=["average", "median", "std"])
    if not analyzer.has_extension("noise_levels"):
        analyzer.compute("noise_levels")
    if not analyzer.has_extension("spike_amplitudes"):
        analyzer.compute("spike_amplitudes", peak_sign="both")
    if not analyzer.has_extension("quality_metrics"):
        analyzer.compute("quality_metrics", qm_params=QUALITY_METRICS_PARAMS_BOTH)
    return analyzer


def _get_quality_metrics_lookup(analyzer):
    qmetrics = {}
    if not analyzer.has_extension("quality_metrics"):
        return qmetrics
    qm_df = analyzer.get_extension("quality_metrics").get_data()
    if not isinstance(qm_df, pd.DataFrame):
        return qmetrics
    if "unit_id" not in qm_df.columns:
        qm_df = qm_df.reset_index()
        if "unit_id" not in qm_df.columns and "index" in qm_df.columns:
            qm_df = qm_df.rename(columns={"index": "unit_id"})
    if "unit_id" not in qm_df.columns:
        return qmetrics
    for row in qm_df.itertuples(index=False):
        try:
            uid = int(getattr(row, "unit_id"))
        except Exception:
            continue
        qmetrics[uid] = {
            "firing_rate": getattr(row, "firing_rate", np.nan),
            "snr": getattr(row, "snr", np.nan),
        }
    return qmetrics


def save_prepare_stage_unit_summaries(output_folder: Path, analyzer):
    unit_figure_folder = get_unit_curation_figure_folder(output_folder)
    mapping_csv = output_folder / "merged_unit_id_mapping.csv"
    if not mapping_csv.exists():
        # TODO_PLACEHOLDER: mapping_source_file
        # Need merged-unit channel metadata to label prepare-stage summaries consistently.
        raise FileNotFoundError(f"Required mapping file not found for prepare figures: {mapping_csv}")

    mapping_df = pd.read_csv(mapping_csv)
    required_cols = {"merged_unit_id", "hw_channel", "sg_channel", "shank_id", "local_channel_on_shank"}
    if not required_cols.issubset(set(mapping_df.columns)):
        # TODO_PLACEHOLDER: mapping_source_file
        # Missing mapping columns prevents deterministic summary naming in prepare stage.
        raise ValueError(
            f"Missing required columns in {mapping_csv}. "
            f"Required: {sorted(required_cols)}; found: {list(mapping_df.columns)}"
        )
    mapping_by_unit = {
        int(r.merged_unit_id): {
            "hw_channel": int(r.hw_channel),
            "sg_channel": int(r.sg_channel),
            "shank_id": int(r.shank_id),
            "local_channel_on_shank": int(r.local_channel_on_shank),
        }
        for r in mapping_df.itertuples(index=False)
    }
    qmetrics_by_unit = _get_quality_metrics_lookup(analyzer)

    # Regenerate root summary PNGs so labels and overlays are consistent for current analyzer.
    for p in unit_figure_folder.glob("unit_summary_*.png"):
        p.unlink()

    for unit_id in analyzer.sorting.get_unit_ids():
        uid = int(unit_id)
        if uid not in mapping_by_unit:
            # TODO_PLACEHOLDER: mapping_source_file
            # Without merged metadata row, prepare-stage summary naming would be ambiguous.
            raise ValueError(f"No merged-unit mapping found for prepare-stage unit_id={uid}")
        meta = mapping_by_unit[uid]
        qm = qmetrics_by_unit.get(uid, {"firing_rate": np.nan, "snr": np.nan})
        firing_rate = qm.get("firing_rate", np.nan)
        snr = qm.get("snr", np.nan)
        fr_text = f"{float(firing_rate):.3f}" if pd.notna(firing_rate) else "nan"
        snr_text = f"{float(snr):.3f}" if pd.notna(snr) else "nan"

        sw.plot_unit_summary(analyzer, unit_id=unit_id)
        fig = plt.gcf()
        fig.suptitle(
            f"Unit {uid} | shank {meta['shank_id']}, ch {meta['local_channel_on_shank']}, "
            f"sg {meta['sg_channel']}, hw {meta['hw_channel']} | FR={fr_text} Hz, SNR={snr_text}",
            fontsize=11,
        )
        fig_name = (
            f"unit_summary_shank{meta['shank_id']}_"
            f"ch{meta['local_channel_on_shank']}_"
            f"sg{meta['sg_channel']}_"
            f"{uid}.png"
        )
        plt.savefig(unit_figure_folder / fig_name, dpi=300)
        plt.close()

    print(f"Saved pre-curation unit summaries with FR/SNR: {unit_figure_folder}")


def save_unit_figures(output_folder: Path, analyzer):
    unit_figure_folder = output_folder / "unit_figure_wfcurated_for_unit_curation"
    unit_figure_folder.mkdir(parents=True, exist_ok=True)
    (unit_figure_folder / "good_units").mkdir(exist_ok=True)
    (unit_figure_folder / "bad_units").mkdir(exist_ok=True)
    for i in range(5):
        (unit_figure_folder / f"merge{i}").mkdir(exist_ok=True)

    analyzer_channel_ids = np.array(analyzer.channel_ids)
    for unit_id in analyzer.sorting.get_unit_ids():
        waveforms = analyzer.get_extension("waveforms").get_waveforms_one_unit(unit_id)
        if waveforms.size == 0:
            continue

        # Global channel index (dense waveforms), then map to global channel ID.
        channel_idx = int(np.argmax(np.max(np.abs(waveforms.mean(axis=0)), axis=0)))
        channel_id_global = analyzer_channel_ids[channel_idx]
        try:
            channel_trodes = int(channel_id_global) + 1
        except Exception:
            channel_trodes = channel_idx + 1

        waveforms_single = waveforms[:, :, channel_idx]
        avg_wave = np.mean(waveforms_single, axis=0)
        time_axis = np.linspace(-1, 2, len(avg_wave))

        plt.figure(figsize=(4, 6))
        for wf in waveforms_single:
            plt.plot(time_axis, wf, color="gray", alpha=0.05)
        plt.plot(time_axis, avg_wave, color="red", linewidth=2)
        plt.title(f"Unit {unit_id} - Channel {channel_trodes}")
        plt.ylim(-470, 300)

        fig_path = unit_figure_folder / f"Unit_{unit_id}_TrodesChannel_{channel_trodes}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()

    placeholder_note = unit_figure_folder / "TODO_shank_channel_labels.txt"
    with open(placeholder_note, "w", encoding="utf-8") as f:
        f.write(
            "TODO: print (shank 0-31, channel 0-11) on saved figures.\n"
            "Placeholder implementation point: after channel_trodes is determined, map channel->(shank, local_ch)\n"
            "from probe geometry and append '(Shank s, Ch c)' to figure title/filename.\n"
        )
    return unit_figure_folder


def bootstrap_unit_figures_from_raw_units(output_folder: Path):
    shank_raw_units_folders = sorted(
        [p for p in output_folder.glob("shank_*/raw_units") if p.is_dir()],
        key=lambda p: p.as_posix(),
    )
    if not shank_raw_units_folders:
        # Backward-compatibility fallback for older single-folder outputs.
        raw_units_folder = output_folder / "raw_units"
        if raw_units_folder.exists():
            shank_raw_units_folders = [raw_units_folder]
    if not shank_raw_units_folders:
        return None

    unit_figure_folder = output_folder / "unit_figure_wfcurated_for_unit_curation"
    unit_figure_folder.mkdir(parents=True, exist_ok=True)
    (unit_figure_folder / "good_units").mkdir(exist_ok=True)
    (unit_figure_folder / "bad_units").mkdir(exist_ok=True)
    for i in range(5):
        (unit_figure_folder / f"merge{i}").mkdir(exist_ok=True)

    copied = 0
    collisions = 0
    for raw_units_folder in shank_raw_units_folders:
        for src in raw_units_folder.glob("unit_summary_*.png"):
            m = re.search(r"unit_summary_(\d+)", src.name)
            if not m:
                continue
            uid = int(m.group(1))
            dst = unit_figure_folder / f"Unit_{uid}_TrodesChannel_0.png"
            if dst.exists():
                collisions += 1
                continue
            shutil.copy2(src, dst)
            copied += 1
    if copied > 0:
        print(
            f"Bootstrapped {copied} unit figures from {len(shank_raw_units_folders)} raw_units folder(s) into curation folder."
        )
    if collisions > 0:
        # TODO_PLACEHOLDER: merge_unit_id_namespace
        # Multiple shanks can reuse local unit ids; collisions are skipped to avoid overwrite.
        print(f"[warning] Skipped {collisions} unit figure collisions while bootstrapping from shank raw_units.")
    return unit_figure_folder


def load_labels(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"labelsByUnit": {}}


def save_labels(labels: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def get_unit_curation_figure_folder(output_folder: Path) -> Path:
    unit_figure_folder = output_folder / "unit_summary_before_curation"
    unit_figure_folder.mkdir(parents=True, exist_ok=True)
    (unit_figure_folder / "good_units").mkdir(exist_ok=True)
    (unit_figure_folder / "bad_units").mkdir(exist_ok=True)
    for i in range(5):
        (unit_figure_folder / f"merge{i}").mkdir(exist_ok=True)
    return unit_figure_folder


def sync_legacy_unit_figure_folder(output_folder: Path, source_folder: Path):
    """
    Keep backward compatibility with run_unit_level_curation_from_pngs(), which
    still reads from output_folder/unit_figure_wfcurated_for_unit_curation.
    """
    legacy_folder = output_folder / "unit_figure_wfcurated_for_unit_curation"
    legacy_folder.mkdir(parents=True, exist_ok=True)
    (legacy_folder / "good_units").mkdir(exist_ok=True)
    (legacy_folder / "bad_units").mkdir(exist_ok=True)
    for i in range(5):
        (legacy_folder / f"merge{i}").mkdir(exist_ok=True)

    # Clear previous mirrored png files.
    for sub in [legacy_folder, legacy_folder / "good_units", legacy_folder / "bad_units"]:
        for p in sub.glob("*.png"):
            p.unlink()
    for i in range(5):
        for p in (legacy_folder / f"merge{i}").glob("*.png"):
            p.unlink()

    def _legacy_name_for_png(p: Path):
        uid = parse_unit_id_from_png_name(p.name)
        if uid is None:
            return None
        # Keep legacy naming expected by run_unit_level_curation_from_pngs():
        # Unit_<uid>_TrodesChannel_<ch>.png
        m = re.search(r"TrodesChannel_(\d+)", p.name)
        trodes_ch = int(m.group(1)) if m else 0
        return f"Unit_{uid}_TrodesChannel_{trodes_ch}.png"

    # Mirror root and label folders using legacy-compatible names.
    for p in source_folder.glob("*.png"):
        legacy_name = _legacy_name_for_png(p)
        if legacy_name is not None:
            shutil.copy2(p, legacy_folder / legacy_name)
    for sub_name in ["good_units", "bad_units"]:
        src_sub = source_folder / sub_name
        dst_sub = legacy_folder / sub_name
        if src_sub.exists():
            for p in src_sub.glob("*.png"):
                legacy_name = _legacy_name_for_png(p)
                if legacy_name is not None:
                    shutil.copy2(p, dst_sub / legacy_name)
    for i in range(5):
        src_sub = source_folder / f"merge{i}"
        dst_sub = legacy_folder / f"merge{i}"
        if src_sub.exists():
            for p in src_sub.glob("*.png"):
                legacy_name = _legacy_name_for_png(p)
                if legacy_name is not None:
                    shutil.copy2(p, dst_sub / legacy_name)

    print(f"Synchronized legacy curation folder: {legacy_folder}")
    return legacy_folder


def parse_unit_id_from_png_name(name: str):
    m = re.search(r"unit_summary_shank(\d+)_ch(\d+)_sg(\d+)_(\d+)$", Path(name).stem)
    if m:
        return int(m.group(4))
    m = re.search(r"unit_summary_(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"Unit_(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def parse_shank_channel_from_png_name(name: str):
    """
    Parse (shank, channel) metadata from known png naming conventions.
    Returns (shank, ch) where each can be None if unavailable.
    """
    stem = Path(name).stem
    m = re.search(r"unit_summary_shank(\d+)_ch(\d+)_sg(\d+)_\d+$", stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"shank(\d+)_ch(\d+)", stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def collect_root_unit_pngs(unit_figure_folder: Path):
    unit_files = []
    for p in sorted(unit_figure_folder.glob("*.png")):
        uid = parse_unit_id_from_png_name(p.name)
        if uid is not None:
            unit_files.append((uid, p))
    return unit_files


def _build_prepare_ui_metadata(output_folder: Path, sorting, rec_good):
    metadata_by_unit = {}

    mapping_csv = output_folder / "merged_unit_id_mapping.csv"
    if mapping_csv.exists():
        try:
            mapping_df = pd.read_csv(mapping_csv)
            for r in mapping_df.itertuples(index=False):
                uid = int(getattr(r, "merged_unit_id"))
                metadata_by_unit[uid] = {
                    "shank_id": getattr(r, "shank_id", np.nan),
                    "local_channel_on_shank": getattr(r, "local_channel_on_shank", np.nan),
                    "sg_channel": getattr(r, "sg_channel", np.nan),
                    "hw_channel": getattr(r, "hw_channel", np.nan),
                    "firing_rate": np.nan,
                    "snr": np.nan,
                }
        except Exception as exc:
            print(f"[warning] Could not load prepare UI mapping metadata from {mapping_csv}: {exc}")

    merged_metrics_csv = output_folder / "merged_unit_metrics_for_match.csv"
    if merged_metrics_csv.exists():
        try:
            metrics_df = pd.read_csv(merged_metrics_csv)
            for r in metrics_df.itertuples(index=False):
                uid = int(getattr(r, "unit_id"))
                row = metadata_by_unit.setdefault(
                    uid,
                    {
                        "shank_id": np.nan,
                        "local_channel_on_shank": np.nan,
                        "sg_channel": np.nan,
                        "hw_channel": np.nan,
                        "firing_rate": np.nan,
                        "snr": np.nan,
                    },
                )
                row["firing_rate"] = getattr(r, "firing_rate", np.nan)
                row["snr"] = getattr(r, "snr", np.nan)
        except Exception as exc:
            print(f"[warning] Could not load prepare UI metrics from {merged_metrics_csv}: {exc}")

    # Fallback: compute FR/SNR from analyzer only if metrics file missing FR/SNR.
    needs_qm_fallback = any(
        pd.isna(v.get("firing_rate", np.nan)) or pd.isna(v.get("snr", np.nan))
        for v in metadata_by_unit.values()
    ) or len(metadata_by_unit) == 0
    if needs_qm_fallback:
        try:
            analyzer = create_or_load_analyzer(output_folder, sorting, rec_good)
            qm_lookup = _get_quality_metrics_lookup(analyzer)
            for uid, qm in qm_lookup.items():
                row = metadata_by_unit.setdefault(
                    int(uid),
                    {
                        "shank_id": np.nan,
                        "local_channel_on_shank": np.nan,
                        "sg_channel": np.nan,
                        "hw_channel": np.nan,
                        "firing_rate": np.nan,
                        "snr": np.nan,
                    },
                )
                if pd.isna(row.get("firing_rate", np.nan)):
                    row["firing_rate"] = qm.get("firing_rate", np.nan)
                if pd.isna(row.get("snr", np.nan)):
                    row["snr"] = qm.get("snr", np.nan)
        except Exception as exc:
            print(f"[warning] Could not load analyzer fallback for prepare UI FR/SNR: {exc}")

    return metadata_by_unit


def label_units_with_buttons(unit_figure_folder: Path, labels_json: Path, unit_metadata_by_id: dict | None = None):
    labels = load_labels(labels_json)
    labels_by_unit = labels.setdefault("labelsByUnit", {})
    units = collect_root_unit_pngs(unit_figure_folder)
    todo = [(uid, p) for uid, p in units if str(uid) not in labels_by_unit]

    if not todo:
        print("All unit PNGs already labeled in curation_labels.json.")
        return labels

    merge1_folder = unit_figure_folder / "merge1"
    merge2_folder = unit_figure_folder / "merge2"
    merge1_folder.mkdir(exist_ok=True)
    merge2_folder.mkdir(exist_ok=True)

    # Remove explicit MUA button and make MUA default on Skip.
    default_skip_label = "mua"
    state = {"idx": 0, "merge1": False, "merge2": False}
    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.16)

    def show_current():
        ax.clear()
        uid, img_path = todo[state["idx"]]
        shank, ch = parse_shank_channel_from_png_name(img_path.name)
        shank_ch_text = f" (shank={shank}, ch={ch})" if shank is not None and ch is not None else ""
        meta = unit_metadata_by_id.get(int(uid), {}) if isinstance(unit_metadata_by_id, dict) else {}
        sg = meta.get("sg_channel", np.nan)
        hw = meta.get("hw_channel", np.nan)
        fr = meta.get("firing_rate", np.nan)
        snr = meta.get("snr", np.nan)
        sg_text = "nan" if pd.isna(sg) else f"{int(sg)}"
        hw_text = "nan" if pd.isna(hw) else f"{int(hw)}"
        fr_text = "nan" if pd.isna(fr) else f"{float(fr):.3f}"
        snr_text = "nan" if pd.isna(snr) else f"{float(snr):.3f}"
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(
            f"Unit {uid}{shank_ch_text} | sg={sg_text}, hw={hw_text}, "
            f"FR={fr_text} Hz, SNR={snr_text} ({state['idx'] + 1}/{len(todo)})",
            fontsize=13,
        )
        fig.canvas.draw_idle()

    def maybe_copy_to_merge_folders(img_path: Path):
        if state.get("merge1", False):
            shutil.copy2(img_path, merge1_folder / img_path.name)
        if state.get("merge2", False):
            shutil.copy2(img_path, merge2_folder / img_path.name)

    def make_callback(label_value):
        def callback(event):
            uid, img_path = todo[state["idx"]]
            labels_by_unit[str(uid)] = label_value
            maybe_copy_to_merge_folders(img_path)
            save_labels(labels, labels_json)
            state["idx"] += 1
            if state["idx"] >= len(todo):
                print("Labeling complete.")
                plt.close(fig)
            else:
                show_current()
        return callback

    def on_back(event):
        if state["idx"] > 0:
            state["idx"] -= 1
            show_current()

    def on_skip(event):
        uid, img_path = todo[state["idx"]]
        labels_by_unit[str(uid)] = default_skip_label
        maybe_copy_to_merge_folders(img_path)
        save_labels(labels, labels_json)
        state["idx"] += 1
        if state["idx"] >= len(todo):
            print(f"Reached end (remaining units labeled as default='{default_skip_label}').")
            plt.close(fig)
        else:
            show_current()

    def on_merge_toggle(label):
        if label == "Merge1":
            state["merge1"] = not state["merge1"]
        elif label == "Merge2":
            state["merge2"] = not state["merge2"]

    btn_specs = [
        (0.15, "Good", "lightgreen", make_callback("good")),
        (0.38, "Noise", "lightsalmon", make_callback("noise")),
        (0.72, "Back", "lightgray", on_back),
        (0.85, "Skip=MUA", "whitesmoke", on_skip),
    ]
    buttons = []
    for x, text, color, cb in btn_specs:
        ax_btn = fig.add_axes([x, 0.03, 0.13, 0.05])
        btn = Button(ax_btn, text, color=color, hovercolor="lightskyblue")
        btn.on_clicked(cb)
        buttons.append(btn)

    ax_chk = fig.add_axes([0.02, 0.02, 0.11, 0.09])
    merge_checks = CheckButtons(ax_chk, ["Merge1", "Merge2"], [False, False])
    merge_checks.on_clicked(on_merge_toggle)
    buttons.append(merge_checks)

    show_current()
    plt.show()
    return labels


def sync_label_folders_from_json(unit_figure_folder: Path, labels_json: Path):
    labels = load_labels(labels_json)
    labels_by_unit = labels.get("labelsByUnit", {})
    good_units_folder = unit_figure_folder / "good_units"
    bad_units_folder = unit_figure_folder / "bad_units"
    good_units_folder.mkdir(exist_ok=True)
    bad_units_folder.mkdir(exist_ok=True)

    # If no labels are present, keep any manually curated folder organization.
    if len(labels_by_unit) == 0:
        print(
            f"No labels found in {labels_json}; preserving existing "
            f"{good_units_folder.name}/{bad_units_folder.name} contents."
        )
        return

    for folder in (good_units_folder, bad_units_folder):
        for p in folder.glob("*.png"):
            p.unlink()

    for uid, p in collect_root_unit_pngs(unit_figure_folder):
        label = str(labels_by_unit.get(str(uid), "")).lower()
        if label == "good":
            shutil.copy2(p, good_units_folder / p.name)
        elif label in {"mua", "noise"}:
            # Keep root folder complete while synchronizing label folders deterministically.
            shutil.copy2(p, bad_units_folder / p.name)

    print(f"Synchronized label folders from {labels_json}")


def sync_labels_json_from_folders(unit_figure_folder: Path, labels_json: Path):
    """
    Apply-stage source of truth: infer labels from good_units/bad_units folder PNGs.
    - good_units -> "good"
    - bad_units -> "noise"
    """
    good_units_folder = unit_figure_folder / "good_units"
    bad_units_folder = unit_figure_folder / "bad_units"
    good_units_folder.mkdir(exist_ok=True)
    bad_units_folder.mkdir(exist_ok=True)

    labels_by_unit = {}
    conflicts = []

    for p in sorted(good_units_folder.glob("*.png")):
        uid = parse_unit_id_from_png_name(p.name)
        if uid is None:
            continue
        labels_by_unit[str(int(uid))] = "good"

    for p in sorted(bad_units_folder.glob("*.png")):
        uid = parse_unit_id_from_png_name(p.name)
        if uid is None:
            continue
        key = str(int(uid))
        if key in labels_by_unit and labels_by_unit[key] != "noise":
            conflicts.append(int(uid))
        labels_by_unit[key] = "noise"

    if conflicts:
        # Deterministic conflict policy for manual folder edits:
        # if a unit appears in both folders, bad/noise wins to avoid false accepts.
        print(
            f"[warning] {len(conflicts)} unit(s) found in both good_units and bad_units; "
            f"using bad_units label. Examples: {conflicts[:10]}"
        )

    payload = {"labelsByUnit": labels_by_unit}
    save_labels(payload, labels_json)
    print(
        f"Synchronized {labels_json} from folders: "
        f"good={sum(1 for v in labels_by_unit.values() if v == 'good')}, "
        f"bad={sum(1 for v in labels_by_unit.values() if v != 'good')}"
    )
    return payload


def build_curation_json_from_labels_and_merges(output_folder: Path):
    unit_figure_folder = get_unit_curation_figure_folder(output_folder)
    labels_json = output_folder / "curation_labels.json"
    curation_json = output_folder / "curation.json"

    labels = load_labels(labels_json)
    labels_by_unit_raw = labels.get("labelsByUnit", {})
    # Include units from root + label folders so moved rejected units are retained.
    all_units = [uid for uid, _ in collect_root_unit_pngs(unit_figure_folder)]
    for sub in ("good_units", "bad_units"):
        sub_folder = unit_figure_folder / sub
        if sub_folder.exists():
            for p in sorted(sub_folder.glob("*.png")):
                uid = parse_unit_id_from_png_name(p.name)
                if uid is not None:
                    all_units.append(uid)
    labelsByUnit = {}
    for uid in sorted(set(all_units)):
        v = str(labels_by_unit_raw.get(str(uid), "")).lower()
        labelsByUnit[str(uid)] = ["accept"] if v == "good" else ["reject"]

    mergeGroups = []
    for sub in sorted(unit_figure_folder.iterdir()):
        if sub.is_dir() and sub.name.lower().startswith("merge"):
            merge_units_by_channel = {}
            skipped_no_channel = []
            for f in sorted(sub.glob("*.png")):
                uid = parse_unit_id_from_png_name(f.name)
                if uid is None:
                    continue
                shank, ch = parse_shank_channel_from_png_name(f.name)
                if shank is None or ch is None:
                    skipped_no_channel.append((uid, f.name))
                    continue
                key = (int(shank), int(ch))
                merge_units_by_channel.setdefault(key, set()).add(int(uid))

            for key in sorted(merge_units_by_channel.keys()):
                uniq = sorted(merge_units_by_channel[key])
                if len(uniq) > 1:
                    mergeGroups.append(uniq)

            if len(skipped_no_channel) > 0:
                # TODO_PLACEHOLDER: merge_policy_cross_channel
                # Channel metadata is required to enforce same-channel-only merge groups.
                preview = ", ".join([f"{uid}:{name}" for uid, name in skipped_no_channel[:5]])
                print(
                    f"[warning] Skipped {len(skipped_no_channel)} merge candidate(s) in {sub.name} "
                    f"due to missing channel metadata. Examples [{preview}]"
                )

    curation_data = {
        "labelsByUnit": labelsByUnit,
        "mergeGroups": mergeGroups,
    }
    with open(curation_json, "w", encoding="utf-8") as f:
        json.dump(curation_data, f, indent=2)
    print(f"Wrote translated curation JSON: {curation_json}")
    return curation_json


def write_stage1_prompt(output_folder: Path):
    unit_figure_folder = get_unit_curation_figure_folder(output_folder)
    msg = (
        "Stage 1 complete.\n"
        f"Please organize unit figure folders under: {unit_figure_folder}\n"
        f"Label file: {output_folder / 'curation_labels.json'}\n"
        "  - good_units/: accepted units\n"
        "  - bad_units/: rejected units (optional)\n"
        "  - merge0..merge4/: units to merge (optional)\n"
        "    Merge rule: only units on the same (shank, channel) are merged.\n"
        "    You can place mixed channels in one merge folder; the script splits them by channel.\n"
        "    PNG names must include channel metadata (e.g., unit_summary_shank*_ch*_*). Otherwise skipped.\n"
        "Then set CURATION_STAGE='apply_curation' and rerun.\n"
    )
    with open(output_folder / "CURATION_NEXT_STEPS.txt", "w", encoding="utf-8") as f:
        f.write(msg)
    print(msg)


def apply_curation_and_save_outputs(
    output_folder: Path,
    sorting,
    rec_good,
    plot_figures: bool = True,
    plot_waveform_figures: bool = True,
    verbose_plot_logs: bool = True,
):
    curation_json = output_folder / "curation.json"
    if not curation_json.exists():
        raise FileNotFoundError(f"Curation json not found: {curation_json}")

    accepted_sorting = scur.apply_sortingview_curation(
        sorting,
        uri_or_json=curation_json,
        include_labels=["accept"],
        skip_merge=False,
    )

    curated_analyzer_folder = output_folder / "CuratedAnalyzer2"
    curated_analyzer_zarr = output_folder / "Curated_analyzer_full.zarr"
    if curated_analyzer_zarr.exists():
        curated_analyzer = si.load_sorting_analyzer(
            folder=curated_analyzer_zarr,
            format="zarr",
            load_extensions=True,
        )
        print(f"Loaded existing curated analyzer: {curated_analyzer_zarr}")
    else:
        curated_analyzer = si.create_sorting_analyzer(
            sorting=accepted_sorting,
            recording=rec_good,
            format="binary_folder",
            folder=str(curated_analyzer_folder),
            overwrite=True,
        )
        print(f"Created curated analyzer at: {curated_analyzer_folder}")

    # Ensure required extensions exist whether analyzer was loaded or newly created.
    if not curated_analyzer.has_extension("random_spikes"):
        curated_analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    if not curated_analyzer.has_extension("waveforms"):
        curated_analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
    if not curated_analyzer.has_extension("templates"):
        curated_analyzer.compute("templates", operators=["average", "median", "std"])
    if not curated_analyzer.has_extension("noise_levels"):
        curated_analyzer.compute("noise_levels")
    if not curated_analyzer.has_extension("spike_amplitudes"):
        curated_analyzer.compute("spike_amplitudes", peak_sign="both")
    if not curated_analyzer.has_extension("unit_locations"):
        curated_analyzer.compute("unit_locations", method="monopolar_triangulation")
    if not curated_analyzer.has_extension("correlograms"):
        curated_analyzer.compute("correlograms", window_ms=50.0, bin_ms=1.0, method="auto")
    if not curated_analyzer.has_extension("isi_histograms"):
        curated_analyzer.compute("isi_histograms", window_ms=50.0, bin_ms=1.0, method="auto")
    if not curated_analyzer.has_extension("quality_metrics"):
        curated_analyzer.compute("quality_metrics", qm_params=QUALITY_METRICS_PARAMS_BOTH)
    if not curated_analyzer.has_extension("template_similarity"):
        curated_analyzer.compute("template_similarity", method="cosine_similarity")

    if not curated_analyzer_zarr.exists():
        curated_analyzer.save_as(folder=curated_analyzer_zarr, format="zarr")
        print(f"Saved curated analyzer zarr: {curated_analyzer_zarr}")

    spk_vec = accepted_sorting.to_spike_vector()
    channel_ids = rec_good.get_channel_ids()
    templates = curated_analyzer.get_extension("templates").get_data()
    best_channel_index_per_unit = templates.min(axis=1).argmin(axis=1).astype(np.int32)
    best_channel_id_per_unit = np.array([channel_ids[i] for i in best_channel_index_per_unit])

    new_dtype = spk_vec.dtype.descr + [("channel_id", "<i8")]
    spk_vec_full = np.zeros(spk_vec.shape, dtype=new_dtype)
    for name in spk_vec.dtype.names:
        spk_vec_full[name] = spk_vec[name]
    spk_vec_full["channel_id"] = best_channel_id_per_unit[spk_vec["unit_index"]]

    spk_vec_save_path = output_folder / "SpkVec_full.npy"
    np.save(spk_vec_save_path, spk_vec_full)
    print(f"Saved spike vector: {spk_vec_save_path}")

    if PLOT_POSTCURATION_FIG4_POPULATION_METRICS and (plot_figures or plot_waveform_figures):
        try:
            from lsnet_curation_unit_summary_fig4 import save_postcuration_fig4_unit_metric_plots

            save_postcuration_fig4_unit_metric_plots(
                output_folder,
                accepted_sorting,
                rec_good,
                curated_analyzer,
            )
        except Exception as exc:
            print(f"[warning] Fig4-style population unit metric plots failed: {exc}")

    if PLOT_POSTCURATION_ALL_UNITS_GRID_WAVEFORM_ISI and (plot_figures or plot_waveform_figures):
        try:
            from lsnet_curation_all_units_grid import save_postcuration_all_units_grid

            save_postcuration_all_units_grid(
                output_folder,
                accepted_sorting,
                rec_good,
                curated_analyzer,
                rec_path=infer_recording_dir_from_output(output_folder),
                ms_before=1.0,
                ms_after=2.0,
                ordered_channel_ids=None,
                best_channel_id_per_unit=best_channel_id_per_unit,
            )
        except Exception as exc:
            print(f"[warning] ALL_units_grid_waveformISI plot failed: {exc}")

    if plot_figures or plot_waveform_figures:
        save_fig_folder = output_folder / "unit_summary_curated"
        save_fig_folder.mkdir(parents=True, exist_ok=True)
        mapping_csv = output_folder / "merged_unit_id_mapping.csv"
        if not mapping_csv.exists():
            # TODO_PLACEHOLDER: mapping_source_file
            # Need merged-unit channel metadata to label curated unit summaries consistently.
            raise FileNotFoundError(f"Required mapping file not found for curated summaries: {mapping_csv}")
        mapping_df = pd.read_csv(mapping_csv)
        required_cols = {"merged_unit_id", "hw_channel", "sg_channel", "shank_id", "local_channel_on_shank"}
        if not required_cols.issubset(set(mapping_df.columns)):
            # TODO_PLACEHOLDER: mapping_source_file
            # Missing required mapping columns prevents deterministic curated summary naming.
            raise ValueError(
                f"Missing required columns in {mapping_csv}. "
                f"Required: {sorted(required_cols)}; found: {list(mapping_df.columns)}"
            )
        mapping_by_unit = {
            int(r.merged_unit_id): {
                "hw_channel": int(r.hw_channel),
                "sg_channel": int(r.sg_channel),
                "shank_id": int(r.shank_id),
                "local_channel_on_shank": int(r.local_channel_on_shank),
            }
            for r in mapping_df.itertuples(index=False)
        }
        # Fill in mapping for accepted units missing from CSV (e.g. merged units) using same inference as post-curation export.
        accepted_unit_ids_plot = np.asarray(accepted_sorting.get_unit_ids(), dtype=np.int64)
        mapped_unit_ids_plot = set(mapping_by_unit.keys())
        missing_units_plot = sorted(set(accepted_unit_ids_plot.tolist()) - mapped_unit_ids_plot)
        if missing_units_plot and len(best_channel_id_per_unit) == len(accepted_unit_ids_plot):
            analyzer_hw_by_unit_plot = {
                int(uid): int(ch)
                for uid, ch in zip(
                    accepted_unit_ids_plot.tolist(),
                    best_channel_id_per_unit.astype(np.int64).tolist(),
                )
            }
            hw_meta_cols = ["sg_channel", "shank_id", "local_channel_on_shank"]
            hw_meta_lookup_plot = {}
            for hw, grp in mapping_df.groupby("hw_channel"):
                uniq = grp[hw_meta_cols].drop_duplicates()
                if len(uniq) == 1:
                    row = uniq.iloc[0]
                    hw_meta_lookup_plot[int(hw)] = {
                        "sg_channel": int(row["sg_channel"]),
                        "shank_id": int(row["shank_id"]),
                        "local_channel_on_shank": int(row["local_channel_on_shank"]),
                    }
            for uid in missing_units_plot:
                hw = analyzer_hw_by_unit_plot.get(int(uid), None)
                if hw is None:
                    continue
                meta = hw_meta_lookup_plot.get(int(hw), None)
                if meta is None:
                    continue
                mapping_by_unit[int(uid)] = {
                    "hw_channel": int(hw),
                    "sg_channel": int(meta["sg_channel"]),
                    "shank_id": int(meta["shank_id"]),
                    "local_channel_on_shank": int(meta["local_channel_on_shank"]),
                }
            n_inferred = sum(1 for u in missing_units_plot if int(u) in mapping_by_unit)
            if n_inferred > 0:
                print(
                    f"[curated_plot] Inferred {n_inferred} missing merged-unit mapping(s) for plot export."
                )
        for unit_id in accepted_sorting.get_unit_ids():
            uid = int(unit_id)
            if uid not in mapping_by_unit:
                # TODO_PLACEHOLDER: mapping_source_file
                # Missing merged-unit metadata for this accepted unit; skip unit-level plots for this unit.
                print(f"[warning] No merged-unit mapping found for curated unit_id={uid}; skipping plot export for this unit.")
                continue
            meta = mapping_by_unit[uid]
            unit_t0 = pd.Timestamp.now()
            summary_saved_t = unit_t0
            if plot_figures:
                sw.plot_unit_summary(curated_analyzer, unit_id=unit_id)
                plot_ready_t = pd.Timestamp.now()
                fig = plt.gcf()
                fig.suptitle(
                    f"Unit {uid} | shank {meta['shank_id']}, ch {meta['local_channel_on_shank']}, "
                    f"sg {meta['sg_channel']}, hw {meta['hw_channel']}",
                    fontsize=11,
                )
                fig_name = (
                    f"unit_summary_shank{meta['shank_id']}_"
                    f"ch{meta['local_channel_on_shank']}_"
                    f"sg{meta['sg_channel']}_"
                    f"hw{meta['hw_channel']}_"
                    f"{uid}.png"
                )
                plt.savefig(save_fig_folder / fig_name)
                summary_saved_t = pd.Timestamp.now()
                plt.close()
                if verbose_plot_logs:
                    print(
                        f"[curated_plot] unit={uid} summary done | "
                        f"plot={(plot_ready_t - unit_t0).total_seconds():.2f}s | "
                        f"save={(summary_saved_t - plot_ready_t).total_seconds():.2f}s"
                    )

            if plot_waveform_figures:
                # Also save per-spike waveform overlays on best channel for quick visual QC.
                waveforms = curated_analyzer.get_extension("waveforms").get_waveforms_one_unit(unit_id)
                if waveforms.size == 0:
                    if verbose_plot_logs:
                        print(f"[curated_plot] unit={uid} waveform skipped (empty waveforms).")
                    continue
                channel_ids_cur = np.array(curated_analyzer.channel_ids)
                channel_idx = int(np.argmax(np.max(np.abs(waveforms.mean(axis=0)), axis=0)))
                waveforms_single = waveforms[:, :, channel_idx]
                avg_wave = np.mean(waveforms_single, axis=0)
                time_axis = np.linspace(-1.0, 2.0, len(avg_wave))

                plt.figure(figsize=(4, 6))
                for wf in waveforms_single:
                    plt.plot(time_axis, wf, color="gray", alpha=0.05)
                plt.plot(time_axis, avg_wave, color="red", linewidth=2)
                try:
                    trodes_channel = int(channel_ids_cur[channel_idx]) + 1
                except Exception:
                    trodes_channel = channel_idx + 1
                plt.title(f"Unit {uid} - Channel {trodes_channel}")
                plt.ylim(-470, 300)
                wf_name = (
                    f"unit_summary_shank{meta['shank_id']}_"
                    f"ch{meta['local_channel_on_shank']}_"
                    f"sg{meta['sg_channel']}_"
                    f"hw{meta['hw_channel']}_"
                    f"{uid}_wf.png"
                )
                plt.savefig(save_fig_folder / wf_name, dpi=300)
                wf_saved_t = pd.Timestamp.now()
                plt.close()
                if verbose_plot_logs:
                    print(
                        f"[curated_plot] unit={uid} waveform done | "
                        f"save={(wf_saved_t - summary_saved_t).total_seconds():.2f}s | "
                        f"total={(wf_saved_t - unit_t0).total_seconds():.2f}s"
                    )
            elif verbose_plot_logs:
                print(f"[curated_plot] unit={uid} waveform plotting skipped by config.")
        if plot_figures:
            print(f"Saved curated unit summaries: {save_fig_folder}")
        if plot_waveform_figures:
            print(f"Saved curated unit waveform overlays: {save_fig_folder}")

    return accepted_sorting, curated_analyzer, spk_vec_full, best_channel_id_per_unit


def save_postcuration_channel_matrix_and_exports(
    output_folder: Path,
    accepted_sorting,
    rec_good,
    best_channel_id_per_unit: np.ndarray,
    debug_trace_mapping: bool = False,
):
    accepted_unit_ids = np.asarray(accepted_sorting.get_unit_ids(), dtype=np.int64)
    mapping_csv = output_folder / "merged_unit_id_mapping.csv"
    if not mapping_csv.exists():
        # TODO_PLACEHOLDER: mapping_source_file
        # merged_unit_id_mapping.csv is required to export hw/sg/(shank,ch) consistently.
        raise FileNotFoundError(f"Required mapping file not found: {mapping_csv}")

    mapping_df = pd.read_csv(mapping_csv)
    required_cols = {
        "merged_unit_id",
        "hw_channel",
        "sg_channel",
        "shank_id",
        "local_channel_on_shank",
    }
    if not required_cols.issubset(set(mapping_df.columns)):
        # TODO_PLACEHOLDER: mapping_source_file
        # Required mapping columns are missing for post-curation channel exports.
        raise ValueError(
            f"Missing required columns in {mapping_csv}. "
            f"Required: {sorted(required_cols)}; found: {list(mapping_df.columns)}"
        )

    map_cur = mapping_df[mapping_df["merged_unit_id"].isin(accepted_unit_ids)].copy()
    mapped_unit_ids = set(map_cur["merged_unit_id"].astype(np.int64).tolist())
    missing_units = sorted(set(accepted_unit_ids.tolist()) - mapped_unit_ids)

    analyzer_hw_by_unit = {}
    if len(best_channel_id_per_unit) == len(accepted_unit_ids):
        analyzer_hw_by_unit = {
            int(uid): int(ch)
            for uid, ch in zip(accepted_unit_ids.tolist(), best_channel_id_per_unit.astype(np.int64).tolist())
        }
    elif len(missing_units) > 0:
        # TODO_PLACEHOLDER: mapping_source_file
        # Unable to infer missing merged-unit mappings without aligned analyzer best-channel vector.
        raise ValueError(
            f"Cannot infer missing merged-unit mappings: best_channel_id_per_unit length "
            f"({len(best_channel_id_per_unit)}) != accepted units ({len(accepted_unit_ids)})."
        )

    if missing_units:
        # Infer missing merged-unit mapping rows by best HW channel.
        # This is primarily for merged units created in curation where merged IDs
        # do not exist in the original merged_unit_id_mapping.csv.
        hw_meta_cols = ["sg_channel", "shank_id", "local_channel_on_shank"]
        hw_meta_lookup = {}
        inconsistent_hw = []
        for hw, grp in mapping_df.groupby("hw_channel"):
            uniq = grp[hw_meta_cols].drop_duplicates()
            if len(uniq) == 1:
                row = uniq.iloc[0]
                hw_meta_lookup[int(hw)] = {
                    "sg_channel": int(row["sg_channel"]),
                    "shank_id": int(row["shank_id"]),
                    "local_channel_on_shank": int(row["local_channel_on_shank"]),
                }
            else:
                inconsistent_hw.append(int(hw))
        if inconsistent_hw:
            # TODO_PLACEHOLDER: merge_policy_cross_channel
            # HW channel should map deterministically to SG/shank/local; ambiguity blocks safe inference.
            raise ValueError(
                f"Cannot infer mappings for missing units: {len(inconsistent_hw)} HW channel(s) have "
                f"ambiguous SG/shank/local metadata. Examples: {inconsistent_hw[:10]}"
            )

        inferred_rows = []
        unresolved_units = []
        trace_rows = []
        for uid in missing_units:
            hw = analyzer_hw_by_unit.get(int(uid), None)
            if hw is None:
                unresolved_units.append(int(uid))
                trace_rows.append({"unit_id": int(uid), "status": "missing_best_hw_channel"})
                continue
            meta = hw_meta_lookup.get(int(hw), None)
            if meta is None:
                unresolved_units.append(int(uid))
                trace_rows.append({"unit_id": int(uid), "status": "missing_hw_lookup", "hw_channel": int(hw)})
                continue
            inferred = {
                "merged_unit_id": int(uid),
                "hw_channel": int(hw),
                "sg_channel": int(meta["sg_channel"]),
                "shank_id": int(meta["shank_id"]),
                "local_channel_on_shank": int(meta["local_channel_on_shank"]),
            }
            inferred_rows.append(inferred)
            trace_rows.append(
                {
                    "unit_id": int(uid),
                    "status": "inferred",
                    "hw_channel": int(hw),
                    "sg_channel": int(meta["sg_channel"]),
                    "shank_id": int(meta["shank_id"]),
                    "local_channel_on_shank": int(meta["local_channel_on_shank"]),
                }
            )

        if debug_trace_mapping and len(trace_rows) > 0:
            trace_df = pd.DataFrame(trace_rows)
            print("[debug] post-curation mapping inference trace:")
            print(trace_df.to_string(index=False))

        if inferred_rows:
            inferred_df = pd.DataFrame(inferred_rows)
            map_cur = pd.concat([map_cur, inferred_df], ignore_index=True, sort=False)
            print(
                f"[warning] Inferred {len(inferred_rows)} missing merged-unit mapping row(s) "
                f"using analyzer best-channel mapping."
            )

        if unresolved_units:
            # TODO_PLACEHOLDER: mapping_source_file
            # Remaining units have no resolvable channel metadata and cannot be exported safely.
            raise ValueError(
                f"Missing merged-unit mappings for {len(unresolved_units)} accepted unit(s) after inference. "
                f"Examples: {unresolved_units[:10]}"
            )

    map_cur["merged_unit_id"] = map_cur["merged_unit_id"].astype(np.int64)
    map_cur["hw_channel"] = map_cur["hw_channel"].astype(np.int64)
    map_cur["sg_channel"] = map_cur["sg_channel"].astype(np.int64)
    map_cur["shank_id"] = map_cur["shank_id"].astype(np.int64)
    map_cur["local_channel_on_shank"] = map_cur["local_channel_on_shank"].astype(np.int64)

    # Optional consistency check against analyzer-derived best channels.
    if len(best_channel_id_per_unit) == len(accepted_unit_ids):
        map_hw_by_unit = {
            int(r.merged_unit_id): int(r.hw_channel) for r in map_cur.itertuples(index=False)
        }
        hw_mismatch_units = [
            uid for uid in accepted_unit_ids.tolist()
            if uid in map_hw_by_unit and uid in analyzer_hw_by_unit and map_hw_by_unit[uid] != analyzer_hw_by_unit[uid]
        ]
        if hw_mismatch_units:
            print(
                f"[warning] {len(hw_mismatch_units)} accepted unit(s) have HW-channel mismatch between "
                "mapping csv and analyzer-derived best channel."
            )
            map_rows_by_unit = map_cur.set_index("merged_unit_id")
            mismatch_rows = []
            for uid in hw_mismatch_units:
                if uid not in map_rows_by_unit.index:
                    continue
                r = map_rows_by_unit.loc[uid]
                # Handle the unlikely duplicated merged_unit_id case.
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                mismatch_rows.append(
                    {
                        "unit_id": int(uid),
                        "mapped_hw_channel": int(r["hw_channel"]),
                        "analyzer_best_hw_channel": int(analyzer_hw_by_unit[uid]),
                        "mapped_sg_channel": int(r["sg_channel"]),
                        "mapped_shank_id": int(r["shank_id"]),
                        "mapped_local_channel_on_shank": int(r["local_channel_on_shank"]),
                        "hw_delta": int(analyzer_hw_by_unit[uid] - int(r["hw_channel"])),
                    }
                )
            if mismatch_rows:
                mismatch_df = pd.DataFrame(mismatch_rows).sort_values("unit_id")
                print("[warning] Detailed HW-channel mismatch rows:")
                print(mismatch_df.to_string(index=False))

    # Save per-unit channel metadata (hw, sg, shank, local ch).
    unit_channel_map_csv = output_folder / "postcuration_unit_channel_mapping.csv"
    map_cur.sort_values("merged_unit_id").to_csv(unit_channel_map_csv, index=False)

    # Build unique channel summaries.
    hw_counts = (
        map_cur.groupby("hw_channel", as_index=False)
        .size()
        .rename(columns={"size": "n_units"})
        .sort_values("hw_channel")
    )
    sg_counts = (
        map_cur.groupby("sg_channel", as_index=False)
        .size()
        .rename(columns={"size": "n_units"})
        .sort_values("sg_channel")
    )
    shank_ch_counts = (
        map_cur.groupby(["shank_id", "local_channel_on_shank"], as_index=False)
        .size()
        .rename(columns={"size": "n_units"})
        .sort_values(["shank_id", "local_channel_on_shank"])
    )

    hw_csv = output_folder / "units_per_hw_channel_after_curation.csv"
    sg_csv = output_folder / "units_per_sg_channel_after_curation.csv"
    shank_ch_csv = output_folder / "units_per_shank_channel_after_curation.csv"
    hw_counts.to_csv(hw_csv, index=False)
    sg_counts.to_csv(sg_csv, index=False)
    shank_ch_counts.to_csv(shank_ch_csv, index=False)

    # Backward-compatible file names.
    channels_with_units = hw_counts["hw_channel"].to_numpy(dtype=np.int64)
    units_per_channel = hw_counts["n_units"].to_numpy(dtype=np.int64)
    channels_with_units_path = output_folder / "channels_with_units_after_curation.npy"
    np.save(channels_with_units_path, channels_with_units)
    channel_unit_count_csv = output_folder / "units_per_channel_after_curation.csv"
    hw_counts.rename(columns={"hw_channel": "channel_id"}).to_csv(channel_unit_count_csv, index=False)

    sg_channels_with_units_path = output_folder / "sg_channels_with_units_after_curation.npy"
    np.save(sg_channels_with_units_path, sg_counts["sg_channel"].to_numpy(dtype=np.int64))
    shank_ch_pairs_path = output_folder / "shank_channel_pairs_with_units_after_curation.npy"
    np.save(
        shank_ch_pairs_path,
        shank_ch_counts[["shank_id", "local_channel_on_shank"]].to_numpy(dtype=np.int64),
    )

    good_channels_for_sorting = None
    run_meta_path = output_folder / "preprocessing_metadata_for_run.json"
    if run_meta_path.exists():
        with open(run_meta_path, "r", encoding="utf-8") as f:
            run_meta = json.load(f)
        params = run_meta.get("params", {})
        good_channels_for_sorting = (
            params.get("good_channel_ids")
            or params.get("good_channels_fed_to_sorting")
            or run_meta.get("good_channel_ids")
            or run_meta.get("good_channels_fed_to_sorting")
        )

    # TODO_PLACEHOLDER: good_channels_metadata_key
    # If metadata keys differ across pipeline versions, this fallback uses rec_good channels.
    if good_channels_for_sorting is None:
        good_channels_for_sorting = rec_good.get_channel_ids()
    good_channels_for_sorting = np.asarray(good_channels_for_sorting, dtype=np.int64)
    good_channels_path = output_folder / "good_channels_fed_to_sorting.npy"
    np.save(good_channels_path, good_channels_for_sorting)

    # Build SG -> (shank, local_channel) map from ChMap384_32sh_SG.csv (12x32 block, SG+1 values).
    sg_map_csv = Path(r"E:\Centimani\Recording\Scripts\ChMap384_32sh_SG.csv")
    if not sg_map_csv.exists():
        # TODO_PLACEHOLDER: mapping_source_file
        # Channel map file is required for HW->SG->(shank,ch) projection of good channels.
        raise FileNotFoundError(f"Required SG map csv not found: {sg_map_csv}")
    sg_map_df = pd.read_csv(sg_map_csv, header=None).iloc[:12, :32]
    sg_to_shank_ch = {}
    for local_ch in range(sg_map_df.shape[0]):
        for shank_id in range(sg_map_df.shape[1]):
            val = sg_map_df.iat[local_ch, shank_id]
            if pd.isna(val):
                continue
            sg_plus_1 = int(val)
            if sg_plus_1 <= 0:
                continue
            sg_to_shank_ch[sg_plus_1 - 1] = (int(shank_id), int(local_ch))

    # Convert HW -> SG using the same reverse conversion convention from sorting pipeline.
    total_channels = 384
    num_cards = total_channels // 32
    good_sg_channels = np.array(
        [(int(hw) % num_cards) * 32 + (int(hw) // num_cards) for hw in good_channels_for_sorting],
        dtype=np.int64,
    )
    good_pairs = [sg_to_shank_ch.get(int(sg), None) for sg in good_sg_channels]
    good_pairs = [p for p in good_pairs if p is not None]
    if len(good_pairs) > 0:
        good_shanks = np.array([p[0] for p in good_pairs], dtype=np.int64)
        good_local_ch = np.array([p[1] for p in good_pairs], dtype=np.int64)
    else:
        good_shanks = np.array([], dtype=np.int64)
        good_local_ch = np.array([], dtype=np.int64)
    good_shank_ch_pairs_path = output_folder / "good_shank_channel_pairs_fed_to_sorting.npy"
    np.save(good_shank_ch_pairs_path, np.column_stack((good_shanks, good_local_ch)) if len(good_shanks) > 0 else np.empty((0, 2), dtype=np.int64))

    # Plot 12x32 matrix indexed by (shank, local_channel_on_shank).
    n_rows, n_cols = 12, 32
    matrix = np.zeros((n_rows, n_cols), dtype=np.int64)
    valid = (
        (shank_ch_counts["local_channel_on_shank"] >= 0)
        & (shank_ch_counts["local_channel_on_shank"] < n_rows)
        & (shank_ch_counts["shank_id"] >= 0)
        & (shank_ch_counts["shank_id"] < n_cols)
    )
    for r in shank_ch_counts[valid].itertuples(index=False):
        matrix[int(r.local_channel_on_shank), int(r.shank_id)] = int(r.n_units)

    grid_rows, grid_cols = np.indices((n_rows, n_cols))
    hi_mask = matrix > 0
    hi_rows = grid_rows[hi_mask]
    hi_cols = grid_cols[hi_mask]
    hi_counts = matrix[hi_mask]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(
        grid_cols.ravel(),
        grid_rows.ravel(),
        c="lightgray",
        s=36,
        marker="s",
        edgecolors="none",
        label="All (shank, ch)",
    )
    sc = ax.scatter(
        hi_cols,
        hi_rows,
        c=hi_counts,
        cmap="viridis",
        s=70,
        marker="s",
        edgecolors="black",
        linewidths=0.3,
        label="(shank, ch) with units",
    )
    if len(good_shanks) > 0:
        ax.scatter(
            good_shanks,
            good_local_ch,
            s=85,
            marker="o",
            facecolors="none",
            edgecolors="red",
            linewidths=1.1,
            label="Good channels fed to sorting",
        )
    ax.set_title("Post-curation units per (shank, ch) matrix (12 x 32)")
    ax.set_xlabel("shank_id (0-31)")
    ax.set_ylabel("local_channel_on_shank (0-11)")
    ax.set_xticks(np.arange(0, n_cols, 1))
    ax.set_yticks(np.arange(0, n_rows, 1))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.colorbar(sc, ax=ax, label="Number of units")
    # Move legend to top right, outside the axis
    # Place the legend at the upper right, but shift it further right outside the axis.
    # bbox_to_anchor specifies the anchor point for the legend; (1.14, 1) means 1.14 times the width of the axis to the right (x) and 1 times the height (y).
    # This moves the legend outside the plot area, to the right of the axes.
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1.18))
    fig.tight_layout()

    matrix_plot_path = output_folder / "postcuration_channel_unit_matrix_12x32.png"
    fig.savefig(matrix_plot_path, dpi=300)
    # Also save as SVG
    matrix_plot_svg_path = output_folder / "postcuration_channel_unit_matrix_12x32.svg"
    fig.savefig(matrix_plot_svg_path)
    plt.close(fig)
    print(f"Saved channel matrix plot (indexed by shank/ch): {matrix_plot_path}")
    print(f"Saved channel matrix plot SVG (indexed by shank/ch): {matrix_plot_svg_path}")

    total_units = int(accepted_sorting.get_num_units())
    total_units_path = output_folder / "total_units_after_curation.txt"
    with open(total_units_path, "w", encoding="utf-8") as f:
        f.write(f"{total_units}\n")

    summary_json_path = output_folder / "postcuration_channel_unit_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "channels_with_units_after_curation_hw": hw_counts["hw_channel"].astype(int).tolist(),
                "channels_with_units_after_curation_sg": sg_counts["sg_channel"].astype(int).tolist(),
                "shank_channel_pairs_with_units_after_curation": (
                    shank_ch_counts[["shank_id", "local_channel_on_shank"]]
                    .astype(int)
                    .values
                    .tolist()
                ),
                "units_per_hw_channel_after_curation": {
                    str(int(ch)): int(nu) for ch, nu in zip(hw_counts["hw_channel"], hw_counts["n_units"])
                },
                "units_per_sg_channel_after_curation": {
                    str(int(ch)): int(nu) for ch, nu in zip(sg_counts["sg_channel"], sg_counts["n_units"])
                },
                "units_per_shank_channel_after_curation": {
                    f"{int(r.shank_id)}_{int(r.local_channel_on_shank)}": int(r.n_units)
                    for r in shank_ch_counts.itertuples(index=False)
                },
                "total_units_after_curation": total_units,
                "good_channels_fed_to_sorting": good_channels_for_sorting.astype(int).tolist(),
                "good_sg_channels_fed_to_sorting": good_sg_channels.astype(int).tolist(),
                "good_shank_channel_pairs_fed_to_sorting": (
                    np.column_stack((good_shanks, good_local_ch)).astype(int).tolist()
                    if len(good_shanks) > 0
                    else []
                ),
            },
            f,
            indent=2,
        )

    print(f"Saved per-unit channel mapping: {unit_channel_map_csv}")
    print(f"Saved HW units-per-channel csv: {hw_csv}")
    print(f"Saved SG units-per-channel csv: {sg_csv}")
    print(f"Saved shank/ch units-per-channel csv: {shank_ch_csv}")
    print(f"Saved channels-with-units HW npy: {channels_with_units_path}")
    print(f"Saved channels-with-units SG npy: {sg_channels_with_units_path}")
    print(f"Saved shank/ch-pairs npy: {shank_ch_pairs_path}")
    print(f"Saved good shank/ch-pairs npy: {good_shank_ch_pairs_path}")
    print(f"Saved total units: {total_units_path}")
    print(f"Saved good channels fed to sorting: {good_channels_path}")
    print(f"Saved summary json: {summary_json_path}")


def run_curation_pipeline_once():
    output_folder, sorting, rec_good = resolve_output_sorting_recording()
    print(f"Using output folder: {output_folder}")
    print(f"Loaded sorting units: {sorting.get_num_units()}")

    if CURATION_STAGE == "prepare_figures":
        # Optional waveform-level curation: default skipped.
        if RUN_WAVEFORM_LEVEL_CURATION:
            if USER_STIM_TS_PATH is None:
                raise ValueError("USER_STIM_TS_PATH is required when RUN_WAVEFORM_LEVEL_CURATION=True.")
            stim_ts_sorted = load_stim_ts(Path(USER_STIM_TS_PATH), USER_STIM_TS_MAT_VAR)
            sorting_wfcurated, analyzer_wfcurated, we, output_folder_out = run_waveform_curation_and_unit_figures(
                recording=rec_good,
                sorting=sorting,
                stim_ts_sorted=stim_ts_sorted,
                output_folder=output_folder,
                rec_file_name=USER_RECORDING_DIR.name,  # placeholder for future use
                rec_all=rec_good,  # placeholder for future channel naming customization
                fs=int(rec_good.get_sampling_frequency()),
            )
            print(f"Waveform-level curation enabled: {sorting_wfcurated.get_num_units()} units")
        else:
            print("Waveform-level curation skipped (RUN_WAVEFORM_LEVEL_CURATION=False).")

        if REGENERATE_PREPARE_STAGE_PLOTS:
            analyzer_prepare = create_or_load_analyzer(output_folder, sorting, rec_good)
            save_prepare_stage_unit_summaries(output_folder, analyzer_prepare)
        else:
            print(
                "Skipping prepare-stage unit_summary_before_curation regeneration; "
                "preserving existing figures from sorting pipeline."
            )
        unit_figure_folder = get_unit_curation_figure_folder(output_folder)
        root_unit_pngs = collect_root_unit_pngs(unit_figure_folder)
        if len(root_unit_pngs) == 0:
            raise FileNotFoundError(
                f"No unit PNGs found under {unit_figure_folder}. "
                "Expected merged sorting figures like unit_summary_<unit_id>.png."
            )

        labels_json = output_folder / "curation_labels.json"
        if RUN_LABEL_UI_IN_PREPARE:
            prepare_ui_metadata = {}
            if SHOW_PREPARE_UI_STATS:
                prepare_ui_metadata = _build_prepare_ui_metadata(output_folder, sorting, rec_good)
            label_units_with_buttons(unit_figure_folder, labels_json, unit_metadata_by_id=prepare_ui_metadata)
        else:
            print("Skipping button labeling UI in prepare stage.")
        sync_label_folders_from_json(unit_figure_folder, labels_json)
        write_stage1_prompt(output_folder)
        return

    if CURATION_STAGE == "apply_curation":
        sorting_for_stage2 = sorting
        unit_figure_folder = get_unit_curation_figure_folder(output_folder)
        labels_json = output_folder / "curation_labels.json"
        sync_labels_json_from_folders(unit_figure_folder, labels_json)
        build_curation_json_from_labels_and_merges(output_folder)

        accepted_sorting, curated_analyzer, spk_vec_full, best_channel_id_per_unit = apply_curation_and_save_outputs(
            output_folder=output_folder,
            sorting=sorting_for_stage2,
            rec_good=rec_good,
            plot_figures=PLOT_POSTCURATION_SUMMARY_PLOTS,
            plot_waveform_figures=PLOT_POSTCURATION_WAVEFORMS,
            verbose_plot_logs=PRINT_CURATED_PLOT_LOGS,
        )

        save_postcuration_channel_matrix_and_exports(
            output_folder=output_folder,
            accepted_sorting=accepted_sorting,
            rec_good=rec_good,
            best_channel_id_per_unit=best_channel_id_per_unit,
            debug_trace_mapping=DEBUG_TRACE_POSTCURATION_MAPPING,
        )

        print(f"Accepted units: {accepted_sorting.get_num_units()}")
        print(f"Spk_Vec shape: {spk_vec_full.shape}")
        return

    raise ValueError("CURATION_STAGE must be either 'prepare_figures' or 'apply_curation'.")


def main():
    global USER_OUTPUT_FOLDER, USER_RECORDING_DIR, CURATION_STAGE
    if os.environ.get("LSNET_PIPELINE_OUTPUT_FOLDER", "").strip():
        apply_env_config()
    else:
        prompt_interactive_config()

    if USER_OUTPUT_FOLDER is None:
        raise ValueError(
            "USER_OUTPUT_FOLDER is unset after configuration. "
            "Set LSNET_PIPELINE_OUTPUT_FOLDER or complete interactive prompts."
        )

    folders = resolve_output_folders_for_run()
    for idx, folder in enumerate(folders):
        USER_OUTPUT_FOLDER = folder
        USER_RECORDING_DIR = infer_recording_dir_from_output(folder)
        print(f"\n{'=' * 60}\nRun {idx + 1}/{len(folders)}: {folder}\n{'=' * 60}\n")
        run_curation_pipeline_once()


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as exc:
        print(f"[error] {exc}")
        sys.exit(1)
