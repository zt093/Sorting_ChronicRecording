from __future__ import annotations

"""
Plot a 12 x 32 post-curation channel matrix for one selected Sorting output folder.

Data source:
- Reads only from the selected folder's `units_alignment_summary`.
- `unique_units_summary.json` provides the kept/good units.
- `discarded_units_summary.json` provides the discarded units.

Counting rule:
- Counts are keyed by `sg_channel` from the summary JSON rows.
- `n_units_in_channel` = rows from unique + discarded for that SG channel.
- `n_good_units_in_channel` = rows from unique only for that SG channel.

Plot:
- Uses `LSNET_probe.json` to map each SG channel onto the 12 x 32 grid
  (`shank_id`, `local_channel_on_shank`).
- Draws all channels as gray squares, channels with any units as blue hollow
  circles, and channels with good units as filled markers colored by count.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SORTING_CHECK_DIR = SCRIPT_DIR.parent
REPO_ROOT = SORTING_CHECK_DIR.parent
if str(SORTING_CHECK_DIR) not in sys.path:
    sys.path.insert(0, str(SORTING_CHECK_DIR))

from Units_alignment_UI import choose_output_root, safe_int  # noqa: E402


NUM_SHANKS = 32
CHANNELS_PER_SHANK = 12
PLOT_TITLE = "Post-curation units per channel matrix (12 x 32)"
COLORBAR_LABEL = "Number of good units"
OUTPUT_FILENAME = "post_curation_units_per_channel_matrix.png"
PROBE_FILE = REPO_ROOT / "LSNET_probe.json"
ALIGNMENT_FOLDER_NAME = "units_alignment_summary"
UNIQUE_SUMMARY_NAME = "unique_units_summary.json"
DISCARDED_SUMMARY_NAME = "discarded_units_summary.json"


def build_channel_grid_dataframe(
    num_shanks: int = NUM_SHANKS,
    channels_per_shank: int = CHANNELS_PER_SHANK,
) -> pd.DataFrame:
    probe_grid_df = load_probe_channel_grid(PROBE_FILE)
    if not probe_grid_df.empty:
        return probe_grid_df

    return pd.DataFrame(
        [
            {
                "shank_id": shank_id,
                "local_channel_on_shank": local_channel_on_shank,
                "sg_channel": np.nan,
                "is_present": True,
            }
            for shank_id in range(num_shanks)
            for local_channel_on_shank in range(channels_per_shank)
        ]
    )


def load_probe_channel_grid(probe_path: Path) -> pd.DataFrame:
    if not probe_path.exists():
        return pd.DataFrame()

    try:
        payload = json.loads(probe_path.read_text(encoding="utf-8"))
        probe_payload = payload["probes"][0]
        device_channel_indices = probe_payload["device_channel_indices"]
        shank_ids = probe_payload["shank_ids"]
        contact_positions = probe_payload["contact_positions"]
    except Exception:
        return pd.DataFrame()

    probe_df = pd.DataFrame(
        {
            "sg_channel": [safe_int(value) for value in device_channel_indices],
            "shank_id": [safe_int(value) for value in shank_ids],
            "x": [position[0] for position in contact_positions],
            "y": [position[1] for position in contact_positions],
        }
    )
    probe_df = probe_df.dropna(subset=["sg_channel", "shank_id", "y"]).copy()
    probe_df["sg_channel"] = probe_df["sg_channel"].astype(int)
    probe_df["shank_id"] = probe_df["shank_id"].astype(int)
    probe_df = probe_df.sort_values(["shank_id", "y"], ascending=[True, True]).reset_index(drop=True)
    probe_df["local_channel_on_shank"] = probe_df.groupby("shank_id").cumcount()
    probe_df["is_present"] = True
    return probe_df[["shank_id", "local_channel_on_shank", "sg_channel", "is_present"]]


def _load_rows_from_path(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def build_post_curation_channel_counts(output_root: Path) -> pd.DataFrame:
    summary_root = output_root / ALIGNMENT_FOLDER_NAME
    unique_units_path = summary_root / UNIQUE_SUMMARY_NAME
    discarded_units_path = summary_root / DISCARDED_SUMMARY_NAME

    unique_rows = _load_rows_from_path(unique_units_path)
    discarded_rows = _load_rows_from_path(discarded_units_path)

    if not unique_rows and not discarded_rows:
        raise FileNotFoundError(
            "No unit summary rows were found. "
            f"Checked: {unique_units_path} and {discarded_units_path}"
        )

    rows: list[dict[str, int]] = []

    for row in unique_rows:
        sg_channel = safe_int(row.get("sg_channel"))
        if sg_channel is None:
            continue
        rows.append(
            {
                "sg_channel": sg_channel,
                "n_units_in_channel": 1,
                "n_good_units_in_channel": 1,
            }
        )

    for row in discarded_rows:
        sg_channel = safe_int(row.get("sg_channel"))
        if sg_channel is None:
            continue
        rows.append(
            {
                "sg_channel": sg_channel,
                "n_units_in_channel": 1,
                "n_good_units_in_channel": 0,
            }
        )

    counts_df = (
        pd.DataFrame(rows)
        .groupby(["sg_channel"], as_index=False)
        .sum()
        if rows
        else pd.DataFrame(
            columns=[
                "sg_channel",
                "n_units_in_channel",
                "n_good_units_in_channel",
            ]
        )
    )

    grid_df = build_channel_grid_dataframe()
    merged_df = grid_df.merge(
        counts_df,
        on=["sg_channel"],
        how="left",
    )
    merged_df["n_units_in_channel"] = merged_df["n_units_in_channel"].fillna(0).astype(int)
    merged_df["n_good_units_in_channel"] = merged_df["n_good_units_in_channel"].fillna(0).astype(int)
    return merged_df.sort_values(["local_channel_on_shank", "shank_id"]).reset_index(drop=True)


def plot_post_curation_channel_matrix(
    channel_df: pd.DataFrame,
    *,
    title: str = PLOT_TITLE,
    cmap: str = "viridis",
) -> tuple[plt.Figure, plt.Axes]:
    required_columns = {
        "shank_id",
        "local_channel_on_shank",
        "n_units_in_channel",
        "n_good_units_in_channel",
    }
    missing_columns = required_columns.difference(channel_df.columns)
    if missing_columns:
        raise ValueError(f"channel_df is missing required columns: {sorted(missing_columns)}")

    fig, ax = plt.subplots(figsize=(18, 7))

    ax.scatter(
        channel_df["shank_id"],
        channel_df["local_channel_on_shank"],
        marker="s",
        s=34,
        facecolor="#d9d9d9",
        edgecolor="none",
        alpha=0.8,
        label="All channels",
        zorder=1,
    )

    unit_channels = channel_df[channel_df["n_units_in_channel"] > 0]
    if not unit_channels.empty:
        ax.scatter(
            unit_channels["shank_id"],
            unit_channels["local_channel_on_shank"],
            marker="o",
            s=90,
            facecolors="none",
            edgecolors="#1f77b4",
            linewidths=1.6,
            label="Channels with units",
            zorder=2,
        )

    good_unit_channels = channel_df[channel_df["n_good_units_in_channel"] > 0]
    good_units_artist = None
    if not good_unit_channels.empty:
        norm = Normalize(
            vmin=1,
            vmax=max(1, int(good_unit_channels["n_good_units_in_channel"].max())),
        )
        good_units_artist = ax.scatter(
            good_unit_channels["shank_id"],
            good_unit_channels["local_channel_on_shank"],
            c=good_unit_channels["n_good_units_in_channel"],
            cmap=cmap,
            norm=norm,
            marker="o",
            s=52,
            edgecolors="black",
            linewidths=0.3,
            label="Good units in the channel",
            zorder=3,
        )
        colorbar = fig.colorbar(good_units_artist, ax=ax, pad=0.02)
        colorbar.set_label(COLORBAR_LABEL)

    ax.set_title(title, loc="left")
    ax.set_xlabel("shank_id (0-31)")
    ax.set_ylabel("local_channel_on_shank (0-11)")
    ax.set_xlim(-0.5, NUM_SHANKS - 0.5)
    ax.set_ylim(-0.5, CHANNELS_PER_SHANK - 0.5)
    ax.set_xticks(np.arange(NUM_SHANKS))
    ax.set_yticks(np.arange(CHANNELS_PER_SHANK))
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_facecolor("white")

    legend_handles = [
        Patch(facecolor="#d9d9d9", edgecolor="none", label="All channels"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="#1f77b4",
            markeredgewidth=1.6,
            markersize=8,
            label="Channels with units",
        ),
    ]
    if good_units_artist is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=plt.get_cmap(cmap)(0.7),
                markeredgecolor="black",
                markeredgewidth=0.3,
                markersize=7,
                label="Good units in the channel",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.18),
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#b0b0b0",
        columnspacing=1.0,
        handletextpad=0.6,
        borderpad=0.5,
        fontsize=9,
    )

    fig.tight_layout()
    return fig, ax


def build_and_save_post_curation_plot(output_root: Path) -> tuple[pd.DataFrame, Path]:
    channel_df = build_post_curation_channel_counts(output_root)
    fig, _ax = plot_post_curation_channel_matrix(channel_df)

    stats_root = output_root / "stats"
    stats_root.mkdir(parents=True, exist_ok=True)
    figure_path = stats_root / OUTPUT_FILENAME
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    return channel_df, figure_path


def main() -> None:
    output_root = choose_output_root()
    channel_df, figure_path = build_and_save_post_curation_plot(output_root)

    print(f"Saved figure to: {figure_path}")
    print(
        "Counted channels with units: "
        f"{int((channel_df['n_units_in_channel'] > 0).sum())} / {len(channel_df)}"
    )
    print(
        "Counted channels with good units: "
        f"{int((channel_df['n_good_units_in_channel'] > 0).sum())} / {len(channel_df)}"
    )


if __name__ == "__main__":
    main()
