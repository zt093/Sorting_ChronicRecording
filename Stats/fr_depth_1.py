sorting_analyzer = r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\SNr1\SNr1_20260422\curated_analyzer"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, Polygon
import spikeinterface.core as si

analyzer = si.load_sorting_analyzer(sorting_analyzer)
probe = analyzer.get_probe()
templates_ext = analyzer.get_extension("templates")
templates = templates_ext.get_data()  # (n_units, n_samples, n_channels)
unit_locs = analyzer.get_extension("unit_locations").get_data()[:, :2]  # (n_units, 2) µm

unit_ids = analyzer.unit_ids
contact_positions = probe.contact_positions[:, :2]  # (n_channels, 2)

# Firing rate per unit (Hz)
spike_counts = analyzer.sorting.count_num_spikes_per_unit()
total_duration = analyzer.get_total_duration()
firing_rates = np.array([spike_counts[uid] / total_duration for uid in unit_ids])

# Trough-to-peak duration per unit (ms), measured on each unit's best channel
sampling_rate = analyzer.sampling_frequency


def compute_cv2(spike_train):
    """CV2 = mean(2*|ISI_{i+1} - ISI_i| / (ISI_{i+1} + ISI_i)) over adjacent ISI pairs."""
    if len(spike_train) < 3:
        return np.nan
    isi = np.diff(spike_train).astype(float)
    if len(isi) < 2:
        return np.nan
    return float(np.mean(2 * np.abs(np.diff(isi)) / (isi[:-1] + isi[1:])))


def trough_to_peak_ms(waveform, fs):
    trough_idx = int(np.argmin(waveform))
    peak_idx   = int(np.argmax(waveform))

    if abs(waveform[trough_idx]) >= abs(waveform[peak_idx]):
        # negative spike: find repolarisation peak after the trough
        if trough_idx >= len(waveform) - 1:
            return np.nan
        post_peak_idx = trough_idx + int(np.argmax(waveform[trough_idx:]))
        return (post_peak_idx - trough_idx) / fs * 1000.0
    else:
        # positive spike: find repolarisation trough after the peak
        if peak_idx >= len(waveform) - 1:
            return np.nan
        post_trough_idx = peak_idx + int(np.argmin(waveform[peak_idx:]))
        return (post_trough_idx - peak_idx) / fs * 1000.0

# Physical contact geometry (square pads)
contact_size = 20.0      # um, side length
contact_pitch = 25.0     # um, center-to-center spacing


def get_best_channel(template):
    ptp = np.ptp(template, axis=0)
    return int(np.argmax(ptp))


x_pos = contact_positions[:, 0]
y_pos = contact_positions[:, 1]
x_min, x_max = x_pos.min(), x_pos.max()
y_min, y_max = y_pos.min(), y_pos.max()

# Probe outline: rectangular shank with a pointed tip
# Larger y = deeper = closer to tip (tip at bottom of plot)
shank_halfwidth = (x_max - x_min) / 2 + contact_size * 1.5
shank_center_x = (x_min + x_max) / 2
probe_left = shank_center_x - shank_halfwidth
probe_right = shank_center_x + shank_halfwidth
probe_top = y_min - contact_size * 2      # surface end (small y, top of plot)
probe_bottom = y_max + contact_size * 2   # deep end (large y, bottom of plot)
tip_y = probe_bottom + shank_halfwidth * 1.2  # tip, deepest (largest y)

probe_outline = np.array([
    [probe_left, probe_top],
    [probe_right, probe_top],
    [probe_right, probe_bottom],
    [shank_center_x, tip_y],
    [probe_left, probe_bottom],
])

cv2_vals = np.array([
    compute_cv2(analyzer.sorting.get_unit_spike_train(uid, segment_index=0))
    for uid in unit_ids
])

fig, (ax, ax_fr, ax_tp, ax_cv2) = plt.subplots(
    1, 4, figsize=(12, 12), sharey=True,
    gridspec_kw={"width_ratios": [3, 2, 2, 2], "wspace": 0.08},
)

ax.add_patch(Polygon(probe_outline, closed=True, fill=False,
                     edgecolor="black", lw=1.5, zorder=1))

half = contact_size / 2
for cx, cy in contact_positions:
    ax.add_patch(Rectangle((cx - half, cy - half), contact_size, contact_size,
                           fill=False, edgecolor="black", lw=1.0, zorder=2))

# Waveform overlay scaling: span ~1 pitch wide and tall enough to see clearly
x_scale = contact_pitch * 1.6
y_scale = (contact_pitch * 2.5) / max(np.max(np.abs(templates)), 1e-9)

colors = cm.tab20(np.linspace(0, 1, len(unit_ids)))
rng = np.random.default_rng(0)
rng.shuffle(colors)

# Stagger waveforms along x so they don't overlap.
# Each unit is randomly placed on the left or right of its best channel.
# Units sharing the same best channel get larger deterministic offsets,
# and every unit additionally gets a small random jitter.
best_channels = [get_best_channel(templates[i]) for i in range(len(unit_ids))]
ch_to_units = {}
for i, ch in enumerate(best_channels):
    ch_to_units.setdefault(ch, []).append(i)

stack_step = x_scale * 0.6           # offset for units on the same channel
jitter_amp = x_scale * 0.25          # random jitter applied to every unit
unit_jitter = rng.uniform(-jitter_amp, jitter_amp, size=len(unit_ids))
# Balanced left/right assignment: half -1, half +1, then shuffled
n_units = len(unit_ids)
unit_side = np.array([-1] * (n_units // 2) + [1] * (n_units - n_units // 2))
rng.shuffle(unit_side)

for ch, idxs in ch_to_units.items():
    # Alternate sides for stacked units on the same channel
    for k, i in enumerate(idxs):
        side = unit_side[i]
        template = templates[i]
        cx, cy = unit_locs[i]
        waveform = template[:, ch]
        n_samples = len(waveform)
        t = np.linspace(0, x_scale, n_samples)
        base_offset = half + k * stack_step + unit_jitter[i]
        # Shift the waveform left or right WITHOUT flipping time
        if side > 0:
            x_start = cx + base_offset
        else:
            x_start = cx - base_offset - x_scale
        ax.plot(x_start + t, cy - waveform * y_scale,
                color=colors[i], lw=1.4, zorder=3)

# 100 uV scale bar (templates are in uV)
scale_uv = 100.0
bar_len = scale_uv * y_scale
bar_x = probe_right + x_scale + contact_size * 1.5
bar_y = y_min
ax.plot([bar_x, bar_x], [bar_y, bar_y - bar_len],
        color="black", lw=2, zorder=4)
ax.text(bar_x + contact_size * 0.4, bar_y - bar_len / 2,
        f"{int(scale_uv)} µV", ha="left", va="center", fontsize=10)

ax.set_xlim(probe_left - x_scale - contact_size,
            bar_x + contact_size * 4)
ax.set_ylim(tip_y + contact_size, probe_top - contact_size)  # inverted: tip (large y) at bottom
ax.set_aspect("equal")
ax.axis("off")

# --- Firing rate vs depth side panel ---
unit_depths = unit_locs[:, 1]

# Lollipops: stem from x=0 to firing rate, dot at the rate
for i, uid in enumerate(unit_ids):
    ax_fr.hlines(unit_depths[i], 0, firing_rates[i],
                 color=colors[i], lw=1.2, alpha=0.7, zorder=2)
ax_fr.scatter(firing_rates, unit_depths, c=colors, s=45,
              edgecolor="black", lw=0.5, zorder=3)

ax_fr.set_xlabel("firing rate (Hz)")
ax_fr.set_xlim(left=0)
ax_fr.spines["top"].set_visible(False)
ax_fr.spines["right"].set_visible(False)
ax_fr.tick_params(left=True, labelleft=True)
ax_fr.set_ylabel("depth (µm)")

# --- Trough-to-peak vs depth side panel ---
trough_to_peak = np.array([
    trough_to_peak_ms(templates[i, :, best_channels[i]], sampling_rate)
    for i in range(len(unit_ids))
])

for i, uid in enumerate(unit_ids):
    ax_tp.hlines(unit_depths[i], 0, trough_to_peak[i],
                 color=colors[i], lw=1.2, alpha=0.7, zorder=2)
ax_tp.scatter(trough_to_peak, unit_depths, c=colors, s=45,
              edgecolor="black", lw=0.5, zorder=3)

ax_tp.set_xlabel("trough-to-peak (ms)")
ax_tp.set_xlim(left=0)
ax_tp.spines["top"].set_visible(False)
ax_tp.spines["right"].set_visible(False)
ax_tp.tick_params(left=False, labelleft=False)
ax_tp.set_ylabel("")

# --- CV2 vs depth side panel ---
for i, uid in enumerate(unit_ids):
    ax_cv2.hlines(unit_depths[i], 0, cv2_vals[i],
                  color=colors[i], lw=1.2, alpha=0.7, zorder=2)
ax_cv2.scatter(cv2_vals, unit_depths, c=colors, s=45,
               edgecolor="black", lw=0.5, zorder=3)
ax_cv2.axvline(1.0, color="gray", lw=0.8, linestyle="--")  # CV2=1 → Poisson
ax_cv2.set_xlabel("CV2")
ax_cv2.set_xlim(left=0)
ax_cv2.spines["top"].set_visible(False)
ax_cv2.spines["right"].set_visible(False)
ax_cv2.tick_params(left=False, labelleft=False)
ax_cv2.set_ylabel("")

# Print a summary table
print(f"{'unit':>6}  {'depth(um)':>9}  {'rate(Hz)':>9}  {'t2p(ms)':>8}  {'CV2':>6}")
for i, uid in enumerate(unit_ids):
    print(f"{str(uid):>6}  {unit_depths[i]:9.1f}  "
          f"{firing_rates[i]:9.2f}  {trough_to_peak[i]:8.3f}  {cv2_vals[i]:6.3f}")

plt.tight_layout()
save_path = r"C:\Users\Windows\SpikeSorting\random\probe_waveforms.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Done. Saved to {save_path}")
