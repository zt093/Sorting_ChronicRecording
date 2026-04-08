import gc
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import numpy as np
import pandas as pd
import probeinterface as pi
import spikeinterface.extractors as se
import spikeinterface.full as si
from spikeinterface.core import BaseRecording
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries

if not hasattr(pi, "read_spikegadgets"):
    def _probeinterface_read_spikegadgets_stub(*args, **kwargs):
        return None

    pi.read_spikegadgets = _probeinterface_read_spikegadgets_stub


SPIKEGADGETS_TARGET_SAMPLING_FREQUENCY = 30000.0
DEBUG_PRINT_FIRST_N_SHANKS = 4
DEBUG_PRINT_FIRST_N_CHANNEL_IDS = 8
class CustomSamplingFrequencyRecording(BaseRecording):
    """Wrap a recording while overriding only its reported sampling frequency."""

    def __init__(self, recording, new_sampling_frequency: float):
        BaseRecording.__init__(
            self,
            sampling_frequency=new_sampling_frequency,
            channel_ids=recording.channel_ids,
            dtype=recording.get_dtype(),
        )

        for segment in recording._recording_segments:
            self.add_recording_segment(segment)

        self._kwargs = getattr(recording, "_kwargs", {})
        for key in recording.get_property_keys():
            values = recording.get_property(key)
            self.set_property(key, values)


def read_impedance(imp_txt_path: Path) -> pd.DataFrame:
    """Read impedance data from a text file and return a DataFrame."""
    with open(imp_txt_path, "r") as f:
        lines = f.readlines()

    impedance_entries = []
    for line in lines:
        match = re.match(r"NT\s+(\d+),\s+Channel\s+\d+:\s+(\d+)", line.strip())
        if match:
            sg_ch = int(match.group(1))
            imp = float(match.group(2))
            impedance_entries.append((sg_ch - 1, imp))

    data_sorted = sorted(impedance_entries, key=lambda x: x[0])

    result = {}
    for k, v in data_sorted:
        result[k] = v

    impedance_entries = [(k, result[k]) for k in sorted(result.keys())]
    return pd.DataFrame(impedance_entries, columns=["SG_index", "impedance_ohm"])


def read_impedance_file(imp_txt_path: Path, rec_probe, ref_probe_path: Path, debug: bool = False):
    """
    Read impedance data and map it to recording channels via position matching.

    Mapping logic:
    1. For each channel in rec_probe, get its (x, y) position.
    2. Find the contact in ref_probe with the same position.
    3. Get that contact's device_channel_index.
    4. Look up impedance where SG_index matches that reference channel.
    """
    imp_df = read_impedance(imp_txt_path)
    if debug:
        print(f"[DEBUG] Loaded impedance data: {len(imp_df)} entries")
        print(
            "[DEBUG] Impedance SG_index range: "
            f"{imp_df['SG_index'].min()} - {imp_df['SG_index'].max()}"
        )

    ref_probe_group = pi.read_probeinterface(ref_probe_path)
    ref_probe = ref_probe_group.probes[0]
    if debug:
        print(f"[DEBUG] Loaded ref_probe: {len(ref_probe.device_channel_indices)} channels")

    rec_positions = rec_probe.contact_positions
    rec_channels = rec_probe.device_channel_indices
    if debug:
        print(f"[DEBUG] rec_probe: {len(rec_channels)} channels")

    ref_positions = ref_probe.contact_positions
    ref_channels = ref_probe.device_channel_indices

    impedance_map = {}
    matched_count = 0
    no_position_match = 0
    no_impedance_found = 0

    if debug:
        print("\n[DEBUG] === Position Matching Process ===")
        print(f"{'rec_ch':<8} {'rec_pos':<20} {'ref_ch':<8} {'dist':<10} {'imp (ohm)':<12} {'status'}")
        print("-" * 80)

    for i, rec_ch in enumerate(rec_channels):
        rec_pos = rec_positions[i]
        distances = np.sqrt(np.sum((ref_positions - rec_pos) ** 2, axis=1))
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist < 1.0:
            ref_ch_matched = ref_channels[min_idx]
            imp_row = imp_df[imp_df["SG_index"] == ref_ch_matched]
            if not imp_row.empty:
                imp_value = imp_row["impedance_ohm"].values[0]
                impedance_map[rec_ch] = imp_value
                matched_count += 1
                status = "OK"
            else:
                impedance_map[rec_ch] = np.nan
                no_impedance_found += 1
                imp_value = "N/A"
                status = "NO_IMP"
        else:
            impedance_map[rec_ch] = np.nan
            no_position_match += 1
            ref_ch_matched = "N/A"
            imp_value = "N/A"
            status = "NO_POS"

        if debug:
            pos_str = f"({rec_pos[0]:.1f}, {rec_pos[1]:.1f})"
            imp_str = f"{imp_value}" if isinstance(imp_value, str) else f"{imp_value:.0f}"
            print(f"{rec_ch:<8} {pos_str:<20} {ref_ch_matched:<8} {min_dist:<10.2f} {imp_str:<12} {status}")

    if debug:
        print("-" * 80)
        print(
            f"[DEBUG] Summary: {matched_count} matched, "
            f"{no_position_match} no position match, "
            f"{no_impedance_found} no impedance found"
        )

    return impedance_map


class EphysToNWBConverter:
    """Unified converter for SpikeGadgets and Ripple data to NWB."""

    def __init__(self, recording_method: str, chunk_duration: float = 240.0):
        if recording_method not in ["spikegadget", "ripple"]:
            raise ValueError("Recording method must be 'spikegadget' or 'ripple'")
        self.recording_method = recording_method
        self.chunk_duration = chunk_duration
        self._data_dset_path = "/acquisition/ElectricalSeries/data"

    @staticmethod
    def _forward_conversion(hw_chan: int, totalchan: int) -> int:
        """Map hardware-channel order into SpikeGadgets .rec channel order."""
        num_cards = totalchan // 32
        return ((hw_chan % 32) * num_cards) + (hw_chan // 32)

    def _apply_spikegadgets_channel_reorder(self, recording):
        """Reorder SpikeGadgets channels to match the HW→.rec mapping used elsewhere."""
        totalchan = recording.get_num_channels()
        if totalchan == 0:
            return recording

        channel_ids_rec = list(recording.get_channel_ids())
        new_hw_chans = [
            self._forward_conversion(hw_chan, totalchan)
            for hw_chan in range(totalchan)
        ]
        new_channel_order = [channel_ids_rec[c] for c in new_hw_chans]
        return recording.select_channels(new_channel_order)

    def _get_recording(self, data_file: Path):
        """Load recording using SpikeInterface with system-specific tweaks."""
        if self.recording_method == "spikegadget":
            recording = si.read_spikegadgets(data_file)
            original_sampling_frequency = float(recording.get_sampling_frequency())
            if original_sampling_frequency != SPIKEGADGETS_TARGET_SAMPLING_FREQUENCY:
                print(
                    "Correcting SpikeGadgets sampling frequency "
                    f"from {original_sampling_frequency} Hz "
                    f"to {SPIKEGADGETS_TARGET_SAMPLING_FREQUENCY} Hz"
                )
                recording = CustomSamplingFrequencyRecording(
                    recording, SPIKEGADGETS_TARGET_SAMPLING_FREQUENCY
                )
            recording = self._apply_spikegadgets_channel_reorder(recording)
        else:
            recording = se.read_blackrock(data_file)
            # Ripple channel IDs start at 1; remap to 0-indexed strings to match probe
            old_ids = recording.get_channel_ids()
            new_ids = [str(i) for i in range(len(old_ids))]
            recording = recording.rename_channels(new_ids)
        return recording

    def _get_conversion_offset(self, recording):
        """Infer conversion/offset from SpikeInterface if available."""
        gains = recording.get_channel_gains()
        offsets = recording.get_channel_offsets()
        if gains is None or len(gains) == 0:
            conversion = 1.0
        else:
            conversion = float(gains[0]) / 1e6
        if offsets is None or len(offsets) == 0:
            offset = 0.0
        else:
            offset = float(offsets[0]) / 1e6
        return conversion, offset

    def _prepare_nwb_object(self, data_file: Path, electrode_df: "pd.DataFrame",
                            metadata: dict):
        """
        Build an NWBFile with electrode metadata but no ElectricalSeries data.

        Returns
        -------
        nwbfile, electrode_table_region, channel_ids,
        (sampling_freq, num_frames, conversion, offset)
        """
        metadata = metadata or {}
        electrode_location = metadata.get("electrode_location") or "unknown"

        session_start_time = self.get_timestamp(data_file)
        nwbfile = NWBFile(
            session_description=metadata.get("session_desc", "NWB file for ephys data"),
            identifier=str(uuid4()),
            session_start_time=session_start_time,
            experimenter=[metadata.get("experimenter", "Zhang, Xiaorong")],
            lab=metadata.get("lab", "XL Lab"),
            institution=metadata.get("institution", "Rice University"),
            experiment_description=metadata.get("exp_desc", "None"),
            session_id=metadata.get("session_id", "None"),
        )

        device = nwbfile.create_device(name="LSNET", description="--", manufacturer="XieLab")
        nwbfile.add_electrode_column(name="device_channel_index",
                                     description="Original device channel index")
        nwbfile.add_electrode_column(name="label", description="label of electrode")

        shank_ids = sorted(electrode_df["shank_id"].unique().tolist())
        electrode_groups = {}
        for shank_id in shank_ids:
            electrode_groups[shank_id] = nwbfile.create_electrode_group(
                name=f"shank{shank_id}",
                description=f"electrode group for shank {shank_id}",
                device=device,
                location=electrode_location,
            )

        for _, row in electrode_df.iterrows():
            nwbfile.add_electrode(
                group=electrode_groups[row["shank_id"]],
                label=f"shank{row['shank_id']}:ch{row['device_channel_index']}",
                location=electrode_location,
                rel_x=float(row["x"]),
                rel_y=float(row["y"]),
                imp=float(row["impedance_ohm"]) if not np.isnan(row["impedance_ohm"]) else np.nan,
                device_channel_index=int(row["device_channel_index"]),
            )

        electrode_table_region = nwbfile.create_electrode_table_region(
            list(range(len(electrode_df))), "all electrodes"
        )

        recording = self._get_recording(data_file)
        sampling_freq = recording.get_sampling_frequency()
        num_frames = recording.get_num_frames()
        conversion, offset = self._get_conversion_offset(recording)
        channel_ids = self._resolve_shank_channel_ids(
            recording, electrode_df["device_channel_index"].tolist()
        )
        del recording
        gc.collect()

        return nwbfile, electrode_table_region, channel_ids, (sampling_freq, num_frames, conversion, offset)

    def _write_nwb_with_data(self, nwbfile, nwb_path: Path, electrode_table_region,
                              first_chunk, sampling_freq: float,
                              conversion: float, offset: float, is_chunked: bool):
        """Attach ElectricalSeries to nwbfile and write to disk."""
        if is_chunked:
            data_io = H5DataIO(
                data=first_chunk,
                maxshape=(None, first_chunk.shape[1]),
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )
        else:
            data_io = H5DataIO(
                data=first_chunk,
                maxshape=(None, first_chunk.shape[1]),
                chunks=True,
            )
        electrical_series = ElectricalSeries(
            name="ElectricalSeries",
            data=data_io,
            electrodes=electrode_table_region,
            starting_time=0.0,
            rate=sampling_freq,
            conversion=conversion,
            offset=offset,
        )
        nwbfile.add_acquisition(electrical_series)
        with NWBHDF5IO(nwb_path, "w") as io:
            io.write(nwbfile)

    def _normalize_channel_ids(self, recording, channel_ids):
        """Match channel id types with recording.get_channel_ids()."""
        rec_ids = recording.get_channel_ids()
        if len(rec_ids) == 0:
            return channel_ids
        if isinstance(rec_ids[0], str):
            return [str(ch) for ch in channel_ids]
        return [int(ch) for ch in channel_ids]

    def _resolve_shank_channel_ids(self, recording, sg_channel_indices):
        """
        Resolve shank channels by treating probe SG channels as indices into the
        reordered recording channel-id list.
        """
        rec_ids = list(recording.get_channel_ids())
        resolved_channel_ids = []

        for sg_idx in sg_channel_indices:
            sg_idx_int = int(sg_idx)
            if sg_idx_int < 0 or sg_idx_int >= len(rec_ids):
                raise IndexError(
                    f"Probe SG channel index {sg_idx_int} is out of bounds for "
                    f"recording with {len(rec_ids)} channels"
                )
            resolved_channel_ids.append(rec_ids[sg_idx_int])

        return self._normalize_channel_ids(recording, resolved_channel_ids)

    def _read_recording_chunk(self, recording, channel_ids=None,
                              start_frame: int = 0, end_frame: int = None):
        if end_frame is None:
            end_frame = recording.get_num_frames()
        if channel_ids:
            trace = recording.get_traces(
                channel_ids=channel_ids,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        else:
            trace = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
        return trace

    def _read_recording_chunk_from_file(self, data_file: Path, channel_ids=None,
                                        start_frame: int = 0, end_frame: int = None):
        """Open recording fresh to avoid reader caching across chunks."""
        recording = self._get_recording(data_file)
        if channel_ids:
            channel_ids = self._normalize_channel_ids(recording, channel_ids)
        trace = self._read_recording_chunk(
            recording,
            channel_ids=channel_ids,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        del recording
        gc.collect()
        return trace

    def _estimate_file_size_gb(self, num_frames: int, num_channels: int, dtype_size: int = 2):
        total_bytes = num_frames * num_channels * dtype_size
        return total_bytes / (1024 ** 3)

    def _print_chunk_timing(self, read_sec: float, slice_sec: float, write_sec: float):
        total_sec = read_sec + slice_sec + write_sec
        print(
            f"    Timing: read {read_sec:.2f}s | "
            f"slice {slice_sec:.2f}s | "
            f"write {write_sec:.2f}s | "
            f"total {total_sec:.2f}s"
        )

    def _should_use_chunked_processing(self, num_frames: int, num_channels: int,
                                       threshold_gb: float = 1.0):
        # Use float32 (4 bytes) for estimation since SpikeInterface often converts
        estimated_size = self._estimate_file_size_gb(num_frames, num_channels, dtype_size=4)
        return estimated_size > threshold_gb

    def get_timestamp(self, file_path: Path) -> datetime:
        """Extract the recording start time from filename if available."""
        # Expected format: YYYYMMDD_HHMMSS_<description>.rec (e.g., 20251218_152916_stim3.rec)
        match = re.search(r"(\d{8})_(\d{6})", file_path.name)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        print("Warning: filename lacks timestamp; using current time.")
        return datetime.now()

    def initiate_nwb(self, data_file: Path, nwb_path: Path, electrode_df: pd.DataFrame,
                     metadata: dict = None, has_multiple_files: bool = False):
        """Create and write an NWB file for a single shank or all channels."""
        metadata = metadata or {}
        print("Initiating NWB file...")

        nwbfile, electrode_table_region, channel_ids, \
            (sampling_freq, num_frames, conversion, offset) = \
            self._prepare_nwb_object(data_file, electrode_df, metadata)

        use_chunked = self._should_use_chunked_processing(num_frames, len(channel_ids))
        if has_multiple_files and not use_chunked:
            print("Multiple files detected - enabling chunked dataset for appending")
            use_chunked = True

        estimated_size = self._estimate_file_size_gb(num_frames, len(channel_ids), dtype_size=4)
        print(f"Estimated memory: ~{estimated_size:.2f} GB (float32)")
        print(f"Duration: {num_frames / sampling_freq:.2f} seconds")
        print(f"Processing mode: {'Chunked' if use_chunked else 'Direct'}")

        if use_chunked:
            chunk_frames = int(self.chunk_duration * sampling_freq)
            num_chunks = int(np.ceil(num_frames / chunk_frames))
            print(f"Processing {num_chunks} chunks of {self.chunk_duration}s each...")

            read_t0 = perf_counter()
            first_chunk = self._read_recording_chunk_from_file(
                data_file, channel_ids, 0, min(chunk_frames, num_frames)
            )
            read_sec = perf_counter() - read_t0
            print(f"Writing chunk 1/{num_chunks}...")
            write_t0 = perf_counter()
            self._write_nwb_with_data(nwbfile, nwb_path, electrode_table_region,
                                      first_chunk, sampling_freq, conversion, offset,
                                      is_chunked=True)
            write_sec = perf_counter() - write_t0
            self._print_chunk_timing(read_sec, 0.0, write_sec)
            del first_chunk, nwbfile
            gc.collect()

            for i in range(1, num_chunks):
                start_frame = i * chunk_frames
                end_frame = min((i + 1) * chunk_frames, num_frames)
                print(f"Processing chunk {i+1}/{num_chunks} "
                      f"(frames {start_frame}-{end_frame})...")
                read_t0 = perf_counter()
                chunk_data = self._read_recording_chunk_from_file(
                    data_file, channel_ids, start_frame, end_frame
                )
                read_sec = perf_counter() - read_t0
                write_t0 = perf_counter()
                self._append_nwb_h5(nwb_path, chunk_data)
                write_sec = perf_counter() - write_t0
                self._print_chunk_timing(read_sec, 0.0, write_sec)
                del chunk_data
                gc.collect()
        else:
            read_t0 = perf_counter()
            trace = self._read_recording_chunk_from_file(data_file, channel_ids)
            read_sec = perf_counter() - read_t0
            print("Writing NWB file...")
            write_t0 = perf_counter()
            self._write_nwb_with_data(nwbfile, nwb_path, electrode_table_region,
                                      trace, sampling_freq, conversion, offset,
                                      is_chunked=False)
            write_sec = perf_counter() - write_t0
            self._print_chunk_timing(read_sec, 0.0, write_sec)

        return channel_ids

    def append_nwb(self, nwb_path: Path, data_file: Path, channel_ids: list):
        """Append additional recording data to an existing NWB file."""
        recording = self._get_recording(data_file)
        channel_ids = self._normalize_channel_ids(recording, channel_ids)

        sampling_freq = recording.get_sampling_frequency()
        num_frames = recording.get_num_frames()

        use_chunked = self._should_use_chunked_processing(num_frames, len(channel_ids))
        estimated_size = self._estimate_file_size_gb(num_frames, len(channel_ids))
        print(f"Appending file size: ~{estimated_size:.2f} GB")
        print(f"Processing mode: {'Chunked' if use_chunked else 'Direct'}")

        if use_chunked:
            chunk_frames = int(self.chunk_duration * sampling_freq)
            num_chunks = int(np.ceil(num_frames / chunk_frames))
            print(f"Appending {num_chunks} chunks...")
            del recording
            gc.collect()
            for i in range(num_chunks):
                start_frame = i * chunk_frames
                end_frame = min((i + 1) * chunk_frames, num_frames)
                print(f"Appending chunk {i+1}/{num_chunks} "
                      f"(frames {start_frame}-{end_frame})...")
                read_t0 = perf_counter()
                chunk_data = self._read_recording_chunk_from_file(
                    data_file, channel_ids, start_frame, end_frame
                )
                read_sec = perf_counter() - read_t0
                write_t0 = perf_counter()
                self._append_nwb_h5(nwb_path, chunk_data)
                write_sec = perf_counter() - write_t0
                self._print_chunk_timing(read_sec, 0.0, write_sec)
                del chunk_data
                gc.collect()
        else:
            read_t0 = perf_counter()
            trace = self._read_recording_chunk_from_file(data_file, channel_ids)
            read_sec = perf_counter() - read_t0
            write_t0 = perf_counter()
            self._append_nwb_h5(nwb_path, trace)
            write_sec = perf_counter() - write_t0
            self._print_chunk_timing(read_sec, 0.0, write_sec)
            del trace
            gc.collect()

    def convert_all_shanks(self, data_files: list, shank_configs: list,
                           metadata: dict = None,
                           first_file_max_duration_s: float | None = None):
        """
        Convert all shanks simultaneously, reading each raw-file chunk only once.

        Each chunk is read with the union of all shank channel IDs, then sliced
        per-shank in memory — eliminating the N_shanks × N_chunks redundant disk
        reads that occur when shanks are processed sequentially.

        Parameters
        ----------
        data_files : list of Path
            Raw recording files in chronological order.
        shank_configs : list of dict
            Each entry has keys: 'shank_id', 'electrode_df', 'nwb_path'.
        metadata : dict, optional
            Session-level metadata forwarded to each NWB file.
        first_file_max_duration_s : float, optional
            If set, only convert up to this many seconds from the first data file
            and skip any later files.
        """
        metadata = metadata or {}
        first_file = data_files[0]
        limit_first_file = (
            first_file_max_duration_s is not None and first_file_max_duration_s > 0
        )
        if limit_first_file:
            print(
                f"Limiting conversion to the first {first_file_max_duration_s:.2f} s "
                f"of {first_file.name}"
            )
            data_files = [first_file]
        has_multi = len(data_files) > 1

        # ── Phase 1: build NWBFile metadata objects (header reads only) ──
        print("Preparing NWB structures for all shanks...")
        prepared = []   # (nwbfile, electrode_table_region, channel_ids, nwb_path)
        rec_meta = None
        for cfg in shank_configs:
            print(f"  Shank {cfg['shank_id']} → {cfg['nwb_path'].name}")
            nwbfile, etr, ch_ids, meta = self._prepare_nwb_object(
                first_file, cfg["electrode_df"], metadata
            )
            prepared.append((nwbfile, etr, ch_ids, cfg["nwb_path"]))
            if rec_meta is None:
                rec_meta = meta

        sampling_freq, num_frames, conversion, offset = rec_meta
        if limit_first_file:
            max_frames = min(
                num_frames,
                int(round(first_file_max_duration_s * sampling_freq))
            )
            if max_frames <= 0:
                raise ValueError("first_file_max_duration_s must produce at least one frame")
            print(
                f"Using {max_frames} / {num_frames} frames from the first file "
                f"({max_frames / sampling_freq:.2f} s)"
            )
            num_frames = max_frames
        chunk_frames = int(self.chunk_duration * sampling_freq)

        # ── Phase 2: build union channel list and per-shank column indices ──
        seen, all_ch = set(), []
        for _, _, ch_ids, _ in prepared:
            for ch in ch_ids:
                if ch not in seen:
                    seen.add(ch)
                    all_ch.append(ch)

        ch_to_col = {ch: i for i, ch in enumerate(all_ch)}
        col_idxs = [
            np.array([ch_to_col[ch] for ch in ch_ids])
            for _, _, ch_ids, _ in prepared
        ]

        use_chunked = self._should_use_chunked_processing(num_frames, len(all_ch))
        if has_multi:
            use_chunked = True

        # ── Phase 3: read each chunk once, fan out to all shanks ──
        for file_idx, data_file in enumerate(data_files):
            if file_idx == 0:
                file_frames = num_frames
            else:
                rec = self._get_recording(data_file)
                file_frames = rec.get_num_frames()
                del rec
                gc.collect()

            n_chunks = int(np.ceil(file_frames / chunk_frames))
            print(f"\n{'='*60}")
            print(f"File {file_idx+1}/{len(data_files)}: {data_file.name} "
                  f"— {n_chunks} chunk(s)")
            print(f"{'='*60}")

            for ci in range(n_chunks):
                start = ci * chunk_frames
                end = min((ci + 1) * chunk_frames, file_frames)
                print(f"  Chunk {ci+1}/{n_chunks} (frames {start}–{end})…")

                # Single read covering all shanks
                read_t0 = perf_counter()
                chunk_all = self._read_recording_chunk_from_file(
                    data_file, all_ch, start, end
                )
                read_sec = perf_counter() - read_t0

                is_first = (file_idx == 0 and ci == 0)
                is_ch = use_chunked or n_chunks > 1
                slice_sec = 0.0
                write_sec = 0.0

                for s, (nwbfile, etr, ch_ids, nwb_path) in enumerate(prepared):
                    slice_t0 = perf_counter()
                    chunk_sh = chunk_all[:, col_idxs[s]]
                    slice_sec += perf_counter() - slice_t0
                    write_t0 = perf_counter()
                    if is_first:
                        self._write_nwb_with_data(
                            nwbfile, nwb_path, etr,
                            chunk_sh, sampling_freq, conversion, offset,
                            is_chunked=is_ch,
                        )
                    else:
                        self._append_nwb_h5(nwb_path, chunk_sh)
                    write_sec += perf_counter() - write_t0

                self._print_chunk_timing(read_sec, slice_sec, write_sec)

                del chunk_all
                gc.collect()

            if file_idx == 0:
                # Release NWBFile objects — data already written to disk
                prepared = [(None, None, ch_ids, p) for _, _, ch_ids, p in prepared]
                gc.collect()

    def _append_nwb_dset(self, dset, data_to_append, append_axis: int) -> None:
        """Append data along a specified axis in an HDF5 dataset."""
        dset_shape = dset.shape
        dset_len = dset_shape[append_axis]
        app_len = data_to_append.shape[append_axis]
        new_len = dset_len + app_len

        slicer = [slice(None)] * len(dset_shape)
        slicer[append_axis] = slice(-app_len, None)

        dset.resize(new_len, axis=append_axis)
        dset[tuple(slicer)] = data_to_append

    def _append_nwb_h5(self, nwb_path: Path, data_to_append) -> None:
        """Append chunk data directly via h5py to avoid loading NWB object."""
        import h5py

        with h5py.File(nwb_path, "a") as h5file:
            dset = h5file[self._data_dset_path]
            self._append_nwb_dset(dset, data_to_append, 0)


def load_bad_ch(bad_file: Path) -> list:
    if not bad_file.exists():
        print(f"No bad channels file found at {bad_file}. Using all channels.")
        return []
    with open(bad_file, "r") as f:
        bad_channels = [int(line.strip()) for line in f if line.strip()]
    return bad_channels


def _build_electrode_df(probe, impedance_data=None, bad_ch_ids=None):
    device_channel_indices = probe.device_channel_indices
    positions = probe.contact_positions
    shank_ids = probe.shank_ids
    if shank_ids is None:
        shank_ids = np.zeros(len(device_channel_indices), dtype=int).astype(str)
    else:
        shank_ids = np.array(shank_ids, dtype=str)

    impedance_data = impedance_data or {}
    impedances = np.array([
        impedance_data.get(str(ch), impedance_data.get(int(ch), np.nan))
        for ch in device_channel_indices
    ])

    electrode_df = pd.DataFrame({
        "device_channel_index": device_channel_indices,
        "x": positions[:, 0],
        "y": positions[:, 1],
        "shank_id": shank_ids,
        "impedance_ohm": impedances,
    })

    if bad_ch_ids:
        electrode_df = electrode_df[
            ~electrode_df["device_channel_index"].isin(bad_ch_ids)
        ]

    return electrode_df


def _collect_data_files(data_path: Path, recording_method: str) -> list:
    if data_path.is_file():
        return [data_path]

    if recording_method == "spikegadget":
        data_files = sorted(data_path.glob("*.rec"))
    else:
        data_files = sorted(
            p for p in data_path.iterdir()
            if p.suffix.lower() in {".ns6", ".ns5", ".ns2", ".ns1"}
        )

    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_path}")

    return data_files


def _to_nwb_name(stem: str) -> str:
    if stem.startswith("Chronic_Rec"):
        return stem.replace("Chronic_Rec", "NWB", 1)
    return f"NWB_{stem}"


def _build_output_folder(
    data_path: Path, data_files: list[Path], output_root: Path | None = None
) -> Path:
    base_parent = output_root if output_root is not None else (
        data_path.parent if data_path.is_file() else data_path
    )

    if len(data_files) == 1:
        folder_name = _to_nwb_name(data_files[0].stem)
    else:
        first_name = _to_nwb_name(data_files[0].stem)
        last_name = _to_nwb_name(data_files[-1].stem)
        folder_name = f"{first_name}_{last_name}"

    return base_parent / folder_name


def _forward_conversion(hw_chan: int, totalchan: int) -> int:
    """Map hardware-channel order into SpikeGadgets .rec channel order."""
    num_cards = totalchan // 32
    return ((hw_chan % 32) * num_cards) + (hw_chan // 32)


def _print_shank_namespace_preview(
    shank_configs: list[dict], total_channels: int, limit: int = DEBUG_PRINT_FIRST_N_SHANKS
) -> None:
    """Print the first few shanks in the SG/probe namespace and legacy .rec indexing."""
    if total_channels <= 0 or limit <= 0 or not shank_configs:
        return

    print("\n=== Shank channel preview ===")
    for cfg in shank_configs[:limit]:
        shank_id = cfg["shank_id"]
        electrode_df = cfg["electrode_df"]
        sg_channels = [int(ch) for ch in electrode_df["device_channel_index"].tolist()]
        channel_locations = [
            (float(row["x"]), float(row["y"]))
            for _, row in electrode_df.iterrows()
        ]
        print(f"Shank {shank_id}:")
        print(f"  SG/probe channels: {sg_channels}")
        hw_channels = [_forward_conversion(ch, total_channels) for ch in sg_channels]
        print(f"  Legacy .rec indices: {hw_channels}")
        print(f"  Channel locations: {channel_locations}")


def _print_resolved_shank_channel_id_preview(
    shank_configs: list[dict], limit: int = DEBUG_PRINT_FIRST_N_SHANKS
) -> None:
    """Print the resolved recording channel IDs that will be sliced for the first shanks."""
    if limit <= 0 or not shank_configs:
        return

    print("\n=== Resolved shank channel IDs used for slicing ===")
    for cfg in shank_configs[:limit]:
        shank_id = cfg["shank_id"]
        sg_channels = [int(ch) for ch in cfg["electrode_df"]["device_channel_index"].tolist()]
        resolved_channel_ids = cfg.get("resolved_channel_ids", [])
        print(f"Shank {shank_id}:")
        print(f"  SG channel indices: {sg_channels}")
        print(f"  Recording channel IDs used: {resolved_channel_ids}")


def _print_recording_channel_id_preview(recording, limit: int = DEBUG_PRINT_FIRST_N_CHANNEL_IDS) -> None:
    """Print the first few channel IDs from the loaded recording."""
    if limit <= 0:
        return

    channel_ids = list(recording.get_channel_ids())
    if not channel_ids:
        print("\nNo recording channel IDs found.")
        return

    preview = channel_ids[:limit]
    print(f"\nFirst {len(preview)} recording channel IDs: {preview}")


def _format_elapsed_time(elapsed_seconds: float) -> str:
    """Format elapsed wall-clock time as HH:MM:SS.ss."""
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def main():
    # ------------------------------------------------------------------
    # Configuration for this setup
    # Use one fixed probe JSON for both SpikeGadgets and Ripple recordings.
    # Update these paths to match your environment.
    RECORDING_METHOD = "spikegadget"  # "spikegadget" or "ripple"
    PROBE_FILE = Path(r"E:\Curtis\spikeinterface\LSNET_probe.json")
    IMPEDANCE_FILE = Path(r"E:\Curtis\spikeinterface\imp_09222025_LSNET18.txt")
    # ------------------------------------------------------------------

    # Old interactive recording-system selection kept for reference:
    # print("\nWhich recording system do you want to convert?")
    # print("  1. SpikeGadgets (SG)")
    # print("  2. Ripple")
    # choice = input("Enter choice (1 or 2): ").strip()
    # if choice == "1":
    #     recording_method = "spikegadget"
    # elif choice == "2":
    #     recording_method = "ripple"
    # else:
    #     print("Invalid choice. Exiting.")
    #     sys.exit(1)

    recording_method = RECORDING_METHOD
    if recording_method not in {"spikegadget", "ripple"}:
        print(f"Invalid RECORDING_METHOD: {recording_method}")
        sys.exit(1)

    # User inputs
    rec_path_str = input("\nEnter recording file path (or folder): ").strip().strip('"').strip("'")
    rec_path = Path(rec_path_str)
    if not rec_path.exists():
        print(f"Recording path not found: {rec_path}")
        sys.exit(1)

    source_folder = rec_path.parent if rec_path.is_file() else rec_path

    # Old probe-selection logic kept for reference, but not used in this setup:
    # tri_num = input("Enter probe trident number: ").strip()
    # script_dir = Path(__file__).resolve().parent
    # probe_folder = script_dir.parent / "LSNET_probes"
    # probe_file = probe_folder / f"LSNET_probe_tri{tri_num}_{probe_suffix}.json"

    probe_suffix = "SG" if recording_method == "spikegadget" else "Ripple"
    probe_file = PROBE_FILE

    if not probe_file.exists():
        print(f"Probe file not found: {probe_file}")
        sys.exit(1)

    print(f"Loading probe: {probe_file}")
    probe_group = pi.read_probeinterface(probe_file)
    probe = probe_group.probes[0]

    # Old interactive impedance-file input kept for reference:
    # imp_txt_str = input(
    #     "Enter impedance file path (press Enter to skip): "
    # ).strip().strip('"').strip("'")
    imp_txt_path = IMPEDANCE_FILE
    ref_probe_path = probe_file
    if imp_txt_path and imp_txt_path.exists() and ref_probe_path.exists():
        impedance_data = read_impedance_file(
            imp_txt_path,
            rec_probe=probe,
            ref_probe_path=ref_probe_path,
        )
        print(f"Loaded impedance data for {len(impedance_data)} channels")
    else:
        if imp_txt_path and not imp_txt_path.exists():
            print(f"No impedance file found at {imp_txt_path}")
        if imp_txt_path and not ref_probe_path.exists():
            print(f"No reference probe found at {ref_probe_path}")
        impedance_data = None

    bad_folder = source_folder / f"bad_channel_screening_{probe_suffix}"
    bad_file = bad_folder / "bad_channels.txt"
    bad_ch_ids = load_bad_ch(bad_file)

    chunk_duration_input = input(
        "Enter chunk duration in seconds for large files (default: 240): "
    ).strip()
    chunk_duration = float(chunk_duration_input) if chunk_duration_input else 240.0
    first_file_only_duration_input = input(
        "Enter seconds to convert from only the first recording "
        "(press Enter for all data): "
    ).strip()
    first_file_only_duration = (
        float(first_file_only_duration_input)
        if first_file_only_duration_input else None
    )

    electrode_location = input("Please enter the electrode location (default: unknown): ").strip() or "unknown"
    exp_desc = input("Please enter the experiment description (default: None): ").strip() or "None"

    converter = EphysToNWBConverter(recording_method, chunk_duration=chunk_duration)

    # Build electrode table and decide shanks
    electrode_df_all = _build_electrode_df(
        probe, impedance_data=impedance_data, bad_ch_ids=bad_ch_ids
    )
    unique_shanks = sorted(electrode_df_all["shank_id"].unique().tolist())
    shank_str = input(
        f"Enter shank numbers (e.g. 0,1,2) or press Enter for all {unique_shanks}: "
    ).strip()
    if shank_str:
        shanks = [str(x) for x in re.findall(r"\d+", shank_str)]
    else:
        shanks = unique_shanks

    data_files = _collect_data_files(rec_path, recording_method)
    first_file = data_files[0]
    print(f"All data files: {data_files}")
    output_root = rec_path.parent.parent if rec_path.is_file() else rec_path.parent
    output_folder = _build_output_folder(rec_path, data_files, output_root=output_root)
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_folder}")

    # Use the output folder name as the session description for combined runs.
    session_description = output_folder.name

    shank_configs = []
    for shank_id in shanks:
        electrode_df = electrode_df_all[electrode_df_all["shank_id"] == shank_id].copy()
        if electrode_df.empty:
            print(f"Shank {shank_id} has no channels after filtering. Skipping.")
            continue
        # Sort channels by depth (y, shallow to deep)
        electrode_df = electrode_df.sort_values("y", ascending=True).reset_index(drop=True)
        nwb_path = output_folder / f"{session_description}_sh{shank_id}.nwb"
        shank_configs.append({
            "shank_id": shank_id,
            "electrode_df": electrode_df,
            "nwb_path": nwb_path,
        })

    if recording_method == "spikegadget" and shank_configs:
        first_recording = converter._get_recording(first_file)
        _print_recording_channel_id_preview(first_recording)
        total_channels = first_recording.get_num_channels()
        for cfg in shank_configs:
            sg_channel_indices = cfg["electrode_df"]["device_channel_index"].tolist()
            cfg["resolved_channel_ids"] = converter._resolve_shank_channel_ids(
                first_recording, sg_channel_indices
            )
        _print_resolved_shank_channel_id_preview(shank_configs)
        del first_recording
        gc.collect()
        _print_shank_namespace_preview(shank_configs, total_channels)

    conversion_start_datetime = datetime.now()
    conversion_start_perf = perf_counter()
    print(
        "\nStarting NWB conversion at "
        f"{conversion_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    converter.convert_all_shanks(
        data_files,
        shank_configs,
        metadata={
            "session_desc": session_description,
            "electrode_location": electrode_location,
            "exp_desc": exp_desc,
        },
        first_file_max_duration_s=first_file_only_duration,
    )

    conversion_end_datetime = datetime.now()
    total_runtime_sec = perf_counter() - conversion_start_perf

    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print(f"Conversion started: {conversion_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Conversion ended:   {conversion_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total conversion time: {_format_elapsed_time(total_runtime_sec)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
