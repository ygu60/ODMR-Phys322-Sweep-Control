"""
First software pass: per-sweep voltage-to-frequency averaging and
normalization.

Reads the raw per-sample traces exported by odmr_labview_replica.py
(--save-traces -> odmr_labview_replica_traces.csv, columns run_index,
time_s, ch1_V, ch2_V). Sweeps can optionally be filtered out before
averaging: --min-diff-mv excludes sweeps whose overall CH1 max-min range is
too small (probably no visible feature), while --max-jump-mv excludes
sweeps with an irregular single-sample spike/glitch (a jump between
adjacent samples larger than the threshold, as distinct from a broad
max-min range). For each remaining sweep:

  1. Fit CH2 (VCO tuning voltage) to a straight line vs. sample index, then
     convert that fit - not the raw per-sample CH2 value - to frequency:

         freq_GHz = V_TO_F_SLOPE_GHZ_PER_V * V_ch2_fit + V_TO_F_INTERCEPT_GHZ

     (same calibration as odmr_averaged_sweep.py). The ramp is linear by
     design, so this recovers a smooth, strictly monotonic frequency axis.
     Using each raw (noisy) CH2 sample directly doesn't: at high
     --min-record-length the true per-sample voltage step becomes smaller
     than CH2's own measurement noise, so sorting samples by their raw
     computed frequency scrambles time-adjacency - points that end up next
     to each other after sorting can come from far-apart, uncorrelated
     times, producing a jagged reconstruction even when the real (time
     domain) signal is smooth on the scope.
  2. Interpolate that sweep's CH1 (detector) signal onto a common frequency
     grid.
  3. Normalize per-run (--normalize-per-run, on by default): rescale the
     sweep so its own mean maps to 1 - intensity(f) becomes
     1 + (intensity(f) - mean)/|mean| - so sweep-to-sweep gain/offset
     drift (e.g. laser RIN) doesn't bias the average computed next. Mean,
     not max, is used as the per-sweep reference since max is sensitive to
     a single noisy sample.

Then, across all sweeps:

  2. Average: I(f) = (1/N) * sum_i I_i(f), with per-frequency standard
     deviation as well.
  3. Normalize: contrast(f) = 100 * (I(f) - I_max) / |I_max|, where I_max is
     the averaged curve's own maximum (assumed off-resonance baseline), so
     the baseline reads as 0% and the resonance dip reads as %-below-baseline
     regardless of whether the detector signal itself is positive or negative.

Requires: pip install numpy matplotlib

All CSV output goes to csv_output/, all PNG output to png_output/ (created
automatically if missing) - matching odmr_labview_replica.py's layout.

Usage:
  python odmr_labview_voltage_freq.py
  python odmr_labview_voltage_freq.py --input csv_output/odmr_labview_replica_traces.csv --grid-points 500
"""

import argparse
import csv
import datetime
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Same voltage -> RF frequency calibration as odmr_averaged_sweep.py.
V_TO_F_SLOPE_GHZ_PER_V = 0.1201024911
V_TO_F_INTERCEPT_GHZ = 1.933

# --- Output folders ---
CSV_DIR = "csv_output"
PNG_DIR = "png_output"

DEFAULT_INPUT_CSV = os.path.join(CSV_DIR, "odmr_labview_replica_traces.csv")

# Experiment condition label for output filenames - fixed for now (not yet a
# CLI flag); update this if you run at a different field.
FIELD_LABEL = "0Field"


def output_base_name(n_sweeps):
    """ODMR_Trace_<field>_N<sweeps>_<MMDDYY>_<HHMM>, e.g.
    ODMR_Trace_0Field_N1000_091526_1627."""
    timestamp = datetime.datetime.now().strftime("%m%d%y_%H%M")
    return f"ODMR_Trace_{FIELD_LABEL}_N{n_sweeps}_{timestamp}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Raw traces CSV from odmr_labview_replica.py --save-traces.")
    p.add_argument("--grid-points", type=int, default=1000, help="Number of points in the common frequency grid.")
    p.add_argument("--min-diff-mv", type=float, default=None,
                   help="Only include sweeps whose CH1 max-min difference exceeds this threshold (mV) - "
                        "same quality filter as odmr_labview_replica.py's --min-diff-mv. Unset: use all sweeps.")
    p.add_argument("--max-jump-mv", type=float, default=30,
                   help="Exclude sweeps with a single-sample-to-sample CH1 jump larger than this (mV) - "
                        "filters out irregular spiking/glitches, as distinct from --min-diff-mv's broad "
                        "max-min range. Unset: no spike filtering.")
    p.add_argument("--normalize-per-run", action=argparse.BooleanOptionalAction, default=False,
                   help="Rescale each sweep to its own mean before averaging, so sweep-to-sweep "
                        "gain/offset drift (e.g. laser RIN) doesn't bias the average (default: on).")
    p.add_argument("--despike-window", type=int, default=5,
                   help="Rolling-median window (samples, odd) used to detect per-sample spikes within "
                        "each kept sweep - distinct from --max-jump-mv, which rejects a whole sweep. "
                        "0 disables despiking (default: 5).")
    p.add_argument("--despike-threshold-mv", type=float, default=15.0,
                   help="A sample is replaced by its local rolling-median value if it deviates from that "
                        "median by more than this (mV) (default: 15).")
    p.add_argument("--diagnostic-traces", type=int, default=10,
                   help="When --min-diff-mv/--max-jump-mv filter out sweeps, plot this many kept and "
                        "this many discarded raw CH1-vs-frequency traces side by side for inspection "
                        "(0 disables this diagnostic plot).")
    return p.parse_args()


def load_sweeps(path):
    """Read the long-format traces CSV and return {run_index: (ch1_V array, ch2_V array)}."""
    sweeps = defaultdict(lambda: ([], []))
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if "ch2_V" not in header:
            raise SystemExit(
                f"{path} has no ch2_V column - rerun odmr_labview_replica.py with "
                "--ch2-enabled --save-traces to capture the tuning voltage."
            )
        run_idx_col = header.index("run_index")
        ch1_col = header.index("ch1_V")
        ch2_col = header.index("ch2_V")

        for row in reader:
            run_idx = int(row[run_idx_col])
            ch1, ch2 = sweeps[run_idx]
            ch1.append(float(row[ch1_col]))
            ch2.append(float(row[ch2_col]))

    return {run_idx: (np.array(ch1), np.array(ch2)) for run_idx, (ch1, ch2) in sweeps.items()}


def despike(v, window, threshold_mv):
    """Replace samples that deviate from their local rolling median by more
    than threshold_mv with that median - cleans up isolated per-sample
    glitches within an otherwise-good sweep, in original (time) sample
    order, before frequency sorting/interpolation. Returns (cleaned, n_replaced)."""
    if window <= 1 or len(v) < window:
        return v, 0

    pad = window // 2
    padded = np.pad(v, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    local_median = np.median(windows, axis=1)

    outliers = np.abs(v - local_median) * 1e3 > threshold_mv
    cleaned = v.copy()
    cleaned[outliers] = local_median[outliers]
    return cleaned, int(np.sum(outliers))


def smooth_freq_axis(ch2_v):
    """Fit CH2 to a straight line vs. sample index and convert that fit -
    not the raw per-sample value - to frequency. The ramp is linear by
    design, so this recovers a smooth, strictly monotonic frequency axis;
    using each raw (noisy) CH2 sample directly lets scope measurement noise
    dominate the sort order once the true per-sample voltage step is small
    (i.e. at high --min-record-length), scrambling time-adjacency and
    producing a jagged reconstruction of an otherwise-smooth signal."""
    idx = np.arange(len(ch2_v))
    slope, intercept = np.polyfit(idx, ch2_v, 1)
    ch2_fit = slope * idx + intercept
    return V_TO_F_SLOPE_GHZ_PER_V * ch2_fit + V_TO_F_INTERCEPT_GHZ


def plot_kept_vs_discarded(kept, discarded, n, output_path):
    """Plot up to n raw CH1-vs-frequency traces each from kept and
    discarded sweeps, side by side, so a filter's effect can be inspected."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    for ax, group, title, color in (
        (ax1, kept, "Kept", "tab:blue"),
        (ax2, discarded, "Discarded", "tab:red"),
    ):
        run_ids = list(group)[:n]
        for run_idx in run_ids:
            ch1_v, ch2_v = group[run_idx]
            freq_ghz = smooth_freq_axis(ch2_v)
            order = np.argsort(freq_ghz)
            ax.plot(freq_ghz[order], ch1_v[order], color=color, alpha=0.6, linewidth=0.8)
        ax.set_xlabel("Microwave drive frequency (GHz)")
        ax.set_title(f"{title} ({len(run_ids)}/{len(group)} shown)")
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("CH1 detector output (V)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved kept-vs-discarded diagnostic plot to {output_path}")


def main():
    args = parse_args()
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(PNG_DIR, exist_ok=True)
    sweeps = load_sweeps(args.input)
    n_loaded = len(sweeps)
    print(f"Loaded {n_loaded} sweeps from {args.input}")

    discarded = {}
    if args.min_diff_mv is not None or args.max_jump_mv is not None:
        kept = {}
        for run_idx, (ch1_v, ch2_v) in sweeps.items():
            if args.min_diff_mv is not None and (np.max(ch1_v) - np.min(ch1_v)) * 1e3 <= args.min_diff_mv:
                discarded[run_idx] = (ch1_v, ch2_v)
                continue
            if args.max_jump_mv is not None and np.max(np.abs(np.diff(ch1_v))) * 1e3 > args.max_jump_mv:
                discarded[run_idx] = (ch1_v, ch2_v)
                continue
            kept[run_idx] = (ch1_v, ch2_v)
        sweeps = kept
        print(
            f"Kept {len(sweeps)}/{n_loaded} sweeps "
            f"(min_diff_mv={args.min_diff_mv}, max_jump_mv={args.max_jump_mv})"
        )
        if not sweeps:
            raise SystemExit("No sweeps passed the filters - nothing to average.")

    # N reflects the actual number of sweeps used (post-filtering), so the
    # filename alone tells you what went into the average.
    base_name = output_base_name(len(sweeps))
    output_csv = os.path.join(CSV_DIR, f"{base_name}.csv")
    output_png = os.path.join(PNG_DIR, f"{base_name}.png")
    output_kept_vs_discarded_png = os.path.join(PNG_DIR, f"{base_name}_kept_vs_discarded.png")

    if args.diagnostic_traces > 0 and discarded:
        plot_kept_vs_discarded(sweeps, discarded, args.diagnostic_traces, output_kept_vs_discarded_png)

    # Convert each sweep's tuning voltage to frequency via a linear fit
    # (see smooth_freq_axis), sorted ascending so interpolation is
    # well-defined.
    sweep_freqs = []
    sweep_intensities = []
    n_despiked = 0
    for ch1_v, ch2_v in sweeps.values():
        if args.despike_window > 0:
            # Despike in original (time) sample order, before frequency
            # sorting - an isolated glitch is a neighbor in time, not
            # necessarily in the (noisy) frequency ordering.
            ch1_v, n = despike(ch1_v, args.despike_window, args.despike_threshold_mv)
            n_despiked += n

        freq_ghz = smooth_freq_axis(ch2_v)
        order = np.argsort(freq_ghz)
        intensity = ch1_v[order]

        if args.normalize_per_run:
            # Rescale this sweep to its own mean, so its typical level maps
            # to 1 regardless of that sweep's absolute gain/offset. Mean
            # (not max) as the baseline reference, since max is sensitive to
            # a single noisy sample - whatever that sweep's largest noise
            # spike happens to be - while mean is far more robust (same
            # convention as odmr_averaged_sweep.py's deviation-from-mean).
            # abs() in the denominator keeps a dip reading as "below 1" even
            # if the sweep's baseline is negative.
            baseline = np.mean(intensity)
            intensity = 1.0 + (intensity - baseline) / abs(baseline) if baseline else np.zeros_like(intensity)

        sweep_freqs.append(freq_ghz[order])
        sweep_intensities.append(intensity)

    if args.despike_window > 0:
        print(
            f"Despiked {n_despiked} samples across {len(sweeps)} sweeps "
            f"(window={args.despike_window}, threshold={args.despike_threshold_mv:.0f} mV)"
        )

    # Common frequency grid: the intersection of all sweeps' frequency
    # ranges, so every grid point can be interpolated without extrapolating.
    grid_min = max(f[0] for f in sweep_freqs)
    grid_max = min(f[-1] for f in sweep_freqs)
    if grid_min >= grid_max:
        raise SystemExit("Sweeps' frequency ranges don't overlap - nothing to average.")
    freq_grid = np.linspace(grid_min, grid_max, args.grid_points)

    interpolated = np.array([
        np.interp(freq_grid, f, i) for f, i in zip(sweep_freqs, sweep_intensities)
    ])

    # Average and normalize.
    i_avg = np.mean(interpolated, axis=0)
    i_sd = np.std(interpolated, axis=0)
    i_max = np.max(i_avg)
    # Divide by abs(i_max), not i_max, so the sign convention (0% at
    # baseline, negative % at a dip) holds even when the detector signal
    # itself is negative-going.
    contrast_pct = 100.0 * (i_avg - i_max) / abs(i_max) if i_max else np.zeros_like(i_avg)

    # Relative noise: SD as a percentage of the local mean. If this stays
    # roughly flat through the dip, the dip's noise is scaling down along
    # with its signal (shot noise / laser RIN, etc.) - evidence the dip is a
    # real reduction in signal rather than an unrelated measurement artifact.
    relative_noise_pct = 100.0 * i_sd / np.abs(i_avg)

    avg_col = "detector_norm_avg" if args.normalize_per_run else "detector_V_avg"
    sd_col = "detector_norm_sd" if args.normalize_per_run else "detector_V_sd"
    avg_unit = "ratio to per-run baseline" if args.normalize_per_run else "V"

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["freq_GHz", avg_col, sd_col, "contrast_pct", "relative_noise_pct"])
        for fq, avg, sd, c, rn in zip(freq_grid, i_avg, i_sd, contrast_pct, relative_noise_pct):
            writer.writerow([fq, avg, sd, c, rn])
    print(f"Saved averaged, voltage-derived spectrum to {output_csv}")

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(freq_grid, i_avg, linewidth=1)
    ax1.fill_between(freq_grid, i_avg - i_sd, i_avg + i_sd, alpha=0.25)
    ax1.set_xlabel("Microwave drive frequency (GHz)")
    ax1.set_ylabel(f"Detector output ({avg_unit})")
    title_suffix = ", per-run normalized" if args.normalize_per_run else ""
    ax1.set_title(f"Averaged detector response vs. drive frequency (N={len(sweeps)} sweeps{title_suffix})")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    print(f"Saved plot to {output_png}")

    dip_idx = np.argmin(contrast_pct)
    print(
        f"\nBaseline (max) detector level: {i_max:.5f} {avg_unit}\n"
        f"Deepest dip: {contrast_pct[dip_idx]:.2f}% below baseline at {freq_grid[dip_idx]:.4f} GHz"
    )

    plt.show()


if __name__ == "__main__":
    main()
