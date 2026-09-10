"""
First software pass: per-sweep voltage-to-frequency averaging and
normalization.

Reads the raw per-sample traces exported by odmr_labview_replica.py
(--save-traces -> odmr_labview_replica_traces.csv, columns run_index,
time_s, ch1_V, ch2_V) and, for each sweep:

  1. Convert CH2 (VCO tuning voltage) to frequency directly - voltage, not
     time, is the independent variable here, since CH2 is the actual
     per-sample tuning voltage rather than an assumed-linear time ramp:

         freq_GHz = V_TO_F_SLOPE_GHZ_PER_V * V_ch2 + V_TO_F_INTERCEPT_GHZ

     (same calibration as odmr_averaged_sweep.py). Samples are sorted by
     frequency first, since per-sample scope noise on CH2 can make the raw
     voltage-to-frequency mapping non-monotonic within a sweep.
  2. Interpolate that sweep's CH1 (detector) signal onto a common frequency
     grid.

Then, across all sweeps:

  2. Average: I(f) = (1/N) * sum_i I_i(f), with per-frequency standard
     deviation as well.
  3. Normalize: contrast(f) = 100 * (I(f) - I_max) / I_max, where I_max is
     the averaged curve's own maximum (assumed off-resonance baseline), so
     the baseline reads as 0% and the resonance dip reads as %-below-baseline.

Requires: pip install numpy matplotlib

All CSV output goes to csv_output/, all PNG output to png_output/ (created
automatically if missing) - matching odmr_labview_replica.py's layout.

Usage:
  python odmr_labview_voltage_freq.py
  python odmr_labview_voltage_freq.py --input csv_output/odmr_labview_replica_traces.csv --grid-points 500
"""

import argparse
import csv
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
OUTPUT_CSV = os.path.join(CSV_DIR, "odmr_labview_voltage_freq.csv")
OUTPUT_PNG = os.path.join(PNG_DIR, "odmr_labview_voltage_freq.png")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Raw traces CSV from odmr_labview_replica.py --save-traces.")
    p.add_argument("--grid-points", type=int, default=500, help="Number of points in the common frequency grid.")
    p.add_argument("--min-diff-mv", type=float, default=None,
                   help="Only include sweeps whose CH1 max-min difference exceeds this threshold (mV) - "
                        "same quality filter as odmr_labview_replica.py's --min-diff-mv. Unset: use all sweeps.")
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


def main():
    args = parse_args()
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(PNG_DIR, exist_ok=True)
    sweeps = load_sweeps(args.input)
    n_loaded = len(sweeps)
    print(f"Loaded {n_loaded} sweeps from {args.input}")

    output_csv, output_png = OUTPUT_CSV, OUTPUT_PNG
    if args.min_diff_mv is not None:
        sweeps = {
            run_idx: (ch1_v, ch2_v)
            for run_idx, (ch1_v, ch2_v) in sweeps.items()
            if (np.max(ch1_v) - np.min(ch1_v)) * 1e3 > args.min_diff_mv
        }
        print(f"Kept {len(sweeps)}/{n_loaded} sweeps with CH1 diff > {args.min_diff_mv:.0f} mV")
        if not sweeps:
            raise SystemExit("No sweeps passed the --min-diff-mv filter - nothing to average.")
        root, ext = os.path.splitext(OUTPUT_CSV)
        output_csv = f"{root}_filtered{ext}"
        root, ext = os.path.splitext(OUTPUT_PNG)
        output_png = f"{root}_filtered{ext}"

    # Convert each sweep's tuning voltage to frequency, sorted ascending so
    # interpolation is well-defined despite per-sample noise on CH2.
    sweep_freqs = []
    sweep_intensities = []
    for ch1_v, ch2_v in sweeps.values():
        freq_ghz = V_TO_F_SLOPE_GHZ_PER_V * ch2_v + V_TO_F_INTERCEPT_GHZ
        order = np.argsort(freq_ghz)
        sweep_freqs.append(freq_ghz[order])
        sweep_intensities.append(ch1_v[order])

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
    contrast_pct = 100.0 * (i_avg - i_max) / i_max if i_max else np.zeros_like(i_avg)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["freq_GHz", "detector_V_avg", "detector_V_sd", "contrast_pct"])
        for fq, avg, sd, c in zip(freq_grid, i_avg, i_sd, contrast_pct):
            writer.writerow([fq, avg, sd, c])
    print(f"Saved averaged, voltage-derived spectrum to {output_csv}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(freq_grid, i_avg, linewidth=1)
    ax1.fill_between(freq_grid, i_avg - i_sd, i_avg + i_sd, alpha=0.25)
    ax1.set_ylabel("Detector output (V)")
    ax1.set_title(f"Averaged detector response vs. drive frequency (N={len(sweeps)} sweeps)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(freq_grid, contrast_pct, linewidth=1)
    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax2.set_xlabel("Microwave drive frequency (GHz)")
    ax2.set_ylabel("Contrast vs. baseline max (%)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    print(f"Saved plot to {output_png}")

    dip_idx = np.argmin(contrast_pct)
    print(
        f"\nBaseline (max) detector level: {i_max:.5f} V\n"
        f"Deepest dip: {contrast_pct[dip_idx]:.2f}% below baseline at {freq_grid[dip_idx]:.4f} GHz"
    )

    plt.show()


if __name__ == "__main__":
    main()
