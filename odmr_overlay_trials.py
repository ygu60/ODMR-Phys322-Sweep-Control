"""
Overlay every trial condition's averaged ODMR spectrum on one plot,
normalized to percent contrast (not raw voltage), so conditions with wildly
different absolute baseline levels can be visually compared on a single
figure - intended as a publication-ready summary figure for a paper.

Reads odmr_voltage_freq_analysis.py's output spectrum CSVs (columns
freq_GHz, detector_V_avg, detector_V_sd, contrast_pct, relative_noise_pct)
and plots contrast_pct vs. freq_GHz for every condition found. Raw voltage
isn't comparable across conditions here (autoscale/detector gain differs
run to run), but contrast_pct is already normalized to each condition's own
baseline, so it's the right quantity to overlay.

Conditions are color-coded on a single-hue sequential ("ordinal") blue ramp
(light = low value, dark = high value) with a colorbar, not a per-line
legend - a legend with 10 tiny color swatches is hard to match to lines by
eye, and the conditions are a numerically ordered physical quantity (e.g.
increasing current), not arbitrary categories, so color-as-position is the
right encoding (see the project's dataviz-skill notes: one hue, light->dark,
for ordinal data).

Auto-discovers one condition per immediate subfolder of --base-dir (the
subfolder name is used as both the label and, after stripping --unit, the
numeric value used for sorting/coloring - e.g. "0.300A" -> 0.3). Within each
subfolder, if multiple analysis-output spectra exist (e.g. from re-running
the analysis), picks the one with the largest N (most kept sweeps).

Produces two figures:
  - A grid of small multiples (2 rows x 5 columns for the 10-condition case),
    one condition per panel, sharing one y-scale so the shrinking dip depth
    reads directly from panel to panel - past ~4 converging series on one
    axis, small multiples is the right call (dataviz-skill anti-patterns:
    converging end-labels turn into noise). The x-axis is cropped to the
    union of each condition's own feature region (auto-detected, not
    hardcoded), not the full swept range, since most of every sweep is flat
    baseline that just wastes width in a small panel.
  - The original single-axis overlay, all conditions on one shared frequency
    scale with a colorbar - useful for comparing absolute frequency
    alignment across conditions, which the grid's per-panel framing doesn't
    show as directly.

Requires: pip install numpy matplotlib

Usage:
  python odmr_overlay_trials.py --base-dir "csv_output/09172026 Outputs"
"""

import argparse
import csv
import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

PNG_DIR = "png_output"

# Sequential blue ramp (light -> dark), ordinal steps 250-700 from the
# project's validated palette (dataviz skill, references/palette.md) -
# one hue, monotone lightness, light end still clears the 2:1 contrast
# floor for an ordinal ramp on a light surface.
BLUE_RAMP_HEX = [
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True,
                   help="Directory whose immediate subfolders are the conditions to overlay, e.g. "
                        "'csv_output/<date> Outputs' containing 0A/, 0.100A/, ... as produced by "
                        "odmr_averaged_sweep.py/odmr_voltage_freq_analysis.py's FIELD_LABEL folders.")
    p.add_argument("--unit", default="A",
                   help="Unit suffix stripped from folder names to parse the numeric condition value "
                        "for coloring/sorting (default: 'A', e.g. '0.300A' -> 0.3).")
    p.add_argument("--output", default=None,
                   help="Output PNG path for the grid plot. Default: "
                        "png_output/odmr_overlay_grid_<base-dir name>.png. The single-axis overlay "
                        "always goes to png_output/odmr_overlay_single_<base-dir name>.png.")
    p.add_argument("--ncols", type=int, default=5,
                   help="Number of grid columns (default: 5, giving 2 rows for 10 conditions).")
    p.add_argument("--feature-frac", type=float, default=0.2,
                   help="Fraction of each condition's own peak deviation from its median baseline "
                        "used to detect that condition's feature region for the shared, cropped x-axis "
                        "(default: 0.2). Lower = wider auto-detected region (more sensitive to noise "
                        "in the shallowest conditions); higher = tighter crop.")
    return p.parse_args()


def find_spectrum_csv(folder):
    """Return the analysis-output spectrum CSV in this folder with the most
    kept sweeps (largest N in its filename), or None if none found. Skips
    raw-traces files (run_index/time_s/ch1_V/ch2_V columns) and per-run
    summary files (run_index/ch1_max_V/... columns) by checking the header -
    only a spectrum CSV starts with freq_GHz."""
    best_path, best_n = None, -1
    for name in os.listdir(folder):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(folder, name)
        try:
            with open(path, newline="") as f:
                header = next(csv.reader(f))
        except (StopIteration, OSError):
            continue
        if not header or header[0] != "freq_GHz":
            continue
        m = re.search(r"_N(\d+)_", name)
        n = int(m.group(1)) if m else 0
        if n > best_n:
            best_path, best_n = path, n
    return best_path


def load_spectrum(path):
    freq, contrast = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        fi = header.index("freq_GHz")
        ci = header.index("contrast_pct")
        for row in reader:
            freq.append(float(row[fi]))
            contrast.append(float(row[ci]))
    return np.array(freq), np.array(contrast)


def feature_x_range(conditions, feature_frac):
    """Union, across conditions, of each condition's own region where it
    deviates from its own median baseline by more than feature_frac of its
    own peak deviation - a data-driven crop for the shared x-axis instead of
    a hardcoded GHz window that would need updating as the feature widens
    with current."""
    ranges = []
    for _, _, freq, contrast in conditions:
        baseline = np.median(contrast)
        deviation = np.abs(contrast - baseline)
        peak = deviation.max()
        if peak <= 0:
            continue
        mask = deviation > feature_frac * peak
        if np.any(mask):
            ranges.append((freq[mask].min(), freq[mask].max()))
    data_lo = min(np.min(c[2]) for c in conditions)
    data_hi = max(np.max(c[2]) for c in conditions)
    if not ranges:
        return data_lo, data_hi
    x_lo = min(r[0] for r in ranges)
    x_hi = max(r[1] for r in ranges)
    pad = 0.2 * (x_hi - x_lo)
    return max(data_lo, x_lo - pad), min(data_hi, x_hi + pad)


def plot_grid(conditions, cmap, norm, ncols, feature_frac):
    """Small multiples: one panel per condition, sharing a y-scale so the
    shrinking dip depth is directly comparable panel to panel, laid out in a
    grid instead of one axis (past ~4 converging series, small multiples
    beats direct end-labels or a legend - dataviz-skill anti-patterns), with
    the shared x-axis cropped to where the features actually are."""
    n = len(conditions)
    ncols = max(1, ncols)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows),
        sharex=True, sharey=True,
    )
    axes_flat = np.atleast_1d(axes).flatten()

    x_lo, x_hi = feature_x_range(conditions, feature_frac)

    for ax, (value, label, freq, contrast) in zip(axes_flat, conditions):
        color = cmap(norm(value))
        ax.plot(freq, contrast, linewidth=1.6, color=color, zorder=2)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=0)
        ax.set_title(label, fontsize=10, color="#52514e")
        ax.grid(True, axis="x", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    axes_flat[0].set_xlim(x_lo, x_hi)
    fig.supxlabel("Microwave drive frequency (GHz)")
    fig.supylabel("Contrast vs. baseline max (%)")
    fig.suptitle("ODMR contrast vs. drive frequency across conditions")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    return fig


def plot_single_overlay(conditions, cmap, norm, unit):
    """All conditions on one shared axis, color-coded on the ordinal blue
    ramp with a colorbar (not a 10-swatch legend, which is hard to match to
    lines by eye) - the original overlay view, kept alongside the small-
    multiples grid since the two answer different questions: this one shows
    every curve's absolute frequency alignment on one scale, the grid shows
    per-condition shape/depth clearly without overlap."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for value, label, freq, contrast in conditions:
        ax.plot(freq, contrast, linewidth=1.6, color=cmap(norm(value)), zorder=2)
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=0)
    ax.set_xlabel("Microwave drive frequency (GHz)")
    ax.set_ylabel("Contrast vs. baseline max (%)")
    ax.set_title("ODMR contrast vs. drive frequency across conditions")
    ax.grid(True, axis="x", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    values = [c[0] for c in conditions]
    cbar = fig.colorbar(sm, ax=ax, ticks=values)
    cbar.ax.set_yticklabels([c[1] for c in conditions])
    cbar.set_label(f"Condition ({unit})" if unit else "Condition")
    cbar.outline.set_visible(False)

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    os.makedirs(PNG_DIR, exist_ok=True)

    conditions = []  # (numeric_value, label, freq, contrast)
    for name in sorted(os.listdir(args.base_dir)):
        folder = os.path.join(args.base_dir, name)
        if not os.path.isdir(folder):
            continue
        csv_path = find_spectrum_csv(folder)
        if csv_path is None:
            print(f"Skipping {folder}: no analysis-output spectrum CSV found.")
            continue
        numeric_str = name[: -len(args.unit)] if args.unit and name.endswith(args.unit) else name
        try:
            value = float(numeric_str)
        except ValueError:
            value = float(len(conditions))  # fall back to discovery order
        freq, contrast = load_spectrum(csv_path)
        conditions.append((value, name, freq, contrast))
        print(f"Loaded {name} from {csv_path} ({len(freq)} points)")

    if not conditions:
        raise SystemExit(f"No conditions with a spectrum CSV found under {args.base_dir}.")

    conditions.sort(key=lambda c: c[0])
    values = [c[0] for c in conditions]
    vmin, vmax = min(values), max(values)
    # Guard a single-condition or all-equal-value edge case (Normalize needs
    # vmin != vmax).
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    cmap = mcolors.LinearSegmentedColormap.from_list("blue_ordinal", BLUE_RAMP_HEX)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    base_name = re.sub(r"\s+", "_", os.path.basename(args.base_dir.rstrip("/\\")))

    grid_fig = plot_grid(conditions, cmap, norm, args.ncols, args.feature_frac)
    grid_path = args.output or os.path.join(PNG_DIR, f"odmr_overlay_grid_{base_name}.png")
    grid_fig.savefig(grid_path, dpi=200)
    print(f"Saved small-multiples grid plot to {grid_path}")

    overlay_fig = plot_single_overlay(conditions, cmap, norm, args.unit)
    overlay_path = os.path.join(PNG_DIR, f"odmr_overlay_single_{base_name}.png")
    overlay_fig.savefig(overlay_path, dpi=200)
    print(f"Saved single-axis overlay plot to {overlay_path}")

    plt.show()


if __name__ == "__main__":
    main()
