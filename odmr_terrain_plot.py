"""
3D terrain surface of ODMR contrast vs. drive frequency and current, built
from the same per-condition spectrum CSVs as odmr_overlay_trials.py.

Each condition has its own frequency sampling (freq_GHz differs slightly run
to run), so before a surface can be drawn every condition's contrast_pct is
linearly interpolated (np.interp) onto one shared frequency grid. That grid
is clipped to the *intersection* of all conditions' measured frequency
ranges, never extrapolated past what a condition actually swept.

The current (amperage) axis is only 10 real measurements wide. matplotlib's
plot_surface draws each patch between adjacent rows as a flat bilinear quad -
exactly "linear interpolation only between adjacent measured amperages," with
nothing smoothed or fabricated between them. No spline/smoothing is applied
across the amperage axis for that reason: a smooth spline through 10 points
would imply structure between conditions that was never measured.

Requires: pip install numpy matplotlib

Usage:
  python odmr_terrain_plot.py --base-dir "csv_output/09172026 Outputs"
"""

import argparse
import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from odmr_overlay_trials import BLUE_RAMP_HEX, load_conditions

PNG_DIR = "png_output"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True,
                   help="Directory whose immediate subfolders are the conditions to plot, e.g. "
                        "'csv_output/<date> Outputs' containing 0A/, 0.100A/, ... as produced by "
                        "odmr_averaged_sweep.py/odmr_voltage_freq_analysis.py's FIELD_LABEL folders.")
    p.add_argument("--unit", default="A",
                   help="Unit suffix stripped from folder names to parse the numeric condition value "
                        "for the current axis (default: 'A', e.g. '0.300A' -> 0.3).")
    p.add_argument("--output", default=None,
                   help="Output PNG path. Default: png_output/odmr_terrain_<base-dir name>.png")
    p.add_argument("--n-freq-points", type=int, default=1200,
                   help="Number of points on the shared, interpolated frequency grid (default: 1200).")
    p.add_argument("--elev", type=float, default=28, help="3D view elevation angle in degrees (default: 28).")
    p.add_argument("--azim", type=float, default=-55, help="3D view azimuth angle in degrees (default: -55).")
    p.add_argument("--alpha", type=float, default=0.75,
                   help="Surface opacity, 0-1 (default: 0.75) - lower makes the far side of the "
                        "surface visible through the near side, which helps read the terrain's shape.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(PNG_DIR, exist_ok=True)

    conditions = load_conditions(args.base_dir, args.unit)
    if len(conditions) < 2:
        raise SystemExit("Need at least 2 conditions to build a surface across current.")

    amps = np.array([c[0] for c in conditions])

    # Intersection of every condition's measured frequency range - the grid
    # never extrapolates past what a condition actually swept.
    freq_lo = max(np.min(c[2]) for c in conditions)
    freq_hi = min(np.max(c[2]) for c in conditions)
    freq_grid = np.linspace(freq_lo, freq_hi, args.n_freq_points)

    z = np.array([np.interp(freq_grid, c[2], c[3]) for c in conditions])
    x, y = np.meshgrid(freq_grid, amps)

    # Sequential encoding for magnitude: light = near-zero contrast, dark =
    # the deepest dip - the ramp reversed so the darkest step lands on the
    # most negative z, matching the project's one-hue sequential convention.
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_terrain", list(reversed(BLUE_RAMP_HEX)))
    norm = mcolors.Normalize(vmin=z.min(), vmax=z.max())

    # cstride thins the drawn mesh along the (dense, interpolated) frequency
    # axis; rstride stays 1 since the amperage axis only has 10 real rows to
    # begin with. A visible hairline mesh (edgecolor) marks every quad
    # boundary so the grid structure - and which lines are the 10 real
    # measured rows - reads clearly instead of a flat-looking blob.
    cstride = max(1, args.n_freq_points // 200)

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, z, cmap=cmap, norm=norm, rstride=1, cstride=cstride,
                            edgecolor="#2c2c2a", linewidth=0.15, alpha=args.alpha,
                            antialiased=True)

    ax.set_xlabel("Microwave drive frequency (GHz)")
    ax.set_ylabel(f"Current ({args.unit})" if args.unit else "Current")
    ax.set_zlabel("Contrast vs. baseline max (%)")
    ax.set_title("ODMR contrast surface: frequency x current")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.grid(True)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Contrast vs. baseline max (%)")
    cbar.outline.set_visible(False)

    fig.tight_layout()
    base_name = re.sub(r"\s+", "_", os.path.basename(args.base_dir.rstrip("/\\")))
    output_path = args.output or os.path.join(PNG_DIR, f"odmr_terrain_{base_name}.png")
    fig.savefig(output_path, dpi=200)
    print(f"Saved terrain surface plot to {output_path}")


if __name__ == "__main__":
    main()
