"""
Fit each trial condition's ODMR spectrum to a sum of N Lorentzian dips
(the lab handout's Eq. 5 two-Lorentzian model, generalized to as many lines
as are actually resolved in that condition's spectrum):

    f(nu) = C * sum_i (Gamma/2)^2 / ((nu-nu_i)^2 + (Gamma/2)^2)

and use the OUTERMOST pair of fitted line centers to back out the magnetic
field at the sample, which is then compared to the current driving the
electromagnet.

Why adaptive N, not a fixed 2 or 4 lines: at low current only 2 dips are
resolved, matching Eq. 5 directly. Above ~0.4A, the diamond's 4 NV-axis
families project the applied field differently, and more than 2 (up to 4,
per the handout's Eq. 2) or even more sub-features become visible. A fixed
2-line fit forced onto a 4+-line spectrum finds a degenerate optimum -
either it averages across everything (drifting instead of widening as
current increases), or, if seeded from the outer dips, it can collapse into
two unrealistically narrow spikes sitting exactly on 2 points and treating
the rest as baseline. Fitting exactly as many lines as find_peaks resolves
avoids the model/data mismatch: each real dip gets its own line, contrast C
and linewidth Gamma are still shared across every line in a condition (as in
the handout's model), and the fit's R^2 reflects genuine fit quality rather
than a lines-count mismatch.

Our contrast_pct column is the fractional intensity drop expressed as a
(negative) percent, so the fitted model is
  contrast_pct(nu) = -100 * f(nu) + b
with a small constant offset b absorbing any residual baseline tilt.

Field extraction: regardless of how many total lines are resolved, the
OUTERMOST two always correspond to the NV centers whose axis is aligned with
the field (full splitting), per the handout's identification of dips 1 and 4
in Eq. (2) - the inner lines belong to the other 3 (of 4) tetrahedral axes,
which see a reduced (B/3) projection. So the outer-pair splitting is:
  nu_outer_max - nu_outer_min = 2 * sqrt(E^2 + (gamma_e * B)^2)
where E is the strain/electric-field splitting (present even at B=0) and
gamma_e = 28.025 GHz/T is the electron gyromagnetic ratio. E is taken from
the zero-current trial's own fit (assumed field-independent), then every
other trial's measured outer splitting is converted to B by solving for it.

Requires: pip install numpy scipy matplotlib

Usage:
  python odmr_bfield_fit.py --base-dir "csv_output/09172026 Outputs" \
      --coil-length-cm <winding length>
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from odmr_overlay_trials import BLUE_RAMP_HEX, GRIDLINE, load_conditions

PNG_DIR = "png_output"
CSV_DIR = "csv_output"

GAMMA_E_GHZ_PER_T = 28.025  # electron gyromagnetic ratio / 2pi
MU0 = 4 * np.pi * 1e-7  # T*m/A

SERIES_MEASURED = "#2a78d6"  # categorical slot 1 (identity: measured)
SERIES_THEORY = "#eb6834"    # categorical slot 2 (identity: theory)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True,
                   help="Directory whose immediate subfolders are the conditions to fit, e.g. "
                        "'csv_output/<date> Outputs' containing 0A/, 0.100A/, ...")
    p.add_argument("--unit", default="A", help="Unit suffix stripped from folder names (default: 'A').")
    p.add_argument("--zero-label", default="0A",
                   help="Label of the zero-current condition, used as the B=0 strain-splitting "
                        "reference (default: '0A').")
    p.add_argument("--coil-turns", type=float, default=470,
                   help="Number of turns in the electromagnet coil (default: 470).")
    p.add_argument("--coil-diameter-cm", type=float, default=3.2,
                   help="Coil tube diameter in cm (default: 3.2). Only affects the finite-solenoid "
                        "correction term; negligible once length >> diameter.")
    p.add_argument("--coil-length-cm", type=float, default=None,
                   help="Winding length in cm (the span of tube actually wound with wire). Required "
                        "to draw the theoretical B-vs-current line; omitted by default since it "
                        "wasn't given.")
    p.add_argument("--ncols", type=int, default=5, help="Grid columns for the per-condition fit plot.")
    p.add_argument("--max-lines", type=int, default=6,
                   help="Cap on how many Lorentzian lines to fit per condition, even if more dips are "
                        "detected (default: 6). Keeps the fit from chasing noise wiggles as if they "
                        "were real resolved transitions.")
    return p.parse_args()


def n_lorentzian_pct(nu, *params):
    """params = [C, gamma, b, nu_1, nu_2, ..., nu_N] - one shared contrast and
    linewidth (as in the handout's model), one center per resolved dip."""
    C, gamma, b = params[0], params[1], params[2]
    centers = params[3:]
    half_g_sq = (gamma / 2) ** 2
    total = np.zeros_like(nu, dtype=float)
    for nu_i in centers:
        total += half_g_sq / ((nu - nu_i) ** 2 + half_g_sq)
    return -100.0 * C * total + b


def detect_dip_centers(freq, contrast, max_lines):
    """Seed one line center per resolved dip (not just the 2 deepest) - see
    module docstring for why fitting the true number of dips, rather than a
    fixed 2 or 4, avoids the model/data mismatch that caused the 2-line fit
    to either drift or collapse to degenerate narrow spikes once more than 2
    real dips are present. Falls back to a symmetric 2-line guess around the
    single deepest point when fewer than 2 dips are resolved."""
    depth = -contrast
    prominence = max(0.08 * depth.max(), 1e-6)
    min_distance = max(int(0.01 * len(freq)), 3)
    peaks, props = find_peaks(depth, prominence=prominence, distance=min_distance)
    if len(peaks) < 2:
        i = int(np.argmax(depth))
        return np.array(sorted([freq[i] - 0.003, freq[i] + 0.003]))
    if len(peaks) > max_lines:
        order = np.argsort(props["prominences"])[::-1]
        peaks = np.sort(peaks[order[:max_lines]])
    return freq[peaks]


def fit_condition(freq, contrast, max_lines=6):
    centers0 = detect_dip_centers(freq, contrast, max_lines)
    n_lines = len(centers0)
    depth = -contrast
    gamma0 = 0.01
    C0 = max(depth.max() / 100.0, 1e-4)
    b0 = float(np.median(contrast))

    # A near-zero floor on Gamma lets the optimizer collapse a line to an
    # unrealistically narrow spike sitting exactly on one data point,
    # ignoring the dip's actual (much broader) shape - a degenerate fit that
    # technically lands on the right frequency but represents nothing
    # physical. Flooring Gamma at the linewidth actually seen in the clean
    # low-current spectra (~0.01 GHz) rules that degenerate solution out.
    p0 = [C0, gamma0, b0] + list(centers0)
    lo = [0.0, 0.008, -5.0] + [freq.min()] * n_lines
    hi = [1.0, 0.05, 5.0] + [freq.max()] * n_lines
    popt, pcov = curve_fit(n_lorentzian_pct, freq, contrast, p0=p0, bounds=(lo, hi), maxfev=60000)

    residual = contrast - n_lorentzian_pct(freq, *popt)
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((contrast - np.mean(contrast)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return popt, pcov, r2, n_lines


def outer_splitting_and_uncertainty(popt, pcov):
    """Separation between the two OUTERMOST fitted line centers (regardless
    of how many total lines were fit) and its 1-sigma uncertainty. The outer
    pair always corresponds to the field-aligned NV axis population (Eq. 2's
    dips 1 and 4), so this is the quantity that maps directly to B with no
    projection factor."""
    centers = popt[3:]
    idx_min = int(np.argmin(centers)) + 3
    idx_max = int(np.argmax(centers)) + 3
    splitting = popt[idx_max] - popt[idx_min]
    var = pcov[idx_max, idx_max] + pcov[idx_min, idx_min] - 2 * pcov[idx_max, idx_min]
    sigma = np.sqrt(var) if var > 0 else float("nan")
    return splitting, sigma


def field_from_splitting(splitting_ghz, sigma_splitting_ghz, e0_ghz, sigma_e0_ghz):
    """Solve nu_+ - nu_- = 2*sqrt(E^2 + (gamma_e*B)^2) for B, in mT, with
    1-sigma uncertainty propagated from the splitting and E0 uncertainties."""
    gamma_e_ghz_per_mt = GAMMA_E_GHZ_PER_T / 1000.0
    half_s = splitting_ghz / 2.0
    inside = half_s ** 2 - e0_ghz ** 2
    if inside <= 0:
        return 0.0, 0.0  # B pinned to 0 (splitting below the strain floor); no meaningful uncertainty
    b = np.sqrt(inside) / gamma_e_ghz_per_mt
    d_b_d_s = (half_s / 2.0) / (gamma_e_ghz_per_mt * np.sqrt(inside))
    d_b_d_e0 = -e0_ghz / (gamma_e_ghz_per_mt * np.sqrt(inside))
    sigma_b = np.sqrt((d_b_d_s * sigma_splitting_ghz) ** 2 + (d_b_d_e0 * sigma_e0_ghz) ** 2)
    return b, sigma_b


def solenoid_end_field_mt(current_a, turns, length_m, radius_m):
    """On-axis field at the very end of a finite solenoid (the sample sits
    right at the coil's end, not its center, so this is half - and a bit
    less, via the finite-radius correction - of the long-solenoid interior
    field)."""
    return MU0 * turns * current_a / (2 * np.sqrt(length_m ** 2 + radius_m ** 2)) * 1000.0


def plot_fit_grid(conditions, fits, ncols):
    n = len(conditions)
    ncols = max(1, ncols)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.8 * nrows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (value, label, freq, contrast), (popt, pcov, r2, n_lines) in zip(axes_flat, conditions, fits):
        ax.plot(freq, contrast, linewidth=1.2, color="#898781", zorder=1, label="data")
        nu_fit = np.linspace(freq.min(), freq.max(), 600)
        ax.plot(nu_fit, n_lorentzian_pct(nu_fit, *popt), linewidth=1.6,
                 color=SERIES_MEASURED, zorder=2, label="fit")
        ax.set_title(f"{label}  (N={n_lines}, R²={r2:.3f})", fontsize=9, color="#52514e")
        ax.grid(True, axis="x", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)
    axes_flat[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.supxlabel("Microwave drive frequency (GHz)")
    fig.supylabel("Contrast vs. baseline max (%)")
    fig.suptitle("Adaptive N-Lorentzian fit per condition")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    return fig


def plot_b_vs_current(rows, theory_fn, unit):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    amps = np.array([r["current"] for r in rows])
    b_meas = np.array([r["B_measured_mT"] for r in rows])
    b_unc = np.array([r["B_unc_mT"] for r in rows])
    ax.errorbar(amps, b_meas, yerr=b_unc, fmt="o", color=SERIES_MEASURED, ecolor=SERIES_MEASURED,
                elinewidth=1, capsize=3, markersize=5, zorder=3, label="Measured (ODMR fit)")

    if theory_fn is not None:
        i_line = np.linspace(0, max(amps.max(), 1e-9), 200)
        ax.plot(i_line, theory_fn(i_line), linewidth=1.6, linestyle="--",
                 color=SERIES_THEORY, zorder=2, label="Theory (solenoid end field)")

    # Scale the y-axis to the data points themselves, not the error bars - a
    # single poorly-conditioned fit (e.g. an odd line count with one weakly
    # constrained outer line) can give an absurdly large formal uncertainty
    # that would otherwise squash every other point into an unreadable strip.
    y_lo = min(0.0, b_meas.min())
    y_hi = b_meas.max()
    y_pad = 0.15 * max(y_hi - y_lo, 1e-6)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    ax.set_xlabel(f"Current ({unit})" if unit else "Current")
    ax.set_ylabel("Magnetic field at sample (mT)")
    ax.set_title("ODMR-derived field vs. drive current")
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    os.makedirs(PNG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    conditions = load_conditions(args.base_dir, args.unit)
    fits = [fit_condition(freq, contrast, args.max_lines) for _, _, freq, contrast in conditions]

    zero_idx = next((i for i, c in enumerate(conditions) if c[1] == args.zero_label), None)
    if zero_idx is None:
        raise SystemExit(f"Zero-current condition '{args.zero_label}' not found among "
                          f"{[c[1] for c in conditions]}.")
    s0, sigma_s0 = outer_splitting_and_uncertainty(fits[zero_idx][0], fits[zero_idx][1])
    e0 = s0 / 2.0
    sigma_e0 = sigma_s0 / 2.0
    print(f"Zero-field strain splitting from {args.zero_label}: "
          f"2E = {s0*1e3:.3f} +/- {sigma_s0*1e3:.3f} MHz")

    rows = []
    print(f"\n{'label':>8}  {'N':>2}  {'C':>7}  {'D (GHz)':>9}  {'E (GHz)':>9}  {'Gamma (GHz)':>11}  "
          f"{'R2':>6}  {'B (mT)':>8}")
    for (value, label, freq, contrast), (popt, pcov, r2, n_lines) in zip(conditions, fits):
        C, gamma, b = popt[0], popt[1], popt[2]
        centers = popt[3:]
        nu_plus, nu_minus = max(centers), min(centers)
        splitting, sigma_s = outer_splitting_and_uncertainty(popt, pcov)
        b_field, sigma_b = field_from_splitting(splitting, sigma_s, e0, sigma_e0)
        d_ghz = (nu_plus + nu_minus) / 2.0  # midpoint, matching the paper's D (zero-field splitting)
        e_ghz = splitting / 2.0             # matching the paper's E (strain/Zeeman-combined half-splitting)
        rows.append({
            "label": label, "current": value, "n_lines": n_lines, "C": C,
            "nu_plus_GHz": nu_plus, "nu_minus_GHz": nu_minus,
            "D_GHz": d_ghz, "E_GHz": e_ghz,
            "splitting_MHz": splitting * 1e3, "splitting_unc_MHz": sigma_s * 1e3,
            "Gamma_GHz": gamma, "r_squared": r2,
            "B_measured_mT": b_field, "B_unc_mT": sigma_b,
        })
        print(f"{label:>8}  {n_lines:>2}  {C:7.4f}  {d_ghz:9.4f}  {e_ghz:9.4f}  {gamma:11.4f}  "
              f"{r2:6.3f}  {b_field:8.4f}")

    base_name = re.sub(r"\s+", "_", os.path.basename(args.base_dir.rstrip("/\\")))

    csv_path = os.path.join(CSV_DIR, f"odmr_bfield_fit_{base_name}.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        f.write(",".join(fieldnames) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in fieldnames) + "\n")
    print(f"Saved fit results to {csv_path}")

    grid_fig = plot_fit_grid(conditions, fits, args.ncols)
    grid_path = os.path.join(PNG_DIR, f"odmr_bfield_fits_{base_name}.png")
    grid_fig.savefig(grid_path, dpi=200)
    print(f"Saved per-condition fit grid to {grid_path}")

    theory_fn = None
    if args.coil_length_cm is not None:
        length_m = args.coil_length_cm / 100.0
        radius_m = (args.coil_diameter_cm / 100.0) / 2.0
        theory_fn = lambda i_a: solenoid_end_field_mt(i_a, args.coil_turns, length_m, radius_m)
    else:
        print("No --coil-length-cm given: skipping the theoretical B-vs-current overlay.")

    bvi_fig = plot_b_vs_current(rows, theory_fn, args.unit)
    bvi_path = os.path.join(PNG_DIR, f"odmr_bfield_vs_current_{base_name}.png")
    bvi_fig.savefig(bvi_path, dpi=200)
    print(f"Saved B-vs-current comparison plot to {bvi_path}")


if __name__ == "__main__":
    main()
