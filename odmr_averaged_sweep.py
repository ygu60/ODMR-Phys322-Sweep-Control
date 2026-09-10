"""
ODMR averaged-sweep capture.

Collects several hundred single-shot sweeps of:
  CHAN1 - PDA36A2 amplifier/detector output (the fluorescence-proportional signal)
  CHAN2 - the 33220A sawtooth drive (V_tune to the VCO)
triggered so every sweep starts at the same point in the ramp, averages them
point-by-point to beat down noise, converts the averaged sawtooth voltage into
microwave drive frequency, and plots detector output vs. frequency looking for
a resonance dip (analogous to the ODMR literature - e.g. Zhang et al., Am. J.
Phys. 86, 225 (2018), which reports an ~8% fluorescence dip on resonance).

This is a dedicated DATA-COLLECTION script, distinct from the live console
monitor - because averaging requires hundreds of individually-triggered
:DIGITIZE acquisitions, the scope will be taken out of continuous run for the
duration of the capture (each :DIGITIZE arms, waits for the sawtooth trigger,
and grabs one sweep). Normal continuous display is restored at the end.

Requires: pip install pyvisa pyvisa-py numpy matplotlib

Usage:
  python odmr_averaged_sweep.py
"""

import csv
import time
import numpy as np
import pyvisa
import matplotlib.pyplot as plt

# --- Instrument addresses ---
SCOPE_ADDR = "USB0::0x2A8D::0x0396::CN63257664::0::INSTR"
AWG_ADDR = "GPIB0::5::INSTR"

# --- Sawtooth drive on the 33220A (VCO tuning voltage sweep) ---
FREQ_HZ = 100            # sawtooth sweep rate
AMPL_VPP = 1.5           # 1.5 V peak-to-peak
OFFSET_V = 4.0           # 4 V offset -> ramps 3.25 V to 4.75 V
RAMP_SYMMETRY_PCT = 100  # 100 = rising sawtooth
RAMP_PERIOD_S = 1.0 / FREQ_HZ
RAMP_VMIN = OFFSET_V - AMPL_VPP / 2.0
RAMP_VMAX = OFFSET_V + AMPL_VPP / 2.0

# --- Voltage -> RF frequency calibration (from band-edge measurements) ---
V_TO_F_SLOPE_GHZ_PER_V = 0.1201024911
V_TO_F_INTERCEPT_GHZ = 1.933

# --- Averaging ---
NUM_AVERAGES = 100      # "several hundred" sweeps
POINTS_PER_SWEEP = 2000  # per-acquisition record length (kept modest so 300+
                         # USB transfers don't take forever)

# --- Scope channel settings ---
CH1_COUPLING = "DC"

# Capture slightly less than one full period. The flyback isn't perfectly
# instantaneous (finite AWG reset time, plus small period mismatch between the
# AWG's actual output and the host-computed RAMP_PERIOD_S), so a window sized
# to exactly one period ends up running past the reset and picking up the
# leading rise of the next cycle. Trimming the window leaves margin so it
# always ends before that happens, at the cost of a bit of the ramp's tail.
CAPTURE_FRACTION_OF_PERIOD = 0.93
TIMEBASE_RANGE_S = RAMP_PERIOD_S * CAPTURE_FRACTION_OF_PERIOD

# Trigger just after the ramp resets (slightly above the ramp's minimum) on a
# rising edge of CHAN2, so every capture starts at the same phase of the sweep.
TRIGGER_LEVEL_V = RAMP_VMIN

TIMEOUT_MS = 5000

OUTPUT_CSV = "odmr_averaged_sweep.csv"
OUTPUT_PNG = "odmr_averaged_sweep.png"


def setup_awg(inst):
    inst.write("*RST")
    time.sleep(0.5)
    inst.write("FUNC RAMP")
    inst.write(f"FUNC:RAMP:SYMMETRY {RAMP_SYMMETRY_PCT}")
    inst.write(f"FREQ {FREQ_HZ}")
    inst.write(f"VOLT {AMPL_VPP}")
    inst.write(f"VOLT:OFFS {OFFSET_V}")
    inst.write("OUTP ON")
    err = inst.query("SYST:ERR?").strip()
    print(
        f"[33220A] Rising sawtooth {FREQ_HZ} Hz, {AMPL_VPP} Vpp, {OFFSET_V} V offset "
        f"({RAMP_VMIN:.3f}-{RAMP_VMAX:.3f} V). SYST:ERR? -> {err}"
    )


def setup_scope(inst):
    inst.write(":CHAN1:DISPLAY ON")
    inst.write(":CHAN2:DISPLAY ON")

    # Force DC coupling on CHAN1 before autoscaling, so autoscale sizes the
    # vertical scale/offset for the actual DC-coupled detector signal rather
    # than whatever coupling was left over from a previous run.
    inst.write(f":CHAN1:COUPLING {CH1_COUPLING}")

    # Autoscale both channels so CHAN1's V/div and offset are sized to
    # whatever the detector signal actually is, not a guessed fixed range.
    inst.write(":AUTOSCALE CHAN1,CHAN2")
    time.sleep(2)

    ch1_scale = float(inst.query(":CHAN1:SCALE?").strip())
    ch1_offset = float(inst.query(":CHAN1:OFFSET?").strip())

    # Deterministic timebase: one clean rising ramp per trigger, starting
    # right at the trigger point (reference at the left edge of the record).
    inst.write(f":TIMEBASE:RANGE {TIMEBASE_RANGE_S}")
    inst.write(":TIMEBASE:REFERENCE LEFT")
    inst.write(":TIMEBASE:POSITION 0")

    # Trigger on CHAN2 rising through the start of the ramp.
    inst.write(":TRIGGER:MODE EDGE")
    inst.write(":TRIGGER:EDGE:SOURCE CHAN2")
    inst.write(f":TRIGGER:EDGE:LEVEL {TRIGGER_LEVEL_V}")
    inst.write(":TRIGGER:EDGE:SLOPE POSITIVE")

    # Max vertical resolution per acquisition, modest record length so 300+
    # single-shot transfers over USB stay reasonably fast.
    inst.write(":ACQUIRE:TYPE HRESOLUTION")
    inst.write(":WAV:FORMAT WORD")
    inst.write(":WAV:BYTEORDER LSBFIRST")
    inst.write(":WAV:POINTS:MODE NORMAL")
    inst.write(f":WAV:POINTS {POINTS_PER_SWEEP}")

    err = inst.query("SYST:ERR?").strip()
    print(
        f"[DSOX1204G] Timebase {TIMEBASE_RANGE_S * 1e3:.3f} ms, trigger CHAN2 "
        f"rising @ {TRIGGER_LEVEL_V:.3f} V, High-Res, {POINTS_PER_SWEEP} pts/sweep. "
        f"CHAN1 autoscaled to {ch1_scale * 1e3:.2f} mV/div, {ch1_offset * 1e3:.2f} mV offset. "
        f"SYST:ERR? -> {err}"
    )


def acquire_single_sweep(inst):
    """Arm a single trigger, wait for it, digitize CHAN1+CHAN2 together, and
    return {ch: (t, v)} for both channels from this one sweep."""
    inst.write(":DIGITIZE CHAN1,CHAN2")  # blocks until acquisition completes

    results = {}
    for ch in (1, 2):
        inst.write(f":WAV:SOURCE CHAN{ch}")
        preamble = inst.query(":WAV:PREAMBLE?").strip().split(",")
        xincrement = float(preamble[4])
        xorigin = float(preamble[5])
        xreference = float(preamble[6])
        yincrement = float(preamble[7])
        yorigin = float(preamble[8])
        yreference = float(preamble[9])

        raw = inst.query_binary_values(":WAV:DATA?", datatype="h", is_big_endian=False, container=np.array)

        t = (np.arange(len(raw)) - xreference) * xincrement + xorigin
        v = (raw.astype(float) - yreference) * yincrement + yorigin
        results[ch] = (t, v)

    return results


def main():
    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(SCOPE_ADDR)
    scope.timeout = TIMEOUT_MS
    awg = rm.open_resource(AWG_ADDR)
    awg.timeout = TIMEOUT_MS

    print("[DSOX1204G] IDN:", scope.query("*IDN?").strip())
    print("[33220A] IDN:", awg.query("*IDN?").strip())

    setup_awg(awg)
    setup_scope(scope)

    t_ref = None
    sum_v1 = None
    sum_v2 = None
    n_captured = 0

    print(f"\nCapturing {NUM_AVERAGES} triggered sweeps and averaging...\n")

    try:
        for i in range(NUM_AVERAGES):
            data = acquire_single_sweep(scope)
            t1, v1 = data[1]
            t2, v2 = data[2]

            if t_ref is None:
                n = len(v1)
                t_ref = t1[:n]
                sum_v1 = np.zeros(n)
                sum_v2 = np.zeros(n)

            n = min(len(v1), len(v2), len(sum_v1))
            sum_v1[:n] += v1[:n]
            sum_v2[:n] += v2[:n]
            n_captured += 1

            if (i + 1) % 25 == 0 or (i + 1) == NUM_AVERAGES:
                print(f"\r  {i + 1}/{NUM_AVERAGES} sweeps captured", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\nStopped early after {n_captured} sweeps - averaging what we have.")
    finally:
        try:
            awg.write("OUTP OFF")
        except Exception:
            pass
        scope.write(":RUN")  # restore normal continuous display
        scope.close()
        awg.close()
        rm.close()

    if n_captured == 0:
        print("No sweeps captured - nothing to plot.")
        return

    avg_v1 = sum_v1 / n_captured  # averaged detector output
    avg_v2 = sum_v2 / n_captured  # averaged sawtooth (V_tune) - raw, for reference

    # The ramp is driven linearly in time by design, but per-sample scope
    # noise on CHAN2 is comparable in size to the ramp's tiny per-sample
    # voltage step, so avg_v2 itself is not perfectly monotonic. Using it
    # directly as the frequency axis makes freq_ghz wiggle backward from one
    # sample to the next, which shows up as zig-zag spikes when plotted
    # against the (also noisy) detector trace. Fit a line to avg_v2 vs. time
    # instead - since the ramp is linear by construction - to get a smooth,
    # strictly monotonic voltage (and therefore frequency) axis.
    ramp_slope, ramp_intercept = np.polyfit(t_ref, avg_v2, 1)
    v_tune_fit = ramp_slope * t_ref + ramp_intercept
    freq_ghz = V_TO_F_SLOPE_GHZ_PER_V * v_tune_fit + V_TO_F_INTERCEPT_GHZ

    # Deviation from the mean: subtract the averaged trace's own mean level
    # so the dip shows up as a negative excursion around zero.
    mean_v1 = np.mean(avg_v1)
    dev_v1 = avg_v1 - mean_v1

    # Save raw + deviation-from-mean averaged data for the lab report.
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "v_tune_V", "freq_GHz", "detector_V", "detector_V_dev_from_mean"])
        for t, vt, fq, vd, dv in zip(t_ref, avg_v2, freq_ghz, avg_v1, dev_v1):
            writer.writerow([t, vt, fq, vd, dv])
    print(f"\nSaved averaged data to {OUTPUT_CSV}")

    # Plot detector output's deviation from its mean vs. drive frequency - look for the resonance dip.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(freq_ghz, dev_v1, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Microwave drive frequency (GHz)")
    ax.set_ylabel("Detector output, deviation from mean (V)")
    ax.set_title(f"Detector response deviation from mean vs. drive frequency (N={n_captured} sweeps)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"Saved plot to {OUTPUT_PNG}")

    dip_idx = np.argmin(dev_v1)
    dip_depth_pct = 100 * (mean_v1 - avg_v1[dip_idx]) / mean_v1 if mean_v1 else float("nan")
    print(
        f"\nMean detector level: {mean_v1:.5f} V\n"
        f"Deepest dip: {dev_v1[dip_idx]:.5f} V below mean at {freq_ghz[dip_idx]:.4f} GHz "
        f"({dip_depth_pct:.2f}% below mean)"
    )

    plt.show()


if __name__ == "__main__":
    main()
