"""
General-purpose max/min contrast acquisition, replicating the LabVIEW VI
architecture described in the lab's IO Architecture instructions (Figures 9
and 10):

  - Function generator block (Fig. 9, top left): configures the AWG's
    waveform, frequency, amplitude, offset, ramp symmetry, and burst mode.
    Runs once at the start of the program. If --no-change-pulse is passed
    (the VI's "change pulse" button off), this block is skipped entirely and
    whatever the AWG is already outputting is left alone.

  - Oscilloscope control block (Fig. 9, remainder): configures CHAN1/CHAN2
    coupling, probe attenuation, on/off, vertical range/offset, acquisition
    type, and the trigger (source, level, slope, holdoff), then digitizes
    --runs single-shot acquisitions ("Runs" / "Current #").

  - Data-processing block (Fig. 10): for each run, records that waveform's
    max and min ("Average Max" / "Average Minimum" boxes acting on the
    current run), then across all runs computes the overall Average Max 2 /
    Average Minimum 2 / Difference (mV) / Percent Diff.

Unlike odmr_averaged_sweep.py (which is hardwired to the sawtooth-driven
ODMR frequency sweep), this script is a generic single-channel-of-interest
contrast measurement: point it at CHAN1 for any max/min-style measurement
and it captures/reports the same statistics as the VI.

Not replicated: the VI's live front-panel graphs update after every run, and
the "All runs + Average and SD curves" / "Create Histogram" plots - this
script is CSV-output only and prints per-run progress to the console.
"Serial Configuration" (VISA resource setup) is handled transparently by
pyvisa and has no separate control here.

Pass --save-traces to additionally dump every run's raw per-sample CH1/CH2
voltages to csv_output/odmr_labview_replica_traces.csv - odmr_labview_voltage_freq.py
consumes that file to convert each sweep's CH2 tuning voltage to frequency
and average the sweeps on a common frequency grid.

All CSV output goes to csv_output/ (created automatically if missing).

Requires: pip install pyvisa pyvisa-py numpy

Usage:
  python odmr_labview_replica.py --runs 100 --waveform ramp --freq-hz 100 --ampl-vpp 1.5 --offset-v 4.0 --ch2-enabled
"""

import argparse
import csv
import os
import time

import numpy as np
import pyvisa

DEFAULT_SCOPE_ADDR = "USB0::0x2A8D::0x0396::CN63257664::0::INSTR"
DEFAULT_AWG_ADDR = "GPIB0::5::INSTR"

# --- Output folder ---
CSV_DIR = "csv_output"

OUTPUT_CSV = os.path.join(CSV_DIR, "odmr_labview_replica.csv")
OUTPUT_TRACES_CSV = os.path.join(CSV_DIR, "odmr_labview_replica_traces.csv")

TIMEOUT_MS = 5000

# Capture slightly less than one full ramp period - same rationale as
# odmr_averaged_sweep.py's CAPTURE_FRACTION_OF_PERIOD.
CAPTURE_FRACTION_OF_PERIOD = 0.93


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    addr = p.add_argument_group("Instrument addresses")
    addr.add_argument("--scope-addr", default=DEFAULT_SCOPE_ADDR)
    addr.add_argument("--awg-addr", default=DEFAULT_AWG_ADDR)

    fg = p.add_argument_group("Function generator (Fig. 9, top left)")
    fg.add_argument("--change-pulse", dest="change_pulse", action="store_true", default=True,
                     help="Reconfigure the AWG at startup (default: on).")
    fg.add_argument("--no-change-pulse", dest="change_pulse", action="store_false",
                     help="Leave the AWG's current output untouched (VI's 'change pulse' button off).")
    fg.add_argument("--reset", dest="reset_awg", action="store_true", default=True,
                     help="Send *RST to the AWG before configuring it (default: on).")
    fg.add_argument("--no-reset", dest="reset_awg", action="store_false")
    fg.add_argument("--waveform", choices=["sine", "ramp"], default="ramp",
                     help="1=Sine, 4=Ramp in the VI's waveform selector.")
    fg.add_argument("--freq-hz", type=float, default=20.0,
                     help="Sawtooth sweep rate (default: 20 Hz, slowed from an earlier 100 Hz default "
                          "to give the VCO/detector more time per frequency step to settle, reducing "
                          "dynamic distortion of the resonance dip).")
    fg.add_argument("--ampl-vpp", type=float, default=1.5)
    fg.add_argument("--offset-v", type=float, default=4.0, help="DC offset, at center.")
    fg.add_argument("--ramp-symmetry-pct", type=float, default=100.0,
                     help="Only used when --waveform ramp; 100 = rising sawtooth.")
    fg.add_argument("--burst", action="store_true", default=False)
    fg.add_argument("--burst-phase-deg", type=float, default=0.0)

    sc = p.add_argument_group("Oscilloscope channels (Fig. 9, remainder)")
    sc.add_argument("--ch1-enabled", action=argparse.BooleanOptionalAction, default=True)
    sc.add_argument("--ch2-enabled", action=argparse.BooleanOptionalAction, default=False)
    sc.add_argument("--ch1-coupling", choices=["DC", "AC"], default="DC")
    sc.add_argument("--ch2-coupling", choices=["DC", "AC"], default="DC")
    sc.add_argument("--ch1-probe-atten", type=float, default=1.0)
    sc.add_argument("--ch2-probe-atten", type=float, default=1.0)
    sc.add_argument("--ch1-range-v", type=float, default=2.0, help="Full-scale vertical range.")
    sc.add_argument("--ch2-range-v", type=float, default=2.0)
    sc.add_argument("--ch1-offset-v", type=float, default=0.0)
    sc.add_argument("--ch2-offset-v", type=float, default=0.0)
    sc.add_argument("--autoscale", action=argparse.BooleanOptionalAction, default=True,
                     help="Autoscale enabled channels' vertical range/offset instead of using "
                          "the --chN-range-v/--chN-offset-v values (default: on).")

    acq = p.add_argument_group("Acquisition / trigger")
    acq.add_argument("--acquisition-type", choices=["sample", "average", "hresolution", "peak"], default="sample")
    acq.add_argument("--time-per-record-s", type=float, default=None,
                      help="Timebase full-record span, seconds. Unset: derived from --freq-hz as "
                           f"{CAPTURE_FRACTION_OF_PERIOD}/freq_hz, i.e. slightly less than one full "
                           "ramp period (capturing exactly one period risks running past the ramp's "
                           "reset into the next cycle's leading edge).")
    acq.add_argument("--min-record-length", type=int, default=10000,
                      help="Points per acquisition (default: 10000, raised from an earlier 2000 default "
                           "for finer frequency resolution across the same swept span).")
    acq.add_argument("--trigger-source", choices=["CHAN1", "CHAN2"], default="CHAN1")
    acq.add_argument("--trigger-level-v", type=float, default=0.0)
    acq.add_argument("--trigger-slope", choices=["positive", "negative"], default="negative")
    acq.add_argument("--trigger-holdoff-s", type=float, default=0.0)
    acq.add_argument("--runs", type=int, default=100, help="Number of single-shot acquisitions to average.")
    acq.add_argument("--save-traces", action="store_true", default=False,
                      help="Also save raw per-sample traces (time_s, chN_V per run) to "
                           f"{OUTPUT_TRACES_CSV}, e.g. for voltage-to-frequency sweep averaging downstream.")

    args = p.parse_args()
    if args.time_per_record_s is None:
        args.time_per_record_s = CAPTURE_FRACTION_OF_PERIOD / args.freq_hz
    return args


ACQ_TYPE_SCPI = {
    "sample": "NORMAL",
    "average": "AVERAGE",
    "hresolution": "HRESOLUTION",
    "peak": "PEAK",
}


def setup_awg(inst, args):
    if not args.change_pulse:
        print("[AWG] --no-change-pulse: leaving current AWG output untouched.")
        return

    if args.reset_awg:
        inst.write("*RST")
        time.sleep(0.5)

    inst.write("FUNC SIN" if args.waveform == "sine" else "FUNC RAMP")
    if args.waveform == "ramp":
        inst.write(f"FUNC:RAMP:SYMMETRY {args.ramp_symmetry_pct}")
    inst.write(f"FREQ {args.freq_hz}")
    inst.write(f"VOLT {args.ampl_vpp}")
    inst.write(f"VOLT:OFFS {args.offset_v}")

    inst.write(f"BURST:STATE {'ON' if args.burst else 'OFF'}")
    if args.burst:
        inst.write(f"BURST:PHASE {args.burst_phase_deg}")

    inst.write("OUTP ON")
    err = inst.query("SYST:ERR?").strip()
    print(
        f"[AWG] {args.waveform} {args.freq_hz} Hz, {args.ampl_vpp} Vpp, "
        f"{args.offset_v} V offset, burst={'ON' if args.burst else 'OFF'}. SYST:ERR? -> {err}"
    )


def setup_scope(inst, args):
    channel_settings = (
        (1, args.ch1_enabled, args.ch1_coupling, args.ch1_probe_atten, args.ch1_range_v, args.ch1_offset_v),
        (2, args.ch2_enabled, args.ch2_coupling, args.ch2_probe_atten, args.ch2_range_v, args.ch2_offset_v),
    )

    # Coupling and probe attenuation affect what a correct autoscale looks
    # like, so set those before autoscaling rather than after.
    for ch, enabled, coupling, atten, rng, offset in channel_settings:
        inst.write(f":CHAN{ch}:DISPLAY {'ON' if enabled else 'OFF'}")
        if not enabled:
            continue
        inst.write(f":CHAN{ch}:COUPLING {coupling}")
        inst.write(f":CHAN{ch}:PROBE {atten}")
        if not args.autoscale:
            inst.write(f":CHAN{ch}:RANGE {rng}")
            inst.write(f":CHAN{ch}:OFFSET {offset}")

    enabled_channels = [ch for ch, enabled, *_ in channel_settings if enabled]
    if args.autoscale:
        inst.write(":AUTOSCALE " + ",".join(f"CHAN{ch}" for ch in enabled_channels))
        time.sleep(2)
        for ch in enabled_channels:
            scale = float(inst.query(f":CHAN{ch}:SCALE?").strip())
            offset = float(inst.query(f":CHAN{ch}:OFFSET?").strip())
            print(f"[Scope] CHAN{ch} autoscaled to {scale * 1e3:.2f} mV/div, {offset * 1e3:.2f} mV offset.")

    inst.write(f":TIMEBASE:RANGE {args.time_per_record_s}")
    inst.write(":TIMEBASE:REFERENCE LEFT")
    inst.write(":TIMEBASE:POSITION 0")

    inst.write(":TRIGGER:MODE EDGE")
    inst.write(f":TRIGGER:EDGE:SOURCE {args.trigger_source}")
    inst.write(f":TRIGGER:EDGE:LEVEL {args.trigger_level_v}")
    inst.write(f":TRIGGER:EDGE:SLOPE {'POSITIVE' if args.trigger_slope == 'positive' else 'NEGATIVE'}")
    if args.trigger_holdoff_s > 0:
        inst.write(f":TRIGGER:HOLDOFF {args.trigger_holdoff_s}")

    inst.write(f":ACQUIRE:TYPE {ACQ_TYPE_SCPI[args.acquisition_type]}")
    inst.write(":WAV:FORMAT WORD")
    inst.write(":WAV:BYTEORDER LSBFIRST")
    inst.write(":WAV:POINTS:MODE NORMAL")
    inst.write(f":WAV:POINTS {args.min_record_length}")

    err = inst.query("SYST:ERR?").strip()
    print(
        f"[Scope] {args.time_per_record_s * 1e3:.3f} ms/record, {args.min_record_length} pts, "
        f"trigger {args.trigger_source} {args.trigger_slope} @ {args.trigger_level_v:.3f} V, "
        f"acquisition={args.acquisition_type}. SYST:ERR? -> {err}"
    )


def acquire_single_run(inst, channels):
    """Arm a single trigger, wait for it, digitize the requested channels
    together, and return {ch: (t, v)}."""
    src = ",".join(f"CHAN{ch}" for ch in channels)
    inst.write(f":DIGITIZE {src}")  # blocks until acquisition completes

    results = {}
    for ch in channels:
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
    args = parse_args()

    os.makedirs(CSV_DIR, exist_ok=True)

    channels = []
    if args.ch1_enabled:
        channels.append(1)
    if args.ch2_enabled:
        channels.append(2)
    if not channels:
        raise SystemExit("At least one of --ch1-enabled/--ch2-enabled must be set.")

    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(args.scope_addr)
    scope.timeout = TIMEOUT_MS
    awg = rm.open_resource(args.awg_addr)
    awg.timeout = TIMEOUT_MS

    print("[Scope] IDN:", scope.query("*IDN?").strip())
    print("[AWG] IDN:", awg.query("*IDN?").strip())

    setup_awg(awg, args)
    setup_scope(scope, args)

    t_ref = None
    traces = {ch: [] for ch in channels}  # traces[ch][run] -> v array
    run_max = []   # per-run max of channel 1 ("Average Max" input)
    run_min = []   # per-run min of channel 1 ("Average Minimum" input)
    n_captured = 0

    print(f"\nCapturing {args.runs} triggered runs...\n")

    try:
        for i in range(args.runs):
            data = acquire_single_run(scope, channels)
            t1, v1 = data[1]

            if t_ref is None:
                t_ref = t1

            n = len(t_ref)
            for ch in channels:
                t_ch, v_ch = data[ch]
                traces[ch].append(v_ch[:n])

            run_max.append(float(np.max(v1)))
            run_min.append(float(np.min(v1)))
            n_captured += 1

            if (i + 1) % 10 == 0 or (i + 1) == args.runs:
                print(f"\r  Run {i + 1}/{args.runs} captured", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\nStopped early after {n_captured} runs - processing what we have.")
    finally:
        if args.change_pulse:
            # Only turn the AWG output back off if we're the ones who turned
            # it on; --no-change-pulse means "leave it as I found it."
            try:
                awg.write("OUTP OFF")
            except Exception:
                pass
        scope.write(":RUN")  # restore normal continuous display
        scope.close()
        awg.close()
        rm.close()

    if n_captured == 0:
        print("No runs captured - nothing to process.")
        return

    run_max = np.array(run_max[:n_captured])
    run_min = np.array(run_min[:n_captured])
    run_diff_mv = (run_max - run_min) * 1e3

    # "Average Max 2" / "Average Minimum 2" / "Difference (mV)" / "Percent Diff":
    # the per-run max/min values, averaged across all runs.
    avg_max_2 = float(np.mean(run_max))
    avg_min_2 = float(np.mean(run_min))
    difference_mv = (avg_max_2 - avg_min_2) * 1e3
    # abs() so Percent Diff stays a positive fraction of the signal's scale
    # even if the detector's baseline (avg_max_2) is negative-going.
    percent_diff = 100.0 * difference_mv / abs(avg_max_2 * 1e3) if avg_max_2 else float("nan")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_index", "ch1_max_V", "ch1_min_V", "diff_mV"])
        for i, (mx, mn, df) in enumerate(zip(run_max, run_min, run_diff_mv)):
            writer.writerow([i, mx, mn, df])
    print(f"\n\nSaved per-run stats to {OUTPUT_CSV}")

    if args.save_traces:
        with open(OUTPUT_TRACES_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_index", "time_s"] + [f"ch{ch}_V" for ch in channels])
            for i in range(n_captured):
                for j, t in enumerate(t_ref):
                    writer.writerow([i, t] + [traces[ch][i][j] for ch in channels])
        print(f"Saved raw per-sample traces to {OUTPUT_TRACES_CSV}")

    print(
        f"\nAverage Max 2:  {avg_max_2:.5f} V\n"
        f"Average Min 2:  {avg_min_2:.5f} V\n"
        f"Difference:     {difference_mv:.3f} mV\n"
        f"Percent Diff:   {percent_diff:.2f} %"
    )


if __name__ == "__main__":
    main()
