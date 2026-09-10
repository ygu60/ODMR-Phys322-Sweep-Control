# ODMR Averaged Sweep

Data-collection script for an optically detected magnetic resonance (ODMR)
measurement: sweep a microwave drive frequency across a resonance and look
for the fractional dip in fluorescence, analogous to Zhang et al., *Am. J.
Phys.* 86, 225 (2018), which reports an ~8% fluorescence dip on resonance.

## Setup

- **PDA36A2** photodetector/amplifier — fluorescence-proportional signal, read
  on scope **CHAN1**.
- **Keysight 33220A** function generator — drives a rising sawtooth ramp
  (`V_tune`) into a VCO to sweep the microwave frequency; also monitored on
  scope **CHAN2**.
- **Keysight DSOX1204G** oscilloscope — captures both channels together, one
  triggered single-shot acquisition per sweep.

The scope triggers on the rising edge of CHAN2 at the start of each ramp, so
every capture starts at the same phase of the sweep. A voltage-to-frequency
calibration (`V_TO_F_SLOPE_GHZ_PER_V`, `V_TO_F_INTERCEPT_GHZ`), obtained from
band-edge measurements, converts the sawtooth's tuning voltage into the
corresponding microwave drive frequency.

## What the script does

1. Configures the 33220A to output the sawtooth sweep and the DSOX1204G to
   trigger and digitize one sweep at a time.
2. Repeats `NUM_AVERAGES` single-shot, triggered acquisitions of CHAN1 +
   CHAN2, summing point-by-point as it goes (averaging beats down noise on
   the detector signal, which is otherwise too small/noisy to see a clean
   dip in a single sweep).
3. Fits a line to the averaged sawtooth (CHAN2) vs. time — the ramp is linear
   by design, and fitting removes residual per-sample scope noise so the
   derived frequency axis is smooth and strictly monotonic — then converts it
   to drive frequency via the calibration constants.
4. Expresses the averaged detector trace as deviation from its own mean, so
   the resonance shows up as a negative excursion around zero.
5. Saves the averaged data to CSV and a plot (detector deviation vs. drive
   frequency) to PNG, and prints the frequency and depth of the deepest dip.
6. Restores the scope to normal continuous-run display when done (including
   on Ctrl-C, which stops early and still averages/plots whatever was
   captured).

## Usage

```
pip install pyvisa pyvisa-py numpy matplotlib
python odmr_averaged_sweep.py
```

Update `SCOPE_ADDR` / `AWG_ADDR` at the top of the script for your VISA
resource strings, and the sawtooth/calibration constants if the sweep range
or setup changes.

## Output

- `odmr_averaged_sweep.csv` — `time_s, v_tune_V, freq_GHz, detector_V,
  detector_V_dev_from_mean` for every point in the averaged trace.
- `odmr_averaged_sweep.png` — detector deviation-from-mean vs. drive
  frequency, i.e. the ODMR sweep plot.

Both are git-ignored since they're per-run measurement data, not source.
