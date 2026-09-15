"""
Read-only diagnostic: connects to the scope and prints CH2's actual measured
voltage (Vpp, Vmax, Vmin, Vavg), its current coupling/scale/offset, and the
currently configured trigger settings - no configuration is changed.

Run this while the AWG is actively driving CH2 (i.e. right after starting
odmr_averaged_sweep.py or odmr_labview_replica.py, or with the AWG output
left on), so the scope has a live signal to measure.

Usage:
  python odmr_diagnose_ch2.py
"""

import pyvisa

SCOPE_ADDR = "USB0::0x2A8D::0x0396::CN63257664::0::INSTR"


def main():
    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(SCOPE_ADDR)
    scope.timeout = 5000

    print("[Scope] IDN:", scope.query("*IDN?").strip())

    print("\n--- CH2 current settings ---")
    print("Coupling:", scope.query(":CHAN2:COUPLING?").strip())
    print("Display:", scope.query(":CHAN2:DISPLAY?").strip())
    print("Probe attenuation:", scope.query(":CHAN2:PROBE?").strip())
    print("Scale (V/div):", scope.query(":CHAN2:SCALE?").strip())
    print("Offset (V):", scope.query(":CHAN2:OFFSET?").strip())
    print("Range (V):", scope.query(":CHAN2:RANGE?").strip())

    print("\n--- CH2 measured voltage (live) ---")
    for meas in ["VPP", "VMAX", "VMIN", "VAVERAGE", "VTOP", "VBASE"]:
        try:
            val = scope.query(f":MEASURE:{meas}? CHAN2").strip()
            print(f"{meas}: {val}")
        except Exception as e:
            print(f"{meas}: query failed ({e})")

    print("\n--- Current trigger configuration ---")
    print("Mode:", scope.query(":TRIGGER:MODE?").strip())
    print("Sweep:", scope.query(":TRIGGER:SWEEP?").strip())
    print("Edge source:", scope.query(":TRIGGER:EDGE:SOURCE?").strip())
    print("Edge level (V):", scope.query(":TRIGGER:EDGE:LEVEL?").strip())
    print("Edge slope:", scope.query(":TRIGGER:EDGE:SLOPE?").strip())

    scope.close()
    rm.close()


if __name__ == "__main__":
    main()
