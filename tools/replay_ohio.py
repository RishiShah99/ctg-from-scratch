#!/usr/bin/env python3
"""Stream a held-out Ohio T1DM test patient's CSV over UART to the ESP32.

Reads data/ohio_t1dm.csv, filters to one patient, and emits the four-
command UART protocol (g/b/m/t) understood by esp32/src/main.cpp:

    g <mg_dL>          triggers inference
    b <t_min> <units>  bolus event
    m <t_min> <carbs>  meal event
    t <t_min>          set device clock

Every event line (g/b/m) is preceded by a `t <t_min>` advance so the
firmware's `t_min > g_now_min` future-event guard never trips. Boluses
and meals that precede the first glucose sample in the Ohio CSV
(treatments are logged before the sensor starts streaming) used to be
silently dropped; the pre-emit fix back-fills them in order. At
`--speedup 1` the gap between glucose samples is the trainer's 5-min
cadence (300 s wall time); higher speedup compresses that
proportionally.

Requires pyserial (`pip install pyserial`). No other dependencies.

Usage:
    python tools/replay_ohio.py --csv data/ohio_t1dm.csv \\
        --patient 591 --port COM7 --baud 115200 --speedup 60

Only patients 591 and 596 are accepted — the other Ohio patients are
training data and replaying them would conflate train/test.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import re
import sys
import time

try:
    import serial
except ImportError:
    sys.stderr.write(
        "error: pyserial not installed. Run: pip install pyserial\n")
    sys.exit(2)


TEST_PATIENTS = {"591", "596"}

# Matches the device's per-prediction emit line, e.g.
#   cgm=142.0 logit=1.8432 prob=0.8634 alert=1 iob=0.34 cob=1.20 t=1280
EMIT_RE = re.compile(
    r"^cgm=(?P<cgm>[-\d.]+)\s+"
    r"logit=(?P<logit>[-\d.]+)\s+"
    r"prob=(?P<prob>[-\d.]+)\s+"
    r"alert=(?P<alert>\d+)\s+"
    r"iob=(?P<iob>[-\d.]+)\s+"
    r"cob=(?P<cob>[-\d.]+)\s+"
    r"t=(?P<t_min>\d+)\s*$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Path to data/ohio_t1dm.csv")
    p.add_argument("--patient", required=True,
                   help="Patient id (591 or 596).")
    p.add_argument("--port", required=True,
                   help="Serial port path, e.g. COM7 or /dev/ttyACM0.")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--speedup", type=float, default=1.0,
                   help="Replay speed multiplier. 1.0 = realtime "
                        "(300 s/sample); 60.0 = one sample / 5 s.")
    p.add_argument("--from-t-min", type=int, default=0,
                   help="Skip records with t_min strictly less than this.")
    p.add_argument("--max-windows", type=int, default=0,
                   help="Stop after this many inference responses "
                        "(0 = no limit).")
    p.add_argument("--echo", action="store_true",
                   help="Print every TX and RX line to stderr.")
    return p.parse_args()


def load_patient_records(csv_path: str, patient: str, from_t_min: int):
    """Return list of (t_min, kind, value) tuples, sorted by t_min.

    kind ∈ {'g', 'b', 'm'}; value is mg/dL, units, or grams.
    """
    if patient not in TEST_PATIENTS:
        sys.stderr.write(
            f"error: patient {patient} is not in the test set. "
            f"Accepted: {sorted(TEST_PATIENTS)}\n")
        sys.exit(2)

    records = []
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["patient_id"] != patient:
                continue
            t_min = int(row["t_min"])
            if t_min < from_t_min:
                continue
            event = (row.get("event") or "").strip()
            amount = row.get("amount") or ""
            glucose = row.get("glucose") or ""
            if event == "bolus":
                records.append((t_min, "b", float(amount)))
            elif event == "meal":
                records.append((t_min, "m", float(amount)))
            elif glucose:
                records.append((t_min, "g", float(glucose)))
    records.sort(key=lambda r: (r[0], {"g": 2, "b": 0, "m": 1}[r[1]]))
    return records


def wait_for_banner(ser: serial.Serial, echo: bool, timeout_s: float = 5.0):
    """Read lines until we see 'OSDN load:'. Bail if it never appears."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        line = ser.readline().decode("utf-8", errors="replace").rstrip()
        if not line:
            continue
        if echo:
            sys.stderr.write(f"<< {line}\n")
        if "OSDN load:" in line:
            return line
    sys.stderr.write(
        "error: no 'OSDN load:' banner from device within "
        f"{timeout_s} s. Wrong port? Wrong firmware? Reset the board.\n")
    sys.exit(3)


def main() -> int:
    args = parse_args()
    records = load_patient_records(args.csv, args.patient, args.from_t_min)
    if not records:
        sys.stderr.write(
            f"error: no records found for patient {args.patient} "
            f"at or after t_min={args.from_t_min}.\n")
        return 2

    sys.stderr.write(
        f"loaded {len(records)} records for patient {args.patient} "
        f"(t_min range: {records[0][0]}..{records[-1][0]})\n")

    if args.speedup <= 0:
        sys.stderr.write("error: --speedup must be > 0.\n")
        return 2
    inter_sample_s = max(5.0 * 60.0 / args.speedup, 0.0)

    sys.stderr.write(
        f"opening {args.port} @ {args.baud}; "
        f"inter-sample wait = {inter_sample_s:.3f} s "
        f"(speedup={args.speedup}×)\n")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"replay_{args.patient}_{stamp}.csv")
    sys.stderr.write(f"capture file: {out_path}\n")

    with serial.Serial(args.port, args.baud, timeout=2.0) as ser:
        time.sleep(2.0)
        wait_for_banner(ser, args.echo)

        def send(line: str):
            if args.echo:
                sys.stderr.write(f">> {line}\n")
            ser.write((line + "\n").encode("utf-8"))

        def drain():
            """Pull any pending RX lines; return list of parsed emits."""
            emits = []
            while True:
                raw = ser.readline().decode("utf-8", errors="replace").rstrip()
                if not raw:
                    return emits
                if args.echo:
                    sys.stderr.write(f"<< {raw}\n")
                m = EMIT_RE.match(raw)
                if m:
                    emits.append(m.groupdict())

        send("t 0")
        time.sleep(0.05)
        drain()

        capture_cols = ["t_min", "cgm", "logit", "prob", "alert", "iob", "cob"]
        captured: list[dict] = []
        windows_emitted = 0

        try:
            for t_min, kind, value in records:
                # Pre-emit `t` before every event so the firmware's
                # `t_min > g_now_min` future-event guard accepts events
                # that precede the first glucose sample (Ohio sessions
                # often start with several boluses/meals before the CGM
                # comes online). See results/step7_review.md MAJOR-2.
                send(f"t {t_min}")
                if kind in ("b", "m"):
                    send(f"{kind} {t_min} {value:.4f}")
                elif kind == "g":
                    send(f"g {value:.4f}")
                    if inter_sample_s > 0:
                        time.sleep(inter_sample_s)
                    for emit in drain():
                        captured.append({
                            "t_min": int(emit["t_min"]),
                            "cgm":   float(emit["cgm"]),
                            "logit": float(emit["logit"]),
                            "prob":  float(emit["prob"]),
                            "alert": int(emit["alert"]),
                            "iob":   float(emit["iob"]),
                            "cob":   float(emit["cob"]),
                        })
                        windows_emitted += 1
                        if args.max_windows and \
                           windows_emitted >= args.max_windows:
                            raise StopIteration
        except StopIteration:
            sys.stderr.write(
                f"max_windows={args.max_windows} reached, stopping.\n")
        except KeyboardInterrupt:
            sys.stderr.write("interrupted; flushing capture.\n")

        # One last drain in case responses are still in the buffer.
        for emit in drain():
            captured.append({
                "t_min": int(emit["t_min"]),
                "cgm":   float(emit["cgm"]),
                "logit": float(emit["logit"]),
                "prob":  float(emit["prob"]),
                "alert": int(emit["alert"]),
                "iob":   float(emit["iob"]),
                "cob":   float(emit["cob"]),
            })

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=capture_cols)
        writer.writeheader()
        for row in captured:
            writer.writerow(row)

    sys.stderr.write(
        f"done. {len(captured)} predictions captured -> {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
