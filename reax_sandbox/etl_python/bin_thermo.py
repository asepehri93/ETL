import argparse
import csv
import math
from typing import Dict, List, Optional, Tuple


def parse_lammps_log_thermo(log_path: str) -> Tuple[List[str], List[List[float]]]:
    """
    Parse the LAST thermo block in a LAMMPS log file.

    Assumes thermo output format:
      <header columns...>
      <numeric rows...>
      Loop time ...
    """
    with open(log_path, "r") as f:
        lines = f.readlines()

    # Find all thermo headers by looking for a line that starts with 'Step' (common convention)
    header_idxs = [i for i, ln in enumerate(lines) if ln.strip().startswith("Step")]
    if not header_idxs:
        raise RuntimeError("Could not find thermo header starting with 'Step' in log file.")

    header_i = header_idxs[-1]
    headers = lines[header_i].strip().split()

    rows: List[List[float]] = []
    for ln in lines[header_i + 1 :]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("Loop time"):
            break
        parts = s.split()
        if len(parts) != len(headers):
            # Not a thermo row
            continue
        try:
            row = [float(x) for x in parts]
        except ValueError:
            continue
        rows.append(row)

    if not rows:
        raise RuntimeError("Parsed zero thermo rows from log file.")

    return headers, rows


def bin_by_time(
    headers: List[str],
    rows: List[List[float]],
    bin_width_ps: float,
    time_col: str = "Time",
) -> List[Dict[str, float]]:
    if time_col not in headers:
        raise KeyError(f"'{time_col}' column not found in headers: {headers}")

    time_idx = headers.index(time_col)
    bins: Dict[int, List[List[float]]] = {}

    for row in rows:
        t_ps = float(row[time_idx])
        b = int(math.floor(t_ps / bin_width_ps))
        bins.setdefault(b, []).append(row)

    out: List[Dict[str, float]] = []
    for b in sorted(bins.keys()):
        chunk = bins[b]
        d: Dict[str, float] = {"bin": float(b), "t0_ps": b * bin_width_ps, "t1_ps": (b + 1) * bin_width_ps}
        for j, h in enumerate(headers):
            xs = [r[j] for r in chunk]
            mean = sum(xs) / len(xs)
            var = sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)
            d[f"{h}_mean"] = mean
            d[f"{h}_std"] = math.sqrt(var)
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description="Bin LAMMPS thermo output by physical time (ps).")
    ap.add_argument("log_lammps", help="Path to log.lammps")
    ap.add_argument("--bin-ps", type=float, default=0.5, help="Bin width in ps")
    ap.add_argument("--out", default="thermo_binned.csv", help="Output CSV filename")
    args = ap.parse_args()

    headers, rows = parse_lammps_log_thermo(args.log_lammps)
    binned = bin_by_time(headers, rows, bin_width_ps=args.bin_ps, time_col="Time")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(binned[0].keys()))
        w.writeheader()
        w.writerows(binned)

    print(f"Wrote {len(binned)} bins to {args.out}")


if __name__ == "__main__":
    main()

