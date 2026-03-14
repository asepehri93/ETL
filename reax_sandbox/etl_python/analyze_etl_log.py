#!/usr/bin/env python3
"""
Analyze ETL controller logs with time-weighted statistics and ramp-phase summaries.

Features:
- Time-weighted statistics (accounts for variable dt)
- Phase-based analysis for temperature ramps (e.g., 300-1000K, 1000-2000K, 2000-3000K)
- Comparison between baseline and ETL runs
- CSV output for further analysis

Usage:
    # Single log analysis
    python analyze_etl_log.py outputs_etl_dt/etl_log.csv

    # Compare ETL run against baseline
    python analyze_etl_log.py outputs_etl_dt/etl_log.csv --baseline outputs_baseline_conservative/etl_log.csv

    # Analyze with custom phase boundaries (T in K)
    python analyze_etl_log.py outputs_etl_dt/etl_log.csv --phases 300,1000,2000,3000
"""
import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class LogRecord:
    """Single row from etl_log.csv."""
    step_index: int
    time_fs: float
    time_ps: float
    dt_fs: float
    tol: float
    lang_damp_fs: float
    baro_pdamp_fs: float
    T_target: float
    Sbar: float
    temp: float
    press: float
    pe: float
    ke: float
    etotal: float


def _safe_float(row: dict, key: str, default: float = float("nan")) -> float:
    """Extract float from row, returning default if missing or invalid."""
    if key not in row:
        return default
    try:
        val = row[key]
        if val in ("", "nan", "NaN", None):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def load_etl_log(filepath: str) -> List[LogRecord]:
    """Load etl_log.csv into a list of LogRecord objects."""
    records = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = LogRecord(
                step_index=int(row.get("step_index", 0)),
                time_fs=_safe_float(row, "time_fs"),
                time_ps=_safe_float(row, "time_ps"),
                dt_fs=_safe_float(row, "dt_fs"),
                tol=_safe_float(row, "tol"),
                lang_damp_fs=_safe_float(row, "lang_damp_fs"),
                baro_pdamp_fs=_safe_float(row, "baro_pdamp_fs"),
                T_target=_safe_float(row, "T_target"),
                Sbar=_safe_float(row, "Sbar"),
                temp=_safe_float(row, "temp"),
                press=_safe_float(row, "press"),
                pe=_safe_float(row, "pe"),
                ke=_safe_float(row, "ke"),
                etotal=_safe_float(row, "etotal"),
            )
            records.append(rec)
    return records


def time_weighted_stats(values: List[float], weights: List[float]) -> Dict[str, float]:
    """
    Compute time-weighted statistics.
    
    Returns: dict with count, total_weight, min, max, mean, std
    """
    valid = [(v, w) for v, w in zip(values, weights) if not math.isnan(v) and w > 0]
    if not valid:
        return {"count": 0, "total_weight": 0.0}
    
    vs, ws = zip(*valid)
    total_w = sum(ws)
    mean = sum(v * w for v, w in valid) / total_w
    var = sum(w * (v - mean) ** 2 for v, w in valid) / total_w
    
    return {
        "count": len(valid),
        "total_weight": total_w,
        "min": min(vs),
        "max": max(vs),
        "mean": mean,
        "std": math.sqrt(var),
    }


def simple_stats(values: List[float]) -> Dict[str, float]:
    """Compute simple statistics (unweighted)."""
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return {"count": 0}
    
    n = len(valid)
    mean = sum(valid) / n
    var = sum((v - mean) ** 2 for v in valid) / max(1, n - 1)
    
    return {
        "count": n,
        "min": min(valid),
        "max": max(valid),
        "mean": mean,
        "std": math.sqrt(var),
    }


def segment_by_T_target(
    records: List[LogRecord],
    boundaries: List[float],
) -> Dict[str, List[LogRecord]]:
    """
    Segment records into temperature phases.
    
    Args:
        records: List of LogRecord
        boundaries: Temperature boundaries in K (e.g., [300, 1000, 2000, 3000])
    
    Returns:
        Dict mapping phase labels to record lists
    """
    phases = {}
    for i in range(len(boundaries) - 1):
        T_lo, T_hi = boundaries[i], boundaries[i + 1]
        label = f"{int(T_lo)}-{int(T_hi)}K"
        phases[label] = [r for r in records if T_lo <= r.T_target < T_hi]
    
    # Include the upper boundary in the last phase
    if len(boundaries) >= 2:
        T_lo, T_hi = boundaries[-2], boundaries[-1]
        label = f"{int(T_lo)}-{int(T_hi)}K"
        phases[label].extend([r for r in records if r.T_target == T_hi])
    
    return phases


def segment_by_ramp_direction(records: List[LogRecord]) -> Dict[str, List[LogRecord]]:
    """
    Segment records by ramp direction for triangle ramps.
    
    Detects the peak temperature point and splits into:
    - "heating": records before the peak (T increasing)
    - "cooling": records after the peak (T decreasing)
    
    For linear ramps (monotonic T), returns all records as "heating" or "cooling"
    depending on the direction.
    """
    if not records:
        return {"heating": [], "cooling": []}
    
    T_targets = [r.T_target for r in records]
    
    # Find the index of maximum temperature
    max_T = max(T_targets)
    max_idx = T_targets.index(max_T)
    
    # Check if it's a triangle ramp (T goes up then down)
    T_start = T_targets[0]
    T_end = T_targets[-1]
    
    if T_start < max_T and T_end < max_T and max_idx > 0 and max_idx < len(records) - 1:
        # Triangle ramp detected
        heating = records[:max_idx + 1]
        cooling = records[max_idx:]  # Include peak in both for continuity
        return {"heating": heating, "cooling": cooling}
    elif T_start < T_end:
        # Linear ramp (heating)
        return {"heating": records, "cooling": []}
    elif T_start > T_end:
        # Linear ramp (cooling)
        return {"heating": [], "cooling": records}
    else:
        # Constant temperature
        return {"constant": records}


def format_stats(stats: Dict[str, float], precision: int = 4) -> str:
    """Format statistics dict as a readable string."""
    if stats.get("count", 0) == 0:
        return "(no data)"
    parts = []
    for k, v in stats.items():
        if k == "count":
            parts.append(f"n={int(v)}")
        elif k == "total_weight":
            parts.append(f"Σdt={v:.2f}fs")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.{precision}g}")
    return ", ".join(parts)


def analyze_single(records: List[LogRecord], label: str = "") -> Dict:
    """Analyze a single ETL log and return summary dict."""
    if not records:
        return {"label": label, "error": "No records"}
    
    dts = [r.dt_fs for r in records]
    tols = [r.tol for r in records]
    Sbars = [r.Sbar for r in records]
    T_targets = [r.T_target for r in records]
    temps = [r.temp for r in records]
    pes = [r.pe for r in records]
    kes = [r.ke for r in records]
    etotals = [r.etotal for r in records]
    lang_damps = [r.lang_damp_fs for r in records]
    
    # Total simulation time
    total_time_fs = sum(dts)
    total_time_ps = total_time_fs / 1000.0
    
    # Count dt changes (hysteresis effectiveness)
    dt_changes = sum(1 for i in range(1, len(dts)) if abs(dts[i] - dts[i-1]) > 1e-10)
    
    # Count tol changes
    tol_changes = sum(1 for i in range(1, len(tols)) 
                      if not math.isnan(tols[i]) and not math.isnan(tols[i-1])
                      and abs(tols[i] - tols[i-1]) > 1e-12)
    
    summary = {
        "label": label,
        "total_chunks": len(records),
        "total_time_ps": total_time_ps,
        "dt_changes": dt_changes,
        "tol_changes": tol_changes,
        "dt_fs": time_weighted_stats(dts, dts),
        "tol": time_weighted_stats(tols, dts),
        "Sbar": time_weighted_stats(Sbars, dts),
        "T_target": simple_stats(T_targets),
        "temp": time_weighted_stats(temps, dts),
        "pe": time_weighted_stats(pes, dts),
        "ke": time_weighted_stats(kes, dts),
        "etotal": time_weighted_stats(etotals, dts),
        "lang_damp_fs": simple_stats(lang_damps),
    }
    
    return summary


def print_summary(summary: Dict, title: str = "Summary"):
    """Print a formatted summary."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    
    if "error" in summary:
        print(f"Error: {summary['error']}")
        return
    
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Total time: {summary['total_time_ps']:.3f} ps")
    print(f"dt changes: {summary['dt_changes']}")
    print(f"tol changes: {summary['tol_changes']}")
    
    print("\n--- Controller Parameters (time-weighted) ---")
    print(f"  dt_fs:        {format_stats(summary['dt_fs'])}")
    print(f"  tol:          {format_stats(summary['tol'])}")
    print(f"  Sbar:         {format_stats(summary['Sbar'])}")
    print(f"  lang_damp_fs: {format_stats(summary['lang_damp_fs'])}")
    
    print("\n--- Thermodynamics (time-weighted) ---")
    print(f"  T_target (K): {format_stats(summary['T_target'])}")
    print(f"  temp (K):     {format_stats(summary['temp'])}")
    print(f"  pe (kcal/mol):    {format_stats(summary['pe'])}")
    print(f"  ke (kcal/mol):    {format_stats(summary['ke'])}")
    print(f"  etotal (kcal/mol): {format_stats(summary['etotal'])}")


def print_phase_summaries(
    records: List[LogRecord],
    boundaries: List[float],
):
    """Print phase-by-phase summaries for temperature ramp."""
    phases = segment_by_T_target(records, boundaries)
    
    print(f"\n{'=' * 60}")
    print("PHASE-BY-PHASE ANALYSIS (Temperature Ramp)")
    print(f"{'=' * 60}")
    
    for phase_label, phase_records in phases.items():
        if not phase_records:
            print(f"\n[{phase_label}] No data")
            continue
        
        summary = analyze_single(phase_records, phase_label)
        
        print(f"\n[{phase_label}]")
        print(f"  Chunks: {summary['total_chunks']}, Time: {summary['total_time_ps']:.3f} ps")
        print(f"  dt_fs: {format_stats(summary['dt_fs'], 3)}")
        print(f"  tol:   {format_stats(summary['tol'], 2)}")
        print(f"  Sbar:  {format_stats(summary['Sbar'], 2)}")
        print(f"  temp:  {format_stats(summary['temp'], 1)}")
        print(f"  etotal: {format_stats(summary['etotal'], 2)}")


def compare_runs(
    test_records: List[LogRecord],
    baseline_records: List[LogRecord],
    test_label: str = "ETL",
    baseline_label: str = "Baseline",
):
    """Compare ETL run against baseline."""
    test_summary = analyze_single(test_records, test_label)
    baseline_summary = analyze_single(baseline_records, baseline_label)
    
    print(f"\n{'=' * 60}")
    print(f"COMPARISON: {test_label} vs {baseline_label}")
    print(f"{'=' * 60}")
    
    # Time comparison
    test_time = test_summary["total_time_ps"]
    base_time = baseline_summary["total_time_ps"]
    print(f"Total simulated time: {test_label}={test_time:.3f}ps, {baseline_label}={base_time:.3f}ps")
    
    # Chunk count (fewer chunks = more efficient)
    test_chunks = test_summary["total_chunks"]
    base_chunks = baseline_summary["total_chunks"]
    chunk_ratio = test_chunks / base_chunks if base_chunks > 0 else float("inf")
    print(f"Chunks: {test_label}={test_chunks}, {baseline_label}={base_chunks} (ratio={chunk_ratio:.2f})")
    
    # Mean dt comparison
    test_dt_mean = test_summary["dt_fs"].get("mean", 0)
    base_dt_mean = baseline_summary["dt_fs"].get("mean", 0)
    dt_ratio = test_dt_mean / base_dt_mean if base_dt_mean > 0 else float("inf")
    print(f"Mean dt_fs: {test_label}={test_dt_mean:.4f}, {baseline_label}={base_dt_mean:.4f} (ratio={dt_ratio:.2f})")
    
    # Temperature tracking error
    test_T = test_summary["temp"]
    base_T = baseline_summary["temp"]
    if test_T.get("count", 0) > 0 and base_T.get("count", 0) > 0:
        T_error = abs(test_T["mean"] - base_T["mean"])
        print(f"Mean temp: {test_label}={test_T['mean']:.1f}K, {baseline_label}={base_T['mean']:.1f}K (diff={T_error:.1f}K)")
    
    # Energy comparison
    test_E = test_summary["etotal"]
    base_E = baseline_summary["etotal"]
    if test_E.get("count", 0) > 0 and base_E.get("count", 0) > 0:
        E_error = abs(test_E["mean"] - base_E["mean"])
        E_rel_error = E_error / abs(base_E["mean"]) * 100 if base_E["mean"] != 0 else 0
        print(f"Mean etotal: {test_label}={test_E['mean']:.2f}, {baseline_label}={base_E['mean']:.2f}")
        print(f"  Energy difference: {E_error:.2f} kcal/mol ({E_rel_error:.3f}%)")


def write_summary_csv(summary: Dict, filepath: str):
    """Write summary statistics to CSV for further analysis."""
    flat = {"label": summary.get("label", "")}
    for key in ["total_chunks", "total_time_ps", "dt_changes", "tol_changes"]:
        flat[key] = summary.get(key, "")
    
    for stat_key in ["dt_fs", "tol", "Sbar", "T_target", "temp", "pe", "ke", "etotal", "lang_damp_fs"]:
        stats = summary.get(stat_key, {})
        for k, v in stats.items():
            flat[f"{stat_key}_{k}"] = v
    
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        writer.writeheader()
        writer.writerow(flat)
    
    print(f"Wrote summary to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ETL logs with time-weighted statistics and ramp-phase summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("etl_log_csv", help="Path to etl_log.csv")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline etl_log.csv for comparison")
    parser.add_argument(
        "--phases",
        type=str,
        default="300,1000,2000,3000",
        help="Comma-separated temperature boundaries in K for phase analysis",
    )
    parser.add_argument("--summary-csv", type=str, default=None, help="Output summary to CSV file")
    parser.add_argument("--quiet", action="store_true", help="Only print comparison/summary, skip full stats")
    parser.add_argument("--triangle", action="store_true", help="Analyze as triangle ramp (heating vs cooling phases)")
    args = parser.parse_args()

    # Load main log
    records = load_etl_log(args.etl_log_csv)
    label = os.path.basename(os.path.dirname(args.etl_log_csv)) or "ETL"
    
    if not args.quiet:
        summary = analyze_single(records, label)
        print_summary(summary, f"Analysis: {args.etl_log_csv}")
    
    # Phase analysis (only if T_target varies)
    T_targets = [r.T_target for r in records if not math.isnan(r.T_target)]
    if T_targets:
        T_range = max(T_targets) - min(T_targets)
        if T_range > 100:  # Only do phase analysis if T varies significantly
            # Triangle ramp analysis (heating vs cooling)
            if args.triangle:
                direction_phases = segment_by_ramp_direction(records)
                print(f"\n{'=' * 60}")
                print("TRIANGLE RAMP ANALYSIS (Heating vs Cooling)")
                print(f"{'=' * 60}")
                for phase_name, phase_records in direction_phases.items():
                    if phase_records:
                        phase_summary = analyze_single(phase_records, f"{label}_{phase_name}")
                        print(f"\n--- {phase_name.upper()} PHASE ({len(phase_records)} chunks) ---")
                        T_vals = [r.T_target for r in phase_records]
                        print(f"  T_target: {min(T_vals):.1f} -> {max(T_vals):.1f} K")
                        dt_vals = [r.dt_fs for r in phase_records]
                        print(f"  dt_fs: {min(dt_vals):.4f} - {max(dt_vals):.4f} (mean: {sum(dt_vals)/len(dt_vals):.4f})")
                        Sbar_vals = [r.Sbar for r in phase_records]
                        print(f"  Sbar: {min(Sbar_vals):.1f} - {max(Sbar_vals):.1f}")
            else:
                # Temperature boundary-based phase analysis
                boundaries = [float(x) for x in args.phases.split(",")]
                print_phase_summaries(records, boundaries)
    
    # Comparison with baseline
    if args.baseline:
        baseline_records = load_etl_log(args.baseline)
        baseline_label = os.path.basename(os.path.dirname(args.baseline)) or "Baseline"
        compare_runs(records, baseline_records, label, baseline_label)
    
    # Optional CSV output
    if args.summary_csv:
        summary = analyze_single(records, label)
        write_summary_csv(summary, args.summary_csv)


if __name__ == "__main__":
    main()
