#!/usr/bin/env python3
"""
Analyze accuracy metrics for ETL comparison runs.

Computes:
1. Thermo fidelity: Time-binned T(t), P(t) mean ± std vs target
2. Charge stats: Per-type mean, variance, max |q| over trajectory
3. RDF computation: Pair distribution functions from dump files
4. Species counting: Track coordination numbers over time

Usage:
    # Analyze single run
    python analyze_accuracy.py outputs_etl_dt/

    # Compare multiple runs against reference
    python analyze_accuracy.py outputs_etl_dt/ outputs_etl_qeq/ --ref outputs_baseline_safe/

    # Generate comparison report
    python analyze_accuracy.py --all-outputs --ref outputs_baseline_safe/
"""
import argparse
import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ThermoRecord:
    """Thermodynamic data from etl_log.csv."""
    time_ps: float
    T_target: float
    T_measured: float
    pressure: float
    pe: float
    ke: float
    etotal: float


@dataclass
class ChargeStats:
    """Charge statistics per atom type."""
    atom_type: int
    mean: float
    std: float
    min_q: float
    max_q: float
    count: int


@dataclass
class RDFResult:
    """Radial distribution function result."""
    pair: str
    r: np.ndarray
    g_r: np.ndarray
    coordination: float


def parse_etl_log(log_path: str) -> List[ThermoRecord]:
    """Parse etl_log.csv and extract thermo data."""
    records = []
    if not os.path.exists(log_path):
        return records
    
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append(ThermoRecord(
                    time_ps=float(row.get('time_ps', 0)),
                    T_target=float(row.get('T_target', 0)),
                    T_measured=float(row.get('temp', 0)),
                    pressure=float(row.get('press', 0)),
                    pe=float(row.get('pe', 0)),
                    ke=float(row.get('ke', 0)),
                    etotal=float(row.get('etotal', 0)),
                ))
            except (ValueError, KeyError):
                continue
    return records


def compute_thermo_fidelity(records: List[ThermoRecord], 
                            bin_width_ps: float = 1.0) -> Dict:
    """
    Compute time-binned temperature and pressure statistics.
    
    Returns dict with:
        - T_bins: list of (t_center, T_target, T_mean, T_std, T_error)
        - P_bins: list of (t_center, P_mean, P_std)
        - overall_T_error: mean absolute error in T
        - overall_T_std: overall T standard deviation
    """
    if not records:
        return {}
    
    # Group by time bins
    bins = defaultdict(list)
    for r in records:
        bin_idx = int(r.time_ps / bin_width_ps)
        bins[bin_idx].append(r)
    
    T_bins = []
    P_bins = []
    T_errors = []
    
    for bin_idx in sorted(bins.keys()):
        recs = bins[bin_idx]
        t_center = (bin_idx + 0.5) * bin_width_ps
        
        T_target = np.mean([r.T_target for r in recs])
        T_measured = [r.T_measured for r in recs]
        T_mean = np.mean(T_measured)
        T_std = np.std(T_measured)
        T_error = abs(T_mean - T_target)
        T_errors.append(T_error)
        
        P_measured = [r.pressure for r in recs]
        P_mean = np.mean(P_measured)
        P_std = np.std(P_measured)
        
        T_bins.append((t_center, T_target, T_mean, T_std, T_error))
        P_bins.append((t_center, P_mean, P_std))
    
    # Energy drift / stability
    etotal_vals = [r.etotal for r in records]
    etotal_std = float(np.std(etotal_vals)) if len(etotal_vals) > 1 else 0.0
    etotal_drift = (records[-1].etotal - records[0].etotal) if len(records) >= 2 else 0.0
    
    # Phase-resolved T error (plateau schedule: 60% cold, 20% ramp, 20% hot)
    t_max = max(r.time_ps for r in records) if records else 0.0
    cold = [r for r in records if r.time_ps <= t_max * 0.6]
    ramp = [r for r in records if t_max * 0.6 < r.time_ps <= t_max * 0.8]
    hot = [r for r in records if r.time_ps > t_max * 0.8]
    
    def phase_T_error(phase_recs):
        if not phase_recs:
            return float('nan')
        return float(np.mean([abs(r.T_measured - r.T_target) for r in phase_recs]))
    
    return {
        'T_bins': T_bins,
        'P_bins': P_bins,
        'overall_T_error': np.mean(T_errors) if T_errors else 0.0,
        'overall_T_std': np.std([r.T_measured for r in records]) if records else 0.0,
        'etotal_std': etotal_std,
        'etotal_drift': etotal_drift,
        'T_error_cold': phase_T_error(cold),
        'T_error_ramp': phase_T_error(ramp),
        'T_error_hot': phase_T_error(hot),
    }


def parse_dump_charges(dump_path: str) -> List[Dict[int, List[float]]]:
    """
    Parse LAMMPS dump file and extract charges per atom type.
    
    Returns list of snapshots, each a dict: {atom_type: [charges]}
    """
    snapshots = []
    
    if not os.path.exists(dump_path):
        return snapshots
    
    with open(dump_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if lines[i].startswith('ITEM: TIMESTEP'):
            i += 2  # Skip timestep value
            continue
        elif lines[i].startswith('ITEM: NUMBER OF ATOMS'):
            n_atoms = int(lines[i + 1].strip())
            i += 2
            continue
        elif lines[i].startswith('ITEM: BOX BOUNDS'):
            i += 4  # Skip box bounds
            continue
        elif lines[i].startswith('ITEM: ATOMS'):
            header = lines[i].split()[2:]  # Skip "ITEM:" and "ATOMS"
            
            # Find column indices
            type_idx = header.index('type') if 'type' in header else None
            q_idx = header.index('q') if 'q' in header else None
            
            if type_idx is None or q_idx is None:
                i += 1
                continue
            
            snapshot = defaultdict(list)
            for j in range(n_atoms):
                if i + 1 + j >= len(lines):
                    break
                parts = lines[i + 1 + j].split()
                if len(parts) > max(type_idx, q_idx):
                    atom_type = int(parts[type_idx])
                    charge = float(parts[q_idx])
                    snapshot[atom_type].append(charge)
            
            snapshots.append(dict(snapshot))
            i += n_atoms + 1
        else:
            i += 1
    
    return snapshots


def compute_charge_stats(snapshots: List[Dict[int, List[float]]]) -> List[ChargeStats]:
    """Compute charge statistics across all snapshots."""
    if not snapshots:
        return []
    
    # Aggregate charges by type
    all_charges = defaultdict(list)
    for snap in snapshots:
        for atom_type, charges in snap.items():
            all_charges[atom_type].extend(charges)
    
    stats = []
    for atom_type, charges in sorted(all_charges.items()):
        if charges:
            stats.append(ChargeStats(
                atom_type=atom_type,
                mean=np.mean(charges),
                std=np.std(charges),
                min_q=min(charges),
                max_q=max(charges),
                count=len(charges),
            ))
    
    return stats


def parse_dump_positions(dump_path: str) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Parse LAMMPS dump file and extract positions and types.
    
    Returns list of (types, positions, box) for each snapshot.
    """
    snapshots = []
    
    if not os.path.exists(dump_path):
        return snapshots
    
    with open(dump_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    n_atoms = 0
    box = np.zeros((3, 2))
    
    while i < len(lines):
        if lines[i].startswith('ITEM: TIMESTEP'):
            i += 2
            continue
        elif lines[i].startswith('ITEM: NUMBER OF ATOMS'):
            n_atoms = int(lines[i + 1].strip())
            i += 2
            continue
        elif lines[i].startswith('ITEM: BOX BOUNDS'):
            for dim in range(3):
                lo, hi = map(float, lines[i + 1 + dim].split()[:2])
                box[dim] = [lo, hi]
            i += 4
            continue
        elif lines[i].startswith('ITEM: ATOMS'):
            header = lines[i].split()[2:]
            
            type_idx = header.index('type') if 'type' in header else None
            x_idx = header.index('x') if 'x' in header else None
            y_idx = header.index('y') if 'y' in header else None
            z_idx = header.index('z') if 'z' in header else None
            
            if None in (type_idx, x_idx, y_idx, z_idx):
                i += 1
                continue
            
            types = np.zeros(n_atoms, dtype=int)
            positions = np.zeros((n_atoms, 3))
            
            for j in range(n_atoms):
                if i + 1 + j >= len(lines):
                    break
                parts = lines[i + 1 + j].split()
                if len(parts) > max(type_idx, x_idx, y_idx, z_idx):
                    types[j] = int(parts[type_idx])
                    positions[j] = [float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])]
            
            snapshots.append((types.copy(), positions.copy(), box.copy()))
            i += n_atoms + 1
        else:
            i += 1
    
    return snapshots


def compute_rdf(snapshots: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                type_pair: Tuple[int, int],
                r_max: float = 10.0,
                n_bins: int = 100) -> Optional[RDFResult]:
    """
    Compute radial distribution function for a pair of atom types.
    
    Args:
        snapshots: List of (types, positions, box) tuples
        type_pair: (type1, type2) pair to compute RDF for
        r_max: Maximum distance
        n_bins: Number of bins
    
    Returns:
        RDFResult with r values and g(r)
    """
    if not snapshots:
        return None
    
    type1, type2 = type_pair
    dr = r_max / n_bins
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    
    histogram = np.zeros(n_bins)
    n_pairs_total = 0
    volume_total = 0
    
    for types, positions, box in snapshots:
        box_lengths = box[:, 1] - box[:, 0]
        volume = np.prod(box_lengths)
        volume_total += volume
        
        # Find atoms of each type
        idx1 = np.where(types == type1)[0]
        idx2 = np.where(types == type2)[0]
        
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        
        # Compute pairwise distances with periodic boundary conditions
        for i in idx1:
            for j in idx2:
                if type1 == type2 and j <= i:
                    continue
                
                dr_vec = positions[j] - positions[i]
                # Minimum image convention
                dr_vec -= box_lengths * np.round(dr_vec / box_lengths)
                dist = np.linalg.norm(dr_vec)
                
                if dist < r_max:
                    bin_idx = int(dist / dr)
                    if 0 <= bin_idx < n_bins:
                        histogram[bin_idx] += 1
        
        if type1 == type2:
            n_pairs_total += len(idx1) * (len(idx1) - 1) // 2
        else:
            n_pairs_total += len(idx1) * len(idx2)
    
    if n_pairs_total == 0 or volume_total == 0:
        return None
    
    # Normalize to g(r)
    n_snapshots = len(snapshots)
    avg_volume = volume_total / n_snapshots
    avg_n_pairs = n_pairs_total / n_snapshots
    
    # Average density of type2 atoms
    n_type2 = np.mean([np.sum(types == type2) for types, _, _ in snapshots])
    rho = n_type2 / avg_volume
    
    # Shell volume normalization
    shell_volumes = 4.0 / 3.0 * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)
    
    # g(r) = histogram / (n_snapshots * n_type1 * rho * shell_volume)
    n_type1 = np.mean([np.sum(types == type1) for types, _, _ in snapshots])
    g_r = histogram / (n_snapshots * n_type1 * rho * shell_volumes + 1e-10)
    
    # Compute coordination number (integral of g(r) * 4πr²ρ dr up to first minimum or r_max)
    coord = np.trapz(g_r * 4 * np.pi * r_centers**2 * rho, r_centers)
    
    pair_name = f"{type1}-{type2}"
    return RDFResult(pair=pair_name, r=r_centers, g_r=g_r, coordination=coord)


def analyze_output_dir(output_dir: str, compute_rdf_flag: bool = True) -> Dict:
    """
    Analyze a single output directory.
    
    Returns dict with all metrics.
    """
    results = {'dir': output_dir}
    
    # Thermo analysis from etl_log.csv
    log_path = os.path.join(output_dir, 'etl_log.csv')
    thermo_records = parse_etl_log(log_path)
    if thermo_records:
        results['thermo'] = compute_thermo_fidelity(thermo_records)
        results['n_records'] = len(thermo_records)
    
    # Charge analysis from dump file
    dump_path = os.path.join(output_dir, 'dump.reax')
    if not os.path.exists(dump_path):
        # Try other common dump names
        for name in ['dump.lammpstrj', 'dump.reax.lammpstrj', 'dump.custom']:
            alt_path = os.path.join(output_dir, name)
            if os.path.exists(alt_path):
                dump_path = alt_path
                break
    
    if os.path.exists(dump_path):
        charge_snapshots = parse_dump_charges(dump_path)
        if charge_snapshots:
            results['charge_stats'] = compute_charge_stats(charge_snapshots)
            results['n_charge_snapshots'] = len(charge_snapshots)
        
        # RDF analysis
        if compute_rdf_flag:
            position_snapshots = parse_dump_positions(dump_path)
            if position_snapshots:
                results['rdfs'] = {}
                # Common pairs for CHO systems
                type_pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
                for pair in type_pairs:
                    rdf = compute_rdf(position_snapshots, pair)
                    if rdf is not None:
                        results['rdfs'][rdf.pair] = rdf
    
    return results


def compare_results(results: List[Dict], ref_results: Optional[Dict] = None) -> str:
    """Generate comparison report."""
    lines = []
    lines.append("=" * 70)
    lines.append("ETL ACCURACY COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Summary table
    lines.append("THERMO FIDELITY SUMMARY")
    lines.append("-" * 70)
    lines.append(f"{'Run':<30} {'T_err(K)':<10} {'T_std(K)':<10} {'E_drift':<12} {'E_std':<12} {'Records':<8}")
    lines.append("-" * 70)
    
    for r in results:
        name = os.path.basename(r.get('dir', 'unknown'))
        thermo = r.get('thermo', {})
        T_err = thermo.get('overall_T_error', float('nan'))
        T_std = thermo.get('overall_T_std', float('nan'))
        E_drift = thermo.get('etotal_drift', float('nan'))
        E_std = thermo.get('etotal_std', float('nan'))
        n_rec = r.get('n_records', 0)
        lines.append(f"{name:<30} {T_err:<10.1f} {T_std:<10.1f} {E_drift:<12.2f} {E_std:<12.2f} {n_rec:<8}")
    
    lines.append("")
    lines.append("PHASE-RESOLVED T ERROR (cold / ramp / hot) [K]")
    lines.append("-" * 70)
    for r in results:
        name = os.path.basename(r.get('dir', 'unknown'))
        thermo = r.get('thermo', {})
        Tc = thermo.get('T_error_cold', float('nan'))
        Tr = thermo.get('T_error_ramp', float('nan'))
        Th = thermo.get('T_error_hot', float('nan'))
        lines.append(f"  {name:<28} cold: {Tc:.1f}   ramp: {Tr:.1f}   hot: {Th:.1f}")
    
    lines.append("")
    lines.append("CHARGE STATISTICS")
    lines.append("-" * 70)
    
    for r in results:
        name = os.path.basename(r.get('dir', 'unknown'))
        stats = r.get('charge_stats', [])
        if stats:
            lines.append(f"\n{name}:")
            lines.append(f"  {'Type':<6} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
            for s in stats:
                lines.append(f"  {s.atom_type:<6} {s.mean:<10.4f} {s.std:<10.4f} {s.min_q:<10.4f} {s.max_q:<10.4f}")
    
    # RDF comparison (if reference provided)
    if ref_results and 'rdfs' in ref_results:
        lines.append("")
        lines.append("RDF COMPARISON (vs reference)")
        lines.append("-" * 70)
        lines.append(f"{'Run':<30} {'Pair':<8} {'Coord_ref':<12} {'Coord':<12} {'Diff':<10}")
        lines.append("-" * 70)
        
        for r in results:
            name = os.path.basename(r.get('dir', 'unknown'))
            rdfs = r.get('rdfs', {})
            for pair, ref_rdf in ref_results['rdfs'].items():
                if pair in rdfs:
                    rdf = rdfs[pair]
                    diff = abs(rdf.coordination - ref_rdf.coordination)
                    lines.append(f"{name:<30} {pair:<8} {ref_rdf.coordination:<12.2f} {rdf.coordination:<12.2f} {diff:<10.4f}")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze accuracy metrics for ETL comparison runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_dirs", nargs="*", help="Output directories to analyze")
    parser.add_argument("--ref", type=str, help="Reference output directory for comparison")
    parser.add_argument("--all-outputs", action="store_true", 
                        help="Auto-detect all outputs_* directories")
    parser.add_argument("--no-rdf", action="store_true", help="Skip RDF computation (faster)")
    parser.add_argument("--output", "-o", type=str, help="Write report to file")
    
    args = parser.parse_args()
    
    output_dirs = list(args.output_dirs)
    
    if args.all_outputs:
        for name in os.listdir('.'):
            if name.startswith('outputs_') and os.path.isdir(name):
                if name not in output_dirs:
                    output_dirs.append(name)
    
    if not output_dirs:
        print("No output directories specified. Use --all-outputs or provide paths.")
        return
    
    print(f"Analyzing {len(output_dirs)} directories...")
    
    results = []
    for d in sorted(output_dirs):
        print(f"  Processing {d}...")
        r = analyze_output_dir(d, compute_rdf_flag=not args.no_rdf)
        results.append(r)
    
    ref_results = None
    if args.ref:
        print(f"  Processing reference {args.ref}...")
        ref_results = analyze_output_dir(args.ref, compute_rdf_flag=not args.no_rdf)
    
    report = compare_results(results, ref_results)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport written to {args.output}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
