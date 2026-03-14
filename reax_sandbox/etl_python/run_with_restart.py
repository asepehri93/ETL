#!/usr/bin/env python3
"""
Restart-based workflow for ETL comparison runs.

This script ensures all runs (baseline, ETL variants) start from the same
exact microscopic state by using LAMMPS restart files.

Workflow:
1. Generate data file (if needed)
2. Write restart: setup + warm-up + write_restart
3. Run comparison cases from the same restart

Usage:
    # Write restart file (with optional longer warmup)
    python run_with_restart.py --write-restart --warmup-ps 5.0

    # Baselines (conservative/moderate/aggressive dt and QEq)
    python run_with_restart.py --baseline-conservative --t-ps 50
    python run_with_restart.py --baseline-moderate --t-ps 50
    python run_with_restart.py --baseline-aggressive --t-ps 50

    # ETL variants
    python run_with_restart.py --etl-dt --t-ps 50
    python run_with_restart.py --etl-qeq --t-ps 50
    python run_with_restart.py --qeq-only --t-ps 50  # fixed dt + adaptive QEq
    python run_with_restart.py --etl-full --t-ps 50

    # Temperature ramp (300 K -> 3000 K over t-ps)
    python run_with_restart.py --etl-dt --t-ps 50 --T-start 300 --T-end 3000 --ramp
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from lammps import lammps
except ImportError:
    print("LAMMPS Python module not found. Please install it.")
    sys.exit(1)

from etl_controller import (
    ETLController, ETLParams, make_linear_ramp, make_constant_temp, 
    make_triangle_ramp, make_plateau_ramp_plateau, make_hat_schedule,
    AdaptiveTempController
)
from convert_dumps_to_xyz import convert_directory


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SANDBOX_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SANDBOX_DIR)

DEFAULT_FFIELD = os.path.join(PROJECT_ROOT, "ffield.txt")
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "data.gasmix_cho.lmp")
DEFAULT_RESTART = os.path.join(SCRIPT_DIR, "restart.equil")

# Default temperatures
T_DEFAULT = 1500.0
T_RAMP_START = 300.0
T_RAMP_END = 3000.0
WARMUP_PS = 5.0  # Increased default warmup for better equilibration

# Baseline presets: (dt_fs, qeq_tol)
BASELINE_PRESETS = {
    "safe": (0.1, 1.0e-6),            # Primary reference (standard practice for reactive ReaxFF)
    "aggressive": (0.25, 1.0e-5),     # Common aggressive practice
    "ultrafast": (0.50, 1.0e-4),      # Ultra-fast (expected to crash/degrade - shows ETL's value)
    "conservative": (0.05, 1.0e-7),   # Ultra-conservative (accuracy validation only)
}


def make_lmp(log_file: str = "log.lammps", screen_file: str = "screen.txt") -> "lammps":
    """Create a new LAMMPS instance."""
    lmp = lammps(cmdargs=["-log", log_file, "-screen", screen_file])
    return lmp


def setup_reaxff(lmp: "lammps", data_file: str, ffield_path: str) -> None:
    """Setup ReaxFF system from a data file."""
    lmp.command("units real")
    lmp.command("atom_style charge")
    lmp.command("boundary p p p")
    lmp.command(f'read_data "{data_file}"')
    lmp.command("pair_style reaxff NULL")
    lmp.command(f'pair_coeff * * "{ffield_path}" C H O')
    lmp.command("neighbor 2.0 bin")
    lmp.command("neigh_modify every 1 delay 0 check yes")
    lmp.command("fix q all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 200")


def reinit_reaxff_after_restart(lmp: "lammps", ffield_path: str) -> None:
    """
    Reinitialize ReaxFF after reading a restart file.
    
    Restart files don't store pair_style/pair_coeff, so we must redefine them.
    Fixes may be preserved in the restart, so we unfix before redefining.
    """
    lmp.command("pair_style reaxff NULL")
    lmp.command(f'pair_coeff * * "{ffield_path}" C H O')
    lmp.command("neighbor 2.0 bin")
    lmp.command("neigh_modify every 1 delay 0 check yes")
    try:
        lmp.command("unfix q")
    except Exception:
        pass
    lmp.command("fix q all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 200")


def ensure_integrator_and_thermostat(
    lmp: "lammps",
    T: float = T_DEFAULT,
    damp: float = 100.0,
    seed: int = 90421,
) -> None:
    """
    Ensure NVE + Langevin thermostat are defined.
    
    If Python will control Langevin adaptively, it will redefine 'fix lang' later.
    """
    try:
        lmp.command("unfix nve")
    except Exception:
        pass
    try:
        lmp.command("unfix lang")
    except Exception:
        pass

    lmp.command("fix nve all nve")
    lmp.command(f"fix lang all langevin {T} {T} {damp} {seed}")


def write_restart_file(
    data_file: str = DEFAULT_DATA,
    ffield_path: str = DEFAULT_FFIELD,
    restart_file: str = DEFAULT_RESTART,
    warmup_ps: float = WARMUP_PS,
    T: float = T_DEFAULT,
) -> None:
    """
    Create a restart file after equilibration.
    
    Steps:
    1. Read data file
    2. Setup ReaxFF
    3. Minimize
    4. Setup integrator + thermostat
    5. Initialize velocities
    6. Run warmup
    7. Write restart
    """
    out_dir = os.path.dirname(restart_file) or SCRIPT_DIR
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, "log.warmup")
    screen_file = os.path.join(out_dir, "screen.warmup")

    lmp = make_lmp(log_file, screen_file)

    print(f"Setting up ReaxFF from {data_file}...")
    setup_reaxff(lmp, data_file, ffield_path)

    print("Running minimization...")
    lmp.command("min_style cg")
    lmp.command("minimize 1.0e-4 1.0e-6 500 5000")

    print(f"Setting up integrator + Langevin at {T} K...")
    ensure_integrator_and_thermostat(lmp, T=T)

    print("Initializing velocities...")
    lmp.command(f"velocity all create {T} 492783 mom yes rot yes dist gaussian")

    lmp.command("thermo 100")
    lmp.command("thermo_style custom step time temp press pe ke etotal")

    warmup_fs = warmup_ps * 1000.0
    dt_fs = 0.1
    warmup_steps = int(warmup_fs / dt_fs)

    print(f"Running warmup for {warmup_ps} ps ({warmup_steps} steps)...")
    lmp.command(f"timestep {dt_fs}")
    lmp.command(f"run {warmup_steps}")

    print(f"Writing restart to {restart_file}...")
    lmp.command(f'write_restart "{restart_file}"')

    lmp.close()
    print("Done writing restart file.")


def run_from_restart(
    restart_file: str,
    ffield_path: str,
    out_dir: str,
    t_ps: float,
    adapt_dt: bool,
    adapt_qeq: bool,
    adapt_langevin: bool = False,
    adapt_T: bool = False,
    fixed_dt_fs: float = 0.1,
    fixed_tol: float = 1.0e-6,
    T: float = T_DEFAULT,
    T_start: Optional[float] = None,
    T_end: Optional[float] = None,
    use_ramp: bool = False,
    ramp_style: str = "linear",
    dt_hysteresis: float = 0.05,
    dt_target0_fs: Optional[float] = 0.15,
    snap_every_ps: float = 0.5,
    lang_damp_fs: float = 100.0,
    label: str = "",
) -> float:
    """
    Run simulation from a restart file with the specified ETL settings.
    
    Args:
        restart_file: Path to LAMMPS restart file
        ffield_path: Path to ReaxFF force field file
        out_dir: Output directory for logs and dumps
        t_ps: Simulation time in picoseconds
        adapt_dt: Enable adaptive timestep
        adapt_qeq: Enable adaptive QEq tolerance
        adapt_langevin: Enable adaptive Langevin damping
        adapt_T: Enable adaptive temperature stepping (ETL-based)
        fixed_dt_fs: Fixed timestep (used when adapt_dt=False)
        fixed_tol: Fixed QEq tolerance (used when adapt_qeq=False)
        T: Constant temperature (used when not ramping)
        T_start: Starting temperature for ramp
        T_end: Ending temperature for ramp
        use_ramp: If True, use linear temperature ramp (fixed schedule)
        dt_hysteresis: Minimum relative change to update dt (0.05 = 5%)
        label: Optional label for logging
    
    Returns:
        Wall time in seconds
    """
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, "log.lammps")
    screen_file = os.path.join(out_dir, "screen.txt")

    lmp = make_lmp(log_file, screen_file)

    print(f"Reading restart from {restart_file}...")
    lmp.command(f'read_restart "{restart_file}"')

    print("Reinitializing ReaxFF...")
    reinit_reaxff_after_restart(lmp, ffield_path)

    lmp.command("thermo 100")
    lmp.command("thermo_style custom step time temp press pe ke etotal")

    # Determine temperature schedule
    T_s = T_start if T_start is not None else T_RAMP_START
    T_e = T_end if T_end is not None else T_RAMP_END
    
    if adapt_T:
        # Adaptive temperature stepping - no fixed schedule
        T_schedule = None
        T_init = T_s
        T_ref = (T_s + T_e) / 2
        print(f"Adaptive temperature stepping: {T_s} K -> {T_e} K (adaptive pace)")
    elif use_ramp:
        if ramp_style == "triangle":
            T_schedule = make_triangle_ramp(T_s, T_e, t_ps)
            T_init = T_s
            T_ref = (T_s + T_e) / 2
            print(f"Triangle temperature ramp: {T_s} K -> {T_e} K -> {T_s} K over {t_ps} ps")
        elif ramp_style == "plateau":
            # Use 60% cold, 20% ramp, 20% hot to emphasize calm-phase savings
            T_schedule = make_plateau_ramp_plateau(T_s, T_e, t_ps, low_frac=0.60, high_frac=0.20)
            T_init = T_s
            T_ref = (T_s + T_e) / 2
            print(f"Plateau temperature ramp: {T_s} K (60%) -> ramp (20%) -> {T_e} K (20%) over {t_ps} ps")
        elif ramp_style == "hat":
            T_schedule = make_hat_schedule(T_s, T_e, t_ps,
                                           cold1_frac=0.30, up_frac=0.20,
                                           down_frac=0.20, cold2_frac=0.30)
            T_init = T_s
            T_ref = (T_s + T_e) / 2
            print(f"Hat schedule: {T_s} K (30%) -> ramp up to {T_e} K (20%) "
                  f"-> ramp down to {T_s} K (20%) -> {T_s} K (30%) over {t_ps} ps")
        else:
            T_schedule = make_linear_ramp(T_s, T_e, t_ps)
            T_init = T_s
            T_ref = (T_s + T_e) / 2
            print(f"Linear temperature ramp: {T_s} K -> {T_e} K over {t_ps} ps")
    else:
        T_schedule = make_constant_temp(T)
        T_init = T
        T_ref = T
        print(f"Constant temperature: {T} K")

    # Setup integrator; Langevin will be controlled by ETLController
    if not adapt_langevin:
        ensure_integrator_and_thermostat(lmp, T=T_init, damp=lang_damp_fs)
    else:
        for fix_id in ("nve", "lang", "int"):
            try:
                lmp.command(f"unfix {fix_id}")
            except Exception:
                pass
        lmp.command("fix nve all nve")

    params = ETLParams(
        dt_target0_fs=dt_target0_fs,  # Auto-calibrate Delta_l from this target
        T_target=T_ref,  # Reference T for kBT scaling
        lang_temp=T_init,  # Initial thermostat temperature
        lang_damp_fixed_fs=lang_damp_fs,
        tol_fixed=fixed_tol,
        out_dir=out_dir,
        snap_every_ps=snap_every_ps,
        chunk_steps=10,
        dt_hysteresis=dt_hysteresis,
        # Adaptive temperature stepping parameters
        T_start=T_s,
        T_end=T_e,
    )

    controller = ETLController(
        lmp=lmp,
        params=params,
        adapt_dt=adapt_dt,
        adapt_qeq=adapt_qeq,
        adapt_langevin=adapt_langevin,
        adapt_T=adapt_T,
        fixed_dt_fs=fixed_dt_fs,
        T_schedule=T_schedule,
    )

    mode_str = []
    if label:
        mode_str.append(f"[{label}]")
    if adapt_dt:
        mode_str.append("adaptive-dt")
    else:
        mode_str.append(f"fixed-dt={fixed_dt_fs}fs")
    if adapt_qeq:
        mode_str.append("adaptive-qeq")
    else:
        mode_str.append(f"fixed-qeq={fixed_tol:.0e}")
    if adapt_langevin:
        mode_str.append("adaptive-langevin")
    if adapt_T:
        mode_str.append("adaptive-T")
    elif use_ramp:
        mode_str.append("fixed-T-ramp")
    print(f"Running {' + '.join(mode_str)} for {t_ps} ps...")

    wall_start = time.time()
    controller.run(t_ps)
    wall_end = time.time()

    wall_time = wall_end - wall_start

    lmp.close()

    print(f"Done. Wall time: {wall_time:.2f} s")
    return wall_time


def main():
    parser = argparse.ArgumentParser(
        description="Restart-based ETL comparison workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create restart file with 5ps warmup
    python run_with_restart.py --write-restart --warmup-ps 5.0 --T 300

    # Run baselines (safe is the primary reference)
    python run_with_restart.py --baseline-safe --t-ps 15 --ramp --ramp-style plateau
    python run_with_restart.py --baseline-aggressive --t-ps 15 --ramp --ramp-style plateau

    # Run ETL variants with plateau temperature ramp
    python run_with_restart.py --etl-dt --t-ps 15 --ramp --ramp-style plateau
    python run_with_restart.py --etl-qeq --t-ps 15 --ramp --ramp-style plateau

    # Run clean comparison suite (safe, aggressive, etl-dt, etl-qeq)
    python run_with_restart.py --all --t-ps 15 --ramp --ramp-style plateau
""",
    )

    # Workflow steps
    parser.add_argument("--write-restart", action="store_true", help="Write restart file from data")
    parser.add_argument("--warmup-ps", type=float, default=WARMUP_PS, help="Warmup time before writing restart (ps)")

    # Baseline modes
    parser.add_argument("--baseline-safe", action="store_true", help="Run safe baseline (dt=0.1 fs, tol=1e-6) - primary reference")
    parser.add_argument("--baseline-aggressive", action="store_true", help="Run aggressive baseline (dt=0.25 fs, tol=1e-5) - common fast practice")
    parser.add_argument("--baseline-ultrafast", action="store_true", help="Run ultrafast baseline (dt=0.5 fs, tol=1e-4) - expected to fail/degrade")
    parser.add_argument("--baseline-conservative", action="store_true", help="Run conservative baseline (dt=0.05 fs, tol=1e-7) - accuracy validation")

    # ETL modes
    parser.add_argument("--etl-dt", action="store_true", help="Run ETL with adaptive dt only (fixed QEq, fixed damping)")
    parser.add_argument("--etl-qeq", action="store_true", help="Run ETL with adaptive dt + adaptive QEq (fixed damping)")
    parser.add_argument("--qeq-only", action="store_true", help="Run fixed dt + adaptive QEq (isolation case, not in --all)")
    parser.add_argument("--etl-full", action="store_true", help="Run full ETL with adaptive Langevin (advanced, not in --all)")
    parser.add_argument("--etl-adaptive-T", action="store_true", help="Run ETL with adaptive temperature stepping (advanced, not in --all)")

    # Batch mode
    parser.add_argument("--all", action="store_true", help="Run clean comparison: safe, aggressive, ultrafast, etl-dt, etl-qeq")

    # Simulation parameters
    parser.add_argument("--t-ps", type=float, default=50.0, help="Simulation time in ps (default: 50)")
    parser.add_argument("--T", type=float, default=T_DEFAULT, help="Target temperature in K (constant mode)")

    # Temperature ramp parameters
    parser.add_argument("--ramp", action="store_true", help="Use temperature ramp")
    parser.add_argument("--ramp-style", type=str, choices=["linear", "triangle", "plateau", "hat"], default="plateau",
                        help="Ramp style: linear, triangle, plateau, or hat (cold->up->down->cold)")
    parser.add_argument("--T-start", type=float, default=T_RAMP_START, help="Ramp start temperature (K)")
    parser.add_argument("--T-end", type=float, default=T_RAMP_END, help="Ramp end temperature (K)")

    # ETL tuning
    parser.add_argument("--lang-damp", type=float, default=100.0, help="Langevin damping time in fs (default: 100)")
    parser.add_argument("--dt-hysteresis", type=float, default=0.0, help="dt hysteresis threshold (0 = disabled)")
    parser.add_argument("--dt-target", type=float, default=0.20, help="Target dt for Delta_l auto-calibration (fs) - calibrated at cold start")

    # File paths
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Data file path")
    parser.add_argument("--ffield", type=str, default=DEFAULT_FFIELD, help="Force field file path")
    parser.add_argument("--restart", type=str, default=DEFAULT_RESTART, help="Restart file path")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (auto-generated if not set)")
    parser.add_argument("--snap-every-ps", type=float, default=0.5, help="Dump snapshot interval in ps (default: 0.5)")

    # Post-processing
    parser.add_argument("--write-xyz", action="store_true", help="Convert dumps to XYZ trajectory after each run")

    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)

    if args.write_restart:
        # Use T_start for warmup if ramping, else T
        warmup_T = args.T_start if args.ramp else args.T
        write_restart_file(
            data_file=args.data,
            ffield_path=args.ffield,
            restart_file=args.restart,
            warmup_ps=args.warmup_ps,
            T=warmup_T,
        )
        return

    if not os.path.exists(args.restart):
        print(f"Restart file not found: {args.restart}")
        print("Run with --write-restart first, or specify --restart path.")
        sys.exit(1)

    # Expand --all into clean comparison suite (fixed damping for all)
    if args.all:
        args.baseline_safe = True
        args.baseline_aggressive = True
        args.baseline_ultrafast = True
        args.etl_dt = True
        args.etl_qeq = True

    # Common kwargs for all runs
    common_kwargs = dict(
        restart_file=args.restart,
        ffield_path=args.ffield,
        t_ps=args.t_ps,
        T=args.T,
        T_start=args.T_start,
        T_end=args.T_end,
        use_ramp=args.ramp,
        ramp_style=getattr(args, 'ramp_style', 'linear'),
        dt_hysteresis=args.dt_hysteresis,
        dt_target0_fs=getattr(args, 'dt_target', 0.15),
        snap_every_ps=getattr(args, 'snap_every_ps', 0.5),
        lang_damp_fs=getattr(args, 'lang_damp', 100.0),
    )

    wall_times = {}
    output_dirs = []  # Track output directories for XYZ conversion

    # Safe baseline (primary reference)
    if args.baseline_safe:
        dt, tol = BASELINE_PRESETS["safe"]
        out_dir = args.out_dir or "outputs_baseline_safe"
        wt = run_from_restart(
            out_dir=out_dir,
            adapt_dt=False,
            adapt_qeq=False,
            fixed_dt_fs=dt,
            fixed_tol=tol,
            label="safe",
            **common_kwargs,
        )
        wall_times["baseline_safe"] = wt
        output_dirs.append(out_dir)

    # Aggressive baseline (stress test)
    if args.baseline_aggressive:
        dt, tol = BASELINE_PRESETS["aggressive"]
        out_dir = args.out_dir or "outputs_baseline_aggressive"
        wt = run_from_restart(
            out_dir=out_dir,
            adapt_dt=False,
            adapt_qeq=False,
            fixed_dt_fs=dt,
            fixed_tol=tol,
            label="aggressive",
            **common_kwargs,
        )
        wall_times["baseline_aggressive"] = wt
        output_dirs.append(out_dir)

    # Ultrafast baseline (expected to fail/degrade - demonstrates ETL's value)
    if args.baseline_ultrafast:
        dt, tol = BASELINE_PRESETS["ultrafast"]
        out_dir = args.out_dir or "outputs_baseline_ultrafast"
        try:
            wt = run_from_restart(
                out_dir=out_dir,
                adapt_dt=False,
                adapt_qeq=False,
                fixed_dt_fs=dt,
                fixed_tol=tol,
                label="ultrafast",
                **common_kwargs,
            )
            wall_times["baseline_ultrafast"] = wt
        except Exception as e:
            print(f"[ultrafast] CRASHED as expected: {e}")
            wall_times["baseline_ultrafast"] = float('inf')  # Mark as crashed
        output_dirs.append(out_dir)

    # Conservative baseline (accuracy validation)
    if args.baseline_conservative:
        dt, tol = BASELINE_PRESETS["conservative"]
        out_dir = args.out_dir or "outputs_baseline_conservative"
        wt = run_from_restart(
            out_dir=out_dir,
            adapt_dt=False,
            adapt_qeq=False,
            fixed_dt_fs=dt,
            fixed_tol=tol,
            label="conservative",
            **common_kwargs,
        )
        wall_times["baseline_conservative"] = wt
        output_dirs.append(out_dir)

    # ETL(dt) - adaptive dt, fixed QEq
    if args.etl_dt:
        _, tol = BASELINE_PRESETS["safe"]
        out_dir = args.out_dir or "outputs_etl_dt"
        wt = run_from_restart(
            out_dir=out_dir,
            adapt_dt=True,
            adapt_qeq=False,
            fixed_tol=tol,
            label="ETL(dt)",
            **common_kwargs,
        )
        wall_times["etl_dt"] = wt
        output_dirs.append(out_dir)

    # ETL(dt+QEq) - adaptive dt + adaptive QEq
    if args.etl_qeq:
        out_dir = args.out_dir or "outputs_etl_qeq"
        wt = run_from_restart(
            out_dir=out_dir,
            adapt_dt=True,
            adapt_qeq=True,
            label="ETL(dt+QEq)",
            **common_kwargs,
        )
        wall_times["etl_qeq"] = wt
        output_dirs.append(out_dir)

    # QEq-only isolation - fixed dt + adaptive QEq
    if args.qeq_only:
        dt, _ = BASELINE_PRESETS["safe"]
        out_dir = args.out_dir or "outputs_qeq_only"
        wt = run_from_restart(
            out_dir=out_dir,
            adapt_dt=False,
            adapt_qeq=True,
            fixed_dt_fs=dt,
            label="QEq-only",
            **common_kwargs,
        )
        wall_times["qeq_only"] = wt
        output_dirs.append(out_dir)

    # ETL(full) - adaptive dt + adaptive QEq + adaptive Langevin
    if args.etl_full:
        out_dir = args.out_dir or "outputs_etl_full"
        wt = run_from_restart(
            out_dir=out_dir,
            adapt_dt=True,
            adapt_qeq=True,
            adapt_langevin=True,
            label="ETL(full)",
            **common_kwargs,
        )
        wall_times["etl_full"] = wt
        output_dirs.append(out_dir)

    # ETL(adaptive-T) - adaptive dt + adaptive QEq + adaptive temperature stepping (Phase 2b)
    if args.etl_adaptive_T:
        out_dir = args.out_dir or "outputs_etl_adaptive_T"
        wt = run_from_restart(
            out_dir=out_dir,
            adapt_dt=True,
            adapt_qeq=True,
            adapt_T=True,
            label="ETL(adaptive-T)",
            **common_kwargs,
        )
        wall_times["etl_adaptive_T"] = wt
        output_dirs.append(out_dir)

    if wall_times:
        print("\n" + "=" * 50)
        print("WALL TIME SUMMARY")
        print("=" * 50)
        for name, wt in sorted(wall_times.items()):
            print(f"  {name:25s}: {wt:8.2f} s")

        # Compute speedups vs conservative baseline (the reference)
        ref_key = "baseline_conservative"
        if ref_key in wall_times and len(wall_times) > 1:
            ref_wt = wall_times[ref_key]
            print("\n" + "=" * 50)
            print(f"SPEEDUPS vs {ref_key}")
            print("=" * 50)
            for name, wt in sorted(wall_times.items()):
                if name != ref_key:
                    speedup = ref_wt / wt if wt > 0 else float("inf")
                    print(f"  {name:25s}: {speedup:6.2f}x")

    # Convert dumps to XYZ if requested
    if getattr(args, 'write_xyz', False) and output_dirs:
        print("\n" + "=" * 50)
        print("CONVERTING DUMPS TO XYZ")
        print("=" * 50)
        for out_dir in output_dirs:
            dumps_dir = os.path.join(out_dir, "dumps")
            if os.path.isdir(dumps_dir):
                xyz_file = os.path.join(out_dir, "trajectory.xyz")
                print(f"  Converting {dumps_dir} -> {xyz_file}")
                try:
                    convert_directory(dumps_dir, xyz_file)
                except Exception as e:
                    print(f"    Warning: conversion failed: {e}")


if __name__ == "__main__":
    main()
