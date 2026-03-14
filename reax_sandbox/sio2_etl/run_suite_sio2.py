#!/usr/bin/env python3
"""
SiO2 full suite: baselines + ETL runs at 10 ps (hat schedule, 300→4500→300 K).

Uses shared restart from silica.data for reproducibility. Monitors dt, tol, Sbar,
charge dynamics (q_t1_mean, q_t2_mean, q_std), and T in real time.

Progress is printed every 0.2 ps (with flush) so runs don't appear stalled.
Expect ~1.5–2 h wall time per 10 ps run (ReaxFF + 576 atoms); full suite ~7–10 h.

Usage:
  # Create shared restart (once)
  python run_suite_sio2.py --write-restart

  # Run full suite (10 ps, hat, snap=0.01 ps, damp=100 fs)
  python run_suite_sio2.py --all

  # Run full suite in parallel (each case in a separate process)
  python run_suite_sio2.py --all --parallel

  # Single mode
  python run_suite_sio2.py --baseline-moderate --out-dir out_moderate

  # Larger system (1500 atoms): only hat and constant_T, only baseline_moderate and etl_full (placeholders for other 5 cases)
  python run_suite_sio2.py --write-restart --large   # create restart_sio2_1500.rst from silica_1500.data (once)
  python run_suite_sio2.py --large-suite             # run hat + constant_T, moderate + etl_full; outputs under outputs_sio2_large_*

Baselines: reference (dt=0.1 fs), moderate (dt=0.25 fs), aggressive (dt=0.5 fs). Speedups reported vs reference and vs moderate.
"""
import argparse
import os
import subprocess
import sys
import time

# Allow importing from etl_python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ETL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "etl_python")
if ETL_DIR not in sys.path:
    sys.path.insert(0, ETL_DIR)

try:
    from lammps import lammps
except ImportError:
    print("LAMMPS Python module not found.")
    sys.exit(1)

from etl_controller import (
    ETLController,
    ETLParams,
    make_constant_temp,
    make_hat_schedule,
)

# SiO2 paths
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "silica.data")
DEFAULT_FFIELD = os.path.join(SCRIPT_DIR, "ffield.reax.CHOFe")
DEFAULT_RESTART = os.path.join(SCRIPT_DIR, "restart.sio2")

# Suite defaults (match your 4500 K test); 10 ps keeps full suite manageable
T_START = 300.0
T_END = 4500.0
T_PS = 10.0
SNAP_EVERY_PS = 0.05   # Baseline runs used 0.01 for dense trajectories; 0.05 reduces I/O
LANG_DAMP_FS = 100.0
# reference = 0.1 fs (CHO-style reference for larger speedup); moderate = 0.25 fs; aggressive = 0.5 fs
BASELINE_PRESETS = {
    "reference": (0.10, 1.0e-6),
    "moderate": (0.25, 1.0e-5),
    "aggressive": (0.50, 1.0e-4),
}
# Si = type 1, O = type 2 (from silica.data)
TYPE_MASS_SIO2 = {1: 28.0855, 2: 15.9994}

# Larger system (e.g. 1500 atoms): data and restart file names (in SCRIPT_DIR)
LARGE_DATA = "silica_1500.data"
LARGE_RESTART = "restart_sio2_1500.rst"

# All 7 case names (for placeholder summary when running reduced large-suite)
ALL_CASE_NAMES = [
    "baseline_reference",
    "baseline_moderate",
    "baseline_aggressive",
    "etl_dt",
    "etl_dt_ramp",
    "etl_qeq",
    "etl_full",
]


def make_lmp(log_file: str, screen_file: str):
    return lammps(cmdargs=["-log", log_file, "-screen", screen_file])


def ensure_integrator_and_thermostat(lmp, T: float = 300.0, damp: float = 100.0, seed: int = 48279):
    for fix_id in ("nve", "lang"):
        try:
            lmp.command(f"unfix {fix_id}")
        except Exception:
            pass
    lmp.command("fix nve all nve")
    lmp.command(f"fix lang all langevin {T} {T} {damp} {seed}")


def write_restart_sio2(
    data_file: str = DEFAULT_DATA,
    ffield_path: str = DEFAULT_FFIELD,
    restart_file: str = DEFAULT_RESTART,
    warmup_ps: float = 1.0,
    T: float = T_START,
):
    os.makedirs(os.path.dirname(restart_file) or ".", exist_ok=True)
    log_file = os.path.join(os.path.dirname(restart_file), "log.warmup_sio2")
    lmp = make_lmp(log_file, os.path.join(os.path.dirname(restart_file), "screen.warmup_sio2"))

    print("Setting up SiO2 from data...")
    lmp.command("units real")
    lmp.command("atom_style full")
    lmp.command("boundary p p p")
    lmp.command(f'read_data "{data_file}"')
    lmp.command("pair_style reaxff NULL safezone 3.0 mincap 150")
    lmp.command(f'pair_coeff * * "{ffield_path}" Si O')
    lmp.command("neighbor 2.0 bin")
    lmp.command("neigh_modify every 1 delay 0 check yes")
    lmp.command("fix qeq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 400")

    print("Minimizing...")
    lmp.command("min_style cg")
    lmp.command("minimize 1.0e-4 1.0e-6 200 2000")
    lmp.command("reset_timestep 0")
    lmp.command(f"velocity all create {T} 48279 dist gaussian")
    ensure_integrator_and_thermostat(lmp, T=T, damp=LANG_DAMP_FS)
    lmp.command("thermo 100")
    lmp.command("thermo_style custom step time temp press pe ke etotal")

    warmup_steps = int(warmup_ps * 1000.0 / 0.5)
    lmp.command("timestep 0.5")
    print(f"Warmup {warmup_ps} ps ({warmup_steps} steps)...")
    lmp.command(f"run {warmup_steps}")
    lmp.command("reset_timestep 0")
    print(f"Writing restart to {restart_file}...")
    lmp.command(f'write_restart "{restart_file}"')
    lmp.close()
    print("Done.")


def reinit_reaxff_sio2(lmp, ffield_path: str):
    try:
        lmp.command("unfix q")
    except Exception:
        pass
    lmp.command("pair_style reaxff NULL safezone 3.0 mincap 150")
    lmp.command(f'pair_coeff * * "{ffield_path}" Si O')
    lmp.command("neighbor 2.0 bin")
    lmp.command("neigh_modify every 1 delay 0 check yes")


def progress_monitor(label: str, interval_ps: float = 0.2):
    """Print progress every interval_ps (default 0.2 ps). Shows ramp (schedule position) and ETL step savings vs safe."""
    last_printed = [0.0]  # use list so closure can mutate

    def callback(row):
        t_ps = row.get("time_ps", 0)
        if t_ps - last_printed[0] < interval_ps and last_printed[0] > 0:
            return
        last_printed[0] = t_ps
        dt = row.get("dt_fs", 0)
        tol = row.get("tol", 0)
        Sbar = row.get("Sbar", 0)
        T_t = row.get("T_target", 0)
        T_m = row.get("temp", 0)
        q1 = row.get("q_t1_mean", float("nan"))
        q2 = row.get("q_t2_mean", float("nan"))
        q_std = row.get("q_std", float("nan"))
        schedule_ps = row.get("schedule_ps", row.get("time_ps", 0))
        step_savings = row.get("step_savings_pct")
        base = f"  [{label}] t={t_ps:.2f} ps  dt={dt:.3f} fs  tol={tol:.2e}  Sbar={Sbar:.1f}  T={T_t:.0f}/{T_m:.0f} K  q_Si={q1:.3f} q_O={q2:.3f} σ_q={q_std:.3f}  ramp={schedule_ps:.2f} ps"
        if step_savings is not None:
            base += f"  savings={step_savings:.1f}%"
        print(base, flush=True)
        sys.stdout.flush()
    return callback


def run_from_restart_sio2(
    restart_file: str,
    ffield_path: str,
    out_dir: str,
    t_ps: float = T_PS,
    adapt_dt: bool = False,
    adapt_qeq: bool = False,
    adapt_ramp: bool = False,
    fixed_dt_fs: float = 0.1,
    fixed_tol: float = 1.0e-6,
    T_start: float = T_START,
    T_end: float = T_END,
    snap_every_ps: float = SNAP_EVERY_PS,
    lang_damp_fs: float = LANG_DAMP_FS,
    dt_target0_fs: float = 0.35,
    step_savings_reference_dt_fs: float = 0.1,
    schedule_category: str = "hat",
    T_constant: float = 300.0,
    label: str = "",
) -> float:
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "log.lammps")
    screen_file = os.path.join(out_dir, "screen.txt")
    lmp = make_lmp(log_file, screen_file)

    print(f"Reading restart from {restart_file}...")
    lmp.command(f'read_restart "{restart_file}"')
    reinit_reaxff_sio2(lmp, ffield_path)
    lmp.command("thermo 100")
    lmp.command("thermo_style custom step time temp press pe ke etotal")

    if schedule_category == "hat":
        T_schedule = make_hat_schedule(T_start, T_end, t_ps, 0.20, 0.30, 0.30, 0.20)
        T_init = T_start
        T_ref = (T_start + T_end) / 2.0
        schedule_desc = f"hat 300→{T_end}→300 K, 20/30/30/20"
    elif schedule_category == "milder":
        T_high_milder = 2000.0
        T_schedule = make_hat_schedule(T_start, T_high_milder, t_ps, 0.25, 0.25, 0.25, 0.25)
        T_init = T_start
        T_ref = (T_start + T_high_milder) / 2.0
        schedule_desc = f"milder 300→{T_high_milder}→300 K, 25/25/25/25"
    elif schedule_category == "constant_T":
        T_schedule = make_constant_temp(T_constant)
        T_init = T_constant
        T_ref = T_constant
        schedule_desc = f"constant T={T_constant} K"
    else:
        raise ValueError(f"Unknown schedule_category: {schedule_category}")

    ensure_integrator_and_thermostat(lmp, T=T_init, damp=lang_damp_fs)

    params = ETLParams(
        dt_target0_fs=dt_target0_fs,
        T_target=T_ref,
        lang_temp=T_init,
        lang_damp_fixed_fs=lang_damp_fs,
        tol_fixed=fixed_tol,
        out_dir=out_dir,
        snap_every_ps=snap_every_ps,
        chunk_steps=20,
        T_start=T_start,
        T_end=T_end,
        step_savings_reference_dt_fs=step_savings_reference_dt_fs,
        log_charge_stats=True,
        progress_callback=progress_monitor(label),
    )

    controller = ETLController(
        lmp=lmp,
        params=params,
        type_mass_g_per_mol=TYPE_MASS_SIO2,
        adapt_dt=adapt_dt,
        adapt_qeq=adapt_qeq,
        adapt_langevin=False,
        adapt_T=False,
        adapt_ramp=adapt_ramp,
        fixed_dt_fs=fixed_dt_fs,
        T_schedule=T_schedule,
    )

    mode_str = []
    if label:
        mode_str.append(f"[{label}]")
    mode_str.append("adaptive-dt" if adapt_dt else f"fixed-dt={fixed_dt_fs}fs")
    mode_str.append("adaptive-qeq" if adapt_qeq else f"fixed-tol={fixed_tol:.0e}")
    if adapt_ramp:
        mode_str.append("adaptive-ramp")
    print(f"Running {' '.join(mode_str)} for {t_ps} ps ({schedule_desc})...", flush=True)

    wall_start = time.time()
    controller.run(t_ps)
    wall_time = time.time() - wall_start
    lmp.close()
    print(f"  Done. Wall time: {wall_time:.2f} s", flush=True)
    try:
        with open(os.path.join(out_dir, "wall_time.txt"), "w") as f:
            f.write(f"{wall_time:.6f}\n")
    except Exception:
        pass
    return wall_time


def main():
    parser = argparse.ArgumentParser(description="SiO2 suite: 10 ps hat schedule, baselines + ETL")
    parser.add_argument("--write-restart", action="store_true", help="Create restart from silica.data")
    parser.add_argument("--all", action="store_true", help="Run full suite: reference, moderate, aggressive, etl-dt, etl-dt-ramp, etl-qeq, etl-full")
    parser.add_argument("--parallel", action="store_true", help="Run each selected case in a separate process (use with --all or individual flags)")
    parser.add_argument("--baseline-reference", action="store_true", help="Reference baseline dt=0.1 fs")
    parser.add_argument("--baseline-moderate", action="store_true", help="Moderate baseline dt=0.25 fs")
    parser.add_argument("--baseline-aggressive", action="store_true", help="Aggressive baseline dt=0.5 fs")
    parser.add_argument("--etl-dt", action="store_true")
    parser.add_argument("--etl-dt-ramp", action="store_true", help="ETL(dt+ramp): adaptive dt and ramp, fixed QEq (no adaptive QEq)")
    parser.add_argument("--etl-qeq", action="store_true")
    parser.add_argument("--etl-full", action="store_true", help="ETL(dt+QEq+ramp): adaptive dt, QEq, and ramp rate")
    parser.add_argument("--t-ps", type=float, default=T_PS)
    parser.add_argument("--T-end", type=float, default=T_END)
    parser.add_argument("--snap-every-ps", type=float, default=SNAP_EVERY_PS)
    parser.add_argument("--lang-damp", type=float, default=LANG_DAMP_FS)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--ffield", default=DEFAULT_FFIELD)
    parser.add_argument("--restart", default=DEFAULT_RESTART)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--out-prefix", type=str, default=None, help="Prefix for output dirs (e.g. v2) so baseline is not overwritten")
    parser.add_argument("--schedule", type=str, default="hat", choices=("hat", "milder", "constant_T"),
                        help="Schedule category: hat (300→T_end→300), milder (300→2000→300), constant_T")
    parser.add_argument("--all-schedules", action="store_true",
                        help="Run full suite for each schedule (hat, milder, constant_T) in turn")
    parser.add_argument("--T-constant", type=float, default=300.0, help="Temperature (K) for constant_T schedule")
    parser.add_argument("--large", action="store_true",
                        help="Use larger system: silica_1500.data and restart_sio2_1500.rst (create with --write-restart --large)")
    parser.add_argument("--large-suite", action="store_true",
                        help="Run reduced suite for larger system: only hat and constant_T schedules, only baseline_moderate and etl_full (implies --large)")
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)

    # --large-suite implies --large (same data/restart and output prefix)
    if args.large_suite:
        args.large = True
    if args.large:
        args.data = os.path.join(SCRIPT_DIR, LARGE_DATA)
        args.restart = os.path.join(SCRIPT_DIR, LARGE_RESTART)
        if args.out_prefix is None:
            args.out_prefix = "large"

    prefix = f"{args.out_prefix}_" if args.out_prefix else ""

    if args.write_restart:
        write_restart_sio2(
            data_file=args.data,
            ffield_path=args.ffield,
            restart_file=args.restart,
        )
        return

    if not os.path.exists(args.restart):
        print(f"Restart not found: {args.restart}")
        print("Run with --write-restart first.")
        sys.exit(1)

    # Large suite: only hat and constant_T; only baseline_moderate and etl_full (placeholders for other 5)
    if args.large_suite:
        args.baseline_reference = False
        args.baseline_aggressive = False
        args.etl_dt = False
        args.etl_dt_ramp = False
        args.etl_qeq = False
        args.baseline_moderate = True
        args.etl_full = True
        schedules_to_run = ["hat", "constant_T"]
        print("Large-suite mode: schedules=hat, constant_T; cases=baseline_moderate, etl_full (others placeholder)", flush=True)
    elif args.all_schedules:
        schedules_to_run = ["hat", "milder", "constant_T"]
    else:
        schedules_to_run = [args.schedule]

    def common_argv(schedule):
        return [
            "--t-ps", str(args.t_ps),
            "--T-end", str(args.T_end),
            "--snap-every-ps", str(args.snap_every_ps),
            "--lang-damp", str(args.lang_damp),
            "--restart", args.restart,
            "--ffield", args.ffield,
            "--schedule", schedule,
            "--T-constant", str(args.T_constant),
        ]

    if args.all:
        args.baseline_reference = True
        args.baseline_moderate = True
        args.baseline_aggressive = True
        args.etl_dt = True
        args.etl_dt_ramp = True
        args.etl_qeq = True
        args.etl_full = True

    for schedule in schedules_to_run:
        common = dict(
            restart_file=args.restart,
            ffield_path=args.ffield,
            t_ps=args.t_ps,
            T_start=T_START,
            T_end=args.T_end,
            snap_every_ps=args.snap_every_ps,
            lang_damp_fs=args.lang_damp,
            step_savings_reference_dt_fs=0.1,
            schedule_category=schedule,
            T_constant=args.T_constant,
        )
        wall_times = {}
        prefix = f"{args.out_prefix}_" if args.out_prefix else ""
        prefix_sched = f"{prefix}{schedule}_"

        def all_cases():
            out = []
            if args.baseline_reference:
                out.append(("baseline_reference", ["--baseline-reference"], f"outputs_sio2_{prefix_sched}baseline_reference"))
            if args.baseline_moderate:
                out.append(("baseline_moderate", ["--baseline-moderate"], f"outputs_sio2_{prefix_sched}baseline_moderate"))
            if args.baseline_aggressive:
                out.append(("baseline_aggressive", ["--baseline-aggressive"], f"outputs_sio2_{prefix_sched}baseline_aggressive"))
            if args.etl_dt:
                out.append(("etl_dt", ["--etl-dt"], f"outputs_sio2_{prefix_sched}etl_dt"))
            if args.etl_dt_ramp:
                out.append(("etl_dt_ramp", ["--etl-dt-ramp"], f"outputs_sio2_{prefix_sched}etl_dt_ramp"))
            if args.etl_qeq:
                out.append(("etl_qeq", ["--etl-qeq"], f"outputs_sio2_{prefix_sched}etl_qeq"))
            if args.etl_full:
                out.append(("etl_full", ["--etl-full"], f"outputs_sio2_{prefix_sched}etl_full"))
            return out

        def read_existing_wall_time(out_dir):
            wt_file = os.path.join(SCRIPT_DIR, out_dir, "wall_time.txt")
            if not os.path.isfile(wt_file):
                return None
            try:
                with open(wt_file) as f:
                    return float(f.read().strip().splitlines()[0])
            except Exception:
                return None

        # Hat only: skip cases that already have matching output (avoid re-running)
        use_parallel = args.parallel or args.all_schedules
        if use_parallel and (args.baseline_reference or args.baseline_moderate or args.baseline_aggressive or args.etl_dt or args.etl_dt_ramp or args.etl_qeq or args.etl_full):
            cases_to_run = []
            for name, case_flags, out_dir in all_cases():
                if schedule == "hat" and read_existing_wall_time(out_dir) is not None:
                    wall_times[name] = read_existing_wall_time(out_dir)
                    print(f"[hat] Skipping {name} (existing output in {out_dir})", flush=True)
                else:
                    cases_to_run.append((name, case_flags, out_dir))
            cases = cases_to_run
            script = os.path.abspath(os.path.join(SCRIPT_DIR, "run_suite_sio2.py"))
            cmd_base = [sys.executable, script]
            procs = []
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            print(f"\n--- Schedule: {schedule} ---", flush=True)
            for name, case_flags, out_dir in cases:
                argv = case_flags + common_argv(schedule) + ["--out-dir", out_dir]
                cmd = cmd_base + argv
                print(f"Starting {name} (out_dir={out_dir})...", flush=True)
                # Use main process stdout/stderr so parallel runs show output in Cursor terminal
                procs.append((name, out_dir, subprocess.Popen(
                    cmd, cwd=SCRIPT_DIR, env=env,
                    stdout=sys.stdout, stderr=sys.stderr,
                )))
            print("Waiting for all parallel runs to finish...", flush=True)
            for name, out_dir, p in procs:
                p.wait()
                if p.returncode != 0:
                    print(f"  [{name}] exited with code {p.returncode}", flush=True)
            for name, out_dir, _ in procs:
                wt = read_existing_wall_time(out_dir)
                wall_times[name] = wt if wt is not None else float("inf")
            if wall_times:
                print("\n" + "=" * 50)
                print(f"WALL TIME SUMMARY (schedule={schedule})")
                print("=" * 50)
                names = ALL_CASE_NAMES if args.large_suite else sorted(wall_times.keys())
                for name in names:
                    wt = wall_times.get(name)
                    if wt is None:
                        print(f"  {name:25s}: (placeholder)")
                    elif wt != float("inf"):
                        print(f"  {name:25s}: {wt:8.2f} s")
                    else:
                        print(f"  {name:25s}: (crashed)")
                ref_ref = wall_times.get("baseline_reference")
                ref_mod = wall_times.get("baseline_moderate")
                if ref_ref and ref_ref != float("inf") and len(wall_times) > 1:
                    print("\nSpeedups vs baseline_reference (0.1 fs):")
                    for name in sorted(wall_times.keys()):
                        if name != "baseline_reference":
                            wt = wall_times[name]
                            if wt != float("inf"):
                                print(f"  {name:25s}: {ref_ref/wt:.2f}x")
                if ref_mod and ref_mod != float("inf") and len(wall_times) > 1:
                    print("\nSpeedups vs baseline_moderate (0.25 fs):")
                    for name in sorted(wall_times.keys()):
                        if name != "baseline_moderate":
                            wt = wall_times[name]
                            if wt != float("inf"):
                                print(f"  {name:25s}: {ref_mod/wt:.2f}x")
            continue

        if len(schedules_to_run) > 1:
            print(f"\n--- Schedule: {schedule} ---", flush=True)

        def run_or_skip(name, out_dir, run_fn):
            if schedule == "hat" and read_existing_wall_time(out_dir) is not None:
                wall_times[name] = read_existing_wall_time(out_dir)
                print(f"[hat] Skipping {name} (existing output in {out_dir})", flush=True)
            else:
                wall_times[name] = run_fn()

        if args.baseline_reference:
            dt, tol = BASELINE_PRESETS["reference"]
            out = args.out_dir or f"outputs_sio2_{prefix_sched}baseline_reference"
            run_or_skip("baseline_reference", out, lambda: run_from_restart_sio2(
                out_dir=out, adapt_dt=False, adapt_qeq=False,
                fixed_dt_fs=dt, fixed_tol=tol, label="reference", **common
            ))

        if args.baseline_moderate:
            dt, tol = BASELINE_PRESETS["moderate"]
            out = args.out_dir or f"outputs_sio2_{prefix_sched}baseline_moderate"
            run_or_skip("baseline_moderate", out, lambda: run_from_restart_sio2(
                out_dir=out, adapt_dt=False, adapt_qeq=False,
                fixed_dt_fs=dt, fixed_tol=tol, label="moderate", **common
            ))

        if args.baseline_aggressive:
            dt, tol = BASELINE_PRESETS["aggressive"]
            out = args.out_dir or f"outputs_sio2_{prefix_sched}baseline_aggressive"
            def run_agg():
                try:
                    return run_from_restart_sio2(
                        out_dir=out, adapt_dt=False, adapt_qeq=False,
                        fixed_dt_fs=dt, fixed_tol=tol, label="aggressive", **common
                    )
                except Exception as e:
                    print(f"[aggressive] Crashed: {e}")
                    return float("inf")
            run_or_skip("baseline_aggressive", out, run_agg)

        if args.etl_dt:
            _, tol = BASELINE_PRESETS["moderate"]
            out = args.out_dir or f"outputs_sio2_{prefix_sched}etl_dt"
            run_or_skip("etl_dt", out, lambda: run_from_restart_sio2(
                out_dir=out, adapt_dt=True, adapt_qeq=False, fixed_tol=tol,
                label="ETL(dt)", **common
            ))

        if args.etl_dt_ramp:
            _, tol = BASELINE_PRESETS["moderate"]
            out = args.out_dir or f"outputs_sio2_{prefix_sched}etl_dt_ramp"
            run_or_skip("etl_dt_ramp", out, lambda: run_from_restart_sio2(
                out_dir=out, adapt_dt=True, adapt_qeq=False, adapt_ramp=True, fixed_tol=tol,
                label="ETL(dt+ramp)", **common
            ))

        if args.etl_qeq:
            out = args.out_dir or f"outputs_sio2_{prefix_sched}etl_qeq"
            run_or_skip("etl_qeq", out, lambda: run_from_restart_sio2(
                out_dir=out, adapt_dt=True, adapt_qeq=True,
                label="ETL(dt+QEq)", **common
            ))

        if args.etl_full:
            out = args.out_dir or f"outputs_sio2_{prefix_sched}etl_full"
            run_or_skip("etl_full", out, lambda: run_from_restart_sio2(
                out_dir=out, adapt_dt=True, adapt_qeq=True, adapt_ramp=True,
                label="ETL(dt+QEq+ramp)", **common
            ))

        if wall_times:
            print("\n" + "=" * 50)
            print(f"WALL TIME SUMMARY (schedule={schedule})")
            print("=" * 50)
            names = ALL_CASE_NAMES if args.large_suite else sorted(wall_times.keys())
            for name in names:
                wt = wall_times.get(name)
                if wt is None:
                    print(f"  {name:25s}: (placeholder)")
                elif wt != float("inf"):
                    print(f"  {name:25s}: {wt:8.2f} s")
                else:
                    print(f"  {name:25s}: (crashed)")
            ref_ref = wall_times.get("baseline_reference")
            ref_mod = wall_times.get("baseline_moderate")
            if ref_ref and ref_ref != float("inf") and len(wall_times) > 1:
                print("\nSpeedups vs baseline_reference (0.1 fs):")
                for name in (ALL_CASE_NAMES if args.large_suite else sorted(wall_times.keys())):
                    if name != "baseline_reference":
                        wt = wall_times.get(name)
                        if wt is not None and wt != float("inf"):
                            print(f"  {name:25s}: {ref_ref/wt:.2f}x")
            if ref_mod and ref_mod != float("inf") and len(wall_times) > 1:
                print("\nSpeedups vs baseline_moderate (0.25 fs):")
                for name in (ALL_CASE_NAMES if args.large_suite else sorted(wall_times.keys())):
                    if name != "baseline_moderate":
                        wt = wall_times.get(name)
                        if wt is not None and wt != float("inf"):
                            print(f"  {name:25s}: {ref_mod/wt:.2f}x")


if __name__ == "__main__":
    main()
