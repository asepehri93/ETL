#!/usr/bin/env python3
"""
SiO2 ReaxFF baseline simulation (no ETL).

Runs amorphous silica through a hat-shaped temperature schedule:
  cold1 (30%) -> ramp up (20%) -> ramp down (20%) -> cold2 (30%)

This validates that the system is thermally stable under the temperature
schedule before ETL is added.
"""
import argparse
import csv
import math
import os
import sys
import time

try:
    from lammps import lammps
except ImportError:
    print("LAMMPS Python module not found. Please install it.")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "silica.data")
# Official LAMMPS tutorial ReaxFF for SiO2: lammpstutorials-inputs/tutorial5/ffield.reax.CHOFe (Si,O in pair_coeff)
DEFAULT_FFIELD = os.path.join(SCRIPT_DIR, "ffield.reax.CHOFe")


def make_hat_schedule(T_low, T_high, t_ps,
                      cold1_frac=0.30, up_frac=0.20,
                      down_frac=0.20, cold2_frac=0.30):
    total = cold1_frac + up_frac + down_frac + cold2_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    def schedule(time_ps):
        if t_ps <= 0:
            return T_low
        frac = min(1.0, max(0.0, time_ps / t_ps))
        if frac < cold1_frac:
            return T_low
        elif frac < cold1_frac + up_frac:
            p = (frac - cold1_frac) / up_frac if up_frac > 0 else 1.0
            return T_low + p * (T_high - T_low)
        elif frac < cold1_frac + up_frac + down_frac:
            p = (frac - cold1_frac - up_frac) / down_frac if down_frac > 0 else 1.0
            return T_high - p * (T_high - T_low)
        else:
            return T_low
    return schedule


def run_baseline(data_file, ffield_path, out_dir, t_ps,
                 T_start, T_end, dt_fs, tol, lang_damp_fs,
                 snap_every_ps, chunk_steps):
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, "log.lammps")
    screen_file = os.path.join(out_dir, "screen.txt")

    lmp = lammps(cmdargs=["-log", log_file, "-screen", screen_file])

    # Setup
    lmp.command("units real")
    lmp.command("atom_style full")
    lmp.command("boundary p p p")
    lmp.command(f'read_data "{data_file}"')

    lmp.command("pair_style reaxff NULL safezone 3.0 mincap 150")
    lmp.command(f'pair_coeff * * "{ffield_path}" Si O')
    lmp.command("neighbor 2.0 bin")
    lmp.command("neigh_modify every 1 delay 0 check yes")
    lmp.command(f"fix qeq all qeq/reaxff 1 0.0 10.0 {tol} reaxff maxiter 400")

    # Minimise
    lmp.command("min_style cg")
    lmp.command("minimize 1.0e-4 1.0e-6 200 2000")
    lmp.command("reset_timestep 0")

    # Initialise velocities
    lmp.command(f"velocity all create {T_start} 48279 dist gaussian")

    # Thermostat (NVE + Langevin)
    lmp.command("fix nve all nve")
    lmp.command(f"fix lang all langevin {T_start} {T_start} {lang_damp_fs} 48279")

    lmp.command(f"timestep {dt_fs}")
    lmp.command("thermo 100")
    lmp.command("thermo_style custom step time temp press pe ke etotal")

    # Brief equilibration at low T
    lmp.command("run 200 pre yes post no")
    lmp.command("reset_timestep 0")

    # Temperature schedule
    T_schedule = make_hat_schedule(T_start, T_end, t_ps)

    # Dump directory
    snap_dir = os.path.join(out_dir, "dumps")
    os.makedirs(snap_dir, exist_ok=True)
    snap_idx = 0
    next_snap_fs = snap_every_ps * 1000.0

    # CSV log
    csv_path = os.path.join(out_dir, "baseline_log.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "step", "time_fs", "time_ps", "dt_fs", "tol",
        "T_target", "temp", "press", "pe", "ke", "etotal",
    ])
    writer.writeheader()

    target_fs = t_ps * 1000.0
    time_fs = 0.0
    step = 0
    prev_T_target = None

    print(f"Hat schedule: {T_start} K (30%) -> {T_end} K (20%) -> {T_start} K (20%) -> {T_start} K (30%)")
    print(f"Running baseline for {t_ps} ps (dt={dt_fs} fs, tol={tol}, damp={lang_damp_fs} fs)...")

    wall_start = time.time()

    while time_fs < target_fs:
        time_ps = time_fs / 1000.0
        T_target = T_schedule(time_ps)

        if prev_T_target is None or abs(T_target - prev_T_target) > 0.1:
            try:
                lmp.command("unfix lang")
            except Exception:
                pass
            lmp.command(f"fix lang all langevin {T_target} {T_target} {lang_damp_fs} 48279")
            prev_T_target = T_target

        remaining_fs = target_fs - time_fs
        nsteps = min(chunk_steps, max(1, int(math.ceil(remaining_fs / dt_fs))))
        lmp.command(f"run {nsteps} pre yes post no")

        time_fs += dt_fs * nsteps
        step += nsteps

        temp = lmp.get_thermo("temp")
        press = lmp.get_thermo("press")
        pe = lmp.get_thermo("pe")
        ke = lmp.get_thermo("ke")
        etotal = lmp.get_thermo("etotal")

        writer.writerow({
            "step": step, "time_fs": time_fs, "time_ps": time_fs / 1000.0,
            "dt_fs": dt_fs, "tol": tol,
            "T_target": T_target, "temp": temp, "press": press,
            "pe": pe, "ke": ke, "etotal": etotal,
        })

        while time_fs >= next_snap_fs:
            t_ps_snap = next_snap_fs / 1000.0
            fname = os.path.join(snap_dir, f"frame_{snap_idx:06d}_{t_ps_snap:010.3f}ps.dump")
            lmp.command(f"write_dump all custom {fname} id type q x y z")
            snap_idx += 1
            next_snap_fs += snap_every_ps * 1000.0

    wall_time = time.time() - wall_start
    csv_file.close()
    lmp.close()

    print(f"Done. Wall time: {wall_time:.2f} s")
    print(f"Log: {csv_path}")
    print(f"Dumps: {snap_dir}/ ({snap_idx} frames)")
    return wall_time


def main():
    parser = argparse.ArgumentParser(description="SiO2 ReaxFF baseline (no ETL)")
    parser.add_argument("--t-ps", type=float, default=20.0)
    parser.add_argument("--T-start", type=float, default=300.0)
    parser.add_argument("--T-end", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.5, help="Timestep in fs")
    parser.add_argument("--tol", type=float, default=1.0e-6, help="QEq tolerance")
    parser.add_argument("--lang-damp", type=float, default=10.0, help="Langevin damping (fs)")
    parser.add_argument("--snap-every-ps", type=float, default=0.5)
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--ffield", type=str, default=DEFAULT_FFIELD)
    parser.add_argument("--out-dir", type=str, default="outputs_sio2_baseline")
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)
    run_baseline(
        data_file=args.data, ffield_path=args.ffield,
        out_dir=args.out_dir, t_ps=args.t_ps,
        T_start=args.T_start, T_end=args.T_end,
        dt_fs=args.dt, tol=args.tol,
        lang_damp_fs=args.lang_damp,
        snap_every_ps=args.snap_every_ps,
        chunk_steps=args.chunk_steps,
    )


if __name__ == "__main__":
    main()
