#!/usr/bin/env python3
"""
Reproduce the old CHO ETL+QEq run (isolated from current SiO2/ETL work).
Uses logic extracted from ETL_dt_qeq_08242025.ipynb to run:
  - Fixed dt=0.1 fs baseline
  - ETL(dt+QEq) with dt_target0=0.30, dt_max=0.30, tol_max=5e-4

Run from repo root or old_etl:
  cd old_etl && python run_cho_reproduce.py [--t-ps 5]
  (default 5 ps for a quick check; use 50 to match the original 4x run)

Output: log_fixed_repro.json, log_etl_qeq_repro.json, and printed speedup.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Paths: run from repo root or from old_etl
SCRIPT_DIR = Path(__file__).resolve().parent
OLD_ETL = SCRIPT_DIR
PROJECT_ROOT = OLD_ETL.parent
DATA_V3 = OLD_ETL / "data.gasmix_cho_v3.lmp"
DATA_FALLBACK = OLD_ETL / "data.gasmix_cho.lmp"
FF = OLD_ETL / "ffield.reax.cho"
kB = 0.0019872041  # kcal/mol-K


def _ensure_data():
    """Ensure CHO data file exists; copy from reax_sandbox if needed. Returns relative path for LAMMPS (no spaces)."""
    if DATA_V3.exists():
        return "data.gasmix_cho_v3.lmp"
    if DATA_FALLBACK.exists():
        return "data.gasmix_cho.lmp"
    src = PROJECT_ROOT / "reax_sandbox" / "etl_python" / "data.gasmix_cho.lmp"
    if src.exists():
        shutil.copy(src, DATA_V3)
        return "data.gasmix_cho_v3.lmp"
    raise FileNotFoundError(
        f"No CHO data file. Put data.gasmix_cho_v3.lmp or data.gasmix_cho.lmp in {OLD_ETL}, "
        f"or ensure {src} exists to copy."
    )


# ----- ETL-dt-only (chunked) -----
@dataclass
class ETLCfg:
    T_target: float
    Delta_l: float | None = None
    dt_min: float = 0.02
    dt_max: float = 0.50
    dt_growth: float = 1.5
    dt_shrink: float = 2.0
    dt_target0: float = 0.10
    chunk: int = 10
    log_every: int = 1000
    mass_by_type: dict | None = None


class ETLTimestepOnly:
    def __init__(self, lmp, cfg: ETLCfg):
        self.lmp = lmp
        self.cfg = cfg
        self.step = 0
        self.dt_prev = cfg.dt_min
        if self.cfg.mass_by_type is None:
            self.cfg.mass_by_type = {1: 12.011, 2: 1.00784, 3: 15.999}
        if self.cfg.Delta_l is None:
            Sbar = self._Sbar_F2_over_m()
            self.cfg.Delta_l = self.cfg.dt_target0 * math.sqrt(Sbar / (kB * self.cfg.T_target))
            self.cfg.Delta_l = max(1e-6, min(self.cfg.Delta_l, 5.0))

    def _forces(self):
        f = self.lmp.extract_atom("f", 3)
        n = self.lmp.get_natoms()
        return np.array([[f[i][0], f[i][1], f[i][2]] for i in range(n)], dtype=float)

    def _types(self):
        t = self.lmp.extract_atom("type", 0)
        n = self.lmp.get_natoms()
        return np.array([int(t[i]) for i in range(n)], dtype=int)

    def _masses(self):
        types = self._types()
        return np.array([self.cfg.mass_by_type[int(t)] for t in types], dtype=float)

    def _Sbar_F2_over_m(self):
        F = self._forces()
        m = self._masses()
        N = max(len(m), 1)
        Stot = float(((F ** 2).sum(axis=1) / np.maximum(m, 1e-12)).sum())
        return Stot / (3.0 * N)

    def _choose_dt(self, Sbar: float) -> float:
        cfg = self.cfg
        if Sbar <= 0.0:
            dt = cfg.dt_max
        else:
            dt = cfg.Delta_l * math.sqrt((kB * cfg.T_target) / Sbar)
        dt = max(min(dt, self.dt_prev * cfg.dt_growth), self.dt_prev / cfg.dt_shrink)
        dt = min(max(dt, cfg.dt_min), cfg.dt_max)
        return dt

    def run_until(self, t_ps: float) -> dict:
        cum_fs = 0.0
        log = {"dt_fs": [], "cum_fs": []}
        while cum_fs < t_ps * 1000.0:
            Sbar = self._Sbar_F2_over_m()
            dt = self._choose_dt(Sbar)
            remaining = max(0.0, t_ps * 1000.0 - cum_fs)
            nsteps = int(min(self.cfg.chunk, math.ceil(remaining / max(dt, 1e-12))))
            self.lmp.command(f"timestep {dt:.6f}")
            self.lmp.command(f"run {nsteps} post no pre no")
            self.dt_prev = dt
            self.step += nsteps
            cum_fs += nsteps * dt
            if self.cfg.log_every and (self.step % self.cfg.log_every == 0):
                T = self.lmp.get_thermo("temp")
                print(f"[ETL] step={self.step} dt={dt:.3f} fs T≈{T:.1f}K cum={cum_fs/1000:.3f}ps")
            log["dt_fs"].append(dt)
            log["cum_fs"].append(cum_fs)
        return log


# ----- ETL + QEq -----
@dataclass
class ETLCfgQEq:
    T_target: float
    Delta_l: float | None = None
    dt_min: float = 0.02
    dt_max: float = 0.50
    dt_growth: float = 1.5
    dt_shrink: float = 2.0
    dt_target0: float = 0.10
    chunk: int = 10
    log_every: int = 1000
    mass_by_type: dict | None = None
    qeq_fix_id: str = "q"
    qeq_nevery: int = 1
    qeq_cutlo: float = 0.0
    qeq_cuthi: float = 10.0
    qeq_maxiter: int = 200
    tol_min: float = 1e-6
    tol_max: float = 1e-4
    alpha_qeq: float = 0.3
    cal_every: int = 50
    tol_hysteresis: float = 1.5


class ETLTimestepQEq:
    def __init__(self, lmp, cfg: ETLCfgQEq):
        self.lmp = lmp
        self.cfg = cfg
        self.step = 0
        self.dt_prev = cfg.dt_min
        if self.cfg.mass_by_type is None:
            self.cfg.mass_by_type = {1: 12.011, 2: 1.00784, 3: 15.999}
        self._A = None
        self._B = None
        self._last_tol = None
        self._ensure_qeq_fix(self.cfg.tol_min)
        self.lmp.command("run 0 post no pre no")
        if self.cfg.Delta_l is None:
            Sbar = self._Sbar_F2_over_m()
            self.cfg.Delta_l = self.cfg.dt_target0 * math.sqrt(Sbar / (kB * self.cfg.T_target))
            self.cfg.Delta_l = max(1e-6, min(self.cfg.Delta_l, 5.0))

    def _forces(self):
        f = self.lmp.extract_atom("f", 3)
        n = self.lmp.get_natoms()
        return np.array([[f[i][0], f[i][1], f[i][2]] for i in range(n)], dtype=float)

    def _types(self):
        t = self.lmp.extract_atom("type", 0)
        n = self.lmp.get_natoms()
        return np.array([int(t[i]) for i in range(n)], dtype=int)

    def _masses(self):
        types = self._types()
        return np.array([self.cfg.mass_by_type[int(t)] for t in types], dtype=float)

    def _Sbar_F2_over_m(self) -> float:
        F = self._forces()
        m = self._masses()
        N = max(len(m), 1)
        Stot = float(((F ** 2).sum(axis=1) / np.maximum(m, 1e-12)).sum())
        return Stot / (3.0 * N)

    def _choose_dt(self, Sbar: float) -> float:
        cfg = self.cfg
        if Sbar <= 0.0:
            dt = cfg.dt_max
        else:
            dt = cfg.Delta_l * math.sqrt((kB * cfg.T_target) / Sbar)
        dt = max(min(dt, self.dt_prev * cfg.dt_growth), self.dt_prev / cfg.dt_shrink)
        dt = min(max(dt, cfg.dt_min), cfg.dt_max)
        return dt

    def _ensure_qeq_fix(self, tol: float):
        tol = min(max(tol, self.cfg.tol_min), self.cfg.tol_max)
        if (self._last_tol is None) or (
            max(self._last_tol, tol) / max(min(self._last_tol, tol), 1e-30) >= self.cfg.tol_hysteresis
        ):
            try:
                self.lmp.command(f"unfix {self.cfg.qeq_fix_id}")
            except Exception:
                pass
            self.lmp.command(
                f"fix {self.cfg.qeq_fix_id} all qeq/reaxff "
                f"{self.cfg.qeq_nevery} {self.cfg.qeq_cutlo} {self.cfg.qeq_cuthi} "
                f"{tol} reaxff maxiter {self.cfg.qeq_maxiter}"
            )
            self._last_tol = tol

    def _calibrate_model(self):
        tight = self.cfg.tol_min
        loose = min(self.cfg.tol_max, max(self.cfg.tol_min * 10.0, 1e-4))
        self._ensure_qeq_fix(loose)
        self.lmp.command("run 0 post no pre no")
        F_loose = self._forces()
        self._ensure_qeq_fix(tight)
        self.lmp.command("run 0 post no pre no")
        F_tight = self._forces()
        self._ensure_qeq_fix(loose)
        self.lmp.command("run 0 post no pre no")
        dF = F_loose - F_tight
        m = self._masses()
        N = max(len(m), 1)
        err_bar_loose = float(((dF ** 2).sum(axis=1) / np.maximum(m, 1e-12)).sum()) / (3.0 * N)
        EPS = 1e-30
        err_bar_loose = max(err_bar_loose, EPS)
        err_bar_tight = max(err_bar_loose * 1e-3, EPS)
        if err_bar_loose <= 1e-10:
            self._A = self._B = None
            self._ensure_qeq_fix(self.cfg.tol_max)
            return
        x1, y1 = math.log10(loose), math.log10(err_bar_loose)
        x2, y2 = math.log10(tight), math.log10(err_bar_tight)
        B = (y1 - y2) / (x1 - x2)
        A = y1 - B * x1
        self._A, self._B = A, B

    def _pick_tol_for_dt(self, dt: float) -> float:
        if self._A is None or self._B is None:
            return self.cfg.tol_max
        cap = (
            self.cfg.alpha_qeq
            * (self.cfg.Delta_l ** 2)
            * (kB * self.cfg.T_target)
            / max(dt ** 2, 1e-30)
        )
        cap = max(min(cap, 1e20), 1e-30)
        log10_tol = (math.log10(cap) - self._A) / self._B
        tol = 10.0 ** log10_tol
        return min(max(tol, self.cfg.tol_min), self.cfg.tol_max)

    def run_until(self, t_ps: float) -> dict:
        if self.cfg.cal_every != 0:
            self._calibrate_model()
        cum_fs = 0.0
        log = {"dt_fs": [], "cum_fs": [], "tol": []}
        while cum_fs < t_ps * 1000.0:
            Sbar = self._Sbar_F2_over_m()
            dt = self._choose_dt(Sbar)
            tol = self._pick_tol_for_dt(dt)
            self._ensure_qeq_fix(tol)
            if self.cfg.cal_every and (
                (self.step == 0) or ((self.step // self.cfg.chunk) % self.cfg.cal_every == 0)
            ):
                self._calibrate_model()
                tol = self._pick_tol_for_dt(dt)
                self._ensure_qeq_fix(tol)
            remaining = max(0.0, t_ps * 1000.0 - cum_fs)
            nsteps = int(min(self.cfg.chunk, math.ceil(remaining / max(dt, 1e-12))))
            self.lmp.command(f"timestep {dt:.6f}")
            self.lmp.command(f"run {nsteps} post no pre no")
            self.dt_prev = dt
            self.step += nsteps
            cum_fs += nsteps * dt
            if self.cfg.log_every and (self.step % self.cfg.log_every == 0):
                T = self.lmp.get_thermo("temp")
                print(f"[ETL+QEq] step={self.step} dt={dt:.3f} fs tol~{tol:.1e} T≈{T:.1f}K cum={cum_fs/1000:.3f}ps")
            log["dt_fs"].append(dt)
            log["cum_fs"].append(cum_fs)
            log["tol"].append(tol)
        return log


# ----- Run helpers -----
SEED = 492783
T_TARGET = 1500.0


def setup_common(lmp, data_path: str, ff_path: str):
    lmp.command("units real")
    lmp.command("atom_style charge")
    lmp.command("boundary p p p")
    lmp.command(f"read_data {data_path}")
    lmp.command("pair_style reaxff NULL")
    lmp.command(f"pair_coeff * * {ff_path} C H O")
    lmp.command("fix q all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 200")
    lmp.command("neighbor 2.0 bin")
    lmp.command("neigh_modify every 1 delay 0 check yes")
    lmp.command(f"velocity all create {T_TARGET} {SEED} mom yes rot yes dist gaussian")
    lmp.command("fix int all nve")
    lmp.command(f"fix lang all langevin {T_TARGET} {T_TARGET} 100.0 90421")
    lmp.command("thermo_style custom step temp press pe ke etotal")
    lmp.command("thermo 500")
    lmp.command("run 0")


def write_restart(path: str, data_path: str, ff_path: str):
    from lammps import lammps
    lmp = lammps(cmdargs=["-log", str(OLD_ETL / "log.write_restart"), "-screen", "none"])
    setup_common(lmp, data_path, ff_path)
    lmp.command("timestep 0.10")
    lmp.command("thermo 1000")
    lmp.command("run 5000")
    lmp.command(f"write_restart {path}")
    lmp.close()


def ensure_integrator_and_thermostat(lmp, T: float):
    for fx in ("int", "lang"):
        try:
            lmp.command(f"unfix {fx}")
        except Exception:
            pass
    lmp.command("fix int all nve")
    lmp.command(f"fix lang all langevin {T} {T} 100.0 90421")
    lmp.command("neighbor 2.0 bin")
    lmp.command("neigh_modify every 1 delay 0 check yes")


def reinit_reaxff_after_restart(lmp, ffpath: str):
    lmp.command("pair_style reaxff NULL")
    lmp.command(f"pair_coeff * * {ffpath} C H O")
    try:
        lmp.command("unfix q")
    except Exception:
        pass
    lmp.command("fix q all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 200")


def run_fixed_dt(restart: str, data_path: str, ff_path: str, dt_fs: float, t_ps: float, logpath: str) -> float:
    from lammps import lammps
    steps = int(round((t_ps * 1000.0) / dt_fs))
    (OLD_ETL / "dumps_fixed").mkdir(exist_ok=True)
    lmp = lammps(cmdargs=["-log", str(OLD_ETL / "log.fixed_repro"), "-screen", "none"])
    lmp.command(f"read_restart {restart}")
    reinit_reaxff_after_restart(lmp, ff_path)
    ensure_integrator_and_thermostat(lmp, T_TARGET)
    lmp.command("reset_timestep 0")
    lmp.command("thermo_style custom step temp press pe ke etotal")
    lmp.command("thermo 500")
    steps_per_snap = max(1, int(round((0.05 * 1000.0) / dt_fs)))
    lmp.command(f"dump dfix all custom {steps_per_snap} dumps_fixed/fixed_repro_*.dump id type q x y z")
    lmp.command("dump_modify dfix sort id")
    lmp.command(f"timestep {dt_fs}")
    t0 = time.time()
    lmp.command(f"run {steps} post no pre no")
    t1 = time.time()
    lmp.close()
    with open(logpath, "w") as f:
        json.dump({"steps": steps, "dt_fs": dt_fs, "wall_s": t1 - t0}, f, indent=2)
    print(f"[fixed] ran {steps} steps @ {dt_fs} fs → ~{t_ps:.3f} ps; wall={t1-t0:.1f}s")
    return t1 - t0


def run_etl_dt_qeq(restart: str, ff_path: str, t_ps: float, logpath: str) -> float:
    from lammps import lammps
    lmp = lammps(cmdargs=["-log", str(OLD_ETL / "log.etlqeq_repro"), "-screen", "none"])
    lmp.command(f"read_restart {restart}")
    reinit_reaxff_after_restart(lmp, ff_path)
    ensure_integrator_and_thermostat(lmp, T_TARGET)
    lmp.command("reset_timestep 0")
    lmp.command("thermo 100000000")
    lmp.command("thermo_style custom step temp pe ke etotal press")
    lmp.command("dump detl all custom 10000 dumps/etlqeq_repro.*.dump id type q x y z")
    lmp.command("dump_modify detl sort id")
    ctl = ETLTimestepQEq(
        lmp,
        ETLCfgQEq(
            T_target=T_TARGET,
            Delta_l=None,
            dt_target0=0.30,
            dt_min=0.03,
            dt_max=0.30,
            dt_growth=1.5,
            dt_shrink=2.0,
            chunk=20,
            log_every=1000,
            mass_by_type={1: 12.011, 2: 1.00784, 3: 15.999},
            qeq_fix_id="q",
            qeq_nevery=1,
            qeq_cutlo=0.0,
            qeq_cuthi=10.0,
            qeq_maxiter=200,
            tol_min=1e-6,
            tol_max=5e-4,
            alpha_qeq=0.30,
            cal_every=100,
            tol_hysteresis=2.0,
        ),
    )
    t0 = time.time()
    log = ctl.run_until(t_ps)
    t1 = time.time()
    lmp.close()
    with open(logpath, "w") as f:
        json.dump({"etl_qeq": log, "wall_s": t1 - t0}, f, indent=2)
    print(f"[ETL+QEq] reached {t_ps:.3f} ps in {len(log['dt_fs'])} chunks; wall={t1-t0:.1f}s")
    return t1 - t0


def main():
    ap = argparse.ArgumentParser(description="Reproduce old CHO ETL+QEq run (fixed vs ETL+QEq).")
    ap.add_argument("--t-ps", type=float, default=5.0, help="Physical time in ps (default 5; use 50 to match original).")
    ap.add_argument("--skip-fixed", action="store_true", help="Skip fixed-dt run (use existing log_fixed_repro.json).")
    args = ap.parse_args()
    t_ps = args.t_ps

    os.chdir(OLD_ETL)
    (OLD_ETL / "dumps").mkdir(exist_ok=True)
    (OLD_ETL / "dumps_fixed").mkdir(exist_ok=True)
    (OLD_ETL / "restarts").mkdir(exist_ok=True)

    data_path = _ensure_data()
    ff_path = "ffield.reax.cho"
    if not (OLD_ETL / ff_path).exists():
        print(f"Missing {FF}", file=sys.stderr)
        sys.exit(1)

    rst_rel = "restarts/start_repro.rst"
    rst = str(OLD_ETL / "restarts" / "start_repro.rst")
    if not os.path.exists(rst):
        print("Creating restart...")
        write_restart(rst_rel, data_path, ff_path)

    log_fixed = str(OLD_ETL / "log_fixed_repro.json")
    log_etl = str(OLD_ETL / "log_etl_qeq_repro.json")

    if not args.skip_fixed:
        wall_fixed = run_fixed_dt(rst_rel, data_path, ff_path, dt_fs=0.1, t_ps=t_ps, logpath=log_fixed)
    else:
        with open(log_fixed) as f:
            wall_fixed = json.load(f)["wall_s"]
        print(f"[fixed] (skipped) using wall_s={wall_fixed:.1f} from {log_fixed}")

    wall_etl = run_etl_dt_qeq(rst_rel, ff_path, t_ps=t_ps, logpath=log_etl)

    speedup = wall_fixed / wall_etl
    print("\n" + "=" * 50)
    print(f"CHO reproduction ({t_ps} ps): fixed wall={wall_fixed:.1f}s  ETL+QEq wall={wall_etl:.1f}s")
    print(f"Speedup (fixed/ETL+QEq): {speedup:.2f}x")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
