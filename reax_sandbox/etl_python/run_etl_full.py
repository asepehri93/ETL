#!/usr/bin/env python3
"""
Stage D: Full ETL — adaptive dt + adaptive QEq tol + adaptive Langevin coupling

This uses in.reax_etl_full.lmp which does NOT define Langevin; Python owns it.
"""
import argparse
import os

from etl_controller import ETLController, ETLParams


DEFAULT_T_TARGET = 1500.0


def _require_lammps():
    try:
        from lammps import lammps  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "LAMMPS Python bindings not available. You need a LAMMPS build with the Python module.\n"
        ) from e
    return lammps


def make_lmp(*, out_dir: str, input_file: str = "in.reax_etl_full.lmp"):
    here = os.path.abspath(os.path.dirname(__file__))
    os.chdir(here)

    lammps = _require_lammps()
    out_dir_abs = os.path.abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    log_path = os.path.join(out_dir_abs, "log.lammps")
    screen_path = os.path.join(out_dir_abs, "screen.txt")

    lmp = lammps(cmdargs=["-log", log_path, "-screen", screen_path])
    lmp.file(input_file)
    return lmp


def main():
    ap = argparse.ArgumentParser(description="Stage D: Full ETL (dt + QEq + Langevin coupling)")
    ap.add_argument("--t-ps", type=float, default=50.0, help="Run length in ps")
    ap.add_argument("--T", type=float, default=DEFAULT_T_TARGET, help="Target temperature in K")
    ap.add_argument(
        "--dt-target0-fs",
        type=float,
        default=0.25,
        help="Auto-calibrate Delta_l so the first proposed dt ~= this value.",
    )
    args = ap.parse_args()

    out_dir = "etl_full_outputs"
    lmp = make_lmp(out_dir=out_dir)

    params = ETLParams(
        out_dir=out_dir,
        dt_target0_fs=args.dt_target0_fs,
        T_target=args.T,
        lang_temp=args.T,
    )
    controller = ETLController(
        lmp,
        params=params,
        adapt_dt=True,
        adapt_qeq=True,
        adapt_langevin=True,
        adapt_barostat=False,
    )
    controller.run(args.t_ps)


if __name__ == "__main__":
    main()
