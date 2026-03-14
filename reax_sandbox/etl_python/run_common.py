import os
from typing import Optional

from etl_controller import ETLController, ETLParams


DEFAULT_T_TARGET = 1500.0


def _require_lammps():
    try:
        from lammps import lammps  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "LAMMPS Python bindings not available. You need a LAMMPS build with the Python module.\n"
            "Common options:\n"
            "- conda-forge: install a LAMMPS package that ships Python bindings\n"
            "- build LAMMPS from source with the PYTHON package enabled and ensure 'lammps' is on PYTHONPATH\n"
        ) from e
    return lammps


def make_lmp(*, out_dir: str, input_file: str = "in.reax_etl.lmp"):
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


def run_case(
    *,
    out_dir: str,
    t_ps: float,
    adapt_dt: bool,
    adapt_qeq: bool,
    fixed_dt_fs: float = 0.1,
    tol_fixed: float = 1.0e-6,
    dt_target0_fs: Optional[float] = None,
    T_target: float = DEFAULT_T_TARGET,
):
    lmp = make_lmp(out_dir=out_dir)

    params = ETLParams(
        out_dir=out_dir,
        tol_fixed=tol_fixed,
        dt_target0_fs=dt_target0_fs,
        T_target=T_target,
        lang_temp=T_target,
    )
    controller = ETLController(
        lmp,
        params=params,
        adapt_dt=adapt_dt,
        adapt_qeq=adapt_qeq,
        fixed_dt_fs=fixed_dt_fs,
    )
    controller.run(t_ps)
