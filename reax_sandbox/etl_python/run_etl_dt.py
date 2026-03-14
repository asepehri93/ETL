#!/usr/bin/env python3
"""Stage B: ETL adaptive dt + fixed QEq tolerance"""
import argparse

from run_common import run_case, DEFAULT_T_TARGET


def main():
    ap = argparse.ArgumentParser(description="Stage B: ETL adaptive dt + fixed QEq tolerance")
    ap.add_argument("--t-ps", type=float, default=50.0, help="Run length in ps")
    ap.add_argument("--tol", type=float, default=1.0e-6, help="Fixed QEq tolerance")
    ap.add_argument("--T", type=float, default=DEFAULT_T_TARGET, help="Target temperature in K")
    ap.add_argument(
        "--dt-target0-fs",
        type=float,
        default=0.25,
        help="Auto-calibrate Delta_l so the first proposed dt ~= this value.",
    )
    args = ap.parse_args()

    run_case(
        out_dir="etl_dt_outputs",
        t_ps=args.t_ps,
        adapt_dt=True,
        adapt_qeq=False,
        tol_fixed=args.tol,
        dt_target0_fs=args.dt_target0_fs,
        T_target=args.T,
    )


if __name__ == "__main__":
    main()
