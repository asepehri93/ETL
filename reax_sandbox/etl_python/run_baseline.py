#!/usr/bin/env python3
"""Stage A: baseline fixed dt + fixed QEq tolerance"""
import argparse

from run_common import run_case, DEFAULT_T_TARGET


def main():
    ap = argparse.ArgumentParser(description="Stage A: baseline fixed dt + fixed QEq tolerance")
    ap.add_argument("--t-ps", type=float, default=50.0, help="Run length in ps")
    ap.add_argument("--dt-fs", type=float, default=0.1, help="Fixed timestep in fs")
    ap.add_argument("--tol", type=float, default=1.0e-6, help="Fixed QEq tolerance")
    ap.add_argument("--T", type=float, default=DEFAULT_T_TARGET, help="Target temperature in K")
    args = ap.parse_args()

    run_case(
        out_dir="baseline_outputs",
        t_ps=args.t_ps,
        adapt_dt=False,
        adapt_qeq=False,
        fixed_dt_fs=args.dt_fs,
        tol_fixed=args.tol,
        T_target=args.T,
    )


if __name__ == "__main__":
    main()
